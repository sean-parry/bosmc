from collections import namedtuple

import pyro
import pyro.distributions as dist
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan
from bosmc.smc.smc_kernel import SMCKernel
from pyro.infer.mcmc.util import initialize_model
import torch
import math


def _logaddexp(x, y):
    minval, maxval = (x, y) if x < y else (y, x)
    return (minval - maxval).exp().log1p() + maxval

_TreeInfo = namedtuple(
    "TreeInfo",
    [
        "z_left",
        "r_left",
        "r_left_unscaled",
        "z_left_grads",
        "z_right",
        "r_right", 
        "r_right_unscaled",
        "z_right_grads",
        "z_proposal",
        "z_proposal_pe",
        "z_proposal_grads",
        "r_sum",
        "weight",
        "turning",
        "diverging",
        "sum_accept_probs",
        "num_proposals",
    ],
)


class NUTSSMCKernelAR(NUTS):
    def sample_initial(self, *args, **kwargs):
        """Sample initial parameters from the prior."""
        if self._initial_params is None:
            raise RuntimeError("Kernel not properly initialized. Call setup() first.")
        return self._initial_params.copy()

    def sample(self, params):
        """
        Returns new param and incremental log weigth
        """
        z, potential_energy, z_grads = self._fetch_from_cache()

        initial_potential_energy = potential_energy
        # recompute PE when cache is cleared
        if z is None:
            z = params
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, potential_energy, z_grads)
        # return early if no sample sites
        elif len(z) == 0:
            self._t += 1
            self._mean_accept_prob = 1.0
            if self._t > self._warmup_steps:
                self._accept_cnt += 1
            return z
        r, r_unscaled = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r_unscaled) + potential_energy

        if self.use_multinomial_sampling:
            log_slice = -energy_current
        else:
            # Rather than sampling the slice variable from `Uniform(0, exp(-energy))`, we can
            # sample log_slice directly using `energy`, so as to avoid potential underflow or
            # overflow issues ([2]).
            slice_exp_term = pyro.sample(
                "slicevar_exp_t={}".format(self._t),
                dist.Exponential(scalar_like(energy_current, 1.0)),
            )
            log_slice = -energy_current - slice_exp_term

        z_left = z_right = z
        r_left = r_right = r
        r_left_unscaled = r_right_unscaled = r_unscaled
        z_left_grads = z_right_grads = z_grads
        accepted = False
        r_sum = r_unscaled
        sum_accept_probs = 0.0
        num_proposals = 0
        tree_weight = scalar_like(
            energy_current, 0.0 if self.use_multinomial_sampling else 1.0
        )

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation.
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            # doubling process, stop when turning or diverging
            tree_depth = 0
            while tree_depth < self._max_tree_depth:
                direction = pyro.sample(
                    "direction_t={}_treedepth={}".format(self._t, tree_depth),
                    dist.Bernoulli(probs=scalar_like(tree_weight, 0.5)),
                )
                direction = int(direction.item())
                if (
                    direction == 1
                ):  # go to the right, start from the right leaf of current tree
                    new_tree = self._build_tree(
                        z_right,
                        r_right,
                        z_right_grads,
                        log_slice,
                        direction,
                        tree_depth,
                        energy_current,
                    )
                    # update leaf for the next doubling process
                    z_right = new_tree.z_right
                    r_right = new_tree.r_right
                    r_right_unscaled = new_tree.r_right_unscaled
                    z_right_grads = new_tree.z_right_grads
                else:  # go the the left, start from the left leaf of current tree
                    new_tree = self._build_tree(
                        z_left,
                        r_left,
                        z_left_grads,
                        log_slice,
                        direction,
                        tree_depth,
                        energy_current,
                    )
                    z_left = new_tree.z_left
                    r_left = new_tree.r_left
                    r_left_unscaled = new_tree.r_left_unscaled
                    z_left_grads = new_tree.z_left_grads

                sum_accept_probs = sum_accept_probs + new_tree.sum_accept_probs
                num_proposals = num_proposals + new_tree.num_proposals

                # stop doubling
                if new_tree.diverging:
                    if self._t >= self._warmup_steps:
                        self._divergences.append(self._t - self._warmup_steps)
                    break

                if new_tree.turning:
                    break

                tree_depth += 1

                if self.use_multinomial_sampling:
                    new_tree_prob = (new_tree.weight - tree_weight).exp()
                else:
                    new_tree_prob = new_tree.weight / tree_weight
                rand = pyro.sample(
                    "rand_t={}_treedepth={}".format(self._t, tree_depth),
                    dist.Uniform(
                        scalar_like(new_tree_prob, 0.0), scalar_like(new_tree_prob, 1.0)
                    ),
                )
                if rand < new_tree_prob:
                    accepted = True
                    z = new_tree.z_proposal
                    z_grads = new_tree.z_proposal_grads
                    self._cache(z, new_tree.z_proposal_pe, z_grads)

                r_sum = {
                    site_names: r_sum[site_names] + new_tree.r_sum[site_names]
                    for site_names in r_unscaled
                }
                if self._is_turning(
                    r_left_unscaled, r_right_unscaled, r_sum
                ):  # stop doubling
                    break
                else:  # update tree_weight
                    if self.use_multinomial_sampling:
                        tree_weight = _logaddexp(tree_weight, new_tree.weight)
                    else:
                        tree_weight = tree_weight + new_tree.weight

        accept_prob = sum_accept_probs / num_proposals

        self._t += 1
        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t
            self._adapter.step(self._t, z, accept_prob, z_grads)
        self._mean_accept_prob += (accept_prob.item() - self._mean_accept_prob) / n

        incremental_log_weight = 0.0
        if accepted:
            _, final_potential_energy, _ = self._fetch_from_cache()
            incremental_log_weight = initial_potential_energy - final_potential_energy

        return z.copy(), incremental_log_weight

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def logging(self):
        return {"step_size": "{:.2e}".format(self.step_size), "iteration": self._t}

    def diagnostics(self):
        return {"step_size": self.step_size, "iteration": self._t}
    
# Example usage
def _test_gp_target():
    import pyro.distributions as dist
    from bosmc.smc.api import SMC  # Import your SMC class
    from botorch.models.fully_bayesian import SaasPyroModel
    # from pyro.infer.mcmc.nuts import NUTS
    # from pyro.infer.mcmc import MCMC

    num_datapoints: int = 10
    dim: int = 3
    train_X = torch.rand((num_datapoints, dim), dtype=torch.float64)
    train_y = torch.rand((num_datapoints, 1), dtype=torch.float64)

    pyro_model = SaasPyroModel()
    pyro_model.set_inputs(train_X, train_y)
    
    print("Running NUTS SMC...")
    
    nuts_smc_kernel = NUTSSMCKernelAR(pyro_model.sample, step_size=0.1, max_tree_depth=5)
    smc = SMC(nuts_smc_kernel, num_samples=100, num_iters=50, ess_threshold=0.5)

    smc.run()
    
    print("NUTS SMC estimate:", smc.get_samples())
    smc.summary()

def _test_nuts_bernouli_mcmc():
    import pyro.distributions as dist
    from pyro.infer.mcmc.nuts import NUTS
    from pyro.infer.mcmc import MCMC

    true_coefs = torch.tensor([1., 2., 3.])
    data = torch.randn(2000, 3)
    dim = 3
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = torch.zeros(dim)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y
    
    print("Running SMC...")
    nuts = NUTS(model, step_size=0.3, max_tree_depth=20)
    mcmc = MCMC(nuts, warmup_steps= 100 ,num_samples=5)
    mcmc.run(data)
    
    print("\nTrue coefficients:", true_coefs)
    print("SMC estimate:", mcmc.get_samples()['beta'].mean(0))
    mcmc.summary()
    
    return

def _test_nuts_bernouli_smc():
    import pyro.distributions as dist
    from bosmc.smc.api import SMC

    true_coefs = torch.tensor([1., 2., 3.])
    data = torch.randn(2000, 3)
    dim = 3
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = torch.zeros(dim)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y
    
    print("Running SMC...")
    nuts = NUTSSMCKernelAR(model, step_size=0.3, max_tree_depth=20)
    smc = SMC(nuts, num_iters= 200, num_samples=100)
    smc.run(data)
    
    print("\nTrue coefficients:", true_coefs)
    print("SMC estimate:", smc.get_samples()['beta'].mean(0))
    smc.summary()
    
    return

if __name__ == '__main__':
    _test_nuts_bernouli_smc()
    #_test_gp_target()