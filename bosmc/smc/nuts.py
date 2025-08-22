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


class NUTSSMCKernel(NUTS):
    def sample_initial(self, *args, **kwargs):
        """Sample initial parameters from the prior."""
        if self._initial_params is None:
            raise RuntimeError("Kernel not properly initialized. Call setup() first.")
        return self._initial_params.copy()

    def sample(self, params):
        """
        Sample new parameters using NUTS trajectory, without accept/reject.
        
        Args:
            params: Current parameters
            
        Returns:
            new_params: New parameter values from NUTS trajectory
            incremental_log_weight: Log weight increment
        """
        z, potential_energy, z_grads = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, potential_energy, z_grads)
        
        # Sample momentum
        r, r_unscaled = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r_unscaled) + potential_energy

        # Set up slice variable
        if self.use_multinomial_sampling:
            log_slice = -energy_current
        else:
            # Sample slice variable
            slice_exp = torch.exp(torch.ones_like(energy_current))
            log_slice = -energy_current - slice_exp

        # Initialize tree
        z_left = z_right = z
        r_left = r_right = r
        r_left_unscaled = r_right_unscaled = r_unscaled
        z_left_grads = z_right_grads = z_grads
        r_sum = r_unscaled
        tree_weight = scalar_like(energy_current, 0.0 if self.use_multinomial_sampling else 1.0)
        
        z_proposal = z
        z_proposal_pe = potential_energy
        z_proposal_grads = z_grads

        # Build NUTS tree
        for tree_depth in range(self._max_tree_depth):
            # Choose direction (deterministic for reproducibility in SMC)
            direction = 1 if (self._t + tree_depth) % 2 == 0 else -1
            
            if direction == 1:
                new_tree = self._build_tree(
                    z_right, r_right, z_right_grads, log_slice, direction, tree_depth, energy_current
                )
                z_right = new_tree.z_right
                r_right = new_tree.r_right
                r_right_unscaled = new_tree.r_right_unscaled
                z_right_grads = new_tree.z_right_grads
            else:
                new_tree = self._build_tree(
                    z_left, r_left, z_left_grads, log_slice, direction, tree_depth, energy_current
                )
                z_left = new_tree.z_left
                r_left = new_tree.r_left
                r_left_unscaled = new_tree.r_left_unscaled
                z_left_grads = new_tree.z_left_grads

            # Stop if diverging or turning
            if new_tree.diverging or new_tree.turning:
                break

            # Update proposal (deterministic selection based on tree weight)
            if self.use_multinomial_sampling:
                new_tree_prob = (new_tree.weight - tree_weight).exp()
            else:
                new_tree_prob = new_tree.weight / tree_weight if tree_weight != 0 else 0.5
                
            if new_tree_prob > 0.5:  # Deterministic selection
                z_proposal = new_tree.z_proposal
                z_proposal_pe = new_tree.z_proposal_pe
                z_proposal_grads = new_tree.z_proposal_grads

            # Update tree weight and momentum sum
            r_sum = {
                site_name: r_sum[site_name] + new_tree.r_sum[site_name]
                for site_name in r_unscaled
            }
            
            if self._is_turning(r_left_unscaled, r_right_unscaled, r_sum):
                break
            else:
                if self.use_multinomial_sampling:
                    tree_weight = _logaddexp(tree_weight, new_tree.weight)
                else:
                    tree_weight = tree_weight + new_tree.weight

        # Compute incremental log weight
        energy_proposal = z_proposal_pe + self._kinetic_energy(r_unscaled)  # Keep same momentum for fair comparison
        incremental_log_weight = energy_current - energy_proposal

        # Update internal state
        self._energy_last = z_proposal_pe
        self._t += 1

        return z_proposal.copy(), incremental_log_weight

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
    
    nuts_smc_kernel = NUTSSMCKernel(pyro_model.sample, step_size=0.1, max_tree_depth=5)
    smc = SMC(nuts_smc_kernel, num_samples=100, num_iters=50, ess_threshold=0.5)

    smc.run()
    
    print("NUTS SMC estimate:", smc.get_samples())
    smc.summary()


if __name__ == '__main__':
    _test_gp_target()