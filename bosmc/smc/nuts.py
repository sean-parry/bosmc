from collections import namedtuple

import pyro
import pyro.distributions as dist
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.hmc import HMC
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


class NUTSSMCKernel(SMCKernel):
    """
    NUTS kernel adapted for Sequential Monte Carlo.
    
    Unlike standard NUTS, this version:
    - Does not use accept/reject in sample()
    - Takes current parameters and returns new parameters with incremental log weight
    - Uses the NUTS trajectory to propose new states
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        step_size=1.0,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        transforms=None,
        max_plate_nesting=None,
        jit_compile=False,
        jit_options=None,
        ignore_jit_warnings=False,
        target_accept_prob=0.8,
        max_tree_depth=10,
        init_strategy=init_to_uniform,
    ):
        self.model = model
        self.potential_fn = potential_fn
        self.step_size = step_size
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.full_mass = full_mass
        self.use_multinomial_sampling = use_multinomial_sampling
        self.transforms = transforms
        self.max_plate_nesting = max_plate_nesting
        self.jit_compile = jit_compile
        self.jit_options = jit_options or {}
        self.ignore_jit_warnings = ignore_jit_warnings
        self.target_accept_prob = target_accept_prob
        self._max_tree_depth = max_tree_depth
        self.init_strategy = init_strategy
        
        # SMC-specific attributes
        self._t = 0
        self._warmup_steps = 0
        self._max_sliced_energy = 1000
        self._energy_last = None
        
        # These will be set during setup
        self._initial_params = None
        self._prototype_trace = None
        
        # Mass matrix and adaptation (simplified for SMC)
        self.mass_matrix_adapter = None
        self.inverse_mass_matrix = {}
        
        super().__init__()

    def setup(self, warmup_steps, *args, **kwargs):
        """Initialize the kernel."""
        self._warmup_steps = warmup_steps
        
        # Initialize model
        (
            self._initial_params,
            self.potential_fn,
            self.transforms,
            self._prototype_trace,
        ) = initialize_model(
            self.model,
            model_args=args,
            model_kwargs=kwargs,
        )
        
        # Initialize energy
        self._energy_last = self.potential_fn(self._initial_params)
        
        # Set up mass matrix (simplified - use identity)
        self.inverse_mass_matrix = {}
        for name, param in self._initial_params.items():
            self.inverse_mass_matrix[name] = torch.ones_like(param)
        
        # Simple mass matrix adapter mock
        self.mass_matrix_adapter = SimpleMassMatrixAdapter(self.inverse_mass_matrix)

    def sample_initial(self, *args, **kwargs):
        """Sample initial parameters."""
        if self._initial_params is None:
            raise RuntimeError("Kernel not properly initialized. Call setup() first.")
        return self._initial_params.copy()

    def _kinetic_energy(self, r_unscaled):
        """Compute kinetic energy."""
        energy = 0.0
        for name, r in r_unscaled.items():
            energy += 0.5 * (r * self.inverse_mass_matrix[name] * r).sum()
        return energy

    def _sample_r(self, name_prefix="r"):
        """Sample momentum variables."""
        r = {}
        r_unscaled = {}
        for site_name, param in self._initial_params.items():
            r_unscaled[site_name] = torch.randn_like(param)
            r[site_name] = r_unscaled[site_name] / torch.sqrt(self.inverse_mass_matrix[site_name])
        return r, r_unscaled

    def _is_turning(self, r_left_unscaled, r_right_unscaled, r_sum):
        """Check if trajectory is making a U-turn."""
        left_angle = 0.0
        right_angle = 0.0
        for site_name, value in r_sum.items():
            rho = value - (r_left_unscaled[site_name] + r_right_unscaled[site_name]) / 2
            left_angle += r_left_unscaled[site_name].dot(rho)
            right_angle += r_right_unscaled[site_name].dot(rho)
        return (left_angle <= 0) or (right_angle <= 0)

    def _build_basetree(self, z, r, z_grads, log_slice, direction, energy_current):
        """Build base tree (single leapfrog step)."""
        step_size = self.step_size if direction == 1 else -self.step_size
        
        # Simple velocity verlet step (simplified version)
        z_new = {}
        r_new = {}
        for name, param in z.items():
            # Half step for momentum
            r_half = r[name] - 0.5 * step_size * z_grads[name]
            # Full step for position  
            z_new[name] = param + step_size * r_half / self.inverse_mass_matrix[name]
        
        # Compute new gradients and energy
        z_grads_new, potential_energy = potential_grad(self.potential_fn, z_new)
        
        # Complete momentum step
        for name in r.keys():
            r_new[name] = r_half - 0.5 * step_size * z_grads_new[name]
        
        r_new_unscaled = self.mass_matrix_adapter.unscale(r_new)
        energy_new = potential_energy + self._kinetic_energy(r_new_unscaled)
        
        # Handle NaN case
        energy_new = (
            scalar_like(energy_new, float("inf"))
            if torch_isnan(energy_new)
            else energy_new
        )
        
        sliced_energy = energy_new + log_slice
        diverging = sliced_energy > self._max_sliced_energy
        delta_energy = energy_new - energy_current
        accept_prob = (-delta_energy).exp().clamp(max=1.0)

        if self.use_multinomial_sampling:
            tree_weight = -sliced_energy
        else:
            tree_weight = scalar_like(sliced_energy, 1.0 if sliced_energy <= 0 else 0.0)

        r_sum = r_new_unscaled
        return _TreeInfo(
            z_new, r_new, r_new_unscaled, z_grads_new,
            z_new, r_new, r_new_unscaled, z_grads_new,
            z_new, potential_energy, z_grads_new,
            r_sum, tree_weight, False, diverging, accept_prob, 1,
        )

    def _build_tree(self, z, r, z_grads, log_slice, direction, tree_depth, energy_current):
        """Build NUTS tree recursively."""
        if tree_depth == 0:
            return self._build_basetree(z, r, z_grads, log_slice, direction, energy_current)

        # Build first half
        half_tree = self._build_tree(z, r, z_grads, log_slice, direction, tree_depth - 1, energy_current)
        z_proposal = half_tree.z_proposal
        z_proposal_pe = half_tree.z_proposal_pe
        z_proposal_grads = half_tree.z_proposal_grads

        # Check stopping conditions
        if half_tree.turning or half_tree.diverging:
            return half_tree

        # Build second half
        if direction == 1:
            z = half_tree.z_right
            r = half_tree.r_right
            z_grads = half_tree.z_right_grads
        else:
            z = half_tree.z_left
            r = half_tree.r_left
            z_grads = half_tree.z_left_grads
            
        other_half_tree = self._build_tree(z, r, z_grads, log_slice, direction, tree_depth - 1, energy_current)

        # Combine trees
        if self.use_multinomial_sampling:
            tree_weight = _logaddexp(half_tree.weight, other_half_tree.weight)
        else:
            tree_weight = half_tree.weight + other_half_tree.weight
            
        sum_accept_probs = half_tree.sum_accept_probs + other_half_tree.sum_accept_probs
        num_proposals = half_tree.num_proposals + other_half_tree.num_proposals
        
        r_sum = {
            site_name: half_tree.r_sum[site_name] + other_half_tree.r_sum[site_name]
            for site_name in self.inverse_mass_matrix
        }

        # Choose proposal
        if self.use_multinomial_sampling:
            other_half_prob = (other_half_tree.weight - tree_weight).exp()
        else:
            other_half_prob = (
                other_half_tree.weight / tree_weight if tree_weight > 0 
                else scalar_like(tree_weight, 0.0)
            )
            
        # Use deterministic selection based on weight for SMC
        if other_half_prob > 0.5:
            z_proposal = other_half_tree.z_proposal
            z_proposal_pe = other_half_tree.z_proposal_pe
            z_proposal_grads = other_half_tree.z_proposal_grads

        # Set tree boundaries
        if direction == 1:
            z_left = half_tree.z_left
            r_left = half_tree.r_left
            r_left_unscaled = half_tree.r_left_unscaled
            z_left_grads = half_tree.z_left_grads
            z_right = other_half_tree.z_right
            r_right = other_half_tree.r_right
            r_right_unscaled = other_half_tree.r_right_unscaled
            z_right_grads = other_half_tree.z_right_grads
        else:
            z_left = other_half_tree.z_left
            r_left = other_half_tree.r_left
            r_left_unscaled = other_half_tree.r_left_unscaled
            z_left_grads = other_half_tree.z_left_grads
            z_right = half_tree.z_right
            r_right = half_tree.r_right
            r_right_unscaled = half_tree.r_right_unscaled
            z_right_grads = half_tree.z_right_grads

        # Check for U-turn
        turning = other_half_tree.turning or self._is_turning(r_left_unscaled, r_right_unscaled, r_sum)
        diverging = other_half_tree.diverging

        return _TreeInfo(
            z_left, r_left, r_left_unscaled, z_left_grads,
            z_right, r_right, r_right_unscaled, z_right_grads,
            z_proposal, z_proposal_pe, z_proposal_grads,
            r_sum, tree_weight, turning, diverging,
            sum_accept_probs, num_proposals,
        )

    def sample(self, params):
        """
        Sample new parameters using NUTS trajectory, without accept/reject.
        
        Args:
            params: Current parameters
            log_weight: Current log weight (for context, not used in computation)
            
        Returns:
            new_params: New parameter values from NUTS trajectory
            incremental_log_weight: Log weight increment
        """
        z = params
        z_grads, potential_energy = potential_grad(self.potential_fn, z)
        
        # Sample momentum
        r, r_unscaled = self._sample_r()
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


class SimpleMassMatrixAdapter:
    """Simplified mass matrix adapter for SMC NUTS."""
    
    def __init__(self, inverse_mass_matrix):
        self.inverse_mass_matrix = inverse_mass_matrix
    
    def unscale(self, r):
        """Convert scaled momentum to unscaled."""
        r_unscaled = {}
        for name, momentum in r.items():
            r_unscaled[name] = momentum * torch.sqrt(self.inverse_mass_matrix[name])
        return r_unscaled
    
    def kinetic_grad(self, r_unscaled):
        """Compute kinetic gradient (not used in simplified version)."""
        return r_unscaled
    
# Example usage
def _test_gp_target():
    import pyro.distributions as dist
    from bosmc.smc.api import SMC  # Import your SMC class
    from botorch.models.fully_bayesian import SaasPyroModel

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
    
    print("NUTS SMC estimate:", smc.get_samples().mean(0))
    smc.summary()


if __name__ == '__main__':
    _test_gp_target()