import copy
import json
import logging
import queue
import signal
import threading
import warnings
import tqdm
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, List, Callable
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.multiprocessing as mp

import pyro
import pyro.poutine as poutine

from pyro.infer.mcmc.nuts import NUTS # should be changed for smc kernel
from pyro.infer.mcmc.hmc import HMC # should be changed for smc kernel
from bosmc.smc.smc_kernel import SMCKernel
from pyro.infer.mcmc.util import (
    diagnostics,
    diagnostics_from_stats,
    print_summary,
    select_samples,
)

from pyro.infer.mcmc.api import _Worker, _UnarySampler, _MultiSampler
from pyro.ops.streaming import CountMeanVarianceStats, StatsOfDict, StreamingStats
from pyro.util import optional

import math
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
from bosmc.smc.smc_kernel import SMCKernel
from pyro.infer.mcmc.util import initialize_model

def _update_particle_static(args):
    """
    Static worker function for process pool. 
    Takes a tuple (kernel, particle) and returns the result of the kernel's sample method.
    """
    kernel, particle = args
    # Note: Any setup or context needed by the kernel must be initialized here
    # or be part of the kernel object itself.
    return kernel.sample(particle)


class AbstractSMC(ABC):
    """
    Base class for SMC methods.
    """

    def __init__(self, 
                 kernels: list[SMCKernel],
                 transforms,
                 ) -> None:
        self.kernels = kernels
        self.kernel = self.kernels[0] if kernels else None
        self.transforms = transforms

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self):
        raise NotImplementedError

    def _set_transforms(self, *args, **kwargs):
        # Use `kernel.transforms` when available
        if getattr(self.kernel, "transforms", None) is not None:
            self.transforms = self.kernel.transforms
        # Else, get transforms from model (e.g. in multiprocessing).
        elif self.kernel.model:
            warmup_steps = 0
            self.kernel.setup(warmup_steps, *args, **kwargs)
            self.transforms = self.kernel.transforms
        # Assign default value
        else:
            self.transforms = {}

    def _validate_kernel(self, initial_params):
        """
        Validate SMC kernel setup
        """
        if not hasattr(self.kernel, 'sample'):
            raise ValueError("Kernel must implement sample() method")

    def _validate_initial_params(self, initial_params):
        pass


class SMC(AbstractSMC):
    """
    Sequential Monte Carlo sampler.
    """

    def __init__(
        self,
        kernel,
        num_samples: int,
        num_iters: int,
        ess_threshold: float = 0.5,
        initial_params: Optional[Dict] = None,
        hook_fn: Optional[Callable] = None,
        disable_progbar: bool = False,
        disable_validation: bool = True,
        transforms: Optional[Dict] = None,
    ):
        # Create a separate kernel for each particle
        kernels = [copy.deepcopy(kernel) for _ in range(num_samples)]
        super().__init__(kernels, transforms)
        
        self.num_iters = num_iters
        self.num_samples = num_samples
        self.ess_threshold = ess_threshold
        self.disable_validation = disable_validation
        self.disable_progbar = disable_progbar
        self.hook_fn = hook_fn
        self.initial_params = initial_params
        
        self._samples = None
        self._weights = None
        self._log_weights = None
        self._args = None
        self._kwargs = None
        self._diagnostics = {}
        self._resampling_counts = []
            
        self._validate_kernel(initial_params)
        
    def _validate_initial_params(self, initial_params: Dict):
        """Validate that initial parameters have correct dimensions for particles."""
        if initial_params is None:
            return
            
        for name, param in initial_params.items():
            if param.shape[0] != self.num_samples:
                raise ValueError(
                    f"Initial parameter '{name}' has batch size {param.shape[0]} "
                    f"but expected {self.num_samples} particles."
                )
    
    def _compute_ess(self, log_weights: torch.Tensor) -> float:
        """Compute effective sample size from log weights."""
        weights = torch.exp(log_weights)
        
        ess = 1.0 / torch.sum(weights ** 2).item()
        return ess
    
    def _resample_particles(self, particles: List[Dict], log_weights: torch.Tensor):
        """Slow placeholder implementation of stratified resampling scheme"""
        # Normalize weights
        log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
        weights = torch.exp(log_weights_normalized)
        
        # Systematic resampling
        indices = self._systematic_resample(weights)
        
        # Resample particles
        resampled_particles = [particles[i] for i in indices]
        
        # Resample kernels (create new copies to reset their state)
        
        # Reset weights to uniform after resampling
        uniform_log_weight = -torch.log(torch.tensor(self.num_samples, dtype=torch.float))
        new_log_weights = torch.full((self.num_samples,), uniform_log_weight)
        
        return resampled_particles, new_log_weights
    
    def _systematic_resample(self, weights: torch.Tensor) -> torch.Tensor:
        """Systematic resampling algorithm."""
        num_samples = len(weights)
        indices = torch.zeros(num_samples, dtype=torch.long)
        
        # Generate systematic samples
        u = torch.rand(1).item() / num_samples
        cumsum = torch.cumsum(weights, dim=0)
        
        i, j = 0, 0
        while i < num_samples and j < num_samples:
            while j < num_samples and cumsum[j] < u:
                j += 1
            if j < num_samples:
                indices[i] = j
                u += 1.0 / num_samples
                i += 1
            else:
                # Handle edge case - fill remaining with last valid index
                indices[i:] = num_samples - 1
                break
                
        return indices
    
    @poutine.block
    def run(self, *args, **kwargs):
        """Run SMC to generate samples."""
        self._args, self._kwargs = args, kwargs

        def particle_updater(i):
            new_particle, inc_log_weight = self.kernels[i].sample(particles[i])
            return new_particle, inc_log_weight
        
        with optional(
            pyro.validation_enabled(not self.disable_validation),
            self.disable_validation is not None,
        ):
            args = [arg.detach() if torch.is_tensor(arg) else arg for arg in args]
            
            # Initialize all kernels
            for i, kernel in enumerate(self.kernels):
                kernel.setup(0, *args, **kwargs)  # No warmup for SMC
            
            # Initialize particles - each particle is a separate dict
            if self.initial_params is not None:
                self._validate_initial_params(self.initial_params)
                particles = []
                for i in range(self.num_samples):
                    particle = {}
                    for name, param in self.initial_params.items():
                        particle[name] = param[i].clone()
                    particles.append(particle)
            else:
                particles = []
                for i in range(self.num_samples):
                    particles.append(self.kernels[i].sample_initial(*args, **kwargs))
            
            # Initialize uniform weights
            log_weights = torch.full(
                (self.num_samples,), 
                -torch.log(torch.tensor(self.num_samples, dtype=torch.float))
            )
            
            # Storage for diagnostics
            particle_history = []
            weight_history = []
            
            # Main SMC loop
            for iteration in tqdm.tqdm(iterable = range(self.num_iters), 
                                       disable=self.disable_progbar):
                # Propagate each particle with its corresponding kernel
                new_particles = []
                incremental_log_weights = torch.zeros(self.num_samples)
                
                kernel_particle_pairs = zip(self.kernels, particles)
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(_update_particle_static, kernel_particle_pairs))
                            
                new_particles, incremental_log_weights = zip(*results)
            
                # Convert results back to the correct format
                particles = list(new_particles)
                incremental_log_weights = torch.tensor(incremental_log_weights, dtype = torch.float32)
                
                # Update log weights
                log_weights = log_weights + incremental_log_weights
                # normalize
                log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
                
                # Compute ESS
                ess = self._compute_ess(log_weights)
                ess_threshold_actual = self.ess_threshold * self.num_samples
                
                # I think there's an issue with weight eq since i resample every step
                # maybe some lower step size runs would be good
                if ess < ess_threshold_actual:
                    particles, log_weights = self._resample_particles(
                        new_particles, log_weights
                    )
                    self._resampling_counts.append(iteration)
                else:
                    particles = new_particles
                
                # Store for diagnostics
                particle_history.append([{k: v.clone() for k, v in p.items()} for p in particles])
                weight_history.append(log_weights.clone())
                
                # Call hook function if provided
                if self.hook_fn is not None:
                    self.hook_fn(self.kernels[0], particles, "sample", iteration)
            
            # Convert final particles to batched format for output
            final_samples = {}
            if particles:
                # Get parameter names from first particle
                param_names = particles[0].keys()
                for name in param_names:
                    # Stack all particles for this parameter
                    param_values = torch.stack([p[name] for p in particles])
                    final_samples[name] = param_values
            
            # Store final results
            self._samples = final_samples
            self._log_weights = log_weights
            self._weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
            
            # Store full history for diagnostics
            self._particle_history = particle_history
            self._weight_history = weight_history
            
            # Set transforms
            if self.transforms is None:
                self._set_transforms(*args, **kwargs)

            # Transform samples back to constrained space
            for name, z in self._samples.items():
                if name in self.transforms:
                    self._samples[name] = self.transforms[name].inv(z)
    
    def get_samples(self, num_samples: Optional[int] = None, group_by_chain: bool = False):
        """Get samples from the SMC run."""
        if self._samples is None:
            raise RuntimeError("No samples available. Run SMC first.")
        
        if num_samples is None:
            return self._samples
        
        # Resample according to final weights
        indices = torch.multinomial(self._weights, num_samples, replacement=True)
        
        resampled = {}
        for name, values in self._samples.items():
            resampled[name] = values[indices]
        
        return resampled
    
    def get_weighted_samples(self):
        """Get samples with their corresponding weights."""
        if self._samples is None or self._weights is None:
            raise RuntimeError("No samples available. Run SMC first.")
        
        return self._samples, self._weights
    
    def diagnostics(self):
        """Gets diagnostics statistics from the SMC run."""
        if self._samples is None:
            raise RuntimeError("No samples available. Run SMC first.")
        
        # Basic diagnostics
        final_ess = self._compute_ess(self._log_weights)
        
        diag = {
            'final_ess': final_ess,
            'effective_sample_size_ratio': final_ess / self.num_samples,
            'num_resampling_steps': len(self._resampling_counts),
            'resampling_iterations': self._resampling_counts,
        }
        
        # Add standard MCMC-style diagnostics for the final samples
        mcmc_diag = diagnostics({k: v.unsqueeze(0) for k, v in self._samples.items()})
        diag.update(mcmc_diag)
        
        return diag

    def summary(self, prob: float = 0.9):
        """Prints a summary table displaying diagnostics."""
        if self._samples is None:
            raise RuntimeError("No samples available. Run SMC first.")
        
        print("SMC Summary:")
        print(f"Number of particles: {self.num_samples}")
        print(f"Number of iterations: {self.num_iters}")
        print(f"Final ESS: {self._compute_ess(self._log_weights):.2f}")
        print(f"Number of resampling steps: {len(self._resampling_counts)}")
        print()
        
        # Print parameter summary using final particles
        print_summary({k: v.unsqueeze(0) for k, v in self._samples.items()}, prob=prob)


def _test_nuts():
    import pyro.distributions as dist
    from bosmc.smc.nuts import NUTSSMCKernel

    true_coefs = torch.tensor([1., 2., 3.])
    data = torch.randn(2000, 3)
    dim = 3
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data, idx=1):
        coefs_mean = torch.zeros(dim)
        name = f"beta_{idx}" if idx is not None else "beta"
        coefs = pyro.sample(name, dist.Normal(coefs_mean, torch.ones(dim)))
        y = pyro.sample("y", dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y
    
    print("Running SMC...")
    nuts = NUTSSMCKernel(model, step_size=0.3, max_tree_depth=20)
    smc = SMC(nuts, num_samples=100, num_iters=100, ess_threshold=0.5)
    smc.run(data)
    
    print("\nTrue coefficients:", true_coefs)
    print("SMC estimate:", smc.get_samples()['beta'].mean(0))
    smc.summary()
    
    return

def _test_rw():
    import pyro.distributions as dist
    from bosmc.smc.rw import RandomWalkSMCKernel

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
    rw_smc_kernel = RandomWalkSMCKernel(model)
    smc = SMC(rw_smc_kernel, num_samples=200, num_iters=100, ess_threshold=0.5)
    smc.run(data)
    
    print("\nTrue coefficients:", true_coefs)
    print("SMC estimate:", smc.get_samples()['beta'].mean(0))
    smc.summary()
    
    return

if __name__ == '__main__':
    _test_nuts()