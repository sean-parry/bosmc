# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
from bosmc.smc.smc_kernel import SMCKernel
from pyro.infer.mcmc.util import initialize_model


class RandomWalkSMCKernel(SMCKernel):
    r"""
    :param model: Python callable containing Pyro primitives.
    :param float init_step_size: A positive float that controls the initial step size. Defaults to 0.1.
    """

    def __init__(self, 
                 model, 
                 init_step_size: float = 0.1,
                 ) -> None:
        if not isinstance(init_step_size, float) or init_step_size <= 0.0:
            raise ValueError("init_step_size must be a positive float.")
        
        self.model = model
        self.init_step_size = init_step_size
        self._t = 0
        self._log_step_size = math.log(init_step_size)
        self._accept_cnt = 0
        self._energy_last = None
        
        # These will be set during setup
        self._initial_params = None
        self.potential_fn = None
        self.transforms = None
        self._prototype_trace = None
        
        super().__init__()

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
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
        # Initialize energy for this kernel
        self._energy_last = self.potential_fn(self._initial_params)

    def sample_initial(self, *args, **kwargs):
        """Sample initial parameters from the prior."""
        if self._initial_params is None:
            raise RuntimeError("Kernel not properly initialized. Call setup() first.")
        return self._initial_params.copy()

    def sample(self, params):
        """
        Sample new parameters and compute incremental log weight.
        
        Args:
            params: Current parameters
            log_weight: Current log weight (not used in computation, just passed through context)
            
        Returns:
            new_params: New parameter values
            incremental_log_weight: Log weight increment for this transition
        """
        if self.potential_fn is None:
            raise RuntimeError("Kernel not properly initialized. Call setup() first.")
            
        step_size = math.exp(self._log_step_size)
        
        # Current energy
        energy_current = self.potential_fn(params)
        
        # Propose new parameters
        new_params = {}
        for k, v in params.items():
            new_params[k] = v + step_size * torch.randn(v.shape, dtype=v.dtype, device=v.device)
        
        # New energy
        energy_proposal = self.potential_fn(new_params)
        
        # Weight update: exp(-energy_new) / exp(-energy_old) = exp(energy_old - energy_new)
        incremental_log_weight = energy_current - energy_proposal
        
        # Update stored energy for this kernel
        self._energy_last = energy_proposal
        
        # Adaptation during warmup
        if self._t <= self._warmup_steps:
            adaptation_speed = max(0.001, 0.1 / math.sqrt(1 + self._t))
            # Simple step size adaptation (increase step size over time)
            self._log_step_size += adaptation_speed * 0.1
        
        self._t += 1
        
        return new_params, incremental_log_weight

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def logging(self):
        return {"step_size": "{:.2e}".format(math.exp(self._log_step_size))}

    def diagnostics(self):
        return {"step_size": math.exp(self._log_step_size), "iteration": self._t}