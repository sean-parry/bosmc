import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import pyro
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    Warp,
)
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.transforms.utils import kumaraswamy_warp, subset_transform
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
from botorch.models.fully_bayesian import PyroModel, SaasPyroModel, SaasFullyBayesianSingleTaskGP
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import LinearKernel, MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import dist, Kernel
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.models.exact_gp import ExactGP
from pyro.ops.integrator import register_exception_handler
from torch import Tensor


class WeightedGaussianMixturePosterior(GaussianMixturePosterior):
    r"""A weighted Gaussian mixture posterior.
    
    Extends GaussianMixturePosterior to support weighted mixture statistics
    using SMC weights or other importance weights.
    """
    
    def __init__(self, distribution: MultivariateNormal, weights: Optional[Tensor] = None) -> None:
        r"""A weighted posterior for a fully Bayesian model.
        
        Args:
            distribution: A GPyTorch MultivariateNormal (single-output case)
            weights: Optional weights for each MCMC sample. If None, uses uniform weights.
        """
        super().__init__(distribution=distribution)
        
        # Handle weights
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=distribution.mean.dtype, device=distribution.mean.device)
            else:
                weights = weights.to(dtype=distribution.mean.dtype, device=distribution.mean.device)
            
            # Normalize weights to sum to 1
            self.weights = weights / weights.sum()
        else:
            # Use uniform weights
            num_samples = self._mean.shape[MCMC_DIM]
            self.weights = torch.ones(num_samples, dtype=distribution.mean.dtype, device=distribution.mean.device) / num_samples
    
    @property
    def mixture_mean(self) -> Tensor:
        r"""The weighted posterior mean for the mixture of models."""
        if self._mixture_mean is None:
            # Expand weights to match the shape needed for broadcasting
            weight_shape = [1] * len(self._mean.shape)
            weight_shape[MCMC_DIM] = len(self.weights)
            expanded_weights = self.weights.view(*weight_shape)
            
            # Weighted mean
            self._mixture_mean = (self._mean * expanded_weights).sum(dim=MCMC_DIM)
        return self._mixture_mean
    
    @property
    def mixture_variance(self) -> Tensor:
        r"""The weighted posterior variance for the mixture of models."""
        if self._mixture_variance is None:
            # Expand weights to match the shape needed for broadcasting
            weight_shape = [1] * len(self._mean.shape)
            weight_shape[MCMC_DIM] = len(self.weights)
            expanded_weights = self.weights.view(*weight_shape)
            
            # Weighted variance using law of total variance for mixtures
            # Var[X] = E[Var[X|component]] + Var[E[X|component]]
            
            # E[Var[X|component]] - weighted average of variances
            weighted_variance_term = (self._variance * expanded_weights).sum(dim=MCMC_DIM)
            
            # Var[E[X|component]] - variance of weighted means
            weighted_mean_sq = (self._mean.pow(2) * expanded_weights).sum(dim=MCMC_DIM)
            mean_variance_term = weighted_mean_sq - self.mixture_mean.pow(2)
            
            self._mixture_variance = weighted_variance_term + mean_variance_term
        return self._mixture_variance
    
    @property
    def mixture_covariance_matrix(self) -> Tensor:
        r"""The weighted posterior covariance matrix for the mixture of models."""
        if self._mixture_covariance_matrix is None:
            # Expand weights to match the shape needed for broadcasting
            weight_shape = [1] * len(self._mean.shape)
            weight_shape[MCMC_DIM] = len(self.weights)
            expanded_weights = self.weights.view(*weight_shape)
            
            # Weighted average of covariance matrices
            weighted_cov_term = (self._covariance_matrix * expanded_weights.unsqueeze(-1)).sum(dim=MCMC_DIM)
            
            # Covariance due to mean differences (between-component covariance)
            mean_diff = self._mean - self.mixture_mean.unsqueeze(MCMC_DIM)
            weighted_outer_products = (
                torch.matmul(mean_diff, mean_diff.transpose(-1, -2)) * 
                expanded_weights.unsqueeze(-1)
            ).sum(dim=MCMC_DIM)
            
            self._mixture_covariance_matrix = weighted_cov_term + weighted_outer_products
        return self._mixture_covariance_matrix


class SMCFullyBayesianSingleTaskGP(SaasFullyBayesianSingleTaskGP):
    """
    A fully Bayesian single-task GP model that uses SMC samples with proper weighting.
    
    This model uses weighted samples from Sequential Monte Carlo to create a proper
    Bayesian mixture posterior that accounts for particle weights.
    
    Example:
        >>> smc_gp = SMCFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> # Run SMC sampling
        >>> samples, weights = smc.get_weighted_samples()
        >>> smc_gp.load_smc_samples(samples, weights)
        >>> posterior = smc_gp.posterior(test_X)
    """

    _is_fully_bayesian = True
    _is_ensemble = True
    _pyro_model_class: type[PyroModel] = SaasPyroModel
    smc_weights: torch.Tensor = None
    
    def _check_if_fitted(self) -> None:
        """Check if model has been fitted with SMC samples."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`load_smc_samples` with SMC samples and weights."
            )
    
    def load_smc_samples(self, smc_samples: Dict[str, Tensor], weights: Tensor) -> None:
        """
        Load SMC samples and weights into the model.
        
        Args:
            smc_samples: Dictionary of parameter samples from SMC
            weights: Normalized weights for each particle/sample
        """
        self.smc_weights = weights

        print(self.pyro_model.load_mcmc_samples(mcmc_samples=smc_samples))

        # if unpacking issues occurs 3 to 4 then add 'input_transform' to the tuple
        (self.mean_module, self.covar_module, self.likelihood, input_transform) = (
            self.pyro_model.load_mcmc_samples(mcmc_samples=smc_samples)
        )
        if input_transform is not None:
            if hasattr(self, "input_transform"):
                tfs = [self.input_transform]
                if isinstance(input_transform, ChainedInputTransform):
                    tfs.extend(list(input_transform.values()))
                else:
                    tfs.append(input_transform)
                self.input_transform = ChainedInputTransform(
                    **{f"tf{i}": tf for i, tf in enumerate(tfs)}
                )
            else:
                self.input_transform = input_transform

    def postprocess_smc_samples(
        self, smc_samples: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        r"""Post-process the SMC Samples."""
        inv_length_sq = (
            smc_samples["kernel_tausq"].unsqueeze(-1)
            * smc_samples["_kernel_inv_length_sq"]
        )
        smc_samples["lengthscale"] = inv_length_sq.rsqrt()
        # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
        # into the final model.
        del smc_samples["kernel_tausq"], smc_samples["_kernel_inv_length_sq"]
        return smc_samples
    
    def set_weigths(self, weights: torch.Tensor) -> None:
        self.smc_weights = weights
    
    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Forward pass through the batched GP model.
        
        Returns unweighted multivariate normal predictor
        """
        self._check_if_fitted()
        
        # Transform inputs
        X = self.transform_inputs(X)
        
        # Add batch dimension if needed
        if X.ndim == 2:
            X = X.unsqueeze(0).expand(len(self.smc_weights), -1, -1)
            
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)
    
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> WeightedGaussianMixturePosterior:
        """
        Compute weighted posterior over model outputs.
        
        Args:
            X: Test points (batch_shape x q x d)
            output_indices: Output indices to compute posterior over
            observation_noise: Whether to include observation noise
            posterior_transform: Optional posterior transform
            
        Returns:
            WeightedGaussianMixturePosterior
            functionally the same as GaussianMixturePosterior
        """
        self._check_if_fitted()
        
        # Transform inputs
        X = self.transform_inputs(X)
        
        # Add batch dimension for each SMC sample
        if X.ndim == 2:
            X = X.unsqueeze(MCMC_DIM)  # Add MCMC dimension
            X = X.expand(len(self.smc_weights), -1, -1)
            
        # Get forward prediction
        with torch.no_grad():
            mvn = self(X)
            if observation_noise:
                mvn = self.likelihood(mvn)
            
        # Apply posterior transform if provided
        if posterior_transform is not None:
            mvn = posterior_transform(mvn)
            
        # Return weighted mixture posterior
        return WeightedGaussianMixturePosterior(
            distribution=mvn,
            weights=self.smc_weights,
        )
    
    @property 
    def batch_shape(self) -> torch.Size:
        """Batch shape of the model."""
        if self.smc_weights is not None:
            return torch.Size([len(self.smc_weights)])
        return torch.Size([])
    
    def train(self, mode: bool = True):
        """Put model in train mode."""
        if hasattr(self, 'training'):
            self.training = mode
        if mode:
            # Reset fitted components when entering train mode
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None
            self.smc_weights = None
        return self
    
    def eval(self):
        """Put model in eval mode."""
        return self.train(mode=False)
    
    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> "SMCFullyBayesianSingleTaskGP":
        """Condition model on additional observations."""
        # This would need to be implemented properly for your use case
        # For now, create a new model with combined data
        if hasattr(self, 'train_X_original'):
            combined_X = torch.cat([self.train_X_original, X], dim=0)
            combined_Y = torch.cat([self.train_Y_original, Y], dim=0)
            
            new_model = self.__class__(
                train_X=combined_X,
                train_Y=combined_Y,
                train_Yvar=None,  # You may need to handle this properly
                outcome_transform=getattr(self, 'outcome_transform', None),
                input_transform=getattr(self, 'input_transform', None),
                kernel_type=self.kernel_type
            )
            
            # If this model is fitted, you'd need to handle transferring samples
            if self.smc_weights is not None:
                # This is a simplified approach - you might want to rerun SMC
                print("Warning: conditioning on observations with fitted SMC model requires re-running SMC")
            
            return new_model
        else:
            raise RuntimeError("Cannot condition model - original training data not available")
