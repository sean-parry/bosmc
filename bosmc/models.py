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
    
    def __init__(
            self, 
            distribution: MultivariateNormal, 
            weights: Optional[Tensor] = None
        ) -> None:
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


class SaaSSMCFullyBayesianSingleTaskGP(SaasFullyBayesianSingleTaskGP):
    """A fully Bayesian GP model that uses an SMC approach."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        """SaaSSMCFullyBayesianSingleTaskGP.

        Args:
            train_X: Training inputs.
            train_Y: Training targets.
            outcome_transform: The outcome transform.
            input_transform: The input transform.
            train_Yvar: Optional training observation variance.
        """
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            train_Yvar=train_Yvar,
        )
        self.smc_weights = None  # Add a property to store the SMC weights

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> WeightedGaussianMixturePosterior:
        """Computes the posterior using the SMC samples and weights.

        Args:
            X: The test inputs.
            output_indices: The output indices.
        """
        distribution = super().posterior(X, output_indices).distribution
        return WeightedGaussianMixturePosterior(distribution=distribution, weights=self.smc_weights)

    def load_smc_samples(self, smc_samples: Dict[str, Tensor], weights: Tensor) -> None:
        """Loads SMC samples and their corresponding weights into the model.

        Args:
            smc_samples: A dictionary of SMC samples.
            weights: A tensor of SMC weights.
        """
        # Call the parent class's method to load the samples
        super().load_mcmc_samples(smc_samples)
        # Store the weights separately for the custom posterior
        self.set_weights(weights)

    def set_weights(self, weights: Tensor) -> None:
        """Sets the SMC weights for the model.

        Args:
            weights: A tensor of SMC weights.
        """
        self.smc_weights = weights

    def train(self, mode: bool = True) -> "SaaSSMCFullyBayesianSingleTaskGP":
        """Put model in train mode."""
        return super().train(mode=mode)

    def eval(self) -> "SaaSSMCFullyBayesianSingleTaskGP":
        """Put model in eval mode."""
        return self.train(mode=False)
    
    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> "SaaSSMCFullyBayesianSingleTaskGP":
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
            )
            
            # If this model is fitted, you'd need to handle transferring samples
            if self.smc_weights is not None:
                # This is a simplified approach - you might want to rerun SMC
                print("Warning: conditioning on observations with fitted SMC model requires re-running SMC")
            
            return new_model
        else:
            raise RuntimeError("Cannot condition model - original training data not found.")

