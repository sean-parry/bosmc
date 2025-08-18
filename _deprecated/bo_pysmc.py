import numpy as np
import numpy.typing as npt

import torch
import gpytorch
import botorch
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction

from gpytorch.priors import UniformPrior

import pyro

from tests.target_functions import BaseTarget, Branin

from tqdm import tqdm

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 likelihood: gpytorch.likelihoods._GaussianLikelihoodBase = gpytorch.likelihoods.GaussianLikelihood()):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.MaternKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def set_hyperparameters(self,
                            outputscale: torch.Tensor,
                            lengthscale: torch.Tensor
                            ) -> None:
        self.base_kernel.lengthscale = torch.tensor([[lengthscale]])
        self.covar_module.outputscale = outputscale

    def forward(self, 
                x: torch.Tensor,
                ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class WeightedAcq(AcquisitionFunction):
    def __init__(self, models, weights, best_f):
        super().__init__(model=None) # don't need model
        self.acq_fns = [LogExpectedImprovement(model=model, best_f=best_f) for model in models]
        self.weights = weights

    def forward(self, X):
        acq_vals = torch.tensor([acq_fn.forward(X) for acq_fn in self.acq_fns])
        return torch.sum(acq_vals * self.weights)


class Target_SMC():
    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(X_train = X_train,
                                  y_train = y_train,
                                  likelihood = self.likelihood)
        return
    
    def mll(self,
            x: torch.Tensor
            ) -> np.float64:
        self.model.set_hyperparameters(outputscale=x[0], lengthscale=x[1])

        self.model.train()
        self.likelihood.train()
        
        mll_fn = gpytorch.ExactMarginalLogLikelihood(self.likelihood, self.model)
        output = self.model(self.X_train)
        mll = mll_fn(output, self.y_train)
        return mll.item()
    
    def mll_grad(self,
                 x: torch.Tensor,
                 ) -> npt.NDArray[np.float64]:
        self.model.set_hyperparameters(outputscale=x[0], lengthscale=x[1])

        self.model.train()
        self.likelihood.train()
        
        mll_fn = gpytorch.ExactMarginalLogLikelihood(self.likelihood, self.model)
        output = self.model(self.X_train)
        loss = -1*mll_fn(output, self.y_train)

        loss.backward()

        # feels like a terrible way of getting the params for some reason .named_params() returns something weird though
        param_names = ['covar_module.raw_outputscale', 'base_kernel.raw_lengthscale']
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param

        #print(f'the grads are {np.array([param_dict[name].grad.norm().item() for name in param_names])}')
        return np.array([param_dict[name].grad.item() for name in param_names])


    def logpdf(self,
               X: npt.NDArray[np.float64],
               phi: float = 0.0
               ) -> npt.NDArray[np.float64]:
        X = torch.tensor(data = X, dtype=torch.float64)
        mlls = []
        for x in X:
            mlls.append(self.mll(x))
        return np.array(mlls)

    def logpdfgrad(self,
                   X: npt.NDArray[np.float64],
                   phi: float = 0.0
                   ) -> npt.NDArray[np.float64]:
        X = torch.tensor(data = X, dtype=torch.float64)
        mll_grads = []
        for x in X:
            mll_grads.append(self.mll_grad(x))
        return np.array(mll_grads)


class BO_SMC():
    def __init__(self,
                 target: BaseTarget,
                 X_train: torch.Tensor = None,
                 y_train: torch.Tensor = None,
                 random_seed: int = None,
                 ) -> None:
        """
        Inputs,
        target, inherits from BaseTarget (required)\n
        X_train, y_train torch tensors of data if None ensure at least
        3 random evals are performed with self.random_eval_iters before running
        bo_iters\n
        random_seed used to set torch.Generator seed in random evals\n
        """
        self.target = target
        self.X_train = X_train
        self.y_train = y_train
        self.random_generator = (torch.Generator().manual_seed(random_seed) if random_seed is not None
                                 else torch.Generator().manual_seed(torch.seed()))
        return
    
    def random_eval_iters(self,
                          n_iters: int,
                          ) -> None:
        """
        Adds random evaluations of the target, to the dataset (X_train, y_train),
        Only usable for X_train and y_train = None
        """
        assert self.X_train is None and self.y_train is None
        rand_vals = torch.rand(size = (n_iters, self.target.dim), 
                               generator=self.random_generator)
        X = self.target.bounds[0] + (self.target.bounds[1]-self.target.bounds[0]) * rand_vals
        y = torch.tensor([self.target.sample(x) for x in X])
        self.X_train = X
        self.y_train = y.reshape(1,-1)
        return
        
    def run_smc(self, 
                n_samples: int, 
                n_iters: int,
                ) -> tuple[list[Model], torch.Tensor]:
        """
        returns a list of models and the weights of each model 
        to be used in a weighted acquisition funciton
        """

        """SingleTaskGP(train_X=self.X_train,
                              train_Y=self.y_train,
                              input_transform=Normalize(d=self.target.dim),
                              outcome_transform=Standardize(m=1),)"""
        return

    def bo_iters(self,
                 n_iters: int,
                 ) -> None:
        for _ in tqdm(range(n_iters)):
            models, weights = self.run_smc()
            best_f = self.y_train.max()
            acq_fn = WeightedAcq(models=models, weights=weights, best_f=best_f)
            x_star, acq_val = optimize_acqf(acq_fn, bounds=self.target.bounds, q=1, num_restarts=5, raw_samples=20)
            x_star = x_star
            y_star = self.target.sample(x_star[0])
            self.X_train = torch.cat((self.X_train, x_star))
            self.y_train = torch.cat((self.y_train, y_star))
        return


def test_model() -> None:
    import tests.utils as test_utils
    X_train, y_train = test_utils.dummy_data(n_iters = 20, seed = None)
    y_train = y_train.squeeze(-1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    model.set_hyperparameters(outputscale = torch.tensor(1.0),
                              lengthscale = torch.tensor([[1.0]]))

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    output = model(X_train)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = -1*mll(output, y_train)

    #optimizer.zero_grad()

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'{name}: has value {param.data} and gradient norm = {param.grad.norm().item()})')
        else:
            print(f'{name}: no gradient')
    return


def test_smc_target():
    import tests.utils as test_utils
    X_train, y_train = test_utils.dummy_data(n_iters = 20, seed = None)
    y_train = y_train.squeeze(-1)
    target = Target_SMC(X_train, y_train)
    X = np.array([[1.0, 1.0],[2.0, 2.0],[0.1, 0.1]])
    logpdfs = target.logpdfgrad(X)
    print(logpdfs)


def main():
    test_smc_target()
    #test_model()

    return

if __name__ == '__main__':
    main()