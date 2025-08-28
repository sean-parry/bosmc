# provides function comparing smc, mcmc and ADAM
from tests.target_functions.base import BaseTarget
from tqdm import tqdm

import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from bosmc.fit import fit_fully_bayesian_model_nuts_smc
from bosmc.models import SaaSSMCFullyBayesianSingleTaskGP


class Dataset():
    X: torch.Tensor | None = None
    y: torch.Tensor | None = None
    def __init__(
            self, 
            target: BaseTarget,
        ) -> None:
        self.target = target
        return

    def random_evals(
            self, 
            seed: int, 
            n_iters: int
        ) -> None:
        assert self.x is None and self.y is None
        random_gen = torch.Generator().manual_seed(seed)
        rand_vals = torch.rand(size = (n_iters, self.target.dim), 
                               generator=random_gen)
        X = self.target.bounds[0] + (self.target.bounds[1]-self.target.bounds[0]) * rand_vals
        y = torch.tensor([self.target.sample(x) for x in X])
        self.X = X
        self.y = y.reshape(-1, 1)

    def eval_x(
            self,
            x_star: torch.Tensor,
    ) -> None:
        y_star = self.target.sample(x_star)
        self.X = torch.cat((self.X, x_star))
        self.y = torch.cat((self.y, y_star.reshape((-1,1))))
        return
    

def trad_loop(
        target: BaseTarget,
        seed: int,
        n_random_evals: int,
        n_bo_evals: int,
    ) -> None:
    assert target.num_evals == 0, 'target must not have been evaluated'
    save_name = f'trad_{target.target_name}_{seed}_{n_random_evals}_{n_bo_evals}'
    dataset = Dataset(target)
    dataset.random_evals(seed, n_random_evals)

    for _ in tqdm(range(n_bo_evals)):
        gp = SingleTaskGP(train_X=dataset.X,
                          train_Y=dataset.y,
                          input_transform=Normalize(d=target.dim),
                          outcome_transform=Standardize(m=1),)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        logEI = LogExpectedImprovement(model=gp, best_f=dataset.y.max())
        x_star, acq_val = optimize_acqf(logEI, bounds=target.bounds, q=1, num_restarts=5, raw_samples=20)
        x_star = x_star[0]
        dataset.eval_x(x_star)

def mcmc_loop(
        target: BaseTarget,
        seed: int,
        n_random_evals: int,
        n_bo_evals: int,
        warm_up_steps: int,
        num_samples: int,
        thinning: int,
    ) -> None:
    save_name = f'mcmc_{target.target_name}_{seed}_{n_random_evals}_{n_bo_evals}'
    dataset = Dataset(target)
    dataset.random_evals(seed, n_random_evals)

    for _ in tqdm(range(n_bo_evals)):
        model = SaasFullyBayesianSingleTaskGP(
            train_X=dataset.X,
            train_Y=dataset.y,
            input_transform=Normalize(d=target.dim),
            outcome_transform=Standardize(m=1),)
        fit_fully_bayesian_model_nuts(
            model = model,
            warmup_steps=warm_up_steps,
            num_samples=num_samples,
            thinning=thinning,
            disable_progbar=True,
        )
        logEI = LogExpectedImprovement(model=model.posterior, best_f=dataset.y.max())
        
        x_star, acq_val = optimize_acqf(logEI, bounds=target.bounds, q=1, num_restarts=5, raw_samples=20)
        x_star = x_star[0]
        dataset.eval_x(x_star)

    
def smc_loop(
        target: BaseTarget,
        seed: int,
        n_random_evals: int,
        n_bo_evals: int,
        warm_up_steps: int,
        num_samples: int,
    ) -> None:
    save_name = f'smc_{target.target_name}_{seed}_{n_random_evals}_{n_bo_evals}'
    dataset = Dataset(target)
    dataset.random_evals(seed, n_random_evals)

    for _ in tqdm(range(n_bo_evals)):
        model = SaaSSMCFullyBayesianSingleTaskGP(
            train_X=dataset.X,
            train_Y=dataset.y,
            input_transform=Normalize(d=target.dim),
            outcome_transform=Standardize(m=1),)
        fit_fully_bayesian_model_nuts_smc(
            model = model,
            warmup_steps=warm_up_steps,
            num_samples=num_samples,
            disable_progbar=True,
        )
        logEI = LogExpectedImprovement(model=model.posterior, best_f=dataset.y.max())
        
        x_star, acq_val = optimize_acqf(logEI, bounds=target.bounds, q=1, num_restarts=5, raw_samples=20)
        x_star = x_star[0]
        dataset.eval_x(x_star)

def _benchmark_test():
    seed = 30
    

    return

if __name__ == '__main__':
    trad_loop()