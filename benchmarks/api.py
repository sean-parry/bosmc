# provides function comparing smc, mcmc and ADAM
from tests.target_functions.base import BaseTarget
from tqdm import tqdm

import pickle
import os

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

DATA_DEVICE = torch.device("cpu") # pyro is quicker with the data on the cpu
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # but the model on the gpu

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
        assert self.X is None and self.y is None
        random_gen = torch.Generator(device=DATA_DEVICE).manual_seed(seed)
        rand_vals = torch.rand(size = (n_iters, self.target.dim), 
                               generator=random_gen,
                               device=DATA_DEVICE)
        a = self.target.bounds[0].to(DATA_DEVICE)
        b = self.target.bounds[1].to(DATA_DEVICE)
        X = (a + (b-a)) * rand_vals
        y = torch.tensor([self.target.sample(x) for x in X], device=DATA_DEVICE)
        self.X = X
        self.y = y.reshape(-1, 1)

    def eval_x(
            self,
            x_star: torch.Tensor,
    ) -> None:
        y_star = self.target.sample(x_star).to(DATA_DEVICE)
        self.X = torch.cat((self.X, x_star.unsqueeze(0)))
        self.y = torch.cat((self.y, y_star.reshape((-1,1))))
        return
    

def trad_loop(
        target: BaseTarget,
        seed: int,
        n_random_evals: int,
        n_bo_evals: int,
        disable_prog_bar: bool = True,
    ) -> str:
    assert target.num_evals == 0, 'target must not have been evaluated'

    save_name = f'trad_{target.target_name}_{seed}_{n_random_evals}_{n_bo_evals}.pkl'
    if os.path.exists(save_name):
        return save_name
    
    dataset = Dataset(target)
    dataset.random_evals(seed, n_random_evals)

    for _ in tqdm(range(n_bo_evals), disable = disable_prog_bar):
        gp = SingleTaskGP(
            train_X=dataset.X,
            train_Y=dataset.y,
            input_transform=Normalize(d=target.dim),
            outcome_transform=Standardize(m=1),
        )
        
        gp.to(MODEL_DEVICE)

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        logEI = LogExpectedImprovement(model=gp, best_f=dataset.y.max())
        x_star, acq_val = optimize_acqf(logEI, bounds=target.bounds, q=1, num_restarts=5, raw_samples=20)
        x_star = x_star[0].to(DATA_DEVICE)
        dataset.eval_x(x_star)

    results = target.get_results()
    with open(save_name, "wb") as f:
        pickle.dump(results, f)

    return save_name

def mcmc_loop(
        target: BaseTarget,
        seed: int,
        n_random_evals: int,
        n_bo_evals: int,
        warm_up_steps: int,
        num_samples: int,
        thinning: int,
        disable_prog_bar: bool = True,
    ) -> str:
    assert target.num_evals == 0, 'target must not have been evaluated'

    save_name = f'mcmc_{target.target_name}_{seed}_{n_random_evals}_{n_bo_evals}.pkl'
    if os.path.exists(save_name):
        return save_name
    
    dataset = Dataset(target)
    dataset.random_evals(seed, n_random_evals)

    for _ in tqdm(range(n_bo_evals), disable = disable_prog_bar):
        model = SaasFullyBayesianSingleTaskGP(
            train_X=dataset.X,
            train_Y=dataset.y,
            input_transform=Normalize(d=target.dim),
            outcome_transform=Standardize(m=1),
        )
        
        model.to(MODEL_DEVICE)

        fit_fully_bayesian_model_nuts(
            model = model,
            warmup_steps=warm_up_steps,
            num_samples=num_samples,
            thinning=thinning,
            disable_progbar=True,
        )
        logEI = LogExpectedImprovement(model=model, best_f=dataset.y.max())
        
        x_star, acq_val = optimize_acqf(logEI, bounds=target.bounds, q=1, num_restarts=5, raw_samples=20)
        x_star = x_star[0].to(DATA_DEVICE)
        dataset.eval_x(x_star)

    results = target.get_results()
    with open(save_name, "wb") as f:
        pickle.dump(results, f)

    return save_name

    
def smc_loop(
        target: BaseTarget,
        seed: int,
        n_random_evals: int,
        n_bo_evals: int,
        warm_up_steps: int,
        num_samples: int,
        disable_prog_bar: bool = True,
    ) -> str:
    assert target.num_evals == 0, 'target must not have been evaluated'

    save_name = f'smc_{target.target_name}_{seed}_{n_random_evals}_{n_bo_evals}.pkl'
    if os.path.exists(save_name):
        return save_name
    
    dataset = Dataset(target)
    dataset.random_evals(seed, n_random_evals)

    for _ in tqdm(range(n_bo_evals), disable = disable_prog_bar):

        model = SaaSSMCFullyBayesianSingleTaskGP(
            train_X=dataset.X,
            train_Y=dataset.y,
            input_transform=Normalize(d=target.dim),
            outcome_transform=Standardize(m=1),
        )

        model.to(MODEL_DEVICE)
        
        fit_fully_bayesian_model_nuts_smc(
            model = model,
            num_iters=warm_up_steps,
            num_samples=num_samples,
            disable_progbar=False,
        )
        logEI = LogExpectedImprovement(model=model, best_f=dataset.y.max())
        
        x_star, acq_val = optimize_acqf(logEI, bounds=target.bounds, q=1, num_restarts=5, raw_samples=20)
        x_star = x_star[0].to(DATA_DEVICE)
        dataset.eval_x(x_star)

    results = target.get_results()
    with open(save_name, "wb") as f:
        pickle.dump(results, f)

    return save_name

def run_benchmarks_for_trad():
    RANDOM_EVALS = 3
    BO_EVLAS = 97
    from tests.target_functions.branin import Branin
    target_type = Branin

    for i in tqdm(range(32)):
        trad_loop(
            target=target_type(),
            seed = i,
            n_random_evals = RANDOM_EVALS,
            n_bo_evals = BO_EVLAS
        )

def run_benchmarks_for_mcmc():
    RANDOM_EVALS = 3
    BO_EVLAS = 97
    WARMUP_STEPS = 128
    NUM_SAMPLES = 128
    from tests.target_functions.branin import Branin
    target_type = Branin

    for i in tqdm(range(32)):
        mcmc_loop(
            target=target_type(),
            seed = i,
            n_random_evals = RANDOM_EVALS,
            n_bo_evals = BO_EVLAS,
            warm_up_steps= WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=1,
            disable_prog_bar = False,
        )

def run_benchmarks_for_smc():
    RANDOM_EVALS = 3
    BO_EVLAS = 97
    NUM_ITERS = 128
    NUM_SAMPLES = 128
    from tests.target_functions.branin import Branin
    target_type = Branin

    for i in tqdm(range(32)):
        smc_loop(
            target=target_type(),
            seed = i,
            n_random_evals = RANDOM_EVALS,
            n_bo_evals = BO_EVLAS,
            warm_up_steps = NUM_ITERS,
            num_samples = NUM_SAMPLES,
            disable_prog_bar = True,
        )
        

def run_benchmarks_for_smc_seed_list(rand_vals: list[int]):
    RANDOM_EVALS = 3
    BO_EVLAS = 97
    NUM_ITERS = 128
    NUM_SAMPLES = 128
    from tests.target_functions.branin import Branin
    target_type = Branin

    for rand_val in rand_vals:
        print(rand_val)
        smc_loop(
            target=target_type(),
            seed = rand_val,
            n_random_evals = RANDOM_EVALS,
            n_bo_evals = BO_EVLAS,
            warm_up_steps = NUM_ITERS,
            num_samples = NUM_SAMPLES,
            disable_prog_bar = False,
        )




def _benchmark_test():
    from tests.target_functions.branin import Branin
    seed = 30
    target_type = Branin

    file_name = trad_loop(
        target=target_type(),
        seed = seed,
        n_random_evals= 5,
        n_bo_evals = 10
    )

    with open(file_name, "rb") as f:
        res: dict = pickle.load(f)

    for key, value in res.items():
        print(f"{key}:{type(value)} of type {type(value[0])}")


    

    return

def main():
    run_benchmarks_for_smc()

if __name__ == '__main__':
    main()