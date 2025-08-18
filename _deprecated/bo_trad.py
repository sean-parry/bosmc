import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from tests.target_functions import BaseTarget, Branin

from tqdm import tqdm

class TradOpt():
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
        self.y_train = y.reshape(-1,1)
        print(self.X_train, self.y_train)
        return
        
    
    def bo_iters(self,
                 n_iters: int,
                 ) -> None:
        for _ in tqdm(range(n_iters)):
            gp = SingleTaskGP(train_X=self.X_train,
                              train_Y=self.y_train,
                              input_transform=Normalize(d=self.target.dim),
                              outcome_transform=Standardize(m=1),)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            #print(gp)
            #print(self.y_train.max())
            logEI = LogExpectedImprovement(model=gp, best_f=self.y_train.max())
            x_star, acq_val = optimize_acqf(logEI, bounds=self.target.bounds, q=1, num_restarts=5, raw_samples=20)
            x_star = x_star
            y_star = self.target.sample(x_star[0])
            self.X_train = torch.cat((self.X_train, x_star))
            self.y_train = torch.cat((self.y_train, y_star.reshape((-1,1))))
        return


def main() -> None:
    target = Branin()
    bo = TradOpt(target = target)
    bo.random_eval_iters(n_iters=3)
    bo.bo_iters(100)
    print(target.regret_arr)
    return

if __name__ == '__main__':
    main()