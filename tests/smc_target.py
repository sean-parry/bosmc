from bosmc.bo_smc import Target_SMC
from bosmc.target_function.branin import Branin
from bosmc.target_function.base import BaseTarget

import torch

def dummy_data(n_iters: int = 20,
               target: BaseTarget = Branin(),
               seed: int = None
               ) -> tuple[torch.Tensor]:
    random_generator = (torch.Generator().manual_seed(seed) if seed is not None
                        else torch.Generator().manual_seed(torch.seed()))
    
    rand_vals = torch.rand(size = (n_iters, target.dim), 
                           generator = random_generator)
    
    X = target.bounds[0] + (target.bounds[1] - target.bounds[0]) * rand_vals
    y = torch.tensor([target.sample(x) for x in X])

    y = y.reshape(-1,1)
    return X, y

def test():
    X, y = dummy_data()
    target_smc = Target_SMC(X_train = X,
                            y_train = y)
    hyperparams = torch.tensor([[1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0]])
    target_smc.logpdf(X = hyperparams)