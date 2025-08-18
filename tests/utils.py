from bosmc.target_function.base import BaseTarget
from bosmc.target_function.branin import Branin

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