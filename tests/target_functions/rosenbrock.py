from tests.target_functions.base import BaseTarget

import torch
from botorch.test_functions.synthetic import Rosenbrock as _Rosenbrock


class Rosenbrock(BaseTarget):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()

        self.dim = dim
        self.bounds = torch.tensor([[-5]*dim, [10]*dim]).to(torch.double)
        self.target_name = f'hartmann{dim}'

        # values for tracking
        self.num_evals = 0
        self.eval_positions: list[list] = []
        self.eval_values: list[float] = []

        self.regret_arr = []
        self.optimal_val = 0.0

        self.bo_hartmann = _Rosenbrock(dim)
        return
    
    def _update_regret(self):
        if self.regret_arr:
            regret = min(self.regret_arr[-1], abs(self.optimal_val-self.eval_values[-1]))
        else:
            regret = abs(self.optimal_val-self.eval_values[-1])
        self.regret_arr.append(regret)

    def get_results(self) -> dict:
        return {'eval_positions': self.eval_positions,
                'eval_values': self.eval_values,
                'regret': self.regret_arr}
    
    def sample(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns value to maximise
        """
        ans = self.bo_hartmann.forward(x, noise = False)

        self.eval_positions.append(x)
        self.eval_values.append(ans)
        self._update_regret()

        return -ans
    

def main():
    dim = 4
    minima = torch.tensor(
        data = [1]*dim,
        dtype = torch.float64
        )
    random_val = torch.rand(size = (dim,))

    target = Rosenbrock(dim)
    val_rand = target.sample(random_val)
    val_opt = target.sample(minima)
    print(f'The optimal value was evaluated it is{val_opt}'
          f'additional a random val was evaled {val_rand}')
    
    print(f'The results dict for this target is:\n{target.get_results()}')

if __name__ == '__main__':
    main()