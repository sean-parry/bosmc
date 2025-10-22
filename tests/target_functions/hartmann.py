from tests.target_functions.base import BaseTarget

import torch
from botorch.test_functions.synthetic import Hartmann as _Hartmann


class Hartmann(BaseTarget):
    def __init__(self, dim: int = 6) -> None:
        super().__init__()

        self.dim = dim
        self.bounds = torch.tensor([[0]*dim, [1]*dim]).to(torch.double)
        self.target_name = f'hartmann{dim}'

        # values for tracking
        self.num_evals = 0
        self.eval_positions: list[list] = []
        self.eval_values: list[float] = []

        self.regret_arr = []
        self.optimal_val = -3.32237

        self.bo_hartmann = _Hartmann(dim)
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
    minima = torch.tensor(
        data = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
        dtype = torch.float64
        )
    random_val = torch.rand(size = (6,))

    target = Hartmann()
    val_rand = target.sample(random_val)
    val_opt = target.sample(minima)
    print(f'The optimal value was evaluated it is{val_opt}'
          f'additional a random val was evaled {val_rand}')
    
    print(f'The results dict for this target is:\n{target.get_results()}')

if __name__ == '__main__':
    main()
