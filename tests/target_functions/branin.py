import torch

from tests.target_functions.base import BaseTarget

from scipy.optimize import Bounds

class Branin(BaseTarget):
    """
    2 dimensional branin target toy problem, limits should be x0 [-5, 10] x1 [0, 15]

    function to minimise with
    f(x*) = 0.397887 at x* = (-pi, 12.275), (+pi, 2.275), and (9.42478, 2.475)

    Ref:
    [1] https://scikit-optimize.github.io/stable/_modules/skopt/benchmarks.html#branin
    """
    def __init__(self):
        # arguments required from base class
        super().__init__()
        self.dim = 2
        self.bounds = torch.tensor([[-5, 0], [10, 15]]).to(torch.double)

        # values for tracking
        self.num_evals = 0
        self.eval_positions: list[list] = []
        self.eval_values: list[float] = []

        self.regret_arr = []
        self.optimal_val = 0.397887

        # constants for the calculation
        self._a = 1.0
        self._b = 5.1 / (4.0 * (torch.pi ** 2))
        self._c = 5.0 / torch.pi
        self._r = 6.0
        self._s = 10.0
        self._t = 1.0 / (8.0 * torch.pi)

    def _update_regret(self):
        if self.regret_arr:
            regret = min(self.regret_arr[-1], abs(self.optimal_val-self.eval_values[-1]))
        else:
            regret = abs(self.optimal_val-self.eval_values[-1])
        self.regret_arr.append(regret)

    def sample(self,
               x: torch.Tensor
               ) -> torch.Tensor:
        """
        R2 to R
        returns -ans, (this is a function to maximize)
        """
        self.num_evals += 1
        self.eval_positions.append(x)

        x0, x1 = x
        ans = self._a*(x1 - self._b*x0**2 + self._c*x0 - self._r)**2 + self._s*(1-self._t) * torch.cos(x0) + self._s

        self.eval_values.append(ans)
        self._update_regret()

        return  -ans