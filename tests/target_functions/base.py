import torch

from abc import ABC, abstractmethod

class BaseTarget(ABC):
    """
    The target is the way of the end user seeeing where evaluations were performed, see branin.py 
    as an example of how to track the evaluations done to your target
    """
    def __init__(self):
        """
        dim is the expected length of the vector input x in self.sample(x), can be left as None
        provided either bounds, or constraints are not None
        bounds are simple box constraints set on x, [[min0, min1],[max0, max1]
        """
        self.dim: int | None = None
        self.bounds: torch.Tensor | None = None
        pass

    @abstractmethod
    def sample(self,
               x: torch.Tensor,
               ) -> torch.Tensor:
        """
        returns a float, a value to maximise.
        """
        pass