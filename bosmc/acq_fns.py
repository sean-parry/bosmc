import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction

class WeightedLogEI(AcquisitionFunction):
    def __init__(self, models, weights, best_f):
        super().__init__(model=None) # don't need model
        self.acq_fns = [LogExpectedImprovement(model=model, best_f=best_f) for model in models]
        self.weights = weights

    def forward(self, X):
        acq_vals = torch.tensor([acq_fn.forward(X) for acq_fn in self.acq_fns])
        return torch.sum(acq_vals * self.weights)
