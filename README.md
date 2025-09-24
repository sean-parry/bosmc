# BOSMC

BOSMC uses a Sequantial Monte-Carlo (SMC) Sampler with a No U-Turn Sampler (NUTs) proposal, to create a gaussian mixture predictor to be used with botorch acquisition functions for bayesian optimization. Usage is intentionally very similar to using botorch's fully bayesian models.

example usage:
```python
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from bosmc.models import SMCFullyBayesianSingleTaskGP
from bosmc.fit import fit_fully_bayes_model_nuts, fit_fully_bayes_model_rw

# use your existing data (or do at least 3 random evals of your target)
n_datapoints, dims = 10, 3
train_X = torch.rand((n_datapoints, dim))
train_Y = torch.rand((n_datapoints, 1))

# bosmc functionality
model = SMCFullyBayesianSingleTaskGP(train_X, train_y)
fit_fully_bayes_model_nuts(model)

# then use model normally in botorch acquisition functions e.g.
ei = LogExpectedImprovement(model=model, best_f=best_f, maximize=True)
best_f = train_Y.max().item()
bounds = torch.tensor([[0.0], [1.0]])
x_star, acq_value = optimize_acqf(
    acq_function=ei,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)
print("Next candidate:", x_star)
```