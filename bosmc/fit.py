from bosmc.smc.api import SMC
from bosmc.smc.nuts import NUTSSMCKernel
from bosmc.smc.rw import RandomWalkSMCKernel
from bosmc.models import SMCFullyBayesianSingleTaskGP, WeightedGaussianMixturePosterior

from botorch.models.transforms import Normalize, Standardize

def fit_fully_bayes_model_nut_smc(
        model: SMCFullyBayesianSingleTaskGP,
        max_tree_depth: int = 6,
        num_iters: int = 128,
        num_samples: int = 128,
        disable_progbar: bool = False,
        jit_compile: bool = False,
    ) -> None:

    model.train()

    nuts = NUTSSMCKernel(
        model.pyro_model.sample,
        jit_compile=jit_compile,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=max_tree_depth
    )
    smc = SMC(
        kernel = nuts,
        num_iters=num_iters,
        num_samples=num_samples,
        disable_progbar=disable_progbar
    )

    smc.run()

    samples, weights = smc.get_weighted_samples()

    model.load_smc_samples(samples, weights)

    model.eval()


def fit_fully_bayes_model_rw_smc(
        model: SMCFullyBayesianSingleTaskGP,
        num_iters: int = 64,
        num_samples: int = 64,
        disable_progbar: bool = False,
    ) -> None:

    model.train()

    rw = RandomWalkSMCKernel(
        model.pyro_model.sample,
        init_step_size=5.0
    )
    smc = SMC(
        kernel = rw,
        num_iters=num_iters,
        num_samples=num_samples,
        disable_progbar=disable_progbar
    )

    smc.run()

    samples, weights = smc.get_weighted_samples()

    samples = model.pyro_model.postprocess_mcmc_samples(samples)

    model.load_mcmc_samples(samples)

    model.set_weigths(weights)

    model.eval()

def _test():
    import torch
    n_data_points = 5
    dim = 1
    train_X = torch.rand((n_data_points, dim), dtype=torch.float64)
    train_y = torch.rand((n_data_points, 1), dtype=torch.float64)
    model = SMCFullyBayesianSingleTaskGP(train_X, 
                                         train_y,
                                         input_transform=Normalize(d=dim),
                                         outcome_transform=Standardize(m=1),)
    
    fit_fully_bayes_model_rw_smc(model = model,
                                 num_iters = 8,
                                 num_samples = 4,
                                 disable_progbar=True)
    
    test_data: int = 1
    gauss_mix = model.posterior(torch.rand((test_data, dim), dtype=torch.float64))
    return