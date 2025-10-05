import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Adjust file patterns as needed
smc_file = sorted(glob.glob("_branin_data/smc_branin_*_3_97.pkl"))
mcmc_files = sorted(glob.glob("_branin_data/mcmc_branin_*_3_97.pkl"))
trad_files = sorted(glob.glob("_branin_data/trad_branin_*_3_97.pkl"))

def load_metric(file_list, key='regret'):
    """Load the chosen metric (e.g. 'regret' or 'eval_values') from each pickle file."""
    all_runs = []
    for f in file_list:
        with open(f, "rb") as handle:
            data = pickle.load(handle)  # dictionary
            metric = np.array(data[key])  # pick the key
            all_runs.append(metric)
    return np.stack(all_runs)  # shape = (n_seeds, n_iterations)

# Load regret across seeds for each method
mcmc_regret = load_metric(mcmc_files, key='regret')
trad_regret = load_metric(trad_files, key='regret')
smc_regret = load_metric(smc_file, key='regret')

# Compute mean and std across seeds
mcmc_mean, mcmc_std = mcmc_regret.mean(axis=0), mcmc_regret.std(axis=0)
trad_mean, trad_std = trad_regret.mean(axis=0), trad_regret.std(axis=0)
smc_mean, smc_std = smc_regret.mean(axis=0), smc_regret.std(axis=0)


def plot_w_uncertainty():
    # x-axis starting at 1
    x = np.arange(1, len(mcmc_mean) + 1)

    # Compute global min/max for setting y-limits
    y_min = min(
        np.min(mcmc_mean - mcmc_std),
        np.min(trad_mean - trad_std),
        np.min(smc_mean - smc_std)
    )
    y_max = max(
        np.max(mcmc_mean + mcmc_std),
        np.max(trad_mean + trad_std),
        np.max(smc_mean + smc_std)
    )

    # Add some padding to the y-range for visibility
    y_min *= 0.8
    y_max *= 1.2

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(x, mcmc_mean, label='MCMC Branin', color='blue')
    plt.fill_between(x, mcmc_mean - mcmc_std, mcmc_mean + mcmc_std, color='blue', alpha=0.2)

    plt.plot(x, trad_mean, label='Traditional Branin', color='red')
    plt.fill_between(x, trad_mean - trad_std, trad_mean + trad_std, color='red', alpha=0.2)

    plt.plot(x, smc_mean, label='SMC Branin', color='green')
    plt.fill_between(x, smc_mean - smc_std, smc_mean + smc_std, color='green', alpha=0.2)

    plt.xlim((1, x[-1]))
    plt.ylim((1e-5, y_max))

    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title('Bayesian Optimisation (mean ± std) across seeds')
    plt.yscale('log')  # logarithmic y-axis
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()

def plot_wo_uncertainty(): 
    x = np.arange(1, len(mcmc_mean) + 1)

    # Compute y_max from means only (we keep your lower bound at 1e-5)
    y_max = max(np.max(mcmc_mean), np.max(trad_mean), np.max(smc_mean)) * 1.2

    # Plot (means only)
    plt.figure(figsize=(8, 5))

    plt.plot(x, mcmc_mean, label='MCMC, 256 iters', color='blue')
    plt.plot(x, trad_mean, label='ADAM Optimizer', color='red')
    plt.plot(x, smc_mean,  label='SMC, 128 iters', color='green')

    plt.xlim((1, x[-1]))
    plt.ylim((1e-4, y_max))

    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title('Bayesian Optimisation, mean of multiple runs for ADAM, and MCMC, single run for SMC')
    plt.yscale('log')  # logarithmic y-axis
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()

plot_wo_uncertainty()