import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Locate files
mcmc_files = sorted(glob.glob("_branin_data/mcmc_branin_*_3_97.pkl"))
trad_files = sorted(glob.glob("_branin_data/trad_branin_*_3_97.pkl"))
smc_file = glob.glob("_branin_data/smc_branin_*_3_97.pkl")[0]  # only one SMC file

def load_metric(file_list, key='regret'):
    """Load a metric (like regret) from a list of pickle files."""
    all_runs = []
    for f in file_list:
        with open(f, "rb") as handle:
            data = pickle.load(handle)
            metric = np.array(data[key])
            all_runs.append(metric)
    return np.stack(all_runs)

# Load metrics
mcmc_regret = load_metric(mcmc_files, key='regret')
trad_regret = load_metric(trad_files, key='regret')

# Compute means and stds
mcmc_mean = mcmc_regret.mean(axis=0)
mcmc_std = mcmc_regret.std(axis=0)

trad_mean = trad_regret.mean(axis=0)
trad_std = trad_regret.std(axis=0)

# Load the single SMC regret
with open(smc_file, "rb") as handle:
    smc_regret = np.array(pickle.load(handle)['regret'])

# X-axis values
x = np.arange(1, len(mcmc_mean) + 1)

# Plotting
plt.figure(figsize=(8, 5))

# MCMC
plt.plot(x, mcmc_mean, label='MCMC, 256 iters, 28 runs', color='blue')
plt.fill_between(x, mcmc_mean - mcmc_std, mcmc_mean + mcmc_std, color='blue', alpha=0.2)

# Traditional
plt.plot(x, trad_mean, label='Traditional, 32 runs', color='red')
plt.fill_between(x, trad_mean - trad_std, trad_mean + trad_std, color='red', alpha=0.2)

# SMC - only mean, no error bars
plt.plot(x, smc_regret, label='SMC, 128 iters, 1 run', color='green', linestyle='--')

# Final touches
plt.xlim((1, x[-1]))
plt.xlabel('Iteration')
plt.ylabel('Regret')
plt.title('Bayesian Optimisation (mean ± std for MCMC & Traditional)')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()