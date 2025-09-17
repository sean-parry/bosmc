import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Adjust file patterns as needed
mcmc_files = sorted(glob.glob("mcmc_branin_*_3_97.pkl"))
trad_files = sorted(glob.glob("trad_branin_*_3_97.pkl"))

def load_metric(file_list, key='regret'):
    """Load the chosen metric (e.g. 'regret' or 'eval_values') from each pickle file."""
    all_runs = []
    for f in file_list:
        with open(f, "rb") as handle:
            data = pickle.load(handle)  # dictionary
            metric = np.array(data[key])  # pick the key
            all_runs.append(metric)
    return np.stack(all_runs)  # shape = (n_seeds, n_iterations)

# Load regret across seeds for both methods
mcmc_regret = load_metric(mcmc_files, key='regret')
trad_regret = load_metric(trad_files, key='regret')

# Compute mean and std across seeds
mcmc_mean = mcmc_regret.mean(axis=0)
mcmc_std = mcmc_regret.std(axis=0)

trad_mean = trad_regret.mean(axis=0)
trad_std = trad_regret.std(axis=0)

# x-axis starting at 1
x = np.arange(1, len(mcmc_mean) + 1)  # 1,2,3,... instead of 0,1,2,...

# Plot
plt.figure(figsize=(8, 5))

plt.plot(x, mcmc_mean, label='MCMC Branin', color='blue')
plt.fill_between(x, mcmc_mean - mcmc_std, mcmc_mean + mcmc_std, color='blue', alpha=0.2)

plt.plot(x, trad_mean, label='Traditional Branin', color='red')
plt.fill_between(x, trad_mean - trad_std, trad_mean + trad_std, color='red', alpha=0.2)

plt.xlim((1, x[-1]))

plt.xlabel('Iteration')
plt.ylabel('Regret')
plt.title('Bayesian Optimisation (mean ± std) across seeds')
plt.yscale('log')  # logarithmic y-axis
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()