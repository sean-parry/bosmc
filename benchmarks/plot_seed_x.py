import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# File pattern that matches only seed 3 (because of _branin_3_3_97.pkl)
mcmc_file = glob.glob("_branin_data/mcmc_branin_2_3_97.pkl")[0]
trad_file = glob.glob("_branin_data/trad_branin_3_3_97.pkl")[0]
smc_file  = glob.glob("_branin_data/smc_branin_3_3_97.pkl")[0]

def load_single_run(file_path, key='regret'):
    with open(file_path, "rb") as handle:
        data = pickle.load(handle)
        return np.array(data[key])

# Load regret values
mcmc_regret = load_single_run(mcmc_file, key='regret')
trad_regret = load_single_run(trad_file, key='regret')
smc_regret  = load_single_run(smc_file,  key='regret')

# X-axis
x = np.arange(1, len(mcmc_regret) + 1)

# Plotting
plt.figure(figsize=(8, 5))

plt.plot(x, mcmc_regret, label='MCMC (Seed 3)', color='blue')
plt.plot(x, trad_regret, label='Traditional (Seed 3)', color='red')
plt.plot(x, smc_regret, label='SMC (Seed 3)', color='green', linestyle='--')

plt.xlim((1, x[-1]))
plt.xlabel('Iteration')
plt.ylabel('Regret')
plt.title('Bayesian Optimisation – Seed 3 Only')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()