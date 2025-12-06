import numpy as np

# task parameters
r = 0.05                  # risk-free rate
sigma2 = 0.125            # variance
sigma = np.sqrt(sigma2)   # volatility
mu_star = r - sigma2 / 2  # -0.0125
S0 = 100                  # initial stock price
K = 100                   # strike price
T = 1.0                   # time to maturity
C = np.inf                # upper barrier; use np.inf for brak bariery

# simulation parameters
n_steps = 100             # number of time steps
R = 50000                 # number of simulations

# reproducibility
seed = 12345
