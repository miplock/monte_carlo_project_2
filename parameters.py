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
# dodatkowe bariery do wizualizacji trajektorii
BARRIERS = (105.0, 120.0)

# simulation parameters
n_steps = 12              # number of time steps
R = 100000                # number of simulations
m = 20                    # number of strata for stratified Monte Carlo

# reproducibility
seed = 12345
