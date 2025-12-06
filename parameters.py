import numpy as np

# task parameters
r = 0.05
sigma2 = 0.125
sigma = np.sqrt(sigma2)
mu_star = r - sigma2 / 2  # -0.0125
S0 = 100
K = 100

# Set seed for reproducibility
rng = np.random.default_rng(12345)
