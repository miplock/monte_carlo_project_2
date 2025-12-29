from models import simulate_gbm_paths
from payoffs import payoff_up_and_out_call
import numpy as np


def crude_monte_carlo(
    S0: float,
    mu_star: float,
    sigma: float,
    r: float,
    K: float,
    C: float,
    T: float,
    n_steps: int,
    R: int,
    seed: int,
):
    """
    Estimate the price of an up-and-out call option using the crude Monte Carlo
    method.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu_star : float
        Drift parameter for the GBM simulation (possibly under an adjusted
        measure).
    sigma : float
        Volatility of the underlying asset.
    r : float
        Risk-free interest rate used for discounting.
    K : float
        Strike price of the option.
    C : float
        Upper barrier level for the up-and-out call.
    T : float
        Time to maturity (in years).
    n_steps : int
        Number of time steps per simulated path.
    R : int
        Number of Monte Carlo simulated paths.
    seed : int
        Seed used to initialize random number generation inside the GBM
        simulator.

    Returns
    -------
    est : float
        Monte Carlo estimate of the option price.
    se : float
        Standard error of the estimator.
    var : float
        Sample variance of the discounted payoffs.

    Notes
    -----
    The function delegates the heavy numerical work to two Numba-compiled
    routines:
        - simulate_gbm_paths(...)
        - payoff_up_and_out_call(...)
    Both must be jitted; `crude_monte_carlo` itself does not need to be.
    """

    # Simulate GBM price paths (Numba-jitted)
    S_paths = simulate_gbm_paths(
        S0, mu_star, sigma, T, n_steps, R, seed
    )

    # Compute discounted option payoffs (Numba-jitted)
    discounted_payoffs = payoff_up_and_out_call(S_paths, K, r, C, T)

    # Monte Carlo estimators
    est = float(discounted_payoffs.mean())
    var = float(discounted_payoffs.var(ddof=1))
    se = float(np.sqrt(var / R))

    return est, se, var
