from models import simulate_gbm_paths
from payoffs import payoff_up_and_out_call
import numpy as np


def control_variate_monte_carlo(
    S0: float,
    mu_star: float,
    sigma: float,
    r: float,
    K: float,
    T: float,
    R: int,
    seed: int,
):
    """
    Estimate the price of a European call option using a control variate
    Monte Carlo estimator with the Brownian motion value B(T) as the control
    variate.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu_star : float
        Drift parameter under the chosen measure (often r - sigma^2/2).
    sigma : float
        Volatility of the underlying asset.
    r : float
        Risk-free interest rate used for discounting.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    R : int
        Number of Monte Carlo simulated paths.
    seed : int
        Seed for the GBM simulator’s random number generator.

    Returns
    -------
    est_cv : float
        Control variate Monte Carlo estimate of the option price.
    se_cv : float
        Standard error of the estimator.
    var_cv : float
        Sample variance of the control-variate-adjusted payoffs.

    Notes
    -----
    Control variate:
        Y = discounted payoff of the European call.
        X = B(T), reconstructed from the final GBM price:

            S(T) = S0 * exp(mu_star * T + sigma * B(T))
            => B(T) = (log(S(T)/S0) - mu_star * T) / sigma

        Since E[B(T)] = 0, the optimal control variate estimator is:

            beta* = Cov(Y, X) / Var(X)
            Y_cv  = Y - beta* * X
    """

    n_steps = 1
    C = np.inf

    # 1. Simulate GBM price paths (Numba-jitted)
    S_paths = simulate_gbm_paths(
        S0, mu_star, sigma, T, n_steps, R, seed
    )

    # 2. Compute discounted payoffs of the European call (Numba-jitted)
    discounted_payoffs = payoff_up_and_out_call(S_paths, K, r, C, T)
    Y = discounted_payoffs

    # 3. Extract S(T) and reconstruct B(T) from the terminal GBM value
    S_T = S_paths[:, -1]

    # Reconstruct Brownian motion at time T
    B_T = (np.log(S_T / S0) - mu_star * T) / sigma
    X = B_T  # control variate

    # 4. Compute optimal beta* using sample covariance and variance
    cov_YX = np.cov(Y, X, ddof=1)[0, 1]
    var_X = np.var(X, ddof=1)

    beta_star = 0.0 if var_X == 0.0 else cov_YX / var_X

    # 5. Apply control variate correction (E[X] = 0)
    Y_cv = Y - beta_star * X

    # 6. Monte Carlo estimators
    est_cv = float(Y_cv.mean())
    var_cv = float(Y_cv.var(ddof=1))
    se_cv = float(np.sqrt(var_cv / R))

    return est_cv, se_cv, var_cv
