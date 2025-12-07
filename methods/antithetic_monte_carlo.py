import numpy as np

from models.simulate_gbm_paths import _simulate_gbm_paths_numba_impl
from payoffs import payoff_up_and_out_call


def antithetic_monte_carlo(
    S0: float,
    mu_star: float,
    sigma: float,
    r: float,
    K: float,
    C: float,
    T: float,
    n_steps: int,
    R: int,
    seed: int | None,
):
    """
    Estimate the option price using antithetic variates on the Brownian
    increments Z. Designed for n_steps = 1 (as in the task), but works for
    general n_steps as well.

    Parameters mirror the other estimators. R must be even so that each
    draw Z has a paired -Z.
    """
    if R % 2 != 0:
        raise ValueError("R must be even for antithetic sampling.")

    rng = np.random.default_rng(seed)
    half = R // 2

    # Generate Z and its negation to reduce variance.
    Z_half = rng.standard_normal((half, n_steps))
    Z = np.vstack((Z_half, -Z_half))

    # Keep the numba-compiled GBM simulator fast by reusing the low-level impl.
    S_paths = _simulate_gbm_paths_numba_impl(
        S0, mu_star, sigma, T, n_steps, Z
    )

    discounted_payoffs = payoff_up_and_out_call(S_paths, K, r, C, T)

    est = float(discounted_payoffs.mean())
    var = float(discounted_payoffs.var(ddof=1))
    se = float(np.sqrt(var / R))

    return est, se, var
