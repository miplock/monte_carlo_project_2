import math
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def _simulate_gbm_paths_numba_impl(
    S0: float,
    mu_star: float,
    sigma: float,
    T: float,
    n_steps: int,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Internal Numba-compiled implementation for GBM path simulation.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    R = Z.shape[0]

    # Output: (R, n_steps)
    S_paths = np.empty((R, n_steps), dtype=np.float64)

    for i in nb.prange(R):
        B_t = 0.0
        for j in range(n_steps):
            B_t += sqrt_dt * Z[i, j]        # Brownian motion increment
            t = (j + 1) * dt
            S_paths[i, j] = S0 * math.exp(mu_star * t + sigma * B_t)

    return S_paths


def simulate_gbm_paths(
    S0: float,
    mu_star: float,
    sigma: float,
    T: float,
    n_steps: int,
    R: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate multiple Geometric Brownian Motion (GBM) paths using Numba.

    This function generates R independent paths of a GBM process on a uniform
    time grid with `n_steps` steps over the horizon [0, T]. The implementation
    is JIT-compiled with Numba and uses explicit Python loops instead of
    large intermediate matrices, which can significantly speed up
    simulations for large R and n_steps.

    The simulated process follows:
        S(t) = S0 * exp(mu_star * t + sigma * B(t)),

    where B(t) is a standard Brownian motion.

    Parameters
    ----------
    S0 : float
        Initial asset price S(0).
    mu_star : float
        Drift parameter in the exponent of the GBM.
    sigma : float
        Volatility parameter of the GBM.
    T : float
        Final time horizon.
    n_steps : int
        Number of time steps in each simulated path.
    R : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility. If None, the RNG is not explicitly
        seeded inside the compiled function.

    Returns
    -------
    np.ndarray
        Array of shape (R, n_steps) containing simulated GBM paths.
        Row i corresponds to the i-th simulated trajectory.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((R, n_steps))
    return _simulate_gbm_paths_numba_impl(S0, mu_star, sigma, T, n_steps, Z)
