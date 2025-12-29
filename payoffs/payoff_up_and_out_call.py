import numpy as np
from numba import njit, prange


@njit(parallel=True)
def payoff_up_and_out_call(S_paths: np.ndarray,
                           K: float,
                           r: float,
                           C: float = np.inf,
                           T: float = 1.0) -> np.ndarray:
    """
    Compute discounted payoffs of an up-and-out barrier call option
    for a set of simulated price paths over a maturity T = 1, using Numba
    for JIT compilation and parallelization.

    Parameters
    ----------
    S_paths : np.ndarray
        2D array of shape (R, n_steps) containing simulated underlying
        asset price paths. Each row represents one path. For best
        performance, use a C-contiguous float64 array.
    K : float
        Strike price of the call option.
    r : float
        Risk-free interest rate used for discounting.
    C : float, optional
        Upper barrier level. If the asset price touches or exceeds this
        barrier at any point, the option is knocked out.
        Defaults to np.inf (i.e., no barrier).
    T : float, optional
        Time to maturity (in years). Defaults to 1.0.

    Returns
    -------
    np.ndarray
        1D array of shape (R,) containing the discounted option payoffs.
    """
    n_paths, n_steps = S_paths.shape
    discounted_payoffs = np.empty(n_paths)
    discount = np.exp(-r * T)

    barrier_is_finite = np.isfinite(C)

    for i in prange(n_paths):
        alive = True

        if barrier_is_finite:
            # Check barrier along the path
            for t in range(n_steps):
                if S_paths[i, t] >= C:
                    alive = False
                    break

        if alive:
            S_T = S_paths[i, n_steps - 1]
            intrinsic = S_T - K
            if intrinsic > 0.0:
                discounted_payoffs[i] = discount * intrinsic
            else:
                discounted_payoffs[i] = 0.0
        else:
            discounted_payoffs[i] = 0.0

    return discounted_payoffs
