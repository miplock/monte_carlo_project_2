import numpy as np
from scipy.stats import norm

from models.simulate_gbm_paths import _simulate_gbm_paths_numba_impl
from payoffs import payoff_up_and_out_call


def _stratified_normals(R: int,
                        n_steps: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Generate a matrix of stratified N(0, 1) samples of shape (R, n_steps).

    For each time step j, [0, 1] is split into R equal strata. One uniform
    sample is drawn from every stratum, the samples are shuffled to avoid
    correlation across time steps, and then mapped to the normal distribution
    via the inverse CDF.
    """
    Z = np.empty((R, n_steps), dtype=np.float64)

    for j in range(n_steps):
        u = (np.arange(R) + rng.random(R)) / R
        rng.shuffle(u)
        Z[:, j] = norm.ppf(u)

    return Z


def stratified_monte_carlo(
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
    Estimate the up-and-out call price using stratified sampling on the normal
    increments of the GBM.

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
        Number of Monte Carlo simulated paths (strata).
    seed : int | None
        Seed used to initialize random number generation.

    Returns
    -------
    est : float
        Stratified Monte Carlo estimate of the option price.
    se : float
        Standard error of the estimator.
    var : float
        Sample variance of the discounted payoffs.
    """
    rng = np.random.default_rng(seed)

    # Pre-generate stratified normals for every time step to keep the GBM
    # simulator jitted and fast.
    Z = _stratified_normals(R=R, n_steps=n_steps, rng=rng)
    S_paths = _simulate_gbm_paths_numba_impl(
        S0, mu_star, sigma, T, n_steps, Z
    )

    discounted_payoffs = payoff_up_and_out_call(S_paths, K, r, C, T)

    est = float(discounted_payoffs.mean())
    var = float(discounted_payoffs.var(ddof=1))
    se = float(np.sqrt(var / R))

    return est, se, var
