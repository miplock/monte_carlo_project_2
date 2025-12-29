import numpy as np
from scipy.stats import chi2

from payoffs import payoff_up_and_out_call


def _sample_ring_stratum(
    i: int,
    count: int,
    n_steps: int,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample `count` vectors Z in R^{n_steps} from stratum i (1..m) of N(0, I),
    where strata are concentric rings defined by ||Z|| (chi-square quantiles).
    """
    if count <= 0:
        return np.empty((0, n_steps), dtype=np.float64)

    # 1) random direction on unit sphere
    xi = rng.standard_normal((count, n_steps))
    norms = np.linalg.norm(xi, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    X = xi / norms[:, None]

    # 2) stratified radius via chi-square quantiles
    U = rng.random(count)
    p = (i - 1) / m + U / m
    D2 = chi2.ppf(p, df=n_steps)
    D = np.sqrt(D2)

    return X * D[:, None]


def _stratified_normals(
    R: int, n_steps: int, m: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Proportional allocation version (roughly equal counts per stratum).
    Returns Z of shape (R, n_steps).
    """
    if m <= 0:
        raise ValueError("Number of strata m must be positive.")
    if R <= 0:
        raise ValueError("Number of samples R must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if m > R:
        raise ValueError("Number of strata m cannot exceed number of samples R.")

    counts = np.full(m, R // m, dtype=np.int64)
    counts[: R % m] += 1

    Z = np.empty((R, n_steps), dtype=np.float64)
    idx = 0
    for i, c in enumerate(counts, start=1):
        Zi = _sample_ring_stratum(
            i=i, count=int(c), n_steps=n_steps, m=m, rng=rng
        )
        Z[idx: idx + c, :] = Zi
        idx += c

    rng.shuffle(Z, axis=0)
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
    m: int,
    R: int,
    seed: int | None,
    allocation: str = "proportional",  # "proportional" or "optimal"
    pilot_per_stratum: int | None = None,
):
    """
    Stratified Monte Carlo with 'ring' strata as in the PDF.

    Parameters (same as before) + optional:
    - allocation: "proportional" (default) or "optimal"
    - pilot_per_stratum: if allocation="optimal", how many pilot samples per
                         stratum to estimate within-stratum std dev; default
                         is auto.

    Returns
    -------
    est : float
        Stratified estimate of option price.
    se : float
        Estimated standard error of the stratified estimator.
    var : float
        Estimated variance of the stratified estimator (i.e., se^2).
    """
    rng = np.random.default_rng(seed)
    if allocation not in {"proportional", "optimal"}:
        raise ValueError('allocation must be "proportional" or "optimal"')
    if m <= 0 or R <= 0 or n_steps <= 0:
        raise ValueError("m, R, n_steps must be positive.")
    if m > R:
        raise ValueError("m cannot exceed R.")

    dt = T / n_steps
    t = (np.arange(1, n_steps + 1, dtype=np.float64) * dt)  # (n_steps,)
    p_i = 1.0 / m

    def _payoffs_from_Z(Z: np.ndarray) -> np.ndarray:
        # Brownian values at grid: B(t_k) = sqrt(dt) * sum_{j<=k} Z_j
        B = np.cumsum(Z, axis=1) * np.sqrt(dt)
        S_paths = S0 * np.exp(mu_star * t[None, :] + sigma * B)
        return payoff_up_and_out_call(S_paths, K, r, C, T)

    # -------- allocation: proportional (≈ equal counts) --------
    if allocation == "proportional":
        Z = _stratified_normals(R=R, n_steps=n_steps, m=m, rng=rng)
        Y = _payoffs_from_Z(Z)
        est = float(Y.mean())
        var_payoff = float(Y.var(ddof=1))
        se = float(np.sqrt(var_payoff / R))
        var = float(se * se)
        return est, se, var

    # -------- allocation: optimal (Neyman) --------
    # 1) pilot to estimate within-stratum std devs
    if pilot_per_stratum is None:
        # small, but not too small; keep it cheap and stable
        pilot_per_stratum = max(20, min(200, R // (5 * m)))
    pilot_per_stratum = int(pilot_per_stratum)
    if pilot_per_stratum < 2:
        pilot_per_stratum = 2

    s = np.zeros(m, dtype=np.float64)  # within-stratum std dev estimates
    for i in range(1, m + 1):
        Zp = _sample_ring_stratum(
            i=i, count=pilot_per_stratum, n_steps=n_steps, m=m, rng=rng
        )
        Yp = _payoffs_from_Z(Zp)
        # ddof=1 needs at least 2 samples
        s[i - 1] = float(Yp.std(ddof=1))

    # 2) Neyman allocation: R_i ∝ p_i * s_i; here p_i=1/m, so ∝ s_i
    # handle edge case where all s_i=0
    if np.all(s == 0.0):
        counts = np.full(m, R // m, dtype=np.int64)
        counts[: R % m] += 1
    else:
        weights = s / s.sum()
        raw = weights * R
        counts = np.floor(raw).astype(np.int64)

        # ensure at least 1 per stratum
        counts = np.maximum(counts, 1)

        # fix total to exactly R
        diff = int(R - counts.sum())
        if diff > 0:
            # add remaining to largest fractional parts (or just to largest weights)
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for k in range(diff):
                counts[order[k % m]] += 1
        elif diff < 0:
            # remove extras from strata with largest counts (but keep >=1)
            order = np.argsort(-counts)
            to_remove = -diff
            k = 0
            while to_remove > 0 and k < m * 10:
                j = order[k % m]
                if counts[j] > 1:
                    counts[j] -= 1
                    to_remove -= 1
                k += 1
            if counts.sum() != R:
                # final fallback: proportional
                counts = np.full(m, R // m, dtype=np.int64)
                counts[: R % m] += 1

    # 3) final stratified sampling with optimal counts; compute weighted
    #    estimator + stratified variance
    means = np.zeros(m, dtype=np.float64)
    vars_within = np.zeros(m, dtype=np.float64)

    for i, R_i in enumerate(counts, start=1):
        Zi = _sample_ring_stratum(
            i=i, count=int(R_i), n_steps=n_steps, m=m, rng=rng
        )
        Yi = _payoffs_from_Z(Zi)

        means[i - 1] = float(Yi.mean())
        # within-stratum sample variance
        vars_within[i - 1] = float(Yi.var(ddof=1)) if R_i > 1 else 0.0

    est = float(np.sum(p_i * means))

    # Estimated variance of the stratified estimator:
    # Var_hat = sum p_i^2 * s_i^2 / R_i  (use within-stratum sample variances)
    var = float(np.sum((p_i * p_i) * vars_within / counts))
    se = float(np.sqrt(var))
    return est, se, var
