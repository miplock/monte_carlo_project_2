import math


def _norm_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution function Φ(x).

    Uses the error function identity:
        Φ(x) = 0.5 * (1 + erf(x / sqrt(2))).

    Parameters
    ----------
    x:
        Point at which to evaluate Φ.

    Returns
    -------
    float
        The probability that a standard normal variable is <= x.
    """
    # Convert from N(0, 1) CDF to erf-based closed form.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """Price a European call option using the Black–Scholes model.

    Assumes:
    - Underlying follows geometric Brownian motion.
    - Constant risk-free rate r and volatility sigma.
    - No dividends (or equivalently, continuous dividend yield q = 0).

    Handles edge cases cleanly:
    - If T <= 0, returns intrinsic value max(S0 - K, 0).
    - If sigma <= 0, treats the underlying as deterministic under r.

    Parameters
    ----------
    S0:
        Current spot price of the underlying (must be >= 0).
    K:
        Strike price (must be > 0 for log(S0 / K) to be defined).
    r:
        Continuously-compounded risk-free rate.
    sigma:
        Volatility (annualized, must be >= 0).
    T:
        Time to maturity in years (must be >= 0).

    Returns
    -------
    float
        The Black–Scholes price of the European call option.
    """
    # At/after expiry the option is worth its intrinsic value.
    if T <= 0.0:
        return max(S0 - K, 0.0)

    # No randomness: under risk-neutral measure S_T is deterministic.
    if sigma <= 0.0:
        # Risk-neutral forward price: F = S0 * exp(rT).
        forward = S0 * math.exp(r * T)

        # Discounted payoff of a call: exp(-rT) * max(F - K, 0).
        return math.exp(-r * T) * max(forward - K, 0.0)

    # Precompute sqrt(T) once to avoid repeating the same work.
    sqrt_T = math.sqrt(T)

    # Compute d1 and d2 from the classic Black–Scholes formulas.
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (
        sigma * sqrt_T
    )
    d2 = d1 - sigma * sqrt_T

    # Price = S0 * Φ(d1) - K * exp(-rT) * Φ(d2).
    discounted_strike = K * math.exp(-r * T)
    return S0 * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
