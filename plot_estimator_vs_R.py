import argparse
from typing import Tuple

import plotly.graph_objects as go

from methods import (
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    crude_monte_carlo,
    stratified_monte_carlo,
)
from parameters import (
    C,
    K,
    S0,
    T,
    mu_star,
    n_steps,
    r,
    seed,
    sigma,
    m,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wygeneruj wykres wartości estymatora "
            "(crude, stratified, antithetic, control variate) "
            "w zależności od liczby ścieżek R."
        )
    )
    parser.add_argument(
        "--method",
        choices=["crude", "stratified", "antithetic", "control"],
        default="crude",
        help="Który estymator uruchomić.",
    )
    parser.add_argument(
        "--R-values",
        nargs="+",
        type=int,
        default=[200, 500, 1_000, 5_000, 10_000],
        help="Lista wartości R (liczba ścieżek) do przetestowania.",
    )
    parser.add_argument(
        "--output",
        default="estimator_vs_R.html",
        help="Ścieżka do pliku HTML z wykresem.",
    )
    parser.add_argument(
        "--vary-seed",
        action="store_true",
        help=(
            "Jeśli ustawione, dodaje numer iteracji do seed, "
            "żeby uniezależnić próby."
        ),
    )
    return parser.parse_args()


def run_estimator(method: str, R_value: int, seed_offset: int) -> Tuple[float, float]:
    # Niewielka zmiana seed zmniejsza korelację między próbami dla kolejnych R.
    run_seed = seed + seed_offset if seed is not None else None

    if method == "crude":
        est, se, _ = crude_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=n_steps,
            R=R_value,
            seed=run_seed,
        )
    elif method == "control":
        est, se, _ = control_variate_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=n_steps,
            R=R_value,
            seed=run_seed,
        )
    elif method == "antithetic":
        est, se, _ = antithetic_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=n_steps,
            R=R_value,
            seed=run_seed,
        )
    else:
        est, se, _ = stratified_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=n_steps,
            m=m,
            R=R_value,
            seed=run_seed,
        )

    return est, se


def main() -> None:
    args = parse_args()
    R_values = [int(v) for v in args.R_values]

    estimates: list[float] = []
    ses: list[float] = []

    for idx, R_value in enumerate(R_values):
        est, se = run_estimator(
            method=args.method,
            R_value=R_value,
            seed_offset=idx if args.vary_seed else 0,
        )
        estimates.append(est)
        ses.append(se)
        print(f"R={R_value:>6}: est={est:.6f}, se={se:.6f}")

    fig = go.Figure(
        go.Scatter(
            x=R_values,
            y=estimates,
            mode="lines+markers",
            name=f"{args.method} estimator",
            error_y=dict(
                type="data",
                array=ses,
                visible=True,
                thickness=1.2,
                width=3,
                color="#444",
            ),
        )
    )

    fig.update_layout(
        title="Wartość estymatora vs R",
        xaxis=dict(
            title="Liczba ścieżek R (skala log)",
            type="log",
            gridcolor="rgba(0,0,0,0.15)",
        ),
        yaxis=dict(
            title="Szacowana wartość opcji",
            gridcolor="rgba(0,0,0,0.15)",
        ),
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )

    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Wykres zapisany do: {args.output}")


if __name__ == "__main__":
    main()
