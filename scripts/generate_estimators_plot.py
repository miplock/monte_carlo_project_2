import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_estimator_vs_R import run_estimator

DEFAULT_R_VALUES: List[int] = [200, 500, 1_000, 5_000, 10_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wygeneruj wykres wartości estymatorów (crude i control variate) "
            "w zależności od liczby ścieżek R i zapisz go jako .pkl."
        )
    )
    parser.add_argument(
        "--R-values",
        nargs="+",
        type=int,
        default=DEFAULT_R_VALUES,
        help="Lista wartości R (liczba ścieżek) do przetestowania.",
    )
    parser.add_argument(
        "--output",
        default="plots/estimators_vs_R.pkl",
        help="Ścieżka do pliku .pkl z zapisanym wykresem Plotly.",
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


def gather_estimates(
    method: str, R_values: List[int], vary_seed: bool
) -> Tuple[List[float], List[float]]:
    estimates: List[float] = []
    ses: List[float] = []

    for idx, R_value in enumerate(R_values):
        est, se = run_estimator(
            method=method,
            R_value=R_value,
            seed_offset=idx if vary_seed else 0,
        )
        estimates.append(est)
        ses.append(se)
        print(f"{method:7} | R={R_value:>6}: est={est:.6f}, se={se:.6f}")

    return estimates, ses


def build_figure(R_values: List[int], vary_seed: bool) -> go.Figure:
    crude_estimates, crude_ses = gather_estimates("crude", R_values, vary_seed)
    control_estimates, control_ses = gather_estimates(
        "control", R_values, vary_seed
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=R_values,
            y=crude_estimates,
            mode="lines+markers",
            name="Crude estimator",
            error_y=dict(
                type="data",
                array=crude_ses,
                visible=True,
                thickness=1.2,
                width=3,
                color="#444",
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=R_values,
            y=control_estimates,
            mode="lines+markers",
            name="Control variate estimator",
            error_y=dict(
                type="data",
                array=control_ses,
                visible=True,
                thickness=1.2,
                width=3,
                color="#1f77b4",
            ),
        )
    )

    fig.update_layout(
        title="Wartość estymatorów vs R",
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

    return fig


def main() -> None:
    args = parse_args()
    R_values = [int(v) for v in args.R_values]

    fig = build_figure(R_values, args.vary_seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(fig, f)

    print(f"Wykres zapisany do: {output_path}")


if __name__ == "__main__":
    main()
