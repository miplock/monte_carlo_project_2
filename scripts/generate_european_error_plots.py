import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import black_scholes_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Plotly error plots for European estimator replications."
        )
    )
    parser.add_argument(
        "--input",
        default="data/european_estimator_replications_M1000.pkl",
        help="Input .pkl with estimator replications.",
    )
    parser.add_argument(
        "--hist-output",
        default="plots/european_errors_hist.pkl",
        help="Output .pkl path for histogram figure.",
    )
    parser.add_argument(
        "--box-output",
        default="plots/european_errors_box.pkl",
        help="Output .pkl path for box plot figure.",
    )
    parser.add_argument(
        "--nbinsx",
        type=int,
        default=40,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--barmode",
        choices=["overlay", "group"],
        default="group",
        help="Histogram bar mode.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.6,
        help="Histogram opacity in overlay mode.",
    )
    return parser.parse_args()


def _label_for_estimator(key: str) -> str:
    labels = {
        "crude": "Crude",
        "stratified_proportional": "Stratified (proportional)",
        "stratified_optimal": "Stratified (optimal)",
        "antithetic": "Antithetic",
        "control": "Control variate",
    }
    return labels.get(key, key)


def build_histogram(
    errors_by_label: Dict[str, List[float]],
    nbinsx: int,
    opacity: float,
    barmode: str,
) -> go.Figure:
    fig = go.Figure()
    for label, errors in errors_by_label.items():
        fig.add_trace(
            go.Histogram(
                x=errors,
                name=label,
                opacity=opacity,
                nbinsx=nbinsx,
            )
        )

    fig.update_layout(
        title="Error histograms vs Black-Scholes",
        xaxis_title="Error (estimate - Black-Scholes)",
        yaxis_title="Count",
        barmode=barmode,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def build_boxplot(errors_by_label: Dict[str, List[float]]) -> go.Figure:
    fig = go.Figure()
    for label, errors in errors_by_label.items():
        fig.add_trace(
            go.Box(
                y=errors,
                name=label,
                boxmean=True,
            )
        )

    fig.update_layout(
        title="Error boxplots vs Black-Scholes",
        yaxis_title="Error (estimate - Black-Scholes)",
        xaxis_title="Estimator",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("rb") as f:
        import pickle

        payload = pickle.load(f)

    params = payload["parameters"]
    estimates = payload["estimates"]

    reference = black_scholes_call(
        S0=params["S0"],
        K=params["K"],
        r=params["r"],
        sigma=params["sigma"],
        T=params["T"],
    )

    errors_by_label = {
        _label_for_estimator(key): (np.array(vals) - reference).tolist()
        for key, vals in estimates.items()
    }

    hist_fig = build_histogram(
        errors_by_label, args.nbinsx, args.opacity, args.barmode
    )
    box_fig = build_boxplot(errors_by_label)

    hist_path = Path(args.hist_output)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("wb") as f:
        import pickle

        pickle.dump(hist_fig, f)

    box_path = Path(args.box_output)
    box_path.parent.mkdir(parents=True, exist_ok=True)
    with box_path.open("wb") as f:
        import pickle

        pickle.dump(box_fig, f)

    print(f"Histogram saved to: {hist_path}")
    print(f"Box plot saved to: {box_path}")


if __name__ == "__main__":
    main()
