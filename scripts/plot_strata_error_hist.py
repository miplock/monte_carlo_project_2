import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build histogram of estimation errors for different m."
    )
    parser.add_argument(
        "--input",
        default="plots/strata_stats.pkl",
        help="Input .pkl with saved stats and errors.",
    )
    parser.add_argument(
        "--output",
        default="plots/strata_errors_hist.pkl",
        help="Output .pkl path for the Plotly figure.",
    )
    parser.add_argument(
        "--nbinsx",
        type=int,
        default=30,
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

def build_histogram(
    errors_by_m: Dict[int, List[float]],
    nbinsx: int,
    opacity: float,
    barmode: str,
) -> go.Figure:
    fig = go.Figure()
    for m, errors in sorted(errors_by_m.items()):
        fig.add_trace(
            go.Histogram(
                x=errors,
                name=f"m={m}",
                opacity=opacity,
                nbinsx=nbinsx,
            )
        )

    fig.update_layout(
        title="Histogram of estimation errors (stratified MC)",
        xaxis_title="Error (estimate - Black-Scholes)",
        yaxis_title="Count",
        barmode=barmode,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("rb") as f:
        payload = pickle.load(f)
    if "errors_by_m" not in payload:
        raise KeyError("Input .pkl does not contain errors_by_m.")
    errors_by_m = payload["errors_by_m"]

    fig = build_histogram(errors_by_m, args.nbinsx, args.opacity, args.barmode)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(fig, f)
    print(f"Histogram saved to: {output_path}")


if __name__ == "__main__":
    main()
