import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build box plot of estimation errors for different m."
    )
    parser.add_argument(
        "--input",
        default="plots/strata_stats.pkl",
        help="Input .pkl with saved stats and errors.",
    )
    parser.add_argument(
        "--output",
        default="plots/strata_errors_box.pkl",
        help="Output .pkl path for the Plotly figure.",
    )
    return parser.parse_args()

def build_boxplot(errors_by_m: Dict[int, List[float]]) -> go.Figure:
    fig = go.Figure()
    for m, errors in sorted(errors_by_m.items()):
        fig.add_trace(
            go.Box(
                y=errors,
                name=f"m={m}",
                boxmean=True,
            )
        )

    fig.update_layout(
        title="Error distribution by strata count (stratified MC)",
        yaxis_title="Error (estimate - Black-Scholes)",
        xaxis_title="Number of strata m",
        template="plotly_white",
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

    fig = build_boxplot(errors_by_m)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(fig, f)
    print(f"Box plot saved to: {output_path}")


if __name__ == "__main__":
    main()
