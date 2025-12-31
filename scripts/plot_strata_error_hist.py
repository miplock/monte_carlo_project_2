import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import black_scholes_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build histogram of estimation errors."
    )
    parser.add_argument(
        "--input",
        default="data/strata_stats.pkl",
        help="Input .pkl with saved stats/estimates.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Reference .pkl with I_ref for barrier option.",
    )
    parser.add_argument(
        "--barrier",
        type=float,
        default=None,
        help="Barrier level to select when input contains multiple barriers.",
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

def _label_for_estimator(key: str) -> str:
    labels = {
        "crude": "Crude",
        "stratified_proportional": "Stratified (proportional)",
        "stratified_optimal": "Stratified (optimal)",
        "antithetic": "Antithetic",
        "control": "Control variate",
    }
    return labels.get(key, key)


def _load_errors(
    payload: dict,
    ref_path: str | None,
    barrier: float | None,
) -> Tuple[Dict[str, List[float]], str]:
    if "errors_by_m" in payload:
        errors_by_m = payload["errors_by_m"]
        errors_by_label = {
            f"m={m}": errors_by_m[m] for m in sorted(errors_by_m)
        }
        return errors_by_label, "Histogram of estimation errors (stratified MC)"

    if "estimates" not in payload:
        raise KeyError("Input .pkl does not contain errors_by_m or estimates.")

    estimates = payload["estimates"]
    if "barriers" in payload:
        if barrier is None:
            raise ValueError("Barrier option requires --barrier.")
        if ref_path is None:
            raise ValueError("Barrier option requires --ref with I_ref.")
        barriers = {float(b): b for b in payload["barriers"]}
        if float(barrier) not in barriers:
            raise KeyError(f"Barrier {barrier} not found in input.")
        barrier_key = barriers[float(barrier)]
        estimates = estimates[barrier_key]
        with Path(ref_path).open("rb") as f:
            ref_payload = pickle.load(f)
        if "I_ref" not in ref_payload:
            raise KeyError("Reference .pkl does not contain I_ref.")
        reference = float(ref_payload["I_ref"])
        title = f"Histogram of estimation errors (barrier C={barrier:g})"
    else:
        params = payload["parameters"]
        reference = black_scholes_call(
            S0=params["S0"],
            K=params["K"],
            r=params["r"],
            sigma=params["sigma"],
            T=params["T"],
        )
        title = "Histogram of estimation errors (European)"

    errors_by_label = {
        _label_for_estimator(key): (np.array(vals) - reference).tolist()
        for key, vals in estimates.items()
    }
    return errors_by_label, title


def build_histogram(
    errors_by_label: Dict[str, List[float]],
    nbinsx: int,
    opacity: float,
    barmode: str,
    title: str,
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
        title=title,
        xaxis_title="Error (estimate - reference)",
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

    errors_by_label, title = _load_errors(payload, args.ref, args.barrier)

    fig = build_histogram(
        errors_by_label,
        args.nbinsx,
        args.opacity,
        args.barmode,
        title,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(fig, f)
    print(f"Histogram saved to: {output_path}")


if __name__ == "__main__":
    main()
