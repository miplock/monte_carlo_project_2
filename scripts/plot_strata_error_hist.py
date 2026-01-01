import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        help="Histogram bar mode (ignored for single-series plots).",
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


def _kde_curve(values: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.array([]), np.array([])
    std = float(values.std(ddof=1))
    if std == 0.0:
        return np.array([]), np.array([])
    bw = 1.06 * std * (values.size ** (-1 / 5))
    if bw <= 0.0:
        return np.array([]), np.array([])
    grid = np.linspace(values.min() - 3 * std, values.max() + 3 * std, points)
    diffs = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return grid, density


def _color_for_label(label: str, idx: int) -> str:
    estimator_colors = {
        "Crude": "#1f77b4",
        "Stratified (proportional)": "#2ca02c",
        "Stratified (optimal)": "#17becf",
        "Antithetic": "#ff7f0e",
        "Control variate": "#9467bd",
    }
    if label in estimator_colors:
        return estimator_colors[label]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#8c564b"]
    return palette[idx % len(palette)]


def _to_rgba(color: str, alpha: float) -> str:
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return color


def build_histogram(
    errors_by_label: Dict[str, List[float]],
    nbinsx: int,
    opacity: float,
    barmode: str,
    title: str,
) -> go.Figure:
    labels = list(errors_by_label.keys())
    cols = 2
    rows = (len(labels) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)

    for idx, label in enumerate(labels, start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        errors = errors_by_label[label]
        color = _color_for_label(label, idx - 1)

        grid, density = _kde_curve(np.array(errors, dtype=float))
        if grid.size:
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=density,
                    mode="lines",
                    name="KDE",
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=_to_rgba(color, 0.15),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Histogram(
                x=errors,
                name=label,
                opacity=opacity,
                nbinsx=nbinsx,
                histnorm="probability density",
                marker=dict(color=color, line=dict(color="#444", width=1)),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode=barmode,
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=False,
        height=300 * rows,
    )
    fig.update_xaxes(title_text="Error (estimate - reference)")
    fig.update_yaxes(title_text="Density")
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
