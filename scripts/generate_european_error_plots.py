import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def _kde_curve(values: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
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
        title=dict(text="Error histograms vs Black-Scholes", x=0.5),
        barmode=barmode,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=50),
        height=300 * rows,
    )
    fig.update_xaxes(title_text="Error (estimate - Black-Scholes)")
    fig.update_yaxes(title_text="Density")
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

    box_fig = build_boxplot(errors_by_label)

    hist_fig = build_histogram(
        errors_by_label, args.nbinsx, args.opacity, args.barmode
    )

    hist_path = Path(args.hist_output)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("wb") as f:
        import pickle

        pickle.dump(hist_fig, f)
    print(f"Histogram saved to: {hist_path}")

    box_path = Path(args.box_output)
    box_path.parent.mkdir(parents=True, exist_ok=True)
    with box_path.open("wb") as f:
        import pickle

        pickle.dump(box_fig, f)

    print(f"Box plot saved to: {box_path}")


if __name__ == "__main__":
    main()
