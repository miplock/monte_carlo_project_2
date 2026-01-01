import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import black_scholes_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary statistics from estimator replications."
    )
    parser.add_argument(
        "--input",
        default="data/european_estimator_replications_M1000.pkl",
        help="Input .pkl with estimator replications.",
    )
    parser.add_argument(
        "--output",
        default="data/european_estimator_stats.pkl",
        help="Output .pkl path for saved stats.",
    )
    return parser.parse_args()


def _compute_stats(
    estimates: np.ndarray, ses: np.ndarray, true_value: float
) -> Dict[str, float]:
    mean_est = float(np.mean(estimates))
    std_est = float(np.std(estimates, ddof=1))
    bias = float(mean_est - true_value)
    mse = float(np.mean((estimates - true_value) ** 2))

    lower = estimates - 1.96 * ses
    upper = estimates + 1.96 * ses
    coverage = float(np.mean((lower <= true_value) & (true_value <= upper)))

    return {
        "mean": mean_est,
        "std": std_est,
        "bias": bias,
        "mse": mse,
        "ci_coverage_95": coverage,
    }


def _format_scientific(value: float) -> str:
    if value == 0:
        return "0.000 &middot; 10<sup>0</sup>"
    sign = "-" if value < 0 else ""
    abs_val = abs(value)
    exp = int(np.floor(np.log10(abs_val))) + 1
    mantissa = abs_val / (10 ** exp)
    return f"{sign}{mantissa:.3f} &middot; 10<sup>{exp}</sup>"


def _format_default(value: float) -> str:
    return f"{value:.3f}"


def _format_coverage(value: float) -> str:
    text = f"{value:.3f}"
    text = text[1:] if text.startswith("0") else text
    return text


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("rb") as f:
        import pickle

        payload = pickle.load(f)

    params = payload["parameters"]
    estimates = payload["estimates"]
    ses = payload["ses"]

    true_value = black_scholes_call(
        S0=params["S0"],
        K=params["K"],
        r=params["r"],
        sigma=params["sigma"],
        T=params["T"],
    )

    method_keys = [
        "crude",
        "stratified_proportional",
        "stratified_optimal",
        "antithetic",
        "control",
    ]
    method_keys = [k for k in method_keys if k in estimates]

    stats: Dict[str, Dict[str, float]] = {}
    for key in method_keys:
        est_arr = np.array(estimates[key], dtype=float)
        se_arr = np.array(ses[key], dtype=float)
        stats[key] = _compute_stats(est_arr, se_arr, true_value)

    labels = {
        "crude": "crude",
        "stratified_proportional": "stratified (proportional)",
        "stratified_optimal": "stratified (optimal)",
        "antithetic": "antitetic",
        "control": "control variate",
    }
    stat_order = [
        ("MEAN", "mean"),
        ("SD", "std"),
        ("BIAS", "bias"),
        ("MSE", "mse"),
        ("CI-COVER-95%", "ci_coverage_95"),
    ]

    display_table = [["METHOD"] + [name for name, _ in stat_order]]
    for key in method_keys:
        stat_vals = stats[key]
        method_name = labels.get(key, key)
        row = [method_name]
        for col_name, stat_key in stat_order:
            value = stat_vals[stat_key]
            if stat_key == "ci_coverage_95":
                row.append(_format_coverage(value))
            elif stat_key == "bias":
                row.append(_format_scientific(value))
            else:
                row.append(_format_default(value))
        display_table.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(
            {
                "true_value": true_value,
                "parameters": params,
                "stats": stats,
                "display_table": display_table,
            },
            f,
        )
    print(f"Stats saved to: {output_path}")


if __name__ == "__main__":
    main()
