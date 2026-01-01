import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary statistics for barrier estimator replications."
    )
    parser.add_argument(
        "--input",
        default="data/barrier_estimator_replications_M1000.pkl",
        help="Input .pkl with estimator replications.",
    )
    parser.add_argument(
        "--refs",
        nargs="+",
        required=True,
        help="Reference .pkl files (I_ref) in the same order as barriers.",
    )
    parser.add_argument(
        "--output",
        default="data/barrier_estimator_stats.pkl",
        help="Output .pkl path for saved stats.",
    )
    return parser.parse_args()


def _compute_stats(
    estimates: np.ndarray, ses: np.ndarray, ref_value: float
) -> Dict[str, float]:
    mean_est = float(np.mean(estimates))
    std_est = float(np.std(estimates, ddof=1))
    bias = float(mean_est - ref_value)
    mse = float(np.mean((estimates - ref_value) ** 2))

    lower = estimates - 1.96 * ses
    upper = estimates + 1.96 * ses
    coverage = float(np.mean((lower <= ref_value) & (ref_value <= upper)))

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

    barriers = payload["barriers"]
    if len(barriers) != len(args.refs):
        raise ValueError("Number of --refs must match number of barriers.")

    references: Dict[float, float] = {}
    for barrier, ref_path in zip(barriers, args.refs):
        with Path(ref_path).open("rb") as f:
            ref_payload = pickle.load(f)
        if "I_ref" not in ref_payload:
            raise KeyError(f"{ref_path} does not contain I_ref.")
        references[float(barrier)] = float(ref_payload["I_ref"])

    estimates = payload["estimates"]
    ses = payload["ses"]

    stats_by_barrier: Dict[float, Dict[str, Dict[str, float]]] = {}
    display_tables: Dict[float, List[List[str]]] = {}
    labels = {
        "crude": "crude",
        "stratified_proportional": "stratified (proportional)",
        "stratified_optimal": "stratified (optimal)",
    }
    stat_order = [
        ("MEAN", "mean"),
        ("SD", "std"),
        ("BIAS", "bias"),
        ("MSE", "mse"),
        ("CI-COVER-95%", "ci_coverage_95"),
    ]
    method_keys = [
        "crude",
        "stratified_proportional",
        "stratified_optimal",
    ]
    for barrier in barriers:
        barrier_key = float(barrier)
        stats_by_barrier[barrier_key] = {}
        table = [["METHOD"] + [name for name, _ in stat_order]]
        available_keys = [k for k in method_keys if k in estimates[barrier]]
        for key in available_keys:
            est_arr = np.array(estimates[barrier][key], dtype=float)
            se_arr = np.array(ses[barrier][key], dtype=float)
            stats_row = _compute_stats(
                est_arr, se_arr, references[barrier_key]
            )
            stats_by_barrier[barrier_key][key] = stats_row

            method_name = labels.get(key, key)
            row = [method_name]
            for col_name, stat_key in stat_order:
                value = stats_row[stat_key]
                if stat_key == "ci_coverage_95":
                    row.append(_format_coverage(value))
                elif stat_key in {"std", "bias", "mse"}:
                    row.append(_format_scientific(value))
                else:
                    row.append(_format_default(value))
            table.append(row)
        display_tables[barrier_key] = table

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(
            {
                "parameters": payload["parameters"],
                "barriers": barriers,
                "references": references,
                "stats": stats_by_barrier,
                "display_tables": display_tables,
            },
            f,
        )
    print(f"Stats saved to: {output_path}")


if __name__ == "__main__":
    main()
