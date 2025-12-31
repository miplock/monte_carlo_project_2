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

    stats: Dict[str, Dict[str, float]] = {}
    for key in estimates:
        est_arr = np.array(estimates[key], dtype=float)
        se_arr = np.array(ses[key], dtype=float)
        stats[key] = _compute_stats(est_arr, se_arr, true_value)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(
            {
                "true_value": true_value,
                "parameters": params,
                "stats": stats,
            },
            f,
        )
    print(f"Stats saved to: {output_path}")


if __name__ == "__main__":
    main()
