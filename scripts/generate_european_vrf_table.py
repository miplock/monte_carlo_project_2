import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VRF table for European estimator replications."
    )
    parser.add_argument(
        "--input",
        default="data/european_estimator_replications_M1000.pkl",
        help="Input .pkl with estimator replications.",
    )
    parser.add_argument(
        "--output",
        default="data/european_vrf_table.pkl",
        help="Output .pkl path for saved VRF table.",
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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("rb") as f:
        import pickle

        payload = pickle.load(f)

    estimates: Dict[str, List[float]] = payload["estimates"]
    method_keys = [
        "crude",
        "stratified_proportional",
        "stratified_optimal",
        "antithetic",
        "control",
    ]
    method_keys = [k for k in method_keys if k in estimates]
    labels = [_label_for_estimator(k) for k in method_keys]

    variances = {}
    for key in method_keys:
        values = np.array(estimates[key], dtype=float)
        variances[key] = float(values.var(ddof=1))

    table: List[List[str]] = []
    for i, key_i in enumerate(method_keys):
        row: List[str] = []
        var_i = variances[key_i]
        for j, key_j in enumerate(method_keys):
            if i > j:
                row.append("-")
            elif i == j:
                row.append("1")
            else:
                var_j = variances[key_j]
                if var_j == 0.0:
                    row.append("&infin;")
                else:
                    row.append(f"{var_i / var_j:.3f}")
        table.append(row)

    display_table = [["VRF = VAR(ROW) / VAR(COL)"] + labels]
    for i, method in enumerate(labels):
        display_table.append([method] + table[i])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(
            {
                "methods": labels,
                "variances": {labels[i]: variances[k] for i, k in enumerate(method_keys)},
                "vrf_table": table,
                "display_table": display_table,
            },
            f,
        )
    print(f"VRF table saved to: {output_path}")


if __name__ == "__main__":
    main()
