import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VRF tables for barrier estimator replications."
    )
    parser.add_argument(
        "--input",
        default="data/barrier_estimator_replications_M1000.pkl",
        help="Input .pkl with barrier estimator replications.",
    )
    parser.add_argument(
        "--output",
        default="data/barrier_vrf_tables.pkl",
        help="Output .pkl path for saved VRF tables.",
    )
    return parser.parse_args()


def _label_for_estimator(key: str) -> str:
    labels = {
        "crude": "Crude",
        "stratified_proportional": "Stratified (proportional)",
        "stratified_optimal": "Stratified (optimal)",
    }
    return labels.get(key, key)


def _build_table(estimates: Dict[str, List[float]]) -> Dict[str, List[List[str]]]:
    method_keys = [
        "crude",
        "stratified_proportional",
        "stratified_optimal",
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

    return {
        "methods": labels,
        "variances": {labels[i]: variances[k] for i, k in enumerate(method_keys)},
        "vrf_table": table,
        "display_table": display_table,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("rb") as f:
        import pickle

        payload = pickle.load(f)

    barriers = payload["barriers"]
    estimates = payload["estimates"]

    tables_by_barrier = {}
    for barrier in barriers:
        tables_by_barrier[float(barrier)] = _build_table(estimates[barrier])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(
            {
                "barriers": barriers,
                "tables": tables_by_barrier,
            },
            f,
        )
    print(f"VRF tables saved to: {output_path}")


if __name__ == "__main__":
    main()
