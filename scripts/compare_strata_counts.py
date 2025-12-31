import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import stratified_monte_carlo
from models import black_scholes_call
from parameters import K, R as DEFAULT_R, S0, T, mu_star, n_steps, r, seed, sigma


@dataclass
class Summary:
    m: int
    bias: float
    mae: float
    rmse: float
    mean_est: float
    mean_se: float
    std_est: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stratified Monte Carlo for different numbers of strata m."
        )
    )
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="List of m values (strata counts) to test.",
    )
    parser.add_argument(
        "--R",
        type=int,
        default=DEFAULT_R,
        help="Number of paths per run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of repeated runs per m (different seeds).",
    )
    parser.add_argument(
        "--allocation",
        choices=["proportional", "optimal"],
        default="proportional",
        help="Allocation strategy for stratified sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="Base seed for reproducibility (default: parameters.py seed).",
    )
    parser.add_argument(
        "--output",
        default="data/strata_stats.pkl",
        help="Output .pkl path for saved statistics.",
    )
    return parser.parse_args()


def run_for_m(
    m: int,
    R_value: int,
    runs: int,
    allocation: str,
    reference: float,
    base_seed: int,
) -> tuple[Summary, List[float]]:
    estimates: List[float] = []
    ses: List[float] = []
    errors: List[float] = []

    for idx in range(runs):
        run_seed = base_seed + idx
        est, se, _ = stratified_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=float("inf"),
            T=T,
            n_steps=n_steps,
            m=m,
            R=R_value,
            seed=run_seed,
            allocation=allocation,
        )
        estimates.append(est)
        ses.append(se)
        errors.append(est - reference)

    mean_est = float(np.mean(estimates))
    mean_se = float(np.mean(ses))
    std_est = float(np.std(estimates, ddof=1)) if runs > 1 else 0.0
    bias = float(np.mean(errors))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    summary = Summary(
        m=m,
        bias=bias,
        mae=mae,
        rmse=rmse,
        mean_est=mean_est,
        mean_se=mean_se,
        std_est=std_est,
    )
    return summary, errors


def main() -> None:
    args = parse_args()
    base_seed = 0 if args.seed is None else int(args.seed)
    reference = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    print(
        "Stratified Monte Carlo comparison\n"
        f"- reference (Black-Scholes call): {reference:.6f}\n"
        f"- R per run: {args.R}\n"
        f"- runs per m: {args.runs}\n"
        f"- allocation: {args.allocation}\n"
        f"- base seed: {base_seed}\n"
        f"- m values: {args.m_values}\n"
    )

    summaries = []
    errors_by_m = {}
    for m in args.m_values:
        if m > args.R:
            raise ValueError("m cannot exceed R.")
        summary, errors = run_for_m(
            m,
            args.R,
            args.runs,
            args.allocation,
            reference,
            base_seed,
        )
        summaries.append(summary)
        errors_by_m[m] = errors

    print("m | mean_est | mean_se | std_est | bias | mae | rmse")
    print("--+----------+---------+--------+------+-----+------")
    for s in summaries:
        print(
            f"{s.m:2d} | {s.mean_est:8.6f} | {s.mean_se:7.6f} | "
            f"{s.std_est:7.6f} | {s.bias:5.3f} | {s.mae:5.3f} | {s.rmse:6.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": reference,
        "R": args.R,
        "runs": args.runs,
        "allocation": args.allocation,
        "base_seed": base_seed,
        "m_values": args.m_values,
        "summaries": [s.__dict__ for s in summaries],
        "errors_by_m": errors_by_m,
    }
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(payload, f)
    print(f"Stats saved to: {output_path}")


if __name__ == "__main__":
    main()
