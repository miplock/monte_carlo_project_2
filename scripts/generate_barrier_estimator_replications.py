import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import crude_monte_carlo, stratified_monte_carlo
from parameters import (
    K,
    R as DEFAULT_R,
    S0,
    T,
    mu_star,
    n_steps,
    r,
    seed,
    sigma,
    m,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate M replications for crude and stratified estimators "
            "for barrier options with n_steps from parameters.py."
        )
    )
    parser.add_argument(
        "--M",
        type=int,
        default=100,
        help="Number of replications per estimator and barrier.",
    )
    parser.add_argument(
        "--R",
        type=int,
        default=DEFAULT_R,
        help="Number of paths per replication.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=m,
        help="Number of strata for stratified estimators.",
    )
    parser.add_argument(
        "--barriers",
        nargs="+",
        type=float,
        default=[105.0, 120.0],
        help="Barrier levels C to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="Base seed for reproducibility (default: parameters.py seed).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pkl path for saved replications.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_seed = 0 if args.seed is None else int(args.seed)

    output_name = (
        Path(f"data/barrier_estimator_replications_M{args.M}.pkl")
        if args.output is None
        else Path(args.output)
    )

    estimates: Dict[float, Dict[str, List[float]]] = {}
    ses: Dict[float, Dict[str, List[float]]] = {}

    total_tasks = args.M * len(args.barriers)
    start = time.perf_counter()
    done = 0

    for barrier in args.barriers:
        estimates[barrier] = {
            "crude": [],
            "stratified_proportional": [],
            "stratified_optimal": [],
        }
        ses[barrier] = {key: [] for key in estimates[barrier]}

        for idx in range(args.M):
            run_seed = base_seed + idx

            est, se, _ = crude_monte_carlo(
                S0=S0,
                mu_star=mu_star,
                sigma=sigma,
                r=r,
                K=K,
                C=barrier,
                T=T,
                n_steps=n_steps,
                R=args.R,
                seed=run_seed,
            )
            estimates[barrier]["crude"].append(est)
            ses[barrier]["crude"].append(se)

            est, se, _ = stratified_monte_carlo(
                S0=S0,
                mu_star=mu_star,
                sigma=sigma,
                r=r,
                K=K,
                C=barrier,
                T=T,
                n_steps=n_steps,
                m=args.m,
                R=args.R,
                seed=run_seed,
                allocation="proportional",
            )
            estimates[barrier]["stratified_proportional"].append(est)
            ses[barrier]["stratified_proportional"].append(se)

            est, se, _ = stratified_monte_carlo(
                S0=S0,
                mu_star=mu_star,
                sigma=sigma,
                r=r,
                K=K,
                C=barrier,
                T=T,
                n_steps=n_steps,
                m=args.m,
                R=args.R,
                seed=run_seed,
                allocation="optimal",
            )
            estimates[barrier]["stratified_optimal"].append(est)
            ses[barrier]["stratified_optimal"].append(se)

            done += 1
            elapsed = time.perf_counter() - start
            avg = elapsed / done
            eta = avg * (total_tasks - done)
            pct = 100.0 * done / total_tasks
            sys.stdout.write(
                f"\rProgress: {done}/{total_tasks} ({pct:5.1f}%) | "
                f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    total_elapsed = time.perf_counter() - start

    payload = {
        "parameters": {
            "S0": S0,
            "mu_star": mu_star,
            "sigma": sigma,
            "r": r,
            "K": K,
            "T": T,
            "n_steps": n_steps,
            "R": args.R,
            "m": args.m,
        },
        "barriers": args.barriers,
        "M": args.M,
        "base_seed": base_seed,
        "estimates": estimates,
        "ses": ses,
    }

    output_name.parent.mkdir(parents=True, exist_ok=True)
    with output_name.open("wb") as f:
        import pickle

        pickle.dump(payload, f)
    print(f"Replications saved to: {output_name}")

    timing_output = output_name.with_name(f"{output_name.stem}_timing.pkl")
    timing_payload = {
        "output_data_file": str(output_name),
        "M": args.M,
        "barriers": args.barriers,
        "total_tasks": total_tasks,
        "elapsed_seconds": total_elapsed,
        "avg_seconds_per_task": total_elapsed / total_tasks if total_tasks else 0.0,
    }
    with timing_output.open("wb") as f:
        import pickle

        pickle.dump(timing_payload, f)
    print(f"Timing saved to: {timing_output}")


if __name__ == "__main__":
    main()
