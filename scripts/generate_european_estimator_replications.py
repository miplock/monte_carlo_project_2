import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import (
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    crude_monte_carlo,
    stratified_monte_carlo,
)
from parameters import K, R as DEFAULT_R, S0, T, mu_star, n_steps, r, seed, sigma, m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate M replications for European option estimators "
            "(n_steps=1, C=inf) and save them to a .pkl file."
        )
    )
    parser.add_argument(
        "--M",
        type=int,
        default=50,
        help="Number of replications per estimator.",
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


def _ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def main() -> None:
    args = parse_args()
    base_seed = 0 if args.seed is None else int(args.seed)

    C = float("inf")
    steps = 1

    R_even = _ensure_even(args.R)

    output_name = (
        Path(f"data/european_estimator_replications_M{args.M}.pkl")
        if args.output is None
        else Path(args.output)
    )

    estimates: Dict[str, List[float]] = {
        "crude": [],
        "stratified_proportional": [],
        "stratified_optimal": [],
        "antithetic": [],
        "control": [],
    }
    ses: Dict[str, List[float]] = {
        key: [] for key in estimates
    }

    start = time.perf_counter()
    for idx in range(args.M):
        run_seed = base_seed + idx

        est, se, _ = crude_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=steps,
            R=args.R,
            seed=run_seed,
        )
        estimates["crude"].append(est)
        ses["crude"].append(se)

        est, se, _ = stratified_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=steps,
            m=args.m,
            R=args.R,
            seed=run_seed,
            allocation="proportional",
        )
        estimates["stratified_proportional"].append(est)
        ses["stratified_proportional"].append(se)

        est, se, _ = stratified_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            C=C,
            T=T,
            n_steps=steps,
            m=args.m,
            R=args.R,
            seed=run_seed,
            allocation="optimal",
        )
        estimates["stratified_optimal"].append(est)
        ses["stratified_optimal"].append(se)

        est, se, _ = antithetic_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            T=T,
            R=R_even,
            seed=run_seed,
        )
        estimates["antithetic"].append(est)
        ses["antithetic"].append(se)

        est, se, _ = control_variate_monte_carlo(
            S0=S0,
            mu_star=mu_star,
            sigma=sigma,
            r=r,
            K=K,
            T=T,
            R=args.R,
            seed=run_seed,
        )
        estimates["control"].append(est)
        ses["control"].append(se)

        elapsed = time.perf_counter() - start
        done = idx + 1
        avg = elapsed / done
        eta = avg * (args.M - done)
        pct = 100.0 * done / args.M
        sys.stdout.write(
            f"\rProgress: {done}/{args.M} ({pct:5.1f}%) | "
            f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s"
        )
        sys.stdout.flush()

    sys.stdout.write("\n")

    payload = {
        "parameters": {
            "S0": S0,
            "mu_star": mu_star,
            "sigma": sigma,
            "r": r,
            "K": K,
            "T": T,
            "C": C,
            "n_steps": steps,
            "R": args.R,
            "R_even": R_even,
            "m": args.m,
        },
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


if __name__ == "__main__":
    main()
