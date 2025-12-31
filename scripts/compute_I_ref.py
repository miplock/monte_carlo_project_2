import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import simulate_gbm_paths
from parameters import C as DEFAULT_C, K, R as DEFAULT_R, S0, T, mu_star, n_steps, r, seed, sigma
from payoffs import payoff_up_and_out_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute I_ref using crude Monte Carlo with a large R_ref, "
            "processing paths in chunks."
        )
    )
    parser.add_argument(
        "--R-ref",
        type=int,
        default=10_000_000,
        help="Total number of Monte Carlo paths.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of paths per chunk (controls memory use).",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_C,
        help="Barrier level (use inf for European).",
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
        help="Output .pkl path for saved I_ref.",
    )
    return parser.parse_args()


def _combine_moments(
    n_total: int, mean_total: float, m2_total: float, values: np.ndarray
) -> tuple[int, float, float]:
    n_chunk = len(values)
    if n_chunk == 0:
        return n_total, mean_total, m2_total

    mean_chunk = float(values.mean())
    var_chunk = float(values.var(ddof=0))

    if n_total == 0:
        return n_chunk, mean_chunk, var_chunk * n_chunk

    delta = mean_chunk - mean_total
    n_new = n_total + n_chunk
    mean_new = mean_total + delta * n_chunk / n_new
    m2_new = (
        m2_total
        + var_chunk * n_chunk
        + delta * delta * n_total * n_chunk / n_new
    )
    return n_new, mean_new, m2_new


def main() -> None:
    args = parse_args()
    base_seed = 0 if args.seed is None else int(args.seed)

    r_ref = int(args.R_ref)
    exp = int(round(np.log10(r_ref))) if r_ref > 0 else 0
    if r_ref > 0 and 10**exp == r_ref:
        r_label = f"10^{exp}"
    else:
        r_label = str(r_ref)

    if args.output is None:
        barrier_label = "inf" if np.isinf(args.C) else f"{args.C:g}"
        output_name = Path(
            f"data/I_ref_crude_C{barrier_label}_R{r_label}.pkl"
        )
        timing_name = Path(
            f"data/I_ref_crude_C{barrier_label}_R{r_label}_time.pkl"
        )
    else:
        output_name = Path(args.output)
        timing_name = output_name.with_name(
            f"{output_name.stem}_time{output_name.suffix}"
        )

    total_paths = r_ref
    chunk_size = int(args.chunk_size)
    if total_paths <= 0 or chunk_size <= 0:
        raise ValueError("R_ref and chunk_size must be positive.")

    n_total = 0
    mean_total = 0.0
    m2_total = 0.0

    start = time.perf_counter()
    processed = 0
    chunk_idx = 0

    while processed < total_paths:
        current = min(chunk_size, total_paths - processed)
        run_seed = base_seed + chunk_idx

        S_paths = simulate_gbm_paths(
            S0, mu_star, sigma, T, n_steps, current, run_seed
        )
        payoffs = payoff_up_and_out_call(S_paths, K, r, args.C, T)

        n_total, mean_total, m2_total = _combine_moments(
            n_total, mean_total, m2_total, payoffs
        )

        processed += current
        chunk_idx += 1

        elapsed = time.perf_counter() - start
        avg = elapsed / processed
        eta = avg * (total_paths - processed)
        pct = 100.0 * processed / total_paths
        sys.stdout.write(
            f"\rProgress: {processed}/{total_paths} ({pct:5.1f}%) | "
            f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s"
        )
        sys.stdout.flush()

    sys.stdout.write("\n")

    if n_total < 2:
        var = 0.0
    else:
        var = m2_total / (n_total - 1)
    se = float(np.sqrt(var / n_total)) if n_total > 0 else 0.0

    payload = {
        "I_ref": float(mean_total),
        "se": se,
        "var": float(var),
        "R_ref": total_paths,
        "chunk_size": chunk_size,
        "base_seed": base_seed,
        "parameters": {
            "S0": S0,
            "mu_star": mu_star,
            "sigma": sigma,
            "r": r,
            "K": K,
            "T": T,
            "C": args.C,
            "n_steps": n_steps,
        },
    }

    total_elapsed = time.perf_counter() - start

    output_name.parent.mkdir(parents=True, exist_ok=True)
    with output_name.open("wb") as f:
        import pickle

        pickle.dump(payload, f)
    print(f"I_ref saved to: {output_name}")

    timing_payload = {
        "seconds": float(total_elapsed),
        "R_ref": total_paths,
        "C": float(args.C),
        "chunk_size": chunk_size,
        "base_seed": base_seed,
    }
    with timing_name.open("wb") as f:
        import pickle

        pickle.dump(timing_payload, f)
    print(f"Timing saved to: {timing_name}")


if __name__ == "__main__":
    main()
