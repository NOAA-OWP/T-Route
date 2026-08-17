#!/usr/bin/env python3
"""Tier B kernel microbenchmark for the Muskingum-Cunge perf work.

Replays the compute_network_structured() calls captured by
harvest_kernel_inputs.py straight into the compiled function, over
`--warmup` + `--runs` timed iterations. No IO, no joblib, no config
parsing, no network construction (just the kernel-dominated hot path),
so it is a fast, low-noise speed metric for MC-kernel changes.

`compute_network_structured` is the Cython routine that runs the
timestep loop, the per-reach loop, and every Muskingum-Cunge Fortran
call, so its wall time tracks kernel speed directly.

Correctness here is a SANITY gate (no new NaN/Inf, no gross divergence
from the harvested baseline). The authoritative correctness gate is the
full-output comparison in bench_e2e.py.

Usage:
    python benchmark/bench_kernel.py
    python benchmark/bench_kernel.py --runs 15 --label step1-O3 --json
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
from harvest_kernel_inputs import CALLS_PKL, summarize_result, qlat_from_call  # noqa: E402

RESULTS_DIR = BENCH_DIR / "results"

# Tier B sanity-gate threshold on the sampled-value relative drift.
REL_DRIFT_GATE = 1e-2

# compute_network_structured returns an 11-tuple; element 1 is the
# flowveldepth output (flow/velocity/depth), the kernel's actual product.
# The other elements are data-assimilation / upstream-bookkeeping arrays
# that an isolated single-subnetwork replay does not reproduce, so they are
# not kernel-correctness signals. bench_e2e.py is the authoritative gate.
FLOWVELDEPTH_IDX = 1


def replay_once(calls: list[dict], fn) -> tuple[float, list]:
    """Replay every captured call once on fresh argument copies.

    Args are deep-copied before the timer starts so in-place mutation
    cannot leak between iterations; the copy cost is excluded from the
    measured time.

    Note: because the deepcopy is excluded, the returned (measured) time can
    diverge noticeably from the apparent wall time of this function when the
    harvested arrays are large, the copy can take longer than the kernel
    replay itself. That is intentional: we want the pure kernel time, not the
    setup cost. If you instrument this function's total wall time externally,
    expect it to exceed the reported number.
    """
    fresh = [(copy.deepcopy(c["args"]), copy.deepcopy(c["kwargs"]))
             for c in calls]
    t0 = time.perf_counter()
    results = [fn(*a, **k) for a, k in fresh]
    return time.perf_counter() - t0, results


def compare(
    fresh_results: list, calls: list[dict],
) -> tuple[list[str], float, int]:
    """Compare the replayed flowveldepth output to the harvested baseline.

    Returns:
        lines:           per-call drift summary strings, one per replayed
                         call, ready to print.
        worst_rel:       max sampled relative drift across all calls
                         (NaN values count as 0).
        new_nan_total:   total NaN+Inf entries observed in the replayed
                         output that were finite in the baseline.
    """
    lines: list[str] = []
    worst_rel = 0.0
    new_nan_total = 0
    for ci, (res, call) in enumerate(zip(fresh_results, calls)):
        f = summarize_result(res)[FLOWVELDEPTH_IDX]
        b = call["result_summary"][FLOWVELDEPTH_IDX]
        new_nan = (max(0, f["n_nan"] - b["n_nan"])
                   + max(0, f["n_inf"] - b["n_inf"]))
        fs, bs = np.asarray(f["sample"]), np.asarray(b["sample"])
        if fs.shape == bs.shape and fs.size:
            rel = float(np.nanmax(np.abs(fs - bs)
                                  / np.maximum(np.abs(bs), 1e-6)))
        else:
            rel = float("nan")
        worst_rel = max(worst_rel, 0.0 if np.isnan(rel) else rel)
        new_nan_total += new_nan
        lines.append(f"call{ci} flowveldepth: new_nan/inf={new_nan} "
                     f"sample_rel={rel:.3e}")
    return lines, worst_rel, new_nan_total


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BENCH_DIR,
            text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=15, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=3, help="discarded warmups")
    ap.add_argument("--label", default=None, help="label for the JSON record")
    ap.add_argument("--json", action="store_true", help="write results/<label>.json")
    args = ap.parse_args()

    if not CALLS_PKL.exists():
        sys.exit("ERROR: no kernel_calls.pkl. Run: "
                 "python benchmark/harvest_kernel_inputs.py")
    payload = pickle.load(open(CALLS_PKL, "rb"))
    calls = payload["calls"]
    invocations = sum(int(qlat_from_call(c["args"], c["kwargs"]).size) for c in calls)
    print(f"loaded {len(calls)} call(s) "
          f"({payload['n_calls_kept']}/{payload['n_calls_total']} from harvest), "
          f"{invocations:,} kernel invocations / iteration")

    from troute.routing.fast_reach.mc_reach import compute_network_structured

    samples: list[float] = []
    last_results: list = []
    for i in range(args.warmup + args.runs):
        elapsed, results = replay_once(calls, compute_network_structured)
        if i >= args.warmup:
            samples.append(elapsed)
            last_results = results
        kind = "warmup" if i < args.warmup else "timed "
        print(f"  [{kind} {i:2d}] {elapsed * 1000:9.2f} ms")

    stats = {
        "min": min(samples), "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }
    print(f"\nkernel microbenchmark  ({args.runs} timed, {args.warmup} warmup)")
    print(f"  min={stats['min']*1000:.2f} ms  median={stats['median']*1000:.2f} ms  "
          f"mean={stats['mean']*1000:.2f} ms  stdev={stats['stdev']*1000:.3f} ms")

    report, worst_rel, new_nan = compare(last_results, calls)
    if new_nan:
        status = "FAIL (new NaN/Inf)"
    elif worst_rel > REL_DRIFT_GATE:
        status = "WARN (drift > gate)"
    else:
        status = "PASS"
    print(f"\ncorrectness sanity: {status}  "
          f"(worst sampled rel drift {worst_rel:.3e}, gate {REL_DRIFT_GATE:.0e})")
    for line in report:
        print(f"  {line}")
    print("  reading the numbers: the ms above are the kernel-only replay time")
    print(f"  per iteration over the harvested {invocations:,} MC-kernel")
    print("  invocations, with no IO/joblib/Python pipeline, so a lower value")
    print("  means a faster kernel. 'rel drift' is a sanity check of the replayed")
    print("  output vs the harvested baseline (should be ~0 for an unchanged")
    print("  kernel); bench_e2e.py is the authoritative full-output gate.")

    print("\nRESULTS.md kernel column:")
    print(f"  median {stats['median']*1000:.2f} ms   ({status})")

    if args.json:
        RESULTS_DIR.mkdir(exist_ok=True)
        label = args.label or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        rec = {"label": label, "git_sha": _git_sha(),
               "utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
               "runs": args.runs, "warmup": args.warmup,
               "invocations_per_iter": invocations,
               "ms": {k: v * 1000 for k, v in stats.items()},
               "worst_rel_drift": worst_rel, "status": status}
        (RESULTS_DIR / f"{label}.kernel.json").write_text(json.dumps(rec, indent=2))
        print(f"json -> {RESULTS_DIR / (label + '.kernel.json')}")

    return 0 if not new_nan else 1


if __name__ == "__main__":
    raise SystemExit(main())
