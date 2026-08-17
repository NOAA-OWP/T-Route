#!/usr/bin/env python3
"""Tier B harvest: capture real compute_network_structured() calls.

`compute_network_structured` (Cython, mc_reach.pyx) is the kernel-dominated
hot path: it runs the timestep loop, the per-reach loop, and every
Muskingum-Cunge Fortran call. This script runs one max_loop chunk of the
nhf_subset_ohio case in-process, intercepts each `compute_network_structured` call,
and pickles its arguments + return value.

bench_kernel.py then replays those calls straight into the compiled
function for a fast, IO-free, low-noise, kernel-isolated microbenchmark.

Interception: troute.routing.compute dispatches the kernel through the
`_compute_func_map` dict, so we swap the "V02-structured" entry for a
recording wrapper. cpu_pool=1 (pinned in nhf_subset_ohio.yaml) keeps joblib on the
sequential backend, so the wrapper runs in-process and the capture works.

Usage:
    python benchmark/harvest_kernel_inputs.py
    python benchmark/harvest_kernel_inputs.py --nts 288
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
from bench_e2e import resolved_config  # noqa: E402

DATA_DIR = BENCH_DIR / "data"
CALLS_PKL = DATA_DIR / "kernel_calls.pkl"


def sample_indices(size: int, n: int = 4096) -> np.ndarray:
    """Deterministic, shape-derived indices for a result-array sample."""
    if size <= n:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, n).astype(np.int64)


def summarize_result(result) -> list[dict]:
    """Compact summary of a compute_network_structured() return value.

    The full per-timestep output is ~88 MB/call; storing it would bloat
    the pickle. Tier B's correctness check is a sanity gate (no new
    NaN/Inf, no gross divergence); bench_e2e.py is the authoritative
    full-output correctness gate. We keep only per-array stats plus a
    deterministic value sample for a quantitative drift estimate.
    """
    summary = []
    for elem in result:
        if isinstance(elem, np.ndarray) and elem.size:
            flat = elem.ravel()
            is_float = elem.dtype.kind == "f"
            finite = np.isfinite(flat) if is_float else np.ones(flat.size, bool)
            summary.append({
                "kind": "ndarray",
                "shape": tuple(elem.shape),
                "dtype": str(elem.dtype),
                "n_nan": int(np.isnan(flat).sum()) if is_float else 0,
                "n_inf": int(np.isinf(flat).sum()) if is_float else 0,
                "sum": float(flat[finite].sum()) if finite.any() else 0.0,
                "min": float(flat[finite].min()) if finite.any() else 0.0,
                "max": float(flat[finite].max()) if finite.any() else 0.0,
                "sample": flat[sample_indices(flat.size)].astype(np.float64),
            })
        else:
            summary.append({"kind": "other", "repr": repr(elem)[:80]})
    return summary


def qlat_from_call(call_args, call_kwargs):
    """Return the kernel's qlat_values argument regardless of call convention.

    Legacy compute.py passes the kernel inputs positionally (qlat_values is
    index 9). The refactored compute.py calls ``compute_func(**vars(package))``,
    so every argument arrives in call_kwargs and call_args is empty.
    """
    if len(call_args) > 9:
        return call_args[9]
    return call_kwargs.get("qlat_values")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nts", type=int, default=288,
                    help="timesteps to harvest (default 288 = one 24h chunk)")
    ap.add_argument("--max-calls", type=int, default=3,
                    help="keep the N largest compute_network_structured calls "
                         "(default 3) to bound the pickle size")
    args = ap.parse_args()

    if not (DATA_DIR / "MANIFEST.json").exists():
        sys.exit("ERROR: benchmark dataset missing. Run prep_ohio_data.py first.")

    tmpdir = Path(tempfile.mkdtemp(prefix="harvest_"))
    out_dir = tmpdir / "out"
    out_dir.mkdir()
    config = resolved_config(out_dir, args.nts)

    # Import compute, then swap the dispatch-map entry for a recorder.
    # `by-subnetwork-jit-clustered` makes one call per subnetwork; we keep
    # only the N largest (by kernel work = qlat_values.size) so the pickle
    # stays bounded while still exercising a representative input range.
    import troute.routing.compute as compute_mod
    captured: list[tuple[int, dict]] = []  # (work, snapshot), largest-first
    n_seen = 0
    original = compute_mod._compute_func_map["V02-structured"]

    def recording_wrapper(*call_args, **call_kwargs):
        nonlocal n_seen
        n_seen += 1
        qlat = qlat_from_call(call_args, call_kwargs)  # (segments, chunk-steps)
        work = int(getattr(qlat, "size", 0))
        smallest_kept = min((c[0] for c in captured), default=-1)
        keep = len(captured) < args.max_calls or work > smallest_kept
        # Snapshot inputs BEFORE the call (it may mutate args in place).
        snap_args = copy.deepcopy(call_args) if keep else None
        snap_kwargs = copy.deepcopy(call_kwargs) if keep else None
        result = original(*call_args, **call_kwargs)
        if keep:
            captured.append((work, {"args": snap_args, "kwargs": snap_kwargs,
                                    "result_summary": summarize_result(result)}))
            captured.sort(key=lambda c: c[0], reverse=True)
            del captured[args.max_calls:]
        return result

    compute_mod._compute_func_map["V02-structured"] = recording_wrapper

    print(f"Harvesting compute_network_structured calls (nts={args.nts}) ...")
    from nwm_routing.nhf_routing import nhf_routing
    nhf_routing(["-f", str(config)])

    if not captured:
        sys.exit("ERROR: no compute_network_structured calls were captured "
                 "(is cpu_pool=1 and compute_kernel=V02-structured?)")

    calls = [snap for _work, snap in captured]
    DATA_DIR.mkdir(exist_ok=True)
    payload = {"nts": args.nts, "n_calls_total": n_seen,
               "n_calls_kept": len(calls), "calls": calls}
    with open(CALLS_PKL, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = CALLS_PKL.stat().st_size / (1024 * 1024)
    for i, (work, snap) in enumerate(captured):
        qlat = qlat_from_call(snap["args"], snap["kwargs"])
        print(f"  kept call {i}: qlat_values shape={getattr(qlat, 'shape', '?')}"
              f"  ({work} kernel invocations)")
    print(f"\nKept {len(calls)}/{n_seen} call(s) -> {CALLS_PKL} ({size_mb:.1f} MB)")
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
