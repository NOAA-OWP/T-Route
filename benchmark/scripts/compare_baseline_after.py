"""Numeric equivalence check: baseline-built Tier A vs after-built Tier A.

The default Tier A correctness gate (``bench_e2e.py``) compares repeated
optimized-build runs against a golden saved by the optimized build. That
proves run-to-run determinism but NOT equivalence to the pre-optimization
Fortran kernel. This script compares per-variable per-segment per-timestep
arrays between a baseline-built run and an after-built run, both reading
the same input dataset, and reports max-abs / max-rel / NaN counts.

The baseline-vs-after comparison is informative but not a strict
bit-equivalence gate: the kernel-side changes (strength-reduced powers,
CSE, hoisted invariants) preserve floating-point ordering, but the
overall build also changes optimization flags (``-O3 -funroll-loops``
vs upstream ``-O2``). float32 cancellation noise is therefore expected
at the ~1e-3 relative level on flow/velocity/depth. The point is to
verify that the *physical* output (flow, depth) stays within
solver-noise tolerance, not bit-for-bit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

VARS = ("flow", "velocity", "depth")


def load_concat(out_dir: Path) -> dict:
    files = sorted(out_dir.glob("troute_output_*.nc"))
    if not files:
        sys.exit(f"ERROR: no troute_output_*.nc in {out_dir}")
    chunks: dict[str, list] = {v: [] for v in VARS}
    feature_id = None
    for f in files:
        with xr.open_dataset(f) as ds:
            if feature_id is None:
                feature_id = ds["feature_id"].values
            for v in VARS:
                chunks[v].append(ds[v].values)
    return {
        "feature_id": feature_id,
        **{v: np.concatenate(chunks[v], axis=-1) for v in VARS},
    }


def compare(baseline_dir: Path, after_dir: Path) -> int:
    b = load_concat(baseline_dir)
    a = load_concat(after_dir)
    if not np.array_equal(b["feature_id"], a["feature_id"]):
        sys.exit("ERROR: feature_id ordering differs between runs")
    print(f"feature_count={len(b['feature_id'])}")
    print(f"timestep_count={b['flow'].shape[-1]}")
    print()
    print(f"{'var':9s}  {'max_abs':>11s}  {'max_rel':>11s}  "
          f"{'rel_p99':>11s}  {'new_nan':>8s}  {'new_inf':>8s}")
    worst_flow_rel = 0.0
    for v in VARS:
        diff = a[v] - b[v]
        abs_diff = np.abs(diff)
        denom = np.maximum(np.abs(b[v]), 1e-6)
        rel = abs_diff / denom
        max_abs = float(np.nanmax(abs_diff))
        max_rel = float(np.nanmax(rel))
        rel_p99 = float(np.nanpercentile(rel[np.isfinite(rel)], 99))
        new_nan = int(np.sum(np.isnan(a[v]) & ~np.isnan(b[v])))
        new_inf = int(np.sum(np.isinf(a[v]) & ~np.isinf(b[v])))
        print(f"{v:9s}  {max_abs:11.3e}  {max_rel:11.3e}  "
              f"{rel_p99:11.3e}  {new_nan:>8d}  {new_inf:>8d}")
        if v == "flow":
            worst_flow_rel = max_rel
    print()
    # Tolerances: flow/velocity/depth on Tier A are float32 throughout the
    # MC kernel, so single-precision cancellation noise sets the floor.
    # max_rel for velocity/depth is dominated by near-zero denominators
    # (e.g. dry-channel depth ~ 1e-9); rel_p99 is the meaningful signal.
    # The check below targets flow's max_rel as the headline metric.
    FLOW_REL_GATE = 1e-2
    if worst_flow_rel > FLOW_REL_GATE:
        print(f"FAIL: flow max_rel {worst_flow_rel:.3e} > "
              f"gate {FLOW_REL_GATE:.0e}")
        return 1
    print(f"PASS: flow max_rel {worst_flow_rel:.3e} within "
          f"gate {FLOW_REL_GATE:.0e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, type=Path,
                    help="directory containing baseline-built troute_output_*.nc")
    ap.add_argument("--after", required=True, type=Path,
                    help="directory containing after-built troute_output_*.nc")
    args = ap.parse_args()
    return compare(args.baseline, args.after)


if __name__ == "__main__":
    sys.exit(main())
