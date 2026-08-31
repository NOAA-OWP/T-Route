#!/usr/bin/env python3
"""Compare two benchmark runs (baseline vs candidate) and flag regressions.

Reads the JSON produced by bench_e2e.py / bench_kernel.py / bench_conus.py for
a baseline tag and a candidate tag, then gates on:

  * performance: candidate wall / kernel time vs baseline (default: fail if
                 the candidate is more than 5 percent slower)
  * accuracy:    candidate Tier A output drift vs the baseline's own output
                 (default: fail on any new NaN or > 1e-3 relative drift)
  * memory:      (Tier C only) candidate peak tree PSS vs baseline

Exits non-zero if any gate fails, so it can act as a pre-PR / CI check.
regression_check.sh wires the two runs up automatically; this script is the
pure comparison step and also runs by hand on any two existing result tags.

Reading the ratio column: time rows show speedup = baseline / candidate
(higher is faster, > 1.0 means the candidate is faster). The memory row shows
growth = candidate / baseline (higher means more memory, so lower is better).

Usage:
    python benchmark/compare_runs.py --baseline regress-base --candidate regress-cand
    python benchmark/compare_runs.py --baseline regress-base --candidate regress-cand \
        --conus --max-slowdown 1.03 --max-rel 1e-9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
VARS = ("flow", "velocity", "depth")


def _load(results_dir: Path, tag: str, suffix: str):
    p = results_dir / f"{tag}-{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    help="baseline result tag (e.g. regress-base)")
    ap.add_argument("--candidate", required=True,
                    help="candidate result tag (e.g. regress-cand)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--max-slowdown", type=float, default=1.05,
                    help="fail if candidate/baseline time exceeds this "
                         "(default 1.05, i.e. 5 percent slower)")
    ap.add_argument("--max-rel", type=float, default=1e-3,
                    help="fail if Tier A relative drift vs baseline exceeds this "
                         "(default 1e-3)")
    ap.add_argument("--max-mem-growth", type=float, default=1.05,
                    help="(Tier C) fail if candidate peak PSS exceeds baseline "
                         "by this factor (default 1.05)")
    ap.add_argument("--conus", action="store_true",
                    help="also compare Tier C (CONUS) wall + memory")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    fails: list[str] = []
    warns: list[str] = []
    infos: list[str] = []
    rows: list[tuple[str, str, str, str, str]] = []

    def perf_row(check: str, base, cand, unit: str) -> None:
        if base is None or cand is None:
            rows.append((check, "-", "-", "-", "MISSING"))
            warns.append(f"{check}: missing result JSON")
            return
        speedup = base / cand if cand else float("inf")   # > 1 => faster
        slow = cand / base if base else float("inf")       # > 1 => slower
        if slow > args.max_slowdown:
            verdict = "FAIL slower"
            fails.append(f"{check}: candidate {slow:.2f}x baseline time "
                         f"(gate {args.max_slowdown:.2f}x)")
        elif slow > 1.0:
            verdict = "ok (slower)"
        else:
            verdict = "ok (faster)"
        rows.append((check, f"{base:.2f}{unit}", f"{cand:.2f}{unit}",
                     f"{speedup:.2f}x", verdict))

    # --- performance: Tier A wall, Tier B kernel, optional Tier C wall ---
    ba, ca = _load(rd, args.baseline, "A"), _load(rd, args.candidate, "A")
    perf_row("Tier A wall",
             ba["stats"]["wall_s"]["median"] if ba else None,
             ca["stats"]["wall_s"]["median"] if ca else None, " s")

    bb, cb = _load(rd, args.baseline, "B.kernel"), _load(rd, args.candidate, "B.kernel")
    perf_row("Tier B kernel",
             bb["ms"]["median"] if bb else None,
             cb["ms"]["median"] if cb else None, " ms")

    if args.conus:
        bc, cc = _load(rd, args.baseline, "C.conus"), _load(rd, args.candidate, "C.conus")
        perf_row("Tier C wall",
                 bc["metrics"]["wall_s"] if bc else None,
                 cc["metrics"]["wall_s"] if cc else None, " s")
        if bc and cc:
            bm = bc["metrics"]["peak_tree_pss_mb"] / 1024
            cm = cc["metrics"]["peak_tree_pss_mb"] / 1024
            grow = cm / bm if bm else float("inf")
            degraded = bc["metrics"].get("pss_degraded") or cc["metrics"].get("pss_degraded")
            if grow > args.max_mem_growth:
                verdict = "FAIL grew"
                fails.append(f"Tier C PSS: candidate {grow:.2f}x baseline "
                             f"(gate {args.max_mem_growth:.2f}x)")
            else:
                verdict = "ok"
            note = " [RSS]" if degraded else ""
            rows.append(("Tier C peak PSS", f"{bm:.1f} GB", f"{cm:.1f} GB",
                         f"{grow:.2f}x{note}", verdict))

    # --- accuracy: candidate Tier A output vs the baseline's own output ---
    # The relative-drift gate is on FLOW only: it is the conserved routing
    # output and well-conditioned. velocity/depth relative error explodes at
    # near-dry nodes (tiny denominators) even when two builds are physically
    # identical, so they are reported for information and gated only on new NaN.
    def _rel(v):
        return float((corr.get(v, {}) or {}).get("max_rel", 0.0) or 0.0)

    if ca is None:
        rows.append(("Accuracy (flow rel)", "-", "-", "-", "MISSING"))
        warns.append("Accuracy: candidate Tier A JSON missing")
    else:
        corr = ca.get("correctness") or {}
        if not corr:
            rows.append(("Accuracy (flow rel)", "-", "-", "-", "N/A"))
            warns.append("Accuracy: candidate has no correctness report; was the "
                         "baseline output captured as the golden reference?")
        else:
            new_nan = sum(int((corr.get(v, {}) or {}).get("new_nan", 0) or 0)
                          for v in VARS)
            flow_rel = _rel("flow")
            if new_nan > 0:
                verdict = "FAIL new NaN"
                fails.append(f"Accuracy: {new_nan} new NaN/Inf vs baseline output")
            elif flow_rel > args.max_rel:
                verdict = "FAIL drift"
                fails.append(f"Accuracy: flow rel drift {flow_rel:.2e} > gate "
                             f"{args.max_rel:.0e}")
            elif flow_rel > 0:
                verdict = "warn (changed)"
                warns.append(f"Accuracy: flow changed vs baseline "
                             f"(rel {flow_rel:.2e}); confirm this is intended")
            else:
                verdict = "ok (identical)"
            rows.append(("Accuracy (flow rel)", "reference",
                         f"{flow_rel:.1e} (NaN {new_nan})", "-", verdict))
            infos.append(f"velocity/depth rel drift (informational, gated only on "
                         f"NaN): velocity {_rel('velocity'):.1e}, depth "
                         f"{_rel('depth'):.1e}")

    # --- report ---
    print(f"\nregression check: baseline='{args.baseline}'  "
          f"candidate='{args.candidate}'")
    print(f"gates: slowdown <= {args.max_slowdown:.2f}x | rel drift <= "
          f"{args.max_rel:.0e} | mem growth <= {args.max_mem_growth:.2f}x\n")
    w = max(len(r[0]) for r in rows)
    print(f"  {'check'.ljust(w)}  {'baseline':>12}  {'candidate':>20}  "
          f"{'speedup':>9}  verdict")
    print(f"  {'-' * w}  {'-' * 12}  {'-' * 20}  {'-' * 9}  {'-' * 13}")
    for check, b, c, ratio, verdict in rows:
        print(f"  {check.ljust(w)}  {b:>12}  {c:>20}  {ratio:>9}  {verdict}")

    print()
    if fails:
        print("REGRESSION DETECTED:")
        for f in fails:
            print(f"  - {f}")
    if warns:
        print("warnings:")
        for x in warns:
            print(f"  - {x}")
    if infos:
        print("info:")
        for x in infos:
            print(f"  - {x}")
    if not fails:
        print("OK: no regression beyond the gates"
              + (" (see warnings above)" if warns else ""))
    print("\nnote: single-machine wall times are noisy and thermal-sensitive; "
          "re-run or set BENCH_COOLDOWN for borderline perf calls. The accuracy "
          "gate is deterministic.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
