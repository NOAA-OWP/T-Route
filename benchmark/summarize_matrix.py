#!/usr/bin/env python3
"""Summarize the 3-way benchmark matrix, isolating code vs Python-version wins.

Reads results/<tag>-A.json, <tag>-B.kernel.json, <tag>-C.conus.json for each
tag and prints a comparison table plus an attribution:

    baseline    = ngwpc/development code, Python 3.9   (no code change)
    after-py39  = optimized code,        Python 3.9    (baseline  -> after-py39  = CODE win)
    after-py311 = optimized code,        Python 3.11   (after-py39 -> after-py311 = PYTHON win)

Usage: python benchmark/summarize_matrix.py [baseline after-py39 after-py311]
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(tag):
    def j(suffix):
        p = R / f"{tag}-{suffix}"
        return json.loads(p.read_text()) if p.exists() else None

    a, b, c = j("A.json"), j("B.kernel.json"), j("C.conus.json")
    m = (c or {}).get("metrics", {})
    return {
        "tierA_wall": a["stats"]["wall_s"]["median"] if a else None,
        "tierA_drift": a["correctness"]["flow"]["max_rel"] if a else None,
        "tierB_ms": b["ms"]["median"] if b else None,
        "tierC_wall": m.get("wall_s"),
        "tierC_cpu": m.get("cpu_s"),
        "tierC_main_gb": m["rss_mb"] / 1024 if "rss_mb" in m else None,
        "tierC_pss_gb": m["peak_tree_pss_mb"] / 1024 if "peak_tree_pss_mb" in m else None,
    }


def fmt(v, nd):
    if v is None:
        return "-"
    return f"{v:.{nd}e}" if nd >= 5 else f"{v:.{nd}f}"


def ratio(slow, fast):
    """How much `fast` improves on `slow` (>1 = improvement)."""
    return f"{slow / fast:.2f}x" if slow and fast else "-"


ROWS = [
    ("Tier A wall (s)", "tierA_wall", 2),
    ("Tier A drift vs golden", "tierA_drift", 6),
    ("Tier B kernel (ms)", "tierB_ms", 1),
    ("Tier C wall (s)", "tierC_wall", 1),
    ("Tier C CPU (s)", "tierC_cpu", 1),
    ("Tier C main RSS (GB)", "tierC_main_gb", 2),
    ("Tier C PSS (GB, TRUE)", "tierC_pss_gb", 2),
]

ATTRIB = [
    ("Tier A wall", "tierA_wall"),
    ("Tier B kernel", "tierB_ms"),
    ("Tier C wall", "tierC_wall"),
    ("Tier C PSS", "tierC_pss_gb"),
]


def main():
    tags = sys.argv[1:] or ["baseline", "after-py39", "after-py311"]
    d = {t: load(t) for t in tags}
    base, c39, c311 = (d.get(t, {}) for t in ("baseline", "after-py39", "after-py311"))

    print(f"{'metric':<26}{'baseline':>12}{'after-py39':>12}{'after-py311':>13}")
    print("-" * 63)
    for label, key, nd in ROWS:
        print(f"{label:<26}{fmt(base.get(key), nd):>12}"
              f"{fmt(c39.get(key), nd):>12}{fmt(c311.get(key), nd):>13}")

    print("\nattribution  (code = baseline->after-py39, python = after-py39->after-py311):")
    for label, key in ATTRIB:
        bk, c9, c11 = base.get(key), c39.get(key), c311.get(key)
        print(f"  {label:<14}  code {ratio(bk, c9):>7}   "
              f"python {ratio(c9, c11):>7}   total {ratio(bk, c11):>7}")


if __name__ == "__main__":
    main()
