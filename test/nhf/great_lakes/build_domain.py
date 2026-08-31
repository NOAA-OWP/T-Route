"""Build the Great Lakes test domain for ``run_test.py``.

The Great Lakes drainage is far too large to subset by outlet (the documented
``subset_nhf.py --outlet-fp-id`` approach), so this carves a small slice instead:
the fp_id-bearing Great Lakes (4800002/4800004/4800006 -- 4800007/Lake Ontario
has no fp_id and is forced via the Ontario outflow file) plus a few downstream
hops, which is all ``run_test.py`` needs to watch the DA-forced outflow
propagate.

A pre-built ``domain/nhf.gpkg`` is committed so the test runs out of the box;
re-run this to regenerate it from a newer NHF release:

    python build_domain.py --source-gpkg /path/to/nhf.gpkg
"""
import argparse
import sqlite3
import sys
from pathlib import Path

# subset_nhf lives one directory up (test/nhf/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subset_nhf as S  # noqa: E402

# Great Lakes lake ids that carry an fp_id and can be anchored to a flowpath.
GREAT_LAKES_FP_BEARING_IDS = (4800002, 4800004, 4800006)


def great_lakes_fp_ids(source_gpkg: str) -> list[int]:
    conn = sqlite3.connect(source_gpkg)
    placeholders = ",".join("?" * len(GREAT_LAKES_FP_BEARING_IDS))
    rows = conn.execute(
        f"SELECT fp_id FROM lakes WHERE lake_id IN ({placeholders}) "
        "AND fp_id IS NOT NULL",
        GREAT_LAKES_FP_BEARING_IDS,
    ).fetchall()
    conn.close()
    return [int(r[0]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-gpkg", required=True,
                    help="full NHF geopackage to carve the domain from")
    ap.add_argument("--out-gpkg", default="domain/nhf.gpkg",
                    help="output domain path (default: domain/nhf.gpkg)")
    ap.add_argument("--downstream-hops", type=int, default=30,
                    help="how many hops below each Great Lake to include "
                         "(default 30)")
    args = ap.parse_args()

    gl_fps = great_lakes_fp_ids(args.source_gpkg)
    if not gl_fps:
        sys.exit("ERROR: no fp_id-bearing Great Lakes found in the source "
                 "geopackage (expected lake_ids 4800002/4800004/4800006)")
    print(f"Great Lakes flowpaths: {gl_fps}")

    fp_ids = S.get_downstream_fp_ids(args.source_gpkg, gl_fps, args.downstream_hops)
    print(f"downstream domain (<= {args.downstream_hops} hops): {len(fp_ids)} flowpaths")

    layers = S.extract_layers(args.source_gpkg, fp_ids)
    S.write_gpkg(layers, Path(args.out_gpkg))


if __name__ == "__main__":
    main()
