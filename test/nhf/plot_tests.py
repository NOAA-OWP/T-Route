"""Plot hydrographs for each PEAK_BOUNDS reach in every NHF test case.

Usage
-----
    python -m test.nhf._plot_tests
    python -m test.nhf._plot_tests --test conecuh ciss_creek
"""

import argparse
from pathlib import Path

from .conecuh_case.test_conecuh import PEAK_BOUNDS as conecuh_bounds, DATA_DIR as conecuh_dir
from .patuxent.test_patuxent import PEAK_BOUNDS as patuxent_bounds, DATA_DIR as patuxent_dir
from .ciss_creek.test_ciss_creek import PEAK_BOUNDS as ciss_creek_bounds, DATA_DIR as ciss_creek_dir
from .great_lakes.test_great_lakes import PEAK_BOUNDS as great_lakes_bounds, DATA_DIR as great_lakes_dir
from .four_lakes.test_four_lakes import PEAK_BOUNDS as four_lakes_bounds, DATA_DIR as four_lakes_dir
from .utils.plotting import plot_reach

TESTS: dict[str, tuple[dict[int, tuple[float, float]], Path]] = {
    "conecuh": (conecuh_bounds, conecuh_dir),
    "patuxent": (patuxent_bounds, patuxent_dir),
    "ciss_creek": (ciss_creek_bounds, ciss_creek_dir),
    "great_lakes": (great_lakes_bounds, great_lakes_dir),
    "four_lakes": (four_lakes_bounds, four_lakes_dir),
}
ALL_TESTS = list(TESTS.keys())


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        nargs="+",
        choices=ALL_TESTS,
        metavar="NAME",
        help=f"Tests to plot (default: all).  Choices: {', '.join(ALL_TESTS)}",
    )
    args = parser.parse_args()

    tests = args.test if args.test else ALL_TESTS

    for name in tests:
        peak_bounds, data_dir = TESTS[name]
        output_dir = data_dir / "output"
        if not output_dir.exists():
            print(f"[{name}] No output directory at {output_dir}, skipping.")
            continue
        for reach_id in peak_bounds:
            out_path = data_dir / f"hydrograph_{reach_id}.png"
            print(f"[{name}] Plotting reach {reach_id} -> {out_path}")
            plot_reach(str(output_dir), reach_id, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
