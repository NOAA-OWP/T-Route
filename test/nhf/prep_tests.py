import argparse
from pathlib import Path

from .conecuh_case.test_conecuh import setup as conecuh_setup
from .patuxent.test_patuxent import setup as patuxent_setup
from .ciss_creek.test_ciss_creek import setup as ciss_creek_setup
from .great_lakes.test_great_lakes import setup as great_lakes_setup
from .four_lakes.test_four_lakes import setup as four_lakes_setup
from .conus.test_conus import setup as conus_setup
from .conus.test_conus_reservoir_da import setup as conus_reservoir_da_setup
from .old_river.test_old_river import setup as old_river_setup

NHF_GPKG_DEFAULT = "/hydrofabric/nhf_1.2.1.gpkg"
FUNC_LOOKUP = {
    "conecuh": conecuh_setup,
    "patuxent": patuxent_setup,
    "ciss_creek": ciss_creek_setup,
    "great_lakes": great_lakes_setup,
    "four_lakes": four_lakes_setup,
    "conus": conus_setup,
    "conus_reservoir_da": conus_reservoir_da_setup,
    "old_river": old_river_setup
}
ALL_TESTS = list(FUNC_LOOKUP.keys())

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nhf-gpkg",
        default=NHF_GPKG_DEFAULT,
        help="Path to your NHF geopackage.",
    )
    parser.add_argument(
        "--test",
        nargs="+",
        choices=ALL_TESTS,
        metavar="NAME",
        help=f"Tests to prep (default: all).  Choices: {', '.join(ALL_TESTS)}",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force regeneration even if output already exists.",
    )
    args = parser.parse_args()

    tests = set(args.test) if args.test else set(ALL_TESTS)

    # Validate gpkg for tests that need it
    gpkg_path = Path(args.nhf_gpkg)
    if not gpkg_path.exists():
        parser.error(f"--nhf-gpkg '{args.nhf_gpkg}' does not exist. ")

    for test in tests:
        # gpkg_path, not NHF_GPKG_DEFAULT: --nhf-gpkg was validated above and then
        # thrown away, so every prep read the container path regardless of what was
        # passed and failed anywhere else.
        FUNC_LOOKUP[test](gpkg_path, refresh=args.refresh)


if __name__ == "__main__":
    main()
