from pathlib import Path

import pandas as pd
import pytest

from ..utils.integration_helpers import (
    assert_peak_bounds,
    assert_lakeout,
    delete_outputs,
    has_files,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config
from ..utils.make_forcing import build_forcing_dataset
from ..utils.subset_nhf import subset_nhf

OUTLET_FP_ID = 1288454913281725
START_TIME = "2000-01-01 00:00"
END_TIME = "2000-01-03 00:00"
FORCING_MODE = "pulse"
PEAK_QLAT = 2000

PEAK_BOUNDS: dict[int, tuple[float, float]] = {
    OUTLET_FP_ID: (0.9 * 1504, 1.1 * 1504),
}
# Every lake in the domain must route an outflow in this range. The bound is the same
# for all of them, so listing ids bought nothing and cost portability: `nhf_lake_id` is
# assigned per hydrofabric build, and 2 of the 5 hardcoded here moved between nhf 1.2.1
# and 1.2.2, which failed as though routing had broken. Read them from the domain the
# test actually runs on; a lake going missing still fails, on the count check below.
def _lakeout_bounds() -> dict[int, tuple[float, float]]:
    import geopandas as gpd

    lakes = gpd.read_file(CFG.domain_path, layer="lakes", ignore_geometry=True)
    return {int(i): (0.1, PEAK_QLAT) for i in lakes["nhf_lake_id"].dropna()}

RUNOUT_PERIOD = int(
    (pd.Timestamp(END_TIME) - pd.Timestamp(START_TIME)).total_seconds() / 3600 / 2
)
END_TIME_WITH_RUNOUT = (
    pd.Timestamp(END_TIME) + pd.Timedelta(hours=RUNOUT_PERIOD)
).strftime("%Y-%m-%d %H:%M")

DATA_DIR = Path(__file__).parent / "data"
CFG = Config(DATA_DIR, START_TIME, END_TIME_WITH_RUNOUT, lakeout_output="lakeout")


def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for a standard test case."""
    if refresh or not CFG.config_path.exists():
        CFG.write_yaml()

    if refresh or not CFG.domain_path.exists():
        subset_nhf(source_gpkg, CFG.domain_path, OUTLET_FP_ID)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG.channel_forcing_dir,
            CFG.domain_path,
            RUNOUT_PERIOD,
            peak_qlat=PEAK_QLAT,
        )


@pytest.mark.integration
def test_ciss_creek():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_peak_bounds(CFG.output_dir, PEAK_BOUNDS)
    bounds = _lakeout_bounds()
    assert bounds, "domain has no lakes; the lakeout assertion would be vacuous"
    assert_lakeout(
        CFG.lakeout_dir,
        expected_feature_count=len(bounds),
        outflow_bounds=bounds,
    )
