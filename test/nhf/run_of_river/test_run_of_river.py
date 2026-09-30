"""A VPU 02 domain holding dams the hydrofabric flags run-of-river and dams it does not.

Of its 8 routable lakes, 4 are flagged low-head dams and none is ``da_type`` 4. The flagged
dams' flowpaths must survive the whole network build and route as channel reaches with no
level pool behind them.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from ..utils.integration_helpers import (
    delete_outputs,
    has_files,
    load_lakeout,
    load_output,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config
from ..utils.make_forcing import build_forcing_dataset
from ..utils.subset_nhf import subset_nhf

OUTLET_FP_ID = 1285238895019143
START_TIME = "2011-09-05 00:00"
END_TIME = "2011-09-10 00:00"
FORCING_MODE = "retro"

# Flagged in the lakes layer, so each leaves the reservoir set and its flowpath routes
# as Muskingum-Cunge. None carries da_type 4, so RFC precedence holds none back.
FLAGGED_LAKE_IDS = [
    1285240199076119, 1285267302172926, 1285270064192495, 1285271314977256,
]

RUNOUT_PERIOD = int(
    (pd.Timestamp(END_TIME) - pd.Timestamp(START_TIME)).total_seconds() / 3600 / 2
)
END_TIME_WITH_RUNOUT = (
    pd.Timestamp(END_TIME) + pd.Timedelta(hours=RUNOUT_PERIOD)
).strftime("%Y-%m-%d %H:%M")

DATA_DIR = Path(__file__).parent / "data"
CFG = Config(DATA_DIR, START_TIME, END_TIME_WITH_RUNOUT, lakeout_output="lakeout")


def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for the run-of-river case."""
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
        )


def _lakes() -> pd.DataFrame:
    return gpd.read_file(CFG.domain_path, layer="lakes", ignore_geometry=True)


@pytest.mark.integration
def test_the_domain_carries_both_arms():
    """A domain with nothing flagged, or everything, would pass the test below for the
    wrong reason."""
    skip_if_not_built(CFG)
    lakes = _lakes()
    flagged = lakes["run_of_river"].astype(bool)
    assert set(lakes.loc[flagged, "nhf_lake_id"]) == set(FLAGGED_LAKE_IDS)
    assert (~flagged).sum() > 0, "no retained reservoir to contrast against"


@pytest.mark.integration
def test_a_flagged_dam_is_not_a_reservoir_in_the_routed_output():
    """A flagged dam must be absent from lakeout, which holds one record per routed
    reservoir; a retained lake must be present, or the absence means nothing."""
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)

    ds = load_lakeout(CFG.lakeout_dir)
    routed = {int(i) for i in ds["feature_id"].to_numpy()}
    lakes = _lakes()
    retained = {
        int(i) for i in lakes.loc[~lakes["run_of_river"].astype(bool), "nhf_lake_id"]
    }

    assert routed & retained, "no retained reservoir reached lakeout"
    still_reservoirs = routed & set(FLAGGED_LAKE_IDS)
    assert not still_reservoirs, (
        f"flagged dam(s) {sorted(still_reservoirs)} were routed as level pools"
    )

    # Absence from lakeout also holds if the flowpath vanished during the build, so
    # each one has to be routed.
    flagged_fps = {
        int(f) for f in lakes.loc[lakes["nhf_lake_id"].isin(FLAGGED_LAKE_IDS), "fp_id"]
    }
    streamed = {int(i) for i in load_output(CFG.output_dir)["feature_id"].to_numpy()}
    missing = flagged_fps - streamed
    assert not missing, f"flagged dam flowpath(s) {sorted(missing)} left the model"


@pytest.mark.integration
def test_run_of_river():
    """The domain routes end to end with the flagged dams as channel."""
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert has_files(CFG.output_dir, "*.nc"), "the run produced no output"
