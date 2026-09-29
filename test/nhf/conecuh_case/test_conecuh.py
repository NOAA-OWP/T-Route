from pathlib import Path

import pandas as pd
import pytest

from ..utils.integration_helpers import (
    assert_peak_bounds,
    delete_outputs,
    has_files,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config
from ..utils.make_forcing import build_forcing_dataset
from ..utils.subset_nhf import subset_nhf

OUTLET_FP_ID = 1270581653591645
START_TIME = "2009-12-12 00:00"
END_TIME = "2009-12-29 00:00"
FORCING_MODE = "retro"

# Widened from (1830, 1850), which was +/-0.54% -- 20x tighter than any other NHF
# case. Collapsing a lake's flowpaths into its level pool moves this outlet peak
# 1840.761 -> 1870.914 (+1.64%) at unchanged peak time and +0.09% volume: the wave
# sharpens because 167 attenuating in-lake links became weir discharge. 1840 is the
# pre-collapse peak, kept as the reference.
PEAK_BOUNDS: dict[int, tuple[float, float]] = {
    OUTLET_FP_ID: (0.9*1840, 1.1*1840),
}

RUNOUT_PERIOD = int(
    (pd.Timestamp(END_TIME) - pd.Timestamp(START_TIME)).total_seconds() / 3600 / 2
)
END_TIME_WITH_RUNOUT = (
    pd.Timestamp(END_TIME) + pd.Timedelta(hours=RUNOUT_PERIOD)
).strftime("%Y-%m-%d %H:%M")

DATA_DIR = Path(__file__).parent / "data"
CFG = Config(DATA_DIR, START_TIME, END_TIME_WITH_RUNOUT)


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
        )


@pytest.mark.integration
def test_conecuh():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_peak_bounds(CFG.output_dir, PEAK_BOUNDS)
