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

OUTLET_FP_ID = 1284196257037837
START_TIME = "2011-09-05 00:00"
END_TIME = "2011-09-15 00:00"
FORCING_MODE = "retro"

PEAK_BOUNDS: dict[int, tuple[float, float]] = {
    1284687521058505: (0.9*0.46, 1.1*0.46),    # T. Howard Duckett Dam
    # Inflows to the Rocky Gorge lake, main stem and a tributary. Verified
    # unchanged with and without the collapse, so they pin routing, not this
    # feature: 100.656044 vs 100.656036 (float noise) and 8.699928 exactly.
    1284666761425097: (0.9*100.6, 1.1*100.6),
    1284666836882012: (0.9*8.7, 1.1*8.7),
}

# The old inflow bound, 1284687464436834 ("Duckett Dam Inflow", ~18.6), is gone: it
# sits inside the dam's polygon, so it now reports the reservoir's release (~0.46)
# like the other 8 absorbed flowpaths there. Duckett has no replacement -- no MC
# reach within 8 hops upstream, it is fed by lateral runoff -- hence the check moved
# to the other lake.

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
def test_patuxent():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_peak_bounds(CFG.output_dir, PEAK_BOUNDS)
