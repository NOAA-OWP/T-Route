from pathlib import Path

import pytest

from ..utils.integration_helpers import (
    assert_output_dimensions_and_validity,
    delete_outputs,
    has_files,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config
from ..utils.make_forcing import build_forcing_dataset

START_TIME = "2000-01-01 00:00"
END_TIME = "2000-01-01 04:00"
FORCING_MODE = "constant"
CONSTANT_QLAT = 10

DATA_DIR = Path(__file__).parent / "data" / "base"
CFG = Config(DATA_DIR, START_TIME, END_TIME, max_loop_size=1)

def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for a standard test case."""
    if refresh or not CFG.config_path.exists():
        CFG.write_yaml()

    if refresh or not CFG.domain_path.exists():
        if CFG.domain_path.exists():
            CFG.domain_path.unlink()
        CFG.domain_path.symlink_to(source_gpkg)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG.channel_forcing_dir,
            CFG.domain_path,
            constant_qlat=CONSTANT_QLAT,
        )


@pytest.mark.integration
def test_conus():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_output_dimensions_and_validity(
        CFG.output_dir, CFG.domain_path, CFG.config_path
    )
