from pathlib import Path

import pandas as pd
import pytest

from test import find_cwd


@pytest.fixture
def expected_nhd_preprocessed_outputs(nhd_test_network):
    cwd = find_cwd(nhd_test_network["cwd"])
    return {
        "qlats": pd.read_csv(cwd / "test/troute-nwm/data/nhd/qlats.csv"),
    }


@pytest.fixture
def expected_q0(nhd_test_network):
    cwd = find_cwd(nhd_test_network["cwd"])
    expected_q0 = pd.read_parquet(
        cwd / "test/troute-nwm/data/nhd/q0_nwm_route_results.parquet"
    )
    return expected_q0
