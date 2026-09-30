import pandas as pd
import pytest
import os
from pathlib import Path
from typing import List, Dict


@pytest.fixture
def q_lateral_hy_features(hyfeatures_test_network) -> pd.DataFrame:
    cwd = hyfeatures_test_network["cwd"]
    return pd.read_parquet(
        cwd / "test/troute-routing/data/q_lateral_hy_features.parquet"
    )


@pytest.fixture
def da_test_data(nhd_test_network) -> List[Dict[str, str]]:
    return [
        {
            "usgs_timeslice_files": [
                "2021-08-23_00:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_00:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_00:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_00:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_01:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_01:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_01:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_01:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_02:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_02:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_02:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_02:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_03:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_03:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_03:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_03:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_04:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_04:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_04:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_04:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_05:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_05:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_05:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_05:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_06:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_06:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_06:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_06:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_07:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_07:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_07:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_07:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_08:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_08:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_08:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_08:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_09:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_09:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_09:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_09:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_10:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_10:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_10:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_10:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_11:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_11:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_11:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_11:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_12:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_12:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_12:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_12:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_13:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_13:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_13:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_13:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_14:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_14:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_14:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_14:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_15:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_15:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_15:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_15:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_16:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_16:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_16:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_16:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_17:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_17:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_17:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_17:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_18:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_18:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_18:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_18:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_19:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_19:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_19:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_19:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_20:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_20:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_20:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_20:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_21:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_21:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_21:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_21:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_22:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_22:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_22:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_22:45:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_23:00:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_23:15:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_23:30:00.15min.usgsTimeSlice.ncdf",
                "2021-08-23_23:45:00.15min.usgsTimeSlice.ncdf",
            ]
        }
    ]
