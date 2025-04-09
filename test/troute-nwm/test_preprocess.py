import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from nwm_routing.preprocess import nwm_forcing_preprocess
from test import temporarily_change_dir


def test_nhd_preprocess(
    nhd_test_network: Dict[str, Any],
    nhd_built_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
    nhd_validation_files: Dict[str, Any],
    expected_nhd_preprocessed_outputs: Dict[str, Any],
):
    path = nhd_test_network["path"]
    forcing_parameters = nhd_test_network["forcing_parameters"]
    hybrid_parameters = nhd_test_network["hybrid_parameters"]
    compute_parameters = nhd_test_network["compute_parameters"]
    data_assimilation_parameters = nhd_test_network["data_assimilation_parameters"]
    segment_index = nhd_built_test_network["param_df"].index
    link_gage_df = nhd_built_test_network["link_gage_df"]
    usgs_lake_gage_crosswalk = nhd_built_test_network["usgs_lake_gage_crosswalk"]
    usace_lake_gage_crosswalk = nhd_built_test_network["usace_lake_gage_crosswalk"]
    link_lake_crosswalk = nhd_built_test_network["link_lake_crosswalk"]

    run_sets = [nhd_qlat_data]
    da_sets = [{"usgs_timeslice_files": []}]

    t0 = warmstart_nhd_test["t0"]
    lastobs_df = warmstart_nhd_test["lastobs_df"]

    break_network_at_waterbodies = nhd_built_test_network[
        "break_network_at_waterbodies"
    ]

    cpu_pool = compute_parameters.get("cpu_pool", None)

    with temporarily_change_dir(path):
        (
            qlats,
            usgs_df,
            reservoir_usgs_df,
            reservoir_usgs_param_df,
            reservoir_usace_df,
            reservoir_usace_param_df,
            coastal_boundary_depth_df,
        ) = nwm_forcing_preprocess(
            run_sets[0],
            forcing_parameters,
            hybrid_parameters,
            da_sets[0] if data_assimilation_parameters else {},
            data_assimilation_parameters,
            break_network_at_waterbodies,
            segment_index,
            link_gage_df,
            usgs_lake_gage_crosswalk,
            usace_lake_gage_crosswalk,
            link_lake_crosswalk,
            lastobs_df.index,
            cpu_pool,
            t0,
        )

    pd.testing.assert_frame_equal(
        expected_nhd_preprocessed_outputs["qlats"]
        .rename(columns=lambda x: int(x))
        .reset_index(drop=True),
        qlats.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )
