import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from nwm_routing.__main__ import new_nwm_q0, nwm_route
from nwm_routing.preprocess import nwm_forcing_preprocess
from test import find_cwd, temporarily_change_dir


def test_nwm_route_execution(
    nhd_test_network: Dict[str, Any], 
    nhd_built_test_network: Dict[str, Any], 
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
    nhd_validation_files: Dict[str, Any],
    expected_q0: pd.DataFrame,
):
    """Test the main routing computation"""
    path = nhd_test_network["path"]

    connections = nhd_built_test_network["connections"]
    rconn = nhd_built_test_network["rconn"]
    wbody_conn = nhd_built_test_network["wbody_conn"]
    reaches_bytw = nhd_built_test_network["reaches_bytw"]
    independent_networks = nhd_built_test_network["independent_networks"]
    param_df = nhd_built_test_network["param_df"]
    unrefactored_topobathy_df = nhd_built_test_network["unrefactored_topobathy_df"]
    refactored_reaches = nhd_built_test_network["refactored_reaches"]
    refactored_diffusive_domain = nhd_built_test_network["refactored_diffusive_domain"]
    topobathy_df = nhd_built_test_network["topobathy_df"]
    diffusive_network_data = nhd_built_test_network["diffusive_network_data"]
    waterbody_type_specified = nhd_built_test_network["waterbody_type_specified"]
    waterbody_types_df = nhd_built_test_network["waterbody_types_df"]

    forcing_parameters = nhd_test_network["forcing_parameters"]
    compute_parameters = nhd_test_network["compute_parameters"]
    waterbody_parameters = nhd_test_network["waterbody_parameters"]

        
    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    cpu_pool = compute_parameters.get("cpu_pool", None)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)

    run_sets = [nhd_qlat_data]
    t0 = run_sets[0].get("t0")
    dt = forcing_parameters.get('dt')
    nts = run_sets[0].get("nts")

    t0 = warmstart_nhd_test["t0"]
    q0 = warmstart_nhd_test["q0"]
    lastobs_df = warmstart_nhd_test["lastobs_df"]
    waterbodies_df = warmstart_nhd_test["waterbodies_df"]
    da_parameter_dict = warmstart_nhd_test["da_parameter_dict"]

    subnetwork_list = [None, None, None]

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

    break_network_at_waterbodies = nhd_built_test_network["break_network_at_waterbodies"]

    cpu_pool = compute_parameters.get("cpu_pool", None)

    with temporarily_change_dir(path):
        (
            qlats, 
            usgs_df, 
            reservoir_usgs_df, 
            reservoir_usgs_param_df,
            reservoir_usace_df,
            reservoir_usace_param_df,
            coastal_boundary_depth_df
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

        run_results, subnetwork_list = nwm_route(
            connections,
            rconn,
            wbody_conn,
            reaches_bytw,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            cpu_pool,
            t0,
            dt,
            nts,
            qts_subdivisions,
            independent_networks,
            param_df,
            q0,
            qlats,
            usgs_df,
            lastobs_df,
            reservoir_usgs_df,
            reservoir_usgs_param_df,
            reservoir_usace_df,
            reservoir_usace_param_df,
            pd.DataFrame(), #empty dataframe for RFC data...not needed unless running via BMI
            pd.DataFrame(), #empty dataframe for RFC param data...not needed unless running via BMI
            pd.DataFrame(), #empty dataframe for great lakes data...
            pd.DataFrame(), #empty dataframe for great lakes param data...
            pd.DataFrame(), #empty dataframe for great lakes climatology data...
            da_parameter_dict,
            assume_short_ts,
            return_courant,
            waterbodies_df,
            waterbody_parameters,
            waterbody_types_df,
            waterbody_type_specified,
            diffusive_network_data,
            topobathy_df,
            refactored_diffusive_domain,
            refactored_reaches,
            subnetwork_list,
            coastal_boundary_depth_df,
            unrefactored_topobathy_df,
            firstRun=False,
            logFileName ='troute_run_log.txt'             
        )

        q0 = new_nwm_q0(run_results)

    pd.testing.assert_frame_equal(
       q0,
       expected_q0,
       check_dtype=False,
       check_exact=False,
       rtol=1e-5
   )
