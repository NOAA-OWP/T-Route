import argparse
import time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import math

import asyncio
import concurrent.futures

## network and reach utilities
import troute.nhd_network as nhd_network
import troute.nhd_io as nhd_io
import troute.nhd_network_utilities_v02 as nnu
import build_tests  # TODO: Determine whether and how to incorporate this into setup.py
import sys

import troute.routing.diffusive_utils as diff_utils

from .input import _input_handler_v02, _input_handler_v03
from .preprocess import (
    nwm_network_preprocess,
    nwm_initial_warmstate_preprocess,
    nwm_forcing_preprocess,
)
from .output import nwm_output_generator

from troute.routing.compute import compute_nhd_routing_v02


def _handle_args_v03(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--custom-input-file",
        dest="custom_input_file",
        help="OR... please enter the path of a .yaml or .json file containing a custom supernetwork information. See for example test/input/yaml/CustomInput.yaml and test/input/json/CustomInput.json.",
    )
    return parser.parse_args(argv)


def _handle_args_v02(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debuglevel",
        help="Set the debuglevel",
        dest="debuglevel",
        choices=[0, 1, 2, 3],
        default=0,
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output (leave blank for quiet output)",
        dest="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--qlat-dt",
        "--qlateral-time-step",
        help="Set the default qlateral timestep length",
        dest="qdt",
        default=3600,
    )
    parser.add_argument(
        "--qN",
        "--qts-subdivisions",
        help="number of simulation timesteps per qlateral timestep",
        dest="qts_subdivisions",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dt",
        "--simulation-time-step",
        help="Set the default simulation timestep length",
        dest="dt",
        default=300,
    )
    parser.add_argument(
        "--nts",
        "--number-of-simulation-timesteps",
        help="Set the number of timesteps to execute. If used with ql_file or ql_folder, nts must be less than len(ql) x qN.",
        dest="nts",
        default=144,
        type=int,
    )

    # change this so after --test, the user enters a test choice
    parser.add_argument(
        "--test",
        help="Select a test case, routing results will be compared against WRF hydro for parity",
        choices=["pocono1"],
        dest="test_case",
    )

    parser.add_argument(
        "--sts",
        "--assume-short-ts",
        help="Use the previous timestep value for upstream flow",
        dest="assume_short_ts",
        action="store_true",
    )
    parser.add_argument(
        "--courant",
        "--return-courant-metrics",
        help="Return Courant evaluation metrics for each segment/timestep",
        dest="return_courant",
        action="store_true",
    )
    parser.add_argument(
        "-ocsv",
        "--write-output-csv",
        nargs="?",
        help="Write csv output files to this folder (omit flag for no csv writing)",
        dest="csv_output_folder",
        const="../../test/output/text",
    )
    parser.add_argument(
        "-t",
        "--showtiming",
        help="Set the showtiming (leave blank for no timing information)",
        dest="showtiming",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--break-at-waterbodies",
        help="Use the waterbodies in the route-link dataset to divide the computation (leave blank for no splitting)",
        dest="break_network_at_waterbodies",
        action="store_true",
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        help="Use the parallel computation engine (omit flag for serial computation)",
        dest="parallel_compute_method",
        const="by-network",
    )
    parser.add_argument(
        "--subnet-size",
        help="Set the target size (number of segments) for grouped subnetworks.",
        dest="subnetwork_target_size",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--cpu-pool",
        help="Assign the number of cores to multiprocess across.",
        dest="cpu_pool",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--compute-method",
        nargs="?",
        help="Use the cython version of the compute_network code [options: 'V02-caching'; 'V02-structured'; 'V02-structured-obj' ... ).",
        dest="compute_method",
        default="VO2-caching",
    )
    supernetwork_arg_group = parser.add_mutually_exclusive_group()
    supernetwork_arg_group.add_argument(
        "-n",
        "--supernetwork",
        help="Choose from among the pre-programmed supernetworks (Pocono_TEST1, Pocono_TEST2, LowerColorado_Conchos_FULL_RES, Brazos_LowerColorado_ge5, Brazos_LowerColorado_FULL_RES, Brazos_LowerColorado_Named_Streams, CONUS_ge5, Mainstems_CONUS, CONUS_Named_Streams, CONUS_FULL_RES_v20)",
        choices=[
            "Pocono_TEST1",
            "Pocono_TEST2",
            "LowerColorado_Conchos_FULL_RES",
            "Brazos_LowerColorado_ge5",
            "Brazos_LowerColorado_FULL_RES",
            "Brazos_LowerColorado_Named_Streams",
            "CONUS_ge5",
            "Mainstems_CONUS",
            "CONUS_Named_Streams",
            "CONUS_FULL_RES_v20",
            "CapeFear_FULL_RES",
            "Florence_FULL_RES",
        ],
        # TODO: accept multiple or a Path (argparse Action perhaps)
        # action='append',
        # nargs=1,
        dest="supernetwork",
        default="Pocono_TEST1",
    )
    supernetwork_arg_group.add_argument(
        "-f",
        "--custom-input-file",
        dest="custom_input_file",
        help="OR... please enter the path of a .yaml or .json file containing a custom supernetwork information. See for example test/input/yaml/CustomInput.yaml and test/input/json/CustomInput.json.",
    )
    parser.add_argument(
        "--wrf-hydro-channel-restart-file",
        dest="wrf_hydro_channel_restart_file",
        help="provide a WRF-Hydro channel warm state file (may be the same as waterbody restart file)",
    )
    parser.add_argument(
        "--wrf-hydro-channel-ID-crosswalk-file",
        dest="wrf_hydro_channel_ID_crosswalk_file",
        help="provide an xarray-readable file that defines the order of the outputs in the channel restart file. Specify the ID field with --wrf_hydro_channel_ID_crosswalk_file_field_name",
    )
    parser.add_argument(
        "--wrf-hydro-channel-ID-crosswalk-file-field-name",
        dest="wrf_hydro_channel_ID_crosswalk_file_field_name",
        help="Name of the column providing the channel segment IDs in the channel crosswalk file",
        default="ID",
    )
    parser.add_argument(
        "--wrf-hydro-channel-restart-upstream-flow-field-name",
        dest="wrf_hydro_channel_restart_upstream_flow_field_name",
        help="Name of the column providing the upstream flow at the beginning of the simulation.",
        default="qlink1",
    )
    parser.add_argument(
        "--wrf-hydro-channel-restart-downstream-flow-field-name",
        dest="wrf_hydro_channel_restart_downstream_flow_field_name",
        help="Name of the column providing the downstream flow at the beginning of the simulation.",
        default="qlink2",
    )
    parser.add_argument(
        "--wrf-hydro-channel-restart-depth-flow-field-name",
        dest="wrf_hydro_channel_restart_depth_flow_field_name",
        help="Name of the column providing the depth of flow at the beginning of the simulation.",
        default="hlink",
    )
    # TODO: Refine exclusivity of ql args (currently not going to accept more than one arg; more than one is needed for qlw, for instance.)
    ql_arg_group = parser.add_mutually_exclusive_group()
    ql_arg_group.add_argument(
        "--qlc",
        "--constant_qlateral",
        help="Constant qlateral to apply to all time steps at all segments",
        dest="qlat_const",
        type=float,
        default=10,
    )
    ql_arg_group.add_argument(
        "--qlf",
        "--single_file_qlateral",
        help="QLaterals arranged with segment IDs as rows and timesteps as columns in a single .csv",
        dest="qlat_input_file",
    )
    ql_arg_group.add_argument(
        "--qlw",
        "--ql_wrf_hydro_folder",
        help="QLaterals in separate netcdf files as found in standard WRF-Hydro output",
        dest="qlat_input_folder",
    )
    ql_arg_group.add_argument(
        "--qlic",
        "--qlat_file_index_col",
        help="QLateral index column number",
        dest="qlat_file_index_col",
        default="feature_id",
    )
    ql_arg_group.add_argument(
        "--qlvc",
        "--qlat_file_value_col",
        help="QLateral value column number",
        dest="qlat_file_value_col",
        default="q_lateral",
    )
    parser.add_argument(
        "--qlat_file_pattern_filter",
        help="Provide a globbing pattern to identify files in the Wrf-Hydro qlateral output file folder",
        dest="qlat_file_pattern_filter",
        default="q_lateral",
    )
    parser.add_argument("--ql", help="QLat input data", dest="ql", default=None)

    parser.add_argument(
        "--data_assimilation_folder_path",
        help="Provide a path to a folder containing the usgs time slice files",
        dest="data_assimilation_parameters_folder",
        default=None,
    )
    parser.add_argument(
        "--data_assimilation_filter",
        help="Provide a glob pattern filter for ncdf files (e.g., 2020-03-21*.usgsTimeSlice.ncdf)",
        dest="data_assimilation_filter",
        default=None,
    )
    parser.add_argument(
        "--data_assimilation_csv",
        help="Provide a csv with the timeslices prepared for use",
        dest="data_assimilation_csv",
        default=None,
    )
    return parser.parse_args(argv)


def main_v02(argv):
    args = _handle_args_v02(argv)
    (
        supernetwork_parameters,
        waterbody_parameters,
        forcing_parameters,
        restart_parameters,
        output_parameters,
        run_parameters,
        parity_parameters,
        data_assimilation_parameters,
        diffusive_parameters,
        coastal_parameters,
    ) = _input_handler_v02(args)

    verbose = run_parameters.get("verbose", None)
    showtiming = run_parameters.get("showtiming", None)
    if showtiming:
        main_start_time = time.time()

    _results = _run_everything_v02(
        supernetwork_parameters,
        waterbody_parameters,
        forcing_parameters,
        restart_parameters,
        output_parameters,
        run_parameters,
        parity_parameters,
        data_assimilation_parameters,
        diffusive_parameters,
        coastal_parameters,
    )

    _handle_output_v02(
        _results,
        run_parameters,
        supernetwork_parameters,
        restart_parameters,
        output_parameters,
        parity_parameters,
    )

    if verbose:
        print("process complete")
    if showtiming:
        print("%s seconds." % (time.time() - main_start_time))


def _run_everything_v02(
    supernetwork_parameters,
    waterbody_parameters,
    forcing_parameters,
    restart_parameters,
    output_parameters,
    run_parameters,
    parity_parameters,
    data_assimilation_parameters,
    diffusive_parameters,
    coastal_parameters,
):

    dt = run_parameters.get("dt", None)
    nts = run_parameters.get("nts", None)
    verbose = run_parameters.get("verbose", None)
    showtiming = run_parameters.get("showtiming", None)
    debuglevel = run_parameters.get("debuglevel", 0)
    break_network_at_waterbodies = run_parameters.get(
        "break_network_at_waterbodies", False
    )
    break_network_at_gages = supernetwork_parameters.get(
        "break_network_at_gages", False
    )

    if verbose:
        print("creating supernetwork connections set")
    if showtiming:
        start_time = time.time()

    # STEP 1: Build basic network connections graph
    connections, param_df, wbody_conn, gages = nnu.build_connections(
        supernetwork_parameters
    )
    if break_network_at_waterbodies:
        connections = nhd_network.replace_waterbodies_connections(
            connections, wbody_conn
        )

    if verbose:
        print("supernetwork connections set complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    ################################
    ## STEP 3a: Read waterbody parameter file
    # waterbodies_values = supernetwork_values[12]
    # waterbodies_segments = supernetwork_values[13]
    # connections_tailwaters = supernetwork_values[4]

    waterbody_type_specified = False

    if break_network_at_waterbodies:
        # Read waterbody parameters
        waterbodies_df = nhd_io.read_waterbody_df(
            waterbody_parameters, {"level_pool": wbody_conn.values()}
        )

        # Remove duplicate lake_ids and rows
        waterbodies_df = (
            waterbodies_df.reset_index()
            .drop_duplicates(subset="lake_id")
            .set_index("lake_id")
        )

        # Declare empty dataframe
        waterbody_types_df = pd.DataFrame()

        # Check if hybrid-usgs, hybrid-usace, or rfc type reservoirs are set to true
        wbtype = "hybrid_and_rfc"
        wb_params_hybrid_and_rfc = waterbody_parameters.get(
            wbtype, defaultdict(list)
        )  # TODO: Convert these to `get` statments

        wbtype = "level_pool"
        wb_params_level_pool = waterbody_parameters.get(
            wbtype, defaultdict(list)
        )  # TODO: Convert these to `get` statments

        waterbody_type_specified = False

        if (
            wb_params_hybrid_and_rfc["reservoir_persistence_usgs"]
            or wb_params_hybrid_and_rfc["reservoir_persistence_usace"]
            or wb_params_hybrid_and_rfc["reservoir_rfc_forecasts"]
        ):

            waterbody_type_specified = True

            waterbody_types_df = nhd_io.read_reservoir_parameter_file(
                wb_params_hybrid_and_rfc["reservoir_parameter_file"],
                wb_params_level_pool["level_pool_waterbody_id"],
                wbody_conn.values(),
            )

            # Remove duplicate lake_ids and rows
            waterbody_types_df = (
                waterbody_types_df.reset_index()
                .drop_duplicates(subset="lake_id")
                .set_index("lake_id")
            )

    else:
        # Declare empty dataframe
        waterbody_types_df = pd.DataFrame()
        waterbodies_df = pd.DataFrame()

    # STEP 2: Identify Independent Networks and Reaches by Network
    if showtiming:
        start_time = time.time()
    if verbose:
        print("organizing connections into reaches ...")

    network_break_segments = set()
    if break_network_at_waterbodies:
        network_break_segments = network_break_segments.union(wbody_conn.values())
    if break_network_at_gages:
        network_break_segments = network_break_segments.union(gages.keys())
    independent_networks, reaches_bytw, rconn = nnu.organize_independent_networks(
        connections,
        network_break_segments,
    )
    if verbose:
        print("reach organization complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    if break_network_at_waterbodies:
        ## STEP 3c: Handle Waterbody Initial States
        # TODO: move step 3c into function in nnu, like other functions wrapped in main()
        if showtiming:
            start_time = time.time()
        if verbose:
            print("setting waterbody initial states ...")

        if restart_parameters.get("wrf_hydro_waterbody_restart_file", None):
            waterbodies_initial_states_df = nhd_io.get_reservoir_restart_from_wrf_hydro(
                restart_parameters["wrf_hydro_waterbody_restart_file"],
                restart_parameters["wrf_hydro_waterbody_ID_crosswalk_file"],
                restart_parameters["wrf_hydro_waterbody_ID_crosswalk_file_field_name"],
                restart_parameters["wrf_hydro_waterbody_crosswalk_filter_file"],
                restart_parameters[
                    "wrf_hydro_waterbody_crosswalk_filter_file_field_name"
                ],
            )
        else:
            # TODO: Consider adding option to read cold state from route-link file
            waterbodies_initial_ds_flow_const = 0.0
            waterbodies_initial_depth_const = -1e9
            # Set initial states from cold-state
            waterbodies_initial_states_df = pd.DataFrame(
                0,
                index=waterbodies_df.index,
                columns=[
                    "qd0",
                    "h0",
                ],
                dtype="float32",
            )
            # TODO: This assignment could probably by done in the above call
            waterbodies_initial_states_df["qd0"] = waterbodies_initial_ds_flow_const
            waterbodies_initial_states_df["h0"] = waterbodies_initial_depth_const
            waterbodies_initial_states_df["index"] = range(
                len(waterbodies_initial_states_df)
            )

        waterbodies_df = pd.merge(
            waterbodies_df, waterbodies_initial_states_df, on="lake_id"
        )

        if verbose:
            print("waterbody initial states complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
            start_time = time.time()

    # STEP 4: Handle Channel Initial States
    if showtiming:
        start_time = time.time()
    if verbose:
        print("setting channel initial states ...")

    q0 = nnu.build_channel_initial_state(restart_parameters, param_df.index)
    # STEP 4a: Set Channel States and T0
    if restart_parameters.get("wrf_hydro_channel_restart_file",None):
        channel_initial_states_file = restart_parameters["wrf_hydro_channel_restart_file"]
        t0_str = nhd_io.get_param_str(channel_initial_states_file, "Restart_Time")
    else:
        t0_str = "2015-08-16_00:00:00"

    t0 = datetime.strptime(t0_str, "%Y-%m-%d_%H:%M:%S")
    run_parameters["t0"] = t0

    if verbose:
        print("channel initial states complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))
        start_time = time.time()

    # STEP 5: Read (or set) QLateral Inputs
    if showtiming:
        start_time = time.time()
    if verbose:
        print("creating qlateral array ...")

    forcing_parameters["qts_subdivisions"] = run_parameters["qts_subdivisions"]
    forcing_parameters["nts"] = run_parameters["nts"]
    qlats = nnu.build_qlateral_array(
        forcing_parameters,
        param_df.index,
        nts,
        run_parameters.get("qts_subdivisions", 1),
    )

    if verbose:
        print("qlateral array complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    # STEP 6
    data_assimilation_csv = data_assimilation_parameters.get(
        "data_assimilation_csv", None
    )
    data_assimilation_folder = data_assimilation_parameters.get(
        "data_assimilation_timeslices_folder", None
    )
    lastobs_file = data_assimilation_parameters.get("wrf_hydro_lastobs_file", None)

    if data_assimilation_csv or data_assimilation_folder or lastobs_file:
        if showtiming:
            start_time = time.time()
        if verbose:
            print("creating usgs time_slice data array ...")

            usgs_df, lastobs_df, da_parameter_dict = nnu.build_data_assimilation(
                data_assimilation_parameters,
                run_parameters
            )

        if verbose:
            print("usgs array complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))

    else:
        usgs_df = pd.DataFrame()
        lastobs_df = pd.DataFrame()
        da_parameter_dict = {}

    ################### Main Execution Loop across ordered networks
    if showtiming:
        start_time = time.time()
    if verbose:
        if run_parameters.get("return_courant", False):
            print(
                f"executing routing computation, with Courant evaluation metrics returned"
            )
        else:
            print(f"executing routing computation ...")

    # TODO: align compute_kernel and compute_method in run_parameters
    if run_parameters.get("compute_kernel", None):
        compute_func = run_parameters.get("compute_kernel", None)
    else:
        compute_func = run_parameters.get("compute_method", None)
    # TODO: Remove below. --compute-method=V02-structured-obj did not work on command line
    # compute_func = fast_reach.compute_network_structured_obj

    results = compute_nhd_routing_v02(
        connections,
        rconn,
        wbody_conn,
        reaches_bytw,
        compute_func,
        run_parameters.get("parallel_compute_method", None),
        run_parameters.get("subnetwork_target_size", 1),
        # The default here might be the whole network or some percentage...
        run_parameters.get("cpu_pool", None),
        run_parameters.get("dt"),
        run_parameters.get("nts", 1),
        run_parameters.get("qts_subdivisions", 1),
        independent_networks,
        param_df,
        q0,
        qlats,
        usgs_df,
        lastobs_df,
        da_parameter_dict,
        run_parameters.get("assume_short_ts", False),
        run_parameters.get("return_courant", False),
        waterbodies_df,
        waterbody_parameters,  # TODO: Can we remove the dependence on this input? It's like passing argv down into the compute kernel -- seems like we can strip out the specifically needed items.
        waterbody_types_df,
        waterbody_type_specified,
        diffusive_parameters,
    )

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    return results


def _handle_output_v02(
    results,
    run_parameters,
    supernetwork_parameters,
    restart_parameters,
    output_parameters,
    parity_parameters,
):
    ################### Output Handling
    dt = run_parameters.get("dt", None)
    nts = run_parameters.get("nts", None)
    t0 = run_parameters.get("t0", None)
    verbose = run_parameters.get("verbose", None)
    showtiming = run_parameters.get("showtiming", None)
    debuglevel = run_parameters.get("debuglevel", 0)

    if showtiming:
        start_time = time.time()
    if verbose:
        print(f"Handling output ...")

    csv_output = output_parameters.get("csv_output", None)
    if csv_output:
        csv_output_folder = output_parameters["csv_output"].get(
            "csv_output_folder", None
        )
        csv_output_segments = csv_output.get("csv_output_segments", None)

    if (debuglevel <= -1) or csv_output:

        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()

        flowveldepth = pd.concat(
            [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
            copy=False,
        )

        if run_parameters.get("return_courant", False):
            courant_columns = pd.MultiIndex.from_product(
                [range(nts), ["cn", "ck", "X"]]
            ).to_flat_index()
            courant = pd.concat(
                [
                    pd.DataFrame(r[2], index=r[0], columns=courant_columns)
                    for r in results
                ],
                copy=False,
            )

        if csv_output_folder:
            # create filenames
            # TO DO: create more descriptive filenames
            if supernetwork_parameters.get("title_string", None):
                filename_fvd = (
                    "flowveldepth_" + supernetwork_parameters["title_string"] + ".csv"
                )
                filename_courant = (
                    "courant_" + supernetwork_parameters["title_string"] + ".csv"
                )
            else:
                run_time_stamp = datetime.now().isoformat()
                filename_fvd = "flowveldepth_" + run_time_stamp + ".csv"
                filename_courant = "courant_" + run_time_stamp + ".csv"

            output_path = Path(csv_output_folder).resolve()

            flowveldepth = flowveldepth.sort_index()
            flowveldepth.to_csv(output_path.joinpath(filename_fvd))

            if run_parameters.get("return_courant", False):
                courant = courant.sort_index()
                courant.to_csv(output_path.joinpath(filename_courant))

            # TODO: need to identify the purpose of these outputs
            # if the need is to output the usgs_df dataframe,
            # then that could be done in the dataframe IO somewhere above.
            # usgs_df_filtered = usgs_df[usgs_df.index.isin(csv_output_segments)]
            # usgs_df_filtered.to_csv(output_path.joinpath("usgs_df.csv"))

        if debuglevel <= -1:
            print(flowveldepth)

    # directory containing WRF Hydro restart files
    wrf_hydro_restart_read_dir = output_parameters.get(
        "wrf_hydro_channel_restart_source_directory", None
    )
    wrf_hydro_restart_write_dir = output_parameters.get(
        "wrf_hydro_channel_restart_output_directory", wrf_hydro_restart_read_dir
    )
    if wrf_hydro_restart_read_dir:

        wrf_hydro_channel_restart_new_extension = output_parameters.get(
            "wrf_hydro_channel_restart_new_extension", "TRTE"
        )

        # list of WRF Hydro restart files
        wrf_hydro_restart_files = sorted(
            Path(wrf_hydro_restart_read_dir).glob(
                output_parameters["wrf_hydro_channel_restart_pattern_filter"]
                + "[!"
                + wrf_hydro_channel_restart_new_extension
                + "]"
            )
        )

        if len(wrf_hydro_restart_files) > 0:
            qvd_columns = pd.MultiIndex.from_product(
                [range(nts), ["q", "v", "d"]]
            ).to_flat_index()

            flowveldepth = pd.concat(
                [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
                copy=False,
            )
            nhd_io.write_channel_restart_to_wrf_hydro(
                flowveldepth,
                wrf_hydro_restart_files,
                # TODO: remove this dependence on the restart_parameters
                Path(wrf_hydro_restart_write_dir),
                restart_parameters.get("wrf_hydro_channel_restart_file"),
                run_parameters.get("dt"),
                run_parameters.get("nts"),
                t0,
                restart_parameters.get("wrf_hydro_channel_ID_crosswalk_file"),
                restart_parameters.get(
                    "wrf_hydro_channel_ID_crosswalk_file_field_name"
                ),
                wrf_hydro_channel_restart_new_extension,
            )
        else:
            # print error and raise exception
            str = "WRF Hydro restart files not found - Aborting restart write sequence"
            raise AssertionError(str)

    chrtout_read_folder = output_parameters.get(
        "wrf_hydro_channel_output_source_folder", None
    )
    chrtout_write_folder = output_parameters.get(
        "wrf_hydro_channel_final_output_folder", chrtout_read_folder
    )
    if chrtout_read_folder:
        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()

        flowveldepth = pd.concat(
            [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
            copy=False,
        )
        wrf_hydro_channel_output_new_extension = output_parameters.get(
            "wrf_hydro_channel_output_new_extension", "TRTE"
        )
        chrtout_files = sorted(
            Path(chrtout_read_folder).glob(
                output_parameters["wrf_hydro_channel_output_file_pattern_filter"]
            )
        )
        nhd_io.write_q_to_wrf_hydro(
            flowveldepth,
            chrtout_files,
            Path(chrtout_write_folder),
            run_parameters["qts_subdivisions"],
            wrf_hydro_channel_output_new_extension,
        )

    if verbose:
        print("output complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    ################### Parity Check

    if (
        "parity_check_input_folder" in parity_parameters
        or "parity_check_file" in parity_parameters
        or "parity_check_waterbody_file" in parity_parameters
    ):

        if verbose:
            print(
                "conducting parity check, comparing WRF Hydro results against t-route results"
            )
        if showtiming:
            start_time = time.time()

        parity_parameters["nts"] = nts
        parity_parameters["dt"] = dt

        build_tests.parity_check(
            parity_parameters,
            results,
        )

        if verbose:
            print("parity check complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))


def nwm_route(
    downstream_connections,
    upstream_connections,
    waterbodies_in_connections,
    reaches_bytw,
    parallel_compute_method,
    compute_kernel,
    subnetwork_target_size,
    cpu_pool,
    dt,
    nts,
    qts_subdivisions,
    independent_networks,
    param_df,
    q0,
    qlats,
    usgs_df,
    lastobs_df,
    da_parameter_dict,
    assume_short_ts,
    return_courant,
    waterbodies_df,
    waterbody_parameters,
    waterbody_types_df,
    waterbody_type_specified,
    diffusive_parameters,
    showtiming=False,
    verbose=False,
    debuglevel=0,
):

    ################### Main Execution Loop across ordered networks
    if showtiming:
        start_time = time.time()
    if verbose:
        if return_courant:
            print(
                f"executing routing computation, with Courant evaluation metrics returned"
            )
        else:
            print(f"executing routing computation ...")

    # TODO: Remove below. --compute-kernel=V02-structured-obj did not work on command line
    # compute_func = fast_reach.compute_network_structured_obj

    results = compute_nhd_routing_v02(
        downstream_connections,
        upstream_connections,
        waterbodies_in_connections,
        reaches_bytw,
        compute_kernel,
        parallel_compute_method,
        subnetwork_target_size,  # The default here might be the whole network or some percentage...
        cpu_pool,
        dt,
        nts,
        qts_subdivisions,
        independent_networks,
        param_df,
        q0,
        qlats,
        usgs_df,
        lastobs_df,
        da_parameter_dict,
        assume_short_ts,
        return_courant,
        waterbodies_df,
        waterbody_parameters,
        waterbody_types_df,
        waterbody_type_specified,
        diffusive_parameters,
    )

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    return results


def new_nwm_q0(run_results):
    """
    Prepare a new q0 dataframe with initial flow and depth to act as
    a warmstate for the next simulation chunk.
    """
    return pd.concat(
        # TODO: we only need two fields, technically, and the restart file produced by WRF-Hydro
        # actually contains a field qu0, which is never used for restart (the qu0 can be obtained
        # as the qd0 from the topologically upstream segments, just like during the calculation).
        # In any case, the qu0 currently in the WRF-Hydro output is populated with the same value
        # as the qd0.
        # [pd.DataFrame(d[:,-3::2], index=i, columns=["qd0", "h0"]) for i, d in run_results],
        # [pd.DataFrame(r[1][:,-3:], index=r[0], columns=["qu0", "v0", "h0"]) for r in run_results],
        [
            pd.DataFrame(
                r[1][:, [-3, -3, -1]], index=r[0], columns=["qu0", "qd0", "h0"]
            )
            for r in run_results
        ],
        copy=False,
    )


def get_waterbody_water_elevation(waterbodies_df, q0):
    """
    Update the starting water_elevation of each lake/reservoir
    with depth values from q0
    """
    waterbodies_df.update(q0)

    return waterbodies_df


def update_lookback_hours(dt, nts, waterbody_parameters):
    """
    Update the lookback hours that an RFC type reservoir searches in reverse
    from the model start time to find a time series file. The update is based
    on the total hours ran in the prior loop.
    """

    waterbody_parameters['hybrid_and_rfc']['reservoir_rfc_forecasts_lookback_hours'] = \
    waterbody_parameters['hybrid_and_rfc']['reservoir_rfc_forecasts_lookback_hours'] + \
    math.ceil((dt * nts) / 3600)

    return waterbody_parameters


def new_lastobs(run_results, time_increment):
    """
    Creates new "lastobs" dataframe for the next simulation chunk.

    run_results - output from the compute kernel sequence, organized
        (because that is how it comes out of the kernel) by network.
        For each item in the result, there are four elements, the
        fourth of which is a tuple containing: 1) a list of the
        segments ids where data assimilation was performed (if any)
        in that network; 2) a list of the last valid observation
        applied at that segment; 3) a list of the time in seconds
        from the beginning of the last simulation that the
        observation was applied.
    time_increment - length of the prior simulation. To prepare the
        next lastobs state, we have to convert the time since the prior
        simulation start to a time since the new simulation start.
        If the most recent observation was right at the end of the
        prior loop, then the value in the incoming run_result will
        be equal to the time_increment and the output value will be
        zero. If observations were not present at the last timestep,
        the last obs time will be calculated to a negative value --
        the number of seconds ago that the last valid observation
        was used for assimilation.
    """
    df = pd.concat(
        [
            pd.DataFrame(
                # TODO: Add time_increment (or subtract?) from time_since_lastobs
                np.array([rr[3][1],rr[3][2]]).T,
                index=rr[3][0],
                columns=["time_since_lastobs", "lastobs_discharge"]
            )
            for rr in run_results
            if not rr[3][0].size == 0
        ],
        copy=False,
    )
    df["time_since_lastobs"] = df["time_since_lastobs"] - time_increment
    return df


def main_v03(argv):
    """
    Handles the creation of the input parameter dictionaries
    from an input file and then sequences the execution of the
    t-route routing agorithm on a series of execution loops.
    """
    args = _handle_args_v03(argv)
    (
        log_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        diffusive_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v03(args)

    verbose = log_parameters.get("verbose", None)
    showtiming = log_parameters.get("showtiming", None)
    debuglevel = log_parameters.get("debuglevel", 0)

    if showtiming:
        main_start_time = time.time()

    (
        connections,
        param_df,
        wbody_conn,
        waterbodies_df,
        waterbody_types_df,
        break_network_at_waterbodies,
        waterbody_type_specified,
        independent_networks,
        reaches_bytw,
        rconn,
    ) = nwm_network_preprocess(
        supernetwork_parameters,
        waterbody_parameters,
        showtiming=showtiming,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    # TODO: This function modifies one of its arguments (waterbodies_df), which is somewhat poor practice given its otherwise functional nature. Consider refactoring
    waterbodies_df, q0, t0, lastobs_df, da_parameter_dict = nwm_initial_warmstate_preprocess(
        break_network_at_waterbodies,
        restart_parameters,
        data_assimilation_parameters,
        param_df.index,
        waterbodies_df,
        segment_list=None,
        wbodies_list=None,
        showtiming=showtiming,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    # The inputs below assume a very pedantic setup
    # with each run set explicitly defined, so...
    # TODO: Make this more flexible.
    run_sets = forcing_parameters.get("qlat_forcing_sets", False)

    if "data_assimilation_parameters" in compute_parameters:
        if "data_assimilation_sets" in data_assimilation_parameters:
            da_sets = data_assimilation_parameters.get("data_assimilation_sets", [])
        else:
            da_sets = [{} for _ in run_sets]

    if "wrf_hydro_parity_check" in output_parameters:
        parity_sets = parity_parameters.get("parity_check_compare_file_sets", [])
    else:
        parity_sets = []

    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    cpu_pool = compute_parameters.get("cpu_pool", None)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)

    qlats, usgs_df = nwm_forcing_preprocess(
        run_sets[0],
        forcing_parameters,
        da_sets[0] if data_assimilation_parameters else {},
        data_assimilation_parameters,
        break_network_at_waterbodies,
        param_df.index,
        lastobs_df.index,
        t0,
        showtiming,
        verbose,
        debuglevel,
    )

    for run_set_iterator, run in enumerate(run_sets):

        t0 = run.get("t0")
        dt = run.get("dt")
        nts = run.get("nts")

        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        run_results = nwm_route(
            connections,
            rconn,
            wbody_conn,
            reaches_bytw,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            cpu_pool,
            dt,
            nts,
            qts_subdivisions,
            independent_networks,
            param_df,
            q0,
            qlats,
            usgs_df,
            lastobs_df,
            da_parameter_dict,
            assume_short_ts,
            return_courant,
            waterbodies_df,
            waterbody_parameters,
            waterbody_types_df,
            waterbody_type_specified,
            diffusive_parameters,
            showtiming,
            verbose,
            debuglevel,
        )

        # No forcing to prepare for the last loop
        if run_set_iterator < len(run_sets) - 1:
            qlats, usgs_df = nwm_forcing_preprocess(
                run_sets[run_set_iterator + 1],
                forcing_parameters,
                da_sets[run_set_iterator + 1] if data_assimilation_parameters else {},
                data_assimilation_parameters,
                break_network_at_waterbodies,
                param_df.index,
                lastobs_df.index,
                t0 + timedelta(seconds = dt * nts),
                showtiming,
                verbose,
                debuglevel,
            )

            q0 = new_nwm_q0(run_results)

            if data_assimilation_parameters:
                lastobs_df = new_lastobs(run_results, dt * nts)

            # TODO: Confirm this works with Waterbodies turned off
            waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)

            if waterbody_type_specified:
                waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)

        nwm_output_generator(
            run,
            run_results,
            supernetwork_parameters,
            output_parameters,
            parity_parameters,
            restart_parameters,
            parity_sets[run_set_iterator] if parity_parameters else {},
            qts_subdivisions,
            compute_parameters.get("return_courant", False),
            showtiming,
            verbose,
            debuglevel,
        )

    # nwm_final_output_generator()

    if verbose:
        print("process complete")
    if showtiming:
        print("%s seconds." % (time.time() - main_start_time))


async def main_v03_async(argv):
    """
    Handles the creation of the input parameter dictionaries
    from an input file and then sequences the execution of the
    t-route routing agorithm on a series of execution loops.
    """
    args = _handle_args_v03(argv)  # async shares input framework with non-async
    (
        log_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        diffusive_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v03(args)

    verbose = log_parameters.get("verbose", None)
    showtiming = log_parameters.get("showtiming", None)
    debuglevel = log_parameters.get("debuglevel", 0)

    if showtiming:
        main_start_time = time.time()

    (
        connections,
        param_df,
        wbody_conn,
        waterbodies_df,
        waterbody_types_df,
        break_network_at_waterbodies,
        waterbody_type_specified,
        independent_networks,
        reaches_bytw,
        rconn,
    ) = nwm_network_preprocess(
        supernetwork_parameters,
        waterbody_parameters,
        showtiming=showtiming,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    # TODO: This function modifies one of its arguments (waterbodies_df), which is somewhat poor practice given its otherwise functional nature. Consider refactoring
    waterbodies_df, q0, t0, lastobs_df, da_parameter_dict = nwm_initial_warmstate_preprocess(
        break_network_at_waterbodies,
        restart_parameters,
        data_assimilation_parameters,
        param_df.index,
        waterbodies_df,
        segment_list=None,
        wbodies_list=None,
        showtiming=showtiming,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    # The inputs below assume a very pedantic setup
    # with each run set explicitly defined, so...
    # TODO: Make this more flexible.
    run_sets = forcing_parameters.get("qlat_forcing_sets", False)

    if "data_assimilation_parameters" in compute_parameters:
        if "data_assimilation_sets" in data_assimilation_parameters:
            da_sets = data_assimilation_parameters.get("data_assimilation_sets", [])
        else:
            da_sets = [{} for _ in run_sets]

    if "wrf_hydro_parity_check" in output_parameters:
        parity_sets = parity_parameters.get("parity_check_compare_file_sets", [])
    else:
        parity_sets = []

    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    # TODO: Determine parameterization of the CPU and Threading pools
    # TODO: Make sure default values from dict.get for pool sizes work
    # e.g., is this valid: `ThreadPoolExecutor(max_workers=None)`?
    COMPUTE_cpu_pool = compute_parameters.get("cpu_pool", None)
    # IO_cpu_pool = compute_parameters.get("cpu_pool_IO", None)
    IO_cpu_pool = COMPUTE_cpu_pool
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)

    FORCE_KERNEL_THREAD = False
    if FORCE_KERNEL_THREAD:
        pool_IO = None
        pool_Processing = None
    else:
        pool_IO = concurrent.futures.ThreadPoolExecutor(max_workers=IO_cpu_pool)
        pool_Processing = concurrent.futures.ThreadPoolExecutor(max_workers=COMPUTE_cpu_pool)

    loop = asyncio.get_running_loop()

    forcings_task = loop.run_in_executor(
        pool_IO,
        nwm_forcing_preprocess,
        run_sets[0],
        forcing_parameters,
        da_sets[0] if data_assimilation_parameters else {},
        data_assimilation_parameters,
        break_network_at_waterbodies,
        param_df.index,
        lastobs_df.index,
        t0,
        showtiming,
        verbose,
        debuglevel,
    )

    run_set_iterator = 0
    for run_set_iterator, run in enumerate(run_sets[:-1]):

        t0 = run.get("t0")
        dt = run.get("dt")
        nts = run.get("nts")

        qlats, usgs_df = await forcings_task

        # TODO: confirm utility of visual parity check in async execution
        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        model_task = loop.run_in_executor(
            pool_Processing,
            nwm_route,
            connections,
            rconn,
            wbody_conn,
            reaches_bytw,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            COMPUTE_cpu_pool,
            dt,
            nts,
            qts_subdivisions,
            independent_networks,
            param_df,
            q0,
            qlats,
            usgs_df,
            lastobs_df,
            da_parameter_dict,
            assume_short_ts,
            return_courant,
            waterbodies_df,
            waterbody_parameters,
            waterbody_types_df,
            waterbody_type_specified,
            diffusive_parameters,
            showtiming,
            verbose,
            debuglevel,
        )

        forcings_task = loop.run_in_executor(
            pool_IO,
            nwm_forcing_preprocess,
            run_sets[run_set_iterator + 1],
            forcing_parameters,
            da_sets[run_set_iterator + 1] if data_assimilation_parameters else {},
            data_assimilation_parameters,
            break_network_at_waterbodies,
            param_df.index,
            lastobs_df.index,
            t0 + timedelta(seconds = dt * nts),
            showtiming,
            verbose,
            debuglevel,
        )

        run_results = await model_task

        q0 = new_nwm_q0(run_results)

        if data_assimilation_parameters:
            lastobs_df = new_lastobs(run_results, dt * nts)

        # TODO: Confirm this works with Waterbodies turned off
        waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)

        if waterbody_type_specified:
            waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)

        output_task = loop.run_in_executor(
            pool_IO,
            nwm_output_generator,
            run,
            run_results,
            supernetwork_parameters,
            output_parameters,
            parity_parameters,
            restart_parameters,
            parity_sets[run_set_iterator] if parity_parameters else {},
            qts_subdivisions,
            compute_parameters.get("return_courant", False),
            showtiming,
            verbose,
            debuglevel,
        )

    # For the last loop, no next forcing or warm state is needed for execution.
    run_set_iterator += 1
    run = run_sets[run_set_iterator]

    t0 = run.get("t0")
    dt = run.get("dt")
    nts = run.get("nts")

    qlats, usgs_df = await forcings_task

    # TODO: confirm utility of visual parity check in async execution
    if parity_sets:
        parity_sets[run_set_iterator]["dt"] = dt
        parity_sets[run_set_iterator]["nts"] = nts

    model_task = loop.run_in_executor(
        pool_Processing,
        nwm_route,
        connections,
        rconn,
        wbody_conn,
        reaches_bytw,
        parallel_compute_method,
        compute_kernel,
        subnetwork_target_size,
        COMPUTE_cpu_pool,
        dt,
        nts,
        qts_subdivisions,
        independent_networks,
        param_df,
        q0,
        qlats,
        usgs_df,
        lastobs_df,
        da_parameter_dict,
        assume_short_ts,
        return_courant,
        waterbodies_df,
        waterbody_parameters,
        waterbody_types_df,
        waterbody_type_specified,
        diffusive_parameters,
        showtiming,
        verbose,
        debuglevel,
    )

    # nwm_final_output_generator()
    run_results = await model_task

    # These warmstates are never used for modeling, but
    # should be availble for last outputs.
    q0 = new_nwm_q0(run_results)

    if data_assimilation_parameters:
        lastobs_df = new_lastobs(run_results, dt * nts)

    waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)

    if waterbody_type_specified:
        waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)

    output_task = await loop.run_in_executor(
        pool_IO,
        nwm_output_generator,
        run,
        run_results,
        supernetwork_parameters,
        output_parameters,
        parity_parameters,
        restart_parameters,
        parity_sets[run_set_iterator] if parity_parameters else {},
        qts_subdivisions,
        compute_parameters.get("return_courant", False),
        showtiming,
        verbose,
        debuglevel,
    )

    if verbose:
        print("process complete")
    if showtiming:
        print("%s seconds." % (time.time() - main_start_time))

        """
        Asynchronous execution Psuedocode
        Sync1: Prepare first warmstate from files
        Sync1: Prepare first forcing from files

        For first forcing set
            Sync2a: run model
            Sync2b: begin preparing next forcing
            Sync3a - AFTER Sync2a, prepare next warmstate (last state of model run)
            Sync3b: write any output from Sync2a
            Loop has to wait for Sync2a+b+Sync3a, does not have to wait for Sync3b
                  if next forcing prepared
        """

    pool_IO.shutdown(wait=True)
    pool_Processing.shutdown(wait=True)


if __name__ == "__main__":
    v_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    v_parser.add_argument(
        "-V",
        "--input-version",
        default=3,
        nargs="?",
        choices=[2, 3, 4],
        type=int,
        help="Use version 2 or 3 of the input format. Default 3",
    )
    v_args = v_parser.parse_known_args()
    if v_args[0].input_version == 4:
        coroutine = main_v03_async(v_args[1])
        asyncio.run(coroutine)
        # loop.run_until_complete(coroutine)
    if v_args[0].input_version == 3:
        main_v03(v_args[1])
    if v_args[0].input_version == 2:
        print("Running main v02")
        main_v02(v_args[1])
