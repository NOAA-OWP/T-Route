# command line args and usage with: python compute_nhd_routing_SingleSeg.py --help
# example to run test: python compute_nhd_routing_SingleSeg.py -v --test
# example usage: python compute_nhd_routing_SingleSeg.py -v -t -w -onc -n Mainstems_CONUS
#                python compute_nhd_routing_SingleSeg.py -n Mainstems_CONUS --nts 1440 --parallel &
#                python compute_nhd_routing_SingleSeg.py -v --parallel
#                python compute_nhd_routing_SingleSeg.py -v -w --parallel
#                python compute_nhd_routing_SingleSeg.py -v -w -n Brazos_LowerColorado_Named_Streams
#                python compute_nhd_routing_SingleSeg.py -v -w -n Brazos_LowerColorado_Named_Streams --parallel
#                python compute_nhd_routing_SingleSeg.py -f ../../test/input/json/CustomInput.json
#                python compute_nhd_routing_SingleSeg.py -f ../../test/input/json/CustomInput_short.json

# a notebook-based version of very similar code is found here:
# https://github.com/NOAA-OWP/t-route/blob/master/notebooks/compute_nhd_routing_v2_clean_with_lateral_inflow_data.ipynb

## Parallel execution
import os
import sys
import time
import traceback
import numpy as np
import argparse
import pathlib
import pandas as pd
import netCDF4
import csv
from datetime import datetime
import multiprocessing
import glob
import xarray as xr
from tqdm import tqdm


def _handle_args():
    # TODO: Convert to global argparser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debuglevel",
        help="Set the debuglevel",
        dest="debuglevel",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output (leave blank for quiet output)",
        dest="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--test",
        "--run-pocono2-test-example",
        help="Use the data values stored in the repository for a test of the Pocono network",
        dest="run_pocono2_test",
        action="store_true",
    )
    parser.add_argument(
        "--test-full-pocono",
        "--run-pocono1-test-example",
        help="Use the data values stored in the repository for a test of the Mainstems_CONUS network",
        dest="run_pocono1_test",
        action="store_true",
    )
    parser.add_argument(
        "--sts",
        "--assume-short-ts",
        help="Use the previous timestep value for upstream flow",
        dest="assume_short_ts",
        action="store_true",
    )
    parser.add_argument(
        "--network-only",
        help="Perform the network decomposition, then stop before executing the routing computation",
        dest="do_network_analysis_only",
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
        "-onc",
        "--write-output-nc",
        nargs="?",
        help="Write netcdf output files to this folder (omit flag for no netcdf writing)",
        dest="nc_output_folder",
        const="../../test/output/nc",
    )
    parser.add_argument(
        "--dt",
        "--qlateral-time-step",
        help="Set the default timestep length",
        dest="dt",
        default=300,
    )
    parser.add_argument(
        "--nts",
        "--number-of-qlateral-timesteps",
        help="Set the number of timesteps to execute. If used with ql_file or ql_folder, nts must be less than len(ql) x qN.",
        dest="nts",
        default=144,
    )
    parser.add_argument(
        "--qN",
        "--qts_subdivisions",
        help="number of simulation timesteps per qlateral timestep",
        dest="qts_subdivisions",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--wrf_hydro_channel_restart_file",
        dest="wrf_hydro_channel_restart_file",
        help="provide a WRF-Hydro channel warm state file (may be the same as waterbody restart file)",
    )
    parser.add_argument(
        "--wrf_hydro_channel_ID_crosswalk_file",
        dest="wrf_hydro_channel_ID_crosswalk_file",
        help="provide an xarray-readable file that defines the order of the outputs in the channel restart file. Specify the ID field with --wrf_hydro_channel_ID_crosswalk_file_field_name",
    )
    parser.add_argument(
        "--wrf_hydro_channel_ID_crosswalk_file_field_name",
        dest="wrf_hydro_channel_ID_crosswalk_file_field_name",
        help="Name of the column providing the channel segment IDs in the channel crosswalk file",
        default="ID",
    )
    parser.add_argument(
        "--wrf_hydro_waterbody_restart_file",
        dest="wrf_hydro_waterbody_restart_file",
        help="provide a WRF-Hydro waterbody warm state file (may be the same as channel restart file)",
    )
    parser.add_argument(
        "--wrf_hydro_waterbody_ID_crosswalk_file",
        dest="wrf_hydro_waterbody_ID_crosswalk_file",
        help="provide an xarray-readable file that defines the order of the outputs in the waterbody restart file. Specify the ID field with --wrf_hydro_waterbody_ID_crosswalk_file_field_name",
    )
    parser.add_argument(
        "--wrf_hydro_waterbody_ID_crosswalk_file_field_name",
        dest="wrf_hydro_waterbody_ID_crosswalk_file_field_name",
        help="Name of the column providing the waterbody segment IDs in the waterbody crosswalk file",
        default="ID",
    )
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
    parser.add_argument(
        "--qlat_file_pattern_filter",
        help="Provide a globbing pattern to identify files in the Wrf-Hydro qlateral output file folder",
        dest="qlat_file_pattern_filter",
        default="/*.CHRTOUT_DOMAIN1",
    )
    parser.add_argument(
        "-t",
        "--showtiming",
        help="Set the showtiming (omit flag for no timing information)",
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
        "--sort-ordered-networks",
        help="Sort networks by size, largest to smallest, so the deepest ones (network['maximum_reach_seqorder']) start earliest in a given network order",
        dest="sort_networks",
        action="store_true",
    )
    supernetwork_arg_group = parser.add_mutually_exclusive_group()
    supernetwork_arg_group.add_argument(
        "-n",
        "--supernetwork",
        help="Choose from among the pre-programmed supernetworks (Pocono_TEST1, Pocono_TEST2, LowerColorado_Conchos_FULL_RES, Brazos_LowerColorado_ge5, Brazos_LowerColorado_FULL_RES, Brazos_LowerColorado_Named_Streams, CONUS_ge5, Mainstems_CONUS, CONUS_Named_Streams, CONUS_FULL_RES_v20",
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
        "--parallel",
        help="Use the parallel computation engine (omit flag for serial computation)",
        dest="parallel_compute",
        action="store_true",
    )
    parser.add_argument(
        "--cpu-pool",
        help="Assign the number of cores to multiprocess across.",
        dest="cpu_pool",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-p",
        "--percentage_complete",
        help="Prints the percentage complete of the network computation.",
        dest="percentage_complete",
        action="store_true",
    )

    args = parser.parse_args()

    # TODO: Add any other checking
    # TODO: This check is probably no longer needed; 'custom' functionality is managed by the 'custom-input-file' argument
    if args.supernetwork == "custom" and not args.customnetworkfile:
        parser.error(
            r"If 'custom' is selected for the supernetwork, you must enter a path to a supernetwork-describing .yaml or .json file"
        )

    return args


root = pathlib.Path("../..").resolve()

sys.setrecursionlimit(4000)
sys.path.append(os.path.join(root, r"src", r"python_framework_v01"))
sys.path.append(os.path.join(root, r"src", r"python_framework_v02", "troute"))
sys.path.append(os.path.join(root, r"src", r"python_routing_v01"))
fortran_routing_dir = os.path.join(
    root, r"src", r"fortran_routing", r"mc_pylink_v00", r"MC_singleSeg_singleTS"
)
fortran_reservoir_dir = os.path.join(
    root, r"src", r"fortran_routing", r"mc_pylink_v00", r"Reservoir_singleTS"
)
sys.path.append(fortran_routing_dir)

## Muskingum Cunge
COMPILE = True


def in_wsl() -> bool:
    """
    WSL is thought to be the only common Linux kernel with Microsoft in the name.
    """

    return "microsoft" in (os.uname().release).lower()


if COMPILE:
    try:
        import subprocess

        fortran_compile_call = []
        fortran_compile_call.append(r"f2py3")
        fortran_compile_call.append(r"-c")
        fortran_compile_call.append(r"varPrecision.f90")
        fortran_compile_call.append(r"MCsingleSegStime_f2py_NOLOOP.f90")
        fortran_compile_call.append(r"-m")
        fortran_compile_call.append(r"mc_sseg_stime")
        if in_wsl():  # disable optimization for compiling on WSL
            fortran_compile_call.append(r"--noopt")
        else:
            fortran_compile_call.append(r"--opt='-O1'")
        subprocess.run(
            fortran_compile_call,
            cwd=fortran_routing_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from mc_sseg_stime import muskingcunge_module as mc
    except Exception as e:
        print(e)
        traceback.print_exc()
else:
    from mc_sseg_stime import muskingcunge_module as mc

connections = None
networks = None
qlateral = None
waterbodies_df = None
waterbody_initial_states_df = None
channel_initial_states_df = None

time_index = 0  # time
flowval_index = 1  # flowval
velval_index = 2  # velval
depthval_index = 3  # depthval
qlatval_index = 4  # qlatval
storageval_index = 5  # storageval
qlatCumval_index = 6  # qlatCumval
kinCelerity_index = 7  # ck
courant_index = 8  # cn
X_index = 9  # X

## network and reach utilities
import nhd_network_utilities_v01 as nnu
import nhd_io as nio
import nhd_reach_utilities as nru

sys.path.append(fortran_reservoir_dir)
if COMPILE:
    try:
        import subprocess

        fortran_compile_call = []
        fortran_compile_call.append(r"f2py3")
        fortran_compile_call.append(r"-c")
        fortran_compile_call.append(r"varPrecision.f90")
        fortran_compile_call.append(r"module_levelpool.f90")
        fortran_compile_call.append(r"-m")
        fortran_compile_call.append(r"pymodule_levelpool")
        if in_wsl():  # disable optimization for compiling on WSL
            fortran_compile_call.append(r"--noopt")
        else:
            fortran_compile_call.append(r"--opt='-O1'")
        subprocess.run(
            fortran_compile_call,
            cwd=fortran_reservoir_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from pymodule_levelpool import module_levelpool as rc
    except Exception as e:
        print(e)
        traceback.print_exc()
else:
    from pymodule_levelpool import module_levelpool as rc


def compute_network(
    flowveldepth_connect,
    terminal_segment,
    supernetwork_parameters,
    waterbody_parameters,
    waterbody,
    nts=1,
    dt=60,
    qts_subdivisions=1,
    verbose=False,
    debuglevel=0,
    csv_output=None,
    nc_output_folder=None,
    assume_short_ts=False,
):
    global connections
    global networks
    global qlateral

    network = networks[terminal_segment]
    flowveldepth = {
        connection: np.zeros(np.array([nts, 10]))
        for connection in (network["all_segments"])
    }

    if debuglevel <= -1:
        print(
            f"\nExecuting simulation on network {terminal_segment} beginning with streams of order {network['maximum_reach_seqorder']}"
        )

    ordered_reaches = {}
    for head_segment, reach in network["reaches"].items():
        if reach["seqorder"] not in ordered_reaches:
            ordered_reaches[reach["seqorder"]] = []
        ordered_reaches[reach["seqorder"]].append((head_segment, reach))

    # initialize write to files variable
    # TODO: Make one dictionary containing all output options
    # The options below would become:
    # writeToCSV = out_opt.get(csv_output,None)
    # writeToNETCDF = out_opt.get(nc_output,None)
    writeToCSV = csv_output
    writeToNETCDF = nc_output_folder

    for ts in range(0, nts):
        for x in range(network["maximum_reach_seqorder"], -1, -1):
            for head_segment, reach in ordered_reaches[x]:
                # print(f'timestep: {ts}\n')
                # for loops should contain both waterbodies and mc_reach as it loops entire network
                # TODO: Prune these inputs
                qup_reach, quc_reach = compute_reach_upstream_flows(
                    flowveldepth_connect=flowveldepth_connect,
                    flowveldepth=flowveldepth,
                    head_segment=head_segment,
                    reach=reach,
                    network=network,
                    supernetwork_parameters=supernetwork_parameters,
                    waterbody=waterbody,
                    ts=ts,
                    dt=dt,
                    verbose=verbose,
                    debuglevel=debuglevel,
                    assume_short_ts=assume_short_ts,
                )
                if waterbody:
                    compute_level_pool_reach_up2down(
                        flowveldepth=flowveldepth,
                        qlateral=qlateral,
                        qup_reach=qup_reach,
                        quc_reach=quc_reach,
                        head_segment=head_segment,
                        reach=reach,
                        network=network,
                        supernetwork_parameters=supernetwork_parameters,
                        waterbody_parameters=waterbody_parameters,
                        waterbody=waterbody,
                        ts=ts,
                        dt=dt,
                        qts_subdivisions=qts_subdivisions,
                        verbose=verbose,
                        debuglevel=debuglevel,
                        assume_short_ts=assume_short_ts,
                    )

                else:
                    compute_mc_reach_up2down(
                        flowveldepth=flowveldepth,
                        qlateral=qlateral,
                        qup_reach=qup_reach,
                        quc_reach=quc_reach,
                        head_segment=head_segment,
                        reach=reach,
                        supernetwork_parameters=supernetwork_parameters,
                        ts=ts,
                        dt=dt,
                        qts_subdivisions=qts_subdivisions,
                        verbose=verbose,
                        debuglevel=debuglevel,
                        assume_short_ts=assume_short_ts,
                    )

    if writeToCSV:
        for x in range(network["maximum_reach_seqorder"], -1, -1):
            for head_segment, reach in ordered_reaches[x]:
                writeArraytoCSV(
                    connections=connections,
                    flowveldepth=flowveldepth,
                    reach=reach,
                    verbose=verbose,
                    debuglevel=debuglevel,
                    outputOptions=writeToCSV,
                )

    if writeToNETCDF:
        writeArraytoNC(
            connections=connections,
            flowveldepth=flowveldepth,
            network=network,
            nts=nts,
            qts_subdivisions=qts_subdivisions,
            dt=dt,
            verbose=verbose,
            debuglevel=debuglevel,
            pathToOutputFile=writeToNETCDF,
        )

    return {terminal_segment: flowveldepth[terminal_segment]}


# TODO: generalize with a direction flag
def compute_reach_upstream_flows(
    flowveldepth_connect,
    flowveldepth,
    head_segment=None,
    reach=None,
    network=None,
    supernetwork_parameters=None,
    waterbody=None,
    ts=0,
    dt=60,
    verbose=False,
    debuglevel=0,
    assume_short_ts=False,
):
    global connections
    global channel_initial_states_df

    # upstream flow per reach
    qup = 0.0
    quc = 0.0
    ########################
    if waterbody:
        upstreams_list = set()
        for rr in network["receiving_reaches"]:
            for us in connections[rr]["upstreams"]:
                upstreams_list.add(us)
        # this step was critical -- there were receiving reaches that were junctions
        # with only one of their upstreams out of the network. The other was inside
        # the network, so it caused a lookup error.
        upstreams_list = upstreams_list - network["all_segments"]
        us_flowveldepth = flowveldepth_connect

    elif head_segment in network["receiving_reaches"]:
        # TODO: confirm this logic, to make sure we don't double count the head
        upstreams_list = connections[reach["reach_head"]]["upstreams"]
        us_flowveldepth = flowveldepth
        us_flowveldepth.update(flowveldepth_connect)

    else:
        upstreams_list = connections[reach["reach_head"]]["upstreams"]
        us_flowveldepth = flowveldepth

    for us in upstreams_list:
        if us != supernetwork_parameters["terminal_code"]:  # Not Headwaters
            quc += us_flowveldepth[us][ts][flowval_index]

            if ts == 0:
                # Initialize qup from warm state array
                qup += channel_initial_states_df.loc[us, "qd0"]
            else:
                qup += us_flowveldepth[us][ts - 1][flowval_index]

    if assume_short_ts:
        quc = qup

    return quc, qup


# TODO: generalize with a direction flag
def compute_mc_reach_up2down(
    flowveldepth,
    qlateral,
    qup_reach,
    quc_reach,
    head_segment=None,
    reach=None,
    supernetwork_parameters=None,
    ts=0,
    dt=60,
    qts_subdivisions=1,
    verbose=False,
    debuglevel=0,
    assume_short_ts=False,
):
    global connections

    if debuglevel <= -2:
        print(
            f"\nreach: {head_segment} (order: {reach['seqorder']} n_segs: {len(reach['segments'])})"
        )

    qup = qup_reach
    quc = quc_reach

    current_segment = reach["reach_head"]
    # next_segment = connections[current_segment]["downstream"]

    while True:
        data = connections[current_segment]["data"]
        # for now treating as constant per reach
        bw = data[supernetwork_parameters["bottomwidth_col"]]
        tw = data[supernetwork_parameters["topwidth_col"]]
        twcc = data[supernetwork_parameters["topwidthcc_col"]]
        dx = data[supernetwork_parameters["length_col"]]
        n_manning = data[supernetwork_parameters["manningn_col"]]
        n_manning_cc = data[supernetwork_parameters["manningncc_col"]]
        cs = data[supernetwork_parameters["ChSlp_col"]]
        s0 = data[supernetwork_parameters["slope_col"]]

        # TODO: update this extremely simplistic handling of timestep adjustment
        # allow shorter timesteps
        qts = int(ts / qts_subdivisions)
        qlat = qlateral[current_segment][qts]

        if ts == 0:
            # initialize from initial states
            qdp = channel_initial_states_df.loc[current_segment, "qd0"]
            velp = 0
            depthp = channel_initial_states_df.loc[current_segment, "h0"]
        else:
            qdp = flowveldepth[current_segment][ts - 1][flowval_index]
            velp = 0  # flowveldepth[current_segment]["velval"][-1]
            depthp = flowveldepth[current_segment][ts - 1][depthval_index]

        # run M-C model
        qdc, velc, depthc, ck, cn, X = singlesegment(
            dt=dt,
            qup=qup,
            quc=quc,
            qdp=qdp,
            qlat=qlat,
            dx=dx,
            bw=bw,
            tw=tw,
            twcc=twcc,
            n_manning=n_manning,
            n_manning_cc=n_manning_cc,
            cs=cs,
            s0=s0,
            velp=velp,
            depthp=depthp,
        )
        # storage
        volumec = dt * (quc - qdc + qlat)
        # TODO: This qlatCum is invalid as a cumulative value unless time is factored in
        qlatCum = qlat * dt
        if ts > 0:
            volumec = volumec + flowveldepth[current_segment][ts - 1][storageval_index]
            qlatCum = qlatCum + flowveldepth[current_segment][ts - 1][qlatCumval_index]

        # for next segment qup / quc use the just-now and previously
        # calculated flow values from the current segment
        qup = qdp
        quc = qdc
        if assume_short_ts:
            # if we are assuming time steps are short, we can
            # assume that the previous flow value sufficiently
            # represents both the previous and current state.
            # This approximation is entirely un-necessary in
            # this framework, which preserves the connectivity
            # between reach segments in time and in space, but
            # it allows us to approximate the behavior of the
            # previous version of the model, which made this
            # assumption out of necessity to allow out-of-order
            # computation on segments within a given time step,
            # which was a peculiar requirement of the parallel-
            # ization scheme used during execution.
            quc = qup = qdp

        """
            One time step, moving from one upstream segment 
            to the calculation on it's downstream neighbor...
            
            Normal calculation:

            current_segment (upstream)
            qup      qdp╮
             │  Q-->  │ ┊
             │━━━━━━━━│ ╰->╮
             │        │    ┊ next_segment (downstream)
            quc      qdc╮  ╰-qup      qdp
                        ┊     │  Q-->  │
                        ╰->╮  │━━━━━━━━│
                           ┊  │        │  
                           ╰-quc      qdc

            Short-time-step calculation:

            current_segment (upstream)
            qup      qdp╮
             │  Q-->  │ ┊
             │━━━━━━━━│ ╰->╮
             │        │    ┊ next_segment (downstream)
            quc      qdc   ├-qup      qdp
                           ┊  │  Q-->  │
                           ┊  │━━━━━━━━│
                           ┊  │        │  
                           ╰-quc      qdc
        """
        # update flowveldepth values for currentsegment for current timestep
        flowveldepth[current_segment][ts] = [
            ts * dt,
            qdc,
            velc,
            depthc,
            qlat,
            volumec,
            qlatCum,
            ck,
            cn,
            X,
        ]

        next_segment = connections[current_segment]["downstream"]
        if current_segment == reach["reach_tail"]:
            if debuglevel <= -2:
                print(f"{current_segment} (tail)")
            break
        if debuglevel <= -3:
            print(f"{current_segment} --> {next_segment}\n")
        # current_segment = next_segment
        # next_segment = connections[current_segment]["downstream"]
        current_segment = next_segment
    # end loop initialized the MC vars
    # end while loop


def compute_level_pool_reach_up2down(
    flowveldepth,
    qlateral,
    qup_reach,
    quc_reach,
    head_segment=None,
    reach=None,
    network=None,
    supernetwork_parameters=None,
    waterbody_parameters=None,
    waterbody=None,
    ts=0,
    dt=60,
    qts_subdivisions=1,
    verbose=False,
    debuglevel=0,
    assume_short_ts=False,
):
    global connections
    global waterbodies_df
    global waterbody_initial_states_df

    if debuglevel <= -2:
        print(
            f"\nreach: {head_segment} (order: {reach['seqorder']} n_segs: {len(reach['segments'])})"
        )

    qup = qup_reach
    quc = quc_reach

    current_segment = reach["reach_tail"]
    if ts == 0:
        # Initialize from warm state
        depthp = waterbody_initial_states_df.loc[waterbody, "h0"]
    else:
        depthp = flowveldepth[current_segment][ts - 1][depthval_index]

    # This Qlat gathers all segments of the waterbody
    qts = int(ts / qts_subdivisions)
    qlat = sum([qlateral[seg][qts] for seg in network["all_segments"]])
    if debuglevel <= -2:
        print(f"executing reservoir computation on waterbody: {waterbody}")

    wb_params = waterbody_parameters["level_pool"]
    ln = waterbody
    qi0 = qup
    qi1 = quc
    ql = qlat
    dt = dt  # current timestep length
    ar = waterbodies_df.loc[waterbody, wb_params["level_pool_waterbody_area"]]
    we = waterbodies_df.loc[waterbody, wb_params["level_pool_weir_elevation"]]
    maxh = waterbodies_df.loc[
        waterbody, wb_params["level_pool_waterbody_max_elevation"]
    ]
    wc = waterbodies_df.loc[waterbody, wb_params["level_pool_outfall_weir_coefficient"]]
    wl = waterbodies_df.loc[waterbody, wb_params["level_pool_outfall_weir_length"]]
    # TODO: find the right value for this variable -- it should be in the parameter file!
    dl = (
        10 * wl
    )  # waterbodies_df.loc[waterbody, wb_params["level_pool_overall_dam_length"]]
    oe = waterbodies_df.loc[waterbody, wb_params["level_pool_orifice_elevation"]]
    oc = waterbodies_df.loc[waterbody, wb_params["level_pool_orifice_coefficient"]]
    oa = waterbodies_df.loc[waterbody, wb_params["level_pool_orifice_area"]]

    qdc, depthc = rc.levelpool_physics(
        dt, qi0, qi1, ql, ar, we, maxh, wc, wl, dl, oe, oc, oa, depthp
    )
    velc = 0  # We assume a zero velocity for level-pool reaches

    volumec = dt * (quc - qdc + qlat)
    # TODO: This qlatCum is invalid as a cumulative value unless time is factored in
    qlatCum = qlat * dt
    if ts > 0:
        volumec = volumec + flowveldepth[current_segment][ts - 1][storageval_index]
        qlatCum = qlatCum + flowveldepth[current_segment][ts - 1][qlatCumval_index]

    # TODO: There may be a more useful output value to provide here.
    # celerity values are nullified for reservoirs.
    ck = 0
    cn = 0
    X = 0

    flowveldepth[current_segment][ts] = [
        ts * dt,
        qdc,
        velc,
        depthc,
        qlat,
        volumec,
        qlatCum,
        ck,
        cn,
        X,
    ]


# ### Psuedocode
# Write Array  to CSV file
# arguments reach , pathToOutputFile
# using global connections and flowveldepth.
def writeArraytoCSV(
    connections,
    flowveldepth,
    reach,
    verbose=False,
    debuglevel=0,
    outputOptions={"csv_output_folder": "../../test/output/text"},
):

    # define CSV file Header
    header = ["time", "qlat", "q", "v", "d", "storage", "qlatCum", "ck", "cn", "X"]

    # Loop over reach segments
    current_segment = reach["reach_head"]
    next_segment = connections[current_segment]["downstream"]
    pathToOutputFile = outputOptions["csv_output_folder"]
    csv_output_segments = set(
        outputOptions.get("csv_output_segments", reach["segments"])
    )

    while True:
        if current_segment in csv_output_segments:
            filename = f"{pathToOutputFile}/{current_segment}.csv"  #
            if verbose:
                print(f"writing segment output to --> {filename}")
            with open(filename, "w+") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
                csvwriter.writerow(header)
                csvwriter.writerows(
                    zip(
                        flowveldepth[current_segment][:, time_index],
                        flowveldepth[current_segment][:, qlatval_index],
                        flowveldepth[current_segment][:, flowval_index],
                        flowveldepth[current_segment][:, velval_index],
                        flowveldepth[current_segment][:, depthval_index],
                        flowveldepth[current_segment][:, storageval_index],
                        flowveldepth[current_segment][:, qlatCumval_index],
                        flowveldepth[current_segment][:, kinCelerity_index],
                        flowveldepth[current_segment][:, courant_index],
                        flowveldepth[current_segment][:, X_index],
                    )
                )

        if current_segment == reach["reach_tail"]:
            if debuglevel <= -2:
                print(f"{current_segment} (tail)")
            break
        if debuglevel <= -3:
            print(f"{current_segment} --> {next_segment}\n")
        current_segment = next_segment
        next_segment = connections[current_segment]["downstream"]


# ### Psuedocode
# Write Array  to Arrays for Netcdf file and then call writeNC function to write output data to NC file
# arguments network, number of timesteps (nts),  timestep in seconds  (dt),  pathToOutputFile
# using global connections and flowveldepth.
def writeArraytoNC(
    connections,
    flowveldepth,
    network,
    nts,
    qts_subdivisions=1,
    dt=60,
    verbose=False,
    debuglevel=0,
    pathToOutputFile="../../test/output/text/",
):
    # create  array variables to copy from python "flowveldepth" which is global
    flowveldepth_data = {
        "segment": [],
        "time": [],
        "qlatval": [],
        "qlatCumval": [],
        "flowval": [],
        "depthval": [],
        "velval": [],
        "storageval": [],
        "ckval": [],
        "cnval": [],
        "xval": [],
    }

    ordered_reaches = {}
    for head_segment, reach in network["reaches"].items():
        if reach["seqorder"] not in ordered_reaches:
            ordered_reaches[reach["seqorder"]] = []
        ordered_reaches[reach["seqorder"]].append((head_segment, reach))

    # get data into array - preparation step
    TIME_WRITTEN = False
    write_segment = None
    for x in range(network["maximum_reach_seqorder"], -1, -1):
        for head_segment, reach in ordered_reaches[x]:
            current_segment = reach["reach_head"]
            # TODO If we use numpy arrays, will this append work?
            while True:
                # appending data from each segments to a single list  "flowveldepth_data"
                # preserving ordering same as segment in a reach
                flowveldepth_data["qlatval"].append(
                    flowveldepth[current_segment][:, qlatval_index]
                )
                flowveldepth_data["qlatCumval"].append(
                    flowveldepth[current_segment][:, qlatCumval_index]
                )
                flowveldepth_data["flowval"].append(
                    flowveldepth[current_segment][:, flowval_index]
                )
                flowveldepth_data["storageval"].append(
                    flowveldepth[current_segment][:, storageval_index]
                )
                flowveldepth_data["depthval"].append(
                    flowveldepth[current_segment][:, depthval_index]
                )
                flowveldepth_data["velval"].append(
                    flowveldepth[current_segment][:, velval_index]
                )
                flowveldepth_data["ckval"].append(
                    flowveldepth[current_segment][:, kinCelerity_index]
                )
                flowveldepth_data["cnval"].append(
                    flowveldepth[current_segment][:, courant_index]
                )
                flowveldepth_data["xval"].append(
                    flowveldepth[current_segment][:, X_index]
                )

                # write segment flowveldepth_data['segment']
                flowveldepth_data["segment"].append(current_segment)
                if not TIME_WRITTEN:
                    # write time only once - for first segment
                    flowveldepth_data["time"].append(
                        flowveldepth[current_segment][:, time_index]
                    )
                    TIME_WRITTEN = True

                if current_segment == reach["reach_tail"]:
                    write_segment = current_segment
                    if debuglevel <= -2:
                        print(f"{current_segment} (tail)")
                    break
                next_segment = connections[current_segment]["downstream"]
                if debuglevel <= -3:
                    print(f"{current_segment} --> {next_segment}\n")
                current_segment = next_segment

    # check number of timesteps should match the time the data is written
    if len(flowveldepth_data["time"][0]) != nts:
        print(
            f"Number of timesteps  {nts} does not match data timesteps {len(flowveldepth_data['time'][0])}\n"
        )
        return

    writeNC(
        flowveldepth_data=flowveldepth_data,
        segment_count=network["total_segment_count"],
        terminal_segment=write_segment,
        nts=nts,
        dt=dt,
        pathToOutputFile=pathToOutputFile,
        verbose=verbose,
        debuglevel=debuglevel,
    )


# ### Psuedocode
# Write  netcdf  file
# arguments flowveldepth , segment_count, terminal_segment (for filename and as identifier of reach)
#           number of timesteps (nts),  timestep in seconds  (dt),  verbose , debuglevel , pathToOutputFile
def writeNC(
    flowveldepth_data=None,
    segment_count=0,
    terminal_segment=None,
    nts=0,
    dt=60,
    pathToOutputFile="../../test/output/text",
    verbose=False,
    debuglevel=0,
):
    # start writing data to nc file
    filename = f"{pathToOutputFile}/{terminal_segment}.nc"  # ncfile'
    if verbose:
        print(f"writing netcdf output to --> {filename}")
    ncfile = netCDF4.Dataset(filename, mode="w", format="NETCDF4")
    # segcount = total segments for the current reach
    segcount = ncfile.createDimension("stations", segment_count)  # segment
    # timecount = number of timesteps
    timecount = ncfile.createDimension(
        "time", nts
    )  # unlimited axis (can be appended to).
    # analysis time =  current time , hence count =1
    analysistimecount = ncfile.createDimension(
        "analysistime", 1
    )  # unlimited axis (can be appended to).
    ncfile.title = f"Result of MC for Reach with terminal segment {terminal_segment}"
    ncfile.subtitle = "MC Python Module"
    ncfile.anything = f"streamflow , depth, velocity and lateral flow for Reach with terminal segment {terminal_segment}"
    # write streamflow
    flow = ncfile.createVariable(
        "flow", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    flow[:, :] = np.transpose(np.array(flowveldepth_data["flowval"], dtype=float))
    flow.units = "cu ft/s"
    flow.standard_name = "streamflow"  # this is a CF standard name
    # write volume
    storage = ncfile.createVariable(
        "storage", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    storage[:, :] = np.transpose(np.array(flowveldepth_data["storageval"], dtype=float))
    storage.units = "cu ft"
    storage.standard_name = "storage"  # this is a CF standard name
    # write depth
    depth = ncfile.createVariable(
        "depth", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    depth[:, :] = np.transpose(np.array(flowveldepth_data["depthval"], dtype=float))
    depth.units = "ft"  #
    depth.standard_name = "depth"  # this is a CF standard name
    # write velocity
    velocity = ncfile.createVariable(
        "velocity", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    velocity.units = "ft/s"  #
    velocity[:, :] = np.transpose(np.array(flowveldepth_data["velval"], dtype=float))
    velocity.standard_name = "velocity"  # this is a CF standard name
    # write  lateral flow (input from NWM)
    lateralflow = ncfile.createVariable(
        "lateralflow", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    lateralflow[:, :] = np.transpose(
        np.array(flowveldepth_data["qlatval"], dtype=float)
    )
    lateralflow.units = "cu ft/s"  #
    lateralflow.standard_name = "lateralflow"  # this is a CF standard name
    # write  Cummulitive lateral flow (input from NWM)
    lateralCumflow = ncfile.createVariable(
        "Cummulativelateralflow", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    lateralCumflow[:, :] = np.transpose(
        np.array(flowveldepth_data["qlatCumval"], dtype=float)
    )
    lateralCumflow.units = "cu ft/s"  #
    lateralCumflow.standard_name = (
        "Cummulativelateralflow"  # this is a CF standard name
    )

    # write  kinematic celerity
    celerity = ncfile.createVariable(
        "celerity", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    celerity.units = "ft/s"  #
    celerity[:, :] = np.transpose(np.array(flowveldepth_data["ckval"], dtype=float))
    celerity.standard_name = "celerity"

    # write  courant number
    courant = ncfile.createVariable(
        "courant", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    courant.units = "-"  #
    courant[:, :] = np.transpose(np.array(flowveldepth_data["cnval"], dtype=float))
    courant.standard_name = "courant number"

    # write  X parameter
    X_param = ncfile.createVariable(
        "X", np.float64, ("time", "stations")
    )  # note: unlimited dimension is leftmost
    X_param.units = "-"  #
    X_param[:, :] = np.transpose(np.array(flowveldepth_data["xval"], dtype=float))
    X_param.standard_name = "MC X parameter"

    # write time in seconds since  TODO get time from lateral flow from NWM
    time = ncfile.createVariable("time", np.float64, "time")
    time.units = "seconds since 2011-08-27 00:00:00"  ## TODO get time fron NWM as argument to this function
    time.long_name = "time"
    time[:] = flowveldepth_data["time"]
    # write segment ids
    segment = ncfile.createVariable("station_id", np.int, ("stations"))
    segment.long_name = "feature id"
    segment[:] = flowveldepth_data["segment"]
    # write analysis time = current time = time MC model is run
    analysistime = ncfile.createVariable("analysis_time", np.double, "analysistime")
    analysistime.long_name = "forecast_reference_time"
    analysistime.units = " minutes since " + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    analysistime[:] = 0
    #
    ncfile.close()
    if debuglevel <= -1:
        print(f"{filename} closed!")


## call to singlesegment MC Fortran Module
def singlesegment(
    dt=60,  # dt
    qup=None,  # qup
    quc=None,  # quc
    qdp=None,  # qdp
    qlat=None,  # ql
    dx=None,  # dx
    bw=None,  # bw
    tw=None,  # tw
    twcc=None,  # twcc
    n_manning=None,  #
    n_manning_cc=None,  # ncc
    cs=None,  # cs
    s0=None,  # s0
    velp=None,  # velocity at previous time step
    depthp=None,  # depth at previous time step
):
    # call Fortran routine
    return mc.muskingcungenwm(
        dt,
        qup,
        quc,
        qdp,
        qlat,
        dx,
        bw,
        tw,
        twcc,
        n_manning,
        n_manning_cc,
        cs,
        s0,
        0,
        depthp,
    )
    # return qdc, vel, depth, ck, cn, X


def sort_ordered_network(l, reverse=False):
    key = lambda x: x[1]["maximum_reach_seqorder"]
    l.sort(key=key, reverse=reverse)
    return l


# Main Routine
def main():
    args = _handle_args()

    global connections
    global networks
    global qlateral
    global waterbodies_df
    global waterbody_initial_states_df
    global channel_initial_states_df

    supernetwork = args.supernetwork
    custom_input_file = args.custom_input_file
    supernetwork_parameters = None
    waterbody_parameters = None
    if custom_input_file:
        (
            supernetwork_parameters,
            waterbody_parameters,
            forcing_parameters,
            restart_parameters,
            output_parameters,
            run_parameters,
            parity_parameters,
        ) = nio.read_custom_input(custom_input_file)

        break_network_at_waterbodies = run_parameters.get(
            "break_network_at_waterbodies", None
        )

        dt = run_parameters.get("dt", None)
        nts = run_parameters.get("nts", None)
        qts_subdivisions = run_parameters.get("qts_subdivisions", None)
        debuglevel = -1 * int(run_parameters.get("debuglevel", 0))
        verbose = run_parameters.get("verbose", None)
        showtiming = run_parameters.get("showtiming", None)
        percentage_complete = run_parameters.get("percentage_complete", None)
        do_network_analysis_only = run_parameters.get("do_network_analysis_only", None)
        assume_short_ts = run_parameters.get("assume_short_ts", None)
        parallel_compute = run_parameters.get("parallel_compute", None)
        cpu_pool = run_parameters.get("cpu_pool", None)
        sort_networks = run_parameters.get("sort_networks", None)

        csv_output = output_parameters.get("csv_output", None)
        nc_output_folder = output_parameters.get("nc_output_folder", None)

        qlat_const = forcing_parameters.get("qlat_const", None)
        qlat_input_file = forcing_parameters.get("qlat_input_file", None)
        qlat_input_folder = forcing_parameters.get("qlat_input_folder", None)
        qlat_file_pattern_filter = forcing_parameters.get(
            "qlat_file_pattern_filter", None
        )
        qlat_file_index_col = forcing_parameters.get("qlat_file_index_col", None)
        qlat_file_value_col = forcing_parameters.get("qlat_file_value_col", None)

        wrf_hydro_channel_restart_file = restart_parameters.get(
            "wrf_hydro_channel_restart_file", None
        )
        wrf_hydro_channel_ID_crosswalk_file = restart_parameters.get(
            "wrf_hydro_channel_ID_crosswalk_file", None
        )
        wrf_hydro_channel_ID_crosswalk_file_field_name = restart_parameters.get(
            "wrf_hydro_channel_ID_crosswalk_file_field_name", None
        )
        wrf_hydro_channel_restart_upstream_flow_field_name = restart_parameters.get(
            "wrf_hydro_channel_restart_upstream_flow_field_name", None
        )
        wrf_hydro_channel_restart_downstream_flow_field_name = restart_parameters.get(
            "wrf_hydro_channel_restart_downstream_flow_field_name", None
        )
        wrf_hydro_channel_restart_depth_flow_field_name = restart_parameters.get(
            "wrf_hydro_channel_restart_depth_flow_field_name", None
        )

        wrf_hydro_waterbody_restart_file = restart_parameters.get(
            "wrf_hydro_waterbody_restart_file", None
        )
        wrf_hydro_waterbody_ID_crosswalk_file = restart_parameters.get(
            "wrf_hydro_waterbody_ID_crosswalk_file", None
        )
        wrf_hydro_waterbody_ID_crosswalk_file_field_name = restart_parameters.get(
            "wrf_hydro_waterbody_ID_crosswalk_file_field_name", None
        )
        wrf_hydro_waterbody_crosswalk_filter_file = restart_parameters.get(
            "wrf_hydro_waterbody_crosswalk_filter_file", None
        )
        wrf_hydro_waterbody_crosswalk_filter_file_field_name = restart_parameters.get(
            "wrf_hydro_waterbody_crosswalk_filter_file_field_name", None
        )

    # Any specific commandline arguments will override the file
    # TODO: There are probably some pathological collisions that could
    # arise from this ordering ... check these out.

    else:
        break_network_at_waterbodies = args.break_network_at_waterbodies

        dt = int(args.dt)
        nts = int(args.nts)
        qts_subdivisions = args.qts_subdivisions
        qlat_const = float(args.qlat_const)
        qlat_input_folder = args.qlat_input_folder
        qlat_input_file = args.qlat_input_file
        qlat_file_pattern_filter = args.qlat_file_pattern_filter

        wrf_hydro_channel_restart_file = args.wrf_hydro_channel_restart_file
        wrf_hydro_channel_ID_crosswalk_file = args.wrf_hydro_channel_ID_crosswalk_file
        wrf_hydro_channel_ID_crosswalk_file_field_name = (
            args.wrf_hydro_channel_ID_crosswalk_file_field_name
        )

        wrf_hydro_waterbody_restart_file = args.wrf_hydro_waterbody_restart_file
        wrf_hydro_waterbody_ID_crosswalk_file = (
            args.wrf_hydro_waterbody_ID_crosswalk_file
        )
        wrf_hydro_waterbody_ID_crosswalk_file_field_name = (
            args.wrf_hydro_waterbody_ID_crosswalk_file_field_name
        )

        debuglevel = -1 * int(args.debuglevel)
        verbose = args.verbose
        showtiming = args.showtiming
        percentage_complete = args.percentage_complete
        do_network_analysis_only = args.do_network_analysis_only
        if args.csv_output_folder:
            csv_output = {"csv_output_folder": args.csv_output_folder}
        else:
            csv_output = None
        nc_output_folder = args.nc_output_folder
        assume_short_ts = args.assume_short_ts
        parallel_compute = args.parallel_compute
        sort_networks = args.sort_networks
        cpu_pool = args.cpu_pool

    run_pocono2_test = args.run_pocono2_test
    run_pocono1_test = args.run_pocono1_test

    if run_pocono2_test:
        if verbose:
            print("running test case for Pocono_TEST2 domain")
        # Overwrite the following test defaults
        supernetwork = "Pocono_TEST2"
        break_network_at_waterbodies = False
        qts_subdivisions = 1  # change qts_subdivisions = 1 as  default
        dt = 300 / qts_subdivisions
        nts = 144 * qts_subdivisions
        csv_output = {"csv_output_folder": os.path.join(root, "test", "output", "text")}
        nc_output_folder = os.path.join(root, "test", "output", "text")
        # test 1. Take lateral flow from re-formatted wrf-hydro output from Pocono Basin simulation
        qlat_input_file = os.path.join(
            root, r"test/input/geo/PoconoSampleData2/Pocono_ql_testsamp1_nwm_mc.csv"
        )

    elif run_pocono1_test:
        # NOTE: The test case for the Pocono basin was derived from this
        # resource on HydroShare, developed by aaraney and sourced from the
        # wrf_hydro_nwm_public repository on GitHub
        # see: https://www.hydroshare.org/resource/03ca354200e540018d44183598890448/
        # By downloading aaraney's docker job scheduler repo from GitHub, one can
        # execute the WRF-Hydro model that generated the test results
        # see: https://github.com/aaraney/NWM-Dockerized-Job-Scheduler
        if verbose:
            print("running test case for Pocono_TEST1 domain")
        # Overwrite the following test defaults

        NWM_test_path = os.path.join(
            root, "test/input/geo/NWM_2.1_Sample_Datasets/Pocono_TEST1/"
        )
        lakeparm_file = os.path.join(
            NWM_test_path,
            "primary_domain",
            "DOMAIN",
            "LAKEPARM.nc",
        )
        routelink_file = os.path.join(
            NWM_test_path,
            "primary_domain",
            "DOMAIN",
            "Route_Link.nc",
        )
        time_string = "2017-12-31_06-00_DOMAIN1"
        wrf_hydro_restart_file = os.path.join(
            NWM_test_path, "example_RESTART", "HYDRO_RST." + time_string
        )
        supernetwork_parameters = {
            "title_string": "Custom Input Example (using Pocono Test Example datafile)",
            "geo_file_path": routelink_file,
            "cols_as_text": False,
            "key_col": 16,  # "link",
            "downstream_col": 22,  # "to",
            "length_col": 3,  # "Length",
            "manningn_col": 18,  # "n",
            "manningncc_col": 19,  # "nCC",
            "slope_col": 8,  # "So",
            "bottomwidth_col": 0,  # "BtmWdth",
            "topwidth_col": 9,  # "TopWdth",
            "topwidthcc_col": 10,  # "TopWdthCC",
            "waterbody_col": 6,  # "NHDWaterbodyComID",
            "waterbody_null_code": -9999,
            "MusK_col": 4,  # "MusK",
            "MusX_col": 5,  # "MusX",
            "ChSlp_col": 1,  # "ChSlp",
            "terminal_code": 0,
            "driver_string": "NetCDF",
            "layer_string": 0,
        }
        waterbody_parameters = {
            "level_pool": {
                "level_pool_waterbody_parameter_file_path": lakeparm_file,
                "level_pool_waterbody_id": "lake_id",
                "level_pool_waterbody_area": "LkArea",
                "level_pool_weir_elevation": "WeirE",
                "level_pool_waterbody_max_elevation": "LkMxE",
                "level_pool_outfall_weir_coefficient": "WeirC",
                "level_pool_outfall_weir_length": "WeirL",
                "level_pool_overall_dam_length": "DamL",
                "level_pool_orifice_elevation": "OrificeE",
                "level_pool_orifice_coefficient": "OrificeC",
                "level_pool_orifice_area": "OrificeA",
            }
        }
        break_network_at_waterbodies = True
        qts_subdivisions = 12
        dt = 3600 / qts_subdivisions
        nts = 24 * qts_subdivisions
        csv_output = None
        nc_output_folder = None
        # build a time string to specify input date
        wrf_hydro_channel_restart_file = wrf_hydro_restart_file
        wrf_hydro_channel_ID_crosswalk_file = routelink_file
        wrf_hydro_channel_ID_crosswalk_file_field_name = "link"
        wrf_hydro_channel_restart_upstream_flow_field_name = "qlink1"
        wrf_hydro_channel_restart_downstream_flow_field_name = "qlink2"
        wrf_hydro_channel_restart_depth_flow_field_name = "hlink"
        wrf_hydro_waterbody_restart_file = wrf_hydro_restart_file
        wrf_hydro_waterbody_ID_crosswalk_file = lakeparm_file
        wrf_hydro_waterbody_ID_crosswalk_file_field_name = "lake_id"
        wrf_hydro_waterbody_crosswalk_filter_file = routelink_file
        wrf_hydro_waterbody_crosswalk_filter_file_field_name = "NHDWaterbodyComID"
        # wrf_hydro_waterbody_crosswalk_file_output_order_field= "AscendingIndex"
        qlat_input_folder = os.path.join(
            root, "test/input/geo/NWM_2.1_Sample_Datasets/Pocono_TEST1/example_CHRTOUT/"
        )
        qlat_file_pattern_filter = "/*.CHRTOUT_DOMAIN1"
        qlat_file_index_col = "feature_id"
        qlat_file_value_col = "q_lateral"

    if showtiming:
        program_start_time = time.time()
    if verbose:
        print(f"begin program t-route ...")

    # STEP 1: Read the supernetwork dataset and build the connections graph
    if verbose:
        print("creating supernetwork connections set")
    if showtiming:
        start_time = time.time()

    if supernetwork_parameters:
        supernetwork_values = nnu.get_nhd_connections(
            supernetwork_parameters=supernetwork_parameters,
            verbose=False,
            debuglevel=debuglevel,
        )

    else:
        test_folder = os.path.join(root, r"test")
        geo_input_folder = os.path.join(test_folder, r"input", r"geo")
        supernetwork_parameters, supernetwork_values = nnu.set_networks(
            supernetwork=supernetwork,
            geo_input_folder=geo_input_folder,
            verbose=False,
            debuglevel=debuglevel,
        )
        waterbody_parameters = nnu.set_waterbody_parameters(
            supernetwork=supernetwork,
            geo_input_folder=geo_input_folder,
            verbose=False,
            debuglevel=debuglevel,
        )

    if verbose:
        print("supernetwork connections set complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    connections = supernetwork_values[0]

    # STEP 2: Separate the networks and build the sub-graph of reaches within each network
    if showtiming:
        start_time = time.time()
    if verbose:
        print("organizing connections into reaches ...")
    networks = nru.compose_networks(
        supernetwork_values,
        break_network_at_waterbodies=break_network_at_waterbodies,
        verbose=False,
        debuglevel=debuglevel,
        showtiming=showtiming,
    )

    if verbose:
        print("reach organization complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))
        start_time = time.time()

    # STEP 3: Organize Network for Waterbodies
    if break_network_at_waterbodies:
        if showtiming:
            start_time = time.time()
        if verbose:
            print("reading waterbody parameter file ...")

        ## STEP 3a: Read waterbody parameter file
        waterbodies_values = supernetwork_values[12]
        waterbodies_segments = supernetwork_values[13]
        connections_tailwaters = supernetwork_values[4]

        waterbodies_df = nio.read_waterbody_df(
            waterbody_parameters,
            waterbodies_values,
        )
        waterbodies_df = waterbodies_df.sort_index(axis="index").sort_index(
            axis="columns"
        )

        nru.order_networks(connections, networks, connections_tailwaters)

        if verbose:
            print("waterbodies complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
            start_time = time.time()

        ## STEP 3b: Order subnetworks above and below reservoirs
        if showtiming:
            start_time = time.time()
        if verbose:
            print("ordering waterbody subnetworks ...")

        max_network_seqorder = -1
        for network in networks:
            max_network_seqorder = max(
                networks[network]["network_seqorder"], max_network_seqorder
            )
        ordered_networks = {}

        for terminal_segment, network in networks.items():
            if network["network_seqorder"] not in ordered_networks:
                ordered_networks[network["network_seqorder"]] = []
            ordered_networks[network["network_seqorder"]].append(
                (terminal_segment, network)
            )

        if verbose:
            print("ordering waterbody subnetworks complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
            start_time = time.time()

    else:
        # If we are not splitting the networks, we can put them all in one order
        max_network_seqorder = 0
        ordered_networks = {}
        ordered_networks[0] = [
            (terminal_segment, network)
            for terminal_segment, network in networks.items()
        ]

    if do_network_analysis_only:
        sys.exit()

    if break_network_at_waterbodies:
        ## STEP 3c: Handle Waterbody Initial States
        if showtiming:
            start_time = time.time()
        if verbose:
            print("setting waterbody initial states ...")

        if wrf_hydro_waterbody_restart_file:

            waterbody_initial_states_df = nio.get_reservoir_restart_from_wrf_hydro(
                wrf_hydro_waterbody_restart_file,
                wrf_hydro_waterbody_ID_crosswalk_file,
                wrf_hydro_waterbody_ID_crosswalk_file_field_name,
                wrf_hydro_waterbody_crosswalk_filter_file,
                wrf_hydro_waterbody_crosswalk_filter_file_field_name,
            )
        else:
            # TODO: Consider adding option to read cold state from route-link file
            waterbody_initial_ds_flow_const = 0.0
            waterbody_initial_depth_const = 0.0
            # Set initial states from cold-state
            waterbody_initial_states_df = pd.DataFrame(
                0,
                index=waterbodies_df.index,
                columns=[
                    "qd0",
                    "h0",
                ],
                dtype="float32",
            )
            # TODO: This assignment could probably by done in the above call
            waterbody_initial_states_df["qd0"] = waterbody_initial_ds_flow_const
            waterbody_initial_states_df["h0"] = waterbody_initial_depth_const
            waterbody_initial_states_df["index"] = range(
                len(waterbody_initial_states_df)
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

    if wrf_hydro_channel_restart_file:

        channel_initial_states_df = nio.get_channel_restart_from_wrf_hydro(
            wrf_hydro_channel_restart_file,
            wrf_hydro_channel_ID_crosswalk_file,
            wrf_hydro_channel_ID_crosswalk_file_field_name,
            wrf_hydro_channel_restart_upstream_flow_field_name,
            wrf_hydro_channel_restart_downstream_flow_field_name,
            wrf_hydro_channel_restart_depth_flow_field_name,
        )
    else:
        # TODO: Consider adding option to read cold state from route-link file
        channel_initial_us_flow_const = 0.0
        channel_initial_ds_flow_const = 0.0
        channel_initial_depth_const = 0.0
        # Set initial states from cold-state
        channel_initial_states_df = pd.DataFrame(
            0,
            index=connections.keys(),
            columns=[
                "qu0",
                "qd0",
                "h0",
            ],
            dtype="float32",
        )
        channel_initial_states_df["qu0"] = channel_initial_us_flow_const
        channel_initial_states_df["qd0"] = channel_initial_ds_flow_const
        channel_initial_states_df["h0"] = channel_initial_depth_const
        channel_initial_states_df["index"] = range(len(channel_initial_states_df))

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

    # initialize qlateral dict
    qlateral = {}

    if qlat_input_folder:
        qlat_files = glob.glob(qlat_input_folder + qlat_file_pattern_filter)
        qlat_df = nio.get_ql_from_wrf_hydro(
            qlat_files=qlat_files,
            index_col=qlat_file_index_col,
            value_col=qlat_file_value_col,
        )

    elif qlat_input_file:
        qlat_df = nio.get_ql_from_csv(qlat_input_file)

    else:
        qlat_df = pd.DataFrame(
            qlat_const, index=connections.keys(), columns=range(nts), dtype="float32"
        )

    for index, row in qlat_df.iterrows():
        qlateral[index] = row.tolist()

    if verbose:
        print("qlateral array complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))
        start_time = time.time()

    # STEP 6: Sort the ordered networks
    if sort_networks:
        if showtiming:
            start_time = time.time()
        if verbose:
            print("sorting the ordered networks ...")

        for nsq in range(max_network_seqorder, -1, -1):
            sort_ordered_network(ordered_networks[nsq], True)

        if verbose:
            print("sorting complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
            start_time = time.time()

    # Define them pool after we create the static global objects (and collect the garbage)
    if parallel_compute:
        import gc

        gc.collect()
        pool = multiprocessing.Pool(cpu_pool)

    flowveldepth_connect = (
        {}
    )  # dict to contain values to transfer from upstream to downstream networks

    ################### Main Execution Loop across ordered networks
    if showtiming:
        main_start_time = time.time()
    if verbose:
        print(f"executing routing computation ...")

    progress_count = 0
    if percentage_complete:
        for nsq in range(max_network_seqorder, -1, -1):
            for terminal_segment, network in ordered_networks[nsq]:
                progress_count += len(network["all_segments"])
        pbar = tqdm(total=(progress_count))

    for nsq in range(max_network_seqorder, -1, -1):

        if parallel_compute:
            nslist = []
        results = []

        current_index_total = 0

        for terminal_segment, network in ordered_networks[nsq]:

            if percentage_complete:
                if current_index_total == 0:
                    pbar.update(0)

            if break_network_at_waterbodies:
                waterbody = waterbodies_segments.get(terminal_segment)
            else:
                waterbody = None
            if not parallel_compute:  # serial execution
                if showtiming:
                    start_time = time.time()
                if verbose:
                    print(
                        f"routing ordered reaches for terminal segment {terminal_segment} ..."
                    )

                results.append(
                    compute_network(
                        flowveldepth_connect=flowveldepth_connect,
                        terminal_segment=terminal_segment,
                        supernetwork_parameters=supernetwork_parameters,
                        waterbody_parameters=waterbody_parameters,
                        waterbody=waterbody,
                        nts=nts,
                        dt=dt,
                        qts_subdivisions=qts_subdivisions,
                        verbose=verbose,
                        debuglevel=debuglevel,
                        csv_output=csv_output,
                        nc_output_folder=nc_output_folder,
                        assume_short_ts=assume_short_ts,
                    )
                )

                if showtiming:
                    print("... complete in %s seconds." % (time.time() - start_time))
                if percentage_complete:
                    pbar.update(len(network["all_segments"]))

            else:  # parallel execution
                nslist.append(
                    [
                        flowveldepth_connect,
                        terminal_segment,
                        supernetwork_parameters,  # TODO: This should probably be global...
                        waterbody_parameters,
                        waterbody,
                        nts,
                        dt,
                        qts_subdivisions,
                        verbose,
                        debuglevel,
                        csv_output,
                        nc_output_folder,
                        assume_short_ts,
                    ]
                )

        if parallel_compute:
            if verbose:
                print(f"routing ordered reaches for networks of order {nsq} ... ")
            if debuglevel <= -2:
                print(f"reaches to be routed include:")
                print(f"{[network[0] for network in ordered_networks[nsq]]}")
            # with pool:
            # with multiprocessing.Pool() as pool:
            results = pool.starmap(compute_network, nslist)

            if showtiming:
                print("... complete in %s seconds." % (time.time() - start_time))
            if percentage_complete:
                # import pdb; pdb.set_trace()
                pbar.update(
                    sum(
                        len(network[1]["all_segments"])
                        for network in ordered_networks[nsq]
                    )
                )
                # print(f"{[network[0] for network in ordered_networks[nsq]]}")
        if (
            nsq > 0
        ):  # We skip this step for zero-order networks, i.e., those that have no downstream dependents
            flowveldepth_connect = (
                {}
            )  # There is no need to preserve previously passed on values -- so we clear the dictionary
            for i, (terminal_segment, network) in enumerate(ordered_networks[nsq]):
                # seg = network["reaches"][network["terminal_reach"]]["reach_tail"]
                seg = terminal_segment
                flowveldepth_connect[seg] = {}
                flowveldepth_connect[seg] = results[i][seg]
                # TODO: The value passed here could be much more specific to
                # TODO: exactly and only the most recent time step for the passing reach

    if parallel_compute:
        pool.close()

    if percentage_complete:
        pbar.close()

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - main_start_time))
    if verbose:
        print("program complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - program_start_time))


if __name__ == "__main__":
    main()
