import argparse
import time
import math
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures

from troute.NHDNetwork import NHDNetwork
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from troute.DataAssimilation import DataAssimilation

import numpy as np
import pandas as pd

from .input import _input_handler_v03, _input_handler_v04
from .preprocess import (
    nwm_network_preprocess,
    nwm_initial_warmstate_preprocess,
    nwm_forcing_preprocess,
    unpack_nwm_preprocess_data,
)
from .output import nwm_output_generator
from .log_level_set import log_level_set
from troute.routing.compute import compute_nhd_routing_v02, compute_diffusive_routing, compute_log_mc, compute_log_diff

import troute.nhd_io as nhd_io
import troute.nhd_network_utilities_v02 as nnu
import troute.hyfeature_network_utilities as hnu
import sys


LOG = logging.getLogger('')

'''
High level orchestration of ngen t-route simulations for NWM application
'''
def main_v04(argv):

    args = _handle_args_v03(argv)
    
    # unpack user inputs
    (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        hybrid_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v04(args)
    
    run_parameters = {
        'dt': forcing_parameters.get('dt'),
        'nts': forcing_parameters.get('nts'),
        'cpu_pool': compute_parameters.get('cpu_pool'),
    }
    
    showtiming = log_parameters.get("showtiming", None)
    

    task_times = {}
    task_times['forcing_time'] = 0
    task_times['route_time'] = 0
    task_times['output_time'] = 0
    main_start_time = time.time()
    
    cpu_pool = compute_parameters.get("cpu_pool", None)
 
    # Build routing network data objects. Network data objects specify river 
    # network connectivity, channel geometry, and waterbody parameters. Also
    # perform initial warmstate preprocess.
    
    network_start_time = time.time()
    
    #if "ngen_nexus_file" in supernetwork_parameters:
    if supernetwork_parameters["network_type"] == 'HYFeaturesNetwork':
        network = HYFeaturesNetwork(supernetwork_parameters,
                                    waterbody_parameters,
                                    data_assimilation_parameters,
                                    restart_parameters,
                                    compute_parameters,
                                    forcing_parameters,
                                    hybrid_parameters,
                                    preprocessing_parameters,
                                    output_parameters,
                                    verbose=True, showtiming=showtiming)
        duplicate_ids_df = network._duplicate_ids_df
        
    elif supernetwork_parameters["network_type"] == 'NHDNetwork':
        network = NHDNetwork(supernetwork_parameters,
                             waterbody_parameters,
                             restart_parameters,
                             forcing_parameters,
                             compute_parameters,
                             data_assimilation_parameters,
                             hybrid_parameters,
                             output_parameters,
                             verbose=True,
                             showtiming=showtiming,          
                            )
        duplicate_ids_df = pd.DataFrame()
    
    
    network_end_time = time.time()
    task_times['network_creation_time'] = network_end_time - network_start_time
    
    # Create run_sets: sets of forcing files for each loop
    run_sets = network.build_forcing_sets()
    
    # Create da_sets: sets of TimeSlice files for each loop
    if "data_assimilation_parameters" in compute_parameters:
        da_sets = hnu.build_da_sets(data_assimilation_parameters, run_sets, network.t0)
        
    # Create parity_sets: sets of CHRTOUT files against which to compare t-route flows
    if output_parameters.get("wrf_hydro_parity_check"):
        parity_sets = nnu.build_parity_sets(parity_parameters, run_sets)
    else:
        parity_sets = []

    # Create forcing data within network object for first loop iteration
    network.assemble_forcings(run_sets[0],)
    
    # Create data assimilation object from da_sets for first loop iteration
    data_assimilation = DataAssimilation(
        network,
        data_assimilation_parameters,
        run_parameters,
        waterbody_parameters,
        from_files=True,
        value_dict=None,
        da_run=da_sets[0],
        )

    
    forcing_end_time = time.time()
    task_times['forcing_time'] += forcing_end_time - network_end_time

    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)
        
    logFileName = 'NONE'    
    kernelTalks = log_parameters.get("log_directory", None)
    if kernelTalks:
        logFileName = kernelTalks+'/kernelTalks.log'
        with open(logFileName, 'w') as preRunLog:
            preRunLog.write("************************************************************\n") 
            preRunLog.write("Pre- and post run parameter and run statistics output file. \n") 
            preRunLog.write("************************************************************\n")         
            preRunLog.write("\n")
            preRunLog.write("-----\n")
    
            if (restart_parameters['lite_channel_restart_file']==None):
                outPutStr = "No channel restart file: cold start."
                preRunLog.write(outPutStr+"\n") 
                LOG.info(outPutStr)
            else:
                outPutStr = "Warmstart - restart file: "+restart_parameters['lite_channel_restart_file']
                preRunLog.write(outPutStr+" \n") 
                LOG.info(outPutStr)
    
            if (restart_parameters['lite_waterbody_restart_file']==None):
                outPutStr = "No waterbody restart file."
                preRunLog.write(outPutStr+"\n") 
                LOG.info(outPutStr)
            else:
                outPutStr = "Waterbody restart file: "+restart_parameters['lite_waterbody_restart_file']
                preRunLog.write(outPutStr+" \n")
                LOG.info(outPutStr)

            preRunLog.write("-----\n")
            preRunLog.write("\n")
            preRunLog.close()

    # Pass empty subnetwork list to nwm_route. These objects will be calculated/populated
    # on first iteration of for loop only. For additional loops this will be passed
    # to function from inital loop.     
    subnetwork_list = [None, None, None]

    # Flag for first run for param output
    firstRun = True
    # Disable in case there is no log file
    if (not kernelTalks):
        firstRun = False

    for run_set_iterator, run in enumerate(run_sets):
        
        t0 = run.get("t0")
        dt = run.get("dt")
        nts = run.get("nts")

        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        
        route_start_time = time.time()

        run_results = nwm_route(
            network.connections, 
            network.reverse_network, 
            network.waterbody_connections, 
            network.reaches_by_tailwater,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            cpu_pool,
            network.t0,
            dt,
            nts,
            qts_subdivisions,
            network.independent_networks, 
            network.dataframe,
            network.q0,
            network._qlateral,
            data_assimilation.usgs_df,
            data_assimilation.lastobs_df,
            data_assimilation.reservoir_usgs_df,
            data_assimilation.reservoir_usgs_param_df,
            data_assimilation.reservoir_usace_df,
            data_assimilation.reservoir_usace_param_df,
            data_assimilation.reservoir_rfc_df,
            data_assimilation.reservoir_rfc_param_df,
            data_assimilation.great_lakes_df,
            data_assimilation.great_lakes_param_df,
            network.great_lakes_climatology_df,
            data_assimilation.assimilation_parameters,
            assume_short_ts,
            return_courant,
            network.waterbody_dataframe,
            data_assimilation_parameters,
            network.waterbody_types_dataframe,
            network.waterbody_type_specified,
            network.diffusive_network_data,
            network.topobathy_df,
            network.refactored_diffusive_domain,
            network.refactored_reaches,
            subnetwork_list,
            network.coastal_boundary_depth_df,
            network.unrefactored_topobathy_df,
            firstRun,
            logFileName            
        )
      
        # returns list, first item is run result, second item is subnetwork items
        subnetwork_list = run_results[1]
        run_results = run_results[0]

        route_end_time = time.time()
        task_times['route_time'] += route_end_time - route_start_time

        # create initial conditions for next loop itteration
        network.new_q0(run_results)
        network.update_waterbody_water_elevation()    
        
        # update reservoir parameters and lastobs_df
        data_assimilation.update_after_compute(run_results, dt*nts)

        # TODO move the conditional call to write_lite_restart to nwm_output_generator.
        if output_parameters:
            if output_parameters['lite_restart'] is not None:
                nhd_io.write_lite_restart(
                    network.q0, 
                    network._waterbody_df, 
                    t0 + timedelta(seconds = dt * nts), 
                    output_parameters['lite_restart']
                )                    

        # Prepare input forcing for next time loop simulation when mutiple time loops are presented.
        if run_set_iterator < len(run_sets) - 1:
            # update t0
            network.new_t0(dt,nts)
            
            # update forcing data
            network.assemble_forcings(run_sets[run_set_iterator + 1],)
            
            # get reservoir DA initial parameters for next loop iteration
            data_assimilation.update_for_next_loop(
                network,
                da_sets[run_set_iterator + 1])
            
            
            forcing_end_time = time.time()
            task_times['forcing_time'] += forcing_end_time - route_end_time

        output_start_time = time.time()  
        
        #TODO Update this to work with either network type...
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
            cpu_pool,
            network.waterbody_dataframe,
            network.waterbody_types_dataframe,
            duplicate_ids_df,
            data_assimilation_parameters,
            data_assimilation.lastobs_df,
            network.link_gage_df,
            network.link_lake_crosswalk,
            network.nexus_dict,
            network.poi_nex_df, 
            logFileName            
        )
        

        output_end_time = time.time()
        task_times['output_time'] += output_end_time - output_start_time
    
        firstRun = False
    
    # end of for run_set_iterator, run in enumerate(run_sets):
    
    
    task_times['total_time'] = time.time() - main_start_time

    LOG.debug("process complete in %s seconds." % (time.time() - main_start_time))

    LOG.info('************ TIMING SUMMARY ************')
    LOG.info('----------------------------------------')
    LOG.info(
        'Network graph construction: {} secs, {} %'\
        .format(
            round(task_times['network_creation_time'], 2),
            round(task_times['network_creation_time'] / task_times['total_time'] * 100, 2)
        )
    )
    LOG.info(
        'Forcing array construction: {} secs, {} %'\
        .format(
            round(task_times['forcing_time'], 2),
            round(task_times['forcing_time'] / task_times['total_time'] * 100, 2)
        )
    ) 
    LOG.info(
        'Routing computations: {} secs, {} %'\
        .format(
            round(task_times['route_time'], 2),
            round(task_times['route_time'] / task_times['total_time'] * 100, 2)
        )
    ) 
    LOG.info(
        'Output writing: {} secs, {} %'\
        .format(
            round(task_times['output_time'], 2),
            round(task_times['output_time'] / task_times['total_time'] * 100, 2)
        )
    )
    LOG.info('----------------------------------------')
    LOG.info(
        'Total execution time: {} secs'\
        .format(
            round(task_times['network_creation_time'], 2) +
            round(task_times['forcing_time'], 2) +
            round(task_times['route_time'], 2) +
            round(task_times['output_time'], 2)
        )
    )
    
    if showtiming and log_parameters.get('log_level') not in ['DEBUG', 'INFO']:
        print('************ TIMING SUMMARY ************')
        print('----------------------------------------')
        print(
            'Network graph construction: {} secs, {} %'\
            .format(
                round(task_times['network_creation_time'],2),
                round(task_times['network_creation_time']/task_times['total_time'] * 100,2)
            )
        )
        print(
            'Forcing array construction: {} secs, {} %'\
            .format(
                round(task_times['forcing_time'],2),
                round(task_times['forcing_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Routing computations: {} secs, {} %'\
            .format(
                round(task_times['route_time'],2),
                round(task_times['route_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Output writing: {} secs, {} %'\
            .format(
                round(task_times['output_time'],2),
                round(task_times['output_time']/task_times['total_time'] * 100,2)
            )
        )
        print('----------------------------------------')
        print(
            'Total execution time: {} secs'\
            .format(
                round(task_times['network_creation_time'],2) +
                round(task_times['forcing_time'],2) +
                round(task_times['route_time'],2) +
                round(task_times['output_time'],2)
            )
        ) 


'''
NGEN functions (_v02)
'''
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

    _results, _link_gage_df = _run_everything_v02(
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
        _link_gage_df,
        run_parameters,
        supernetwork_parameters,
        restart_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
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

    verbose = run_parameters.get("verbose", None)
    showtiming = run_parameters.get("showtiming", None)
    debuglevel = run_parameters.get("debuglevel", 0)
    break_network_at_waterbodies = run_parameters.get(
        "break_network_at_waterbodies", False
    )
    #NJF hacking this into supernetwork parameters to
    #avoid having to pass runtime params to network init, which deosn't seem quite right...
    supernetwork_parameters["break_network_at_waterbodies"] = break_network_at_waterbodies
    forcing_parameters["qts_subdivisions"] = run_parameters["qts_subdivisions"]
    forcing_parameters["nts"] = run_parameters["nts"]
    
    # STEP 1: Build network
    if "ngen_nexus_file" in supernetwork_parameters:
        network = HYFeaturesNetwork(supernetwork_parameters,
                                    waterbody_parameters=waterbody_parameters,
                                    restart_parameters=restart_parameters,
                                    forcing_parameters=forcing_parameters,
                                    verbose=verbose, showtiming=showtiming)
    else:
        network = NHDNetwork(supernetwork_parameters,
                             waterbody_parameters=waterbody_parameters,
                             restart_parameters=restart_parameters,
                             forcing_parameters=forcing_parameters,
                             verbose=verbose, showtiming=showtiming)

    synthetic_wb_segments = supernetwork_parameters.get("synthetic_wb_segments", None)
    synthetic_wb_id_offset = supernetwork_parameters.get("synthetic_wb_id_offset", 9.99e11)
    if synthetic_wb_segments:
        network.set_synthetic_wb_segments( synthetic_wb_segments, synthetic_wb_id_offset )
    #FIXME
    network.astype("float32", ["dx", "n", "ncc", "s0", "bw", "tw", "twcc", "musk", "musx", "cs", "alt"])

    #TODO: This could probably done in networkwork construction,
    #But requires a hefty refactoring of network construction to get everything
    #built in the correct order...for now just leave it as is...
    if break_network_at_waterbodies:
        network.replace_waterbodies()
    
    run_parameters["t0"] = network.t0

    # to the total number of columns (hours) multiplied 
    # by qts_subdivisions, number of timesteps per forcing 
    # (qlateral) timestep.
    # NJF adjusted based on ngen usage.  This probably isn't
    # appropriate in all situations.
    # TODO allow a network to "override" nts
    qlats = network.qlateral
    if "ngen_nexus_file" in supernetwork_parameters:
        if len(network.qlateral.columns) != run_parameters["nts"] * run_parameters.get("qts_subdivisions", 1):
            print("WARNING: Lateral flow time series is larger than provided nts. Adjusting nts.\n"+\
                "If this was unintended, double check the configuration number of time steps and the "+
                "lateral flow input time series")
            run_parameters["nts"] = (len(network.qlateral.columns)) \
        * run_parameters.get("qts_subdivisions", 1)


    # STEP 6
    #TODO factory create the DA object from params to pick the right one
    #i.e. EmptyDA, NudgingDA, NewAndImprovedDA...
    data_assimilation = NudgingDA(data_assimilation_parameters, run_parameters)

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
        network.connections,
        network.reverse_network,
        network.waterbody_connections,
        network.reaches_by_tailwater,
        compute_func,
        run_parameters.get("parallel_compute_method", None),
        run_parameters.get("subnetwork_target_size", 1),
        # The default here might be the whole network or some percentage...
        run_parameters.get("cpu_pool", None),
        run_parameters.get("dt"),
        run_parameters.get("nts", 1),
        run_parameters.get("qts_subdivisions", 1),
        network.independent_networks,
        network.dataframe,
        network.q0,
        network.qlateral,
        data_assimilation.usgs_df,
        data_assimilation.last_obs,
        data_assimilation.asssimilation_parameters,
        run_parameters.get("assume_short_ts", False),
        run_parameters.get("return_courant", False),
        network.waterbody_dataframe,
        waterbody_parameters,  # TODO: Can we remove the dependence on this input? It's like passing argv down into the compute kernel -- seems like we can strip out the specifically needed items.
        network.waterbody_types_dataframe,
        not network.waterbody_types_dataframe.index.empty,
        diffusive_parameters,
    )

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    return results, pd.DataFrame.from_dict(network.gages)


def _handle_output_v02(
    results,
    link_gage_df,
    run_parameters,
    supernetwork_parameters,
    restart_parameters,
    output_parameters,
    parity_parameters,
    data_assimilation_parameters,
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
    csv_output_folder = None
    if csv_output:
        csv_output_folder = output_parameters["csv_output"].get(
            "csv_output_folder", None
        )
        csv_output_segments = csv_output.get("csv_output_segments", None)
        
    if (debuglevel <= -1) or csv_output_folder:

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
            
            if verbose:
                print("- writing flow, velocity, and depth results to .csv")
                
            # create filenames
            # TO DO: create more descriptive filenames
            extension = ".csv"
            extension = ".h5"
            if supernetwork_parameters.get("title_string", None):
                filename_fvd = (
                    "flowveldepth_" + supernetwork_parameters["title_string"] + extension
                )
                filename_courant = (
                    "courant_" + supernetwork_parameters["title_string"] + extension
                )
            else:
                run_time_stamp = datetime.now().isoformat()
                filename_fvd = "flowveldepth_" + run_time_stamp + extension
                filename_courant = "courant_" + run_time_stamp + extension

            output_path = Path(csv_output_folder).resolve()

            flowveldepth = flowveldepth.sort_index()
            
            # no csv_output_segments are specified, then write results for all segments
            if not csv_output_segments:
                csv_output_segments = flowveldepth.index
            
            #flowveldepth.loc[csv_output_segments].to_csv(output_path.joinpath(filename_fvd))
            flowveldepth.loc[csv_output_segments].to_hdf(output_path.joinpath(filename_fvd), key="qvd")
            if run_parameters.get("return_courant", False):
                courant = courant.sort_index()
                courant.loc[csv_output_segments].to_csv(output_path.joinpath(filename_courant))

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
            
            if verbose:
                print("- writing restart files")
                
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
        
        if verbose:
            print("- writing results to CHRTOUT")
        
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

    data_assimilation_folder = data_assimilation_parameters.get(
    "data_assimilation_timeslices_folder", None
    )
    lastobs_output_folder = data_assimilation_parameters.get(
    "lastobs_output_folder", None
    )
    if data_assimilation_folder and lastobs_output_folder:
        # create a new lastobs DataFrame from the last itteration of run results
        # lastobs_df = new_lastobs(run_results, dt * nts)
        # lastobs_df_copy = lastobs_df.copy()
        lastobs_df = new_lastobs(results, dt * nts)
        nhd_io.lastobs_df_output(
            lastobs_df,
            dt,
            nts,
            t0,
            link_gage_df['gages'],
            lastobs_output_folder,
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

'''
Version 3 and earlier
'''
def _handle_args_v03(argv):
    '''
    Handle command line input argument - filepath of configuration file
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--custom-input-file",
        dest="custom_input_file",
        help="Path of a .yaml or .json file containing model configuration parameters. See doc/v3_doc.yaml",
    )
    return parser.parse_args(argv)

def nwm_route(
    downstream_connections,
    upstream_connections,
    waterbodies_in_connections,
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
    reservoir_rfc_df,
    reservoir_rfc_param_df,
    great_lakes_df,
    great_lakes_param_df,
    great_lakes_climatology_df,
    da_parameter_dict,
    assume_short_ts,
    return_courant,
    waterbodies_df,
    data_assimilation_parameters,
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
    logFileName='troute_run_log.txt',  
    flowveldepth_interorder={},
    from_files=False,
):

    ################### Main Execution Loop across ordered networks      
    start_time = time.time()

    if return_courant:
        LOG.info(
            f"executing routing computation, with Courant evaluation metrics returned"
        )
    else:
        LOG.info(f"executing routing computation ...")

    if (firstRun):
        compute_log_mc(
            logFileName,
            downstream_connections,
            upstream_connections,
            waterbodies_in_connections,
            reaches_bytw,
            compute_kernel,
            parallel_compute_method,
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
            reservoir_rfc_df,
            reservoir_rfc_param_df,
            assume_short_ts,
            waterbodies_df,
            data_assimilation_parameters,
            waterbody_types_df,
            waterbody_type_specified,
        )

    start_time_mc = time.time()
    results = compute_nhd_routing_v02(
        downstream_connections,
        upstream_connections,
        waterbodies_in_connections,
        reaches_bytw,
        compute_kernel,
        parallel_compute_method,
        subnetwork_target_size,  # The default here might be the whole network or some percentage...
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
        reservoir_rfc_df,
        reservoir_rfc_param_df,
        great_lakes_df,
        great_lakes_param_df,
        great_lakes_climatology_df,
        da_parameter_dict,
        assume_short_ts,
        return_courant,
        waterbodies_df,
        data_assimilation_parameters,
        waterbody_types_df,
        waterbody_type_specified,
        subnetwork_list,
        flowveldepth_interorder,
        from_files = from_files,
    )
    LOG.debug("MC computation complete in %s seconds." % (time.time() - start_time_mc))
    # returns list, first item is run result, second item is subnetwork items
    subnetwork_list = results[1]
    results = results[0]
    
    # run diffusive side of a hybrid simulation
    if diffusive_network_data:
        start_time_diff = time.time()
        '''
        # retrieve MC-computed streamflow value at upstream boundary of diffusive mainstem
        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()
        flowveldepth = pd.concat(
            [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
            copy=False,
        )
        '''
        #upstream_boundary_flow={}
        #for tw,v in  diffusive_network_data.items():
        #    upstream_boundary_link     = diffusive_network_data[tw]['upstream_boundary_link']
        #    flow_              = flowveldepth.loc[upstream_boundary_link][0::3]
            # the very first value at time (0,q) is flow value at the first time step after initial time.
        #    upstream_boundary_flow[tw] = flow_         
          
        if (firstRun):
            compute_log_diff(
                logFileName,
                diffusive_network_data,
                topobathy_df,
                refactored_diffusive_domain,
                refactored_reaches,                
                coastal_boundary_depth_df,
                unrefactored_topobathy_df,                
            )          

        # call diffusive wave simulation and append results to MC results
        results.extend(
            compute_diffusive_routing(
                results,
                diffusive_network_data,
                cpu_pool,
                t0,
                dt,
                nts,
                q0,
                qlats,
                qts_subdivisions,
                usgs_df,
                lastobs_df,
                da_parameter_dict,
                waterbodies_df,
                topobathy_df,
                refactored_diffusive_domain,
                refactored_reaches,
                coastal_boundary_depth_df,
                unrefactored_topobathy_df,
            )
        )
        LOG.debug("Diffusive computation complete in %s seconds." % (time.time() - start_time_diff))

    else:

        if (firstRun):
            with open(logFileName, 'a') as preRunLog:
                preRunLog.write("**********************\n") 
                preRunLog.write("No diffusive routing. \n") 
                preRunLog.write("**********************\n")     
            preRunLog.close()            

    LOG.debug("ordered reach computation complete in %s seconds." % (time.time() - start_time))

    return results, subnetwork_list


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

def set_reservoir_da_prams(run_results):
    '''
    Update persistence reservoir DA parameters for subsequent loops
    '''
    
    reservoir_usgs_param_df = pd.DataFrame(data = [], 
                                           index = [], 
                                           columns = [
                                               'update_time', 'prev_persisted_outflow', 
                                               'persistence_update_time', 'persistence_index'
                                           ]
                                          )
    reservoir_usace_param_df = pd.DataFrame(data = [], 
                                           index = [], 
                                           columns = [
                                               'update_time', 'prev_persisted_outflow', 
                                               'persistence_update_time', 'persistence_index'
                                           ]
                                          )
    
    for r in run_results:
        
        if len(r[4][0]) > 0:
            tmp_usgs = pd.DataFrame(data = r[4][1], index = r[4][0], columns = ['update_time'])
            tmp_usgs['prev_persisted_outflow'] = r[4][2]
            tmp_usgs['persistence_update_time'] = r[4][4]
            tmp_usgs['persistence_index'] = r[4][3]
            reservoir_usgs_param_df = pd.concat([reservoir_usgs_param_df, tmp_usgs])
        
        if len(r[5][0]) > 0:
            tmp_usace = pd.DataFrame(data = r[5][1], index = r[5][0], columns = ['update_time'])
            tmp_usace['prev_persisted_outflow'] = r[5][2]
            tmp_usace['persistence_update_time'] = r[5][4]
            tmp_usace['persistence_index'] = r[5][3]
            reservoir_usace_param_df = pd.concat([reservoir_usace_param_df, tmp_usace])
    
    return reservoir_usgs_param_df, reservoir_usace_param_df


def update_lookback_hours(dt, nts, waterbody_parameters):
    """
    Update the lookback hours that an RFC type reservoir searches in reverse
    from the model start time to find a time series file. The update is based
    on the total hours ran in the prior loop.
    """

    waterbody_parameters['rfc']['reservoir_rfc_forecasts_lookback_hours'] = \
    waterbody_parameters['rfc']['reservoir_rfc_forecasts_lookback_hours'] + \
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
    High level orchestration of t-route simulations for NWM application
    """
    args = _handle_args_v03(argv)
    
    # unpack user inputs
    (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        hybrid_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v03(args)
    
    showtiming = log_parameters.get("showtiming", None)
    if showtiming:
        task_times = {}
        task_times['initial_condition_time'] = 0
        task_times['forcing_time'] = 0
        task_times['route_time'] = 0
        task_times['output_time'] = 0
        main_start_time = time.time()

    if showtiming:
        network_start_time = time.time()

    # Build routing network data objects. Network data objects specify river 
    # network connectivity, channel geometry, and waterbody parameters.
    if preprocessing_parameters.get('use_preprocessed_data', False): 
        
        # get data from pre-processed file
        (
            connections,
            param_df,
            wbody_conn,
            waterbodies_df,
            waterbody_types_df,
            break_network_at_waterbodies,
            waterbody_type_specified,
            link_lake_crosswalk,
            independent_networks,
            reaches_bytw,
            rconn,
            link_gage_df,
            usgs_lake_gage_crosswalk, 
            usace_lake_gage_crosswalk,
            diffusive_network_data,
            topobathy_df,
            refactored_diffusive_domain,
            refactored_reaches,
            unrefactored_topobathy_df,
        ) = unpack_nwm_preprocess_data(
            preprocessing_parameters
        )
    else:
        
        # build data objects from scratch
        (
            connections,
            param_df,
            wbody_conn,
            waterbodies_df,
            waterbody_types_df,
            break_network_at_waterbodies,
            waterbody_type_specified,
            link_lake_crosswalk,
            independent_networks,
            reaches_bytw,
            rconn,
            link_gage_df,
            usgs_lake_gage_crosswalk, 
            usace_lake_gage_crosswalk,
            diffusive_network_data,
            topobathy_df,
            refactored_diffusive_domain,
            refactored_reaches,
            unrefactored_topobathy_df,
        ) = nwm_network_preprocess(
            supernetwork_parameters,
            waterbody_parameters,
            preprocessing_parameters,
            compute_parameters,
            data_assimilation_parameters,
        )
    
    if showtiming:
        network_end_time = time.time()
        task_times['network_time'] = network_end_time - network_start_time

    # list of all segments in the domain (MC + diffusive)
    segment_index = param_df.index
    if diffusive_network_data:
        for tw in diffusive_network_data:
            segment_index = segment_index.append(
                pd.Index(diffusive_network_data[tw]['mainstem_segs'])
            ) 

    # TODO: This function modifies one of its arguments (waterbodies_df), which is somewhat poor practice given its otherwise functional nature. Consider refactoring
    waterbodies_df, q0, t0, lastobs_df, da_parameter_dict = nwm_initial_warmstate_preprocess(
        break_network_at_waterbodies,
        restart_parameters,
        data_assimilation_parameters,
        segment_index,
        waterbodies_df,
        link_lake_crosswalk,
    )

    if showtiming:
        ic_end_time = time.time()
        task_times['initial_condition_time'] += ic_end_time - network_end_time

    # Create run_sets: sets of forcing files for each loop
    run_sets = nnu.build_forcing_sets(forcing_parameters, t0)

    # Create da_sets: sets of TimeSlice files for each loop
    if "data_assimilation_parameters" in compute_parameters:
        da_sets = nnu.build_da_sets(data_assimilation_parameters, run_sets, t0)
        
    # Create parity_sets: sets of CHRTOUT files against which to compare t-route flows
    if "wrf_hydro_parity_check" in output_parameters:
        parity_sets = nnu.build_parity_sets(parity_parameters, run_sets)
    else:
        parity_sets = []
    
    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    cpu_pool = compute_parameters.get("cpu_pool", None)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    return_courant = compute_parameters.get("return_courant", False)

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
    
        
    if showtiming:
        forcing_end_time = time.time()
        task_times['forcing_time'] += forcing_end_time - ic_end_time

    logFileName = 'kernelTalks.log'

    kernelTalks = log_parameters.get("log_directory", None)

    if (kernelTalks):

        logFileName = kernelTalks+'/'+logFileName

        with open(logFileName, 'w') as preRunLog:
            preRunLog.write("************************************************************\n") 
            preRunLog.write("Pre- and post run parameter and run statistics output file. \n") 
            preRunLog.write("************************************************************\n")     
        preRunLog.close()

    # Pass empty subnetwork list to nwm_route. These objects will be calculated/populated
    # on first iteration of for loop only. For additional loops this will be passed
    # to function from inital loop. 
    subnetwork_list = [None, None, None]

    # Flag for first run for param output
    firstRun = True
    # Disable in case there is no log file
    if (not kernelTalks):
        firstRun = False

    for run_set_iterator, run in enumerate(run_sets):

        t0 = run.get("t0")
        dt = run.get("dt")
        nts = run.get("nts")

        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        if showtiming:
            route_start_time = time.time()
        
        run_results = nwm_route(
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
            firstRun,
            logFileName              
        )
        
        # returns list, first item is run result, second item is subnetwork items
        subnetwork_list = run_results[1]
        run_results = run_results[0]
        
        if showtiming:
            route_end_time = time.time()
            task_times['route_time'] += route_end_time - route_start_time

        # create initial conditions for next loop itteration
        q0 = new_nwm_q0(run_results)
        waterbodies_df = get_waterbody_water_elevation(waterbodies_df, q0)

        
        # get reservoir DA initial parameters for next loop itteration
        reservoir_usgs_param_df, reservoir_usace_param_df = set_reservoir_da_prams(run_results)
        
        # TODO move the conditional call to write_lite_restart to nwm_output_generator.
        if "lite_restart" in output_parameters:
            nhd_io.write_lite_restart(
                q0, 
                waterbodies_df, 
                t0 + timedelta(seconds = dt * nts), 
                output_parameters['lite_restart']
            )
        
        if run_set_iterator < len(run_sets) - 1:
            (
                qlats, 
                usgs_df, 
                reservoir_usgs_df, 
                _,
                reservoir_usace_df,
                _,
                coastal_boundary_depth_df,
            ) = nwm_forcing_preprocess(
                run_sets[run_set_iterator + 1],
                forcing_parameters,
                hybrid_parameters,
                da_sets[run_set_iterator + 1] if data_assimilation_parameters else {},
                data_assimilation_parameters,
                break_network_at_waterbodies,
                segment_index,
                link_gage_df,
                usgs_lake_gage_crosswalk, 
                usace_lake_gage_crosswalk,
                link_lake_crosswalk,
                lastobs_df.index,
                cpu_pool,
                t0 + timedelta(seconds = dt * nts),
            )
            
            # if there are no TimeSlice files available for hybrid reservoir DA in the next loop, 
            # but there are DA parameters from the previous loop, then create a
            # dummy observations df. This allows the reservoir persistence to continue across loops.
            # USGS Reservoirs
            if not waterbody_types_df.empty:
                if 2 in waterbody_types_df['reservoir_type'].unique():
                    if reservoir_usgs_df.empty and len(reservoir_usgs_param_df.index) > 0:
                        reservoir_usgs_df = pd.DataFrame(
                            data    = np.nan, 
                            index   = reservoir_usgs_param_df.index, 
                            columns = [t0]
                        )

                # USACE Reservoirs   
                if 3 in waterbody_types_df['reservoir_type'].unique():
                    if reservoir_usace_df.empty and len(reservoir_usace_param_df.index) > 0:
                        reservoir_usace_df = pd.DataFrame(
                            data    = np.nan, 
                            index   = reservoir_usace_param_df.index, 
                            columns = [t0]
                        )

                # update RFC lookback hours if there are RFC-type reservoirs in the simulation domain
                if 4 in waterbody_types_df['reservoir_type'].unique():
                    waterbody_parameters = update_lookback_hours(dt, nts, waterbody_parameters)     

            if showtiming:
                forcing_end_time = time.time()
                task_times['forcing_time'] += forcing_end_time - route_end_time
  
            if showtiming:
                ic_end_time = time.time()
                task_times['initial_condition_time'] += ic_end_time - forcing_end_time

        if showtiming:
            ic_start_time = time.time()
        
        # if streamflow DA is ON, then create a new lastobs dataframe
        if data_assimilation_parameters:
            streamflow_da = data_assimilation_parameters.get('streamflow_da',False)
            if streamflow_da:
                if streamflow_da.get('streamflow_nudging', False):
                    lastobs_df = new_lastobs(run_results, dt * nts)
            
        if showtiming:
            ic_end_time = time.time()
            task_times['initial_condition_time'] += ic_end_time - ic_start_time
        
        if (not kernelTalks):        
            logFileName = 'NONE'

        duplicate_ids_df = pd.DataFrame()

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
            cpu_pool,
            waterbodies_df,
            waterbody_types_df,
            duplicate_ids_df,
            data_assimilation_parameters,
            lastobs_df,
            link_gage_df,
            link_lake_crosswalk,
            logFileName
        )
        
        if showtiming:
            output_end_time = time.time()
            task_times['output_time'] += output_end_time - ic_end_time
            
    if showtiming:
        task_times['total_time'] = time.time() - main_start_time

    LOG.debug("process complete in %s seconds." % (time.time() - main_start_time))
    
    if showtiming:
        print('************ TIMING SUMMARY ************')
        print('----------------------------------------')
        print(
            'Network graph construction: {} secs, {} %'\
            .format(
                round(task_times['network_time'],2),
                round(task_times['network_time']/task_times['total_time'] * 100,2)
            )
        )
        print(
            'Initial condition handling: {} secs, {} %'\
            .format(
                round(task_times['initial_condition_time'],2),
                round(task_times['initial_condition_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Forcing array construction: {} secs, {} %'\
            .format(
                round(task_times['forcing_time'],2),
                round(task_times['forcing_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Routing computations: {} secs, {} %'\
            .format(
                round(task_times['route_time'],2),
                round(task_times['route_time']/task_times['total_time'] * 100,2)
            )
        ) 
        print(
            'Output writing: {} secs, {} %'\
            .format(
                round(task_times['output_time'],2),
                round(task_times['output_time']/task_times['total_time'] * 100,2)
            )
        )
        print('----------------------------------------')
        print(
            'Total execution time: {} secs'\
            .format(
                round(task_times['network_time'],2) +
                round(task_times['initial_condition_time'],2) +
                round(task_times['forcing_time'],2) +
                round(task_times['route_time'],2) +
                round(task_times['output_time'],2)
            )
        )

async def main_v03_async(argv):
    """
    Handles the creation of the input parameter dictionaries
    from an input file and then sequences the execution of the
    t-route routing agorithm on a series of execution loops.
    """
    args = _handle_args_v03(argv)  # async shares input framework with non-async
    (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        hybrid_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    ) = _input_handler_v03(args)

    showtiming = log_parameters.get("showtiming", None)

    
    main_start_time = time.time()

    if preprocessing_parameters.get('use_preprocessed_data', False): 
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
            link_gage_df,
        ) = unpack_nwm_preprocess_data(
            preprocessing_parameters
        )
    else:
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
            link_gage_df,
        ) = nwm_network_preprocess(
            supernetwork_parameters,
            waterbody_parameters,
            preprocessing_parameters,
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
    )

    # Create run_sets: sets of forcing files for each loop
    run_sets = nnu.build_forcing_sets(forcing_parameters, t0)

    # Create da_sets: sets of TimeSlice files for each loop
    if "data_assimilation_parameters" in compute_parameters: 
        da_sets = nnu.build_da_sets(data_assimilation_parameters, run_sets, t0)
        
    # Create parity_sets: sets of CHRTOUT files against which to compare t-route flows
    if "wrf_hydro_parity_check" in output_parameters:
        parity_sets = nnu.build_parity_sets(parity_parameters, run_sets)
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
        IO_cpu_pool,
        t0,
    )

    run_set_iterator = 0
    for run_set_iterator, run in enumerate(run_sets[:-1]):
                    
        dt = forcing_parameters.get("dt", None)
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
            IO_cpu_pool,
            t0 + timedelta(seconds = dt * nts),
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
            IO_cpu_pool,
            data_assimilation_parameters,
            lastobs_df,
            link_gage_df,
        )

    # For the last loop, no next forcing or warm state is needed for execution.
    run_set_iterator += 1
    run = run_sets[run_set_iterator]

    dt = forcing_parameters.get("dt", None)
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
        IO_cpu_pool,
        data_assimilation_parameters,
        lastobs_df,
        link_gage_df,
    )

    
    LOG.debug("process complete in %s seconds." % (time.time() - main_start_time))

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
        default=4,
        nargs="?",
        choices=[2, 3, 4],
        type=int,
        help="Use version 3 or 4 of the input format. Default 4",
    )
    v_args = v_parser.parse_known_args()
    '''
    if v_args[0].input_version == 4:
        LOG.info("Running main v03 - async looping")
        coroutine = main_v03_async(v_args[1])
        asyncio.run(coroutine)
        # loop.run_until_complete(coroutine)
    '''
    if v_args[0].input_version == 3:
        LOG.info("Running main v03 - looping")
        main_v03(v_args[1])
    if v_args[0].input_version == 4:
        LOG.info("Running main v04 - looping")
        main_v04(v_args[1])
