"""A file to contain the main function for running nhf routing"""
import argparse
import time

import numpy as np

from .flow_scaling_utils import append_nonrouting_to_run_results
from .input import _input_handler_v04
from .nwm_route import nwm_route
from .output import nwm_output_generator

from troute.NHF import NHF
from troute.DataAssimilation import DataAssimilation

import troute.nhd_network_utilities_v02 as nnu
import troute.hyfeature_network_utilities as hnu

import logging
LOG = logging.getLogger("TROUTE")

def nhf_routing(argv):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--custom-input-file",
        dest="custom_input_file",
        help="Path of a .yaml or .json file containing model configuration parameters. See doc/v3_doc.yaml",
    )
    args = parser.parse_args(argv)

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
    
    network = NHF(supernetwork_parameters,
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

        run_results, subnetwork_list = nwm_route(
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
            network._dataframe,
            network.q0,
            network._qlateral,
            network._eloss,
            forcing_parameters.get("ssout"),
            data_assimilation.usgs_df,
            data_assimilation.lastobs_df,
            data_assimilation.reservoir_usgs_df,
            data_assimilation.reservoir_usgs_param_df,
            data_assimilation.reservoir_usace_df,
            data_assimilation.reservoir_usace_param_df,
            data_assimilation.reservoir_usbr_df,
            data_assimilation.reservoir_usbr_param_df,
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
            logFileName,
            # flowveldepth_interorder=network.flowveldepth_interorder,
            qlat_add_loc = "bottom"  # All NHF lats go in bottom
        )
        
        route_end_time = time.time()
        task_times['route_time'] += route_end_time - route_start_time

        # create initial conditions for next loop itteration
        network.new_q0(run_results)
        network.update_waterbody_water_elevation()    
        
        # update reservoir parameters and lastobs_df
        data_assimilation.update_after_compute(run_results, dt*nts)

        # TODO move the conditional call to write_lite_restart to nwm_output_generator.
        # if output_parameters:
        #     if output_parameters['lite_restart'] is not None:
        #         nhd_io.write_lite_restart(
        #             network.q0, 
        #             network._waterbody_df, 
        #             t0 + timedelta(seconds = dt * nts), 
        #             output_parameters['lite_restart']
        #         )                    

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

        if network.poi_nex_dict:
            poi_crosswalk = network.poi_nex_dict
        else:
            poi_crosswalk = dict()

        output_start_time = time.time() 
                
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
            poi_crosswalk,
            logFileName,
            fp_outlet_crosswalk=network.fp_outlet_crosswalk,
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

