"""A file to contain the main function for running nhf routing"""
import argparse
import time

import numpy as np
import pandas as pd

from .flow_scaling_utils import append_nonrouting_to_run_results
from .input import _input_handler_v04
from .nwm_route import nwm_route
from .output import nwm_output_generator
from .scaling_da_apply import (
    build_scaling_da,
    merge_injected_obs,
    network_gage_segments,
    should_seed_state,
)

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

    # Build the simple-scaling DA once (trees + gage crosswalk) if enabled.
    scaling_da = build_scaling_da(
        network, supernetwork_parameters, data_assimilation_parameters,
        cpu_pool=compute_parameters.get("cpu_pool"),
    )
    # Every execution plan breaks reaches at the network's gages, the treewise plan
    # used by serial/by-network/bmi included, so the in-kernel override lands at a
    # gaged nexus and propagates downstream under any compute method. The split set is
    # network.gages, which is static, so it does not depend on which observations
    # happened to arrive in the window the plan was built for.

    # A traced travel-time shift reads the kernel's Courant number (r[2]) for
    # each reach's transit, 1/cn timesteps. Forced on for ROUTING only:
    # nwm_output_generator keeps reading the config value below, so this does not
    # start writing Courant output files.
    #
    # Kept as TWO flags, not one. The user's request is permanent and feeds the
    # output writer; the trace's is dropped as soon as its cache is full. Folding
    # them together meant a run that explicitly asked for Courant output stopped
    # getting it after the first window, with the writer still expecting it.
    _courant_for_user = return_courant
    _courant_for_trace = scaling_da is not None and scaling_da.travel_time_lag

    # A window whose spread is deferred for its halo is held here until the next
    # window's innovation arrives; see the flush at the top of the loop body.
    _pending = None

    def _flush_output(_run, _res, _iter):
        nwm_output_generator(
            _run, _res, supernetwork_parameters, output_parameters, parity_parameters,
            restart_parameters, parity_sets[_iter] if parity_parameters else {},
            qts_subdivisions, compute_parameters.get("return_courant", False), cpu_pool,
            network.waterbody_dataframe, network.waterbody_types_dataframe,
            duplicate_ids_df, data_assimilation_parameters,
            data_assimilation.lastobs_df, network.link_gage_df,
            network.link_lake_crosswalk, network.nexus_dict, poi_crosswalk, logFileName,
            fp_outlet_crosswalk=network.fp_outlet_crosswalk,
        )

    for run_set_iterator, run in enumerate(run_sets):

        t0 = run.get("t0")
        dt = run.get("dt")
        nts = run.get("nts")

        if parity_sets:
            parity_sets[run_set_iterator]["dt"] = dt
            parity_sets[run_set_iterator]["nts"] = nts

        # In-kernel downstream propagation: inject the accepted gage obs into the MC
        # nudging override BEFORE routing, so the gage correction propagates downstream
        # through Muskingum-Cunge to the inter-gage reach. Re-injected every loop, since
        # DataAssimilation rebuilds the frame each time.
        if scaling_da is not None:
            # da_sets[i] is the SAME per-window TimeSlice file list the nudging
            # path consumes, so both read the identical files with the identical
            # lookback padding.
            data_assimilation._usgs_df = merge_injected_obs(
                scaling_da.build_usgs_df(
                    t0, dt, nts, da_sets[run_set_iterator] if da_sets else None
                ),
                data_assimilation.usgs_df,
            )

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
            # The trace reads the Courant field ONCE, to fill its cache. After
            # that the kernel would allocate and fill an [n_seg, nts*3] block per
            # window that nothing reads, which is the whole of the trace's
            # measured overhead. Re-evaluated per window rather than hoisted:
            # the cache is filled during the first window's DA, so the flag has
            # to be able to change after it. The user's own request never drops.
            (_courant_for_user or (
                _courant_for_trace
                and not scaling_da._trace_cached(scaling_da.trees)
            )),
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
            qlat_add_loc = "bottom",  # All NHF lats go in bottom
            diversion_da=network.diversion_da,
            # Static split points for the cached execution plan: every gage the
            # network carries, not just those with observations this window.
            gage_segments=network_gage_segments(network)
        )
        
        route_end_time = time.time()
        task_times['route_time'] += route_end_time - route_start_time

        # The upstream spread runs exactly ONCE per window under both arms; only its
        # POSITION differs. The prognostic arm places it BEFORE the warmstate snapshot
        # so the corrected discharge lands in q0, but only on the FINAL window -- see
        # should_seed_state() for why re-seeding every window degrades the correction.
        # NOTE: this driver keeps no restart of its own (write_lite_restart below is
        # commented out); forecast restarts go through the BMI driver, which persists q0.
        seed_state = should_seed_state(scaling_da, run_set_iterator, len(run_sets))
        # HALO: this window's innovation is what the PREVIOUS window's backward shift
        # needs to read past its own end. Flush the pending window now that we have it,
        # so its tail uses real observations instead of persistence -- that fallback is
        # what made the lagged result depend on max_loop_size, a memory knob.
        if _pending is not None:
            _p_run, _p_res, _p_nts, _p_t0, _p_it = _pending
            # The halo is THIS window's innovation, so the deferred window's
            # tail reads real observations instead of persistence.
            scaling_da.apply_in_kernel(
                _p_res, _p_nts, dt, _p_t0,
                halo=scaling_da.gather_innovation(run_results),
            )
            _flush_output(_p_run, _p_res, _p_it)
            _pending = None
        if seed_state:
            scaling_da.apply_in_kernel(run_results, nts, dt, t0, seed_untimed=True)

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

        # Output pass. The gage segment and the inter-gage reach below it were already
        # corrected in-kernel (downstream propagation via routing, injected above).
        # new_q0 has already copied the warmstate, so this pass spreads the recorded
        # innovation UPSTREAM into run_results for the output writer only.
        #
        # This is the diagnostic arm's ONLY placement, and it is also where the
        # prognostic arm lands on every window EXCEPT the last (see seed_state above),
        # so the written output carries the upstream correction in every window under
        # both arms. The `not seed_state` guard is what keeps apply_in_kernel to exactly
        # one call per window: it reconstructs the gage background as
        # (Q_analyzed - nudge) at the tree root and reads interior segments as-is, so a
        # second call in the same window would spread on top of already-corrected
        # interior flow.
        # Non-final windows WAIT for the next window's halo (flushed above); the
        # spread here is output-only, so deferring it cannot disturb the routing
        # chain -- new_q0 has already taken the uncorrected warmstate. The costs
        # (two windows of run_results resident; a crash in window k+1 loses
        # window k's unwritten output) are the halo's price.
        if scaling_da is not None and not seed_state:
            _pending = (run, run_results, nts, t0, run_set_iterator)
            continue

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

