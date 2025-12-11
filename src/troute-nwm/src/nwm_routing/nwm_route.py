"""A file to hold the nwm routing function"""
import logging
import time

from troute.routing.compute import compute_nhd_routing_v02, compute_diffusive_routing, compute_log_mc, compute_log_diff


LOG = logging.getLogger('')

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
    eloss_df,
    ssout,
    usgs_df,
    lastobs_df,
    reservoir_usgs_df,
    reservoir_usgs_param_df,
    reservoir_usace_df,
    reservoir_usace_param_df,
    reservoir_usbr_df,
    reservoir_usbr_param_df,
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
            reservoir_usbr_df,
            reservoir_usbr_param_df,
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
        eloss_df,
        ssout,
        usgs_df,
        lastobs_df,
        reservoir_usgs_df,
        reservoir_usgs_param_df,
        reservoir_usace_df,
        reservoir_usace_param_df,
        reservoir_usbr_df,
        reservoir_usbr_param_df,
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
