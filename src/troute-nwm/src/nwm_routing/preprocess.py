import time
import pathlib
import logging
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import xarray as xr

import troute.nhd_network_utilities_v02 as nnu
import troute.nhd_network as nhd_network
import troute.nhd_io as nhd_io

import ewts
LOG = ewts.get_logger(ewts.T_ROUTE_ID)

def _reindex_link_to_lake_id(target_df, crosswalk):
    '''
    Utility function for replacing link ID index values
    with lake ID values in a dataframe. This is used to 
    reinedex dataframes used for streamflow DA such that 
    data from data from gages located at waterbody outlets
    can be assimilated. 
    
    Arguments:
    ----------
    - target_df (DataFrame): Data frame to be reinexed
    - crosswalk      (dict): Relates lake ids to outlet link ids
    
    Returns:
    --------
    - target_df (DataFrame): Re-indexed with lake ids replacing 
                             link ids
    '''

    # evaluate intersection of link ids and target_df index values
    # i.e. what are the index positions of link ids that need replacing?
    linkids = np.fromiter(crosswalk.values(), dtype = int)
    gageidxs = target_df.index.to_numpy()
    lake_index_intersect = np.intersect1d(
        gageidxs, 
        linkids, 
        return_indices = True
    )

    # replace link ids with lake IDs in the target_df index array
    lakeids = np.fromiter(crosswalk.keys(), dtype = int)
    gageidxs[lake_index_intersect[1]] = lakeids[lake_index_intersect[2]]

    # (re) set the target_df index
    target_df.set_index(gageidxs, inplace = True)
    
    return target_df

def nwm_network_preprocess(
    supernetwork_parameters,
    waterbody_parameters,
    preprocessing_parameters,
    compute_parameters,
    data_assimilation_parameters,
):
    '''
    Creation of routing network data objects. Logical ordering of lower-level
    function calls that build individual network data objects.
    
    Arguments
    ---------
    supernetwork_parameters      (dict): user input data re network extent
    waterbody_parameters         (dict): user input data re waterbodies
    preprocessing_parameters     (dict): user input data re preprocessing
    compute_parameters           (dict): user input data re compute configuration
    data_assimilation_parameters (dict): user input data re data assimilation
    
    Returns
    -------
    connections                 (dict of int: [int]): {segment id: [downsteram adjacent segment ids]}
    param_df                             (DataFrame): Hydraulic geometry and roughness parameters, by segment
    wbody_conn                    (dict of int: int): {segment id: associated lake id}
    waterbodies_df                       (DataFrame): Waterbody (reservoir) parameters
    waterbody_types_df                   (DataFrame): Waterbody type codes (1 - levelpool, 2 - USGS, 3 - USACE, 4 - RFC)
    break_network_at_waterbodies              (bool): If True, waterbodies occpy reaches of their own
    waterbody_type_specified                  (bool): If True, more than just levelpool waterbodies exist
    link_lake_crosswalk           (dict of int: int): {lake id: outlet segment id}
    independent_networks (dict of int: {int: [int]}): {tailwater id: {segment id: [upstream adjacent segment ids]}}
    reaches_bytw              (dict of int: [[int]]): {tailwater id: list or reach lists}
    rconn                       (dict of int: [int]): {segment id: [upstream adjacent segment ids]}
    pd.DataFrame.from_dict(gages)        (DataFrame): Gage ids and corresponding segment ids at which they are located
    diffusive_network_data            (dict or None): Network data objects for diffusive domain
    topobathy_df                         (DataFrame): Natural cross section data for diffusive domain
    
    Notes
    -----
    - waterbody_type_specified is likely an excessive return and can be removed and inferred from the 
      contents of waterbody_types_df
    - The values of the link_lake_crosswalk dictionary are the downstream-most segments within 
      the waterbody extent to which waterbody data are written. They are NOT the first segments 
      downsteram of the waterbody 
    '''

    #============================================================================
    # Establish diffusive domain for MC/diffusive hybrid simulations
    
    hybrid_params = compute_parameters.get("hybrid_parameters", False)
    if hybrid_params:
        # switch parameters
        # if run_hybrid = False, run MC only
        # if run_hybrid = True, if use_topobathy = False, run MC+diffusive on RouteLink.nc
        #    "      "      "  , if use_topobathy = True,  if run_refactored_network = False, run MC+diffusive on original hydrofabric
        #    "      "      "  , if use_topobathy = True,  if run_refactored_network = True,  run MC+diffusive on refactored hydrofabric
        run_hybrid             = hybrid_params.get('run_hybrid_routing', False)
        use_topobathy          = hybrid_params.get('use_natl_xsections', False) 
        run_refactored         = hybrid_params.get('run_refactored_network', False)
        
        # file path parameters of non-refactored hydrofabric defined by RouteLink.nc
        domain_file            = hybrid_params.get("diffusive_domain",   None)
        topobathy_file         = hybrid_params.get("topobathy_domain",   None)
        
        # file path parameters of refactored hydrofabric for diffusive wave channel routing 
        refactored_domain_file    = hybrid_params.get("refactored_domain", None)
        refactored_topobathy_file = hybrid_params.get("refactored_topobathy_domain", None)
        #-------------------------------------------------------------------------
        # for non-refactored hydofabric defined by RouteLink.nc
        # TODO: By default, make diffusive available for both non-refactored and refactored hydrofabric for now. Place a switch in the future. 
        if run_hybrid and domain_file:
            
            LOG.info('reading diffusive domain extent for MC/Diffusive hybrid simulation')
            
            # read diffusive domain dictionary from yaml or json
            diffusive_domain = nhd_io.read_diffusive_domain(domain_file)           
            
            if use_topobathy and topobathy_file:
                
                LOG.debug('Natural cross section data on original hydrofabric are provided.')
                
                # read topobathy domain netcdf file, set index to 'comid'
                # TODO: replace 'link' with a user-specified indexing variable name.
                # ... if for whatever reason there is not a `link` variable in the 
                # ... dataframe returned from read_netcdf, then the code would break here.
                topobathy_df = (nhd_io.read_netcdf(topobathy_file).set_index('link'))
                
                # TODO: Request GID make comID variable an integer in their product, so
                # we do not need to change variable types, here.
                topobathy_df.index = topobathy_df.index.astype(int)
                
            else:
                topobathy_df = pd.DataFrame()
                LOG.debug('No natural cross section topobathy data provided. Hybrid simualtion will run on compound trapezoidal geometry.')
             
            # initialize a dictionary to hold network data for each of the diffusive domains
            diffusive_network_data = {}
        
        else:
            diffusive_domain       = None
            diffusive_network_data = None
            topobathy_df              = pd.DataFrame()
            LOG.info('No diffusive domain file specified in configuration file. This is an MC-only simulation')
        unrefactored_topobathy_df = pd.DataFrame()    
        #-------------------------------------------------------------------------
        # for refactored hydofabric 
        if run_hybrid and run_refactored and refactored_domain_file:
            
            LOG.info('reading refactored diffusive domain extent for MC/Diffusive hybrid simulation')
            
            # read diffusive domain dictionary from yaml or json
            refactored_diffusive_domain = nhd_io.read_diffusive_domain(refactored_domain_file)           
            
            if use_topobathy and refactored_topobathy_file:
                
                LOG.debug('Natural cross section data of refactored hydrofabric are provided.')
                
                # read topobathy domain netcdf file, set index to 'comid'
                # TODO: replace 'link' with a user-specified indexing variable name.
                # ... if for whatever reason there is not a `link` variable in the 
                # ... dataframe returned from read_netcdf, then the code would break here.
                topobathy_df = (nhd_io.read_netcdf(refactored_topobathy_file).set_index('link'))

                # unrefactored_topobaty_data is passed to diffusive kernel to provide thalweg elevation of unrefactored topobathy 
                # for crosswalking water elevations between non-refactored and refactored hydrofabrics. 
                unrefactored_topobathy_df       = (nhd_io.read_netcdf(topobathy_file).set_index('link'))
                unrefactored_topobathy_df.index = unrefactored_topobathy_df.index.astype(int)
                
            else:
                topobathy_df               = pd.DataFrame()
                LOG.debug('No natural cross section topobathy data of refactored hydrofabric provided. Hybrid simualtion will run on compound trapezoidal geometry.')
             
            # initialize a dictionary to hold network data for each of the diffusive domains
            refactored_diffusive_network_data = {}
        
        else:
            refactored_diffusive_domain       = None
            refactored_diffusive_network_data = None
            refactored_reaches                = {}
            LOG.info('No refactored diffusive domain file specified in configuration file. This is an MC-only simulation')     
   
    else:
        diffusive_domain                  = None
        diffusive_network_data            = None
        topobathy_df                      = pd.DataFrame()
        unrefactored_topobathy_df         = pd.DataFrame() 
        refactored_diffusive_domain       = None
        refactored_diffusive_network_data = None   
        refactored_reaches                = {}
        LOG.info('No hybrid parameters specified in configuration file. This is an MC-only simulation')

    #============================================================================
    # Build network connections graph, assemble parameter dataframe, 
    # establish segment-waterbody, and segment-gage mappings
    LOG.info("creating network connections graph")
    start_time = time.time()
    
    connections, param_df, wbody_conn, gages = nnu.build_connections(
        supernetwork_parameters,
    )
    
    link_gage_df = pd.DataFrame.from_dict(gages)
    link_gage_df.index.name = 'link'
    break_network_at_waterbodies = waterbody_parameters.get(
        "break_network_at_waterbodies", False
    )
    
    # if streamflow DA, then break network at gages
    break_network_at_gages = False
    streamflow_da = data_assimilation_parameters.get('streamflow_da', False)
    if streamflow_da:
        break_network_at_gages = streamflow_da.get('streamflow_nudging', False)

    if not wbody_conn: 
        # Turn off any further reservoir processing if the network contains no 
        # waterbodies
        break_network_at_waterbodies = False

    # if waterbodies are being simulated, adjust the connections graph so that 
    # waterbodies are collapsed to single nodes. Also, build a mapping between 
    # waterbody outlet segments and lake ids
    if break_network_at_waterbodies:
        connections, link_lake_crosswalk = nhd_network.replace_waterbodies_connections(
            connections, wbody_conn
        )
    else:
        link_lake_crosswalk = None

    LOG.debug("network connections graph created in %s seconds." % (time.time() - start_time))

    #============================================================================
    # Retrieve and organize waterbody parameters

    waterbody_type_specified = False
    if break_network_at_waterbodies:
        
        # Read waterbody parameters from LAKEPARM file
        level_pool_params = waterbody_parameters.get('level_pool', defaultdict(list))
        waterbodies_df = nhd_io.read_lakeparm(
            level_pool_params['level_pool_waterbody_parameter_file_path'],
            level_pool_params.get("level_pool_waterbody_id", 'lake_id'),
            wbody_conn.values()
        )

        # Remove duplicate lake_ids and rows
        waterbodies_df = (
            waterbodies_df.reset_index()
            .drop_duplicates(subset="lake_id")
            .set_index("lake_id")
        )

        # Declare empty dataframe
        waterbody_types_df = pd.DataFrame()

        # Check if hybrid-usgs or hybrid-usace reservoir DA is set to True
        reservoir_da = data_assimilation_parameters.get(
            'reservoir_da', 
            {}
        )
        
        if reservoir_da:
            usgs_hybrid  = reservoir_da.get(
                'reservoir_persistence_usgs', 
                False
            )
            usace_hybrid = reservoir_da.get(
                'reservoir_persistence_usace', 
                False
            )
            param_file   = reservoir_da.get(
                'gage_lakeID_crosswalk_file',
                None
            )
        else:
            param_file = None
            usace_hybrid = False
            usgs_hybrid = False
            
        # check if RFC-type reservoirs are set to true
        rfc_params = waterbody_parameters.get('rfc')
        if rfc_params:
            rfc_forecast = rfc_params.get(
                'reservoir_rfc_forecasts',
                False
            )
            param_file = rfc_params.get('reservoir_parameter_file',None)
        else:
            rfc_forecast = False

        if (param_file and reservoir_da) or (param_file and rfc_forecast):
            waterbody_type_specified = True
            (
                waterbody_types_df, 
                usgs_lake_gage_crosswalk, 
                usace_lake_gage_crosswalk
            ) = nhd_io.read_reservoir_parameter_file(
                param_file,
                usgs_hybrid,
                usace_hybrid,
                rfc_forecast,
                level_pool_params.get("level_pool_waterbody_id", 'lake_id'),
                reservoir_da.get('crosswalk_usgs_gage_field', 'usgs_gage_id'),
                reservoir_da.get('crosswalk_usgs_lakeID_field', 'usgs_lake_id'),
                reservoir_da.get('crosswalk_usace_gage_field', 'usace_gage_id'),
                reservoir_da.get('crosswalk_usace_lakeID_field', 'usace_lake_id'),
                wbody_conn.values(),
            )
        else:
            waterbody_type_specified = True
            waterbody_types_df = pd.DataFrame(data = 1, index = waterbodies_df.index, columns = ['reservoir_type'])
            usgs_lake_gage_crosswalk = None
            usace_lake_gage_crosswalk = None

    else:
        # Declare empty dataframes
        waterbody_types_df = pd.DataFrame()
        waterbodies_df = pd.DataFrame()
        usgs_lake_gage_crosswalk = None
        usace_lake_gage_crosswalk = None

    #============================================================================
    # build diffusive domain data and edit MC domain data for hybrid simulation
    
    #

    if diffusive_domain:
        rconn_diff0 = nhd_network.reverse_network(connections)
        refactored_reaches = {}
        
        for tw in diffusive_domain:
            mainstem_segs = diffusive_domain[tw]['links']
            # we want mainstem_segs start at a mainstem link right after the upstream boundary mainstem link, which is
            # in turn not under any waterbody. This boundary mainstem link should be turned into a tributary segment.
            upstream_boundary_mainstem_link = diffusive_domain[tw]['upstream_boundary_link_mainstem']         
            if upstream_boundary_mainstem_link[0] in mainstem_segs:
                mainstem_segs.remove(upstream_boundary_mainstem_link[0])
            
            # ===== build diffusive network data objects ==== 
            diffusive_network_data[tw] = {}

            # add diffusive domain segments
            diffusive_network_data[tw]['mainstem_segs'] =  mainstem_segs

            # diffusive domain tributary segments
            trib_segs = []

            for seg in mainstem_segs:
                us_list = rconn_diff0[seg]
                for u in us_list:
                    if u not in mainstem_segs:
                        trib_segs.append(u) 

            diffusive_network_data[tw]['tributary_segments'] = trib_segs
            # diffusive domain connections object
            diffusive_network_data[tw]['connections'] = {k: connections[k] for k in (mainstem_segs + trib_segs)}       

            # diffusive domain reaches and upstream connections. 
            # break network at tributary segments
            _, reaches, rconn_diff = nnu.organize_independent_networks(
                diffusive_network_data[tw]['connections'],
                set(trib_segs),
                set(),
            )
            
            diffusive_network_data[tw]['rconn'] = rconn_diff
            diffusive_network_data[tw]['reaches'] = reaches[tw]
            
            # RouteLink parameters
            diffusive_network_data[tw]['param_df'] = param_df.filter(
                (mainstem_segs + trib_segs),
                axis = 0,
            )

            diffusive_network_data[tw]['upstream_boundary_link'] = upstream_boundary_mainstem_link
            
            if refactored_diffusive_domain: 
                diffusive_parameters = {'geo_file_path': refactored_topobathy_file}
                refactored_connections = nnu.build_refac_connections(diffusive_parameters)

                # list of stream segments of a single refactored diffusive domain 
                rlinks_tw = refactored_diffusive_domain[tw]['rlinks']
                refactored_connections_tw = {}
                
                # Subset a connection dictionary (upstream segment as key : downstream segments as values) from refactored_connections
                # for a single refactored diffusive domain defined by a current tw. 
                for k in rlinks_tw:
                    if k in refactored_connections.keys():
                        refactored_connections_tw[k] = refactored_connections[k]

                refac_tw = refactored_diffusive_domain[tw]['refac_tw']
                refactored_diffusive_network_data[refac_tw] = {}                
                refactored_diffusive_network_data[refac_tw]['tributary_segments'] = trib_segs
                refactored_diffusive_network_data[refac_tw]['connections'] = refactored_connections_tw                 

                for k in trib_segs:
                    refactored_diffusive_network_data[refac_tw]['connections'][k]= [refactored_diffusive_domain[tw]['incoming_tribs'][k]]

                # diffusive domain reaches and upstream connections. 
                # break network at tributary segments
                _, refactored_reaches_batch, refactored_conn_diff = nnu.organize_independent_networks(
                                                            refactored_diffusive_network_data[refac_tw]['connections'],
                                                            set(trib_segs),
                                                            set(),
                                                            )

                refactored_reaches[refac_tw] = refactored_reaches_batch[refac_tw]
                refactored_diffusive_network_data[refac_tw]['mainstem_segs'] = refactored_diffusive_domain[tw]['rlinks']
                refactored_diffusive_network_data[refac_tw]['upstream_boundary_link'] = diffusive_network_data[tw]['upstream_boundary_link'] 
            else:
                refactored_reaches={}

            # ==== remove diffusive domain segs from MC domain ====        
            # drop indices from param_df
            param_df = param_df.drop(mainstem_segs)

            # remove keys from connections dictionary
            for s in mainstem_segs:
                connections.pop(s)

            # update downstream connections of trib segs
            for us in trib_segs:
                connections[us] = []

    #============================================================================
    # Identify Independent Networks and Reaches by Network
    LOG.info("organizing connections into reaches ...")
    start_time = time.time() 
    gage_break_segments = set()
    wbody_break_segments = set()
    if break_network_at_waterbodies:
        wbody_break_segments = wbody_break_segments.union(wbody_conn.values())
        
    if break_network_at_gages:
        gage_break_segments = gage_break_segments.union(gages['gages'].keys())
 
    independent_networks, reaches_bytw, rconn = nnu.organize_independent_networks(
        connections,
        wbody_break_segments,
        gage_break_segments,
    )
    
    LOG.debug("reach organization complete in %s seconds." % (time.time() - start_time))
    
    if preprocessing_parameters.get('preprocess_only', False):

        LOG.debug("saving preprocessed network data to disk for future use")
        # todo: consider a better default than None
        destination_folder = preprocessing_parameters.get('preprocess_output_folder', None)
        if destination_folder:

            output_filename = preprocessing_parameters.get(
                'preprocess_output_filename', 
                'preprocess_output'
            )

            outputs = {}
            outputs.update(
                {'connections': connections,
                 'param_df': param_df,
                 'wbody_conn': wbody_conn,
                 'waterbodies_df': waterbodies_df,
                 'waterbody_types_df': waterbody_types_df,
                 'break_network_at_waterbodies': break_network_at_waterbodies,
                 'waterbody_type_specified': waterbody_type_specified,
                 'link_lake_crosswalk': link_lake_crosswalk,
                 'independent_networks': independent_networks,
                 'reaches_bytw': reaches_bytw,
                 'rconn': rconn,
                 'link_gage_df': link_gage_df,
                 'usgs_lake_gage_crosswalk': usgs_lake_gage_crosswalk, 
                 'usace_lake_gage_crosswalk': usace_lake_gage_crosswalk,
                 'diffusive_network_data': diffusive_network_data,
                 'topobathy_data': topobathy_df,
                }
            )
            try:
                np.save(
                    pathlib.Path(destination_folder).joinpath(output_filename),
                    outputs
                )
            except:
                LOG.critical('Canonot find %s. Aborting preprocessing routine' % pathlib.Path(destination_folder))
                quit()
                
            LOG.debug(
                "writing preprocessed network data to %s"\
                % pathlib.Path(destination_folder).joinpath(output_filename + '.npy'))
            LOG.critical(
                "Preprocessed network data written to %s aborting preprocessing sequence" \
                % pathlib.Path(destination_folder).joinpath(output_filename + '.npy'))
            quit()

        else:
            LOG.critical(
                "No destination folder specified for preprocessing. Please specify preprocess_output_folder in configuration file. Aborting preprocessing routine"
            )
            quit()

    return (
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
    )

def unpack_nwm_preprocess_data(preprocessing_parameters):
    
    preprocess_filepath = preprocessing_parameters.get('preprocess_source_file',None)
    if preprocess_filepath:
        try:
            inputs = np.load(pathlib.Path(preprocess_filepath),allow_pickle='TRUE').item()
        except:
            LOG.critical('Canonot find %s' % pathlib.Path(preprocess_filepath))
            quit()
              
        connections                  = inputs.get('connections',None)            
        param_df                     = inputs.get('param_df',None)
        wbody_conn                   = inputs.get('wbody_conn',None)
        waterbodies_df               = inputs.get('waterbodies_df',None)
        waterbody_types_df           = inputs.get('waterbody_types_df',None)
        break_network_at_waterbodies = inputs.get('break_network_at_waterbodies',None)
        waterbody_type_specified     = inputs.get('waterbody_type_specified',None)
        link_lake_crosswalk          = inputs.get('link_lake_crosswalk', None)
        independent_networks         = inputs.get('independent_networks',None)
        reaches_bytw                 = inputs.get('reaches_bytw',None)
        rconn                        = inputs.get('rconn',None)
        gages                        = inputs.get('link_gage_df',None)
        usgs_lake_gage_crosswalk     = inputs.get('usgs_lake_gage_crosswalk',None)
        usace_lake_gage_crosswalk    = inputs.get('usace_lake_gage_crosswalk',None)
        diffusive_network_data       = inputs.get('diffusive_network_data',None)
        topobathy_df                 = inputs.get('topobathy_data',None)        
        refactored_diffusive_domain  = inputs.get('refactored_diffusive_domain',None)
        refactored_reaches           = inputs.get('refactored_reaches',None)
        unrefactored_topobathy_df    = inputs.get('unrefactored_topobathy',None)

    else:
        LOG.critical("use_preprocessed_data = True, but no preprocess_source_file is specified. Aborting the simulation.")
        quit()
                         
    return (
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
        gages,
        usgs_lake_gage_crosswalk, 
        usace_lake_gage_crosswalk,
        diffusive_network_data,
        topobathy_df,
        refactored_diffusive_domain,
        refactored_reaches,
        unrefactored_topobathy_df,
    )


def nwm_initial_warmstate_preprocess(
    break_network_at_waterbodies,
    restart_parameters,
    data_assimilation_parameters,
    segment_index,
    waterbodies_df,
    link_lake_crosswalk,
):

    '''
    Assemble model initial condition data:
        - waterbody inital states (outflow and pool elevation)
        - channel initial states (flow and depth)
        - initial time
        - most recent gage observations
    Additionally, a dictionary of data assimilation parameters is assembled.
    
    Arguments
    ---------
    - break_network_at_waterbodies (bool): If True, waterbody initial states will
                                           be appended to the waterbody parameter
                                           dataframe. If False, waterbodies will
                                           not be simulated and the waterbody
                                           parameter datataframe wil not be changed
    
    - restart_parameters           (dict): User-input simulation restart 
                                           parameters
    
    - data_assimilation_parameters (dict): User-input data assimilation 
                                           parameters
    
    - segment_index        (Pandas Index): All segment IDs in the simulation 
                                           doamin
    
    - waterbodies_df   (Pandas DataFrame): Waterbody parameters
    
    - link_lake_crosswalk          (dict): Crosswalking between lake ids and the link
                                           id of the lake outlet segment
    
    Returns
    -------
    - waterbodies_df (Pandas DataFrame): Waterbody parameters with initial
                                         states (outflow and pool elevation)
    
    - q0             (Pandas DataFrame): Initial flow and depth states for each
                                         segment in the model domain
    
    - t0                     (datetime): Datetime of the model initialization
    
    - lastobs_df     (Pandas DataFrame): Last gage observations data for DA
    
    - da_parameter_dict          (dict): Data assimilation parameters
    
    Notes
    -----
    - I don't know if it makes sense to create the da_parameter_dict, here. 
      Consider moving the creation to another location..
    '''

    #----------------------------------------------------------------------------
    # Assemble waterbody initial states (outflow and pool elevation
    #----------------------------------------------------------------------------
    
    if break_network_at_waterbodies:

        start_time = time.time()
        LOG.info("setting waterbody initial states ...")

        # if a lite restart file is provided, read initial states from it.
        if restart_parameters.get("lite_waterbody_restart_file", None):
            
            waterbodies_initial_states_df, _ = nhd_io.read_lite_restart(
                restart_parameters['lite_waterbody_restart_file']
            )
            
        # read waterbody initial states from WRF-Hydro type restart file
        elif restart_parameters.get("wrf_hydro_waterbody_restart_file", None):
            waterbodies_initial_states_df = nhd_io.get_reservoir_restart_from_wrf_hydro(
                restart_parameters["wrf_hydro_waterbody_restart_file"],
                restart_parameters["wrf_hydro_waterbody_ID_crosswalk_file"],
                restart_parameters.get("wrf_hydro_waterbody_ID_crosswalk_file_field_name", 'lake_id'),
                restart_parameters["wrf_hydro_waterbody_crosswalk_filter_file"],
                restart_parameters.get(
                    "wrf_hydro_waterbody_crosswalk_filter_file_field_name",
                    'NHDWaterbodyComID'
                ),
            )
        
        # if no restart file is provided, default initial states
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

        LOG.debug(
            "waterbody initial states complete in %s seconds."\
            % (time.time() - start_time))
        start_time = time.time()

    #----------------------------------------------------------------------------
    # Assemble channel initial states (flow and depth)
    # also establish simulation initialization timestamp
    #----------------------------------------------------------------------------
    
    start_time = time.time()
    LOG.info("setting channel initial states ...")

    # if lite restart file is provided, the read channel initial states from it
    if restart_parameters.get("lite_channel_restart_file", None):
        
        q0, t0 = nhd_io.read_lite_restart(
            restart_parameters['lite_channel_restart_file']
        )
        t0_str = None

    # build initial states from user-provided restart parameters
    else:
        q0 = nnu.build_channel_initial_state(restart_parameters, segment_index)

        # get initialization time from restart file
        if restart_parameters.get("wrf_hydro_channel_restart_file", None):
            channel_initial_states_file = restart_parameters[
                "wrf_hydro_channel_restart_file"
            ]
            t0_str = nhd_io.get_param_str(
                channel_initial_states_file, 
                "Restart_Time"
            )
        else:
            t0_str = "2015-08-16_00:00:00"

        # convert timestamp from string to datetime
        t0 = datetime.strptime(t0_str, "%Y-%m-%d_%H:%M:%S")

    # get initial time from user inputs
    if restart_parameters.get("start_datetime", None):
        t0_str = restart_parameters.get("start_datetime")
        
        def _try_parsing_date(text):
            for fmt in (
                "%Y-%m-%d_%H:%M", 
                "%Y-%m-%d_%H:%M:%S", 
                "%Y-%m-%d %H:%M", 
                "%Y-%m-%d %H:%M:%S", 
                "%Y/%m/%d %H:%M", 
                "%Y/%m/%d %H:%M:%S"
            ):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    pass
            LOG.error('No valid date format found for start_datetime input. Please use format YYYY-MM-DD_HH:MM')
            quit()
            
        t0 = _try_parsing_date(t0_str)
    else:
        if t0_str == "2015-08-16_00:00:00":
            LOG.info('No user-input start_datetime and no restart file, start time arbitrarily 2015-08-16_00:00:00')
        else:
            LOG.info('No user-specified start_datetime, continuing with start time from restart file: %s', t0_str)
        
    #----------------------------------------------------------------------------
    # Assemble streamflow DA lastobs data
    #----------------------------------------------------------------------------
    
    lastobs_df, da_parameter_dict = nnu.build_data_assimilation_lastobs(
        data_assimilation_parameters
    )
    
    # replace link ids with lake ids, for gages at waterbody outlets, 
    # otherwise, gage data will not be assimilated at waterbody outlet
    # segments.
    if link_lake_crosswalk:
        lastobs_df = _reindex_link_to_lake_id(lastobs_df, link_lake_crosswalk)

    LOG.debug(
        "channel initial states complete in %s seconds."\
        % (time.time() - start_time)
    )
    start_time = time.time()

    return waterbodies_df, q0, t0, lastobs_df, da_parameter_dict
    # TODO: This returns a full dataframe (waterbodies_df) with the
    # merged initial states for waterbodies, but only the
    # initial state values (q0; not merged with the channel properties)
    # for the channels --
    # That is because that is how they are used downstream. Need to
    # trace that back and decide if there is one of those two ways
    # that is optimal and make both returns that way.


def nwm_forcing_preprocess(
    run,
    forcing_parameters,
    hybrid_parameters,
    da_run,
    data_assimilation_parameters,
    break_network_at_waterbodies,
    segment_index,
    link_gage_df,
    usgs_lake_gage_crosswalk, 
    usace_lake_gage_crosswalk,
    link_lake_crosswalk,
    lastobs_index,
    cpu_pool,
    t0,
):
    """
    Assemble model forcings. Forcings include hydrological lateral inflows (qlats)
    and observations for streamflow and/or reservoir data assimilation schemes
    
    Aguments
    --------
    - run                          (dict): List of forcing files pertaining
                                           to a single run-set
    
    - forcing_parameters           (dict): User-input simulation forcing 
                                           parameters
    
    - da_run                       (dict): Lists of TimeSlice files for a
                                           single run-set
    
    - data_assimilation_parameters (dict): User-input parameters controlling
                                           data assimilation routines
    
    - break_network_at_waterbodies (bool): ????
    
    - segment_index                    (): ????
    
    - link_gage_df     (Pandas DataFrame): Crosswalking between segment ID and
                                           USGS gage IDs in the model domain
                                           
    - link_lake_crosswalk          (dict): Crosswalking between lake ids and the link
                                           id of the lake outlet segment
    
    - lastobs_index        (Pandas Index): ????
    
    - cpu_pool                      (int): Number of CPUs in the process-parall
                                           pool
    
    - t0                       (datetime): Timestamp of the simualtion initial 
                                           condition

    Returns
    -------
    - qlats_df           (Pandas DataFrame): Lateral inflow data, indexed by 
                                             segment ID
                                             
    - usgs_df            (Pandas DataFrame): Streamflow DA observations, indexed by
                                             segment ID
                                             
    - reservoir_usgs_df  (Pandas DataFrame): Reservoir DA observations 
                                             (USGS persistence reservoirs), indexed 
                                             by waterbody ID
                                             
    - reservoir_usace_df (Pandas DataFrame): Reservoir DA observations 
                                             (USACE persistence reservoirs), indexed 
                                             by waterbody ID
    
    Notes
    -----
    - This function adds new keys to the `run` input dictionary
    
    """

    # Unpack user-specified forcing parameters
    dt                           = forcing_parameters.get("dt", None)
    qts_subdivisions             = forcing_parameters.get("qts_subdivisions", None)
    qlat_input_folder            = forcing_parameters.get("qlat_input_folder", None)
    qlat_file_index_col          = forcing_parameters.get("qlat_file_index_col", "feature_id")
    qlat_file_value_col          = forcing_parameters.get("qlat_file_value_col", "q_lateral")
    qlat_file_gw_bucket_flux_col = forcing_parameters.get("qlat_file_gw_bucket_flux_col", "qBucket")
    qlat_file_terrain_runoff_col = forcing_parameters.get("qlat_file_terrain_runoff_col", "qSfcLatRunoff")

    # TODO: find a better way to deal with these defaults and overrides.
    run["t0"]                           = run.get("t0", t0)
    run["nts"]                          = run.get("nts")
    run["dt"]                           = run.get("dt", dt)
    run["qts_subdivisions"]             = run.get("qts_subdivisions", qts_subdivisions)
    run["qlat_input_folder"]            = run.get("qlat_input_folder", qlat_input_folder)
    run["qlat_file_index_col"]          = run.get("qlat_file_index_col", qlat_file_index_col)
    run["qlat_file_value_col"]          = run.get("qlat_file_value_col", qlat_file_value_col)
    run["qlat_file_gw_bucket_flux_col"] = run.get("qlat_file_gw_bucket_flux_col", qlat_file_gw_bucket_flux_col)
    run["qlat_file_terrain_runoff_col"] = run.get("qlat_file_terrain_runoff_col", qlat_file_terrain_runoff_col)

    #---------------------------------------------------------------------------
    # Assemble lateral inflow data
    #---------------------------------------------------------------------------
    
    start_time = time.time()
    LOG.info("Creating a DataFrame of lateral inflow forcings ...")

    qlats_df = nnu.build_qlateral_array(
        run,
        cpu_pool,
        segment_index,
    )

    LOG.debug(
        "lateral inflow DataFrame creation complete in %s seconds." \
        % (time.time() - start_time)
    )
    
    #---------------------------------------------------------------------------
    # Assemble streamflow DA observation data
    #---------------------------------------------------------------------------
    
    # isolate user-input parameters for streamflow data assimilation
    streamflow_da_parameters = data_assimilation_parameters.get(
        'streamflow_da', 
        None
    )
    
    # determine if user explictly requests streamflow DA
    nudging = False
    if streamflow_da_parameters:
        nudging = streamflow_da_parameters.get('streamflow_nudging', False)
        
    # if user requested nudging and a specified a USGS TimeSlice directory, 
    # then build and return USGS dataframe
    if nudging and da_run['usgs_timeslice_files']:
        
        start_time = time.time()
        LOG.info(
            "Creating a DataFrame of USGS gage observations for streamflow DA ..."
        )
        
        usgs_timeslices_folder = data_assimilation_parameters.get(
                                    "usgs_timeslices_folder", 
                                    None
                                )
        lastobs_file           = streamflow_da_parameters.get(
                                    "wrf_hydro_lastobs_file", 
                                    None
                                )
        lastobs_start          = data_assimilation_parameters.get(
            "wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time",
                                    0
                                )
        lastobs_type           = data_assimilation_parameters.get(
                                    "wrf_lastobs_type", 
                                    "error-based"
                                )
        crosswalk_file         = streamflow_da_parameters.get(
                                    "gage_segID_crosswalk_file", 
                                    None
                                )
        crosswalk_gage_field   = streamflow_da_parameters.get(
                                    'crosswalk_gage_field',
                                    'gages'
                                )
        crosswalk_segID_field  = streamflow_da_parameters.get(
                                    'crosswalk_segID_field',
                                    'link'
                                )
        da_decay_coefficient   = data_assimilation_parameters.get(
                                    "da_decay_coefficient",
                                    120
                                )
        qc_threshold           = data_assimilation_parameters.get(
                                    "qc_threshold",
                                    1
                                )
        interpolation_limit    = data_assimilation_parameters.get(
                                    "interpolation_limit_min",
                                    59
                                )
        
        # TODO: join timeslice folder and files into complete path upstream
        usgs_timeslices_folder = pathlib.Path(usgs_timeslices_folder)
        usgs_files = [usgs_timeslices_folder.joinpath(f) for f in 
                      da_run['usgs_timeslice_files']]
        
        if usgs_files:
            usgs_df = (
                nhd_io.get_obs_from_timeslices(
                    link_gage_df,
                    crosswalk_gage_field,
                    crosswalk_segID_field,
                    usgs_files,
                    qc_threshold,
                    interpolation_limit,
                    run["dt"],
                    run["t0"],
                    cpu_pool
                ).
                loc[link_gage_df.index]
            )

        else:
            usgs_df = pd.DataFrame()
        
        LOG.debug(
            "Streamflow DA USGS observation DataFrame creation complete in %s seconds." \
            % (time.time() - start_time)
        )
                 
    else: 
        usgs_df = pd.DataFrame()
    
    #---------------------------------------------------------------------------
    # Assemble reservoir data assimilation observation data
    #---------------------------------------------------------------------------
    
    # isolate user-input parameters for reservoir data assimilation
    reservoir_da_parameters = data_assimilation_parameters.get(
        'reservoir_da', 
        None
    )
    
    # check if user explictly requests USGS and/or USACE reservoir DA
    usgs_persistence  = False
    usace_persistence = False
    if reservoir_da_parameters:
        usgs_persistence  = reservoir_da_parameters.get(
            'reservoir_persistence_usgs', 
            False
        )
        usace_persistence = reservoir_da_parameters.get(
            'reservoir_persistence_usace', 
            False
        )
        
    #---------------------------------------------
    # observations for USGS reservoir DA
    #---------------------------------------------
    
    # create crosswalking dataframes and identify USGS lakeIDs in the model domain
    if usgs_persistence:
        
        gage_lake_df = (
            usgs_lake_gage_crosswalk.
            reset_index().
            set_index(['usgs_gage_id']) # <- TODO use input parameter for this
        )
        
        # build dataframe that crosswalks gageIDs to segmentIDs
        gage_link_df = (
            link_gage_df['gages'].
            reset_index().
            set_index(['gages'])
        )
        
        # extract the USGS lakeIDs within the model domain
        usgs_lakes_in_domain = (
            gage_lake_df.join(gage_link_df, how = 'inner')['usgs_lake_id'].
            to_numpy()
        )
    
    # if USGS TimeSlices have already been opened and assembled for 
    # streamflow DA, then simply take TimeSlice observations from `usgs_df`,
    # no need to open TimeSlice files again.
    if usgs_persistence and nudging and usgs_df.empty == False:
        
        start_time = time.time()
        LOG.info(
            "Creating a DataFrame of USGS gage observations for Reservoir DA ..."
        )
        
        
        # build dataframe that crosswalks segmentIDs to lakeIDs
        link_lake_df = (
            gage_lake_df.join(gage_link_df, how = 'inner').
            reset_index(drop=True).set_index('link')
        )
        
        # resample `usgs_df` to 15 minute intervals
        usgs_df_15min = (
            usgs_df.
            transpose().
            resample('15min').asfreq().
            transpose()
        )
        
        # subset and re-index `usgs_df`, using the segID <> lakeID crosswalk
        reservoir_usgs_df = (
            usgs_df_15min.join(link_lake_df, how = 'inner').
            reset_index(drop=True).
            set_index('usgs_lake_id')
        )
        
        LOG.debug(
            "Reservoir DA USGS observation DataFrame creation complete in %s seconds." \
            % (time.time() - start_time)
        )
                    
    elif usgs_persistence and not nudging:
        
        start_time = time.time()
        LOG.info("Creating a DataFrame of USGS gage observations for Reservoir DA ...")
        
        usgs_timeslices_folder = data_assimilation_parameters.get(
                                   "usgs_timeslices_folder",
                                   None
                                )
        crosswalk_file         = reservoir_da_parameters.get(
                                    "gage_lakeID_crosswalk_file", 
                                    None
                                )
        crosswalk_gage_field   = streamflow_da_parameters.get(
                                    'crosswalk_usgs_gage_field',
                                    'usgs_gage_id'
                                )
        crosswalk_lakeID_field  = streamflow_da_parameters.get(
                                    'crosswalk_usgs_lakeID_field',
                                    'usgs_lake_id'
                                )
        qc_threshold            = data_assimilation_parameters.get(
                                    "qc_threshold",
                                    1
                                )
        interpolation_limit     = data_assimilation_parameters.get(
                                    "interpolation_limit_min",
                                    59
                                )
        
        # TODO: join timeslice folder and files into complete path upstream in workflow
        usgs_timeslices_folder = pathlib.Path(usgs_timeslices_folder)
        usgs_files = [usgs_timeslices_folder.joinpath(f) for f in
                      da_run['usgs_timeslice_files']]
                
        if usgs_files:
            
            reservoir_usgs_df = nhd_io.get_obs_from_timeslices(
                usgs_lake_gage_crosswalk,
                crosswalk_gage_field,
                crosswalk_lakeID_field,
                usgs_files,
                qc_threshold,
                interpolation_limit,
                900,                      # 15 minutes, as secs
                run["t0"],
                cpu_pool
            )
            
        else:
            reservoir_usgs_df = pd.DataFrame()
            
        LOG.debug(
            "Reservoir DA USGS observation DataFrame creation complete in %s seconds." \
            % (time.time() - start_time)
        )  
        
    else:
        reservoir_usgs_df = pd.DataFrame()
    
     # replace link ids with lake ids, for gages at waterbody outlets, 
    # otherwise, gage data will not be assimilated at waterbody outlet
    # segments.
    if link_lake_crosswalk:
        usgs_df = _reindex_link_to_lake_id(usgs_df, link_lake_crosswalk)
                    
    # create USGS reservoir hybrid DA initial parameters dataframe    
    if reservoir_usgs_df.empty == False:
        reservoir_usgs_param_df = pd.DataFrame(
            data = 0, 
            index = reservoir_usgs_df.index ,
            columns = ['update_time']
        )
        reservoir_usgs_param_df['prev_persisted_outflow'] = np.nan
        reservoir_usgs_param_df['persistence_update_time'] = 0
        reservoir_usgs_param_df['persistence_index'] = 0
    else:
        reservoir_usgs_param_df = pd.DataFrame()

    #---------------------------------------------
    # observations for USACE reservoir DA
    #---------------------------------------------  
    
    if usace_persistence:
        start_time = time.time()
        LOG.info("Creating a DataFrame of USACE gage observations for Reservoir DA ...")
        
        usace_timeslices_folder = data_assimilation_parameters.get(
                                   "usace_timeslices_folder",
                                   None
                                )
        crosswalk_file          = reservoir_da_parameters.get(
                                    "gage_lakeID_crosswalk_file", 
                                    None
                                )
        crosswalk_gage_field    = streamflow_da_parameters.get(
                                    'crosswalk_usace_gage_field',
                                    'usace_gage_id'
                                )
        crosswalk_lakeID_field   = streamflow_da_parameters.get(
                                    'crosswalk_usace_lakeID_field',
                                    'usace_lake_id'
                                )
        qc_threshold            = data_assimilation_parameters.get(
                                    "qc_threshold",
                                    1
                                )
        interpolation_limit     = data_assimilation_parameters.get(
                                    "interpolation_limit_min",
                                    59
                                )
        
        # TODO: join timeslice folder and files into complete path upstream in workflow
        usace_timeslices_folder = pathlib.Path(usace_timeslices_folder)
        usace_files = [usace_timeslices_folder.joinpath(f) for f in da_run['usace_timeslice_files']]
                
        if usace_files:
            reservoir_usace_df = nhd_io.get_obs_from_timeslices(
                usace_lake_gage_crosswalk,
                crosswalk_gage_field,
                crosswalk_lakeID_field,
                usace_files,
                qc_threshold,
                interpolation_limit,
                900,                      # 15 minutes, as secs
                run["t0"],
                cpu_pool
            )
        else:
            reservoir_usace_df = pd.DataFrame()
            
        LOG.debug(
            "Reservoir DA USACE observation DataFrame creation complete in %s seconds." \
            % (time.time() - start_time)
        )
        
    else:
        reservoir_usace_df = pd.DataFrame()
        
    # create USACE reservoir hybrid DA initial parameters dataframe    
    if reservoir_usace_df.empty == False:
        reservoir_usace_param_df = pd.DataFrame(
            data = 0, 
            index = reservoir_usace_df.index ,
            columns = ['update_time']
        )
        reservoir_usace_param_df['prev_persisted_outflow'] = np.nan
        reservoir_usace_param_df['persistence_update_time'] = 0
        reservoir_usace_param_df['persistence_index'] = 0
    else:
        reservoir_usace_param_df = pd.DataFrame()

    #---------------------------------------------------------------------------
    # Assemble coastal coupling data [WIP]
    coastal_boundary_elev_files = forcing_parameters.get('coastal_boundary_input_file', None) 
    coastal_boundary_domain_files = hybrid_parameters.get('coastal_boundary_domain', None)    

    if coastal_boundary_elev_files:
        start_time = time.time()    
        LOG.info("creating coastal dataframe ...")
        
        coastal_boundary_domain   = nhd_io.read_coastal_boundary_domain(coastal_boundary_domain_files)          
        coastal_boundary_depth_df = nhd_io.build_coastal_ncdf_dataframe(
                                                    coastal_boundary_elev_files,
                                                    coastal_boundary_domain,
                                                    )
                
        LOG.debug(
            "coastal boundary elevation observation DataFrame creation complete in %s seconds." \
            % (time.time() - start_time)
        )            
    else:
        coastal_boundary_depth_df = pd.DataFrame()
 
    #---------------------------------------------------------------------------
    # Trim the time-extent of the streamflow_da usgs_df
    # what happens if there are timeslice files missing on the front-end? 
    # if the first column is some timestamp greater than t0, then this will throw
    # an error. Need to think through this more. 
    if not usgs_df.empty:
        usgs_df = usgs_df.loc[:,t0:]
    
    return qlats_df, usgs_df, reservoir_usgs_df, reservoir_usgs_param_df, reservoir_usace_df, reservoir_usace_param_df, coastal_boundary_depth_df
