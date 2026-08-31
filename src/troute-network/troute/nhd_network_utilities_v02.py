import json
import pathlib
from functools import partial
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np
import netCDF4
from joblib import delayed, Parallel

import troute.nhd_io as nhd_io
import troute.nhd_network as nhd_network

LOG = logging.getLogger("TROUTE")

def build_connections(supernetwork_parameters):
    '''
    Construct network connections network, parameter dataframe, waterbody mapping, 
    and gage mapping. This is an intermediate-level function that calls several 
    lower level functions to read data, conduct network operations, and extract mappings.
    
    Arguments
    ---------
    supernetwork_parameters (dict): User input network parameters
    
    Returns:
    --------
    connections (dict int: [int]): Network connections
    param_df          (DataFrame): Geometry and hydraulic parameters
    wbodies       (dict, int: int): segment-waterbody mapping
    gages         (dict, int: int): segment-gage mapping
    
    '''
    
    # crosswalking dictionary between variables names in input dataset and 
    # variable names recognized by troute.routing module.
    cols = supernetwork_parameters.get(
        'columns', 
        {
        'key'       : 'link',
        'downstream': 'to',
        'dx'        : 'Length',
        'n'         : 'n',
        'ncc'       : 'nCC',
        's0'        : 'So',
        'bw'        : 'BtmWdth',
        'waterbody' : 'NHDWaterbodyComID',
        'gages'     : 'gages',
        'tw'        : 'TopWdth',
        'twcc'      : 'TopWdthCC',
        'alt'       : 'alt',
        'musk'      : 'MusK',
        'musx'      : 'MusX',
        'cs'        : 'ChSlp',
        }
    )
    
    # numeric code used to indicate network terminal segments
    terminal_code = supernetwork_parameters.get("terminal_code", 0)

    # read parameter dataframe 
    param_df = nhd_io.read(pathlib.Path(supernetwork_parameters["geo_file_path"]))

    # select the column names specified in the values in the cols dict variable
    param_df = param_df[list(cols.values())]
    
    # rename dataframe columns to keys in the cols dict variable
    param_df = param_df.rename(columns=nhd_network.reverse_dict(cols))
    
    # handle synthetic waterbody segments
    synthetic_wb_segments = supernetwork_parameters.get("synthetic_wb_segments", None)
    synthetic_wb_id_offset = supernetwork_parameters.get("synthetic_wb_id_offset", 9.99e11)
    if synthetic_wb_segments:
        # rename the current key column to key32
        key32_d = {"key":"key32"}
        param_df = param_df.rename(columns=key32_d)
        # create a key index that is int64
        # copy the links into the new column
        param_df["key"] = param_df.key32.astype("int64")
        # update the values of the synthetic reservoir segments
        fix_idx = param_df.key.isin(set(synthetic_wb_segments))
        param_df.loc[fix_idx,"key"] = (param_df[fix_idx].key + synthetic_wb_id_offset).astype("int64")

    # set parameter dataframe index as segment id number, sort
    param_df = param_df.set_index("key").sort_index()

    # get and apply domain mask
    if "mask_file_path" in supernetwork_parameters:
        data_mask = nhd_io.read_mask(
            pathlib.Path(supernetwork_parameters["mask_file_path"]),
            layer_string=supernetwork_parameters.get("mask_layer_string", None),
        )
        data_mask = data_mask.set_index(data_mask.columns[0])
        param_df = param_df.filter(data_mask.index, axis=0)

    # map segment ids to waterbody ids
    wbodies = {}
    if "waterbody" in cols:
        wbodies = nhd_network.extract_waterbody_connections(
            param_df[["waterbody"]]
        )
        param_df = param_df.drop("waterbody", axis=1)

    # map segment ids to gage ids
    gages = {}
    if "gages" in cols:
        gages = nhd_network.gage_mapping(param_df[["gages"]])
        param_df = param_df.drop("gages", axis=1)
        
    # There can be an externally determined terminal code -- that's this first value
    terminal_codes = set()
    terminal_codes.add(terminal_code)
    # ... but there may also be off-domain nodes that are not explicitly identified
    # but which are terminal (i.e., off-domain) as a result of a mask or some other
    # an interior domain truncation that results in a
    # otherwise valid node value being pointed to, but which is masked out or
    # being intentionally separated into another domain.
    terminal_codes = terminal_codes | set(
        param_df[~param_df["downstream"].isin(param_df.index)]["downstream"].values
    )
    
    # build connections dictionary
    connections = nhd_network.extract_connections(
        param_df, "downstream", terminal_codes=terminal_codes
    )
    param_df = param_df.drop("downstream", axis=1)

    param_df = param_df.astype("float32")

    return connections, param_df, wbodies, gages

def organize_independent_networks(connections, wbody_break_segments, gage_break_segments):
    '''
    Build reverse network connections, independent drainage networks, and network reaches.
    Reaches are defined as linearly connected segments between two junctions or break points.
    
    Arguments:
    ----------
    connections (dict, int: [int]): downstream network connections
    wbody_break_segments     (set): waterbody segments to break reaches at inlets/outlets
    gage_break_segments      (set): gage ids to break reaches at gages
    
    Returns:
    --------
    independent_networks (dict, {int: {int: [int]}}): reverse network connections dictionaries 
                                                      for each indepent network
    reaches_bytw         (dict): list of reaches, by independent network tailwaters
    rconn                (dict): reverse network connections
    
    '''

    # reverse network connections graph - identify upstream adjacents of each segment
    rconn = nhd_network.reverse_network(connections)
    
    # identify independent drainage networks
    independent_networks = nhd_network.reachable_network(rconn)
    
    # construct network reaches
    reaches_bytw = {}
    for tw, net in independent_networks.items():
        
        # break reaches at waterbody inlets/outlets, gages and junctions
        if wbody_break_segments and gage_break_segments:
            
            path_func = partial(
                nhd_network.split_at_gages_waterbodies_and_junctions, 
                gage_break_segments, 
                wbody_break_segments, 
                net
            )
            
        # break reaches at gages and junctions
        elif gage_break_segments and not wbody_break_segments:
            
            path_func = partial(
                nhd_network.split_at_gages_and_junctions, 
                gage_break_segments, 
                net
            )
        
        # break network at waterbody inlets/outlets and junctions
        elif wbody_break_segments and not gage_break_segments:
            
            path_func = partial(
                nhd_network.split_at_waterbodies_and_junctions, 
                wbody_break_segments, 
                net
            )
            
        # break reaches at junctions
        else:
            # Fast path: dfs_decomposition's path_func=None inlines the
            # equivalent of ``partial(nhd_network.split_at_junction, net)``.
            path_func = None

        # construct network reaches with depth-first search alg.
        reaches_bytw[tw] = nhd_network.dfs_decomposition(net, path_func)

    return independent_networks, reaches_bytw, rconn


def build_channel_initial_state(
    restart_parameters, segment_index=pd.Index([])
):

    channel_restart_file = restart_parameters.get("channel_restart_file", None)

    wrf_hydro_channel_restart_file = restart_parameters.get(
        "wrf_hydro_channel_restart_file", None
    )
    
    hyfeature_channel_restart_file = restart_parameters.get(
        "hyfeature_channel_restart_file", None
    )

    if channel_restart_file:
        q0 = nhd_io.get_channel_restart_from_csv(channel_restart_file)
        
    elif wrf_hydro_channel_restart_file:
        q0 = nhd_io.get_channel_restart_from_wrf_hydro(
            restart_parameters["wrf_hydro_channel_restart_file"],
            restart_parameters["wrf_hydro_channel_ID_crosswalk_file"],
            restart_parameters.get("wrf_hydro_channel_ID_crosswalk_file_field_name", 'link'),
            restart_parameters.get("wrf_hydro_channel_restart_upstream_flow_field_name", 'qlink1'),
            restart_parameters.get("wrf_hydro_channel_restart_downstream_flow_field_name", 'qlink2'),
            restart_parameters.get("wrf_hydro_channel_restart_depth_flow_field_name", 'hlink'),
        )
    elif hyfeature_channel_restart_file: 
        # FIXME: fix this cold start when acutal restart file for hyfeature network rolls in.
        q0 = pd.DataFrame(
            0, index=segment_index, columns=["qu0", "qd0", "h0"], dtype="float32",
        )
    else:
        # Set cold initial state
        # assume to be zero
        # 0, index=connections.keys(), columns=["qu0", "qd0", "h0",], dtype="float32"
        q0 = pd.DataFrame(
            0, index=segment_index, columns=["qu0", "qd0", "h0"], dtype="float32",
        )
  
    # TODO: If needed for performance improvement consider filtering mask file on read.
    if not segment_index.empty:
        q0 = q0[q0.index.isin(segment_index)]

    return q0


def build_forcing_sets(
    forcing_parameters,
    t0
):
    
    run_sets = forcing_parameters.get("qlat_forcing_sets", None)
    qlat_input_folder = forcing_parameters.get("qlat_input_folder", None)
    nts = forcing_parameters.get("nts", None)
    max_loop_size = forcing_parameters.get("max_loop_size", 12)
    dt = forcing_parameters.get("dt", None)
    
    try:
        qlat_input_folder = pathlib.Path(qlat_input_folder)
        assert qlat_input_folder.is_dir() == True
    except TypeError:
        raise TypeError("Aborting simulation because no qlat_input_folder is specified in the forcing_parameters section of the .yaml control file.") from None
    except AssertionError:
        raise AssertionError("Aborting simulation because the qlat_input_folder:", qlat_input_folder,"does not exist. Please check the the qlat_input_folder variable is correctly entered in the .yaml control file") from None

    forcing_glob_filter = forcing_parameters.get("qlat_file_pattern_filter", "*.CHRTOUT_DOMAIN1")
        
    # TODO: Throw errors if insufficient input data are available
    if run_sets:
        
        # append final_timestamp variable to each set_list
        qlat_input_folder = pathlib.Path(qlat_input_folder)
        for (s, _) in enumerate(run_sets):
            final_chrtout = qlat_input_folder.joinpath(run_sets[s]['qlat_files'
                    ][-1])
            final_timestamp_str = nhd_io.get_param_str(final_chrtout,
                    'model_output_valid_time')
            run_sets[s]['final_timestamp'] = \
                datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')
            
    elif qlat_input_folder:
        
        # Construct run_set dictionary from user-specified parameters

        # get the first and seconded files from an ordered list of all forcing files
        qlat_input_folder = pathlib.Path(qlat_input_folder)
        all_files = sorted(qlat_input_folder.glob(forcing_glob_filter))
        first_file = all_files[0]
        second_file = all_files[1]

        # Deduce the timeinterval of the forcing data from the output timestamps of the first
        # two ordered CHRTOUT files
        t1 = nhd_io.get_param_str(first_file, "model_output_valid_time")
        t1 = datetime.strptime(t1, "%Y-%m-%d_%H:%M:%S")
        t2 = nhd_io.get_param_str(second_file, "model_output_valid_time")
        t2 = datetime.strptime(t2, "%Y-%m-%d_%H:%M:%S")
        dt_qlat_timedelta = t2 - t1
        dt_qlat = dt_qlat_timedelta.seconds

        # determine qts_subdivisions
        qts_subdivisions = dt_qlat / dt
        if dt_qlat % dt == 0:
            qts_subdivisions = dt_qlat / dt

        # the number of files required for the simulation
        nfiles = int(np.ceil(nts / qts_subdivisions))

        # list of forcing file datetimes
        datetime_list = [t0 + dt_qlat_timedelta * (n + 1) for n in
                         range(nfiles)]
        datetime_list_str = [datetime.strftime(d, '%Y%m%d%H%M') for d in
                             datetime_list]

        # list of forcing files
        forcing_filename_list = [d_str + ".CHRTOUT_DOMAIN1" for d_str in
                                 datetime_list_str]
        
        # check that all forcing files exist
        for f in forcing_filename_list:
            try:
                J = pathlib.Path(qlat_input_folder.joinpath(f))     
                assert J.is_file() == True
            except AssertionError:
                raise AssertionError("Aborting simulation because forcing file", J, "cannot be not found.") from None
 
        # build run sets list
        run_sets = []
        k = 0
        j = 0
        nts_accum = 0
        nts_last = 0
        while k < len(forcing_filename_list):
            run_sets.append({})

            if k + max_loop_size < len(forcing_filename_list):
                run_sets[j]['qlat_files'] = forcing_filename_list[k:k
                    + max_loop_size]
            else:
                run_sets[j]['qlat_files'] = forcing_filename_list[k:]

            nts_accum += len(run_sets[j]['qlat_files']) * qts_subdivisions
            if nts_accum <= nts:
                run_sets[j]['nts'] = int(len(run_sets[j]['qlat_files'])
                                         * qts_subdivisions)
            else:
                run_sets[j]['nts'] = int(nts - nts_last)

            final_chrtout = qlat_input_folder.joinpath(run_sets[j]['qlat_files'
                    ][-1])
            final_timestamp_str = nhd_io.get_param_str(final_chrtout,
                    'model_output_valid_time')
            run_sets[j]['final_timestamp'] = \
                datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')

            nts_last = nts_accum
            k += max_loop_size
            j += 1
    return run_sets

def build_qlateral_array(
    forcing_parameters,
    cpu_pool,
    segment_index=pd.Index([]),
    ts_iterator=None,
    file_run_size=None,
):
    # TODO: set default/optional arguments

    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    nts = forcing_parameters.get("nts", 1)
    qlat_input_folder = forcing_parameters.get("qlat_input_folder", None)
    qlat_input_file = forcing_parameters.get("qlat_input_file", None)
    if qlat_input_folder:
        qlat_input_folder = pathlib.Path(qlat_input_folder)
        if "qlat_files" in forcing_parameters:
            qlat_files = forcing_parameters.get("qlat_files")
            qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
        elif "qlat_file_pattern_filter" in forcing_parameters:
            qlat_file_pattern_filter = forcing_parameters.get(
                "qlat_file_pattern_filter", "*CHRT_OUT*"
            )
            qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))

        qlat_file_index_col = forcing_parameters.get(
            "qlat_file_index_col", "feature_id"
        )
        qlat_file_value_col = forcing_parameters.get("qlat_file_value_col", "q_lateral")
        gw_bucket_col = forcing_parameters.get("qlat_file_gw_bucket_flux_col","qBucket")
        terrain_ro_col = forcing_parameters.get("qlat_file_terrain_runoff_col","qSfcLatRunoff")

        # Parallel reading of qlateral data from CHRTOUT
        with Parallel(n_jobs=cpu_pool) as parallel:

            jobs = []
            for f in qlat_files:
                jobs.append(
                    delayed(nhd_io.get_ql_from_chrtout)
                    (f, qlat_file_value_col, gw_bucket_col, terrain_ro_col)
                )
            ql_list = parallel(jobs)

        # get feature_id from a single CHRTOUT file
        with netCDF4.Dataset(qlat_files[0]) as ds:
            idx = ds.variables[qlat_file_index_col][:].filled()

        # package data into a DataFrame
        qlat_df = pd.DataFrame(
            np.stack(ql_list).T,
            index = idx,
            columns = range(len(qlat_files))
        )    
        qlat_df = qlat_df[qlat_df.index.isin(segment_index)]

    elif qlat_input_file:
        qlat_df = nhd_io.get_ql_from_csv(qlat_input_file)

    else:
        qlat_const = forcing_parameters.get("qlat_const", 0)
        qlat_df = pd.DataFrame(
            qlat_const,
            index=segment_index,
            columns=range(nts // qts_subdivisions),
            dtype="float32",
        )

    # TODO: Make a more sophisticated date-based filter
    max_col = 1 + nts // qts_subdivisions
    if len(qlat_df.columns) > max_col:
        qlat_df.drop(qlat_df.columns[max_col:], axis=1, inplace=True)

    if not segment_index.empty:
        qlat_df = qlat_df[qlat_df.index.isin(segment_index)]

    return qlat_df


def build_parity_sets(parity_parameters, run_sets):
    
    parity_sets = parity_parameters.get("parity_check_compare_file_sets", None)
    
    if parity_sets:
        pass
    
    else:
    
        parity_sets = []
        for (i, set_dict) in enumerate(run_sets):
            parity_sets.append({})
            parity_sets[i]['validation_files'] = run_sets[i]['qlat_files']
    
    return parity_sets

def _check_timeslice_exists(filenames, timeslices_folder):
    """
    Check that each TimeSlice file in a list of files exists. Return a list of 
    available files.
    
    Arguments
    ---------
    - filenames               (list of str): TimeSlice filenames
    - timeslices_folder (pathlib.PosixPath): TimeSlice directory
    
    Returns
    -------
    - filenames_existing (list of chr): Existing TimeSlice filenames in 
                                        TimeSlice directory
    
    Notes
    -----
    - This is a utility function used by build_da_sets
    
    """
    
    # check that all USGS TimeSlice files in the set actually exist
    drop_list = []
    for f in filenames:
        try:
            J = pathlib.Path(timeslices_folder.joinpath(f))     
            assert J.is_file() == True
        except AssertionError:
            LOG.warning("Missing TimeSlice file %s", J)
            drop_list.append(f)

    # Assemble a list of existing TimeSlice files, only
    filenames_existing = [x for x in filenames if x not in drop_list]   
    
    return filenames_existing

def build_da_sets(da_params, run_sets, t0):
    """
    Create set lists of USGS and/or USACE TimeSlice files used for 
    streamflow and reservoir data assimilation
    
    Arguments
    --------
    - da_params (dict): user-input data assimilation parameters
    - run_sets (list) : forcing files for each run set in the simlation
    - t0 (datetime)   : model initialization time
    
    Returns
    -------
    - da_sets (list)  : lists of USGS and USACE TimeSlice files for each run set 
                        in the simulation
    
    Notes
    -----
    
    """
    
    # check for user-input usgs and usace timeslice directories
    usgs_timeslices_folder = da_params.get(
        "usgs_timeslices_folder",
        None
    )
    usace_timeslices_folder = da_params.get(
        "usace_timeslices_folder",
        None
    )

    # User-specified DA ON/OFF preferences
    usace_da = False
    usgs_da = False
    reservoir_da = da_params.get('reservoir_da', False)
    if reservoir_da:
        usgs_da = reservoir_da.get('reservoir_persistence_usgs', False)
        usace_da = reservoir_da.get('reservoir_persistence_usace', False)
    
    nudging = False
    streamflow_da = da_params.get('streamflow_da', False)
    if streamflow_da:
        nudging = streamflow_da.get('streamflow_nudging', False)
        
    if not usgs_da and not usace_da and not nudging:
        # if all DA capabilities are OFF, return empty dictionary
        da_sets = [{} for _ in run_sets]
    
    # if no user-input timeslice folders, a list of empty dictionaries
    elif not usgs_timeslices_folder and not usace_timeslices_folder:
        # if no timeslice folders, return empty dictionary
        da_sets = [{} for _ in run_sets]
        
    # if user-input timeslice folders are present, build TimeSlice sets
    else:
        
        # create Path objects for each TimeSlice directory
        if usgs_timeslices_folder:
            usgs_timeslices_folder = pathlib.Path(usgs_timeslices_folder)
        if usace_timeslices_folder:
            usace_timeslices_folder = pathlib.Path(usace_timeslices_folder)
        
        # the number of timeslice files appended to the front- and back-ends
        # of the TimeSlice file interpolation stack
        pad_hours = da_params.get("timeslice_lookback_hours",0)
        timeslice_pad = pad_hours * 4 # number of 15 minute TimeSlices in the pad

        # timedelta of TimeSlice data - typically 15 minutes
        dt_timeslice = timedelta(minutes = 15)

        da_sets = [] # initialize list to store TimeSlice set lists

        # Loop through each run set and build lists of available TimeSlice files
        for (i, set_dict) in enumerate(run_sets):
            
            # Append an empty dictionary to the loop, which be used to hold
            # lists of USGS and USACE TimeSlice files.
            da_sets.append({})

            # timestamps of TimeSlice files desired for run set i
            timestamps = pd.date_range(
                t0 - dt_timeslice * timeslice_pad,
                run_sets[i]['final_timestamp'] + dt_timeslice * 4,
                freq=dt_timeslice
            )

            # identify available USGS TimeSlices in run set i
            if (usgs_timeslices_folder and nudging) or (usgs_timeslices_folder and usgs_da):
                filenames_usgs = (timestamps.strftime('%Y-%m-%d_%H:%M:%S') 
                            + '.15min.usgsTimeSlice.ncdf').to_list()
                
                # identify available USGS TimeSlices
                filenames_usgs = _check_timeslice_exists(
                    filenames_usgs, 
                    usgs_timeslices_folder
                )
                
                # Add available TimeSlices to da_sets list
                da_sets[i]['usgs_timeslice_files'] = filenames_usgs
                
            # identify available USACE TimeSlices in run set i
            if usace_timeslices_folder and usace_da:
                filenames_usace = (timestamps.strftime('%Y-%m-%d_%H:%M:%S') 
                            + '.15min.usaceTimeSlice.ncdf').to_list()
                
                # identify available USACE TimeSlices
                filenames_usace = _check_timeslice_exists(
                    filenames_usace, 
                    usace_timeslices_folder
                )
                
                # Add available TimeSlices to da_sets list
                da_sets[i]['usace_timeslice_files'] = filenames_usace

            # reset initialization time for loop set i+1
            t0 = run_sets[i]['final_timestamp']
            
    return da_sets
    
def build_data_assimilation(data_assimilation_parameters, run_parameters):
    lastobs_df, da_parameter_dict = build_data_assimilation_lastobs(data_assimilation_parameters)
    usgs_df = build_data_assimilation_usgs_df(data_assimilation_parameters, run_parameters, lastobs_df.index)
    return usgs_df, lastobs_df, da_parameter_dict

'''
def build_streamflow_da_data(
    streamflow_da_parameters,
    run_parameters,
    lastobs_index=None,
):
    """
    Construct DataFrame of USGS gage observations for streamflow DA.
    
    Arguments
    ---------
    - streamflow_da_parameters (dict): Parameters controlling DA data assembly
    - run_parameters           (dict): Simulation run parameters
    - lastobs_index    (Pandas Index): ????
    
    Returns
    -------
    - usgs_df (DataFrame): qa/qc'd and interpolated USGS gage observations from 
                           USGS TimeSlice files for streamflow DA
    
    Notes
    -----
    
    """

    # directory containing USGS TimeSlice files
    usgs_timeslices_folder = streamflow_da_parameters.get(
        "usgs_timeslices_folder", None
    )

    # initialize empty dataframe to contain gage observations
    usgs_df = pd.DataFrame()

    if usgs_timeslices_folder:
        
        # convert timeslices_folder from str to PosixPath
        usgs_timeslices_folder = pathlib.Path(
            usgs_timeslices_folder
        )
        
        if "usgs_timeslice_files" in streamflow_da_parameters:
                
            usgs_files = streamflow_da_parameters.get("usgs_timeslice_files", None)
            usgs_files = [usgs_timeslices_folder.joinpath(f) for f in usgs_files]
            
            if usgs_files: 

                usgs_df = nhd_io.get_obs_from_timeslices(
                    streamflow_da_parameters["crosswalk_file"],
                    streamflow_da_parameters["crosswalk_gage_field"],
                    streamflow_da_parameters["crosswalk_segID_field"],
                    usgs_files,
                    streamflow_da_parameters["qc_threshold"],
                    streamflow_da_parameters["interpolation_limit"],
                    run_parameters["dt"],
                    run_parameters["t0"],
                )
        
    if not isinstance(lastobs_index, pd.Index):
        lastobs_index = pd.Index()
        
    if not lastobs_index.empty:
        if not usgs_df.empty and not usgs_df.index.equals(lastobs_index):
            LOG.warning("USGS Dataframe Index Does Not Match Last Observations Dataframe Index")
            usgs_df = usgs_df.loc[lastobs_index]

    return usgs_df
'''

def build_data_assimilation_lastobs(data_assimilation_parameters):
    '''
    A middle man function that assembles inputs for and calls
    nhd_io.build_lastobs_df, which constructs the lastobs dataframe used for 
    streamflow data assimilation. 
    
    Also, this function creates a dictionary of data assimilation parameters
    that gets passed down to the compute kernels. 
    
    Arguments
    ---------
    - data_assimilation_parameters (dict): user-input data assimilation parameters
    
    Returns
    -------
    - lastobs_df (Pandas DataFrame):
    
    - da_parameter_dict      (dict):
    
    Notes
    -----
    - TODO: package additional parameters into da_parameter_dict
    '''
    
    # TODO: Fix the Logic here according to the following:
    # If there are any observations for data assimilation, there
    # needs to be a complete set in the first time set or else
    # there must be a "LastObs". If there is a complete set in
    # the first time step, the LastObs is optional. If there are
    # no observations for assimilation, there can be a LastObs
    # with an empty usgs dataframe.

    lastobs_df = pd.DataFrame()
    
    streamflow_da_parameters = data_assimilation_parameters.get(
        'streamflow_da',
        None
    )
    
    if streamflow_da_parameters:
        
        # determine if user explictly requests streamflow DA
        nudging = streamflow_da_parameters.get(
            'streamflow_nudging', 
            False
        )
            
        if nudging:
            
            lastobs_file = streamflow_da_parameters.get(
                "wrf_hydro_lastobs_file",
                None
            )
            
            lastobs_start = streamflow_da_parameters.get(
                "wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time",
                0
            )
            
            lastobs_type = streamflow_da_parameters.get(
                "wrf_lastobs_type", 
                "error-based"
            )
        
            lastobs_crosswalk_file = streamflow_da_parameters.get(
                "gage_segID_crosswalk_file",
                None
            )

            if lastobs_file:
                lastobs_df = nhd_io.build_lastobs_df(
                    lastobs_file,
                    lastobs_crosswalk_file,
                    lastobs_start,
                )

    da_parameter_dict = {}
    da_parameter_dict["da_decay_coefficient"] = data_assimilation_parameters.get(
        "da_decay_coefficient", 
        120)
    da_parameter_dict["diffusive_streamflow_nudging"] = streamflow_da_parameters.get(
        "diffusive_streamflow_nudging", False) 
    
    # TODO: Add parameters here for interpolation length (14/59), QC threshold (1.0)

    return lastobs_df, da_parameter_dict


def build_data_assimilation_csv(data_assimilation_parameters):

    usgs_df = nhd_io.get_usgs_df_from_csv(
        data_assimilation_parameters["data_assimilation_csv"],
        data_assimilation_parameters["wrf_hydro_da_channel_ID_crosswalk_file"],
    )

    return usgs_df


def build_data_assimilation_folder(data_assimilation_parameters, run_parameters):

    usgs_timeslices_folder = pathlib.Path(
        data_assimilation_parameters["data_assimilation_timeslices_folder"],
    ).resolve()
    if "data_assimilation_filter" in data_assimilation_parameters:
        da_glob_filter = data_assimilation_parameters["data_assimilation_filter"]
        usgs_files = sorted(usgs_timeslices_folder.glob(da_glob_filter))
    elif "usgs_timeslice_files" in data_assimilation_parameters:
        usgs_files = data_assimilation_parameters.get("usgs_timeslice_files", None)
        usgs_files = [usgs_timeslices_folder.joinpath(f) for f in usgs_files]
    else:
        LOG.warning("No Files Found for DA")
        # TODO: Handle this with a real exception

    if usgs_files:
        usgs_df = nhd_io.get_obs_from_timeslices(
            data_assimilation_parameters["wrf_hydro_da_channel_ID_crosswalk_file"],
            usgs_files,
            data_assimilation_parameters.get("qc_threshold", 1),
            data_assimilation_parameters.get("data_assimilation_interpolation_limit", 59),
            run_parameters["dt"],
            run_parameters["t0"],
        )
    else:
        usgs_df = pd.DataFrame()

    return usgs_df


def build_refac_connections(diff_network_parameters):
    '''
    Construct network connections network for refacored dataset. 
    
    Arguments
    ---------
    diff_network_parameters (dict): User input network parameters
    
    Returns:
    --------
    connections (dict int: [int]): Network connections
    '''

    # crosswalking dictionary between variables names in input dataset and 
    # variable names recognized by troute.routing module.
    cols = diff_network_parameters.get(
        'columns', 
        {
        'key'       : 'link',
        'downstream': 'to',
        'dx'        : 'Length',
        'n'         : 'n',
        'waterbody' : 'NHDWaterbodyComID',
        'gages'     : 'gages',
        'alt'      : 'z',
        'line_d'    : 'xid_d',        
        }
    )

    # read parameter dataframe 
    param_df = nhd_io.read(pathlib.Path(diff_network_parameters["geo_file_path"]))

    # numeric code used to indicate network terminal segments
    terminal_code = set(param_df.to.unique()) - set(param_df.link.unique())

    # select the column names specified in the values in the cols dict variable
    param_df = param_df[list(cols.values())]

    # rename dataframe columns to keys in the cols dict variable
    param_df = param_df.rename(columns=nhd_network.reverse_dict(cols))

    # set parameter dataframe index as segment id number, sort
    param_df = param_df.set_index("key").sort_index()  
    
    # There can be an externally determined terminal code -- that's this first value
    terminal_codes = set()
    terminal_codes.update(terminal_code)
    # ... but there may also be off-domain nodes that are not explicitly identified
    # but which are terminal (i.e., off-domain) as a result of a mask or some other
    # an interior domain truncation that results in a
    # otherwise valid node value being pointed to, but which is masked out or
    # being intentionally separated into another domain.
    terminal_codes = terminal_codes | set(
        param_df[~param_df["downstream"].isin(param_df.index)]["downstream"].values
    )

    param_df_unique = param_df.drop_duplicates("downstream")
    # build connections dictionary
    connections = nhd_network.extract_connections(
        param_df_unique, "downstream", terminal_codes=terminal_codes
    )

    return connections
