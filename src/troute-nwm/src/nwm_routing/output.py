import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import troute.nhd_io as nhd_io
from build_tests import parity_check
import logging
from troute.nhd_io import updated_flowveldepth

LOG = logging.getLogger('')

def _reindex_lake_to_link_id(target_df, crosswalk):
    '''
    Utility function for replacing lake ID index values
    with link ID values in a dataframe. This is used to 
    reinedex results dataframes
    
    Arguments:
    ----------
    - target_df (DataFrame): Data frame to be reinexed
    - crosswalk      (dict): Relates lake ids to outlet link ids
    
    Returns:
    --------
    - target_df (DataFrame): Re-indexed with link ids replacing 
                             lake ids
    '''

    # evaluate intersection of lake ids and target_df index values
    # i.e. what are the index positions of lake ids that need replacing?
    lakeids = np.fromiter(crosswalk.keys(), dtype = int)
    idxs = target_df.index.to_numpy()
    lake_index_intersect = np.intersect1d(
        idxs, 
        lakeids, 
        return_indices = True
    )

    # replace lake ids with link IDs in the target_df index array
    linkids = np.fromiter(crosswalk.values(), dtype = int)
    idxs[lake_index_intersect[1]] = linkids[lake_index_intersect[2]]

    # (re) set the target_df index
    target_df.set_index(idxs, inplace = True)

    return target_df


def _parquet_output_format_converter(df, start_datetime, dt, configuration, prefix_ids, nexus_dict):
    '''
    Utility function for convert flowveldepth dataframe
    to a timeseries and to match parquet input format
    of TEEHR

    Arguments:
    ----------
    - df (DataFrame): Data frame to be converted
    - start_datetime: Date time from restart parameters
    - dt: Time step
    - configuration: configuration (for instance- short_range, medium_range)

    Returns:
    --------
    - timeseries_df (DataFrame): Converted timeseries data frame
    '''
    nex_id = {}
    if prefix_ids == 'nex' and nexus_dict:
        for key, val in nexus_dict.items():
            nex_key = int(key.split('-')[-1])
            nex_id[nex_key] = [int(v.split('-')[-1]) for v in val] 
    
    df = updated_flowveldepth(df, nex_id, seg_id = list(), mask_list = None)
    df = df.reset_index().drop('Type', axis=1).set_index('featureID')
    variable_to_name_map = {"q": "streamflow", "d": "depth", "v": "velocity"}
    variable_to_units_map = {"streamflow": "m3/s", "velocity": "m/s", "depth": "m"}

    # Prepare the location_id with prefix
    df.index.name = 'location_id'
    df.reset_index(inplace=True)
    if nexus_dict:
        location_ids = prefix_ids + '-' + df['location_id'].astype(str)
    else:
        location_ids = df['location_id'].astype(str)    

    # Flatten the dataframe using NumPy
    num_locations = df.shape[0]
    num_time_variables = df.shape[1] - 1
    num_records = num_locations * num_time_variables

    # Prepare timestep and variable arrays
    times = df.columns[1:]
    timesteps = np.array([t[0] for t in times], dtype=int)
    variables = np.array([t[1] for t in times])

    # Preallocate arrays
    location_ids_repeated = np.tile(location_ids, num_time_variables)
    value_time = np.empty(num_records, dtype='datetime64[us]')
    variable_names = np.empty(num_records, dtype=object)
    units = np.empty(num_records, dtype=object)
    values = np.empty(num_records, dtype=float)

    # Calculate value_time, variable_names, units, and values in a vectorized manner
    for i in range(num_time_variables):
        start_idx = i * num_locations
        end_idx = start_idx + num_locations
        value_time[start_idx:end_idx] = start_datetime + pd.to_timedelta((timesteps[i] + 1) * dt, unit='s')
        variable_name = variable_to_name_map[variables[i]]
        unit = variable_to_units_map[variable_name]
        variable_names[start_idx:end_idx] = variable_name
        units[start_idx:end_idx] = unit
        values[start_idx:end_idx] = df.iloc[:, i + 1].values

    # Create the resulting DataFrame
    timeseries_df = pd.DataFrame({
        'location_id': location_ids_repeated,
        'value': values,
        'value_time': value_time,
        'variable_name': variable_names,
        'units': units,
        'reference_time': start_datetime.date(),
        'configuration': configuration
    })
    
    return timeseries_df


def nwm_output_generator(
    run,
    results,
    supernetwork_parameters,
    output_parameters,
    parity_parameters,
    restart_parameters,
    parity_set,
    qts_subdivisions,
    return_courant,
    cpu_pool,
    waterbodies_df,
    waterbody_types_df,
    duplicate_ids_df,
    data_assimilation_parameters=False,
    lastobs_df = None,
    link_gage_df = None,
    link_lake_crosswalk = None,
    nexus_dict = None,
    poi_crosswalk = None,
    logFileName='NONE' 
):
  
    dt = run.get("dt")
    nts = run.get("nts")
    t0 = run.get("t0")

    if parity_parameters:
        parity_check_waterbody_file = parity_parameters.get("parity_check_waterbody_file", None)
        parity_check_file = parity_parameters.get("parity_check_file", None)
        parity_check_input_folder = parity_parameters.get("parity_check_input_folder", None)
        parity_check_file_index_col = parity_parameters.get(
            "parity_check_file_index_col", None
        )
        parity_check_file_value_col = parity_parameters.get(
            "parity_check_file_value_col", None
        )
        parity_check_compare_node = parity_parameters.get("parity_check_compare_node", None)

        # TODO: find a better way to deal with these defaults and overrides.
        parity_set["parity_check_waterbody_file"] = parity_set.get(
            "parity_check_waterbody_file", parity_check_waterbody_file
        )
        parity_set["parity_check_file"] = parity_set.get(
            "parity_check_file", parity_check_file
        )
        parity_set["parity_check_input_folder"] = parity_set.get(
            "parity_check_input_folder", parity_check_input_folder
        )
        parity_set["parity_check_file_index_col"] = parity_set.get(
            "parity_check_file_index_col", parity_check_file_index_col
        )
        parity_set["parity_check_file_value_col"] = parity_set.get(
            "parity_check_file_value_col", parity_check_file_value_col
        )
        parity_set["parity_check_compare_node"] = parity_set.get(
            "parity_check_compare_node", parity_check_compare_node
        )

    ################### Output Handling
    
    start_time = time.time()

    LOG.info(f"Handling output ...")

    csv_output = output_parameters.get("csv_output", None)
    csv_output_folder = None
    parquet_output = output_parameters.get("parquet_output", None)
    parquet_output_folder = None
    rsrto = output_parameters.get("hydro_rst_output", None)
    chrto = output_parameters.get("chrtout_output", None)
    test = output_parameters.get("test_output", None)
    chano = output_parameters.get("chanobs_output", None)
    wbdyo = output_parameters.get("lakeout_output", None)
    stream_output = output_parameters.get("stream_output", None)
    lastobso = output_parameters.get("lastobs_output", None)

    if csv_output:
        csv_output_folder = output_parameters["csv_output"].get(
            "csv_output_folder", None
        )
        csv_output_segments = csv_output.get("csv_output_segments", None)

    if parquet_output:
        parquet_output_folder = output_parameters["parquet_output"].get(
            "parquet_output_folder", None
        )
        parquet_output_segments = parquet_output.get("parquet_output_segments", None)

    if csv_output_folder or parquet_output_folder or rsrto or chrto or chano or test or wbdyo or stream_output:

        start = time.time()
        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()

        flowveldepth = pd.concat(
            [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
            copy=False,
        )

        if wbdyo and not waterbodies_df.empty:
            
            # create waterbody dataframe for output to netcdf file
            i_columns = pd.MultiIndex.from_product(
                [range(nts), ["i"]]
            ).to_flat_index()

            wbdy = pd.concat(
                [pd.DataFrame(r[6], index=r[0], columns=i_columns) for r in results],
                copy=False,
            )

            wbdy_id_list = waterbodies_df.index.values.tolist()
            flow_df = flowveldepth.loc[wbdy_id_list]
            wbdy = wbdy.loc[wbdy_id_list]
            
            # Replace synthetic waterbody IDs (made from duplicate IDs) with
            # original waterbody IDs (if duplicates exist):
            if not duplicate_ids_df.empty:
                flow_df = flow_df.rename(index=dict(duplicate_ids_df[['synthetic_ids','lake_id']].values))
                wbdy = wbdy.rename(index=dict(duplicate_ids_df[['synthetic_ids','lake_id']].values))

            timestep, variable = zip(*flow_df.columns.tolist())
            timestep_index = np.where(((np.array(list(set(list(timestep)))) + 1) * dt) % (dt * qts_subdivisions) == 0)
            ts_set = set(timestep_index[0].tolist())
            flow_df_col_index = [i for i, e in enumerate(timestep) if e in ts_set]
            flow_df = flow_df.iloc[:,flow_df_col_index]

            timestep, variable = zip(*wbdy.columns.tolist())
            timestep_index = np.where(((np.array(list(set(list(timestep)))) + 1) * dt) % (dt * qts_subdivisions) == 0)
            ts_set = set(timestep_index[0].tolist())
            wbdy_col_index = [i for i, e in enumerate(timestep) if e in ts_set]
            i_df = wbdy.iloc[:,wbdy_col_index]
            q_df = flow_df.iloc[:,0::3]
            d_df = flow_df.iloc[:,2::3]

        # replace waterbody lake_ids with outlet link ids
        if (link_lake_crosswalk):
            flowveldepth = _reindex_lake_to_link_id(flowveldepth, link_lake_crosswalk)
            
        # todo: create a unit test by saving FVD array to disk and then checking that
        # it matches FVD array from parent branch or other configurations. 
        # flowveldepth.to_pickle(output_parameters['test_output'])

        if return_courant:
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
            
            # replace waterbody lake_ids with outlet link ids
            if link_lake_crosswalk:
                # (re) set the flowveldepth index
                courant.set_index(fvdidxs, inplace = True)
            
        LOG.debug("Constructing the FVD DataFrame took %s seconds." % (time.time() - start))
    
    if stream_output:
        stream_output_directory = stream_output['stream_output_directory']
        stream_output_mask = stream_output.get('mask_output',)
        stream_output_timediff = stream_output['stream_output_time']
        stream_output_type = stream_output['stream_output_type']
        stream_output_internal_frequency = stream_output['stream_output_internal_frequency']
        if stream_output_mask:
            stream_output_mask = Path(stream_output_mask)
        
        nudge = np.concatenate([r[8] for r in results])
        usgs_positions_id = np.concatenate([r[3][0] for r in results])
        nhd_io.write_flowveldepth(
            Path(stream_output_directory),
            stream_output_mask, 
            flowveldepth, 
            nudge, 
            usgs_positions_id, 
            t0, 
            dt,
            int(stream_output_timediff), 
            stream_output_type,
            stream_output_internal_frequency,
            cpu_pool = cpu_pool,
            poi_crosswalk = poi_crosswalk,
            nexus_dict= nexus_dict,
            )

        if (not logFileName == 'NONE'):
            with open(logFileName, 'a') as preRunLog:
                preRunLog.write("\n") 
                preRunLog.write("-----\n") 
                preRunLog.write("Output of flow velocity depth files into folder: "+str(Path(stream_output_directory))+"\n") 
                preRunLog.write("-----\n") 
                nTimeBins = int( len(flowveldepth.columns)/3)
                fCalc = int(dt/60)
                preRunLog.write("Internal computation of FVD data every "+str(stream_output_internal_frequency)+" minutes\n")
                preRunLog.write("Output of FVD data every "+str(fCalc)+" minutes\n")
                preRunLog.write("Writing "+str(nTimeBins)+" time bins for "+str(len(flowveldepth.index))+" segments per FVD output file\n")               
            preRunLog.close()      

    if test:
        flowveldepth.to_pickle(Path(test))
    
    if wbdyo and not waterbodies_df.empty:
        
        time_index, tmp_variable = map(list,zip(*i_df.columns.tolist()))
        if not duplicate_ids_df.empty:
            output_waterbodies_df = waterbodies_df.rename(index=dict(duplicate_ids_df[['synthetic_ids','lake_id']].values))
            output_waterbody_types_df = waterbody_types_df.rename(index=dict(duplicate_ids_df[['synthetic_ids','lake_id']].values))
        else:
            output_waterbodies_df = waterbodies_df
            output_waterbody_types_df = waterbody_types_df
        LOG.info("- writing t-route flow results to LAKEOUT files")
        start = time.time()
        for i in range(i_df.shape[1]):              
            nhd_io.write_waterbody_netcdf(
                wbdyo, 
                i_df.iloc[:,[i]],
                q_df.iloc[:,[i]],
                d_df.iloc[:,[i]],
                output_waterbodies_df,
                output_waterbody_types_df,
                t0, 
                dt, 
                nts,
                time_index[i],
            )
        
        if (not logFileName == 'NONE'):
            with open(logFileName, 'a') as preRunLog:
                preRunLog.write("\n") 
                preRunLog.write("-----\n") 
                preRunLog.write("Output of waterbody files into folder: "+str(wbdyo)+"\n") 
                preRunLog.write("-----\n") 
            preRunLog.close()  
        
        LOG.debug("writing LAKEOUT files took a total time of %s seconds." % (time.time() - start))
    
    if rsrto:

        LOG.info("- writing restart files")
        start = time.time()
        
        wrf_hydro_restart_dir = rsrto.get(
            "wrf_hydro_channel_restart_source_directory", None
        )

        if wrf_hydro_restart_dir:

            restart_pattern_filter = rsrto.get("wrf_hydro_channel_restart_pattern_filter", "HYDRO_RST.*")
            # list of WRF Hydro restart files
            wrf_hydro_restart_files = sorted(
                Path(wrf_hydro_restart_dir).glob(
                    restart_pattern_filter
                )
            )

            if len(wrf_hydro_restart_files) > 0:
                nhd_io.write_hydro_rst(
                    flowveldepth,
                    wrf_hydro_restart_files,
                    restart_parameters.get("wrf_hydro_channel_restart_file"),
                    dt,
                    nts,
                    t0,
                    restart_parameters.get("wrf_hydro_channel_ID_crosswalk_file"),
                    restart_parameters.get(
                        "wrf_hydro_channel_ID_crosswalk_file_field_name"
                    ),
                )

                if (not logFileName == 'NONE'):
                    with open(logFileName, 'a') as preRunLog:
                        preRunLog.write("\n") 
                        preRunLog.write("-----\n") 
                        preRunLog.write("Output of wrf hydro restart files into folder: "+str(Path(wrf_hydro_restart_dir))+"\n") 
                        preRunLog.write("-----\n") 
                    preRunLog.close()  

            else:
                LOG.critical('Did not find any restart files in wrf_hydro_channel_restart_source_directory. Aborting restart write sequence.')

        else:
            LOG.critical('wrf_hydro_channel_restart_source_directory not specified in configuration file. Aborting restart write sequence.')

        LOG.debug("writing restart files took %s seconds." % (time.time() - start))

    if chrto:

        LOG.info("- writing t-route flow results to CHRTOUT files")
        start = time.time()
        
        chrtout_read_folder = chrto.get(
            "wrf_hydro_channel_output_source_folder", None
        )

        if chrtout_read_folder:
            
            chrtout_files = sorted(
                Path(chrtout_read_folder) / f for f in run["qlat_files"]
            )

            nhd_io.write_chrtout(
                flowveldepth,
                chrtout_files,
                qts_subdivisions,
                cpu_pool,
            )
        
            if (not logFileName == 'NONE'):
                with open(logFileName, 'a') as preRunLog:
                    preRunLog.write("\n") 
                    preRunLog.write("-----\n") 
                    preRunLog.write("Output of FVD files into folder: "+str(chrtout_read_folder)+"\n") 
                    preRunLog.write("-----\n") 
                preRunLog.close()  

        LOG.debug("writing CHRTOUT files took a total time of %s seconds." % (time.time() - start))

    if csv_output_folder: 
    
        LOG.info("- writing flow, velocity, and depth results to .csv")
        start = time.time()

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

        # no csv_output_segments are specified, then write results for all segments
        if not csv_output_segments:
            csv_output_segments = flowveldepth.index
        
        flowveldepth = flowveldepth.sort_index()
        flowveldepth.loc[csv_output_segments].to_csv(output_path.joinpath(filename_fvd))

        if return_courant:
            courant = courant.sort_index()
            courant.loc[csv_output_segments].to_csv(output_path.joinpath(filename_courant))

        LOG.debug("writing CSV file took %s seconds." % (time.time() - start))
        # usgs_df_filtered = usgs_df[usgs_df.index.isin(csv_output_segments)]
        # usgs_df_filtered.to_csv(output_path.joinpath("usgs_df.csv"))

    if parquet_output_folder:

        LOG.info("- writing flow, velocity, and depth results to .parquet")
        start = time.time()

        # create filenames
        # TO DO: create more descriptive filenames
        if supernetwork_parameters.get("title_string", None):
            filename_fvd = (
                    "flowveldepth_" + supernetwork_parameters["title_string"] + ".parquet"
            )
            filename_courant = (
                    "courant_" + supernetwork_parameters["title_string"] + ".parquet"
            )
        else:
            run_time_stamp = datetime.now().isoformat()
            filename_fvd = "flowveldepth_" + run_time_stamp + ".parquet"
            filename_courant = "courant_" + run_time_stamp + ".parquet"

        output_path = Path(parquet_output_folder).resolve()

        # no parquet_output_segments are specified, then write results for all segments
        if not parquet_output_segments:
            parquet_output_segments = flowveldepth.index

        flowveldepth = flowveldepth.sort_index()
        configuration = output_parameters["parquet_output"].get("configuration")
        prefix_ids = output_parameters["parquet_output"].get("prefix_ids")
        timeseries_df = _parquet_output_format_converter(flowveldepth, restart_parameters.get("start_datetime"), dt,
                                                         configuration, prefix_ids, nexus_dict)

        parquet_output_segments_str = [prefix_ids + '-' + str(segment) for segment in parquet_output_segments]
        timeseries_df.loc[timeseries_df['location_id'].isin(parquet_output_segments_str)].to_parquet(
            output_path.joinpath(filename_fvd), allow_truncated_timestamps=True)

        if return_courant:
            courant = courant.sort_index()
            timeseries_courant = _parquet_output_format_converter(courant, restart_parameters.get("start_datetime"), dt,
                                                                  configuration, prefix_ids)
            timeseries_courant.loc[timeseries_courant['location_id'].isin(parquet_output_segments_str)].to_parquet(
                output_path.joinpath(filename_courant), allow_truncated_timestamps=True)

        LOG.debug("writing parquet file took %s seconds." % (time.time() - start))

    if chano:

        LOG.info("- writing t-route flow results at gage locations to CHANOBS file")
        start = time.time()
        
        # replace waterbody lake_ids with outlet link ids
        if link_lake_crosswalk:
            link_gage_df = _reindex_lake_to_link_id(link_gage_df, link_lake_crosswalk)

        if isinstance(chano['chanobs_output_directory'], Path):
            chano['chanobs_output_directory'] = str(chano['chanobs_output_directory']) + '/'
            chano['chanobs_filepath'] = str(chano['chanobs_filepath'])

        nhd_io.write_chanobs(
            Path(chano['chanobs_output_directory'] + chano['chanobs_filepath']),
            flowveldepth, 
            link_gage_df, 
            t0, 
            dt, 
            nts,
            # TODO allow user to pass a list of segments at which they would like to print results
            # rather than just printing at gages. 
        )

        if (not logFileName == 'NONE'):
            with open(logFileName, 'a') as preRunLog:
                preRunLog.write("\n") 
                preRunLog.write("-----\n") 
                preRunLog.write("Output of results at gage locations into folder: "+str(Path(chano['chanobs_output_directory'])+"\n"))
                preRunLog.write("-----\n") 
            preRunLog.close()  

        LOG.debug("writing flow data to CHANOBS took %s seconds." % (time.time() - start))       

    if lastobso:      
        # Write out LastObs as netcdf when using main_v04 or troute_model with HYfeature.
        # This is only needed if 1) streamflow nudging is ON and 2) a lastobs output
        # folder is provided by the user.
        start = time.time()
        lastobs_output_folder = None
        nudging_true = None
        streamflow_da = data_assimilation_parameters.get('streamflow_da', None)
        if streamflow_da:
            lastobs_output_folder = output_parameters.get("lastobs_output", None)

            nudging_true = streamflow_da.get('streamflow_nudging', None)

        if nudging_true and lastobs_output_folder and not lastobs_df.empty:

            LOG.info("- writing lastobs files")
            start = time.time()

            nhd_io.lastobs_df_output(
                lastobs_df,
                dt,
                nts,
                t0,
                link_gage_df['gages'],
                lastobs_output_folder,
            )

            if (not logFileName == 'NONE'):
                with open(logFileName, 'a') as preRunLog:
                    preRunLog.write("\n") 
                    preRunLog.write("-----\n") 
                    preRunLog.write("Output of lastobs data into folder: "+str(lastobs_output_folder)+"\n") 
                    preRunLog.write("-----\n") 
                preRunLog.close()  

            LOG.debug("writing lastobs files took %s seconds." % (time.time() - start))

    # if 'flowveldepth' in locals():
    #    LOG.debug(flowveldepth)

    LOG.debug("output complete in %s seconds." % (time.time() - start_time))

    ################### Parity Check

    if parity_set:

        LOG.info(
            "conducting parity check, comparing WRF Hydro results against t-route results"
        )
    
        start_time = time.time()
        
        parity_check(
            parity_set, results,
        )

        LOG.debug("parity check complete in %s seconds." % (time.time() - start_time))
