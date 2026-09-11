import json
import pathlib
from functools import partial
from datetime import datetime, timedelta
import logging
import os

import pandas as pd
import numpy as np
import netCDF4
from joblib import delayed, Parallel
import pyarrow as pa
import pyarrow.parquet as pq

import troute.nhd_io as nhd_io

LOG = logging.getLogger("TROUTE")

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
    usbr_timeslices_folder = da_params.get(
        "usbr_timeslices_folder",
        None
    )
    canada_timeslices_folder = da_params.get(
        "canada_timeslices_folder",
        None
    )
    LakeOntario_outflow = da_params.get(
        "LakeOntario_outflow",
        None
    )     
    # User-specified DA ON/OFF preferences
    usace_da = False
    usgs_da = False
    # Bound here, not only inside the reservoir_persistence_da branch below: the
    # `not usbr_da` test further down runs unconditionally, so a config with
    # streamflow nudging and no reservoir persistence DA raised UnboundLocalError
    # before it could build a single TimeSlice set.
    usbr_da = False
    GreatLakes_da = False
    reservoir_persistence_da = da_params.get('reservoir_da', {}).get('reservoir_persistence_da', False)
    if reservoir_persistence_da:
        usgs_da = reservoir_persistence_da.get('reservoir_persistence_usgs', False)
        usace_da = reservoir_persistence_da.get('reservoir_persistence_usace', False)
        usbr_da = reservoir_persistence_da.get('reservoir_persistence_usbr', False)
        GreatLakes_da = reservoir_persistence_da.get('reservoir_persistence_greatLake', False)

    nudging = False
    streamflow_da = da_params.get('streamflow_da', False)
    if streamflow_da:
        nudging = streamflow_da.get('streamflow_nudging', False)

    # The scaling DA is a third TimeSlice consumer and does not require nudging;
    # without it here a scaling-only config got no per-window file list and fell
    # back to globbing the whole directory. Same gate exists in
    # nhd_network_utilities_v02.build_da_sets -- keep both.
    scaling = bool(streamflow_da) and streamflow_da.get('streamflow_scaling', False)

    if (not usgs_da and not usace_da and not usbr_da and not GreatLakes_da
            and not nudging and not scaling):
        # if all DA capabilities are OFF, return empty dictionary
        da_sets = [{} for _ in run_sets]
    
    # if no user-input timeslice folders, a list of empty dictionaries
    elif not usgs_timeslices_folder and not usace_timeslices_folder and not usbr_timeslices_folder and not canada_timeslices_folder:
        # if no timeslice folders, return empty dictionary
        da_sets = [{} for _ in run_sets]
        
    # if user-input timeslice folders are present, build TimeSlice sets
    else:
        
        # create Path objects for each TimeSlice directory
        if usgs_timeslices_folder:
            usgs_timeslices_folder = pathlib.Path(usgs_timeslices_folder)
        if usace_timeslices_folder:
            usace_timeslices_folder = pathlib.Path(usace_timeslices_folder)
        if usbr_timeslices_folder:
            usbr_timeslices_folder = pathlib.Path(usbr_timeslices_folder)
        if canada_timeslices_folder:
            canada_timeslices_folder = pathlib.Path(canada_timeslices_folder)
        if LakeOntario_outflow:
            LakeOntario_outflow = pathlib.Path(LakeOntario_outflow)

        # timedelta of TimeSlice data. The convention is 15 minutes, and the filenames
        # say so, but an observation record can arrive at any cadence: hourly USGS data in
        # 15 minute-named files is common. Read the cadence from the directory rather than
        # assuming it, or three names in four are enumerated, missed and warned about.
        dt_timeslice = _timeslice_cadence(
            usgs_timeslices_folder or usace_timeslices_folder or usbr_timeslices_folder
        )

        # the number of timeslice files appended to the front- and back-ends
        # of the TimeSlice file interpolation stack
        pad_hours = da_params.get("timeslice_lookback_hours",0)
        timeslice_pad = int(round(pad_hours * 3600 / dt_timeslice.total_seconds()))

        da_sets = [] # initialize list to store TimeSlice set lists

        # Loop through each run set and build lists of available TimeSlice files
        for (i, set_dict) in enumerate(run_sets):
            
            # Append an empty dictionary to the loop, which be used to hold
            # lists of USGS and USACE TimeSlice files.
            da_sets.append({})

            # timestamps of TimeSlice files desired for run set i
            timestamps = pd.date_range(
                t0 - dt_timeslice * timeslice_pad,
                run_sets[i]['final_timestamp'] + _timeslice_lead_out(dt_timeslice),
                freq=dt_timeslice
            )

            # identify available USGS TimeSlices in run set i
            if usgs_timeslices_folder and (nudging or usgs_da or scaling):
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

            # identify available USACE TimeSlices in run set i
            if usbr_timeslices_folder and usbr_da:
                filenames_usbr = (timestamps.strftime('%Y-%m-%d_%H:%M:%S') 
                            + '.15min.usbrTimeSlice.ncdf').to_list()
                
                # identify available USACE TimeSlices
                filenames_usbr = _check_timeslice_exists(
                    filenames_usbr, 
                    usbr_timeslices_folder
                )
                
                # Add available TimeSlices to da_sets list
                da_sets[i]['usbr_timeslice_files'] = filenames_usbr

            # identify available USGS TimeSlices in run set i
            if canada_timeslices_folder and GreatLakes_da:
                filenames_canada = (timestamps.strftime('%Y-%m-%d_%H:%M:%S') 
                            + '.15min.wscTimeSlice.ncdf').to_list()

                # identify available USGS TimeSlices
                filenames_canada = _check_timeslice_exists(
                    filenames_canada, 
                    canada_timeslices_folder
                )
                # Add available TimeSlices to da_sets list
                da_sets[i]['canada_timeslice_files'] = filenames_canada
            if LakeOntario_outflow:
                da_sets[i]['LakeOntario_outflow'] = LakeOntario_outflow                
            # reset initialization time for loop set i+1
            t0 = run_sets[i]['final_timestamp']
            
    return da_sets

# Lead-out past the end of a run set, so the interpolator brackets the final model
# timestep. One hour, which is what the old ``dt_timeslice * 4`` meant at the assumed
# 15 minute cadence. Wall-clock like ``timeslice_lookback_hours`` at the front.
_TIMESLICE_LEAD_OUT = timedelta(hours=1)


def _timeslice_lead_out(dt_timeslice):
    """Lead-out at *dt_timeslice* spacing, never less than one file.

    A cadence coarser than the lead-out would round down to zero files past the end of
    the run, leaving the last timestep nothing to interpolate against.
    """
    return max(_TIMESLICE_LEAD_OUT, dt_timeslice)


def _timeslice_cadence(timeslices_folder, default=timedelta(minutes=15)):
    """The spacing between the TimeSlice files in *timeslices_folder*.

    Taken as the most common gap between consecutive file timestamps, which is robust to
    a missing file here and there. Falls back to *default* when the directory is absent,
    empty or holds a single file, so callers behave as they did before.

    Arguments
    ---------
    - timeslices_folder (pathlib.PosixPath): TimeSlice directory
    - default                   (timedelta): cadence to assume when it cannot be measured

    Returns
    -------
    - cadence (timedelta)
    """
    if not timeslices_folder:
        return default
    folder = pathlib.Path(timeslices_folder)
    if not folder.is_dir():
        return default

    stamps = []
    for f in folder.glob("*TimeSlice.ncdf"):
        try:
            stamps.append(datetime.strptime(f.name[:19], "%Y-%m-%d_%H:%M:%S"))
        except ValueError:
            continue
    if len(stamps) < 2:
        return default

    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if not gaps:
        return default
    return max(set(gaps), key=gaps.count)


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
    
    # The caller spells the cadence token as 15min because that is the NWM
    # convention, but the producer writes whatever its observations actually are
    # (the hourly USGS store writes 60min). Match on the timestamp and the family
    # and accept any cadence, so an hourly directory is not silently empty.
    filenames_existing = []
    substituted = []
    for f in filenames:
        J = pathlib.Path(timeslices_folder).joinpath(f)
        if J.is_file():
            filenames_existing.append(f)
            continue
        stamp, _, rest = f.partition(".")
        cadence, _, family = rest.partition(".")
        match = sorted(pathlib.Path(timeslices_folder).glob(f"{stamp}.*.{family}"))
        if not match:
            LOG.warning("Missing TimeSlice file %s", J)
            continue
        if len(match) > 1:
            # Two products for one timestamp is not the NWM layout, and picking
            # by name would silently prefer whichever sorts first. Rare enough to
            # report every time.
            LOG.warning(
                "Ambiguous TimeSlice for %s: %s; using %s",
                stamp, [m.name for m in match], match[0].name,
            )
        else:
            substituted.append((cadence, match[0].name.partition(".")[2].partition(".")[0]))
        filenames_existing.append(match[0].name)

    if substituted:
        # One line, not one per file: a directory written at another cadence
        # substitutes EVERY file, on every window, for every arm. Every DISTINCT
        # pair is named, so a directory that changes product mid-window shows both.
        pairs = ", ".join(f"{req} -> {used}" for req, used in sorted(set(substituted)))
        LOG.info(
            "TimeSlice: %d file(s) matched at a different cadence (%s)",
            len(substituted), pairs,
        )

    return filenames_existing

def build_forcing_sets(
    forcing_parameters,
    t0
):

    run_sets           = forcing_parameters.get("qlat_forcing_sets", None)
    nexus_input_folder = forcing_parameters.get("nexus_input_folder", None)
    nts                = forcing_parameters.get("nts", None)
    max_loop_size      = forcing_parameters.get("max_loop_size", 12)
    if not max_loop_size:  # 0 is the schema's "automatic"
        max_loop_size = 24
        LOG.info("max_loop_size not set; using %d forcing file(s) per window.", max_loop_size)
    dt                 = forcing_parameters.get("dt", None)

    try:
        nexus_input_folder = pathlib.Path(nexus_input_folder)
        assert nexus_input_folder.is_dir() == True
    except TypeError:
        raise TypeError("Aborting simulation because no nexus_input_folder is specified in the forcing_parameters section of the .yaml control file.") from None
    except AssertionError:
        raise AssertionError("Aborting simulation because the nexus_input_folder:", qlat_input_folder,"does not exist. Please check the the nexus_input_folder variable is correctly entered in the .yaml control file") from None

    forcing_glob_filter = forcing_parameters.get("nexus_file_pattern_filter", "*.NEXOUT")

    if forcing_glob_filter=="nex-*":
        print("Reformating qlat nexus files as hourly binary files...")
        binary_folder = forcing_parameters.get('binary_nexus_file_folder', None)
        nexus_files = nexus_input_folder.glob(forcing_glob_filter)

        #Check that directory/files specified will work
        if not binary_folder:
            raise(RuntimeError("No output binary qlat folder supplied in config"))
        elif not os.path.exists(binary_folder):
            raise(RuntimeError("Output binary qlat folder supplied in config does not exist"))
        elif len(list(pathlib.Path(binary_folder).glob('*.parquet'))) != 0:
            raise(RuntimeError("Output binary qlat folder supplied in config is not empty (already contains '.parquet' files)"))

        #Add tnx for backwards compatability
        nexus_files_list = list(nexus_files) + list(nexus_input_folder.glob('tnx*.csv'))
        #Convert files to binary hourly files, reset nexus input information
        nexus_input_folder, forcing_glob_filter = nex_files_to_binary(nexus_files_list, binary_folder)
        forcing_parameters["nexus_input_folder"] = nexus_input_folder
        forcing_parameters["nexus_file_pattern_filter"] = forcing_glob_filter
        
    # TODO: Throw errors if insufficient input data are available
    if run_sets:        
        #FIXME: Change it for hyfeature
        '''
        # append final_timestamp variable to each set_list
        qlat_input_folder = pathlib.Path(qlat_input_folder)
        for (s, _) in enumerate(run_sets):
            final_chrtout = qlat_input_folder.joinpath(run_sets[s]['qlat_files'
                    ][-1])
            final_timestamp_str = nhd_io.get_param_str(final_chrtout,
                    'model_output_valid_time')
            run_sets[s]['final_timestamp'] = \
                datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')
        '''  
    elif nexus_input_folder:        
        # Construct run_set dictionary from user-specified parameters

        # get the first and seconded files from an ordered list of all forcing files
        nexus_input_folder = pathlib.Path(nexus_input_folder)
        all_files          = sorted(nexus_input_folder.glob(forcing_glob_filter))
        first_file         = all_files[0]
        second_file        = all_files[1]

        # Deduce the timeinterval of the forcing data from the output timestamps of the first
        # two ordered CHRTOUT files
        df     = read_file(first_file)
        t1_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")
        t1     = datetime.strptime(t1_str,"%Y-%m-%d_%H:%M:%S")
        df     = read_file(second_file)
        t2_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")
        t2     = datetime.strptime(t2_str,"%Y-%m-%d_%H:%M:%S")
        dt_qlat_timedelta = t2 - t1
        dt_qlat = dt_qlat_timedelta.seconds

        # determine qts_subdivisions
        qts_subdivisions = dt_qlat / dt
        if dt_qlat % dt == 0:
            qts_subdivisions = dt_qlat / dt
        # make sure that qts_subdivisions = dt_qlat / dt
        forcing_parameters['qts_subdivisions']= qts_subdivisions

        # the number of files required for the simulation
        nfiles = int(np.ceil(nts / qts_subdivisions))
        
        # list of forcing file datetimes
        #datetime_list = [t0 + dt_qlat_timedelta * (n + 1) for n in
        #                 range(nfiles)]
        # ** Correction ** Because qlat file at time t is constantly applied throughout [t, t+1],
        #               ** n + 1 should be replaced by n
        datetime_list = [t0 + dt_qlat_timedelta * (n) for n in
                         range(nfiles)]        
        datetime_list_str = [datetime.strftime(d, '%Y%m%d%H%M') for d in
                             datetime_list]

        # list of forcing files
        forcing_filename_list = [d_str + forcing_glob_filter[1:] for d_str in
                                 datetime_list_str]
        
        # check that all forcing files exist
        for f in forcing_filename_list:
            try:
                J = pathlib.Path(nexus_input_folder.joinpath(f))     
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
                run_sets[j]['nexus_files'] = forcing_filename_list[k:k
                    + max_loop_size]
            else:
                run_sets[j]['nexus_files'] = forcing_filename_list[k:]

            nts_accum += len(run_sets[j]['nexus_files']) * qts_subdivisions
            if nts_accum <= nts:
                run_sets[j]['nts'] = int(len(run_sets[j]['nexus_files'])
                                         * qts_subdivisions)
            else:
                run_sets[j]['nts'] = int(nts - nts_last)

            final_nexout        = nexus_input_folder.joinpath(run_sets[j]['nexus_files'
                    ][-1])            
            #final_timestamp_str = nhd_io.get_param_str(final_nexout,
            #        'model_output_valid_time')
            df                  = read_file(final_nexout)
            final_timestamp_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")           
            
            run_sets[j]['final_timestamp'] = \
                datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')

            nts_last = nts_accum
            k += max_loop_size
            j += 1

    return run_sets

def build_qlateral_array(
    run,
    cpu_pool,
    nexus_to_upstream_flowpath_dict, 
    segment_index=pd.Index([]),
    ts_iterator=None,
    file_run_size=None,
):
    # TODO: set default/optional arguments
    qts_subdivisions = run.get("qts_subdivisions", 1)
    nts = run.get("nts", 1)
    nexus_input_folder = run.get("nexus_input_folder", None)
    qlat_input_file = run.get("qlat_input_file", None)

    if nexus_input_folder:
        nexus_input_folder = pathlib.Path(nexus_input_folder)
        if "nexus_files" in run:
            nexus_files = run.get("nexus_files")
            nexus_files = [nexus_input_folder.joinpath(f) for f in nexus_files]
        elif "nexus_file_pattern_filter" in run:
            nexus_file_pattern_filter = run.get(
                "nexus_file_pattern_filter", "*NEXOUT*"
            )
            nexus_files = sorted(nexus_input_folder.glob(nexus_file_pattern_filter))

        qlat_file_index_col = run.get(
            "qlat_file_index_col", "feature_id"
        )
        qlat_file_value_col = run.get("qlat_file_value_col", "q_lateral")
        gw_bucket_col = run.get("qlat_file_gw_bucket_flux_col","qBucket")
        terrain_ro_col = run.get("qlat_file_terrain_runoff_col","qSfcLatRunoff")


        #nexuses_lateralflows_df = nhd_io.get_ql_from_csv(nexus_files[0])
        '''
        # Parallel reading of qlateral data from CHRTOUT
        with Parallel(n_jobs=cpu_pool) as parallel:
            jobs = []
            for f in qlat_files:
                jobs.append(
                    #delayed(nhd_io.get_ql_from_chrtout)
                    #(f, qlat_file_value_col, gw_bucket_col, terrain_ro_col)
                    delayed(nhd_io.get_ql_from_csv)
                    (f)                    
                )
            ql_list = parallel(jobs)

        # get feature_id from a single CHRTOUT file
        with netCDF4.Dataset(qlat_files[0]) as ds:
            idx = ds.variables[qlat_file_index_col][:].filled()

        # package data into a DataFrame
        qlats_df = pd.DataFrame(
            np.stack(ql_list).T,
            index = idx,
            columns = range(len(qlat_files))
        )    
        '''
        dfs=[]
        for f in nexus_files:
            df = read_file(f).set_index(['feature_id']) 
            dfs.append(df)
        
        # lateral flows [m^3/s] are stored at NEXUS points with NEXUS ids
        nexuses_lateralflows_df = pd.concat(dfs, axis=1)  
        
        # Take flowpath ids entering NEXUS and replace NEXUS ids by the upstream flowpath ids 
        qlats_df = pd.concat( (nexuses_lateralflows_df.loc[int(k)].rename(v)
                            for k,v in nexus_to_upstream_flowpath_dict.items() ), axis=1
                            ).T 
        qlats_df.columns=range(len(nexus_files))
        qlats_df = qlats_df[qlats_df.index.isin(segment_index)]
 
        # The segment_index has the full network set of segments/flowpaths. 
        # Whereas the set of flowpaths that are downstream of nexuses is a 
        # subset of the segment_index. Therefore, all of the segments/flowpaths
        # that are not accounted for in the set of flowpaths downstream of
        # nexuses need to be added to the qlateral dataframe and padded with
        # zeros.
        all_df = pd.DataFrame( np.zeros( (len(segment_index), len(qlats_df.columns)) ), index=segment_index,
                    columns=qlats_df.columns )
        all_df.loc[ qlats_df.index ] = qlats_df
        qlats_df = all_df.sort_index()    
    elif qlat_input_file:
        qlats_df = nhd_io.get_ql_from_csv(qlat_input_file)
    else:
        qlat_const = run.get("qlat_const", 0)
        qlats_df = pd.DataFrame(
            qlat_const,
            index=segment_index,
            columns=range(nts // qts_subdivisions),
            dtype="float32",
        )

    # TODO: Make a more sophisticated date-based filter
    max_col = 1 + nts // qts_subdivisions
    if len(qlats_df.columns) > max_col:
        qlats_df.drop(qlats_df.columns[max_col:], axis=1, inplace=True)

    if not segment_index.empty:
        qlats_df = qlats_df[qlats_df.index.isin(segment_index)]

    return qlats_df

def nex_files_to_binary(nexus_files, binary_folder):
    for f in nexus_files:
        # read the csv file
        df = pd.read_csv(f, usecols=[1,2], names=['Datetime','qlat'])
        
        # convert and reformat datetime column
        df['Datetime']= pd.to_datetime(df['Datetime']).dt.strftime("%Y%m%d%H%M")

        # reformat the dataframe
        df['feature_id'] = get_id_from_filename(f)
        df = df.pivot(index="feature_id", columns="Datetime", values="qlat")
        df.columns.name = None

        for col in df.columns:
            table_new = pa.Table.from_pandas(df.loc[:, [col]])
            
            if not os.path.exists(f'{binary_folder}/{col}NEXOUT.parquet'):
                pq.write_table(table_new, f'{binary_folder}/{col}NEXOUT.parquet')
            
            else:
                table_old = pq.read_table(f'{binary_folder}/{col}NEXOUT.parquet')
                table = pa.concat_tables([table_old,table_new])
                pq.write_table(table, f'{binary_folder}/{col}NEXOUT.parquet')
    
    nexus_input_folder = binary_folder
    forcing_glob_filter = '*NEXOUT.parquet'

    return nexus_input_folder, forcing_glob_filter

def get_id_from_filename(file_name):
    id = os.path.splitext(file_name)[0].split('-')[1].split('_')[0]
    return int(id)

def read_file(file_name):
    extension = file_name.suffix
    if extension=='.csv':
        df = pd.read_csv(file_name)
    elif extension=='.parquet':
        df = pq.read_table(file_name).to_pandas().reset_index()
        df.index.name = None
    
    return df
