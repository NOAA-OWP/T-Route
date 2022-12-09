import pathlib
import logging
import os

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import troute.nhd_io as nhd_io


LOG = logging.getLogger('')


def build_forcing_sets(
    forcing_parameters,
    t0
):

    run_sets           = forcing_parameters.get("qlat_forcing_sets", None)
    nexus_input_folder = forcing_parameters.get("nexus_input_folder", None)
    nts                = forcing_parameters.get("nts", None)
    max_loop_size      = forcing_parameters.get("max_loop_size", 12)
    dt                 = forcing_parameters.get("dt", None)

    if nexus_input_folder is not None:
        nexus_input_folder = pathlib.Path(nexus_input_folder)
        if not nexus_input_folder.is_dir():
            raise ValueError(f"Aborting simulation because the nexus_input_folder: {nexus_input_folder} does not exist. Please check the the nexus_input_folder variable is correctly entered in the .yaml control file")
    else:
        raise TypeError("Aborting simulation because no nexus_input_folder is specified in the forcing_parameters section of the .yaml control file.")
    

    forcing_glob_filter = forcing_parameters.get("nexus_file_pattern_filter", "*.NEXOUT")

    if forcing_glob_filter=="nex-*":
        print("Reformating qlat nexus files as hourly binary files...")
        binary_folder = forcing_parameters.get('binary_nexus_file_folder', None)
        nexus_files = nexus_input_folder.glob(forcing_glob_filter)

        #Check that directory/files specified will work
        if not binary_folder:
            raise RuntimeError("No output binary qlat folder supplied in config")
        elif not os.path.exists(binary_folder):
            raise RuntimeError("Output binary qlat folder supplied in config does not exist")
        else:
            try:
                next(pathlib.Path(binary_folder).glob('*.parquet'))
                raise RuntimeError("Output binary qlat folder supplied in config is not empty (already contains '.parquet' files)")
            except StopIteration:
                # Directory is empty
                pass

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
        pass
    elif nexus_input_folder:        
        # Construct run_set dictionary from user-specified parameters

        # get the first and seconded files from an ordered list of all forcing files
        nexus_input_folder = pathlib.Path(nexus_input_folder)
        all_files          = sorted(nexus_input_folder.glob(forcing_glob_filter))
        first_file         = all_files[0]
        second_file        = all_files[1]

        # Deduce the timeinterval of the forcing data from the output timestamps of the first
        # two ordered CHRTOUT files
        df = read_file(first_file)
        t1 = pd.to_datetime(df.columns[1])
        df = read_file(second_file)
        t2 = pd.to_datetime(df.columns[1])
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
        # Existence of each file is checked, raising an error if an expected file is missing.
        forcing_filenames = []
        for n in range(nfiles):
            fn_dt = (t0 + dt_qlat_timedelta * n).strftime("%Y%m%d%H%M")
            fn = nexus_input_folder.joinpath(fn_dt + forcing_glob_filter[1:])
            if fn.is_file():
                forcing_filenames.append(fn)
            else:
                raise FileNotFoundError(f"Forcing file {fn} is missing. Aborting simulation.")

        # build run sets list
        run_sets = []
        k = 0
        j = 0
        nts_accum = 0
        nts_last = 0
        while k < len(forcing_filenames):
            run_sets.append({})

            if k + max_loop_size < len(forcing_filenames):
                run_sets[j]['nexus_files'] = forcing_filenames[k:k
                    + max_loop_size]
            else:
                run_sets[j]['nexus_files'] = forcing_filenames[k:]

            nts_accum += len(run_sets[j]['nexus_files']) * qts_subdivisions
            if nts_accum <= nts:
                run_sets[j]['nts'] = int(len(run_sets[j]['nexus_files'])
                                         * qts_subdivisions)
            else:
                run_sets[j]['nts'] = int(nts - nts_last)

            final_nexout = nexus_input_folder.joinpath(run_sets[j]['nexus_files'][-1]) 
            #final_timestamp_str = nhd_io.get_param_str(final_nexout,
            #        'model_output_valid_time')
            df = read_file(final_nexout)
            run_sets[j]['final_timestamp'] = pd.to_datetime(df.columns[1])

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
    else:
        raise ValueError(f"Unknown file suffix: {extension}")
    return df