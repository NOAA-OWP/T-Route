import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import troute.nhd_io as nhd_io
from build_tests import parity_check
import logging


LOG = logging.getLogger('')

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
    data_assimilation_parameters=False,
    lastobs_df=None,
    link_gage_df=None,
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
    rsrto = output_parameters.get("hydro_rst_output", None)
    chrto = output_parameters.get("chrtout_output", None)
    test = output_parameters.get("test_output", None)
    chano = output_parameters.get("chanobs_output", None)

    if csv_output:
        csv_output_folder = output_parameters["csv_output"].get(
            "csv_output_folder", None
        )
        csv_output_segments = csv_output.get("csv_output_segments", None)

    
    if csv_output_folder or rsrto or chrto or chano or test:
        
        start = time.time()
        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()

        flowveldepth = pd.concat(
            [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
            copy=False,
        )
        
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
            
        LOG.debug("Constructing the FVD DataFrame took %s seconds." % (time.time() - start))

    if test:
        flowveldepth.to_pickle(Path(test))
    
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
        
    if chano:
        
        LOG.info("- writing t-route flow results to CHANOBS file")
        
        gage_feature_id = link_gage_df.index.to_numpy(dtype = "int32")
        gage_flow_data = flowveldepth.loc[link_gage_df.index].iloc[:,::3].to_numpy(dtype="float32") 
        
        gage_flow_time = [t0 + timedelta(seconds = (i+1) * dt) for i in range(nts)]
#         gage_flow_time = np.arange(
#             np.datetime64(t0 + timedelta(seconds = dt)),
#             np.datetime64(t0 + timedelta(seconds = (nts+1)*dt)),
#             np.timedelta64(dt, 's')
#         )
        
        import netCDF4
        from cftime import date2num
        
        with netCDF4.Dataset(
            filename = '/glade/scratch/adamw/chanobs_dump/chanobs_test',
            mode = 'w',
            format = "NETCDF4"
        ) as f:
            
            # =========== DIMENSIONS ===============
            _ = f.createDimension("time", nts)
            _ = f.createDimension("feature_id", len(gage_feature_id))
            _ = f.createDimension("reference_time", 1)
            
            # =========== reference_time VARIABLE ===============
            REF_TIME = f.createVariable(
                varname = "reference_time",
                datatype = 'int32',
                dimensions = ("reference_time",),
            )
            REF_TIME[:] = date2num(
                t0, 
                units = "minutes since 1970-01-01 00:00:00 UTC",
                calendar = "gregorian"
            )
            f['reference_time'].setncatts(
                {
                    'long_name': 'vaild output time',
                    'standard_name': 'time',
                    'units': 'minutes since 1970-01-01 00:00:00 UTC'
                }
            )
            
            # =========== time VARIABLE ===============
            TIME = f.createVariable(
                varname = "time",
                datatype = 'int32',
                dimensions = ("time",),
            )
            TIME[:] = date2num(
                gage_flow_time, 
                units = "minutes since 1970-01-01 00:00:00 UTC",
                calendar = "gregorian"
            )
            f['time'].setncatts(
                {
                    'long_name': 'model initialization time',
                    'standard_name': 'forecast_reference_time',
                    'units': 'minutes since 1970-01-01 00:00:00 UTC'
                }
            )
            
            # =========== feature_id VARIABLE ===============
            FEATURE_ID = f.createVariable(
                varname = "feature_id",
                datatype = 'int32',
                dimensions = ("feature_id",),
            )
            FEATURE_ID[:] = gage_feature_id
            f['feature_id'].setncatts(
                {
                    'long_name': 'Reach ID',
                    'comment': 'NHDPlusv2 ComIDs within CONUS, arbitrary Reach IDs outside of CONUS',
                    'cf_role:': 'timeseries_id'
                }
            )
            
            # =========== streamflow VARIABLE ===============            
            y = f.createVariable(
                    varname = "streamflow",
                    datatype = "f4",
                    dimensions = ("time", "feature_id"),
                    fill_value = np.nan
                )
            y[:] = gage_flow_data
            
            # =========== GLOBAL ATTRIBUTES ===============  
            f.setncatts(
                {
                    'model_initialization_time': t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    'model_output_valid_time': gage_flow_time[0].strftime('%Y-%m-%d_%H:%M:%S'),
                    'model_total_valid_times': len(gage_flow_time)
                }
            )
            
        
        import pdb; pdb.set_trace()
        # find gage locations
        
        

    # Write out LastObs as netcdf.
    # Assumed that this capability is only needed for AnA simulations
    # i.e. when timeslice files are provided to support DA
    data_assimilation_folder = data_assimilation_parameters.get(
    "data_assimilation_timeslices_folder", None
    )
    lastobs_output_folder = data_assimilation_parameters.get(
    "lastobs_output_folder", None
    )
    # if lastobs_output_folder:
    #     warnings.warn("No LastObs output folder directory specified in input file - not writing out LastObs data")
    if data_assimilation_folder and lastobs_output_folder:
        
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
        
        LOG.debug("writing lastobs files took %s seconds." % (time.time() - start))

    if 'flowveldepth' in locals():
        LOG.debug(flowveldepth)

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
