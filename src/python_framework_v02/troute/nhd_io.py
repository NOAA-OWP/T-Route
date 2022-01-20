import zipfile
import xarray as xr
import pandas as pd
import geopandas as gpd
import json
import yaml
import numpy as np
from toolz import compose
import dask.array as da
import sys
import math
from datetime import *
import pathlib
import netCDF4
import time
import logging
from joblib import delayed, Parallel
from cftime import date2num



LOG = logging.getLogger('')

from troute.nhd_network import reverse_dict


def read_netcdf(geo_file_path):
    with xr.open_dataset(geo_file_path) as ds:
        return ds.to_dataframe()


def read_csv(geo_file_path, header="infer", layer_string=None):
    if geo_file_path.suffix == ".zip":
        if layer_string is None:
            raise ValueError("layer_string is needed if reading from compressed csv")
        with zipfile.ZipFile(geo_file_path, "r") as zcsv:
            with zcsv.open(layer_string) as csv:
                return pd.read_csv(csv, header=header)
    else:
        return pd.read_csv(geo_file_path, header=header)


def read_geopandas(geo_file_path, layer_string=None, driver_string=None):
    return gpd.read_file(geo_file_path, driver=driver_string, layer_string=layer_string)


def read(geo_file_path, layer_string=None, driver_string=None):
    if geo_file_path.suffix == ".nc":
        return read_netcdf(geo_file_path)
    else:
        return read_geopandas(
            geo_file_path, layer_string=layer_string, driver_string=driver_string
        )


def read_mask(path, layer_string=None):
    return read_csv(path, header=None, layer_string=layer_string)


def read_custom_input_new(custom_input_file):
    if custom_input_file[-4:] == "yaml":
        with open(custom_input_file) as custom_file:
            data = yaml.load(custom_file, Loader=yaml.SafeLoader)
    else:
        with open(custom_input_file) as custom_file:
            data = json.load(custom_file)
    log_parameters = data.get("log_parameters", {})
    network_topology_parameters = data.get("network_topology_parameters", None)
    supernetwork_parameters = network_topology_parameters.get(
        "supernetwork_parameters", None
    )
    preprocessing_parameters = network_topology_parameters.get(
        "preprocessing_parameters", {}
    )
    if not preprocessing_parameters:
        preprocessing_parameters = {}
    waterbody_parameters = network_topology_parameters.get("waterbody_parameters", None)
    compute_parameters = data.get("compute_parameters", {})
    forcing_parameters = compute_parameters.get("forcing_parameters", {})
    restart_parameters = compute_parameters.get("restart_parameters", {})
    diffusive_parameters = compute_parameters.get("diffusive_parameters", {})
    data_assimilation_parameters = compute_parameters.get(
        "data_assimilation_parameters", {}
    )
    output_parameters = data.get("output_parameters", {})
    parity_parameters = output_parameters.get("wrf_hydro_parity_check", {})

    # TODO: add error trapping for potentially missing files
    return (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        diffusive_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
    )

def read_diffusive_domain(domain_file):
    if domain_file[-4:] == "yaml":
        with open(domain_file) as domain:
            data = yaml.load(domain, Loader=yaml.SafeLoader)
    else:
        with open(domain_file) as domain:
            data = json.load(domain)
            
    return data

def read_custom_input(custom_input_file):
    if custom_input_file[-4:] == "yaml":
        with open(custom_input_file) as custom_file:
            data = yaml.load(custom_file, Loader=yaml.SafeLoader)
    else:
        with open(custom_input_file) as custom_file:
            data = json.load(custom_file)
    supernetwork_parameters = data.get("supernetwork_parameters", None)
    waterbody_parameters = data.get("waterbody_parameters", {})
    forcing_parameters = data.get("forcing_parameters", {})
    restart_parameters = data.get("restart_parameters", {})
    output_parameters = data.get("output_parameters", {})
    run_parameters = data.get("run_parameters", {})
    parity_parameters = data.get("parity_parameters", {})
    data_assimilation_parameters = data.get("data_assimilation_parameters", {})
    diffusive_parameters = data.get("diffusive_parameters", {})
    coastal_parameters = data.get("coastal_parameters", {})

    # TODO: add error trapping for potentially missing files
    return (
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


def replace_downstreams(data, downstream_col, terminal_code):
    ds0_mask = data[downstream_col] == terminal_code
    new_data = data.copy()
    new_data.loc[ds0_mask, downstream_col] = ds0_mask.index[ds0_mask]

    # Also set negative any nodes in downstream col not in data.index
    new_data.loc[~data[downstream_col].isin(data.index), downstream_col] *= -1
    return new_data


def read_waterbody_df(waterbody_parameters, waterbodies_values, wbtype="level_pool"):
    """
    General waterbody dataframe reader. At present, only level-pool
    capability exists.
    """
    if wbtype == "level_pool":
        wb_params = waterbody_parameters[wbtype]
        return read_level_pool_waterbody_df(
            wb_params["level_pool_waterbody_parameter_file_path"],
            wb_params["level_pool_waterbody_id"],
            waterbodies_values[wbtype],
        )


def read_level_pool_waterbody_df(
    parm_file, lake_index_field="lake_id", lake_id_mask=None
):
    """
    Reads LAKEPARM file and prepares a dataframe, filtered
    to the relevant reservoirs, to provide the parameters
    for level-pool reservoir computation.

    Completely replaces the read_waterbody_df function from prior versions
    of the v02 routing code.

    Prior version filtered the dataframe as opposed to the dataset as in this version.
    with xr.open_dataset(parm_file) as ds:
        df1 = ds.to_dataframe()
    df1 = df1.set_index(lake_index_field).sort_index(axis="index")
    if lake_id_mask is None:
        return df1
    else:
        return df1.loc[lake_id_mask]
    """

    # TODO: avoid or parameterize "feature_id" or ... return to name-blind dataframe version
    with xr.open_dataset(parm_file) as ds:
        ds = ds.swap_dims({"feature_id": lake_index_field})
        df1 = ds.sel({lake_index_field: list(lake_id_mask)}).to_dataframe()
    df1 = df1.sort_index(axis="index")
    return df1


def read_reservoir_parameter_file(
    reservoir_parameter_file, lake_index_field="lake_id", lake_id_mask=None
):
    """
    Reads reservoir parameter file, which is separate from the LAKEPARM file.
    This function is only called if Hybrid Persistence or RFC type reservoirs
    are active.
    """
    with xr.open_dataset(reservoir_parameter_file) as ds:
        ds = ds.swap_dims({"feature_id": lake_index_field})

        ds_new = ds["reservoir_type"]

        df1 = ds_new.sel({lake_index_field: list(lake_id_mask)}).to_dataframe()

    return df1


def get_ql_from_csv(qlat_input_file, index_col=0):
    """
    qlat_input_file: comma delimted file with header giving timesteps, rows for each segment
    index_col = 0: column/field in the input file with the segment/link id
    """
    ql = pd.read_csv(qlat_input_file, index_col=index_col)
    ql.index = ql.index.astype(int)
    ql = ql.sort_index(axis="index")
    return ql.astype("float32")


def read_qlat(path):
    """
    retained for backwards compatibility with early v02 files
    """
    return get_ql_from_csv(path)

def get_ql_from_chrtout(
    f,
    qlateral_varname = "q_lateral",
    qbucket_varname="qBucket",
    runoff_varname = "qSfcLatRunoff",
):
    '''
    Return an array of qlateral data from a single CHRTOUT netCDF4 file.
    If the lateral inflow variable is not present in the file, then calculate
    lateral inflow as the sum of qbucket and surface runoff.
    
    Arguments
    ---------
    f (Path): 
    qlateral_varname (string): lateral inflow variable name
    qbucket_varname (string): Groundwater bucket flux variable name
    runoff_varname (string): surface runoff variable name
    
    NOTES:
    - This is very bespoke to WRF-Hydro
    '''
    with netCDF4.Dataset(
        filename = f,
        mode = 'r',
        format = "NETCDF4"
    ) as ds:
        
        all_variables = list(ds.variables.keys())
        if qlateral_varname in all_variables:
            dat = ds.variables[qlateral_varname][:].filled(fill_value = 0.0)
        
        else:
            dat = ds.variables[qbucket_varname][:].filled(fill_value = 0.0) + \
                ds.variables[runoff_varname][:].filled(fill_value = 0.0)
        
    return dat

# TODO: Generalize this name -- perhaps `read_wrf_hydro_chrt_mf()`
def get_ql_from_wrf_hydro_mf(
    qlat_files,
    index_col="feature_id",
    value_col="q_lateral",
    gw_col="qBucket",
    runoff_col = "qSfcLatRunoff",
):
    """
    qlat_files: globbed list of CHRTOUT files containing desired lateral inflows
    index_col: column/field in the CHRTOUT files with the segment/link id
    value_col: column/field in the CHRTOUT files with the lateral inflow value
    gw_col: column/field in the CHRTOUT files with the groundwater bucket flux value
    runoff_col: column/field in the CHRTOUT files with the runoff from terrain routing value

    In general the CHRTOUT files contain one value per time step. At present, there is
    no capability for handling non-uniform timesteps in the qlaterals.

    The qlateral may also be input using comma delimited file -- see
    `get_ql_from_csv`


    Note/Todo:
    For later needs, filtering for specific features or times may
    be accomplished with one of:
        ds.loc[{selectors}]
        ds.sel({selectors})
        ds.isel({selectors})

    Returns from these selection functions are sub-datasets.

    For example:
    ```
    (Pdb) ds.sel({"feature_id":[4186117, 4186169],"time":ds.time.values[:2]})['q_lateral'].to_dataframe()
                                     latitude  longitude  q_lateral
    time                feature_id
    2018-01-01 13:00:00 4186117     41.233807 -75.413895   0.006496
    2018-01-02 00:00:00 4186117     41.233807 -75.413895   0.006460
    ```

    or...
    ```
    (Pdb) ds.sel({"feature_id":[4186117, 4186169],"time":[np.datetime64('2018-01-01T13:00:00')]})['q_lateral'].to_dataframe()
                                     latitude  longitude  q_lateral
    time                feature_id
    2018-01-01 13:00:00 4186117     41.233807 -75.413895   0.006496
    ```
    """

    with xr.open_mfdataset(
        qlat_files,
        combine="nested",
        concat_dim="time",
#         data_vars=["q_lateral","qBucket","qSfcLatRunoff"],
        coords="minimal",
        compat="override",
        # parallel=True,
    ) as ds:
        
        # if forcing file contains a variable with the specified value_col name, 
        # then use it, otherwise compute q_lateral as the sum of qBucket and qSfcLatRunoff
        try:
            qlateral_data = ds[value_col].values.T
        except:
            qlateral_data = ds[gw_col].values.T + ds[runoff_col].values.T
            
        try:
            ql = pd.DataFrame(
                qlateral_data,
                index=ds[index_col].values[0],
                columns=ds.time.values,
                # dtype=float,
            )
        except:
            ql = pd.DataFrame(
                qlateral_data,
                index=ds[index_col].values,
                columns=ds.time.values,
                # dtype=float,
            )

    return ql


def drop_all_coords(ds):
    return ds.reset_coords(drop=True)

def write_chanobs(
    chanobs_filepath, 
    flowveldepth, 
    link_gage_df, 
    t0, 
    dt, 
    nts
):
    
    '''
    Write results at gage locations to netcdf.
    If the user specified file does not exist, create it. 
    If the user specified file already exiss, append it. 
    
    Arguments
    -------------
        chanobs_filepath (Path or string) - 
        flowveldepth (DataFrame) - t-route flow velocity and depth results
        link_gage_df (DataFrame) - linkIDs of gages in network
        t0 (datetime) - initial time
        dt (int) - timestep duration (seconds)
        nts (int) - number of timesteps in simulation
        
    Returns
    -------------
    
    '''
    
    # array of segment linkIDs at gage locations. Results from these segments will be written
    gage_feature_id = link_gage_df.index.to_numpy(dtype = "int32")
    
    # array of simulated flow data at gage locations
    gage_flow_data = flowveldepth.loc[link_gage_df.index].iloc[:,::3].to_numpy(dtype="float32") 
    
    # array of simulation time
    gage_flow_time = [t0 + timedelta(seconds = (i+1) * dt) for i in range(nts)]
    
    if not chanobs_filepath.is_file():
        
        # if no chanobs file exists, create a new one
        # open netCDF4 Dataset in write mode
        with netCDF4.Dataset(
            filename = chanobs_filepath,
            mode = 'w',
            format = "NETCDF4"
        ) as f:

            # =========== DIMENSIONS ===============
            _ = f.createDimension("time", None)
            _ = f.createDimension("feature_id", len(gage_feature_id))
            _ = f.createDimension("reference_time", 1)

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
            y[:] = gage_flow_data.reshape(
                len(gage_flow_time),
                len(gage_feature_id)
            )

            # =========== GLOBAL ATTRIBUTES ===============  
            f.setncatts(
                {
                    'model_initialization_time': t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    'model_output_valid_time': gage_flow_time[0].strftime('%Y-%m-%d_%H:%M:%S'),
                }
            )
            
    else:
        
        # append data to chanobs file
        # open netCDF4 Dataset in r+ mode to append
        with netCDF4.Dataset(
            filename = chanobs_filepath,
            mode = 'r+',
            format = "NETCDF4"
        ) as f:

            # =========== format variable data to be appended =============== 
            time_new = date2num(
                gage_flow_time, 
                units = "minutes since 1970-01-01 00:00:00 UTC",
                calendar = "gregorian"
            )

            flow_new = gage_flow_data.reshape(
                len(gage_flow_time),
                len(gage_feature_id)
            )
            
            # =========== append new flow data =============== 
            tshape = len(f.dimensions['time'])
            f['time'][tshape:(tshape+nts)] = time_new
            f['streamflow'][tshape:(tshape+nts)] = flow_new
            
def write_to_netcdf(f, variables, datatype = 'f4'):
    
    '''
    Quickly append or overwrite variable data in NetCDF files by leveraging the netCDF4 library. 
    For additional documentation on netCDF4: https://unidata.github.io/netcdf4-python/#version-157
    
    Arguments:
    ----------
    f (Path): Name of netCDF file to hold dataset. Can also be a python 3 pathlib instance
    variables (dict): dictionary keys are variable names (strings), dictionary values are tuples:
                           (
                               variable data (numpy array - 1D must be same size as variable dimension), 
                               variable dimension name (string) that already exist in netCDF file, 
                               variable attributes (dict, keys are attribute names and values are attribute contents),
                           )
    datatype: numpy datatype object, or a string that describes a numpy dtype object.
              Supported specifiers include: 'S1' or 'c' (NC_CHAR), 'i1' or 'b' or 'B' (NC_BYTE),
              'u1' (NC_UBYTE), 'i2' or 'h' or 's' (NC_SHORT), 'u2' (NC_USHORT), 'i4' or 'i' or 'l' (NC_INT),
              'u4' (NC_UINT), 'i8' (NC_INT64), 'u8' (NC_UINT64), 'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE)
    
    NOTES:
    - the netCDF files we want to append/edit must have write permission!
    '''
    
    with netCDF4.Dataset(
        filename = f,
        mode = 'r+',
        format = "NETCDF4"
    ) as ds:

        for varname, (vardata, dim, attrs) in variables.items():
            
            # check that dimension exists
            if dim not in list(ds.dimensions.keys()):
                LOG.error("The dimensions %s could not be found in file %s" % (dim, f))
                LOG.error("Aborting writing process for %s. No data were written to this file" % f)
                return        
            
            # check that dimension size and variable data size agree
            dim_size = ds.dimensions[dim].size
            if vardata.size != dim_size:
                LOG.error("Cannot write data of size %d to variable with dimension size of %d" % (vardata.size, dim_size))
                LOG.error("Aborting writing process for %s. No data were written to this file" % f)
                return
            
            # check that varname doesn't already exist
            # if it does, then overwrite it
            if varname in list(ds.variables.keys()):

                ds[varname][:] = vardata

            # if variable does not exist, create new one
            else:

                # create a new variable
                y = ds.createVariable(
                    varname = varname,
                    datatype = datatype,
                    dimensions = (dim,),
                    fill_value = np.nan
                )

                # write data to new variable
                y[:] = vardata

                # include variable attributes
                ds[varname].setncatts(attrs)
                
def write_chrtout(    
    flowveldepth,
    chrtout_files,
    qts_subdivisions,
    cpu_pool,
):
    
    LOG.debug("Starting the write_chrtout function") 
    
    # count the number of simulated timesteps
    nsteps = len(flowveldepth.loc[:,::3].columns)
    
    # determine how many files to write results out to
    nfiles_to_write = int(np.floor(nsteps / qts_subdivisions))
    
    if nfiles_to_write >= 1:
        
        LOG.debug("%d CHRTOUT files will be written." % (nfiles_to_write))
        LOG.debug("Extracting flow DataFrame on qts_subdivisions from FVD DataFrame")
        start = time.time()
        
        flow = flowveldepth.loc[:, ::3].iloc[:, qts_subdivisions-1::qts_subdivisions]
        
        LOG.debug("Extracting flow DataFrame took %s seconds." % (time.time() - start))
        
        varname = 'streamflow_troute'
        dim = 'feature_id'
        attrs = {
            'long_name': 'River Flow',
            'units': 'm3 s-1',
            'coordinates': 'latitude longitude',
            'grid_mapping': 'crs',
            'valid_range': np.array([0,50000], dtype = 'float32'),
        }
        
        LOG.debug("Reindexing the flow DataFrame to align with `feature_id` dimension in CHRTOUT files")
        start = time.time()
        
        with xr.open_dataset(chrtout_files[0]) as ds:
            newindex = ds.feature_id.values
            
        qtrt = flow.reindex(newindex).to_numpy().astype("float32")
        
        LOG.debug("Reindexing the flow DataFrame took %s seconds." % (time.time() - start))
        
        LOG.debug("Writing t-route data to %d CHRTOUT files" % (nfiles_to_write))
        start = time.time()
        with Parallel(n_jobs=cpu_pool) as parallel:
        
            jobs = []
            for i, f in enumerate(chrtout_files[:nfiles_to_write]):

                s = time.time()
                variables = {
                    varname: (qtrt[:,i], dim, attrs)
                }
                jobs.append(delayed(write_to_netcdf)(f, variables))
                LOG.debug("Writing %s." % (f))
                
            parallel(jobs)
               
        LOG.debug("Writing t-route data to %d CHRTOUT files took %s seconds." % (nfiles_to_write, (time.time() - start)))
        
    else:
        LOG.debug("Simulation duration is less than one qts_subdivision. No CHRTOUT files written.")

def get_ql_from_wrf_hydro(qlat_files, index_col="station_id", value_col="q_lateral"):
    """
    qlat_files: globbed list of CHRTOUT files containing desired lateral inflows
    index_col: column/field in the CHRTOUT files with the segment/link id
    value_col: column/field in the CHRTOUT files with the lateral inflow value

    In general the CHRTOUT files contain one value per time step. At present, there is
    no capability for handling non-uniform timesteps in the qlaterals.

    The qlateral may also be input using comma delimited file -- see
    `get_ql_from_csv`
    """

    li = []

    for filename in qlat_files:
        with xr.open_dataset(filename) as ds:
            df1 = ds[["time", value_col]].to_dataframe()

        li.append(df1)

    frame = pd.concat(li, axis=0, ignore_index=False)
    mod = frame.reset_index()
    ql = mod.pivot(index=index_col, columns="time", values=value_col)

    return ql


def read_netcdfs(paths, dim, transform_func=None):
    def process_one_path(path):
        with xr.open_dataset(path) as ds:
            if transform_func is not None:
                ds = transform_func(ds)
            ds.load()
            return ds

    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim, combine_attrs = "override")
    return combined


def preprocess_time_station_index(xd):
    stationId_da_mask = list(
        map(compose(bytes.isalnum, bytes.strip), xd.stationId.values)
    )
    stationId = list(map(bytes.strip, xd.stationId[stationId_da_mask].values))
    #stationId_int = xd.stationId[stationId_da_mask].values.astype(int)

    unique_times_str = np.unique(xd.time.values).tolist()

    unique_times = np.array(unique_times_str, dtype="str")

    center_time = xd.sliceCenterTimeUTC

    tmask = []
    for t in unique_times_str:
        tmask.append(xd.time == t)

    data_var_dict = {}
    # TODO: make this input parameters
    data_vars = ("discharge", "discharge_quality")

    for v in data_vars:
        vals = []
        for i, t in enumerate(unique_times_str):
            vals.append(np.where(tmask[i],xd[v].values[stationId_da_mask],np.nan))
        combined = np.vstack(vals).T
        data_var_dict[v] = (["stationId","time"], combined)

    return xr.Dataset(
        data_vars=data_var_dict,
        coords={"stationId": stationId, "time": unique_times},
        attrs={"sliceCenterTimeUTC": center_time},
    )


def get_nc_attributes(nc_list, attribute, file_selection=[0, -1]):
    rv = []
    for fs in file_selection:
        try:
            rv.append(get_attribute(nc_list[fs], attribute))
        except:
            rv.append(-1)
    return rv


def get_attribute(nc_file, attribute):
    # TODO: consider naming as get_nc_attribute
    # -- this is really specific to a netcdf file.

    with xr.open_dataset(nc_file) as xd:
        return xd.attrs[attribute]


def build_filtered_gage_df(segment_gage_df, gage_col="gages"):
    """
    segment_gage_df - dataframe indexed by segment with at least
        one column, gage_col, with the gage ids by segment.
    gage_col - the name of the column containing the gages
        to filter.
    """
    # TODO: use this function to filter the inputs
    # coming into the da read routines below.
    gage_list = list(map(bytes.strip, segment_gage_df[gage_col].values))
    gage_mask = list(map(bytes.isalnum, gage_list))
    segment_gage_df = segment_gage_df.loc[gage_mask, [gage_col]]
    segment_gage_df[gage_col] = segment_gage_df[gage_col].map(bytes.strip)
    return segment_gage_df.to_dict()


def build_lastobs_df(
        lastobsfile,
        crosswalk_file,
        time_shift           = 0,
        crosswalk_gage_field = "gages",
        crosswalk_link_field = "link",
        model_discharge_id   = "model_discharge",
        obs_discharge_id     = "discharge",
        time_idx_id          = "timeInd",
        station_id           = "stationId",
        station_idx_id       = "stationIdInd",
        time_id              = "time",
        discharge_nan        = -9999.0,
        ref_t_attr_id        = "modelTimeAtOutput",
        route_link_idx       = "feature_id",
    ):
    '''
    Constructs a DataFame of "lastobs" data used in streamflow DA routine
    "lastobs" information is just like it sounds. It is the magnitude and
    timing of the last valid observation at each gage in the model domain. 
    We use this information to jump start initialize the DA process, both 
    for forecast and AnA simulations. 
    
    Arguments
    ---------
    
    Returns
    -------
    
    Notes
    -----
    
    '''
    
    # open crosswalking file and construct dataframe relating gageID to segmentID
    with xr.open_dataset(crosswalk_file) as ds:
        gage_list = list(map(bytes.strip, ds[crosswalk_gage_field].values))
        gage_mask = list(map(bytes.isalnum, gage_list))
        gage_da   = list(map(bytes.strip, ds[crosswalk_gage_field][gage_mask].values))
        data_var_dict = {
            crosswalk_gage_field: gage_da,
            crosswalk_link_field: ds[crosswalk_link_field].values[gage_mask],
        }
        gage_link_df = pd.DataFrame(data = data_var_dict).set_index([crosswalk_gage_field])
            
    with xr.open_dataset(lastobsfile) as ds:
        
        gages    = np.char.strip(ds[station_id].values)
        
        ref_time = datetime.strptime(ds.attrs[ref_t_attr_id], "%Y-%m-%d_%H:%M:%S")
        
        last_model = ds[model_discharge_id][:,-1].values
        last_model[np.where(last_model == discharge_nan)] = np.nan
        
        last_ts = ds[time_idx_id].values[-1]
        
        df_discharge = (
            ds[obs_discharge_id].to_dataframe().                 # discharge to MultiIndex DF
            replace(to_replace = discharge_nan, value = np.nan). # replace null values with nan
            unstack(level = 0)                                   # unstack to single Index (timeInd)    
        )
        
        last_obs_index = (
            df_discharge.
            apply(pd.Series.last_valid_index).                   # index of last non-nan value, each gage
            to_numpy()                                           # to numpy array
        )
        last_obs_index = np.nan_to_num(last_obs_index, nan = last_ts).astype(int)
                        
        last_observations = []
        lastobs_times     = []
        for i, idx in enumerate(last_obs_index):
            last_observations.append(df_discharge.iloc[idx,i])
            lastobs_times.append(ds.time.values[i, idx].decode('utf-8'))
            
        last_observations = np.array(last_observations)
        lastobs_times     = pd.to_datetime(
            np.array(lastobs_times), 
            format="%Y-%m-%d_%H:%M:%S", 
            errors = 'coerce'
        )

        lastobs_times = (lastobs_times - ref_time).total_seconds()
        lastobs_times = lastobs_times - time_shift

    data_var_dict = {
        'gages'               : gages,
        'last_model_discharge': last_model,
        'time_since_lastobs'  : lastobs_times,
        'lastobs_discharge'   : last_observations
    }

    lastobs_df = (
        pd.DataFrame(data = data_var_dict).
        set_index('gages').
        join(gage_link_df, how = 'inner').
        reset_index().
        set_index(crosswalk_link_field)
    )
    lastobs_df = lastobs_df[
        [
            'gages',
            'last_model_discharge',
            'time_since_lastobs',
            'lastobs_discharge',
        ]
    ]
    
    return lastobs_df


def get_usgs_df_from_csv(usgs_csv, routelink_subset_file, index_col="link"):
    """
    routelink_subset_file - provides the gage-->segment crosswalk. Only gages that are represented in the
    crosswalk will be brought into the evaluation.
    usgs_csv - csv file with SEGMENT IDs in the left-most column labeled with "link",
                        and date-headed values from time-slice files in the format
                        "2018-09-18 00:00:00"

    It is assumed that the segment crosswalk and interpolation have both
    already been performed, so we do not need to comprehend
    the potentially non-numeric byte-strings associated with gage IDs, nor
    do we need to interpolate anything here as when we read from the timeslices.

    If that were necessary, we might use a solution such as proposed here:
    https://stackoverflow.com/a/35058538
    note that explicit typing of the index cannot be done on read and
    requires a two-line solution such as:
    ```
    df2 = pd.read_csv(usgs_csv, dtype={index_col:bytes})
    df2 = df2.set_index(index_col)
    ```
    """

    df2 = pd.read_csv(usgs_csv, index_col=index_col)

    with xr.open_dataset(routelink_subset_file) as ds:
        gage_list = list(map(bytes.strip, ds.gages.values))
        gage_mask = list(map(bytes.isdigit, gage_list))

        gage_da = ds[index_col][gage_mask].values.astype(int)

        data_var_dict = {}
        data_vars = ("gages", "to", "ascendingIndex")
        for v in data_vars:
            data_var_dict[v] = ([index_col], ds[v].values[gage_mask])
        ds = xr.Dataset(data_vars=data_var_dict, coords={index_col: gage_da})
        df = ds.to_dataframe()

    usgs_df = df.join(df2)
    usgs_df = usgs_df.drop(["gages", "ascendingIndex", "to"], axis=1)

    return usgs_df


def _read_timeslice_file(f):

    with netCDF4.Dataset(
        filename = f,
        mode = 'r',
        format = "NETCDF4"
    ) as ds:
        
        discharge = ds.variables['discharge'][:].filled(fill_value = np.nan)
        stns      = ds.variables['stationId'][:].filled(fill_value = np.nan)
        t         = ds.variables['time'][:].filled(fill_value = np.nan)
        qual      = ds.variables['discharge_quality'][:].filled(fill_value = np.nan)
        
    stationId = np.apply_along_axis(''.join, 1, stns.astype(np.str))
    time_str = np.apply_along_axis(''.join, 1, t.astype(np.str))
    stationId = np.char.strip(stationId)
    
    timeslice_observations = (pd.DataFrame({
                                'stationId' : stationId,
                                'datetime'  : time_str,
                                'discharge' : discharge
                            }).
                             set_index(['stationId', 'datetime']).
                             unstack(1, fill_value = np.nan)['discharge'])
    
    observation_quality = (pd.DataFrame({
                                'stationId' : stationId,
                                'datetime'  : time_str,
                                'quality'   : qual/100
                            }).
                             set_index(['stationId', 'datetime']).
                             unstack(1, fill_value = np.nan)['quality'])
    
    return timeslice_observations, observation_quality

def get_obs_from_timeslices(
    crosswalk_file,
    crosswalk_gage_field,
    crosswalk_dest_field,
    timeslice_files,
    qc_threshold,
    interpolation_limit,
    frequency_secs,
    t0,
    cpu_pool
):
    """
    Read observations from TimeSlice files, interpolate available observations
    and organize into a Pandas DataFrame
    
    Aguments
    --------
    - crosswalk_file               (str): full directory path to channel segment 
                                          cross walk file (RouteLink.nc)
                                          
    - crosswalk_gage_field         (str): fieldname of gage ID data in crosswalk file
    
    - crosswalk_dest_field         (str): fieldname of destination data in crosswalk 
                                          file. For streamflow DA, this is the field
                                          containing segment IDs. For reservoir DA, 
                                          this is the field containing waterbody IDs.
                                          
    - timeslice_files (list of PosixPath): Full paths to existing TimeSlice files
    
    - qc_threshold                  (int): Numerical observation quality 
                                           threshold. Observations with quality
                                           flags less than this value will be 
                                           removed and replaced witn nan.
                                           
    - interpolation_limit           (int): Maximum gap duration (minuts) over 
                                           which observations may be interpolated 
                                           
    - t0                       (datetime): Initialization time of simulation set
    
    - cpu_pool                      (int): Number of CPUs used for parallel 
                                           TimeSlice reading
    
    Returns
    -------
    - observation_df_new (Pandas DataFrame): 
    
    Notes
    -----
    The max_fill is applied when the series is being considered at a 1 minute interval
    so 14 minutes ensures no over-interpolation with 15-minute gage records, but creates
    square-wave signals at gages reporting hourly...
    therefore, we advise a 59 minute gap filling tolerance.
    
    """
    # TODO: 
    # - Parallelize this reading for speedup using netCDF4
    # - Generecize the function to read USACE or USGS data
    # - only return gages that are in the model domain (consider mask application)
        
    # open TimeSlce files, organize data into dataframes
    with Parallel(n_jobs=cpu_pool) as parallel:
        jobs = []
        for f in timeslice_files:
            jobs.append(delayed(_read_timeslice_file)(f))
        timeslice_dataframes = parallel(jobs)
        
    # create lists of observations and obs quality dataframes returned 
    # from _read_timeslice_file
    timeslice_obs_frames = []
    timeslice_qual_frames = []
    for d in timeslice_dataframes:
        timeslice_obs_frames.append(d[0])  # TimeSlice gage observation
        timeslice_qual_frames.append(d[1]) # TimeSlice observation qual
        
    # concatenate dataframes
    timeslice_obs_df  = pd.concat(timeslice_obs_frames, axis = 1)
    timeslice_qual_df = pd.concat(timeslice_qual_frames, axis = 1)   
        
    # Open the cross walk file and build a dataframe of gages and their corresponding
    # linkID locations. This dataframe will be joined to the `timeslice_obs_df`, so 
    # that gage observations are indexed by linkID, rather than gageID. 
    #
    # TODO: Re-write this section to allow the crosswalk file to be flexile. For 
    # reservoir DA, we would use a differenct cross walk (gage<>waterbody) than 
    # we would for streamflow DA. 
    #
    with xr.open_dataset(crosswalk_file) as ds:
        
        # assemble list of gage IDs as bytestrings, this list includes
        # the entire gages variable from the crosswalk file, most of which
        # are empty because most segments are not co-loacated with gages
        gage_list = list(map(bytes.strip, ds[crosswalk_gage_field].values))

        # create mask for populated (alphanumeric) gage IDs
        gage_mask = list(map(bytes.isalnum, gage_list))
        
        # apply mask to gage_list, isolating only populated gage IDs
        gage_da   = list(map(bytes.strip, ds[crosswalk_gage_field][gage_mask].values))
        
        # convert gage IDs to unicode strings
        gage_da   = np.asarray(gage_da).astype('<U15')

        # package crosswalk data into a dictionary
        data_var_dict = {
            crosswalk_gage_field: gage_da,
            crosswalk_dest_field: ds[crosswalk_dest_field].values[gage_mask]
            }
        
        # construct crosswalk dataframe, set gage ID field as index
        df = (pd.DataFrame(data = data_var_dict).
              set_index(crosswalk_gage_field))
        
    # join crosswalk data with timeslice data, indexed on crosswalk destination field
    observation_df = (df.join(timeslice_obs_df).
               reset_index().
               rename(columns={"index": crosswalk_gage_field}).
               set_index(crosswalk_dest_field).
               drop([crosswalk_gage_field], axis=1))

    observation_qual_df = (df.join(timeslice_qual_df).
               reset_index().
               rename(columns={"index": crosswalk_gage_field}).
               set_index(crosswalk_dest_field).
               drop([crosswalk_gage_field], axis=1))

    # ---- Laugh testing ------
    # screen-out erroneous qc flags
    observation_qual_df = (observation_qual_df.
                           mask(observation_qual_df < 0, np.nan).
                           mask(observation_qual_df > 1, np.nan)
                          )

    # screen-out poor quality flow observations
    observation_df = (observation_df.
                      mask(observation_qual_df < qc_threshold, np.nan).
                      mask(observation_df <= 0, np.nan)
                     )

    # ---- Interpolate USGS observations to the input frequency (frequency_secs)
    observation_df_T = observation_df.transpose()             # transpose, making time the index
    observation_df_T.index = pd.to_datetime(
        observation_df_T.index, format = "%Y-%m-%d_%H:%M:%S"  # index variable as type datetime
    )
    
    # specify resampling frequency 
    frequency = str(int(frequency_secs/60))+"min"    
    
    # interpolate and resample frequency
    observation_df_T = (observation_df_T.resample('min').
                        interpolate(
                            limit = interpolation_limit, 
                            limit_direction = 'both'
                        ).
                        resample(frequency).
                        asfreq()
                       )
    
    # re-transpose, making link the index
    observation_df_new = observation_df_T.transpose()

    return observation_df_new


def get_param_str(target_file, param):
    # TODO: remove this duplicate function
    return get_attribute(target_file, param)


# TODO: Move channel restart above usgs to keep order with execution script
def get_channel_restart_from_csv(
    channel_initial_states_file,
    index_col=0,
    default_us_flow_column="qu0",
    default_ds_flow_column="qd0",
    default_depth_column="h0",
):
    """
    channel_initial_states_file: CSV standard restart file
    index_col = 0: column/field in the input file with the segment/link id
    NOT USED YET default_us_flow_column: name used in remainder of program to refer to this column of the dataset
    NOT USED YET default_ds_flow_column: name used in remainder of program to refer to this column of the dataset
    NOT USED YET default_depth_column: name used in remainder of program to refer to this column of the dataset
    """
    q0 = pd.read_csv(channel_initial_states_file, index_col=index_col)
    q0.index = q0.index.astype(int)
    q0 = q0.sort_index(axis="index")
    return q0.astype("float32")


def get_channel_restart_from_wrf_hydro(
    channel_initial_states_file,
    crosswalk_file,
    channel_ID_column,
    us_flow_column="qlink1",
    ds_flow_column="qlink2",
    depth_column="hlink",
    default_us_flow_column="qu0",
    default_ds_flow_column="qd0",
    default_depth_column="h0",
):
    """
    channel_initial_states_file: WRF-HYDRO standard restart file
    crosswalk_file: File containing channel IDs IN THE ORDER of the Restart File
    channel_ID_column: field in the crosswalk file to assign as the index of the restart values
    us_flow_column: column in the restart file to use for upstream flow initial state
    ds_flow_column: column in the restart file to use for downstream flow initial state
    depth_column: column in the restart file to use for depth initial state
    default_us_flow_column: name used in remainder of program to refer to this column of the dataset
    default_ds_flow_column: name used in remainder of program to refer to this column of the dataset
    default_depth_column: name used in remainder of program to refer to this column of the dataset

    The Restart file gives hlink, qlink1, and qlink2 values for channels --
    the order is simply the same as that found in the Route-Link files.
    *Subnote 1*: The order of these values is NOT the order found in the CHRTOUT files,
    though the number of outputs is the same as in those files.
    *Subnote 2*: If there is no mask applied, then providing the path to the RouteLink file should
    be sufficient for the crosswalk file. If there is a mask applied (so that
    not all segments in the RouteLink file are used in the routing computations) then
    a pre-processing step will be need to provide only the relevant segments in the
    crosswalk file.
    """

    with xr.open_dataset(crosswalk_file) as xds:
        xdf = xds[channel_ID_column].to_dataframe()
    xdf = xdf.reset_index()
    xdf = xdf[[channel_ID_column]]
    with xr.open_dataset(channel_initial_states_file) as qds:
        if depth_column in qds:
            qdf2 = qds[[us_flow_column, ds_flow_column, depth_column]].to_dataframe()
        else:
            qdf2 = qds[[us_flow_column, ds_flow_column]].to_dataframe()
            qdf2[depth_column] = 0
    qdf2 = qdf2.reset_index()
    qdf2 = qdf2[[us_flow_column, ds_flow_column, depth_column]]
    qdf2.rename(
        columns={
            us_flow_column: default_us_flow_column,
            ds_flow_column: default_ds_flow_column,
            depth_column: default_depth_column,
        },
        inplace=True,
    )
    qdf2[channel_ID_column] = xdf
    qdf2 = qdf2.reset_index().set_index([channel_ID_column])

    q_initial_states = qdf2

    q_initial_states = q_initial_states.drop(columns="index")

    return q_initial_states


def read_lite_restart(
    file
):
    '''
    Open lite restart pickle files. Can open either waterbody_restart or channel_restart
    
    Arguments
    -----------
        file (string): File path to lite restart file
        
    Returns
    ----------
        df (DataFrame): restart states
        t0 (datetime): restart datetime
    '''
    
    # open pickle file to pandas DataFrame
    df = pd.read_pickle(pathlib.Path(file))
    
    # extract restart time as datetime object
    t0 = df['time'].iloc[0].to_pydatetime()
    
    return df.drop(columns = 'time') , t0
    

def write_lite_restart(
    q0, 
    waterbodies_df, 
    t0, 
    restart_parameters
):
    '''
    Save initial conditions dataframes as pickle files
    
    Arguments
    -----------
        q0 (DataFrame):
        waterbodies_df (DataFrame):
        t0 (datetime.datetime):
        restart_parameters (string):
        
    Returns
    -----------
        
    '''
    
    output_directory = restart_parameters.get('lite_restart_output_directory', None)
    if output_directory:
        
        # create pathlib object for output directory
        output_path = pathlib.Path(output_directory)
        
        # create restart filenames
        t0_str = t0.strftime("%Y%m%d%H%M")
        channel_restart_filename = 'channel_restart_'+t0_str
        waterbody_restart_filename = 'waterbody_restart_'+t0_str
        
        q0_out = q0.copy()
        q0_out['time'] = t0
        q0_out.to_pickle(pathlib.Path.joinpath(output_path, channel_restart_filename))
        LOG.debug('Dropped lite channel restart file %s' % pathlib.Path.joinpath(output_path, channel_restart_filename))

        if not waterbodies_df.empty:
            wbody_initial_states = waterbodies_df.loc[:,['qd0','h0']]
            wbody_initial_states['time'] = t0
            wbody_initial_states.to_pickle(pathlib.Path.joinpath(output_path, waterbody_restart_filename))
            LOG.debug('Dropped lite waterbody restart file %s' % pathlib.Path.joinpath(output_path, channel_restart_filename))
        else:
            LOG.debug('No lite waterbody restart file dropped becuase waterbodies are either turned off or do not exist in this domain.')
        
    else:
        LOG.error("Not writing lite restart files. No lite_restart_output_directory variable was not specified in configuration file.")
    

def write_hydro_rst(
    data,
    restart_files,
    channel_initial_states_file,
    dt_troute,
    nts_troute,
    t0,
    crosswalk_file,
    channel_ID_column,
    restart_file_dimension_var="links",
    troute_us_flow_var_name="qlink1_troute",
    troute_ds_flow_var_name="qlink2_troute",
    troute_depth_var_name="hlink_troute",
):
    """
    Write t-route flow and depth data to WRF-Hydro restart files. 

    Agruments
    ---------
        data (Data Frame): t-route simulated flow, velocity and depth data
        restart_files (list): globbed list of WRF-Hydro restart files
        channel_initial_states_file (str): WRF-HYDRO standard restart file used to initiate t-route simulation
        dt_troute (int): timestep of t-route simulation (seconds)
        nts_troute (int): number of t-route simulation timesteps
        crosswalk_file (str): File containing reservoir IDs IN THE ORDER of the Restart File
        channel_ID_column (str): field in the crosswalk file to assign as the index of the restart values
        troute_us_flow_var_name (str):
        troute_ds_flow_var_name (str):
        troute_depth_var_name (str):

    Returns
    -------
    """
    
    # Assemble the simulation tme domain
    t0_array = np.array(t0, dtype=np.datetime64)
    troute_dt = np.timedelta64(dt_troute, "s")
    troute_timestamps = (t0_array + troute_dt) + np.arange(nts_troute) * troute_dt
    
    LOG.debug('t-route intialized at %s' % (np.datetime_as_string(t0_array)))
    LOG.debug('t-route first simulated time at %s' % (np.datetime_as_string(troute_timestamps[0])))
    LOG.debug('t-route final simulated time at %s' % (np.datetime_as_string(troute_timestamps[-1])))
    
    # check the Restart_Time of each restart file in the restart directory
    LOG.debug("Looking for restart files that need to be appended")
    start = time.time()
    files_to_append = []
    write_index = []
    for f in restart_files:
        
        # open the restart file and get the Restart_Time attribute
        with xr.open_dataset(f) as ds:
            t = np.array(ds.Restart_Time.replace("_", " "), dtype=np.datetime64)
            
        # check if the Restart_Time is within the t-route model domain
        a = np.where(troute_timestamps == t)[0].tolist()
        if a:
            files_to_append.append(f)
            write_index.append(a[0])
            
    LOG.debug('Found %d restart files to append.' % len(files_to_append))
    LOG.debug('It took %s seconds to find restart files that need to be appended' % (time.time() - start))

    if len(files_to_append) == 0:
        return
    else: # contune on to append restart files
        
        LOG.debug('Retrieving index ordering used restart files')
        start = time.time()
        # extract ordered feature_ids from crosswalk file
        # TODO: Find out why we re-index this dataset when it
        # already has a segment index.
        with xr.open_dataset(crosswalk_file) as xds:
            xdf = xds[channel_ID_column].to_dataframe()
        xdf = xdf.reset_index()
        xdf = xdf[[channel_ID_column]]
        LOG.debug('Retrieving index ordering took %s seconds' % (time.time() - start))

        LOG.debug('Begining the restart writing process')
        start = time.time()
        for i, f in enumerate(files_to_append):
            
            LOG.debug('Preparing data for- and writing data to- %s' % f)
            # extract and reindex depth data
            qtrt = (
                data.iloc[:,::3]
                .iloc[:, a[i]]
                .reindex(xdf.link)
                .to_numpy()
                .astype("float32")
                .reshape(len(xdf.link,))
            )

            # extract and reindex depth data
            htrt = (
                data.iloc[:, 2::3]
                .iloc[:, a[i]]
                .reindex(xdf.link)
                .to_numpy()
                .astype("float32")
                .reshape(len(xdf.link,))
            )
            
            # assemble variables dictionary with content to be written out
            variables = {
                troute_us_flow_var_name: (qtrt, restart_file_dimension_var, {}),
                troute_ds_flow_var_name: (qtrt, restart_file_dimension_var, {}),
                troute_depth_var_name: (htrt, restart_file_dimension_var, {}),
            }
            
            # append restart data to netcdf restart files
            write_to_netcdf(f, variables)
        
        LOG.debug('Restart writing process completed in % seconds.' % (time.time() - start))
            
def get_reservoir_restart_from_wrf_hydro(
    waterbody_intial_states_file,
    crosswalk_file,
    waterbody_ID_field,
    crosswalk_filter_file=None,
    crosswalk_filter_file_field=None,
    waterbody_flow_column="qlakeo",
    waterbody_depth_column="resht",
    default_waterbody_flow_column="qd0",
    default_waterbody_depth_column="h0",
):
    """
    waterbody_intial_states_file: WRF-HYDRO standard restart file
    crosswalk_file: File containing reservoir IDs IN THE ORDER of the Restart File
    waterbody_ID_field: field in the crosswalk file to assign as the index of the restart values
    crosswalk_filter_file: file containing only reservoir ids needed in the crosswalk file (see notes below)
    crosswalk_filter_file_field: waterbody id field in crosswalk filter file
    waterbody_flow_column: column in the restart file to use for upstream flow initial state
    waterbody_depth_column: column in the restart file to use for downstream flow initial state
    default_waterbody_flow_column: name used in remainder of program to refer to this column of the dataset
    default_waterbody_depth_column: name used in remainder of program to refer to this column of the dataset

    The Restart file gives qlakeo and resht values for waterbodies.
    The order of values in the file is the same as the order in the LAKEPARM file from WRF-Hydro.
    However, there are many instances where only a subset of waterbodies described in the lakeparm
    file are used. In these cases, a filter file must be provided which specifies
    which of the reservoirs in the crosswalk file are to be used.
    """

    with xr.open_dataset(crosswalk_file) as xds:
        X = xds[waterbody_ID_field]

        if crosswalk_filter_file:
            with xr.open_dataset(crosswalk_filter_file) as fds:
                xdf = X.loc[X.isin(fds[crosswalk_filter_file_field])].to_dataframe()
        else:
            xdf = X.to_dataframe()

    xdf = xdf.reset_index()[waterbody_ID_field]

    # read initial states from r&r output
    with xr.open_dataset(waterbody_intial_states_file) as resds:
        resdf = resds[[waterbody_flow_column, waterbody_depth_column]].to_dataframe()
    resdf = resdf.reset_index()
    resdf = resdf[[waterbody_flow_column, waterbody_depth_column]]
    resdf.rename(
        columns={
            waterbody_flow_column: default_waterbody_flow_column,
            waterbody_depth_column: default_waterbody_depth_column,
        },
        inplace=True,
    )

    mod = resdf.join(xdf).reset_index().set_index(waterbody_ID_field)
    init_waterbody_states = mod

    return init_waterbody_states


def build_coastal_dataframe(coastal_boundary_elev):
    coastal_df = pd.read_csv(
        coastal_boundary_elev, sep="  ", header=None, engine="python"
    )
    return coastal_df


def build_coastal_ncdf_dataframe(coastal_ncdf):
    with xr.open_dataset(coastal_ncdf) as ds:
        coastal_ncdf_df = ds[["elev", "depth"]]
        return coastal_ncdf_df.to_dataframe()


def lastobs_df_output(
    lastobs_df,
    dt,
    nts,
    t0,
    gages,
    lastobs_output_folder=False,
):

    # join gageIDs to lastobs_df
    lastobs_df = lastobs_df.join(gages)

    # timestamp of last simulation timestep
    modelTimeAtOutput = t0 + timedelta(seconds = nts * dt)
    modelTimeAtOutput_str = modelTimeAtOutput.strftime('%Y-%m-%d_%H:%M:%S')

    # timestamp of last observation
    var = [timedelta(seconds=d) for d in lastobs_df.time_since_lastobs.fillna(0)]
    lastobs_timestamp = [modelTimeAtOutput - d for d in var]
    lastobs_timestamp_str = [d.strftime('%Y-%m-%d_%H:%M:%S') for d in lastobs_timestamp]

    # create xarray Dataset similarly structured to WRF-generated lastobs netcdf files
    ds = xr.Dataset(
        {
            "stationId": (["stationIdInd"], lastobs_df["gages"].to_numpy(dtype = '|S15')),
            "time": (["stationIdInd"], np.asarray(lastobs_timestamp_str,dtype = '|S19')),
            "discharge": (["stationIdInd"], lastobs_df["lastobs_discharge"].to_numpy()),
        }
    )
    ds.attrs["modelTimeAtOutput"] = "example attribute"

    # write-out LastObs file as netcdf
    output_path = pathlib.Path(lastobs_output_folder + "/nudgingLastObs." + modelTimeAtOutput_str + ".nc").resolve()
    ds.to_netcdf(str(output_path))
