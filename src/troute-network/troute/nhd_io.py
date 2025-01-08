import zipfile
import json
import sys
import math
import pathlib
import logging
from datetime import *
import time

import yaml
import xarray as xr
import pandas as pd
import numpy as np
from toolz import compose
import netCDF4
from joblib import delayed, Parallel
from cftime import date2num
import dateutil.parser as dparser
from datetime import datetime, timedelta

from troute.nhd_network import reverse_dict

LOG = logging.getLogger('')

def read_netcdf(geo_file_path):
    '''
    Open a netcdf file with xarray and convert to dataframe
    
    Arguments
    ---------
    geo_file_path (str or pathlib.Path): netCDF filepath
    
    Returns
    -------
    ds.to_dataframe() (DataFrame): netCDF contents
    
    Notes
    -----
    - When handling large volumes of netCDF files, xarray is not the most efficient.
    
    '''
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

def read(geo_file_path, layer_string=None, driver_string=None):
    if geo_file_path.suffix == ".nc":
        return read_netcdf(geo_file_path)
    else:
        raise RuntimeError("Cannot read file {}. Unsupported file type {}.".format(geo_file_path, geo_file_path.suffix))


def read_mask(path, layer_string=None):
    return read_csv(path, header=None, layer_string=layer_string)

def read_config_file(custom_input_file):
    '''
    Read-in data from user-created configuration file.
    
    Arguments
    ---------
    custom_input_file (str): configuration filepath, either .yaml or .json
    
    Returns
    -------
    log_parameters               (dict): Input parameters re logging
    preprocessing_parameters     (dict): Input parameters re preprocessing
    supernetwork_parameters      (dict): Input parameters re network extent
    waterbody_parameters         (dict): Input parameters re waterbodies
    compute_parameters           (dict): Input parameters re computation settings
    forcing_parameters           (dict): Input parameters re model forcings
    restart_parameters           (dict): Input parameters re model restart
    hybrid_parameters            (dict): Input parameters re diffusive wave model
    output_parameters            (dict): Input parameters re output writing
    parity_parameters            (dict): Input parameters re parity assessment
    data_assimilation_parameters (dict): Input parameters re data assimilation

    '''
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
    # add attributes when HYfeature network is selected
    if supernetwork_parameters['geo_file_path'][-4:] == "gpkg":
        supernetwork_parameters["title_string"]       = "HY_Features Test"
        supernetwork_parameters["geo_file_path"]      = supernetwork_parameters['geo_file_path']
        supernetwork_parameters["flowpath_edge_list"] = None    
        routelink_attr = {
                        #link????
                        "key": "id",
                        "downstream": "toid",
                        "dx": "length_m",
                        "n": "n",  # TODO: rename to `manningn`
                        "ncc": "nCC",  # TODO: rename to `mannningncc`
                        "s0": "So",
                        "bw": "BtmWdth",  # TODO: rename to `bottomwidth`
                        #waterbody: "NHDWaterbodyComID",
                        "tw": "TopWdth",  # TODO: rename to `topwidth`
                        "twcc": "TopWdthCC",  # TODO: rename to `topwidthcc`
                        "alt": "alt",
                        "musk": "MusK",
                        "musx": "MusX",
                        "cs": "ChSlp"  # TODO: rename to `sideslope`
                        }
        if not supernetwork_parameters.get('columns',None):
            supernetwork_parameters["columns"]         = routelink_attr 
        supernetwork_parameters["waterbody_null_code"] = -9999
        supernetwork_parameters["terminal_code"]       =  0
        supernetwork_parameters["driver_string"]       = "NetCDF"
        supernetwork_parameters["layer_string"]        = 0
        
    preprocessing_parameters = network_topology_parameters.get(
        "preprocessing_parameters", {}
    )        
    #waterbody_parameters = network_topology_parameters.get(
    #    "waterbody_parameters", None
    #)
    waterbody_parameters = network_topology_parameters.get(
        "waterbody_parameters", {}
    )
    compute_parameters = data.get("compute_parameters", {})
    forcing_parameters = compute_parameters.get("forcing_parameters", {})
    restart_parameters = compute_parameters.get("restart_parameters", {})
    hybrid_parameters = compute_parameters.get("hybrid_parameters", {})
    data_assimilation_parameters = compute_parameters.get(
        "data_assimilation_parameters", {}
    )
    output_parameters = data.get("output_parameters", {})
    parity_parameters = output_parameters.get("wrf_hydro_parity_check", {})

    return (
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
    )

def read_diffusive_domain(domain_file):
    '''
    Read diffusive domain data from .ymal or .json file.
    
    Arguments
    ---------
    domain_file (str or pathlib.Path): Path of diffusive domain file
    
    Returns
    -------
    data (dict int: [int]): domain tailwater segments: list of segments in domain 
                            (includeing tailwater segment) 
    
    '''
    if domain_file[-4:] == "yaml":
        with open(domain_file) as domain:
            data = yaml.load(domain, Loader=yaml.SafeLoader)
    else:
        with open(domain_file) as domain:
            data = json.load(domain)
            
    return data

def read_coastal_boundary_domain(domain_file):
    '''
    Read coastal boundary domain from .ymal or .json file.
    
    Arguments
    ---------
    domain_file (str or pathlib.Path): Path of coastal boundary domain file
    
    Returns
    -------
    data (dict int: int): diffusive domain tailwater segments: coastal domain segments 
                        
    
    '''

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

def read_lakeparm(
    parm_file, 
    lake_index_field="lake_id", 
    lake_id_mask=None
):
    """
    Reads LAKEPARM file and prepares a dataframe, filtered
    to the relevant reservoirs, to provide the parameters
    for level-pool reservoir computation.
    Completely replaces the read_waterbody_df function from prior versions
    of the v02 routing code.
    
    Arguments:
    ----------
    parm_file (str or pathlib.Path): Path to LAKEPARM file
    lake_index_field           (str): waterbody id dimension name (default: 'lake_id')
    lake_id_mask       (list of int): lake ids in simulation domain
    
    Returns:
    df1 (DataFrame):
    """

    # TODO: avoid or parameterize "feature_id" or ... return to name-blind dataframe version
    with xr.open_dataset(parm_file) as ds:
        ds = ds.swap_dims({"feature_id": lake_index_field})
        lake_id_values = list(lake_id_mask)
        ds_filtered = ds.where(ds.lake_id.isin(lake_id_values), drop=True)
        df1 = ds_filtered.to_dataframe()
        # df1 = ds.sel({lake_index_field: list(lake_id_mask)}).to_dataframe()

    df1 = df1.sort_index(axis="index")
    return df1


def read_reservoir_parameter_file(
    reservoir_parameter_file, 
    usgs_hybrid,
    usace_hybrid,
    rfc_forecast,
    lake_index_field="lake_id", 
    usgs_gage_id_field = "usgs_gage_id",
    usgs_lake_id_field = "usgs_lake_id",
    usace_gage_id_field = "usace_gage_id",
    usace_lake_id_field = "usace_lake_id",
    lake_id_mask=None,
):

    """
    Reads reservoir parameter file, which is separate from the LAKEPARM file.
    Extracts reservoir "type" codes and returns in a DataFrame
    type 1: Levelool
    type 2: USGS Hybrid Persistence
    type 3: USACE Hybrid Persistence
    type 4: RFC
    This function is only called if Hybrid Persistence or RFC type reservoirs
    are active.
    
    Arguments
    ---------
    - reservoir_parameter_file (str): full file path of the reservoir parameter
                                      file
    
    - usgs_hybrid          (boolean): If True, then USGS Hybrid DA will be coded
    
    - usace_hybrid         (boolean): If True, then USACE Hybrid DA will be coded
    
    - rfc_forecast         (boolean): If True, then RFC Forecast DA will be coded
    
    - lake_index_field         (str): field containing lake IDs in reservoir 
                                      parameter file
    
    - lake_id_mask     (dict_values): Waterbody IDs in the model domain 
    
    Returns
    -------
    - df1 (Pandas DataFrame): Reservoir type codes, indexed by lake_id
    
    Notes
    -----
    
    """
    with xr.open_dataset(reservoir_parameter_file) as ds:
        ds = ds.swap_dims({"feature_id": lake_index_field})
        ds_new = ds["reservoir_type"]
        df1 = ds_new.sel({lake_index_field: list(lake_id_mask)}).to_dataframe()
        
        ds_vars = [i for i in ds.data_vars] 
        
        if (usgs_gage_id_field in ds_vars) and (usgs_lake_id_field in ds_vars):
            usgs_crosswalk = pd.DataFrame(
                data = ds[usgs_gage_id_field].to_numpy(), 
                index = ds[usgs_lake_id_field].to_numpy(), 
                columns = [usgs_gage_id_field]
            )
            usgs_crosswalk.index.name = usgs_lake_id_field
            usgs_crosswalk[usgs_gage_id_field] = usgs_crosswalk[usgs_gage_id_field].apply(lambda x: x.decode('utf-8')).str.strip()
        else:
            usgs_crosswalk = None
        
        if (usace_gage_id_field in ds_vars) and (usace_lake_id_field in ds_vars):
            usace_crosswalk = pd.DataFrame(
                data = ds[usace_gage_id_field].to_numpy(), 
                index = ds[usace_lake_id_field].to_numpy(), 
                columns = [usace_gage_id_field]
            )
            usace_crosswalk.index.name = usace_lake_id_field
            usace_crosswalk[usace_gage_id_field] = usace_crosswalk[usace_gage_id_field].apply(lambda x: x.decode('utf-8')).str.strip()
        else:
            usace_crosswalk = None
        
    # drop duplicate indices
    df1 = (df1.reset_index()
           .drop_duplicates(subset="lake_id")
           .set_index("lake_id")
           .sort_index()
          )
    
    # recode to levelpool (1) for reservoir DA types set to false
    if usgs_hybrid == False:
        df1[df1['reservoir_type'] == 2] = 1
    if usace_hybrid == False:
        df1[df1['reservoir_type'] == 3] = 1
    if rfc_forecast == False:
        df1[df1['reservoir_type'] == 4] = 1
    
    return df1, usgs_crosswalk, usace_crosswalk


def get_ql_from_csv(nhd_input_file, index_col=0):
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
    Lateral inflows to any segment are calculated as the sum of qBucket
    and qSfcLatRunoff variables. In the event that one or both of these
    variables are not available, then the q_lateral variable is used. In
    the event that none of these variables are available an error will be
    thrown and the simulation will stop. 
    
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
        if qbucket_varname in all_variables and runoff_varname in all_variables:
            dat = ds.variables[qbucket_varname][:].filled(fill_value = 0.0) + \
                ds.variables[runoff_varname][:].filled(fill_value = 0.0)
        else:
            dat = ds.variables[qlateral_varname][:].filled(fill_value = 0.0)
        
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
    # TODO: for and if statements are improvised just in case when link of gage location is not in flowveldepth, which should be fixed
    link_na = []
    for link in link_gage_df.index:
        if link not in flowveldepth.index:
            link_na.append(link)
    link_gage_df_nona = link_gage_df.drop(link_na)

    # array of segment linkIDs at gage locations. Results from these segments will be written
    #gage_feature_id = link_gage_df.index.to_numpy(dtype = "int64")    
    gage_feature_id = link_gage_df_nona.index.to_numpy(dtype = "int64")
    
    # array of simulated flow data at gage locations    
    #gage_flow_data = flowveldepth.loc[link_gage_df.index].iloc[:,::3].to_numpy(dtype="float32") 
    gage_flow_data = flowveldepth.loc[link_gage_df_nona.index].iloc[:,::3].to_numpy(dtype="float32") 
    
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
                datatype = 'int64',
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
            y[:] = gage_flow_data.T
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

            flow_new = gage_flow_data.T
            
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

        with xr.open_dataset(chrtout_files[0],engine='netcdf4') as ds:
            newindex = ds.feature_id.values
            
        qtrt = flow.reindex(newindex).to_numpy().astype("float32")
        
        LOG.debug("Reindexing the flow DataFrame took %s seconds." % (time.time() - start))
        
        LOG.debug("Writing t-route data to %d CHRTOUT files" % (nfiles_to_write))
        start = time.time()
        try:
            with Parallel(n_jobs=cpu_pool) as parallel:
            
                jobs = []
                for i, f in enumerate(chrtout_files[:nfiles_to_write]):

                    s = time.time()
                    variables = {
                        varname: (qtrt[:,i], dim, attrs)
                    }
                    jobs.append(delayed(write_to_netcdf)(f, variables))
                    #LOG.debug("Writing %s." % (f))
                    
                parallel(jobs)
        except:
            for i, f in enumerate(chrtout_files[:nfiles_to_write]):
                s = time.time()
                variables = {
                    varname: (qtrt[:i], dim, attrs)
                }
                write_to_netcdf(f, variables)
                LOG.debug("Writing %s." % (f))
               
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

def build_lastobs_df(
        lastobsfile,
        crosswalk_file,
        time_shift           = 0,
        crosswalk_gage_field = "gages",
        crosswalk_link_field = "link",
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
   
    if discharge.size != 0 and stns.size != 0 and t.size != 0:        
        stationId = np.apply_along_axis(''.join, 1, stns.astype(str))
        time_str = np.apply_along_axis(''.join, 1, t.astype(str))
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
    else:
        timeslice_observations = pd.DataFrame()
        observation_quality = pd.DataFrame()
    return timeslice_observations, observation_quality

def _interpolate_one(df, interpolation_limit, frequency):
    
    interp_out = (df.resample('min').
                        interpolate(
                            limit = interpolation_limit, 
                            limit_direction = 'both'
                        ).
                        resample(frequency).
                        asfreq().
                        to_numpy()
                       )
    return interp_out

def get_obs_from_timeslices(
    crosswalk_df,
    crosswalk_gage_field,
    crosswalk_dest_field,
    timeslice_files,
    qc_threshold,
    interpolation_limit,
    frequency_secs,
    t0,
    cpu_pool, 
):
    """
    Read observations from TimeSlice files, interpolate available observations
    and organize into a Pandas DataFrame
    
    Aguments
    --------                                          
    - crosswalk_gage_field         (str): fieldname of gage ID data in crosswalk dataframe
    
    - crosswalk_dest_field         (str): fieldname of destination data in link_gage_df. 
                                          For streamflow DA, this is the field
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
                                           TimeSlice reading and interolation
    
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

    all_empty = all(df.empty for tuple in timeslice_dataframes for df in tuple)
    if all_empty:
        LOG.debug(f'{crosswalk_gage_field} DataFrames is empty, check timeslice files.')
        return pd.DataFrame()
       
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
      
    # Link <> gage crosswalk data
    df = crosswalk_df.reset_index()
    df[crosswalk_gage_field] = np.asarray(df[crosswalk_gage_field]).astype('<U15')
    df = df.set_index(crosswalk_gage_field)
    df.index = df.index.str.strip()
    # join crosswalk data with timeslice data, indexed on crosswalk destination field
    observation_df = (df.join(timeslice_obs_df).
               reset_index().
               set_index(crosswalk_dest_field).
               select_dtypes(include='number'))

    observation_qual_df = (df.join(timeslice_qual_df).
               reset_index().
               set_index(crosswalk_dest_field).
               select_dtypes(include='number'))

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
    buffer_df = observation_df_T.resample(frequency).asfreq()
    with Parallel(n_jobs=cpu_pool) as parallel:
        
        jobs = []
        interp_chunks = ()
        step = 200
        for a, i in enumerate(range(0, len(observation_df_T.columns), step)):
            
            start = i
            if (i+step-1) < buffer_df.shape[1]:
                stop = i+(step)
            else:
                stop = buffer_df.shape[1]
                
            jobs.append(
                delayed(_interpolate_one)(observation_df_T.iloc[:,start:stop], interpolation_limit, frequency)
            )
            
        interp_chunks = parallel(jobs)

    observation_df_T = pd.DataFrame(
        data = np.concatenate(interp_chunks, axis = 1), 
        columns = buffer_df.columns, 
        index = buffer_df.index
    )
    
    # re-transpose, making link the index
    observation_df_new = observation_df_T.transpose()

    return observation_df_new


def get_GL_obs_from_timeslices(
    crosswalk_df,
    timeslice_files,
    crosswalk_gage_field='gages',
    crosswalk_dest_field='link',
    qc_threshold=1,
    cpu_pool=1, 
):
    """
    Read observations from TimeSlice files and organize into a Pandas DataFrame.
    This is specific to observations needed for Great Lakes data assimilation, 
    so no interpolation happens.
    
    Aguments
    --------                                          
    - crosswalk_gage_field         (str): fieldname of gage ID data in crosswalk dataframe
    - crosswalk_dest_field         (str): fieldname of destination data in link_gage_df. 
                                          For streamflow DA, this is the field
                                          containing segment IDs. For reservoir DA, 
                                          this is the field containing waterbody IDs.                                 
    - timeslice_files (list of PosixPath): Full paths to existing TimeSlice files
    - qc_threshold                  (int): Numerical observation quality 
                                           threshold. Observations with quality
                                           flags less than this value will be 
                                           removed and replaced witn nan.                                    
    - cpu_pool                      (int): Number of CPUs used for parallel 
                                           TimeSlice reading and interolation
    
    Returns
    -------
    - observation_df_new (Pandas DataFrame): 
    
    Notes
    -----
    """
    # open TimeSlce files, organize data into dataframes
    with Parallel(n_jobs=cpu_pool) as parallel:
        jobs = []
        for f in timeslice_files:
            jobs.append(delayed(_read_timeslice_file)(f))
        timeslice_dataframes = parallel(jobs)

    all_empty = all(df.empty for tuple in timeslice_dataframes for df in tuple)
    if all_empty:
        LOG.debug(f'{crosswalk_gage_field} DataFrames is empty, check timeslice files.')
        return pd.DataFrame()

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

    # Link <> gage crosswalk data
    df = crosswalk_df.reset_index()
    df = df.set_index(crosswalk_gage_field)
    df.index = df.index.str.strip()
    # join crosswalk data with timeslice data, indexed on crosswalk destination field
    observation_df = (df.join(timeslice_obs_df).
               reset_index().
               set_index(crosswalk_dest_field).
               select_dtypes(include='number'))

    observation_qual_df = (df.join(timeslice_qual_df).
               reset_index().
               set_index(crosswalk_dest_field).
               select_dtypes(include='number'))

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

    return observation_df


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


def build_coastal_ncdf_dataframe(
                                coastal_files, 
                                coastal_boundary_domain,
                                ):

    # retrieve coastal elevation, topo depth, and temporal data
    ds = netCDF4.Dataset(filename = coastal_files,  mode = 'r', format = "NETCDF4")
    
    tws = list(coastal_boundary_domain.keys())
    coastal_boundary_nodes = list(coastal_boundary_domain.values())
    
    elev_NAVD88 = ds.variables['elev'][:, coastal_boundary_nodes].filled(fill_value = np.nan)
    depth_bathy = ds.variables['depth'][coastal_boundary_nodes].filled(fill_value = np.nan)
    timesteps   = ds.variables['time'][:]
    if len(timesteps) > 1:
        dt_schism = timesteps[1]-timesteps[0]
    else:
        raise RuntimeError("schism provided less than 2 time steps")
    
    start_date = ds.variables['time'].units
    start_date   = dparser.parse(start_date,fuzzy=True)
    dt_timeslice = timedelta(minutes=dt_schism/60.0)
    tfin         =  start_date + dt_timeslice*len(timesteps)
    timestamps   = pd.date_range(start_date, tfin, freq=dt_timeslice)
    timestamps   = timestamps.strftime('%Y-%m-%d %H:%M:%S')
    
    # create a dataframe of water depth at coastal domain nodes
    timeslice_schism_list=[]
    for t in range(0, len(timesteps)+1):
        timeslice= np.full(len(tws), timestamps[t])
        if t==0:
            depth = np.nan
        else:   
            depth =  elev_NAVD88[t-1,:] + depth_bathy
        
        timeslice_schism  = (pd.DataFrame({
                                'stationId' : tws,
                                'datetime'  : timeslice,
                                'depth'     : depth
                            }).
                             set_index(['stationId', 'datetime']).
                             unstack(1, fill_value = np.nan)['depth'])

        timeslice_schism_list.append(timeslice_schism)
    
    coastal_boundary_depth_df = pd.concat(timeslice_schism_list, axis=1, ignore_index=False)
    
    # linearly extrapolate depth value at start date
    coastal_boundary_depth_df.iloc[:,0] = 2.0*coastal_boundary_depth_df.iloc[:,1] - coastal_boundary_depth_df.iloc[:,2]   

    return coastal_boundary_depth_df    
 
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
    lastobs_timestamp_str_array = np.asarray(lastobs_timestamp_str,dtype = '|S19').reshape(len(lastobs_timestamp_str),1)

    # create xarray Dataset similarly structured to WRF-generated lastobs netcdf files
    ds = xr.Dataset(
        {
            "stationId": (["stationIdInd"], lastobs_df["gages"].to_numpy(dtype = '|S15')),
            "time": (["stationIdInd", "timeInd"], np.asarray(lastobs_timestamp_str_array,dtype = '|S19')),
            "discharge": (["stationIdInd", "timeInd"], lastobs_df["lastobs_discharge"].to_numpy().reshape(len(lastobs_df["lastobs_discharge"]),1)),
        }
    )
    ds.attrs["modelTimeAtOutput"] = modelTimeAtOutput_str

    # write-out LastObs file as netcdf
    if isinstance(lastobs_output_folder, pathlib.Path):
        lastobs_output_folder = str(lastobs_output_folder)
    output_path = pathlib.Path(lastobs_output_folder + "/nudgingLastObs." + modelTimeAtOutput_str + ".nc").resolve()
    ds.to_netcdf(str(output_path))

def write_waterbody_netcdf(
    wbdy_filepath, 
    i_df,
    q_df,
    d_df,
    waterbodies_df,
    waterbody_types_df,
    t0, 
    dt, 
    nts,
    time_index
):
    
    '''
    Write results of waterbodies to netcdf. 
    Written as one file per time step.
    
    Arguments
    -------------
        wbdy_filepath (Path or string) - directory where files will be created
        wbdy_df (DataFrame) - t-route waterbody inflow, outflow, and depth results
        waterbodies_df (DataFrame) - linkIDs of waterbodies in network
        t0 (datetime) - initial time
        dt (int) - timestep duration (seconds)
        nts (int) - number of timesteps in simulation
        time_index (ind) - specific timestep for this file
        
    Returns
    -------------
    
    '''
    i_df.index.name = 'lake_id'
    i_df.columns = ['i']
    i_df = i_df.sort_index()
    q_df.index.name = 'lake_id'
    q_df.columns = ['q']
    q_df = q_df.sort_index()
    d_df.index.name = 'lake_id'
    d_df.columns = ['d']
    d_df = d_df.sort_index()
    
    # array of segment linkIDs at gage locations. Results from these segments will be written
    waterbodies_df = waterbodies_df[['lat','lon','crs']].sort_index()
    wbdy_feature_id = waterbodies_df.index.to_numpy(dtype = "int64")

    # dataframe of waterbody types
    waterbody_types_df = waterbody_types_df.sort_index()
    wbdy_type = waterbody_types_df.reservoir_type.to_numpy(dtype = 'int32')

    # array of simulation time
    wbdy_time = [t0 + timedelta(seconds = (time_index + 1) * dt)]
    
    if wbdy_filepath:
        
        # open netCDF4 Dataset in write mode
        with netCDF4.Dataset(
            filename = str(wbdy_filepath) + '/' + str(wbdy_time[0].strftime('%Y%m%d%H%M')) + '.LAKEOUT.nc',
            mode = 'w',
            format = "NETCDF4"
        ) as f:

            # =========== DIMENSIONS ===============
            _ = f.createDimension("time", 1)
            _ = f.createDimension("feature_id", len(wbdy_feature_id))
            _ = f.createDimension("reference_time", 1)
            _ = f.createDimension("latitude", len(wbdy_feature_id))
            _ = f.createDimension("longitude", len(wbdy_feature_id))

            # =========== time VARIABLE ===============
            TIME = f.createVariable(
                varname = "time",
                datatype = 'int32',
                dimensions = ("time",),
            )
            TIME[:] = date2num(
                wbdy_time, 
                units = "minutes since 1970-01-01 00:00:00 UTC",
                calendar = "gregorian"
            )
            f['time'].setncatts(
                {
                    'long_name': 'valid output time',
                    'standard_name': 'time',
                    'valid_min': date2num(
                        t0, 
                        units = "minutes since 1970-01-01 00:00:00 UTC",
                        calendar = "gregorian"
                    ),
                    'valid_max': date2num(
                        wbdy_time[0], 
                        units = "minutes since 1970-01-01 00:00:00 UTC",
                        calendar = "gregorian"
                    ),
                    'units': 'minutes since 1970-01-01T00:00:00+00:00',
                    'calendar': 'proleptic_gregorian'
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
                    'long_name': 'model initialization time',
                    'standard_name': 'forecast_reference_time',
                    'units': 'minutes since 1970-01-01T00:00:00+00:00',
                    'calendar': 'proleptic_gregorian'
                }
            )

            # =========== feature_id VARIABLE ===============
            FEATURE_ID = f.createVariable(
                varname = "feature_id",
                datatype = 'int64',
                dimensions = ("feature_id",),
            )
            FEATURE_ID[:] = wbdy_feature_id
            f['feature_id'].setncatts(
                {
                    'long_name': 'Lake ComID',
                    'comment': '',
                    'cf_role:': 'timeseries_id'
                }
            )

            # =========== latitude VARIABLE ===============
            LATITUDE = f.createVariable(
                varname = "latitude",
                datatype = 'f4',
                dimensions = ("feature_id",),
                fill_value = np.nan
            )
            LATITUDE[:] = waterbodies_df.lat.tolist()
            f['latitude'].setncatts(
                {
                    'long_name': 'Lake latitude',
                    'standard_name': 'latitude',
                    'units': 'degrees_north'
                }
            )
            
            # =========== longitude VARIABLE ===============
            LONGITUDE = f.createVariable(
                varname = "longitude",
                datatype = 'f4',
                dimensions = ("feature_id",),
                fill_value = np.nan
            )
            LONGITUDE[:] = waterbodies_df.lon.tolist()
            f['longitude'].setncatts(
                {
                    'long_name': 'Lake longitude',
                    'standard_name': 'longitude',
                    'units': 'degrees_east'
                }
            )
            
            # =========== reservoir type VARIABLE ===============            
            res_type = f.createVariable(
                    varname = "reservoir_type",
                    datatype = "i4",
                    dimensions = ("feature_id")
                )
            res_type[:] = wbdy_type
            f['reservoir_type'].setncatts(
                {
                    'coordinates': 'latitude longitude',
                    'long_name': 'reservoir_type',
                    'flag_values': [1, 2, 3, 4],
                    'flag_meanings': 'Level_pool USGS-persistence USACE-persistence RFC-forecasts'
                }
            )
            
            # =========== crs VARIABLE ===============            
            crs = f.createVariable(
                    varname = "crs",
                    datatype = "S1",
                    dimensions = ()
                )
            crs[:] = np.array(waterbodies_df['crs'].iat[0],dtype = '|S1')
            f['crs'].setncatts(
                {
                    'transform_name': 'latitude longitude',
                    'grid_mapping_name': 'latitude longitude',
                    'esri_pe_string': 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]];-400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119521E-09;0.001;0.001;IsHighPrecision',
                    'spatial_ref': 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]];-400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119521E-09;0.001;0.001;IsHighPrecision',
                    'long_name': 'CRS definition',
                    'longitude_of_prime_meridian': 0.0,
                    '_CoordinateAxes': 'latitude longitude',
                    'semi_major_axis': 6378137.0,
                    'semi_minor_axis': 6356752.5,
                    'inverse_flattening': 298.25723
                }
            )
            
            # =========== inflow VARIABLE ===============            
            inflow = f.createVariable(
                    varname = "inflow",
                    datatype = "f4",
                    dimensions = ("feature_id"),
                    fill_value = -999900
                )

            inflow[:] = i_df.i.tolist()
            f['inflow'].setncatts(
                {
                    'long_name': 'Lake Inflow',
                    'units': 'm3 s-1',
                    'grid_mapping': 'crs',
                    'valid_range': [-1000000,1000000],
                    'coordinates': 'latitude longitude',
                    'add_offset': 0.0,
                    'scale_factor': 0.01,
                    'missing_value': -999900
                }
            )

            # =========== outflow VARIABLE ===============            
            outflow = f.createVariable(
                    varname = "outflow",
                    datatype = "f4",
                    dimensions = ("feature_id"),
                    fill_value = np.nan
                )
            outflow[:] = q_df.q.tolist()
            f['outflow'].setncatts(
                {
                    'long_name': 'Lake Outflow',
                    'units': 'm3 s-1',
                    'grid_mapping': 'crs',
                    'valid_range': [-1000000,1000000],
                    'coordinates': 'latitude longitude',
                    'add_offset': 0.0,
                    'scale_factor': 0.01,
                    'missing_value': -999900
                }
            )

            # =========== depth VARIABLE ===============            
            depth = f.createVariable(
                    varname = "water_sfc_elev",
                    datatype = "f4",
                    dimensions = ("feature_id"),
                    fill_value = np.nan
                )
            depth[:] = d_df.d.tolist()
            f['water_sfc_elev'].setncatts(
                {
                    'long_name': 'Water Surface Elevation',
                    'units': 'm',
                    'comment': 'If reservoir_type = 4, water_sfc_elev is invalid because this value corresponds only to level pool',
                    'coordinates': 'latitude longitude'
                }
            )

            # =========== GLOBAL ATTRIBUTES ===============  
            f.setncatts(
                {
                    'TITLE': 'OUTPUT FROM T-ROUTE',
                    'featureType': 'timeSeries',
                    'proj4': '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@',
                    'model_initialization_time': t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    'station_dimension': 'lake_id',
                    'model_output_valid_time': wbdy_time[0].strftime('%Y-%m-%d_%H:%M:%S'),
                    'model_total_valid_times': nts,
                    'Conventions': 'CF-1.6',
                    'code_version': '',
                    'model_output_type': 'reservoir',
                    'model_configuration': ''
                }
            )


def write_flowveldepth_csv_pkl(stream_output_directory, file_name,
                              flow, velocity, depth, nudge_df, timestamps,
                              t0, poi_crosswalk):
    
    formatted_times = [str(timedelta(seconds=t)) for t in timestamps]
    
    df_list = []
    
    for i in range(len(formatted_times)):
        # Construct a temporary DataFrame for each column set
        df_temp = pd.DataFrame({
            't0': str(t0),
            'time': formatted_times[i],
            'flow': flow.iloc[:, i],
            'velocity': velocity.iloc[:, i],
            'depth': depth.iloc[:, i],
            'nudge': nudge_df.iloc[:, i]  
        }, index=flow.index)
        df_list.append(df_temp)

    # Concatenate all temporary DataFrames vertically
    df = pd.concat(df_list)
    
    df['current_time'] = pd.to_datetime(df['t0']) + pd.to_timedelta(df['time'])
    df = df[['current_time', 'flow', 'velocity', 'depth', 'nudge']]

    # File format handling
    file_format = file_name.split('.')[-1]
    if file_format == 'csv':
        df.to_csv(f"{stream_output_directory}/{file_name}", index=True)
    elif file_format == 'pkl':
        df.to_pickle(f"{stream_output_directory}/{file_name}")

def write_flowveldepth_netcdf(stream_output_directory, file_name,
                              flow, velocity, depth, nudge_df, timestamps,
                              t0, poi_crosswalk):
   
    # Find max lenght of string variables:
    max_Type_str_len = max(len(str(x)) for x in flow.index.get_level_values('Type'))
    max_poi_str_len = max(len(str(x)) for x in poi_crosswalk.poi_id)
    
    # Open netCDF4 Dataset in write mode
    with netCDF4.Dataset(
        filename=f"{stream_output_directory}/{file_name}",
        mode='w',
        format='NETCDF4'
    ) as ncfile:

        # ============ DIMENSIONS ===================
        _ = ncfile.createDimension('feature_id', len(flow))
        _ = ncfile.createDimension('time', len(timestamps))
        _ = ncfile.createDimension('type_strlen', max_Type_str_len)
        #_ = ncfile.createDimension('gage', gage)
        #_ = ncfile.createDimension('nudge_timestep', nudge_timesteps)  # Add dimension for nudge time steps
        
        # =========== time VARIABLE ===============
        TIME = ncfile.createVariable(
            varname = "time",
            datatype = 'float64',
            dimensions = ("time",),
            fill_value = -9999.0
        )
        TIME[:] = timestamps
        ncfile['time'].setncatts(
            {
                'long_name': 'valid output time',
                'standard_name': 'time',
                'units': f'seconds since {t0.strftime("%Y-%m-%d %H:%M:%S")}',
                'missing_value': -9999.0
                #'calendar': 'proleptic_gregorian'
            }
        )

        # # =========== reference_time VARIABLE ===============
        # REF_TIME = ncfile.createVariable(
        #     varname = "reference_time",
        #     datatype = 'S1',
        #     dimensions = ("reference_time",),
        # )
        # REF_TIME[:] = t0.strftime('%Y-%m-%d_%H:%M:%S')
        # ncfile['reference_time'].setncatts(
        #     {
        #         'long_name': 'reference time',
        #         'standard_name': 'reference_time',
        #     }
        # )

        # =========== feature_id VARIABLE ===============
        FEATURE_ID = ncfile.createVariable(
            varname = "feature_id",
            datatype = 'int64',
            dimensions = ("feature_id",),
        )
        FEATURE_ID[:] = flow.index.get_level_values('featureID')
        ncfile['feature_id'].setncatts(
            {
                'long_name': 'Segment ID',
            }
        )
        # =========== type VARIABLE ===============
        TYPE = ncfile.createVariable(
            varname="type",
            datatype=f'S{max_Type_str_len}',
            dimensions=("feature_id",),
        )
        TYPE[:] = np.array(flow.index.get_level_values('Type').astype(str).tolist(), dtype=f'S{max_Type_str_len}')
        ncfile['type'].setncatts(
            {
                'long_name': 'Type',
            }
        )

        # =========== flow VARIABLE ===============            
        flow_var = ncfile.createVariable(
            varname = "flow",
            datatype = "f4",
            dimensions = ("feature_id", "time"),
            fill_value = -9999.0
            )

        flow_var[:] = flow.to_numpy(dtype=np.float32)
        ncfile['flow'].setncatts(
            {
                'long_name': 'Flow',
                'units': 'm3 s-1',
                'missing_value': -9999.0
            }
        )

        # =========== velocity VARIABLE ===============            
        velocity_var = ncfile.createVariable(
            varname = "velocity",
            datatype = "f4",
            dimensions = ("feature_id", "time"),
            fill_value = -9999.0
            )
        velocity_var[:] = velocity.to_numpy(dtype=np.float32)
        ncfile['velocity'].setncatts(
            {
                'long_name': 'Velocity',
                'units': 'm/s',
                'missing_value': -9999.0
            }
        )

        # =========== depth VARIABLE ===============            
        depth_var = ncfile.createVariable(
            varname = "depth",
            datatype = "f4",
            dimensions = ("feature_id", "time"),
            fill_value = -9999.0
            )
        depth_var[:] = depth.to_numpy(dtype=np.float32)
        ncfile['depth'].setncatts(
            {
                'long_name': 'Depth',
                'units': 'm',
                'missing_value': -9999.0
            }
        )
        
        # =========== nudge VARIABLE ===============            
        nudge = ncfile.createVariable(
            varname = "nudge",
            datatype = "f4",
            dimensions = ("feature_id", "time"),
            fill_value = -9999.0
            )
        nudge[:] = nudge_df.to_numpy(dtype=np.float32)
        ncfile['nudge'].setncatts(
            {
                'long_name': 'Streamflow Nudge Value',
                'units': 'm3 s-1',
                'missing_value': -9999.0
            }
        )
        
        # ============= POI VARIABLE ===============
        TYPE = ncfile.createVariable(
            varname="POI",
            datatype=f'S{max_poi_str_len}',
            dimensions=("feature_id",),
        )
        TYPE[:] = np.array(poi_crosswalk.poi_id.astype(str).tolist(), dtype=f'S{max_poi_str_len}')
        ncfile['type'].setncatts(
            {
                'long_name': 'POI ID',
            }
        )
        
        # =========== GLOBAL ATTRIBUTES ===============
        ncfile.setncatts(
            {
                'TITLE': 'OUTPUT FROM T-ROUTE',
                'file_reference_time': t0.strftime('%Y-%m-%d_%H:%M:%S'),
                'code_version': '',
            }
        )
def stream_output_mask_reader(stream_output_mask):
    if not stream_output_mask:
        return {}
    with open(stream_output_mask, 'r') as file:
        mask_list = yaml.safe_load(file)
    
    return mask_list

def mask_find_seg(mask_list, nexus_dict, poi_crosswalk):
    seg_id = []
    nex_id = {}
    
    if not mask_list:
        return nex_id, seg_id
    if 'wb' in mask_list and mask_list['wb']:
        if 9999 not in mask_list['wb']: 
            seg_id.extend(mask_list['wb'])
        else:
            seg_id.extend([9999])

    if 'nex' in mask_list and mask_list['nex']:
        if 9999 in mask_list['nex']:
            
            for key, val in nexus_dict.items():
                nex_key = int(key.split('-')[-1])
                nex_id[nex_key] = [int(float(v.split('-')[-1])) for v in val]
        else:
            for key in mask_list['nex']:
                item = f'nex-{key}'
                nex_id[key] = [int(val.split('-')[-1]) for val in nexus_dict.get(item, [])]

    return nex_id, seg_id

def updated_flowveldepth(flowveldepth, nex_id, seg_id, mask_list):
    flowveldepth = flowveldepth.copy(deep=True)
    flowveldepth.index.name = 'featureID'
    flowveldepth['Type'] = 'wb'
    flowveldepth.set_index('Type', append=True, inplace=True)

    def create_mask(ids):
        if 9999 in ids:
            ids = flowveldepth.index.get_level_values('featureID')
        
        masked = (flowveldepth.index.get_level_values('featureID').isin(ids)) & (flowveldepth.index.get_level_values('Type') == 'wb')
        return masked
    
    flowveldepth_seg = flowveldepth[create_mask(seg_id)] if seg_id else pd.DataFrame(columns=flowveldepth.reset_index().columns).set_index(['featureID','Type'])

    if nex_id:    
        nex_df = pd.DataFrame([(nex, wb) for nex, wbs in nex_id.items() for wb in wbs], columns=['nex', 'featureID'])
        flowveldepth_reset = flowveldepth.reset_index()
        merge_flowveldepth_reset = flowveldepth_reset.merge(nex_df, on='featureID', how='left')
        
        merge_flowveldepth_reset = merge_flowveldepth_reset.dropna(subset=['nex'])
        merge_flowveldepth_reset['nex'] = merge_flowveldepth_reset['nex'].astype(int)
               
        def optimized_v_data(merge_flowveldepth_reset, v_columns):
            # Ensure v_columns is a list
            if isinstance(v_columns, tuple):
                v_columns = [v_columns]
            
            # Calculate the size of each group by 'nex', to find out the duplicate
            group_sizes = merge_flowveldepth_reset.groupby('nex').size()

            unique_nex = group_sizes[group_sizes == 1].index
            non_unique_nex = group_sizes[group_sizes > 1].index
            filtered_unique_df = merge_flowveldepth_reset[merge_flowveldepth_reset['nex'].isin(unique_nex)]
            filtered_non_unique_df = merge_flowveldepth_reset[merge_flowveldepth_reset['nex'].isin(non_unique_nex)]
            filtered_non_unique_df.loc[:, v_columns] = np.nan
            filtered_non_unique_df = filtered_non_unique_df.drop_duplicates(subset=['nex'])

            combined_df = pd.concat([filtered_unique_df, filtered_non_unique_df])
            v_data = combined_df.set_index('nex')[v_columns]

            return v_data
       
               
        def optimized_agg(merge_flowveldepth_reset):
            # Identify columns for each operation
            q_columns = [col for col in merge_flowveldepth_reset.columns if col[1] == 'q']
            v_columns = [col for col in merge_flowveldepth_reset.columns if col[1] == 'v']
            d_columns = [col for col in merge_flowveldepth_reset.columns if col[1] == 'd']
            
            # Sum of flowrate columns
            sum_q = merge_flowveldepth_reset.groupby('nex')[q_columns].sum()
            
            # Handle velocity in a way that if there are 2 segments coming into one put NaN value
            v_data = optimized_v_data(merge_flowveldepth_reset, v_columns) 

            # Average depth columns 
            avg_d = merge_flowveldepth_reset.groupby('nex')[d_columns].mean()

            # Combine all results into a single DataFrame
            all_nex_data = pd.concat([sum_q, v_data, avg_d], axis=1)

            return all_nex_data
            
        all_nex_data = optimized_agg(merge_flowveldepth_reset)
        all_nex_data.index = all_nex_data.index.rename('featureID')
        all_nex_data['Type'] = 'nex'        
        # Set the new 'Type' column as an index
        all_nex_data = all_nex_data.set_index('Type', append=True)
        
    else:
        all_nex_data = pd.DataFrame()
    
    if not flowveldepth_seg.empty or not all_nex_data.empty:
        flowveldepth = pd.concat([flowveldepth_seg, all_nex_data])
    
    return flowveldepth

def write_flowveldepth(
    stream_output_directory,
    stream_output_mask,
    flowveldepth,
    nudge,
    usgs_positions_id,
    t0,
    dt,
    stream_output_timediff,
    stream_output_type,
    stream_output_internal_frequency = 5,
    cpu_pool = 1,
    poi_crosswalk = None,
    nexus_dict= None,
    ):
    '''
    Write the results of flowveldepth and nudge to netcdf- break. 
    Arguments
    -------------
    stream_output_directory (Path or string) - directory where file will be created
    flowveldepth (DataFrame) -  including flowrate, velocity, and depth for each time step
    nudge (numpy.ndarray) - nudge data with shape (76, 289)
    usgs_positions_id (array) - Position ids of usgs gages
    '''
    
    mask_list = stream_output_mask_reader(stream_output_mask)
    nex_id, seg_id = mask_find_seg(mask_list, nexus_dict, poi_crosswalk)
    flowveldepth = updated_flowveldepth(flowveldepth, nex_id, seg_id, mask_list)
    
    # timesteps, variable = zip(*flowveldepth.columns.tolist())
    # timesteps = list(timesteps)
    n_timesteps = flowveldepth.shape[1]//3
    ts = stream_output_internal_frequency//(dt//60)
    ind = [i for i in range(ts-1,n_timesteps,ts)]
    timestamps_sec =  [(i+1)*dt for i in ind]
    
    flow = flowveldepth.iloc[:,0::3].iloc[:,ind]
    velocity = flowveldepth.iloc[:,1::3].iloc[:,ind]
    depth = flowveldepth.iloc[:,2::3].iloc[:,ind]

    # Create POI-nexus point crosswalk dataframe (if necessary)
    if not poi_crosswalk.empty:
        idx = flowveldepth.reset_index()[['featureID','Type']]
        poi_crosswalk_masked = poi_crosswalk.copy()
        poi_crosswalk_masked[['Type', 'featureID']] = poi_crosswalk_masked['id'].str.split('-', expand=True)
        poi_crosswalk_masked['Type'] = 'nex' # Types can be 'nex', 'tnx', etc. This just sets them all to generic 'nex'.
        poi_crosswalk_masked['featureID'] = poi_crosswalk_masked.featureID.astype(int)
        poi_crosswalk_masked = idx.set_index(['featureID','Type']).join(poi_crosswalk_masked[['featureID','Type','poi_id']].set_index(['featureID','Type']))
    else:
        poi_crosswalk_masked = poi_crosswalk.copy()
        poi_crosswalk_masked['poi_id'] = [" " for _ in range(len(flowveldepth))]

    # Check if the first column of nudge is all zeros
    if np.all(nudge[:, 0] == 0):
        # Drop the first column
        nudge = nudge[:, 1:]
    nudge_df = pd.DataFrame(data=nudge, index=usgs_positions_id).iloc[:,ind]
    empty_ids = list(set(flowveldepth.index).difference(set(nudge_df.index)))
    empty_df = pd.DataFrame(index=empty_ids, columns=nudge_df.columns).fillna(-9999.0)
    nudge_df = pd.concat([nudge_df, empty_df]).loc[flowveldepth.index]
    file_name_time = t0
    jobs = []
    
    if stream_output_timediff > 0:
        ts_per_file = stream_output_timediff*60//stream_output_internal_frequency
        
        num_files = flowveldepth.shape[1]//3*dt//(stream_output_timediff*60*60)
        if num_files==0:
            num_files=1
        
        for _ in range(num_files):
            filename = 'troute_output_' + file_name_time.strftime('%Y%m%d%H%M') + stream_output_type
            args = (stream_output_directory,filename,
                    flow.iloc[:,0:ts_per_file],
                    velocity.iloc[:,0:ts_per_file],
                    depth.iloc[:,0:ts_per_file],
                    nudge_df.iloc[:,0:ts_per_file],
                    timestamps_sec[0:ts_per_file],t0,
                    poi_crosswalk_masked)
            if stream_output_type == '.nc':
                if cpu_pool > 1 & num_files > 1:
                    jobs.append(delayed(write_flowveldepth_netcdf)(*args))
                else:
                    write_flowveldepth_netcdf(*args)
            else:
                if cpu_pool > 1 & num_files > 1:
                    jobs.append(delayed(write_flowveldepth_csv_pkl)(*args))
                else:
                    write_flowveldepth_csv_pkl(*args)

            flow = flow.iloc[:,ts_per_file:]
            velocity = velocity.iloc[:,ts_per_file:]
            depth = depth.iloc[:,ts_per_file:]
            nudge_df = nudge_df.iloc[:,ts_per_file:]
            timestamps_sec = timestamps_sec[ts_per_file:]
            file_name_time = file_name_time + timedelta(hours=stream_output_timediff)
    
    elif stream_output_timediff == -1:
     
        filename = 'troute_output_' + file_name_time.strftime('%Y%m%d%H%M') + stream_output_type
        args = (stream_output_directory,filename,
                flow,
                velocity,
                depth,
                nudge_df,
                timestamps_sec,
                t0,
                poi_crosswalk_masked)
        if stream_output_type == '.nc':
            if cpu_pool > 1:
                jobs.append(delayed(write_flowveldepth_netcdf)(*args))
            else:
                write_flowveldepth_netcdf(*args)
        else:
            if cpu_pool > 1:
                jobs.append(delayed(write_flowveldepth_csv_pkl)(*args))
            else:
                write_flowveldepth_csv_pkl(*args)

    if cpu_pool > 1:
        try:
            # Execute all jobs in parallel
            with Parallel(n_jobs=cpu_pool) as parallel:
                parallel(jobs)
        except Exception as e:
            LOG.error("Error during parallel writing output: %s", e)   
    
    LOG.debug("Completed the write_flowveldepth_netcdf function") 

    


#TODO Delete these function. They've been replace by the above two functions. Keep here for
# now until functions above are properly vetted.


# def helper_write_flowveldepth(flowveldepth, subset_df, current_time_step, 
#                   stream_output_directory, stream_output_type, gage, 
#                   nudge_timesteps, time_dim, stream_output_internal_frequency, time_steps,
#                   counter, t0, stream_output_timediff):
#     '''
#     Just a helper function for 'write_flowveldepth_netcdf' function for parallel computing.
#     '''
    
#     if stream_output_directory:
#             if stream_output_type =='.nc':
#                 file_name = f"{current_time_step}.flowveldepth.nc"

#             elif stream_output_type=='.csv':
#                 file_name = f"{current_time_step}.flowveldepth.csv"
#                 # Save the data to CSV file
#                 subset_df.to_csv(f"{stream_output_directory}/{file_name}", index=True)
#                 LOG.debug(f"Flowveldepth data saved as CSV files in {stream_output_directory}")

#             elif stream_output_type=='.pkl':
#                 file_name = f"{current_time_step}.flowveldepth.pkl"
#                 # Save the data to Pickle file
#                 subset_df.to_pickle(f"{stream_output_directory}/{file_name}")
#                 LOG.debug(f"Flowveldepth data saved as PICKLE files in {stream_output_directory}")
                
#             else:
#                 print('WRONG FORMAT')
        
#     if stream_output_directory:
#         if (stream_output_type =='.nc'):
#             # Open netCDF4 Dataset in write mode
#             with netCDF4.Dataset(
#                 filename=f"{stream_output_directory}/{file_name}",
#                 mode='w',
#                 format='NETCDF4'
#             ) as ncfile:

#                 # ============ DIMENSIONS ===================
#                 _ = ncfile.createDimension('feature_id', None)
#                 _ = ncfile.createDimension('time_step (sec)', subset_df.iloc[:, 0::4].shape[1])
#                 _ = ncfile.createDimension('gage', gage)
#                 _ = ncfile.createDimension('nudge_timestep', nudge_timesteps)  # Add dimension for nudge time steps
                
#                 # =========== q,v,d,ndg VARIABLES ===============
#                 for counters, var in enumerate(['flowrate', 'velocity', 'depth', 'nudge']):
#                     QVD = ncfile.createVariable(
#                         varname=var,
#                         datatype=np.float32,
#                         dimensions=('feature_id', 'time_step (sec)',),
#                     )

#                     QVD.units = 'm3/s m/s m m3/s'
#                     QVD.description = f'Data for {var}'
                    
#                     # Prepare data for writing
#                     data_array = subset_df.iloc[:, counters::4].to_numpy(dtype=np.float32)
                    
#                     # Set data for each feature_id and time_step
#                     ncfile.variables[var][:] = data_array
#                 feature_id = ncfile.createVariable(
#                     varname='feature_id',
#                     datatype=np.int32,
#                     dimensions=('feature_id',),
#                 )
#                 feature_id[:] = flowveldepth.index.to_numpy(dtype=np.int32)
#                 feature_id.units = 'None'
#                 feature_id.description = 'Feature IDs'
#                 ###
#                 time_step = ncfile.createVariable(
#                     varname='time_step (sec)',
#                     datatype=np.int32,
#                     dimensions=('time_step (sec)',),
#                 )
#                 time_step[:] = np.array(time_dim[:subset_df.iloc[:, 0::4].shape[1]], dtype=np.int32)
#                 time_step.units = 'sec'
#                 time_step.description = 'time stamp'
#                 # =========== GLOBAL ATTRIBUTES ===============
#                 ncfile.setncatts(
#                     {
#                         'TITLE': 'OUTPUT FROM T-ROUTE',
#                         'Time step (sec)': f'{stream_output_internal_frequency}',
#                         'model_initialization_time': t0.strftime('%Y-%m-%d_%H:%M:%S'),
#                         'model_reference_time': time_steps[counter].strftime('%Y-%m-%d_%H:%M:%S'),
#                         'comment': f'The file includes {stream_output_timediff} hour data which includes {len(time_dim)} timesteps',
#                         'code_version': '',
#                     }
#                 )
#                 LOG.debug(f"Flowveldepth data saved as NetCDF files in {stream_output_directory}")

# def write_flowveldepth_netcdf(stream_output_directory, 
#                               flowveldepth, 
#                               nudge, 
#                               usgs_positions_id, 
#                               t0, 
#                               stream_output_timediff, 
#                               stream_output_type,
#                               stream_output_internal_frequency = 5,
#                               cpu_pool = 6):
#     '''
#     Write the results of flowveldepth and nudge to netcdf- break. 
#     Arguments
#     -------------
#     stream_output_directory (Path or string) - directory where file will be created
#     flowveldepth (DataFrame) -  including flowrate, velocity, and depth for each time step
#     nudge (numpy.ndarray) - nudge data with shape (76, 289)
#     usgs_positions_id (array) - Position ids of usgs gages
#     '''
#     # Number of timesteps and features
#     print('step 1...')
#     nsteps = len(flowveldepth.columns) // 3
#     num_features = len(flowveldepth)
#     nstep_nc = 12 * stream_output_timediff
#     # Check if the first column of nudge is all zeros
#     if np.all(nudge[:, 0] == 0):
#         # Drop the first column
#         nudge = nudge[:, 1:]

#     gage, nudge_timesteps = nudge.shape
#     print('step 2...')
#     #--------- Add 'nudge' column based on usgs_positions_id----------
    
#     # Create a copy of the flowveldepth DataFrame to add 'ndg' columns
#     qvd_ndg = flowveldepth.copy()
#     # Create a list for names of the columns for nudge values
#     ndg_columns = [(j,'ndg') for j in range(nsteps)]

#     if len(usgs_positions_id)>0:
#         # Add 'ndg' columns based on usgs_positions_id
#         for i, usgs_id in enumerate(usgs_positions_id):
#             # Extract the corresponding nudge values for the usgs_id
#             nudge_values = nudge[i]

#             # Assign nudge values to 'ndg' columns for the corresponding row
#             qvd_ndg.loc[usgs_id, ndg_columns] = nudge_values
#     else:
#         print('step 3...')
#         qvd_ndg.loc[:, ndg_columns] = -9999.0

#     print('step 4...')
#     new_order = [(i, attr) for i in range(0, nsteps) for attr in ['q', 'v', 'd', 'ndg']]
#     # Reorder the columns
#     qvd_ndg = qvd_ndg[new_order]
    
#     # renaming the columns based on timestamp
#     print('step 5...')
#     column_name_timeStamp = [t0 + timedelta(minutes=(i * 5)) for i in range(nsteps)]
#     new_column_name_timeStamp = [(attr + '_' +str(cnt) + '_' + times.strftime('%Y%m%d%H%M')) for cnt, times in enumerate(column_name_timeStamp) for attr in ['q', 'v', 'd', 'ndg']]
#     qvd_ndg.columns = new_column_name_timeStamp

#     # Create time step values based on t0
#     print('step 6...')
#     time_steps = [t0 + timedelta(hours= (i * stream_output_timediff)) for i in range(nsteps//nstep_nc)]
#     time_dim = [t * stream_output_internal_frequency*60 for t in range(1, int(stream_output_timediff * 60 / stream_output_internal_frequency) + 1)]
#     jobs = []
    
#     for counter, i in enumerate(range(0, nsteps, nstep_nc)):
        
        
#         # Define the range of columns for this file
#         start_col = i * 4
#         end_col = min((i + nstep_nc) * 4 , nsteps * 4)
#         selected_col = stream_output_internal_frequency // 5
#         # Create a subset DataFrame for the current range of columns
#         # subset_df = qvd_ndg.iloc[:, start_col:end_col]
#         # Create a list of column names to keep
#         columns_to_keep = [col for col in qvd_ndg.columns[start_col:end_col] if int(col.split('_')[1]) % selected_col == 0]
#         subset_df = qvd_ndg[columns_to_keep]
#         subset_df.columns = ['_'.join([col.split('_')[0], col.split('_')[2]]) for col in subset_df.columns]
        
#         # Create the file name based on the current time step
#         import pdb; pdb.set_trace()
#         current_time_step = time_steps[counter].strftime('%Y%m%d%H%M')
        
#         args = (flowveldepth, subset_df, current_time_step, 
#         stream_output_directory, stream_output_type, gage, 
#         nudge_timesteps, time_dim, stream_output_internal_frequency, time_steps,
#         counter, t0, stream_output_timediff)
        
#         if cpu_pool > 1:
#             jobs.append(delayed(helper_write_flowveldepth)(*args))
#             LOG.debug(f"Job for step {counter} added for parallel processing.")
#         else:
#             helper_write_flowveldepth(*args)

#     if cpu_pool > 1:
#         try:
#             # Execute all jobs in parallel
#             with Parallel(n_jobs=cpu_pool) as parallel:
#                 parallel(jobs)
#         except Exception as e:
#             LOG.error("Error during parallel processing: %s", e)   
    
#     LOG.debug("Completed the write_flowveldepth_netcdf function") 
