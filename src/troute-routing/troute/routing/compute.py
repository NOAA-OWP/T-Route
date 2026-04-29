from __future__ import annotations
from collections import defaultdict
from itertools import chain
from functools import partial
from typing import Literal, TYPE_CHECKING, Iterable
from joblib import delayed, Parallel
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import copy
import os.path

import troute.nhd_network as nhd_network
from troute.routing.fast_reach.mc_reach import compute_network_structured
import troute.routing.diffusive_utils_v02 as diff_utils
from troute.routing.fast_reach import diffusive

import logging
import ewts
LOG = ewts.get_logger(ewts.T_ROUTE_ID)

if TYPE_CHECKING:
    from typing import Annotated
    from numpy.typing import NDArray


_compute_func_map = defaultdict(
    compute_network_structured,
    {
        "V02-structured": compute_network_structured,
    },
)


def _format_qlat_start_time(qlat_start_time):
    if not isinstance(qlat_start_time,datetime):
        try:
            return datetime.strptime(qlat_start_time, '%Y-%m-%d %H:%M:%S')
        except:  # TODO: make sure this doesn't introduce a silent error
            return datetime.now()

    else:
        return qlat_start_time


def _build_reach_type_list(reach_list, wbodies_segs):

    reach_type_list = [
                1 if (set(reaches) & wbodies_segs) else 0 for reaches in reach_list
            ]

    return list(zip(reach_list, reach_type_list))


def _prep_da_dataframes(
    usgs_df,
    lastobs_df,
    param_df_sub_idx,
    exclude_segments=None,
    ):
    """
    Produce, based on the segments in the param_df_sub_idx (which is a subset
    representing a subnetwork of the larger collection of all segments),
    a subset of the relevant usgs gage observation time series
    and the relevant last-valid gage observation from any
    prior model execution.
    
    exclude_segments (list): segments to exclude from param_df_sub when searching for gages
                             This catches and excludes offnetwork upstreams segments from being
                             realized as locations for DA substitution. Else, by-subnetwork
                             parallel executions fail.

    Cases to consider:
    USGS_DF, LAST_OBS
    Yes, Yes: Analysis and Assimilation; Last_Obs used to fill gaps in the front of the time series
    No, Yes: Forecasting mode;
    Yes, No; Cold-start case;
    No, No: Open-Loop;

    For both cases where USGS_DF is present, there is a sub-case where the length of the observed
    time series is as long as the simulation.

    """
    
    subnet_segs = param_df_sub_idx
    # segments in the subnetwork ONLY, no offnetwork upstreams included
    if exclude_segments:
        subnet_segs = param_df_sub_idx.difference(set(exclude_segments))
    
    # NOTE: Uncomment to easily test no observations...
    # usgs_df = pd.DataFrame()
    if not usgs_df.empty and not lastobs_df.empty:
        # index values for last obs are not correct, but line up correctly with usgs values. Switched
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                       )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        usgs_segs = (usgs_df.index.
                     intersection(subnet_segs).
                     reindex(lastobs_segs)[0].
                     to_list()
                    )
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
    elif usgs_df.empty and not lastobs_df.empty:
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                       )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        # Create a completely empty list of gages -- the .shape[1] attribute
        # will be == 0, and that will trigger a reference to the lastobs.
        # in the compute kernel below.
        usgs_df_sub = pd.DataFrame(index=lastobs_df_sub.index,columns=[])
        usgs_segs = lastobs_segs
        da_positions_list_byseg = param_df_sub_idx.get_indexer(lastobs_segs)
    elif not usgs_df.empty and lastobs_df.empty:
        usgs_segs = list(usgs_df.index.intersection(subnet_segs))
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
        lastobs_df_sub = pd.DataFrame(index=usgs_df_sub.index,columns=["discharge","time","model_discharge"])
    else:
        usgs_df_sub = pd.DataFrame()
        lastobs_df_sub = pd.DataFrame()
        da_positions_list_byseg = []

    return usgs_df_sub, lastobs_df_sub, da_positions_list_byseg


def _prep_da_positions_byreach(reach_list, gage_index):
    """
    produce a list of indexes of the reach_list identifying reaches with gages
    and a corresponding list of indexes of the gage_list of the gages in
    the order they are found in the reach_list.
    """
    reach_key = []
    reach_gage = []
    for i, r in enumerate(reach_list):
        for s in r:
            if s in gage_index:
                reach_key.append(i)
                reach_gage.append(s)
    gage_reach_i = gage_index.get_indexer(reach_gage)

    return reach_key, gage_reach_i

def _prep_reservoir_da_dataframes(reservoir_usgs_df,
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
                                  waterbody_types_df_sub,
                                  t0, 
                                  from_files,
                                  exclude_segments=None):
    '''
    Helper function to build reservoir DA data arrays for routing computations

    Arguments
    ---------
    reservoir_usgs_df        (DataFrame): gage flow observations at USGS-type reservoirs
    reservoir_usgs_param_df  (DataFrame): USGS reservoir DA state parameters
    reservoir_usace_df       (DataFrame): gage flow observations at USACE-type reservoirs
    reservoir_usace_param_df (DataFrame): USACE reservoir DA state parameters
    reservoir_usbr_df       (DataFrame): gage flow observations at USBR-type reservoirs
    reservoir_usbr_param_df (DataFrame): USBR reservoir DA state parameters
    reservoir_rfc_df         (DataFrame): gage flow observations and forecasts at RFC-type reservoirs
    reservoir_rfc_param_df   (DataFrame): RFC reservoir DA state parameters
    waterbody_types_df_sub   (DataFrame): type-codes for waterbodies in sub domain
    t0                        (datetime): model initialization time

    Returns
    -------
    * there are many returns, because we are passing explicit arrays to mc_reach cython code
    reservoir_usgs_df_sub                 (DataFrame): gage flow observations for USGS-type reservoirs in sub domain
    reservoir_usgs_df_time                  (ndarray): time in seconds from model initialization time
    reservoir_usgs_update_time              (ndarray): update time (sec) to search for new observation at USGS reservoirs
    reservoir_usgs_prev_persisted_flow      (ndarray): previously persisted outflow rates at USGS reservoirs
    reservoir_usgs_persistence_update_time  (ndarray): update time (sec) of persisted value at USGS reservoirs
    reservoir_usgs_persistence_index        (ndarray): index denoting elapsed persistence epochs at USGS reservoirs
    reservoir_usace_df_sub                (DataFrame): gage flow observations for USACE-type reservoirs in sub domain
    reservoir_usace_df_time                 (ndarray): time in seconds from model initialization time
    reservoir_usace_update_time             (ndarray): update time (sec) to search for new observation at USACE reservoirs
    reservoir_usace_prev_persisted_flow     (ndarray): previously persisted outflow rates at USACE reservoirs
    reservoir_usace_persistence_update_time (ndarray): update time (sec) of persisted value at USACE reservoirs
    reservoir_usace_persistence_index       (ndarray): index denoting elapsed persistence epochs at USACE reservoirs
    reservoir_usbr_df_sub                (DataFrame): gage flow observations for USBR-type reservoirs in sub domain
    reservoir_usbr_df_time                 (ndarray): time in seconds from model initialization time
    reservoir_usbr_update_time             (ndarray): update time (sec) to search for new observation at USBR reservoirs
    reservoir_usbr_prev_persisted_flow     (ndarray): previously persisted outflow rates at USBR reservoirs
    reservoir_usbr_persistence_update_time (ndarray): update time (sec) of persisted value at USBR reservoirs
    reservoir_usbr_persistence_index       (ndarray): index denoting elapsed persistence epochs at USBR reservoirs
    '''
    if not reservoir_usgs_df.empty:
        usgs_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==2
                                ].index
        if exclude_segments:
            usgs_wbodies_sub = list(set(usgs_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usgs_df_sub = reservoir_usgs_df.loc[usgs_wbodies_sub]
        reservoir_usgs_df_time = []
        for timestamp in reservoir_usgs_df.columns:
            reservoir_usgs_df_time.append((timestamp - t0).total_seconds())
        reservoir_usgs_df_time = np.array(reservoir_usgs_df_time)
        reservoir_usgs_update_time = reservoir_usgs_param_df['update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_prev_persisted_flow = reservoir_usgs_param_df['prev_persisted_outflow'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_update_time = reservoir_usgs_param_df['persistence_update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_index = reservoir_usgs_param_df['persistence_index'].loc[usgs_wbodies_sub].to_numpy()
    else:
        reservoir_usgs_df_sub = pd.DataFrame()
        reservoir_usgs_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 2] = 1

    # select USACE reservoir DA data waterbodies in sub-domain
    if not reservoir_usace_df.empty:
        usace_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==3
                                ].index
        if exclude_segments:
            usace_wbodies_sub = list(set(usace_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usace_df_sub = reservoir_usace_df.loc[usace_wbodies_sub]
        reservoir_usace_df_time = []
        for timestamp in reservoir_usace_df.columns:
            reservoir_usace_df_time.append((timestamp - t0).total_seconds())
        reservoir_usace_df_time = np.array(reservoir_usace_df_time)
        reservoir_usace_update_time = reservoir_usace_param_df['update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_prev_persisted_flow = reservoir_usace_param_df['prev_persisted_outflow'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_update_time = reservoir_usace_param_df['persistence_update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_index = reservoir_usace_param_df['persistence_index'].loc[usace_wbodies_sub].to_numpy()
    else: 
        reservoir_usace_df_sub = pd.DataFrame()
        reservoir_usace_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 3] = 1

    # select USBR reservoir DA data waterbodies in sub-domain
    if not reservoir_usbr_df.empty:
        usbr_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==7
                                ].index
        if exclude_segments:
            usbr_wbodies_sub = list(set(usbr_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usbr_df_sub = reservoir_usbr_df.loc[usbr_wbodies_sub]
        reservoir_usbr_df_time = []
        for timestamp in reservoir_usbr_df.columns:
            reservoir_usbr_df_time.append((timestamp - t0).total_seconds())
        reservoir_usbr_df_time = np.array(reservoir_usbr_df_time)
        reservoir_usbr_update_time = reservoir_usbr_param_df['update_time'].loc[usbr_wbodies_sub].to_numpy()
        reservoir_usbr_prev_persisted_flow = reservoir_usbr_param_df['prev_persisted_outflow'].loc[usbr_wbodies_sub].to_numpy()
        reservoir_usbr_persistence_update_time = reservoir_usbr_param_df['persistence_update_time'].loc[usbr_wbodies_sub].to_numpy()
        reservoir_usbr_persistence_index = reservoir_usbr_param_df['persistence_index'].loc[usbr_wbodies_sub].to_numpy()
    else: 
        reservoir_usbr_df_sub = pd.DataFrame()
        reservoir_usbr_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usbr_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usbr_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usbr_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usbr_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 7] = 1
    
    # RFC reservoirs
    if not reservoir_rfc_df.empty:
        rfc_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==4
            ].index
        if exclude_segments:
            rfc_wbodies_sub = list(set(rfc_wbodies_sub).difference(set(exclude_segments)))
        reservoir_rfc_df_sub = reservoir_rfc_df.loc[rfc_wbodies_sub]
        reservoir_rfc_totalCounts = reservoir_rfc_param_df['totalCounts'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_file = reservoir_rfc_param_df['file'].loc[rfc_wbodies_sub].to_list()
        reservoir_rfc_use_forecast = reservoir_rfc_param_df['use_rfc'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_timeseries_idx = reservoir_rfc_param_df['timeseries_idx'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_update_time = reservoir_rfc_param_df['update_time'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_da_timestep = reservoir_rfc_param_df['da_timestep'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_persist_days = reservoir_rfc_param_df['rfc_persist_days'].loc[rfc_wbodies_sub].to_numpy()
    else:
        reservoir_rfc_df_sub = pd.DataFrame()
        reservoir_rfc_totalCounts = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_file = []
        reservoir_rfc_use_forecast = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_timeseries_idx = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_da_timestep = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_persist_days = pd.DataFrame().to_numpy().reshape(0,)
        if not from_files:
            if not waterbody_types_df_sub.empty:
                waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 4] = 1
    
    # Great Lakes
    if not great_lakes_df.empty:
        gl_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==6
            ].index
        if exclude_segments:
            gl_wbodies_sub = list(set(gl_wbodies_sub).difference(set(exclude_segments)))
        gl_df_sub = great_lakes_df[great_lakes_df['lake_id'].isin(gl_wbodies_sub)]
        gl_climatology_df_sub = great_lakes_climatology_df.loc[gl_wbodies_sub]
        gl_param_df_sub = great_lakes_param_df[great_lakes_param_df['lake_id'].isin(gl_wbodies_sub)]
        gl_parm_lake_id_sub = gl_param_df_sub.lake_id.to_numpy()
        gl_param_flows_sub = gl_param_df_sub.previous_assimilated_outflows.to_numpy()
        gl_param_time_sub = gl_param_df_sub.previous_assimilated_time.to_numpy()
        gl_param_update_time_sub = gl_param_df_sub.update_time.to_numpy()
    else:
        gl_df_sub = pd.DataFrame(columns=['lake_id','time','Discharge'])
        gl_climatology_df_sub = pd.DataFrame()
        gl_parm_lake_id_sub = pd.DataFrame().to_numpy().reshape(0,)
        gl_param_flows_sub = pd.DataFrame().to_numpy().reshape(0,)
        gl_param_time_sub = pd.DataFrame().to_numpy().reshape(0,)
        gl_param_update_time_sub = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 6] = 1

    return (
        reservoir_usgs_df_sub, reservoir_usgs_df_time, reservoir_usgs_update_time, reservoir_usgs_prev_persisted_flow, reservoir_usgs_persistence_update_time, reservoir_usgs_persistence_index,
        reservoir_usace_df_sub, reservoir_usace_df_time, reservoir_usace_update_time, reservoir_usace_prev_persisted_flow, reservoir_usace_persistence_update_time, reservoir_usace_persistence_index,
        reservoir_usbr_df_sub, reservoir_usbr_df_time, reservoir_usbr_update_time, reservoir_usbr_prev_persisted_flow, reservoir_usbr_persistence_update_time, reservoir_usbr_persistence_index,
        reservoir_rfc_df_sub, reservoir_rfc_totalCounts, reservoir_rfc_file, reservoir_rfc_use_forecast, reservoir_rfc_timeseries_idx, reservoir_rfc_update_time, reservoir_rfc_da_timestep, reservoir_rfc_persist_days,
        gl_df_sub, gl_parm_lake_id_sub, gl_param_flows_sub, gl_param_time_sub, gl_param_update_time_sub, gl_climatology_df_sub,
        waterbody_types_df_sub
        )


def compute_log_mc(
    fileName,
    connections,
    rconn,
    wbody_conn,
    reaches_bytw,
    compute_func_name,
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
):
    
    # TODO: do something with param_df, reservoir_XXX_param_df, or delete them as args

    # append parameters and some statistics to log file
    with open(fileName, 'a') as preRunLog:

        preRunLog.write("*******************\n") 
        preRunLog.write("Compute Parameters:\n") 
        preRunLog.write("*******************\n") 
        preRunLog.write("\n")   
        preRunLog.write("General Compute Parameters:\n")
        preRunLog.write("\n")  
        preRunLog.write("Parallel Compute Method: "+parallel_compute_method+'\n')
        preRunLog.write("Compute Kernel Name: "+compute_func_name+'\n')
        preRunLog.write("Assume Short Timescale: "+str(assume_short_ts)+'\n')
        preRunLog.write("Subnetwork Target Size: "+str(subnetwork_target_size)+'\n')
        preRunLog.write("CPU Pool: "+str(cpu_pool)+'\n')
        #preRunLog.write("\n")
        #preRunLog.write("Restart Parameters:\n")  
        #preRunLog.write("\n")
        preRunLog.write("Start_datetime: "+str(t0)+'\n')
        preRunLog.write("Coldstart: "+str(((q0==0).all()).all())+'\n')
        preRunLog.write("\n")
        preRunLog.write("Forcing Parameters:\n")
        preRunLog.write("\n")           
        preRunLog.write("qts subdivisions: "+str(qts_subdivisions)+'\n')
        preRunLog.write("dt [sec]: "+str(dt)+'\n')
        preRunLog.write("nts: "+str(nts)+'\n')
        preRunLog.write("\n")
        preRunLog.write("Data Assimilation Parameters:\n")
        preRunLog.write("\n")
    
        if ('usgs_timeslices_folder' in data_assimilation_parameters.keys()):           
            preRunLog.write("usgs timeslice folder: "+str(data_assimilation_parameters['usgs_timeslices_folder'])+'\n')
        if ('usace_timeslices_folder' in data_assimilation_parameters.keys()):    
            preRunLog.write("usace timeslice folder: "+str(data_assimilation_parameters['usace_timeslices_folder'])+'\n')
        preRunLog.write("-----\n")
        preRunLog.write("Streamflow DA\n")
        if ('streamflow_da' in data_assimilation_parameters.keys()):  
            outPutStr = "Streamflow nudging: "+str(data_assimilation_parameters['streamflow_da']['streamflow_nudging'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            outPutStr = "Diffusive streamflow nudging: "+str(data_assimilation_parameters['streamflow_da']['diffusive_streamflow_nudging'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            preRunLog.write("Lastobs file: "+str(data_assimilation_parameters['streamflow_da']['lastobs_file'])+'\n')
            preRunLog.write("-----\n")
            preRunLog.write("Reservoir DA\n")
            outPutStr = "Reservoir persistence USGS: "+str(data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usgs'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            outPutStr = "Reservoir persistence USACE: "+str(data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usace'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            outPutStr = "Reservoir persistence USBR: "+str(data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usbr'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            preRunLog.write("Reservoir RFC forecasts: "+str(data_assimilation_parameters['reservoir_da']['reservoir_rfc_da']['reservoir_rfc_forecasts'])+'\n')

        preRunLog.write("\n")                   
        preRunLog.write("****************************\n") 
        preRunLog.write("Network Topology Parameters:\n") 
        preRunLog.write("****************************\n") 
        preRunLog.write("\n")   
        preRunLog.write("General network:\n")
        preRunLog.write("Number of downstream connections: "+str(len(connections))+'\n')
        preRunLog.write("Number of upstream connections: "+str(len(rconn))+'\n')
        preRunLog.write("Number of waterbody connections: "+str(len(wbody_conn))+'\n')
        preRunLog.write("Number of reaches by tailwater: "+str(len(reaches_bytw))+'\n')
        preRunLog.write("Number of independent networks: "+str(len(independent_networks))+'\n')
        preRunLog.write("Number of waterbodies: "+str(len(waterbodies_df.index))+'\n')
        preRunLog.write("Waterbody type specified: "+str(waterbody_type_specified)+'\n')
        nH20_types = len(waterbody_types_df.value_counts())
        if (nH20_types>0):
            for nH20 in range(nH20_types):
                preRunLog.write("Type: "+str(waterbody_types_df.value_counts().index[nH20][0]))
                preRunLog.write("   Number of waterbodies: "+str(waterbody_types_df.value_counts().values[nH20])+'\n')
        preRunLog.write("-----\n")
        preRunLog.write("Gages and relations with waterbodies:\n")
        preRunLog.write("Number of USGS gages in network: "+str(len(usgs_df.index))+'\n')
        preRunLog.write("Number of USGS gage time bins in network: "+str(len(usgs_df.columns))+'\n')
        preRunLog.write("Lastobs files, number of gages: "+str(len(lastobs_df.index))+'\n')
        preRunLog.write("Number of USGS gages in waterbodies: "+str(len(reservoir_usgs_df.index))+'\n')
        preRunLog.write("Number of USACE gages in waterbodies: "+str(len(reservoir_usace_df.index))+'\n')
        preRunLog.write("Number of USBR gages in waterbodies: "+str(len(reservoir_usbr_df.index))+'\n')
        preRunLog.write("Number of RFC gages in waterbodies: "+str(len(reservoir_rfc_df.index))+'\n')
        preRunLog.write("\n")        

    preRunLog.close()


def compute_log_diff(
    fileName,
    diffusive_network_data,
    topobathy_df,
    refactored_diffusive_domain,
    refactored_reaches,                
    coastal_boundary_depth_df,
    unrefactored_topobathy_df,                
):

    # TODO: do something with refactored_diffusive_domain, refactored_reaches, unrefactored_topobathy_df, or delete args

    # append parameters and some statistics to log file
    with open(fileName, 'a') as preRunLog:

        preRunLog.write("*******************\n") 
        preRunLog.write("Diffusive Routing :\n") 
        preRunLog.write("*******************\n") 
        nTw = len(diffusive_network_data)
        preRunLog.write("\n")   
        outPutStr = "Number of diffusive tailwaters: "+str(nTw)
        preRunLog.write(outPutStr+'\n')
        LOG.info(outPutStr)
        preRunLog.write("-----\n")  

        twList = [key for key in diffusive_network_data]

        for i_nTw in range(nTw):  
            outPutStr = "Tailwater number and ID: "+str(i_nTw+1)+"   "+str(twList[i_nTw])
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)
            diffNw = diffusive_network_data[twList[i_nTw]]
            nMainSegs = len(diffNw['mainstem_segs'])
            firstSeg = diffNw['mainstem_segs'][0]
            lastSeg = diffNw['mainstem_segs'][-1]
            preRunLog.write("Number of mainstem segments: "+str(nMainSegs)+"\n")
            preRunLog.write("First and last segment ID: "+str(firstSeg)+"   "+str(lastSeg)+"\n")
            nTribSegs = len(diffNw['tributary_segments'])
            preRunLog.write("Number of tributary segments: "+str(nTribSegs)+"\n")
            connGraphLength = len(diffNw['connections'])
            revConnGraphLength = len(diffNw['rconn'])
            preRunLog.write("Connections in network: "+str(connGraphLength)+"\n")
            preRunLog.write("Reverse connections in network: "+str(revConnGraphLength)+"\n")
            paramDf_Columns = [column for column in diffNw['param_df'].columns]
            preRunLog.write("Diffusive parameters:\n")
    
            for paramDf_Col in paramDf_Columns:
                preRunLog.write(str(paramDf_Col)+"  ")
            preRunLog.write("\n")
            preRunLog.write("-----\n")

        if (not topobathy_df.empty):    
            topoIDs = topobathy_df.index
            topoTraces = len(topoIDs)
            topoTracesUnique = len(set(topoIDs))
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "Number of topobathy profiles: "+str(topoTraces)
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)
            preRunLog.write("Number of segment IDs with topobathy profiles: "+str(topoTracesUnique)+"\n")
            preRunLog.write("-----\n")
        else:
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "No topobathy profiles."
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)
            preRunLog.write("-----\n")            

        if (not coastal_boundary_depth_df.empty):    
            coastalIDs = coastal_boundary_depth_df.index
            coastalTraces = len(coastalIDs)
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "Number of segments with coastal boundary condition: "+str(coastalTraces)
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)            
            preRunLog.write("-----\n")
        else:
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "No coastal boundary condition."
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)  
            preRunLog.write("-----\n")   

        preRunLog.write("\n")  


def compute_nhd_routing_v02(
    connections,
    rconn,
    wbody_conn,
    reaches_bytw,
    compute_func_name,
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
    flowveldepth_interorder = {},
    from_files = True,
    qlat_add_loc: Literal["top", "middle", "bottom"] = "middle",
):

    da_decay_coefficient = da_parameter_dict.get("da_decay_coefficient", 0)
    param_df["dt"] = dt
    param_df = param_df.astype("float32")
    
    start_time = time.time()
    compute_func = _compute_func_map[compute_func_name]
    if qlat_add_loc == "top":
        qlat_add_loc_c = 0
    elif qlat_add_loc == "middle":
        qlat_add_loc_c = 1
    elif qlat_add_loc == "bottom":
        qlat_add_loc_c = 2
    else:
        raise ValueError(f"qlat_add_loc was {qlat_add_loc}, but must be one of 'top', 'middle', or 'bottom'")
    if parallel_compute_method == "by-subnetwork-jit-clustered":
        
        # Create subnetwork objects if they have not already been created
        if not subnetwork_list[0] or not subnetwork_list[1]:
            networks_with_subnetworks_ordered_jit = nhd_network.build_subnetworks(
                connections, rconn, subnetwork_target_size
            )
            subnetworks_only_ordered_jit = defaultdict(dict)
            subnetworks = defaultdict(dict)
            for tw, ordered_network in networks_with_subnetworks_ordered_jit.items():
                intw = independent_networks[tw]
                for order, subnet_sets in ordered_network.items():
                    subnetworks_only_ordered_jit[order].update(subnet_sets)
                    for subn_tw, subnetwork in subnet_sets.items():
                        subnetworks[subn_tw] = {k: intw[k] for k in subnetwork}
            
            reaches_ordered_bysubntw = defaultdict(dict)
            for order, ordered_subn_dict in subnetworks_only_ordered_jit.items():
                for subn_tw, subnet in ordered_subn_dict.items():
                    conn_subn = {k: connections[k] for k in subnet if k in connections}
                    rconn_subn = {k: rconn[k] for k in subnet if k in rconn}
                    
                    if not waterbodies_df.empty and not usgs_df.empty:
                        path_func = partial(
                            nhd_network.split_at_gages_waterbodies_and_junctions,
                            set(usgs_df.index.values),
                            set(waterbodies_df.index.values),
                            rconn_subn
                            )

                    elif waterbodies_df.empty and not usgs_df.empty:
                        path_func = partial(
                            nhd_network.split_at_gages_and_junctions,
                            set(usgs_df.index.values),
                            rconn_subn
                            )

                    elif not waterbodies_df.empty and usgs_df.empty:
                        path_func = partial(
                            nhd_network.split_at_waterbodies_and_junctions,
                            set(waterbodies_df.index.values),
                            rconn_subn
                            )

                    else:
                        path_func = partial(nhd_network.split_at_junction, rconn_subn)

                    reaches_ordered_bysubntw[order][
                        subn_tw
                    ] = nhd_network.dfs_decomposition(rconn_subn, path_func)

            cluster_threshold = 0.65  # When a job has a total segment count 65% of the target size, compute it
            # Otherwise, keep adding reaches.

            reaches_ordered_bysubntw_clustered = defaultdict(dict)

            for order in subnetworks_only_ordered_jit:
                cluster = 0
                reaches_ordered_bysubntw_clustered[order][cluster] = {
                    "segs": [],
                    "upstreams": {},
                    "tw": [],
                    "subn_reach_list": [],
                }
                for twi, (subn_tw, subn_reach_list) in enumerate(
                    reaches_ordered_bysubntw[order].items(), 1
                ):
                    segs = list(chain.from_iterable(subn_reach_list))
                    reaches_ordered_bysubntw_clustered[order][cluster]["segs"].extend(segs)
                    reaches_ordered_bysubntw_clustered[order][cluster]["upstreams"].update(
                        subnetworks[subn_tw]
                    )

                    reaches_ordered_bysubntw_clustered[order][cluster]["tw"].append(subn_tw)
                    reaches_ordered_bysubntw_clustered[order][cluster][
                        "subn_reach_list"
                    ].extend(subn_reach_list)

                    if (
                        len(reaches_ordered_bysubntw_clustered[order][cluster]["segs"])
                        >= cluster_threshold * subnetwork_target_size
                    ) and (
                        twi
                        < len(reaches_ordered_bysubntw[order])
                        # i.e., we haven't reached the end
                        # TODO: perhaps this should be a while condition...
                    ):
                        cluster += 1
                        reaches_ordered_bysubntw_clustered[order][cluster] = {
                            "segs": [],
                            "upstreams": {},
                            "tw": [],
                            "subn_reach_list": [],
                        }
            

            # save subnetworks_only_ordered_jit and reaches_ordered_bysubntw_clustered in a list
            # to be passed on to next loop. Create a deep copy of this list to prevent it from being
            # altered before being returned
            subnetwork_list = [subnetworks_only_ordered_jit, reaches_ordered_bysubntw_clustered]
            subnetwork_list = copy.deepcopy(subnetwork_list)

        else:
            subnetworks_only_ordered_jit, reaches_ordered_bysubntw_clustered = copy.deepcopy(subnetwork_list)
        
        if 1 == 1:
            LOG.info("JIT Preprocessing time %s seconds." % (time.time() - start_time))
            LOG.info("starting Parallel JIT calculation")
        
        start_para_time = time.time()
        # if 1 == 1:
        with Parallel(n_jobs=cpu_pool, backend="loky") as parallel:
            results_subn = defaultdict(list)

            for order in range(max(subnetworks_only_ordered_jit.keys()), -1, -1):
                jobs = []
                for cluster, clustered_subns in reaches_ordered_bysubntw_clustered[
                    order
                ].items():
                    segs = clustered_subns["segs"]

                    offnetwork_upstreams = set()
                    segs_set = set(segs)
                    for seg in segs:
                        for us in rconn[seg]:
                            if us not in segs_set:
                                offnetwork_upstreams.add(us)

                    segs.extend(offnetwork_upstreams)

                    common_segs = list(param_df.index.intersection(segs))
                    wbodies_segs = set(segs).symmetric_difference(common_segs)

                    #Declare empty dataframe
                    waterbody_types_df_sub = pd.DataFrame()

                    if not waterbodies_df.empty:
                        lake_segs = list(waterbodies_df.index.intersection(segs))
                        waterbodies_df_sub = waterbodies_df.loc[
                            lake_segs,
                            [
                                "LkArea",
                                "LkMxE",
                                "OrificeA",
                                "OrificeC",
                                "OrificeE",
                                "WeirC",
                                "WeirE",
                                "WeirL",
                                "ifd",
                                "qd0",
                                "h0",
                            ],
                        ]

                        #If reservoir types other than Level Pool are active
                        if not waterbody_types_df.empty:
                            waterbody_types_df_sub = waterbody_types_df.loc[
                                lake_segs,
                                [
                                    "reservoir_type",
                                ],
                            ]

                    else:
                        lake_segs = []
                        waterbodies_df_sub = pd.DataFrame()

                    param_df_sub = param_df.loc[
                        common_segs,
                        ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
                    ].sort_index()

                    param_df_sub_super = param_df_sub.reindex(
                        param_df_sub.index.tolist() + lake_segs
                    ).sort_index()

                    for us_subn_tw in offnetwork_upstreams:
                        if us_subn_tw in flowveldepth_interorder:
                            subn_tw_sortposition = param_df_sub_super.index.get_loc(
                                us_subn_tw
                            )
                            flowveldepth_interorder[us_subn_tw][
                                "position_index"
                            ] = subn_tw_sortposition

                    subn_reach_list = clustered_subns["subn_reach_list"]

                    upstreams = clustered_subns["upstreams"]

                    subn_reach_list_with_type = _build_reach_type_list(subn_reach_list, wbodies_segs)

                    qlat_sub = qlats.loc[param_df_sub.index]
                    q0_sub = q0.loc[param_df_sub.index]
                    eloss_sub = eloss_df.loc[param_df_sub.index]
                                        
                    param_df_sub = param_df_sub.reindex(
                        param_df_sub.index.tolist() + lake_segs
                    ).sort_index()

                    usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(usgs_df, lastobs_df, param_df_sub.index, offnetwork_upstreams)
                    da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(subn_reach_list, lastobs_df_sub.index)

                    qlat_sub = qlat_sub.reindex(param_df_sub.index)
                    q0_sub = q0_sub.reindex(param_df_sub.index)
                    eloss_sub = eloss_sub.reindex(param_df_sub.index)

                    # prepare reservoir DA data
                    (reservoir_usgs_df_sub, 
                     reservoir_usgs_df_time,
                     reservoir_usgs_update_time,
                     reservoir_usgs_prev_persisted_flow,
                     reservoir_usgs_persistence_update_time,
                     reservoir_usgs_persistence_index,
                     reservoir_usace_df_sub, 
                     reservoir_usace_df_time,
                     reservoir_usace_update_time,
                     reservoir_usace_prev_persisted_flow,
                     reservoir_usace_persistence_update_time,
                     reservoir_usace_persistence_index,
                     reservoir_usbr_df_sub, 
                     reservoir_usbr_df_time,
                     reservoir_usbr_update_time,
                     reservoir_usbr_prev_persisted_flow,
                     reservoir_usbr_persistence_update_time,
                     reservoir_usbr_persistence_index,
                     reservoir_rfc_df_sub, 
                     reservoir_rfc_totalCounts, 
                     reservoir_rfc_file, 
                     reservoir_rfc_use_forecast, 
                     reservoir_rfc_timeseries_idx, 
                     reservoir_rfc_update_time, 
                     reservoir_rfc_da_timestep, 
                     reservoir_rfc_persist_days,
                     gl_df_sub, 
                     gl_parm_lake_id_sub, 
                     gl_param_flows_sub, 
                     gl_param_time_sub, 
                     gl_param_update_time_sub,
                     gl_climatology_df_sub,
                     waterbody_types_df_sub,
                     ) = _prep_reservoir_da_dataframes(
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
                        waterbody_types_df_sub, 
                        t0,
                        from_files,
                        offnetwork_upstreams
                    )
                    
                    # results_subn[order].append(
                    #     compute_func(
                    jobs.append(
                        delayed(compute_func)(
                            nts,
                            dt,
                            qts_subdivisions,
                            subn_reach_list_with_type,
                            upstreams,
                            param_df_sub.index.values,
                            param_df_sub.columns.values,
                            param_df_sub.values,
                            q0_sub.values.astype("float32"),
                            qlat_sub.values.astype("float32"),
                            eloss_sub.values.astype("float32"),
                            ssout,
                            lake_segs, 
                            waterbodies_df_sub.values,
                            data_assimilation_parameters,
                            waterbody_types_df_sub.values.astype("int32"),
                            waterbody_type_specified,
                            t0.strftime('%Y-%m-%d_%H:%M:%S'),
                            usgs_df_sub.values.astype("float32"),
                            # flowveldepth_interorder,  # obtain keys and values from this dataset
                            np.array(da_positions_list_byseg, dtype="int32"),
                            np.array(da_positions_list_byreach, dtype="int32"),
                            np.array(da_positions_list_bygage, dtype="int32"),
                            lastobs_df_sub.get(
                                "lastobs_discharge",
                                pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32"),
                            ).values.astype("float32"),
                            lastobs_df_sub.get(
                                "time_since_lastobs",
                                pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32"),
                            ).values.astype("float32"),
                            da_decay_coefficient,
                            # USGS Hybrid Reservoir DA data
                            reservoir_usgs_df_sub.values.astype("float32"),
                            reservoir_usgs_df_sub.index.values.astype("int32"),
                            reservoir_usgs_df_time.astype('float32'),
                            reservoir_usgs_update_time.astype('float32'),
                            reservoir_usgs_prev_persisted_flow.astype('float32'),
                            reservoir_usgs_persistence_update_time.astype('float32'),
                            reservoir_usgs_persistence_index.astype('float32'),
                            # USACE Hybrid Reservoir DA data
                            reservoir_usace_df_sub.values.astype("float32"),
                            reservoir_usace_df_sub.index.values.astype("int32"),
                            reservoir_usace_df_time.astype('float32'),
                            reservoir_usace_update_time.astype("float32"),
                            reservoir_usace_prev_persisted_flow.astype("float32"),
                            reservoir_usace_persistence_update_time.astype("float32"),
                            reservoir_usace_persistence_index.astype("float32"),
                            # USBR Hybrid Reservoir DA data
                            reservoir_usbr_df_sub.values.astype("float32"),
                            reservoir_usbr_df_sub.index.values.astype("int32"),
                            reservoir_usbr_df_time.astype("float32"),
                            reservoir_usbr_update_time.astype("float32"),
                            reservoir_usbr_prev_persisted_flow.astype("float32"),
                            reservoir_usbr_persistence_update_time.astype("float32"),
                            reservoir_usbr_persistence_index.astype("float32"),
                            # RFC Reservoir DA data
                            reservoir_rfc_df_sub.values.astype("float32"),
                            reservoir_rfc_df_sub.index.values.astype("int32"),
                            reservoir_rfc_totalCounts.astype("int32"),
                            reservoir_rfc_file,
                            reservoir_rfc_use_forecast.astype("int32"),
                            reservoir_rfc_timeseries_idx.astype("int32"),
                            reservoir_rfc_update_time.astype("float32"),
                            reservoir_rfc_da_timestep.astype("int32"),
                            reservoir_rfc_persist_days.astype("int32"),
                            # Great Lakes DA data
                            gl_df_sub.lake_id.values.astype("int32"),
                            gl_df_sub.time.values.astype("int32"),
                            gl_df_sub.Discharge.values.astype("float32"),
                            gl_parm_lake_id_sub.astype("int32"),
                            gl_param_flows_sub.astype("float32"),
                            gl_param_time_sub.astype("int32"),
                            gl_param_update_time_sub.astype("int32"),
                            gl_climatology_df_sub.values.astype("float32"),
                            {
                                us: fvd
                                for us, fvd in flowveldepth_interorder.items()
                                if us in offnetwork_upstreams
                            },
                            assume_short_ts,
                            return_courant,
                            from_files = from_files,
                            qlat_add_loc = qlat_add_loc_c
                        )
                    )
                results_subn[order] = parallel(jobs)
   
                if order > 0:  # This is not needed for the last rank of subnetworks
                    for ci, (cluster, clustered_subns) in enumerate(
                        reaches_ordered_bysubntw_clustered[order].items()
                    ):
                        for subn_tw in clustered_subns["tw"]:
                            # TODO: This index step is necessary because we sort the segment index
                            # TODO: I think there are a number of ways we could remove the sorting step
                            #       -- the binary search could be replaced with an index based on the known topology
                            flowveldepth_interorder[subn_tw] = {}
                            subn_tw_sortposition = (
                                results_subn[order][ci][0].tolist().index(subn_tw)
                            )
                            flowveldepth_interorder[subn_tw]["results"] = results_subn[
                                order
                            ][ci][1][subn_tw_sortposition]
                            # what will it take to get just the tw FVD values into an array to pass to the next loop?
                            # There will be an empty array initialized at the top of the loop, then re-populated here.
                            # we don't have to bother with populating it after the last group

        results = []
        for order in subnetworks_only_ordered_jit:
            results.extend(results_subn[order])

        if 1 == 1:
            LOG.info("PARALLEL TIME %s seconds." % (time.time() - start_para_time))
        
    elif parallel_compute_method == "by-subnetwork-jit":
        # Create subnetwork objects if they have not already been created
        if not subnetwork_list[0] or not subnetwork_list[1] or not subnetwork_list[2]:
            networks_with_subnetworks_ordered_jit = nhd_network.build_subnetworks(
                connections, rconn, subnetwork_target_size
            )
            subnetworks_only_ordered_jit = defaultdict(dict)
            subnetworks = defaultdict(dict)
            for tw, ordered_network in networks_with_subnetworks_ordered_jit.items():
                intw = independent_networks[tw]
                for order, subnet_sets in ordered_network.items():
                    subnetworks_only_ordered_jit[order].update(subnet_sets)
                    for subn_tw, subnetwork in subnet_sets.items():
                        subnetworks[subn_tw] = {k: intw[k] for k in subnetwork}

            reaches_ordered_bysubntw = defaultdict(dict)
            for order, ordered_subn_dict in subnetworks_only_ordered_jit.items():
                for subn_tw, subnet in ordered_subn_dict.items():
                    conn_subn = {k: connections[k] for k in subnet if k in connections}
                    rconn_subn = {k: rconn[k] for k in subnet if k in rconn}
                    if not waterbodies_df.empty and not usgs_df.empty:
                        path_func = partial(
                            nhd_network.split_at_gages_waterbodies_and_junctions,
                            set(usgs_df.index.values),
                            set(waterbodies_df.index.values),
                            rconn_subn
                            )

                    elif waterbodies_df.empty and not usgs_df.empty:
                        path_func = partial(
                            nhd_network.split_at_gages_and_junctions,
                            set(usgs_df.index.values),
                            rconn_subn
                            )

                    elif not waterbodies_df.empty and usgs_df.empty:
                        path_func = partial(
                            nhd_network.split_at_waterbodies_and_junctions,
                            set(waterbodies_df.index.values),
                            rconn_subn
                            )

                    else:
                        path_func = partial(nhd_network.split_at_junction, rconn_subn)
                    reaches_ordered_bysubntw[order][
                        subn_tw
                    ] = nhd_network.dfs_decomposition(rconn_subn, path_func)

            # save subnetworks_only_ordered_jit and reaches_ordered_bysubntw_clustered in a list
            # to be passed on to next loop. Create a deep copy of this list to prevent it from being
            # altered before being returned
            subnetwork_list = [subnetworks_only_ordered_jit, reaches_ordered_bysubntw, subnetworks]
            subnetwork_list = copy.deepcopy(subnetwork_list)

        else:
            subnetworks_only_ordered_jit, reaches_ordered_bysubntw, subnetworks = subnetwork_list
            
        if 1 == 1:
            LOG.info("JIT Preprocessing time %s seconds." % (time.time() - start_time))
            LOG.info("starting Parallel JIT calculation")

        start_para_time = time.time()
        with Parallel(n_jobs=cpu_pool, backend="loky") as parallel:
            results_subn = defaultdict(list)
            # Preserve any pre-populated entries (e.g., upstream inflow
            # virtual segments from NHF catchment discharge routing).
            _prepopulated_fvd = dict(flowveldepth_interorder)
            flowveldepth_interorder = dict(_prepopulated_fvd)

            for order in range(max(subnetworks_only_ordered_jit.keys()), -1, -1):
                jobs = []
                for twi, (subn_tw, subn_reach_list) in enumerate(
                    reaches_ordered_bysubntw[order].items(), 1
                ):
                    # TODO: Confirm that a list here is best -- we are sorting,
                    # so a set might be sufficient/better
                    segs = list(chain.from_iterable(subn_reach_list))

                    # Identify virtual segments (those with pre-filled FVD
                    # data) that were absorbed into this subnetwork.
                    fvd_keys = set(flowveldepth_interorder.keys())
                    cluster_virtual_segs = fvd_keys & set(segs)
                    if cluster_virtual_segs:
                        segs = [s for s in segs if s not in cluster_virtual_segs]
                        subn_reach_list = [
                            [s for s in reach if s not in cluster_virtual_segs]
                            for reach in subn_reach_list
                        ]
                        subn_reach_list = [r for r in subn_reach_list if r]

                    offnetwork_upstreams = set()
                    segs_set = set(segs)
                    for seg in segs:
                        for us in rconn[seg]:
                            if us not in segs_set:
                                offnetwork_upstreams.add(us)

                    offnetwork_upstreams |= cluster_virtual_segs
                    segs.extend(offnetwork_upstreams)

                    common_segs = list(param_df.index.intersection(segs))
                    wbodies_segs = set(segs).symmetric_difference(common_segs)

                    #Declare empty dataframe
                    waterbody_types_df_sub = pd.DataFrame()

                    if not waterbodies_df.empty:
                        lake_segs = list(waterbodies_df.index.intersection(segs))
                        waterbodies_df_sub = waterbodies_df.loc[
                            lake_segs,
                            [
                                "LkArea",
                                "LkMxE",
                                "OrificeA",
                                "OrificeC",
                                "OrificeE",
                                "WeirC",
                                "WeirE",
                                "WeirL",
                                "ifd",
                                "qd0",
                                "h0",
                            ],
                        ]

                        #If reservoir types other than Level Pool are active
                        if not waterbody_types_df.empty:
                            waterbody_types_df_sub = waterbody_types_df.loc[
                                lake_segs,
                                [
                                    "reservoir_type",
                                ],
                            ]

                    else:
                        lake_segs = []
                        waterbodies_df_sub = pd.DataFrame()

                    param_df_sub = param_df.loc[
                        common_segs,
                        ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
                    ].sort_index()

                    param_df_sub_super = param_df_sub.reindex(
                        param_df_sub.index.tolist() + lake_segs
                    ).sort_index()

                    for us_subn_tw in offnetwork_upstreams:
                        if us_subn_tw in flowveldepth_interorder:
                            subn_tw_sortposition = param_df_sub_super.index.get_loc(
                                us_subn_tw
                            )
                            flowveldepth_interorder[us_subn_tw][
                                "position_index"
                            ] = subn_tw_sortposition

                    subn_reach_list_with_type = _build_reach_type_list(subn_reach_list, wbodies_segs)

                    qlat_sub = qlats.loc[param_df_sub.index]
                    q0_sub = q0.loc[param_df_sub.index]

                    param_df_sub = param_df_sub.reindex(
                        param_df_sub.index.tolist() + lake_segs
                    ).sort_index()

                    usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(usgs_df, lastobs_df, param_df_sub.index, offnetwork_upstreams)
                    da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(subn_reach_list, lastobs_df_sub.index)

                    qlat_sub = qlat_sub.reindex(param_df_sub.index)
                    q0_sub = q0_sub.reindex(param_df_sub.index)
                    
                    # prepare reservoir DA data
                    (reservoir_usgs_df_sub, 
                     reservoir_usgs_df_time,
                     reservoir_usgs_update_time,
                     reservoir_usgs_prev_persisted_flow,
                     reservoir_usgs_persistence_update_time,
                     reservoir_usgs_persistence_index,
                     reservoir_usace_df_sub, 
                     reservoir_usace_df_time,
                     reservoir_usace_update_time,
                     reservoir_usace_prev_persisted_flow,
                     reservoir_usace_persistence_update_time,
                     reservoir_usace_persistence_index,
                     reservoir_usbr_df_sub, 
                     reservoir_usbr_df_time,
                     reservoir_usbr_update_time,
                     reservoir_usbr_prev_persisted_flow,
                     reservoir_usbr_persistence_update_time,
                     reservoir_usbr_persistence_index,
                     reservoir_rfc_df_sub, 
                     reservoir_rfc_totalCounts, 
                     reservoir_rfc_file, 
                     reservoir_rfc_use_forecast, 
                     reservoir_rfc_timeseries_idx, 
                     reservoir_rfc_update_time, 
                     reservoir_rfc_da_timestep, 
                     reservoir_rfc_persist_days,
                     gl_df_sub, 
                     gl_parm_lake_id_sub, 
                     gl_param_flows_sub, 
                     gl_param_time_sub, 
                     gl_param_update_time_sub,
                     gl_climatology_df_sub,
                     waterbody_types_df_sub,
                     ) = _prep_reservoir_da_dataframes(
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
                        waterbody_types_df_sub, 
                        t0,
                        from_files,
                        offnetwork_upstreams
                    )

                    jobs.append(
                        delayed(compute_func)(
                            nts,
                            dt,
                            qts_subdivisions,
                            subn_reach_list_with_type,
                            subnetworks[subn_tw],
                            param_df_sub.index.values,
                            param_df_sub.columns.values,
                            param_df_sub.values,
                            q0_sub.values.astype("float32"),
                            qlat_sub.values.astype("float32"),
                            eloss_sub.values.astype("float32"),
                            ssout,
                            lake_segs,
                            waterbodies_df_sub.values,
                            data_assimilation_parameters,
                            waterbody_types_df_sub.values.astype("int32"),
                            waterbody_type_specified,
                            t0.strftime('%Y-%m-%d_%H:%M:%S'),
                            usgs_df_sub.values.astype("float32"),
                            # flowveldepth_interorder,  # obtain keys and values from this dataset
                            np.array(da_positions_list_byseg, dtype="int32"),
                            np.array(da_positions_list_byreach, dtype="int32"),
                            np.array(da_positions_list_bygage, dtype="int32"),
                            lastobs_df_sub.get(
                                "lastobs_discharge",
                                pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32"),
                            ).values.astype("float32"),
                            lastobs_df_sub.get(
                                "time_since_lastobs",
                                pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32"),
                            ).values.astype("float32"),
                            da_decay_coefficient,
                            # USGS Hybrid Reservoir DA data
                            reservoir_usgs_df_sub.values.astype("float32"),
                            reservoir_usgs_df_sub.index.values.astype("int32"),
                            reservoir_usgs_df_time.astype('float32'),
                            reservoir_usgs_update_time.astype('float32'),
                            reservoir_usgs_prev_persisted_flow.astype('float32'),
                            reservoir_usgs_persistence_update_time.astype('float32'),
                            reservoir_usgs_persistence_index.astype('float32'),
                            # USACE Hybrid Reservoir DA data
                            reservoir_usace_df_sub.values.astype("float32"),
                            reservoir_usace_df_sub.index.values.astype("int32"),
                            reservoir_usace_df_time.astype('float32'),
                            reservoir_usace_update_time.astype("float32"),
                            reservoir_usace_prev_persisted_flow.astype("float32"),
                            reservoir_usace_persistence_update_time.astype("float32"),
                            reservoir_usace_persistence_index.astype("float32"),
                            # USBR Hybrid Reservoir DA data
                            reservoir_usbr_df_sub.values.astype("float32"),
                            reservoir_usbr_df_sub.index.values.astype("int32"),
                            reservoir_usbr_df_time.astype('float32'),
                            reservoir_usbr_update_time.astype("float32"),
                            reservoir_usbr_prev_persisted_flow.astype("float32"),
                            reservoir_usbr_persistence_update_time.astype("float32"),
                            reservoir_usbr_persistence_index.astype("float32"),
                            # RFC Reservoir DA data
                            reservoir_rfc_df_sub.values.astype("float32"),
                            reservoir_rfc_df_sub.index.values.astype("int32"),
                            reservoir_rfc_totalCounts.astype("int32"),
                            reservoir_rfc_file,
                            reservoir_rfc_use_forecast.astype("int32"),
                            reservoir_rfc_timeseries_idx.astype("int32"),
                            reservoir_rfc_update_time.astype("float32"),
                            reservoir_rfc_da_timestep.astype("int32"),
                            reservoir_rfc_persist_days.astype("int32"),
                            # Great Lakes DA data
                            gl_df_sub.lake_id.values.astype("int32"),
                            gl_df_sub.time.values.astype("int32"),
                            gl_df_sub.Discharge.values.astype("float32"),
                            gl_parm_lake_id_sub.astype("int32"),
                            gl_param_flows_sub.astype("float32"),
                            gl_param_time_sub.astype("int32"),
                            gl_param_update_time_sub.astype("int32"),
                            gl_climatology_df_sub.values.astype("float32"),
                            {
                                us: fvd
                                for us, fvd in flowveldepth_interorder.items()
                                if us in offnetwork_upstreams
                            },
                            assume_short_ts,
                            return_courant,
                            from_files=from_files,
                            qlat_add_loc=qlat_add_loc_c
                        )
                    )

                results_subn[order] = parallel(jobs)

                if order > 0:  # This is not needed for the last rank of subnetworks
                    # Start with pre-populated entries (e.g., catchment inflow virtual segments)
                    flowveldepth_interorder = dict(_prepopulated_fvd)
                    for twi, subn_tw in enumerate(reaches_ordered_bysubntw[order]):
                        # TODO: This index step is necessary because we sort the segment index
                        # TODO: I think there are a number of ways we could remove the sorting step
                        #       -- the binary search could be replaced with an index based on the known topology
                        flowveldepth_interorder[subn_tw] = {}
                        subn_tw_sortposition = (
                            results_subn[order][twi][0].tolist().index(subn_tw)
                        )
                        flowveldepth_interorder[subn_tw]["results"] = results_subn[
                            order
                        ][twi][1][subn_tw_sortposition]
                        # what will it take to get just the tw FVD values into an array to pass to the next loop?
                        # There will be an empty array initialized at the top of the loop, then re-populated here.
                        # we don't have to bother with populating it after the last group

        results = []
        for order in subnetworks_only_ordered_jit:
            results.extend(results_subn[order])

        if 1 == 1:
            LOG.info("PARALLEL TIME %s seconds." % (time.time() - start_para_time))

    elif parallel_compute_method == "by-network":
        with Parallel(n_jobs=cpu_pool, backend="loky") as parallel:
            jobs = []
            for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
                # The X_sub lines use SEGS...
                # which is now invalid with the wbodies included.
                # So we define "common_segs" to identify regular routing segments
                # and wbodies_segs for the waterbody reaches/segments
                segs = list(chain.from_iterable(reach_list))
                common_segs = param_df.index.intersection(segs)
                # Assumes everything else is a waterbody...
                wbodies_segs = set(segs).symmetric_difference(common_segs)

                #Declare empty dataframe
                waterbody_types_df_sub = pd.DataFrame()

                # If waterbody parameters exist
                if not waterbodies_df.empty:

                    lake_segs = list(waterbodies_df.index.intersection(segs))

                    waterbodies_df_sub = waterbodies_df.loc[
                        lake_segs,
                        [
                            "LkArea",
                            "LkMxE",
                            "OrificeA",
                            "OrificeC",
                            "OrificeE",
                            "WeirC",
                            "WeirE",
                            "WeirL",
                            "ifd",
                            "qd0",
                            "h0",
                        ],
                    ]

                    #If reservoir types other than Level Pool are active
                    if not waterbody_types_df.empty:
                        waterbody_types_df_sub = waterbody_types_df.loc[
                            lake_segs,
                            [
                                "reservoir_type",
                            ],
                        ]

                else:
                    lake_segs = []
                    waterbodies_df_sub = pd.DataFrame()

                param_df_sub = param_df.loc[
                    common_segs,
                    ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
                ].sort_index()

                reaches_list_with_type = _build_reach_type_list(reach_list, wbodies_segs)

                # qlat_sub = qlats.loc[common_segs].sort_index()
                # q0_sub = q0.loc[common_segs].sort_index()
                qlat_sub = qlats.loc[param_df_sub.index]
                q0_sub = q0.loc[param_df_sub.index]

                param_df_sub = param_df_sub.reindex(
                    param_df_sub.index.tolist() + lake_segs
                ).sort_index()

                usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(usgs_df, lastobs_df, param_df_sub.index)
                da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(reach_list, lastobs_df_sub.index)

                qlat_sub = qlat_sub.reindex(param_df_sub.index)
                q0_sub = q0_sub.reindex(param_df_sub.index)
                
                # prepare reservoir DA data
                (reservoir_usgs_df_sub, 
                 reservoir_usgs_df_time,
                 reservoir_usgs_update_time,
                 reservoir_usgs_prev_persisted_flow,
                 reservoir_usgs_persistence_update_time,
                 reservoir_usgs_persistence_index,
                 reservoir_usace_df_sub, 
                 reservoir_usace_df_time,
                 reservoir_usace_update_time,
                 reservoir_usace_prev_persisted_flow,
                 reservoir_usace_persistence_update_time,
                 reservoir_usace_persistence_index,
                 reservoir_usbr_df_sub, 
                 reservoir_usbr_df_time,
                 reservoir_usbr_update_time,
                 reservoir_usbr_prev_persisted_flow,
                 reservoir_usbr_persistence_update_time,
                 reservoir_usbr_persistence_index,
                 reservoir_rfc_df_sub, 
                 reservoir_rfc_totalCounts, 
                 reservoir_rfc_file, 
                 reservoir_rfc_use_forecast, 
                 reservoir_rfc_timeseries_idx, 
                 reservoir_rfc_update_time, 
                 reservoir_rfc_da_timestep, 
                 reservoir_rfc_persist_days,
                 gl_df_sub, 
                 gl_parm_lake_id_sub, 
                 gl_param_flows_sub,
                 gl_param_time_sub,
                 gl_param_update_time_sub,
                 gl_climatology_df_sub,
                 waterbody_types_df_sub,
                ) = _prep_reservoir_da_dataframes(
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
                    waterbody_types_df_sub, 
                    t0,
                    from_files,
                    set()
                    )

                jobs.append(
                    delayed(compute_func)(
                        nts,
                        dt,
                        qts_subdivisions,
                        reaches_list_with_type,
                        independent_networks[tw],
                        param_df_sub.index.values.astype("int64"),
                        param_df_sub.columns.values,
                        param_df_sub.values,
                        q0_sub.values.astype("float32"),
                        qlat_sub.values.astype("float32"),
                        eloss_sub.values.astype("float32"),
                        ssout,
                        lake_segs,
                        waterbodies_df_sub.values,
                        data_assimilation_parameters,
                        waterbody_types_df_sub.values.astype("int32"),
                        waterbody_type_specified,
                        t0.strftime('%Y-%m-%d_%H:%M:%S'),
                        usgs_df_sub.values.astype("float32"),
                        np.array(da_positions_list_byseg, dtype="int32"),
                        np.array(da_positions_list_byreach, dtype="int32"),
                        np.array(da_positions_list_bygage, dtype="int32"),
                        lastobs_df_sub.get("lastobs_discharge", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                        lastobs_df_sub.get("time_since_lastobs", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                        da_decay_coefficient,
                        # USGS Hybrid Reservoir DA data
                        reservoir_usgs_df_sub.values.astype("float32"),
                        reservoir_usgs_df_sub.index.values.astype("int32"),
                        reservoir_usgs_df_time.astype('float32'),
                        reservoir_usgs_update_time.astype('float32'),
                        reservoir_usgs_prev_persisted_flow.astype('float32'),
                        reservoir_usgs_persistence_update_time.astype('float32'),
                        reservoir_usgs_persistence_index.astype('float32'),
                        # USACE Hybrid Reservoir DA data
                        reservoir_usace_df_sub.values.astype("float32"),
                        reservoir_usace_df_sub.index.values.astype("int32"),
                        reservoir_usace_df_time.astype('float32'),
                        reservoir_usace_update_time.astype("float32"),
                        reservoir_usace_prev_persisted_flow.astype("float32"),
                        reservoir_usace_persistence_update_time.astype("float32"),
                        reservoir_usace_persistence_index.astype("float32"),
                        # USBR Hybrid Reservoir DA data
                        reservoir_usbr_df_sub.values.astype("float32"),
                        reservoir_usbr_df_sub.index.values.astype("int32"),
                        reservoir_usbr_df_time.astype('float32'),
                        reservoir_usbr_update_time.astype("float32"),
                        reservoir_usbr_prev_persisted_flow.astype("float32"),
                        reservoir_usbr_persistence_update_time.astype("float32"),
                        reservoir_usbr_persistence_index.astype("float32"),
                        # RFC Reservoir DA data
                        reservoir_rfc_df_sub.values.astype("float32"),
                        reservoir_rfc_df_sub.index.values.astype("int32"),
                        reservoir_rfc_totalCounts.astype("int32"),
                        reservoir_rfc_file,
                        reservoir_rfc_use_forecast.astype("int32"),
                        reservoir_rfc_timeseries_idx.astype("int32"),
                        reservoir_rfc_update_time.astype("float32"),
                        reservoir_rfc_da_timestep.astype("int32"),
                        reservoir_rfc_persist_days.astype("int32"),
                        # Great Lakes DA data
                        gl_df_sub.lake_id.values.astype("int32"),
                        gl_df_sub.time.values.astype("int32"),
                        gl_df_sub.Discharge.values.astype("float32"),
                        gl_parm_lake_id_sub.astype("int32"),
                        gl_param_flows_sub.astype("float32"),
                        gl_param_time_sub.astype("int32"),
                        gl_param_update_time_sub.astype("int32"),
                        gl_climatology_df_sub.values.astype("float32"),
                        {},
                        assume_short_ts,
                        return_courant,
                        from_files=from_files,
                    )
                )

            results = parallel(jobs)

    elif parallel_compute_method == "serial":
        results = []
        for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
            # The X_sub lines use SEGS...
            # which becomes invalid with the wbodies included.
            # So we define "common_segs" to identify regular routing segments
            # and wbodies_segs for the waterbody reaches/segments
            segs = list(chain.from_iterable(reach_list))
            common_segs = param_df.index.intersection(segs)
            # Assumes everything else is a waterbody...
            wbodies_segs = set(segs).symmetric_difference(common_segs)

            #Declare empty dataframe
            waterbody_types_df_sub = pd.DataFrame()

            # If waterbody parameters exist
            if not waterbodies_df.empty:

                lake_segs = list(waterbodies_df.index.intersection(segs))

                waterbodies_df_sub = waterbodies_df.loc[
                    lake_segs,
                    [
                        "LkArea",
                        "LkMxE",
                        "OrificeA",
                        "OrificeC",
                        "OrificeE",
                        "WeirC",
                        "WeirE",
                        "WeirL",
                        "ifd",
                        "qd0",
                        "h0",
                    ],
                ]

                #If reservoir types other than Level Pool are active
                if not waterbody_types_df.empty:
                    waterbody_types_df_sub = waterbody_types_df.loc[
                        lake_segs,
                        [
                            "reservoir_type",
                        ],
                    ]

            else:
                lake_segs = []
                waterbodies_df_sub = pd.DataFrame()

            param_df_sub = param_df.loc[
                common_segs,
                ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
            ].sort_index()

            reaches_list_with_type = _build_reach_type_list(reach_list, wbodies_segs)

            # qlat_sub = qlats.loc[common_segs].sort_index()
            # q0_sub = q0.loc[common_segs].sort_index()
            qlat_sub = qlats.loc[param_df_sub.index]
            q0_sub = q0.loc[param_df_sub.index]

            param_df_sub = param_df_sub.reindex(
                param_df_sub.index.tolist() + lake_segs
            ).sort_index()

            usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(usgs_df, lastobs_df, param_df_sub.index)
            da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(reach_list, lastobs_df_sub.index)

            qlat_sub = qlat_sub.reindex(param_df_sub.index)
            q0_sub = q0_sub.reindex(param_df_sub.index)
            
            # prepare reservoir DA data
            (reservoir_usgs_df_sub, 
             reservoir_usgs_df_time,
             reservoir_usgs_update_time,
             reservoir_usgs_prev_persisted_flow,
             reservoir_usgs_persistence_update_time,
             reservoir_usgs_persistence_index,
             reservoir_usace_df_sub, 
             reservoir_usace_df_time,
             reservoir_usace_update_time,
             reservoir_usace_prev_persisted_flow,
             reservoir_usace_persistence_update_time,
             reservoir_usace_persistence_index,
             reservoir_usbr_df_sub, 
             reservoir_usbr_df_time,
             reservoir_usbr_update_time,
             reservoir_usbr_prev_persisted_flow,
             reservoir_usbr_persistence_update_time,
             reservoir_usbr_persistence_index,
             reservoir_rfc_df_sub, 
             reservoir_rfc_totalCounts, 
             reservoir_rfc_file, 
             reservoir_rfc_use_forecast, 
             reservoir_rfc_timeseries_idx, 
             reservoir_rfc_update_time, 
             reservoir_rfc_da_timestep, 
             reservoir_rfc_persist_days,
             gl_df_sub,
             gl_parm_lake_id_sub, 
             gl_param_flows_sub,
             gl_param_time_sub,
             gl_param_update_time_sub,
             gl_climatology_df_sub,
             waterbody_types_df_sub,
             ) = _prep_reservoir_da_dataframes(
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
                waterbody_types_df_sub, 
                t0,
                from_files,
                )
            
            results.append(
                compute_func(
                    nts,
                    dt,
                    qts_subdivisions,
                    reaches_list_with_type,
                    independent_networks[tw],
                    param_df_sub.index.values.astype("int64"),
                    param_df_sub.columns.values,
                    param_df_sub.values,
                    q0_sub.values.astype("float32"),
                    qlat_sub.values.astype("float32"),
                    eloss_sub.values.astype("float32"),
                    ssout,
                    lake_segs,
                    waterbodies_df_sub.values,
                    data_assimilation_parameters,
                    waterbody_types_df_sub.values.astype("int32"),
                    waterbody_type_specified,
                    t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    usgs_df_sub.values.astype("float32"),
                    np.array(da_positions_list_byseg, dtype="int32"),
                    np.array(da_positions_list_byreach, dtype="int32"),
                    np.array(da_positions_list_bygage, dtype="int32"),
                    lastobs_df_sub.get("lastobs_discharge", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                    lastobs_df_sub.get("time_since_lastobs", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                    da_decay_coefficient,
                    # USGS Hybrid Reservoir DA data
                    reservoir_usgs_df_sub.values.astype("float32"),
                    reservoir_usgs_df_sub.index.values.astype("int32"),
                    reservoir_usgs_df_time.astype('float32'),
                    reservoir_usgs_update_time.astype('float32'),
                    reservoir_usgs_prev_persisted_flow.astype('float32'),
                    reservoir_usgs_persistence_update_time.astype('float32'),
                    reservoir_usgs_persistence_index.astype('float32'),
                    # USACE Hybrid Reservoir DA data
                    reservoir_usace_df_sub.values.astype("float32"),
                    reservoir_usace_df_sub.index.values.astype("int32"),
                    reservoir_usace_df_time.astype('float32'),
                    reservoir_usace_update_time.astype("float32"),
                    reservoir_usace_prev_persisted_flow.astype("float32"),
                    reservoir_usace_persistence_update_time.astype("float32"),
                    reservoir_usace_persistence_index.astype("float32"),
                    # USBR Hybrid Reservoir DA data
                    reservoir_usbr_df_sub.values.astype("float32"),
                    reservoir_usbr_df_sub.index.values.astype("int32"),
                    reservoir_usbr_df_time.astype('float32'),
                    reservoir_usbr_update_time.astype("float32"),
                    reservoir_usbr_prev_persisted_flow.astype("float32"),
                    reservoir_usbr_persistence_update_time.astype("float32"),
                    reservoir_usbr_persistence_index.astype("float32"),
                    # RFC Reservoir DA data
                    reservoir_rfc_df_sub.values.astype("float32"),
                    reservoir_rfc_df_sub.index.values.astype("int32"),
                    reservoir_rfc_totalCounts.astype("int32"),
                    reservoir_rfc_file,
                    reservoir_rfc_use_forecast.astype("int32"),
                    reservoir_rfc_timeseries_idx.astype("int32"),
                    reservoir_rfc_update_time.astype("float32"),
                    reservoir_rfc_da_timestep.astype("int32"),
                    reservoir_rfc_persist_days.astype("int32"),
                    # Great Lakes DA data
                    gl_df_sub.lake_id.values.astype("int32"),
                    gl_df_sub.time.values.astype("int32"),
                    gl_df_sub.Discharge.values.astype("float32"),
                    gl_parm_lake_id_sub.astype("int32"),
                    gl_param_flows_sub.astype("float32"),
                    gl_param_time_sub.astype("int32"),
                    gl_param_update_time_sub.astype("int32"),
                    gl_climatology_df_sub.values.astype("float32"),
                    {},
                    assume_short_ts,
                    return_courant,
                    from_files=from_files,
                )
            )


    elif parallel_compute_method == "bmi":
        results = []
        for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
            # The X_sub lines use SEGS...
            # which becomes invalid with the wbodies included.
            # So we define "common_segs" to identify regular routing segments
            # and wbodies_segs for the waterbody reaches/segments
            segs = list(chain.from_iterable(reach_list))
            offnetwork_upstreams = set(flowveldepth_interorder.keys())
            segs.extend(offnetwork_upstreams)
            common_segs = param_df.index.intersection(segs)
            # Assumes everything else is a waterbody...
            wbodies_segs = set(segs).symmetric_difference(common_segs)

            #Declare empty dataframe
            waterbody_types_df_sub = pd.DataFrame()

            # If waterbody parameters exist
            if not waterbodies_df.empty:

                lake_segs = list(waterbodies_df.index.intersection(segs))

                waterbodies_df_sub = waterbodies_df.loc[
                    lake_segs,
                    [
                        "LkArea",
                        "LkMxE",
                        "OrificeA",
                        "OrificeC",
                        "OrificeE",
                        "WeirC",
                        "WeirE",
                        "WeirL",
                        "ifd",
                        "qd0",
                        "h0",
                    ],
                ]

                #If reservoir types other than Level Pool are active
                if not waterbody_types_df.empty:
                    waterbody_types_df_sub = waterbody_types_df.loc[
                        lake_segs,
                        [
                            "reservoir_type",
                        ],
                    ]

            else:
                lake_segs = []
                waterbodies_df_sub = pd.DataFrame()

            param_df_sub = param_df.loc[
                common_segs,
                ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
            ].sort_index()

            reaches_list_with_type = _build_reach_type_list(reach_list, wbodies_segs)

            # qlat_sub = qlats.loc[common_segs].sort_index()
            # q0_sub = q0.loc[common_segs].sort_index()

            qlat_sub = qlats.loc[param_df_sub.drop(offnetwork_upstreams).index]
            q0_sub = q0.loc[param_df_sub.index]

            param_df_sub = param_df_sub.reindex(
                param_df_sub.index.tolist() + lake_segs
            ).sort_index()
            
            for us_subn_tw in offnetwork_upstreams:
                subn_tw_sortposition = param_df_sub.index.get_loc(
                    us_subn_tw
                )
                flowveldepth_interorder[us_subn_tw][
                    "position_index"
                ] = subn_tw_sortposition

            usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(usgs_df, lastobs_df, param_df_sub.index)
            da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(reach_list, lastobs_df_sub.index)

            qlat_sub = qlat_sub.reindex(np.setdiff1d(param_df_sub.index, offnetwork_upstreams))
            q0_sub = q0_sub.reindex(param_df_sub.index)
            
            # prepare reservoir DA data
            (reservoir_usgs_df_sub, 
             reservoir_usgs_df_time,
             reservoir_usgs_update_time,
             reservoir_usgs_prev_persisted_flow,
             reservoir_usgs_persistence_update_time,
             reservoir_usgs_persistence_index,
             reservoir_usace_df_sub, 
             reservoir_usace_df_time,
             reservoir_usace_update_time,
             reservoir_usace_prev_persisted_flow,
             reservoir_usace_persistence_update_time,
             reservoir_usace_persistence_index,
             reservoir_usbr_df_sub, 
             reservoir_usbr_df_time,
             reservoir_usbr_update_time,
             reservoir_usbr_prev_persisted_flow,
             reservoir_usbr_persistence_update_time,
             reservoir_usbr_persistence_index,
             waterbody_types_df_sub,
             ) = _prep_reservoir_da_dataframes(
                reservoir_usgs_df,
                reservoir_usgs_param_df,
                reservoir_usace_df, 
                reservoir_usace_param_df,
                reservoir_usbr_df, 
                reservoir_usbr_param_df,
                waterbody_types_df_sub, 
                t0,
                from_files,
            )
            
            results.append(
                compute_func(
                    nts,
                    dt,
                    qts_subdivisions,
                    reaches_list_with_type,
                    independent_networks[tw],
                    param_df_sub.index.values.astype("int64"),
                    param_df_sub.columns.values,
                    param_df_sub.values,
                    q0_sub.values.astype("float32"),
                    qlat_sub.values.astype("float32"),
                    eloss_sub.values.astype("float32"),
                    ssout,
                    lake_segs,
                    waterbodies_df_sub.values,
                    data_assimilation_parameters,
                    waterbody_types_df_sub.values.astype("int32"),
                    waterbody_type_specified,
                    t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    usgs_df_sub.values.astype("float32"),
                    np.array(da_positions_list_byseg, dtype="int32"),
                    np.array(da_positions_list_byreach, dtype="int32"),
                    np.array(da_positions_list_bygage, dtype="int32"),
                    lastobs_df_sub.get("lastobs_discharge", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                    lastobs_df_sub.get("time_since_lastobs", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
                    da_decay_coefficient,
                    # USGS Hybrid Reservoir DA data
                    reservoir_usgs_df_sub.values.astype("float32"),
                    reservoir_usgs_df_sub.index.values.astype("int32"),
                    reservoir_usgs_df_time.astype('float32'),
                    reservoir_usgs_update_time.astype('float32'),
                    reservoir_usgs_prev_persisted_flow.astype('float32'),
                    reservoir_usgs_persistence_update_time.astype('float32'),
                    reservoir_usgs_persistence_index.astype('float32'),
                    # USACE Hybrid Reservoir DA data
                    reservoir_usace_df_sub.values.astype("float32"),
                    reservoir_usace_df_sub.index.values.astype("int32"),
                    reservoir_usace_df_time.astype('float32'),
                    reservoir_usace_update_time.astype("float32"),
                    reservoir_usace_prev_persisted_flow.astype("float32"),
                    reservoir_usace_persistence_update_time.astype("float32"),
                    reservoir_usace_persistence_index.astype("float32"),
                    # USBR Hybrid Reservoir DA data
                    reservoir_usbr_df_sub.values.astype("float32"),
                    reservoir_usbr_df_sub.index.values.astype("int32"),
                    reservoir_usbr_df_time.astype('float32'),
                    reservoir_usbr_update_time.astype("float32"),
                    reservoir_usbr_prev_persisted_flow.astype("float32"),
                    reservoir_usbr_persistence_update_time.astype("float32"),
                    reservoir_usbr_persistence_index.astype("float32"),
                    {
                        us: fvd
                        for us, fvd in flowveldepth_interorder.items()
                        if us in offnetwork_upstreams
                    },
                    assume_short_ts,
                    return_courant,
                )
            )

    return RoutingResultsCollection(results), subnetwork_list

def compute_diffusive_routing(
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
    topobathy,
    refactored_diffusive_domain,
    refactored_reaches,
    coastal_boundary_depth_df, 
    unrefactored_topobathy,
    ):

    results_diffusive = []
    for tw in diffusive_network_data: # <------- TODO - by-network parallel loop, here.
        trib_segs = None
        trib_flow = None
        # extract junction inflows from results array
        for j, i in enumerate(results):
            x = np.in1d(i[0], diffusive_network_data[tw]['tributary_segments'])
            if sum(x) > 0:
                if j == 0:
                    trib_segs = i[0][x]
                    trib_flow = i[1][x, ::3]
                else:
                    if trib_segs is None:
                        trib_segs = i[0][x]
                        trib_flow = i[1][x, ::3]                        
                    else:
                        trib_segs = np.append(trib_segs, i[0][x])
                        trib_flow = np.append(trib_flow, i[1][x, ::3], axis = 0)  

        # create DataFrame of junction inflow data            
        junction_inflows = pd.DataFrame(data = trib_flow, index = trib_segs)

        if not topobathy.empty:
            # create topobathy data for diffusive mainstem segments related to this given tw segment        
            if refactored_diffusive_domain:
                topobathy_bytw               = topobathy.loc[refactored_diffusive_domain[tw]['rlinks']] 
                # TODO: missing topobathy data in one of diffuisve domains, so inactivate the next line for now. 
                #unrefactored_topobathy_bytw  = unrefactored_topobathy.loc[diffusive_network_data[tw]['mainstem_segs']]
                unrefactored_topobathy_bytw = pd.DataFrame()
            else:
                topobathy_bytw               = topobathy.loc[diffusive_network_data[tw]['mainstem_segs']] 
                unrefactored_topobathy_bytw = pd.DataFrame()
            
        else:
            topobathy_bytw = pd.DataFrame()
            unrefactored_topobathy_bytw = pd.DataFrame()

        # diffusive streamflow DA activation switch
        #if da_parameter_dict['diffusive_streamflow_nudging']==True:
        if 'diffusive_streamflow_nudging' in da_parameter_dict:
            diffusive_usgs_df = usgs_df
        else:
            diffusive_usgs_df = pd.DataFrame()

        # tw in refactored hydrofabric
        if refactored_diffusive_domain:
            refactored_tw = refactored_diffusive_domain[tw]['refac_tw']
            refactored_diffusive_domain_bytw = refactored_diffusive_domain[tw]
            refactored_reaches_byrftw        = refactored_reaches[refactored_tw]
        else:
            refactored_diffusive_domain_bytw = None
            refactored_reaches_byrftw        = None
  
        # coastal boundary depth input data at TW
        if tw in coastal_boundary_depth_df.index:
            coastal_boundary_depth_bytw_df = coastal_boundary_depth_df.loc[tw].to_frame().T
        else:
            coastal_boundary_depth_bytw_df = pd.DataFrame()

        # temporary: column names of qlats from HYfeature are currently timestamps. To be consistent with qlats from NHD
        # the column names need to be changed to intergers from zero incrementing by 1
        diffusive_qlats = qlats.copy()
        diffusive_qlats.columns = range(diffusive_qlats.shape[1])  

        # build diffusive inputs
        diffusive_inputs = diff_utils.diffusive_input_data_v02(
            tw,
            diffusive_network_data[tw]['connections'],
            diffusive_network_data[tw]['rconn'],
            diffusive_network_data[tw]['reaches'],
            diffusive_network_data[tw]['mainstem_segs'],
            diffusive_network_data[tw]['tributary_segments'],
            None, # place holder for diffusive parameters
            diffusive_network_data[tw]['param_df'],
            diffusive_qlats,
            q0,
            junction_inflows,
            qts_subdivisions,
            t0,
            nts,
            dt,
            waterbodies_df,
            topobathy_bytw,
            diffusive_usgs_df,
            refactored_diffusive_domain_bytw,
            refactored_reaches_byrftw, 
            coastal_boundary_depth_bytw_df,
            unrefactored_topobathy_bytw,
        )

        # run the simulation
        out_q, out_elv, out_depth = diffusive.compute_diffusive(diffusive_inputs)

        # unpack results
        rch_list, dat_all = diff_utils.unpack_output(
            diffusive_inputs['pynw'], 
            diffusive_inputs['ordered_reaches'], 
            out_q, 
            out_depth, #out_elv
        )
        
        # mask segments for which we already have MC solution
        x = np.in1d(rch_list, diffusive_network_data[tw]['tributary_segments'])
        
        results_diffusive.append(
            (
                rch_list[~x], dat_all[~x,3:], 0,
                # place-holder for streamflow DA parameters
                (np.asarray([]), np.asarray([]), np.asarray([])),
                # place-holder for reservoir DA parameters
                (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])),
                (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])),
                # place holder for reservoir inflows
                np.zeros(dat_all[~x,3::3].shape),
                # place-holder for rfc DA parameters
                (np.asarray([]), np.asarray([]), np.asarray([])),
                # place-holder for nudge values
                (np.empty(shape=(0, nts + 1), dtype='float32')),
                # place-holder for great lakes DA values/parameters
                (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])),
            )
        )

    return results_diffusive


class _RoutingResultsParser:
    def __init__(self, raw_results: tuple):
        self._raw = raw_results

    def __getitem__(self, index: int):
        return self._raw[index]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    @property
    def ids(self) -> np.ndarray[tuple[int], np.intp]:
        """Segment IDs as 1D array"""
        return self._raw[0]

    def _append(self, a: np.ndarray, b: np.ndarray):
        axis = len(a.shape) - 1
        return np.concatenate([a, b], axis=axis)

    def append(self, other: _RoutingResultsParser):
        copy = list(self)
        # skip first element as that is the ID
        for i in range(1, len(self)):
            copy[i] = self._append(self[i], other[i])
        return self.__class__(copy)

    @classmethod
    def merge(cls, to_merge: list[_RoutingResultsParser]):
        size = len(to_merge[0])
        data = [None] * size
        for i in range(size):
            data[i] = np.concatenate([r[i] for r in to_merge], axis=0)
        return cls(data)

    def align_ids(self, source: _RoutingResultsParser):
        if self.ids.size and not np.array_equal(self.ids, source.ids):
            copy = [None] * len(self)
            sorter = np.argsort(source.ids)
            for i, item in enumerate(self):
                copy[i] = item[sorter]
            return self.__class__(copy)
        return self

    def _set_index(self, value, index: int):
        if isinstance(self._raw, tuple):
            self._raw = list(self._raw)
        self._raw[index] = value


class RoutingResultsCollection:
    def __init__(self, results: Iterable[tuple]):
        self.results = [RoutingResults(r) for r in results]

    def __getitem__(self, index: int):
        return self.results[index]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def flow_velocity_depth(self, nts: int, drop_ql: bool = False):
        columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d", "ql"]]
        ).to_flat_index()
        dfs = []
        for result in self.results:
            df = pd.DataFrame(
                result.flow,
                index=result.ids,
                columns=columns,
            )
            dfs.append(df)
        flowveldepth = pd.concat(dfs, copy=False)
        if drop_ql:
            flowveldepth = flowveldepth.drop(columns=[
                col for col in flowveldepth.columns if col[1] == "ql"
            ])
        return flowveldepth

    def waterbodies(self, nts: int):
        columns = pd.MultiIndex.from_product(
            [range(nts), ["i"]]
        ).to_flat_index()
        dfs = []
        for result in self.results:
            df = pd.DataFrame(
                result.upstream,
                index=result.ids,
                columns=columns,
            )
            dfs.append(df)
        return pd.concat(dfs, copy=False)

    def courant(self, nts: int):
        columns = pd.MultiIndex.from_product(
            [range(nts), ["cn", "ck", "X"]]
        ).to_flat_index()
        dfs = []
        for result in self.results:
            df = pd.DataFrame(
                result.courant,
                index=result.ids,
                columns=columns,
            )
            dfs.append(df)
        return pd.concat(dfs, copy=False)

    def nudge(self):
        return np.concatenate(
            [result.nudge for result in self.results]
        )

    def usgs_position_ids(self):
        return np.concatenate(
            [result.usgs_reservoir.ids for result in self.results]
        )

    def merged_results(self) -> RoutingResults:
        """Merge the separate results into one single results."""
        if len(self.results) > 1:
            merged = RoutingResults([None] * len(self.results[0]))
            merged.ids = np.concatenate([r.ids for r in self.results])
            merged.flow = np.concatenate([r.flow for r in self.results], axis=0)
            merged.courant = 0 # fix when this is no longer a placeholder
            merged.lastobs = RoutingLastObs.merge([r.lastobs for r in self.results])
            merged.usgs_reservoir = RoutingReservoir.merge([r.usgs_reservoir for r in self.results])
            merged.usace_reservoir = RoutingReservoir.merge([r.usace_reservoir for r in self.results])
            merged.usbr_reservoir = RoutingReservoir.merge([r.usbr_reservoir for r in self.results])
            merged.upstream = np.concatenate([r.upstream for r in self.results], axis=0)
            merged.rfc_reservoir = RoutingRfc.merge([r.rfc_reservoir for r in self.results])
            merged.nudge = np.concatenate([r.nudge for r in self.results], axis=0)
            merged.great_lakes = RoutingGreatLakes.merge([r.great_lakes for r in self.results])
            return merged
        return self.results[0]

    def append_timesteps(self, other: RoutingResultsCollection):
        a = self.merged_results()
        b = other.merged_results()
        b = b.align_ids(a)
        return RoutingResultsCollection([a.append(b)])


class RoutingResults(_RoutingResultsParser):
    def align_ids(self, source: RoutingResults):
        if not np.array_equal(self.ids, source.ids):
            self = RoutingResults(list(self))
            sorter = np.argsort(source.ids)
            self.ids = self.ids[sorter]
            self.flow = self.flow[sorter]
            if self.upstream.size > 0:
                self.upstream = self.upstream[sorter]
            if self.nudge.size > 0:
                self.nudge = self.nudge[sorter]
        self.usgs_reservoir = self.usgs_reservoir.align_ids(source.usgs_reservoir)
        self.usace_reservoir = self.usace_reservoir.align_ids(source.usace_reservoir)
        self.usbr_reservoir = self.usbr_reservoir.align_ids(source.usbr_reservoir)
        self.rfc_reservoir = self.rfc_reservoir.align_ids(source.rfc_reservoir)
        self.great_lakes = self.great_lakes.align_ids(source.great_lakes)
        return self

    def append(self, other: RoutingResults):
        appended = RoutingResults(list(self))
        appended.flow = self._append(self.flow, other.flow)
        appended.lastobs = self.lastobs.append(other.lastobs)
        appended.usgs_reservoir = self.usgs_reservoir.append(other.usgs_reservoir)
        appended.usace_reservoir = self.usace_reservoir.append(other.usace_reservoir)
        appended.usbr_reservoir = self.usbr_reservoir.append(other.usbr_reservoir)
        appended.upstream = self._append(self.upstream, other.upstream)
        appended.rfc_reservoir = self.rfc_reservoir.append(other.rfc_reservoir)
        # remove leading timestep from other's nudge
        appended.nudge = self._append(self.nudge, other.nudge[:, 1:])
        appended.great_lakes = self.great_lakes.append(other.great_lakes)
        return appended

    @property
    def ids(self) -> np.ndarray[tuple[int], np.intp]:
        """Catchment IDs as 1D array"""
        return self._raw[0]
    @ids.setter
    def ids(self, value):
        self._set_index(value, 0)

    @property
    def flow(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        """Flow velocity depth 2D array: (num_ids, nts * 4)"""
        return self._raw[1]
    @flow.setter
    def flow(self, value):
        self._set_index(value, 1)

    @property
    def courant(self) -> Literal[0]:
        """Is currently a placeholder, so the value will always be 0."""
        return self._raw[2]
    @courant.setter
    def courant(self, value):
        self._set_index(value, 2)

    @property
    def lastobs(self):
        return RoutingLastObs(self._raw[3])
    @lastobs.setter
    def lastobs(self, value):
        self._set_index(list(value), 3)

    @property
    def usgs_reservoir(self):
        return RoutingReservoir(self._raw[4])
    @usgs_reservoir.setter
    def usgs_reservoir(self, value):
        self._set_index(list(value), 4)

    @property
    def usace_reservoir(self):
        return RoutingReservoir(self._raw[5])
    @usace_reservoir.setter
    def usace_reservoir(self, value):
        self._set_index(list(value), 5)

    @property
    def usbr_reservoir(self):
        return RoutingReservoir(self._raw[6])
    @usbr_reservoir.setter
    def usbr_reservoir(self, value):
        self._set_index(list(value), 6)

    @property
    def upstream(self) -> np.ndarray[tuple[int], np.float32]:
        """Upstream 2D array: (num_ids, nts)"""
        return self._raw[7]
    @upstream.setter
    def upstream(self, value):
        self._set_index(value, 7)

    @property
    def rfc_reservoir(self):
        return RoutingRfc(self._raw[8])
    @rfc_reservoir.setter
    def rfc_reservoir(self, value):
        self._set_index(value, 8)

    @property
    def nudge(self) -> np.ndarray[tuple[int, int], np.float32]:
        """Nudge 2D array: (num_ids, nts + 1)"""
        return self._raw[9]
    @nudge.setter
    def nudge(self, value):
        self._set_index(value, 9)

    @property
    def great_lakes(self):
        return RoutingGreatLakes(self._raw[10])
    @great_lakes.setter
    def great_lakes(self, value):
        self._set_index(list(value), 10)


class RoutingLastObs(_RoutingResultsParser):
    @property
    def times(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[1]

    @property
    def values(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[2]


class RoutingReservoir(_RoutingResultsParser):
    @property
    def update_times(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[1]

    @property
    def persisted_outflow(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[2]

    @property
    def persistence_index(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[3]

    @property
    def persistence_update_time(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[4]


class RoutingRfc(_RoutingResultsParser):
    @property
    def update_times(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[1]

    @property
    def timeseries(self) -> np.ndarray[tuple[int], np.intp]:
        return self._raw[2]


class RoutingGreatLakes(_RoutingResultsParser):
    @property
    def outflows(self) -> np.ndarray[tuple[int], np.float32]:
        return self._raw[1]

    @property
    def timestamps(self) -> np.ndarray[tuple[int], np.intp]:
        return self._raw[2]

    @property
    def update_times(self) -> np.ndarray[tuple[int], np.intp]:
        return self._raw[3]
