from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from functools import partial
from tlz import concat
from joblib import delayed, Parallel
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import copy

import troute.nhd_network as nhd_network
from troute.routing.fast_reach.mc_reach import compute_network_structured
import troute.routing.diffusive_utils_v02 as diff_utils
from troute.routing.fast_reach import diffusive

import logging

LOG = logging.getLogger('')

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Mapping, List
    from pathlib import Path

_compute_func_map = defaultdict(
    compute_network_structured,
    {
        "V02-structured": compute_network_structured,
    },
)

# ------------------------------------------------------------------------------
# Abstract Compute Class:
# Define all slots and __init__, then pass function definitions to child classes
# ------------------------------------------------------------------------------
class AbstractCompute(ABC):
    """
    This just defines all of the slots that are be used by child classes.
    These need to be defined in a parent class so each child class can be
    combined into a single DataAssimilation object without getting a 
    'multiple-inheritance' error.
    """
    __slots__ = ["_connections", "_rconn", "_reaches_bytw", "_compute_func_name", "_subnetwork_target_size", "_cpu_pool", "_t0",
                 "_dt", "_nts", "_qts_subdivisions", "_independent_networks", "_param_df", "_q0", "_qlats", "_usgs_df", "_lastobs_df",
                 "_reservoir_usgs_df", "_reservoir_usgs_param_df", "_reservoir_usace_df", "_reservoir_usace_param_df",
                 "_reservoir_rfc_df", "_reservoir_rfc_param_df", "_da_parameter_dict", "_assume_short_ts", "_return_courant",
                 "_waterbodies_df", "_data_assimilation_parameters", "_waterbody_types_df", "_waterbody_type_specified",
                 "_subnetwork_list", "_flowveldepth_interorder", "_from_files", "_results",]
    
    def __init__(self, 
                 connections: dict[int, List[int]],
                 rconn: dict[int, List[int]],
                 reaches_bytw: dict[int, List[List[int]]],
                 compute_func_name: str,
                 parallel_compute_method: str,
                 subnetwork_target_size: int,
                 cpu_pool: int,
                 t0: datetime,
                 dt: int,
                 nts: int,
                 qts_subdivisions: int,
                 independent_networks: dict[int, dict[int, List[int]]],
                 param_df: pd.DataFrame,
                 q0: pd.DataFrame,
                 qlats: pd.DataFrame,
                 usgs_df: pd.DataFrame,
                 lastobs_df: pd.DataFrame,
                 reservoir_usgs_df: pd.DataFrame,
                 reservoir_usgs_param_df: pd.DataFrame,
                 reservoir_usace_df: pd.DataFrame,
                 reservoir_usace_param_df: pd.DataFrame,
                 reservoir_rfc_df: pd.DataFrame,
                 reservoir_rfc_param_df: pd.DataFrame,
                 da_parameter_dict: dict,
                 assume_short_ts: bool,
                 return_courant: bool,
                 waterbodies_df: pd.DataFrame,
                 data_assimilation_parameters: dict,
                 waterbody_types_df: pd.DataFrame,
                 waterbody_type_specified: bool,
                 subnetwork_list: list,
                 flowveldepth_interorder: dict = {},
                 from_files: bool = True) -> None:
        """
        Run subnetworking pre-processing, then computing.
        """
        self._connections = connections
        self._rconn = rconn
        self._reaches_bytw = reaches_bytw
        self._compute_func_name = compute_func_name
        self._subnetwork_target_size = subnetwork_target_size
        self._cpu_pool = cpu_pool
        self._t0 = t0
        self._dt = dt
        self._nts = nts
        self._qts_subdivisions = qts_subdivisions
        self._independent_networks = independent_networks
        self._param_df = param_df
        self._q0 = q0
        self._qlats = qlats
        self._usgs_df = usgs_df
        self._lastobs_df = lastobs_df
        self._reservoir_usgs_df = reservoir_usgs_df
        self._reservoir_usgs_param_df = reservoir_usgs_param_df
        self._reservoir_usace_df = reservoir_usace_df
        self._reservoir_usace_param_df = reservoir_usace_param_df
        self._reservoir_rfc_df = reservoir_rfc_df
        self._reservoir_rfc_param_df = reservoir_rfc_param_df
        self._da_parameter_dict = da_parameter_dict
        self._assume_short_ts = assume_short_ts
        self._return_courant = return_courant
        self._waterbodies_df = waterbodies_df
        self._data_assimilation_parameters = data_assimilation_parameters
        self._waterbody_types_df = waterbody_types_df
        self._waterbody_type_specified = waterbody_type_specified
        self._subnetwork_list = subnetwork_list
        self._flowveldepth_interorder = flowveldepth_interorder
        self._from_files = from_files
        
        self._compute_func = _compute_func_map[compute_func_name]
        
        # Subset the full domain into sub-networks (if applicable).
        self._subset_domain()
        
        # Perform routing
        self._route()
    
    def subset_by_reach(self, reach_list):
        # The X_sub lines use SEGS...
        # which becomes invalid with the wbodies included.
        # So we define "common_segs" to identify regular routing segments
        # and wbodies_segs for the waterbody reaches/segments
        segs = list(chain.from_iterable(reach_list))
        segs, offnetwork_upstreams = self._add_offnetwork_upstreams()
        
        common_segs = self._param_df.index.intersection(segs)
        # Assumes everything else is a waterbody...
        wbodies_segs = set(segs).symmetric_difference(common_segs)

        #Declare empty dataframe
        waterbody_types_df_sub = pd.DataFrame()

        # If waterbody parameters exist
        if not self._waterbodies_df.empty:

            lake_segs = list(self._waterbodies_df.index.intersection(segs))

            waterbodies_df_sub = self._waterbodies_df.loc[
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
            if not self._waterbody_types_df.empty:
                waterbody_types_df_sub = self._waterbody_types_df.loc[
                    lake_segs,
                    [
                        "reservoir_type",
                    ],
                ]

        else:
            lake_segs = []
            waterbodies_df_sub = pd.DataFrame()

        param_df_sub = self._param_df.loc[
            common_segs,
            ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
        ].sort_index()

        self._prep_flowveldepth_interorder()
        
        reaches_list_with_type = _build_reach_type_list(reach_list, wbodies_segs)

        # qlat_sub = qlats.loc[common_segs].sort_index()
        # q0_sub = q0.loc[common_segs].sort_index()
        qlat_sub = self._qlats.loc[param_df_sub.index]
        q0_sub = self._q0.loc[param_df_sub.index]

        param_df_sub = param_df_sub.reindex(
            param_df_sub.index.tolist() + lake_segs
        ).sort_index()

        usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(self._usgs_df, self._lastobs_df, param_df_sub.index)
        da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(reach_list, lastobs_df_sub.index)

        qlat_sub = qlat_sub.reindex(param_df_sub.index)
        q0_sub = q0_sub.reindex(param_df_sub.index)
        
        # prepare reservoir DA data
        (
            reservoir_usgs_df_sub, 
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
            reservoir_rfc_df_sub, 
            reservoir_rfc_totalCounts, 
            reservoir_rfc_file, 
            reservoir_rfc_use_forecast, 
            reservoir_rfc_timeseries_idx, 
            reservoir_rfc_update_time, 
            reservoir_rfc_da_timestep, 
            reservoir_rfc_persist_days,
            waterbody_types_df_sub,
            ) = _prep_reservoir_da_dataframes(
            self._reservoir_usgs_df,
            self._reservoir_usgs_param_df,
            self._reservoir_usace_df, 
            self._reservoir_usace_param_df,
            self._reservoir_rfc_df,
            self._reservoir_rfc_param_df,
            waterbody_types_df_sub, 
            self._t0,
            self._from_files,
            )
        
        return (
            self._nts,
            self._dt,
            self._qts_subdivisions,
            reaches_list_with_type,
            self._independent_networks[tw],
            param_df_sub.index.values.astype("int64"),
            param_df_sub.columns.values,
            param_df_sub.values,
            q0_sub.values.astype("float32"),
            qlat_sub.values.astype("float32"),
            lake_segs,
            waterbodies_df_sub.values,
            self._data_assimilation_parameters,
            waterbody_types_df_sub.values.astype("int32"),
            self._waterbody_type_specified,
            self._t0.strftime('%Y-%m-%d_%H:%M:%S'),
            usgs_df_sub.values.astype("float32"),
            np.array(da_positions_list_byseg, dtype="int32"),
            np.array(da_positions_list_byreach, dtype="int32"),
            np.array(da_positions_list_bygage, dtype="int32"),
            lastobs_df_sub.get("lastobs_discharge", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
            lastobs_df_sub.get("time_since_lastobs", pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32")).values.astype("float32"),
            self._da_parameter_dict.get("da_decay_coefficient", 0),
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
            {},
            self._assume_short_ts,
            self._return_courant,
            self._from_files,
            )
            
    @property
    def get_output(self,):
        return self._results
    
    @abstractmethod
    def _add_offnetwork_upstreams(self,):
        pass
    
    @abstractmethod
    def _subset_domain(self,):
        pass
    
    @abstractmethod
    def _route(self,):
        pass


# -----------------------------------------------------------------------------
# Compute class definitions:
#   1. serial
#   2. by_network
#   3. by_subnetwork_jit
#   4. by_subnetwork_jit_clustered: 
# -----------------------------------------------------------------------------
class serial(AbstractCompute):
    """
    Serial compute class.
    """
    
    def _subset_domain(self,):
        pass
        
    def _route(self,):
        # Initalize an empty results list
        self._results = []
        
        # Cycle through each reach and route
        for twi, (tw, reach_list) in enumerate(self._reaches_bytw.items(), 1):
            routing_args = self._subset_by_reach(reach_list)
            self._results.append(self._compute_func(routing_args))
    
    
class by_network(serial):
    """
    By Network compute class.
    
    #NOTE I think this can be a subclass of serial. It
    # just needs to route networks in parallel. -shorvath.
    """
    
    def _subset_domain(self,):
        #TODO Define subsetting method for by-network
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        #TODO Define routing compute method for by-network
        self._results = []
    
    
class by_subnetwork_jit(by_network):
    """
    By Network JIT compute class.
    
    #NOTE I think this can be a subclass of by_network. It
    # just needs a couple extra steps to handle 'order',
    # e.g., 'flowveldepth_interorder'. -shorvath.
    """

    def _subset_domain(self,):
        #TODO Define subsetting method for by-network-jit
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        #TODO Define routing compute method for by-network-jit
        self._results = []
    
    
class by_subnetwork_jit_clustered(by_subnetwork_jit):
    """
    By Network JIT Clustered compute class.
    
    #NOTE I think this can be a subclass of by_subnetwork_jit. It
    # just needs a couple extra steps to cluster subnetworks. -shorvath.
    """
    
    def _subset_domain(self,):
        #TODO Define subsetting method for by-network-jit-clustered
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        #TODO Define routing compute method for by-network-jit-clustered
        self._results = []




#############################################################################
# Helper Functions
#############################################################################
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
                                  reservoir_rfc_df,
                                  reservoir_rfc_param_df,
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

    return (
        reservoir_usgs_df_sub, reservoir_usgs_df_time, reservoir_usgs_update_time, reservoir_usgs_prev_persisted_flow, reservoir_usgs_persistence_update_time, reservoir_usgs_persistence_index,
        reservoir_usace_df_sub, reservoir_usace_df_time, reservoir_usace_update_time, reservoir_usace_prev_persisted_flow, reservoir_usace_persistence_update_time, reservoir_usace_persistence_index,
        reservoir_rfc_df_sub, reservoir_rfc_totalCounts, reservoir_rfc_file, reservoir_rfc_use_forecast, reservoir_rfc_timeseries_idx, reservoir_rfc_update_time, reservoir_rfc_da_timestep, reservoir_rfc_persist_days,
        waterbody_types_df_sub
        )