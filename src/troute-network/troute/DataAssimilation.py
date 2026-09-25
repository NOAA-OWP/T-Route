import math
import troute.nhd_io as nhd_io
import pandas as pd
from typing import NamedTuple
import numpy as np
import pathlib
import xarray as xr
from datetime import datetime, timedelta
from abc import ABC
from joblib import delayed, Parallel
import re
import time
import logging

LOG = logging.getLogger("TROUTE")

from troute.routing.fast_reach.reservoir_RFC_da import (
    ABSURD_DISCHARGE_CMS,
    _RFC_FILENAME,
    _validate_RFC_data,
    read_rfc_timeseries,
)

from troute.network import bmi_array2df as a2df

# set legacy run flag: option to pass data frames through BMI formalism 
# not to be used in regular BMI runs any longer, only for debugging
legacy_bmi_df = False

# -----------------------------------------------------------------------------
# Abstract DA Class:
#   Define all slots and pass function definitions to child classes
# -----------------------------------------------------------------------------
class AbstractDA(ABC):
    """
    This just defines all of the slots that are be used by child classes.
    These need to be defined in a parent class so each child class can be
    combined into a single DataAssimilation object without getting a 
    'multiple-inheritance' error.
    """
    __slots__ = ["_usgs_df", "_usbr_df", "_last_obs_df", "_da_parameter_dict",
                 "_diversion_seed_in", "_diversion_rows", "_diversion_applied",
                 "_diversion_scan_empty_before",
                 "_reservoir_usgs_df", "_reservoir_usgs_param_df", 
                 "_reservoir_usace_df", "_reservoir_usace_param_df",
                 "_reservoir_usbr_df", "_reservoir_usbr_param_df", 
                 "_reservoir_rfc_df", "_reservoir_rfc_synthetic",
                 "_reservoir_rfc_param_df", "_great_lakes_df", "_great_lakes_param_df",
                 "_dateNull", 
                 "_datesSecondsArray_usgs", "_nDates_usgs", "_stationArray_usgs", 
                 "_stationStringLengthArray_usgs", "_nStations_usgs",
                 "_usgs_Array"
                 "_datesSecondsArray_reservoir_usgs", "nDates_reservoir_usgs",
                 "_stationArray_reservoir_usgs", "_stationStringLengthArray_reservoir_usgs",
                 "_nStations_reservoir_usgs",
                 "_usgs_reservoir_Array",
                 "_datesSecondsArray_reservoir_usace", "_nDates_reservoir_usace",
                 "_stationArray_reservoir_usace", "_stationStringLengthArray_reservoir_usace",
                 "_nStations_reservoir_usace", 
                 "_usace_reservoir_Array",
                 "_rfc_da_timestep", "_rfc_totalCounts", "_rfc_synthetic_values",
                 "_rfc_discharges", "_rfc_timeseries_idx", "_rfc_use_rfc",
                 "_rfc_Datetime", "_rfc_timeSteps", "_rfc_StationId_array",
                 "_rfc_StationId_stringLengths", "_rfc_List_array",
                 "_rfc_List_stringLengths",
                 "_lastObs_gageArray", "_lastObs_gageStringLengths", "_lastObs_timeSince",
                 "_lastObs_discharge",
                 "_q0_columnArray", "_q0_columnLengthArray", "_q0_nCol", "_q0_indexArray",
                 "_q0_nIndex", "_q0_Array",
                 "_waterbodyLR_columnArray", "_waterbodyLR_columnLengthArray", 
                 "_waterbodyLR_nCol", "_waterbodyLR_indexArray", "_waterbodyLR_nIndex",
                 "_waterbodyLR_Array", "_rfc_timeseries_df"
                 ]


# -----------------------------------------------------------------------------
# Base DA class definitions:
#   1. NudgingDA: streamflow nudging from usgs gages
#   2. PersistenceDA: USGS and USACE reservoir persistence
#   3. RFCDA: RFC forecasts for reservoirs
# -----------------------------------------------------------------------------
class NudgingDA(AbstractDA):
    """
    
    """
    def __init__(self, network, from_files, value_dict, da_run=[]):
        LOG.info("NudgingDA class is Started.")
        main_start_time = time.time()
        data_assimilation_parameters = self._data_assimilation_parameters
        run_parameters = self._run_parameters
        
        # isolate user-input parameters for streamflow data assimilation
        streamflow_da_parameters = data_assimilation_parameters.get('streamflow_da') or {}

        da_parameter_dict = {"da_decay_coefficient": data_assimilation_parameters.get("da_decay_coefficient", 120),
                             "diffusive_streamflow_nudging": False}
        
        # determine if user explictly requests streamflow DA
        nudging = False
        if streamflow_da_parameters:
            nudging = streamflow_da_parameters.get('streamflow_nudging', False)
            
            da_parameter_dict["diffusive_streamflow_nudging"] = streamflow_da_parameters.get("diffusive_streamflow_nudging", False)
        
        self._da_parameter_dict = da_parameter_dict

        self._last_obs_df = pd.DataFrame()
        self._usgs_df = pd.DataFrame()
        self._usbr_df = pd.DataFrame()
        self._canada_df = pd.DataFrame()
        self._canada_is_created = False  
        usgs_df = pd.DataFrame()

        # If streamflow nudging is turned on, create lastobs_df and usgs_df:
        if nudging:

            if not from_files:

                if (legacy_bmi_df):

                    # THIS LINE WAS REPLACED BY THE BMI TRANSPORT STUFF
                    usgs_df = value_dict['usgs_df']

                else:

                    # check if there are data transported
                    usgs_Array = value_dict['usgs_Array']
                    if len(usgs_Array) >0:

                        dateNull = value_dict['dateNull']    

                        datesSecondsArray_usgs = value_dict['datesSecondsArray_usgs']
                        nDates_usgs = value_dict['nDates_usgs']
                        stationArray_usgs = value_dict['stationArray_usgs']
                        stationStringLengthArray_usgs = value_dict['stationStringLengthArray_usgs']
                        nStations_usgs = value_dict['nStations_usgs']
                
                        # Unflatten the arrays
                        df_raw_usgs = a2df._unflatten_array(usgs_Array,\
                                    nDates_usgs, nStations_usgs)

                        # Decode time/date axis
                        timeAxisName = 'time'
                        freqString = '5T'
                        df_withDates_usgs = a2df._time_retrieve_from_arrays(\
                            df_raw_usgs, dateNull, datesSecondsArray_usgs, \
                            timeAxisName, freqString)

                        # Decode station ID axis
                        stationAxisName = 'stationId'
                        usgs_df = a2df._stations_retrieve_from_arrays(\
                                df_withDates_usgs, stationArray_usgs, \
                                stationStringLengthArray_usgs, stationAxisName)
                
                #usgs_df = usgs_df.join(network.link_gage_df.reset_index().set_index('gages'),how='inner').set_index('link').sort_index()
                self._usgs_df = network.link_gage_df.reset_index().set_index('gages').join(usgs_df).set_index('link').sort_index()
                
                # Next is lastobs - can also be implemented following bmi_array2df module
                lastobs = streamflow_da_parameters.get("lastobs_file", False)
                
                self._last_obs_df = pd.DataFrame()
                if lastobs:
                    lastobs_df = value_dict['lastobs_df']
                    lastobs_df_ids = value_dict['lastobs_df_index']
                    lastobs_df = pd.DataFrame(data=lastobs_df.reshape(len(lastobs_df_ids),-1),
                                              index=lastobs_df_ids,
                                              columns=['gages','time_since_lastobs','lastobs_discharge']).set_index('gages')
                    link_gage_df = network.link_gage_df.reset_index().set_index('gages')
                    col_name = link_gage_df.columns[0]
                    link_gage_df[col_name] = link_gage_df[col_name].astype(int)
                    gages_dict = link_gage_df.to_dict().get(col_name)
                    lastobs_df = lastobs_df.rename(index=gages_dict)
                    # remove 'nan' values from index
                    temp_df = lastobs_df.reset_index()
                    temp_df = temp_df[temp_df['gages']!='nan']
                    temp_df['gages'] = temp_df['gages'].astype(int)
                    lastobs_df = temp_df.set_index('gages').dropna()
                    '''
                    link_lake_dict = pd.DataFrame.from_dict(network.link_lake_crosswalk, orient='index').reset_index()
                    link_lake_dict.columns = ['lake_id', 'link']
                    link_lake_dict = link_lake_dict.set_index('link').to_dict().get('lake_id')
                    '''
                    
                    self._last_obs_df = _reindex_link_to_lake_id(lastobs_df, network.link_lake_crosswalk)
            
            else:
                #TODO: Is it a sustainable to figure out if using NHD or HYfeature based on lastobs_crosswalk_file?
                lastobs_crosswalk_file = streamflow_da_parameters.get("gage_segID_crosswalk_file", None)
                if lastobs_crosswalk_file:
                    # lastobs Dataframe for NHD Hydrofabric
                    lastobs_file = streamflow_da_parameters.get("lastobs_file", None)
                    lastobs_crosswalk_file = streamflow_da_parameters.get("gage_segID_crosswalk_file", None)
                    lastobs_start = streamflow_da_parameters.get("wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time", 0)
                    
                    if lastobs_file:
                        self._last_obs_df = build_lastobs_df(
                            lastobs_file,
                            lastobs_crosswalk_file,
                            lastobs_start,
                        )
                else:
                    # lastobs Dataframe for HYfeature HYdrofabric
                    lastobs_file = (data_assimilation_parameters.get('streamflow_da') or {}).get('lastobs_file', False)              
                    if lastobs_file:
                        lastobs_df = _read_lastobs_file(lastobs_file)
                        lastobs_df = lastobs_df.set_index('gages')
                        link_gage_df = network.link_gage_df.reset_index().set_index('gages')
                        col_name = link_gage_df.columns[0]
                        link_gage_df[col_name] = link_gage_df[col_name].astype(int)
                        gages_dict = link_gage_df.to_dict().get(col_name)
                        lastobs_df = lastobs_df.rename(index=gages_dict)
                        # remove 'nan' values from index
                        temp_df = lastobs_df.reset_index()
                        temp_df = temp_df[temp_df['gages']!='nan']
                        temp_df['gages'] = temp_df['gages'].astype(int)
                        lastobs_df = temp_df.set_index('gages').dropna()
                        self._last_obs_df = lastobs_df
                   
                # replace link ids with lake ids, for gages at waterbody outlets, 
                # otherwise, gage data will not be assimilated at waterbody outlet
                # segments because connections dic has replaced all link ids within
                # waterbodies with related lake ids.
                if network.link_lake_crosswalk:
                    self._last_obs_df = _reindex_link_to_lake_id(self._last_obs_df, network.link_lake_crosswalk)
                
                self._usgs_df = _create_usgs_df(data_assimilation_parameters, streamflow_da_parameters, run_parameters, network, da_run)
                if ('canada_timeslice_files' in da_run) and (not network.canadian_gage_df.empty):
                    self._canada_df = _create_canada_df(data_assimilation_parameters, streamflow_da_parameters, run_parameters, network, da_run)
                    self._canada_is_created = True

        # Hold the diversion gage's last observation where the reader left none.
        # Observations always win.
        diversion_da_parameters = data_assimilation_parameters.get('diversion_da', {}) or {}
        self._diversion_seed_in = {}
        self._diversion_rows = {}
        self._diversion_applied = {}
        self._diversion_scan_empty_before = {}
        if diversion_da_parameters.get('diversion_gage_crosswalk'):
            self._usgs_df = _read_diversion_observations(
                self._usgs_df, diversion_da_parameters, data_assimilation_parameters,
                network, run_parameters, da_run,
            )
            self._usgs_df, self._diversion_rows = _fill_diversion_row(
                self._usgs_df,
                diversion_da_parameters,
                network,
                run_parameters,
                seed_in=self._diversion_seed_in,
                nts=(da_run or {}).get('nts'),
            )
        LOG.debug("NudgingDA class is completed in %s seconds." % (time.time() - main_start_time))
        
    def update_after_compute(self, run_results, time_increment):
        '''
        Function to update data assimilation object after running routing module.
        
        Arguments:
        ----------
        - run_results                  (list): output from the compute kernel sequence,
                                               organized (because that is how it comes 
                                               out of the kernel) by network.
                                               For each item in the result, there are 
                                               seven elements, the fifth (usgs) and sixth 
                                               (usace) of which are lists of five elements 
                                               containing: 1) a list of the segments ids 
                                               where data assimilation was performed (if any) 
                                               in that network; 2) a list of the lupdate time; 
                                               3) a list of the previously persisted outflow; 
                                               4) a list of the persistence index; 5) a list 
                                               of the persistence update time.
        
        Returns:
        --------
        - data_assimilation               (Object): Object containing all data assimilation information
            - lastobs_df               (DataFrame): Last gage observations data for DA
        '''
        streamflow_da_parameters = self._data_assimilation_parameters.get('streamflow_da') or {}

        if streamflow_da_parameters:
            # Scaling drives the same override, so the kernel records the same
            # tuple. Harvesting it carries stale-obs decay across a window boundary.
            if (
                streamflow_da_parameters.get('streamflow_nudging', False)
                or streamflow_da_parameters.get('streamflow_scaling', False)
            ):
                self._last_obs_df = new_lastobs(run_results, time_increment)
        if (self._data_assimilation_parameters.get('diversion_da') or {}).get('diversion_gage_crosswalk'):
            # The amount each donor gave up at the last step, restored into qdp at
            # the next window's first step: exact through a clamp or a missing
            # boundary observation.
            self._diversion_applied = new_diversion_applied(run_results)

    @property
    def diversion_applied(self) -> dict:
        """Donor segment id -> the subtraction applied at the last routed step."""
        return getattr(self, "_diversion_applied", {}) or {}

    def update_for_next_loop(self, network, da_run,):
        '''
        Function to update data assimilation object for the next loop iteration. This is assumed
        to not be needed when t-route is run through the BMI.
        
        Arguments:
        ----------
        - network                    (Object): network object created from abstract class
        - da_run                       (list): list of data assimilation files separated
                                               by for loop chunks
        
        Returns:
        --------
        - data_assimilation               (Object): Object containing all data assimilation information
            - usgs_df                  (DataFrame): dataframe of USGS gage observations
        '''
        data_assimilation_parameters = self._data_assimilation_parameters
        run_parameters = self._run_parameters

        # update usgs_df if it is not empty
        # A section the configuration omits arrives as None from the schema.
        streamflow_da_parameters = data_assimilation_parameters.get('streamflow_da') or {}
        
        if streamflow_da_parameters.get('streamflow_nudging', False):
            self._usgs_df = _create_usgs_df(data_assimilation_parameters, streamflow_da_parameters, run_parameters, network, da_run)
            if ('canada_timeslice_files' in da_run) and (not network.canadian_gage_df.empty):
                self._canada_df = _create_canada_df(data_assimilation_parameters, streamflow_da_parameters, run_parameters, network, da_run)
            else:
                self._canada_df = pd.DataFrame()

        # Re-apply the diversion fill for the next loop iteration
        diversion_da_parameters = data_assimilation_parameters.get('diversion_da', {}) or {}
        if diversion_da_parameters.get('diversion_gage_crosswalk'):
            # The seed for this window is the latest observation at or before its start:
            # the reader's frame is rebuilt per window, so the hold needs memory, and a
            # report the last frame carried past this start is this window's to read.
            self._diversion_seed_in = self.diversion_seed_at(network.t0)
            if not (streamflow_da_parameters or {}).get('streamflow_nudging', False):
                # With nudging off nothing above rebuilds this frame, and the kernel
                # indexes it from zero every loop, so it is cleared and rebuilt at the
                # advanced t0; the seed carries the last observation across.
                self._usgs_df = pd.DataFrame()
            self._usgs_df = _read_diversion_observations(
                self._usgs_df, diversion_da_parameters, data_assimilation_parameters,
                network, run_parameters, da_run,
            )
            self._diversion_seed_in = _seed_from_record(
                self._usgs_df, diversion_da_parameters, data_assimilation_parameters,
                network, self._diversion_seed_in, self._diversion_scan_empty_before,
            )
            self._usgs_df, self._diversion_rows = _fill_diversion_row(
                self._usgs_df,
                diversion_da_parameters,
                network,
                run_parameters,
                seed_in=self._diversion_seed_in,
                nts=(da_run or {}).get('nts'),
            )

    def seed_from_record(self, network):
        """Seed a gage with no carried seed and no report at or before t0 from the record.

        Called by the drivers once any checkpoint is restored, so a checkpointed cycle
        never scans; a scan that found nothing is remembered, so later windows do not
        repeat it.
        """
        params = self._data_assimilation_parameters.get('diversion_da', {}) or {}
        if not params.get('diversion_gage_crosswalk'):
            return
        seeded = _seed_from_record(
            self._usgs_df, params, self._data_assimilation_parameters, network,
            self._diversion_seed_in, self._diversion_scan_empty_before,
        )
        if seeded != self._diversion_seed_in:
            self._diversion_seed_in = seeded
            self.refill_diversion_rows()

    def diversion_seed_at(self, tau=None):
        """The last diversion observation known at ``tau`` (all columns when None).

        The latest, by timestamp, of the seed this frame inherited and the finite
        pre-fill cells of its rows. Time decides, so a lookback report cannot displace
        a newer inherited seed and a report after ``tau`` is invisible to a checkpoint
        taken at ``tau``.
        """
        return _diversion_seed_at(self._diversion_rows, self._diversion_seed_in, tau)

    def refill_diversion_rows(self):
        """Re-run the fill from the pre-fill rows and the current seed, in place.

        A BMI restore lands after the frame was filled at init, so restoring the seed
        alone would change nothing.
        """
        params = self._data_assimilation_parameters.get('diversion_da', {}) or {}
        horizon = _diversion_horizon(params)
        for link_id, row in self._diversion_rows.items():
            filled, _ = _hold_and_fill(
                row.raw, self._diversion_seed_in.get(link_id), horizon=horizon,
            )
            # On the frame's own columns: the raw row keeps the reader's lookback,
            # which the frame no longer has once trimmed to t0, and writing it back
            # would append those columns out of order.
            self._usgs_df.loc[link_id, :] = filled.reindex(self._usgs_df.columns).to_numpy()


class PersistenceDA(AbstractDA):
    """
    
    """
    def __init__(self, network, from_files, value_dict, da_run=[]):
        LOG.info("PersistenceDA class is started.")
        PersistenceDA_start_time = time.time()
        data_assimilation_parameters = self._data_assimilation_parameters
        run_parameters = self._run_parameters

        # isolate user-input parameters for reservoir data assimilation
        reservoir_da_parameters = (data_assimilation_parameters.get('reservoir_da') or {}).get('reservoir_persistence_da', None)
        streamflow_da_parameters = data_assimilation_parameters.get('streamflow_da') or {}

        # check if user explictly requests USGS and/or USACE reservoir DA
        usgs_persistence  = False
        usace_persistence = False
        usbr_persistence = False
        if reservoir_da_parameters:
            usgs_persistence  = reservoir_da_parameters.get('reservoir_persistence_usgs', False)
            usace_persistence = reservoir_da_parameters.get('reservoir_persistence_usace', False)
            usbr_persistence = reservoir_da_parameters.get('reservoir_persistence_usbr', False)

        #--------------------------------------------------------------------------------
        # Assemble Reservoir dataframes
        #--------------------------------------------------------------------------------
        reservoir_usgs_df = pd.DataFrame()
        reservoir_usgs_param_df = pd.DataFrame()
        reservoir_usace_df = pd.DataFrame()
        reservoir_usace_param_df = pd.DataFrame()
        reservoir_usbr_df = pd.DataFrame()
        reservoir_usbr_param_df = pd.DataFrame()

        if not from_files:

            if usgs_persistence:

                if (legacy_bmi_df):

                    # THIS LINE WAS REPLACED BY THE BMI TRANSPORT STUFF
                    reservoir_usgs_df = value_dict['reservoir_usgs_df']

                else:

                    usgs_reservoir_Array = value_dict['usgs_reservoir_Array']
                    if len(usgs_reservoir_Array) >0:

                        dateNull = value_dict['dateNull']    

                        datesSecondsArray_reservoir_usgs = value_dict['datesSecondsArray_reservoir_usgs']
                        nDates_reservoir_usgs = value_dict['nDates_reservoir_usgs']
                        stationArray_reservoir_usgs = value_dict['stationArray_reservoir_usgs']
                        stationStringLengthArray_reservoir_usgs = value_dict['stationStringLengthArray_reservoir_usgs']
                        nStations_reservoir_usgs = value_dict['nStations_reservoir_usgs']
 
                        # Unflatten the arrays
                        df_raw_reservoirUsgs = a2df._unflatten_array(\
                                        usgs_reservoir_Array,\
                                        nDates_reservoir_usgs,\
                                        nStations_reservoir_usgs)

                        # Decode time/date axis
                        timeAxisName = 'time'
                        freqString = '15T'
                        df_withDates_reservoirUsgs = a2df._time_retrieve_from_arrays(\
                                df_raw_reservoirUsgs, dateNull, \
                                datesSecondsArray_reservoir_usgs, \
                                timeAxisName, freqString)

                        # Decode station ID axis
                        stationAxisName = 'stationId'
                        reservoir_usgs_df = a2df._stations_retrieve_from_arrays\
                                (df_withDates_reservoirUsgs, stationArray_reservoir_usgs, \
                                stationStringLengthArray_reservoir_usgs, stationAxisName)

                reservoir_usgs_df = (
                    network.usgs_lake_gage_crosswalk.
                    reset_index().
                    set_index('usgs_gage_id').
                    join(reservoir_usgs_df).
                    set_index('usgs_lake_id')
                    )

                self._usgs_df = _reindex_link_to_lake_id(self._usgs_df, network.link_lake_crosswalk)
                
                # create reservoir persistence DA initial parameters dataframe    
                if not reservoir_usgs_df.empty:
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

            if usace_persistence: 

                if (legacy_bmi_df):

                    # THIS LINE WAS REPLACED BY THE BMI TRANSPORT STUFF
                    reservoir_usace_df = value_dict['reservoir_usace_df']

                else:
                
                    # first check if there are data transported
                    usace_reservoir_Array = value_dict['usace_reservoir_Array']
                    if len(usace_reservoir_Array) >0:

                        dateNull = value_dict['dateNull']    

                        datesSecondsArray_reservoir_usace = value_dict['datesSecondsArray_reservoir_usace']
                        nDates_reservoir_usace = value_dict['nDates_reservoir_usace']
                        stationArray_reservoir_usace = value_dict['stationArray_reservoir_usace']
                        stationStringLengthArray_reservoir_usace = value_dict['stationStringLengthArray_reservoir_usace']
                        nStations_reservoir_usace = value_dict['nStations_reservoir_usace']
                
                        # Unflatten the arrays
                        df_raw_reservoirUsace = a2df._unflatten_array(\
                                            usace_reservoir_Array,\
                                            nDates_reservoir_usace,\
                                            nStations_reservoir_usace)

                        # Decode time/date axis
                        timeAxisName = 'time'
                        freqString = '15T'
                        df_withDates_reservoirUsace = a2df._time_retrieve_from_arrays\
                                (df_raw_reservoirUsace, dateNull, \
                                datesSecondsArray_reservoir_usace, 
                                timeAxisName, freqString)

                        # Decode station ID axis
                        stationAxisName = 'stationId'
                        reservoir_usace_df = a2df._stations_retrieve_from_arrays\
                                (df_withDates_reservoirUsace,\
                                stationArray_reservoir_usace, \
                                stationStringLengthArray_reservoir_usace, \
                                stationAxisName)                

                reservoir_usace_df = (
                    network.usace_lake_gage_crosswalk.
                    reset_index().
                    set_index('usace_gage_id').
                    join(reservoir_usace_df).
                    set_index('usace_lake_id')
                    )

                # create reservoir hybrid DA initial parameters dataframe    
                if not reservoir_usace_df.empty:
                    reservoir_usace_param_df = pd.DataFrame(
                        data = 0, 
                        index = reservoir_usace_df.index,
                        columns = ['update_time']
                    )
                    reservoir_usace_param_df['prev_persisted_outflow'] = np.nan
                    reservoir_usace_param_df['persistence_update_time'] = 0
                    reservoir_usace_param_df['persistence_index'] = 0
                else:
                    reservoir_usace_param_df = pd.DataFrame()
            
            if usbr_persistence:

                # USBR observations, gage-indexed, before the lake crosswalk is applied.
                # Defined on every branch so the join below cannot raise NameError.
                usbr_station_df = pd.DataFrame()

                if (legacy_bmi_df):

                    # THIS LINE WAS REPLACED BY THE BMI TRANSPORT STUFF
                    usbr_station_df = value_dict['reservoir_usbr_df']

                else:

                    usbr_reservoir_Array = value_dict['usbr_reservoir_Array']
                    if len(usbr_reservoir_Array) >0:

                        dateNull = value_dict['dateNull']    

                        datesSecondsArray_reservoir_usbr = value_dict['datesSecondsArray_reservoir_usbr']
                        nDates_reservoir_usbr = value_dict['nDates_reservoir_usbr']
                        stationArray_reservoir_usbr = value_dict['stationArray_reservoir_usbr']
                        stationStringLengthArray_reservoir_usbr = value_dict['stationStringLengthArray_reservoir_usbr']
                        nStations_reservoir_usbr = value_dict['nStations_reservoir_usbr']
 
                        # Unflatten the arrays
                        df_raw_reservoirUsbr = a2df._unflatten_array(\
                                        usbr_reservoir_Array,\
                                        nDates_reservoir_usbr,\
                                        nStations_reservoir_usbr)

                        # Decode time/date axis
                        timeAxisName = 'time'
                        freqString = '15T'
                        df_withDates_reservoirUsbr = a2df._time_retrieve_from_arrays(\
                                df_raw_reservoirUsbr, dateNull, \
                                datesSecondsArray_reservoir_usbr, \
                                timeAxisName, freqString)

                        # Decode station ID axis
                        stationAxisName = 'stationId'
                        usbr_station_df = a2df._stations_retrieve_from_arrays\
                                (df_withDates_reservoirUsbr, stationArray_reservoir_usbr, \
                                stationStringLengthArray_reservoir_usbr, stationAxisName)

                # Join the USBR observations decoded just above, NOT reservoir_usgs_df.
                # Joining the USGS frame here assimilated USGS discharge into USBR
                # reservoirs (or produced no USBR observations at all when the two gage
                # sets did not overlap).
                reservoir_usbr_df = (
                    network.usbr_lake_gage_crosswalk.
                    reset_index().
                    set_index('usbr_gage_id').
                    join(usbr_station_df).
                    set_index('usbr_lake_id')
                    )

                self._usbr_df = _reindex_link_to_lake_id(self._usbr_df, network.link_lake_crosswalk)
                
                # create reservoir persistence DA initial parameters dataframe    
                if not reservoir_usbr_df.empty:
                    reservoir_usbr_param_df = pd.DataFrame(
                    data = 0, 
                    index = reservoir_usbr_df.index ,
                        columns = ['update_time']
                    )
                    reservoir_usbr_param_df['prev_persisted_outflow'] = np.nan
                    reservoir_usbr_param_df['persistence_update_time'] = 0
                    reservoir_usbr_param_df['persistence_index'] = 0
                else:
                    reservoir_usbr_param_df = pd.DataFrame()
                
        else:

            if usgs_persistence:
                # if usgs_df is already created, make reservoir_usgs_df from that rather than reading in data again.
                # Gate on nudging actually being on, not merely on the frame being
                # non-empty: with nudging off the diversion's fill also
                # populates this frame, and it holds only the diversion gage's row, so
                # taking this shortcut derived reservoir observations from it and left
                # USGS reservoir persistence with nothing.
                if (streamflow_da_parameters or {}).get('streamflow_nudging', False) and not self._usgs_df.empty:
                    
                    gage_lake_df = (
                        network.usgs_lake_gage_crosswalk.
                        reset_index().
                        set_index(['usgs_gage_id']) # <- TODO use input parameter for this
                    )
                    
                    # build dataframe that crosswalks gageIDs to segmentIDs
                    gage_link_df = (
                        network.link_gage_df['gages'].
                        reset_index().
                        set_index(['gages'])
                    )
                    
                    # build dataframe that crosswalks segmentIDs to lakeIDs
                    link_lake_df = (
                        gage_lake_df.
                        join(gage_link_df, how = 'inner').
                        reset_index().set_index('link').
                        drop(['index'], axis = 1)
                    )

                    # resample `usgs_df` to 15 minute intervals
                    usgs_df_15min = (
                        self._usgs_df.
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
                    
                    # replace link ids with lake ids, for gages at waterbody outlets, 
                    # otherwise, gage data will not be assimilated at waterbody outlet
                    # segments.
                    if network.link_lake_crosswalk:
                        self._usgs_df = _reindex_link_to_lake_id(self._usgs_df, network.link_lake_crosswalk)
            
                    # create reservoir hybrid DA initial parameters dataframe    
                    if not reservoir_usgs_df.empty:
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
                    
                else:
                    (
                        reservoir_usgs_df,
                        reservoir_usgs_param_df
                    ) = _create_reservoir_df(
                        data_assimilation_parameters,
                        reservoir_da_parameters,
                        streamflow_da_parameters,
                        run_parameters,
                        network,
                        da_run,
                        lake_gage_crosswalk = network.usgs_lake_gage_crosswalk,
                        res_source = 'usgs')
            else:
                reservoir_usgs_df = pd.DataFrame()
                reservoir_usgs_param_df = pd.DataFrame()
                
            if usace_persistence:
                (
                    reservoir_usace_df,
                    reservoir_usace_param_df
                ) = _create_reservoir_df(
                    data_assimilation_parameters,
                    reservoir_da_parameters,
                    streamflow_da_parameters,
                    run_parameters,
                    network,
                    da_run,
                    lake_gage_crosswalk = network.usace_lake_gage_crosswalk,
                    res_source = 'usace')
            else:
                reservoir_usace_df = pd.DataFrame()
                reservoir_usace_param_df = pd.DataFrame()
            
            if usbr_persistence:
                # if usbr_df is already created, make reservoir_usbr_df from that rather than reading in data again
                if not self._usbr_df.empty: 
                    
                    gage_lake_df = (
                        network.usbr_lake_gage_crosswalk
                        .reset_index()
                        .set_index(['usbr_gage_id']) # <- TODO use input parameter for this
                    )
                    
                    # build dataframe that crosswalks gageIDs to segmentIDs
                    gage_link_df = (
                        network.link_gage_df['gages'].
                        reset_index().
                        set_index(['gages'])
                    )
                    
                    # build dataframe that crosswalks segmentIDs to lakeIDs
                    link_lake_df = (
                        gage_lake_df.
                        join(gage_link_df, how = 'inner').
                        reset_index().set_index('link').
                        drop(['index'], axis = 1)
                    )

                    # resample `usbr_df` to 15 minute intervals
                    usbr_df_15min = (
                        self._usbr_df.
                        transpose().
                        resample('15min').asfreq().
                        transpose()
                    )                     
                    
                    # subset and re-index `usbr_df`, using the segID <> lakeID crosswalk
                    reservoir_usbr_df = (
                        usbr_df_15min.join(link_lake_df, how = 'inner').
                        reset_index(drop=True).
                        set_index('usbr_lake_id')
                    )
                    
                    # replace link ids with lake ids, for gages at waterbody outlets, 
                    # otherwise, gage data will not be assimilated at waterbody outlet
                    # segments.
                    if network.link_lake_crosswalk:
                        self._usbr_df = _reindex_link_to_lake_id(self._usbr_df, network.link_lake_crosswalk)
            
                    # create reservoir hybrid DA initial parameters dataframe    
                    if not reservoir_usbr_df.empty:
                        reservoir_usbr_param_df = pd.DataFrame(
                            data = 0, 
                            index = reservoir_usbr_df.index ,
                            columns = ['update_time']
                        )
                        reservoir_usbr_param_df['prev_persisted_outflow'] = np.nan
                        reservoir_usbr_param_df['persistence_update_time'] = 0
                        reservoir_usbr_param_df['persistence_index'] = 0
                    else:
                        reservoir_usbr_param_df = pd.DataFrame()
                    
                else:
                    (
                        reservoir_usbr_df,
                        reservoir_usbr_param_df
                    ) = _create_reservoir_df(
                        data_assimilation_parameters,
                        reservoir_da_parameters,
                        streamflow_da_parameters,
                        run_parameters,
                        network,
                        da_run,
                        lake_gage_crosswalk = network.usbr_lake_gage_crosswalk,
                        res_source = 'usbr')
            else:
                reservoir_usbr_df = pd.DataFrame()
                reservoir_usbr_param_df = pd.DataFrame()
        
        self._reservoir_usgs_df = reservoir_usgs_df
        self._reservoir_usgs_param_df = reservoir_usgs_param_df
        self._reservoir_usace_df = reservoir_usace_df
        self._reservoir_usace_param_df = reservoir_usace_param_df
        self._reservoir_usbr_df = reservoir_usbr_df
        self._reservoir_usbr_param_df = reservoir_usbr_param_df

        # Trim the time-extent of the streamflow_da usgs_df
        # what happens if there are timeslice files missing on the front-end? 
        # if the first column is some timestamp greater than t0, then this will throw
        # an error. Need to think through this more. 
        if not self._usgs_df.empty:
            self._usgs_df = self._usgs_df.loc[:,network.t0:]
        LOG.debug("PersistenceDA class is completed in %s seconds." % (time.time() - PersistenceDA_start_time))
    
    def update_after_compute(self, run_results,):
        '''
        Function to update data assimilation object after running routing module.
        
        Arguments:
        ----------
        - run_results (list): output from the compute kernel sequence,
                              organized (because that is how it comes 
                              out of the kernel) by network.
                              For each item in the result, there are 
                              seven elements, the fifth (usgs) and sixth 
                              (usace) of which are lists of five elements 
                              containing: 1) a list of the segments ids 
                              where data assimilation was performed (if any) 
                              in that network; 2) a list of the lupdate time; 
                              3) a list of the previously persisted outflow; 
                              4) a list of the persistence index; 5) a list 
                              of the persistence update time.
        
        Returns:
        --------
        - data_assimilation               (Object): Object containing all data assimilation information
            - reservoir_usgs_param_df  (DataFrame): USGS reservoir DA parameters
            - reservoir_usace_param_df (DataFrame): USACE reservoir DA parameters
            - reservoir_usbr_param_df (DataFrame): USBR reservoir DA parameters
        '''
        # get reservoir DA initial parameters for next loop itteration
        self._reservoir_usgs_param_df, self._reservoir_usace_param_df,  self._reservoir_usbr_param_df = _set_persistence_reservoir_da_params(run_results)

    def update_for_next_loop(self, network, da_run,):
        '''
        Function to update data assimilation object for the next loop iteration.
        
        Arguments:
        ----------
        - network                    (Object): network object created from abstract class
        - da_run                       (list): list of data assimilation files separated
                                               by for loop chunks
        
        Returns:
        --------
        - data_assimilation               (Object): Object containing all data assimilation information
            - reservoir_usgs_df        (DataFrame): USGS reservoir observations
            - reservoir_usace_df       (DataFrame): USACE reservoir observations
        '''
        LOG.debug(' Update for next loop started')
        data_assimilation_parameters = self._data_assimilation_parameters
        run_parameters = self._run_parameters

        # update usgs_df if it is not empty
        streamflow_da_parameters = data_assimilation_parameters.get('streamflow_da') or {}
        reservoir_da_parameters = data_assimilation_parameters.get('reservoir_da') or {}
        reservoir_persistence = reservoir_da_parameters.get('reservoir_persistence_da') or {}
        
        # Same gate as in __init__: the diversion's fill can make this
        # frame non-empty while holding only the diversion gage's row.
        if (streamflow_da_parameters or {}).get('streamflow_nudging', False) and not self.usgs_df.empty:

            if reservoir_persistence.get('reservoir_persistence_usgs', False):
                
                gage_lake_df = (
                    network.usgs_lake_gage_crosswalk.
                    reset_index().
                    set_index(['usgs_gage_id']) # <- TODO use input parameter for this
                )
                
                # build dataframe that crosswalks gageIDs to segmentIDs
                gage_link_df = (
                    network.link_gage_df['gages'].
                    reset_index().
                    set_index(['gages'])
                )
                
                # build dataframe that crosswalks segmentIDs to lakeIDs
                link_lake_df = (
                    gage_lake_df.
                    join(gage_link_df, how = 'inner').
                    reset_index().set_index('link').
                    drop(['index'], axis = 1)
                )
                
                # resample `usgs_df` to 15 minute intervals
                usgs_df_15min = (
                    self._usgs_df.
                    transpose().
                    resample('15min').asfreq().
                    transpose()
                )
                
                # subset and re-index `usgs_df`, using the segID <> lakeID crosswalk
                self._reservoir_usgs_df = (
                        usgs_df_15min.join(link_lake_df, how = 'inner').
                        reset_index(drop=True).
                        set_index('usgs_lake_id')
                    )
                
                # replace link ids with lake ids, for gages at waterbody outlets, 
                # otherwise, gage data will not be assimilated at waterbody outlet
                # segments.
                if network.link_lake_crosswalk:
                    self._usgs_df = _reindex_link_to_lake_id(self.usgs_df, network.link_lake_crosswalk)
        
        # reservoir_persistence_usgs lives under reservoir_persistence_da, the same
        # nesting the USACE branch below reads. Looked up one level too high this
        # branch never fired, so whenever the frame above was not refreshed the
        # type-2 reservoir observations stayed frozen on the first window's values
        # while the kernel kept recomputing offsets against the current t0.
        elif reservoir_persistence.get('reservoir_persistence_usgs', False):
            (
                self._reservoir_usgs_df,
                _,
            ) = _create_reservoir_df(
                data_assimilation_parameters,
                reservoir_da_parameters,
                streamflow_da_parameters,
                run_parameters,
                network,
                da_run,
                lake_gage_crosswalk = network.usgs_lake_gage_crosswalk,
                res_source = 'usgs')
            
            # replace link ids with lake ids, for gages at waterbody outlets, 
            # otherwise, gage data will not be assimilated at waterbody outlet
            # segments.
            if network.link_lake_crosswalk:
                self._usgs_df = _reindex_link_to_lake_id(self.usgs_df, network.link_lake_crosswalk)
        
        # USACE
        if reservoir_persistence.get('reservoir_persistence_usace', False):
            
            (
                self._reservoir_usace_df,
                _,
            ) = _create_reservoir_df(
                data_assimilation_parameters,
                reservoir_da_parameters,
                streamflow_da_parameters,
                run_parameters,
                network,
                da_run,
                lake_gage_crosswalk = network.usace_lake_gage_crosswalk,
                res_source = 'usace')

        # USBR. Without this, type-7 reservoirs kept the first window's observations
        # for the whole run: the next loop's usbr timeslices were never read.
        if reservoir_persistence.get('reservoir_persistence_usbr', False):

            (
                self._reservoir_usbr_df,
                _,
            ) = _create_reservoir_df(
                data_assimilation_parameters,
                reservoir_da_parameters,
                streamflow_da_parameters,
                run_parameters,
                network,
                da_run,
                lake_gage_crosswalk = network.usbr_lake_gage_crosswalk,
                res_source = 'usbr')

        # if there are no TimeSlice files available for hybrid reservoir DA in the next loop,
        # but there are DA parameters from the previous loop, then create a
        # dummy observations df. This allows the reservoir persistence to continue across loops.
        # USGS Reservoirs
        if not network.waterbody_types_dataframe.empty:
            if 2 in network.waterbody_types_dataframe['reservoir_type'].unique():
                if self.reservoir_usgs_df.empty and len(self.reservoir_usgs_param_df.index) > 0:
                    self._reservoir_usgs_df = pd.DataFrame(
                        data    = np.nan, 
                        index   = self.reservoir_usgs_param_df.index, 
                        columns = [network.t0]
                    )

            # USACE Reservoirs
            if 3 in network.waterbody_types_dataframe['reservoir_type'].unique():
                if self.reservoir_usace_df.empty and len(self.reservoir_usace_param_df.index) > 0:
                    self._reservoir_usace_df = pd.DataFrame(
                        data    = np.nan,
                        index   = self.reservoir_usace_param_df.index,
                        columns = [network.t0]
                    )

            # USBR Reservoirs. Same continuation rule as USGS and USACE above: with
            # no timeslices this loop but persistence state carried from the last
            # one, a dummy frame lets that persistence continue instead of stalling.
            if 7 in network.waterbody_types_dataframe['reservoir_type'].unique():
                if self.reservoir_usbr_df.empty and len(self.reservoir_usbr_param_df.index) > 0:
                    self._reservoir_usbr_df = pd.DataFrame(
                        data    = np.nan,
                        index   = self.reservoir_usbr_param_df.index,
                        columns = [network.t0]
                    )

        # Trim the time-extent of the streamflow_da usgs_df
        # what happens if there are timeslice files missing on the front-end? 
        # if the first column is some timestamp greater than t0, then this will throw
        # an error. Need to think through this more. 
        if not self._usgs_df.empty:
            self._usgs_df = self._usgs_df.loc[:,network.t0:]

class great_lake(AbstractDA):
    '''
    Here is a list of the waterbody IDs and the gage they should correspond to:
    4800002 -> 04127885 -> 
    4800004 -> 04159130 -> 13196034 (segment_id)
    4800006 -> 02HA013
    4800007 -> IJC file    
    '''
    def __init__(self, network, from_files, value_dict, da_run):
        LOG.info("great_lake class is started.")
        great_lake_start_time = time.time()
        greatLake = False
        data_assimilation_parameters = self._data_assimilation_parameters
        run_parameters = self._run_parameters
        reservoir_persistence_da = (data_assimilation_parameters.get('reservoir_da') or {}).get('reservoir_persistence_da', {})

        self._great_lakes_df = pd.DataFrame()
        self._great_lakes_param_df = pd.DataFrame()
        
        if reservoir_persistence_da:
            greatLake = reservoir_persistence_da.get('reservoir_persistence_greatLake', False)

        if greatLake:

            GL_crosswalk_df = pd.DataFrame(
                {
                    'link': [4800002,4800004,4800006],
                    'gages': ['04127885','04159130','02HA013']
                }
            ).set_index('link')
            
            self._great_lakes_df, self._great_lakes_param_df = _create_GL_dfs(
                GL_crosswalk_df,
                data_assimilation_parameters,
                run_parameters,
                da_run,
                network.t0,
            )
            
        LOG.debug("great_lake class is completed in %s seconds." % (time.time() - great_lake_start_time))
    
    def update_after_compute(self, run_results, time_increment):
        '''
        Function to update data assimilation object after running routing module.
        
        Arguments:
        ----------
        - run_results (list): output from the compute kernel sequence,
                              organized (because that is how it comes 
                              out of the kernel) by network.
                              For each item in the result, there are 
                              10 elements, the 9th  of which are lists of 
                              four elements containing: 
                              1) a list of the segments ids where data 
                              assimilation was performed (if any) in that network; 
                              2) a list of the previously persisted outflow;
                              3) a list of the previously assimilated observation times; 
                              4) a list of the update time.
        
        Returns:
        --------
        - data_assimilation            (Object): Object containing all data assimilation information
            - _great_lakes_param_df (DataFrame): Great Lakes reservoir DA parameters
        '''
        # get reservoir DA initial parameters for next loop iteration
        great_lakes_param_df = pd.DataFrame()
        tmp_list = []
        for r in run_results:   # Corresponds to the ordering of the outflows from line 843 troute/routing/fast_reach/mc_reach.pyx
            
            if len(r[10][0]) > 0:
                tmp_df = pd.DataFrame(data = r[10][0], columns = ['lake_id'])
                tmp_df['previous_assimilated_outflows'] = r[10][1]
                tmp_df['previous_assimilated_time'] = r[10][2]
                tmp_df['update_time'] = r[10][3]
                tmp_list.append(tmp_df)
        
        if tmp_list:
            great_lakes_param_df = pd.concat(tmp_list)
            great_lakes_param_df['previous_assimilated_time'] = great_lakes_param_df['previous_assimilated_time'] - time_increment
            great_lakes_param_df['update_time'] = great_lakes_param_df['update_time'] - time_increment
        
        self._great_lakes_param_df = great_lakes_param_df

    def update_for_next_loop(self, network, da_run,):
        '''
        Function to update data assimilation object for the next loop iteration.
        
        Arguments:
        ----------
        - network                    (Object): network object created from abstract class
        - da_run                       (list): list of data assimilation files separated
                                               by for loop chunks
        
        Returns:
        --------
        - data_assimilation               (Object): Object containing all data assimilation information
            - reservoir_usgs_df        (DataFrame): USGS reservoir observations
            - reservoir_usace_df       (DataFrame): USACE reservoir observations
        '''
        greatLake = False
        data_assimilation_parameters = self._data_assimilation_parameters
        run_parameters = self._run_parameters
        reservoir_persistence_da = (data_assimilation_parameters.get('reservoir_da') or {}).get('reservoir_persistence_da', {})

        if reservoir_persistence_da:
            greatLake = reservoir_persistence_da.get('reservoir_persistence_greatLake', False)
        
        if greatLake:

            GL_crosswalk_df = pd.DataFrame(
                {
                    'link': [4800002,4800004,4800006],
                    'gages': ['04127885','04159130','02HA013']
                }
            ).set_index('link')
            
            self._great_lakes_df, _ = _create_GL_dfs(
                GL_crosswalk_df,
                data_assimilation_parameters,
                run_parameters,
                da_run,
                network.t0,
            )
            

class RFCDA(AbstractDA):
    """
    
    """
    def __init__(self, network, from_files, value_dict):
        LOG.info("RFCDA class is started.")
        RFCDA_start_time = time.time()
        rfc_parameters = (self._data_assimilation_parameters.get('reservoir_da') or {}).get('reservoir_rfc_da', None)

        # check if user explictly requests RFC reservoir DA
        rfc  = False
        if rfc_parameters:
            rfc = rfc_parameters.get('reservoir_rfc_forecasts', False)

        self._reservoir_rfc_df = pd.DataFrame()
        self._reservoir_rfc_param_df = pd.DataFrame()
        rfc_df = pd.DataFrame()

        if not from_files:

            if rfc:

                if (legacy_bmi_df):

                    # THIS LINE WAS REPLACED BY THE BMI TRANSPORT STUFF
                    # Retrieve rfc timeseries dataframe from BMI dictionary
                    rfc_df = value_dict['rfc_timeseries_df']

                else:

                    # check if there are any data transported
                    rfc_StationId_array = value_dict['rfc_StationId_array']
                    if len(rfc_StationId_array) >0:

                        dateNull = value_dict['dateNull']    

                        # RFC timeseries dataframe converted
                        rfc_da_timestep = value_dict['rfc_da_timestep']
                        rfc_totalCounts = value_dict['rfc_totalCounts']
                        rfc_synthetic_values = value_dict['rfc_synthetic_values']
                        rfc_discharges = value_dict['rfc_discharges']
                        rfc_timeseries_idx = value_dict['rfc_timeseries_idx']
                        rfc_use_rfc = value_dict['rfc_use_rfc']
                        rfc_Datetime = value_dict['rfc_Datetime']
                        rfc_timeSteps = value_dict['rfc_timeSteps']
                        rfc_issue_time = value_dict.get('rfc_issue_time')
                
                        rfc_StationId_stringLengths = value_dict['rfc_StationId_stringLengths']
                        rfc_List_array = value_dict['rfc_List_array']
                        rfc_List_stringLengths = value_dict['rfc_List_stringLengths']

                        # Decode rfc timeseries
                        rfc_df = a2df._bmi_reassemble_rfc_timeseries (rfc_da_timestep, \
                                rfc_totalCounts, rfc_synthetic_values, \
                                rfc_discharges, rfc_timeseries_idx, \
                                rfc_use_rfc, rfc_Datetime, rfc_timeSteps, \
                                rfc_StationId_array, rfc_StationId_stringLengths, \
                                rfc_List_array, rfc_List_stringLengths, 
                                dateNull, rfc_issue_time)

                self._reservoir_rfc_df, self._reservoir_rfc_param_df = assemble_rfc_dataframes(
                                                                                            rfc_df, 
                                                                                            network.rfc_lake_gage_crosswalk,
                                                                                            network.t0, 
                                                                                            rfc_parameters,
                                                                                            )

            else: 

                self._reservoir_rfc_df = pd.DataFrame()
                self._reservoir_rfc_param_df = pd.DataFrame()
        
        else:

            if rfc:
                # In order to use only RFC python module (not fortran module), create rfc dataframes from reading files
                #rfc_parameters   = self._data_assimilation_parameters.get('reservoir_da', {}).get('reservoir_rfc_da', {})
                lookback_hrs     = rfc_parameters.get('reservoir_rfc_forecasts_lookback_hours')
                offset_hrs       = rfc_parameters.get('reservoir_rfc_forecasts_offset_hours')
                start_datetime   = network.t0 
                timeseries_end   = start_datetime + timedelta(hours=offset_hrs)
                timeseries_start = timeseries_end - timedelta(hours=lookback_hrs)
                delta            = timedelta(hours=1)
                timeseries_dates = []
                while timeseries_start <= timeseries_end:
                    timeseries_dates.append(timeseries_start.strftime('%Y-%m-%d_%H'))
                    timeseries_start += delta
                rfc_forecast_persist_days = rfc_parameters.get('reservoir_rfc_forecast_persist_days')
                final_persist_datetime = start_datetime + timedelta(days=rfc_forecast_persist_days)
                
                # RFC Observations
                rfc_timeseries_path = str(rfc_parameters.get('reservoir_rfc_forecasts_time_series_path'))
                self._rfc_timeseries_df = _read_timeseries_files(
                    rfc_timeseries_path, timeseries_dates, start_datetime, final_persist_datetime,
                    routing_period=self._run_parameters.get('dt', 300),
                    unavailable_action=rfc_parameters.get(
                        'reservoir_rfc_forecasts_unavailable_action', 'error'),
                    gages=network.rfc_lake_gage_crosswalk['rfc_gage_id'].dropna(),
                )
                self._reservoir_rfc_df, self._reservoir_rfc_param_df = assemble_rfc_dataframes(
                                                                                                self._rfc_timeseries_df, 
                                                                                                network.rfc_lake_gage_crosswalk,
                                                                                                network.t0, 
                                                                                                rfc_parameters,
                                                                                                )
            else:    

                self._reservoir_rfc_df = pd.DataFrame()
                self._reservoir_rfc_param_df = pd.DataFrame()
        LOG.debug("RFCDA class is completed in %s seconds." % (time.time() - RFCDA_start_time))
    
    def update_after_compute(self, run_results):
        '''
        Function to update data assimilation object after running routing module.
        
        Arguments:
        ----------
        - run_results (list): output from the compute kernel sequence,
                              organized (because that is how it comes 
                              out of the kernel) by network.
                              For each item in the result, there are 
                              seven elements, the fifth (usgs) and sixth 
                              (usace) of which are lists of five elements 
                              containing: 1) a list of the segments ids 
                              where data assimilation was performed (if any) 
                              in that network; 2) a list of the lupdate time; 
                              3) a list of the previously persisted outflow; 
                              4) a list of the persistence index; 5) a list 
                              of the persistence update time.
        
        Returns:
        --------
        - data_assimilation               (Object): Object containing all data assimilation information
            - reservoir_usgs_param_df  (DataFrame): USGS reservoir DA parameters
            - reservoir_usace_param_df (DataFrame): USACE reservoir DA parameters
        '''
        # get reservoir DA initial parameters for next loop itteration
        self._reservoir_rfc_param_df = _set_rfc_reservoir_da_params(self._reservoir_rfc_param_df, run_results)


    def update_for_next_loop(self,):
        pass


# --------------------------------------------------------------------
# Combination of base DA classes. This is the DA object that is called
# by t-route.
# --------------------------------------------------------------------
class DataAssimilation(NudgingDA, PersistenceDA, RFCDA):
    """
    
    """
    __slots__ = ["_data_assimilation_parameters", "_run_parameters", "_waterbody_parameters"]

    def __init__(self, network, data_assimilation_parameters, run_parameters, waterbody_parameters,
                 from_files=True, value_dict=None, da_run=[]):

        self._data_assimilation_parameters = data_assimilation_parameters
        self._run_parameters = run_parameters
        self._waterbody_parameters = waterbody_parameters

        NudgingDA.__init__(self, network, from_files, value_dict, da_run)
        PersistenceDA.__init__(self, network, from_files, value_dict, da_run)
        RFCDA.__init__(self, network, from_files, value_dict)
        great_lake.__init__(self, network, from_files, value_dict, da_run)
    
    def update_after_compute(self, run_results, time_increment):
        '''
        
        '''
        NudgingDA.update_after_compute(self, run_results, time_increment)
        PersistenceDA.update_after_compute(self, run_results)
        RFCDA.update_after_compute(self, run_results)
        great_lake.update_after_compute(self, run_results, time_increment)

    def update_for_next_loop(self, network, da_run,):
        '''

        '''
        NudgingDA.update_for_next_loop(self, network, da_run)
        PersistenceDA.update_for_next_loop(self, network, da_run)
        RFCDA.update_for_next_loop(self)
        great_lake.update_for_next_loop(self, network, da_run)
    

    @property
    def assimilation_parameters(self):
        return self._da_parameter_dict
    
    @property
    def lastobs_df(self):
        return self._last_obs_df

    @property
    def usgs_df(self):
        return self._usgs_df

    @property
    def usbr_df(self):
        return self._usbr_df
    
    @property
    def reservoir_usgs_df(self):
        return self._reservoir_usgs_df
    
    @property
    def reservoir_usgs_param_df(self):
        return self._reservoir_usgs_param_df
    
    @property
    def reservoir_usace_df(self):
        return self._reservoir_usace_df
    
    @property
    def reservoir_usace_param_df(self):
        return self._reservoir_usace_param_df
    
    @property
    def reservoir_usbr_df(self):
        return self._reservoir_usbr_df
    
    @property
    def reservoir_usbr_param_df(self):
        return self._reservoir_usbr_param_df
    
    @property
    def reservoir_rfc_df(self):
        return self._reservoir_rfc_df
    
    @property
    def reservoir_rfc_param_df(self):
        return self._reservoir_rfc_param_df
    
    @property
    def great_lakes_df(self):
        return self._great_lakes_df

    @property
    def great_lakes_param_df(self):
        return self._great_lakes_param_df


# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------


class _DiversionRow(NamedTuple):
    """One diversion gage's row before any fill, on the routing grid."""
    raw: pd.Series
    site_no: str


def _diversion_horizon(diversion_da_parameters: dict) -> pd.Timedelta:
    """How long the last observation is held; 0 days holds nothing past its instant."""
    return pd.Timedelta(days=int(diversion_da_parameters.get('diversion_persist_days', 11)))


def _read_diversion_observations(
    usgs_df: pd.DataFrame,
    diversion_da_parameters: dict,
    data_assimilation_parameters: dict,
    network,
    run_parameters: dict,
    da_run: dict | None,
) -> pd.DataFrame:
    """Read the diversion gages' rows from the TimeSlices when nothing else did.

    Streamflow nudging reads every gage's row, the diversion gage's included, so only a
    row that is still missing is read here, which is the case with nudging off. Rows
    already present win.
    """
    site_to_node = getattr(network, "_diversion_site_to_node", {})
    crosswalk = diversion_da_parameters.get("diversion_gage_crosswalk", {})
    wanted = {
        int(site_to_node[str(site)]): str(site)
        for site in crosswalk.values() if str(site) in site_to_node
    }
    present = set() if usgs_df.empty else set(usgs_df.index)
    missing = {link: site for link, site in wanted.items() if link not in present}
    folder = data_assimilation_parameters.get("usgs_timeslices_folder")
    files = (da_run or {}).get("usgs_timeslice_files") or []
    if not missing or not folder or not files:
        return usgs_df
    link_gage_df = pd.DataFrame({"gages": pd.Series(missing)})
    link_gage_df.index.name = "link"
    try:
        obs = nhd_io.get_obs_from_timeslices(
            link_gage_df,
            "gages",
            "link",
            [pathlib.Path(folder).joinpath(f) for f in files],
            data_assimilation_parameters.get("qc_threshold", 1),
            data_assimilation_parameters.get("interpolation_limit_min", 59),
            run_parameters.get("dt"),
            network.t0,
            run_parameters.get("cpu_pool", None),
        )
    except (OSError, ValueError, KeyError) as exc:
        # A file removed or still being written between discovery and this read.
        # The hold carries the last known report; the run goes on.
        LOG.warning(
            "diversion DA: this window's TimeSlices could not be read (%s); holding the "
            "last known report instead", exc,
        )
        return usgs_df
    if obs.empty:
        return usgs_df
    obs = obs.loc[obs.index.intersection(list(missing))]
    if usgs_df.empty or usgs_df.shape[1] == 0:
        return obs
    return usgs_df.combine_first(obs)


def _last_report_before(
    folder: str | pathlib.Path,
    site_no: str,
    before: pd.Timestamp,
    horizon: pd.Timedelta,
    qc_threshold: float,
) -> tuple[pd.Timestamp, float] | None:
    """The gage's newest valid report in the files named for ``[before - horizon, before)``.

    Newest file first, one file at a time, stopping at the first hit, so a gage that
    reported recently costs one read and a quiet one at most the horizon's worth. Valid
    means what the reader accepts: quality in range and at least ``qc_threshold``,
    discharge above zero, and the report's own timestamp inside the window.
    """
    stamped = []
    for path in pathlib.Path(folder).glob("*usgsTimeSlice.ncdf"):
        try:
            stamp = pd.Timestamp(path.name[:19].replace("_", " "))
        except ValueError:
            continue  # not named for an instant
        if before - horizon <= stamp < before:
            stamped.append((stamp, path))
    for _, path in sorted(stamped, reverse=True):
        try:
            obs, qual = nhd_io._read_timeslice_file(str(path))
        except (OSError, ValueError, KeyError) as exc:
            # A shared folder can hold a truncated or foreign file; the scan is a
            # fallback and must not end an hourly cycle over it.
            LOG.warning("diversion DA: skipping unreadable TimeSlice %s (%s)", path.name, exc)
            continue
        if obs.empty or site_no not in obs.index:
            continue
        row = obs.loc[site_no].astype(float)
        if not qual.empty and site_no in qual.index:
            quality = qual.loc[site_no].astype(float)
            quality = quality.mask(quality < 0).mask(quality > 1)
            row = row.mask(quality < qc_threshold)
        row = row.mask(row <= 0)
        when = pd.to_datetime(row.index.astype(str).str.replace("_", " "), errors="coerce")
        row = row[(when >= before - horizon) & (when < before)]
        finite = row.dropna()
        if finite.empty:
            continue
        newest = pd.to_datetime(str(finite.index[-1]).replace("_", " "))
        return newest, float(finite.iloc[-1])
    # Say how far back the folder reaches: the horizon is only as long as the
    # history the deployment stages.
    LOG.warning(
        "diversion gage %s: no report in the TimeSlices between %s and %s; the folder's "
        "oldest file in that span is %s", site_no, before - horizon, before,
        min(stamped)[0] if stamped else "none",
    )
    return None


def _seed_from_record(
    usgs_df: pd.DataFrame,
    diversion_da_parameters: dict,
    data_assimilation_parameters: dict,
    network,
    seed_in: dict | None,
    empty_before: dict | None = None,
) -> dict:
    """Seed a gage with no carried seed and no report in the frame from the TimeSlices.

    A fresh process (a forecast cycle without a checkpoint) reads only the reader's
    lookback, so a gage quiet for longer would have nothing to hold although the
    horizon allows it. The last report within the horizon before t0 stands in.
    """
    seed = dict(seed_in or {})
    folder = data_assimilation_parameters.get("usgs_timeslices_folder")
    horizon = _diversion_horizon(diversion_da_parameters)
    if not folder or horizon <= pd.Timedelta(0):
        return seed
    site_to_node = getattr(network, "_diversion_site_to_node", {})
    t0 = pd.Timestamp(network.t0)
    empty_before = {} if empty_before is None else empty_before
    for site_no in diversion_da_parameters.get("diversion_gage_crosswalk", {}).values():
        site_no = str(site_no)
        if site_no not in site_to_node:
            continue
        link_id = int(site_to_node[site_no])
        if link_id in seed:
            continue
        if link_id in getattr(usgs_df, "index", []):
            # Only reports at or before t0 make a seed: a report later in the window
            # is the hold's successor, not its source.
            row = usgs_df.loc[link_id]
            if isinstance(row.index, pd.DatetimeIndex):
                row = row[row.index <= t0]
            if row.notna().any():
                continue
        if link_id in empty_before and t0 >= empty_before[link_id]:
            # An earlier scan found nothing back to its horizon; anything since sits
            # in a frame this object has already seen.
            continue
        found = _last_report_before(
            folder, site_no, t0, horizon, data_assimilation_parameters.get("qc_threshold", 1)
        )
        if found is None:
            empty_before[link_id] = t0
        else:
            seed[link_id] = found
            LOG.warning(
                "diversion gage %s: no report in this window's TimeSlices; holding the last "
                "one, %.1f at %s, %.1f h before t0", site_no, found[1], found[0],
                (t0 - found[0]).total_seconds() / 3600,
            )
    return seed


def _hold_and_fill(
    raw: pd.Series,
    seed: tuple | None,
    *,
    horizon: pd.Timedelta,
) -> tuple[pd.Series, int]:
    """Fill one row: the reader's value, else the last observation held.

    The hold is causal: a cell takes the latest finite cell BEFORE it, never a later
    one; the inherited seed outranks raw cells older than it and covers its own
    instant, and a report at that same instant wins over it. A held value expires
    ``horizon`` after the observation it came from. Returns the row and the held count.
    """
    filled = raw.copy()
    n_held = 0
    if isinstance(raw.index, pd.DatetimeIndex):
        held = raw.ffill()
        # The time of the observation each column holds from.
        source_time = raw.index.to_series().where(raw.notna()).ffill()
        if seed is not None:
            t_seed, q_seed = seed
            t_seed = pd.Timestamp(t_seed)
            from_seed = (
                raw.isna()
                & (raw.index >= t_seed)
                & (source_time.isna() | (source_time < t_seed)).to_numpy()
            )
            held[from_seed] = float(q_seed)
            source_time[from_seed] = t_seed
        age = raw.index.to_series() - source_time
        expired = raw.isna() & held.notna() & (age > horizon).to_numpy()
        held[expired] = np.nan
        n_held = int(held.notna().sum() - raw.notna().sum())
        filled = held
    return filled, n_held


def _fill_diversion_row(
    usgs_df: pd.DataFrame,
    diversion_da_parameters: dict,
    network,
    run_parameters: dict,
    seed_in: dict | None = None,
    nts: int | None = None,
) -> tuple[pd.DataFrame, dict[int, _DiversionRow]]:
    """Put every diversion gage's row on the routing grid and hold its last observation.

    Returns the frame and the pre-fill rows, from which the next window's seed and a
    BMI restart's re-fill are derived. The reader's columns stop at the record's end
    and alignment would add the missing routing steps as NaN, out of the hold's reach,
    so the row is extended onto the window's grid here.
    """
    crosswalk = diversion_da_parameters.get("diversion_gage_crosswalk", {})
    if not crosswalk:
        return usgs_df, {}
    horizon = _diversion_horizon(diversion_da_parameters)
    seed_in = seed_in or {}

    # One column per routing step of the window plus column 0 for t0: the kernel indexes
    # the frame by routing step, so the spacing is dt whatever it is. nts is the window's
    # length when the caller knows it, the run's total otherwise.
    dt = run_parameters.get("dt", 300)
    steps = nts if nts is not None else run_parameters.get("nts", 0)
    grid = pd.date_range(
        network.t0, periods=max(1, int(steps or 0) + 1), freq=pd.Timedelta(seconds=dt)
    )
    if usgs_df.empty or usgs_df.shape[1] == 0:
        usgs_df = pd.DataFrame(index=pd.Index([], dtype="int64"), columns=grid, dtype=float)
    elif isinstance(usgs_df.columns, pd.DatetimeIndex):
        usgs_df = usgs_df.reindex(columns=usgs_df.columns.union(grid))

    site_to_node: dict[str, int] = getattr(network, "_diversion_site_to_node", {})
    rows: dict[int, _DiversionRow] = {}
    for fp_id, site_no in crosswalk.items():
        site_no = str(site_no)
        if site_no not in site_to_node:
            # The gage never resolved to a routing link, e.g. the network carries no
            # waterbodies and preprocessing returned early. Filling would raise
            # KeyError; skipping silently would leave the donor with no fallback.
            LOG.warning(
                "diversion DA: gage %s has no routing link; the diversion for "
                "flowpath %s will not be applied.", site_no, fp_id,
            )
            continue
        link_id = int(site_to_node[site_no])
        if link_id in usgs_df.index:
            raw = usgs_df.loc[link_id].astype(float).copy()
        else:
            raw = pd.Series(np.nan, index=usgs_df.columns, dtype=float)
        filled, _ = _hold_and_fill(raw, seed_in.get(link_id), horizon=horizon)
        rows[link_id] = _DiversionRow(raw, site_no)
        # Counted over the window's steps, not the reader's lookback columns.
        in_window = filled.index.isin(grid)
        n_total = int(in_window.sum())
        n_held = int((filled.notna() & raw.isna())[in_window].sum())
        if not filled[in_window].notna().any():
            # A run that completes with nothing to divert must say so: a misnamed
            # site, a folder without the station, or a record that has not started.
            LOG.warning(
                "diversion gage %s: no observation in this window and nothing held; "
                "no flow will be diverted", site_no,
            )
        elif n_held:
            # A held value is not an observation. Say how much of the window it
            # covers, so a short gap can be told from a gage outage being papered
            # over up to the horizon.
            LOG.warning(
                "diversion gage %s: %d of %d timesteps held at the last observation "
                "in this window (%.0f%% not observed)",
                site_no, n_held, n_total, 100.0 * n_held / max(n_total, 1),
            )
        if link_id in usgs_df.index:
            usgs_df.loc[link_id] = filled
        else:
            new_row = filled.to_frame().T
            new_row.index = pd.Index([link_id], dtype="int64")
            usgs_df = pd.concat([usgs_df, new_row])

    return usgs_df, rows


def _diversion_seed_at(
    rows: dict[int, _DiversionRow],
    seed_in: dict | None,
    tau=None,
) -> dict[int, tuple[pd.Timestamp, float]]:
    """Latest observation per gage at or before ``tau``, inherited seed included.

    A report at the inherited seed's own instant wins over it: the TimeSlice is the
    current record, the seed a checkpoint's copy of it.
    """
    out = dict(seed_in or {})
    for link_id, row in rows.items():
        raw = row.raw
        if not isinstance(raw.index, pd.DatetimeIndex):
            continue
        if tau is not None:
            raw = raw[raw.index <= pd.Timestamp(tau)]
        finite = raw.dropna()
        if finite.empty:
            continue
        candidate = (finite.index[-1], float(finite.iloc[-1]))
        previous = out.get(link_id)
        if previous is None or candidate[0] >= pd.Timestamp(previous[0]):
            out[link_id] = candidate
    return out


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

def _create_usgs_df(data_assimilation_parameters, streamflow_da_parameters, run_parameters, network, da_run):
    '''
    Function for reading USGS timeslice files and creating a dataframe
    of USGS gage observations. This dataframe is used for streamflow
    nudging and can be used for constructing USGS reservoir dataframes.
    
    Arguments:
    ----------
    - data_assimilation_parameters (dict): user input data re data assimilation
    - streamflow_da_parameters     (dict): user input data re streamflow nudging
    - run_parameters               (dict): user input data re subset of compute configuration
    - network                    (Object): network object created from abstract class
    - da_run                       (list): list of data assimilation files separated by for loop chunks
    
    Returns:
    --------
    - usgs_df (DataFrame): dataframe of USGS gage observations
    '''
    usgs_timeslices_folder = data_assimilation_parameters.get("usgs_timeslices_folder", None)
    #lastobs_file           = streamflow_da_parameters.get("wrf_hydro_lastobs_file", None)
    lastobs_start          = data_assimilation_parameters.get("wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time",0)
    lastobs_type           = data_assimilation_parameters.get("wrf_lastobs_type", "error-based")
    crosswalk_file         = streamflow_da_parameters.get("gage_segID_crosswalk_file", None)
    crosswalk_gage_field   = streamflow_da_parameters.get('crosswalk_gage_field','gages')
    crosswalk_segID_field  = streamflow_da_parameters.get('crosswalk_segID_field','link')
    da_decay_coefficient   = data_assimilation_parameters.get("da_decay_coefficient",120)
    qc_threshold           = data_assimilation_parameters.get("qc_threshold",1)
    interpolation_limit    = data_assimilation_parameters.get("interpolation_limit_min",59)
    LOG.info("Reading and preprocessing usgs timeslice files is started.")
    usgs_df_start_time = time.time()
    # TODO: join timeslice folder and files into complete path upstream
    if usgs_timeslices_folder is None:
        raise ValueError(
            "streamflow_da.streamflow_nudging is enabled but "
            "data_assimilation_parameters.usgs_timeslices_folder is not set, so there is nowhere to "
            "read gage observations from. Set the folder, or disable nudging."
        )
    usgs_timeslices_folder = pathlib.Path(usgs_timeslices_folder)
    usgs_files = [usgs_timeslices_folder.joinpath(f) for f in 
                  da_run['usgs_timeslice_files']]
	
    if usgs_files:
        usgs_df = (
            nhd_io.get_obs_from_timeslices(
                network.link_gage_df,
                crosswalk_gage_field,
                crosswalk_segID_field,
                usgs_files,
                qc_threshold,
                interpolation_limit,
                run_parameters.get("dt"),
                network.t0,
                run_parameters.get("cpu_pool", None)
            ).
            loc[network.link_gage_df.index]
        )

    else:
        usgs_df = pd.DataFrame()
    LOG.debug("Reading and preprocessing usgs timeslice files is completed in %s seconds." % (time.time() - usgs_df_start_time))
    return usgs_df

def _create_LakeOntario_df(run_parameters, t0, da_run):
    LOG.info("Creating Lake Ontario dataframe is started.")
    LakeOntario_df_start_time = time.time()    
    start_time = t0 - pd.Timedelta(weeks = 10)
    nts = run_parameters.get('nts')
    dt = run_parameters.get('dt')
    end_time = t0 + pd.Timedelta(hours = nts/(3600/dt))

    lake_ontario_df = pd.read_csv(da_run.get('LakeOntario_outflow'))
    
    # Increment the date by one day where Hour is "24:00" and set Hour to "00:00"
    mask = lake_ontario_df['Hour'] == '24:00'
    lake_ontario_df.loc[mask, 'Hour'] = '00:00'
    lake_ontario_df.loc[mask, 'Date'] = (pd.to_datetime(lake_ontario_df.loc[mask, 'Date']) + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')

    lake_ontario_df['Hour'] = lake_ontario_df['Hour'].apply(lambda x: re.sub(r':\d{2}$', ':00', x))
    # Combine Date and Hour into Datetime
    lake_ontario_df['Datetime'] = pd.to_datetime(lake_ontario_df['Date'] + ' ' + lake_ontario_df['Hour'], format='%Y-%m-%d %H:%M')
    
    # Set Datetime as index and remove duplicates and unwanted columns
    lake_ontario_df = lake_ontario_df.set_index('Datetime')
    lake_ontario_df = lake_ontario_df.drop_duplicates()
    lake_ontario_df = lake_ontario_df.drop(['Date', 'Hour'], axis=1)

    # Rename outflow column to discharge
    lake_ontario_df = lake_ontario_df.rename(columns={'Outflow(m3/s)': 'Discharge'})
    
    # Filter for needed time stamps
    lake_ontario_df = lake_ontario_df[(lake_ontario_df.index>=start_time) & (lake_ontario_df.index<=end_time)]

    # Add 'link' column with waterbody ID
    lake_ontario_df['link'] = 4800007
    
    # Reset index and convert Datetimes to strings
    lake_ontario_df.reset_index(inplace=True)
    lake_ontario_df['Datetime'] = lake_ontario_df['Datetime'].dt.strftime('%Y-%m-%d_%H:%M:%S')
    
    LOG.debug("Creating Lake Ontario dataframe is completed in %s seconds." % (time.time() - LakeOntario_df_start_time))
    
    return lake_ontario_df

def _create_canada_df(data_assimilation_parameters, streamflow_da_parameters, run_parameters, network, da_run):
    '''
    Function for reading Canadian timeslice files and creating a dataframe
    of Canadian gage observations. This dataframe is used for streamflow
    nudging and can be used for constructing Canadian reservoir dataframes.
    
    Arguments:
    ----------
    - data_assimilation_parameters (dict): user input data re data assimilation
    - streamflow_da_parameters     (dict): user input data re streamflow nudging
    - run_parameters               (dict): user input data re subset of compute configuration
    - network                    (Object): network object created from abstract class
    - da_run                       (list): list of data assimilation files separated by for loop chunks
    
    Returns:
    --------
    - canada_df (DataFrame): dataframe of Canadian gage observations
    '''
    canada_timeslices_folder = data_assimilation_parameters.get("canada_timeslices_folder", None)
    #lastobs_file           = streamflow_da_parameters.get("wrf_hydro_lastobs_file", None)
    lastobs_start          = data_assimilation_parameters.get("wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time",0)
    lastobs_type           = data_assimilation_parameters.get("wrf_lastobs_type", "error-based")
    crosswalk_file         = streamflow_da_parameters.get("gage_segID_crosswalk_file", None)
    crosswalk_gage_field   = streamflow_da_parameters.get('crosswalk_gage_field','gages')
    crosswalk_segID_field  = streamflow_da_parameters.get('crosswalk_segID_field','link')
    da_decay_coefficient   = data_assimilation_parameters.get("da_decay_coefficient",120)
    qc_threshold           = data_assimilation_parameters.get("qc_threshold",1)
    interpolation_limit    = data_assimilation_parameters.get("interpolation_limit_min",59)
    
    # TODO: join timeslice folder and files into complete path upstream
    LOG.info("Reading Canadian timeslice files is started.")
    canada_df_start_time = time.time()
    canada_files = [canada_timeslices_folder.joinpath(f) for f in da_run['canada_timeslice_files']]
    
    if canada_files:
        canada_df = (
            nhd_io.get_obs_from_timeslices(
                network.canadian_gage_df,
                crosswalk_gage_field,
                crosswalk_segID_field,
                canada_files,
                qc_threshold,
                interpolation_limit,
                run_parameters.get("dt"),
                network.t0,
                run_parameters.get("cpu_pool", None)
            ).
            loc[network.canadian_gage_df.index]
        )

    else:
        canada_df = pd.DataFrame()
    LOG.debug("Reading Canadian timeslice files is completed in %s seconds." % (time.time() - canada_df_start_time))
    return canada_df

def _create_reservoir_df(data_assimilation_parameters, reservoir_da_parameters, streamflow_da_parameters, run_parameters, network, da_run, lake_gage_crosswalk, res_source):
    '''
    Function for reading USGS/USACE/USBR timeslice files and creating a dataframe
    of reservoir observations and initial parameters. 
    These dataframes are used for reservoir DA.
    
    Arguments:
    ----------
    - data_assimilation_parameters (dict): user input data re data assimilation
    - reservoir_da_parameters      (dict): user input data re reservoir data assimilation
    - streamflow_da_parameters     (dict): user input data re streamflow nudging
    - run_parameters               (dict): user input data re subset of compute configuration
    - network                    (Object): network object created from abstract class
    - da_run                       (list): list of data assimilation files separated
                                           by for loop chunks
    - lake_gage_crosswalk          (dict): usgs/usace/usbr gage ids and corresponding segment ids at
                                           which they are located
    - res_source                    (str): either 'usgs', 'usace', or 'usbr', specifiying which type of
                                           reservoir dataframe to create (must match lake_gage_crosswalk
    
    Returns:
    --------
    - reservoir_usgs/usace/usbr_df       (DataFrame): USGS/USACE reservoir observations
    - reservoir_usgs/usace/usbr_param_df (DataFrame): USGS/USACE reservoir hybrid DA initial parameters
    '''
    res_timeslices_folder  = data_assimilation_parameters.get(res_source + "_timeslices_folder",None)
    crosswalk_file         = reservoir_da_parameters.get("gage_lakeID_crosswalk_file", None)
    crosswalk_gage_field   = streamflow_da_parameters.get('crosswalk_' + res_source + '_gage_field',res_source + '_gage_id')
    crosswalk_lakeID_field = streamflow_da_parameters.get('crosswalk_' + res_source + '_lakeID_field',res_source + '_lake_id')
    qc_threshold           = data_assimilation_parameters.get("qc_threshold",1)
    interpolation_limit    = data_assimilation_parameters.get("interpolation_limit_min",59)
    LOG.info(f"Reading {res_source} timeslice files and creating a dataframe is started.")
    reservoir_df_start_time = time.time()	
    # TODO: join timeslice folder and files into complete path upstream in workflow
    res_timeslices_folder = pathlib.Path(res_timeslices_folder)
    res_files = [res_timeslices_folder.joinpath(f) for f in
                 da_run[res_source + '_timeslice_files']]
			
    if res_files:
		
        reservoir_df = nhd_io.get_obs_from_timeslices(
            lake_gage_crosswalk,
            crosswalk_gage_field,
            crosswalk_lakeID_field,
            res_files,
            qc_threshold,
            interpolation_limit,
            900,                      # 15 minutes, as secs
            network.t0,
            run_parameters.get("cpu_pool", None)
        )
		
    else:
        reservoir_df = pd.DataFrame() 
	
    # create reservoir hybrid DA initial parameters dataframe    
    if reservoir_df.empty == False:
        reservoir_param_df = pd.DataFrame(
            data = 0, 
            index = reservoir_df.index ,
            columns = ['update_time']
        )
        reservoir_param_df['prev_persisted_outflow'] = np.nan
        reservoir_param_df['persistence_update_time'] = 0
        reservoir_param_df['persistence_index'] = 0
    else:
        reservoir_param_df = pd.DataFrame()
    LOG.debug(f"Reading {res_source} timeslice files is completed in %s seconds." % (time.time() - reservoir_df_start_time))    
    return reservoir_df, reservoir_param_df
    
def _set_persistence_reservoir_da_params(run_results):
    '''
    Update persistence reservoir DA parameters for subsequent loops
    Arguments:
    ----------
    - run_results (list): output from the compute kernel sequence, organized
        (because that is how it comes out of the kernel) by network.
        For each item in the result, there are seven elements, the
        fifth (usgs) and sixth (usace) of which are lists of five elements 
        containing: 1) a list of the segments ids where data assimilation 
        was performed (if any) in that network; 2) a list of the lupdate time; 
        3) a list of the previously persisted outflow; 4) a list of the 
        persistence index; 5) a list of hte persistence update time.
    
    Returns:
    --------
    - reservoir_usgs_param_df (DataFrame): USGS reservoir DA parameters
    - reservoir_usace_param_df (DataFrame): USACE reservoir DA parameters
    '''
    
    # Collect parts and concat ONCE: growing an empty seed frame in the loop is
    # deprecated, and in pandas 3 the seed's object dtypes would win.
    _PERSISTENCE_COLUMNS = [
        'update_time', 'prev_persisted_outflow',
        'persistence_update_time', 'persistence_index',
    ]

    def _empty_persistence_df():
        return pd.DataFrame(data=[], index=[], columns=_PERSISTENCE_COLUMNS)

    usgs_parts, usace_parts, usbr_parts = [], [], []
    
    for r in run_results:   # Corresponds to the ordering of the outflows from line 843 troute/routing/fast_reach/mc_reach.pyx
        
        if len(r[4][0]) > 0:
            tmp_usgs = pd.DataFrame(data = r[4][1], index = r[4][0], columns = ['update_time'])
            tmp_usgs['prev_persisted_outflow'] = r[4][2]
            tmp_usgs['persistence_update_time'] = r[4][4]
            tmp_usgs['persistence_index'] = r[4][3]
            usgs_parts.append(tmp_usgs)
        
        if len(r[5][0]) > 0:
            tmp_usace = pd.DataFrame(data = r[5][1], index = r[5][0], columns = ['update_time'])
            tmp_usace['prev_persisted_outflow'] = r[5][2]
            tmp_usace['persistence_update_time'] = r[5][4]
            tmp_usace['persistence_index'] = r[5][3]
            usace_parts.append(tmp_usace)
    
        if len(r[6][0]) > 0:
            tmp_usbr = pd.DataFrame(data = r[6][1], index = r[6][0], columns = ['update_time'])
            tmp_usbr['prev_persisted_outflow'] = r[6][2]
            tmp_usbr['persistence_update_time'] = r[6][4]
            tmp_usbr['persistence_index'] = r[6][3]
            usbr_parts.append(tmp_usbr)

    # A reservoir that sits on a sub-domain boundary appears as an
    # offnetwork upstream in a downstream compute job AND as a home reach
    # in its own job.  Both jobs run the reservoir through the DA kernel
    # and emit results, so pd.concat above produces duplicate index entries.
    # On the next loop iteration _prep_reservoir_da_dataframes calls
    # .loc[lake_id].to_numpy() and gets a 2-row result for a 1-element
    # index, causing a shape mismatch when the kernel output is unpacked.
    # Deduplicate by keeping the last entry for each lake (both duplicates
    # carry the same computed state, so the choice of first vs. last is
    # inconsequential).
    reservoir_usgs_param_df = pd.concat(usgs_parts) if usgs_parts else _empty_persistence_df()
    reservoir_usace_param_df = pd.concat(usace_parts) if usace_parts else _empty_persistence_df()
    reservoir_usbr_param_df = pd.concat(usbr_parts) if usbr_parts else _empty_persistence_df()

    reservoir_usgs_param_df  = reservoir_usgs_param_df[~reservoir_usgs_param_df.index.duplicated(keep='last')]
    reservoir_usace_param_df = reservoir_usace_param_df[~reservoir_usace_param_df.index.duplicated(keep='last')]
    reservoir_usbr_param_df  = reservoir_usbr_param_df[~reservoir_usbr_param_df.index.duplicated(keep='last')]

    return reservoir_usgs_param_df, reservoir_usace_param_df, reservoir_usbr_param_df

def _set_rfc_reservoir_da_params(reservoir_rfc_param_df, run_results):
    '''
    Update RFC reservoir DA parameters for subsequent loops
    Arguments:
    ----------
    - reservoir_rfc_param_df (DataFrame): RFC reservoir DA parameters
    - run_results                 (list): output from the compute kernel sequence, organized
                                          (because that is how it comes out of the kernel) by network.
                                          For each item in the result, there are seven elements, the
                                          eigth of which is a list of three elements containing: 
                                          1) a list of the segments ids where data assimilation 
                                          was performed (if any) in that network; 
                                          2) a list of the update time; 
                                          3) a list of the rfc timeseries index.
    
    Returns:
    --------
    - reservoir_rfc_param_df (DataFrame): RFC reservoir DA parameters (updated)
    '''
    for r in run_results:   # Corresponds to the ordering of the outflows from line 843 troute/routing/fast_reach/mc_reach.pyx
        if len(r[8][0]) > 0:
            rfc_idx = r[8][0]
            reservoir_rfc_param_df.loc[rfc_idx, 'update_time'] = r[8][1]
            reservoir_rfc_param_df.loc[rfc_idx, 'timeseries_idx'] = r[8][2]
    
    return reservoir_rfc_param_df

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
    LOG.info("Building last observation dataframe is started")
    build_lastobs_start_time = time.time()
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
    LOG.debug(f"Building last observation dataframe completed in %s seconds." % (time.time() - build_lastobs_start_time))
    return lastobs_df

def new_lastobs(run_results, time_increment):
    """
    Creates new "lastobs" dataframe for the next simulation chunk.

    Arguments:
    ----------
    - run_results (list): output from the compute kernel sequence, organized
        (because that is how it comes out of the kernel) by network.
        For each item in the result, there are seven elements, the
        fourth of which is a tuple containing: 1) a list of the
        segments ids where data assimilation was performed (if any)
        in that network; 2) a list of the last valid observation
        applied at that segment; 3) a list of the time in seconds
        from the beginning of the last simulation that the
        observation was applied.
    - time_increment (int): length of the prior simulation. To prepare the
        next lastobs state, we have to convert the time since the prior
        simulation start to a time since the new simulation start.
        If the most recent observation was right at the end of the
        prior loop, then the value in the incoming run_result will
        be equal to the time_increment and the output value will be
        zero. If observations were not present at the last timestep,
        the last obs time will be calculated to a negative value --
        the number of seconds ago that the last valid observation
        was used for assimilation.
        
    Returns:
    --------
    - lastobs_df (DataFrame): Last gage observations data for DA
    """

    columns = ["time_since_lastobs", "lastobs_discharge"]
    # Drop empty frames: pandas lets them decide the result dtype, and concat of
    # nothing but empties raises.
    frames = [
        pd.DataFrame(
            # TODO: Add time_increment (or subtract?) from time_since_lastobs
            np.array([rr[3][1],rr[3][2]]).T,
            index=rr[3][0],
            columns=columns
        )
        for rr in run_results
        if len(rr[3][0]) > 0
    ]
    if not frames:
        return pd.DataFrame(columns=columns)

    df = pd.concat(frames, copy=False)
    df["time_since_lastobs"] = df["time_since_lastobs"] - time_increment

    return df

def new_diversion_applied(run_results) -> dict[int, float]:
    """Donor segment id -> the amount removed at the last step, from the kernel results."""
    out: dict[int, float] = {}
    for rr in run_results:
        if len(rr) > 11 and len(rr[11][0]) > 0:
            out.update(zip(np.asarray(rr[11][0]).tolist(), np.asarray(rr[11][1]).tolist()))
    return out


def read_reservoir_parameter_file(
    reservoir_parameter_file, 
    usgs_hybrid,
    usace_hybrid,
    rfc_forecast,
    lake_index_field = "lake_id", 
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
    LOG.info("Reading and processing reservoir parameter to create crosswalk is started")
    read_reservoir_start_time = time.time()
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
        else:
            usgs_crosswalk = None
        
        if (usace_gage_id_field in ds_vars) and (usace_lake_id_field in ds_vars):
            usace_crosswalk = pd.DataFrame(
                data = ds[usace_gage_id_field].to_numpy(), 
                index = ds[usace_lake_id_field].to_numpy(), 
                columns = [usace_gage_id_field]
            )
            usace_crosswalk.index.name = usace_lake_id_field
        else:
            usace_crosswalk = None
        
    # drop duplicate indices
    df1 = (df1.reset_index()
           .drop_duplicates(subset="lake_id")
           .set_index("lake_id")
          )
    
    # recode to levelpool (1) for reservoir DA types set to false
    if usgs_hybrid == False:
        df1[df1['reservoir_type'] == 2] = 1
    if usace_hybrid == False:
        df1[df1['reservoir_type'] == 3] = 1
    if rfc_forecast == False:
        df1[df1['reservoir_type'] == 4] = 1
    
    LOG.debug(f"Reading and processing reservoir is completed in %s seconds." % (time.time() - read_reservoir_start_time))
    return df1, usgs_crosswalk, usace_crosswalk

def _timeslice_qcqa(discharge, 
                    stns, 
                    t, 
                    qual, 
                    qc_threshold, 
                    frequency_secs, 
                    crosswalk_df, 
                    crosswalk_gage_field='gages',
                    crosswalk_dest_field='link',
                    interpolation_limit=59,
                    cpu_pool=1):
    #FIXME Do we need the following commands? Or something similar? Depends on 
    # what format model engine provides these variables...
    '''
    stationId = np.apply_along_axis(''.join, 1, stns.astype(np.str))
    time_str = np.apply_along_axis(''.join, 1, t.astype(np.str))
    stationId = np.char.strip(stationId)
    '''
    stationId = stns
    time_str = t
    observation_df = (pd.DataFrame({
                                'stationId' : stationId,
                                'datetime'  : time_str,
                                'discharge' : discharge
                            }).
                             set_index(['stationId', 'datetime']).
                             unstack(1, fill_value = np.nan)['discharge'])
    
    observation_qual_df = (pd.DataFrame({
                                'stationId' : stationId,
                                'datetime'  : time_str,
                                'quality'   : qual/100
                            }).
                             set_index(['stationId', 'datetime']).
                             unstack(1, fill_value = np.nan)['quality'])
    
    # Link <> gage crosswalk data
    df = crosswalk_df.reset_index()
    df[crosswalk_gage_field] = np.asarray(df[crosswalk_gage_field]).astype('<U15')
    df = df.set_index(crosswalk_gage_field)
    
    # join crosswalk data with timeslice data, indexed on crosswalk destination field
    observation_df = (df.join(observation_df).
               reset_index().
               set_index(crosswalk_dest_field).
               drop([crosswalk_gage_field], axis=1))

    observation_qual_df = (df.join(observation_qual_df).
               reset_index().
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
    
    # Seconds, not truncated minutes: truncation puts a dt that is not whole minutes
    # on the wrong grid, and any dt under 60 s on no grid at all.
    frequency = f"{int(frequency_secs)}s"
    
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
    observation_df_new = observation_df_T.transpose().loc[crosswalk_df.index]
    observation_df_new.index = observation_df_new.index.astype('int64')

    return observation_df_new

def _interpolate_one(df, interpolation_limit, frequency):

    # Interpolate on the finest grid the minute-based limit and the target step share,
    # so a target that is not whole minutes still gets values rather than NaN.
    step_seconds = int(pd.Timedelta(frequency).total_seconds())
    base_seconds = math.gcd(60, step_seconds) or 60
    interp_out = (df.resample(f'{base_seconds}s').
                        interpolate(
                            limit = interpolation_limit * (60 // base_seconds),
                            limit_direction = 'both'
                        ).
                        resample(frequency).
                        asfreq().
                        to_numpy()
                       )
    return interp_out

def _assemble_lastobs_df(
        discharge, 
        stationIdInd, 
        timeInd, 
        stationId, 
        time, 
        modelTimeAtOutput, 
        gage_link_df, 
        time_shift=0):
    
    gages    = np.char.strip(stationId)
        
    ref_time = datetime.strptime(modelTimeAtOutput, "%Y-%m-%d_%H:%M:%S")
    
    last_ts = timeInd.max()
    
    df_discharge = (
        pd.DataFrame(
        data = {
            'discharge': discharge,
            'stationIdInd': stationIdInd,
            'timeInd': timeInd
            }).
            set_index(['stationIdInd','timeInd']).
            unstack(level=0)
    )

    df_time = (
        pd.DataFrame(
        data = {
            'discharge': time,
            'stationIdInd': stationIdInd,
            'timeInd': timeInd
            }).
            set_index(['stationIdInd','timeInd']).
            unstack(level=1)
    ).to_numpy()
    
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
        lastobs_times.append(df_time[i, idx]) #.decode('utf-8')  <- TODO:decoding was needed in old version, may need again depending on dtype that is provided by model engine
        
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
        join(gage_link_df.reset_index().set_index('gages'), how = 'inner').
        reset_index().
        set_index('link')
    )
    
    lastobs_df = lastobs_df[
        [
            'gages',
            'time_since_lastobs',
            'lastobs_discharge',
        ]
    ]
    lastobs_df.index = lastobs_df.index.astype('int64')
    return lastobs_df

def _rfc_timeseries_qcqa(discharge,stationId,synthetic,totalCounts,timestamp,timestep,lake_number,t0):
    rfc_df = pd.DataFrame(
        {'stationId': stationId,
         'datetime': timestamp,
         'discharge': discharge,
         'synthetic': synthetic
         }
    )
    rfc_df['stationId'] = rfc_df['stationId'].map(bytes.strip)

    validation_df = rfc_df.groupby('stationId').agg(list)
    validation_index = validation_df.index
    use_rfc_df = pd.DataFrame()
    for i in validation_index:
        val_lake_number = lake_number #TODO: placeholder, figure out how to get lake number here...
        val_discharge = validation_df.loc[i].discharge
        val_synthetic = validation_df.loc[i].synthetic
        
        use_rfc = _validate_RFC_data(
            val_lake_number, 
            val_discharge, 
            val_synthetic, 
            '', 
            '', 
            300,
            from_files=False
            )
        
        use_rfc_df = pd.concat([
            use_rfc_df,
            pd.DataFrame({
                'stationId': [i],
                'use_rfc': use_rfc
            })
        ], ignore_index=True)
        
    rfc_df = (rfc_df.
              set_index(['stationId', 'datetime']).
              unstack(1, fill_value = np.nan)['discharge'])
    
    rfc_param_df = (pd.merge(
        pd.DataFrame(
        {'stationId': stationId,
         'idx': rfc_df.columns.get_loc(t0),
         'da_timestep': timestep,
         'totalCounts': totalCounts,
         'update_time': 0,
         }
    ).drop_duplicates(),
    use_rfc_df, on='stationId').
    set_index('stationId'))

    return rfc_df, rfc_param_df


def _rfc_unavailable(msg, action, error=ValueError, warn=True):
    """Raise or warn, by policy, when a forecast a reservoir needs is not usable.

    ``error`` ends the run: enabling RFC DA asserts the forecasts are provisioned.
    ``level_pool`` lets the affected reservoirs run as level pool, so one late gage
    does not break an operational chain of runs. ``warn`` is False where the
    per-run summary already reports the same state.
    """
    if action == 'level_pool':
        if warn:
            LOG.warning("%s Running level pool there instead.", msg)
        return
    raise error(msg)


def _read_timeseries_files(filepath, timeseries_dates, t0, final_persist_datetime,
                           routing_period=300, unavailable_action='error', gages=None):
    """Newest RFC forecast per gage that actually covers t0, as one long frame.

    Newest-first but coverage-gated: the newest issue has the latest slice start, so
    taking it unconditionally passed over an older file that did cover t0.
    """
    # Search for most recent RFC timseries file based on offset hours and lookback window
    # for each location.
    # Names are split on '.' into a fixed five fields, so anything the ingestion did not
    # write (a leftover .gz, a directory) has to be dropped before the split.
    files, ignored = [], []
    for f in sorted(pathlib.Path(filepath).glob('*')):
        keep = f.is_file() and _RFC_FILENAME.fullmatch(f.name)
        (files if keep else ignored).append(f)
    if ignored:
        LOG.warning(
            "reservoir RFC DA: ignoring %d file(s) in %s that are not named "
            "<issue>.<cadence>min.<gage>.RFCTimeSeries.ncdf, e.g. %s",
            len(ignored), filepath, ignored[0].name,
        )
    # create temporary dataframe with file names, split up by location and datetime
    df = pd.DataFrame([f.name.split('.') for f in files], columns=['Datetime','dt','ID','rfc','ext'])
    df = df[df['Datetime'].isin(timeseries_dates)][['ID','Datetime','dt']]
    if gages is not None:
        # The forecast folder is shared across domains and holds hundreds of gages.
        # Only the ones this domain routes can say anything about this run.
        df = df[df['ID'].isin(set(gages))]
    if df.empty:
        msg = (
            f"reservoir RFC DA is enabled but no RFC timeseries file in {filepath} "
            f"is dated within the lookback window {timeseries_dates[0]} to "
            f"{timeseries_dates[-1]} ({len(files)} file(s) present). Provide "
            "forecasts covering the simulation period, or turn off "
            "reservoir_da.reservoir_rfc_da.reservoir_rfc_forecasts."
        )
        _rfc_unavailable(msg, unavailable_action, FileNotFoundError)
        return pd.DataFrame()
    df['Datetime'] = df['Datetime'].apply(lambda _: datetime.strptime(_, '%Y-%m-%d_%H'))

    rfc_df = pd.DataFrame()
    for gage, candidates in df.sort_values('Datetime', ascending=False).groupby('ID'):
        chosen, rejected = None, []
        for stamp, cadence in zip(candidates['Datetime'], candidates['dt']):
            f = f"{stamp.strftime('%Y-%m-%d_%H')}.{cadence}.{gage}.RFCTimeSeries.ncdf"
            record = read_rfc_timeseries(filepath + '/' + f)
            if t0 in record.datetimes:
                chosen = (f, record)
                break
            rejected.append(f"{f} spans {record.datetimes[0]} to {record.datetimes[-1]}")
        if chosen is None:
            msg = (
                f"reservoir RFC DA: none of the {len(rejected)} forecast file(s) for gage "
                f"{gage} in the lookback window cover the simulation start t0={t0}, so no "
                f"forecast can be aligned to the run ({'; '.join(rejected)}). Provide "
                "forecasts covering the simulation period, or turn off "
                "reservoir_da.reservoir_rfc_da.reservoir_rfc_forecasts."
            )
            _rfc_unavailable(msg, unavailable_action)
            continue
        f, record = chosen
        # t0 in the untruncated series; the truncation below only trims the tail.
        timeseries_idx = record.datetimes.get_loc(t0)
        one = pd.DataFrame({
            'stationId': record.station_id,
            'discharges': record.discharges,
            'synthetic_values': record.synthetic,
            'totalCounts': record.total_counts,
            'timeSteps': pd.Timedelta(seconds=record.timestep_seconds),
            'Datetime': record.datetimes,
        })
        # Trim at this forecast's own deadline, which the kernel measures from the issue
        # rather than from t0. The caller passes the allowance as t0 plus persist_days.
        one = one[one['Datetime'] <= record.issue_time + (final_persist_datetime - t0)]
        one['timeseries_idx'] = timeseries_idx
        one['file'] = f
        # Whether there is anything to assimilate is a question about the forecast, so
        # the history in front of t0 must not answer it.
        active = one[one['Datetime'] >= t0]
        use_rfc = _validate_RFC_data(
            record.station_id,
            active.discharges,
            active.synthetic_values,
            filepath,
            f,
            # The run's real timestep, so validation's "longer than an hour" check can fire.
            routing_period,
            False,
            da_time_step=record.timestep_seconds,
        )
        # An absurd value is different: when the sample at the cursor is unusable the
        # kernel walks BACKWARD, so one behind t0 reaches the reservoir just as readily.
        if use_rfc and any(v >= ABSURD_DISCHARGE_CMS for v in one.discharges):
            LOG.warning(
                "reservoir RFC DA: %s holds a discharge at or above the %d cms limit "
                "before t0, which the backtrack can reach; using level pool instead.",
                f, ABSURD_DISCHARGE_CMS,
            )
            use_rfc = False
        one['use_rfc'] = use_rfc
        one['da_timestep'] = record.timestep_seconds
        one['issue_time'] = record.issue_time
        rfc_df = pd.concat([rfc_df, one])
    return rfc_df

def assemble_rfc_dataframes(rfc_timeseries_df, rfc_lake_gage_crosswalk, t0, rfc_parameters):
    action = rfc_parameters.get('reservoir_rfc_forecasts_unavailable_action', 'error')
    # One crosswalk shape from here down: lake id in the INDEX. The NHF builder hands
    # it over reset_index()'d, with the lake id in a column and a RangeIndex, and the
    # gageless lookup below reads the index as lake ids -- so on that path it compared
    # row positions against lake ids, never matched, and a lake the hydrofabric gives
    # no gage took the availability policy (by default, ending the run) instead of the
    # warn-and-level-pool it is meant to take.
    if 'rfc_lake_id' in getattr(rfc_lake_gage_crosswalk, 'columns', ()):
        rfc_lake_gage_crosswalk = rfc_lake_gage_crosswalk.set_index('rfc_lake_id')
    # Retrieve rfc timeseries dataframe from BMI dictionary
    rfc_df = rfc_timeseries_df
    if rfc_df.empty:
        # A BMI caller can transport no station at all.
        msg = (
            "reservoir RFC DA is enabled but no RFC timeseries observations were "
            "provided, so there is nothing to assimilate. Provide forecasts "
            "covering the simulation period, or turn off "
            "reservoir_da.reservoir_rfc_da.reservoir_rfc_forecasts."
        )
        # The reader has already named the folder and the window it searched, and the
        # summary below says the same thing with counts, so warning here says it twice.
        _rfc_unavailable(msg, action, warn=False)
        _report_rfc_selection(None, t0, rfc_parameters, rfc_lake_gage_crosswalk)
        return pd.DataFrame(), pd.DataFrame()
    # Create reservoir_rfc_df dataframe of observations, rows are locations and columns are dates.
    reservoir_rfc_df = rfc_df[['stationId','discharges','Datetime']].sort_values(['stationId','Datetime']).pivot(index='stationId',columns='Datetime').fillna(-999.0)
    reservoir_rfc_df.columns = reservoir_rfc_df.columns.droplevel()
    # Replace gage IDs with lake IDs
    reservoir_rfc_df = (
                        rfc_lake_gage_crosswalk.
                        reset_index().
                        set_index('rfc_gage_id').
                        join(reservoir_rfc_df).
                        set_index('rfc_lake_id')
                        )
    # Create reservoir_rfc_df dataframe of parameters
    if 'issue_time' not in rfc_df:
        # The horizon is measured from the forecast's issue, so a caller that
        # transports observations without it cannot be given one.
        msg = (
            "reservoir RFC DA: the RFC observations carry no issue time, so the "
            "persistence horizon cannot be anchored to the forecast. Read the "
            "forecasts from files, or transport issueTimeUTC alongside them."
        )
        raise ValueError(msg)
    reservoir_rfc_param_df = rfc_df[['stationId','totalCounts','timeseries_idx','file','use_rfc','da_timestep','issue_time']].drop_duplicates().set_index('stationId')
    reservoir_rfc_param_df = (
                            rfc_lake_gage_crosswalk.
                            reset_index().
                            set_index('rfc_gage_id').
                            join(reservoir_rfc_param_df).
                            set_index('rfc_lake_id')
                            )
    # To pass RFC observations to mc_reach they need to be in an array. But the RFC timeseries have different
    # lengths/start dates for each location so the array ends up having many NaN observations after we 
    # pivot the dataframe from long to wide format. We therefore need to adjust the timeseries index and
    # total counts to reflect the new position of t0 in the observation array. 
    if t0 not in reservoir_rfc_df.columns:
        msg = (
            f"reservoir RFC DA: the assembled RFC forecasts span "
            f"{reservoir_rfc_df.columns.min()} to {reservoir_rfc_df.columns.max()} "
            f"and do not cover the simulation start t0={t0}. Provide forecasts "
            "covering the simulation period, or turn off "
            "reservoir_da.reservoir_rfc_da.reservoir_rfc_forecasts."
        )
        _rfc_unavailable(msg, action)
        _report_rfc_selection(None, t0, rfc_parameters, rfc_lake_gage_crosswalk)
        return pd.DataFrame(), pd.DataFrame()
    cadences = set(reservoir_rfc_param_df['da_timestep'].dropna().astype(int))
    if len(cadences) > 1:
        msg = (
            f"reservoir RFC DA: the selected forecasts mix cadences {sorted(cadences)} s. "
            "They share one observation grid and the kernel advances one column per "
            "cadence, so the coarser gages would read padding. Use forecasts of a single "
            "cadence."
        )
        raise ValueError(msg)
    t0_column = reservoir_rfc_df.columns.get_loc(t0)
    rebase = t0_column - reservoir_rfc_param_df['timeseries_idx']
    # totalCounts is consumed as the inclusive last index, so it rebases off C - 1.
    reservoir_rfc_param_df['totalCounts'] = reservoir_rfc_param_df['totalCounts'] - 1 + rebase
    reservoir_rfc_param_df['timeseries_idx'] = t0_column
    # Fill in NaNs with default values.
    # Assign back, not df[col].fillna(inplace=True): pandas 3 copies the
    # intermediate and the fill would stop reaching the frame.
    # Two states share this shape: a lake with no gage at all can never have a
    # forecast, while a gage with no usable file follows the availability policy.
    unmatched = reservoir_rfc_param_df.index[reservoir_rfc_param_df['use_rfc'].isna()]
    gageless = set(
        rfc_lake_gage_crosswalk.index[rfc_lake_gage_crosswalk['rfc_gage_id'].isna()]
    )
    no_gage = sorted(set(unmatched) & gageless)
    no_data = sorted(set(unmatched) - gageless)
    if no_gage:
        LOG.warning(
            "reservoir RFC DA: %d RFC reservoir(s) have no gage in the hydrofabric, so "
            "no forecast can exist for them; they run level pool. Lake id(s): %s",
            len(no_gage), no_gage[:10],
        )
    if no_data:
        msg = (
            f"reservoir RFC DA: {len(no_data)} of {len(reservoir_rfc_param_df)} RFC "
            f"reservoir(s) have a gage but no usable forecast. Lake id(s): "
            f"{no_data[:10]}. Provide forecasts covering the simulation period, or turn "
            "off reservoir_da.reservoir_rfc_da.reservoir_rfc_forecasts."
        )
        _rfc_unavailable(msg, action)
    # eq, not fillna: the column is object dtype after the join, and filling it
    # downcasts silently, which pandas warns about and will change.
    reservoir_rfc_param_df['use_rfc'] = reservoir_rfc_param_df['use_rfc'].eq(True)
    reservoir_rfc_param_df['totalCounts'] = reservoir_rfc_param_df['totalCounts'].fillna(0)
    reservoir_rfc_param_df['da_timestep'] = reservoir_rfc_param_df['da_timestep'].fillna(0)
    # Make sure columns are the correct types
    reservoir_rfc_param_df['totalCounts'] = reservoir_rfc_param_df['totalCounts'].astype(int)
    reservoir_rfc_param_df['da_timestep'] = reservoir_rfc_param_df['da_timestep'].astype(int)
    # A lake with no forecast has no deadline. The packer still casts the column to
    # int32, and NaN there is INT_MIN on some platforms, so give it t0 instead.
    reservoir_rfc_param_df['issue_time'] = reservoir_rfc_param_df['issue_time'].fillna(t0)
    # Seeded at t0, so the first advance is one cadence out, as the standalone does.
    # Float, because the kernel hands a float32 clock back into this column each window.
    reservoir_rfc_param_df['update_time'] = (
        reservoir_rfc_param_df['da_timestep'].astype(float)
    )
    persist_days = rfc_parameters.get('reservoir_rfc_forecast_persist_days', 11)
    # Measured from the forecast's issue, not the run's t0: the horizon caps how long a
    # product may be used. The packer turns it into seconds remaining at each window.
    reservoir_rfc_param_df['persist_until'] = (
        reservoir_rfc_param_df['issue_time'] + timedelta(days=persist_days)
    )
    _report_rfc_selection(reservoir_rfc_param_df, t0, rfc_parameters)

    return reservoir_rfc_df, reservoir_rfc_param_df


def _rfc_search_window(t0, rfc_parameters):
    """The span the selection actually searched, as the reader builds it."""
    lookback = rfc_parameters.get('reservoir_rfc_forecasts_lookback_hours') or 0
    offset = rfc_parameters.get('reservoir_rfc_forecasts_offset_hours') or 0
    end = t0 + timedelta(hours=offset)
    return end - timedelta(hours=lookback), end


def _report_rfc_selection(param_df, t0, rfc_parameters, crosswalk=None):
    """One line per run saying what the RFC DA actually has to work with.

    "Selected", not "applied": whether a forecast reaches a reservoir is decided in the
    kernel, which keeps only the cursor. The searched window is reported because a
    domain running level pool with every issue just outside it looks identical, from
    the outflow alone, to one with no forecasts at all.
    """
    start, end = _rfc_search_window(t0, rfc_parameters)
    selected = (
        param_df['use_rfc'].fillna(False).astype(bool)
        if param_df is not None and not param_df.empty
        else None
    )
    total = (
        len(param_df) if selected is not None
        else (0 if crosswalk is None else len(crosswalk))
    )
    n = int(selected.sum()) if selected is not None else 0
    if not n:
        LOG.info(
            "reservoir RFC DA: no forecast selected for any of %d reservoir(s); all run "
            "level pool. Searched %s to %s.", total, start, end,
        )
        return
    ages = (t0 - param_df.loc[selected, 'issue_time']).dt.total_seconds() / 3600
    LOG.info(
        "reservoir RFC DA: %d of %d reservoir(s) have a forecast selected, %d run level "
        "pool. Selected issues are %.0f to %.0f h old at t0=%s. Searched %s to %s.",
        n, total, total - n, ages.min(), ages.max(), t0, start, end,
    )

def _read_lastobs_file(
        lastobsfile,
        station_id = "stationId",
        ref_t_attr_id = "modelTimeAtOutput",
        time_idx_id = "timeInd",
        obs_discharge_id = "discharge",
        discharge_nan = -9999.0,
        time_shift = 0,
        ):
    LOG.info("Reading last observation file is started")
    read_lastobs_start_time = time.time()
    with xr.open_dataset(lastobsfile) as ds:
        gages = ds[station_id].values
        
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

    lastobs_df = pd.DataFrame(data = data_var_dict)
    lastobs_df['gages'] = lastobs_df['gages'].str.decode('utf-8')

    LOG.debug(f"Reading last observation file is completed in %s seconds." % (time.time() - read_lastobs_start_time))
    return lastobs_df

def _create_GL_dfs(GL_crosswalk_df, data_assimilation_parameters, run_parameters, 
                   da_run, t0):

    # USGS gages:
    usgs_timeslices_folder = data_assimilation_parameters.get("usgs_timeslices_folder", None)

    # TODO: join timeslice folder and files into complete path upstream
    usgs_timeslices_folder = pathlib.Path(usgs_timeslices_folder)
    usgs_files = [usgs_timeslices_folder.joinpath(f) for f in 
                da_run['usgs_timeslice_files']]

    usgs_GL_crosswalk_df = GL_crosswalk_df[GL_crosswalk_df.index.isin([4800002,4800004])]
    if usgs_files:
        usgs_GL_df = (
            nhd_io.get_GL_obs_from_timeslices(
                usgs_GL_crosswalk_df,
                usgs_files,
                cpu_pool=run_parameters.get("cpu_pool", 1),
            )
        )
        usgs_GL_df = pd.melt(usgs_GL_df, 
                             var_name='Datetime',
                             value_name='Discharge',
                             ignore_index=False).dropna().reset_index()
        
    else:
        usgs_GL_df = pd.DataFrame()

    # Canadian gages:
    canadian_timeslices_folder = data_assimilation_parameters.get("canada_timeslices_folder", None)
    canadian_files = [canadian_timeslices_folder.joinpath(f) for f in 
                da_run['canada_timeslice_files']]

    canadian_GL_crosswalk_df = GL_crosswalk_df.loc[[4800006]]

    if canadian_files:
        canadian_GL_df = (
            nhd_io.get_GL_obs_from_timeslices(
                canadian_GL_crosswalk_df,
                canadian_files,
                cpu_pool=run_parameters.get("cpu_pool", 1),
            )
        )
        canadian_GL_df = pd.melt(canadian_GL_df, 
                                 var_name='Datetime',
                                 value_name='Discharge',
                                 ignore_index=False).dropna().reset_index()
        
    else:
        canadian_GL_df = pd.DataFrame()
    
    # Lake Ontario data:
    if 'LakeOntario_outflow' in da_run:
        lake_ontario_df = _create_LakeOntario_df(run_parameters, t0, da_run)
    else:
        lake_ontario_df = pd.DataFrame()

    great_lakes_df = pd.concat(
        [usgs_GL_df, canadian_GL_df, lake_ontario_df]
        ).rename(columns={'link': 'lake_id'}).sort_values(by=['lake_id','Datetime'])
    
    great_lakes_df['time'] = pd.to_datetime(great_lakes_df['Datetime'], format='%Y-%m-%d_%H:%M:%S') - t0
    great_lakes_df['time'] = great_lakes_df.time.dt.total_seconds().astype(int)
    great_lakes_df.drop('Datetime', axis=1, inplace=True)
    
    great_lakes_param_df = pd.DataFrame(great_lakes_df.lake_id.unique(), columns=['lake_id']).sort_values('lake_id')
    great_lakes_param_df['previous_assimilated_outflows'] = np.nan
    great_lakes_param_df['previous_assimilated_time'] = 0
    great_lakes_param_df['update_time'] = 0
    
    return great_lakes_df, great_lakes_param_df
