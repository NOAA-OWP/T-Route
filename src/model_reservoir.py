
import numpy as np

import yaml
from array import array

from troute.network.reservoirs.levelpool.levelpool import MC_Levelpool
from troute.routing.fast_reach.reservoir_hybrid_da import reservoir_hybrid_da
from troute.routing.fast_reach.reservoir_RFC_da import reservoir_RFC_da, preprocess_RFC_data

class reservoir_model():

    def __init__(self, bmi_cfg_file=None):
        """
        
        """
        __slots__ = ['_levelpool','_inflow','_outflow','_water_elevation', '_res_type',
                     '_update_time', '_prev_persisted_flow', '_persistence_update_time',
                     '_persistence_index', '_time', '_time_step','_dynamic_res_type',
                     '_assimilated_value', '_assimilated_source_file',
                     '_levelpool_outflow,', '_timeseries_update_time',
                     '_timeseries_idx', '_rfc_gage_id', '_rfc_timeseries_folder',
                     '_rfc_timeseries_file', '_rfc_timeseries_offset_hours', '_rfc_forecast_persist_days',
                     '_rfc_issue_age_seconds',
                     '_time_step_seconds', '_total_counts', '_timeseries_discharges', 
                     '_use_RFC']
        
        if bmi_cfg_file:
            (bmi_parameters, #TODO We might not need any bmi specific parameters
             log_parameters, #TODO Update all logging/warnings throughout standalone reservoirs...
             compute_parameters,
             rfc_parameters) = _read_config_file(bmi_cfg_file)
            
            if compute_parameters:
                self._time_step = compute_parameters.get("model_time_step", 300)
                self._t0 = compute_parameters.get("model_start_time", None)
                if not self._t0:
                    raise(RuntimeError("No start_datetime provided in config file."))
            if rfc_parameters:
                self._rfc_gage_id = rfc_parameters.get("reservoir_rfc_gage_id", None)
                self._rfc_timeseries_folder = rfc_parameters.get("reservoir_rfc_timeseries_folder", None)
                self._rfc_timeseries_offset_hours = rfc_parameters.get("reservoir_rfc_timeseries_offset_hours", None)
                self._rfc_forecast_persist_days = rfc_parameters.get("reservoir_rfc_forecast_persist_days", 11)
            
        else:
            raise(RuntimeError("No config file provided."))
            
        self._time = 0.0
    

    def preprocess_static_vars(self, values: dict):

        lake_number = values['waterbody_id']
        lake_area = values['LkArea']
        max_depth = values['LkMxE']
        orifice_area = values['OrificeA']
        orifice_coefficient = values['OrificeC']
        orifice_elevation = values['OrificeE']
        weir_coefficient = values['WeirC']
        weir_elevation = values['WeirE']
        weir_length = values['WeirL']
        initial_fractional_depth = values['ifd']
        water_elevation = values['lake_surface__elevation']
        
        args = [lake_area, max_depth, orifice_area,
                orifice_coefficient, orifice_elevation,
                weir_coefficient, weir_elevation, weir_length,
                initial_fractional_depth, 0.0, water_elevation]
        
        upstream_ids = array('l', values['upstream_ids'])
        self._res_type = values['reservoir_type']

        self._levelpool = MC_Levelpool(0, lake_number, upstream_ids, args, 1)
        
        # Set inflow
        self._inflow = values['lake_water~incoming__volume_flow_rate']
        self._inflow_list = self._inflow.tolist()
        self._outflow_list = []
        self._water_elevation_list = []

        # Set data assimilation parameters
        if self._res_type==2 or self._res_type==3:
            self._update_time = 0
            self._prev_persisted_outflow = np.nan
            self._persistence_index = 0
            self._persistence_update_time = 0
        
        if self._res_type==4 or self._res_type==5:
            (self._use_RFC, 
             self._timeseries_discharges, 
             self._timeseries_idx, 
             self._update_time, 
             self._da_time_step, 
             self._total_counts,
             self._rfc_timeseries_file,
             self._rfc_issue_age_seconds) = preprocess_RFC_data(self._t0,
                                     self._rfc_timeseries_offset_hours,
                                     self._rfc_gage_id,
                                     self._rfc_timeseries_folder,
                                     lake_number,
                                     self._time_step)

    def run(self, values: dict,):
        """
        Run this model into the future, updating the state stored in the provided model dict appropriately.
    
        Parameters
        ----------
        values: dict
            The model state data structure.
            
        Returns
        -------
        """
        # Check if new inflow values have been provided
        #TODO Should we come up with more clever way to do cycle through inflows for multiple time steps?
        if len(self._inflow_list)==0:
            self._inflow_list = self._inflow.tolist()

        # Get water elevation before levelpool calculation
        initial_water_elevation = self._levelpool.water_elevation

        # Run routing
        inflow = self._inflow_list.pop(0)
        self._levelpool_outflow, levelpool_water_elevation = self._levelpool.run(inflow, 0.0, self._time_step)

        # Data Assimilation
        if self._res_type==2 or self._res_type==3:
            (
                new_outflow,
                new_persisted_outflow,
                new_water_elevation, 
                new_update_time, 
                new_persistence_index, 
                new_persistence_update_time
            ) = reservoir_hybrid_da(
                self._levelpool.lake_number,        # lake identification number
                values['gage_observations'],        # gage observation values (cms)
                values['gage_time'],                # gage observation times (sec)
                # End of the interval being routed, as mc_reach passes dt*timestep.
                self._time + self._time_step,       # model time (sec)
                self._prev_persisted_outflow,       # previously persisted outflow (cms)
                self._persistence_update_time,      
                self._persistence_index,            # number of sequentially persisted update cycles
                self._levelpool_outflow,            # levelpool simulated outflow (cms)
                inflow,                             # waterbody inflow (cms)
                self._time_step,                    # model timestep = time step of inflow to reservoir (sec)
                self._levelpool.lake_area,          # waterbody surface area (km2)
                self._levelpool.max_depth,          # max waterbody depth (m)
                self._levelpool.orifice_elevation,  # orifice elevation (m)
                initial_water_elevation,            # water surface el., previous timestep (m)
                24.0,                               # gage lookback hours (hrs)
                self._update_time,                  # waterbody update time (sec)
            )
            
            # update levelpool water elevation state
            water_elevation = self._levelpool.assimilate_elevation(new_water_elevation)
            
            # change reservoir_outflow
            self._outflow = new_outflow
            
            # update USGS DA reservoir state arrays
            self._update_time = new_update_time
            self._prev_persisted_outflow = new_persisted_outflow
            self._persistence_index = new_persistence_index
            self._persistence_update_time = new_persistence_update_time
        
        elif self._res_type==4 or self._res_type==5:
            
            (
                new_outflow, 
                new_water_elevation, 
                new_update_time,
                new_timeseries_idx,
                dynamic_reservoir_type, 
                assimilated_value, 
                assimilated_source_file,
            ) = reservoir_RFC_da(
                self._use_RFC,                            # boolean whether to use RFC values or not
                self._timeseries_discharges,              # gage observation values (cms)
                self._timeseries_idx,                     # index of for current time series observation
                self._total_counts,                       # total number of observations in RFC timeseries
                self._time_step,                          # routing period (sec)
                self._time + self._time_step,             # model time (sec)
                self._update_time,                        # time to advance to next time series index
                self._da_time_step,                       # frequency of DA observations (sec)
                self._rfc_forecast_persist_days*24*60*60
                - self._rfc_issue_age_seconds,            # allowance left, measured from the forecast's issue
                self._res_type,                           # reservoir type
                inflow,                                   # waterbody inflow (cms)
                initial_water_elevation,                  # water surface el., previous timestep (m)
                self._levelpool_outflow,                  # levelpool simulated outflow (cms)
                levelpool_water_elevation,                # levelpool simulated water elevation (m)
                self._levelpool.lake_area*1.0e6,          # waterbody surface area (km2 -> m2)
                self._levelpool.max_depth,                # max waterbody depth (m)
                self._rfc_timeseries_file,                # RFC file name
            )

            # update levelpool water elevation state
            water_elevation = self._levelpool.assimilate_elevation(new_water_elevation)
            
            # change reservoir_outflow
            self._outflow = new_outflow

            # update DA reservoir state parameters
            self._update_time = new_update_time
            self._timeseries_idx = new_timeseries_idx
            self._dynamic_res_type = dynamic_reservoir_type
            self._assimilated_value = assimilated_value
            self._assimilated_source_file = assimilated_source_file

        # Set output variables
        values['lake_water~outgoing__volume_flow_rate'] = self._outflow
        values['lake_surface__elevation'] = water_elevation

        # Append outflow list
        self._outflow_list.append(self._outflow)
        self._water_elevation_list.append(water_elevation)

        # update model time
        self._time += self._time_step


# Utility functions -------
def _read_config_file(custom_input_file):
    '''
    Read-in data from user-created configuration file.
    
    Arguments
    ---------
    custom_input_file (str): configuration filepath, .yaml
    
    Returns
    -------
    bmi_parameters               (dict): Input parameters re bmi configuration
    log_parameters               (dict): Input parameters re logging
    compute_parameters           (dict): Input parameters re computation settings
    data_assimilation_parameters (dict): Input parameters re data assimilation

    '''
    with open(custom_input_file) as custom_file:
        data = yaml.load(custom_file, Loader=yaml.SafeLoader)

    bmi_parameters = data.get("bmi_parameters", None)
    log_parameters = data.get("log_parameters", None)
    compute_parameters = data.get("compute_parameters", None)
    data_assimilation_parameters = data.get("reservoir_data_assimilation_parameters", None)
    rfc_parameters = data_assimilation_parameters.get("rfc", None)

    return (
        bmi_parameters,
        log_parameters,
        compute_parameters,
        rfc_parameters,
    )