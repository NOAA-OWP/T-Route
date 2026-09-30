
from array import array
from troute.network.reservoirs.rfc.rfc import MC_RFC
from troute.routing.compute import NWM_DAM_LENGTH_MULTIPLIER
from troute.network.reservoirs.levelpool.levelpool import MC_Levelpool
from troute.routing.fast_reach.reservoir_RFC_da import reservoir_RFC_da, reservoir_RFC_da_v2

import xarray as xr
import numpy as np

reservoir_index_file = "/home/dongha.kim/github/t-route/temp_input/reservoir_index/reservoir_index_Extended_AnA_NWMv2.1.nc"
rfc_timeseries_folder = "/home/dongha.kim/github/t-route/temp_input/rfc_timeseries/"


def create_rfc_values():
    rfc_da_df = xr.open_dataset(rfc_timeseries_folder + '2021-10-20_00.60min.KNFC1.RFCTimeSeries.ncdf', engine="netcdf4").to_dataframe()
    return {
        'lake_surface__elevation': np.array([189.255814]),
        'lake_area': np.array([3.80643]),
        'weir_elevation': np.array([188.205]),
        'weir_coefficient': np.array([0.4]),
        'weir_length': np.array([10.0]),
        'dam_length': np.array([10.0]),
        'orifice_elevation': np.array([156.159999]),
        'orifice_coefficient': np.array([0.1]),
        'orifice_area': np.array([1.0]),
        'max_depth': np.array([193.860001]),
        'lake_number': np.array([347987]),
        'initial_fractional_depth': np.array([0.9]),
        'upstream_ids': np.array([int(1)]),
        'res_type': np.array([4]),
        'lake_water~incoming__volume_flow_rate': np.array([91.2719]),
        'gage_observations': rfc_da_df['discharges'].to_numpy(),
        'gage_time': np.array([0.0]),
        'da_idx': np.array([int(50)]),
        'time_step': np.array([3600]),
        'rfc_forecast_persist_seconds': np.array([950400]), # maximum number of days that RFC-supplied forecast will be used/persisted in a simultion.
        'synthetic_flag': rfc_da_df['synthetic_values'].to_numpy(),
    }

values = create_rfc_values()
# ** lake_area is passed to Fortran levelpool or RFC as the input argument with unit of km^2 (then, internally)
#    converted to m^2) 
args = [values['lake_area'], values['max_depth'], values['orifice_area'], #values['lake_area']*1e6
        values['orifice_coefficient'], values['orifice_elevation'],
        values['weir_coefficient'], values['weir_elevation'], 
        values['weir_length'], values['initial_fractional_depth'], 
        0.0, values['lake_surface__elevation'], NWM_DAM_LENGTH_MULTIPLIER]


#--------------------------------------------------------
# Create fortran object and run 1 timestep (300 sec). q
# Calls def __ini__() of cdef class MC_RFC(Reach)
#--------------------------------------------------------
rfc_module = MC_RFC(
    0,
    values['lake_number'],
    array('l',values['upstream_ids']),
    args,
    4,
    reservoir_index_file,
    "2021-10-20_13:00:00",
    rfc_timeseries_folder,
    28.0,
)

# fortran_outflow, fortran_water_elevation = rfc_module.run(
#    values['lake_water~incoming__volume_flow_rate'], 0.0, 300
#    )

inflow_list =  [187,
                185,
                183,
                181,
                179,
                177,
                175,
                173,
                171,
                169,
                167,
                165,
                163,
                161,
                159,
                157,
                155,
                153,
                151,
                149,
                147,
                145,
                143,
                141,
                139,
                 ]

for inflow in inflow_list:
    fortran_outflow, fortran_water_elevation = rfc_module.run(
        inflow, 0.0, 300
        )
    print(f"inflow:{inflow} outflow:{fortran_outflow} welevation:{fortran_water_elevation}")

#--------------------------------------------------------
# Create python object and run 1 timestep (300 sec)
#--------------------------------------------------------
levelpool = MC_Levelpool(
    0, 
    values['lake_number'], 
    array('l',values['upstream_ids']), 
    args,
    int(1))

# Get water elevation before levelpool calculation
initial_water_elevation = levelpool.water_elevation

# Run routing
levelpool_outflow, levelpool_water_elevation = levelpool.run(
    values['lake_water~incoming__volume_flow_rate'], 0.0, 300
    )


'''
# Data Assimilation
(
    new_outflow, 
    new_water_elevation, 
    new_time_series_update_time,
    new_timeseries_idx,
    dynamic_reservoir_type, 
    assimilated_value, 
    #assimilated_source_file,
) = reservoir_RFC_da(
    levelpool.lake_number,                           # lake identification number
    values['gage_observations'],                     # gage observation values (cms)
    values['da_idx'] - 1,                            # index of for current time series observation
    values['synthetic_flag'],                        # boolean flags indicating synthetic values
    300,                                             # routing period (sec)
    0,                                               # model time (sec)
    0,                                               # time to advance to next time series index
    values['time_step'],                             # frequency of DA observations (sec)
    values['rfc_forecast_persist_seconds'],          # max seconds RFC forecasts will be used/persisted
    int(4),                                          # reservoir type
    values['lake_water~incoming__volume_flow_rate'], # waterbody inflow (cms)
    initial_water_elevation,                         # water surface el., previous timestep (m)
    levelpool_outflow,                                  # levelpool simulated outflow (cms)
    levelpool_water_elevation,                          # levelpool simulated water elevation (m)
    levelpool.lake_area,                             # waterbody surface area (km2)
    levelpool.max_depth,                             # max waterbody depth (m)
)
'''
rfc_timeseries_offset_hours = 28 
rfc_gage_id = "KNFC1"
model_start_date = "2021-10-20_01:00:00"
current_time = 0
rfc_timeseries_offset_hours = 28
rfc_forecast_persist_days = 11

(
    new_outflow, 
    new_water_elevation, 
    new_time_series_update_time,
    new_timeseries_idx,
    dynamic_reservoir_type, 
    assimilated_value, 
) = reservoir_RFC_da_v2(
    levelpool.lake_number,                           # lake identification number
    rfc_gage_id, 
    model_start_date,
    300,                                             # routing period (sec)
    current_time,                                    # 
    rfc_timeseries_offset_hours,                     # offset hours ahead of model start date for searching RFC files backward in time
    rfc_forecast_persist_days,                       # max number of days that RFC-supplied forecast will be used/persisted in simulation
    rfc_timeseries_folder,                           # path to folder containing RFC timeseires files
    int(4),                                          # reservoir type
    values['lake_water~incoming__volume_flow_rate'], # waterbody inflow (cms)
    initial_water_elevation,                         # water surface el., previous timestep (m)
    levelpool_outflow,                               # levelpool simulated outflow (cms)
    levelpool_water_elevation,                       # levelpool simulated water elevation (m)
    levelpool.lake_area,                             # waterbody surface area (km2)
    levelpool.max_depth,                             # ** actually max elevation, not depth, of waterbody (m)
)


# update water elevation and outflow
python_water_elevation = new_water_elevation
python_outflow = new_outflow

#----------------------------------------------------
# Compare values:
#----------------------------------------------------
print('Computed outflow values:')
print(f'  Fortran: {fortran_outflow}')
print(f'  Python:  {python_outflow}')
print('')
print('Computed water elevation values:')
print(f'  Fortran: {fortran_water_elevation}')
print(f'  Python:  {python_water_elevation}')
