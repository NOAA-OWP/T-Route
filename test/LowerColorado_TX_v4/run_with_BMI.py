import sys
import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from datetime import datetime, timedelta
import time

# Compute the base path relative to this script's location
script_dir = os.path.dirname(__file__)
# Correctly navigate to the 'src' directory
t_route_src_path = os.path.join(script_dir, '../../', 'src')
sys.path.append(os.path.abspath(t_route_src_path))

import bmi_troute
import bmi_DAforcing

print(t_route_src_path)

# Define generic troute 'run' function for any configuration
def run_troute(config):    
    print('***************',config,' Run***************', sep='')

    base_path = script_dir

    config_dict = {
        'Standard AnA': {
            'yaml': '/configs/Standard_AnA.yaml',
            'duration': 3,
            'restart': 2
            },
        'ShortRange Forecast': {
            'yaml': '/configs/Shortrange_Forecast.yaml',
            'duration': 18,
            'restart': None
            }
        }
    
    #-------------------------------------
    # Begin our t-route related operations
    #-------------------------------------
    # Initialize DA model with configuration file
    print('Initialize DA Module...')
    DAforcing = bmi_DAforcing.bmi_DAforcing()
    DAforcing.initialize(bmi_cfg_file = base_path + config_dict[config]['yaml'])

    # Initialize troute model with configuration file
    print('Initialize t-route Module...')
    troute = bmi_troute.bmi_troute()
    troute.initialize(bmi_cfg_file = base_path + config_dict[config]['yaml'])

    # The following BMI transport is conducted with df2a ("dataframe to array")
    # from dataframes to numerical numpy arrays in the DA module, and then 
    # back to dataframes in DataAssimilation in t-route proper using a2df
    # ("array to dataframe"). At this stage, the following dataframes are 
    # treated such:
    # - usgs
    # - usgs_reservoir
    # - usace_reservoir
    # - usbr_reservoir
    # - rfc

    # Date reference
    troute.set_value('dateNull', DAforcing.get_value('dateNull'))

    # USGS dataframe
    troute.set_value('datesSecondsArray_usgs', DAforcing.get_value('datesSecondsArray_usgs'))
    troute.set_value('nDates_usgs', DAforcing.get_value('nDates_usgs'))
    troute.set_value('stationArray_usgs', DAforcing.get_value('stationArray_usgs'))
    troute.set_value('stationStringLengthArray_usgs', DAforcing.get_value('stationStringLengthArray_usgs'))
    troute.set_value('nStations_usgs', DAforcing.get_value('nStations_usgs'))
    troute.set_value('usgs_Array', DAforcing.get_value('usgs_Array'))

    # USGS reservoir dataframe
    troute.set_value('datesSecondsArray_reservoir_usgs', DAforcing.get_value('datesSecondsArray_reservoir_usgs'))
    troute.set_value('nDates_reservoir_usgs', DAforcing.get_value('nDates_reservoir_usgs'))
    troute.set_value('stationArray_reservoir_usgs', DAforcing.get_value('stationArray_reservoir_usgs'))
    troute.set_value('stationStringLengthArray_reservoir_usgs', DAforcing.get_value('stationStringLengthArray_reservoir_usgs'))
    troute.set_value('nStations_reservoir_usgs', DAforcing.get_value('nStations_reservoir_usgs'))
    troute.set_value('usgs_reservoir_Array', DAforcing.get_value('usgs_reservoir_Array'))

    # USACE reservoir dataframe
    troute.set_value('datesSecondsArray_reservoir_usace', DAforcing.get_value('datesSecondsArray_reservoir_usace'))
    troute.set_value('nDates_reservoir_usace', DAforcing.get_value('nDates_reservoir_usace'))
    troute.set_value('stationArray_reservoir_usace', DAforcing.get_value('stationArray_reservoir_usace'))
    troute.set_value('stationStringLengthArray_reservoir_usace', DAforcing.get_value('stationStringLengthArray_reservoir_usace'))
    troute.set_value('nStations_reservoir_usace', DAforcing.get_value('nStations_reservoir_usace'))
    troute.set_value('usace_reservoir_Array', DAforcing.get_value('usace_reservoir_Array'))

    # USBR reservoir dataframe
    troute.set_value('datesSecondsArray_reservoir_usbr', DAforcing.get_value('datesSecondsArray_reservoir_usbr'))
    troute.set_value('nDates_reservoir_usbr', DAforcing.get_value('nDates_reservoir_usbr'))
    troute.set_value('stationArray_reservoir_usbr', DAforcing.get_value('stationArray_reservoir_usbr'))
    troute.set_value('stationStringLengthArray_reservoir_usbr', DAforcing.get_value('stationStringLengthArray_reservoir_usbr'))
    troute.set_value('nStations_reservoir_usbr', DAforcing.get_value('nStations_reservoir_usbr'))
    troute.set_value('usbr_reservoir_Array', DAforcing.get_value('usbr_reservoir_Array'))

    # RFC timeseries dataframe converted
    troute.set_value('rfc_da_timestep', DAforcing.get_value('rfc_da_timestep'))
    troute.set_value('rfc_totalCounts', DAforcing.get_value('rfc_totalCounts'))
    troute.set_value('rfc_synthetic_values', DAforcing.get_value('rfc_synthetic_values'))
    troute.set_value('rfc_discharges', DAforcing.get_value('rfc_discharges'))
    troute.set_value('rfc_timeseries_idx', DAforcing.get_value('rfc_timeseries_idx'))
    troute.set_value('rfc_use_rfc', DAforcing.get_value('rfc_use_rfc'))
    troute.set_value('rfc_Datetime', DAforcing.get_value('rfc_Datetime'))
    troute.set_value('rfc_timeSteps', DAforcing.get_value('rfc_timeSteps'))
    troute.set_value('rfc_StationId_array', DAforcing.get_value('rfc_StationId_array'))
    troute.set_value('rfc_StationId_stringLengths', DAforcing.get_value('rfc_StationId_stringLengths'))
    troute.set_value('rfc_List_array', DAforcing.get_value('rfc_List_array'))
    troute.set_value('rfc_List_stringLengths', DAforcing.get_value('rfc_List_stringLengths') ) 

    # For the following dataframes, the BMI transport routines are in place
    # in both directions (df2a and a2df), but only export from the DA module
    # is implemented for the time being. The reverse (import into t-route proper)
    # is to be completed in the near. The following dataframes are 
    # treated such (i.e., only export from DA so far):
    # - lastobs
    # - lite-restart (i.e., q0/to/q0_index and waterbody_df)

    # lastobs dataframe converted
    lastObs_gageArray = DAforcing.get_value('lastObs_gageArray')
    lastObs_gageStringLengths = DAforcing.get_value('lastObs_gageStringLengths')
    lastObs_timeSince = DAforcing.get_value('lastObs_timeSince')
    lastObs_discharge = DAforcing.get_value('lastObs_discharge')

    # lite-restart dataframe converted
    # q0
    q0_columnArray = DAforcing.get_value('q0_columnArray')
    q0_columnLengthArray = DAforcing.get_value('q0_columnLengthArray')
    q0_nCol = DAforcing.get_value('q0_nCol')
    q0_indexArray = DAforcing.get_value('q0_indexArray')
    q0_nIndex = DAforcing.get_value('q0_nIndex')
    q0_Array = DAforcing.get_value('q0_Array')
    # waterbody_df
    waterbodyLR_columnArray = DAforcing.get_value('waterbodyLR_columnArray')
    waterbodyLR_columnLengthArray = DAforcing.get_value('waterbodyLR_columnLengthArray')
    waterbodyLR_nCol = DAforcing.get_value('waterbodyLR_nCol')
    waterbodyLR_indexArray = DAforcing.get_value('waterbodyLR_indexArray')
    waterbodyLR_nIndex = DAforcing.get_value('waterbodyLR_nIndex')
    waterbodyLR_Array = DAforcing.get_value('waterbodyLR_Array')

    # For the time being, these dataframes is still handled "the old way":
    # - lastobs
    # - lite-restart (i.e., q0/to/q0_index and waterbody_df)   

    troute.set_value('waterbody_df', DAforcing.get_value('waterbody_df'))
    troute.set_value('waterbody_df_index', DAforcing.get_value('waterbody_df_ids'))
    troute.set_value('lastobs_df', DAforcing.get_value('lastobs_df'))
    troute.set_value('lastobs_df_index', DAforcing.get_value('lastobs_df_ids'))

    # Set restart values
    troute.set_value('q0', DAforcing.get_value('q0'))
    troute.set_value('q0_index', DAforcing.get_value('q0_ids'))
    troute.set_value('t0', DAforcing.get_value('t0'))

    print('Done transporting data from DA to t-route')

    # Read forcing data first. This info will eventually be provided by model engine.
    timeseries_start = DAforcing._model._t0
    timeseries_end = timeseries_start + timedelta(hours=config_dict[config]['duration'] - 1)
    delta = timedelta(hours=1)
    forcing_files = []
    while timeseries_start <= timeseries_end:
        forcing_files.append(base_path+'/channel_forcing/'+ timeseries_start.strftime('%Y%m%d%H%M') + '.CHRTOUT_DOMAIN1.csv')
        timeseries_start += delta

    # Loop through hourly timesteps calling update_until
    print('Begin routing...')
    for hr in range(config_dict[config]['duration']):
        # Retrieve forcing data
        forcing_vals = pd.read_csv(forcing_files[hr]).set_index('feature_id')
        # Get rid of duplicate nexus points:
        idx = forcing_vals.index.duplicated()
        forcing_vals = forcing_vals.loc[~idx]

        # Full network
        # Set dynamic values
        troute.set_value('land_surface_water_source__volume_flow_rate', np.array(forcing_vals.iloc[:,0]))
        troute.set_value('land_surface_water_source__id', np.array(forcing_vals.index))

        # Set duration to run model for
        nts = 12
        n = nts*300
        
        # Run our model
        troute.update_until(n)

        # Retrieve troute output
        if hr!=config_dict[config]['restart']:
            DAforcing.set_value('nudging', troute.get_value('nudging'))
            DAforcing.set_value('nudging_ids', troute.get_value('nudging_ids'))
            DAforcing.set_value('fvd_results', troute.get_value('fvd_results'))
            DAforcing.set_value('fvd_index', troute.get_value('fvd_index'))
            DAforcing.set_value('t-route_model_time', troute.get_current_time())
            DAforcing.update()

        if hr==config_dict[config]['restart']:
            DAforcing.set_value('q0', troute.get_value('q0'))
            DAforcing.set_value('q0_ids', troute.get_value('q0_index'))

            DAforcing.set_value('waterbody_df', troute.get_value('waterbody_df'))
            DAforcing.set_value('waterbody_df_ids', troute.get_value('waterbody_df_index'))
            DAforcing.set_value('write_lite_restart', 1)

            DAforcing.set_value('lastobs_df', troute.get_value('lastobs_df'))
            DAforcing.set_value('lastobs_df_ids', troute.get_value('lastobs_df_index'))

            DAforcing.set_value('t-route_model_time', troute.get_current_time())
            DAforcing.set_value('nudging', troute.get_value('nudging'))
            DAforcing.set_value('nudging_ids', troute.get_value('nudging_ids'))
            DAforcing.set_value('fvd_results', troute.get_value('fvd_results'))
            DAforcing.set_value('fvd_index', troute.get_value('fvd_index'))
            DAforcing.set_value('t-route_model_time', troute.get_current_time())
            
            DAforcing.update()

    troute._model.print_timing_summary()


##### Run t-route #####
# Time logging
total_start_time = time.time()

# Run 2 configs and save results for analysis:
run_troute('Standard AnA')
run_troute('ShortRange Forecast')

total_time = time.time() - total_start_time

print('Total run:', total_time)
print('Total time: {} secs'.format(round(total_time,2)))

