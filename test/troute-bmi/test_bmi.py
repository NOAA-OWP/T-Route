import pytest
import os
import sys
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import numpy as np

from bmi_troute import bmi_troute, _read_config_file
from test import temporarily_change_dir


def test_read_config(sample_config):
    with temporarily_change_dir(sample_config.parent):
        bmi_parameters = _read_config_file(str(sample_config))
        keys = list(bmi_parameters.keys())
        assert 'flowpath_columns' in keys
        assert 'attributes_columns' in keys
        assert 'waterbody_columns' in keys
        assert 'network_columns' in keys


def test_initialization(sample_config):
    """Test basic initialization of the BMI model."""
    with temporarily_change_dir(sample_config.parent):
        model = bmi_troute()
        model.initialize(str(sample_config))
        
        assert model._model is not None
        assert model._time_units == "s"
        assert model._start_time == 0.0
        
        model.finalize()

def test_basic_variable_getters(initialized_model):
    """Test getting various model variables."""
    # Test getting input variable names
    input_vars = initialized_model.get_input_var_names()
    assert isinstance(input_vars, (list, tuple))
    assert "segment_id" in input_vars
    assert "land_surface_water_source__volume_flow_rate" in input_vars
    
    # Test getting output variable names
    output_vars = initialized_model.get_output_var_names()
    assert isinstance(output_vars, (list, tuple))
    assert "channel_exit_water_x-section__volume_flow_rate" in output_vars


def test_integration_model_update(sample_config, initialized_model, DAforcing):
    # initialized_model.update()

    initialized_model.set_value('dateNull', DAforcing.get_value('dateNull'))

    # initialized_model.set_value('datesSecondsArray_usgs', DAforcing.get_value('datesSecondsArray_usgs'))
    # initialized_model.set_value('nDates_usgs', DAforcing.get_value('nDates_usgs'))
    # initialized_model.set_value('stationArray_usgs', DAforcing.get_value('stationArray_usgs'))
    # initialized_model.set_value('stationStringLengthArray_usgs', DAforcing.get_value('stationStringLengthArray_usgs'))
    # initialized_model.set_value('nStations_usgs', DAforcing.get_value('nStations_usgs'))
    # initialized_model.set_value('usgs_Array', DAforcing.get_value('usgs_Array'))

    # initialized_model.set_value('datesSecondsArray_reservoir_usgs', DAforcing.get_value('datesSecondsArray_reservoir_usgs'))
    # initialized_model.set_value('nDates_reservoir_usgs', DAforcing.get_value('nDates_reservoir_usgs'))
    # initialized_model.set_value('stationArray_reservoir_usgs', DAforcing.get_value('stationArray_reservoir_usgs'))
    # initialized_model.set_value('stationStringLengthArray_reservoir_usgs', DAforcing.get_value('stationStringLengthArray_reservoir_usgs'))
    # initialized_model.set_value('nStations_reservoir_usgs', DAforcing.get_value('nStations_reservoir_usgs'))
    # initialized_model.set_value('usgs_reservoir_Array', DAforcing.get_value('usgs_reservoir_Array'))

    # initialized_model.set_value('datesSecondsArray_reservoir_usace', DAforcing.get_value('datesSecondsArray_reservoir_usace'))
    # initialized_model.set_value('nDates_reservoir_usace', DAforcing.get_value('nDates_reservoir_usace'))
    # initialized_model.set_value('stationArray_reservoir_usace', DAforcing.get_value('stationArray_reservoir_usace'))
    # initialized_model.set_value('stationStringLengthArray_reservoir_usace', DAforcing.get_value('stationStringLengthArray_reservoir_usace'))
    # initialized_model.set_value('nStations_reservoir_usace', DAforcing.get_value('nStations_reservoir_usace'))
    # initialized_model.set_value('usace_reservoir_Array', DAforcing.get_value('usace_reservoir_Array'))

    initialized_model.set_value('rfc_da_timestep', DAforcing.get_value('rfc_da_timestep'))
    initialized_model.set_value('rfc_totalCounts', DAforcing.get_value('rfc_totalCounts'))
    initialized_model.set_value('rfc_synthetic_values', DAforcing.get_value('rfc_synthetic_values'))
    initialized_model.set_value('rfc_discharges', DAforcing.get_value('rfc_discharges'))
    initialized_model.set_value('rfc_timeseries_idx', DAforcing.get_value('rfc_timeseries_idx'))
    initialized_model.set_value('rfc_use_rfc', DAforcing.get_value('rfc_use_rfc'))
    initialized_model.set_value('rfc_Datetime', DAforcing.get_value('rfc_Datetime'))
    initialized_model.set_value('rfc_timeSteps', DAforcing.get_value('rfc_timeSteps'))
    initialized_model.set_value('rfc_StationId_array', DAforcing.get_value('rfc_StationId_array'))
    initialized_model.set_value('rfc_StationId_stringLengths', DAforcing.get_value('rfc_StationId_stringLengths'))
    initialized_model.set_value('rfc_List_array', DAforcing.get_value('rfc_List_array'))
    initialized_model.set_value('rfc_List_stringLengths', DAforcing.get_value('rfc_List_stringLengths') ) 

    initialized_model.set_value('waterbody_df', DAforcing.get_value('waterbody_df'))
    initialized_model.set_value('waterbody_df_index', DAforcing.get_value('waterbody_df_ids'))
    initialized_model.set_value('lastobs_df', DAforcing.get_value('lastobs_df'))
    initialized_model.set_value('lastobs_df_index', DAforcing.get_value('lastobs_df_ids'))

    initialized_model.set_value('q0', DAforcing.get_value('q0'))
    initialized_model.set_value('q0_index', DAforcing.get_value('q0_ids'))
    initialized_model.set_value('t0', DAforcing.get_value('t0'))

    initialized_model.set_value('coastal_boundary_depth_df', pd.DataFrame())
    initialized_model.set_value('unrefactored_topobathy_df', pd.DataFrame())

    # Read forcing data first. This info will eventually be provided by model engine.
    timeseries_start = DAforcing._model._t0
    timeseries_end = timeseries_start + timedelta(hours=18 - 1)
    delta = timedelta(hours=1)
    forcing_files = []
    base_path = sample_config.parent
    while timeseries_start <= timeseries_end:
        forcing_files.append(base_path / 'channel_forcing' / f"{timeseries_start.strftime('%Y%m%d%H%M')}.CHRTOUT_DOMAIN1.csv")
        timeseries_start += delta

    # Loop through hourly timesteps calling update_until
    # print('Begin routing...')
    # for hr in range(18):
    #     # Retrieve forcing data
    #     forcing_vals = pd.read_csv(forcing_files[hr]).set_index('feature_id')
    #     # Get rid of duplicate nexus points:
    #     idx = forcing_vals.index.duplicated()
    #     forcing_vals = forcing_vals.loc[~idx]

    #     # Full network
    #     # Set dynamic values
    #     initialized_model.set_value('land_surface_water_source__volume_flow_rate', np.array(forcing_vals.iloc[:,0]))
    #     initialized_model.set_value('land_surface_water_source__id', np.array(forcing_vals.index))

    #     # Set duration to run model for
    #     nts = 12
    #     n = nts*300
        
    #     # Run our model
    #     initialized_model.update_until(n)

