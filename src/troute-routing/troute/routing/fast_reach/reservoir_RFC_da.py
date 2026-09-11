import os
import xarray as xr
import datetime

def _add_hours(date, hours):
    '''
    Compute a new date after adding hours to a current date
    
    Arguments
    ---------
    date (str): "%Y-%m-%d_%H:%M:%S"
    hours (int)
    
    Returns
    -------
    new_date (str): New date after adding hours to a current date
    
    Notes
    -----
    '''
    dt = datetime.datetime.strptime(date,"%Y-%m-%d_%H:%M:%S") 
    formatted_date = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    td = datetime.timedelta(hours=hours)
    new_date = formatted_date + td
    new_date = new_date.strftime("%Y-%m-%d_%H")
    return new_date

def _search_RFCTimeSeries_files_backward_from_offset_hours(offset_date,
                                                           max_rfc_timeseries_file_search_hours,
                                                           rfc_gage_id,
                                                           rfc_timeseries_folder):    
    '''
    Find a RFCTimeSeries.ncdf moving backing from offset_date by an hourly step until 
    the issue time of a RFCTimeSeries.ncdf is matched with the newly updated offset_date within 
    Arguments
    ---------
    offset_date (str): Offset date in the future from the model start time, after offset by a given offset hours
    max_rfc_timeseries_file_search_hours (int): max time period to search backward in time for a RFC file 
    rfc_timeseries_folder (str): folder path for RFCTimeSeries.ncdf files
    
    Returns
    -------
    rfc_timeseries_offset_file (str):  Selected RFCTimeSeries.ncdf file to be used for DA
    lookback_hours (int): Difference in hours between the initial offset_date and issue date of returned rfc_timeseries_offset_file
    
    Notes
    -----
    '''
    new_rfc_timeseries_offset_date = offset_date    
    for hour in range(0, max_rfc_timeseries_file_search_hours):
        
        rfc_timeseries_offset_file = new_rfc_timeseries_offset_date+"."+"60min"+"."+rfc_gage_id+"."+"RFCTimeSeries.ncdf"
        file_path= os.path.join(rfc_timeseries_folder, rfc_timeseries_offset_file)
        
        if os.path.isfile(file_path):
            lookback_hours = hour
            break
        else:
            old_date = new_rfc_timeseries_offset_date+":00:00"        
            new_rfc_timeseries_offset_date = _add_hours(old_date, -1)
    return rfc_timeseries_offset_file, lookback_hours

def _timeseries_idx_updatetime_totalcounts(lookback_hours,
                                           rfc_da_df,
                                           rfc_timeseries_offset_hours):
    '''
    Arguments
    ---------
    lookback_hours (str)             : Difference in hours between the initial offset_date and 
                                       issue date of returned rfc_timeseries_offset_file
    rfc_da_df (dataframe)            : DataFrame of found RFCTimeSeries.ncdf 
    rfc_timeseries_offset_hours (int): Offset hours ahead of model start date from where searching for 
                                       RFCTimeSeries.ncdf files starts backward in time
    
    Returns
    -------
    timeseries_idx (int)  
    timeseries_update_time (int)
    time_step_seconds (int)
    total_counts (int)                
    
    Notes
    -----
    '''        
    # compute initial value of timeseries_idx
    lookback_seconds = lookback_hours*3600
    string_with_time = str(rfc_da_df.timeSteps[0][0])
    time_step = string_with_time[-8:]
    dt = datetime.datetime.strptime(time_step,"%H:%M:%S") 
    time_step_seconds = dt.hour*3600
    observed_counts = rfc_da_df.observedCounts[0][0]
    timeseries_idx = int(lookback_seconds / time_step_seconds + observed_counts - rfc_timeseries_offset_hours)
        
    # compute initial value of timeseries_update_time
    update_offset_seconds = lookback_seconds % time_step_seconds
    timeseries_update_time = time_step_seconds - update_offset_seconds
    
    #total count of observation + forecast in RFC time series file
    total_counts = rfc_da_df.totalCounts[0][0]

    return timeseries_idx, timeseries_update_time, time_step_seconds, total_counts

def _validate_RFC_data(lake_number, 
                       time_series, 
                       synthetic, 
                       rfc_timeseries_folder, 
                       rfc_timeseries_file,
                       routing_period,
                       from_files=True):
    
    use_RFC = True
    file_path= os.path.join(rfc_timeseries_folder, rfc_timeseries_file)
    
    if all(synthetic)==1:
        use_RFC = False
        print(f"WARNING: RFC Forecast Time Series discharges for reservoir {lake_number} \
              are all synthetic. \
              This reservoir will use level pool calculations instead.")
    elif any(time_series)<0:
        use_RFC = False
        print(f"WARNING: RFC Forecast Time Series discharges for reservoir {lake_number} \
              contains missing or negative values. \
              This reservoir will use level pool calculations instead.")
    elif any(time_series)>=90000:
        use_RFC = False
        print(f"WARNING: RFC Forecast Time Series discharges for reservoir {lake_number} \
              contain one or more values greater than or equal to 90,000 Cubic Meters per \
              Second (twice the Mississippi River historical peak flow). \
              This reservoir will use level pool calculations instead.")
    elif from_files and (os.path.isfile(file_path)==False):
        use_RFC = False
        print(f"WARNING: RFC Forecast Time Series file for reservoir {lake_number} \
              does not exist. \
              This reservoir will use level pool calculations instead.")
    elif routing_period>3600:
        use_RFC = False
        print(f"WARNING: The routing period is greater than one hour. The RFC DA for {lake_number} \
              cannot be utilized as the result. \
              This reservoir will use level pool calculations instead.")        
    
    return use_RFC

def preprocess_RFC_data(model_start_date,
                        rfc_timeseries_offset_hours,
                        rfc_gage_id,
                        rfc_timeseries_folder,
                        lake_number,
                        routing_period):
    # compute a new date after adding hours to a current date
    rfc_timeseries_offset_date = _add_hours(model_start_date, rfc_timeseries_offset_hours)
    
    # search for RFCTimeSeries.ncdf file used for DA and lookback hours from offset date
    rfc_timeseries_file, lookback_hours = _search_RFCTimeSeries_files_backward_from_offset_hours(
                                                                                rfc_timeseries_offset_date, 
                                                                                28,
                                                                                rfc_gage_id,
                                                                                rfc_timeseries_folder)
    
    file_path= os.path.join(rfc_timeseries_folder, rfc_timeseries_file)
    if os.path.isfile(file_path):
        # Two xarray compatibility guards, both needed:
        #   drop_variables: queryTime carries the non CF compliant units
        #     "seconds since 1970-01-01 00:00:00 local TZ". xarray tolerated it
        #     until 2025 and now raises on open. Nothing here reads queryTime.
        #   decode_timedelta: xarray stopped decoding a bare "seconds" unit into
        #     timedelta64 by default. timeSteps is one, and
        #     _timeseries_idx_updatetime_totalcounts parses str(timeSteps)[-8:] as
        #     "%H:%M:%S", which needs the timedelta form and not a raw int.
        rfc_da_df = xr.open_dataset(
            rfc_timeseries_folder + rfc_timeseries_file,
            drop_variables="queryTime",
            decode_timedelta=True,
        ).to_dataframe()
        timeseries_discharges = rfc_da_df.discharges.to_numpy()
        synthetic             = rfc_da_df.synthetic_values.to_numpy()
        # compute initial values of time_series_index, time_series_update_time, and total counts of observed+forecated
        (
            timeseries_idx, 
            timeseries_update_time, 
            time_step_seconds, 
            total_counts) = _timeseries_idx_updatetime_totalcounts(lookback_hours,
                                                                   rfc_da_df, 
                                                                   rfc_timeseries_offset_hours)
        
    else:
        timeseries_discharges = 99999
        synthetic = 1
    
    # check if conditions are met for using RFC DA.
    use_RFC = _validate_RFC_data(lake_number, 
                                 timeseries_discharges, 
                                 synthetic, 
                                 rfc_timeseries_folder, 
                                 rfc_timeseries_file, 
                                 routing_period)
    
    return (use_RFC, 
            timeseries_discharges, 
            timeseries_idx, 
            timeseries_update_time, 
            time_step_seconds, 
            total_counts,
            rfc_timeseries_file)

def reservoir_RFC_da(use_RFC, time_series, timeseries_idx, total_counts, routing_period, current_time,
                     update_time, DA_time_step, rfc_forecast_persist_seconds, reservoir_type, inflow, 
                     water_elevation, levelpool_outflow, levelpool_water_elevation, lake_area, 
                     max_water_elevation, rfc_file):
    '''
    Run RFC DA reservoir module via BMI
    
    Arguments
    ---------
    use_RFC (boolean): Did RFC data pass validation checks
    time_series (numpy array): 1D array of RFC outflow values
    timeseries_idx (int): Index of current time_series value
    routing_period (int): simulation time step of channel routing or time step of inflow to reservoir (sec)
    current_time (int): Time step of RFC DA that is initially set to zero but increase as DA evolves (sec)
    update_time (int): Time at which timeseries_idx should be advanced (sec)
    DA_time_step (int): Time step of time_series data (sec)
    rfc_forecast_persist_seconds (int): max number of seconds that RFC-supplied forecast will be used/persisted in simulation
    reservoir_type (int): reservoir type
    inflow(numpy array): inflow to waterbody [cms] 
    water_elevation (float): water surface el., previous timestep (m)
    levelpool_outflow (float): levelpool simulated outflow (cms) 
    levelpool_water_elevation (float): levelpool simulated water elevation (m)
    lake_area (float): waterbody surface area computed from level pool (km2)
    max_water_elevation (float): max waterbody depth (m)
    rfc_file (str): File name of RFC file
    
    Returns
    -------
    outflow, 
    new_water_elevation, 
    update_time, 
    timeseries_idx, 
    dynamic_reservoir_type, 
    assimilated_value,
    source_file
    
    Notes
    -----
    ''' 
    if use_RFC and (current_time)<=rfc_forecast_persist_seconds:
        if (current_time) >= update_time and timeseries_idx<total_counts:
            # Advance update_time to the next timestep and time_series_idx to next index
            update_time += DA_time_step
            timeseries_idx += 1

        # If reservoir_type is 4 for CONUS RFC reservoirs
        if reservoir_type==4:
            # Set outflow to corresponding discharge from array
            outflow = time_series[timeseries_idx]

        # Else reservoir_type 5 for for Alaska RFC glacier outflows
        else:
            # Set outflow to sum inflow and corresponding discharge from array
            outflow = inflow + time_series[timeseries_idx]
        
        # Update water elevation
        new_water_elevation = water_elevation + ((inflow - outflow)/lake_area) * routing_period

        # Ensure that the water elevation is within the minimum and maximum elevation
        if new_water_elevation < 0.0:
            new_water_elevation = 0.0

        elif new_water_elevation > max_water_elevation:
            new_water_elevation = max_water_elevation
        
        # Set dynamic_reservoir_type to RFC Forecasts Type
        dynamic_reservoir_type = reservoir_type

        # Set the assimilated_value to corresponding discharge from array
        assimilated_value = time_series[timeseries_idx]

        # Set the assimilated_source_file to empty string
        assimilated_source_file = rfc_file

        # Check for outflows less than 0 and cycle backwards in the array until a
        # non-negative value is found. If all previous values are negative, then
        # use level pool outflow.
        if outflow < 0:
            missing_outflow_index = timeseries_idx

            while outflow < 0 and missing_outflow_index > 1:
                missing_outflow_index = missing_outflow_index - 1
                outflow = time_series[missing_outflow_index]
            
            if outflow < 0:
                # If reservoir_type is 4 for CONUS RFC reservoirs
                if reservoir_type == 4:
                    outflow = levelpool_outflow

                # Else reservoir_type 5 for for Alaska RFC glacier outflows
                else:
                    outflow = inflow

                # Update water elevation to levelpool water elevation
                new_water_elevation = levelpool_water_elevation

                # Set dynamic_reservoir_type to levelpool type
                dynamic_reservoir_type = 1

                # Set the assimilated_value to sentinel, -9999.0
                assimilated_value = -9999.0

                # Set the assimilated_source_file to empty string
                assimilated_source_file = ""

    else:
        # If reservoir_type is 4 for CONUS RFC reservoirs
        if reservoir_type == 4:
            outflow = levelpool_outflow

        # Else reservoir_type 5 for for Alaska RFC glacier outflows
        else:
            outflow = inflow

        # Update water elevation to levelpool water elevation
        new_water_elevation = levelpool_water_elevation

        # Set dynamic_reservoir_type to levelpool type
        dynamic_reservoir_type = 1

        # Set the assimilated_value to sentinel, -9999.0
        assimilated_value = -9999.0

        # Set the assimilated_source_file to empty string
        assimilated_source_file = ""
    
    return outflow, new_water_elevation, update_time, timeseries_idx, dynamic_reservoir_type, assimilated_value, assimilated_source_file

