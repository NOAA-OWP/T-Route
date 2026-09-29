import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr


def _flatten_array(dataFrame, dataType):

    # convert to numpy array first
    array1 = dataFrame.to_numpy(copy=True, dtype=dataType)
    # flatten it
    array2 = array1.reshape(-1)

    return array2


def _stringsToBMI(stringList):

    # Generate ASCII array for list of strings (one giant 1D ndarray with ASCII encodings
    # concatenated together), aided by an array of how many strings each string consists 
    #
    nEntries = len(stringList)
    # define array defining string length for each station ID
    stringLengthArray = np.empty(nEntries, dtype=np.int64)
    stringLengthArray[0] = len(stringList[0])

    # start big array with first string
    stringArray = np.empty(stringLengthArray[0], dtype=np.int64)

    # init some aux. variables
    firstIter = True
    index = 0
    # iterate through all stations
    for stringRead in stringList:
        # capture string length of present station
        stringLengthArray[index] = len(stringRead)
        index += 1        
        # convert string to ASCII code
        charArray = np.array(stringRead, dtype='c')
        addArray = np.array([ord(ch) for ch in charArray])        
        # build up big array
        if (firstIter):
            stringArray = addArray
            firstIter = False
        else:
            stringArray = np.concatenate([stringArray, addArray])    

    return (stringArray, stringLengthArray)


def _time_from_df(dataFrame, timeBase):

    # get columns (date-time), extract as list at first
    datesList = (dataFrame.columns).tolist()
    # then subtract timebase
    datesListSeconds = [int((d-timeBase).total_seconds()) for d in datesList]
    nDates = len(datesListSeconds)
    datesSecondsArray = np.array(datesListSeconds)

    return datesSecondsArray, nDates


def _time_stations_from_df(dataFrame, timeBase):

    # get columns (date-time), extract as list at first
    datesList = (dataFrame.columns).tolist()
    # then subtract timebase
    datesListSeconds = [int((d-timeBase).total_seconds()) for d in datesList]
    nDates = len(datesListSeconds)
    datesSecondsArray = np.array(datesListSeconds)

    # get station IDs, as list
    stations = (dataFrame).index.tolist()
    stations = [str(x) for x in stations]
    nStations = len(stations)

    # build station array (one giant 1D ndarray with ASCII encodings
    # concatenated together), aided by an array of how many strings each
    # station ID consists of - completely USGS, USACE, etc agnostic 
    #
    # define array defining string length for each station ID
    stationStringLengthArray = np.empty(nStations, dtype=np.int64)
    stationStringLengthArray[0] = len(stations[0])

    # start big array with ID for first station
    stationArray = np.empty(stationStringLengthArray[0], dtype=np.int64)

    # init some aux. variables
    firstIter = True
    stationIndex = 0
    # iterate through all stations
    for station in stations:
        # capture string length of present station
        stationStringLengthArray[stationIndex] = len(station)
        stationIndex += 1        
        # convert string to ASCII code
        charArray = np.array(station, dtype='c')
        addArray = np.array([ord(ch) for ch in charArray])        
        # build up big array
        if (firstIter):
            stationArray = addArray
            firstIter = False
        else:
            stationArray = np.concatenate([stationArray, addArray])

    return datesSecondsArray, nDates, stationArray, stationStringLengthArray, nStations


def _bmi_disassemble_rfc_timeseries (dataFrame, timeRef):

    # Extract individual columns and convert to ndarrays

    # Column entries that are already float or int:
    rfc_da_timestep = dataFrame["da_timestep"].to_numpy(dtype=int, copy=True)
    rfc_totalCounts = dataFrame["totalCounts"].to_numpy(dtype=int, copy=True)
    rfc_synthetic_values = dataFrame["synthetic_values"].to_numpy(dtype=np.float32, copy=True)
    rfc_discharges = dataFrame["discharges"].to_numpy(dtype=np.float32, copy=True)
    rfc_timeseries_idx = dataFrame["timeseries_idx"].to_numpy(dtype=int, copy=True)

    # Convert Boolean entry to in (use_rfc)
    # first, conversion to list
    boolList = (dataFrame["use_rfc"]).tolist()
    totalDim = len(boolList)
    rfc_use_rfc = np.zeros(totalDim)
    rfc_use_rfc = [int(d) for d in boolList]

    # Datetime and timedelta entries
    # Get into list
    DatetimeList = (dataFrame["Datetime"]).tolist()
    timeStepsList = (dataFrame["timeSteps"]).tolist()
    # Convert to seconds (relative to timeRef for the absolute time stamp)
    DatetimeListSeconds = [int((d-timeRef).total_seconds()) for d in DatetimeList]
    timeStepsListSeconds = [int(d.total_seconds()) for d in timeStepsList]
    # Convert into numpy array
    rfc_Datetime = np.array(DatetimeListSeconds)
    rfc_timeSteps = np.array(timeStepsListSeconds)
    # The horizon is measured from the issue, so it has to cross the transport too.
    rfc_issue_time = np.array(
        [int((d - timeRef).total_seconds()) for d in dataFrame["issue_time"].tolist()]
    )
  
    # String entries: two arrays:
    #   1st array: conversion of entries into ASCII code
    #   2nd array: array with length of entries
    # Extract entries as list
    stationId_List = (dataFrame["stationId"]).tolist()
    file_List = (dataFrame["file"]).tolist()    
    (rfc_StationId_array, rfc_StationId_stringLengths) = _stringsToBMI(stationId_List)
    (rfc_List_array, rfc_List_stringLengths) = _stringsToBMI(file_List)   

    return (rfc_da_timestep, rfc_totalCounts, rfc_synthetic_values, rfc_discharges, \
            rfc_timeseries_idx, rfc_use_rfc, rfc_Datetime, rfc_timeSteps, rfc_StationId_array, \
            rfc_StationId_stringLengths, rfc_List_array, rfc_List_stringLengths, \
            rfc_issue_time)


def _bmi_disassemble_lite_restart (dataFrame, dataType):

    # These are the q0- and waterbody dataframes for "lite-restart"
    # index is int64, columns are strings, and array proper is float32 or float64

    # get columns and convert to BMI compliant arrays
    columnList = (dataFrame.columns).tolist()
    nCol = len(columnList)
    (columnArray, columnLengthArray) = _stringsToBMI(columnList)

    # get index array (already BMI compliant; int64)
    indexArray = (dataFrame.index).to_numpy(dtype=np.int64, copy=True)
    nIndex = len(indexArray)

    mainArray = _flatten_array(dataFrame, dataType)

    return (columnArray, columnLengthArray, nCol, indexArray, nIndex, mainArray)


def _bmi_disassemble_lastObs (lastobs_df):
                
    # Column entries that are already float or int:
    timeSinceArray = lastobs_df["time_since_lastobs"].to_numpy(dtype=np.float64, copy=True)
    lastDischargeArray = lastobs_df["lastobs_discharge"].to_numpy(dtype=np.float64, copy=True)

    # Gage IDs: string entry
    gageList = (lastobs_df["gages"]).tolist()
    (gageArray, gageStringLengthArray) = _stringsToBMI(gageList)  

    return(gageArray, gageStringLengthArray, timeSinceArray, lastDischargeArray)


