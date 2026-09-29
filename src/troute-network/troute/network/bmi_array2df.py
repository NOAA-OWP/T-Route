import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr


def _unflatten_array(array_1D,nx,ny):

    # reshape numpy array
    array_2D = array_1D.reshape(ny,nx)

    # convert to data frame
    df_raw = pd.DataFrame(array_2D)

    return df_raw


def _time_retrieve_from_arrays(dataFrame, timeBase, datesSecondsArrays, \
                               timeAxisName, freqString):

    # convert array to list at first
    datesList = (datesSecondsArrays).tolist()

    # add timebase back in, and convert to datetime format
    datesList = [(timedelta(seconds=d)+timeBase) for d in datesList]

    # convert to DatetimeIndex
    if (freqString == 'None'):

        dateTimeIndex = pd.DatetimeIndex(datesList, name=timeAxisName)

    else:

        dateTimeIndex = pd.DatetimeIndex(datesList, name=timeAxisName, freq=freqString)

    # transpose dataframe
    dataFrameTranspose = dataFrame.T
    # add time axis as index on transposed dataframe (time is in rows)
    dataFrameTranspose = dataFrameTranspose.set_index(dateTimeIndex)
    # revert to original (time is in columns)
    dataFrameWithTime = dataFrameTranspose.T

    return dataFrameWithTime


def _stations_retrieve_from_arrays(dataFrame, stationsArray, stationStringLengthArray, \
                               stationAxisName):

    # first create station outputs as list
    stationsList = []

    # mark place in global stationsArray
    i=0

    for stringLength in stationStringLengthArray:

        # extract small array with ASCII codes of one station ID
        stationID_asArray = stationsArray[i:i+stringLength]
        # increment global pointer along array
        i=i+stringLength
        # convert to list of characters
        firstRun = True
        for asciiCode in stationID_asArray:
            # decode Ascii back to character and stick it to the end of the list
            chrAdd = chr(asciiCode)
            if firstRun == True:
                stationID_asString = chrAdd
                firstRun = False
            else:
                stationID_asString += chrAdd

        # done with one station ID
        stationsList.append(stationID_asString)

    # construct index    
    index = pd.Index(stationsList, name=stationAxisName, dtype=object)

    dataFrame.index = (index)

    return dataFrame


def _BMI_toStrings(stringArray, stringLengthArray):

    # Reassemble list of strings from main array with ASCII encodings and array of 
    # how many characters each string consists 
    

    # first create station outputs as list
    stringList = []

    # mark place in global stationsArray
    i=0

    for stringLength in stringLengthArray:

        # extract small array with ASCII codes of one station ID
        string_asArray = stringArray[i:i+stringLength]
        # increment global pointer along array
        i=i+stringLength
        # convert to list of characters
        firstRun = True
        for asciiCode in string_asArray:
            # decode Ascii back to character and stick it to the end of the list
            chrAdd = chr(asciiCode)
            if firstRun == True:
                string_asString = chrAdd
                firstRun = False
            else:
                string_asString += chrAdd

        # done with one station ID
        stringList.append(string_asString)

    return stringList


def _bmi_reassemble_rfc_timeseries (rfc_da_timestep, rfc_totalCounts, \
                rfc_synthetic_values, rfc_discharges, rfc_timeseries_idx, \
                rfc_use_rfc, rfc_Datetime, rfc_timeSteps, rfc_StationId_array, \
                rfc_StationId_stringLengths, rfc_List_array, \
                rfc_List_stringLengths, timeRef, rfc_issue_time=None):

    # Create empty dataframe with appropriate column names
    columnList = ['stationId','discharges','synthetic_values','totalCounts',\
                  'timeSteps','Datetime','timeseries_idx','file','use_rfc','da_timestep',\
                  'issue_time']
    
    # Build up dataframe
    dataFrame = pd.DataFrame()
    for col in columnList:

        if (col == 'stationId'):
            addedCol = _BMI_toStrings(rfc_StationId_array, rfc_StationId_stringLengths)
        elif (col == 'discharges'):
            addedCol = rfc_discharges
        elif (col == 'synthetic_values'):
            addedCol = rfc_synthetic_values
        elif (col == 'totalCounts'):
            addedCol = rfc_totalCounts
        elif (col == 'timeSteps'):
            datesList = rfc_timeSteps.tolist()
            addedCol = [(timedelta(seconds=d)) for d in datesList]
        elif (col == 'Datetime'):
            datesList = rfc_Datetime.tolist()
            addedCol = [(timedelta(seconds=d)+timeRef) for d in datesList]       
        elif (col == 'timeseries_idx'):
            addedCol = rfc_timeseries_idx
        elif (col == 'file'):
            addedCol = _BMI_toStrings(rfc_List_array, rfc_List_stringLengths)        
        elif (col == 'use_rfc'):
            addedCol = [bool(d) for d in rfc_use_rfc]
        elif (col == 'da_timestep'):
            addedCol = rfc_da_timestep
        elif (col == 'issue_time'):
            if rfc_issue_time is None:
                continue
            addedCol = [(timedelta(seconds=d)+timeRef) for d in rfc_issue_time.tolist()]
        # add the selected column
        dataFrame[col] = addedCol

    return dataFrame


def _bmi_reassemble_lite_restart (columnArray, columnLengthArray, nCol, indexArray,\
                                    nIndex, Array):
    
    # reverse flattening of array proper
    df_raw = _unflatten_array(Array,nCol,nIndex)

    # get column names back as strings
    colList = _BMI_toStrings(columnArray, columnLengthArray)
    colListAttach = pd.Index(colList, dtype=object)

    # transpose dataframe
    df_raw_transpose = df_raw.T
    # add column axis as index on transposed dataframe
    df_raw_transpose.index = colListAttach
    # revert transpose
    df_raw_withCol = df_raw_transpose.T

    # add index
    index = pd.Index(indexArray, dtype=np.int64)
    df_raw_withCol.index = index 

    df_complete = df_raw_withCol

    return df_complete


def _bmi_reassemble_lastObs (gageArray, gageStringLengthArray, \
                             timeSinceArray, lastDischargeArray):

    # Create empty dataframe with appropriate column names
    columnList = ['gages','time_since_lastobs','lastobs_discharge']
    
    # Build up dataframe
    dataFrame = pd.DataFrame()
    for col in columnList:

        if (col == 'gages'):
            addedCol = _BMI_toStrings(gageArray, gageStringLengthArray)
        elif (col == 'time_since_lastobs'):
            addedCol = timeSinceArray
        elif (col == 'lastobs_discharge'):
            addedCol = lastDischargeArray
  
        # add the selected column
        dataFrame[col] = addedCol

    return dataFrame
