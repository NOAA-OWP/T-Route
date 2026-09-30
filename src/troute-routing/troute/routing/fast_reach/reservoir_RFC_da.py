import glob
import logging
import os
import re
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
import datetime
from numpy.typing import NDArray

LOG = logging.getLogger("TROUTE")


class RFCTimeSeries(NamedTuple):
    """One RFC TimeSeries file, normalized.

    The file has no time variable; the axis is rebuilt from ``sliceStartTimeUTC`` and
    ``sliceTimeResolutionMinutes``.
    """

    station_id: str
    datetimes: pd.DatetimeIndex
    discharges: NDArray[np.float64]
    synthetic: NDArray[np.float64]
    total_counts: int
    observed_counts: int
    timestep_seconds: int
    issue_time: pd.Timestamp


# Twice the Mississippi's historical peak. Nothing downstream compensates for a
# value above it, so it disqualifies a forecast wherever the kernel could read it.
ABSURD_DISCHARGE_CMS = 90000

_FILENAME_CADENCE = re.compile(r"\.(\d+)min\.")
# The production RFC ingestion naming: <issue>.<cadence>min.<gage>.RFCTimeSeries.ncdf.
# Matched with fullmatch, since `$` would also accept a trailing newline.
_RFC_FILENAME = re.compile(
    r"\d{4}-\d{2}-\d{2}_\d{2}\.\d+min\.[^.]+\.RFCTimeSeries\.ncdf"
)

# Producers disagree: "seconds" from one, "hours" from the NHF generator.
# Unlabeled files are seconds.
_STEP_UNIT_SECONDS = {
    "s": 1, "sec": 1, "second": 1, "seconds": 1,
    "min": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
}


def read_rfc_timeseries(path: str) -> RFCTimeSeries:
    """Read one RFC TimeSeries file into a normalized record.

    Refuses any disagreement between the three places the cadence is recorded: the
    filename, ``sliceTimeResolutionMinutes`` and ``timeSteps``. Everything downstream
    advances one index per cadence step, so a wrong one assimilates at the wrong rate.
    """
    name = os.path.basename(path)
    # queryTime's units are non-CF and raise on open. timeSteps stays raw: decoded as a
    # timedelta its units move to .encoding and 3600 s reads as 3600 ns.
    with xr.open_dataset(path, drop_variables="queryTime", decode_timedelta=False) as ds:
        # A file can hold several issues; only the newest one is wanted.
        n_series = int(ds.sizes.get("nseries", 1))
        newest = int(ds.attrs.get("newest_forecast", n_series - 1))
        if not 0 <= newest < n_series:
            msg = (
                f"reservoir RFC DA: {name} declares newest_forecast={newest} but holds "
                f"{n_series} series, so the current forecast cannot be identified."
            )
            raise ValueError(msg)

        def _pick(var: str) -> NDArray[Any]:
            """The newest series of *var*, or the whole thing if it has no series axis."""
            data = ds[var]
            return np.asarray((data.isel(nseries=newest) if "nseries" in data.dims else data).values)

        slice_start = datetime.datetime.strptime(
            str(ds.attrs["sliceStartTimeUTC"]), "%Y-%m-%d_%H:%M:%S"
        )
        declared = float(ds.attrs["sliceTimeResolutionMinutes"])
        if not declared.is_integer():
            msg = (
                f"reservoir RFC DA: {name} declares a {declared} min cadence. Cadences "
                "are whole minutes; a fractional one would be truncated and read at the "
                "wrong rate."
            )
            raise ValueError(msg)
        attr_minutes = int(declared)
        # The forecast's own age, which is what the persistence horizon is measured
        # from. Read from the same series as the discharges: a file can hold several.
        raw_issue = _pick("issueTimeUTC").ravel()[0]
        issue_text = (
            raw_issue.decode("utf-8") if isinstance(raw_issue, bytes) else str(raw_issue)
        ).strip()
        issue_time = pd.Timestamp(
            datetime.datetime.strptime(issue_text, "%Y-%m-%d_%H:%M:%S")
        )
        raw_id = _pick("stationId")
        discharges = _pick("discharges").astype(np.float64).ravel()
        synthetic = _pick("synthetic_values").astype(np.float64).ravel()
        total_counts = int(_pick("totalCounts").ravel()[0])
        observed_counts = int(_pick("observedCounts").ravel()[0])
        # Exact seconds, not floored minutes: 3659 s would floor to 60 and pass.
        step_units = str(ds["timeSteps"].attrs.get("units", "seconds")).strip().lower()
        step_seconds = float(_pick("timeSteps").ravel()[0])
    if step_units not in _STEP_UNIT_SECONDS:
        # Refuse rather than guess: an unknown label compares two units below.
        msg = (
            f"reservoir RFC DA: {name} records timeSteps in {step_units!r}, which this "
            f"reader does not recognize (known: {sorted(_STEP_UNIT_SECONDS)}), so the "
            "cadence cannot be cross-checked."
        )
        raise ValueError(msg)
    step_seconds *= _STEP_UNIT_SECONDS[step_units]

    # stationId is fixed-width bytes in some vintages, a char array in others.
    flat = np.asarray(raw_id).ravel()
    station = b"".join(
        v if isinstance(v, bytes) else str(v).encode() for v in flat.tolist()
    ).decode("utf-8").strip()

    from_name = _FILENAME_CADENCE.search(name)
    if from_name is None:
        msg = (
            f"reservoir RFC DA: {name} carries no cadence token in its filename, so the "
            "cadence cannot be cross-checked against the file's own attributes."
        )
        raise ValueError(msg)
    name_minutes = int(from_name.group(1))
    if attr_minutes <= 0 or name_minutes != attr_minutes or step_seconds != attr_minutes * 60:
        msg = (
            f"reservoir RFC DA: {name} disagrees about its own cadence -- filename says "
            f"{name_minutes} min, sliceTimeResolutionMinutes says {attr_minutes} min, "
            f"timeSteps says {step_seconds} s. The forecast index advances one step per "
            "cadence, so reading it from the wrong place assimilates at the wrong rate."
        )
        raise ValueError(msg)
    if 60 % attr_minutes:
        # The ingestion's rule (RFCHelper.makeAllTimeSeries). A cadence that divides 60
        # puts a sample on every hour, which is what lets an hourly run line up with it.
        msg = (
            f"reservoir RFC DA: {name} has a {attr_minutes} min cadence, which does not "
            "divide 60. The forecast retrieval only writes cadences that divide 60 with "
            "no remainder, and a run can only align with one that does."
        )
        raise ValueError(msg)

    return RFCTimeSeries(
        station_id=station,
        datetimes=pd.date_range(slice_start, periods=len(discharges), freq=f"{attr_minutes}min"),
        discharges=discharges,
        synthetic=synthetic,
        total_counts=total_counts,
        observed_counts=observed_counts,
        timestep_seconds=attr_minutes * 60,
        issue_time=issue_time,
    )


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

def _covers(path: str, t0: "datetime.datetime") -> bool:
    """Does this forecast file span t0?

    A read error propagates: the folder reader treats an unreadable forecast as fatal,
    and skipping it here would quietly assimilate an older one instead.
    """
    return pd.Timestamp(t0) in read_rfc_timeseries(path).datetimes


def _search_RFCTimeSeries_files_backward_from_offset_hours(offset_date,
                                                           max_rfc_timeseries_file_search_hours,
                                                           rfc_gage_id,
                                                           rfc_timeseries_folder,
                                                           t0=None):    
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
    # Bound before the loop: the return below runs even when nothing matches.
    lookback_hours = None
    # Inclusive far endpoint, matching the folder reader.
    for hour in range(0, max_rfc_timeseries_file_search_hours + 1):
        # Glob the cadence rather than assuming 60min, as the folder reader does.
        matches = sorted(glob.glob(os.path.join(
            rfc_timeseries_folder,
            f"{new_rfc_timeseries_offset_date}.*.{rfc_gage_id}.RFCTimeSeries.ncdf",
        )))
        # Stop at the first covering match: a corrupt sibling must not abort a search
        # that has already succeeded.
        covering = next(
            (m for m in matches if t0 is None or _covers(m, t0)), None
        )
        if covering is not None:
            rfc_timeseries_offset_file = os.path.basename(covering)
            lookback_hours = hour
            break
        rfc_timeseries_offset_file = (
            new_rfc_timeseries_offset_date+"."+"60min"+"."+rfc_gage_id+"."+"RFCTimeSeries.ncdf"
        )
        old_date = new_rfc_timeseries_offset_date+":00:00"        
        new_rfc_timeseries_offset_date = _add_hours(old_date, -1)
    if lookback_hours is None and t0 is not None:
        LOG.warning(
            "reservoir RFC DA: no forecast for gage %s in the %d h before %s covers "
            "t0=%s; using level pool instead.",
            rfc_gage_id, max_rfc_timeseries_file_search_hours, offset_date, t0,
        )
    return rfc_timeseries_offset_file, lookback_hours

def _validate_RFC_data(lake_number, 
                       time_series, 
                       synthetic, 
                       rfc_timeseries_folder, 
                       rfc_timeseries_file,
                       routing_period,
                       from_files=True,
                       da_time_step=3600):
    
    use_RFC = True
    file_path= os.path.join(rfc_timeseries_folder, rfc_timeseries_file)
    
    if all(synthetic)==1:
        use_RFC = False
        LOG.warning(
            "reservoir RFC DA: the forecast for reservoir %s is entirely synthetic; "
            "using level pool instead.", lake_number,
        )
    # NaN fails every comparison, so excluding it takes an explicit test.
    elif not any(v == v and v >= 0 for v in time_series):
        use_RFC = False
        LOG.warning(
            "reservoir RFC DA: the forecast for reservoir %s holds no usable value, only "
            "missing or negative ones; using level pool instead.", lake_number,
        )
    # ANY, not all: nothing downstream compensates for an absurd value.
    elif any(v >= ABSURD_DISCHARGE_CMS for v in time_series):
        use_RFC = False
        LOG.warning(
            "reservoir RFC DA: the forecast for reservoir %s reaches %.0f cms, at or "
            "above the 90000 cms limit (twice the Mississippi's historical peak); using "
            "level pool instead.", lake_number, max(v for v in time_series if v == v),
        )
    elif from_files and (os.path.isfile(file_path)==False):
        use_RFC = False
        LOG.warning(
            "reservoir RFC DA: no forecast file at %s for reservoir %s; using level pool "
            "instead.", file_path, lake_number,
        )
    # The kernel advances at most one index per routing step, so a routing period above
    # the cadence slows the forecast by that ratio.
    elif routing_period>da_time_step:
        use_RFC = False
        LOG.warning(
            "reservoir RFC DA: the routing period %ss is longer than the %ss forecast "
            "cadence, so reservoir %s would read the forecast slowed by that ratio. Use "
            "a dt of %ss or less; using level pool instead.",
            routing_period, da_time_step, lake_number, da_time_step,
        )

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
                                                                                rfc_timeseries_folder,
                                                                                t0=datetime.datetime.strptime(
                                                                                    model_start_date, "%Y-%m-%d_%H:%M:%S"))
    
    file_path= os.path.join(rfc_timeseries_folder, rfc_timeseries_file)
    # lookback_hours is None when nothing covered t0, and the name handed back can still
    # exist on disk.
    have_file = lookback_hours is not None and os.path.isfile(file_path)
    if have_file:
        record = read_rfc_timeseries(file_path)
        timeseries_discharges = record.discharges
        synthetic = record.synthetic
        total_counts = record.total_counts
        time_step_seconds = record.timestep_seconds
        # Look t0 up on the axis: deriving it arithmetically breaks sub-hourly cadences.
        t0_stamp = pd.Timestamp(datetime.datetime.strptime(model_start_date, "%Y-%m-%d_%H:%M:%S"))
        timeseries_idx = (
            int(record.datetimes.get_loc(t0_stamp)) if t0_stamp in record.datetimes else -1
        )
        timeseries_update_time = time_step_seconds
        # How much of the forecast's allowance is already spent when the run starts.
        # The horizon is measured from the issue, and this caller's clock starts at t0.
        issue_age_seconds = int((t0_stamp - record.issue_time).total_seconds())
    else:
        # No file found. The return below is unconditional, so every name it hands back
        # must be defined and of the type the caller expects.
        timeseries_discharges = []
        synthetic = []
        timeseries_idx = 0
        timeseries_update_time = 0
        time_step_seconds = 0
        total_counts = 0
        issue_age_seconds = 0

    # check if conditions are met for using RFC DA.
    use_RFC = have_file and timeseries_idx >= 0 and _validate_RFC_data(lake_number, 
                                 timeseries_discharges, 
                                 synthetic, 
                                 rfc_timeseries_folder, 
                                 rfc_timeseries_file,
                                 routing_period,
                                 da_time_step=time_step_seconds)
    
    return (use_RFC, 
            timeseries_discharges, 
            timeseries_idx, 
            timeseries_update_time, 
            time_step_seconds, 
            total_counts,
            rfc_timeseries_file,
            issue_age_seconds)

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
    # total_counts is the inclusive last index, not an array bound: the pivot is
    # truncated at the horizon and the rebase is not.
    last_idx = min(int(total_counts), len(time_series) - 1) if use_RFC else -1

    if use_RFC and (current_time)<=rfc_forecast_persist_seconds and last_idx >= 0:
        if (current_time) >= update_time and timeseries_idx<last_idx:
            # Advance update_time to the next timestep and time_series_idx to next index
            update_time += DA_time_step
            timeseries_idx += 1
        if timeseries_idx > last_idx:
            # Carried in from a previous window past the end of this forecast.
            timeseries_idx = last_idx

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
        # `not (x >= 0)` rather than `x < 0`: NaN fails every comparison.
        if not (outflow >= 0):
            missing_outflow_index = timeseries_idx

            while not (outflow >= 0) and missing_outflow_index > 1:
                missing_outflow_index = missing_outflow_index - 1
                outflow = time_series[missing_outflow_index]

            if outflow >= 0:
                # The elevation above came from the sample just walked past.
                new_water_elevation = water_elevation + (
                    (inflow - outflow) / lake_area
                ) * routing_period
                new_water_elevation = min(max(new_water_elevation, 0.0), max_water_elevation)
                assimilated_value = outflow

            if not (outflow >= 0):
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

