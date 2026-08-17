"""Generate DA (data assimilation) forcing files for NHF test cases.

Supports four persistence DA types (USGS, USACE, USBR, Canada/WSC) written as
15-minute timeslice NetCDF files, and RFC forecast files written as hourly
per-station NetCDF files.

Usage examples
--------------
# Write USGS and USACE timeslices for station list:
python make_da.py --da-type usgs usace \\
    --station-ids 04127885 04159130 \\
    --start-time "2020-01-01 00:00" --end-time "2020-01-02 00:00" \\
    --output-dir reservoir_da/

# Write RFC files for a single station:
python make_da.py --da-type rfc \\
    --station-ids RFC000000000004 \\
    --start-time "2020-01-01 00:00" --end-time "2020-01-01 01:00" \\
    --output-dir reservoir_da/ --discharge 12345
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import dataretrieval.nwis as nwis
import dataretrieval.waterdata as waterdata
import numpy as np
import netCDF4
import pandas as pd
import xarray as xr

# DA type → subdirectory name
DA_SUBDIR = {
    "usgs": "usgs_da",
    "usace": "usace_da",
    "usbr": "usbr_da",
    "canada": "canada_da",
    "rfc": "rfc_da",
}

# DA type → timeslice filename suffix
DA_SUFFIX = {
    "usgs": "usgsTimeSlice",
    "usace": "usaceTimeSlice",
    "usbr": "usbrTimeSlice",
    "canada": "wscTimeSlice",
}

DAType = Literal["usgs", "usace", "usbr", "canada", "rfc"]

_CFS_TO_CMS = 0.028316846592
_USGS_DISCHARGE_PARAM = "00060"  # Discharge, cubic feet per second
_USGS_QUALITY_CODE_MAP = {
    "A": 100,   # Approved
    "P": 55,    # Provisional
    "e": 30,    # Estimated
}
_USGS_QUALITY_DEFAULT = 30


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DAConfig:
    """Parameters describing a single DA forcing generation run."""

    da_type: DAType
    """One of 'usgs', 'usace', 'usbr', 'canada', or 'rfc'."""

    station_ids: list[str]
    """List of station identifiers (padded to 15 chars inside the files)."""

    start_time: str
    """Simulation start time, e.g. '2020-01-01 00:00:00'."""

    end_time: str
    """Simulation end time, e.g. '2020-01-02 00:00:00'."""

    output_dir: Path = field(default_factory=lambda: Path("reservoir_da"))
    """Root directory under which per-type subdirs are created."""

    discharge: float = 1.0
    """Constant discharge value written to every observation (m³/s)."""

    discharge_quality: int = 100
    """Quality flag written alongside every discharge value."""

    # RFC-specific options
    rfc_lookback_hours: int = 28
    """Hours of observed lookback written into each RFC file."""

    rfc_forecast_hours: int = 12
    """Hours of synthetic forecast written into each RFC file."""

    rfc_timeslice_resolution_minutes: int = 60
    """Timeslice resolution reported in RFC file attributes (minutes)."""

    @property
    def start_dt(self) -> pd.Timestamp:
        return pd.to_datetime(self.start_time)

    @property
    def end_dt(self) -> pd.Timestamp:
        return pd.to_datetime(self.end_time)



# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

_ID_CHAR_LEN = 15
_TIME_CHAR_LEN = 19


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")


def _write_timeslice_file(
    out_path: Path,
    time_str: str,
    station_ids_chars: np.ndarray,
    discharge: np.ndarray,
    quality: np.ndarray,
    now_str: str,
) -> None:
    """Write one timeslice NetCDF file for the given timestamp."""
    n = len(discharge)
    time_chars = np.array(
        [list(time_str.ljust(_TIME_CHAR_LEN)) for _ in range(n)], dtype="S1"
    )
    with netCDF4.Dataset(str(out_path), "w", format="NETCDF4") as nc:
        nc.fileUpdateTimeUTC = now_str
        nc.sliceCenterTimeUTC = time_str
        nc.sliceTimeResolutionMinutes = "15"

        nc.createDimension("stationIdInd", n)
        nc.createDimension("stationIdStringLength", _ID_CHAR_LEN)
        nc.createDimension("timeStringLength", _TIME_CHAR_LEN)

        v_id = nc.createVariable("stationId", "S1", ("stationIdInd", "stationIdStringLength"))
        v_id[:] = station_ids_chars

        v_t = nc.createVariable("time", "S1", ("stationIdInd", "timeStringLength"))
        v_t[:] = time_chars

        v_q = nc.createVariable("discharge", "f4", ("stationIdInd",))
        v_q[:] = discharge

        v_qc = nc.createVariable("discharge_quality", "i2", ("stationIdInd",))
        v_qc[:] = quality


def write_timeslice_da(cfg: DAConfig) -> None:
    """Write 15-minute timeslice NetCDF files for persistence DA types.

    Covers 'usgs', 'usace', 'usbr', and 'canada' (WSC) DA types.
    Files are written at 15-minute intervals spanning a one-hour buffer
    on each side of [start_time, end_time].
    """
    if cfg.da_type not in DA_SUFFIX:
        raise ValueError(
            f"write_timeslice_da does not handle da_type='{cfg.da_type}'. "
            "Use write_rfc_da for rfc."
        )

    suffix = DA_SUFFIX[cfg.da_type]
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(cfg.station_ids)
    if n == 0:
        return

    # Build 2D char arrays (n, char_len) matching WRF-Hydro convention.
    # _read_timeslice_file expects np.apply_along_axis(''.join, 1, ...) on axis=1.
    station_ids_chars = np.array(
        [list(sid.rjust(_ID_CHAR_LEN)) for sid in cfg.station_ids], dtype="S1"
    )
    discharge = np.full(n, cfg.discharge, dtype=np.float32)
    quality = np.full(n, cfg.discharge_quality, dtype=np.int16)

    ts_range = pd.date_range(
        cfg.start_dt - pd.Timedelta(hours=1),
        cfg.end_dt + pd.Timedelta(hours=1),
        freq="15min",
    )

    now_str = _now_utc()
    for t in ts_range:
        time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
        out_path = out_dir / f"{time_str}.15min.{suffix}.ncdf"
        _write_timeslice_file(out_path, time_str, station_ids_chars, discharge, quality, now_str)


def write_rfc_da(cfg: DAConfig) -> None:
    """Write hourly RFC forecast NetCDF files (one per hour per station).

    Files span a one-hour buffer on each side of [start_time, end_time].
    """
    if cfg.da_type != "rfc":
        raise ValueError(
            f"write_rfc_da requires da_type='rfc', got '{cfg.da_type}'."
        )

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback = cfg.rfc_lookback_hours
    forecast = cfg.rfc_forecast_hours
    n_steps = lookback + forecast + 1

    encoding = {
        "stationId": {"dtype": "S15"},
        "issueTimeUTC": {"dtype": "S19"},
        "discharges": {"dtype": "float32"},
        "synthetic_values": {"dtype": "int8"},
        "totalCounts": {"dtype": "int16"},
        "observedCounts": {"dtype": "int16"},
        "forecastCounts": {"dtype": "int16"},
        "discharge_qualities": {"dtype": "int16"},
    }

    hour_range = pd.date_range(
        cfg.start_dt - pd.Timedelta(hours=1),
        cfg.end_dt + pd.Timedelta(hours=1),
        freq="h",
    )

    now_str = _now_utc()
    for t in hour_range:
        time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
        hour_str = t.strftime("%Y-%m-%d_%H")
        slice_start_str = (t - pd.Timedelta(hours=lookback)).strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        for sid in cfg.station_ids:
            sid_clean = sid.strip()
            out_path = (
                out_dir
                / f"{hour_str}.{cfg.rfc_timeslice_resolution_minutes}min.{sid_clean}.RFCTimeSeries.ncdf"
            )
            discharges = np.full((1, n_steps), cfg.discharge, dtype=np.float32)
            synthetic = np.zeros((1, n_steps), dtype=np.int8)
            synthetic[0, lookback + 1 :] = 1
            ds = xr.Dataset(
                data_vars={
                    "stationId": np.asarray(sid.rjust(15), dtype="S15"),
                    "issueTimeUTC": (
                        ["nseries"],
                        np.asarray([time_str], dtype="S19"),
                    ),
                    "discharges": (["nseries", "forecastInd"], discharges),
                    "synthetic_values": (["nseries", "forecastInd"], synthetic),
                    "totalCounts": (
                        ["nseries"],
                        np.asarray([n_steps], dtype=np.int16),
                    ),
                    "observedCounts": (
                        ["nseries"],
                        np.asarray([lookback + 1], dtype=np.int16),
                    ),
                    "forecastCounts": (
                        ["nseries"],
                        np.asarray([forecast], dtype=np.int16),
                    ),
                    "timeSteps": (
                        ["nseries"],
                        np.asarray(
                            [np.timedelta64(1, "h")], dtype="timedelta64[ns]"
                        ),
                    ),
                    "discharge_qualities": (
                        ["nseries"],
                        np.asarray([cfg.discharge_quality], dtype=np.int16),
                    ),
                    "queryTime": (
                        ["nseries"],
                        np.asarray(
                            [np.datetime64(t.to_pydatetime(), "ns")],
                            dtype="datetime64[ns]",
                        ),
                    ),
                },
                attrs={
                    "fileUpdateTimeUTC": now_str,
                    "sliceStartTimeUTC": slice_start_str,
                    "sliceTimeResolutionMinutes": str(
                        cfg.rfc_timeslice_resolution_minutes
                    ),
                    "missingValue": "-999",
                    "newest_forecast": "0",
                    "NWM_version_number": "v3.0",
                },
            )
            ds.to_netcdf(out_path, encoding=encoding)


def _valid_nwis_site_ids(station_ids: list[str]) -> list[str]:
    """Filter to station IDs that the NWIS IV service will accept.

    NWIS rejects coordinate-style IDs (15-digit lat/lon encodings) in batch
    requests. Standard USGS site numbers are purely numeric and ≤ 15 digits but
    practically 8 digits for streamflow gages.
    """
    return [sid for sid in station_ids if sid.strip().isdigit() and len(sid.strip()) <= 15]


def _fetch_dv_as_15min(
    sid: str,
    fetch_start: str,
    fetch_end: str,
    ts_range: pd.DatetimeIndex,
) -> pd.Series:
    """Fetch USGS daily-value discharge for one station and resample to 15-min.

    Daily values are placed at noon UTC of each day, then linearly interpolated
    to the target 15-minute index. This avoids a step-function artifact at
    midnight while still being a simple, defensible fill strategy.

    Returns a Series (float, m³/s) aligned to ts_range, or all-NaN on failure.
    """
    empty = pd.Series(np.nan, index=ts_range, name=sid)
    try:
        raw, _ = nwis.get_dv(
            sites=[sid],
            parameterCd=_USGS_DISCHARGE_PARAM,
            startDT=pd.Timestamp(fetch_start).strftime("%Y-%m-%d"),
            endDT=pd.Timestamp(fetch_end).strftime("%Y-%m-%d"),
        )
    except Exception:
        return empty

    if raw.empty:
        return empty

    q_col = next((c for c in raw.columns if _USGS_DISCHARGE_PARAM in c and not c.endswith("_cd")), None)
    if q_col is None:
        return empty

    # DV index is date-only; shift to noon UTC so interpolation is centred on the day.
    # DV index may already be tz-aware; ensure UTC then shift to noon.
    dv_index = pd.DatetimeIndex(raw.index)
    if dv_index.tz is None:
        dv_index = dv_index.tz_localize("UTC")
    else:
        dv_index = dv_index.tz_convert("UTC")
    dv_index = dv_index + pd.Timedelta(hours=12)
    q_daily = pd.Series(raw[q_col].astype(float).values * _CFS_TO_CMS, index=dv_index, name=sid)

    # Reindex onto a combined index (daily + 15-min), interpolate, then select 15-min only.
    combined_index = dv_index.union(ts_range).sort_values()
    q_combined = q_daily.reindex(combined_index).interpolate(method="time", limit_direction="both")
    return q_combined.reindex(ts_range)


def _fetch_usgs_observations(
    station_ids: list[str],
    start_time: str,
    end_time: str,
    dv_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull 15-min discharge from NWIS and return (discharge_cms, quality) DataFrames.

    Both are indexed by UTC timestamp with one column per station.
    If *dv_only* is True, skip the IV fetch entirely and derive all values from
    daily means interpolated to 15-minute resolution.
    discharge_cms is in m³/s; quality is an int 0-100. Missing obs are NaN.
    """
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    fetch_start = (start_dt - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    fetch_end = (end_dt + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    ts_range = pd.date_range(fetch_start, fetch_end, freq="15min", tz="UTC")

    empty = pd.DataFrame(np.nan, index=ts_range, columns=station_ids)
    empty_quality = pd.DataFrame(_USGS_QUALITY_DEFAULT, index=ts_range, columns=station_ids)

    valid_ids = _valid_nwis_site_ids(station_ids)
    if not valid_ids:
        return empty, empty_quality

    if dv_only:
        discharge_cms = empty.copy()
        quality_df = empty_quality.copy()
        for sid in valid_ids:
            dv_fill = _fetch_dv_as_15min(sid, fetch_start, fetch_end, ts_range)
            discharge_cms[sid] = dv_fill
            quality_df[sid] = np.where(dv_fill.notna(), _USGS_QUALITY_CODE_MAP["A"], _USGS_QUALITY_DEFAULT)
        return discharge_cms, quality_df

    try:
        raw, _ = nwis.get_iv(
            sites=valid_ids,
            parameterCd=_USGS_DISCHARGE_PARAM,
            startDT=fetch_start,
            endDT=fetch_end,
        )
    except ValueError:
        # Batch rejected — at least one site ID is invalid for the IV service.
        # Retry per-site, silently dropping any that fail.
        frames = []
        accepted = []
        for sid in valid_ids:
            try:
                df, _ = nwis.get_iv(sites=[sid], parameterCd=_USGS_DISCHARGE_PARAM, startDT=fetch_start, endDT=fetch_end)
                if not df.empty:
                    df["site_no"] = sid
                    frames.append(df)
                    accepted.append(sid)
            except (ValueError, Exception):
                pass
        raw = pd.concat(frames) if frames else pd.DataFrame()
        valid_ids = accepted if accepted else valid_ids
    except Exception:
        return empty, empty_quality

    if raw.empty:
        return empty, empty_quality

    # Multi-site responses have a (site_no, datetime) MultiIndex; flatten it.
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level="site_no")
        raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
    else:
        raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")

    # Column names are "00060" for single-site or "{site_no}_00060" for multi-site.
    q_cols = [c for c in raw.columns if c.endswith(_USGS_DISCHARGE_PARAM) and not c.endswith("_cd")]
    cd_cols = [c for c in raw.columns if c.endswith(f"{_USGS_DISCHARGE_PARAM}_cd")]

    site_q: dict[str, pd.Series] = {}
    site_cd: dict[str, pd.Series] = {}

    if "site_no" in raw.columns:
        # multi-site: filter by site_no
        for sid in valid_ids:
            sub = raw[raw["site_no"] == sid]
            q_col = q_cols[0] if len(q_cols) == 1 else next((c for c in q_cols if sid in c), None)
            cd_col = cd_cols[0] if len(cd_cols) == 1 else next((c for c in cd_cols if sid in c), None)
            site_q[sid] = sub[q_col].astype(float) if q_col else pd.Series(dtype=float)
            site_cd[sid] = sub[cd_col] if cd_col else pd.Series(dtype=str)
    else:
        for sid, q_col, cd_col in zip(valid_ids, q_cols, cd_cols):
            site_q[sid] = raw[q_col].astype(float)
            site_cd[sid] = raw[cd_col]

    discharge_cms = empty.copy()
    quality_df = empty_quality.copy()

    for sid in valid_ids:
        q_series = site_q.get(sid, pd.Series(dtype=float)).reindex(ts_range)
        cd_series = site_cd.get(sid, pd.Series(dtype=str)).reindex(ts_range)
        q_cms = q_series * _CFS_TO_CMS

        # Always fetch DV and use it to fill any NaN gaps in the IV series.
        dv_fill = _fetch_dv_as_15min(sid, fetch_start, fetch_end, ts_range)
        nan_mask = q_cms.isna()
        q_cms = q_cms.fillna(dv_fill)

        discharge_cms[sid] = q_cms

        # For timesteps filled from DV, mark quality as estimated.
        iv_quality = (
            cd_series.map(lambda v: _USGS_QUALITY_CODE_MAP.get(str(v).strip(), _USGS_QUALITY_DEFAULT))
            .fillna(_USGS_QUALITY_DEFAULT)
            .astype(int)
        )
        quality_df[sid] = np.where(nan_mask & q_cms.notna(), _USGS_QUALITY_CODE_MAP["e"], iv_quality)

    return discharge_cms, quality_df


def write_usgs_timeslices(
    station_ids: list[str],
    start_time: str,
    end_time: str,
    output_dir: Path | str = Path("usgs_da"),
    discharge_quality: int = 100,
    dv_only: bool = False,
) -> None:
    """Write 15-min USGS timeslice files using real NWIS observations.

    Fetches instantaneous discharge (ft³/s → m³/s) for each station and writes
    one NetCDF per 15-min step, ±1 hour around [start_time, end_time].
    Stations with no observations are written as NaN.
    If *dv_only* is True, skip the IV fetch and derive all values from daily
    means interpolated to 15-minute resolution.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discharge_cms, quality_df = _fetch_usgs_observations(
        station_ids, start_time, end_time, dv_only=dv_only
    )

    suffix = DA_SUFFIX["usgs"]
    now_str = _now_utc()
    station_ids_chars = np.array(
        [list(sid.rjust(_ID_CHAR_LEN)) for sid in station_ids], dtype="S1"
    )

    for t in discharge_cms.index:
        time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
        out_path = out_dir / f"{time_str}.15min.{suffix}.ncdf"

        missing_mask = discharge_cms.loc[t].isna().to_numpy()
        q_row = discharge_cms.loc[t].to_numpy(dtype=np.float32)
        qc_row = quality_df.loc[t].to_numpy(dtype=np.int16)
        qc_row[missing_mask] = discharge_quality  # override quality for gap-filled values

        _write_timeslice_file(out_path, time_str, station_ids_chars, q_row, qc_row, now_str)


def build_da_dataset(cfg: DAConfig) -> None:
    """Dispatch to the appropriate writer based on cfg.da_type."""
    if cfg.da_type == "rfc":
        write_rfc_da(cfg)
    else:
        write_timeslice_da(cfg)


def write_lake_ontario_outflow(
    out_path: Path,
    start_time: str,
    end_time: str,
    outflow: float = 2000.0,
) -> None:
    """Write a Lake Ontario outflow CSV spanning [start_time, end_time]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start_time, end_time, freq="h")
    pd.DataFrame({
        "Date": times.strftime("%Y-%m-%d"),
        "Hour": times.strftime("%H:00"),
        "Outflow(m3/s)": np.repeat(outflow, len(times)),
    }).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DA forcing files for an NHF test case."
    )
    parser.add_argument(
        "--da-type",
        required=True,
        choices=list(DA_SUBDIR),
        help="DA type to generate: usgs, usace, usbr, canada, or rfc.",
    )
    parser.add_argument(
        "--station-ids",
        nargs="+",
        required=True,
        metavar="STATION_ID",
        help="One or more station identifiers.",
    )
    parser.add_argument(
        "--start-time",
        required=True,
        help="Simulation start time (e.g. '2020-01-01 00:00:00').",
    )
    parser.add_argument(
        "--end-time",
        required=True,
        help="Simulation end time (e.g. '2020-01-02 00:00:00').",
    )
    parser.add_argument(
        "--output-dir",
        default="reservoir_da",
        help="Root output directory (default: reservoir_da/).",
    )
    parser.add_argument(
        "--discharge",
        type=float,
        default=1.0,
        help="Constant discharge value (m³/s) written to every observation.",
    )
    parser.add_argument(
        "--discharge-quality",
        type=int,
        default=100,
        help="Quality flag written alongside every discharge value (default: 100).",
    )
    parser.add_argument(
        "--rfc-lookback-hours",
        type=int,
        default=28,
        help="Hours of observed lookback in RFC files (default: 28).",
    )
    parser.add_argument(
        "--rfc-forecast-hours",
        type=int,
        default=12,
        help="Hours of synthetic forecast in RFC files (default: 12).",
    )
    args = parser.parse_args()

    cfg = DAConfig(
        da_type=args.da_type,
        station_ids=args.station_ids,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=Path(args.output_dir),
        discharge=args.discharge,
        discharge_quality=args.discharge_quality,
        rfc_lookback_hours=args.rfc_lookback_hours,
        rfc_forecast_hours=args.rfc_forecast_hours,
    )

    build_da_dataset(cfg)


if __name__ == "__main__":
    main()
