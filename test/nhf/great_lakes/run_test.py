
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import geopandas as gpd

import json
from pathlib import Path
import xarray as xr
import yaml

def get_feature_ids(domain_path: Path) -> np.ndarray:
    return gpd.read_file(domain_path, layer="flowpaths")["fp_id"].values

def write_forcing(forcing_dir: Path, timesteps: int, start_time: str, feature_ids: np.ndarray) -> None:
    forcing_dir.mkdir(exist_ok=True)
    times = pd.date_range(start_time, periods=timesteps, freq="h")
    for i, t in enumerate(times):
        ts = t.strftime("%Y%m%d%H%M")
        fname = forcing_dir / f"{ts}.CHRTOUT_DOMAIN1.csv"
        df = pd.DataFrame({
            "feature_id": feature_ids,
            ts: np.zeros(len(feature_ids)),
        })
        df.to_csv(fname, index=False, float_format="%.15g")

def write_da(da_dir: Path, timesteps: int, start_time: str, station_ids: np.ndarray, suffix: str) -> None:
    da_dir.mkdir(exist_ok=True)
    n = len(station_ids)
    times = pd.date_range(start_time, periods=timesteps, freq="h")
    encoding = {
            "stationId": {"dtype": "S15"},
            "time": {"dtype": "S19"},
            "discharge": {"dtype": "float32"},
            "discharge_quality": {"dtype": "int16"},
        }
    for i, t in enumerate(times):
        ts = t.strftime("%Y-%m-%d_%H:%M:%S")
        fname = da_dir / f"{ts}.{suffix}"
        times = np.repeat(ts, n)
        discharge = np.repeat(2000, n)
        discharge_quality = np.repeat(100, n)
        query_times = np.repeat(t, n)

        ds = xr.Dataset(
            data_vars={
                "stationId": (
                    ["stationIdInd"],
                    np.asarray(station_ids, dtype="S15"),
                ),
                "time": (
                    ["stationIdInd"],
                    np.asarray(times, dtype="S19"),
                ),
                "discharge": (
                    ["stationIdInd"],
                    np.asarray(discharge, dtype=np.float32),
                ),
                "discharge_quality": (
                    ["stationIdInd"],
                    np.asarray(discharge_quality, dtype=np.int16),
                ),
                "queryTime": (
                    ["stationIdInd"],
                    pd.to_datetime(query_times),
                ),
            },
            attrs={
                "fileUpdateTimeUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S"),
                "sliceCenterTimeUTC": t.strftime("%Y-%m-%d_%H:%M:%S"),
                "sliceTimeResolutionMinutes": 15,
            },
        )

        ds.to_netcdf(fname, encoding=encoding)

def write_ontario(fname: str, timesteps: int, start_time: str):
    times = pd.date_range(start_time, periods=timesteps, freq="h")

    df = pd.DataFrame({
        "Date": times.strftime("%Y-%m-%d"),
        "Hour": times.strftime("%H:00"),
        "Outflow(m3/s)": np.repeat(2000, len(times))
    })

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fname, index=False)

def load_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        d = yaml.safe_load(f)
    return d

def run_troute(cfg_path: str):
    args = ["python", "-m", "nwm_routing", "-f", "-V5", cfg_path]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Troute returned nonzero exit code: {result.returncode}")

def main():
    cdir = Path(__file__).parent
    os.chdir(cdir)

    cfg_path = "no_inflow.yaml"
    cfg = load_config(cfg_path)
    domain_path = cfg["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"]
    forcing_dir = Path(cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"])
    da_dir = Path(cfg["compute_parameters"]["data_assimilation_parameters"]["usgs_timeslices_folder"])
    can_da_dir = Path(cfg["compute_parameters"]["data_assimilation_parameters"]["canada_timeslices_folder"])
    ontario_path = Path(cfg["compute_parameters"]["data_assimilation_parameters"]["LakeOntario_outflow"])
    timesteps = cfg["compute_parameters"]["forcing_parameters"]["nts"]
    start_dt = cfg["compute_parameters"]["restart_parameters"]["start_datetime"]
    out_dir = cfg["output_parameters"]["stream_output"]["stream_output_directory"]
    out_ext = cfg["output_parameters"]["stream_output"]["stream_output_type"]
    usgs_station_ids = ["04127885", "04159130"]
    can_station_ids = ["02HA013"]

    fids = get_feature_ids(domain_path)
    write_forcing(forcing_dir, timesteps, start_dt, fids)
    write_da(da_dir, timesteps, start_dt, usgs_station_ids, "15min.usgsTimeSlice.ncdf")
    write_da(can_da_dir, timesteps, start_dt, can_station_ids, "15min.wscTimeSlice.ncdf")
    write_ontario(ontario_path, timesteps, start_dt)

    run_troute(cfg_path)

    out_files = Path(out_dir).glob("*"+out_ext)
    ds = xr.concat([xr.open_dataset(i) for i in out_files], dim="time")
    max_flows = ds["flow"].values.max(axis=0)
    close = np.isclose(max_flows, 2000, atol=5).all()
    if not close:
        raise RuntimeError("Maximum flows for great lakes runs not close to forcing value of 2,000")

if __name__ == "__main__":
    main()