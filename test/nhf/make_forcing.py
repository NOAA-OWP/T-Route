"""Generate channel forcing files for a CONUS test case using retrospective data, and create a config YAML for running the test case."""
import argparse
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dataretrieval import nwis

RETRO_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETROSPECTIVE_LATERAL_FIELD = "q_lateral"
RETROSPECTIVE_FLOW_FIELD = "streamflow"

def create_forcing_dataset(t_start: str, t_end: str, forcing_dir: str, hydrofabric_path: str, retrospective_path: str, forcing_file_pattern: str = "CHRTOUT_DOMAIN1", generate_reference_data: bool = False, reference_dir: Union[str, None] = None, runout_time: int = 0):
    """Create a dataset of channel forcing files from retrospective data."""
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    crosswalk = gpd.read_file(hydrofabric_path, layer="reference_flowpaths")
    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    retro = xr.open_zarr(retrospective_path, storage_options={"anon": True},)

    # Post-process
    feature_ids_retro = crosswalk["ref_fp_id"].values
    crosswalk = pd.merge(crosswalk[["ref_fp_id", "div_id"]], fps[["fp_id", "div_id"]], left_on="div_id", right_on="div_id", how="left")

    # Generate dataset
    iterator = pd.date_range(
        start=t_start,
        end=t_end,
        freq="h"
    )
    for i in iterator:
        print(f"Processing time step {i}...")
        qlat = retro.sel(feature_id=feature_ids_retro, time=i)[RETROSPECTIVE_LATERAL_FIELD].reset_coords(drop=True)
        t_str = i.strftime("%Y%m%d%H%M")
        df = qlat.to_dataframe()
        df = pd.merge(df, crosswalk[["ref_fp_id", "fp_id"]], left_index=True, right_on="ref_fp_id", how="left").rename(columns={"fp_id": "feature_id", RETROSPECTIVE_LATERAL_FIELD: t_str})[["feature_id", t_str]]
        df = df.groupby("feature_id").sum().reset_index()
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)
    for i in range(1, runout_time + 1):
        print(f"Processing runout time step {i}...")
        t_str = (iterator[-1] + pd.Timedelta(hours=i)).strftime("%Y%m%d%H%M")
        df = pd.DataFrame({
            "feature_id": fps["fp_id"],
            t_str: [0.0] * len(fps),
        })
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)

    # Generate reference outputs if requested
    if generate_reference_data and reference_dir is not None:
        gages = gpd.read_file(hydrofabric_path, sql="SELECT * FROM gages WHERE status = 'USGS-active'", ignore_geometry=True)
        data_vars = {}
        fp_ids = []
        site_nos = []
        first = True
        for _, gage in gages.iterrows():
            # Load retrospective flow for the gage's reference flowpath
            if pd.isna(gage["fp_id"]):
                continue
            fp_id = int(gage["fp_id"])
            ref_fp_id = crosswalk.loc[crosswalk["fp_id"] == fp_id, "ref_fp_id"].values[0]
            retro_q = retro.sel(feature_id=ref_fp_id, time=slice(t_start, t_end))[RETROSPECTIVE_FLOW_FIELD].reset_coords(drop=True).to_dataframe()
            retro_q.index = retro_q.index.tz_localize("UTC")

            # Load USGS data, if available
            site_no = gage["site_no"]
            usgs_raw = nwis.get_iv(site=site_no, start=t_start.strftime("%Y-%m-%dT%H:%MZ"), end=t_end.strftime("%Y-%m-%dT%H:%MZ"), parameterCd="00060")[0]
            if "00060" in usgs_raw.columns:
                usgs_q = usgs_raw.rename(columns={"00060": "usgs_q"})
                usgs_q.index = pd.to_datetime(usgs_q.index)
                usgs_q = usgs_q.reindex(retro_q.index, method="nearest", tolerance=pd.Timedelta("15min"))
            else:
                usgs_q = pd.DataFrame({"usgs_q": np.nan}, index=retro_q.index)

            if first:
                time_index = pd.DatetimeIndex(retro_q.index).tz_localize(None)
                first = False
            # Log metadata
            fp_ids.append(fp_id)
            site_nos.append(site_no)

            # Stack along gage dimension
            data_vars.setdefault("retrospective_q", []).append(retro_q[RETROSPECTIVE_FLOW_FIELD].values)
            data_vars.setdefault("usgs_q", []).append(usgs_q["usgs_q"].values)

        # Convert to arrays (gage x time)
        retrospective_array = np.stack(data_vars["retrospective_q"], axis=0)
        usgs_array = np.stack(data_vars["usgs_q"], axis=0)

        # Build xarray dataset
        ds = xr.Dataset(
            {
                "retrospective_q": (("gage", "time"), retrospective_array),
                "usgs_q": (("gage", "time"), usgs_array),
            },
            coords={
                "site_no": ("gage", site_nos),
                "fp_id": ("gage", fp_ids),
                "time": time_index,
            },
        )

        # Save single NetCDF
        ds.to_netcdf(Path(reference_dir) / "gage_reference_data.nc")



def make_config_yaml(config_path: str, hydrofabric_path: str, qlat_input_folder: str, nts: int, restart_time: str, output_dir: str, file_pattern_filter: str = "*.CHRTOUT_DOMAIN1.csv", max_loop_size: int = 288):
    """Create a config YAML for running the test case."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "log_parameters": {
            "showtiming": True,
            "log_level": "DEBUG",
        },
        "network_topology_parameters": {
            "supernetwork_parameters": {
                "geo_file_path": hydrofabric_path,
            },
            "waterbody_parameters": {
                "break_network_at_waterbodies": True,
            },
        },
        "compute_parameters": {
            "parallel_compute_method": "by-subnetwork-jit-clustered",
            "compute_kernel": "V02-structured",
            "assume_short_ts": True,
            "subnetwork_target_size": 10000,
            "cpu_pool": 1,
            "restart_parameters": {
                "start_datetime": restart_time,
            },
            "forcing_parameters": {
                "dt": 300,
                "qlat_input_folder": qlat_input_folder,
                "qlat_file_pattern_filter": file_pattern_filter,
                "nts": nts,
                "max_loop_size": max_loop_size,
            },
            "data_assimilation_parameters": {
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "diffusive_streamflow_nudging": False,
                },
                "reservoir_da": {
                    "reservoir_persistence_da": {
                        "reservoir_persistence_usgs": False,
                    },
                    "reservoir_rfc_da": {
                        "reservoir_rfc_forecasts": False,
                    },
                },
            },
        },
        "output_parameters": {
            "stream_output": {
                "stream_output_directory": output_dir,
                "stream_output_time": -1,
                "stream_output_type": ".nc",
                "stream_output_internal_frequency": 60,
            },
        },
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

def build_forcing_dataset(start_time: str, end_time: str, case_id: str, run_id: str, hf_file: str, generate_reference_data: bool = False, add_runout_period: bool = False):
    """Build the forcing dataset and config YAML for a given case and run."""
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    run_dir = Path(__file__).parent / case_id
    forcing_subdir = f"channel_forcing_{run_id}"
    config_path = run_dir / f"{run_id}.yaml"
    forcing_dir = run_dir / forcing_subdir
    hf_path = run_dir / "domain" / hf_file
    output_dir = f"output_{run_id}/"
    if generate_reference_data:
        reference_dir = run_dir / run_id
        reference_dir.mkdir(parents=True, exist_ok=True)
    else:
        reference_dir = None
    if add_runout_period:
        runout_time = int(((end_dt - start_dt) / 2).total_seconds() / 3600)

    create_forcing_dataset(
        t_start=start_dt,
        t_end=end_dt,
        forcing_dir=forcing_dir,
        hydrofabric_path=hf_path,
        retrospective_path=RETRO_PATH,
        generate_reference_data=generate_reference_data,
        reference_dir=reference_dir,
        runout_time=runout_time,
    )
    dt = 300 # could be an input argument if we want to vary it across runs
    sim_time = (end_dt - start_dt).total_seconds()
    if add_runout_period:
        sim_time += (runout_time + 1) * 3600
    nts = int(sim_time / dt)

    make_config_yaml(
        config_path=config_path,
        hydrofabric_path=f"domain/{hf_file}",
        qlat_input_folder=f"{forcing_subdir}/",
        nts=nts,
        restart_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        output_dir=output_dir
    )

def conecuh_retro():
    """Generate forcing files and config yaml for December 2009 floods."""
    start_time = "2009-12-12"
    end_time = "2009-12-29"
    case_id = "conecuh_case"
    run_id = "retro"
    hf_file =  "02374250.gpkg"

    build_forcing_dataset(
        start_time=start_time,
        end_time=end_time,
        case_id=case_id,
        run_id=run_id,
        hf_file=hf_file,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate forcing dataset and config YAML for a case."
    )

    parser.add_argument(
        "--start-time",
        default="2009-12-12 00:00",
        help="Simulation start time (e.g. '2009-12-12' or '2009-12-12 06:00')",
    )

    parser.add_argument(
        "--end-time",
        default="2009-12-29 00:00",
        help="Simulation end time (e.g. '2009-12-29' or '2009-12-29 12:00')",
    )

    parser.add_argument(
        "--case-id",
        default="conecuh_case",
        help="Case directory name",
    )

    parser.add_argument(
        "--hf-file",
        default="02374250.gpkg",
        help="Hydrofabric file inside domain directory",
    )

    parser.add_argument(
        "--run-id",
        default="retro",
        help="Run identifier.  There can be multiple runs per case.",
    )

    parser.add_argument(
        "--generate-reference-data",
        default=True,
        help="Generate reference data (USGS and retrospective outputs) for testing.",
    )

    parser.add_argument(
        "--add-runout-period",
        default=True,
        help="Generate reference data (USGS and retrospective outputs) for testing.",
    )

    args = parser.parse_args()

    build_forcing_dataset(
        start_time=args.start_time,
        end_time=args.end_time,
        case_id=args.case_id,
        run_id=args.run_id,
        hf_file=args.hf_file,
        generate_reference_data=args.generate_reference_data,
        add_runout_period=args.add_runout_period,
    )


if __name__ == "__main__":
    main()
