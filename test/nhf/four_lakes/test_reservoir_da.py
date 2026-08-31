
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

OUTLET_REACH_ID = 1276182780176988
NHF_PATH = "/hydrofabric/nhf_1.2.1.gpkg"
RESERVOIR_TYPE_MOD = {
    1276675272336236: 2,
    1276673989193624: 3,
    1276672890005227: 4,
    1276185235011805: 7,
}
RESERVOIR_FLOW_VALUES = {
    1276675272336236: 1.0,
    1276673989193624: 1.0,
    1276672890005227: 12345,
    1276185235011805: 1.0,
}
# Synthetic site identifiers assigned to each reservoir for DA matching.
RESERVOIR_SITE_NOS = {
    1276675272336236: "USGS00000000002",
    1276673989193624: "USAC00000000003",
    1276672890005227: "RFC000000000004",
    1276185235011805: "USBR00000000007",
}
# Immediately downstream flowpath fp_id for each reservoir (derived from nexus table).
RESERVOIR_DN_FP = {
    1276675272336236: 1276674107160768,
    1276673989193624: 1276673989391626,
    1276672890005227: 1276672844213843,
    1276185235011805: 1276185303862487,
}
DA_TYPE_LOOKUP = {2: "usgs_da", 3: "usace_da", 4: "rfc_da", 7: "usbr_da"}
DA_TIMESLICE_SUFFIX = {2: "usgsTimeSlice", 3: "usaceTimeSlice", 7: "usbrTimeSlice"}

START_TIME = "2020-01-01 00:00:00"
END_TIME = "2020-01-01 01:00:00"
DT = 300   # seconds per routing timestep
NTS = 12   # (1 hour * 3600 s) / 300 s


@dataclass
class RunContext:
    model_root: Path = field(default_factory=lambda: Path(__file__).parent)
    hf_path: Path = field(default_factory=lambda: Path(__file__).parent / "domain" / "nhf.gpkg")
    forcing_dir: Path = field(default_factory=lambda: Path(__file__).parent / "channel_forcing")
    da_dir: Path = field(default_factory=lambda: Path(__file__).parent / "reservoir_da")
    config_path: Path = field(default_factory=lambda: Path(__file__).parent / "config.yaml")
    output_dir: str = "output"


def subset_nhf(run_context: RunContext):
    """Domain is pre-built; verify it exists."""
    if not run_context.hf_path.exists():
        relative_hf_path = run_context.hf_path.relative_to(run_context.model_root)
        subprocess.run(
            ["python", "../subset_nhf.py", "--source-gpkg", NHF_PATH, "--out-gpkg", str(relative_hf_path), "--outlet-fp-id", str(OUTLET_REACH_ID)],
            cwd=run_context.model_root,
            check=True,
        )


def modify_lakes_table(run_context: RunContext):
    """Update da_type and assign synthetic site_nos in the gpkg reservoir_da table."""
    conn = sqlite3.connect(run_context.hf_path)
    cur = conn.cursor()
    for lake_id, da_type in RESERVOIR_TYPE_MOD.items():
        site_no = RESERVOIR_SITE_NOS[lake_id]
        cur.execute(
            "UPDATE reservoir_da SET da_type=?, site_no=? WHERE nhf_lake_id=?",
            (da_type, site_no, lake_id),
        )
    conn.commit()
    conn.close()


def make_channel_forcing_data(run_context: RunContext):
    """Write zero-valued channel forcing CSVs for every flowpath and timestep."""
    run_context.forcing_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(run_context.hf_path)
    cur = conn.cursor()
    cur.execute("SELECT fp_id FROM flowpaths")
    fp_ids = [int(r[0]) for r in cur.fetchall()]
    conn.close()

    timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq="h")
    for t in timestamps:
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": fp_ids, t_str: 100.0})
        df.to_csv(
            run_context.forcing_dir / f"{t_str}.CHRTOUT_DOMAIN1.csv",
            index=False,
        )


def make_reservoir_da_data(run_context: RunContext):
    """Write constant-flow DA forcing files for each reservoir type."""
    for subdir in DA_TYPE_LOOKUP.values():
        (run_context.da_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Build per-type lists of (site_no, flow).
    type_to_sites: dict[int, list[tuple[str, float]]] = {}
    for lake_id, da_type in RESERVOIR_TYPE_MOD.items():
        type_to_sites.setdefault(da_type, []).append(
            (RESERVOIR_SITE_NOS[lake_id], RESERVOIR_FLOW_VALUES[lake_id])
        )

    start_dt = pd.Timestamp(START_TIME)
    end_dt = pd.Timestamp(END_TIME)
    # Timeslice files: 15-min intervals covering the simulation window plus a
    # one-hour buffer on each side to satisfy any lookback/lookahead padding.
    ts_range = pd.date_range(
        start_dt - pd.Timedelta(hours=1),
        end_dt + pd.Timedelta(hours=1),
        freq="15min",
    )

    encoding_ts = {
        "stationId": {"dtype": "S15"},
        "time": {"dtype": "S19"},
        "discharge": {"dtype": "float32"},
        "discharge_quality": {"dtype": "int16"},
    }
    encoding_rfc = {
        "stationId": {"dtype": "S15"},
        "issueTimeUTC": {"dtype": "S19"},
        "discharges": {"dtype": "float32"},
        "synthetic_values": {"dtype": "int8"},
        "totalCounts": {"dtype": "int16"},
        "observedCounts": {"dtype": "int16"},
        "forecastCounts": {"dtype": "int16"},
        "discharge_qualities": {"dtype": "int16"},
        # timeSteps encoding is intentionally omitted; let xarray infer it for timedelta64.
    }

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")

    for da_type, site_list in type_to_sites.items():
        out_subdir = run_context.da_dir / DA_TYPE_LOOKUP[da_type]

        if da_type in DA_TIMESLICE_SUFFIX:
            # Persistence DA types (USGS=2, USACE=3, USBR=7): one file per 15-min slot.
            suffix = DA_TIMESLICE_SUFFIX[da_type]
            station_ids = np.asarray(
                [sid.rjust(15) for sid, _ in site_list], dtype="S15"
            )
            discharge = np.asarray([flow for _, flow in site_list], dtype=np.float32)
            quality = np.full(len(site_list), 100, dtype=np.int16)
            for t in ts_range:
                time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
                out_path = out_subdir / f"{time_str}.15min.{suffix}.ncdf"
                ds = xr.Dataset(
                    data_vars={
                        "stationId": (["stationIdInd"], station_ids),
                        "time": (
                            ["stationIdInd"],
                            np.asarray([time_str] * len(site_list), dtype="S19"),
                        ),
                        "discharge": (["stationIdInd"], discharge),
                        "discharge_quality": (["stationIdInd"], quality),
                    },
                    attrs={
                        "fileUpdateTimeUTC": now_str,
                        "sliceCenterTimeUTC": time_str,
                        "sliceTimeResolutionMinutes": "15",
                    },
                )
                ds.to_netcdf(out_path, encoding=encoding_ts)

        elif da_type == 4:
            # RFC (type 4): one file per hour per station.
            # Each file covers a multi-hour timeseries starting 28 hours before
            # the issue time (matching the lookback_hours used in the config) so
            # that t0 (START_TIME) is guaranteed to be present in every file.
            # The reference format (BCRT2) uses:
            #   stationId           – scalar (0-dim)
            #   discharges          – (nseries, forecastInd)  one entry per hour
            #   synthetic_values    – (nseries, forecastInd)
            #   totalCounts / observedCounts / forecastCounts / timeSteps /
            #   discharge_qualities / queryTime / issueTimeUTC  – (nseries,) scalars
            lookback_hrs = 28
            forecast_hrs = 12   # enough to cover the simulation and persist window
            n_steps = lookback_hrs + forecast_hrs + 1  # hourly steps

            hour_range = pd.date_range(
                start_dt - pd.Timedelta(hours=1),
                end_dt + pd.Timedelta(hours=1),
                freq="h",
            )
            for t in hour_range:
                time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
                hour_str = t.strftime("%Y-%m-%d_%H")
                slice_start = t - pd.Timedelta(hours=lookback_hrs)
                slice_start_str = slice_start.strftime("%Y-%m-%d_%H:%M:%S")
                for sid, flow in site_list:
                    sid_clean = sid.strip()
                    out_path = (
                        out_subdir
                        / f"{hour_str}.60min.{sid_clean}.RFCTimeSeries.ncdf"
                    )
                    discharges = np.full((1, n_steps), flow, dtype=np.float32)
                    # Mark the forecast portion (after issue time) as synthetic.
                    synthetic = np.zeros((1, n_steps), dtype=np.int8)
                    synthetic[0, lookback_hrs + 1:] = 1
                    ds = xr.Dataset(
                        data_vars={
                            "stationId": np.asarray(sid.rjust(15), dtype="S15"),
                            "issueTimeUTC": (
                                ["nseries"],
                                np.asarray([time_str], dtype="S19"),
                            ),
                            "discharges": (
                                ["nseries", "forecastInd"],
                                discharges,
                            ),
                            "synthetic_values": (
                                ["nseries", "forecastInd"],
                                synthetic,
                            ),
                            "totalCounts": (
                                ["nseries"],
                                np.asarray([n_steps], dtype=np.int16),
                            ),
                            "observedCounts": (
                                ["nseries"],
                                np.asarray([lookback_hrs + 1], dtype=np.int16),
                            ),
                            "forecastCounts": (
                                ["nseries"],
                                np.asarray([forecast_hrs], dtype=np.int16),
                            ),
                            "timeSteps": (
                                ["nseries"],
                                np.asarray(
                                    [np.timedelta64(1, "h")],
                                    dtype="timedelta64[ns]",
                                ),
                            ),
                            "discharge_qualities": (
                                ["nseries"],
                                np.asarray([100], dtype=np.int16),
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
                            "sliceTimeResolutionMinutes": "60",
                            "missingValue": "-999",
                            "newest_forecast": "0",
                            "NWM_version_number": "v3.0",
                        },
                    )
                    ds.to_netcdf(out_path, encoding=encoding_rfc)


def make_config(run_context: RunContext):
    """Write a T-Route config yaml with all four reservoir DA types enabled."""
    config_dict = {
        "log_parameters": {
            "showtiming": True,
            "log_level": "DEBUG",
        },
        "network_topology_parameters": {
            "supernetwork_parameters": {
                "geo_file_path": "domain/nhf.gpkg",
                "network_type": "NHF",
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
                "start_datetime": START_TIME,
            },
            "forcing_parameters": {
                "dt": DT,
                "qlat_input_folder": "channel_forcing/",
                "qlat_file_pattern_filter": "*.CHRTOUT_DOMAIN1.csv",
                "nts": NTS,
                "max_loop_size": NTS,
            },
            "data_assimilation_parameters": {
                "timeslice_lookback_hours": 1,
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "diffusive_streamflow_nudging": False,
                },
                "usgs_timeslices_folder": "reservoir_da/usgs_da/",
                "usace_timeslices_folder": "reservoir_da/usace_da/",
                "usbr_timeslices_folder": "reservoir_da/usbr_da/",
                "reservoir_da": {
                    "reservoir_persistence_da": {
                        "reservoir_persistence_usgs": True,
                        "reservoir_persistence_usace": True,
                        "reservoir_persistence_usbr": True,
                    },
                    "reservoir_rfc_da": {
                        "reservoir_rfc_forecasts": True,
                        "reservoir_rfc_forecasts_time_series_path": "reservoir_da/rfc_da/",
                        "reservoir_rfc_forecasts_lookback_hours": 28,
                        "reservoir_rfc_forecasts_offset_hours": 0,
                        "reservoir_rfc_forecast_persist_days": 11,
                    },
                },
            },
        },
        "output_parameters": {
            "stream_output": {
                "stream_output_directory": run_context.output_dir,
                "stream_output_time": -1,
                "stream_output_type": ".nc",
                "stream_output_internal_frequency": 60,
            },
        },
    }
    with open(run_context.config_path, "w") as f:
        yaml.dump(config_dict, f)


def run_troute(run_context: RunContext):
    """Execute the generated T-Route run."""
    subprocess.run(
        ["python", "-m", "nwm_routing", "-V5", "-f", str(run_context.config_path)],
        cwd=run_context.model_root,
        check=True,
    )


def review_results(run_context: RunContext):
    """Check that flow is 0 everywhere except downstream of a DA reservoir and
    verify each reservoir's outflow matches its expected constant value."""
    output_path = run_context.model_root / run_context.output_dir
    output_files = sorted(output_path.glob("*.nc"))
    assert output_files, f"No output files found in {output_path}"

    output_ds = xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in output_files],
        dim="time",
    )
    feature_ids = set(output_ds["feature_id"].values.tolist())

    # Each immediately-downstream reach must carry the reservoir's DA outflow.
    for lake_id, dn_fp_id in RESERVOIR_DN_FP.items():
        if dn_fp_id not in feature_ids:
            continue
        expected = RESERVOIR_FLOW_VALUES[lake_id]
        actual = output_ds["flow"].sel(feature_id=dn_fp_id).values
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.05,
            atol=1.0,
            err_msg=(
                f"Flow downstream of reservoir {lake_id} (fp {dn_fp_id}) "
                f"should be ~{expected} m³/s"
            ),
        )


def test_reservoir_da():
    run_context = RunContext()
    subset_nhf(run_context)
    modify_lakes_table(run_context)
    make_channel_forcing_data(run_context)
    make_reservoir_da_data(run_context)
    make_config(run_context)
    run_troute(run_context)
    review_results(run_context)


if __name__ == "__main__":
    test_reservoir_da()