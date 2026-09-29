"""Generate gage reference data (retrospective + USGS observed) for an NHF test case.

Reads the t-route config YAML to derive the hydrofabric path, start time, and
end time (from nts and dt).  Only the config path and output directory are
required as CLI arguments.

Usage
-----
    python generate_reference_data.py \\
        --config path/to/config.yaml \\
        --output-dir path/to/reference/

For each USGS-active gage in the hydrofabric, instantaneous-value (IV) discharge
is fetched first.  If IV data are unavailable for the requested period, daily
mean values (DV) are fetched as a fallback and resampled to the retrospective
hourly time index.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio.errors
import xarray as xr
import yaml
from dataretrieval import waterdata

RETRO_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETROSPECTIVE_FLOW_FIELD = "streamflow"


def _fetch_usgs_q(
    site_no: str,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    target_index: pd.DatetimeIndex,
    dv_only: bool = False,
) -> pd.Series:
    """Return a discharge Series aligned to *target_index*.

    Fetches instantaneous values (IV) and daily mean values (DV) in parallel.
    IV observations take priority; any NaN gaps in IV are filled with DV values
    linearly interpolated to the target resolution (daily values are placed at
    noon UTC before interpolation to avoid step-function artefacts at midnight).

    Parameters
    ----------
    site_no:
        USGS site number string.
    t_start, t_end:
        Period of interest (timezone-naive, interpreted as UTC).
    target_index:
        DatetimeIndex (UTC-aware) to align the result to.

    Returns
    -------
    pd.Series with name ``"usgs_q"`` aligned to *target_index*.
    """
    start_str = t_start.strftime("%Y-%m-%dT%H:%MZ")
    end_str = t_end.strftime("%Y-%m-%dT%H:%MZ")
    nan_series = pd.Series(np.nan, index=target_index, name="usgs_q")
    site_id = f"USGS-{site_no}"

    # --- Instantaneous values ---
    iv_series = nan_series.copy()
    if not dv_only:
        try:
            iv_raw = waterdata.get_continuous(
                monitoring_location_id=site_id,
                parameter_code="00060",
                time=f"{start_str}/{end_str}",
            )[0]
            if not iv_raw.empty and "value" in iv_raw.columns:
                iv_q = iv_raw[["time", "value"]].rename(columns={"value": "usgs_q"})
                iv_q["time"] = pd.to_datetime(iv_q["time"], utc=True)
                iv_q = iv_q.set_index("time")
                iv_series = iv_q["usgs_q"].reindex(
                    target_index, method="nearest", tolerance=pd.Timedelta("15min")
                )
        except Exception:
            pass

    # --- Daily values: always fetch and use to fill any NaN gaps in IV ---
    try:
        dv_raw = waterdata.get_daily(
            monitoring_location_id=site_id,
            parameter_code="00060",
            statistic_id="00003",
            time=f"{t_start.strftime('%Y-%m-%d')}/{t_end.strftime('%Y-%m-%d')}",
        )[0]
        if (
            not dv_raw.empty
            and "value" in dv_raw.columns
            and not dv_raw["value"].isna().all()
        ):
            dv_q = dv_raw[["time", "value"]].rename(columns={"value": "usgs_q"})
            dv_q["time"] = pd.to_datetime(dv_q["time"], utc=True)
            dv_q = dv_q.set_index("time")
            # Place daily values at noon UTC then interpolate to the target index.
            dv_q.index = dv_q.index + pd.Timedelta(hours=12)
            combined_index = dv_q.index.union(target_index).sort_values()
            dv_interp = (
                dv_q["usgs_q"]
                .reindex(combined_index)
                .interpolate(method="time", limit_direction="both")
                .reindex(target_index)
            )
            iv_series = iv_series.fillna(dv_interp)
    except Exception:
        pass

    return iv_series


def generate_reference_data(
    hydrofabric_path: Path,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    output_dir: Path,
    dv_only: bool = False,
) -> None:
    """Fetch retrospective and USGS flows for all active gages and write a NetCDF.

    Parameters
    ----------
    hydrofabric_path:
        Path to the NHF GeoPackage containing ``gages``, ``flowpaths``, and
        ``reference_flowpaths`` layers.
    t_start, t_end:
        Simulation window (timezone-naive).
    output_dir:
        Directory where ``gage_reference_data.nc`` will be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gages layer
    try:
        gages = gpd.read_file(
            hydrofabric_path,
            sql="SELECT * FROM gages WHERE status = 'USGS-active'",
            ignore_geometry=True,
        )
    except pyogrio.errors.DataLayerError:
        print("No 'gages' layer found in the hydrofabric — nothing to generate.")
        return

    if gages.empty:
        print("No USGS-active gages found — nothing to generate.")
        return

    # Build crosswalk: fp_id → ref_fp_id
    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    ref_fps = gpd.read_file(hydrofabric_path, layer="reference_flowpaths")
    crosswalk = pd.merge(
        ref_fps[["ref_fp_id", "div_id"]],
        fps[["fp_id", "div_id"]],
        on="div_id",
        how="left",
    )

    retro = xr.open_zarr(RETRO_PATH, storage_options={"anon": True})

    fp_ids: list[int] = []
    site_nos: list[str] = []
    retro_arrays: list[np.ndarray] = []
    usgs_arrays: list[np.ndarray] = []
    time_index: pd.DatetimeIndex | None = None

    for _, gage in gages.iterrows():
        if pd.isna(gage["fp_id"]):
            continue

        fp_id = int(gage["fp_id"])
        xw_row = crosswalk.loc[crosswalk["fp_id"] == fp_id, "ref_fp_id"]
        if xw_row.empty:
            print(
                f"  Skipping gage {gage['site_no']}: no crosswalk entry for fp_id {fp_id}"
            )
            continue
        ref_fp_id = xw_row.values[0]

        # Retrospective flow
        retro_q = (
            retro.sel(feature_id=ref_fp_id, time=slice(t_start, t_end))[
                RETROSPECTIVE_FLOW_FIELD
            ]
            .reset_coords(drop=True)
            .to_dataframe()
        )
        retro_q.index = retro_q.index.tz_localize("UTC")

        if time_index is None:
            time_index = retro_q.index

        # USGS observed flow (IV with DV fallback)
        site_no = gage["site_no"]
        print(f"  Fetching USGS data for site {site_no}...")
        usgs_q = _fetch_usgs_q(site_no, t_start, t_end, retro_q.index, dv_only=dv_only)

        fp_ids.append(fp_id)
        site_nos.append(site_no)
        retro_arrays.append(retro_q[RETROSPECTIVE_FLOW_FIELD].values)
        usgs_arrays.append(usgs_q.values)

    if not fp_ids:
        print("No valid gages processed — output not written.")
        return

    # Strip timezone for NetCDF compatibility
    time_coords = pd.DatetimeIndex(time_index).tz_localize(None)

    ds = xr.Dataset(
        {
            "retrospective_q": (("gage", "time"), np.stack(retro_arrays, axis=0)),
            "usgs_q": (("gage", "time"), np.stack(usgs_arrays, axis=0)),
        },
        coords={
            "site_no": ("gage", site_nos),
            "fp_id": ("gage", fp_ids),
            "time": time_coords,
        },
    )

    out_path = output_dir / "gage_reference_data.nc"
    ds.to_netcdf(out_path)
    print(f"Wrote {out_path}")


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gage reference data from a t-route config YAML."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the t-route config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write gage_reference_data.nc.",
    )
    parser.add_argument(
        "--dv-only",
        action="store_true",
        default=False,
        help="Only fetch daily mean values (DV); skip instantaneous-value (IV) requests.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_config(config_path)

    hydrofabric_path = Path(
        cfg["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"]
    )
    if not hydrofabric_path.is_absolute():
        hydrofabric_path = config_path.parent / hydrofabric_path
    t_start = pd.to_datetime(
        cfg["compute_parameters"]["restart_parameters"]["start_datetime"]
    )
    forcing_params = cfg["compute_parameters"]["forcing_parameters"]
    nts = forcing_params["nts"]
    dt = forcing_params.get("dt", 300)
    t_end = t_start + pd.Timedelta(seconds=nts * dt)

    generate_reference_data(
        hydrofabric_path=hydrofabric_path,
        t_start=t_start,
        t_end=t_end,
        output_dir=Path(args.output_dir),
        dv_only=args.dv_only,
    )


if __name__ == "__main__":
    main()
