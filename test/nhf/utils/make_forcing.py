"""Generate channel forcing files and a config YAML for running an NHF test case.

Three forcing modes are available via --forcing-mode:

  retro   (default) Pull lateral inflows from the NWM v3.0 retrospective Zarr
            store on S3.  Requires --start-time, --end-time, and an NHF gpkg
            with a ``reference_flowpaths`` layer.

  pulse   Apply a synthetic unit-hydrograph pulse scaled to --peak-qlat
            (m³/s, default 10 000) uniformly across all reaches.  The pulse
            shape is a 22-step rising/falling limb padded with leading zeros
            and a long recession tail.  Only --start-time is used to set the
            output filename timestamps.

  constant  Apply a constant qlat of --constant-qlat (m³/s, default 1.0)
            to every reach for every timestep in the [start-time, end-time]
            window.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


### CONSTANT DEFINITION ###

RETRO_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETROSPECTIVE_LATERAL_FIELD = "q_lateral"
RETROSPECTIVE_STREAMFLOW_FIELD = "streamflow"

# Unit-hydrograph shape shared by pulse forcing
_PULSE_SHAPE = np.array(
    [
        0.0,
        0.03,
        0.10,
        0.19,
        0.31,
        0.47,
        0.66,
        0.82,
        0.93,
        0.99,
        1.00,
        0.99,
        0.93,
        0.86,
        0.78,
        0.68,
        0.56,
        0.42,
        0.27,
        0.18,
        0.08,
        0.03,
    ]
)


def _write_runout_steps(
    last_time: pd.Timestamp,
    feature_ids,
    forcing_dir: Path,
    runout_time: int,
    forcing_file_pattern: str,
) -> None:
    """Append zero-qlat runout CSVs after the primary simulation window."""
    for i in range(1, runout_time + 1):
        t_str = (last_time + pd.Timedelta(hours=i)).strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: 0.0})
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)
        print(f"  Wrote {i}/{runout_time} runout timesteps...", end="\r", flush=True)
    if runout_time > 0:
        print(f"  Wrote {runout_time}/{runout_time} runout timesteps.   ")


def create_pulse_forcing_dataset(
    t_start: str,
    forcing_dir: str,
    hydrofabric_path: str,
    t_end: str,
    peak_qlat: float = 10000.0,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
    runout_time: int = 0,
) -> None:
    """Write per-timestep CSVs driven by a synthetic pulse scaled to *peak_qlat*.

    The pulse is applied uniformly to every reach in the hydrofabric.  When
    *t_end* is provided the unit-hydrograph shape is linearly interpolated to
    span exactly the [t_start, t_end] window; otherwise the native shape length
    determines the number of timesteps.
    """
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    feature_ids = fps["fp_id"].astype(int).values

    times = pd.date_range(t_start, t_end, freq="h")
    n = len(times)
    # Linearly interpolate the unit-hydrograph shape to n samples
    xp = np.linspace(0, 1, len(_PULSE_SHAPE))
    xi = np.linspace(0, 1, n)
    shape = np.interp(xi, xp, _PULSE_SHAPE)


    inflows = shape * peak_qlat

    for i, (t, q) in enumerate(zip(times, inflows)):
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: q})
        df.to_csv(
            forcing_dir / f"{t_str}.{forcing_file_pattern}.csv",
            index=False,
            float_format="%.15g",
        )
        print(f"  Wrote {i + 1}/{n} timesteps...", end="\r", flush=True)
    print(f"  Wrote {n}/{n} timesteps.   ")
    _write_runout_steps(
        times[-1], feature_ids, forcing_dir, runout_time, forcing_file_pattern
    )


def create_constant_forcing_dataset(
    t_start: str,
    t_end: str,
    forcing_dir: str,
    hydrofabric_path: str,
    constant_qlat: float = 1.0,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
    runout_time: int = 0,
) -> None:
    """Write per-timestep CSVs with a constant *constant_qlat* at every reach."""
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    feature_ids = fps["fp_id"].astype(int).values

    times = pd.date_range(t_start, t_end, freq="h")
    n = len(times)
    for i, t in enumerate(times):
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: constant_qlat})
        df.to_csv(
            forcing_dir / f"{t_str}.{forcing_file_pattern}.csv",
            index=False,
            float_format="%.15g",
        )
        print(f"  Wrote {i + 1}/{n} timesteps...", end="\r", flush=True)
    print(f"  Wrote {n}/{n} timesteps.   ")
    _write_runout_steps(
        times[-1], feature_ids, forcing_dir, runout_time, forcing_file_pattern
    )


def _retro_slice_to_fp_df(
    retro: xr.Dataset,
    crosswalk: pd.DataFrame,
    time: pd.Timestamp,
    retro_field: str,
    col_name: str,
) -> pd.DataFrame:
    """Select *retro_field* at *time* for the ref_fp_ids in *crosswalk* and
    map back to fp_ids, returning a ``(feature_id, col_name)`` DataFrame."""
    values = (
        retro.sel(feature_id=crosswalk["ref_fp_id"].values, time=time)[retro_field]
        .reset_coords(drop=True)
        .to_dataframe()
    )
    return (
        values.merge(crosswalk[["ref_fp_id", "fp_id"]], left_index=True, right_on="ref_fp_id")
        .rename(columns={"fp_id": "feature_id", retro_field: col_name})[["feature_id", col_name]]
        .assign(feature_id=lambda d: d["feature_id"].astype(int))
        .groupby("feature_id")
        .sum()
        .reset_index()
    )


def _build_crosswalk(hydrofabric_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return the ref_fp_id→fp_id crosswalk and all fp_ids from the hydrofabric."""
    crosswalk = gpd.read_file(hydrofabric_path, layer="reference_flowpaths", ignore_geometry=True)
    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    crosswalk = crosswalk[["ref_fp_id", "div_id"]].merge(
        fps[["fp_id", "div_id"]], on="div_id", how="left"
    )
    return crosswalk, fps["fp_id"].astype(int)


def create_hot_start_file(
    t_start: str,
    restart_dir: str,
    hydrofabric_path: str,
    offnetwork_upstreams: list[int] | None = None,
) -> None:
    """Create a hot-start (warm-start) restart pickle from retrospective streamflow.

    Reads the retrospective streamflow at *t_start* and writes a ``restart.pkl``
    into *restart_dir* containing initial flow conditions (qd0, h0, qu0, ql0)
    keyed by ``feature_id``.
    """
    restart_dir = Path(restart_dir)
    restart_dir.mkdir(parents=True, exist_ok=True)

    crosswalk, _ = _build_crosswalk(hydrofabric_path)
    retro = xr.open_zarr(RETRO_PATH, storage_options={"anon": True})

    t0 = pd.Timestamp(t_start)
    q_out = (
        retro.sel(feature_id=crosswalk["ref_fp_id"].values, time=t0)[RETROSPECTIVE_STREAMFLOW_FIELD]
        .reset_coords(drop=True)
        .to_dataframe()
    )
    q_out = (
        q_out.merge(crosswalk[["ref_fp_id", "fp_id"]], left_index=True, right_on="ref_fp_id")
        .rename(columns={"fp_id": "feature_id", RETROSPECTIVE_STREAMFLOW_FIELD: "qd0"})[["feature_id", "qd0"]]
        .assign(feature_id=lambda d: d["feature_id"].astype(int))
        .groupby("feature_id")
        .max()
        .fillna(0)
        .reset_index()
    )
    if offnetwork_upstreams:
        # If we don't zero these, then the qlat+q0 will double inflows.
        mask = q_out["feature_id"].isin(offnetwork_upstreams)
        q_out.loc[mask, "qd0"] = 0.0

    q_out["h0"] = 0.1
    q_out["qu0"] = q_out["qd0"]  # TODO: Consider actually calculating this
    q_out["ql0"] = 0
    q_out["time"] = t0
    q_out.to_pickle(restart_dir / "restart.pkl")


def _apply_crosswalk(da: xr.DataArray, crosswalk: pd.DataFrame, retro_field: str) -> pd.DataFrame:
    """Convert a bulk-loaded DataArray (time x feature_id) to a per-fp_id DataFrame.

    The DataArray should already be selected to the ref_fp_ids in *crosswalk*.
    Returns a DataFrame indexed by (time, feature_id) with the summed field values,
    ready for per-timestep slicing.
    """
    # reset_coords + reset_index flattens (time, feature_id) MultiIndex to columns
    df = da.reset_coords(drop=True).to_dataframe(name=retro_field).reset_index()
    df = (
        df.merge(crosswalk[["ref_fp_id", "fp_id"]], left_on="feature_id", right_on="ref_fp_id")
        .drop(columns=["feature_id", "ref_fp_id"])
        .rename(columns={"fp_id": "feature_id"})[["time", "feature_id", retro_field]]
        .assign(feature_id=lambda d: d["feature_id"].astype(int))
        .groupby(["time", "feature_id"], sort=False)
        .sum()
        .reset_index()
    )
    return df


def create_forcing_dataset(
    t_start: str,
    t_end: str,
    forcing_dir: str,
    hydrofabric_path: str,
    retrospective_path: str,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
    runout_time: int = 0,
    offnetwork_upstreams: list[int] | None = None,
    max_workers: int = 8,
):
    """Create a dataset of channel forcing files from retrospective data.

    Optimized for cloud Zarr stores: loads the full time×feature_id block in a
    single ``.compute()`` call rather than one S3 round-trip per timestep, then
    fans out CSV writes across *max_workers* threads.
    """
    forcing_dir: Path = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    crosswalk, all_fp_ids = _build_crosswalk(hydrofabric_path)
    retro = xr.open_zarr(retrospective_path, storage_options={"anon": True})

    # Split crosswalk: off-network boundaries use streamflow; everything else uses q_lateral
    offnetwork_set = set(offnetwork_upstreams or [])
    mask = crosswalk["fp_id"].isin(offnetwork_set)
    cw_on = crosswalk[~mask]
    cw_off = crosswalk[mask] if offnetwork_set else None

    # --- Bulk load from S3 Zarr (single batched read per field) ---
    print("Loading lateral inflow data from retrospective store (bulk read)...")
    da_on = retro.sel(
        feature_id=cw_on["ref_fp_id"].values,
        time=slice(t_start, t_end),
    )[RETROSPECTIVE_LATERAL_FIELD].compute()

    df_on = _apply_crosswalk(da_on, cw_on, RETROSPECTIVE_LATERAL_FIELD)
    del da_on  # free memory

    df_off = None
    if cw_off is not None and not cw_off.empty:
        print("Loading off-network streamflow data from retrospective store (bulk read)...")
        da_off = retro.sel(
            feature_id=cw_off["ref_fp_id"].values,
            time=slice(t_start, t_end),
        )[RETROSPECTIVE_STREAMFLOW_FIELD].compute()
        df_off = _apply_crosswalk(da_off, cw_off, RETROSPECTIVE_STREAMFLOW_FIELD)
        del da_off

    # Group by time for fast per-timestep lookup
    on_by_time = {t: grp.drop(columns="time") for t, grp in df_on.groupby("time")}
    off_by_time = (
        {t: grp.drop(columns="time") for t, grp in df_off.groupby("time")}
        if df_off is not None
        else {}
    )

    times = pd.date_range(start=t_start, end=t_end, freq="h")

    def _write_one(t: pd.Timestamp) -> None:
        t_str = t.strftime("%Y%m%d%H%M")
        parts = [on_by_time[t].rename(columns={RETROSPECTIVE_LATERAL_FIELD: t_str})]
        if t in off_by_time:
            parts.append(off_by_time[t].rename(columns={RETROSPECTIVE_STREAMFLOW_FIELD: t_str}))
        pd.concat(parts, ignore_index=True).to_csv(
            forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False
        )

    # --- Parallel CSV writes ---
    print(f"Writing {len(times)} forcing files using {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_write_one, t): t for t in times}
        completed = 0
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                raise RuntimeError(f"Failed writing timestep {futures[fut]}") from exc
            completed += 1
            print(f"  Wrote {completed}/{len(times)} timesteps...", end="\r", flush=True)
    print(f"  Wrote {len(times)}/{len(times)} timesteps.   ")

    _write_runout_steps(
        pd.Timestamp(t_end), all_fp_ids.values, forcing_dir, runout_time, forcing_file_pattern
    )


def build_forcing_dataset(
    forcing_mode: Literal["retro", "pulse", "constant"],
    t_start: str,
    t_end: str,
    forcing_dir: Path,
    hydrofabric_path: Path,
    runout_period: int = 0,
    peak_qlat: float = 0,
    constant_qlat: float = 0,
    offnetwork_upstreams: list[int] | None = None,
) -> None:
    """Build the forcing dataset and config YAML described by config."""
    if forcing_mode == "retro":
        create_forcing_dataset(
            t_start=t_start,
            t_end=t_end,
            forcing_dir=forcing_dir,
            hydrofabric_path=hydrofabric_path,
            retrospective_path=RETRO_PATH,
            runout_time=runout_period,
            offnetwork_upstreams=offnetwork_upstreams,
        )
    elif forcing_mode == "pulse":
        create_pulse_forcing_dataset(
            t_start=t_start,
            t_end=t_end,
            forcing_dir=forcing_dir,
            hydrofabric_path=hydrofabric_path,
            peak_qlat=peak_qlat,
            runout_time=runout_period,
        )
    elif forcing_mode == "constant":
        create_constant_forcing_dataset(
            t_start=t_start,
            t_end=t_end,
            forcing_dir=forcing_dir,
            hydrofabric_path=hydrofabric_path,
            constant_qlat=constant_qlat,
            runout_time=runout_period,
        )
    else:
        raise ValueError(
            f"Unknown forcing_mode '{forcing_mode}'. Choose retro, pulse, or constant."
        )


def main():
    """Enter via CLI."""
    parser = argparse.ArgumentParser(description="Generate forcing dataset for a case.")

    parser.add_argument(
        "--hf-path",
        required=True,
        help="Path to the NHF GeoPackage.",
    )

    parser.add_argument(
        "--forcing-dir",
        required=True,
        help="Directory to write per-timestep forcing CSVs.",
    )

    parser.add_argument(
        "--start-time",
        default="2009-12-12 00:00",
        help="Simulation start time (e.g. '2009-12-12' or '2009-12-12 06:00').",
    )

    parser.add_argument(
        "--end-time",
        default="2009-12-29 00:00",
        help="Simulation end time (e.g. '2009-12-29' or '2009-12-29 12:00').",
    )

    parser.add_argument(
        "--forcing-mode",
        default="retro",
        choices=["retro", "pulse", "constant"],
        help=(
            "Forcing generation mode. "
            "'retro' (default): pull lateral inflows from the NWM v3 retrospective store. "
            "'pulse': apply a synthetic unit-hydrograph pulse to all reaches. "
            "'constant': apply a constant qlat to all reaches for every timestep."
        ),
    )

    parser.add_argument(
        "--runout-period",
        type=int,
        default=0,
        help="Hours of zero-qlat runout to append after the primary simulation window.",
    )

    parser.add_argument(
        "--peak-qlat",
        type=float,
        default=10_000.0,
        help="Peak discharge (m³/s) for the synthetic pulse. Only used when --forcing-mode=pulse.",
    )

    parser.add_argument(
        "--constant-qlat",
        type=float,
        default=1.0,
        help="Constant lateral inflow (m³/s) per reach. Only used when --forcing-mode=constant.",
    )

    args = parser.parse_args()

    build_forcing_dataset(
        forcing_mode=args.forcing_mode,
        t_start=args.start_time,
        t_end=args.end_time,
        forcing_dir=Path(args.forcing_dir),
        hydrofabric_path=Path(args.hf_path),
        runout_period=args.runout_period,
        peak_qlat=args.peak_qlat,
        constant_qlat=args.constant_qlat,
    )


if __name__ == "__main__":
    main()
