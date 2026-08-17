import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import yaml

from nwm_routing.nhf_routing import nhf_routing

from test.nhf.utils.make_configs import Config


def get_usgs_station_ids(domain_path: Path) -> list[str]:
    """Read the gages layer and return USGS site numbers."""
    gages = gpd.read_file(domain_path, layer="gages", ignore_geometry=True)
    return gages["site_no"].dropna().astype(str).tolist()


def run_troute(config_path: Path) -> None:
    """Run t-route from config_path's parent directory, raising on failure."""
    os.chdir(config_path.parent)
    nhf_routing(["-f", config_path.name])


def has_files(directory: Path, pattern: str = "*") -> bool:
    """Return True if *directory* exists and contains at least one matching file."""
    return directory.is_dir() and any(directory.glob(pattern))


def delete_outputs(output_dir: Path) -> None:
    """Remove all .nc files from *output_dir* so stale results cannot mask failures."""
    if output_dir.is_dir():
        for nc_file in output_dir.glob("*.nc"):
            nc_file.unlink()


def skip_if_not_built(cfg: Config) -> None:
    """Skip the calling test if the case data has not been prepped yet."""
    passing = True
    if not cfg.config_path.exists():
        message = f"{cfg.config_path} does not exist"
        passing = False
    elif not has_files(cfg.channel_forcing_dir, cfg.qlat_file_pattern):
        message = f"missing forcing files in {cfg.channel_forcing_dir}"
        passing = False
    elif not cfg.domain_path.exists():
        message = f"{cfg.domain_path} does not exist"
        passing = False
    if not passing:
        pytest.skip(
            message
            + "run: python -m test.nhf._prep_tests --nhf-gpkg <path> --test `case_name`"
        )


def load_output(output_dir: Path) -> xr.Dataset:
    """Load and concatenate all .nc files in output_dir into a single Dataset."""
    nc_files = sorted(output_dir.glob("*.nc"))
    if not nc_files:
        pytest.fail(f"no output .nc files found in {output_dir}")
    return xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in nc_files], dim="time"
    )


def load_lakeout(lakeout_dir: Path) -> xr.Dataset:
    """Load and concatenate all lakeout .nc files into a single Dataset."""
    nc_files = sorted(lakeout_dir.glob("*.nc"))
    if not nc_files:
        pytest.fail(f"No lakeout files found in {lakeout_dir}")
    if len(nc_files) == 1:
        return xr.open_dataset(nc_files[0], engine="netcdf4")
    return xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in nc_files], dim="time"
    )


def assert_lakeout(
    lakeout_dir: Path,
    expected_feature_count: int | None = None,
    outflow_bounds: dict[int, tuple[float, float]] | None = None,
) -> None:
    """Validate lakeout NetCDF files."""
    ds = load_lakeout(lakeout_dir)

    expected_vars = {"reservoir_type", "inflow", "outflow", "water_sfc_elev"}
    violations: list[str] = []

    # Check expected variables exist.
    missing = expected_vars - set(ds.data_vars)
    if missing:
        violations.append(f"missing variables {missing}")

    # Check feature_id dimension.
    if "feature_id" not in ds.dims:
        violations.append("missing 'feature_id' dimension")
    elif expected_feature_count is not None and ds.sizes["feature_id"] != expected_feature_count:
        violations.append(
            f"expected {expected_feature_count} features, "
            f"got {ds.dims['feature_id']}"
        )

    # Check for NaNs in numeric data variables.
    for var in ("inflow", "outflow", "water_sfc_elev"):
        if var in ds.data_vars and np.isnan(ds[var].values).any():
            violations.append(f"NaN values found in '{var}'")

    # Check outflow bounds.
    if outflow_bounds and "outflow" in ds.data_vars and "feature_id" in ds.dims:
        for fid, (lo, hi) in outflow_bounds.items():
            if fid not in ds["feature_id"].values:
                violations.append(f"feature_id {fid} not in dataset")
                continue
            vals = ds["outflow"].sel(feature_id=fid).values
            max_val = float(np.nanmax(vals))
            if not (lo <= max_val <= hi):
                violations.append(
                    f"feature_id {fid} max outflow "
                    f"{max_val:.4f} outside [{lo}, {hi}]"
                )

    ds.close()

    if violations:
        pytest.fail("Lakeout validation failed:\n" + "\n".join(violations))


def assert_peak_bounds(
    output_dir: Path, peak_bounds: dict[int, tuple[float, float]]
) -> None:
    """Assert that peak flow for each feature_id falls within (lo, hi)."""
    ds = load_output(output_dir)

    violations = []
    for feature_id, (lo, hi) in peak_bounds.items():
        flows = ds["flow"].sel(feature_id=feature_id).values
        peak = float(np.nanmax(flows))
        if not (lo <= peak <= hi):
            violations.append(
                f"feature_id {feature_id}: observed peak {peak:.3f} outside [{lo}, {hi}]"
            )
    if violations:
        pytest.fail("Peak flow bounds violated:\n" + "\n".join(violations))


def assert_output_dimensions_and_validity(output_path: Path, hf_path: Path, cfg_path: Path) -> None:
    """Check that output data exists for all required timesteps and reaches and that data has no nans."""
    # Load output
    ds = load_output(output_path)

    # Get expected feature IDs from hydrofabric
    fps = gpd.read_file(hf_path, layer="flowpaths", ignore_geometry=True, columns=["fp_id"])
    expected_ids = set(fps["fp_id"].astype(int).values)

    # Get expected number of timesteps from config
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    nts = config["compute_parameters"]["forcing_parameters"]["nts"]
    dt = config["compute_parameters"]["forcing_parameters"]["dt"]
    output_frequency = config["output_parameters"]["stream_output"]["stream_output_internal_frequency"]
    expected_prints = int((nts * (dt / 60)) / output_frequency)

    violations = []

    # Check timestep count
    actual_prints = ds.sizes["time"]
    if actual_prints != expected_prints:
        violations.append(f"expected {expected_prints} output timesteps, got {actual_prints}")

    # Check that all expected reaches are present
    output_ids = set(ds["feature_id"].values)
    missing_ids = expected_ids - output_ids
    if missing_ids:
        violations.append(f"{len(missing_ids)} reaches missing from output: {sorted(missing_ids)}")

    # Check for NaN values in flow
    nan_count = int(ds["flow"].isnull().sum())
    if nan_count > 0:
        violations.append(f"flow contains {nan_count} NaN values")

    if violations:
        pytest.fail("Output validation failed:\n" + "\n".join(violations))
