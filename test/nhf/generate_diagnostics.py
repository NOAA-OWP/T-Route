import argparse
import os
import random
import warnings
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from troute.NHF import NHF
import xarray as xr
import yaml
from matplotlib import pyplot as plt
from troute.config import Config

SAMPLE_RANDOM_SEED = 11
random.seed(SAMPLE_RANDOM_SEED)

### DATA ACCESS ###


class RunContext:
    """Relevant data for tests of a t-route run.

    This context bundles model outputs, forcings, and network mappings needed
    to compute reach-scale and network-integrated routing performance diagnostics.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.run_id = self.config_path.stem

        # Load yaml
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Emulate running from config dir
        cdir = Path(os.curdir).resolve()
        os.chdir(self.config_path.parent)
        troute_configuration = Config.with_strict_mode(**self.config)
        config_dict = troute_configuration.dict()

        compute_parameters = config_dict.get("compute_parameters")
        network_topology_parameters = config_dict.get("network_topology_parameters")
        output_parameters = config_dict.get("output_parameters")

        preprocessing_parameters = network_topology_parameters.get(
            "preprocessing_parameters"
        )
        supernetwork_parameters = network_topology_parameters.get(
            "supernetwork_parameters"
        )
        waterbody_parameters = network_topology_parameters.get("waterbody_parameters")
        forcing_parameters = compute_parameters.get("forcing_parameters")
        restart_parameters = compute_parameters.get("restart_parameters")
        hybrid_parameters = compute_parameters.get("hybrid_parameters")
        data_assimilation_parameters = compute_parameters.get(
            "data_assimilation_parameters"
        )

        self.network = NHF(
            supernetwork_parameters,
            waterbody_parameters,
            data_assimilation_parameters,
            restart_parameters,
            compute_parameters,
            forcing_parameters,
            hybrid_parameters,
            preprocessing_parameters,
            output_parameters,
            verbose=True,
        )

        self.links_df = self.network._dataframe
        os.chdir(cdir)

    @property
    def result_output_dir(self) -> Path:
        """Directory where test results will be saved."""
        p = self.config_root / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def config_root(self) -> Path:
        """Root directory for config file."""
        return self.config_path.parent

    @property
    def reference_data_path(self) -> Path:
        """Directory where reference data is stored, if it exists."""
        return self.config_root / self.run_id / "gage_reference_data.nc"

    @property
    def diagnostic_plot_dir(self) -> Path:
        """Directory where diagnostic plots are written."""
        p = self.config_root / self.run_id / "diagnostic_plots"
        p.mkdir(exist_ok=True)
        return p

    @property
    def diagnostic_test_path(self) -> Path:
        """Directory where diagnostic plots are written."""
        p = self.config_root / self.run_id / "diagnostic_tests"
        return p

    @cached_property
    def nhf_discretization_len(self) -> float:
        """Target length for routed reaches."""
        return (
            self.config.get("network_topology_parameters")
            .get("supernetwork_parameters")
            .get("nhf_discretization_len", 300.0)
        )

    @cached_property
    def max_stream_order(self) -> int:
        """Maximum stream order in the domain."""
        return self.flowpaths_gdf["stream_order"].max()

    @cached_property
    def dt(self) -> float:
        """Timestep of t-route run in seconds."""
        return self.config["compute_parameters"]["forcing_parameters"]["dt"]

    @cached_property
    def nts(self) -> float:
        """Number of timesteps in the run."""
        return self.config["compute_parameters"]["forcing_parameters"]["nts"]

    @cached_property
    def qts_subdivisions(self) -> int:
        """Number of times each timestep is subdivided for routing."""
        # Note that this follows the logic in AbstractNetwork.py (line 909).
        # If that is ever updated to read YAML data, switch to
        # return self.config["compute_parameters"]["forcing_parameters"]["qts_subdivisions"]
        dt_qlat_timedelta = self.forcing_data.index[1] - self.forcing_data.index[0]
        dt_qlat = dt_qlat_timedelta.seconds
        return dt_qlat / self.dt

    @cached_property
    def _output_paths(self) -> list[Path]:
        """Paths to routed result netCDF files."""
        output_path = (
            self.config_root
            / self.config["output_parameters"]["stream_output"][
                "stream_output_directory"
            ]
        )
        return sorted(output_path.glob("*.nc"))

    @cached_property
    def routed_results(self) -> xr.Dataset:
        """Load results from t-route run."""
        flows = [xr.open_dataset(p, engine="netcdf4") for p in self._output_paths]
        return xr.concat(flows, dim="time")

    def subset_routed_results(self, feature_ids: list[int]) -> None:
        """Load only specific feature IDs from results into memory."""
        subsets = []
        for p in self._output_paths:
            with xr.open_dataset(p, engine="netcdf4") as ds:
                subsets.append(ds.sel(feature_id=feature_ids).load())
        self.__dict__["routed_results"] = xr.concat(subsets, dim="time")

    @cached_property
    def forcing_data(self) -> pd.DataFrame:
        """Load channel forcing data that was used to run t-route."""
        forcing_dir = (
            self.config_root
            / self.config["compute_parameters"]["forcing_parameters"][
                "qlat_input_folder"
            ]
        )
        forcing_files = sorted(forcing_dir.glob("*.csv"))
        t0 = datetime.strptime(forcing_files[0].stem.split(".")[0], "%Y%m%d%H%M")
        tmax = t0 + pd.to_timedelta(self.nts * self.dt, unit="s")
        forcing_files = [
            f
            for f in forcing_files
            if datetime.strptime(f.stem.split(".")[0], "%Y%m%d%H%M") < tmax
        ]
        forcing = pd.concat(
            [pd.read_csv(f).sort_values("feature_id") for f in forcing_files], axis=1
        )
        columns = forcing["feature_id"].values[:, 0]
        forcing = forcing.drop(columns=["feature_id"]).T
        forcing.columns = columns
        forcing.index = pd.to_datetime(forcing.index)
        # Final trim
        forcing = forcing.loc[(forcing.index >= t0) & (forcing.index < tmax)]

        all_fp = forcing.columns.values
        self.network._build_qlateral_array_direct(forcing.T)
        forcing = self.network._qlateral
        forcing = pd.merge(
            forcing,
            self.network.dataframe[["fp_id"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        forcing = forcing.groupby("fp_id").agg("sum")
        forcing = forcing.T
        missing_fps = list(set(all_fp).difference(forcing.columns))
        if missing_fps:
            forcing = pd.concat([forcing, pd.DataFrame(0, index=forcing.index, columns=missing_fps)], axis=1)

        return forcing

    @cached_property
    def hydrofabric_path(self) -> Path:
        """Path to hydrofabric gpkg."""
        return (
            self.config_root
            / self.config["network_topology_parameters"]["supernetwork_parameters"][
                "geo_file_path"
            ]
        )

    @cached_property
    def virtual_flowpaths_gdf(self) -> gpd.GeoDataFrame:
        """Load virtual flowpaths layer."""
        return gpd.read_file(
            self.hydrofabric_path, layer="virtual_flowpaths", ignore_geometry=True
        )

    @cached_property
    def reference_flowpaths_gdf(self) -> gpd.GeoDataFrame:
        """Load reference flowpaths layer."""
        return gpd.read_file(
            self.hydrofabric_path,
            layer="reference_flowpaths",
            ignore_geometry=True,
        )

    @cached_property
    def flowpaths_gdf(self) -> gpd.GeoDataFrame:
        """Load flowpaths layer."""
        return gpd.read_file(self.hydrofabric_path, layer="flowpaths")

    @cached_property
    def fp_length_mapping(self) -> dict[int, float]:
        """Make a mapping from flowpath ID to its length."""
        return dict(
            zip(
                self.flowpaths_gdf["fp_id"].to_numpy(),
                self.flowpaths_gdf["length_km"].to_numpy(),
            )
        )

    @cached_property
    def us_mapping(self) -> dict[int, list[int]]:
        """Make a mapping from flowpath ID to the list of upstream flowpath IDs."""
        df = self.flowpaths_gdf
        return (
            df[df.iloc[:, 1].notna()]
            .groupby(df.columns[1])[df.columns[0]]
            .apply(list)
            .to_dict()
        )


### HELPER FUNCTIONS ###


def sample_reaches(
    run_context: RunContext,
    n_samples: int = 500,
    pct_length: float = 0.1,
    pct_slope: float = 0.1,
) -> dict[int, dict[str, Any]]:
    """Create a balanced random sample of reaches."""
    # Reserve pct_length% of samples for shortest and longest reaches
    # Reserve pct_slope% of samples for flatest and steepest reaches
    # Balance remaining % of samples across stream orders
    n_length = int(n_samples * pct_length)
    n_slope = int(n_samples * pct_slope)
    n_stream_order_reaches = n_samples - n_length - n_slope
    samples_per_stream_order = n_stream_order_reaches // run_context.max_stream_order
    working_reach_list = set()

    mask = (run_context.routed_results["flow"] > 0).any(dim="time")
    nonzero_fps = run_context.routed_results["feature_id"].where(mask, drop=True).values.astype(int)
    in_links = run_context.links_df["fp_id"].unique()
    valid_fp = set(in_links).intersection(nonzero_fps)
    tmp_flowpaths = run_context.flowpaths_gdf[
        run_context.flowpaths_gdf["fp_id"].isin(valid_fp)
    ]


    # Sample from stream orders
    for i in range(1, run_context.max_stream_order + 1):
        stream_order_reaches = tmp_flowpaths.loc[
            tmp_flowpaths["stream_order"] == i, "fp_id"
        ]
        stream_order_sample = random.sample(
            list(stream_order_reaches),
            min(samples_per_stream_order, len(stream_order_reaches)),
        )
        working_reach_list.update(stream_order_sample)

    # Sample from reach lengths
    short_reaches = tmp_flowpaths.nsmallest(n_length // 2, "length_km")["fp_id"]
    long_reaches = tmp_flowpaths.nlargest(n_length - len(short_reaches), "length_km")[
        "fp_id"
    ]
    working_reach_list.update(short_reaches)
    working_reach_list.update(long_reaches)

    # Sample from reach slopes
    flat_reaches = tmp_flowpaths.nsmallest(n_slope // 2, "slope")["fp_id"]
    working_reach_list.update(flat_reaches)
    steep_reaches = tmp_flowpaths.nlargest(n_slope - len(flat_reaches), "slope")[
        "fp_id"
    ]
    working_reach_list.update(steep_reaches)

    # Pad out list with true randoms
    remaining_reaches = n_samples - len(working_reach_list)
    if remaining_reaches > 0:
        options = set(tmp_flowpaths["fp_id"]) - working_reach_list
        random_reaches = random.sample(
            list(options), min(remaining_reaches, len(options))
        )
        working_reach_list.update(random_reaches)

    return attribute_reaches(run_context, working_reach_list)


def attribute_reaches(
    run_context: RunContext, reach_list: set[int]
) -> dict[int, dict[str, Any]]:
    """Make a dictionary mapping reach ID to its attributes."""
    reach_attributes = {}
    for reach in reach_list:
        fp_gdf_row = run_context.flowpaths_gdf.loc[
            run_context.flowpaths_gdf["fp_id"] == reach
        ]
        reach_attributes[reach] = {
            "So": fp_gdf_row["slope"].item(),
            "stream_order": run_context.flowpaths_gdf.loc[
                run_context.flowpaths_gdf["fp_id"] == reach, "stream_order"
            ].item(),
            "dx": fp_gdf_row["length_km"].item() * 1000,
            "n": fp_gdf_row["n"].item(),
            "Cs": fp_gdf_row["chslp"].item(),
            "Bw": fp_gdf_row["btmwdth"].item(),
            "Tw": fp_gdf_row["topwdth"].item(),
            "TwCC": fp_gdf_row["topwdthcc"].item(),
            "nCC": fp_gdf_row["ncc"].item(),
        }
    return reach_attributes


def hydraulic_geometry(
    h: Union[float, np.ndarray],
    bw: float,
    tw: float,
    twcc: float,
    cs: float,
    n: float,
    ncc: float,
    so: float,
    dx: float,
):
    ### DIRECT PORT OF src/kernel/muskingum/MCsingleSegStime_f2py_NOLOOP.f90 ###
    # Convert rise over run to run over rise
    if cs == 0:
        z = 1
    else:
        z = 1 / cs

    # Calculate bankfull depth
    if bw > tw:
        bfd = bw / 0.00001
    elif bw == tw:
        bfd = bw / (2.0 * z)
    else:
        bfd = (tw - bw) / (2.0 * z)

    # depth below and above bankfull
    h_gt_bf = np.maximum(h - bfd, 0.0)
    h_lt_bf = np.minimum(bfd, h)

    # Exception for zero floodplain width
    mask_zero_twcc = (h_gt_bf > 0) & (twcc <= 0)
    h_gt_bf = np.where(mask_zero_twcc, 0.0, h_gt_bf)
    h_lt_bf = np.where(mask_zero_twcc, h, h_lt_bf)

    # Compute geometry
    twl = bw + 2.0 * z * h
    AREA = (bw + h_lt_bf * z) * h_lt_bf
    WP = bw + 2.0 * h_lt_bf * np.sqrt(1.0 + z**2)

    AREAC = twcc * h_gt_bf
    WPC = np.where(h_gt_bf > 0, twcc + 2.0 * h_gt_bf, 0.0)

    A = AREA + AREAC
    P = WP + WPC

    # Hydraulic radius
    R = (AREA + AREAC) / (WP + WPC)

    # Roughness
    n_eff = (WP * n + WPC * ncc) / P

    # Manning discharge
    Q = (1.0 / n_eff) * A * (R ** (2.0 / 3.0)) * np.sqrt(so)

    # Celerity
    term = (5.0 / 3.0) * R ** (2.0 / 3.0) - (2.0 / 3.0) * R ** (5.0 / 3.0) * (
        2.0 * np.sqrt(1.0 + z**2) / (bw + 2.0 * h_lt_bf * z)
    )

    ck_channel = (np.sqrt(so) / n) * term

    ck_fp = (np.sqrt(so) / ncc) * (5.0 / 3.0) * (h_gt_bf ** (2.0 / 3.0))

    with warnings.catch_warnings(record=True):
        ck = np.where(
            (h > bfd) & (twcc > 0) & (ncc > 0),
            (ck_channel * AREA + ck_fp * AREAC) / (A),
            ck_channel,
        )

        ck = np.maximum(ck, 0.0)

        # Muskingum–Cunge X
        X = np.where(
            ck > 0,
            0.5 * (1.0 - (Q / (2.0 * twl * so * ck * dx))),
            0.5,
        )

    return {
        "twl": twl,
        "R": R,
        "AREA": AREA,
        "AREAC": AREAC,
        "WP": WP,
        "WPC": WPC,
        "h_lt_bf": h_lt_bf,
        "h_gt_bf": h_gt_bf,
        "q": Q,
        "ck": ck,
        "x": X,
    }


def collect_needed_feature_ids(
    run_context: RunContext,
    reaches: dict[int, dict],
    max_walk: float = 50.0,
) -> set[int]:
    """Collect all feature IDs that will be accessed during diagnostics."""
    needed = set(reaches.keys())
    for reach_id in reaches:
        _walk_upstream_ids(reach_id, run_context, needed, 0, max_walk)
    return needed


def _walk_upstream_ids(
    reach: int,
    run_context: RunContext,
    needed: set[int],
    cur_dist: float,
    max_walk: float,
) -> None:
    """Recursively collect upstream feature IDs within walk distance."""
    cur_dist += run_context.fp_length_mapping.get(reach, 0)
    if reach not in run_context.us_mapping:
        return
    us = run_context.us_mapping[reach]
    needed.update(us)
    if cur_dist < max_walk:
        for u in us:
            _walk_upstream_ids(u, run_context, needed, cur_dist, max_walk)


def generate_reach_diagnostics(
    run_context: RunContext,
    reach_id: int,
    reach_attributes: dict[str, float],
    max_walk: float = 50.0,
    plot: bool = True,
) -> dict[str, float]:
    """Test mass conservation for a given reach."""
    # Load data
    local_qin, _ = virtual_flowpath_mass_conservation_network(
        reach_id, run_context, max_walk=0
    )
    network_qin, max_walk_actual = virtual_flowpath_mass_conservation_network(
        reach_id, run_context, max_walk=max_walk
    )
    qout_hydrograph = run_context.routed_results.sel(feature_id=reach_id)["flow"].values
    qout_hydrograph = np.nan_to_num(qout_hydrograph)
    outflow = qout_hydrograph.sum()
    qlat, qus = get_inflows(run_context, reach_id)
    qin_hydrograph = qlat + qus

    # Summary stats for routing parameters
    dx_link = run_context.links_df[run_context.links_df["fp_id"] == reach_id][
        "dx"
    ].values[0]
    reach_attributes["dx"] = dx_link
    results = get_routing_stats(
        reach_attributes,
        run_context.routed_results.sel(feature_id=reach_id)["depth"].values,
        qout_hydrograph,
        run_context.dt,
    )
    qreroute = np.ones_like(qout_hydrograph)

    # Check mass conservation at-reach
    results["local_mass_conservation_error"] = (
        100 * abs(local_qin - outflow) / outflow if outflow > 0 else np.nan
    )

    # Check mass conservation across network
    results["network_mass_conservation_error"] = (
        100 * abs(network_qin - outflow) / outflow if outflow > 0 else np.nan
    )
    results["network_mass_conservation_walk_dist"] = max_walk_actual

    # Calculate hydrograph lag
    timesteps = run_context.routed_results["time"].values
    t0 = timesteps[0]
    dt_seconds = (timesteps - t0) / np.timedelta64(1, "s")
    qin_sum = np.sum(qin_hydrograph)
    qout_sum = np.sum(qout_hydrograph)
    inflow_centroid = (
        np.sum(qin_hydrograph * dt_seconds) / qin_sum if qin_sum > 0 else 0
    )
    outflow_centroid = (
        np.sum(qout_hydrograph * dt_seconds) / qout_sum if qout_sum > 0 else 0
    )
    results["hydrograph_lag"] = outflow_centroid - inflow_centroid
    results["normalized_lag"] = results["hydrograph_lag"] / (
        reach_attributes["dx"] * 1000
    )

    # Check other hydrograph stats
    results["negative_qout"] = np.min(qout_hydrograph) < 0
    results["pct_attenuation"] = (
        100 * (1 - (np.max(qout_hydrograph) / np.max(qin_hydrograph)))
        if np.max(qin_hydrograph) > 0
        else np.nan
    )
    results["acceleration"] = np.max(qout_hydrograph) > np.max(qin_hydrograph)

    # Plot triggers
    a = results["pct_attenuation"] < -3
    b = results["local_mass_conservation_error"] > 10
    c = results["network_mass_conservation_error"] > 10
    if a or b or c:
        plot_hydrograps(
            qin_hydrograph, qout_hydrograph, qreroute, reach_id, run_context
        )
    return results


def virtual_flowpath_mass_conservation_network(
    reach: int, run_context: RunContext, cur_dist: float = 0, max_walk: float = 10
):
    cur_dist += run_context.fp_length_mapping[reach]
    out_calc = run_context.forcing_data[reach].values.sum()
    if reach not in run_context.us_mapping:
        return out_calc, cur_dist
    elif cur_dist > max_walk:
        us = run_context.us_mapping[reach]
        us_streamflow = (
            run_context.routed_results["flow"]
            .sel(feature_id=us)
            .sum(dim="feature_id")
            .values
        )
        out_calc += us_streamflow.sum()
        return out_calc, cur_dist
    elif cur_dist < max_walk:
        us = run_context.us_mapping[reach]
        us_q = [
            virtual_flowpath_mass_conservation_network(
                u, run_context, cur_dist, max_walk
            )
            for u in us
        ]
        out_calc += sum([q[0] for q in us_q])
        return out_calc, max([q[1] for q in us_q])


def get_inflows(run_context: RunContext, reach_id: int) -> np.ndarray:
    """Get inflows to a reach by summing forcing and upstream flow."""
    local_qin = run_context.forcing_data[reach_id].values
    if reach_id not in run_context.us_mapping:
        return local_qin, np.zeros_like(local_qin)
    else:
        us = run_context.us_mapping[reach_id]
        us_q = (
            run_context.routed_results["flow"]
            .sel(feature_id=us)
            .sum(dim="feature_id")
            .values
        )
        return local_qin, us_q


def get_routing_stats(
    reach_attributes: dict, depths: np.ndarray, qs: np.ndarray, dt: float
) -> dict[str, float]:
    results = {}

    geom_series = hydraulic_geometry(
        depths,
        reach_attributes["Bw"],
        reach_attributes["Tw"],
        reach_attributes["TwCC"],
        reach_attributes["Cs"],
        reach_attributes["n"],
        reach_attributes["nCC"],
        reach_attributes["So"],
        reach_attributes["dx"],
    )
    geom_series["cn"] = geom_series["ck"] * (dt / reach_attributes["dx"])
    for i in ["cn", "ck", "x"]:
        s = geom_series[i]
        results[f"{i}_mean"] = s.mean()
        results[f"{i}_median"] = np.median(s)
        results[f"{i}_max"] = s.max()
        results[f"{i}_min"] = s.min()
        results[f"{i}_std"] = s.std()

    # Get ratio of reach length to courant ideal
    # Uses method of Ponce and Theurer (1982) Accuracy Criteria in Diffusion Routing
    qref = (qs.max() + qs.min()) / 2
    ref_ind = np.argmin(np.abs(qs - qref))
    qref_actual = qs[ref_ind]
    cref = geom_series["ck"][ref_ind]
    twref = geom_series["twl"][ref_ind]
    if cref > 0:
        dxc = dt * cref
        dxd = (qref_actual / twref) / (reach_attributes["So"] * cref)
        dxmax = 0.5 * (dxc + dxd)
        cmax = geom_series["ck"].max()
        dxmin = cmax * dt
        ideal_dx = max([dxmin, dxmax])
        results["dx_ratio"] = reach_attributes["dx"] / ideal_dx
    else:
        results["dx_ratio"] = None
    return results


### PLOTS ###


def plot_hydrograps(
    qin: np.ndarray,
    qout: np.ndarray,
    qreroute: np.ndarray,
    reach_id: int,
    run_context: RunContext,
):
    fig, ax = plt.subplots()
    timesteps = run_context.routed_results["time"].values
    ax.plot(timesteps, qin, label="qin", color="blue")
    ax.plot(timesteps, qout, label="qout", color="orange")
    ax.plot(timesteps, qreroute, label="reroute", color="green", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(f"Reach {reach_id}")
    ax.legend()
    fig.tight_layout()
    plot_dir = run_context.result_output_dir / "hydrograph_plots"
    plot_dir.mkdir(exist_ok=True)
    fig.savefig(plot_dir / f"{reach_id}.png")
    plt.close(fig)


def generate_reference_gage_plots(run_context: RunContext) -> None:
    """Generate plots comparing t-route to retrospective and USGS observed datasets."""
    # Check if reference data exists. If not, skip.
    if not run_context.reference_data_path.exists():
        return

    ref_out_dir = run_context.result_output_dir / "reference_gages"
    ref_out_dir.mkdir(exist_ok=True)

    ds = xr.open_dataset(run_context.reference_data_path)
    for site_no in ds["gage"].values:
        sub_ds = ds.sel(gage=site_no)
        site_no = sub_ds["site_no"].item()
        fp_id = sub_ds["fp_id"].item()
        last_t = sub_ds["time"].values[-1]

        # Load t-route data
        trdf = (
            run_context.routed_results.sel(feature_id=fp_id)
            .to_dataframe()
            .reset_index()
        )
        trdf = trdf[trdf["time"] <= last_t]  # Remove runout period

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            sub_ds["time"].values,
            sub_ds["usgs_q"],
            label=f"USGS Site No {site_no}",
            c="k",
        )
        ax.plot(
            sub_ds["time"].values,
            sub_ds["retrospective_q"] * 35.31,
            label="NWM Retrospective Streamflow",
            color="k",
            ls="--",
        )  # convert from m3/s to cfs
        ax.plot(
            trdf.time,
            trdf["flow"] * 35.31,
            label="NHF T-route Streamflow",
            color="k",
            marker="o",
            markevery=10,
            markerfacecolor="w",
        )  # convert from m3/s to cfs
        ax.set_xlabel("Time")
        ax.set_ylabel("Discharge (cfs)")
        ax.legend()
        ax.set_facecolor("whitesmoke")
        fig.tight_layout()
        fig.savefig(ref_out_dir / f"{fp_id}.png")


def plot_reach_mass_conservation(df: pd.DataFrame, out_path: Path) -> None:
    """Plot mass conservation across a reach."""
    fig, ax = plt.subplots()
    rng = (
        df["local_mass_conservation_error"].min(),
        df["local_mass_conservation_error"].max(),
    )
    rng = (df["local_mass_conservation_error"].min(), 100)
    sns.kdeplot(
        df,
        x="local_mass_conservation_error",
        hue="stream_order",
        palette="viridis",
        clip=rng,
        ax=ax,
    )
    ax.axvline(0, ls="dashed", c="k", alpha=0.3)
    ax.set_xlabel("Volume Difference Across Reach (as % of outflow volume)")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()
    fig.savefig(out_path / "local_mass_conservation_error.png", dpi=300)


def plot_reach_mass_conservation_vs_dx(df: pd.DataFrame, out_path: Path) -> None:
    """Plot mass conservation across a reach versus reach length."""
    fig, ax = plt.subplots()
    sns.scatterplot(df, x="dx", y="local_mass_conservation_error")
    ax.set_ylabel("Volume Difference Across Reach (as % of outflow volume)")
    ax.set_xlabel("Reach Length (m)")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()
    fig.savefig(out_path / "local_mass_conservation_error_vs_dx.png", dpi=300)


def plot_network_mass_conservation(df: pd.DataFrame, out_path: Path) -> None:
    """Plot mass conservation across a network walk."""
    fig, ax = plt.subplots()
    ax.set_ylabel("Volume Difference Across Reaches (as % of outflow volume)")
    ax.set_xlabel("Maximum Distance Walked Upstream (km)")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    sns.scatterplot(
        df,
        x="network_mass_conservation_walk_dist",
        y="network_mass_conservation_error",
        hue="stream_order",
        palette="viridis",
    )
    fig.tight_layout()
    fig.savefig(out_path / "network_mass_conservation_error_vs_walk_dist.png", dpi=300)


def plot_attenuation(df: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of attenuation values across a reach."""
    fig, ax = plt.subplots()
    min_val = np.floor(df["pct_attenuation"].min() * 2) / 2
    max_val = np.ceil(df["pct_attenuation"].max() * 2) / 2
    if np.isnan(min_val):
        ax.text(x=0, y=0, s="Attenuation was NAN")
    else:
        bins = np.arange(min_val, max_val + 0.5, 0.5)
        sns.histplot(df, x="pct_attenuation", ax=ax, bins=bins)
        ax.axvline(0, ls="dashed", c="k", alpha=0.3)
        ax.set_xlabel("Attenuation (as % of peak inflow)")
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_facecolor("whitesmoke")
        fig.tight_layout()
    fig.savefig(out_path / "attenuation_histogram.png", dpi=300)


def plot_courant(df: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of Courant number for each reach."""
    fig, ax = plt.subplots()
    bounds = (0, df["cn_max"].max())
    sns.kdeplot(
        df, x="cn_median", fill=True, ax=ax, label="median over hydrograph", clip=bounds
    )
    sns.kdeplot(
        df, x="cn_mean", fill=True, ax=ax, label="mean over hydrograph", clip=bounds
    )
    sns.kdeplot(
        df, x="cn_max", fill=True, ax=ax, label="max over hydrograph", clip=bounds
    )
    ax.set_xlabel("Courant Number")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "courant_number_distributions.png", dpi=300)


def plot_celerity_vs_lag(df: pd.DataFrame, out_path: Path) -> None:
    """Plot mean celerity vs hydrograph lag rate."""
    fig, ax = plt.subplots()
    df["inv_normalized_lag"] = 1 / df["normalized_lag"]
    sns.scatterplot(
        df,
        x="mean_celerity",
        y="inv_normalized_lag",
        hue="stream_order",
        palette="viridis",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Hydrograph Translation Rate (m/s)")
    ax.set_xlabel("Mean Celerity Over Hydrograph (m/s)")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()

    fig.savefig(out_path / "hydrograph_lag_and_celerity.png", dpi=300)


def plot_optimal_reach_length(df: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of reach lengths relative to Ponce optimal."""
    fig, ax = plt.subplots()
    df["dx_ratio_clip"] = df["dx_ratio"].clip(0, 75)
    sns.histplot(df, x="dx_ratio_clip", ax=ax, bins=25, stat="percent")
    ax2 = ax.twinx()
    ax2.plot(np.sort(df["dx_ratio_clip"]), np.linspace(0, 100, len(df)), color="k")
    ax.axvline(1, ls="dashed", c="k", alpha=0.3)
    ax.set_xlabel("Ratio of Reach Length to Ideal DX")
    ax.set_ylabel("Percentage of Reaches")
    ax2.set_ylabel("Cumulative Percentage")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()

    fig.savefig(out_path / "dx_ratio_distribution.png", dpi=300)


### MAIN FUNCTIONS ###


def generate_sampled_run_dataset(
    run_context: RunContext,
    n_samples: int = 500,
    generate_plots: bool = False,
    reaches: dict[int, dict[str, Any]] = None,
) -> pd.DataFrame:
    """Build dataset of run diagnostics at the reach and network level."""
    # Sample from reaches if not provided
    if reaches is None:
        reaches = sample_reaches(run_context, n_samples)

    for k, v in reaches.items():
        diagnostics = generate_reach_diagnostics(run_context, k, v, plot=generate_plots)
        reaches[k].update(diagnostics)

    df_reaches = pd.DataFrame.from_dict(reaches, orient="index")
    df_reaches.to_parquet(
        run_context.result_output_dir / f"{run_context.run_id}_reaches.parquet"
    )
    return df_reaches


def create_diagnostics(
    df_reaches: pd.DataFrame,
    run_context: RunContext,
):
    """Generate summary file for sampled run dataset on a t-route run."""
    plot_reach_mass_conservation(df_reaches, run_context.diagnostic_plot_dir)
    plot_reach_mass_conservation_vs_dx(df_reaches, run_context.diagnostic_plot_dir)
    plot_network_mass_conservation(df_reaches, run_context.diagnostic_plot_dir)
    plot_attenuation(df_reaches, run_context.diagnostic_plot_dir)
    plot_courant(df_reaches, run_context.diagnostic_plot_dir)
    plot_optimal_reach_length(df_reaches, run_context.diagnostic_plot_dir)


def process_run(config_key: str, n_samples: int = 500):
    """Export diagnostics for a t-route run."""
    run_context = RunContext(config_key)

    # Sample reaches upfront so we can subset results before loading
    reaches = sample_reaches(run_context, n_samples)
    needed_ids = collect_needed_feature_ids(run_context, reaches)

    # Include gage feature IDs if reference data exists
    if run_context.reference_data_path.exists():
        with xr.open_dataset(run_context.reference_data_path) as ref_ds:
            needed_ids.update(int(fp) for fp in ref_ds["fp_id"].values)

    # Load only the needed subset of routed results
    run_context.subset_routed_results(list(needed_ids))

    generate_reference_gage_plots(run_context)
    df_reaches = generate_sampled_run_dataset(
        run_context, n_samples, generate_plots=False, reaches=reaches
    )
    create_diagnostics(df_reaches, run_context)


def main():
    """Run diagnostics for sampled reaches across all test cases."""
    parser = argparse.ArgumentParser(
        description="Generate forcing dataset and config YAML for a case."
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Path to t-route config yaml.",
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        help="Number of flowpaths to randomly sample for analysis.",
        default=500,
    )

    args = parser.parse_args()

    process_run(args.file, args.n_samples)


if __name__ == "__main__":
    main()
