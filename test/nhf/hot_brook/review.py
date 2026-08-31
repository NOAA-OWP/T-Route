import math
from dataclasses import dataclass

import traceback

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import geopandas as gpd

import pandas as pd
import yaml


G = 9.81  # m/s^2
ANALYSIS_REACH = 712322
LAKE_IDS = [1591, 1581]


@dataclass
class LevelPoolParams:
    LkArea: float
    LkMxE: float

    OrficeE: float
    OrficeC: float
    OrficeA: float

    WeirE: float
    WeirC: float
    WeirL: float

    Dam_Length: float


@dataclass
class LevelPoolState:
    water_elevation: float


def _compute_discharge(H: float, p: LevelPoolParams) -> float:
    """
    Compute reservoir outflow using the same piecewise logic
    as the Fortran LEVELPOOL_PHYSICS implementation.
    """
    dh = H - p.WeirE
    max_weir_depth = p.LkMxE - p.WeirE
    dh = min(dh, max_weir_depth)

    # Orifice discharge
    if H > p.OrficeE:
        q_orifice = p.OrficeC * p.OrficeA * math.sqrt(2.0 * G * max(H - p.OrficeE, 0.0))
    else:
        q_orifice = 0.0

    # Weir discharge
    if dh > 0.0:
        q_weir = p.WeirC * p.WeirL * (dh**1.5)
    else:
        q_weir = 0.0

    # Dam overtopping discharge
    if H > p.LkMxE:
        q_overtop = p.WeirC * (p.WeirL * p.Dam_Length) * ((H - p.LkMxE) ** 1.5)
    else:
        q_overtop = 0.0

    # Match Fortran branching order exactly
    if H > p.LkMxE:
        return q_orifice + q_weir + q_overtop
    elif dh > 0.0:
        return q_orifice + q_weir
    elif H > p.OrficeE:
        return q_orifice
    else:
        return 0.0


def levelpool_step(
    state: LevelPoolState,
    params: LevelPoolParams,
    qi0: float,
    qi1: float,
    ql: float,
    dt: float,
    verbose: bool = False,
):

    H = state.water_elevation

    # Fortran: sap = ar * 1.0E6  (ar is km^2)
    sap = params.LkArea * 1.0e6

    # Inflow interpolation (identical constants)
    It = qi0
    Itdt_3 = qi0 + ((qi1 + ql - qi0) * 0.33)
    Itdt_2_3 = qi0 + ((qi1 + ql - qi0) * 0.67)

    maxWeirDepth = params.LkMxE - params.WeirE

    # -----------------------
    # STAGE 1
    # -----------------------
    dh = H - params.WeirE
    if dh > maxWeirDepth:
        dh = maxWeirDepth

    tmp1 = (
        params.OrficeC * params.OrficeA * math.sqrt(2.0 * 9.81 * (H - params.OrficeE))
    )
    tmp2 = params.WeirC * params.WeirL * (dh ** (3.0 / 2.0))

    if H > params.LkMxE:
        discharge = (
            tmp1
            + tmp2
            + (
                params.WeirC
                * (params.WeirL * params.Dam_Length)
                * ((H - params.LkMxE) ** (3.0 / 2.0))
            )
        )
    elif dh > 0.0:
        discharge = tmp1 + tmp2
    elif H > params.OrficeE:
        discharge = (
            params.OrficeC
            * params.OrficeA
            * math.sqrt(2.0 * 9.81 * (H - params.OrficeE))
        )
    else:
        discharge = 0.0

    dh1 = ((It - discharge) / sap) * dt if sap > 0.0 else 0.0

    # -----------------------
    # STAGE 2
    # -----------------------
    H2 = H + dh1 / 3.0

    dh = H2 - params.WeirE
    if dh > maxWeirDepth:
        dh = maxWeirDepth

    tmp1 = (
        params.OrficeC * params.OrficeA * math.sqrt(2.0 * 9.81 * (H2 - params.OrficeE))
    )
    tmp2 = params.WeirC * params.WeirL * (dh ** (3.0 / 2.0))

    # NOTE: condition uses ORIGINAL H (not H2)
    if H > params.LkMxE:
        discharge = (
            tmp1
            + tmp2
            + (
                params.WeirC
                * (params.WeirL * params.Dam_Length)
                * ((H - params.LkMxE) ** (3.0 / 2.0))
            )
        )
    elif dh > 0.0:
        discharge = tmp1 + tmp2
    elif H2 > params.OrficeE:
        discharge = (
            params.OrficeC
            * params.OrficeA
            * math.sqrt(2.0 * 9.81 * (H2 - params.OrficeE))
        )
    else:
        discharge = 0.0

    dh2 = ((Itdt_3 - discharge) / sap) * dt if sap > 0.0 else 0.0

    # -----------------------
    # STAGE 3
    # -----------------------
    H3 = H + (0.667 * dh2)

    dh = H3 - params.WeirE
    if dh > maxWeirDepth:
        dh = maxWeirDepth

    tmp1 = (
        params.OrficeC * params.OrficeA * math.sqrt(2.0 * 9.81 * (H3 - params.OrficeE))
    )
    tmp2 = params.WeirC * params.WeirL * (dh ** (3.0 / 2.0))

    # NOTE: again uses ORIGINAL H
    if H > params.LkMxE:
        discharge = (
            tmp1
            + tmp2
            + (
                params.WeirC
                * (params.WeirL * params.Dam_Length)
                * ((H - params.LkMxE) ** (3.0 / 2.0))
            )
        )
    elif dh > 0.0:
        discharge = tmp1 + tmp2
    elif H3 > params.OrficeE:
        discharge = (
            params.OrficeC
            * params.OrficeA
            * math.sqrt(2.0 * 9.81 * (H3 - params.OrficeE))
        )
    else:
        discharge = 0.0

    dh3 = ((Itdt_2_3 - discharge) / sap) * dt if sap > 0.0 else 0.0

    # -----------------------
    # FINAL UPDATE
    # -----------------------
    dH = (dh1 / 4.0) + (0.75 * dh3)
    H = H + dH

    # -----------------------
    # FINAL DISCHARGE
    # -----------------------
    dh = H - params.WeirE
    if dh > maxWeirDepth:
        dh = maxWeirDepth

    tmp1 = (
        params.OrficeC * params.OrficeA * math.sqrt(2.0 * 9.81 * (H - params.OrficeE))
    )
    tmp2 = params.WeirC * params.WeirL * (dh ** (3.0 / 2.0))

    if H > params.LkMxE:
        discharge = (
            tmp1
            + tmp2
            + (
                params.WeirC
                * (params.WeirL * params.Dam_Length)
                * ((H - params.LkMxE) ** (3.0 / 2.0))
            )
        )
    elif dh > 0.0:
        discharge = tmp1 + tmp2
    elif H > params.OrficeE:
        discharge = (
            params.OrficeC
            * params.OrficeA
            * math.sqrt(2.0 * 9.81 * (H - params.OrficeE))
        )
    else:
        discharge = 0.0

    # update state
    state.water_elevation = H

    if verbose:
        print("----------------------------------------")
        print("PARAM LEVELPOOL_PHYSICS:")

        print("H   =", state.water_elevation)
        print("dt  =", dt)
        print("qi0 =", qi0)
        print("qi1 =", qi1)
        print("ql  =", ql)

        print("ar  =", params.LkArea)
        print("we  =", params.WeirE)
        print("wc  =", params.WeirC)
        print("wl  =", params.WeirL)
        print("dl  =", params.Dam_Length)

        print("oe  =", params.OrficeE)
        print("oc  =", params.OrficeC)
        print("oa  =", params.OrficeA)

        print("maxh =", params.LkMxE)

        print("----------------------------------------")
        print(f"DEBUG LEVELPOOL_PHYSICS: sap (after conversion) =   {sap}")
        print(f"{discharge=}")
        print(f"{H=}")
        print(f"{dh=}")

    return discharge, H


def main():
    model_root = Path(__file__).parent
    cfg_path = model_root / "synthetic_pulse.yaml"
    cfg = load_config(cfg_path)
    forcing_df = load_forcing(model_root, cfg)
    troute_out = load_results(model_root, cfg)
    lakes = make_lake_objects(model_root, cfg)
    outflows = reroute(cfg, forcing_df, lakes)
    plot(forcing_df, troute_out, outflows, model_root)


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, mode="r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_forcing(model_root: Path, cfg: dict) -> pd.DataFrame:
    forcing_path = model_root / str(
        cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"]
    )
    forcing_files = forcing_path.glob(
        str(cfg["compute_parameters"]["forcing_parameters"]["qlat_file_pattern_filter"])
    )
    forcing_df = pd.concat(
        [pd.read_csv(i).set_index("feature_id") for i in forcing_files], axis=1
    )
    forcing_df = forcing_df.T[ANALYSIS_REACH].sort_index()
    return forcing_df


def load_results(model_root: Path, cfg: dict) -> xr.Dataset:
    output_path = model_root / str(
        cfg["output_parameters"]["stream_output"]["stream_output_directory"]
    )
    output_files = output_path.glob(
        "*" + cfg["output_parameters"]["stream_output"]["stream_output_type"]
    )
    output_ds = [xr.open_dataset(p, engine="netcdf4") for p in sorted(output_files)]
    output_ds = xr.concat(output_ds, dim="time")
    troute_out = output_ds.sel(feature_id=ANALYSIS_REACH)["flow"]
    return troute_out


def make_lake_objects(model_root: Path, cfg: dict) -> dict:
    gpkg_path = model_root / str(
        cfg["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"]
    )
    lakes = {}
    for i in LAKE_IDS:
        query = f"SELECT OrificeE, OrificeC, OrificeA, WeirE, WeirC, WeirL, Dam_Length, LkArea, LkMxE FROM lakes WHERE lake_id = {i}"
        wb_gdf = gpd.read_file(gpkg_path, sql=query, ignore_geometry=True).rename(
            columns={
                "OrificeE": "OrficeE",
                "OrificeC": "OrficeC",
                "OrificeA": "OrficeA",
            }
        )
        wb_gdf = wb_gdf.iloc[0].to_dict()
        wb_gdf["Dam_Length"] = 10
        lakes[i] = {
            "params": LevelPoolParams(**wb_gdf),
            "state": LevelPoolState(water_elevation=wb_gdf["OrficeE"]),
        }
    return lakes


def reroute(cfg: dict, forcing_df: pd.DataFrame, lakes: dict) -> dict:
    dt = cfg["compute_parameters"]["forcing_parameters"]["dt"]
    inflows = forcing_df.values
    inflows = np.repeat(inflows, 12)  # resample for qts_subdivisions

    outflows_1 = []
    outflows_2 = []
    for ind, i in enumerate(inflows):
        qin = i
        ql = 0
        for ind, i in enumerate(lakes):
            try:
                q_out, H = levelpool_step(
                    lakes[i]["state"], lakes[i]["params"], qin, qin, ql, dt
                )
            except ValueError as e:
                print(traceback.format_exc())
                q_out = np.nan
                H = 0
            if ind == 0:
                outflows_1.append(q_out)
            elif ind == 1:
                outflows_2.append(q_out)
            qin = q_out
    outflows_1 = outflows_1[12::12]  # resample for qts_subdivisions
    outflows_2 = outflows_2[12::12]
    # Can't tell if starting at 12 is an indexing bug or a T-Route implementation bug
    return {LAKE_IDS[0]: outflows_1, LAKE_IDS[1]: outflows_2}


def plot(
    forcing_df: pd.DataFrame, troute_out: xr.Dataset, outflows: dict, model_root: Path
) -> None:
    fig, ax = plt.subplots()
    ax.plot(forcing_df.values, label="inflow", c="k")
    ax.plot(troute_out, label="outflow", c="r")

    cs = ["gray", "darkred"]
    for ind, i in enumerate(outflows):
        ax.plot(outflows[i], label=f"python check {i}", c=cs[ind], ls="dashed")

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge (cms)")
    fig.tight_layout()
    fig.savefig(model_root / "validation.png")


if __name__ == "__main__":
    main()
