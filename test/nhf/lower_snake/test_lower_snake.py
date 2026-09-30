"""Lower Granite and Little Goose, run-of-river RFC dams, routed with RFC DA off.

Each should pass its inflow with its level held; the contrast arm clears the flag on the
RFC dams, which makes them level pools. Both arms start each pool at the cold state, the
orifice invert, which the pass-through holds and the level pool fills from.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test.nhf.utils.integration_helpers import (
    delete_outputs,
    has_files,
    load_lakeout,
    run_troute,
    skip_if_not_built,
)
from test.nhf.utils.make_configs import Config
from test.nhf.utils.make_forcing import build_forcing_dataset, create_hot_start_file
from test.nhf.utils.subset_nhf import (
    extract_layers,
    get_downstream_fp_ids,
    get_offnetwork_upstreams,
    write_gpkg,
)

# The Snake and the Clearwater just above their confluence. Lower Granite's outlet is 25
# hops down the Snake and Little Goose's 51, so 55 hops carries both and a few reaches below.
SEED_FP_IDS = [1267774669700543, 1267774688946911]
DEPTH = 55
SITES = ("LGDW1", "LGSW1")
START_TIME = "2011-05-05 00:00"
END_TIME = "2011-05-25 00:00"
HOT_START = "restart.pkl"

DATA_DIR = Path(__file__).parent / "data"
CFG = Config(DATA_DIR, START_TIME, END_TIME, lakeout_output="lakeout",
             restart_dir_name="restart", restart_file_name=HOT_START)
CFG_FULL_POOL = Config(
    DATA_DIR, START_TIME, END_TIME, lakeout_output="lakeout_full_pool",
    restart_dir_name="restart", restart_file_name=HOT_START,
    config_file_name="config_full_pool.yaml", output_dir_name="output_full_pool",
    domain_file_name="nhf_full_pool.gpkg",
)


def setup(source_gpkg: str | Path, refresh: bool = True):
    """Carve the lower Snake, clear the RFC flags in a copy, and build retro forcing."""
    for cfg in (CFG, CFG_FULL_POOL):
        cfg.write_yaml()
    fp_ids = get_downstream_fp_ids(str(source_gpkg), SEED_FP_IDS, DEPTH)
    boundaries = get_offnetwork_upstreams(str(source_gpkg), fp_ids)

    if refresh or not (CFG.domain_path.exists() and CFG_FULL_POOL.domain_path.exists()):
        layers = extract_layers(str(source_gpkg), fp_ids + boundaries)
        write_gpkg(layers, CFG.domain_path)
        rda, lakes = layers["reservoir_da"], layers["lakes"].copy()
        lakes.loc[lakes["nhf_lake_id"].isin(rda.loc[rda["da_type"] == 4, "nhf_lake_id"]),
                  "run_of_river"] = False
        write_gpkg({**layers, "lakes": lakes}, CFG_FULL_POOL.domain_path)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        build_forcing_dataset("retro", START_TIME, END_TIME, CFG.channel_forcing_dir,
                              CFG.domain_path, offnetwork_upstreams=boundaries)
    restart_dir = CFG.root_dir / "restart"
    if refresh or not (restart_dir / HOT_START).exists():
        create_hot_start_file(START_TIME, str(restart_dir), str(CFG.domain_path), boundaries)


def _table(domain: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{domain}?mode=ro", uri=True) as db:
        frame = pd.read_sql(sql, db)
    db.close()
    return frame


def _dams(domain: Path) -> pd.DataFrame:
    """Each dam's lake id, type and flag, read from the domain: lake ids are per build."""
    sites = ",".join(f"'{s}'" for s in SITES)
    return _table(domain, (
        "SELECT r.site_no, r.nhf_lake_id, r.da_type, l.run_of_river FROM reservoir_da r "
        f"JOIN lakes l ON l.nhf_lake_id = r.nhf_lake_id WHERE r.site_no IN ({sites})"
    )).set_index("site_no")


def _route(cfg: Config) -> dict[str, pd.DataFrame]:
    """Run *cfg* and return each dam's hourly inflow, outflow and level."""
    lakeout = cfg.lakeout_dir
    assert lakeout is not None
    delete_outputs(lakeout)
    run_troute(cfg.config_path)
    ds = load_lakeout(lakeout)
    series = {}
    for site, lake in _dams(cfg.domain_path)["nhf_lake_id"].items():
        assert lake in ds["feature_id"], f"{site} ({lake}) is not a routed reservoir"
        one = ds.sel(feature_id=lake)
        series[site] = pd.DataFrame(
            {v: one[v].to_numpy().astype(float) for v in ("inflow", "outflow", "water_sfc_elev")}
        )
    return series


@pytest.mark.integration
def test_the_arms_differ_only_in_the_rfc_dams_flags():
    skip_if_not_built(CFG)
    dams = _dams(CFG.domain_path)
    assert set(dams.index) == set(SITES)
    assert (dams["da_type"] == 4).all(), dams
    assert (dams["run_of_river"] == 1).all(), dams
    assert (_dams(CFG_FULL_POOL.domain_path)["run_of_river"] == 0).all()

    sql = "SELECT nhf_lake_id, run_of_river FROM lakes ORDER BY nhf_lake_id"
    delivered = _table(CFG.domain_path, sql).set_index("nhf_lake_id")["run_of_river"]
    cleared = _table(CFG_FULL_POOL.domain_path, sql).set_index("nhf_lake_id")["run_of_river"]
    changed = set(delivered.index[delivered != cleared])
    assert changed == set(dams["nhf_lake_id"]), f"flags changed on {sorted(changed)}"


@pytest.mark.integration
def test_each_dam_passes_its_inflow_and_holds_its_level():
    skip_if_not_built(CFG)
    for site, s in _route(CFG).items():
        # A flow that barely moves would pass through a level pool too.
        assert s["inflow"].max() > 1.5 * s["inflow"].min(), f"{site}: no flood reached it"
        np.testing.assert_allclose(s["outflow"], s["inflow"], rtol=1e-6, err_msg=site)
        moved = np.ptp(s["water_sfc_elev"])
        assert moved < 1e-4, f"{site}: level moved {moved:.3f} m"


@pytest.mark.integration
def test_with_the_flag_cleared_the_same_dams_route_as_level_pools():
    """What the pass-through test would see from a level pool, so its pass is not vacuous."""
    skip_if_not_built(CFG_FULL_POOL)
    for site, s in _route(CFG_FULL_POOL).items():
        gap = (s["outflow"] - s["inflow"]).abs().max()
        assert gap > 0.1 * s["inflow"].max(), f"{site}: release within {gap:.0f} cms of inflow"
        assert np.ptp(s["water_sfc_elev"]) > 1.0, f"{site}: level held"
