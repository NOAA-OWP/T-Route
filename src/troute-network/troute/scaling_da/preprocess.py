"""One-time static preprocessing for the simple-scaling streamflow DA.

Everything the DA needs that is a property of the network plus the run config --
the gage crosswalk, waterbody stops, observable-station roster, holdout, source
set, and per-gage upstream trees -- is resolved here once, as part of t-route's
data processing. ``NHF.__init__`` calls :func:`build_scaling_da_setup` when the
DA is enabled; the runtime applier consumes the resulting
:class:`ScalingDASetup`. Callers with a network that did not pre-build (tests,
harnesses) go through the same function, so there is one implementation.

Every guard here fails LOUDLY, because its silent form is a run that completes,
writes full output, and exits 0 while assimilating nothing.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from troute.scaling_da.build_trees import build_gage_trees_from_mappings

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from troute.AbstractNetwork import AbstractNetwork
    from troute.scaling_da.gage_tree import GageTree

LOG = logging.getLogger("TROUTE")

# The USGS half of the WRF-Hydro TimeSlice naming convention.
TIMESLICE_GLOB_SUFFIX = "usgsTimeSlice.ncdf"


def timeslice_station_roster(folder: Path, cpu_pool: int = 1) -> set[str]:
    """Every station id appearing in *folder*'s USGS TimeSlice files.

    Reads only ``stationId`` headers. Scans the WHOLE directory: a single file
    carries only the stations reporting at that instant, and a gage missing from
    the roster is excluded from assimilation for the entire run.

    O(files) at startup (~35k for a CONUS year, a few seconds in parallel). Cache
    the roster beside the directory if that ever matters.
    """
    files = sorted(folder.glob(f"*.{TIMESLICE_GLOB_SUFFIX}"))
    if not files:
        return set()

    def _roster_one(path: Path) -> set[str]:
        import netCDF4

        try:
            with netCDF4.Dataset(str(path), "r") as ds:
                stns = ds.variables["stationId"][:]
        except (OSError, KeyError):  # a truncated/foreign file is not fatal
            return set()
        # Decoded EXACTLY as nhd_io._read_timeslice_file does: these ids must
        # match that reader's column labels or the site silently never assimilates.
        ids = np.apply_along_axis("".join, 1, stns.filled(fill_value=np.nan).astype(str))
        ids = np.char.rstrip(np.char.strip(ids), "n")
        return {s for s in (str(i) for i in ids) if s}

    from joblib import Parallel, delayed

    with Parallel(n_jobs=cpu_pool) as parallel:
        rosters = parallel(delayed(_roster_one)(f) for f in files)
    # joblib types elements Optional; _roster_one never returns None.
    return set().union(*(r for r in (rosters or []) if r is not None))


def resolve_gage_crosswalk(network: AbstractNetwork) -> tuple[dict[str, int], dict[str, int]]:
    """``(site_no -> up_node_id, site_no -> fp_id)`` from ``network.gages``.

    The segment crosswalk MUST be the same set the execution plan splits reaches
    at, or the observation is injected at a link the plan never isolated. (A
    private gpkg-derived rule disagreed with the network's for 24% of CONUS
    gages.) The fp crosswalk only serves the frozen synthetic baseline, which is
    indexed by fp_id. Note ``total_da_sqkm`` varies WITHIN a flowpath
    (interpolated per link), so A_o is the at-gage-link area.
    """
    gages = (getattr(network, "gages", None) or {}).get("gages", {})
    if not gages:
        return {}, {}
    df = network.dataframe
    fp_of = df["fp_id"] if "fp_id" in df.columns else None
    seg_of: dict[str, int] = {}
    fp_map: dict[str, int] = {}
    for link, site in gages.items():
        if site is None or (isinstance(site, float) and np.isnan(site)):
            continue
        link_id = int(link)
        seg_of[str(site)] = link_id
        if fp_of is not None and link_id in fp_of.index:
            fp_map[str(site)] = int(fp_of.loc[link_id])
    return seg_of, fp_map


def resolve_waterbody_stops(network: AbstractNetwork) -> frozenset[int]:
    """Segments a tree must not walk through (Edge Case 1).

    ``waterbody_dataframe`` is the populated index on NHF;
    ``link_lake_crosswalk`` is None there, and relying on it alone let trees walk
    onto lake nodes whose NaN drainage area then poisoned the whole subtree.
    """
    waterbody = frozenset(int(s) for s in (network.link_lake_crosswalk or {}))
    wb_df = getattr(network, "waterbody_dataframe", None)
    if wb_df is not None and len(wb_df):
        waterbody |= frozenset(int(s) for s in wb_df.index)
    return waterbody


def resolve_obs_sites(
    gage_seg: Mapping[str, int],
    da_params: Mapping[str, Any],
    cpu_pool: int,
    *,
    synthetic: bool,
) -> tuple[Path | None, set[str]]:
    """``(timeslice folder, crosswalked sites the folder can observe)``.

    Resolved at setup because the source set is the stop set the trees are built
    against, so it must be static. All failure modes raise: a misconfigured
    observation source must not degrade into a silent no-DA run.
    """
    folder = da_params.get("usgs_timeslices_folder")
    if not folder or synthetic:
        return None, set()
    ts_folder = Path(folder)
    if not ts_folder.is_dir():
        raise FileNotFoundError(
            f"data_assimilation_parameters.usgs_timeslices_folder is set to "
            f"{folder!r} but that is not a directory, so the scaling DA has no "
            "observations to assimilate. Point it at a TimeSlice directory, "
            "or disable scaling_da."
        )
    roster = timeslice_station_roster(ts_folder, cpu_pool)
    if not roster:
        raise FileNotFoundError(
            f"no readable *.{TIMESLICE_GLOB_SUFFIX} files in {folder!r}; the "
            "scaling DA would assimilate nothing."
        )
    obs_sites = set(gage_seg) & roster
    if not obs_sites:
        raise ValueError(
            f"none of the {len(gage_seg)} gage(s) in this domain appear in the "
            f"TimeSlice station roster at {folder!r} ({len(roster)} station(s)); "
            "the run would be identical to a no-DA control."
        )
    LOG.info(
        "scaling DA: obs from TimeSlice dir %s -- %d of %d gages have observations",
        folder,
        len(obs_sites),
        len(gage_seg),
    )
    return ts_folder, obs_sites


def resolve_holdout(
    params: Mapping[str, Any], gage_seg: Mapping[str, int]
) -> set[str]:
    """The held-out site set, validated against the domain's crosswalk.

    A holdout that silently withholds nothing (missing file, mistyped id) lets a
    scoring run assimilate the very gages it claims to have excluded; both cases
    raise.
    """
    hf = params.get("holdout_sites_file")
    if not hf:
        return set()
    if not Path(hf).exists():
        raise FileNotFoundError(
            f"streamflow_scaling_parameters.holdout_sites_file not found: {hf}"
        )
    holdout = {ln.strip() for ln in Path(hf).read_text().splitlines() if ln.strip()}
    unknown = holdout - set(gage_seg)
    if unknown:
        raise ValueError(
            f"streamflow_scaling_parameters.holdout_sites_file lists {len(unknown)} id(s) that match "
            f"no gage in this domain's crosswalk, e.g. {sorted(unknown)[:5]}; they "
            "would withhold nothing."
        )
    return holdout


def _read_theta_csv(path: str | None) -> dict[str, float]:
    """``gage id -> theta`` from a two-column CSV, or empty when unset.

    A header is optional: a first row whose second field is not a number is
    treated as one. Ids are kept as STRINGS exactly as written, because gage ids
    are zero-padded ("03031500") and reading them as numbers would silently drop
    the leading zero and match nothing.

    A configured file that is missing or unreadable raises rather than falling
    back to the default exponent everywhere, which would look like it had been
    applied.
    """
    if not path:
        return {}
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"streamflow_scaling_parameters.theta.per_tree_csv not found: {path}"
        )
    out: dict[str, float] = {}
    for lineno, raw in enumerate(src.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ValueError(
                f"{path}:{lineno}: expected 'gage_id,theta', got {raw!r}"
            )
        site, value = parts[0], parts[1]
        try:
            theta = float(value)
        except ValueError:
            if lineno == 1:
                continue  # a header row
            raise ValueError(
                f"{path}:{lineno}: theta must be a number, got {value!r}"
            ) from None
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError(
                f"{path}:{lineno}: theta must be finite and positive, got {theta!r}"
            )
        out[site] = theta
    if not out:
        raise ValueError(
            f"streamflow_scaling_parameters.theta.per_tree_csv has no usable rows: {path}"
        )
    return out


def _theta_by_site(
    network: AbstractNetwork,
    theta_per_tree: Mapping[str, float],
    theta_by_vpu: Mapping[str, float],
    theta_default: float,
    known_sites: Iterable[str] = (),
) -> dict[str, float]:
    """site_no -> theta, most specific source first.

    Order is ``per_tree`` (keyed by gage id), then ``by_vpu`` (keyed by the
    gage's hydrofabric VPU), then the default. One tree carries one exponent,
    which is the finest resolution the closed form admits.

    Keys matching no gage are warned about in both cases: a mistyped id or a
    VPU key that is not zero-padded would otherwise fall back to the default
    everywhere and look like it had been applied.
    """
    by_site: dict[str, float] = {}
    gage_vpu = getattr(network, "gage_vpu", None) or {}

    if theta_by_vpu:
        if gage_vpu:
            by_site.update({
                site: theta_by_vpu[vpu]
                for site, vpu in gage_vpu.items()
                if vpu in theta_by_vpu
            })
            unused = sorted(set(theta_by_vpu) - set(gage_vpu.values()))
            if unused:
                LOG.warning(
                    "scaling DA: theta.by_vpu has %d key(s) matching no gage in this "
                    "domain: %s. The hydrofabric's vpu_id values here are %s.",
                    len(unused), unused, sorted(set(gage_vpu.values()))[:8],
                )
        else:
            LOG.warning(
                "scaling DA: theta.by_vpu is set but the hydrofabric provided no "
                "gage->VPU mapping; those entries cannot be applied.",
            )

    if theta_per_tree:
        # Applied AFTER by_vpu so the more specific key wins.
        by_site.update({str(k): float(v) for k, v in theta_per_tree.items()})
        known = set(map(str, known_sites)) | set(gage_vpu)
        if known:
            unused = sorted(set(map(str, theta_per_tree)) - known)
            if unused:
                LOG.warning(
                    "scaling DA: theta.per_tree_csv has %d gage id(s) matching no gage "
                    "in this domain: %s. Those entries do nothing; check them "
                    "against the ids in the TimeSlice files.",
                    len(unused), unused[:8],
                )
    return by_site


@dataclass(frozen=True)
class ScalingDASetup:
    """Everything static the runtime DA applier consumes off the network."""

    trees: dict[str, GageTree]
    gage_seg: dict[str, int]        # injectable site -> routed segment (deduped)
    gage_fp: dict[str, int]         # site -> fp_id (frozen synthetic baseline reads)
    all_gage_seg: dict[str, int]    # full crosswalk, incl. held-out and co-located
    da_sites: list[str] = field(default_factory=list)  # sorted injectable sites
    obs_sites: set[str] = field(default_factory=set)   # crosswalk ∩ TimeSlice roster
    waterbody: frozenset[int] = frozenset()
    ts_folder: Path | None = None
    theta_default: float = 0.77
    theta_by_vpu: dict[str, float] = field(default_factory=dict)
    theta_per_tree: dict[str, float] = field(default_factory=dict)


def build_scaling_da_setup(
    network: AbstractNetwork,
    params: Mapping[str, Any],
    da_params: Mapping[str, Any] | None = None,
    cpu_pool: int | None = None,
) -> ScalingDASetup:
    """Resolve the scaling DA's static inputs from the network and the config.

    Order matters: the tree stop set must be the ACTUAL DA sources, so
    crosswalk -> obs store -> holdout -> sources -> trees.
    """
    da_params = da_params or {}
    _tb = time.perf_counter()

    # One theta per gage tree (the linear step telescopes only for constant theta).
    theta_cfg = params.get("theta") or {}
    if not isinstance(theta_cfg, dict):
        theta_cfg = theta_cfg.model_dump() if hasattr(theta_cfg, "model_dump") else {}
    theta_default = float(theta_cfg.get("default", 0.77))
    theta_by_vpu = {str(k): float(v) for k, v in (theta_cfg.get("by_vpu") or {}).items()}
    theta_per_tree = _read_theta_csv(theta_cfg.get("per_tree_csv"))

    df = network.dataframe
    if "total_da_sqkm" not in df.columns:
        raise KeyError(
            "scaling DA needs 'total_da_sqkm' on the routed dataframe; "
            "the NHF ingest must read it (nhf_preprocess.LAYERS_TO_READ)."
        )
    waterbody = resolve_waterbody_stops(network)
    gage_seg, gage_fp = resolve_gage_crosswalk(network)
    # Full crosswalk kept so evaluation harnesses can locate held-out gages.
    all_gage_seg = dict(gage_seg)
    if not gage_seg:
        LOG.warning("scaling DA: no gage->segment crosswalk; DA will be inert.")

    synthetic = params.get("synthetic_obs_factor") is not None
    ts_folder, obs_sites = resolve_obs_sites(
        gage_seg, da_params, cpu_pool or 1, synthetic=synthetic
    )

    # The injected set is FIXED across loops (plan determinism): real obs -> the
    # roster's gages; synthetic -> every crosswalked gage.
    sites = obs_sites if not synthetic else set(gage_seg)
    holdout = resolve_holdout(params, gage_seg)
    # A gage crosswalked to a lake id is excluded LOUDLY: the kernel routes
    # that segment as a reservoir object (reservoir DA owns it). A gage on the
    # channel below a reservoir still assimilates.
    on_lake = sorted(s for s in sites if gage_seg.get(s) in waterbody)
    if on_lake:
        LOG.warning(
            "scaling DA: %d gage(s) crosswalk to reservoir-routed segments and "
            "are EXCLUDED from streamflow scaling (reservoir DA owns those "
            "segments): %s",
            len(on_lake), on_lake[:8],
        )
    da_sites = sorted(
        s for s in sites if gage_seg.get(s) not in waterbody and s not in holdout
    )
    if holdout:
        LOG.info(
            "scaling DA: holdout -- %d id(s) requested, %d had observations to "
            "withhold this run: %s",
            len(holdout),
            len(holdout & set(sites)),
            sorted(holdout & set(sites))[:8],
        )

    # Trees: BFS upstream from each SOURCE, stopping only at other sources and at
    # waterbodies. Source-only roots AND stops keep the trees disjoint; stopping
    # at non-source gages truncated the correction's reach for no benefit and
    # nulled held-out evaluations.
    area = {int(s): float(a) for s, a in df["total_da_sqkm"].items()}
    rconn = {int(s): [int(u) for u in ups] for s, ups in network.reverse_network.items()}
    source_segs = frozenset(gage_seg[s] for s in da_sites if s in gage_seg)
    trees = build_gage_trees_from_mappings(
        rconn,
        gage_seg,
        area,
        waterbody_segs=waterbody,
        theta_default=theta_default,
        theta_by_site=_theta_by_site(
            network, theta_per_tree, theta_by_vpu, theta_default,
            known_sites=set(gage_seg) | set(all_gage_seg),
        ),
        gage_stop_segs=source_segs,
        site_filter=frozenset(da_sites),
    )

    # A dropped tree (e.g. NaN area in the subtree) only disables the upstream
    # spread; the at-gage insertion needs no area, so the site stays injectable.
    # Co-located gages still dedupe to one site per segment, in gage_seg's
    # insertion order (the network's active-before-discontinued preference).
    injectable: list[str] = []
    claimed: set[int] = set()
    da_set = set(da_sites)
    for site, seg in gage_seg.items():
        if site not in da_set or seg in claimed:
            continue
        claimed.add(seg)
        injectable.append(site)
    gage_seg = {s: gage_seg[s] for s in injectable}
    da_sites = sorted(injectable)
    downstream_only = [s for s in da_sites if s not in trees]
    if downstream_only:
        LOG.info(
            "scaling DA: %d gage(s) have no usable upstream tree and contribute "
            "the at-gage + downstream correction only, e.g. %s",
            len(downstream_only), downstream_only[:5],
        )

    LOG.info("scaling DA: tree build took %.2f s", time.perf_counter() - _tb)
    LOG.info(
        "scaling DA: %d disjoint source trees covering %d segments "
        "(theta default %.2f, %d theta override(s) configured, %d tree(s) regionalized); "
        "%d gage(s) injected%s",
        len(trees),
        sum(int(t.seg_order.size) for t in trees.values()),
        theta_default,
        len(theta_by_vpu) + len(theta_per_tree),
        sum(1 for t in trees.values() if t.theta != theta_default),
        len(da_sites),
        f" ({len(holdout)} held out)" if holdout else "",
    )
    return ScalingDASetup(
        trees=trees,
        gage_seg=gage_seg,
        gage_fp=gage_fp,
        all_gage_seg=all_gage_seg,
        da_sites=da_sites,
        obs_sites=obs_sites,
        waterbody=waterbody,
        ts_folder=ts_folder,
        theta_default=theta_default,
        theta_by_vpu=theta_by_vpu,
        theta_per_tree=theta_per_tree,
    )
