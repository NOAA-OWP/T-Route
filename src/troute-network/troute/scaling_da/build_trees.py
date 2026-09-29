"""Build per-gage upstream trees from a t-route network for the simple-scaling DA.

Two entry points:

- :func:`build_gage_trees_from_mappings` is the network-agnostic core. It takes
  primitive topology mappings -- reverse connectivity, a gage->segment
  crosswalk, a per-segment drainage-area lookup, and the waterbody stop set --
  and emits one :class:`~troute.scaling_da.gage_tree.GageTree` per gage. This is
  the function the unit tests exercise and the one a standalone hydrofabric
  reader can drive directly.
- :func:`build_gage_trees` is a thin adapter that pulls those mappings off a
  t-route ``AbstractNetwork`` (``reverse_network``, ``link_gage_df``,
  ``link_lake_crosswalk``, and ``dataframe[area_column]``) and delegates to the
  core. The drainage-area column is added to the NHF ingest in a later stage;
  until then the adapter raises a clear error pointing there.

Each gage's BFS halts at every *other* gage and at waterbody-interior segments
(the white paper's Edge Case 1 stop rule), so the resulting trees are mutually
disjoint and the kernel's per-tree corrections sum without overlap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from troute.scaling_da.gage_tree import build_one_gage_tree

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from troute.AbstractNetwork import AbstractNetwork
    from troute.scaling_da.gage_tree import GageTree


__all__ = ["build_gage_trees", "build_gage_trees_from_mappings"]

LOG = logging.getLogger("TROUTE")


def build_gage_trees_from_mappings(
    rconn: Mapping[int, list[int]],
    gage_to_seg: Mapping[str, int],
    area_sqkm: Mapping[int, float],
    *,
    waterbody_segs: Iterable[int] = frozenset(),
    theta_default: float = 0.77,
    theta_by_site: Mapping[str, float] | None = None,
    stop_at_waterbodies: bool = True,
    stop_at_upstream_gages: bool = True,
    gage_stop_segs: Iterable[int] | None = None,
    site_filter: frozenset[str] | None = None,
) -> dict[str, GageTree]:
    """Build per-gage trees from primitive topology mappings.

    Parameters
    ----------
    rconn : Mapping[int, list[int]]
        Reverse connectivity ``segment -> [upstream segments]`` (e.g.
        ``AbstractNetwork.reverse_network``).
    gage_to_seg : Mapping[str, int]
        ``site_no -> routed segment id`` for the gages to assimilate.
    area_sqkm : Mapping[int, float]
        ``segment -> total drainage area`` (km^2). A gage whose segment has a
        missing/non-positive area is skipped (the area-scaling step needs a
        finite positive ``A_o``).
    waterbody_segs : Iterable[int], optional
        Segments inside waterbodies; BFS halts at these (Edge Case 1). Default
        empty.
    theta_default : float, optional
        Region theta applied to every segment in every tree (default ``0.77``;
        Ogden-Dawdy 2003). The ``flow_ratio`` kernel requires it uniform.
    stop_at_waterbodies, stop_at_upstream_gages : bool, optional
        Toggle the two stop rules independently (both on by default).
    gage_stop_segs : Iterable[int] or None, optional
        Explicit set of segments to treat as gage-stops. ``None`` (default) means
        *every* gage segment (production: every USGS gage is a DA source and a
        stop). Pass only the DA-source subset for a held-out evaluation, so a
        correction propagates *through* the held-out gages.
    site_filter : frozenset[str] or None, optional
        Restriction to a subset of ``site_no`` strings; ``None`` (default) keeps
        all.

    Returns
    -------
    dict[str, GageTree]
        One :class:`~troute.scaling_da.gage_tree.GageTree` per kept ``site_no``.
        Co-located gages (sharing a segment) and gages with missing/non-positive
        drainage area are dropped (logged as warnings), so the returned trees are
        mutually disjoint.
    """
    gage_to_seg = {str(site): int(seg) for site, seg in gage_to_seg.items()}
    if gage_stop_segs is None:
        gage_stop_set = frozenset(gage_to_seg.values())
    else:
        gage_stop_set = frozenset(int(s) for s in gage_stop_segs)
    waterbody_set = frozenset(int(s) for s in waterbody_segs)

    trees: dict[str, GageTree] = {}
    claimed_segs: set[int] = set()
    skipped: list[str] = []
    nonfinite: list[str] = []
    collisions: list[str] = []
    for site_no, gage_seg in gage_to_seg.items():
        if site_filter is not None and site_no not in site_filter:
            continue
        # One DA source per segment: co-located gages would build identical trees
        # and double-count. First-seen wins (input order = source preference).
        if gage_seg in claimed_segs:
            collisions.append(site_no)
            continue
        stop: set[int] = set()
        if stop_at_upstream_gages:
            stop |= gage_stop_set - {gage_seg}
        if stop_at_waterbodies:
            stop |= waterbody_set
        # The root is never a stop (evaluation harnesses walk held-out gages
        # through here). Production preprocessing excludes lake-id-crosswalked
        # gages loudly BEFORE this layer; reservoir DA owns those segments.
        stop.discard(gage_seg)
        tree = build_one_gage_tree(
            gage_seg,
            rconn,
            area_sqkm,
            stop_segs=frozenset(stop),
            # One theta per tree, regionalized where the caller resolved one for this
            # gage (from its VPU) and the global default everywhere else.
            theta=(theta_by_site or {}).get(site_no, theta_default),
        )
        if tree.gage_area_sqkm <= 0 or not np.isfinite(tree.gage_area_sqkm):
            skipped.append(site_no)
            continue
        # A NaN area anywhere in the subtree would reach the corrected discharge
        # (and, under state seeding, the warmstate); drop the whole tree instead.
        if not np.isfinite(tree.seg_areas_sqkm).all():
            nonfinite.append(site_no)
            continue
        trees[site_no] = tree
        claimed_segs.add(gage_seg)

    if collisions:
        LOG.warning(
            "build_gage_trees: dropped %d gage(s) co-located on an already-assimilated "
            "flowpath (one DA source per segment): %s",
            len(collisions),
            collisions[:5],
        )
    if skipped:
        LOG.warning(
            "build_gage_trees: skipped %d gage(s) with missing/invalid drainage area: %s",
            len(skipped),
            skipped[:5],
        )
    if nonfinite:
        LOG.warning(
            "build_gage_trees: dropped %d gage tree(s) containing a segment with no "
            "drainage area (NaN would propagate into the corrected discharge): %s",
            len(nonfinite),
            nonfinite[:5],
        )

    return trees


def build_gage_trees(
    network: AbstractNetwork,
    *,
    theta_default: float = 0.77,
    area_column: str = "total_da_sqkm",
    stop_at_waterbodies: bool = True,
    stop_at_upstream_gages: bool = True,
    site_filter: frozenset[str] | None = None,
) -> dict[str, GageTree]:
    """Build per-gage trees from a t-route ``AbstractNetwork``.

    Thin adapter over :func:`build_gage_trees_from_mappings`: it reads the
    primitive topology mappings off the network and delegates.

    Parameters
    ----------
    network : AbstractNetwork
        A t-route network. Reverse connectivity is read from
        ``reverse_network``, the gage->segment crosswalk from ``link_gage_df``,
        the waterbody stop set from ``link_lake_crosswalk``, and per-reach
        drainage area from ``dataframe[area_column]``.
    theta_default : float, optional
        Region theta applied to every segment in every tree (default ``0.77``,
        Ogden-Dawdy 2003).
    area_column : str, optional
        Column of ``network.dataframe`` holding per-reach drainage area
        (``flowpaths.total_da_sqkm`` in the NHF gpkg).
    stop_at_waterbodies, stop_at_upstream_gages : bool, optional
        Toggle the two BFS stop rules independently (both on by default).
    site_filter : frozenset[str] or None, optional
        Restriction to a subset of ``site_no`` strings; ``None`` (default) keeps
        all.

    Returns
    -------
    dict[str, GageTree]
        One :class:`~troute.scaling_da.gage_tree.GageTree` per gage (see
        :func:`build_gage_trees_from_mappings` for the disjointness guarantees).

    Raises
    ------
    KeyError
        If ``area_column`` is absent from ``network.dataframe`` (the drainage-area
        column must be plumbed from the hydrofabric first).
    """
    df = network.dataframe
    if area_column not in df.columns:
        msg = (
            f"network.dataframe has no '{area_column}' column; the simple-scaling DA "
            "needs per-reach drainage area plumbed from the hydrofabric "
            "(flowpaths.total_da_sqkm). See the drainage-area stage."
        )
        raise KeyError(msg)
    area_sqkm = {int(seg): float(a) for seg, a in df[area_column].items()}

    rconn = {int(seg): [int(u) for u in ups] for seg, ups in network.reverse_network.items()}

    # link_gage_df: index 'link' (routed segment), column 'gages' (site_no).
    # Invert to site_no -> segment, dropping the 'nan' sentinel the ingest uses
    # for ungaged links, mirroring DataAssimilation.py's read of link_gage_df.
    link_gage = network.link_gage_df.reset_index().set_index("gages")
    site_col = link_gage.columns[0]
    gage_to_seg = {
        str(site): int(seg) for site, seg in link_gage[site_col].items() if str(site) != "nan"
    }

    crosswalk = network.link_lake_crosswalk or {}
    waterbody_segs = frozenset(int(seg) for seg in crosswalk)

    return build_gage_trees_from_mappings(
        rconn,
        gage_to_seg,
        area_sqkm,
        waterbody_segs=waterbody_segs,
        theta_default=theta_default,
        stop_at_waterbodies=stop_at_waterbodies,
        stop_at_upstream_gages=stop_at_upstream_gages,
        site_filter=site_filter,
    )
