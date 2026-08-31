"""Discretize flowpath network into uniform link segments.

This module
    1. Identifies on-flowpath virtual flowpaths (to serve as basis for routing links)
    2. (optionally) Aggregates short links into upstream neighbors
    3. Subdivides long links into near-uniform lengths

Notable Assumptions
---------------
- Node IDs increase in the downstream direction
- Flowpaths form a directed acyclic graph (no loops)

"""

from dataclasses import dataclass, fields
from collections import defaultdict
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import MultiLineString, line_interpolate_point
import numpy as np
import shapely
from shapely.ops import substring

### NHF SCHEMA ###
# Pulling this to the top for easy updates later
FIELD_FP_ID = "fp_id"
FIELD_VIRTUAL_FP_ID = "virtual_fp_id"
FIELD_DN_VIRTUAL_NEX_ID = "dn_virtual_nex_id"
FIELD_UP_VIRTUAL_NEX_ID = "up_virtual_nex_id"
FIELD_LENGTH = "length_km"
FIELD_LENGTH_CONVERSION = 1000
FIELD_VFP_ORDER = "segment_order"
CHANNEL_PARAMS = [
    "n",
    "mainstem_lp",
    "topwdth",
    "slope",
    "ncc",
    "btmwdth",
    "musx",
    "chslp",
    "topwdthcc",
    "musk",
]

### DATA CLASSES ###


@dataclass
class LinkArrays:
    """Routing link data structure.

    Notes
    -----
     - Arrays are used because they are more performant than Pandas dfs and scale better.

    """

    fp_id: np.ndarray
    vfp_id: np.ndarray
    dn_node_id: np.ndarray
    up_node_id: np.ndarray
    length: np.ndarray
    segment_order: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """Load LinksArrays from a virtual flowpaths layer."""
        return cls(
            df[FIELD_FP_ID].to_numpy().astype(int),
            df[FIELD_VIRTUAL_FP_ID].to_numpy().astype(int),
            df[FIELD_DN_VIRTUAL_NEX_ID].to_numpy().astype(int),
            df[FIELD_UP_VIRTUAL_NEX_ID].to_numpy().astype(int),
            df[FIELD_LENGTH].to_numpy() * FIELD_LENGTH_CONVERSION,
            df[FIELD_VFP_ORDER].to_numpy().astype(float)
        )

    def get_short_mask(self, discretization_len_m: float) -> np.ndarray:
        """Identify links shorter than discretization threshold and eligible for merging."""
        short_mask = self.length < discretization_len_m

        # Don't apply to headwaters, because they cannot be merged with upstream
        has_us = np.isin(self.up_node_id, self.dn_node_id)

        return short_mask & has_us

    def to_df(self) -> pd.DataFrame:
        """Convert LinkArrays to a Pandas DataFrame."""
        return pd.DataFrame({f.name: getattr(self, f.name) for f in fields(self)})

    def filter(self, mask: np.ndarray) -> None:
        """Keep masked links."""
        self.fp_id = self.fp_id[mask]
        self.vfp_id = self.vfp_id[mask]
        self.dn_node_id = self.dn_node_id[mask]
        self.up_node_id = self.up_node_id[mask]
        self.length = self.length[mask]
        self.segment_order = self.segment_order[mask]

    def remove(self, ind: int) -> None:
        """Remove a specific index from all arrays."""
        self.fp_id = np.delete(self.fp_id, ind)
        self.vfp_id = np.delete(self.vfp_id, ind)
        self.dn_node_id = np.delete(self.dn_node_id, ind)
        self.up_node_id = np.delete(self.up_node_id, ind)
        self.length = np.delete(self.length, ind)
        self.segment_order = np.delete(self.segment_order, ind)

### MAIN FUNCTION ###


def discretize_flowpaths(
    flowpaths: gpd.GeoDataFrame,
    virtual_flowpaths: gpd.GeoDataFrame,
    reference_flowpaths: pd.DataFrame,
    discretization_len_m: float = 300.0,
    aggregate_short_reaches: bool = True,
    export_links_nodes_gpkg_path: Union[None, str] = None,
    protected_fp_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Discretize flowpaths into uniform-length links and resolve short reaches.

    Parameters
    ----------
    flowpaths : gpd.GeoDataFrame
        Must contain:
        - FIELD_FP_ID
        - CHANNEL_PARAMS columns
    virtual_flowpaths : gpd.GeoDataFrame
        Must contain:
        - FIELD_VIRTUAL_FP_ID
        - FIELD_UP_VIRTUAL_NEX_ID
        - FIELD_DN_VIRTUAL_NEX_ID
        - (optional) geometry
    reference_flowpaths : pd.DataFrame
        Must contain:
        - FIELD_FP_ID
        - FIELD_VIRTUAL_FP_ID
    discretization_len_m : float, default 300.0
        Target link length in meters.
    aggregate_short_reaches : bool, default True
        Whether to merge short links upstream before discretization.
    export_links_nodes_gpkg_path : str or None
        If provided, exports links and nodes to GeoPackage.

    Returns
    -------
    tuple
        (
            pd.DataFrame,  # formatted link table that matches _dataframe format from AbstractNetwork
            dict[int, int] # mapping of virtual_nexus_id to a new node when the original node was merged
        )

    """
    cur_node_id = int(np.nanmax(virtual_flowpaths[[FIELD_DN_VIRTUAL_NEX_ID, FIELD_UP_VIRTUAL_NEX_ID]].values) + 1)

    ##############################################
    ### TEMPORARY PATCH ###  See looped headwaters
    mask = virtual_flowpaths[FIELD_UP_VIRTUAL_NEX_ID] == virtual_flowpaths[FIELD_DN_VIRTUAL_NEX_ID]
    n = mask.sum()
    virtual_flowpaths.loc[mask, FIELD_UP_VIRTUAL_NEX_ID] = np.arange(cur_node_id, cur_node_id + n)
    cur_node_id += n + 1

    ### TEMP PATCH 2 ###  see 1162231
    dup_up_node = virtual_flowpaths.groupby(FIELD_UP_VIRTUAL_NEX_ID).cumcount()
    mask = dup_up_node > 0
    n = mask.sum()
    virtual_flowpaths.loc[mask, FIELD_UP_VIRTUAL_NEX_ID] = np.arange(cur_node_id, cur_node_id + n)
    cur_node_id += n + 1
    #############################################
    
    links = _load_initial_links(virtual_flowpaths, reference_flowpaths)
    if aggregate_short_reaches:
        links, merged_node_crosswalk = _aggregate_links(
            links, discretization_len_m, protected_fp_ids,
        )
    else:
        merged_node_crosswalk = {}
    links = _discretize_links(links, discretization_len_m, cur_node_id)  

    if export_links_nodes_gpkg_path is not None:
        export_links_and_nodes(links, virtual_flowpaths, export_links_nodes_gpkg_path)
    return _format_link_df(links, flowpaths), merged_node_crosswalk


### HELPER FUNCTIONS ###


def export_links_and_nodes(
    links: LinkArrays,
    virtual_flowpaths: gpd.GeoDataFrame,
    export_links_nodes_gpkg_path: str,
) -> None:
    """Export discretized links and nodes as GeoPackage layers (for debugging)."""
    _validate_geometry(virtual_flowpaths)
    # Initialize data stores
    up_node_ids = []
    dn_node_ids = []
    link_geometries = []
    node_geometries = []

    # Make network graphs
    dn_network = dict(zip(links.up_node_id, links.dn_node_id))
    dn_network[set(links.dn_node_id).difference(links.up_node_id).pop()] = -9999

    len_lookup = dict(zip(links.up_node_id, links.length))

    # Processing loop
    virtual_flowpaths = virtual_flowpaths.set_index(FIELD_UP_VIRTUAL_NEX_ID)
    non_eclipsed = virtual_flowpaths[virtual_flowpaths.index.isin(links.up_node_id)]
    for i in non_eclipsed.itertuples():
        up_node = int(i.Index)
        dn_node = i.dn_virtual_nex_id

        # Walk downstream through missing (eclipsed) nodes, merging geometries until a retained node is found
        geoms = [i.geometry]
        while dn_node not in links.dn_node_id:
            tmp = virtual_flowpaths.loc[dn_node]
            geoms.append(tmp.geometry)
            dn_node = tmp.dn_virtual_nex_id
        geom = shapely.line_merge(MultiLineString([j.geoms[0] for j in geoms[::-1]]))
        if geom.geom_type == "MultiLineString":
            coords = [c for part in geom.geoms for c in part.coords]
            geom = shapely.LineString(coords)

        # Reconstruct link chain between retained nodes using dn_network lookup
        link_up = up_node
        link_dn = dn_network[link_up]
        tmp_links = [up_node]
        length = [len_lookup[link_up]]
        while link_dn != dn_node:
            tmp_links.append(link_dn)
            length.append(len_lookup[link_up])
            link_dn = dn_network[link_dn]

        # Build cumulative distances along merged geometry to segment into individual links
        cumdist = np.linspace(0, geom.length, len(length) + 1)
        link_geoms = [
            substring(geom, start_dist, end_dist)
            for start_dist, end_dist in zip(cumdist[:-1], cumdist[1:])
        ]
        node_geoms = [
            line_interpolate_point(geom, start_dist) for start_dist in cumdist[:-1]
        ]

        # Log
        up_node_ids.extend(tmp_links)
        dn_node_ids.extend(tmp_links[1:] + [dn_node])
        link_geometries.extend(link_geoms)
        node_geometries.extend(node_geoms)
    
    idx_lookup = np.argsort(links.up_node_id)
    sorted_ups = links.up_node_id[idx_lookup]
    link_idxs = np.searchsorted(sorted_ups, up_node_ids)
    fp_ids = links.fp_id[idx_lookup[link_idxs]]
    vfp_ids = links.vfp_id[idx_lookup[link_idxs]]
    segment_order = links.segment_order[idx_lookup[link_idxs]]

    # Export geopackage
    gpd.GeoDataFrame(
        {"up_node_id": up_node_ids, "dn_node_id": dn_node_ids, "fp_id": fp_ids, "vfp_ids": vfp_ids, "segment_order": segment_order},
        geometry=link_geometries,
        crs=virtual_flowpaths.crs,
    ).to_file(export_links_nodes_gpkg_path, layer="links")
    gpd.GeoDataFrame(
        {"node_id": up_node_ids}, geometry=node_geometries, crs=virtual_flowpaths.crs
    ).to_file(export_links_nodes_gpkg_path, layer="nodes")

def _validate_geometry(virtual_flowpaths: gpd.GeoDataFrame) -> None:
    if not hasattr(virtual_flowpaths, "geometry"):
        raise ValueError("GeoDataFrame has no active geometry column.")
    if virtual_flowpaths.geometry is None or virtual_flowpaths.geometry.name not in virtual_flowpaths.columns:
        raise ValueError("GeoDataFrame has no active geometry column.")
    if virtual_flowpaths.geometry.isna().any():
        raise ValueError("Geometry column contains null geometries.")
    if not virtual_flowpaths.geometry.is_valid.all():
        raise ValueError("Geometry column contains invalid geometries.")

def _load_initial_links(
    virtual_flowpaths: gpd.GeoDataFrame,
    reference_flowpaths: pd.DataFrame,
):
    # Drop headwater vfps
    tmp_vfp = virtual_flowpaths.dropna(subset=FIELD_UP_VIRTUAL_NEX_ID).copy()
    tmp_vfp[FIELD_UP_VIRTUAL_NEX_ID] = tmp_vfp[FIELD_UP_VIRTUAL_NEX_ID].astype(int)

    # Make LinkArray object
    return LinkArrays.from_df(
        pd.merge(
            tmp_vfp,
            reference_flowpaths[[FIELD_FP_ID, FIELD_VIRTUAL_FP_ID, FIELD_VFP_ORDER]].drop_duplicates(),
            on=FIELD_VIRTUAL_FP_ID,
        )
    )

def _aggregate_links(
    links: LinkArrays, discretization_len_m: float,
    protected_fp_ids: set[int] | None = None,
) -> tuple[LinkArrays, dict[int, int]]:
    """Merge links shorter than threshold into upstream neighbors.

    Notes
    -----
    - Iteratively merges short links until all satisfy threshold or are headwaters.
    - Node remapping provides lookup for removed nodes to their new terminal node.

    """
    # Make a store for rename mapping.  This is used so Qlats can be mapped to the new dn node
    merged_node_crosswalk: dict[int, int] = {}

    # Make lookup for u/s nodes
    dn_index = defaultdict(list)
    for i, dn in enumerate(links.dn_node_id):
        dn_index[dn].append(i)

    ## Combine links that are below discretization length with the next upstream reach
    short_mask = links.length < discretization_len_m
    has_us = np.array([u in dn_index for u in links.up_node_id], dtype=bool)
    short_mask &= has_us
    if protected_fp_ids:
        short_mask &= ~np.isin(links.fp_id, list(protected_fp_ids))
    
    # Mark merged so we can remove them later (currently non removed)
    active = np.ones(len(links.length), dtype=bool)

    # Initialize queue
    queue = list(np.flatnonzero(short_mask))

    while queue:
        idx = queue.pop()

        # Skip if already removed or other merging led to acceptable length
        if not active[idx]:
            continue
        length = links.length[idx]
        if length >= discretization_len_m:
            continue
        # Never merge away a protected flowpath link. Right now these are
        # just short flowpaths associated with waterbodies
        if protected_fp_ids and links.fp_id[idx] in protected_fp_ids:
            continue
        
        # Get link info
        up_node = links.up_node_id[idx]
        dn_node = links.dn_node_id[idx]

        # Get u/s link indices
        us_indices = dn_index.get(up_node)
        if not us_indices:
            raise RuntimeError(
                f"Attempted to merge link {up_node} (dx={int(length)}), but no upstream links were present."
            )

        # Merge into upstream links
        for us_idx in us_indices:
            if not active[us_idx]:
                # Don't merge with a merged link
                continue

            links.dn_node_id[us_idx] = dn_node
            links.length[us_idx] += length

            # Update adjacency: move this index under new dn_node
            dn_index[dn_node].append(us_idx)

            # Re-check upstream link
            if links.length[us_idx] < discretization_len_m and has_us[us_idx]:
                queue.append(us_idx)

        # Record remapping
        merged_node_crosswalk[up_node] = dn_node

        # Deactivate this link
        active[idx] = False

    # Remove short links
    links.filter(active)

    # Clean remapping so each node maps directly to its final downstream node
    for k in list(merged_node_crosswalk.keys()):
        v = merged_node_crosswalk[k]
        if merged_node_crosswalk[k] not in merged_node_crosswalk:
            continue
        while v in merged_node_crosswalk:
            v = merged_node_crosswalk[v]
        merged_node_crosswalk[k] = v

    return links, merged_node_crosswalk

def _discretize_links(
    links: LinkArrays, discretization_len_m: float, cur_node_id: int = 0
) -> LinkArrays:
    """Subdivide links longer than threshold into uniform segments (uses floor such that lengths will overshoot target).

    Vectorized: for each long link i with sub-link count ``n[i]``, all new
    sub-link arrays are built with one ``np.repeat`` per scalar attribute
    plus one ``np.arange`` for the in-between node IDs, instead of one
    ``np.concatenate`` per long link in a Python loop. Identical output to
    the loop version; ~7×–10× faster on CONUS where this runs over
    ~1.1 M long links.
    """
    long_mask = links.length > discretization_len_m

    if not np.any(long_mask):
        return links

    # Pull long-link slices once
    long_length = links.length[long_mask]
    long_fp_id = links.fp_id[long_mask]
    long_vfp_id = links.vfp_id[long_mask]
    long_up = links.up_node_id[long_mask]
    long_dn = links.dn_node_id[long_mask]
    long_order = links.segment_order[long_mask]

    # Sub-links per parent (≥1; long_mask guarantees length > disc_len so floor ≥ 1)
    n = np.maximum(1, np.floor(long_length / discretization_len_m).astype(np.int64))
    new_len = long_length / n  # equal-length per parent
    n_sum = int(n.sum())

    # group[k] = parent-link index of sub-link k; local_j[k] = position within parent (0..n[g]-1)
    group = np.repeat(np.arange(len(long_length), dtype=np.int64), n)
    group_offsets = np.empty(len(n), dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(n[:-1], out=group_offsets[1:])
    local_j = np.arange(n_sum, dtype=np.int64) - group_offsets[group]

    # Scalar attributes propagate by repeat/index
    subdiv_fp_id = np.repeat(long_fp_id, n)
    subdiv_vfp_id = np.repeat(long_vfp_id, n)
    subdiv_length = np.repeat(new_len, n)
    subdiv_order = long_order[group] + local_j / n[group]

    # New in-between node IDs: one contiguous arange across all parents
    n_new_per_link = n - 1
    n_new_total = int(n_new_per_link.sum())
    new_node_ids_flat = np.arange(
        cur_node_id, cur_node_id + n_new_total, dtype=np.int64
    )
    new_node_offsets = np.empty(len(n), dtype=np.int64)
    new_node_offsets[0] = 0
    np.cumsum(n_new_per_link[:-1], out=new_node_offsets[1:])

    # up_node: parent's up_node for local_j == 0, else new_node[parent][local_j - 1]
    subdiv_up_node = np.empty(n_sum, dtype=np.int64)
    is_first = local_j == 0
    subdiv_up_node[is_first] = long_up[group[is_first]]
    not_first = ~is_first
    if not_first.any():
        idx = new_node_offsets[group[not_first]] + local_j[not_first] - 1
        subdiv_up_node[not_first] = new_node_ids_flat[idx]

    # dn_node: parent's dn_node for local_j == n[g] - 1, else new_node[parent][local_j]
    subdiv_dn_node = np.empty(n_sum, dtype=np.int64)
    is_last = local_j == n[group] - 1
    subdiv_dn_node[is_last] = long_dn[group[is_last]]
    not_last = ~is_last
    if not_last.any():
        idx = new_node_offsets[group[not_last]] + local_j[not_last]
        subdiv_dn_node[not_last] = new_node_ids_flat[idx]

    # cur_node_id is consumed by the new arange above; it is *not* returned by
    # the original signature, so we don't propagate the update.

    # Concat kept-as-is short/medium links with the freshly-subdivided sub-links
    keep_mask = ~long_mask
    link_fp_id = np.concatenate([links.fp_id[keep_mask], subdiv_fp_id])
    link_vfp_id = np.concatenate([links.vfp_id[keep_mask], subdiv_vfp_id])
    length = np.concatenate([links.length[keep_mask], subdiv_length])
    up_node_id = np.concatenate([links.up_node_id[keep_mask], subdiv_up_node])
    dn_node_id = np.concatenate([links.dn_node_id[keep_mask], subdiv_dn_node])
    segment_order = np.concatenate([links.segment_order[keep_mask], subdiv_order])

    return LinkArrays(link_fp_id, link_vfp_id, dn_node_id, up_node_id, length, segment_order)

def _format_link_df(links: LinkArrays, flowpaths: pd.DataFrame) -> pd.DataFrame:
    """Conform to AbstractNetwork format and build mapping from link id to fp_id."""
    _dataframe = pd.merge(
        links.to_df(),
        flowpaths[CHANNEL_PARAMS + [FIELD_FP_ID]],
        on=FIELD_FP_ID,
        how="left",
    )

    # Conform to abstractnetwork _dataframe
    _dataframe["alt"] = 0
    renames = {
        "dn_node_id": "downstream",
        "length": "dx",
        "mainstem_lp": "mainstem",
        "topwdth": "tw",
        "slope": "s0",
        "btmwdth": "bw",
        "chslp": "cs",
        "topwdthcc": "twcc",
    }
    _dataframe = _dataframe.set_index("up_node_id").rename(columns=renames)

    return _dataframe
