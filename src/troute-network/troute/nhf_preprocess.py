import logging
from pathlib import Path
from itertools import starmap
from typing import Optional

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
from troute.rfc_lake_gage_crosswalk import get_great_lakes_climatology
import xarray as xr
from joblib import Parallel, delayed

LOG = logging.getLogger("TROUTE")

# Great Lakes lake ids: present in the NHF lakes layer but carrying no
# level-pool parameters (LkArea is NaN); their flows come exclusively from
# data assimilation, so they are excluded from reservoir routing.
GREAT_LAKES_IDS = (4800002, 4800004, 4800006, 4800007)

# Channel-parameter columns of `flowpaths` consumed by the Muskingum-Cunge
# kernel. Non-finite (NaN/Inf) values in any of these would propagate
# into NaN routing output; guard against them at load time and fail loud.
_FLOWPATHS_CHANNEL_COLS = (
    "length_km", "n", "slope", "topwdth", "btmwdth",
    "topwdthcc", "ncc", "chslp", "musx", "musk", "mainstem_lp",
)
_BAD_FPID_PREVIEW_LIMIT = 10
LAKE_ID_FIELD = "nhf_lake_id"
RECORD_LAKE_ID_FIELD = "og_" + LAKE_ID_FIELD
NATIVE_LAKE_ID_FIELD = "lake_id"  # Index of lake in its source dataset
WATERBODY_DF_FIELDS = [
                LAKE_ID_FIELD,
                NATIVE_LAKE_ID_FIELD,
                "fp_id",
                "virtual_fp_id",
                "ifd",
                "LkArea",
                "LkMxE",
                "OrificeA",
                "OrificeC",
                "OrificeE",
                "WeirC",
                "WeirE",
                "WeirL",
            ]
RESERVOIR_DA_SITE_ID_FIELD = "site_no"
RESERVOIR_DA_SITE_TYPE_FIELD = "da_type"

def _sql_in(field: str, values, quote: bool = False) -> str:
    """An OGR ``where`` clause restricting *field* to *values*."""
    vals = sorted({str(v) for v in values if pd.notna(v)})
    if quote:
        items = ",".join("'" + v.replace("'", "''") + "'" for v in vals)
    else:
        items = ",".join(str(int(float(v))) for v in vals)
    return f"{field} IN ({items})"


def _lake_vfp_clusters(
    waterbody_df: pd.DataFrame,
    crosswalk: "pd.DataFrame | None",
) -> tuple["pd.Series[int]", dict[int, int]]:
    """Cluster lakes that share a virtual flowpath; returns lake and vfp labels.

    Lakes sharing a vfp must collapse together: a vfp belongs to one absorbed link
    set, and two claims would pop the same link out of ``connections`` twice.
    Without a crosswalk this degenerates to ``groupby("virtual_fp_id")``.
    """
    lakes = waterbody_df["virtual_fp_id"].dropna()
    edges = list(
        zip(
            lakes.index.to_numpy().astype(int).tolist(),
            lakes.to_numpy().astype(int).tolist(),
        )
    )
    if crosswalk is not None and not crosswalk.empty:
        # Crosswalk is keyed on the ORIGINAL nhf_lake_id; the waterbody table has
        # synthetic ids by now. Great Lakes carry their native lake_id and so do
        # not match -- they stay anchored on their declared vfp alone.
        record_to_index = pd.Series(
            waterbody_df.index, index=waterbody_df[RECORD_LAKE_ID_FIELD]
        )
        cw = crosswalk.dropna(subset=[LAKE_ID_FIELD, "virtual_fp_id"])
        mapped = record_to_index.reindex(cw[LAKE_ID_FIELD].to_numpy())
        keep = mapped.notna().to_numpy()
        edges.extend(
            zip(
                mapped.to_numpy()[keep].astype(int).tolist(),
                cw["virtual_fp_id"].to_numpy()[keep].astype(int).tolist(),
            )
        )

    # Union-find over the (lake, vfp) graph. Keys are TAGGED: synthetic lake ids
    # are allocated above max(dataframe.index), which bounds them against routing
    # node ids but NOT against virtual_fp_id, so bare ints could merge unrelated
    # lakes through a collision.
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        root = parent.setdefault(x, x)
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for lake, vfp in edges:
        root_l, root_v = find(("lake", lake)), find(("vfp", vfp))
        if root_l != root_v:
            parent[root_l] = root_v

    # An edgeless lake gets a null label so the caller's groupby drops it, matching
    # what groupby("virtual_fp_id") did with its NaN key. Demoting it instead would
    # unhook a Great Lake anchored only by fp_id from its type-6 DA.
    edge_lakes = {lake for lake, _ in edges}
    roots: dict[tuple[str, int], int] = {}

    def _label(root: tuple[str, int]) -> int:
        return roots.setdefault(root, len(roots))

    lake_cluster = pd.Series(
        [
            _label(find(("lake", i))) if i in edge_lakes else pd.NA
            for i in waterbody_df.index.to_numpy().astype(int).tolist()
        ],
        index=waterbody_df.index,
        dtype="Int64",
    )
    vfp_cluster = {vfp: _label(find(("vfp", vfp))) for _, vfp in edges}
    return lake_cluster, vfp_cluster


def _outlet_link_per_lake(
    outlet_vfps: "pd.Series[float]",
    vfp_ids: "NDArray[np.int64]",
    up_nodes: "NDArray[np.int64]",
    downstream: "NDArray[np.int64]",
) -> dict[int, int]:
    """Each lake's outlet link: the one link of its declared flowpath whose
    downstream leaves that flowpath. Empty if any flowpath fails to resolve to
    exactly one, which is the caller's signal to fall back."""
    outlets: dict[int, int] = {}
    for vfp in pd.unique(outlet_vfps.dropna().to_numpy()):
        rows = np.flatnonzero(vfp_ids == vfp)
        if rows.size == 0:
            continue
        internal = set(up_nodes[rows])
        terminal = [row for row in rows if downstream[row] not in internal]
        if len(terminal) != 1:
            return {}
        outlets[int(vfp)] = int(terminal[0])
    return outlets


def _links_by_nearest_outlet(
    links: pd.DataFrame, outlet_vfps: "pd.Series[float]",
) -> dict[int, pd.DataFrame]:
    """Split a cluster's links among its lakes, each taking what drains to IT.

    Multi-source upstream walk from every lake outlet; each link goes to the first
    outlet reaching it. This buys two things over one set per cluster: nothing
    below a lake's outlet is absorbed (the crosswalk lists flowpaths INTERSECTING
    the polygon, and 26% of CONUS lakes have one continuing past the dam), and
    serial lakes chain via topology rather than DataFrame row order. Each returned
    set has exactly one exit by construction.

    Returns ``{outlet virtual_fp_id: links}``; empty means fall back.
    """
    up_nodes = links["up_node_id"].to_numpy()
    downstream = links["downstream"].to_numpy()
    vfp_ids = links["vfp_id"].to_numpy()

    feeders: dict[int, list[int]] = {}
    for row, node in enumerate(downstream):
        feeders.setdefault(node, []).append(row)

    outlets = _outlet_link_per_lake(outlet_vfps, vfp_ids, up_nodes, downstream)
    if not outlets:
        return {}

    # Breadth-first so "first claim" means nearest by link count; a claimed link is
    # not re-expanded, which stops a downstream lake swallowing an upstream one's
    # catchment.
    owner: dict[int, int] = {}
    frontier = [(row, vfp) for vfp, row in outlets.items()]
    for row, vfp in frontier:
        owner.setdefault(row, vfp)
    while frontier:
        nxt: list[tuple[int, int]] = []
        for row, vfp in frontier:
            for feeder in feeders.get(up_nodes[row], ()):
                if feeder not in owner:
                    owner[feeder] = vfp
                    nxt.append((feeder, vfp))
        frontier = nxt

    claimed: dict[int, list[int]] = {}
    for row, vfp in owner.items():
        claimed.setdefault(vfp, []).append(row)
    return {vfp: links.iloc[sorted(rows)] for vfp, rows in claimed.items()}


def _validate_flowpaths_channel_params(flowpaths):
    """Raise if any MC-kernel channel parameter is non-finite (NaN/Inf)."""
    if flowpaths is None or flowpaths.empty:
        return
    cols = [c for c in _FLOWPATHS_CHANNEL_COLS if c in flowpaths.columns]
    if not cols:
        return
    arr = flowpaths[cols].to_numpy(dtype=float, copy=False, na_value=np.nan)
    bad_per_col = ~np.isfinite(arr)
    if not bad_per_col.any():
        return
    bad_row_mask = bad_per_col.any(axis=1)
    bad_count = int(bad_row_mask.sum())
    per_col = {c: int(bad_per_col[:, i].sum())
               for i, c in enumerate(cols) if bad_per_col[:, i].any()}
    bad_fp_ids = (flowpaths.loc[bad_row_mask, "fp_id"].tolist()
                  if "fp_id" in flowpaths.columns else [])
    preview = bad_fp_ids[:_BAD_FPID_PREVIEW_LIMIT]
    more = ("" if len(bad_fp_ids) <= _BAD_FPID_PREVIEW_LIMIT
            else f" (and {len(bad_fp_ids) - _BAD_FPID_PREVIEW_LIMIT} more)")
    raise ValueError(
        f"flowpaths contains {bad_count} of {len(flowpaths)} segments with "
        f"non-finite (NaN/Inf) channel parameter(s); the Muskingum-Cunge "
        f"kernel requires finite values. Affected columns: {per_col}. "
        f"Affected fp_ids{more}: {preview}"
    )

# Layers a geopackage may legitimately omit (read_geo_file loads them as empty
# frames); validation must not reject a valid lake-free / non-reservoir domain.
OPTIONAL_LAYERS: frozenset[str] = frozenset(
    {"lakes", "reservoir_da", "lake_vfp_crosswalk"}
)

# Columns a PRESENT layer may omit: NHF >= 1.1.4 only, consumed only by the
# scaling DA (which raises its own clear error when enabled without them).
OPTIONAL_COLUMNS: dict[str, frozenset[str]] = {
    "flowpaths": frozenset({"total_da_sqkm", "vpu_id"}),
}

# Columns required from layers loaded in FULL (``columns=None``), which name no
# column list for the check above to use. Enforced only when the layer is PRESENT,
# so a gage-free domain stays valid. gages.hy_id is the only key tying a gage to its
# hydrolocation (site_no and gid both repeat); nhf 1.2.3 dropped it, which without
# this surfaces as a KeyError deep inside a pandas merge.
REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "gages": frozenset({"hy_id"}),
}


def _missing_requested_columns(
    available_by_layer: dict[str, set],
) -> dict[str, list]:
    """Pure check: which requested columns are absent from each layer.

    ``available_by_layer`` maps a layer name to the columns it carries; omit a
    layer to signal it is absent. Layers with an explicit ``columns`` list are
    checked against it, ``columns=None`` layers only against ``REQUIRED_COLUMNS``
    and only when present. Returns ``{layer_name: [missing columns]}``.
    """
    missing_by_layer: dict[str, list] = {}
    for name, columns, _ in LAYERS_TO_READ:
        must_have = REQUIRED_COLUMNS.get(name, frozenset())
        available = available_by_layer.get(name)
        if columns is None:
            # Loaded in full: nothing to check unless the schema pins a column,
            # and an absent layer of this kind is not itself an error.
            if must_have and available is not None:
                absent = sorted(c for c in must_have if c not in available)
                if absent:
                    missing_by_layer[name] = absent
            continue
        optional_cols = OPTIONAL_COLUMNS.get(name, frozenset())
        required = [c for c in columns if c not in optional_cols]
        required += [c for c in sorted(must_have) if c not in required]
        if available is None:
            # An absent optional layer is fine; an absent core topology layer is not.
            if name not in OPTIONAL_LAYERS:
                missing_by_layer[name] = required
            continue
        absent = [c for c in required if c not in available]
        if absent:
            missing_by_layer[name] = absent
    return missing_by_layer


def _validate_required_columns(gpkg_path: Path, present_layers: set[str]) -> None:
    """Fail fast -- from layer metadata, before any rows are read -- if the
    geopackage is missing a column the NHF build requests.

    Each layer's requested ``columns`` doubles as its required set (we only
    ask for columns the build consumes downstream). ``present_layers`` is the
    set of layers actually in the geopackage (from ``pyogrio.list_layers``); a
    validated layer absent from it is reported as missing its full requested
    set, and a present one is checked against ``pyogrio.read_info(...)["fields"]``
    (the attribute field names, read without touching the rows). This costs one
    metadata lookup per present validated layer and catches a stale hydrofabric
    (e.g. ``reference_flowpaths`` lacking ``segment_order``) up front, replacing
    a cryptic ``KeyError`` raised deep inside discretization. Layers loaded with
    ``columns=None`` (lakes, gages, hydrolocations, virtual_nexus) name no column
    list, so they are checked only against ``REQUIRED_COLUMNS`` and only when
    present -- their absence stays legal.
    """
    available_by_layer: dict[str, set[str]] = {}
    for name, columns, _ in LAYERS_TO_READ:
        if name not in present_layers:
            continue
        if columns is None and name not in REQUIRED_COLUMNS:
            continue
        available_by_layer[name] = set(
            pyogrio.read_info(gpkg_path, layer=name)["fields"]
        )
    missing_by_layer = _missing_requested_columns(available_by_layer)
    if missing_by_layer:
        details = "; ".join(
            f"{name}: {cols}" for name, cols in missing_by_layer.items()
        )
        raise ValueError(
            "Input geopackage is missing required column(s) needed by the NHF "
            f"network build -> {details}. Usually the hydrofabric predates the "
            "current schema (older datasets lack 'segment_order' in "
            "'reference_flowpaths'); it can also be a newer build that dropped a "
            "required column (nhf 1.2.3 dropped 'hy_id' from 'gages'). Regenerate "
            "or switch to a compatible hydrofabric version."
        )


# Layers to read from the NHF geopackage, as (name, columns, ignore_geometry)
# tuples. ``columns`` is the explicit list of fields to load (and doubles as
# the required-column set validated up front by _validate_required_columns),
# or None to load every field. We read only what the build consumes to cut
# processing time and memory. ``reference_flowpaths`` lists its five consumed
# columns explicitly: `segment_order` is a newer hydrofabric field whose
# absence otherwise fails deep in discretization, and `ref_fp_id` is the join
# key in crosswalk_nex_flowpath_poi.
LAYERS_TO_READ: list[tuple[str, Optional[list[str]], bool]] = [
    (
        "flowpaths",
        ["fp_id", "length_km", "n", "mainstem_lp", "topwdth", "slope",
         "ncc", "btmwdth", "musx", "chslp", "topwdthcc", "musk",
         # scaling-DA-only fields (NHF >= 1.1.4, optional): drainage area for the
         # area scaling, and the VPU the per-tree theta is regionalized from.
         "total_da_sqkm",
         "vpu_id"],
        True,
    ),
    (
        "reference_flowpaths",
        ["ref_fp_id", "fp_id", "virtual_fp_id", "segment_order", "div_id"],
        True,
    ),
    (
        "virtual_flowpaths",
        ["length_km", "virtual_fp_id", "dn_virtual_nex_id",
         "up_virtual_nex_id", "percentage_area_contribution"],
        False,
    ),
    ("virtual_nexus", None, True),
    (
        "lakes",
        WATERBODY_DF_FIELDS + ["hy_id", "ref_fp_id"],
        False),
    ("gages", None, True),
    ("hydrolocations", None, True),
    ("reservoir_da", ["nhf_lake_id", "lake_id", "site_no", "da_type"], True),
    # Every vfp intersecting each lake polygon, one lake to many (NHF >= 1.2.2).
    # Without it _refactor_reservoirs absorbs only the declared outlet vfp and
    # routes the rest of the lake as MC channel.
    ("lake_vfp_crosswalk", ["nhf_lake_id", "virtual_fp_id"], True),
]

def read_qlat_file(f):
    df = read_file(f)

    if df["feature_id"].dtype == str:
        df["feature_id"] = df["feature_id"].str.removeprefix("nex-").astype(int)

    if not df["feature_id"].is_unique:
        raise ValueError(
            f"'feature_id's must be unique. '{f!s}' contains duplicate "
            f"'feature_id's: {df.loc[df['feature_id'].duplicated(), 'feature_id'].to_list()}"
        )

    return df.set_index("feature_id")

def read_ngen_waterbody_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """Reads .gpkg or lake.json file and prepares a dataframe, filtered
    to the relevant reservoirs, to provide the parameters
    for level-pool reservoir computation.
    """

    def node_key_func(x):
        return int(x.split("-")[-1])

    if Path(parm_file).suffix == ".gpkg":
        df = gpd.read_file(parm_file, layer="lakes")

        # The lake key differs by vintage (v2.01: hl_link, no lake_id; v2.2:
        # lake_id, no hl_link); key off what is present, drop with errors="ignore".
        if "lake_id" not in df.columns:
            if "hl_link" not in df.columns:
                msg = (
                    f"{parm_file}: the 'lakes' layer has neither a 'lake_id' nor an "
                    f"'hl_link' column, so the lake id cannot be resolved; columns are "
                    f"{sorted(df.columns)}"
                )
                raise KeyError(msg)
            df = df.rename(columns={"hl_link": "lake_id"})
        df = df.drop(
            ["id", "toid", "hl_id", "hl_reference", "hl_uri", "geometry"],
            axis=1,
            errors="ignore",
        )
        df["lake_id"] = df.lake_id.astype(float).astype(int)
        df = df.set_index("lake_id").drop_duplicates().sort_index()
    elif Path(parm_file).suffix == ".json":
        df = pd.read_json(parm_file, orient="index")
        df.index = df.index.map(node_key_func)
        df.index.name = lake_index_field

    if lake_id_mask:
        df = df.loc[lake_id_mask]
    return df


def read_ngen_waterbody_type_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """ """

    # FIXME: this function is likely not correct. Unclear how we will get
    # reservoir type from the gpkg files. Information should be in 'crosswalk'
    # layer, but as of now (Nov 22, 2022) there doesn't seem to be a differentiation
    # between USGS reservoirs, USACE reservoirs, or RFC reservoirs...
    def node_key_func(x):
        return int(x.split("-")[-1])

    if Path(parm_file).suffix == ".gpkg":
        df = gpd.read_file(parm_file, layer="crosswalk").set_index("id")
    elif Path(parm_file).suffix == ".json":
        df = pd.read_json(parm_file, orient="index")

    df.index = df.index.map(node_key_func)
    df.index.name = lake_index_field
    if lake_id_mask:
        df = df.loc[lake_id_mask]

    return df


def read_geo_file(supernetwork_parameters, cpu_pool):
    geo_file_path = supernetwork_parameters["geo_file_path"]
    if Path(geo_file_path).suffix != ".gpkg":
        raise RuntimeError("Only .gpkg files are currently supported for the geo_file_path parameter.")

    # Inspect the geopackage once up front (metadata only): which layers are
    # present, and -- via _validate_required_columns -- do they carry the
    # columns the build needs. A stale/incomplete hydrofabric fails here,
    # before we pay to read any rows.
    gpkg_layers = {name for name, _ in pyogrio.list_layers(geo_file_path)}
    _validate_required_columns(geo_file_path, gpkg_layers)

    def read_layer(
        name: str, columns: Optional[list[str]], ignore_geometry: bool,
    ) -> tuple[str, pd.DataFrame]:
        return name, gpd.read_file(
            geo_file_path, layer=name, columns=columns,
            ignore_geometry=ignore_geometry,
        )

    # Read present layers in parallel (absent ones become empty frames), pruning
    # OPTIONAL columns to what the layer carries -- gpd.read_file raises on a
    # requested column the layer lacks. Required columns were validated above.
    to_read = []
    for name, columns, ignore_geometry in LAYERS_TO_READ:
        if name not in gpkg_layers:
            continue
        optional = OPTIONAL_COLUMNS.get(name)
        if columns is not None and optional:
            fields = set(pyogrio.read_info(geo_file_path, layer=name)["fields"])
            columns = [c for c in columns if c not in optional or c in fields]
        to_read.append((name, columns, ignore_geometry))
    if not to_read:
        raise ValueError(
            f"None of the expected layers to read were present in the geopackage: "
            f"{[lyr for lyr, _, _ in LAYERS_TO_READ]}. Found layers: {gpkg_layers}."
        )
    table_dict = {lyr: pd.DataFrame() for lyr, *_ in LAYERS_TO_READ}
    with Parallel(n_jobs=min(cpu_pool, len(to_read))) as parallel:
        table_dict.update(
            dict(parallel(starmap(delayed(read_layer), to_read)))
        )

    _validate_flowpaths_channel_params(table_dict.get("flowpaths"))
    return table_dict


def load_bmi_data(
    value_dict,
    bmi_parameters,
):
    # Get the column names that we need from each table of the geopackage
    flowpath_columns = bmi_parameters.get("flowpath_columns")
    attributes_columns = bmi_parameters.get("attributes_columns")
    lakes_columns = bmi_parameters.get("waterbody_columns")
    network_columns = bmi_parameters.get("network_columns")

    # Create dataframes with the relevent columns
    flowpaths = pd.DataFrame(data=None, columns=flowpath_columns)
    for col in flowpath_columns:
        flowpaths[col] = value_dict[col]

    flowpath_attributes = pd.DataFrame(data=None, columns=attributes_columns)
    for col in attributes_columns:
        flowpath_attributes[col] = value_dict[col]
    flowpath_attributes = flowpath_attributes.rename(columns={"attributes_id": "id"})

    lakes = pd.DataFrame(data=None, columns=lakes_columns)
    for col in lakes_columns:
        lakes[col] = value_dict[col]

    network = pd.DataFrame(data=None, columns=network_columns)
    for col in network_columns:
        network[col] = value_dict[col]
    network = network.rename(columns={"network_id": "id"})

    # Merge the two flowpath tables into one
    flowpaths = pd.merge(flowpaths, flowpath_attributes, on="id")

    return flowpaths, lakes, network


def read_file(file_name):
    extension = file_name.suffix
    if extension == ".csv":
        df = pd.read_csv(file_name)
    elif extension == ".parquet":
        df = pq.read_table(file_name).to_pandas().reset_index()
        df.index.name = None
    elif extension == ".nc":
        nc = xr.open_dataset(file_name)
        ts = str(nc.get("time").values)
        df = nc.to_pandas().reset_index()[["feature_id", "q_lateral"]]
        df.rename(columns={"q_lateral": f"{ts}"}, inplace=True)
        df.index.name = None

    return df


def _groupby_to_list_dict(df, key_col, val_col):
    """Vectorized equivalent of ``df.groupby(key_col)[val_col].apply(list).to_dict()``.

    Pandas ``groupby.apply(list)`` builds Python lists per group via a
    per-row Python loop -- the per-row overhead dominates at CONUS scale
    (1.1 M rows = ~700 ms per call). This function does the same work in
    pure numpy: argsort, find group boundaries, split, then tolist. About
    3x faster on uniform-distribution CONUS-shape inputs; can be much
    faster on skewed distributions where pandas' per-group fallback is
    slow.

    Matches pandas semantics for the cases this helper is called on:
      * NaN keys are dropped (pandas ``groupby(..., dropna=True)`` default).
        Without this mask numpy would produce a single ``nan`` key in the
        output dict because ``np.argsort`` sorts NaN to the end and
        ``np.unique`` returns each NaN as its own equality class.
      * Numeric keys are unboxed via ``.item()`` so the dict has python
        ``int`` / ``float`` keys (matches pandas' ``.to_dict()`` boxing).
      * Object-dtype keys (python strings, etc.) are returned as-is.

    The helper is intentionally narrow: it expects the key column to be
    numeric or string. For nullable / extension dtypes (Int64, string[python],
    etc.) fall back to pandas at the caller side, since ``to_numpy()``
    behavior on those types is dtype-dependent and would require a more
    elaborate dispatch.
    """
    if df.empty:
        return {}
    keys = df[key_col].to_numpy()
    vals = df[val_col].to_numpy()
    # Drop rows whose key is NaN/NaT, matching pandas' dropna=True default.
    # pd.notna handles float NaN, datetime NaT, and object None uniformly.
    if keys.dtype.kind in "fcmM" or keys.dtype == object:
        mask = pd.notna(keys)
        if not mask.all():
            keys = keys[mask]
            vals = vals[mask]
    if keys.size == 0:
        return {}
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_vals = vals[order]
    unique_keys, group_starts = np.unique(sorted_keys, return_index=True)
    groups = np.split(sorted_vals, group_starts[1:])
    # Object-dtype keys (e.g. Python strings) iterate as raw Python
    # objects and have no .item(); numpy scalars do. Branch on dtype
    # so we don't silently box numpy ints/floats into numpy scalars.
    if unique_keys.dtype == object:
        return {k: g.tolist() for k, g in zip(unique_keys, groups)}
    return {k.item(): g.tolist() for k, g in zip(unique_keys, groups)}


def _clean_waterbodies(
    waterbody_df: pd.DataFrame, lake_id_field: str = "lake_id"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step-by-step NHF waterbody cleanup, with every dropped category counted
    and logged as a warning so data problems are visible instead of silently
    filtered.

    Steps, in order (mirroring the historical inline cleaning):
      1. lake_id integrity: coerce to numeric (the column is text in
         NHF >= 1.2.0) and drop rows whose lake_id cannot be parsed.
      2. index + dedup: set lake_id as the index and drop duplicated rows
         (pre-existing semantics: duplicates are judged on the parameter
         columns only, since pandas ignores the index).
      3. Great Lakes: extracted for the climatology / data-assimilation
         wiring; they carry no level-pool parameters in NHF and are not
         routed as reservoirs.
      4. elevation consistency: OrificeE <= WeirE <= LkMxE must hold for
         level-pool routing; violating lakes are dropped with a warning
         (previously the whole run failed on the first violation).
      5. virtual_fp_id anchoring: drop lakes with no virtual_fp_id -- they
         cannot be anchored to a routing flowpath (the hydrofabric fix for
         these is tracked upstream).
      6. parameter completeness: drop lakes missing any level-pool
         parameter.

    Returns ``(clean_df, gl_df)``: the routable waterbody table (integer
    lake_id index, sorted, fp_id as int) and the raw Great Lakes rows.
    """
    n_raw = len(waterbody_df)

    # 1. lake_id integrity
    lake_ids = pd.to_numeric(waterbody_df[lake_id_field], errors="coerce")
    n_bad_id = int(lake_ids.isna().sum())
    if n_bad_id:
        LOG.warning(
            "waterbodies: dropped %d of %d lakes with a non-numeric or "
            "missing lake_id", n_bad_id, n_raw,
        )
        waterbody_df = waterbody_df[lake_ids.notna()].copy()
        lake_ids = lake_ids[lake_ids.notna()]
    waterbody_df.loc[:, lake_id_field] = lake_ids.astype(int)

    # 2. index + dedup
    waterbody_df = waterbody_df.set_index(lake_id_field)
    n_before = len(waterbody_df)
    waterbody_df = waterbody_df.drop_duplicates().sort_index()
    n_dup = n_before - len(waterbody_df)
    if n_dup:
        LOG.warning("waterbodies: dropped %d duplicated parameter rows", n_dup)

    # 3. Great Lakes
    gl_str = [str(i) for i in GREAT_LAKES_IDS]
    gl_mask = waterbody_df[NATIVE_LAKE_ID_FIELD].astype(str).isin(gl_str)
    gl_df = waterbody_df[gl_mask].copy()
    if not gl_df.empty:
        LOG.warning(
            "waterbodies: %d Great Lakes present; they carry no level-pool "
            "parameters and are handled separately from level-pool reservoirs "
            "(modeled as reservoir_type 6 via data assimilation when Great "
            "Lakes persistence DA is enabled, otherwise left out of the "
            "reservoir set and routed as MC channels). See preprocess_waterbodies.",
            len(gl_df),
        )
        # Remove the Great Lakes from the level-pool routable set entirely. They
        # re-enter only through _great_lakes_for_da (as reservoir_type 6, by
        # original id). Doing this here -- rather than relying on the dropna in
        # step 6 to drop them for missing parameters -- also prevents a Great Lake
        # that happens to carry valid level-pool parameters from surviving,
        # getting synthetic-renamed, and silently routing as a type-1 reservoir.
        waterbody_df = waterbody_df[~gl_mask]

    # 4. elevation consistency: a level-pool reservoir violating
    # OrificeE <= WeirE <= LkMxE is physically inconsistent and cannot be
    # routed; drop it and warn instead of failing the whole run.
    bad_elev = (waterbody_df["OrificeE"] > waterbody_df["WeirE"]) | (
        waterbody_df["WeirE"] > waterbody_df["LkMxE"]
    )
    n_bad_elev = int(bad_elev.sum())
    if n_bad_elev:
        LOG.warning(
            "waterbodies: dropped %d lakes with inconsistent elevations "
            "(OrificeE <= WeirE <= LkMxE must hold for level-pool routing)",
            n_bad_elev,
        )
        LOG.debug(
            "inconsistent-elevation lake ids: %s",
            waterbody_df.index[bad_elev].tolist(),
        )
        waterbody_df = waterbody_df[~bad_elev]

    # 5. virtual_fp_id anchoring
    fp_na = waterbody_df["virtual_fp_id"].isna()
    n_no_fp = int(fp_na.sum())
    if n_no_fp:
        LOG.warning(
            "waterbodies: dropped %d lakes with no virtual_fp_id (cannot be anchored "
            "to a routing flowpath)", n_no_fp,
        )
        waterbody_df = waterbody_df[~fp_na]

    # 6. parameter completeness
    n_before = len(waterbody_df)
    waterbody_df = waterbody_df.dropna()
    n_no_param = n_before - len(waterbody_df)
    if n_no_param:
        LOG.warning(
            "waterbodies: dropped %d lakes missing level-pool parameters",
            n_no_param,
        )

    waterbody_df = waterbody_df.copy()
    waterbody_df.loc[:, "virtual_fp_id"] = waterbody_df["virtual_fp_id"].astype(int)
    summary = (
        "waterbodies: %d of %d lakes retained for reservoir routing",
        len(waterbody_df), n_raw,
    )
    if len(waterbody_df) < n_raw:
        # Dropped lakes require special care from t-route consumers: their
        # flowpaths route as plain MC channels, not reservoirs.
        LOG.warning(*summary)
    else:
        LOG.info(*summary)
    return waterbody_df, gl_df


def _great_lakes_for_da(gl_df: pd.DataFrame, data_assimilation_parameters: dict) -> tuple[pd.DataFrame, bool]:
    """Select the Great Lakes to include in the routable waterbody set as
    ``reservoir_type`` 6 (data-assimilation driven), and report whether Great
    Lakes persistence DA is enabled.

    The Great Lakes carry no level-pool parameters, so they can only be modeled
    as type-6 reservoirs whose flows come from data assimilation. They are
    included only when Great Lakes persistence DA is enabled; otherwise
    ``compute.py`` demotes the type-6 reservoirs to level pool and the kernel
    crashes on their missing parameters (with DA off their flowpaths still route
    Only Great Lakes with a ``virtual_fp_id`` can be anchored to a flowpath; that
    id is cast to int to match the link table.

    Parameters
    ----------
    gl_df : pandas.DataFrame
        Great Lakes rows extracted in :func:`_clean_waterbodies` (original
        lake-id index, level-pool parameter columns, possibly NaN ``fp_id``).
    data_assimilation_parameters : dict
        The network's data-assimilation configuration.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        ``(anchored_gl_df, gl_da_enabled)`` -- the Great Lakes to re-add (empty
        when DA is disabled or none can be anchored) and the DA-enabled flag.
    """
    gl_da_enabled = bool(
        data_assimilation_parameters.get("reservoir_da", {})
        .get("reservoir_persistence_da", {})
        .get("reservoir_persistence_greatLake", False)
    )
    if not gl_da_enabled or gl_df.empty:
        return gl_df.iloc[0:0].copy(), gl_da_enabled
    # Keep a Great Lake if EITHER anchor is present, deciding per row rather than
    # per column. _refactor_reservoirs resolves reservoirs through the virtual
    # flowpath, and on nhf_1.2.2 the two Great Lakes carrying real USGS gages
    # (04127885 and 04159130) have a null fp_id but a valid virtual_fp_id, so
    # filtering on fp_id alone silently dropped exactly the lakes this function
    # exists to keep. Choosing a single column for the whole frame has the mirror
    # failure: a lake with a valid fp_id and a null virtual_fp_id would be dropped
    # even though it is perfectly anchorable.
    anchors = [c for c in ("virtual_fp_id", "fp_id") if c in gl_df.columns]
    if not anchors:
        return gl_df.iloc[0:0].copy(), gl_da_enabled
    keep = gl_df[anchors].notna().any(axis=1)
    anchored = gl_df[keep].copy()
    for col in anchors:
        # Int64 (nullable) so a lake anchored by only one of the two columns keeps
        # its null in the other rather than forcing the column back to float.
        anchored[col] = anchored[col].astype("Int64")
    return anchored, gl_da_enabled


class NHFPreprocessMixin:
    """Mixin providing preprocessing methods for the NHF class."""

    def crosswalk_nex_flowpath_poi(
        self,
        virtual_flowpaths,
        hydrolocations,
        waterbodies,
        gages,
        reference_flowpaths
    ):
        # Step N2: vectorized replacement for
        #   virtual_flowpaths.groupby("dn_virtual_nex_id")["virtual_fp_id"]
        #       .apply(list).to_dict()
        # which dominated NHF.__init__ at CONUS scale (~10 s per call,
        # 2 calls in this method -- the cProfile-measured ~20 s.)
        self._nexus_dict = _groupby_to_list_dict(
            virtual_flowpaths, "dn_virtual_nex_id", "virtual_fp_id"
        )  # {nex_id: [fp_id, ...]}
        if not hydrolocations.empty:
            if not waterbodies.empty:
                waterbody_ids = hydrolocations.merge(
                    waterbodies,
                    left_on='hy_id',
                    right_on='hy_id',
                    how='right'
                )
            else:
                waterbody_ids = pd.DataFrame(columns=["hy_id", "ref_fp_id"])
            if not hydrolocations.empty and not gages.empty:
                gage_ids = hydrolocations.merge(
                    gages,
                    left_on='hy_id',
                    right_on='hy_id',
                    how='right'
                )
            else:
                gage_ids = pd.DataFrame(columns=["hy_id", "ref_fp_id"])
            # Skip empty placeholders: concatenating them is deprecated and in
            # pandas 3 their object dtypes would win.
            _ref_id_parts = [
                df[["hy_id", "ref_fp_id"]]
                for df in (waterbody_ids, gage_ids)
                if not df.empty
            ]
            hy_id_to_ref_id = (
                pd.concat(_ref_id_parts, copy=True)
                if _ref_id_parts
                else pd.DataFrame(columns=["hy_id", "ref_fp_id"])
            )
            _ref_ids = reference_flowpaths.merge(
                hy_id_to_ref_id,
                left_on='ref_fp_id',
                right_on='ref_fp_id',
                how='right',
            )
            result = _ref_ids.merge(
                virtual_flowpaths,
                left_on='virtual_fp_id',
                right_on='virtual_fp_id',
                how='left',
            )
            # Step N2: same vectorization as the _nexus_dict above.
            self._poi_nex_dict = _groupby_to_list_dict(
                result, "hy_id", "dn_virtual_nex_id"
            )
        else:
            self._poi_nex_dict = None

    def preprocess_waterbodies(self, lakes, lake_vfp_crosswalk=None):
        if not lakes.empty:
            # Add lat, lon, and crs columns for LAKEOUT files:
            if self.output_parameters.get("lakeout_output", None):
                lakes = lakes.to_crs(crs=4326)
                lakes["crs"] = 4326  # Why you need to list crs when your coordinate keys are lat/lon eludes me...
                lakes["lon"] = lakes.geometry.x
                lakes["lat"] = lakes.geometry.y
                lake_cols = WATERBODY_DF_FIELDS + ["crs", "lat", "lon"]
            else:
                lake_cols = WATERBODY_DF_FIELDS

            # Step-by-step cleanup; every dropped category is counted and logged
            # as a warning (see _clean_waterbodies).
            self.waterbody_dataframe = lakes[lake_cols]
            self._waterbody_df, gl_df = _clean_waterbodies(
                self._waterbody_df, LAKE_ID_FIELD
            )

            # Add a large value to the lake_ids to create synthetic IDs and avoid conflicts.
            max_df_id = max(self.dataframe.index) + 1 if not self.dataframe.index.empty else 0
            self.waterbody_dataframe[RECORD_LAKE_ID_FIELD] = self.waterbody_dataframe.index
            self._waterbody_df.index = np.arange(len(self._waterbody_df)) + max_df_id
            self._waterbody_df = self._waterbody_df.rename_axis(LAKE_ID_FIELD)
            # Add conversion back to original nhf_lake_id for lakeout table
            self._duplicate_ids_df = self._waterbody_df.reset_index()[
                [RECORD_LAKE_ID_FIELD, LAKE_ID_FIELD]
            ].rename(
                columns={
                    RECORD_LAKE_ID_FIELD: "lake_id",
                    LAKE_ID_FIELD: "synthetic_ids",
                }
            )

            # Process great lakes reaches, if necessary
            gl_anchored, gl_da_enabled = _great_lakes_for_da(
                gl_df, self.data_assimilation_parameters
            )
            if gl_da_enabled and not gl_df.empty:
                n_unanchored = len(gl_df) - len(gl_anchored)
                if n_unanchored:
                    LOG.warning(
                        "waterbodies: %d Great Lake(s) have no virtual_fp_id and "
                        "cannot be anchored for reservoir_type 6 DA", n_unanchored,
                    )
                if not gl_anchored.empty:
                    # Set nhf_lake_id values to lake_id values,
                    # because those have been hard coded throughout this repo.
                    gl_anchored.index = gl_anchored[NATIVE_LAKE_ID_FIELD].astype(int)
                    gl_anchored.index.name = self.waterbody_dataframe.index.name
                    gl_anchored[RECORD_LAKE_ID_FIELD] = gl_anchored.index.astype(int)
                    collision_mask = self.waterbody_dataframe.index.isin(gl_anchored.index)
                    if collision_mask.any():
                        raise RuntimeError(f"Name collision: nhf_lake_id values of {GREAT_LAKES_IDS} are reserved, but received {self._waterbody_df.loc[collision_mask].index.values}")
                    self.waterbody_dataframe = pd.concat([self.waterbody_dataframe, gl_anchored])
                self.great_lakes_climatology_df = get_great_lakes_climatology()
            else:
                self.great_lakes_climatology_df = pd.DataFrame()

            # Condense flowpaths in a reservoir to single level pool node
            self._refactor_reservoirs(lake_vfp_crosswalk)

            self._waterbody_types_df = pd.DataFrame(
                data=1, index=self.waterbody_dataframe.index, columns=["reservoir_type"]
            ).sort_index()

            # Mark the Great Lakes as reservoir_type 6, matched by NATIVE lake id.
            # Must intersect on gl_anchored, not gl_df: gl_anchored was re-indexed
            # above from nhf_lake_id to the native lake_id (4800002 etc), and the
            # waterbody table now carries those native ids, so intersecting the
            # nhf_lake_id-indexed gl_df against it was always empty and the type-6
            # marking never happened.
            # When GL DA is enabled they were re-added above and survive here as
            # type 6 (so compute.py can link climatology/observations); when GL DA
            # is disabled gl_anchored is empty and so is the intersection (they stay
            # out of the reservoir set). A GL that was demoted in
            # _refactor_reservoirs (no single inlet -> outlet chain) is likewise
            # absent here and not marked type 6.
            gl_present = gl_anchored.index.intersection(self._waterbody_types_df.index)
            self._waterbody_types_df.loc[gl_present, "reservoir_type"] = 6

            self._waterbody_type_specified = True

        else:
            self.data_assimilation_parameters["reservoir_da"]["reservoir_persistence_da"][
                "reservoir_persistence_usgs"
            ] = False
            self.data_assimilation_parameters["reservoir_da"]["reservoir_persistence_da"][
                "reservoir_persistence_usace"
            ] = False
            self.data_assimilation_parameters["reservoir_da"]["reservoir_persistence_da"][
                "reservoir_persistence_usbr"
            ] = False
            self.data_assimilation_parameters["reservoir_da"]["reservoir_persistence_da"][
                "reservoir_persistence_canada"
            ] = False
            self.data_assimilation_parameters["reservoir_da"]["reservoir_rfc_da"]["reservoir_rfc_forecasts"] = False
            self.waterbody_parameters["break_network_at_waterbodies"] = False

            self._waterbody_df = pd.DataFrame()
            self._waterbody_types_df = pd.DataFrame()
            self._waterbody_connections = {}
            self._waterbody_type_specified = False
            self._link_lake_crosswalk = None
            self._duplicate_ids_df = pd.DataFrame()
            self.great_lakes_climatology_df = pd.DataFrame()


    
    def _refactor_reservoirs(self, lake_vfp_crosswalk=None):
        """Refactor network connectivity to explicitly represent reservoirs (waterbodies) and their interactions with flowpaths and links.

        Conceptual model:
            - Multiple flowpaths may exist within a single waterbody.
            - A single flowpath may intersect multiple waterbodies.

        Nothing inside a lake polygon is routed as MC channel: every flowpath
        draining to the lake's outlet is dropped from the link table and replaced
        by its level pool, and whatever discharged into them discharges into the
        pool instead.

        For each waterbody (``wb_group``):
            1. Identify the flowpaths intersecting it, from ``lake_vfp_crosswalk``.
            2. Identify all network links associated with these flowpaths (``all_links``).
            3. Remove all links in ``all_links`` from the network dataframe.
            4. Insert ordered connections for the waterbodies into the network
            connectivity structure.
            5. For each link whose downstream node lies within ``all_links``,
            redirect its connection to the most upstream waterbody.
            6. Create a synthetic headwater link (``qlat_link``) that drains into
            the appropriate waterbody link (``wb_link``).
            7. Redirect all lateral inflows (qlats) from ``all_links`` to ``qlat_link``.

        Parameters
        ----------
        lake_vfp_crosswalk : pandas.DataFrame, optional
            NHF ``lake_vfp_crosswalk``. Omitted or empty absorbs each lake's
            declared outlet flowpath only, leaving the rest of the lake as MC.
        """
        # Precompute every absorbed flowpath's links ONCE: one isin plus one
        # groupby. The original per-waterbody rescan of the full link table was
        # O(n_links x n_waterbodies), tens of minutes at CONUS.
        lake_cluster, vfp_cluster = _lake_vfp_clusters(
            self.waterbody_dataframe, lake_vfp_crosswalk
        )
        wb_links = self.dataframe[
            self.dataframe["vfp_id"].isin(set(vfp_cluster))
        ].reset_index()
        wb_links["cluster"] = wb_links["vfp_id"].map(vfp_cluster)
        links_by_cluster = {cid: sub for cid, sub in wb_links.groupby("cluster")}

        # One work item per level pool: (links to absorb, the lakes it replaces).
        # Cluster so no flowpath is claimed twice, then split back per lake so a
        # lake never moves its discharge point. Lakes sharing one declared outlet
        # stay a single chained group, as before the crosswalk was consumed.
        work: list[tuple[Optional[pd.DataFrame], pd.DataFrame]] = []
        for cid, wb_group in self.waterbody_dataframe.groupby(lake_cluster):
            cluster_links = links_by_cluster.get(cid)
            if cluster_links is None:
                work.append((None, wb_group))
                continue
            by_outlet = _links_by_nearest_outlet(
                cluster_links, wb_group["virtual_fp_id"]
            )
            for outlet_vfp, sub_group in wb_group.groupby("virtual_fp_id"):
                absorbed = by_outlet.get(int(outlet_vfp))
                if absorbed is None:
                    # No crosswalk resolution for this lake: fall back to its own
                    # declared flowpath, the behavior that shipped before.
                    absorbed = cluster_links[cluster_links["vfp_id"] == outlet_vfp]
                    absorbed = None if absorbed.empty else absorbed
                work.append((absorbed, sub_group))

        # Build the connections graph from the full link table up front so the
        # per-waterbody pops below are uniform; dataframe / zero_nodes removals are
        # accumulated and applied once after the loop.
        _ = self.connections

        # Sparse node remap (node_id -> waterbody headwater node) for links that
        # are absorbed into a waterbody; every other node maps to itself. Using a
        # dict instead of a dense np.arange(max_node_id + 1) lookup table avoids
        # allocating a max(node_id)-sized array, which is fatal for large/sparse
        # node ids (NHF >= 1.2.0). Behavior is identical on dense-id datasets.
        node_remap: dict[int, int] = {}
        df_rows = []
        index_vals = []
        skipped_wb: list[int] = []
        nodes_removed: list[int] = []
        downstream_groups = self.dataframe.groupby("downstream").groups
        for all_links, wb_group in work:
            # Until this is implemented in NHF, spoof here
            wb_group["lake_order"] = np.arange(len(wb_group))
            wb_group = wb_group.sort_values("lake_order")

            # None if discretization eliminated every flowpath of this cluster.
            if all_links is None:
                skipped_wb.extend(wb_group.index.astype(int).tolist())
                continue
            ds_set = set(all_links["downstream"]).difference(all_links["up_node_id"])
            us_set = set(all_links["up_node_id"]).difference(all_links["downstream"])
            if len(ds_set) != 1:
                # No single outlet even on the declared-outlet fallback (flowpath
                # merged away in discretization, or it spans a junction). Leave its
                # links in self.dataframe to route as MC, like bandaid() does for
                # problematic NHD/HYFeatures lakes.
                skipped_wb.extend(wb_group.index.astype(int).tolist())
                continue
            # us_set is NOT required to be a singleton: absorbing the whole lake
            # makes many inlets normal, one per tributary arm. Only the outlet must
            # be unique, since that is where the level pool discharges.
            ds = ds_set.pop()

            # This waterbody's up-node ids as native Python ints, computed once and
            # reused below (removal list, connections pops, node remap, crosswalk).
            up_nodes = all_links["up_node_id"].astype(int).tolist()

            # Remove references to those links. dataframe / zero_nodes removals are
            # accumulated for a single post-loop drop; only the connections dict is
            # updated per waterbody here (cheap O(group) dict pops).
            nodes_removed.extend(up_nodes)
            for i in up_nodes:
                self._connections.pop(i)

            # Modify connections to use waterbodies instead
            for i in wb_group.index.values:
                self.connections[i] = [ds]
                ds = i
            for us in us_set:
                for i in downstream_groups.get(us, []):
                    self.connections[i] = [ds]

            # Synthetic headwater reach, cut from the declared outlet flowpath so a
            # single-flowpath lake collapses exactly as before. Any absorbed link
            # will do otherwise: the row only supplies channel parameters for the
            # reach collecting the lake's qlat.
            head_vfp = wb_group["virtual_fp_id"].iloc[0]
            head_rows = all_links[all_links["vfp_id"] == head_vfp]
            # .copy(): "downstream" is rewritten below, and writing into an .iloc[0]
            # view of a sliced frame raises SettingWithCopyWarning. Nothing reads
            # all_links afterwards, so a detached row is what was meant.
            headwater = (all_links if head_rows.empty else head_rows).iloc[0].copy()
            head_id = int(headwater["up_node_id"])
            self.connections[head_id] = [ds]
            headwater["downstream"] = ds
            # Ensure MC kernel won't crash on these reaches
            # TODO: Figure out a way to avoid routing on headwaters altogether.
            row = headwater.drop(labels="up_node_id").fillna(9999).to_dict()
            df_rows.append(row)
            index_vals.append(head_id)

            # TODO: consider putting these within single condensed for loop with above.
            # Reroute all div flows to headwater (every up_node maps to the same
            # head_id, so dict.fromkeys over the int list beats a per-element loop).
            node_remap.update(dict.fromkeys(up_nodes, head_id))

            # Remap outflow from waterbody links onto the waterbody's outlet
            # crosswalk. pop() collapses the in/getitem/del triple lookup into one,
            # and the target id is constant per waterbody so hoist it out. (A
            # waterbody id is always > max(dataframe.index) >= every up_node_id, so
            # the target is never itself one of the popped links.)
            wb_outlet = wb_group.index[0]
            for i in up_nodes:
                moved = self._fp_outlet_crosswalk.pop(i, None)
                if moved is not None:
                    self._fp_outlet_crosswalk[wb_outlet].extend(moved)

        # Apply the accumulated waterbody-link removals to the dataframe and
        # zero_nodes in one shot (instead of one drop per waterbody).
        if nodes_removed:
            self.dataframe = self.dataframe.drop(nodes_removed)
            self.zero_nodes = list(set(self.zero_nodes).difference(nodes_removed))

        # Demote un-routable waterbodies to plain MC channels: drop them from the
        # waterbody set so routing won't model them as reservoirs (their links are
        # already left intact in self.dataframe).
        if skipped_wb:
            LOG.warning(
                "waterbodies: demoted %d lakes to MC channels (their links do "
                "not form a single inlet -> outlet chain: the flowpath was "
                "eliminated in discretization or spans a junction)",
                len(skipped_wb),
            )
            LOG.debug("demoted waterbody ids: %s", sorted(skipped_wb))
            self.waterbody_dataframe = self.waterbody_dataframe.drop(skipped_wb)

        # Apply the sparse remap; unmapped nodes keep their own id (identity).
        if node_remap:
            vfp_nodes = pd.Series(self.vfp_nex_ids)
            self.vfp_nex_ids = (
                vfp_nodes.map(node_remap).fillna(vfp_nodes)
                .to_numpy().astype(self.vfp_nex_ids.dtype)
            )
            # Rebuild the connections graph after the per-waterbody rewiring:
            #  - drop stale keys: downstream_groups is precomputed from the original
            #    dataframe, so rewiring waterbody B can re-add another waterbody A's
            #    already-removed link as a key (connections[i] = [ds_B]); such keys
            #    have no routing data behind them and crash binary_find. A's outflow
            #    still reaches downstream through A's own waterbody chain.
            #  - redirect any edge still pointing at a removed waterbody link to its
            #    replacement headwater (node_remap), so a waterbody draining into
            #    another waterbody's now-removed inlet does not dangle, and
            #  - drop edges that point at a network terminal: the rewiring sets
            #    wb -> outlet directly, but extract_connections represents a
            #    terminal-bound segment as having no downstream ([]); a terminal
            #    left in as a value crashes subnetwork construction.
            terminals = set(self._terminal_codes) if self._terminal_codes else set()
            stale_keys = set(node_remap) - set(index_vals)
            self._connections = {
                k: [r for r in (node_remap.get(x, x) for x in v) if r not in terminals]
                for k, v in self.connections.items()
                if k not in stale_keys
            }
        self._link_lake_crosswalk = None  # Handled by _fp_outlet_crosswalk
        # Identity map of waterbody ids. tolist() once + dict(zip(...)) builds the
        # dict over native Python ints (no per-element numpy-scalar boxing), ~1.8x
        # faster than a comprehension iterating the Index.
        wb_index = self.waterbody_dataframe.index.tolist()
        self.waterbody_connections = dict(zip(wb_index, wb_index))

        # Guard the empty case: with no synthetic reservoir rows the frame has no
        # columns, and astype against the link-table dtypes raises KeyError.
        if df_rows:
            row_df = pd.DataFrame(df_rows, index=index_vals)
            row_df.index.name = self.dataframe.index.name
            row_df = row_df.astype(self.dataframe.dtypes.to_dict())
            self.dataframe = pd.concat([self.dataframe, row_df])

    def preprocess_data_assimilation(self, gages: pd.DataFrame, reservoir_da: pd.DataFrame):
        # Reservoir DA needs lakes; streamflow and diversion DA do not. Returning
        # early here for a lake-free domain used to leave self.gages empty, which
        # silently disabled gage nudging on every network without waterbodies (the
        # Ohio benchmark subset among them) even though its gages layer is fully
        # populated. Skip only the reservoir crosswalks and carry on.
        if reservoir_da.empty or self.waterbody_dataframe.empty:
            self.usgs_lake_gage_crosswalk = pd.DataFrame()
            self.usace_lake_gage_crosswalk = pd.DataFrame()
            self.usbr_lake_gage_crosswalk = pd.DataFrame()
            self.rfc_lake_gage_crosswalk = pd.DataFrame()
            if not self.waterbody_dataframe.empty:
                LOG.warning(
                    "reservoir DA: no reservoir_da records for %d waterbody(s); "
                    "reservoir assimilation is disabled for this run.",
                    len(self.waterbody_dataframe),
                )
            self._preprocess_streamflow_and_diversion_da(gages)
            return

        ### reservoir_da validation and formatting ###
        reservoir_da = reservoir_da.copy()
        if RECORD_LAKE_ID_FIELD not in self.waterbody_dataframe.columns:
            raise KeyError(f"Column {RECORD_LAKE_ID_FIELD} must be in waterbody_dataframe, but only got {self.waterbody_dataframe.columns.to_list()}.")
        if  self.waterbody_dataframe.index.name != LAKE_ID_FIELD:
            raise KeyError(f"Column {LAKE_ID_FIELD} must be index of waterbody_dataframe, but found index '{self.waterbody_dataframe.index.name}'.")
        if LAKE_ID_FIELD not in reservoir_da.columns:
            raise KeyError(f"Column {LAKE_ID_FIELD} must be in reservoir_da, but only got {reservoir_da.columns.to_list()}.")
        reservoir_da[LAKE_ID_FIELD] = reservoir_da[LAKE_ID_FIELD].astype(int)

        # Process great lakes
        gl_str = [str(i) for i in GREAT_LAKES_IDS]
        gl_present = reservoir_da[NATIVE_LAKE_ID_FIELD].astype(str).isin(gl_str)
        if gl_present.any():
            reservoir_da.loc[gl_present, LAKE_ID_FIELD] = reservoir_da.loc[gl_present, NATIVE_LAKE_ID_FIELD].astype(int)

        # In NHF, the reservoir_da table is one-to-one with lakes table.
        if not reservoir_da[LAKE_ID_FIELD].is_unique:
            raise ValueError(
                f"NHF networks must have only one gage per value in {LAKE_ID_FIELD}"
            )
        # Check that all lakes are in reservoir_da table
        id_diff = set(
            self.waterbody_dataframe[RECORD_LAKE_ID_FIELD].to_numpy()
        ).difference(reservoir_da[LAKE_ID_FIELD].to_numpy())
        if len(id_diff) > 0:
            raise ValueError(
                f"Missing {RECORD_LAKE_ID_FIELD} values {id_diff} in reservoir_da table"
            )
        reservoir_da = reservoir_da[
            reservoir_da[LAKE_ID_FIELD].isin(
                self.waterbody_dataframe[RECORD_LAKE_ID_FIELD].to_numpy()
            )
        ]

        # Format reservoir_da table
        reservoir_da = reservoir_da[
            [LAKE_ID_FIELD, NATIVE_LAKE_ID_FIELD, RESERVOIR_DA_SITE_ID_FIELD, RESERVOIR_DA_SITE_TYPE_FIELD]
        ]
        reservoir_da = reservoir_da.set_index(LAKE_ID_FIELD, drop=True)
        reservoir_da = reservoir_da.rename(
            columns={RESERVOIR_DA_SITE_TYPE_FIELD: "reservoir_type"}
        )
        # map new waterbody ids to reservoir da table
        record_to_id_lookup = (
            self.waterbody_dataframe.reset_index()
            .set_index(RECORD_LAKE_ID_FIELD)[LAKE_ID_FIELD]
            .to_dict()
        )
        reservoir_da.index = reservoir_da.index.map(record_to_id_lookup)

        # Join types.  These will be overwritten later based on config.
        self.waterbody_dataframe = self.waterbody_dataframe.merge(
            reservoir_da["reservoir_type"], right_index=True, left_index=True
        )

        # USGS DA
        usgs_da = (
            self.data_assimilation_parameters.get("reservoir_da", {})
            .get("reservoir_persistence_da", {})
            .get("reservoir_persistence_usgs", False)
        )
        type_2_mask = (reservoir_da["reservoir_type"] == 2)
        great_lake_mask = reservoir_da[NATIVE_LAKE_ID_FIELD].isin(["4800002", "4800004"])
        usgs_mask = type_2_mask | great_lake_mask
        usgs_indices = reservoir_da[usgs_mask].index.values
        # Also add some Great Lakes gages, if present
        self.usgs_lake_gage_crosswalk = (
            reservoir_da.loc[usgs_indices, RESERVOIR_DA_SITE_ID_FIELD]
            .reset_index()
            .copy()
        )
        self.usgs_lake_gage_crosswalk = self.usgs_lake_gage_crosswalk.rename(
            columns={
                LAKE_ID_FIELD: "usgs_lake_id",
                RESERVOIR_DA_SITE_ID_FIELD: "usgs_gage_id",
            }
        )
        if not usgs_da:
            # Never demote a Great Lake to level pool. The Great Lakes carry no
            # level-pool parameters, so demoting them crashes the kernel on the
            # missing values (see _great_lakes_for_da). They are gated by
            # reservoir_persistence_greatLake, not by reservoir_persistence_usgs,
            # so a config with USGS persistence off and Great Lake DA on must
            # leave them at type 6.
            demote_indices = reservoir_da[type_2_mask & ~great_lake_mask].index.values
            self.waterbody_dataframe.loc[demote_indices, "reservoir_type"] = 1

        # USACE DA
        usace_da = (
            self.data_assimilation_parameters.get("reservoir_da", {})
            .get("reservoir_persistence_da", {})
            .get("reservoir_persistence_usace", False)
        )
        usace_indices = reservoir_da[reservoir_da["reservoir_type"] == 3].index.values
        self.usace_lake_gage_crosswalk = (
            reservoir_da.loc[usace_indices, RESERVOIR_DA_SITE_ID_FIELD]
            .reset_index()
            .copy()
        )
        self.usace_lake_gage_crosswalk = self.usace_lake_gage_crosswalk.rename(
            columns={
                LAKE_ID_FIELD: "usace_lake_id",
                RESERVOIR_DA_SITE_ID_FIELD: "usace_gage_id",
            }
        )
        if not usace_da:
            self.waterbody_dataframe.loc[usace_indices, "reservoir_type"] = 1

        # RFC DA
        rfc_da = (
            self.data_assimilation_parameters.get("reservoir_da", {})
            .get("reservoir_rfc_da", {})
            .get("reservoir_rfc_forecasts", False)
        )
        rfc_indices = reservoir_da[reservoir_da["reservoir_type"] == 4].index.values
        self.rfc_lake_gage_crosswalk = (
            reservoir_da.loc[rfc_indices, RESERVOIR_DA_SITE_ID_FIELD]
            .reset_index()
            .copy()
        )
        self.rfc_lake_gage_crosswalk = self.rfc_lake_gage_crosswalk.rename(
            columns={
                LAKE_ID_FIELD: "rfc_lake_id",
                RESERVOIR_DA_SITE_ID_FIELD: "rfc_gage_id",
            }
        )
        if not rfc_da:
            self.waterbody_dataframe.loc[rfc_indices, "reservoir_type"] = 1

        # USBR DA
        usbr_da = (
            self.data_assimilation_parameters.get("reservoir_da", {})
            .get("reservoir_persistence_da", {})
            .get("reservoir_persistence_usbr", False)
        )
        usbr_indices = reservoir_da[reservoir_da["reservoir_type"] == 7].index.values
        self.usbr_lake_gage_crosswalk = (
            reservoir_da.loc[usbr_indices, RESERVOIR_DA_SITE_ID_FIELD]
            .reset_index()
            .copy()
        )
        self.usbr_lake_gage_crosswalk = self.usbr_lake_gage_crosswalk.rename(
            columns={
                LAKE_ID_FIELD: "usbr_lake_id",
                RESERVOIR_DA_SITE_ID_FIELD: "usbr_gage_id",
            }
        )
        if not usbr_da:
            self.waterbody_dataframe.loc[usbr_indices, "reservoir_type"] = 1

        self.waterbody_types_dataframe = self.waterbody_dataframe[
            ["reservoir_type"]
        ].copy()

        self._preprocess_streamflow_and_diversion_da(gages)

    def _gage_selection_rank(self, sub: pd.DataFrame) -> pd.DataFrame:
        """Rank co-located gages so the survivor is chosen deterministically.

        Several gages can sit on one virtual flowpath (4096 of them across 1729
        flowpaths on the CONUS hydrofabric), and only one observation can be
        assimilated per routing link. Nothing in the gages layer separates them
        positionally: their ``segment_order`` and ``dn_virtual_nex_id`` are identical
        within every colliding group, so they cannot be placed on distinct sub-links.

        The survivor is therefore chosen by, in order:

        1. an active gage over a discontinued one, since assimilation needs current
           observations;
        2. the gage physically closest to the flowpath's downstream nexus, which is
           the one that best represents flow at the outlet link the gage is placed on;
        3. the lowest site number, purely so the result cannot depend on the order
           rows happen to appear in the geopackage.

        On CONUS this leaves 85 groups undecided after (1) and (2), and none after
        (3). Distance is computed only for colliding gages, so the geometry read
        covers about 1700 nexus points rather than the full 1.65 M.
        """
        sub = sub.copy()
        sub["_inactive"] = (sub["status"] != "USGS-active").astype("int8")
        sub["_dist_m"] = np.inf

        colliding = sub[sub.duplicated("virtual_fp_id", keep=False)]
        nex_ids = sorted(
            {int(x) for x in colliding["dn_virtual_nex_id"].dropna().unique()}
        )
        if nex_ids:
            try:
                geo = self.supernetwork_parameters["geo_file_path"]
                pts = self._read_nexus_points(geo, nex_ids)
                if pts is not None and not pts.empty:
                    gg = gpd.read_file(
                        geo, layer="gages",
                        where=_sql_in("site_no", colliding["site_no"], quote=True),
                    )
                    sub = self._attach_nexus_distance(sub, gg, pts)
            except Exception as exc:  # geometry is an optimisation, never a hard dep
                LOG.warning(
                    "gages: could not rank co-located gages by distance to their "
                    "downstream nexus (%s); falling back to status and site number.",
                    exc,
                )
        return sub.sort_values(["_inactive", "_dist_m", "site_no"])

    @staticmethod
    def _read_nexus_points(geo_file_path, nex_ids: list[int]):
        """Geometry for just the virtual nexus points named, in OGR-sized chunks."""
        frames = []
        for i in range(0, len(nex_ids), 400):
            chunk = nex_ids[i : i + 400]
            frames.append(
                gpd.read_file(
                    geo_file_path, layer="virtual_nexus",
                    where=_sql_in("virtual_nex_id", chunk),
                    columns=["virtual_nex_id"],
                )
            )
        if not frames:
            return None
        return pd.concat(frames).drop_duplicates("virtual_nex_id").set_index("virtual_nex_id")

    @staticmethod
    def _attach_nexus_distance(sub, gage_points, nexus_points):
        """Set ``_dist_m`` to each gage's distance from its downstream nexus."""
        pts = gage_points[["site_no", "geometry"]].dropna(subset=["geometry"])
        merged = sub.merge(pts, on="site_no", how="left", suffixes=("", "_pt"))
        merged = merged.merge(
            nexus_points[["geometry"]].rename(columns={"geometry": "_nex_geom"}),
            left_on="dn_virtual_nex_id", right_index=True, how="left",
        )
        ok = merged["geometry"].notna() & merged["_nex_geom"].notna()
        if not ok.any():
            return sub
        # Equal-area projection so the distance is in metres, not degrees.
        crs = gage_points.crs
        a = gpd.GeoSeries(merged.loc[ok, "geometry"].values, crs=crs).to_crs(5070)
        b = gpd.GeoSeries(merged.loc[ok, "_nex_geom"].values, crs=crs).to_crs(5070)
        merged.loc[ok, "_dist_m"] = a.distance(b, align=False).to_numpy()
        return merged.drop(columns=["_nex_geom"], errors="ignore")

    @staticmethod
    def _one_link_per_gage(sub: pd.DataFrame, label: str, quiet: bool = False) -> pd.DataFrame:
        """Reduce a gage-to-link join to exactly one routing link per gage.

        ``vfp_id`` is not unique in the link table: flowpaths longer than the
        discretization length are split into several routing links that all inherit
        the parent ``vfp_id``. Joining on it alone is therefore one-to-many, and
        indexing the result on ``up_node_id`` registered every gage at every link of
        its flowpath (a median of 8 links per gage on the CONUS hydrofabric). That
        assimilated a single observation at many consecutive segments and split the
        cached execution plan at all of them.

        The gage's own ``segment_order`` is not usable for placement: it is a
        different quantity from the link table's ``segment_order`` and matches only
        about a fifth of gages. The outlet link (highest ``segment_order`` within the
        flowpath) is the placement used elsewhere in this module for diversion gages.

        ``quiet`` suppresses the summary logs for callers that re-resolve the same
        join for a different purpose, so the counts are reported once rather than
        once per caller with a label that misdescribes what was counted.
        """
        if sub.empty:
            return sub
        placed = sub.dropna(subset=["up_node_id"])
        n_unplaced = sub["site_no"].nunique() - placed["site_no"].nunique()
        if n_unplaced and not quiet:
            LOG.warning(
                "gages: %d %s gage(s) have no routing link for their virtual flowpath "
                "and are excluded from streamflow DA", n_unplaced, label,
            )
        if placed.empty:
            return placed
        outlet = placed.loc[placed.groupby("site_no")["link_segment_order"].idxmax()].copy()
        outlet["up_node_id"] = outlet["up_node_id"].astype("int64")
        n_collisions = len(outlet) - outlet["up_node_id"].nunique()
        if n_collisions:
            # Restore the ranking. The groupby above sorts by site_no, which discards
            # the order _gage_selection_rank established, so without this the dedupe
            # below would keep the lowest site number rather than the best gage.
            rank_cols = [c for c in ("_inactive", "_dist_m", "site_no") if c in outlet]
            outlet = outlet.sort_values(rank_cols)
            # Only one observation can be assimilated per link, so co-located gages
            # must be reduced to one. `outlet` arrives ordered by
            # _gage_selection_rank (active, then closest to the downstream nexus,
            # then site number), so keeping the first is a deterministic choice
            # rather than whichever row the geopackage happened to list last.
            dropped = outlet[outlet.duplicated("up_node_id", keep="first")]
            outlet = outlet.drop_duplicates("up_node_id", keep="first")
            if not quiet:
                LOG.warning(
                    "gages: %d %s gage(s) share a routing link with another gage; "
                    "keeping the active gage nearest the downstream nexus and "
                    "dropping the rest, e.g. %s",
                    n_collisions, label, sorted(dropped["site_no"].astype(str))[:5],
                )
        # Worth an INFO line of its own: this count is how many links the cached
        # execution plan splits reaches at, so a jump here is a routing slowdown
        # with no other visible cause.
        if not quiet:
            LOG.info(
                "gages: %d %s gage(s) placed on %d routing link(s)",
                len(outlet), label, outlet["up_node_id"].nunique(),
            )
        return outlet

    def build_gage_vpu_map(self, gages: pd.DataFrame, flowpaths: pd.DataFrame) -> None:
        """Record ``site_no -> vpu_id`` for the scaling DA's per-tree theta.

        Kept as a small dict rather than joined onto the routing table (~1.1M
        redundant strings at CONUS). Empty on hydrofabrics without vpu_id;
        callers fall back to the default theta.
        """
        self.gage_vpu: dict[str, str] = {}
        if gages.empty or flowpaths.empty:
            return
        if "vpu_id" not in flowpaths.columns or "fp_id" not in gages.columns:
            LOG.debug(
                "gage->vpu map: hydrofabric has no vpu_id on flowpaths; every gage "
                "tree will use the default theta."
            )
            return
        fp_to_vpu = (
            flowpaths[["fp_id", "vpu_id"]].dropna().drop_duplicates("fp_id")
            .set_index("fp_id")["vpu_id"]
        )
        joined = gages[["site_no", "fp_id"]].dropna()
        joined = joined.assign(vpu_id=joined["fp_id"].map(fp_to_vpu)).dropna(subset=["vpu_id"])
        self.gage_vpu = {
            str(s).strip(): str(v).strip()
            for s, v in zip(joined["site_no"], joined["vpu_id"], strict=True)
        }
        n_unmapped = gages["site_no"].nunique() - len(self.gage_vpu)
        if n_unmapped > 0:
            LOG.debug(
                "gage->vpu map: %d of %d gage(s) have no vpu_id and will use the "
                "default theta", n_unmapped, gages["site_no"].nunique(),
            )

    def _preprocess_streamflow_and_diversion_da(self, gages: pd.DataFrame) -> None:
        """Resolve the gage crosswalk and the diversion map.

        Independent of reservoir DA: a network with no waterbodies still has gages.
        """
        # Streamflow DA
        if gages.empty:
            self.gages = {}
            # An empty DataFrame, not an empty dict: consumers call .empty on this.
            self.canadian_gage_df = pd.DataFrame(columns=["gages"])
            gages_join = pd.DataFrame()
        else:
            # The gages layer carries its own segment_order, which is a different
            # quantity from the link table's, so rename to avoid a _x/_y collision.
            link_cols = self.dataframe.reset_index()[
                ["vfp_id", "up_node_id", "segment_order"]
            ].rename(columns={"segment_order": "link_segment_order"})
            gages_join = gages.merge(
                link_cols,
                left_on="virtual_fp_id",
                right_on="vfp_id",
                how="left",
            )
            usgs_sub = self._one_link_per_gage(
                self._gage_selection_rank(
                    gages_join[
                        gages_join["status"].isin(["USGS-active", "USGS-discontinued"])
                    ]
                ),
                "USGS",
            )
            canada_sub = self._one_link_per_gage(
                self._gage_selection_rank(
                    gages_join[gages_join["status"] == "CADWR_ENVCA"]
                ),
                "Canadian",
            )
            self.gages = (
                usgs_sub.set_index("up_node_id")[["site_no"]]
                .rename(columns={"site_no": "gages"})
                .rename_axis(None, axis=0)
                .to_dict()
            )
            self.canadian_gage_df = (
                canada_sub.set_index("up_node_id")[["site_no"]]
                .rename(columns={"site_no": "gages"})
            )

        self._resolve_diversion_da(gages_join)

    def _resolve_diversion_da(self, gages_join: pd.DataFrame) -> None:
        """Transform fp_id to self.dataframe link ID and gage site_no to link ID of gage link."""
        self._diversion_site_to_node: dict[str, int] = {}
        self.diversion_da: dict[int, int] = {}
        _diversion_cfg = self.data_assimilation_parameters.get("diversion_da", None)
        if not _diversion_cfg:
            return
        _crosswalk = _diversion_cfg.get("diversion_gage_crosswalk", {})
        if not _crosswalk:
            return

        # Map gage site_no to network links
        # Must use the SAME placement rule as the streamflow crosswalk. The gage's
        # observations are looked up in usgs_df by this link, and usgs_df places the
        # gage at its flowpath OUTLET. Taking the first joined sub-link here instead
        # meant the two disagreed on any multi-link flowpath, so the diversion could
        # not find its observations and silently applied nothing.
        self._diversion_site_to_node = (
            self._one_link_per_gage(self._gage_selection_rank(gages_join), "diversion", quiet=True)
            .set_index("site_no")["up_node_id"]
            .astype("int64")
            .to_dict()
        )

        # Find most downstream routing link for each diversion flowpath
        fp_ids_needed = set(_crosswalk.keys())
        fp_outlet_nodes = (
            self._dataframe[self._dataframe["fp_id"].isin(fp_ids_needed)]
            .groupby("fp_id")["segment_order"]
            .idxmax()
            .astype(int)
            .to_dict()
        )

        # Create updated diversion_da dict
        for fp_id, site_no in _crosswalk.items():
            from_id = fp_outlet_nodes.get(fp_id)
            if from_id is None:
                raise ValueError(
                    f"diversion_gage_crosswalk: fp_id {fp_id} not found in network."
                )
            gage_node = self._diversion_site_to_node.get(site_no)
            if gage_node is None:
                raise ValueError(
                    f"diversion_gage_crosswalk: site_no '{site_no}' not found in gages."
                )
            self.diversion_da[from_id] = gage_node
            LOG.debug(
                "Diversion configured: fp_id %s (node %s) -> gage %s (node %s)",
                fp_id, from_id, site_no, gage_node,
            )
