import logging
from pathlib import Path
from itertools import starmap
from typing import Optional

import geopandas as gpd
import numpy as np
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
LAKE_ID_FIELD = "lake_id"

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

def _missing_requested_columns(
    available_by_layer: dict[str, set],
) -> dict[str, list]:
    """Pure check: which requested columns are absent from each layer.

    ``available_by_layer`` maps a layer name to the set of column names the
    geopackage actually carries for it; omit a layer to signal it is entirely
    absent. Only layers that declare an explicit ``columns`` list are checked
    (``columns=None`` layers are loaded in full and used conditionally).
    Returns ``{layer_name: [missing columns]}`` (empty if nothing is missing).
    """
    missing_by_layer: dict[str, list] = {}
    for name, columns, _ in LAYERS_TO_READ:
        if columns is None:
            continue
        available = available_by_layer.get(name)
        if available is None:
            missing_by_layer[name] = list(columns)
            continue
        absent = [c for c in columns if c not in available]
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
    ``columns=None`` (lakes, gages, hydrolocations, virtual_nexus) are used
    conditionally and not validated.
    """
    available_by_layer: dict[str, set[str]] = {}
    for name, columns, _ in LAYERS_TO_READ:
        if columns is None or name not in present_layers:
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
            f"network build -> {details}. This usually means the hydrofabric "
            "predates the current schema (for example, older datasets lack the "
            "'segment_order' column in 'reference_flowpaths'); regenerate or "
            "update the dataset to a compatible hydrofabric version."
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
         "ncc", "btmwdth", "musx", "chslp", "topwdthcc", "musk"],
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
        [LAKE_ID_FIELD, "fp_id", "virtual_fp_id", "ifd", "LkArea", "LkMxE", "OrificeA",
         "OrificeC", "OrificeE", "WeirC", "WeirE", "WeirL", "hy_id", "ref_fp_id"], 
        True),
    ("gages", None, True),
    ("hydrolocations", None, True),
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

        df = df.drop(["id", "toid", "hl_id", "hl_reference", "hl_uri", "geometry"], axis=1).rename(
            columns={"hl_link": "lake_id"}
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

    # Read present layers in parallel; layers absent from the geopackage become
    # empty DataFrames (they are used conditionally downstream). to_read keeps
    # the full (name, columns, ignore_geometry) tuples so starmap can unpack
    # them into read_layer's arguments.
    to_read = [layer for layer in LAYERS_TO_READ if layer[0] in gpkg_layers]
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
      5. fp_id anchoring: drop lakes with no fp_id -- they cannot be placed
         on a routing flowpath (the hydrofabric fix for these is tracked
         upstream).
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
    waterbody_df[lake_id_field] = lake_ids.astype(int)

    # 2. index + dedup
    waterbody_df = waterbody_df.set_index(lake_id_field)
    n_before = len(waterbody_df)
    waterbody_df = waterbody_df.drop_duplicates().sort_index()
    n_dup = n_before - len(waterbody_df)
    if n_dup:
        LOG.warning("waterbodies: dropped %d duplicated parameter rows", n_dup)

    # 3. Great Lakes
    gl_df = waterbody_df[waterbody_df.index.isin(GREAT_LAKES_IDS)].copy()
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
        waterbody_df = waterbody_df[~waterbody_df.index.isin(GREAT_LAKES_IDS)]

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
    waterbody_df["virtual_fp_id"] = waterbody_df["virtual_fp_id"].astype(int)
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
    as MC channels). Only Great Lakes with an ``fp_id`` can be anchored to a
    flowpath; their ``fp_id`` is cast to int to match the link table.

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
    anchored = gl_df[gl_df["fp_id"].notna()].copy()
    if not anchored.empty:
        anchored["fp_id"] = anchored["fp_id"].astype(int)
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
            hy_id_to_ref_id = pd.concat([waterbody_ids[["hy_id", "ref_fp_id"]].copy(), gage_ids[["hy_id", "ref_fp_id"]]])
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

    def preprocess_waterbodies(self, lakes):
        if not lakes.empty:
            # Step-by-step cleanup; every dropped category is counted and logged
            # as a warning (see _clean_waterbodies).
            self.waterbody_dataframe = lakes[[
                LAKE_ID_FIELD,
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
            ]]
            self._waterbody_df, gl_df = _clean_waterbodies(
                self._waterbody_df, LAKE_ID_FIELD
            )

            # Add a large value to the lake_ids to create synthetic IDs and avoid conflicts.
            max_df_id = max(self.dataframe.index) + 1 if not self.dataframe.index.empty else 0
            self._waterbody_df.index = np.arange(len(self._waterbody_df)) + max_df_id
            self._waterbody_df = self._waterbody_df.rename_axis(LAKE_ID_FIELD)
            self._duplicate_ids_df = pd.DataFrame()  # Relic from how hyfeatures and NHD handled this. We add relationship to _fp_outlet_crosswalk 

            # Process great lakes reaches, if necessary
            gl_anchored, gl_da_enabled = _great_lakes_for_da(
                gl_df, self.data_assimilation_parameters
            )
            if gl_da_enabled and not gl_df.empty:
                n_unanchored = len(gl_df) - len(gl_anchored)
                if n_unanchored:
                    LOG.warning(
                        "waterbodies: %d Great Lake(s) have no fp_id and cannot "
                        "be anchored for reservoir_type 6 DA", n_unanchored,
                    )
                if not gl_anchored.empty:
                    self._waterbody_df = pd.concat([self._waterbody_df, gl_anchored])
                self.great_lakes_climatology_df = get_great_lakes_climatology()
            else:
                self.great_lakes_climatology_df = pd.DataFrame()

            # Condense flowpaths in a reservoir to single level pool node
            self._refactor_reservoirs()

            # Add lat, lon, and crs columns for LAKEOUT files:
            lakeout = self.output_parameters.get("lakeout_output", None)
            if lakeout:
                raise NotImplementedError("The lakeout feature has not been developed for NHF.")

            

            self._waterbody_types_df = pd.DataFrame(
                data=1, index=self.waterbody_dataframe.index, columns=["reservoir_type"]
            ).sort_index()

            # Mark the Great Lakes as reservoir_type 6, matched by ORIGINAL id.
            # When GL DA is enabled they were re-added above and survive here as
            # type 6 (so compute.py can link climatology/observations); when GL DA
            # is disabled they are absent and the intersection is empty (they stay
            # out of the reservoir set). A GL that was demoted in
            # _refactor_reservoirs (no single inlet -> outlet chain) is likewise
            # absent here and not marked type 6.
            gl_present = gl_df.index.intersection(self._waterbody_types_df.index)
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


    
    def _refactor_reservoirs(self):
        """Refactor network connectivity to explicitly represent reservoirs (waterbodies) and their interactions with flowpaths and links.

        Conceptual model:
            - Multiple flowpaths may exist within a single waterbody.
            - A single flowpath may intersect multiple waterbodies.

        For each flowpath containing at least one waterbody (``outlet_fp``):
            1. Identify all flowpaths contained within any waterbody intersecting
            the outlet flowpath (``all_fp``).
            2. Identify all network links associated with these flowpaths (``all_links``).
            3. Remove all links in ``all_links`` from the network dataframe.
            4. For each waterbody associated with ``outlet_fp``, insert ordered
            connections into the network connectivity structure.
            5. For each link whose downstream node lies within ``all_links``,
            redirect its connection to the most upstream waterbody.
            6. Create a synthetic headwater link (``qlat_link``) that drains into
            the appropriate waterbody link (``wb_link``).
            7. Redirect all lateral inflows (qlats) from ``all_fp`` to ``qlat_link``.

        Parameters
        ----------
        lake_id_field : str, optional
            Column name used to uniquely identify waterbodies in the input
            dataframe. Default is ``"lake_id"``.

        """
        # Precompute the routing links of every waterbody flowpath ONCE. The
        # original code re-scanned the full (multi-million-row) link table with
        # `self.dataframe["fp_id"].isin(...)` and dropped from it once per
        # waterbody -- O(n_links x n_waterbodies), tens of minutes at CONUS. A
        # waterbody's links are exactly the links whose virtual_fp_id == its 
        # virtual_fp_id, so one isin (restricted to waterbody virtual_fp_ids) 
        # plus one groupby gives an O(1) lookup.
        wb_vfp_ids = set(self.waterbody_dataframe["virtual_fp_id"].dropna())
        wb_links = self.dataframe[self.dataframe["vfp_id"].isin(wb_vfp_ids)].reset_index()
        links_by_vfp = {vfp: sub for vfp, sub in wb_links.groupby("vfp_id")}

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
        for outlet_vfp, wb_group in self.waterbody_dataframe.groupby("virtual_fp_id"):
            # Until this is implemented in NHF, spoof here
            wb_group["lake_order"] = np.arange(len(wb_group))
            wb_group = wb_group.sort_values("lake_order")

            # Routing links of this waterbody's flowpath (None if its flowpath was
            # eliminated in discretization -> nothing to model as a reservoir).
            all_links = links_by_vfp.get(outlet_vfp)
            if all_links is None:
                skipped_wb.extend(wb_group.index.astype(int).tolist())
                continue
            ds_set = set(all_links["downstream"]).difference(all_links["up_node_id"])
            us_set = set(all_links["up_node_id"]).difference(all_links["downstream"])
            if len(ds_set) != 1 or len(us_set) != 1:
                # This waterbody does not collapse to a single inlet -> outlet
                # chain (its flowpath was eliminated/merged in discretization, or
                # it spans a junction). Skip it: its links are left untouched in
                # self.dataframe and route as plain MC channels. Mirrors the
                # bandaid() fallback used for problematic NHD/HYFeatures lakes.
                skipped_wb.extend(wb_group.index.astype(int).tolist())
                continue
            ds, us = ds_set.pop(), us_set.pop()

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
            for i in downstream_groups.get(us, []):
                self.connections[i] = [ds]

            # Add synthetic headwater reach
            headwater = all_links[all_links["vfp_id"] == outlet_vfp].iloc[0]
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

        row_df = pd.DataFrame(df_rows, index=index_vals)
        row_df.index.name = self.dataframe.index.name
        row_df = row_df.astype(self.dataframe.dtypes.to_dict())
        self.dataframe = pd.concat([self.dataframe, row_df])


    def preprocess_data_assimilation(
        self,
        flowpaths,
        reference_flowpaths,
        virtual_flowpaths,
        virtual_nexus,
        waterbodies,
        gages
    ):
        # TODO enable DA methods
        # gages_df = network[["id", "hl_uri", "hydroseq"]].drop_duplicates()
        # # clear out missing values
        # gages_df = gages_df[~gages_df["hl_uri"].isnull()]
        # gages_df = gages_df[~gages_df["hydroseq"].isnull()]
        # # make 'id' an integer
        # gages_df["id"] = gages_df["id"].str.split("-", expand=True).loc[:, 1].astype(float).astype(int)
        # # split the hl_uri column into type and value
        # gages_df[["type", "value"]] = gages_df.hl_uri.str.split("-", expand=True, n=1)
        # # filter for 'Gages' only
        # gages_df = gages_df[gages_df["type"].isin(["gages", "nid", "usbr"])]
        # # Some IDs have multiple gages associated with them. This will expand the dataframe so
        # # there is a unique row per gage ID. Also adds lake ids to the dataframe for creating
        # # lake-gage crosswalk dataframes.
        # gages_df = gages_df[["id", "value", "hydroseq", "type"]]
        # gages_df["value"] = gages_df.value.str.split(" ")
        # gages_df = (
        #     gages_df.explode(column="value")
        #     .set_index("id")
        #     .join(pd.DataFrame().from_dict(self.waterbody_connections, orient="index", columns=["lake_id"]))
        # )
        # # transform dataframe into a dictionary where key is segment ID and value is gage ID
        # usgs_ind = gages_df.value.str.isnumeric()  # usgs gages used for streamflow DA
        # # Use hydroseq information to determine furthest downstream gage when multiple are present.
        # idx_id = gages_df.index.name
        # if not idx_id:
        #     idx_id = "index"
        # self._gages = (
        #     gages_df.loc[usgs_ind]
        #     .reset_index()
        #     .sort_values("hydroseq")
        #     .drop_duplicates(["value"], keep="last")
        #     .set_index(idx_id)[["value"]]
        #     .rename(columns={"value": "gages"})
        #     .rename_axis(None, axis=0)
        #     .to_dict()
        # )

        # # FIXME: temporary solution, add canadian gage crosswalk dataframe. This should come from
        # # the hydrofabric.
        # self._canadian_gage_link_df = pd.DataFrame(columns=["gages", "link"]).set_index("link")

        # # Find furthest downstream gage and create our lake_gage_df to make crosswalk dataframes.
        # lake_gage_hydroseq_df = gages_df[~gages_df["lake_id"].isnull()][["lake_id", "value", "hydroseq", "type"]].rename(
        #     columns={"value": "gages"}
        # )
        # lake_gage_hydroseq_df["lake_id"] = lake_gage_hydroseq_df["lake_id"].astype(int)
        # lake_gage_df = lake_gage_hydroseq_df[["lake_id", "gages", "type"]].drop_duplicates()
        # lake_gage_hydroseq_df = (
        #     lake_gage_hydroseq_df.groupby(["lake_id", "gages", "type"]).max("hydroseq").reset_index().set_index("lake_id")
        # )

        # # FIXME: temporary solution, handles USGS and USACE reservoirs. Need to update for
        # # RFC reservoirs...
        # # NOTE: In the event a lake ID has multiple gages, this also finds the gage furthest
        # # downstream (based on hydroseq) separately for USGS and USACE crosswalks.
        # usgs_ind = lake_gage_df.gages.str.isnumeric()
        # self._usgs_lake_gage_crosswalk = (
        #     lake_gage_df.loc[usgs_ind]
        #     .drop("type", axis=1)  # dropping type to ensure no dups when merging
        #     .rename(columns={"lake_id": "usgs_lake_id", "gages": "usgs_gage_id"})
        #     .set_index("usgs_lake_id")
        #     .merge(
        #         lake_gage_hydroseq_df.rename_axis("usgs_lake_id").rename(columns={"gages": "usgs_gage_id"}),
        #         on=["usgs_lake_id", "usgs_gage_id"],
        #     )
        #     .sort_values(["usgs_gage_id", "hydroseq"])
        #     .groupby("usgs_lake_id")
        #     .last()
        #     .drop("hydroseq", axis=1)
        # )

        # self._usace_lake_gage_crosswalk = (
        #     lake_gage_df.loc[~usgs_ind]
        #     .drop("type", axis=1)  # dropping type to ensure no dups when merging
        #     .rename(columns={"lake_id": "usace_lake_id", "gages": "usace_gage_id"})
        #     .set_index("usace_lake_id")
        #     .merge(
        #         lake_gage_hydroseq_df.rename_axis("usace_lake_id").rename(columns={"gages": "usace_gage_id"}),
        #         on=["usace_lake_id", "usace_gage_id"],
        #     )
        #     .sort_values(["usace_gage_id", "hydroseq"])
        #     .groupby("usace_lake_id")
        #     .last()
        #     .drop("hydroseq", axis=1)
        # )

        # # Using the USBR type to set the crosswalk
        # self._usbr_lake_gage_crosswalk = (
        #     lake_gage_df[lake_gage_df["type"] == "usbr"]
        #     .drop("type", axis=1)  # dropping type to ensure no dups when merging
        #     .rename(columns={"lake_id": "usbr_lake_id", "gages": "usbr_gage_id"})
        #     .set_index("usbr_lake_id")
        #     .merge(
        #         lake_gage_hydroseq_df.rename_axis("usbr_lake_id").rename(columns={"gages": "usbr_gage_id"}),
        #         on=["usbr_lake_id", "usbr_gage_id"],
        #     )
        #     .sort_values(["usbr_gage_id", "hydroseq"])
        #     .groupby("usbr_lake_id")
        #     .last()
        #     .drop("hydroseq", axis=1)
        # )

        # # Set waterbody types if DA is turned on:
        # usgs_da = (
        #     self.data_assimilation_parameters.get("reservoir_da", {})
        #     .get("reservoir_persistence_da", {})
        #     .get("reservoir_persistence_usgs", False)
        # )
        # usace_da = (
        #     self.data_assimilation_parameters.get("reservoir_da", {})
        #     .get("reservoir_persistence_da", {})
        #     .get("reservoir_persistence_usace", False)
        # )
        # usbr_da = (
        #     self.data_assimilation_parameters.get("reservoir_da", {})
        #     .get("reservoir_persistence_da", {})
        #     .get("reservoir_persistence_usbr", False)
        # )
        # rfc_da = (
        #     self.data_assimilation_parameters.get("reservoir_da", {})
        #     .get("reservoir_rfc_da", {})
        #     .get("reservoir_rfc_forecasts", False)
        # )
        # # NOTE: The order here matters. Some waterbody IDs have both a USGS gage designation and
        # # a NID ID used for USACE gages. It seems the USGS gages should take precedent (based on
        # # gages in timeslice files), so setting type 2 reservoirs second should overwrite type 3
        # # designations
        # # FIXME: Related to FIXME above, but we should re-think how to handle waterbody_types...
        # if usbr_da:
        #     self._waterbody_types_df.loc[self._usace_lake_gage_crosswalk.index, "reservoir_type"] = 7
        # if usace_da:
        #     self._waterbody_types_df.loc[self._usace_lake_gage_crosswalk.index, "reservoir_type"] = 3
        # if usgs_da:
        #     self._waterbody_types_df.loc[self._usgs_lake_gage_crosswalk.index, "reservoir_type"] = 2
        # if rfc_da:
        #     # FIXME: Temporary fix, load predefined rfc lake gage crosswalk info for rfc reservoirs.
        #     # Replace relevant waterbody_types as type 4.
        #     rfc_lake_gage_crosswalk = get_rfc_lake_gage_crosswalk().reset_index()
        #     self._rfc_lake_gage_crosswalk = rfc_lake_gage_crosswalk[
        #         rfc_lake_gage_crosswalk["rfc_lake_id"].isin(self.waterbody_dataframe.index)
        #     ].set_index("rfc_lake_id")
        #     self._waterbody_types_df.loc[self._rfc_lake_gage_crosswalk.index, "reservoir_type"] = 4
        # else:
        #     self._rfc_lake_gage_crosswalk = pd.DataFrame()

        self._gages = {}
        self._usgs_lake_gage_crosswalk = pd.DataFrame({"usgs_lake_id": [4800002, 4800004],"usgs_gage_id": ["04127885", "04159130"]})
        self._usace_lake_gage_crosswalk = pd.DataFrame()
        self._usbr_lake_gage_crosswalk = pd.DataFrame()
        self._rfc_lake_gage_crosswalk = pd.DataFrame()

    ######################################################################
    # FIXME Temporary solution to hydrofabric issues.
    def bandaid(
        self,
    ):
        # Identify waterbody IDs that have problematic data. There are underlying stream
        # segments that should be referenced to the waterbody ID, but are not. This causes
        # our connections dictionary to have multiple downstream segments for waterbodies which
        # is not allowed:
        conn_df = self.dataframe.reset_index()[["key", "downstream"]]
        lake_id = self.waterbody_dataframe.index.unique()

        wbody_conn_df = self.dataframe["waterbody"].dropna().astype(int).reset_index()
        wbody_conn_df = wbody_conn_df[wbody_conn_df["waterbody"].isin(lake_id)]

        conn_df2 = (
            conn_df.merge(wbody_conn_df, on="key", how="left")
            .assign(key=lambda x: x["waterbody"].fillna(x["key"]))
            .drop("waterbody", axis=1)
            .merge(wbody_conn_df.rename(columns={"key": "downstream"}), on="downstream", how="left")
            .assign(downstream=lambda x: x["waterbody"].fillna(x["downstream"]))
            .drop("waterbody", axis=1)
            .drop_duplicates()
            .query("key != downstream")
            .astype(int)
        )

        # Find missing segments
        bad_lake_ids = conn_df2.loc[conn_df2.duplicated(subset=["key"])].key.unique()
        # Drop waterbodies that are problematic. Instead t-route will simply treat them as
        # flowpaths and run MC routing.
        self._waterbody_df = self.waterbody_dataframe.drop(bad_lake_ids)

        # This chunk replaces waterbody_id 1711354 with 1710676. I don't know where the
        # former came from, but the latter is listed in the flowpath_attributes table
        # and exists in NWMv2.1 LAKEPARM file. See hydrofabric github issue 16:
        # https://github.com/NOAA-OWP/hydrofabric/issues/16
        self._dataframe["waterbody"] = self._dataframe["waterbody"].replace("1711354", "1710676")
        self._waterbody_df.rename(index={1711354: 1710676}, inplace=True)

    #######################################################################
