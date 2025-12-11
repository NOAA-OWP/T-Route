"""A file to hold all replace and route (RnR) geospatial scripts"""

import logging

import geopandas as gpd
import pandas as pd
import polars as pl

from troute_rnr.logging import log_function_debug

log = logging.getLogger(__name__)


@log_function_debug()
def to_geopandas(df: pd.DataFrame, crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """Converts the geometries in a pandas df to a geopandas dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The iceberg table you are trying to read from
    crs: str, optional
        A string representing the CRS to set in the gdf, by default "EPSG:5070"

    Returns
    -------
    gpd.DataFrame
        The resulting queried row, but in a geodataframe

    Raises
    ------
    ValueError
        Raised if the table does not have a geometry column
    """
    if "geometry" not in df.columns:
        raise ValueError("The provided table does not have a geometry column.")

    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df["geometry"]), crs=crs)


@log_function_debug()
def get_rnr_segment(
    layers: dict[str, pl.LazyFrame], reach_id: str
) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """Returns a geopackage subset from the hydrofabric based on RnR rules

    Parameters
    ----------
    catalog : Catalog
        The iceberg catalog of the hydrofabric
    reach_id : str
        The reach_id, or hf_id, from the NWPS API

    Returns
    -------
    dict[str, pd.DataFrame | gpd.GeoDataFrame]
        a dictionary of dataframes and geodataframes containing HF layers
    """
    network = layers["network"]
    flowpaths = layers["flowpaths"]
    lakes = layers["lakes"]
    hydrolocations = layers["hydrolocations"]
    divides = layers["divides"]
    nexus = layers["nexus"]
    flowpath_attr = layers["flowpath_attr"]
    pois = layers["pois"]

    origin_row = network.filter(pl.col("hf_id") == reach_id).collect()

    mainstem_features = network.filter(
        (pl.col("hf_mainstem") == origin_row["hf_mainstem"].first())
        & (pl.col("hydroseq") <= origin_row["hydroseq"].first())
    ).collect()
    segment_flowpaths = flowpaths.filter(
        pl.col("divide_id").is_in(mainstem_features["divide_id"].unique().implode())
    ).collect()
    joined_df = mainstem_features.join(segment_flowpaths, on="divide_id", how="full")
    stream_order = joined_df.filter(pl.col("hf_id") == int(reach_id))["order"].first()
    filtered_flowpaths = segment_flowpaths.filter(pl.col("order") == stream_order)

    # Find any lakes contained in the RnR segment
    poi_ids = filtered_flowpaths["poi_id"].filter(filtered_flowpaths["poi_id"].is_not_null()).cast(pl.Int64)
    filtered_lakes = lakes.filter(pl.col("poi_id").is_in(poi_ids.implode())).collect()

    if filtered_lakes.shape[0] > 0:
        # Ensuring we break connectivity at lakes
        lake_ids = filtered_lakes["hf_id"].filter(filtered_lakes["hf_id"].is_not_null())
        network_rows = mainstem_features.filter(pl.col("hf_id").is_in(lake_ids.implode()))
        upstream_lake = network_rows[
            "hf_hydroseq"
        ].max()  # since hydroseq decreases as you go downstream, we want the upstream most value
        mainstem_features = mainstem_features.filter(pl.col("hf_hydroseq").ge(upstream_lake))
        segment_flowpaths = flowpaths.filter(
            pl.col("divide_id").is_in(mainstem_features["divide_id"].unique().implode())
        ).collect()
        joined_df = mainstem_features.join(segment_flowpaths, on="divide_id", how="full")
        stream_order = joined_df.filter(pl.col("hf_id") == int(reach_id))["order"].first()
        filtered_flowpaths = segment_flowpaths.filter(pl.col("order") == stream_order)

        # Find any lakes contained in the RnR segment
        poi_ids = (
            filtered_flowpaths["poi_id"].filter(filtered_flowpaths["poi_id"].is_not_null()).cast(pl.Int64)
        )
        filtered_lakes = lakes.filter(pl.col("poi_id").is_in(poi_ids.implode())).collect()

    # Convert output to geopandas
    filtered_nexus_points = to_geopandas(
        nexus.filter(pl.col("id").is_in(filtered_flowpaths["toid"])).collect().to_pandas()
    )
    filtered_divides = to_geopandas(
        divides.filter(pl.col("divide_id").is_in(filtered_flowpaths["divide_id"])).collect().to_pandas()
    )
    filtered_flowpath_attr = (
        flowpath_attr.filter(pl.col("id").is_in(filtered_flowpaths["id"])).collect().to_pandas()
    )
    filtered_pois = pois.filter(pl.col("poi_id").is_in(poi_ids)).collect().to_pandas()
    filtered_hydrolocations = hydrolocations.filter(pl.col("poi_id").is_in(poi_ids)).collect().to_pandas()
    filtered_network = (
        network.filter(pl.col("id").is_in(pl.concat([filtered_flowpaths["toid"], filtered_flowpaths["id"]])))
        .collect()
        .to_pandas()
    )
    filtered_flowpaths = to_geopandas(filtered_flowpaths.to_pandas())

    layers = {
        "flowpaths": filtered_flowpaths,
        "nexus": filtered_nexus_points,
        "divides": filtered_divides,
        "network": filtered_network,
        "pois": filtered_pois,
        "flowpath-attributes": filtered_flowpath_attr,
        "hydrolocations": filtered_hydrolocations,
    }
    return layers


@log_function_debug()
def find_origin(
    network_table: pl.LazyFrame,
    identifier: str | float,
) -> pl.DataFrame:
    """Find an origin point in the hydrofabric network.

    This function handles the case where multiple records match the identifier.
    It follows the R implementation to select a single origin point based on
    the minimum hydroseq value.

    Parameters
    ----------
    network_table : LazyFrame
        The HF network table from the hydrofabric catalog
    identifier : str | float
        The unique identifier you want to find the origin of
    id_type : IdType, optional
        The network table column you can query from, by default "hl_uri"
    return_all: bool, False
        Returns all origin points (for subsetting)

    Returns
    -------
    pd.DataFrame
        The origin row from the network table

    Raises
    ------
    ValueError
        The provided identifier is not supported
    ValueError
        No origin for the point is found
    ValueError
        Multiple origins for the point are found
    """
    # Get all matching records
    origin_candidates = (
        network_table.filter(pl.col("hf_id").is_not_null() & (pl.col("hf_id") == identifier))
        .select(["id", "toid", "vpuid", "hydroseq", "poi_id", "hl_uri"])
        .collect()
    )

    if origin_candidates.height == 0:
        raise ValueError(f"No origin found for hf_id={identifier}")

    origin = origin_candidates["id"].first()

    return origin
