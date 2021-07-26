"""This module includes a utility function to read next generation water modeling framework catchment lateral flows."""

import xarray as xr
import pandas as pd
import geopandas as gpd
import os


def read_catchment_lateral_flows(path):
    """Read and convert catchment lateral flows to format that can be processed by compute_network
    Args:
       path: Path to lateral flows.

    Returns:
       qlats: dataframe of lateral flows
    """

    ql = []

    catchment_id_list = []

    for file_name in os.listdir(path):
        if file_name.startswith("cat-"):
            file_name_str_list = file_name.split("_")
            catchment_id = int(file_name_str_list[0][4:])
            catchment_id_list.append(catchment_id)
            # Read the second column of a csv file and return a series. The index will be an autoincrementing range.
            catchment_qlats = pd.read_csv(
                os.path.join(path, file_name), usecols=["Flow"], squeeze=True
            )
            catchment_qlats.rename(catchment_id, inplace=True)
            ql.append(catchment_qlats)

    qlats = pd.concat(ql, axis="columns").T

    qlats.index = qlats.index.astype(int)
    qlats.columns = qlats.columns.astype(int)
    qlats = qlats.sort_index(axis="index")
    return qlats.astype("float32")
