from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import xarray as xr
from joblib import Parallel, delayed

# Only read relevant areas of NHF to cut down on processing time and memory footprint.
LAYERS_TO_READ = [
    {
        "name": "flowpaths",
        "columns": [
            "fp_id",
            "length_km",
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
        ],
        "ignore_geometry": True
    },
    {
        "name": "reference_flowpaths",
        "columns": None,  # Loads all
        "ignore_geometry": True
    },
    {
        "name": "virtual_flowpaths",
        "columns": [
            "length_km", 
            "virtual_fp_id", 
            "dn_virtual_nex_id", 
            "up_virtual_nex_id",
            "percentage_area_contribution"
            ],
        "ignore_geometry": False
    },
    {
        "name": "virtual_nexus",
        "columns": None,
        "ignore_geometry": True
    },
    {
        "name": "waterbodies",
        "columns": None,
        "ignore_geometry": True
    },
    {
        "name": "gages",
        "columns": None,
        "ignore_geometry": True
    },
    {
        "name": "hydrolocations",
        "columns": None,
        "ignore_geometry": True
    }
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


def read_geo_file(supernetwork_parameters, waterbody_parameters, compute_parameters, cpu_pool):
    geo_file_path = supernetwork_parameters["geo_file_path"]
    file_type = Path(geo_file_path).suffix
    if file_type == ".gpkg":

        # TODO enable lakes to be read into the routing solution here
        # if waterbody_parameters.get("break_network_at_waterbodies", False):
        #     layers_to_read.extend(["lakes", "nexus"])

        # data_assimilation_parameters = compute_parameters.get("data_assimilation_parameters", {})
        # if any(
        #     [
        #         data_assimilation_parameters.get("streamflow_da", {}).get("streamflow_nudging", False),
        #         data_assimilation_parameters.get("reservoir_da", {}).get("reservoir_persistence_da", False).get("reservoir_persistence_usgs", False),
        #         data_assimilation_parameters.get("reservoir_da", {}).get("reservoir_persistence_da", False).get("reservoir_persistence_usace", False),
        #         data_assimilation_parameters.get("reservoir_da", {}).get("reservoir_persistence_da", False).get("reservoir_persistence_usbr", False),
        #         data_assimilation_parameters.get("reservoir_da", {}).get("reservoir_rfc_da", {}).get("reservoir_rfc_forecasts", False),
        #     ]
        # ):
        #     layers_to_read.append("network")

        # Layers whose geometry we need to preserve for discretization

        def read_layer(lyr):
            try:
                _df = gpd.read_file(geo_file_path, layer=lyr["name"], columns=lyr["columns"], ignore_geometry=lyr["ignore_geometry"])
                return _df
            except pyogrio.errors.DataSourceError as e:
                print(f"Error reading file {geo_file_path}: {e}")
                raise pyogrio.errors.DataSourceError from e
            except pyogrio.errors.DataLayerError:
                return pd.DataFrame()  # Missing layer -> empty DF

        # Retrieve geopackage information using matched layer names
        if cpu_pool > 1:
            with Parallel(n_jobs=min(cpu_pool, len(LAYERS_TO_READ))) as parallel:
                gpkg_list = parallel(delayed(read_layer)(layer) for layer in LAYERS_TO_READ)

            table_dict = {LAYERS_TO_READ[i]["name"]: gpkg_list[i] for i in range(len(LAYERS_TO_READ))}
        else:
            table_dict = {layer["name"]: read_layer(layer) for layer in LAYERS_TO_READ}

    else:
        raise RuntimeError("Unsupported file type: {}".format(file_type))

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
        self._nexus_dict = virtual_flowpaths.groupby("dn_virtual_nex_id")["virtual_fp_id"].apply(list).to_dict()  ##{id: toid}
        if hydrolocations.empty or gages.empty:
            self._poi_nex_dict = None
        else:
            waterbody_ids = hydrolocations.merge(
                waterbodies,
                left_on='hy_id',
                right_on='hy_id',
                how='right'
            )
            gage_ids = hydrolocations.merge(
                gages,
                left_on='hy_id',
                right_on='hy_id',
                how='right'
            )
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
            self._poi_nex_dict = result.groupby("hy_id")["dn_virtual_nex_id"].apply(list).to_dict()

    def preprocess_waterbodies(self, lakes, nexus):
        # TODO work on waterbodies support for NHF
        # If waterbodies are being simulated, create waterbody dataframes and dictionaries
        # if not lakes.empty:
        #     self._waterbody_df = lakes[
        #         [
        #             "lake_id",
        #             "id",
        #             "ifd",
        #             "LkArea",
        #             "LkMxE",
        #             "OrificeA",
        #             "OrificeC",
        #             "OrificeE",
        #             "WeirC",
        #             "WeirE",
        #             "WeirL",
        #         ]
        #     ]

        #     id = self.waterbody_dataframe["id"].str.split("-", expand=True).iloc[:, 1]
        #     self._waterbody_df.loc[:, "id"] = id
        #     self._waterbody_df.loc[:, "id"] = self._waterbody_df.id.astype(float).astype(int)
        #     self._waterbody_df.loc[:, "lake_id"] = self.waterbody_dataframe.lake_id.astype(float).astype(int)
        #     self._waterbody_df = self.waterbody_dataframe.set_index("lake_id").drop_duplicates().sort_index()

        #     # Drop any waterbodies that do not have parameters
        #     self._waterbody_df = self.waterbody_dataframe.dropna()

        #     # Check if there are any lake_ids that are also segment_ids. If so, add a large value
        #     # to the lake_ids:
        #     duplicate_ids = list(set(self.waterbody_dataframe.index).intersection(set(self.dataframe.index)))
        #     self._duplicate_ids_df = pd.DataFrame(
        #         {"lake_id": duplicate_ids, "synthetic_ids": [int(id + 9.99e11) for id in duplicate_ids]}
        #     )
        #     update_dict = dict(self._duplicate_ids_df[["lake_id", "synthetic_ids"]].values)

        #     tmp_wbody_conn = self.dataframe[["waterbody"]].dropna()
        #     tmp_wbody_conn = (
        #         tmp_wbody_conn["waterbody"]
        #         .str.split(",", expand=True)
        #         .reset_index()
        #         .melt(id_vars="key")
        #         .drop("variable", axis=1)
        #         .dropna()
        #         .astype(int)
        #     )
        #     tmp_wbody_conn = tmp_wbody_conn[tmp_wbody_conn["value"].isin(self.waterbody_dataframe.index)]
        #     self._dataframe = (
        #         self.dataframe.reset_index()
        #         .merge(tmp_wbody_conn, how="left", on="key")
        #         .drop("waterbody", axis=1)
        #         .rename(columns={"value": "waterbody"})
        #         .set_index("key")
        #     )

        #     self._waterbody_df = self.waterbody_dataframe.rename(index=update_dict).sort_index()
        #     self._dataframe = self.dataframe.replace({"waterbody": update_dict})

        #     # FIXME temp solution for missing waterbody info in hydrofabric
        #     self.bandaid()

        #     wbody_conn = self.dataframe[["waterbody"]].dropna().astype(int).reset_index()

        #     self._waterbody_connections = (
        #         wbody_conn[wbody_conn["waterbody"].isin(self.waterbody_dataframe.index)]
        #         .set_index("key")["waterbody"]
        #         .to_dict()
        #     )

        #     # if waterbodies are being simulated, adjust the connections graph so that
        #     # waterbodies are collapsed to single nodes. Also, build a mapping between
        #     # waterbody outlet segments and lake ids
        #     break_network_at_waterbodies = self.waterbody_parameters.get("break_network_at_waterbodies", False)
        #     if break_network_at_waterbodies:
        #         self._connections, self._link_lake_crosswalk = replace_waterbodies_connections(
        #             self.connections, self.waterbody_connections
        #         )
        #     else:
        #         self._link_lake_crosswalk = None

        #     # Add lat, lon, and crs columns for LAKEOUT files:
        #     lakeout = self.output_parameters.get("lakeout_output", None)
        #     if lakeout:
        #         lat_lon_crs = lakes[["hl_link", "hl_reference", "geometry"]].rename(columns={"hl_link": "lake_id"})
        #         lat_lon_crs = lat_lon_crs[lat_lon_crs["hl_reference"] == "WBOut"]
        #         lat_lon_crs["lake_id"] = lat_lon_crs.lake_id.astype(float).astype(int)
        #         lat_lon_crs = lat_lon_crs.set_index("lake_id").drop_duplicates().sort_index()
        #         lat_lon_crs = lat_lon_crs[lat_lon_crs.index.isin(self.waterbody_dataframe.index)]
        #         lat_lon_crs = lat_lon_crs.to_crs(crs=4326)
        #         lat_lon_crs["lon"] = lat_lon_crs.geometry.x
        #         lat_lon_crs["lat"] = lat_lon_crs.geometry.y
        #         lat_lon_crs["crs"] = str(lat_lon_crs.crs)
        #         lat_lon_crs = lat_lon_crs[["lon", "lat", "crs"]]

        #         self._waterbody_df = self.waterbody_dataframe.join(lat_lon_crs)
        #     else:
        #         self._waterbody_df["lon"] = np.nan
        #         self._waterbody_df["lat"] = np.nan
        #         self._waterbody_df["crs"] = np.nan

        #     # Add the Great Lakes to the connections dictionary and waterbody dataframe
        #     nexus["WBOut_id"] = nexus["hl_uri"].str.extract(r"WBOut-(\d+)").astype(float)
        #     great_lakes_df = nexus[nexus["WBOut_id"].isin([4800002, 4800004, 4800006, 4800007])][["WBOut_id", "toid"]]
        #     if not great_lakes_df.empty:
        #         great_lakes_df["toid"] = great_lakes_df["toid"].str.extract(r"wb-(\d+)").astype(float)
        #         great_lakes_df = great_lakes_df.astype(int)
        #         great_lakes_df["toid"] = great_lakes_df["toid"].apply(lambda x: [x])
        #         gl_dict = great_lakes_df.set_index("WBOut_id")["toid"].to_dict()
        #         self._connections.update(gl_dict)

        #         gl_wbody_df = pd.DataFrame(
        #             data=np.ones([len(gl_dict), self.waterbody_dataframe.shape[1]]),
        #             index=gl_dict.keys(),
        #             columns=self.waterbody_dataframe.columns,
        #         )
        #         gl_wbody_df.index.name = self.waterbody_dataframe.index.name

        #         self._waterbody_df = pd.concat([self.waterbody_dataframe, gl_wbody_df]).sort_index()

        #         self._gl_climatology_df = get_great_lakes_climatology()

        #     else:
        #         gl_dict = {}
        #         self._gl_climatology_df = pd.DataFrame()

        #     self._waterbody_types_df = pd.DataFrame(
        #         data=1, index=self.waterbody_dataframe.index, columns=["reservoir_type"]
        #     ).sort_index()

        #     # Add Great Lakes waterbody type (6)
        #     self._waterbody_types_df.loc[gl_dict.keys(), "reservoir_type"] = 6

        #     self._waterbody_type_specified = True

        # else:
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
        self._usgs_lake_gage_crosswalk = pd.DataFrame()
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
