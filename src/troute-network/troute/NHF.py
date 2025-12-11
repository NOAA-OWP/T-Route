import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from pprint import pformat
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyarrow.parquet as pq
import xarray as xr
from joblib import Parallel, delayed
from troute.nhd_network import reverse_network

from .AbstractNetwork import AbstractNetwork

__verbose__ = False
__showtiming__ = False

def build_downstream_connections(
    routing_flowpaths: pd.DataFrame,
    all_flowpaths: pd.DataFrame,
    terminal_nexus_ids: Optional[set[int]] = None,
) -> dict[int, list[int]]:
    """
    Build downstream connectivity dictionary mapping flowpath IDs to their downstream flowpath IDs.
    
    The NHF data model uses nexus points as connection nodes between flowpaths:
    - Flowpath A has dn_virtual_nex_id = X (A drains TO nexus X)
    - Flowpath B has up_virtual_nex_id = X (B receives FROM nexus X)
    - Therefore A -> B (A flows into B)
    
    Parameters
    ----------
    routing_flowpaths : pd.DataFrame
        DataFrame containing ONLY routing segment flowpaths (routing_segment == True).
        These are the flowpaths that will appear as keys in the connections dict.
        Must have columns: virtual_fp_id, dn_virtual_nex_id, up_virtual_nex_id
    all_flowpaths : pd.DataFrame
        DataFrame containing ALL virtual flowpaths (both routing and non-routing).
        Used to build the nexus-to-downstream-flowpath mapping.
        Must have columns: virtual_fp_id, dn_virtual_nex_id, up_virtual_nex_id
    terminal_nexus_ids : set[int], optional
        Set of nexus IDs that are terminal (no downstream flowpath).
        If provided, flowpaths draining to these nexuses will have empty downstream lists.
        
    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping each routing virtual_fp_id to a list of downstream virtual_fp_ids.
        Terminal flowpaths will have empty lists [].
        
    Examples
    --------
    >>> all_vfp = pd.DataFrame({
    ...     'virtual_fp_id': [1, 2, 3],
    ...     'dn_virtual_nex_id': [100, 101, 102],
    ...     'up_virtual_nex_id': [None, 100, 101],
    ...     'routing_segment': [True, True, True]
    ... })
    >>> routing_vfp = all_vfp[all_vfp['routing_segment']]
    >>> connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids={102})
    >>> connections
    {1: [2], 2: [3], 3: []}
    """
    if terminal_nexus_ids is None:
        terminal_nexus_ids = set()
    
    # Get the set of routing flowpath IDs for filtering downstream connections
    routing_fp_ids = set(routing_flowpaths['virtual_fp_id'].tolist())
    
    # Map: nexus_id -> list of ROUTING flowpaths that have this nexus as their UPSTREAM nexus
    # We use all_flowpaths to find the mapping, but filter to only routing segments
    routing_vfp = all_flowpaths[all_flowpaths['virtual_fp_id'].isin(routing_fp_ids)]
    nex_to_downstream_fps = (
        routing_vfp[routing_vfp['up_virtual_nex_id'].notna()]
        .groupby('up_virtual_nex_id')['virtual_fp_id']
        .apply(list)
        .to_dict()
    )
    
    # Build connections: for each routing flowpath, find downstream routing flowpath(s)
    connections = {}
    for _, row in routing_flowpaths.iterrows():
        fp_id = row['virtual_fp_id']
        dn_nex = row['dn_virtual_nex_id']
        
        if pd.isna(dn_nex):
            # No downstream nexus
            connections[fp_id] = []
        elif dn_nex in terminal_nexus_ids:
            # Downstream nexus is terminal (no further flowpaths)
            connections[fp_id] = []
        else:
            # Find routing flowpaths whose up_virtual_nex_id == this flowpath's dn_virtual_nex_id
            downstream_fps = nex_to_downstream_fps.get(dn_nex, [])
            connections[fp_id] = downstream_fps
    
    return connections


def build_upstream_terminal(
    virtual_flowpaths: pd.DataFrame,
    terminal_nexus_ids: set[int],
) -> dict[int, set[int]]:
    """
    Build mapping of terminal nexus IDs to their upstream flowpath IDs.
    
    This identifies which flowpaths drain into terminal nexuses (network outlets).
    
    Parameters
    ----------
    virtual_flowpaths : pd.DataFrame
        DataFrame containing virtual flowpath information with columns:
        - virtual_fp_id: unique flowpath identifier
        - dn_virtual_nex_id: downstream nexus ID for each flowpath
    terminal_nexus_ids : set[int]
        Set of nexus IDs that are terminal (no downstream flowpath).
        
    Returns
    -------
    dict[int, set[int]]
        Dictionary mapping each terminal nexus ID to a set of upstream flowpath IDs.
        
    Examples
    --------
    >>> vfp = pd.DataFrame({
    ...     'virtual_fp_id': [1, 2, 3],
    ...     'dn_virtual_nex_id': [100, 101, 101],  # fp2 and fp3 both drain to terminal nex 101
    ... })
    >>> upstream_terminal = build_upstream_terminal(vfp, terminal_nexus_ids={101})
    >>> upstream_terminal
    {101: {2, 3}}
    """
    upstream_terminal = {}
    
    for _, row in virtual_flowpaths.iterrows():
        dn_nex = row['dn_virtual_nex_id']
        if pd.notna(dn_nex) and dn_nex in terminal_nexus_ids:
            fp_id = row['virtual_fp_id']
            upstream_terminal.setdefault(dn_nex, set()).add(fp_id)
    
    return upstream_terminal


def get_terminal_nexus_ids(virtual_nexus: pd.DataFrame) -> set[int]:
    """
    Extract terminal nexus IDs from the virtual nexus DataFrame.
    
    Terminal nexuses are those with no downstream flowpath (dn_virtual_fp_id is NaN).
    
    Parameters
    ----------
    virtual_nexus : pd.DataFrame
        DataFrame containing virtual nexus information with columns:
        - virtual_nex_id: unique nexus identifier
        - dn_virtual_fp_id: downstream flowpath ID (NaN for terminals)
        
    Returns
    -------
    set[int]
        Set of terminal nexus IDs.
    """
    terminals = virtual_nexus[pd.isna(virtual_nexus["dn_virtual_fp_id"])]
    return set(terminals["virtual_nex_id"].tolist())


def validate_connections(connections: dict[int, list[int]]) -> tuple[bool, set[int]]:
    """
    Validate that all downstream flowpath IDs in the connections dictionary
    exist as keys (i.e., they are valid flowpath IDs).
    
    This catches the case where nexus IDs are accidentally used instead of
    flowpath IDs in the connections values.
    
    Parameters
    ----------
    connections : dict[int, list[int]]
        Dictionary mapping flowpath IDs to lists of downstream flowpath IDs.
        
    Returns
    -------
    tuple[bool, set[int]]
        - bool: True if all downstream IDs are valid, False otherwise
        - set[int]: Set of orphaned IDs (downstream IDs not found as keys)
        
    Examples
    --------
    >>> connections = {1: [2], 2: [3], 3: []}
    >>> is_valid, orphaned = validate_connections(connections)
    >>> is_valid
    True
    >>> orphaned
    set()
    
    >>> bad_connections = {1: [2], 2: [999], 3: []}  # 999 doesn't exist as key
    >>> is_valid, orphaned = validate_connections(bad_connections)
    >>> is_valid
    False
    >>> orphaned
    {999}
    """
    all_downstream_fps = set(chain.from_iterable(connections.values()))
    all_fp_ids = set(connections.keys())
    
    orphaned = all_downstream_fps - all_fp_ids
    
    return len(orphaned) == 0, orphaned


def find_headwaters(connections: dict[int, list[int]]) -> set[int]:
    """
    Find headwater flowpaths in the network.
    
    Headwaters are flowpaths that are never referenced as downstream of any other
    flowpath (i.e., they appear as keys but never in any values list).
    
    Parameters
    ----------
    connections : dict[int, list[int]]
        Dictionary mapping flowpath IDs to lists of downstream flowpath IDs.
        
    Returns
    -------
    set[int]
        Set of headwater flowpath IDs.
    """
    all_downstream_fps = set(chain.from_iterable(connections.values()))
    all_fp_ids = set(connections.keys())
    
    return all_fp_ids - all_downstream_fps


def find_tailwaters(connections: dict[int, list[int]]) -> set[int]:
    """
    Find tailwater flowpaths in the network.
    
    Tailwaters are flowpaths with no downstream connections (empty list).
    
    Parameters
    ----------
    connections : dict[int, list[int]]
        Dictionary mapping flowpath IDs to lists of downstream flowpath IDs.
        
    Returns
    -------
    set[int]
        Set of tailwater flowpath IDs.
    """
    return {fp_id for fp_id, downstream in connections.items() if not downstream}


def distribute_qlateral_to_virtual_flowpaths(
    div_lateralflows_df: pd.DataFrame,
    vfp_dataframe: pd.DataFrame,
    flowpath_dict: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Distribute lateral flows from divides to virtual flowpaths by area percentage.
    
    Parameters
    ----------
    div_lateralflows_df : pd.DataFrame
        Lateral flows indexed by div_id, columns are timestamps
    vfp_dataframe : pd.DataFrame
        Virtual flowpath dataframe indexed by virtual_fp_id
    flowpath_dict : dict
        Mapping of {dn_virtual_nex_id: downstream_flowpath_id}
        
    Returns
    -------
    routing_qlats : pd.DataFrame
        Qlats for routing segments, indexed by virtual_fp_id
    flow_scaling_df : pd.DataFrame
        Qlats for non-routing segments, indexed by virtual_fp_id
    """
    vfp_to_div = vfp_dataframe[['div_id', 'percentage_area_contribution', 'dn_virtual_nex_id', 'routing_segment']].copy()
    vfp_to_div['div_id'] = vfp_to_div['div_id'].astype(int)
    
    # Distribute qlat from div to virtual flowpaths by area percentage
    parent_qlats = div_lateralflows_df.reindex(vfp_to_div['div_id'].values)
    parent_qlats.index = vfp_to_div.index  # index is virtual_fp_id
    
    _qlats_df = parent_qlats.mul(vfp_to_div['percentage_area_contribution'].values, axis=0)
    
    # Split routing vs non-routing
    routing_mask = vfp_to_div['routing_segment']
    
    # Routing segments: keep indexed by virtual_fp_id
    routing_qlats = _qlats_df[routing_mask].copy()
    
    # Non-routing segments: remap to downstream routing flowpath
    non_routing_qlats = _qlats_df[~routing_mask].copy()
    dn_nex_ids = vfp_to_div.loc[non_routing_qlats.index, 'dn_virtual_nex_id']
    downstream_fp = dn_nex_ids.map(flowpath_dict)
    non_routing_qlats.index = downstream_fp.values
    
    # Add non-routing flows to their downstream routing flowpaths
    aggregated_non_routing = non_routing_qlats.groupby(non_routing_qlats.index).sum()
    routing_qlats = routing_qlats.add(aggregated_non_routing, fill_value=0)
    
    # Keep original non-routing indexed by virtual_fp_id for flow scaling reference
    flow_scaling_df = _qlats_df[~routing_mask].copy()
    
    return routing_qlats, flow_scaling_df


def read_ngen_waterbody_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """
    Reads .gpkg or lake.json file and prepares a dataframe, filtered
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

        layers_to_read = [
            "flowpaths",
            "reference_flowpaths",
            "virtual_flowpaths",
            "virtual_nexus",
            "waterbodies",
            "gages",
            "hydrolocations"
        ]

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

        def read_layer(layer_name):
            if layer_name:
                try:
                    _df = gpd.read_file(geo_file_path, layer=layer_name)
                    if 'geometry' in _df.columns:
                        _df = _df.drop(columns=["geometry"])
                    return _df
                except pyogrio.errors.DataSourceError as e:
                    print(f"Error reading file {geo_file_path}: {e}")
                    raise pyogrio.errors.DataSourceError from e
                except pyogrio.errors.DataLayerError as e:
                    print(f"Error reading layer {layer_name}: {e}")
                    raise pyogrio.errors.DataLayerError from e

        # Retrieve geopackage information using matched layer names
        if cpu_pool > 1:
            with Parallel(n_jobs=min(cpu_pool, len(layers_to_read))) as parallel:
                gpkg_list = parallel(delayed(read_layer)(layer) for layer in layers_to_read)

            table_dict = {layers_to_read[i]: gpkg_list[i] for i in range(len(layers_to_read))}
        else:
            table_dict = {layer: read_layer(layer) for layer in layers_to_read}

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


class NHF(AbstractNetwork):
    """ """

    __slots__ = [
        "_upstream_terminal",
        "_nexus_latlon",
        "_duplicate_ids_df",
        "_flow_scaling_segment_df"
    ]

    def __init__(
        self,
        supernetwork_parameters,
        waterbody_parameters,
        data_assimilation_parameters,
        restart_parameters,
        compute_parameters,
        forcing_parameters,
        hybrid_parameters,
        preprocessing_parameters,
        output_parameters,
        verbose=False,
        showtiming=False,
        from_files=True,
        value_dict={},
        bmi_parameters={},
    ):
        """ """
        self.supernetwork_parameters = supernetwork_parameters
        self.waterbody_parameters = waterbody_parameters
        self.data_assimilation_parameters = data_assimilation_parameters
        self.restart_parameters = restart_parameters
        self.compute_parameters = compute_parameters
        self.forcing_parameters = forcing_parameters
        self.hybrid_parameters = hybrid_parameters
        self.preprocessing_parameters = preprocessing_parameters
        self.output_parameters = output_parameters
        self.verbose = verbose
        self.showtiming = showtiming

        if self.verbose:
            print("creating NHF supernetwork connections set")
        if self.showtiming:
            start_time = time.time()

        # ------------------------------------------------
        # Load hydrofabric information
        # ------------------------------------------------
        if self.preprocessing_parameters.get("use_preprocessed_data", False):
            raise NotImplementedError("Preprocessed data reads not implemented")
            # self.read_preprocessed_data()
        else:
            # FIXME: Temporary solution, from_files should only be from command line.
            # Update this once ngen framework is capable of providing this info via BMI.
            from_files_copy = from_files
            if not from_files_copy:
                from_files = True
            if from_files:
                nhf = read_geo_file(
                    self.supernetwork_parameters,
                    self.waterbody_parameters,
                    self.compute_parameters,
                    self.compute_parameters.get("cpu_pool", 1),
                )

                # Handle different key column names between flowpaths and flowpath_attributes
                flowpaths = nhf["flowpaths"]
                waterbodies = nhf["waterbodies"]
                gages = nhf["gages"]
                reference_flowpaths = nhf["reference_flowpaths"]
                virtual_flowpaths = nhf["virtual_flowpaths"]
                virtual_nexus = nhf["virtual_nexus"]
                hydrolocations = nhf["hydrolocations"]
            else:
                raise NotImplementedError("BMI loading not implemented for the NHF")
                # flowpaths, lakes, network = load_bmi_data(
                #     value_dict,
                #     bmi_parameters,
                # )
            # FIXME: See FIXME above.
            if not from_files_copy:
                from_files = False

            # Preprocess network objects
            self.preprocess_network(flowpaths, reference_flowpaths, virtual_flowpaths, virtual_nexus)

            self.crosswalk_nex_flowpath_poi(
                virtual_flowpaths, 
                hydrolocations,
                waterbodies,
                gages,
                reference_flowpaths,
            )

            # Preprocess waterbody objects
            self.preprocess_waterbodies(waterbodies, virtual_nexus)

            # Preprocess data assimilation objects #TODO: Move to DataAssimilation.py?
            self.preprocess_data_assimilation(
                flowpaths, 
                reference_flowpaths, 
                virtual_flowpaths, 
                virtual_nexus,
                waterbodies,
                gages
            )


        if self.verbose:
            print("supernetwork connections set complete")
        if self.showtiming:
            print("... in %s seconds." % (time.time() - start_time))

        super().__init__(from_files, value_dict)

        # Create empty dataframe for coastal_boundary_depth_df. This way we can check if
        # it exists, and only read in SCHISM data during 'assemble_forcings' if it doesn't
        self._coastal_boundary_depth_df = pd.DataFrame()

    def extract_waterbody_connections(rows, target_col, waterbody_null=-9999):
        """Extract waterbody mapping from dataframe.
        TODO deprecate in favor of waterbody_connections property"""
        return rows.loc[rows[target_col] != waterbody_null, target_col].astype("int").to_dict()

    @property
    def downstream_flowpath_dict(self):
        return self._flowpath_dict

    @property
    def waterbody_connections(self):
        """
        A dictionary where the keys are the reach/segment id, and the
        value is the id to look up waterbody parameters
        """
        return self._waterbody_connections

    @property
    def gages(self):
        """
        FIXME
        """
        return self._gages

    @property
    def great_lakes_climatology_df(self):
        return pd.DataFrame()

    @property
    def waterbody_null(self):
        return np.nan  # pd.NA

    @property
    def segment_index(self):
        """
            Segment IDs of all reaches in parameter dataframe
            and diffusive domain.
        """
        # list of all segments in the domain (MC + diffusive)
        self._segment_index = self.dataframe[self.dataframe["routing_segment"]].index  # only getting routing_segment == True virtual_flowpaths
        if self._routing.diffusive_network_data:
            for tw in self._routing.diffusive_network_data:
                self._segment_index = self._segment_index.append(
                    pd.Index(self._routing.diffusive_network_data[tw]['mainstem_segs'])
                )
        return self._segment_index


    def preprocess_network(self, flowpaths, reference_flowpaths, virtual_flowpaths, virtual_nexus):
        assert not virtual_flowpaths.empty, "No virtual flowpaths read to memory from .gpkg" 
        vfp_to_fp_map = reference_flowpaths[['virtual_fp_id', 'fp_id', 'div_id']].copy()
        _vfp = virtual_flowpaths.merge(
            vfp_to_fp_map,
            left_on='virtual_fp_id',
            right_on='virtual_fp_id',
            how='left'
        )
        result = _vfp.merge(
            flowpaths,
            left_on='fp_id',
            right_on='fp_id',
            how='left',
            suffixes=('', '_flowpath')  # Keep vfp columns as-is, suffix flowpath columns
        )
        cols_to_drop = [col for col in result.columns if col.endswith('_flowpath')]
        result = result.drop(columns=cols_to_drop)
        self._dataframe = result
        

        # make the flowpath linkage
        self._flowpath_dict = dict(zip(
            result.loc[:, 'dn_virtual_nex_id'], 
            result.loc[:, 'virtual_fp_id']
        ))

        self._dataframe.set_index("virtual_fp_id", inplace=True)
        self._dataframe = self.dataframe.sort_index()
        
        self._terminal_codes = get_terminal_nexus_ids(virtual_nexus)
        
        # Build upstream terminal mapping
        self._upstream_terminal = build_upstream_terminal(virtual_flowpaths, self._terminal_codes)
        
        # Build connections dictionary - only for routing segments
        routing_vfp = virtual_flowpaths[virtual_flowpaths["routing_segment"]]
        self._connections = build_downstream_connections(
            routing_flowpaths=routing_vfp,
            all_flowpaths=virtual_flowpaths,
            terminal_nexus_ids=self._terminal_codes,
        )
        
        # Validate connections
        is_valid, orphaned = validate_connections(self._connections)
        if not is_valid:
            raise ValueError(
                f"Invalid network connections: {len(orphaned)} downstream IDs not found. "
                f"First 10: {list(orphaned)[:10]}"
            )

        # Store a dataframe containing info about nexus points. This will be reprojected to lat/lon
        # and filtered for only diffusive domain tailwaters in AbstractNetwork.py.
        # Location information will be used to advertise tailwater locations of diffusive domains
        # to the model engine/coastal models
        self._nexus_latlon = virtual_nexus

    def crosswalk_nex_flowpath_poi(
        self, 
        virtual_flowpaths, 
        hydrolocations,
        waterbodies,
        gages,
        reference_flowpaths
    ):
        self._nexus_dict = virtual_flowpaths.groupby("dn_virtual_nex_id")["virtual_fp_id"].apply(list).to_dict()  ##{id: toid}
        if not hydrolocations.empty:
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
        else:
            self._poi_nex_dict = None

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

    def build_qlateral_array(
        self,
        run,
    ):
        # TODO: set default/optional arguments
        qts_subdivisions = run.get("qts_subdivisions", 1)
        nts = run.get("nts", 1)
        qlat_input_folder = run.get("qlat_input_folder", None)

        if qlat_input_folder:
            qlat_input_folder = Path(qlat_input_folder)
            if "qlat_files" in run:
                qlat_files = run.get("qlat_files")
                qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
            elif "qlat_file_pattern_filter" in run:
                qlat_file_pattern_filter = run.get("qlat_file_pattern_filter", "*CHRT_OUT*")
                qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))

            dfs = []

            # FIXME Temporary solution to allow t-route to use ngen nex-* output files as forcing files
            # This capability should be here, but we need to think through how to handle all of this
            # data in memory for large domains and many timesteps... - shorvath, Feb 28, 2024
            qlat_file_pattern_filter = self.forcing_parameters.get("qlat_file_pattern_filter", None)
            if qlat_file_pattern_filter == "nex-*":
                raise NotImplementedError("Nex-output not implemented!")
            else:
                for f in qlat_files:
                    df = read_file(f)
                    df["feature_id"] = df["feature_id"].map(
                        lambda x: int(str(x).removeprefix("nex-")) if str(x).startswith("nex") else int(x)
                    )
                    assert df["feature_id"].is_unique, (
                        f"'feature_id's must be unique. '{f!s}' contains duplicate 'feature_id's: {pformat(df.loc[df['feature_id'].duplicated(), 'feature_id'].to_list())}"
                    )
                    df = df.set_index("feature_id")
                    dfs.append(df)

                # lateral flows [m^3/s] indexed by div_id (divide/catchment)
                div_lateralflows_df = pd.concat(dfs, axis=1)

                # Distribute to downstream flowpaths for routing, keep virtual_fp_id for flow scaling
                nexus_to_downstream_fp = dict(zip(
                    self._dataframe['up_virtual_nex_id'].dropna().astype(int),
                    self._dataframe.loc[self._dataframe['up_virtual_nex_id'].notna()].index
                ))
                qlats_df, self._flow_scaling_segment_df = distribute_qlateral_to_virtual_flowpaths(
                    div_lateralflows_df,
                    self._dataframe,
                    nexus_to_downstream_fp,
                )
        else:
            raise ValueError("qlat_input_folder does not exist")
        all_df = pd.DataFrame(
            np.zeros((len(self.segment_index), len(qlats_df.columns))),
                index=self.segment_index,
                columns=qlats_df.columns,
        )
        all_df.loc[qlats_df.index] = qlats_df
        qlats_df = all_df.sort_index()

        # column filtering
        max_col = 1 + nts // qts_subdivisions
        if len(qlats_df.columns) > max_col:
            qlats_df.drop(qlats_df.columns[max_col:], axis=1, inplace=True)

        # final filter to segment_index
        if not self.segment_index.empty:
            qlats_df = qlats_df[qlats_df.index.isin(self.segment_index)]

        self._qlateral = qlats_df



    def build_et_array(
        self,
        run,
    ):
        col_idx = run.get("et_index_name", "divide_id")
        var_idx = run.get("et_var_name", "ACTUAL_ET")
        try:
            ds = run["et_forcing_ds"]
        except KeyError as e:
            raise KeyError("Cannot find et_forcing_ds in runs") from e
        ds_AET = ds[var_idx]

        # mapping catchments to flowpath IDs
        mapping_dict = dict(zip(
            self._dataframe['divide_id'].values,
            self._dataframe.index.values
        ))
        keys = np.array([mapping_dict[key] for key in ds_AET[col_idx].values])
        
        time_strings = pd.to_datetime(ds_AET.time.values).strftime('%Y%m%d%H%M')
        aet_df = pd.DataFrame(
            data=ds_AET.values,
            index=keys,
            columns=time_strings
        )
        
        aet_df.index.name = 'key'
        ordered_aet_df = aet_df.reindex(self._dataframe.index, fill_value=0) # ordering based on the existing 

        # Convert ET into ELOSS
        try:
            A_w = self._dataframe["tw"] * self._dataframe["dx"]
            _E = ordered_aet_df * self.forcing_parameters["peadj"]
            TIMINT = 1 # Hardcoding for hourly
            # _E is in mm/hr. Thus, MM/HR × (1/1000) × (1/3600) -> m/s
            ELOSS_cms = (_E / 1000 / 3600 / TIMINT).mul(A_w.values, axis=0)
            ELOSS_cfs = ELOSS_cms * 35.3147  # since NGEN runs in cfs, converting from cms to cfs. Can make a config setting later.
        except KeyError as e:
            raise KeyError("Cannot find flowpath attributes to map PET. Can you ensure ") from e
        self._eloss = ELOSS_cfs

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

    def write_preprocessed_data(
        self,
    ):
        # LOG.debug("saving preprocessed network data to disk for future use")
        # todo: consider a better default than None
        destination_folder = self.preprocessing_parameters.get("preprocess_output_folder", None)
        if destination_folder:
            output_filename = self.preprocessing_parameters.get("preprocess_output_filename", "preprocess_output")

        outputs = {
            "dataframe": self.dataframe,
            "flowpath_dict": self._flowpath_dict,
            "terminal_codes": self._terminal_codes,
            "upstream_termincal": self._upstream_terminal,
            "connections": self._connections,
            "waterbody_df": self._waterbody_df,
            "waterbody_types_df": self._waterbody_types_df,
            "waterbody_connections": self._waterbody_connections,
            "waterbody_type_specified": self._waterbody_type_specified,
            "link_lake_crosswalk": self._link_lake_crosswalk,
            "gages": self._gages,
            "usgs_lake_gage_crosswalk": self._usgs_lake_gage_crosswalk,
            "usace_lake_gage_crosswalk": self._usace_lake_gage_crosswalk,
            "rfc_lake_gage_crosswalk": self._rfc_lake_gage_crosswalk,
        }
        np.save(Path(destination_folder).joinpath(output_filename), outputs)

    def read_preprocessed_data(
        self,
    ):
        preprocess_filepath = self.preprocessing_parameters.get("preprocess_source_file", None)
        if preprocess_filepath:
            try:
                inputs = np.load(Path(preprocess_filepath), allow_pickle="TRUE").item()
            except:
                # LOG.critical('Canonot find %s' % Path(preprocess_filepath))
                quit()

            self._dataframe = inputs.get("dataframe", None)
            self._flowpath_dict = inputs.get("flowpath_dict", None)
            self._terminal_codes = inputs.get("terminal_codes", None)
            self._upstream_terminal = inputs.get("upstream_termincal", None)
            self._connections = inputs.get("connections", None)
            self._waterbody_df = inputs.get("waterbody_df", None)
            self._waterbody_types_df = inputs.get("waterbody_types_df", None)
            self._waterbody_connections = inputs.get("waterbody_connections", None)
            self._waterbody_type_specified = inputs.get("waterbody_type_specified", None)
            self._link_lake_crosswalk = inputs.get("link_lake_crosswalk", None)
            self._gages = inputs.get("gages", None)
            self._usgs_lake_gage_crosswalk = inputs.get("usgs_lake_gage_crosswalk", None)
            self._usace_lake_gage_crosswalk = inputs.get("usace_lake_gage_crosswalk", None)
            self._usbr_lake_gage_crosswalk = inputs.get("usbr_lake_gage_crosswalk", None)
            self._rfc_lake_gage_crosswalk = inputs.get("rfc_lake_gage_crosswalk", None)


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


def tailwaters(N):
    """
    Find network tailwaters

    Arguments
    ---------
    N (dict, int: [int]): Network connections graph

    Returns
    -------
    (iterable): tailwater segments

    Notes
    -----
    - If reverse connections graph is handed as input, then function
      will return network headwaters.

    """
    tw = chain.from_iterable(N.values()) - N.keys()
    for m, n in N.items():
        if not n:
            tw.add(m)
    return tw


def reservoir_shore(connections, waterbody_nodes):
    wbody_set = set(waterbody_nodes)
    not_in = lambda x: x not in wbody_set

    shore = set()
    for node in wbody_set:
        shore.update(filter(not_in, connections[node]))
    return list(shore)


def reservoir_boundary(connections, waterbodies, n):
    if n not in waterbodies and n in connections:
        return any(x in waterbodies for x in connections[n])
    return False


def reverse_surjective_mapping(d):
    rd = defaultdict(list)
    for src, dst in d.items():
        rd[dst].append(src)
    rd.default_factory = None
    return rd


def separate_waterbodies(connections, waterbodies):
    waterbody_nodes = {}
    for wb, nodes in reverse_surjective_mapping(waterbodies).items():
        waterbody_nodes[wb] = net = {}
        for n in nodes:
            if n in connections:
                net[n] = list(filter(waterbodies.__contains__, connections[n]))
    return waterbody_nodes


def replace_waterbodies_connections(connections, waterbodies):
    """
    Use a single node to represent waterbodies. The node id is the
    waterbody id. Create a cross walk dictionary that relates lake_ids
    to the terminal segments within the waterbody footprint.

    Arguments
    ---------
    - connections (dict):
    - waterbodies (dict): dictionary relating segment linkIDs to the
                          waterbody lake_id that they lie in

    Returns
    -------
    - new_conn  (dict): connections dictionary with waterbodies represented by single nodes.
                        Waterbody node ids are lake_ids
    - link_lake (dict): cross walk dictionary where keys area lake_ids and values are lists
                        of waterbody tailwater nodes (i.e. the nodes connected to the
                        waterbody outlet).
    """
    new_conn = {}
    link_lake = {}
    waterbody_nets = separate_waterbodies(connections, waterbodies)
    rconn = reverse_network(connections)

    for n in connections:
        if n in waterbodies:
            wbody_code = waterbodies[n]
            if wbody_code in new_conn:
                continue

            # get all nodes from waterbody
            wbody_nodes = [k for k, v in waterbodies.items() if v == wbody_code]
            outgoing = reservoir_shore(connections, wbody_nodes)
            new_conn[wbody_code] = outgoing

            if len(outgoing) >= 1:
                if outgoing[0] in waterbodies:
                    new_conn[wbody_code] = [waterbodies.get(outgoing[0])]
                link_lake[wbody_code] = list(set(rconn[outgoing[0]]).intersection(set(wbody_nodes)))[0]
            else:
                subset_dict = {key: value for key, value in connections.items() if key in wbody_nodes}
                link_lake[wbody_code] = list(tailwaters(subset_dict))[0]

        elif reservoir_boundary(connections, waterbodies, n):
            # one of the children of n is a member of a waterbody
            # replace that child with waterbody code.
            new_conn[n] = []

            for child in connections[n]:
                if child in waterbodies:
                    new_conn[n].append(waterbodies[child])
                else:
                    new_conn[n].append(child)
        else:
            # copy to new network unchanged
            new_conn[n] = connections[n]

    return new_conn, link_lake
