from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import pandas as pd

from .AbstractNetwork import AbstractNetwork
from troute.nhf_discretize import discretize_flowpaths

from troute.nhf_preprocess import (
    NHFPreprocessMixin,
    read_geo_file,
    read_qlat_file,
)

import ewts
LOG = ewts.get_logger(ewts.T_ROUTE_ID)


__verbose__ = False
__showtiming__ = False


class NHF(NHFPreprocessMixin, AbstractNetwork):
    """ """

    __slots__ = [
        "_upstream_terminal",
        "_nexus_latlon",
        "_duplicate_ids_df",
        "_flow_scaling_segment_df",
        "_links_df",
        "_nodes_df",
        "_reference_flowpaths",
        "_fp_to_dn_nex",
        "_nex_to_dn_fp",
        "_upstream_inflow_df",
        "_nexus_virtual_seg_ids",
        "_fp_outlet_crosswalk",
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
            LOG.info("creating NHF supernetwork connections set")
        if self.showtiming:
            start_time = time.time()

        # ------------------------------------------------
        # Load hydrofabric information
        # ------------------------------------------------
        if self.preprocessing_parameters.get("use_preprocessed_data", False):
            raise NotImplementedError("Preprocessed data reads not implemented")
            # self.read_preprocessed_data()
        else:
            # NHF always reads topology from .gpkg files, even in BMI mode.
            # The ngen framework provides only qlat data via BMI; network
            # geometry comes from the geopackage specified in supernetwork_parameters.
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

            # Preprocess network objects
            discretization_len = self.supernetwork_parameters.get("nhf_discretization_len", 300.0)
            self.preprocess_network(flowpaths, reference_flowpaths, virtual_flowpaths, discretization_len)

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
            LOG.info("supernetwork connections set complete")
        if self.showtiming:
            LOG.info("... in %s seconds." % (time.time() - start_time))

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
    def fp_outlet_crosswalk(self):
        """Map outlet link_id -> fp_id for reindexing outputs."""
        return self._fp_outlet_crosswalk

    def preprocess_network(self, flowpaths: pd.DataFrame, reference_flowpaths: pd.DataFrame, virtual_flowpaths: pd.DataFrame, discretization_len_m=300.0):
        """Create routing links (self._dataframe) and weighting data to assign fp flows to links."""
        self._dataframe, self.nexus_remapping = discretize_flowpaths(
            flowpaths=flowpaths,
            virtual_flowpaths=virtual_flowpaths,
            reference_flowpaths=reference_flowpaths,
            discretization_len_m=discretization_len_m,
        )
        self._connections = None  # Forces recomputation on first call to self.connections
        self._terminal_codes = set(self._dataframe["downstream"]).difference(self._dataframe.index)  # Outlets
        self._build_fp_outlet_crosswalk(reference_flowpaths, virtual_flowpaths, self.nexus_remapping)
        self._build_div_weighting_matrix(virtual_flowpaths, reference_flowpaths, self.nexus_remapping)

    def _build_fp_outlet_crosswalk(self, reference_flowpaths: pd.DataFrame, virtual_flowpaths: pd.DataFrame, nexus_remapping: dict[int, int]):
        """Build a mapping from routing link ID to fp_id to be used when writing results.
        
        N.B. There are a few strategies one could use to assign an outflow timeseries for a merged flowpath
        A) Use the timeseries from a randomly selected upstream link (will underestimate flow)
        B) Use the timeseries from the next downstream link (will overestimate flow)
        We choose option B.
        """
        # Get fp_ids.  Pull out IDs that were merged
        ids = reference_flowpaths["fp_id"].dropna().astype(int).unique()
        ids_merged = set(ids).difference(self._dataframe["fp_id"])

        # Aggregate links in _dataframe to get one outlet per fp_id.  Assumes node IDs increase in downstream direction.
        mapping_base = self._dataframe.reset_index().groupby("fp_id", sort=False)["up_node_id"].max().reset_index().astype(int)
        link_2_fp = mapping_base.set_index("up_node_id")["fp_id"].to_dict()
        fp_2_link = mapping_base.set_index("fp_id")["up_node_id"].to_dict()

        ### Get vfp not in mapping. Make a mapping from their upstream to the missing fps.  When we go to outputs, the
        ### upstream outflows will be summed and reported for the merged reach.
        # Get vfps and join fp_id
        sub_ref = reference_flowpaths[["virtual_fp_id", "fp_id"]].dropna().drop_duplicates().astype(int)
        tmp_vfp = pd.merge(sub_ref, virtual_flowpaths[["virtual_fp_id", "dn_virtual_nex_id", "up_virtual_nex_id"]], how="left", on="virtual_fp_id")
        tmp_vfp = tmp_vfp.dropna().astype(int)

        # For each merged VFP, find its upstream VFP; keep only cross-fp connections
        cross_fp = pd.merge(
            tmp_vfp[tmp_vfp["fp_id"].isin(ids_merged)],
            tmp_vfp,
            left_on="up_virtual_nex_id",
            right_on="dn_virtual_nex_id",
            suffixes=("_dn", "_up"),
        )
        cross_fp = cross_fp[cross_fp["fp_id_dn"] != cross_fp["fp_id_up"]]

        # Build merged fp → upstream fp mapping, then resolve chains for consecutive merges
        merged_to_upstream = dict(zip(cross_fp["fp_id_dn"], cross_fp["fp_id_up"]))
        merged_mapping = {}
        for merged_fp in ids_merged:
            current = merged_fp
            seen = set()
            while current in ids_merged and current in merged_to_upstream:
                if current in seen:
                    break
                seen.add(current)
                current = merged_to_upstream[current]
            if current in fp_2_link:
                outlet_link = fp_2_link[current]
                merged_mapping.setdefault(outlet_link, []).append(merged_fp)

        # Put all results into the mapping dict
        self._fp_outlet_crosswalk = defaultdict(list)
        # Add link mapping
        for k, v in link_2_fp.items():
            self._fp_outlet_crosswalk[k].append(v)
        # Append virtual mapping
        for k, v in merged_mapping.items():
            self._fp_outlet_crosswalk[k].extend(v)
        

    def _build_div_weighting_matrix(self, virtual_flowpaths: pd.DataFrame, reference_flowpaths: pd.DataFrame, nexus_remapping: dict[int, int]) -> pd.DataFrame:
        """Create weights that can be used to expand div direct runoff into vfp direct runoff.
        
        Channel forcings are supplied at the div/fp level, but because a discretized network is used for routing, those 
        forcings need to be reindexed to their corresponding routing network links.  This could be done with a join or 
        lookup table, but using vectors is more performant at scale. The reindexing vectors are created once per NHF 
        network and are reused for each run set in build_qlateral_array.
        
        This function sets the following class variables
         - self.vfp_divs, which is used with search_sorted to make a vector of div lateral flows matching the order of 
           virtual_flowpaths
         - self.weights, which can then be multiplied by the reindexed div lateral flows to get virtual flowpath 
           specific lateral flows
         - self.vfp_nex_ids, which shows the index of _dataframe for each virtual flowpath and allows us to aggregate 
           multiple lateral flows to a single routing link when necessary
         - self.zero_nodes, which allows us to make a zero dataframe for all routing links that never have any lateral 
           flows. 
        """
        # Make a dataframe for every vfp with percentage_area_contribution, div_id, dn_nex_id, and virtual_fp_id
        vfp_map = pd.merge(reference_flowpaths[["virtual_fp_id", "div_id"]].copy().dropna(subset="virtual_fp_id").drop_duplicates().astype("Int64"), virtual_flowpaths[["virtual_fp_id", "percentage_area_contribution", "dn_virtual_nex_id"]], on="virtual_fp_id", how="left")
        vfp_map["dn_virtual_nex_id"] = vfp_map["dn_virtual_nex_id"].astype(int)

        # Remap down nexuses that changed in discretization
        vfp_map["dn_virtual_nex_id"] = vfp_map["dn_virtual_nex_id"].map(nexus_remapping).fillna(vfp_map["dn_virtual_nex_id"]).astype(int)

        # Remap all down nexuses to their on-network link
        # (explanation) In NHF, flows are added at the downstream end of a virtual_flowpath. To achieve this, we apply 
        # discharges at the link just upstream of the downstream end of the virtual flowpath and use the "bottom" option 
        # for lateral addition location
        vfp_map = pd.merge(vfp_map, self._dataframe.reset_index()[["up_node_id", "downstream", "fp_id"]], how="left", left_on=["dn_virtual_nex_id", "div_id"], right_on=["downstream", "fp_id"])

        # Map all merged vfps to one of their tribs.
        merged_us_lookup = vfp_map.dropna().set_index("dn_virtual_nex_id")["up_node_id"].to_dict() # Dropna subets dict to only non-merged paths
        insna_mask = vfp_map["up_node_id"].isna()
        vfp_map.loc[insna_mask, "up_node_id"] = vfp_map.loc[insna_mask, "dn_virtual_nex_id"].map(merged_us_lookup)

        # Fallback for vfps that hit a headwater
        # (explanation) When a virtual flowpath is the div headwater, there's no link for it to add it's flows to. 
        # Instead, we add them to the next downstream link.
        vfp_map.loc[vfp_map["up_node_id"].isna(), "up_node_id"] = vfp_map.loc[vfp_map["up_node_id"].isna(), "dn_virtual_nex_id"]

        # Cleanup
        vfp_map = vfp_map[["virtual_fp_id", "percentage_area_contribution", "div_id", "up_node_id"]].copy()
        vfp_map["up_node_id"] = vfp_map["up_node_id"].astype(int)

        # Make weights
        self.weights = np.nan_to_num(vfp_map["percentage_area_contribution"].to_numpy())

        # In case NHF percents per div don't sum to 100, distribute remainder evenly
        groups = vfp_map["div_id"].astype("int64").to_numpy()
        known_sum = np.bincount(groups, weights=self.weights)
        vfp_count = np.bincount(groups)
        share = np.divide(1 - known_sum, vfp_count, out=np.zeros_like(vfp_count, dtype=float), where=vfp_count!=0)
        self.weights += share[groups]

        # Set class variables
        self.vfp_nex_ids = vfp_map["up_node_id"].to_numpy()
        self.vfp_divs = vfp_map["div_id"].to_numpy()
        self.weights = self.weights[:, np.newaxis]
        self.zero_nodes = list(set(self._dataframe.index).difference(self.vfp_nex_ids))

    def _load_forcing(self, run: dict[str, Any]) -> pd.DataFrame:
        """Load channel forcing data for a run set."""
        qlat_input_folder = run.get("qlat_input_folder", None)

        if qlat_input_folder:
            qlat_input_folder = Path(qlat_input_folder)
            if "qlat_files" in run:
                qlat_files = run.get("qlat_files")
                qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
            elif "qlat_file_pattern_filter" in run:
                qlat_file_pattern_filter = run.get("qlat_file_pattern_filter", "*CHRT_OUT*")
                qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))
                # TODO: Filter for max_col = 1 + nts // qts_subdivisions

            dfs = []

            # FIXME Temporary solution to allow t-route to use ngen nex-* output files as forcing files
            # This capability should be here, but we need to think through how to handle all of this
            # data in memory for large domains and many timesteps... - shorvath, Feb 28, 2024
            qlat_file_pattern_filter = self.forcing_parameters.get("qlat_file_pattern_filter", None)
            if qlat_file_pattern_filter == "nex-*":
                raise NotImplementedError("Nex-output not implemented!")
            else:
                with ProcessPoolExecutor(max_workers=self.compute_parameters.get("cpu_pool", 1)) as exe:
                    dfs = list(exe.map(read_qlat_file, qlat_files))

            # lateral flows [m^3/s] indexed by div_id (divide/catchment)
            div_direct_runoff = pd.concat(dfs, axis=1)
            return div_direct_runoff

    def build_qlateral_array(self, run: dict[str, Any]) -> None:
        """Expand channel forcings provided at the div/fp level to a dataframe of link qlaterals."""
        # Load qlats
        div_direct_runoff_df = self._load_forcing(run)
        # Expand from divs to link qlats
        self._build_qlateral_array_direct(div_direct_runoff_df)

    def _build_qlateral_array_direct(self, div_direct_runoff_df: pd.DataFrame) -> None:
        """Expand channel forcings provided at the div/fp level to a dataframe of link qlaterals."""
        # Apply flow scaling to expand runoff into virtual flowpaths
        div_ids = div_direct_runoff_df.index.to_numpy()
        div_order = np.argsort(div_ids)
        div_sorted = div_ids[div_order]
        vfp_div_ind = div_order[np.searchsorted(div_sorted, self.vfp_divs)]
        vfp_flows = div_direct_runoff_df.values[vfp_div_ind, :] * self.weights

        # Aggregate by routing link
        unique_ids, inv = np.unique(self.vfp_nex_ids, return_inverse=True)
        out = np.zeros((len(unique_ids), vfp_flows.shape[1]))
        np.add.at(out, inv, vfp_flows)

        # Make qlat dataframe
        qlat_valid = pd.DataFrame(out, index=unique_ids, columns=div_direct_runoff_df.columns)

        # Add empty records for other links
        qlat_zero = pd.DataFrame(0.0, index=self.zero_nodes, columns=div_direct_runoff_df.columns)

        self._qlateral = pd.concat([qlat_valid, qlat_zero]).fillna(0.0)
 

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
