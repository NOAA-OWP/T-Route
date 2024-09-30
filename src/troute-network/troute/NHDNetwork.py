from .AbstractNetwork import AbstractNetwork
import troute.nhd_io as nhd_io
import pandas as pd
import numpy as np
import time
import pathlib
from collections import defaultdict
import netCDF4
from joblib import delayed, Parallel
import pyarrow.parquet as pq

from troute.nhd_network import reverse_dict, extract_waterbody_connections, gage_mapping, extract_connections, replace_waterbodies_connections

__showtiming__ = True #FIXME pass flag
__verbose__ = True #FIXME pass verbosity


class NHDNetwork(AbstractNetwork):
    """
    
    """
    def __init__(
                self, 
                supernetwork_parameters, 
                waterbody_parameters, 
                restart_parameters, 
                forcing_parameters, 
                compute_parameters, 
                data_assimilation_parameters, 
                hybrid_parameters,
                output_parameters, 
                verbose=False, 
                showtiming=False,
                ):
        """
        
        """
        self.supernetwork_parameters = supernetwork_parameters
        self.waterbody_parameters = waterbody_parameters
        self.data_assimilation_parameters = data_assimilation_parameters
        self.restart_parameters = restart_parameters
        self.compute_parameters = compute_parameters
        self.forcing_parameters = forcing_parameters
        self.hybrid_parameters = hybrid_parameters
        self.output_parameters = output_parameters
        self.verbose = verbose
        self.showtiming = showtiming
        self._poi_nex_dict = None
        self._nexus_dict = None
        if self.verbose:
            print("creating supernetwork connections set")
        if self.showtiming:
            start_time = time.time()
        
        #------------------------------------------------
        # Load Geo Data
        #------------------------------------------------

        self.read_geo_file()

        if self.verbose:
            print("supernetwork connections set complete")
        if self.showtiming:
            print("... in %s seconds." % (time.time() - start_time))

        self._flowpath_dict = {}
        self._gl_climatology_df = pd.DataFrame()
        self._canadian_gage_link_df = pd.DataFrame(columns=['gages','link']).set_index('link')
        self._nexus_dict = dict()
        super().__init__()

        # Create empty dataframe for coastal_boundary_depth_df. This way we can check if
        # it exists, and only read in SCHISM data during 'assemble_forcings' if it doesn't
        self._coastal_boundary_depth_df = pd.DataFrame()
        

    def extract_waterbody_connections(rows, target_col, waterbody_null=-9999):
        """Extract waterbody mapping from dataframe.
        TODO deprecate in favor of below property???"""
        return (
            rows.loc[rows[target_col] != waterbody_null, target_col].astype("int").to_dict()
        )

    @property
    def waterbody_connections(self):
        if self._waterbody_connections is None :
            self._waterbody_connections = self._dataframe.loc[
                self._dataframe["waterbody"] != self.waterbody_null, "waterbody"
            ].astype("int").to_dict()
        return self._waterbody_connections

    @property
    def gages(self):
        """
        
        """
        if self._gages is None and "gages" in self._dataframe.columns:
            self._gages = nhd_io.build_filtered_gage_df(self._dataframe[["gages"]])
            
        return self._gages

    @property
    def waterbody_null(self):
        return -9999
    
    @property
    def usgs_lake_gage_crosswalk(self):
        return self._usgs_lake_gage_crosswalk
    
    @property
    def usace_lake_gage_crosswalk(self):
        return self._usace_lake_gage_crosswalk

    #@property
    #def wbody_conn(self):
    #    return self._waterbody_connections    

    def read_geo_file(self,):
        '''
        Construct network connections network, parameter dataframe, waterbody mapping, 
        and gage mapping. This is an intermediate-level function that calls several 
        lower level functions to read data, conduct network operations, and extract mappings.
        
        Arguments
        ---------
        supernetwork_parameters (dict): User input network parameters
        
        Returns:
        --------
        connections (dict int: [int]): Network connections
        param_df          (DataFrame): Geometry and hydraulic parameters
        wbodies       (dict, int: int): segment-waterbody mapping
        gages         (dict, int: int): segment-gage mapping
        
        '''
        
        # crosswalking dictionary between variables names in input dataset and 
        # variable names recognized by troute.routing module.
        cols = self.supernetwork_parameters.get(
            'columns', 
            {
            'key'       : 'link',
            'downstream': 'to',
            'dx'        : 'Length',
            'n'         : 'n',
            'ncc'       : 'nCC',
            's0'        : 'So',
            'bw'        : 'BtmWdth',
            'waterbody' : 'NHDWaterbodyComID',
            'gages'     : 'gages',
            'tw'        : 'TopWdth',
            'twcc'      : 'TopWdthCC',
            'alt'       : 'alt',
            'musk'      : 'MusK',
            'musx'      : 'MusX',
            'cs'        : 'ChSlp',
            }
        )
        # Remove 'mainstem' col if it exists:
        if 'mainstem' in cols:
            del cols['mainstem']
        
        # numeric code used to indicate network terminal segments
        terminal_code = self.supernetwork_parameters.get("terminal_code", 0)

        # read parameter dataframe 
        self._dataframe = nhd_io.read(pathlib.Path(self.supernetwork_parameters["geo_file_path"]))

        # select the column names specified in the values in the cols dict variable
        self._dataframe = self.dataframe[list(cols.values())]
        
        # rename dataframe columns to keys in the cols dict variable
        self._dataframe = self.dataframe.rename(columns=reverse_dict(cols))
        
        # handle synthetic waterbody segments
        synthetic_wb_segments = self.supernetwork_parameters.get("synthetic_wb_segments", None)
        synthetic_wb_id_offset = self.supernetwork_parameters.get("synthetic_wb_id_offset", 9.99e11)
        if synthetic_wb_segments:
            # rename the current key column to key32
            key32_d = {"key":"key32"}
            self._dataframe = self.dataframe.rename(columns=key32_d)
            # create a key index that is int64
            # copy the links into the new column
            self._dataframe["key"] = self.dataframe.key32.astype("int64")
            # update the values of the synthetic reservoir segments
            fix_idx = self.dataframe.key.isin(set(synthetic_wb_segments))
            self._dataframe.loc[fix_idx,"key"] = (self.dataframe[fix_idx].key + synthetic_wb_id_offset).astype("int64")

        # set parameter dataframe index as segment id number, sort
        self._dataframe = self.dataframe.set_index("key").sort_index()

        # get and apply domain mask
        if self.supernetwork_parameters["mask_file_path"]:
            data_mask = nhd_io.read_mask(
                pathlib.Path(self.supernetwork_parameters["mask_file_path"]),
                layer_string=self.supernetwork_parameters.get("mask_layer_string", None),
            )
            data_mask = data_mask.set_index(data_mask.columns[0])
            self._dataframe = self.dataframe.filter(data_mask.index, axis=0)

        # map segment ids to waterbody ids
        self._waterbody_connections = {}
        if "waterbody" in cols:
            self._waterbody_connections = extract_waterbody_connections(
                self.dataframe[["waterbody"]]
            )
            self._dataframe = self.dataframe.drop("waterbody", axis=1)

        # map segment ids to gage ids
        self._gages = {}
        if "gages" in cols:
            self._gages = gage_mapping(self.dataframe[["gages"]])
            self._dataframe = self.dataframe.drop("gages", axis=1)
            
        # There can be an externally determined terminal code -- that's this first value
        self._terminal_codes = set()
        self._terminal_codes.add(terminal_code)
        # ... but there may also be off-domain nodes that are not explicitly identified
        # but which are terminal (i.e., off-domain) as a result of a mask or some other
        # an interior domain truncation that results in a
        # otherwise valid node value being pointed to, but which is masked out or
        # being intentionally separated into another domain.
        self._terminal_codes = self.terminal_codes | set(
            self.dataframe[~self.dataframe["downstream"].isin(self.dataframe.index)]["downstream"].values
        )
        
        # build connections dictionary
        self._connections = extract_connections(
            self.dataframe, "downstream", terminal_codes=self.terminal_codes
        )
        self._dataframe = self.dataframe.drop("downstream", axis=1)

        self._dataframe = self.dataframe.astype("float32")

        break_network_at_waterbodies = self.waterbody_parameters.get(
            "break_network_at_waterbodies", False
        )

        # if waterbodies are being simulated, adjust the connections graph so that 
        # waterbodies are collapsed to single nodes. Also, build a mapping between 
        # waterbody outlet segments and lake ids
        if break_network_at_waterbodies:
            self._connections, self._link_lake_crosswalk = replace_waterbodies_connections(
                self.connections, self.waterbody_connections
            )
        else:
            self._link_lake_crosswalk = None

        #============================================================================
        # Retrieve and organize waterbody parameters

        self._waterbody_type_specified = False
        if break_network_at_waterbodies:
            
            # Read waterbody parameters from LAKEPARM file
            level_pool_params = self.waterbody_parameters.get('level_pool', defaultdict(list))
            self._waterbody_df = nhd_io.read_lakeparm(
                level_pool_params['level_pool_waterbody_parameter_file_path'],
                level_pool_params.get("level_pool_waterbody_id", 'lake_id'),
                self.waterbody_connections.values()
            )

            # Remove duplicate lake_ids and rows
            self._waterbody_df = (
                self.waterbody_dataframe.reset_index()
                .drop_duplicates(subset="lake_id")
                .set_index("lake_id")
                .sort_index()
            )

            # Declare empty dataframe
            self._waterbody_types_df = pd.DataFrame()

            # Check if hybrid-usgs or hybrid-usace reservoir DA is set to True
            reservoir_da = self.data_assimilation_parameters.get(
                'reservoir_da', 
                {}
            )
            
            if reservoir_da:
                usgs_hybrid  = reservoir_da['reservoir_persistence_da'].get(
                    'reservoir_persistence_usgs', 
                    False
                )
                usace_hybrid = reservoir_da['reservoir_persistence_da'].get(
                    'reservoir_persistence_usace', 
                    False
                )
                param_file   = reservoir_da.get(
                    'reservoir_parameter_file',
                    None
                )
            else:
                param_file = None
                usace_hybrid = False
                usgs_hybrid = False
                
            # check if RFC-type reservoirs are set to true
            rfc_params = reservoir_da['reservoir_rfc_da']
            if rfc_params:
                rfc_forecast = rfc_params.get(
                    'reservoir_rfc_forecasts',
                    False
                )
                param_file = reservoir_da.get('reservoir_parameter_file', None)
            else:
                rfc_forecast = False

            if (param_file and reservoir_da) or (param_file and rfc_forecast):
                self._waterbody_type_specified = True
                (
                    self._waterbody_types_df, 
                    self._usgs_lake_gage_crosswalk, 
                    self._usace_lake_gage_crosswalk
                ) = nhd_io.read_reservoir_parameter_file(
                    param_file,
                    usgs_hybrid,
                    usace_hybrid,
                    rfc_forecast,
                    level_pool_params.get("level_pool_waterbody_id", 'lake_id'),
                    reservoir_da.get('crosswalk_usgs_gage_field', 'usgs_gage_id'),
                    reservoir_da.get('crosswalk_usgs_lakeID_field', 'usgs_lake_id'),
                    reservoir_da.get('crosswalk_usace_gage_field', 'usace_gage_id'),
                    reservoir_da.get('crosswalk_usace_lakeID_field', 'usace_lake_id'),
                    self.waterbody_connections.values(),
                )
            else:
                self._waterbody_type_specified = True
                self._waterbody_types_df = pd.DataFrame(data = 1, index = self.waterbody_dataframe.index, columns = ['reservoir_type'])
                self._usgs_lake_gage_crosswalk = None
                self._usace_lake_gage_crosswalk = None

        else:
            # Declare empty dataframes
            self._waterbody_types_df = pd.DataFrame()
            self._waterbody_df = pd.DataFrame()
            self._usgs_lake_gage_crosswalk = None
            self._usace_lake_gage_crosswalk = None
    
    def build_qlateral_array(self, run,):
        
        # TODO: set default/optional arguments
        qts_subdivisions = run.get("qts_subdivisions", 1)
        nts = run.get("nts", 1)
        qlat_input_folder = run.get("qlat_input_folder", None)
        qlat_input_file = run.get("qlat_input_file", None)
        cpu_pool = self.compute_parameters.get('cpu_pool', 1)

        if qlat_input_folder:
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            if "qlat_files" in run:
                qlat_files = run.get("qlat_files")
                qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
            elif "qlat_file_pattern_filter" in run:
                qlat_file_pattern_filter = run.get(
                    "qlat_file_pattern_filter", "*CHRT_OUT*"
                )
                qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))

            qlat_file_index_col = run.get(
                "qlat_file_index_col", "feature_id"
            )

            # Parallel reading of qlateral data from CHRTOUT
            with Parallel(n_jobs=cpu_pool) as parallel:
                jobs = []
                for f in qlat_files:
                    jobs.append(
                        delayed(nhd_io.get_ql_from_chrtout)
                        #(f, qlat_file_value_col, gw_bucket_col, terrain_ro_col)
                        #delayed(nhd_io.get_ql_from_csv)
                        (f)                    
                    )
                ql_list = parallel(jobs)

            # get feature_id from a single CHRTOUT file
            with netCDF4.Dataset(qlat_files[0]) as ds:
                idx = ds.variables[qlat_file_index_col][:].filled()

            # package data into a DataFrame
            qlats_df = pd.DataFrame(
                np.stack(ql_list).T,
                index = idx,
                columns = range(len(qlat_files))
            )

            qlats_df = qlats_df[qlats_df.index.isin(self.segment_index)]

        elif qlat_input_file:
            qlats_df = nhd_io.get_ql_from_csv(qlat_input_file)
        else:
            qlat_const = run.get("qlat_const", 0)
            qlats_df = pd.DataFrame(
                qlat_const,
                index=self.segment_index,
                columns=range(nts // qts_subdivisions),
                dtype="float32",
            )

        # TODO: Make a more sophisticated date-based filter
        max_col = 1 + nts // qts_subdivisions
        if len(qlats_df.columns) > max_col:
            qlats_df.drop(qlats_df.columns[max_col:], axis=1, inplace=True)

        if not self.segment_index.empty:
            qlats_df = qlats_df[qlats_df.index.isin(self.segment_index)]

        self._qlateral = qlats_df

def read_file(file_name):
    extension = file_name.suffix
    if extension=='.csv':
        df = pd.read_csv(file_name)
    elif extension=='.parquet':
        df = pq.read_table(file_name).to_pandas().reset_index()
        df.index.name = None
    
    return df