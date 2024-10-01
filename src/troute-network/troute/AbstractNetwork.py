from abc import ABC, abstractmethod
from functools import partial
import pandas as pd
import numpy as np
import multiprocessing
from datetime import datetime, timedelta

import os
import pathlib
import time
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from troute.nhd_network import extract_connections, replace_waterbodies_connections, reverse_network, reachable_network, split_at_waterbodies_and_junctions, split_at_junction, dfs_decomposition
from troute.nhd_network_utilities_v02 import organize_independent_networks
import troute.nhd_io as nhd_io 
from .AbstractRouting import MCOnly, MCwithDiffusive, MCwithDiffusiveNatlXSectionNonRefactored, MCwithDiffusiveNatlXSectionRefactored

LOG = logging.getLogger('')

class AbstractNetwork(ABC):
    """
    
    """
    __slots__ = ["_dataframe", "_waterbody_connections", "_gages",  
                "_terminal_codes", "_connections", "_waterbody_df", 
                "_waterbody_types_df", "_waterbody_type_specified", "_link_gage_df",
                "_canadian_gage_link_df",
                "_independent_networks", "_reaches_by_tw", "_flowpath_dict",
                "_reverse_network", "_q0", "_t0", "_link_lake_crosswalk",
                "_usgs_lake_gage_crosswalk", "_usace_lake_gage_crosswalk", "_rfc_lake_gage_crosswalk",
                "_qlateral", "_break_segments", "_segment_index", "_coastal_boundary_depth_df",
                "supernetwork_parameters", "waterbody_parameters","data_assimilation_parameters",
                "restart_parameters", "compute_parameters", "forcing_parameters",
                "hybrid_parameters", "preprocessing_parameters", "output_parameters",
                "verbose", "showtiming", "break_points", "_routing", "_gl_climatology_df", "_nexus_dict", "_poi_nex_dict"]

    
    def __init__(self, from_files=True, value_dict={}):

        self._independent_networks = None
        self._reverse_network = None
        self._reaches_by_tw = None
        self._q0 = None
        self._t0 = None
        self._qlateral = None
        self._link_gage_df = None
        #qlat_const = forcing_parameters.get("qlat_const", 0)
        #FIXME qlat_const
        """ Figure out a good way to default initialize to qlat_const/c
        qlat_const = 1.0
        self._qlateral = pd.DataFrame(
            qlat_const,
            index=self._dataframe.index,
            columns=range(nts // qts_subdivisions),
            dtype="float32",
        )
        """
        break_network_at_waterbodies = self.waterbody_parameters.get("break_network_at_waterbodies", False)        
        streamflow_da = self.data_assimilation_parameters.get('streamflow_da', False)
        break_network_at_gages       = False       
        if streamflow_da:
            break_network_at_gages   = streamflow_da.get('streamflow_nudging', False)
        self.break_points            = {"break_network_at_waterbodies": break_network_at_waterbodies,
                                        "break_network_at_gages": break_network_at_gages}

        self._break_segments = set()

        if self.break_points["break_network_at_waterbodies"]:
            self._break_segments = self._break_segments | set(self.waterbody_connections.values())
        if self.break_points["break_network_at_gages"]:
            self._break_segments = self._break_segments | set(self.gages.get('gages',{}).keys())
        
        self.initialize_routing_scheme()

        self.create_independent_networks()

        self.initial_warmstate_preprocess(from_files, value_dict)


    def assemble_forcings(self, run,):
        """
        Assemble model forcings. Forcings include hydrological lateral inflows (qlats)
        and coastal boundary depths for hybrid runs
        
        Aguments
        --------
        - run                          (dict): List of forcing files pertaining to a 
                                               single run-set

        Returns
        -------
        
        Notes
        -----
        
        """
    
        # Unpack user-specified forcing parameters
        dt                           = self.forcing_parameters.get("dt", None)
        qts_subdivisions             = self.forcing_parameters.get("qts_subdivisions", None)
        qlat_input_folder            = self.forcing_parameters.get("qlat_input_folder", None)
        qlat_file_index_col          = self.forcing_parameters.get("qlat_file_index_col", "feature_id")
        qlat_file_value_col          = self.forcing_parameters.get("qlat_file_value_col", "q_lateral")
        qlat_file_gw_bucket_flux_col = self.forcing_parameters.get("qlat_file_gw_bucket_flux_col", "qBucket")
        qlat_file_terrain_runoff_col = self.forcing_parameters.get("qlat_file_terrain_runoff_col", "qSfcLatRunoff")

    
        # TODO: find a better way to deal with these defaults and overrides.
        run["t0"]                           = run.get("t0", self.t0)
        run["dt"]                           = run.get("dt", dt)
        run["qts_subdivisions"]             = run.get("qts_subdivisions", qts_subdivisions)
        run["qlat_input_folder"]            = run.get("qlat_input_folder", qlat_input_folder)
        run["qlat_file_index_col"]          = run.get("qlat_file_index_col", qlat_file_index_col)
        run["qlat_file_value_col"]          = run.get("qlat_file_value_col", qlat_file_value_col)
        run["qlat_file_gw_bucket_flux_col"] = run.get("qlat_file_gw_bucket_flux_col", qlat_file_gw_bucket_flux_col)
        run["qlat_file_terrain_runoff_col"] = run.get("qlat_file_terrain_runoff_col", qlat_file_terrain_runoff_col)
        
        #---------------------------------------------------------------------------
        # Assemble lateral inflow data
        #---------------------------------------------------------------------------

        # Place holder, if reading qlats from a file use this.
        # TODO: add an option for reading qlat data from BMI/model engine
        start_time = time.time()
        LOG.info("Creating a DataFrame of lateral inflow forcings ...")

        self.build_qlateral_array(
            run,
        )
        
        LOG.debug(
            "lateral inflow DataFrame creation complete in %s seconds." \
                % (time.time() - start_time)
                )

        #---------------------------------------------------------------------
        # Assemble coastal coupling data [WIP]
        #---------------------------------------------------------------------
        # Run if coastal_boundary_depth_df has not already been created:
        if self._coastal_boundary_depth_df.empty:
            coastal_boundary_elev_files = self.forcing_parameters.get('coastal_boundary_input_file', None) 
            coastal_boundary_domain_files = self.hybrid_parameters.get('coastal_boundary_domain', None)    
            
            if coastal_boundary_elev_files:
                start_time = time.time()    
                LOG.info("creating coastal dataframe ...")
                '''
                coastal_boundary_domain   = nhd_io.read_coastal_boundary_domain(coastal_boundary_domain_files)          
                self._coastal_boundary_depth_df = nhd_io.build_coastal_ncdf_dataframe(
                    coastal_boundary_elev_files,
                    coastal_boundary_domain,
                )
                '''

                coastal_hy_crosswalk = {}    # default empty

                if (coastal_boundary_domain_files):

                    coastal_boundary_domain   = nhd_io.read_coastal_boundary_domain(str(coastal_boundary_domain_files))          

                    if ('coastal_hy_crosswalk' in coastal_boundary_domain.keys()):

                        coastal_hy_crosswalk = coastal_boundary_domain['coastal_hy_crosswalk']

                self._coastal_boundary_depth_df = read_coastal_output(coastal_boundary_elev_files, coastal_hy_crosswalk)
                
                LOG.debug(
                    "coastal boundary elevation observation DataFrame creation complete in %s seconds." \
                    % (time.time() - start_time)
                )
            else:
                self._coastal_boundary_depth_df = pd.DataFrame()            

    def new_q0(self, run_results):
        """
        Prepare a new q0 dataframe with initial flow and depth to act as
        a warmstate for the next simulation chunk.
        """
        self._q0 = pd.concat(
            [
                pd.DataFrame(
                    r[1][:, [-3, -3, -1]], index=r[0], columns=["qu0", "qd0", "h0"]
                )
                for r in run_results
            ],
            copy=False,
        )
        return self._q0
    
    def update_waterbody_water_elevation(self):           
        """
        Update the starting water_elevation of each lake/reservoir
        with flow and depth values from q0
        """
        self._waterbody_df.update(self._q0)
        
    def new_t0(self, dt, nts):
        """
        Update t0 value for next loop iteration
        """
        self._t0 += timedelta(seconds = dt * nts)

    @property
    def network_break_segments(self):
        """
        """
        return self._break_segments

    @property
    def reverse_network(self):
        """
        
        """
        if self._reverse_network is None:
            self._reverse_network = reverse_network(self.connections)
        return self._reverse_network

    @property
    def independent_networks(self):
        """
        
        """
        if self._independent_networks is None:
            # STEP 2: Identify Independent Networks and Reaches by Network
            if self.showtiming:
                start_time = time.time()
            if self.verbose:
                print("organizing connections into reaches ...")

            self._independent_networks = reachable_network(self.reverse_network)
            
            if self.verbose:
                print("reach organization complete")
            if self.showtiming:
                print("... in %s seconds." % (time.time() - start_time))
        return self._independent_networks
    
    @property
    def reaches_by_tailwater(self):
        """
        
        """
        if self._reaches_by_tw is None:
            self._reaches_by_tw = {}
            for tw, net in self.independent_networks.items():
                if self.network_break_segments:
                    path_func = partial(
                        split_at_waterbodies_and_junctions, self.network_break_segments, net
                    )
                else:
                    path_func = partial(split_at_junction, net)

                self._reaches_by_tw[tw] = dfs_decomposition(net, path_func)
        return self._reaches_by_tw

    @property
    def waterbody_dataframe(self):
        return self._waterbody_df
    
    @property
    def waterbody_types_dataframe(self):
        return self._waterbody_types_df
    
    @property
    def waterbody_type_specified(self):
        if self._waterbody_type_specified is None:
            self._waterbody_type_specified = False
        return self._waterbody_type_specified
    
    @property
    def link_lake_crosswalk(self):
        return self._link_lake_crosswalk

    @property
    def connections(self):
        if self._connections is None:
            self._connections = extract_connections(
                self._dataframe, "downstream", terminal_codes=self._terminal_codes
            )
        return self._connections

    @property
    def qlateral(self):
        """
        
        """
        return self._qlateral

    @property
    def q0(self):
        """
            Initial channel segment flow values
            If not set elsewhere, they are 0
        """
        if self._q0 is None:
            self._q0 =  pd.DataFrame(
            0, index=self._dataframe.index, columns=["qu0", "qd0", "h0"], dtype="float32",
            )
        return self._q0

    @property
    def t0(self):
        """
            Time 0 as a datetime object
            If not set elsewhere, it defaults to "2015-08-16_00:00:00"
        """
        if self._t0 is None:
            self._t0 = datetime.strptime("2015-08-16_00:00:00", "%Y-%m-%d_%H:%M:%S")
        return self._t0

    @t0.setter
    def t0(self, value):
        """
        
        """
        if isinstance(value, datetime):
            self._t0 = value
        else:
            self._t0 = datetime.strptime(value, "%Y-%m-%d_%H:%M:%S")
    
    @property
    def segment_index(self):
        """
            Segment IDs of all reaches in parameter dataframe
            and diffusive domain.
        """
        # list of all segments in the domain (MC + diffusive)
        self._segment_index = self.dataframe.index
        # Skip this process for enabling the exchange of flow between MC and diffusive during runtime.
        if 'hybrid-routing' not in self.compute_parameters['compute_kernel']:
            if self._routing.diffusive_network_data:
                for tw in self._routing.diffusive_network_data:
                    self._segment_index = self._segment_index.append(
                        pd.Index(self._routing.diffusive_network_data[tw]['mainstem_segs'])
                    )

        return self._segment_index
    
    @property
    def link_gage_df(self):
        if self._link_gage_df is None:
            self._link_gage_df = pd.DataFrame.from_dict(self._gages)
            self._link_gage_df.index.name = 'link'
        return self._link_gage_df.copy()

    @property
    def usgs_lake_gage_crosswalk(self):
        return self._usgs_lake_gage_crosswalk
    
    @property
    def usace_lake_gage_crosswalk(self):
        return self._usace_lake_gage_crosswalk
    
    @property
    def rfc_lake_gage_crosswalk(self):
        return self._rfc_lake_gage_crosswalk

    @property
    @abstractmethod
    def waterbody_connections(self):
        pass

    @property
    @abstractmethod
    def waterbody_null(self):
        pass

    @property
    @abstractmethod
    def gages(self):
        pass

    @property
    def dataframe(self):
        return self._dataframe
    
    @property
    def nexus_dict(self):
        return self._nexus_dict
    
    @property
    def poi_nex_dict(self):
        return self._poi_nex_dict

    @property
    def terminal_codes(self):
        return self._terminal_codes
    
    @property
    def coastal_boundary_depth_df(self):
        """
        
        """
        return self._coastal_boundary_depth_df

    @property
    def diffusive_network_data(self):
        return self._routing.diffusive_network_data
    
    @property
    def topobathy_df(self):
        return self._routing.topobathy_df
    
    @property
    def refactored_diffusive_domain(self):
        return self._routing.refactored_diffusive_domain
    
    @property
    def refactored_reaches(self):
        return self._routing.refactored_reaches
    
    @property
    def unrefactored_topobathy_df(self):
        return self._routing.unrefactored_topobathy_df
    
    @property
    def great_lakes_climatology_df(self):
        return self._gl_climatology_df
    
    @property
    def canadian_gage_df(self):
        return self._canadian_gage_link_df

        
    def set_synthetic_wb_segments(self, synthetic_wb_segments, synthetic_wb_id_offset):
        """
        
        """
        self._dataframe.reset_index(inplace=True) #reset index so key is now column
        # rename the current key column to key32
        key32_d = {"key":"key32"}
        self._dataframe = self._dataframe.rename(columns=key32_d)
        # create a key index that is int64
        # copy the links into the new column
        self._dataframe["key"] = self._dataframe.key32.astype("int64")
        # update the values of the synthetic reservoir segments
        fix_idx = self._dataframe.key.isin(set(synthetic_wb_segments))
        self._dataframe.loc[fix_idx,"key"] = (self._dataframe[fix_idx].key + synthetic_wb_id_offset).astype("int64")
        #Reset key to index
        self.set_index("key")
        self.sort_index()

    def replace_waterbodies(self):
        """

        """
        #Make sure to update held state self._connections
        #but pass the property self.connectionsto ensure it gets properly instansiated
        #in case it hasn't already been
        self._connections = replace_waterbodies_connections(
            self.connections, self._waterbody_connections
        )

    def set_index(self, key):
        """
        
        """
        #If the index name is already `key`, don't bother
        if self._dataframe.index.name != key:
            self._dataframe.set_index(key, inplace=True)
    
    def sort_index(self):
        """
        
        """
        self._dataframe = self._dataframe.sort_index()
    
    def drop(self, key, axis=1):
        """
            FIXME can be problematic to drop keys
            before certain properties are intialized...
        """
        self._dataframe.drop(key, axis=axis, inplace=True)

    def astype(self, type, columns=None):
        """
        
        """
        if columns:
            self._dataframe[columns] = self._dataframe[columns].astype(type)
        else:
            self._dataframe = self._dataframe.astype(type)
            
    def initialize_routing_scheme(self,):
        '''
        
        '''
        # Get user inputs from configuration file
        run_hybrid = self.hybrid_parameters.get("run_hybrid_routing", False)
        use_topobathy = self.hybrid_parameters.get('use_natl_xsections', False)
        run_refactored = self.hybrid_parameters.get('run_refactored_network', False)

        routing_type = [run_hybrid, use_topobathy, run_refactored]

        _routing_scheme_map = {
            MCOnly: [False, False, False],
            MCwithDiffusive: [True, False, False],
            MCwithDiffusiveNatlXSectionNonRefactored: [True, True, False],
            MCwithDiffusiveNatlXSectionRefactored: [True, True, True],
            }
        
        # Default to MCOnly routing
        routing_scheme = MCOnly

        # Check user input to determine the routing scheme
        for key, value in _routing_scheme_map.items():
            #if routing_type in value:
            if value==routing_type:
                routing_scheme = key

        routing = routing_scheme(self.hybrid_parameters, self.compute_parameters)

        (
            self._dataframe,
            self._connections
        ) = routing.update_routing_domain(self.dataframe, self.connections, self.waterbody_dataframe)

        self._routing = routing
        hyf = self.supernetwork_parameters.get('network_type', None)=='HYFeaturesNetwork'
        if hyf and routing.diffusive_network_data:
            self.filter_diffusive_nexus_pts()
    
    def create_independent_networks(self,):

        LOG.info("organizing connections into reaches ...")
        start_time = time.time() 
        gage_break_segments = set()
        wbody_break_segments = set()
        
        break_network_at_waterbodies = self.waterbody_parameters.get(
            "break_network_at_waterbodies", False
        )
        
        # if streamflow DA, then break network at gages
        #TODO update to work with HYFeatures, need to determine how we'll do DA...
        break_network_at_gages = False
        
        if break_network_at_waterbodies:
            wbody_break_segments = wbody_break_segments.union(self._waterbody_connections.values())
            
        if break_network_at_gages:
            gage_break_segments = gage_break_segments.union(self.gages['gages'].keys())
    
        (
            self._independent_networks, 
            self._reaches_by_tw, 
            self._reverse_network
            ) = organize_independent_networks(
                self.connections,
                wbody_break_segments,
                gage_break_segments,
                )
        
        LOG.debug("reach organization complete in %s seconds." % (time.time() - start_time))

    def initial_warmstate_preprocess(self, from_files, value_dict):

        '''
        Assemble model initial condition data:
            - waterbody inital states (outflow and pool elevation)
            - channel initial states (flow and depth)
            - initial time
        
        Arguments
        ---------
        
        Returns
        -------
        
        Notes
        -----
        '''

        restart_parameters = self.restart_parameters

        # generalize waterbody ID's to be used with any network
        index_id = self.waterbody_dataframe.index.names[0]

        #----------------------------------------------------------------------------
        # Assemble waterbody initial states (outflow and pool elevation
        #----------------------------------------------------------------------------
        if self.break_points['break_network_at_waterbodies']:

            start_time = time.time()
            LOG.info("setting waterbody initial states ...")

            if from_files:
                # if a lite restart file is provided, read initial states from it.
                if restart_parameters.get("lite_waterbody_restart_file", None):
                    
                    waterbodies_initial_states_df, _ = nhd_io.read_lite_restart(
                        restart_parameters['lite_waterbody_restart_file']
                    )
                    
                # read waterbody initial states from WRF-Hydro type restart file
                elif restart_parameters.get("wrf_hydro_waterbody_restart_file", None):
                    waterbodies_initial_states_df = nhd_io.get_reservoir_restart_from_wrf_hydro(
                        restart_parameters["wrf_hydro_waterbody_restart_file"],
                        restart_parameters["wrf_hydro_waterbody_ID_crosswalk_file"],
                        restart_parameters.get("wrf_hydro_waterbody_ID_crosswalk_file_field_name", index_id),
                        restart_parameters["wrf_hydro_waterbody_crosswalk_filter_file"],
                        restart_parameters.get(
                            "wrf_hydro_waterbody_crosswalk_filter_file_field_name",
                            'NHDWaterbodyComID'
                        ),
                    )
                
                # if no restart file is provided, default initial states
                else:
                    # TODO: Consider adding option to read cold state from route-link file
                    waterbodies_initial_ds_flow_const = 0.0
                    waterbodies_initial_depth_const = -1e9
                    # Set initial states from cold-state
                    waterbodies_initial_states_df = pd.DataFrame(
                        0,
                        index=self.waterbody_dataframe.index,
                        columns=[
                            "qd0",
                            "h0",
                        ],
                        dtype="float32",
                    )
                    # TODO: This assignment could probably by done in the above call
                    waterbodies_initial_states_df["qd0"] = waterbodies_initial_ds_flow_const
                    waterbodies_initial_states_df["h0"] = waterbodies_initial_depth_const
                    waterbodies_initial_states_df["index"] = range(
                        len(waterbodies_initial_states_df)
                    )
            
            else:
                waterbodies_initial_states_df = value_dict['waterbody_df']
                waterbodies_initial_states_ids = value_dict['waterbody_df_index']
                if len(waterbodies_initial_states_df)>0:
                    waterbodies_initial_states_df = pd.DataFrame(
                        data=waterbodies_initial_states_df.reshape(len(waterbodies_initial_states_ids), -1),
                        index=waterbodies_initial_states_ids,
                        columns=['qd0','h0'])
                    waterbodies_initial_states_df["index"] = range(
                        len(waterbodies_initial_states_df)
                    )
                else:
                    # TODO: Consider adding option to read cold state from route-link file
                    waterbodies_initial_ds_flow_const = 0.0
                    waterbodies_initial_depth_const = -1e9
                    # Set initial states from cold-state
                    waterbodies_initial_states_df = pd.DataFrame(
                        0,
                        index=self.waterbody_dataframe.index,
                        columns=[
                            "qd0",
                            "h0",
                        ],
                        dtype="float32",
                    )
                    # TODO: This assignment could probably by done in the above call
                    waterbodies_initial_states_df["qd0"] = waterbodies_initial_ds_flow_const
                    waterbodies_initial_states_df["h0"] = waterbodies_initial_depth_const
                    waterbodies_initial_states_df["index"] = range(
                        len(waterbodies_initial_states_df)
                    )
            
            if len(self.waterbody_dataframe) > 0:
                self._waterbody_df = pd.merge(
                    self.waterbody_dataframe, waterbodies_initial_states_df, on=index_id
                )
            else:
                self._waterbody_df = pd.DataFrame()
            LOG.debug(
                "waterbody initial states complete in %s seconds."\
                % (time.time() - start_time))
            start_time = time.time()

        #----------------------------------------------------------------------------
        # Assemble channel initial states (flow and depth)
        # also establish simulation initialization timestamp
        # 3 Restart Options:
        #   1. From t-route generated lite restart file (network agnostic)
        #   2. From wrf_hydro_restart file (valid for NHDNetwork only)
        #   3. Cold start, requires user specified start datetime
        #----------------------------------------------------------------------------    
        start_time = time.time()
        LOG.info("setting channel initial states ...")
        # if lite restart file is provided, the read channel initial states from it
        if from_files:
            if restart_parameters.get("lite_channel_restart_file", None):
                self._q0, self._t0 = nhd_io.read_lite_restart(
                    restart_parameters['lite_channel_restart_file']
                )
            
            elif restart_parameters.get("wrf_hydro_channel_restart_file", None):
                self._q0 = nhd_io.get_channel_restart_from_wrf_hydro(
                    restart_parameters["wrf_hydro_channel_restart_file"],
                    restart_parameters["wrf_hydro_channel_ID_crosswalk_file"],
                    restart_parameters.get("wrf_hydro_channel_ID_crosswalk_file_field_name", 'link'),
                    restart_parameters.get("wrf_hydro_channel_restart_upstream_flow_field_name", 'qlink1'),
                    restart_parameters.get("wrf_hydro_channel_restart_downstream_flow_field_name", 'qlink2'),
                    restart_parameters.get("wrf_hydro_channel_restart_depth_flow_field_name", 'hlink'),
                    )

                t0_str = nhd_io.get_param_str(
                        restart_parameters["wrf_hydro_channel_restart_file"], 
                        "Restart_Time"
                    )
                
                # convert timestamp from string to datetime
                self._t0 = datetime.strptime(t0_str, "%Y-%m-%d_%H:%M:%S")
            
            else:
                # Set cold initial state
                # assume to be zero
                # 0, index=connections.keys(), columns=["qu0", "qd0", "h0",], dtype="float32"
                self._q0 = pd.DataFrame(
                    0, index=self.segment_index, columns=["qu0", "qd0", "h0"], dtype="float32",
                    )
                
                # get initial time from user inputs
                self._t0 = restart_parameters.get("start_datetime")
        
        else:
            q0 = value_dict['q0']
            q0_ids = value_dict['q0_index']
            if len(q0)>0:
                self._q0 = pd.DataFrame(data=q0.reshape(len(q0_ids), -1),
                                        index=q0_ids,
                                        columns=['qu0','qd0','h0'])
                self._t0 = value_dict['t0']
            else:
                # Set cold initial state
                self._q0 = pd.DataFrame(
                    0, index=self.segment_index, columns=["qu0", "qd0", "h0"], dtype="float32",
                    )
                
                # get initial time from user inputs
                self._t0 = restart_parameters.get("start_datetime")

        LOG.debug(
            "channel initial states complete in %s seconds."\
            % (time.time() - start_time)
        )

    def build_forcing_sets(self,):

        forcing_parameters = self.forcing_parameters
        supernetwork_parameters = self.supernetwork_parameters
        stream_output = self.output_parameters.get('stream_output', None)
        run_sets           = forcing_parameters.get("qlat_forcing_sets", None)
        qlat_input_folder  = forcing_parameters.get("qlat_input_folder", None)
        nts                = forcing_parameters.get("nts", None)
        max_loop_size      = forcing_parameters.get("max_loop_size", 12)
        dt                 = forcing_parameters.get("dt", None)

        try:
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            assert qlat_input_folder.is_dir() == True
        except TypeError:
            raise TypeError("Aborting simulation because no qlat_input_folder is specified in the forcing_parameters section of the .yaml control file.") from None
        except AssertionError:
            raise AssertionError("Aborting simulation because the qlat_input_folder:", qlat_input_folder,"does not exist. Please check the the nexus_input_folder variable is correctly entered in the .yaml control file") from None

        forcing_glob_filter = forcing_parameters["qlat_file_pattern_filter"]
        binary_folder = forcing_parameters.get('binary_nexus_file_folder', None)

        if forcing_glob_filter=="nex-*" and binary_folder:
            print("Reformating qlat nexus files as hourly binary files...")
            qlat_files = qlat_input_folder.glob(forcing_glob_filter)

            #Check that directory/files specified will work
            if not binary_folder:
                raise(RuntimeError("No output binary qlat folder supplied in config"))
            elif not os.path.exists(binary_folder):
                raise(RuntimeError("Output binary qlat folder supplied in config does not exist"))
            
            #Add tnx for backwards compatability
            qlat_files_list = list(qlat_files) + list(qlat_input_folder.glob('tnx*.csv'))
            #Convert files to binary hourly files, reset nexus input information
            qlat_input_folder, forcing_glob_filter = nex_files_to_binary(qlat_files_list, binary_folder)
            forcing_parameters["qlat_input_folder"] = qlat_input_folder
            forcing_parameters["qlat_file_pattern_filter"] = forcing_glob_filter
        
        if forcing_glob_filter=="nex-*":
            all_files = sorted(qlat_input_folder.glob(forcing_glob_filter))
            final_timestamp = pd.read_csv(all_files[0], header=None, index_col=[0]).tail(1).iloc[0,0]
            final_timestamp = datetime.strptime(final_timestamp.strip(), "%Y-%m-%d %H:%M:%S")
            
            all_files = [os.path.basename(f) for f in all_files]
            
            run_sets = [
                {
                    'qlat_files': all_files,
                    'nts': nts,
                    'final_timestamp': final_timestamp
                }
            ]
            
        # TODO: Throw errors if insufficient input data are available
        elif run_sets:        
            #FIXME: Change it for hyfeature
            '''
            # append final_timestamp variable to each set_list
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            for (s, _) in enumerate(run_sets):
                final_chrtout = qlat_input_folder.joinpath(run_sets[s]['qlat_files'
                        ][-1])
                final_timestamp_str = nhd_io.get_param_str(final_chrtout,
                        'model_output_valid_time')
                run_sets[s]['final_timestamp'] = \
                    datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')
            '''  
        elif qlat_input_folder:        
            # Construct run_set dictionary from user-specified parameters

            # get the first and seconded files from an ordered list of all forcing files
            qlat_input_folder = pathlib.Path(qlat_input_folder)
            all_files          = sorted(qlat_input_folder.glob(forcing_glob_filter))
            first_file         = all_files[0]
            second_file        = all_files[1]

            # Deduce the timeinterval of the forcing data from the output timestamps of the first
            # two ordered CHRTOUT files
            if forcing_glob_filter=="*.CHRTOUT_DOMAIN1":
                t1 = nhd_io.get_param_str(first_file, "model_output_valid_time")
                t1 = datetime.strptime(t1, "%Y-%m-%d_%H:%M:%S")
                t2 = nhd_io.get_param_str(second_file, "model_output_valid_time")
                t2 = datetime.strptime(t2, "%Y-%m-%d_%H:%M:%S")
            elif forcing_glob_filter.startswith('*NEXOUT'):
                t1_str = first_file.name.split('NEXOUT', 1)[0]
                t1 = datetime.strptime(t1_str, '%Y%m%d%H%M')
                t2_str = second_file.name.split('NEXOUT', 1)[0]
                t2 = datetime.strptime(t2_str, '%Y%m%d%H%M')
            else:
                df     = read_file(first_file)
                t1_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")
                t1     = datetime.strptime(t1_str,"%Y-%m-%d_%H:%M:%S")
                df     = read_file(second_file)
                t2_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")
                t2     = datetime.strptime(t2_str,"%Y-%m-%d_%H:%M:%S")
            
            dt_qlat_timedelta = t2 - t1
            dt_qlat = dt_qlat_timedelta.seconds

            # determine qts_subdivisions
            qts_subdivisions = dt_qlat / dt
            if dt_qlat % dt == 0:
                qts_subdivisions = int(dt_qlat / dt)
            # make sure that qts_subdivisions = dt_qlat / dt
            forcing_parameters['qts_subdivisions']= qts_subdivisions

            # the number of files required for the simulation
            nfiles = int(np.ceil(nts / qts_subdivisions))
            if stream_output:
                stream_output_time = stream_output.get('stream_output_time', None)
                if stream_output_time and stream_output_time > max_loop_size:
                    max_loop_size = stream_output_time
            # list of forcing file datetimes
            #datetime_list = [t0 + dt_qlat_timedelta * (n + 1) for n in
            #                 range(nfiles)]
            # ** Correction ** Because qlat file at time t is constantly applied throughout [t, t+1],
            #               ** n + 1 should be replaced by n
            datetime_list = [self.t0 + dt_qlat_timedelta * (n) for n in
                            range(nfiles)]        
            datetime_list_str = [datetime.strftime(d, '%Y%m%d%H%M') for d in
                                datetime_list]

            # list of forcing files
            forcing_filename_list = [d_str + forcing_glob_filter[1:] for d_str in
                                    datetime_list_str]
            
            # check that all forcing files exist
            for f in forcing_filename_list:
                try:
                    J = pathlib.Path(qlat_input_folder.joinpath(f))     
                    assert J.is_file() == True
                except AssertionError:
                    raise AssertionError("Aborting simulation because forcing file", J, "cannot be not found.") from None
                    
            # build run sets list
            run_sets = []
            k = 0
            j = 0
            nts_accum = 0
            nts_last = 0
            while k < len(forcing_filename_list):
                run_sets.append({})

                if k + max_loop_size < len(forcing_filename_list):
                    run_sets[j]['qlat_files'] = forcing_filename_list[k:k
                        + max_loop_size]
                else:
                    run_sets[j]['qlat_files'] = forcing_filename_list[k:]

                nts_accum += len(run_sets[j]['qlat_files']) * qts_subdivisions
                if nts_accum <= nts:
                    run_sets[j]['nts'] = int(len(run_sets[j]['qlat_files'])
                                            * qts_subdivisions)
                else:
                    run_sets[j]['nts'] = int(nts - nts_last)

                final_qlat = qlat_input_folder.joinpath(run_sets[j]['qlat_files'][-1]) 
                if forcing_glob_filter=="*.CHRTOUT_DOMAIN1":           
                    final_timestamp_str = nhd_io.get_param_str(final_qlat,'model_output_valid_time')
                elif forcing_glob_filter.startswith('*NEXOUT'):
                    
                    final_timestamp_str = datetime.strptime(
                        final_qlat.name.split('NEXOUT', 1)[0],
                        "%Y%m%d%H%M"
                    ).strftime("%Y-%m-%d_%H:%M:%S")
                else:
                    df = read_file(final_qlat)
                    final_timestamp_str = pd.to_datetime(df.columns[1]).strftime("%Y-%m-%d_%H:%M:%S")           
                
                run_sets[j]['final_timestamp'] = \
                    datetime.strptime(final_timestamp_str, '%Y-%m-%d_%H:%M:%S')

                nts_last = nts_accum
                k += max_loop_size
                j += 1

        return run_sets
    
    def filter_diffusive_nexus_pts(self,):
        # Filter nexus dataframe containing lat/lon of nexus points for just the diffusive
        # domain tailwaters.
        nexus_latlon = self._nexus_latlon
        diff_tw_ids = list(self._routing.diffusive_network_data.keys())
        diff_tw_ids = ['nex-' + str(s) for s in diff_tw_ids]
        nexus_latlon = nexus_latlon[nexus_latlon['id'].isin(diff_tw_ids)]
        nexus_latlon['id'] = nexus_latlon['id'].str.split('-',expand=True).loc[:,1].astype(float).astype(int)
        lat_lon_crs = nexus_latlon[['id','geometry']]
        lat_lon_crs = lat_lon_crs.to_crs(crs=4326)
        lat_lon_crs['lon'] = lat_lon_crs.geometry.x
        lat_lon_crs['lat'] = lat_lon_crs.geometry.y
        lat_lon_crs['crs'] = str(lat_lon_crs.crs)
        lat_lon_crs = lat_lon_crs[['lon','lat','crs']]
        self._nexus_latlon = nexus_latlon[['id']].join(lat_lon_crs)

def get_timesteps_from_nex(nexus_files):
    # Return a list of output files
    # Open the first nexus file and extract each timestamp
    output_file_timestamps = []
    with open(nexus_files[0]) as f:
        for line in f:
            output_file_timestamps.append(line.split(',')[1].strip())
    # Convert and reformat dates in the list
    output_file_timestamps = [pd.to_datetime(i).strftime("%Y%m%d%H%M") for i in output_file_timestamps]

    # Sort the list
    output_file_timestamps.sort()
    return output_file_timestamps


def split_csv_file(nexus_file, binary_folder):
    catchment_id = get_id_from_filename(nexus_file)
    # Split the csv file into multiple csv files
    # Unescaped command: awk -F ', ' '{ filename="test/tempfile_"$1".csv"; print "114085, "$NF >> filename; close(filename)}' nex-114085_output.csv
    cmd = f'awk -F \', \' \'{{ filename="{binary_folder}/tempfile_"$1".csv"; print "{catchment_id}, "$NF >> filename; close(filename) }}\' {nexus_file}'
    os.system(cmd)


def rewrite_to_parquet(file_args, binary_folder):
    tempfile_id, output_file_id = file_args
    # Rewrite the csv file to parquet
    df = pd.read_csv(f'{binary_folder}/tempfile_{tempfile_id}.csv', names=['feature_id', output_file_id])
    df.set_index('feature_id', inplace=True)  # Set feature_id as the index
    df[output_file_id] = df[output_file_id].astype(float)  # Convert output_file_id column to float64
    table_new = pa.Table.from_pandas(df)
    pq.write_table(table_new, f'{binary_folder}/{output_file_id}NEXOUT.parquet')


def nex_files_to_binary(nexus_files, binary_folder):
    # Get the output files
    output_timesteps = get_timesteps_from_nex(nexus_files)
    partial_split_csv_file = partial(split_csv_file, binary_folder=binary_folder)
    # Split the csv file into multiple csv files
    with multiprocessing.Pool() as pool:
        pool.map(partial_split_csv_file, nexus_files)

    # create a list of tuples to simplify pool.map call
    temp_to_timestep_list = list(enumerate(output_timesteps))
    
    partial_rewrite_to_parquet = partial(rewrite_to_parquet, binary_folder=binary_folder)
    # Rewrite the temp csv files to parquet
    with multiprocessing.Pool() as pool:
        pool.map(partial_rewrite_to_parquet, temp_to_timestep_list)

    
    # Clean up the temp files
    os.system(f'rm -rf {binary_folder}/tempfile_*.csv')
    
    nexus_input_folder = binary_folder
    forcing_glob_filter = '*NEXOUT.parquet'
    
    return nexus_input_folder, forcing_glob_filter

def get_id_from_filename(file_name):
    id = os.path.splitext(file_name)[0].split('-')[1].split('_')[0]
    return int(id)

def read_file(file_name):
    extension = file_name.suffix
    if extension=='.csv':
        df = pd.read_csv(file_name)
    elif extension=='.parquet':
        df = pq.read_table(file_name).to_pandas().reset_index()
        df.index.name = None
    elif extension=='.nc':
        df = xr.open_dataset(file_name).to_pandas().reset_index()
        df.index.name = None
    return df

def read_DFlow_output(ds):
    df = ds[['waterlevel','bedlevel']].to_dataframe()
    df['depth'] = df['waterlevel'] - df['bedlevel']
    # df['station_name'] = df['station_name'].str.decode('utf-8').str.split('-',expand=True).loc[:,1].astype(float).astype(int)
    df['station_name'] = df['station_name'].str.decode('utf-8').str.extract(r'(\d+)').astype(float).astype(int)
    df = df.reset_index()[['time','station_name','depth']].set_index(['station_name', 'time']).unstack('time', fill_value = np.nan)['depth']
    return df

def read_SCHISM_output(ds, coastal_hy_crosswalk):

    schism_nodes = coastal_hy_crosswalk
                    
    base_date = ds.time.attrs.get('base_date').split()
    base_date[3] = str(int(float(base_date[3])))
    base_date[4] = str(int(float(base_date[4])))
    base_date = "_".join(base_date)
    base_date = datetime.strptime(base_date, '%Y_%m_%d_%H_%M')

    #Retrieve elevations at specified nodes from 'schism_nodes' dictionary
    elevation_df = ds['elevation'].loc[:,list(schism_nodes.keys())].to_dataframe().reset_index().drop('nSCHISM_hgrid_node', axis=1)


    # if none of the nodes from crosswalk are present, return empty df 
    if (len(elevation_df)==0):

        df=pd.DataFrame()

    else:

        #Replace schism nodes with HYFeatures segment IDs
        df_length = len(elevation_df)
        dict_length = len(schism_nodes.values())
        n = df_length/dict_length
        elevation_df['link'] = list(schism_nodes.values())*int(n)

        #Retrieve depths at specified nodes from 'schism_nodes' dictionary
        depth_df = ds['depth'].loc[list(schism_nodes.keys())].to_dataframe().reset_index().drop('nSCHISM_hgrid_node', axis=1)

        #Replace schism nodes with HYFeatures segment IDs
        df_length = len(depth_df)
        dict_length = len(schism_nodes.values())
        n = df_length/dict_length
        depth_df['link'] = list(schism_nodes.values())*int(n)

        #Combine elevation and depth dataframes, and calculate water depth
        df = pd.merge(elevation_df, depth_df, on='link')
        df['waterdepth'] = df['elevation'] + df['depth']
        df = df.drop(['elevation','depth'], axis=1)

        #Replace 'time' columns of seconds with datetimes
        df['base_date'] = base_date
        df['time'] = pd.to_timedelta(df['time'],'s')
        df['time'] = df['base_date'] + df['time']

        #Drop columns we no longer need
        df = df.drop(['base_date'], axis=1)

        #Reformat dataframe so rows are locations and columns are timestamps
        df = df.set_index(['link','time']).unstack('time', fill_value = np.nan)['waterdepth']

    return df


def read_SCHISM_subset(ds, coastal_hy_crosswalk):

    schism_nodes = coastal_hy_crosswalk
                    
    ds2 = ds.drop_vars(["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y"])

    # crosswalk from node ID to node index (number)
    nl = ds2.node.values.tolist()
    idl = ds2.nodeID.values.tolist()
    idmap = dict(map(lambda i,j : (i,j) , idl,nl))
    
    # get indices for xarray dataset
    snKeys =  list(schism_nodes.keys())
    idSelect=[]
    idHySelect=[]
    for snKey in snKeys:
        if (snKey in idl):
            idSelect.append(idmap[snKey])
            idHySelect.append(schism_nodes[snKey])


    # if none of the nodes from crosswalk are present, return empty df 
    if (len(idSelect)==0):

        df=pd.DataFrame()

    else:

        #Retrieve elevations at specified nodes from 'schism_nodes' dictionary
        elevation_df = ds2['elevation'].loc[:,idSelect].to_dataframe().reset_index()

        #Replace schism nodes with HYFeatures segment IDs
        df_length = len(elevation_df)
        dict_length = len(idSelect)
        nTimes = int(df_length/dict_length)
        elevation_df['link'] = idHySelect*nTimes

        #Retrieve depths at specified nodes from 'schism_nodes' dictionary
        depth_df = ds2['depth'].loc[idSelect].to_dataframe().reset_index()

        #Replace schism nodes with HYFeatures segment IDs
        df_length = len(depth_df)
        dict_length = len(idSelect)
        nTimes = int(df_length/dict_length)
        depth_df['link'] = idHySelect*nTimes

        #Combine elevation and depth dataframes, and calculate water depth
        df = pd.merge(elevation_df, depth_df, on='link')
        df['waterdepth'] = df['elevation'] + df['depth']
        df = df.drop(['elevation','depth','node_x','node_y'], axis=1)

        #Reformat dataframe so rows are locations and columns are timestamps
        df = df.set_index(['link','time']).unstack('time', fill_value = np.nan)['waterdepth']

    return df


def read_coastal_output(filepath, coastal_hy_crosswalk):
    ds = xr.open_dataset(filepath)
    coastal_model_indicator = ds.attrs.get('institution', None)
    if coastal_model_indicator=='Deltares':
        df = read_DFlow_output(ds)
    else:
        if (len(coastal_hy_crosswalk)==0):
            # just create empty coastal dataframe if there is no hy crosswalk
            df = pd.DataFrame()
        else:
            if ("base_date" in ds.time.attrs.keys()):
                df = read_SCHISM_output(ds, coastal_hy_crosswalk)
            else:
                df = read_SCHISM_subset(ds, coastal_hy_crosswalk)

    return df
