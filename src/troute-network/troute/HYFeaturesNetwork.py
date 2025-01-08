from .AbstractNetwork import AbstractNetwork
import pandas as pd
import numpy as np
import geopandas as gpd
import time
import json
from pathlib import Path
import pyarrow.parquet as pq
from itertools import chain
from joblib import delayed, Parallel
from collections import defaultdict
import xarray as xr
from datetime import datetime
from pprint import pformat
import os
import fiona
import troute.nhd_io as nhd_io #FIXME
from troute.nhd_network import reverse_dict, extract_connections, reverse_network, reachable
from .rfc_lake_gage_crosswalk import get_rfc_lake_gage_crosswalk, get_great_lakes_climatology
import re
__verbose__ = False
__showtiming__ = False

def find_layer_name(layers, pattern):
    """
    Find a layer name in the list of layers that matches the regex pattern.
    """
    for layer in layers:
        if re.search(pattern, layer, re.IGNORECASE):
            return layer
    return None

def read_geopkg(file_path, compute_parameters, waterbody_parameters, output_parameters, cpu_pool):
    # Retrieve available layers from the GeoPackage
    available_layers = fiona.listlayers(file_path)

    # patterns for the layers we want to find
    layer_patterns = {
        'flowpaths': r'flow[-_]?paths?|flow[-_]?lines?',
        'flowpath_attributes': r'flow[-_]?path[-_]?attributes?|flow[-_]?line[-_]?attributes?',
        'lakes': r'lakes?',
        'nexus': r'nexus?',
        'network': r'network'
    }

    # Match available layers to the patterns
    matched_layers = {key: find_layer_name(available_layers, pattern) 
                      for key, pattern in layer_patterns.items()}
    
    layers_to_read = ['flowpaths', 'flowpath_attributes']
    
    if waterbody_parameters.get('break_network_at_waterbodies', False):
        layers_to_read.extend(['lakes', 'nexus'])

    data_assimilation_parameters = compute_parameters.get('data_assimilation_parameters', {})
    if any([
        data_assimilation_parameters.get('streamflow_da', {}).get('streamflow_nudging', False),
        data_assimilation_parameters.get('reservoir_da', {}).get('reservoir_persistence_usgs', False),
        data_assimilation_parameters.get('reservoir_da', {}).get('reservoir_persistence_usace', False),
        data_assimilation_parameters.get('reservoir_da', {}).get('reservoir_rfc_da', {}).get('reservoir_rfc_forecasts', False)
    ]):
        layers_to_read.append('network')

    hybrid_parameters = compute_parameters.get('hybrid_parameters', {})
    if hybrid_parameters.get('run_hybrid_routing', False) and 'nexus' not in layers_to_read:
        layers_to_read.append('nexus')

    # If nexus points or POIs are specified in output mask, read nexus table
    if output_parameters.get('stream_output'):
        mask_keys = output_parameters.get('stream_output',{}).get('mask',{}).keys()
        if ('nex' in mask_keys) or ('poi' in mask_keys) & ('nexus' not in layers_to_read):
            layers_to_read.append('nexus')
            
    # Function that read a layer from the geopackage
    def read_layer(layer_name):
        if layer_name:
            try:
                return gpd.read_file(file_path, layer=layer_name)
            except Exception as e:
                print(f"Error reading {layer_name}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
       
    # Retrieve geopackage information using matched layer names
    if cpu_pool > 1:
        with Parallel(n_jobs=min(cpu_pool, len(layers_to_read))) as parallel:
            gpkg_list = parallel(delayed(read_layer)(matched_layers[layer]) for layer in layers_to_read)
        
        table_dict = {layers_to_read[i]: gpkg_list[i] for i in range(len(layers_to_read))}
    else:
        table_dict = {layer: read_layer(matched_layers[layer]) for layer in layers_to_read}
    
    # Handle different key column names between flowpaths and flowpath_attributes
    flowpaths_df = table_dict.get('flowpaths', pd.DataFrame())
    flowpath_attributes_df = table_dict.get('flowpath_attributes', pd.DataFrame())

    # Check if 'link' column exists; drop existing 'id' col; rename 'link' to 'id'
    if 'link' in flowpath_attributes_df.columns:
        # In HF 2.2, a 'link' field was introduced. The field is identical to
        # previous version's 'id' field, but it preferred moving forwards.
        flowpath_attributes_df.drop(columns=['id'], errors='ignore', inplace=True)
        flowpath_attributes_df.rename(columns={'link': 'id'}, inplace=True) 
     
    # NOTE: aaraney: `flowpaths_df` and `flowpath_attributes_df` can share
    # column names but this is not accounted for elsewhere. im not sure if it
    # is okay to assume that if the left and right df have the same `id` field
    # that the other shared columns will match and thus it is safe to set, for
    # example, the left suffix to "".
    # Merge flowpaths and flowpath_attributes 
    flowpaths = pd.merge(
        flowpaths_df, 
        flowpath_attributes_df, 
        on='id', 
        how='inner',
        # NOTE: aaraney: not sure if this is safe
        suffixes=("", "_flowpath_attributes"),
    )

    lakes = table_dict.get('lakes', pd.DataFrame())
    network = table_dict.get('network', pd.DataFrame())
    nexus = table_dict.get('nexus', pd.DataFrame())

    return flowpaths, lakes, network, nexus

def read_json(file_path, edge_list):
    dfs = []
    with open(edge_list) as edge_file:
        edge_data = json.load(edge_file)
        edge_map = {}
        wb_id, toid = edge_data[0].keys()
        for id_dict in edge_data:
            edge_map[ id_dict[wb_id] ] = id_dict[toid]
        with open(file_path) as data_file:
            json_data = json.load(data_file)  
            for key_wb, value_params in json_data.items():
                df = pd.json_normalize(value_params)
                df[wb_id] = key_wb
                df[toid] = edge_map[key_wb]
                dfs.append(df)
        df_main = pd.concat(dfs, ignore_index=True)

    return df_main

def read_geojson(file_path):
    flowpaths = gpd.read_file(file_path)
    return flowpaths

def numeric_id(flowpath):
    id = flowpath['key'].split('-')[-1]
    toid = flowpath['downstream'].split('-')[-1]
    flowpath['key'] = int(float(id))
    flowpath['downstream'] = int(float(toid))
    return flowpath

def read_ngen_waterbody_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """
    Reads .gpkg or lake.json file and prepares a dataframe, filtered
    to the relevant reservoirs, to provide the parameters
    for level-pool reservoir computation.
    """
    def node_key_func(x):
        return int( x.split('-')[-1] )
    if Path(parm_file).suffix=='.gpkg':
        df = gpd.read_file(parm_file, layer='lakes')

        df = (
            df.drop(['id','toid','hl_id','hl_reference','hl_uri','geometry'], axis=1)
            .rename(columns={'hl_link': 'lake_id'})
            )
        df['lake_id'] = df.lake_id.astype(float).astype(int)
        df = df.set_index('lake_id').drop_duplicates().sort_index()
    elif Path(parm_file).suffix=='.json':
        df = pd.read_json(parm_file, orient="index")
        df.index = df.index.map(node_key_func)
        df.index.name = lake_index_field

    if lake_id_mask:
        df = df.loc[lake_id_mask]
    return df

def read_ngen_waterbody_type_df(parm_file, lake_index_field="wb-id", lake_id_mask=None):
    """
    """
    #FIXME: this function is likely not correct. Unclear how we will get 
    # reservoir type from the gpkg files. Information should be in 'crosswalk'
    # layer, but as of now (Nov 22, 2022) there doesn't seem to be a differentiation
    # between USGS reservoirs, USACE reservoirs, or RFC reservoirs...
    def node_key_func(x):
        return int( x.split('-')[-1] )
    
    if Path(parm_file).suffix=='.gpkg':
        df = gpd.read_file(parm_file, layer="crosswalk").set_index('id')
    elif Path(parm_file).suffix=='.json':
        df = pd.read_json(parm_file, orient="index")

    df.index = df.index.map(node_key_func)
    df.index.name = lake_index_field
    if lake_id_mask:
        df = df.loc[lake_id_mask]
        
    return df

def read_geo_file(supernetwork_parameters, waterbody_parameters, compute_parameters, output_parameters, cpu_pool):
        
    geo_file_path = supernetwork_parameters["geo_file_path"]
    flowpaths = lakes = network = pd.DataFrame()
    
    file_type = Path(geo_file_path).suffix
    if(file_type=='.gpkg'):        
        flowpaths, lakes, network, nexus = read_geopkg(geo_file_path,
                                                       compute_parameters,
                                                       waterbody_parameters,
                                                       output_parameters,
                                                       cpu_pool)
    elif(file_type == '.json'):
        edge_list = supernetwork_parameters['flowpath_edge_list']
        flowpaths = read_json(geo_file_path, edge_list)
    elif(file_type=='.geojson'):
        flowpaths = read_geojson(geo_file_path)
    else:
        raise RuntimeError("Unsupported file type: {}".format(file_type))
    
    return flowpaths, lakes, network, nexus

def load_bmi_data(value_dict, bmi_parameters,): 
    # Get the column names that we need from each table of the geopackage
    flowpath_columns = bmi_parameters.get('flowpath_columns')
    attributes_columns = bmi_parameters.get('attributes_columns')
    lakes_columns = bmi_parameters.get('waterbody_columns')
    network_columns = bmi_parameters.get('network_columns')

    # Create dataframes with the relevent columns
    flowpaths = pd.DataFrame(data=None, columns=flowpath_columns)
    for col in flowpath_columns:
        flowpaths[col] = value_dict[col]

    flowpath_attributes = pd.DataFrame(data=None, columns=attributes_columns)
    for col in attributes_columns:
        flowpath_attributes[col] = value_dict[col]
    flowpath_attributes = flowpath_attributes.rename(columns={'attributes_id': 'id'})

    lakes = pd.DataFrame(data=None, columns=lakes_columns)
    for col in lakes_columns:
        lakes[col] = value_dict[col]

    network = pd.DataFrame(data=None, columns=network_columns)
    for col in network_columns:
        network[col] = value_dict[col]
    network = network.rename(columns={'network_id': 'id'})

    # Merge the two flowpath tables into one
    flowpaths = pd.merge(flowpaths, flowpath_attributes, on='id')

    return flowpaths, lakes, network


class HYFeaturesNetwork(AbstractNetwork):
    """
    
    """
    __slots__ = ["_upstream_terminal", "_nexus_latlon", "_duplicate_ids_df",]

    def __init__(self, 
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
                 bmi_parameters={},):
        """
        
        """
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
            print("creating supernetwork connections set")
        if self.showtiming:
            start_time = time.time()
        
        #------------------------------------------------
        # Load hydrofabric information
        #------------------------------------------------
        if self.preprocessing_parameters.get('use_preprocessed_data', False):
            self.read_preprocessed_data()
        else:
            #FIXME: Temporary solution, from_files should only be from command line.
            # Update this once ngen framework is capable of providing this info via BMI.
            from_files_copy = from_files
            if not from_files_copy:
                from_files=True
            if from_files:
                flowpaths, lakes, network, nexus = read_geo_file(
                    self.supernetwork_parameters,
                    self.waterbody_parameters,
                    self.compute_parameters,
                    self.output_parameters,
                    self.compute_parameters.get('cpu_pool', 1)
                )
            else:
                flowpaths, lakes, network = load_bmi_data(
                    value_dict, 
                    bmi_parameters,
                    )
            #FIXME: See FIXME above.
            if not from_files_copy:
                from_files=False

            # Preprocess network objects
            self.preprocess_network(flowpaths, nexus)
            
            self.crosswalk_nex_flowpath_poi(flowpaths, nexus)

            # Preprocess waterbody objects
            self.preprocess_waterbodies(lakes, nexus)

            # Preprocess data assimilation objects #TODO: Move to DataAssimilation.py?
            self.preprocess_data_assimilation(flowpaths)
        
            if self.preprocessing_parameters.get('preprocess_output_folder', None):
                self.write_preprocessed_data()

                if self.preprocessing_parameters.get('preprocess_only', False):
                    #TODO: Add LOG message here...
                    quit()

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
        return (
            rows.loc[rows[target_col] != waterbody_null, target_col].astype("int").to_dict()
        )

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
    def waterbody_null(self):
        return np.nan #pd.NA
    
    
    def preprocess_network(self, flowpaths, nexus):
        self._dataframe = flowpaths
        
        cols = self.supernetwork_parameters.get('columns', None)
        if cols:
            col_idx = list(set(cols.values()).intersection(set(self.dataframe.columns)))
            self._dataframe = self.dataframe[col_idx]
            # Rename parameter columns to standard names: from route-link names
            #        key: "link"
            #        downstream: "to"
            #        dx: "Length"
            #        n: "n"  # TODO: rename to `manningn`
            #        ncc: "nCC"  # TODO: rename to `mannningncc`
            #        s0: "So"  # TODO: rename to `bedslope`
            #        bw: "BtmWdth"  # TODO: rename to `bottomwidth`
            #        waterbody: "NHDWaterbodyComID"
            #        gages: "gages"
            #        tw: "TopWdth"  # TODO: rename to `topwidth`
            #        twcc: "TopWdthCC"  # TODO: rename to `topwidthcc`
            #        alt: "alt"
            #        musk: "MusK"
            #        musx: "MusX"
            #        cs: "ChSlp"  # TODO: rename to `sideslope`
            self._dataframe = self.dataframe.rename(columns=reverse_dict(cols))
        
        # Don't need the string prefix anymore, drop it
        mask = ~ self.dataframe['downstream'].str.startswith("tnx") 
        self._dataframe = self.dataframe.apply(numeric_id, axis=1)
        
        # make the flowpath linkage, ignore the terminal nexus
        self._flowpath_dict = dict(zip(self.dataframe.loc[mask].downstream, self.dataframe.loc[mask].key))
        
        self._dataframe.set_index("key", inplace=True)
        self._dataframe = self.dataframe.sort_index()
        
        # **********  need to be included in flowpath_attributes  *************
        if 'alt' not in self.dataframe.columns:
            self._dataframe['alt'] = 1.0 #FIXME get the right value for this... 

        # Drop 'gages' column if it is present
        if 'gages' in self.dataframe:
            self._dataframe = self.dataframe.drop('gages', axis=1)
        # numeric code used to indicate network terminal segments
        terminal_code = self.supernetwork_parameters.get("terminal_code", 0)

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

        #This is NEARLY redundant to the self.terminal_codes property, but in this case
        #we actually need the mapping of what is upstream of that terminal node as well.
        #we also only want terminals that actually exist based on definition, not user input
        terminal_mask = ~self._dataframe["downstream"].isin(self._dataframe.index)
        terminal = self._dataframe.loc[ terminal_mask ]["downstream"]
        self._upstream_terminal = dict()
        for key, value in terminal.items():
            self._upstream_terminal.setdefault(value, set()).add(key)

        # build connections dictionary
        self._connections = extract_connections(
            self.dataframe, "downstream", terminal_codes=self.terminal_codes
        )
        
        # Store a dataframe containing info about nexus points. This will be reprojected to lat/lon
        # and filtered for only diffusive domain tailwaters in AbstractNetwork.py.
        # Location information will be used to advertise tailwater locations of diffusive domains 
        # to the model engine/coastal models
        self._nexus_latlon = nexus

    def crosswalk_nex_flowpath_poi(self, flowpaths, nexus):
        
        mask_flowpaths = flowpaths['toid'].str.startswith(('nex-', 'tnx-'))
        filtered_flowpaths = flowpaths[mask_flowpaths]
        self._nexus_dict = filtered_flowpaths.groupby('toid')['id'].apply(list).to_dict()  ##{id: toid}
        if 'poi_id' in nexus.columns:
            # self._poi_nex_dict = nexus.groupby('poi_id')['id'].apply(list).to_dict()
            self._poi_nex_df = nexus[~nexus['poi_id'].isna()][['id','poi_id']]
        else:
            self._poi_nex_df = pd.DataFrame()

    def preprocess_waterbodies(self, lakes, nexus):
        # If waterbodies are being simulated, create waterbody dataframes and dictionaries
        if not lakes.empty:
            #NOTE: the below worked for hydrofabric v2.2 for Hawaii, Alaska, and PRVI. But does
            # not work for CONUS.
            # self._waterbody_df = (
            #     lakes[['hl_link','ifd','LkArea','LkMxE','OrificeA',
            #            'OrificeC','OrificeE','WeirC','WeirE','WeirL','id']]
            #     .rename(columns={'hl_link': 'lake_id'})
            #     )
            
            #NOTE: This works for hydrofabric v2.2 CONUS.
            self._waterbody_df = (
                lakes[['lake_id','ifd','LkArea','LkMxE','OrificeA',
                       'OrificeC','OrificeE','WeirC','WeirE','WeirL']]
                )
            
            #NOTE: the below worked for hydrofabric v2.2 for Hawaii, Alaska, and PRVI. But does
            # not work for CONUS.
            # id = self.waterbody_dataframe['id'].str.split('-', expand=True).iloc[:,1]
            # self._waterbody_df['id'] = id
            # self._waterbody_df['id'] = self._waterbody_df.id.astype(float).astype(int)
            
            #NOTE: This works for hydrofabric v2.2 CONUS.
            self._waterbody_df['lake_id'] = self.waterbody_dataframe.lake_id.astype(float).astype(int)
            self._waterbody_df = self.waterbody_dataframe.set_index('lake_id').drop_duplicates().sort_index()
            
            # Drop any waterbodies that do not have parameters
            self._waterbody_df = self.waterbody_dataframe.dropna()
            
            # Check if there are any lake_ids that are also segment_ids. If so, add a large value
            # to the lake_ids:
            duplicate_ids = list(set(self.waterbody_dataframe.index).intersection(set(self.dataframe.index)))
            self._duplicate_ids_df = pd.DataFrame({
                'lake_id': duplicate_ids,
                'synthetic_ids': [int(id + 9.99e11) for id in duplicate_ids]
            })
            update_dict = dict(self._duplicate_ids_df[['lake_id','synthetic_ids']].values)
            
            tmp_wbody_conn = self.dataframe[['waterbody']].dropna()
            tmp_wbody_conn = (
                tmp_wbody_conn['waterbody']
                .str.split(',',expand=True)
                .reset_index()
                .melt(id_vars='key')
                .drop('variable', axis=1)
                .dropna()
                .astype(int)
                )
            tmp_wbody_conn = tmp_wbody_conn[tmp_wbody_conn['value'].isin(self.waterbody_dataframe.index)]
            self._dataframe = (
                self.dataframe
                .reset_index()
                .merge(tmp_wbody_conn, how='left', on='key')
                .drop('waterbody', axis=1)
                .rename(columns={'value': 'waterbody'})
                .set_index('key')
            )

            self._waterbody_df = self.waterbody_dataframe.rename(index=update_dict).sort_index()
            self._dataframe = self.dataframe.replace({'waterbody': update_dict})
            
            #FIXME temp solution for missing waterbody info in hydrofabric
            self.bandaid()
            
            wbody_conn = self.dataframe[['waterbody']].dropna().astype(int).reset_index()
            
            self._waterbody_connections = (
                wbody_conn[wbody_conn['waterbody'].isin(self.waterbody_dataframe.index)]
                .set_index('key')['waterbody']
                .to_dict()
                )
            
            # if waterbodies are being simulated, adjust the connections graph so that 
            # waterbodies are collapsed to single nodes. Also, build a mapping between 
            # waterbody outlet segments and lake ids
            break_network_at_waterbodies = self.waterbody_parameters.get("break_network_at_waterbodies", False)
            if break_network_at_waterbodies:
                self._connections, self._link_lake_crosswalk = replace_waterbodies_connections(
                    self.connections, self.waterbody_connections
                )
            else:
                self._link_lake_crosswalk = None
            
            # Add lat, lon, and crs columns for LAKEOUT files:
            lakeout = self.output_parameters.get("lakeout_output", None)
            if lakeout:
                lat_lon_crs = lakes[['hl_link','hl_reference','geometry']].rename(columns={'hl_link': 'lake_id'})
                lat_lon_crs = lat_lon_crs[lat_lon_crs['hl_reference']=='WBOut']
                lat_lon_crs['lake_id'] = lat_lon_crs.lake_id.astype(float).astype(int)
                lat_lon_crs = lat_lon_crs.set_index('lake_id').drop_duplicates().sort_index()
                lat_lon_crs = lat_lon_crs[lat_lon_crs.index.isin(self.waterbody_dataframe.index)]
                lat_lon_crs = lat_lon_crs.to_crs(crs=4326)
                lat_lon_crs['lon'] = lat_lon_crs.geometry.x
                lat_lon_crs['lat'] = lat_lon_crs.geometry.y
                lat_lon_crs['crs'] = str(lat_lon_crs.crs)
                lat_lon_crs = lat_lon_crs[['lon','lat','crs']]

                self._waterbody_df = self.waterbody_dataframe.join(lat_lon_crs)
            else:
                self._waterbody_df['lon'] = np.nan
                self._waterbody_df['lat'] = np.nan
                self._waterbody_df['crs'] = np.nan
                
            # Add the Great Lakes to the connections dictionary and waterbody dataframe
            #TODO: Update for hydrofabric v2.2.
            # nexus['WBOut_id'] = nexus['hl_uri'].str.extract(r'WBOut-(\d+)').astype(float)
            # great_lakes_df = nexus[nexus['WBOut_id'].isin([4800002,4800004,4800006,4800007])][['WBOut_id','toid']]
            
            #NOTE: Hard-coding the Great Lakes outflow toids. Ideally, these will be provided
            # by the hydrofabric, but current version does not support his (shorvath - 09/25/2024)
            great_lakes_df = pd.DataFrame(
                {'WBOut_id': [4800002, 4800004, 4800006, 4800007],
                 'toid': ['wb-660226', 'wb-653294', 'wb-670032', 'wb-678009']}
            )
            
            if not great_lakes_df.empty:
                great_lakes_df['toid'] = great_lakes_df['toid'].str.extract(r'wb-(\d+)').astype(float)
                great_lakes_df = great_lakes_df.astype(int)
                great_lakes_df['toid'] = great_lakes_df["toid"].apply(lambda x: [x])
                gl_dict = great_lakes_df.set_index('WBOut_id')['toid'].to_dict()
                self._connections.update(gl_dict)
                
                gl_wbody_df = pd.DataFrame(
                    data=np.ones([len(gl_dict), self.waterbody_dataframe.shape[1]]),
                    index=gl_dict.keys(), 
                    columns=self.waterbody_dataframe.columns
                    )
                gl_wbody_df.index.name = self.waterbody_dataframe.index.name
                
                self._waterbody_df = pd.concat(
                    [
                        self.waterbody_dataframe,
                        gl_wbody_df
                    ]
                ).sort_index()
                
                self._gl_climatology_df = get_great_lakes_climatology()
                
            else:
                gl_dict = {}
                self._gl_climatology_df = pd.DataFrame()
            
            self._waterbody_types_df = pd.DataFrame(
                data = 1, 
                index = self.waterbody_dataframe.index, 
                columns = ['reservoir_type']).sort_index()
            
            # Add Great Lakes waterbody type (6)
            self._waterbody_types_df.loc[gl_dict.keys(),'reservoir_type'] = 6
             
            self._waterbody_type_specified = True
            
        else:
            self.data_assimilation_parameters['reservoir_da'] = {
                'reservoir_persistence_da': {'reservoir_persistence_usgs': False,
                                             'reservoir_persistence_usace': False,
                                             'reservoir_persistence_canada': False,},
                'reservoir_rfc_da': {'reservoir_rfc_forecasts': False}
            }
            # self.data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usgs'] = False
            # self.data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usace'] = False
            # self.data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_canada'] = False
            # self.data_assimilation_parameters['reservoir_da']['reservoir_rfc_da']['reservoir_rfc_forecasts'] = False
            self.waterbody_parameters['break_network_at_waterbodies'] = False

            self._waterbody_df = pd.DataFrame()
            self._waterbody_types_df = pd.DataFrame()
            self._waterbody_connections = {}
            self._waterbody_type_specified = False
            self._link_lake_crosswalk = None
            self._duplicate_ids_df = pd.DataFrame()
            self._gl_climatology_df = pd.DataFrame()

        if 'waterbody' in self.dataframe.columns:
            self._dataframe = self.dataframe.drop('waterbody', axis=1)
        
        self._dataframe = self.dataframe.drop_duplicates()

    def preprocess_data_assimilation(self, flowpaths):
        gages_df = flowpaths[~flowpaths['gage'].isna()]
        if not gages_df.empty:
            gages_df = gages_df[['id','gage']]
            gages_df['id'] = gages_df['id'].str.split('-',expand=True).loc[:,1].astype(float).astype(int)
            gages_df.set_index('id', inplace=True)
            '''
            gages_df = network[['id','hl_uri','hydroseq']].drop_duplicates()
            # clear out missing values
            gages_df = gages_df[~gages_df['hl_uri'].isnull()]
            gages_df = gages_df[~gages_df['hydroseq'].isnull()]
            # make 'id' an integer
            gages_df['id'] = gages_df['id'].str.split('-',expand=True).loc[:,1].astype(float).astype(int)
            # split the hl_uri column into type and value
            gages_df[['type','value']] = gages_df.hl_uri.str.split('-',expand=True,n=1)
            # filter for 'Gages' only
            gages_df = gages_df[gages_df['type'].isin(['Gages','NID'])]
            # Some IDs have multiple gages associated with them. This will expand the dataframe so
            # there is a unique row per gage ID. Also adds lake ids to the dataframe for creating 
            # lake-gage crosswalk dataframes.
            gages_df = gages_df[['id','value','hydroseq']]
            gages_df['value'] = gages_df.value.str.split(' ')
            gages_df = gages_df.explode(column='value').set_index('id').join(
                pd.DataFrame().from_dict(self.waterbody_connections,orient='index',columns=['lake_id'])
                )
            # transform dataframe into a dictionary where key is segment ID and value is gage ID
            usgs_ind = gages_df.value.str.isnumeric() #usgs gages used for streamflow DA
            # Use hydroseq information to determine furthest downstream gage when multiple are present.
            idx_id = gages_df.index.name
            if not idx_id:
                idx_id = 'index'
            self._gages = (
                gages_df.loc[usgs_ind].reset_index()
                .sort_values('hydroseq').drop_duplicates(['value'],keep='last')
                .set_index(idx_id)[['value']].rename(columns={'value': 'gages'})
                .rename_axis(None, axis=0).to_dict()
            )
            '''
            
            # transform dataframe into a dictionary where key is segment ID and value is gage ID
            usgs_ind = gages_df.gage.str.isnumeric() #usgs gages used for streamflow DA
            # Use hydroseq information to determine furthest downstream gage when multiple are present.
            idx_id = gages_df.index.name
            if not idx_id:
                idx_id = 'index'
            
            self._gages = (
                gages_df.loc[usgs_ind]
                .rename(columns={'gage': 'gages'})
                .rename_axis(None, axis=0).to_dict()
            )
            
            #FIXME: temporary solution, add canadian gage crosswalk dataframe. This should come from
            # the hydrofabric.
            self._canadian_gage_link_df = pd.DataFrame(columns=['gages','link']).set_index('link')
            
            if 'lake_id' in gages_df.columns:
                # Find furthest downstream gage and create our lake_gage_df to make crosswalk dataframes.
                lake_gage_hydroseq_df = gages_df[~gages_df['lake_id'].isnull()][['lake_id', 'value', 'hydroseq']].rename(columns={'value': 'gages'})
                lake_gage_hydroseq_df['lake_id'] = lake_gage_hydroseq_df['lake_id'].astype(int)
                lake_gage_df = lake_gage_hydroseq_df[['lake_id','gages']].drop_duplicates()
                lake_gage_hydroseq_df = lake_gage_hydroseq_df.groupby(['lake_id','gages']).max('hydroseq').reset_index().set_index('lake_id')

                #FIXME: temporary solution, handles USGS and USACE reservoirs. Need to update for
                # RFC reservoirs...
                #NOTE: In the event a lake ID has multiple gages, this also finds the gage furthest 
                # downstream (based on hydroseq) separately for USGS and USACE crosswalks. 
                usgs_ind = lake_gage_df.gages.str.isnumeric()
                self._usgs_lake_gage_crosswalk = (
                    lake_gage_df.loc[usgs_ind].rename(columns={'lake_id': 'usgs_lake_id', 'gages': 'usgs_gage_id'}).
                    set_index('usgs_lake_id').
                    merge(lake_gage_hydroseq_df.
                        rename_axis('usgs_lake_id').
                        rename(columns={'gages': 'usgs_gage_id'}), on=['usgs_lake_id','usgs_gage_id']).
                    sort_values(['usgs_gage_id','hydroseq']).groupby('usgs_lake_id').
                    last().
                    drop('hydroseq', axis=1)
                )

                self._usace_lake_gage_crosswalk =  (
                    lake_gage_df.loc[~usgs_ind].rename(columns={'lake_id': 'usace_lake_id', 'gages': 'usace_gage_id'}).
                    set_index('usace_lake_id').
                    merge(lake_gage_hydroseq_df.
                        rename_axis('usace_lake_id').
                        rename(columns={'gages': 'usace_gage_id'}), on=['usace_lake_id','usace_gage_id']).
                    sort_values(['usace_gage_id','hydroseq']).groupby('usace_lake_id').
                    last().
                    drop('hydroseq', axis=1)
                )
            else:
                self._usgs_lake_gage_crosswalk = self._usace_lake_gage_crosswalk = pd.DataFrame()
            
            # Set waterbody types if DA is turned on:
            usgs_da = self.data_assimilation_parameters.get('reservoir_da',{}).get('reservoir_persistence_da',{}).get('reservoir_persistence_usgs',False)
            usace_da = self.data_assimilation_parameters.get('reservoir_da',{}).get('reservoir_persistence_da',{}).get('reservoir_persistence_usace',False)
            rfc_da = self.data_assimilation_parameters.get('reservoir_da',{}).get('reservoir_rfc_da',{}).get('reservoir_rfc_forecasts',False)
            #NOTE: The order here matters. Some waterbody IDs have both a USGS gage designation and
            # a NID ID used for USACE gages. It seems the USGS gages should take precedent (based on
            # gages in timeslice files), so setting type 2 reservoirs second should overwrite type 3 
            # designations
            #FIXME: Related to FIXME above, but we should re-think how to handle waterbody_types...
            if usace_da:
                self._waterbody_types_df.loc[self._usace_lake_gage_crosswalk.index,'reservoir_type'] = 3
            if usgs_da:
                self._waterbody_types_df.loc[self._usgs_lake_gage_crosswalk.index,'reservoir_type'] = 2
            if rfc_da:
                #FIXME: Temporary fix, load predefined rfc lake gage crosswalk info for rfc reservoirs.
                # Replace relevant waterbody_types as type 4.
                rfc_lake_gage_crosswalk = get_rfc_lake_gage_crosswalk().reset_index()
                self._rfc_lake_gage_crosswalk = rfc_lake_gage_crosswalk[rfc_lake_gage_crosswalk['rfc_lake_id'].isin(self.waterbody_dataframe.index)].set_index('rfc_lake_id')
                self._waterbody_types_df.loc[self._rfc_lake_gage_crosswalk.index,'reservoir_type'] = 4
            else:
                self._rfc_lake_gage_crosswalk = pd.DataFrame()
            
        else:
            self._gages = {}
            self._usgs_lake_gage_crosswalk = pd.DataFrame()
            self._usace_lake_gage_crosswalk = pd.DataFrame()
            self._rfc_lake_gage_crosswalk = pd.DataFrame()
    
    def build_qlateral_array(self, run,):
        
        # TODO: set default/optional arguments
        qts_subdivisions = run.get("qts_subdivisions", 1)
        nts = run.get("nts", 1)
        qlat_input_folder = run.get("qlat_input_folder", None)
        qlat_input_file = run.get("qlat_input_file", None)

        if qlat_input_folder:
            qlat_input_folder = Path(qlat_input_folder)
            if "qlat_files" in run:
                qlat_files = run.get("qlat_files")
                qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
            elif "qlat_file_pattern_filter" in run:
                qlat_file_pattern_filter = run.get(
                    "qlat_file_pattern_filter", "*CHRT_OUT*"
                )
                qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))
            
            dfs=[]
            
            #FIXME Temporary solution to allow t-route to use ngen nex-* output files as forcing files
            # This capability should be here, but we need to think through how to handle all of this 
            # data in memory for large domains and many timesteps... - shorvath, Feb 28, 2024
            qlat_file_pattern_filter = self.forcing_parameters.get("qlat_file_pattern_filter", None)
            if qlat_file_pattern_filter=="nex-*":
                for f in qlat_files:
                    df = pd.read_csv(f, names=['timestamp', 'qlat'], index_col=[0])
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d%H%M')
                    df = df.set_index('timestamp')
                    df = df.T
                    df.index = [int(os.path.basename(f).split('-')[1].split('_')[0])]
                    df = df.rename_axis(None, axis=1)
                    df.index.name = 'feature_id'
                    dfs.append(df)
                
                # lateral flows [m^3/s] are stored at NEXUS points with NEXUS ids
                nexuses_lateralflows_df = pd.concat(dfs, axis=0) 
            else:
                for f in qlat_files:
                    df = read_file(f)
                    df['feature_id'] = df['feature_id'].map(lambda x: int(str(x).removeprefix('nex-')) if str(x).startswith('nex') else int(x))
                    assert df[
                        "feature_id"
                    ].is_unique, f"'feature_id's must be unique. '{f!s}' contains duplicate 'feature_id's: {pformat(df.loc[df['feature_id'].duplicated(), 'feature_id'].to_list())}"
                    df = df.set_index('feature_id')
                    dfs.append(df)
            
                # lateral flows [m^3/s] are stored at NEXUS points with NEXUS ids
                nexuses_lateralflows_df = pd.concat(dfs, axis=1) 
            
            # Take flowpath ids entering NEXUS and replace NEXUS ids by the upstream flowpath ids
            qlats_df = nexuses_lateralflows_df.rename(index=self.downstream_flowpath_dict)
            qlats_df = qlats_df[qlats_df.index.isin(self.segment_index)]

            '''
            #For a terminal nexus, we want to include the lateral flow from the catchment contributing to that nexus
            #one way to do that is to cheat and put that lateral flow at the upstream...this is probably the simplest way
            #right now.  The other is to create a virtual channel segment downstream to "route" i.e accumulate into
            #but it isn't clear right now how to do that with flow/velocity/depth requirements
            #find the terminal nodes
            for tnx, test_up in self._upstream_terminal.items():
                #first need to ensure there is an upstream location to dump to
                pdb.set_trace()
                for nex in test_up:
                    try:
                        #FIXME if multiple upstreams exist in this case then a choice is to be made as to which it goes into
                        #some cases the choice is easy cause the upstream doesn't exist, but in others, it may not be so simple
                        #in such cases where multiple valid upstream nexuses exist, perhaps the mainstem should be used?
                        pdb.set_trace()
                        qlats_df.loc[up] += nexuses_lateralflows_df.loc[tnx]
                        break #flow added, don't add it again!
                    except KeyError:
                        #this upstream doesn't actually exist on the network (maybe it is a headwater?)
                        #or perhaps the output file doesnt exist?  If this is the case, this isn't a good trap
                        #but for now, add the flow to a known good nexus upstream of the terminal
                        continue
                    #TODO what happens if can't put the qlat anywhere?  Right now this silently ignores the issue...
                qlats_df.drop(tnx, inplace=True)
            '''

            # The segment_index has the full network set of segments/flowpaths. 
            # Whereas the set of flowpaths that are downstream of nexuses is a 
            # subset of the segment_index. Therefore, all of the segments/flowpaths
            # that are not accounted for in the set of flowpaths downstream of
            # nexuses need to be added to the qlateral dataframe and padded with
            # zeros.
            all_df = pd.DataFrame( np.zeros( (len(self.segment_index), len(qlats_df.columns)) ), index=self.segment_index,
                columns=qlats_df.columns )
            all_df.loc[ qlats_df.index ] = qlats_df
            qlats_df = all_df.sort_index()

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

    ######################################################################
    #FIXME Temporary solution to hydrofabric issues.
    def bandaid(self,):
        
        # Identify waterbody IDs that have problematic data. There are underlying stream 
        # segments that should be referenced to the waterbody ID, but are not. This causes
        # our connections dictionary to have multiple downstream segments for waterbodies which
        # is not allowed:
        conn_df = self.dataframe.reset_index()[['key', 'downstream']]
        lake_id = self.waterbody_dataframe.index.unique()

        wbody_conn_df = self.dataframe['waterbody'].dropna().astype(int).reset_index()
        wbody_conn_df = wbody_conn_df[wbody_conn_df['waterbody'].isin(lake_id)]
        
        conn_df2 = (
            conn_df
            .merge(wbody_conn_df, on='key', how='left')
            .assign(key=lambda x: x['waterbody'].fillna(x['key']))
            .drop('waterbody', axis=1)
            .merge(wbody_conn_df.rename(columns={'key': 'downstream'}),
                   on='downstream', how='left')
            .assign(downstream=lambda x: x['waterbody'].fillna(x['downstream']))
            .drop('waterbody', axis=1)
            .drop_duplicates()
            .query('key != downstream')
            .astype(int)
        )
        
        # Find missing segments
        bad_lake_ids = conn_df2.loc[conn_df2.duplicated(subset=['key'])].key.unique()
        # Drop waterbodies that are problematic. Instead t-route will simply treat them as
        # flowpaths and run MC routing.
        self._waterbody_df = self.waterbody_dataframe.drop(bad_lake_ids)

        #This chunk replaces waterbody_id 1711354 with 1710676. I don't know where the 
        #former came from, but the latter is listed in the flowpath_attributes table
        #and exists in NWMv2.1 LAKEPARM file. See hydrofabric github issue 16:
        #https://github.com/NOAA-OWP/hydrofabric/issues/16
        self._dataframe['waterbody'] = self._dataframe['waterbody'].replace('1711354','1710676')
        self._waterbody_df.rename(index={1711354: 1710676}, inplace=True)
    #######################################################################

    def write_preprocessed_data(self,):
        #LOG.debug("saving preprocessed network data to disk for future use")
        # todo: consider a better default than None
        destination_folder = self.preprocessing_parameters.get('preprocess_output_folder', None)
        if destination_folder:

            output_filename = self.preprocessing_parameters.get(
                'preprocess_output_filename', 
                'preprocess_output'
            )

        outputs = {
            'dataframe': self.dataframe,
            'flowpath_dict': self._flowpath_dict,
            'terminal_codes': self._terminal_codes,
            'upstream_termincal': self._upstream_terminal,
            'connections': self._connections,
            'waterbody_df': self._waterbody_df,
            'waterbody_types_df': self._waterbody_types_df,
            'waterbody_connections': self._waterbody_connections,
            'waterbody_type_specified': self._waterbody_type_specified,
            'link_lake_crosswalk': self._link_lake_crosswalk,
            'gages': self._gages,
            'usgs_lake_gage_crosswalk': self._usgs_lake_gage_crosswalk,
            'usace_lake_gage_crosswalk': self._usace_lake_gage_crosswalk,
            'rfc_lake_gage_crosswalk': self._rfc_lake_gage_crosswalk
        }
        np.save(
            Path(destination_folder).joinpath(output_filename),
            outputs
            )
    
    def read_preprocessed_data(self,):
        preprocess_filepath = self.preprocessing_parameters.get('preprocess_source_file',None)
        if preprocess_filepath:
            try:
                inputs = np.load(Path(preprocess_filepath),allow_pickle='TRUE').item()
            except:
                #LOG.critical('Canonot find %s' % Path(preprocess_filepath))
                quit()
                
            self._dataframe = inputs.get('dataframe',None)
            self._flowpath_dict = inputs.get('flowpath_dict',None)
            self._terminal_codes = inputs.get('terminal_codes',None)
            self._upstream_terminal = inputs.get('upstream_termincal',None)
            self._connections = inputs.get('connections',None)
            self._waterbody_df = inputs.get('waterbody_df',None)
            self._waterbody_types_df = inputs.get('waterbody_types_df',None)
            self._waterbody_connections = inputs.get('waterbody_connections',None)
            self._waterbody_type_specified = inputs.get('waterbody_type_specified',None)
            self._link_lake_crosswalk = inputs.get('link_lake_crosswalk',None)
            self._gages = inputs.get('gages',None)
            self._usgs_lake_gage_crosswalk = inputs.get('usgs_lake_gage_crosswalk',None)
            self._usace_lake_gage_crosswalk = inputs.get('usace_lake_gage_crosswalk',None)
            self._rfc_lake_gage_crosswalk = inputs.get('rfc_lake_gage_crosswalk',None)


def read_file(file_name):
    extension = file_name.suffix
    if extension=='.csv':
        df = pd.read_csv(file_name)
    elif extension=='.parquet':
        df = pq.read_table(file_name).to_pandas().reset_index()
        df.index.name = None
    elif extension=='.nc':
        nc = xr.open_dataset(file_name)
        ts = str(nc.get('time').values)
        df = nc.to_pandas().reset_index()[['feature_id', 'q_lateral']]
        df.rename(columns={'q_lateral': f'{ts}'}, inplace=True)
        df.index.name = None

    return df

def tailwaters(N):
    '''
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
      
    '''
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
            
            if len(outgoing)>=1:
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
