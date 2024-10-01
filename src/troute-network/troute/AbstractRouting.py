from abc import ABC, abstractmethod
import logging
import yaml
import json
import xarray as xr
import pandas as pd
from itertools import chain

from troute.nhd_network import reverse_network, reachable
from troute.nhd_network_utilities_v02 import organize_independent_networks, build_refac_connections

LOG = logging.getLogger('')

def read_diffusive_domain(domain_file):
    '''
    Read diffusive domain data from .ymal or .json file.
    
    Arguments
    ---------
    domain_file (str or pathlib.Path): Path of diffusive domain file
    
    Returns
    -------
    data (dict int: [int]): domain tailwater segments: list of segments in domain 
                            (includeing tailwater segment) 
    
    '''
    if domain_file.suffix == ".yaml":
        with open(domain_file) as domain:
            data = yaml.load(domain, Loader=yaml.SafeLoader)
    else:
        with open(domain_file) as domain:
            data = json.load(domain)
            
    return data

def read_netcdf(geo_file_path):
    '''
    Open a netcdf file with xarray and convert to dataframe
    
    Arguments
    ---------
    geo_file_path (str or pathlib.Path): netCDF filepath
    
    Returns
    -------
    ds.to_dataframe() (DataFrame): netCDF contents
    
    Notes
    -----
    - When handling large volumes of netCDF files, xarray is not the most efficient.
    
    '''
    with xr.open_dataset(geo_file_path) as ds:
        return ds.to_dataframe()

def read_parquet(file_path, seg_ids):
    '''
    Open a parquet file with pyarrow read_table and convert to dataframe
    
    Arguments
    ---------
    file_path (str or pathlib.Path): nparquetetCDF filepath
    
    Returns
    -------
    
    ds.to_pandas() (DataFrame): parquet contents 
    
    Notes
    -----
    - It checks if the column 'hy_id' exists in the DataFrame and if there's at least one value in that column that is not numeric (means its like wb-254530)
    

    '''
    df = pd.read_parquet(file_path,
                         columns=['hy_id', 'relative_dist', 'Z', 'roughness', 'cs_id'],
                         filters=[('hy_id', 'in', seg_ids)]).dropna()
    
    if 'hy_id' in df.columns and not df['hy_id'].str.isnumeric().all():
        df['hy_id'] = df['hy_id'].apply(lambda x: x.split('-')[-1])
    return df


class AbstractRouting(ABC):
    """
    
    """
    __slots__ = ["hybrid_params", "compute_params", "_diffusive_domain", "_coastal_boundary_depth_df",
                "_diffusive_network_data", "_topobathy_df", "_refactored_diffusive_domain",
                "_refactored_diffusive_network_data", "_refactored_reaches", 
                "_unrefactored_topobathy_df", "_all_links", "_bad_topobathy_links", "_dataframe"]
    
    def __init__(self):
        """
        
        """
        self._diffusive_domain                  = None
        self._diffusive_network_data            = None
        self._topobathy_df                      = pd.DataFrame()
        self._unrefactored_topobathy_df         = pd.DataFrame() 
        self._refactored_diffusive_domain       = None
        self._refactored_diffusive_network_data = None   
        self._refactored_reaches                = {}
        self._all_links                         = []
        self._bad_topobathy_links               = []
        self._dataframe                         = pd.DataFrame() 

    @abstractmethod
    def update_routing_domain(self, dataframe, connections, waterbody_dataframe):
        pass
    
    @property
    @abstractmethod
    def diffusive_network_data(self):
        pass

    @property
    @abstractmethod
    def topobathy_df(self):
        pass
    
    @property
    @abstractmethod
    def refactored_diffusive_domain(self):
        pass
    
    @property
    @abstractmethod
    def refactored_reaches(self):
        pass
    
    @property
    @abstractmethod
    def unrefactored_topobathy_df(self):
        pass

    @property
    @abstractmethod
    def all_links(self):
        pass

    @property
    @abstractmethod
    def bad_topobathy_links(self):
        pass

    @property
    @abstractmethod
    def dataframe(self):
        pass



class MCOnly(AbstractRouting):

    def __init__(self, _, __):
        self.hybrid_params = None
        self.compute_params = None
        
        super().__init__()

    def update_routing_domain(self, dataframe, connections, waterbody_dataframe):
        return dataframe, connections

    @property
    def diffusive_network_data(self):
        return self._diffusive_network_data

    @property
    def topobathy_df(self):
        return self._topobathy_df
    
    @property
    def refactored_diffusive_domain(self):
        return self._refactored_diffusive_domain
    
    @property
    def refactored_reaches(self):
        return self._refactored_reaches
    
    @property
    def unrefactored_topobathy_df(self):
        return self._unrefactored_topobathy_df
    
    @property
    def unrefactored_topobathy_df(self):
        return self._unrefactored_topobathy_df

    @property
    def all_links(self):
        self._all_links

    @property
    def bad_topobathy_links(self):
        self._bad_topobathy_links

    @property
    def dataframe(self):
        return self._dataframe


class MCwithDiffusive(AbstractRouting):

    def __init__(self, hybrid_params, compute_params):
        self.hybrid_params = hybrid_params
        self.compute_params = compute_params
        
        super().__init__()
    
    def update_routing_domain(self, dataframe, connections, waterbody_dataframe):
        #==========================================================================
        # build diffusive domain data and edit MC domain data for hybrid simulation
        domain_file = self.hybrid_params.get("diffusive_domain", None)
        # self._diffusive_domain = read_diffusive_domain(domain_file)
        diffusive_domain = read_diffusive_domain(domain_file)
        self._diffusive_network_data = {}
        diffusive_domain_all = {}
        rconn_diff0 = reverse_network(connections)
        all_links = []
        #bad_topobathy_links = []

        for tw in diffusive_domain:
            headlink_mainstem, rfc_val, rpu_val = list(diffusive_domain.get(tw).values())
            if 999999 in headlink_mainstem:
                headlink_mainstem.remove(999999)
                targets = headlink_mainstem.copy()
                wbody_ids = waterbody_dataframe.index.tolist()
                targets = targets + wbody_ids
                links = list(reachable(rconn_diff0, sources=[tw], targets=targets).get(tw))
                outlet_ids = [connections.get(id) for id in wbody_ids]
                outlet_ids = list(chain.from_iterable(outlet_ids))
                wbody_and_outlet_ids = wbody_ids + outlet_ids
                links = list(set(links).difference(set(wbody_and_outlet_ids)))
                
                diffusive_domain_all[tw] = {
                    'links': links,
                    'rfc': rfc_val,
                    'rpu': rpu_val,
                    'upstream_boundary_link_mainstem': [],
                    'targets': targets,
                }
                all_links = all_links + links
            else:
                headlink_mainstem = headlink_mainstem[0]
                twlink_mainstem = tw
                diffusive_domain_all[twlink_mainstem] = self.diffusive_domain_by_both_ends_streamid(connections, headlink_mainstem, twlink_mainstem, rfc_val, rpu_val)
                diffusive_domain_all[twlink_mainstem]['targets'] = [headlink_mainstem]
                all_links = all_links + diffusive_domain_all[twlink_mainstem]['links']

        self._diffusive_domain = diffusive_domain_all
        self._all_links = all_links 
        self._dataframe = dataframe  
        
        # When topodata is available, load topobathy data and remove any links for which topo data cannot be obtained
        topobathy_df = self.topobathy_df
 
        for tw in self._diffusive_domain:          
            wbody_ids = waterbody_dataframe.index.tolist()
            targets = self._diffusive_domain[tw]['targets'] + self._bad_topobathy_links
            links = list(reachable(rconn_diff0, sources=[tw], targets=targets).get(tw))
            links = list(set(self._diffusive_domain[tw]['links']).intersection(set(links)))
            outlet_ids = [connections.get(id) for id in wbody_ids]
            outlet_ids = list(chain.from_iterable(outlet_ids))
            wbody_and_outlet_ids = wbody_ids + outlet_ids + self._bad_topobathy_links
            mainstem_segs = list(set(links).difference(set(wbody_and_outlet_ids)))
            
            # we want mainstem_segs start at a mainstem link right after the upstream boundary mainstem link, which is
            # in turn not under any waterbody. This boundary mainstem link should be turned into a tributary segment.
            upstream_boundary_mainstem_link = self._diffusive_domain[tw]['upstream_boundary_link_mainstem']
            for us_link in upstream_boundary_mainstem_link:
                if us_link in mainstem_segs:
                    mainstem_segs.remove(us_link)
            
            # ===== build diffusive network data objects ==== 
            self._diffusive_network_data[tw] = {}

            # add diffusive domain segments
            self._diffusive_network_data[tw]['mainstem_segs'] =  mainstem_segs

            # diffusive domain tributary segments
            trib_segs = []
            
            for seg in mainstem_segs:
                us_list = rconn_diff0[seg]
                for u in us_list:
                    if u not in mainstem_segs:
                        trib_segs.append(u) 

            self._diffusive_network_data[tw]['tributary_segments'] = trib_segs
            # diffusive domain connections object
            self._diffusive_network_data[tw]['connections'] = {k: connections[k] for k in (mainstem_segs + trib_segs)}       

            # make sure that no downstream link below tw
            self._diffusive_network_data[tw]['connections'][tw] = []

            # diffusive domain reaches and upstream connections. 
            # break network at tributary segments
            _, reaches, rconn_diff = organize_independent_networks(
                self._diffusive_network_data[tw]['connections'],
                set(trib_segs),
                set(),
            )
            
            self._diffusive_network_data[tw]['rconn'] = rconn_diff
            self._diffusive_network_data[tw]['reaches'] = reaches[tw]

            # RouteLink parameters
            self._diffusive_network_data[tw]['param_df'] = dataframe.filter(
                (mainstem_segs + trib_segs),
                axis = 0,
            )
            self._diffusive_network_data[tw]['upstream_boundary_link'] = upstream_boundary_mainstem_link

            # ==== remove diffusive domain segs from MC domain ====        
            # drop indices from param_df. Make sure when mainstem_segs accidently includes lake ids, exclude them 
            # from id list to be dropped from dataframe as dataframe only handles channel parameters.     
            # Skip this process for enabling the exchange of flow between MC and diffusive during runtime.
            if 'hybrid-routing' not in self.compute_params['compute_kernel']:
                existing_indicies_in_dataframe = [id for id in mainstem_segs if id in dataframe.index]
                dataframe = dataframe.drop(existing_indicies_in_dataframe)
                
                # remove keys from connections dictionary
                for s in mainstem_segs:
                    connections.pop(s)

                # update downstream connections of trib segs
                for us in trib_segs:
                    connections[us] = []

        return dataframe, connections

    @property
    def diffusive_network_data(self):
        return self._diffusive_network_data

    @property
    def topobathy_df(self):
        return self._topobathy_df
    
    @property
    def refactored_diffusive_domain(self):
        return self._refactored_diffusive_domain
    
    @property
    def refactored_reaches(self):
        return self._refactored_reaches
    
    @property
    def unrefactored_topobathy_df(self):
        return self._unrefactored_topobathy_df

    @property
    def all_links(self):
        self._all_links

    @property
    def bad_topobathy_links(self):
        self._bad_topobathy_links

    @property
    def dataframe(self):
        return self._dataframe
        
    def diffusive_domain_by_both_ends_streamid(self, connections, headlink_mainstem, twlink_mainstem, rfc_val, rpu_val):
        # This function build diffusive_domain using given headwater segment IDs at upper and tailwater at lower ends of mainstem.

        uslink_mainstem = headlink_mainstem
        dslink_mainstem = 1 # initial value
        mainstem_list =[headlink_mainstem]
        while dslink_mainstem != twlink_mainstem:
            try:
                dslink_mainstem = connections[uslink_mainstem][0]
                mainstem_list.append(dslink_mainstem)
                uslink_mainstem = dslink_mainstem   
            except KeyError:
                # Handle the KeyError, e.g., break the loop or log an error
                LOG.debug(f"KeyError: 'connections' does not have a key '{uslink_mainstem}'")
                return None 

        diffusive_domain = {'links':mainstem_list, 'rfc': rfc_val, 'rpu': rpu_val, 'upstream_boundary_link_mainstem':[headlink_mainstem]}
   
        return diffusive_domain


class MCwithDiffusiveNatlXSectionNonRefactored(MCwithDiffusive):

    def __init__(self, hybrid_params, compute_params):

        super().__init__(hybrid_params = hybrid_params, compute_params=compute_params)

    @property
    def topobathy_df(self):
        if self._topobathy_df.empty:
            topobathy_file = self.hybrid_params.get("topobathy_domain", None)
            if topobathy_file.suffix == '.nc':
                self._topobathy_df = read_netcdf(topobathy_file).set_index('link')
                self._topobathy_df.index = self._topobathy_df.index.astype(int)

            elif topobathy_file.suffix == '.parquet':
                seg_ids = []
                for tw in self._diffusive_domain:
                    seg_ids = seg_ids + self._diffusive_domain[tw]['links']
                seg_ids = ['wb-' + str(seg) for seg in seg_ids]
                self._topobathy_df = read_parquet(topobathy_file, seg_ids).set_index('hy_id')
                self._topobathy_df.index = self._topobathy_df.index.astype(int)
        
            # Load topobathy data and remove any links for which topo data cannot be obtained
            topobathy_df = self._topobathy_df
            missing_topo_ids = list(set(self._all_links).difference(set(topobathy_df.index)))
            topo_df_list = []
                
            for key in missing_topo_ids:
                topo_df_list.append(_fill_in_missing_topo_data(key, self._dataframe, topobathy_df))
                
            if len(topo_df_list)==0: 
                new_topo_df = pd.DataFrame() 
            else:
                new_topo_df = pd.concat(topo_df_list)

            bad_topobathy_links = list(set(missing_topo_ids).difference(set(new_topo_df.index)))
            self._bad_topobathy_links =bad_topobathy_links
            self._topobathy_df  = pd.concat([topobathy_df,new_topo_df])
                
            # select topo data with minimum value of cs_id for each segment
            df =  self._topobathy_df
            min_cs_id=df.reset_index().groupby('hy_id')['cs_id'].transform('min')
            mask = df.reset_index()['cs_id'] == min_cs_id
            single_topo_df = df.reset_index()[mask]
            single_topo_df.set_index('hy_id', inplace=True)
            self._topobathy_df= single_topo_df
 
        return self._topobathy_df


class MCwithDiffusiveNatlXSectionRefactored(MCwithDiffusive):

    def __init__(self, hybrid_params, compute_params):
        
        super().__init__(hybrid_params = hybrid_params, compute_params=compute_params)

    @property
    def topobathy_df(self):
        if self._topobathy_df.empty:
            refactored_topobathy_file = self.hybrid_params.get("refactored_topobathy_domain", None)
            self._topobathy_df = read_netcdf(refactored_topobathy_file).set_index('link')
        return self._topobathy_df
    
    @property
    def refactored_diffusive_domain(self):
        if not self._refactored_diffusive_domain:
            refactored_domain_file = self.hybrid_params.get("refactored_domain", None)
            self._refactored_diffusive_domain = read_diffusive_domain(refactored_domain_file)
        return self._refactored_diffusive_domain
    
    @property
    def refactored_reaches(self):
        if not self._refactored_reaches:
            refactored_topobathy_file = self.hybrid_params.get("refactored_topobathy_domain", None)
            diffusive_parameters = {'geo_file_path': refactored_topobathy_file}
            refactored_connections = build_refac_connections(diffusive_parameters)

            for tw in self._diffusive_domain:

                # list of stream segments of a single refactored diffusive domain 
                refac_tw = self.refactored_diffusive_domain[tw]['refac_tw']
                rlinks_tw = self.refactored_diffusive_domain[tw]['rlinks']
                refactored_connections_tw = {}   

                # Subset a connection dictionary (upstream segment as key : downstream segments as values) from refactored_connections
                # for a single refactored diffusive domain defined by a current tw. 
                for k in rlinks_tw:
                    if k in refactored_connections.keys() and k != refac_tw:
                        refactored_connections_tw[k] = refactored_connections[k]
                
                trib_segs = self.diffusive_network_data[tw]['tributary_segments']
                refactored_diffusive_network_data = {}
                refactored_diffusive_network_data[refac_tw] = {}                
                refactored_diffusive_network_data[refac_tw]['tributary_segments'] = trib_segs
                refactored_diffusive_network_data[refac_tw]['connections'] = refactored_connections_tw                 

                for k in trib_segs:
                    refactored_diffusive_network_data[refac_tw]['connections'][k] = [self._refactored_diffusive_domain[tw]['incoming_tribs'][k]]

                # diffusive domain reaches and upstream connections. 
                # break network at tributary segments
                _, refactored_reaches_batch, refactored_conn_diff = organize_independent_networks(
                                                            refactored_diffusive_network_data[refac_tw]['connections'],
                                                            set(trib_segs),
                                                            set(),
                                                            )

                self._refactored_reaches[refac_tw] = refactored_reaches_batch[refac_tw]
                refactored_diffusive_network_data[refac_tw]['mainstem_segs'] = self._refactored_diffusive_domain[tw]['rlinks']
                refactored_diffusive_network_data[refac_tw]['upstream_boundary_link'] = self._diffusive_network_data[tw]['upstream_boundary_link']
        return self._refactored_reaches
    
    @property
    def unrefactored_topobathy_df(self):
        if self._unrefactored_topobathy_df.empty:
            topobathy_file = self.hybrid_params.get("topobathy_domain",   None)
            self._unrefactored_topobathy_df = read_netcdf(topobathy_file).set_index('link')
            self._unrefactored_topobathy_df.index = self._unrefactored_topobathy_df.index.astype(int)
        return self._unrefactored_topobathy_df


def _fill_in_missing_topo_data(original_key, dataframe, topobathy_df):
    rconn_list = []
    key = original_key
    mainstem = dataframe.loc[key].mainstem
    mainstem_df = dataframe[dataframe['mainstem']==mainstem].reset_index()
    
    while key in list(mainstem_df.downstream):
        upstream = mainstem_df[mainstem_df['downstream']==key].key.values.tolist()
        rconn_list = rconn_list + upstream
        key = upstream[0]

    new_key = next((e for e in rconn_list if e in topobathy_df.index), None)
    
    temp_df = pd.DataFrame()
    if new_key:
        temp_df = topobathy_df.loc[[new_key]].reset_index()
        cs_id_max = temp_df['cs_id'].max()
        # Select topobathy data at the most downstream of an upstream mainstem segment
        temp_df = pd.DataFrame(temp_df[temp_df.cs_id==cs_id_max])
        temp_df.cs_id = temp_df.cs_id.replace(cs_id_max,1)
        temp_df.hy_id = original_key
        temp_df = temp_df.set_index('hy_id')
    
    return temp_df
