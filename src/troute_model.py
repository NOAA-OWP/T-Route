import pandas as pd
import numpy as np
import yaml
import time

from nwm_routing.log_level_set import log_level_set
from troute.config import Config
import nwm_routing.__main__ as tr

from troute.network import bmi_array2df as a2df

class troute_model():

    def __init__(self, bmi_cfg_file):
        """
        Read t-route configuration file and store parameters within model.
        Initialize model start time, time step, and default number of time
        steps (1). Create list of static attributes for segments and waterbodies.
        Parameters
        ----------
        bmi_cfg_file: str
            Configuration file (.yaml) that provides all configuration specific parameters.
        Returns
        -------
        """
        __slots__ = ['_log_parameters', '_preprocessing_parameters', '_supernetwork_parameters', 
                     '_waterbody_parameters', '_compute_parameters', '_forcing_parameters', 
                     '_restart_parameters', '_hybrid_parameters', '_output_parameters', 
                     '_parity_parameters', '_data_assimilation_parameters', '_time', 
                     '_segment_attributes', '_waterbody_attributes', '_network',
                     '_data_assimilation', '_fvd', '_lakeout', '_nudge']
        
        (
            self._log_parameters,
            self._preprocessing_parameters, 
            self._supernetwork_parameters, 
            self._waterbody_parameters, 
            self._compute_parameters, 
            self._forcing_parameters, 
            self._restart_parameters, 
            self._hybrid_parameters, 
            self._output_parameters, 
            self._parity_parameters, 
            self._data_assimilation_parameters,
            self._bmi_parameters,
        ) = _read_config_file(bmi_cfg_file)

        self._run_parameters = {
            'dt': self._forcing_parameters.get('dt'),
            'nts': self._forcing_parameters.get('nts'),
            'cpu_pool': self._compute_parameters.get('cpu_pool')
            }

        self._time = 0.0
        self._time_step = self._forcing_parameters.get('dt')
        self._nts = 1

        self._segment_attributes = ['segment_id','segment_toid','dx','n','ncc','s0','bw','tw',
                                    'twcc','alt','musk','musx','cs']
        self._waterbody_attributes = ['waterbody_id','waterbody_toid','LkArea','LkMxE','OrificeA',
                                      'OrificeC','OrificeE','WeirC','WeirE','WeirL','ifd',
                                      'reservoir_type']
        
        self.showtiming = self._log_parameters.get("showtiming", None)
        if self.showtiming:
            self.task_times = {
                'network_time': 0,
                'forcing_time' : 0,
                'output_time' : 0,
                'run_time' :0,
                'data_assimilation_time' : 0,

            }
        
        self._subnetwork_list = [None, None, None]
    
    def preprocess_static_vars(self, values: dict):
        """
        Create the static data structures for the network and data assimilation
        objects. Empty dataframes contining only IDs will be created, as well
        as objects such as connections dictionary.
        ----------
        values: dict
            The static and dynamic values for the model.
        Returns
        -------
        """
        if self.showtiming:
            network_start_time = time.time()

        self._network = tr.HYFeaturesNetwork(
            self._supernetwork_parameters,
            waterbody_parameters=self._waterbody_parameters,
            restart_parameters=self._restart_parameters,
            forcing_parameters=self._forcing_parameters,
            data_assimilation_parameters=self._data_assimilation_parameters,
            compute_parameters=self._compute_parameters,
            hybrid_parameters=self._hybrid_parameters,
            preprocessing_parameters=self._preprocessing_parameters,
            output_parameters=self._output_parameters,
            from_files=False, value_dict=values,
            bmi_parameters=self._bmi_parameters,)

        if (len(self._network._nexus_latlon) > 0):
            values['diffusive_tw_ids'] = self._network._nexus_latlon.id.values
            values['diffusive_tw_latitude'] = self._network._nexus_latlon.lat.values
            values['diffusive_tw_longitude'] = self._network._nexus_latlon.lon.values
        else:
            values['diffusive_tw_ids'] = np.empty(0,dtype=np.int32)
            values['diffusive_tw_latitude'] = np.empty(0,dtype=np.float64)
            values['diffusive_tw_longitude'] = np.empty(0,dtype=np.float64)        

        # Create data assimilation object with IDs but no dynamic variables yet.
        # Dynamic variables will be assigned during 'run' function. 
        self._data_assimilation = tr.DataAssimilation(
            self._network,
            self._data_assimilation_parameters,
            self._run_parameters,
            self._waterbody_parameters,
            from_files=False,
            value_dict=values,
        )

        if len(values['upstream_id'])>0:
            for key in values['upstream_id']:
                del self._network._connections[key]
                del self._network._reverse_network[key]
                for tw in self._network._independent_networks.keys():
                    del self._network._independent_networks[tw][key]
                    for rli, _ in enumerate(self._network._reaches_by_tw[tw]):
                        self._network._reaches_by_tw[tw][rli].remove(key)

        if self.showtiming:
            network_end_time = time.time()
            self.task_times['network_time'] += network_end_time - network_start_time


    def run(self, values: dict, until=300):
        """
        Run this model into the future, updating the state stored in the provided model dict appropriately.
        Note that the model assumes the current values set for input variables are appropriately for the time
        duration of this update (i.e., ``dt``) and do not need to be interpolated any here.
        Parameters
        ----------
        values: dict
            The static and dynamic values for the model.
        dt: int
            The number of seconds into the future to advance the model.
        Returns
        -------
        """
        # Set input data into t-route objects
        # Forcing values:
        if self.showtiming:
            forcing_start_time = time.time()

        qlats_df = pd.DataFrame(values['land_surface_water_source__volume_flow_rate'],
                                index=values['land_surface_water_source__id'])
        qlats_df = qlats_df[qlats_df.index.isin(self._network.segment_index)]
        
        all_df = pd.DataFrame( np.zeros( (len(self._network.segment_index), len(qlats_df.columns)) ), index=self._network.segment_index,
            columns=qlats_df.columns )
        all_df.loc[ qlats_df.index ] = qlats_df
        qlats_df = all_df.sort_index()
        
        if not self._network.segment_index.empty:
            qlats_df = qlats_df[qlats_df.index.isin(self._network.segment_index)]
        
        self._network._qlateral = qlats_df

        #self._network._coastal_boundary_depth_df = pd.DataFrame(values['coastal_boundary_depth'])

        depthArray_coastal = values['depthArray_coastal']

        if (len(depthArray_coastal)>0):

            self._depthArray_coastal = depthArray_coastal
            self._timeArray_coastal = values['timeArray_coastal']
            self._nTimes_coastal = values['nTimes_coastal']
            self._stationArray_coastal = values['stationArray_coastal']
            self._nStations_coastal = values['nStations_coastal']
            self._coastal_timeRef = values['coastal_timeRef']

            # Unflatten the actual depth array
            df_depthArray_coastal = a2df._unflatten_array(self._depthArray_coastal,\
                                    self._nTimes_coastal, self._nStations_coastal)
            
            # Decode time axis
            timeAxisName = 'time'

            freqString = 'None'
            df_withTimes_coastal = a2df._time_retrieve_from_arrays(\
                        df_depthArray_coastal, self._coastal_timeRef, self._timeArray_coastal, \
                        timeAxisName, freqString)

            # Add station IDs as index
            stationAxisName = 'station_name'
            index = pd.Index(self._stationArray_coastal, name=stationAxisName, dtype=np.int64)

            df_withTimes_coastal.index = (index)

            self._network._coastal_boundary_depth_df = df_withTimes_coastal

        if len(values['upstream_id'])>0:
            flowveldepth_interorder = {values['upstream_id'][0]:{"results": values['upstream_fvd']}}
        else:
            flowveldepth_interorder = {}

        if self.showtiming:
            forcing_end_time = time.time()
            self.task_times['forcing_time'] += forcing_end_time - forcing_start_time

        # Trim the time-extent of the streamflow_da usgs_df
        # what happens if there are timeslice files missing on the front-end? 
        # if the first column is some timestamp greater than t0, then this will throw
        # an error. Need to think through this more. 
        if not self._data_assimilation.usgs_df.empty:
            self._data_assimilation._usgs_df = self._data_assimilation.usgs_df.loc[:,self._network.t0:]
        
        if self.showtiming:
            DA_end_time = time.time()
            self.task_times['data_assimilation_time'] += DA_end_time - forcing_end_time

        # Adjust number of steps based on user input
        nts = int(until/self._time_step)

        # Run routing
        (
            self._run_results, 
            self._subnetwork_list
        ) = tr.nwm_route(self._network.connections, 
                         self._network.reverse_network, 
                         self._network.waterbody_connections, 
                         self._network._reaches_by_tw,
                         self._compute_parameters.get('parallel_compute_method','serial'), 
                         self._compute_parameters.get('compute_kernel'),
                         self._compute_parameters.get('subnetwork_target_size'),
                         self._compute_parameters.get('cpu_pool'),
                         self._network.t0,
                         self._time_step,
                         nts,
                         self._forcing_parameters.get('qts_subdivisions', 12), #FIXME
                         self._network.independent_networks, 
                         self._network.dataframe,
                         self._network.q0,
                         self._network._qlateral,
                         self._network._eloss,
                         self._network.forcing_parameters.get("ssout"),
                         self._data_assimilation.usgs_df,
                         self._data_assimilation.lastobs_df,
                         self._data_assimilation.reservoir_usgs_df,
                         self._data_assimilation.reservoir_usgs_param_df,
                         self._data_assimilation.reservoir_usace_df,
                         self._data_assimilation.reservoir_usace_param_df,
                         self._data_assimilation.reservoir_rfc_df,
                         self._data_assimilation.reservoir_rfc_param_df,
                         self._data_assimilation.assimilation_parameters,
                         self._compute_parameters.get('assume_short_ts', False),
                         self._compute_parameters.get('return_courant', False),
                         self._network._waterbody_df,
                         self._waterbody_parameters,
                         self._network._waterbody_types_df,
                         self._network.waterbody_type_specified,
                         self._network.diffusive_network_data,
                         self._network.topobathy_df,
                         self._network.refactored_diffusive_domain,
                         self._network.refactored_reaches,
                         self._subnetwork_list,
                         self._network.coastal_boundary_depth_df,
                         self._network.unrefactored_topobathy_df,
                         flowveldepth_interorder,
                         from_files=False,
                         )
        
        # update initial conditions with results output
        self._network.new_q0(self._run_results)
        '''
        # update offnetwork_upstream initial conditions
        if flowveldepth_interorder:
            self._network._q0 = pd.concat(
                [
                    self._network.q0,
                    pd.concat(
                        [
                            pd.DataFrame(
                                vals['results'][[-3, -3, -1]].reshape(1,3), index=[seg_id], columns=["qu0", "qd0", "h0"]
                            ) 
                            for seg_id, vals in flowveldepth_interorder.items()
                        ],
                        copy=False,
                    )
                ]
            ).sort_index()
        '''
        self._network.update_waterbody_water_elevation()               

        # update t0
        self._network.new_t0(self._time_step, nts)

        # get reservoir DA initial parameters for next loop iteration
        self._data_assimilation.update_after_compute(self._run_results, self._time_step*nts)

        if self.showtiming:
            run_end_time = time.time()
            self.task_times['run_time'] += run_end_time - DA_end_time
         
        # Create output flowveldepth and lakeout arrays
        self._fvd, self._lakeout = _create_output_dataframes(
            self._run_results,
            nts,
            self._network._waterbody_df,
        )
        values['fvd_results'] = self._fvd.values.flatten()
        values['fvd_index'] = self._fvd.index
        values['lakeout'] = self._lakeout.values.flatten()
        values['lakeout_index'] = self._lakeout.index
        values['q0'] = self._network.q0.values.flatten()
        values['q0_index'] = self._network.q0.index
        values['waterbody_df'] = self._network.waterbody_dataframe.values.flatten()
        values['waterbody_df_index'] = self._network.waterbody_dataframe.index
        lastobs_df = self._data_assimilation.lastobs_df.join(self._network.link_gage_df)
        values['lastobs_df'] = lastobs_df.values.flatten()
        values['lastobs_df_index'] = lastobs_df.index

        nudge = np.concatenate([r[8] for r in self._run_results])[:,1:]
        usgs_positions_id = np.concatenate([r[3][0] for r in self._run_results]).astype(int)
        self._nudge = pd.DataFrame(data=nudge, index=usgs_positions_id)
        values['nudging'] = self._nudge.values.flatten()
        values['nudging_ids'] = self._nudge.index

        # Get output from final timestep
        (values['channel_exit_water_x-section__volume_flow_rate'], 
         values['channel_water_flow__speed'], 
         values['channel_water__mean_depth'], 
         values['lake_water~incoming__volume_flow_rate'], 
         values['lake_water~outgoing__volume_flow_rate'], 
         values['lake_surface__elevation'],
        ) = _retrieve_last_output(
            self._run_results, 
            nts, 
            self._network._waterbody_df,)
        
        # update model time
        self._time += self._time_step * nts
        
        if self.showtiming:
            output_end_time = time.time()
            self.task_times['output_time'] += output_end_time - run_end_time

    def print_timing_summary(self,):
            if self.showtiming:
                print('***************** TIMING SUMMARY *****************')
                print('----------------------------------------')
                total_time = (round(self.task_times['network_time'], 2) + 
                              round(self.task_times['data_assimilation_time'], 2) + 
                              round(self.task_times['forcing_time'], 2) + 
                              round(self.task_times['run_time'], 2) + 
                              round(self.task_times['output_time'], 2)
                              )
                
                print(
                    'Network graph construction:: {} secs, {} %'\
                    .format(
                        round(self.task_times['network_time'],2),
                        round(self.task_times['network_time']/total_time * 100,2)
                    )
                )
                print(
                    'Data assimilation construction:: {} secs, {} %'\
                    .format(
                        round(self.task_times['data_assimilation_time'],2),
                        round(self.task_times['data_assimilation_time']/total_time * 100,2)
                    )
                )
                print(
                    'Forcing array construction: {} secs, {} %'\
                    .format(
                        round(self.task_times['forcing_time'],2),
                        round(self.task_times['forcing_time']/total_time * 100,2)
                    )
                )
                print(
                    'Routing computations: {} secs, {} %'\
                    .format(
                        round(self.task_times['run_time'],2),
                        round(self.task_times['run_time']/total_time * 100,2)
                    )
                )
                print(
                    'Output writing: {} secs, {} %'\
                    .format(
                        round(self.task_times['output_time'],2),
                        round(self.task_times['output_time']/total_time * 100,2)
                    )
                )
                print('----------------------------------------')
                print(f'Total execution time: {total_time} secs')
                    
                
# Utility functions -------
def _read_config_file(custom_input_file): #TODO: Update this function, I dont' think
    # we need all of this for BMI. This was taken directly from t-route model...
    '''
    Read-in data from user-created configuration file.
    
    Arguments
    ---------
    custom_input_file (str): configuration filepath, .yaml
    
    Returns
    -------
    preprocessing_parameters     (dict): Input parameters re preprocessing
    supernetwork_parameters      (dict): Input parameters re network extent
    waterbody_parameters         (dict): Input parameters re waterbodies
    compute_parameters           (dict): Input parameters re computation settings
    forcing_parameters           (dict): Input parameters re model forcings
    restart_parameters           (dict): Input parameters re model restart
    hybrid_parameters            (dict): Input parameters re diffusive wave model
    output_parameters            (dict): Input parameters re output writing
    parity_parameters            (dict): Input parameters re parity assessment
    data_assimilation_parameters (dict): Input parameters re data assimilation

    '''
    with open(custom_input_file) as custom_file:
        data = yaml.load(custom_file, Loader=yaml.SafeLoader)

    troute_configuration = Config(**data)
    config_dict = troute_configuration.dict()

    log_parameters = config_dict.get('log_parameters')
    compute_parameters = config_dict.get('compute_parameters')
    network_topology_parameters = config_dict.get('network_topology_parameters')
    output_parameters = config_dict.get('output_parameters')
    bmi_parameters = config_dict.get('bmi_parameters')

    preprocessing_parameters = network_topology_parameters.get('preprocessing_parameters')
    supernetwork_parameters = network_topology_parameters.get('supernetwork_parameters')
    waterbody_parameters = network_topology_parameters.get('waterbody_parameters')
    forcing_parameters = compute_parameters.get('forcing_parameters')
    restart_parameters = compute_parameters.get('restart_parameters')
    hybrid_parameters = compute_parameters.get('hybrid_parameters')
    parity_parameters = output_parameters.get('wrf_hydro_parity_check')
    data_assimilation_parameters = compute_parameters.get('data_assimilation_parameters')

    # configure python logger
    log_level_set(log_parameters)

    return (
        log_parameters,
        preprocessing_parameters,
        supernetwork_parameters,
        waterbody_parameters,
        compute_parameters,
        forcing_parameters,
        restart_parameters,
        hybrid_parameters,
        output_parameters,
        parity_parameters,
        data_assimilation_parameters,
        bmi_parameters,
    )

def _retrieve_last_output(results, nts, waterbodies_df,):
    """
    Retrieve calculated values for the last timestep.
    ----------
    results: list
        The results from nwm_routing.
    nts: int
        The number of time steps the model was run.
    waterbodies_df: pd.DataFrame
        Dataframe containing waterbody parameters (specifically, IDs stored in index)
    link_lake_crosswalk: dict #TODO: Can we remove this?
        Relates lake ids to outlet link ids.
    Returns
    -------
    q_channel_df: pandas.core.series.Series
        Streamflow rate for each segment
    v_channel_df: pandas.core.series.Series
        Streamflow velocity for each segment
    d_channel_df: pandas.core.series.Series
        Streamflow depth for each segment
    i_lakeout_df: pandas.core.series.Series
        Inflow for each waterbody
    q_lakeout_df: pandas.core.series.Series
        Outflow for each waterbody
    d_lakeout_df: pandas.core.series.Series
        Water elevation for each waterbody
    """
    qvd_columns = pd.MultiIndex.from_product(
        [range(int(nts)), ["q", "v", "d"]]
    ).to_flat_index()
    
    flowveldepth = pd.concat(
        [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results], copy=False,
    )
    
    # create waterbody dataframe for output to netcdf file
    i_columns = pd.MultiIndex.from_product(
        [range(int(nts)), ["i"]]
    ).to_flat_index()
    
    wbdy = pd.concat(
        [pd.DataFrame(r[6], index=r[0], columns=i_columns) for r in results],
        copy=False,
    )
    
    wbdy_id_list = waterbodies_df.index.values.tolist()

    i_lakeout_df = wbdy.loc[wbdy_id_list].iloc[:,-1]
    q_lakeout_df = flowveldepth.loc[wbdy_id_list].iloc[:,-3]
    d_lakeout_df = flowveldepth.loc[wbdy_id_list].iloc[:,-1]
    # lakeout = pd.concat([i_df, q_df, d_df], axis=1)
    
    # replace waterbody lake_ids with outlet link ids
    #TODO Update the following line to fit with HyFeatures. Do we need to replace IDs? Or replace
    # waterbody_ids with the downstream segment?
    #flowveldepth = _reindex_lake_to_link_id(flowveldepth, link_lake_crosswalk)
    
    q_channel_df = flowveldepth.iloc[:,-3]
    v_channel_df = flowveldepth.iloc[:,-2]
    d_channel_df = flowveldepth.iloc[:,-1]
    
    segment_ids = flowveldepth.index.values.tolist()

    return q_channel_df, v_channel_df, d_channel_df, i_lakeout_df, q_lakeout_df, d_lakeout_df#, wbdy_id_list, 

def _create_output_dataframes(results, nts, waterbodies_df,):
    """
    Retrieve calculated flowveldepth values and waterbody inflow, outflow, and elevation.
    Parameters
    ----------
    results: list
        The results from nwm_routing.
    nts: int
        The number of time steps the model was run.
    waterbodies_df: pd.DataFrame
        Dataframe containing waterbody parameters (specifically, IDs stored in index)
    Returns
    -------
    flowveldepth: pandas.DataFrame
        Flow, velocity, and depth results
    lakeout: pandas.Dataframe
        Waterbody inflow, outflow, and elevation
    """
    qvd_columns = pd.MultiIndex.from_product(
        [range(int(nts)), ["q", "v", "d"]]
    ).to_flat_index()
    
    flowveldepth = pd.concat(
        [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results], copy=False,
    )
    
    # create waterbody dataframe for output to netcdf file
    i_columns = pd.MultiIndex.from_product(
        [range(int(nts)), ["i"]]
    ).to_flat_index()
    
    wbdy = pd.concat(
        [pd.DataFrame(r[6], index=r[0], columns=i_columns) for r in results],
        copy=False,
    )
    
    wbdy_id_list = waterbodies_df.index.values.tolist()
    
    i_lakeout_df = wbdy.loc[wbdy_id_list]
    q_lakeout_df = flowveldepth.loc[wbdy_id_list].iloc[:,0::3]
    d_lakeout_df = flowveldepth.loc[wbdy_id_list].iloc[:,2::3]
    lakeout = pd.concat([i_lakeout_df, q_lakeout_df, d_lakeout_df], axis=1)
    
    # segment_ids = flowveldepth.index.values.tolist() #TODO: do we need to return segment ids
    # to keep track of order?

    return flowveldepth, lakeout 
