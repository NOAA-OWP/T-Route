import numpy as np
import pandas as pd
from pathlib import Path
import yaml

import nwm_routing.__main__ as tr


class troute_model():

    def __init__(self, bmi_cfg_file):
        """
        
        """
        __slots__ = ['_log_parameters', '_preprocessing_parameters', '_supernetwork_parameters', 
                     '_waterbody_parameters', '_compute_parameters', '_forcing_parameters', 
                     '_restart_parameters', '_hybrid_parameters', '_output_parameters', '_parity_parameters', 
                     '_data_assimilation_parameters', '_time', '_segment_attributes', '_waterbody_attributes',
                     '_network','_data_assimilation']
        
        (
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
                                      'OrificeC','OrificeE','WeirC','WeirE','WeirL','ifd','qd0','h0',
                                      'reservoir_type']
    
    def preprocess_static_vars(self, values: dict):
        self._network = tr.HYFeaturesNetwork(
            self._supernetwork_parameters,
            waterbody_parameters=self._waterbody_parameters,
            restart_parameters=self._restart_parameters,
            forcing_parameters=self._forcing_parameters,
            data_assimilation_parameters=self._data_assimilation_parameters,
            compute_parameters=self._compute_parameters,
            hybrid_parameters=self._hybrid_parameters,
            from_files=False, value_dict=values,
            segment_attributes=self._segment_attributes, 
            waterbody_attributes=self._waterbody_attributes)

        # create empty data assimilation object that will get values during 'update' function. 
        #TODO: Edit this to be done in DataAssimilation module, EmptyDA?
        self._data_assimilation = tr.AllDA(
            self._data_assimilation_parameters,
            self._run_parameters,
            self._waterbody_parameters,
            self._network,
            [])

    def run(self, values: dict, until=300):
        """
        Run this model into the future.
        Run this model into the future, updating the state stored in the provided model dict appropriately.
        Note that the model assumes the current values set for input variables are appropriately for the time
        duration of this update (i.e., ``dt``) and do not need to be interpolated any here.
        Parameters
        ----------
        model: dict
            The model state data structure.
        dt: int
            The number of seconds into the future to advance the model.
        Returns
        -------
        """
        # Set input data into t-route objects
        self._network._qlateral = pd.DataFrame(values['land_surface_water_source__volume_flow_rate'],
                                               index=self._network.segment_index.to_numpy())
        self._network._coastal_boundary_depth_df = pd.DataFrame(values['coastal_boundary__depth'])

        # Create data assimilation object from da_sets for first loop iteration
        #TODO I'm not sure if this is the best way to do this. How will all of these variables be 
        # fed to t-route BMI?
        self._data_assimilation._usgs_df = pd.DataFrame(values['usgs_gage_observation__volume_flow_rate'])
        self._data_assimilation._last_obs_df = pd.DataFrame(values['lastobs__volume_flow_rate'])
        self._data_assimilation._reservoir_usgs_df = pd.DataFrame(values['reservoir_usgs_gage_observation__volume_flow_rate'])
        self._data_assimilation._reservoir_usace_df = pd.DataFrame(values['reservoir_usace_gage_observation__volume_flow_rate'])

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
                         self._data_assimilation.usgs_df,
                         self._data_assimilation.lastobs_df,
                         self._data_assimilation.reservoir_usgs_df,
                         self._data_assimilation.reservoir_usgs_param_df,
                         self._data_assimilation.reservoir_usace_df,
                         self._data_assimilation.reservoir_usace_param_df,
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
                         [], #subnetwork_list,
                         self._network.coastal_boundary_depth_df,
                         self._network.unrefactored_topobathy_df,)
        
        # update initial conditions with results output
        self._network.new_q0(self._run_results)
        self._network.update_waterbody_water_elevation()               
        
        # update t0
        self._network.new_t0(self._time_step, self._nts)

        # get reservoir DA initial parameters for next loop iteration
        self._data_assimilation.update_after_compute(
            self._run_results,
            self._data_assimilation_parameters,
            self._run_parameters,
            )
        
        (values['channel_exit_water_x-section__volume_flow_rate'], 
         values['channel_water_flow__speed'], 
         values['channel_water__mean_depth'], 
         values['lake_water~incoming__volume_flow_rate'], 
         values['lake_water~outgoing__volume_flow_rate'], 
         values['lake_surface__elevation'],
        ) = _create_output_dataframes(
            self._run_results, 
            nts, 
            self._network._waterbody_df,
            self._network.link_lake_crosswalk)
        
        # update model time
        self._time += self._time_step * nts

# Utility functions -------
def _read_config_file(custom_input_file):
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

    network_topology_parameters = data.get("network_topology_parameters", None)
    supernetwork_parameters = network_topology_parameters.get(
        "supernetwork_parameters", None
    )
    # add attributes when HYfeature network is selected
    if supernetwork_parameters['geo_file_path'][-4:] == "gpkg":
        supernetwork_parameters["title_string"]       = "HY_Features Test"
        supernetwork_parameters["geo_file_path"]      = supernetwork_parameters['geo_file_path']
        supernetwork_parameters["flowpath_edge_list"] = None    
        routelink_attr = {
                        #link????
                        "key": "id",
                        "downstream": "toid",
                        "dx": "length_m",
                        "n": "n",  # TODO: rename to `manningn`
                        "ncc": "nCC",  # TODO: rename to `mannningncc`
                        "s0": "So",
                        "bw": "BtmWdth",  # TODO: rename to `bottomwidth`
                        #waterbody: "NHDWaterbodyComID",
                        "tw": "TopWdth",  # TODO: rename to `topwidth`
                        "twcc": "TopWdthCC",  # TODO: rename to `topwidthcc`
                        "alt": "alt",
                        "musk": "MusK",
                        "musx": "MusX",
                        "cs": "ChSlp"  # TODO: rename to `sideslope`
                        }
        supernetwork_parameters["columns"]             = routelink_attr 
        supernetwork_parameters["waterbody_null_code"] = -9999
        supernetwork_parameters["terminal_code"]       =  0
        supernetwork_parameters["driver_string"]       = "NetCDF"
        supernetwork_parameters["layer_string"]        = 0
        
    preprocessing_parameters = network_topology_parameters.get(
        "preprocessing_parameters", {}
    )        
    #waterbody_parameters = network_topology_parameters.get(
    #    "waterbody_parameters", None
    #)
    waterbody_parameters = network_topology_parameters.get(
        "waterbody_parameters", {}
    )
    compute_parameters = data.get("compute_parameters", {})
    forcing_parameters = compute_parameters.get("forcing_parameters", {})
    restart_parameters = compute_parameters.get("restart_parameters", {})
    hybrid_parameters = compute_parameters.get("hybrid_parameters", {})
    data_assimilation_parameters = compute_parameters.get(
        "data_assimilation_parameters", {}
    )
    output_parameters = data.get("output_parameters", {})
    parity_parameters = output_parameters.get("wrf_hydro_parity_check", {})

    return (
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
    )

def _create_output_dataframes(results, nts, waterbodies_df, link_lake_crosswalk):
    
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