"""Basic Model Interface implementation for t-route."""
import yaml
import numpy as np
from bmipy import Bmi
from pathlib import Path

# Here is the model we want to run
from troute_model import troute_model

class bmi_troute(Bmi):

    def __init__(self):
        """Create a Bmi troute model that is ready for initialization."""
        super(bmi_troute, self).__init__()
        self._model = None
        self._values = {}
        #self._var_units = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        #self._grids = {}
        #self._grid_type = {}

        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        self._time_units = "s"

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'T-Route for Next Generation NWM',
        'version':            '',
        'author_name':        '',
        'grid_type':          'scalar', 
        'time_step_size':      1,       
        #'time_step_type':     'donno', #unused  
        #'step_method':        'none',  #unused
        #'time_units':         '1 hour' #NJF Have to drop the 1 for NGEN to recognize the unit
        'time_units':         'seconds' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    _input_var_names = [
        # segment static variables
        'segment_id', 
        'segment_toid',
        'dx',
        'n',
        'ncc',
        's0',
        'bw',
        'tw',
        'twcc',
        'alt',
        'musk',
        'musx',
        'cs',
        # waterbody static variables
        'waterbody_id',
        'waterbody_toid',
        'LkArea',
        'LkMxE',
        'OrificeA',
        'OrificeC',
        'OrificeE',
        'WeirC',
        'WeirE',
        'WeirL',
        'ifd',
        'qd0',
        'h0',
        'reservoir_type',
        # dynamic forcing/DA variables
        'land_surface_water_source__volume_flow_rate',
        'upstream_id',
        'upstream_fvd',
        'coastal_boundary__depth', 
        'usgs_gage_observation__volume_flow_rate',
        'reservoir_usgs_gage_observation__volume_flow_rate',
        'reservoir_usace_gage_observation__volume_flow_rate', 
        'rfc_gage_observation__volume_flow_rate', 
        'lastobs__volume_flow_rate' 
        ]

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['channel_exit_water_x-section__volume_flow_rate',
                         'channel_water_flow__speed',
                         'channel_water__mean_depth',
                         'lake_water~incoming__volume_flow_rate',
                         'lake_water~outgoing__volume_flow_rate',
                         'lake_surface__elevation',
                         'channel_water__assimilated_volume_flow_rate',
                        ]

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    #------------------------------------------------------
    _var_name_units_map = {
        'channel_exit_water_x-section__volume_flow_rate':['streamflow_cms','m3 s-1'],
        'channel_water_flow__speed':['streamflow_ms','m s-1'],
        'channel_water__mean_depth':['streamflow_m','m'],
        'lake_water~incoming__volume_flow_rate':['waterbody_cms','m3 s-1'],
        'lake_water~outgoing__volume_flow_rate':['waterbody_cms','m3 s-1'],
        'lake_surface__elevation':['waterbody_m','m'],
        'channel_water__assimilated_volume_flow_rate':['assimilated_value','m3 s-1'],
        #--------------   Dynamic inputs --------------------------------
        'land_surface_water_source__volume_flow_rate':['streamflow_cms','m3 s-1'],
        'coastal_boundary_depth':['depth_m', 'm'],
        'usgs_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        'reservoir_usgs_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        'reservoir_usace_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        'rfc_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        'lastobs__volume_flow_rate':['streamflow_cms','m3 s-1']
    }

    #------------------------------------------------------
    # A list of static attributes. Not all these need to be used.
    #------------------------------------------------------
    _static_attributes_list = []


    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize(self, bmi_cfg_file=None):
        
        # -------------- Read in the BMI configuration -------------------------#
        bmi_cfg_file = Path(bmi_cfg_file)
        bmi_parameters = _read_config_file(bmi_cfg_file)
        flowpath_columns = bmi_parameters.get('flowpath_columns')
        attributes_columns = bmi_parameters.get('attributes_columns')
        lakes_columns = bmi_parameters.get('waterbody_columns', [])
        network_columns = bmi_parameters.get('network_columns', [])

        # ------------- Initialize t-route model ------------------------------#
        self._model = troute_model(bmi_cfg_file)

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for \
                                         long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for \
                                          long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for \
                                          long_name in self._var_name_units_map.keys()}
    

        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        for col in flowpath_columns + attributes_columns + lakes_columns + network_columns:
            self._values[col] = np.zeros(0)

        #FIXME Do this better..., load size of variables from config file??
        self._values['segment_id'] = np.zeros(0, dtype=int)
        self._values['segment_toid'] = np.zeros(0, dtype=int)
        self._values['dx'] = np.zeros(0)
        self._values['n'] = np.zeros(0)
        self._values['ncc'] = np.zeros(0)
        self._values['s0'] = np.zeros(0)
        self._values['bw'] = np.zeros(0)
        self._values['tw'] = np.zeros(0)
        self._values['twcc'] = np.zeros(0)
        self._values['alt'] = np.zeros(0)
        self._values['musk'] = np.zeros(0)
        self._values['musx'] = np.zeros(0)
        self._values['cs'] = np.zeros(0)
        self._values['waterbody_id'] = np.zeros(0, dtype=int)
        self._values['waterbody_toid'] = np.zeros(0, dtype=int)
        self._values['LkArea'] = np.zeros(0)
        self._values['LkMxE'] = np.zeros(0)
        self._values['OrificeA'] = np.zeros(0)
        self._values['OrificeC'] = np.zeros(0)
        self._values['OrificeE'] = np.zeros(0)
        self._values['WeirC'] = np.zeros(0)
        self._values['WeirE'] = np.zeros(0)
        self._values['WeirL'] = np.zeros(0)
        self._values['ifd'] = np.zeros(0)
        #self._values['qd0'] = np.zeros(1) #TODO will this come from a file or model engine?
        #self._values['h0'] = np.zeros(1) #TODO will this come from a file or model engine?
        self._values['reservoir_type'] = np.zeros(0)
        self._values['waterbody_connections__link'] = np.zeros(0)
        self._values['waterbody_connections__lake'] = np.zeros(0)

        self._values['land_surface_water_source__volume_flow_rate'] = np.zeros(0)
        self._values['land_surface_water_source__id'] = np.zeros(0)
        self._values['coastal_boundary_depth'] = np.zeros(0)

        self._values['depthArray_coastal'] = np.zeros(0)

        self._values['channel_exit_water_x-section__volume_flow_rate'] = np.zeros(0)
        self._values['channel_water_flow__speed'] = np.zeros(0)
        self._values['channel_water__mean_depth'] = np.zeros(0)
        self._values['lake_water~incoming__volume_flow_rate'] = np.zeros(0)
        self._values['lake_water~outgoing__volume_flow_rate'] = np.zeros(0)
        self._values['lake_surface__elevation'] = np.zeros(0)
        self._values['channel_water__assimilated_volume_flow_rate'] = np.zeros(0)

        # upstream values from other bmi run...
        self._values['upstream_id'] = np.zeros(0, dtype=int)
        self._values['upstream_fvd'] = np.zeros(0)

        # flattened fvd values to be used as offnetwork upstream flows
        # number of segments*3(flow, velocity, depth)*12(number of timesteps in 1 hour)
        #TODO: coordinate with model engine team on order of values here. this is a 1d array
        # that cycles through variables first, then time steps, then segment ids.
        self._values['fvd_results'] = np.zeros(0)
        self._values['fvd_index'] = np.zeros(0)
        self._values['lakeout'] = np.zeros(0)
        self._values['lakeout_index'] = np.zeros(0)
        self._values['q0'] = np.zeros(0)
        self._values['q0_index'] = np.zeros(0)
        self._values['t0'] = np.zeros(0)
        self._values['waterbody_df'] = np.zeros(0)
        self._values['waterbody_df_index'] = np.zeros(0)
        self._values['lastobs_df'] = np.zeros(0)
        self._values['lastobs_df_index'] = np.zeros(0)

        # Diffusive domain tailwater locations
        self._values['diffusive_tw_ids'] = np.zeros(0)
        self._values['diffusive_tw_latitude'] = np.zeros(0)
        self._values['diffusive_tw_longitude'] = np.zeros(0)
        
        # Data assimilation values
        self._values['usgs_df'] = np.zeros(0)
        self._values['reservoir_usgs_df'] = np.zeros(0)
        self._values['reservoir_usace_df'] = np.zeros(0)
        self._values['rfc_timeseries_df'] = np.zeros(0)
        self._values['lastobs_df'] = np.zeros(0)
        self._values['nudging'] = np.zeros(0)
        self._values['nudging_ids'] = np.zeros(0)

        # Data assimilation for BMI compliant arrays containing data frames
        #
        # usgs_df
        self._values['datesSecondsArray_usgs'] = np.zeros(0)
        self._values['nDates_usgs'] = np.zeros(0)
        self._values['stationArray_usgs'] = np.zeros(0)
        self._values['stationStringLengthArray_usgs'] = np.zeros(0)
        self._values['nStations_usgs'] = np.zeros(0)
        self._values['usgs_Array'] = np.zeros(0)
        #
        # reservoir_usgs_df
        self._values['datesSecondsArray_reservoir_usgs'] = np.zeros(0)
        self._values['nDates_reservoir_usgs'] = np.zeros(0)
        self._values['stationArray_reservoir_usgs'] = np.zeros(0)
        self._values['stationStringLengthArray_reservoir_usgs'] = np.zeros(0)
        self._values['nStations_reservoir_usgs'] = np.zeros(0)
        self._values['usgs_reservoir_Array'] = np.zeros(0)
        #
        # reservoir_usace_df
        self._values['datesSecondsArray_reservoir_usace'] = np.zeros(0)
        self._values['nDates_reservoir_usace'] = np.zeros(0)
        self._values['stationArray_reservoir_usace'] = np.zeros(0)
        self._values['stationStringLengthArray_reservoir_usace'] = np.zeros(0)
        self._values['nStations_reservoir_usace'] = np.zeros(0)
        self._values['usace_reservoir_Array'] = np.zeros(0)
        #
        # RFC dataframe
        self._values['rfc_da_timestep'] = np.zeros(0)
        self._values['rfc_totalCounts'] = np.zeros(0)
        self._values['rfc_synthetic_values'] = np.zeros(0)
        self._values['rfc_discharges'] = np.zeros(0)
        self._values['rfc_timeseries_idx'] = np.zeros(0)
        self._values['rfc_use_rfc'] = np.zeros(0)
        self._values['rfc_Datetime'] = np.zeros(0)
        self._values['rfc_timeSteps'] = np.zeros(0)
        self._values['rfc_StationId_array'] = np.zeros(0)
        self._values['rfc_StationId_stringLengths'] = np.zeros(0)
        self._values['rfc_List_array'] = np.zeros(0)
        self._values['rfc_List_stringLengths'] = np.zeros(0)
        #
        # lastobs dataframe
        #self._values['lastObs_gageArray'] = np.zeros(0)
        #self._values['lastObs_gageStringLengths'] = np.zeros(0)
        #self._values['lastObs_timeSince'] = np.zeros(0)
        #self._values['lastObs_discharge'] = np.zeros(0)

        '''
        self._values['gage_crosswalk__segID'] = np.zeros(0, dtype=int)
        self._values['gage_crosswalk__gageID'] = np.zeros(0, dtype='<U19')
        self._values['usgs_timeslice_discharge'] = np.zeros(0)
        self._values['usgs_timeslice_stationId'] = np.zeros(0, dtype='<U19')
        self._values['usgs_timeslice_time'] = np.zeros(0, dtype='<U19')
        self._values['usgs_timeslice_discharge_quality'] = np.zeros(0)

        self._values['gages'] = np.zeros(0, dtype='<U19')
        self._values['usace_timeslice_discharge'] = np.zeros(0)
        self._values['usace_timeslice_stationId'] = np.zeros(0, dtype='<U19')
        self._values['usace_timeslice_time'] = np.zeros(0, dtype='<U19')
        self._values['usace_timeslice_discharge_quality'] = np.zeros(0)

        self._values['lastobs_discharge'] = np.zeros(0)
        self._values['lastobs_stationIdInd'] = np.zeros(0, dtype=int)
        self._values['lastobs_timeInd'] = np.zeros(0, dtype=int)
        self._values['lastobs_stationId'] = np.zeros(0, dtype='<U19')
        self._values['lastobs_time'] = np.zeros(0, dtype='<U19')
        self._values['lastobs_modelTimeAtOutput'] = np.zeros(0, dtype='<U19')

        self._values['rfc_discharges'] = np.zeros(0)
        self._values['rfc_stationId'] = np.zeros(0, dtype='<U19')
        self._values['rfc_synthetic_values'] = np.zeros(0)
        self._values['rfc_totalCounts'] = np.zeros(0)
        self._values['rfc_datetime'] = np.zeros(0, dtype='<U19')
        self._values['rfc_timestep'] = np.zeros(0)
        '''

    # Currently utilized BMI functions:
    def update(self):
        """Advance model by one time step."""
        
        self.update_until(self._model._time_step)

    def update_until(self, until):
        """Update model until a particular time.
        Parameters
        ----------
        until : int
            Time to run model until in seconds.
        """
        if self._model._time==0.0:
            self._model.preprocess_static_vars(self._values) 
            
        self._model.run(self._values, until)

    def set_value(self, var_name, src):
        """
        Set model values
        
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        #val = self.get_value_ptr(var_name)
        #val[:] = src.reshape(val.shape)
        
        self._values[var_name] = src

    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        output_df : pd.DataFrame
            Copy of values.
        """
        output_df = self.get_value_ptr(var_name)
        return output_df

    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        return self._model._time

    def get_time_step(self):
        return self._model._time_step

    def get_time_units(self):
        return self._time_units

    def finalize(self):
        """Finalize model."""

        self._model = None

    # BMI functions that are not being used yet...
    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        time_step = self.get_time_step()
        self._model.time_step = time_frac * time_step
        self.update()
        self._model.time_step = time_step

    def get_var_type(self, var_name):
        """Data type of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]

    def get_var_grid(self, var_name):
        """Grid id for a variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Grid id.
        """
        for grid_id, var_name_list in self._grids.items():
            if var_name in var_name_list:
                return grid_id

    def get_grid_rank(self, grid_id):
        """Rank of grid.
        Parameters
        ----------
        grid_id : int
            Identifier of a grid.
        Returns
        -------
        int
            Rank of grid.
        """
        return len(self._model.shape)

    def get_grid_size(self, grid_id):
        """Size of grid.
        Parameters
        ----------
        grid_id : int
            Identifier of a grid.
        Returns
        -------
        int
            Size of grid.
        """
        return int(np.prod(self._model.shape))

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        var_name = self._grids[grid_id][0]
        shape[:] = self.get_value_ptr(var_name).shape
        return shape

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        spacing[:] = self._model.spacing
        return spacing

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        origin[:] = self._model.origin
        return origin

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return self._grid_type[grid_id]

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        """Number of grid nodes.
        Parameters
        ----------
        grid : int
            Identifier of a grid.
        Returns
        -------
        int
            Size of grid.
        """
        return self.get_grid_size(grid)

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("get_grid_z")
        
    def _parse_config(self, cfg):
        cfg_list = [cfg.get('flag'),cfg.get('file')]
        return cfg_list

# Helper functions
def _read_config_file(custom_input_file):
    '''

    '''
    with open(custom_input_file) as custom_file:
        data = yaml.load(custom_file, Loader=yaml.SafeLoader)

    bmi_parameters = data.get("bmi_parameters", None)
    return bmi_parameters
