"""Basic Model Interface implementation for t-route's data assimilation forcing."""

import numpy as np
from bmipy import Bmi
from pathlib import Path

# Here is the model we want to run
from model_DAforcing import DAforcing_model

class bmi_DAforcing(Bmi):

    def __init__(self):
        """Create a Bmi DA forcing model that is ready for initialization."""
        super(bmi_DAforcing, self).__init__()
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
        'model_name':         'DA forcing for Next Generation NWM',
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
        'flowvelocitydepth',
        'flowvelocitydepth_ids',
        'nudging',
        'nudging_ids'
    ]

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = [
        'usgs_df',
        'reservoir_usgs_df',
        'reservoir_usace_df',
        'rfc_timeseries_df',
        'lastobs_df'
    ]

    #---------------------------------------------
    # BMI variable output namesnames (CSDMS standard names)
    #---------------------------------------------
    # USGS DF
    _BMI_output_usgs = [
        'datesSecondsArray_usgs',
        'nDates_usgs',
        'stationArray_usgs',
        'stationStringLengthArray_usgs',
        'nStations_usgs',
        'usgs_Array'
    ]    
    
    # Reservoir USGS DF
    _BMI_output_reservoir_usgs = [
        'datesSecondsArray_reservoir_usgs',
        'nDates_reservoir_usgs',
        'stationArray_reservoir_usgs',
        'stationStringLengthArray_reservoir_usgs',
        'nStations_reservoir_usgs',
        'reservoir_usgs_Array'
    ]

    # Reservoir USACE DF
    _BMI_output_reservoir_usace = [
        'datesSecondsArray_reservoir_usace',
        'nDates_reservoir_usace',
        'stationArray_reservoir_usace',
        'stationStringLengthArray_reservoir_usace',
        'nStations_reservoir_usace',
        'reservoir_usace_Array'
    ]

    # RFC DF
    _BMI_output_reservoir_rfc = [
        'rfc_da_timestep',
        'rfc_totalCounts',
        'rfc_synthetic_values',
        'rfc_discharges',
        'rfc_timeseries_idx',
        'rfc_use_rfc',
        'rfc_Datetime',
        'rfc_da_timestep',
        'rfc_timeSteps',
        'rfc_issue_time',
        'rfc_StationId_array',
        'rfc_StationId_stringLengths',
        'rfc_List_array',
        'rfc_List_stringLengths'
    ]

    # lastobs DF
    _BMI_output_lastobs = [
        'lastObs_gageArray',
        'lastObs_gageStringLengths',
        'lastObs_timeSince',
        'lastObs_discharge'
    ]

    # lite restart dataframes (q0, waterbody_df)
    _BMI_lite_restart = [
        'q0_columnArray',
        'q0_columnLengthArray',
        'q0_nCol',
        'q0_indexArray',
        'q0_nIndex',
        'q0_Array',
        'waterbodyLR_columnArray',
        'waterbodyLR_columnLengthArray',
        'waterbodyLR_nCol',
        'waterbodyLR_indexArray',
        'waterbodyLR_nIndex',
        'waterbodyLR_Array'
    ]

    # BMI support variables
    _BMI_support_vars = [
        'dateNull'
    ]

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    #------------------------------------------------------
    #TODO update all these...
    _var_name_units_map = {
        #'channel_exit_water_x-section__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'channel_water_flow__speed':['streamflow_ms','m s-1'],
        #'channel_water__mean_depth':['streamflow_m','m'],
        #'lake_water~incoming__volume_flow_rate':['waterbody_cms','m3 s-1'],
        #'lake_water~outgoing__volume_flow_rate':['waterbody_cms','m3 s-1'],
        #'lake_surface__elevation':['waterbody_m','m'],
        #--------------   Dynamic inputs --------------------------------
        #'land_surface_water_source__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'coastal_boundary__depth':['depth_m', 'm'],
        #'usgs_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'reservoir_usgs_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'reservoir_usace_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'rfc_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'lastobs__volume_flow_rate':['streamflow_cms','m3 s-1']
        # TODO: RFC unit map
        'waterbody__type_number': ['',''],
        'waterbody__lake_number': ['','string'],  #'lake_number':['',''],
        'waterbody_rfc__use_flag': ['use_RFC','boolean'],     #'use_RFC':['',''], 
        'waterbody_rfc__observed_volume_flow_rate': ['streamflow_cms in timeseries', 'm3 s-1'], # 'rfc_timeseries_discharges':['streamflow_cms','m3 s-1'], 
        'waterbody_rfc__timeseries_index': [' ',' '], #'rfc_timeseries_idx':['time_step_count',''], 
        'waterbody_rfc__timeseries_update_time': ['time', 'sec'],      #'rfc_timeseries_update_time':['time','s'], 
        'waterbody_rfc__da_time_step': ['', 'sec'],  #'rfc_da_time_step':['time_step','s'], 
        'waterbody_rfc__total_count': ['','int'],   #'rfc_total_counts':['count',''],
        'waterbody_rfc__file_of_observed_volume_flow_rate': ['',''],  #'rfc_timeseries_file':['',''],
        'usace_timeslice_discharge':['streamflow_cms','m3 s-1'],
        'usace_timeslice_stationId':['',''],
        'usace_timeslice_time':['time',''],
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
        if bmi_cfg_file:
            bmi_cfg_file = Path(bmi_cfg_file)

        # ------------- Initialize t-route model ------------------------------#
        self._model = DAforcing_model(bmi_cfg_file) 

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for \
                                         long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for \
                                          long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for \
                                          long_name in self._var_name_units_map.keys()}
        

        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        self._values['usgs_df'] = self._model._usgs_df
        self._values['reservoir_usgs_df'] = self._model._reservoir_usgs_df
        self._values['reservoir_usace_df'] = self._model._reservoir_usace_df
        self._values['rfc_timeseries_df'] = self._model._rfc_timeseries_df
        
        self._values['write_lite_restart'] = self._model._write_lite_restart
        if not self._model._lastobs_df.empty:
            self._values['lastobs_df'] = self._model._lastobs_df.values.flatten()
            self._values['lastobs_df_ids'] = self._model._lastobs_df.index
        else:
            self._values['lastobs_df'] = np.zeros(0)
            self._values['lastobs_df_ids'] = np.zeros(0)
        
        if not self._model._q0.empty:
            self._values['q0'] = self._model._q0.values.flatten()
            self._values['q0_ids'] = self._model._q0.index
            self._values['t0'] = self._model._t0
        else:
            self._values['q0'] = np.zeros(0)
            self._values['q0_ids'] = np.zeros(0)
            self._values['t0'] = np.zeros(0)
        
        if not self._model._waterbody_df.empty:
            self._values['waterbody_df'] = self._model._waterbody_df.values.flatten()
            self._values['waterbody_df_ids'] = self._model._waterbody_df.index
        else:
            self._values['waterbody_df'] = np.zeros(0)
            self._values['waterbody_df_ids'] = np.zeros(0)

        #
        # Initialize BMI transport arrays
        #
        # Auxiliary variables
        self._values['dateNull'] = self._model._dateNull
        #
        # USGS dataframe converted
        self._values['datesSecondsArray_usgs'] = self._model._datesSecondsArray_usgs
        self._values['nDates_usgs'] = self._model._nDates_usgs
        self._values['stationArray_usgs'] = self._model._stationArray_usgs
        self._values['stationStringLengthArray_usgs'] = self._model._stationStringLengthArray_usgs
        self._values['nStations_usgs'] = self._model._nStations_usgs
        self._values['usgs_Array'] = self._model._usgsArray
        #
        # USGS reservoir dataframe converted
        self._values['datesSecondsArray_reservoir_usgs'] = self._model._datesSecondsArray_reservoir_usgs
        self._values['nDates_reservoir_usgs'] = self._model._nDates_reservoir_usgs
        self._values['stationArray_reservoir_usgs'] = self._model._stationArray_reservoir_usgs
        self._values['stationStringLengthArray_reservoir_usgs'] = self._model._stationStringLengthArray_reservoir_usgs
        self._values['nStations_reservoir_usgs'] = self._model._nStations_reservoir_usgs
        self._values['usgs_reservoir_Array'] = self._model._reservoirUsgsArray
        #
        # USACE reservoir dataframe converted        
        self._values['datesSecondsArray_reservoir_usace'] = self._model._datesSecondsArray_reservoir_usace
        self._values['nDates_reservoir_usace'] = self._model._nDates_reservoir_usace
        self._values['stationArray_reservoir_usace'] = self._model._stationArray_reservoir_usace
        self._values['stationStringLengthArray_reservoir_usace'] = self._model._stationStringLengthArray_reservoir_usace
        self._values['nStations_reservoir_usace'] = self._model._reservoirUsaceArray
        self._values['usace_reservoir_Array'] = self._model._reservoirUsaceArray
        #
        # RFC timeseries dataframe converted
        self._values['rfc_da_timestep'] = self._model._rfc_da_timestep
        self._values['rfc_totalCounts'] = self._model._rfc_totalCounts
        self._values['rfc_synthetic_values'] = self._model._rfc_synthetic_values
        self._values['rfc_discharges'] = self._model._rfc_discharges
        self._values['rfc_timeseries_idx'] = self._model._rfc_timeseries_idx
        self._values['rfc_use_rfc'] = self._model._rfc_use_rfc
        self._values['rfc_Datetime'] = self._model._rfc_Datetime
        self._values['rfc_timeSteps'] = self._model._rfc_timeSteps
        self._values['rfc_issue_time'] = self._model._rfc_issue_time
        self._values['rfc_StationId_array'] = self._model._rfc_StationId_array
        self._values['rfc_StationId_stringLengths'] = self._model._rfc_StationId_stringLengths
        self._values['rfc_List_array'] = self._model._rfc_List_array
        self._values['rfc_List_stringLengths'] = self._model._rfc_List_stringLengths    
        #
        # lastobs dataframe
        self._values['lastObs_gageArray'] = self._model._lastObs_gageArray
        self._values['lastObs_gageStringLengths'] = self._model._lastObs_gageStringLengths
        self._values['lastObs_timeSince'] = self._model._lastObs_timeSince
        self._values['lastObs_discharge'] = self._model._lastObs_discharge
        #
        # lite restart dataframes
        # q0    
        self._values['q0_columnArray'] = self._model._q0_columnArray
        self._values['q0_columnLengthArray'] = self._model._q0_columnLengthArray
        self._values['q0_nCol'] = self._model._q0_nCol   
        self._values['q0_indexArray'] = self._model._q0_indexArray
        self._values['q0_nIndex'] = self._model._q0_nIndex 
        self._values['q0_Array'] = self._model._q0_Array
        #
        # waterbody_df
        self._values['waterbodyLR_columnArray'] = self._model._waterbodyLR_columnArray
        self._values['waterbodyLR_columnLengthArray'] = self._model._waterbodyLR_columnLengthArray    
        self._values['waterbodyLR_nCol'] = self._model._waterbodyLR_nCol 
        self._values['waterbodyLR_indexArray'] = self._model._waterbodyLR_indexArray
        self._values['waterbodyLR_nIndex'] = self._model._waterbodyLR_nIndex 
        self._values['waterbodyLR_Array'] = self._model._waterbodyLR_Array


        self._values['flowvelocitydepth'] = np.zeros(0)
        self._values['flowvelocitydepth_ids'] = np.zeros(0)
        self._values['lakeout'] = np.zeros(0)
        self._values['lakeout_ids'] = np.zeros(0)
        self._values['nudging'] = np.zeros(0)
        self._values['nudging_ids'] = np.zeros(0)
        self._values['t-route_model_time'] = np.zeros(0)

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
        '''
        output_df = self.get_value_ptr(var_name)
        return output_df
        '''
        return self._values[var_name]

    def update(self):
        """Advance model by one time step."""
        self._model.run(self._values)

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

    def update_until(self, until):
        """Update model until a particular time.
        Parameters
        ----------
        until : int
            Time to run model until in seconds.
        """
        '''
        n_steps = int(until/self._model._time_step)

        for _ in range(int(n_steps)):
            self.update()
        '''
        pass

    def finalize(self):
        """Finalize model."""

        self._model = None
    
    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        '''
        time_step = self.get_time_step()
        self._model.time_step = time_frac * time_step
        self.update()
        self._model.time_step = time_step
        '''
        pass

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