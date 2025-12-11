"""Basic Model Interface implementation for NGEN t-route."""
from __future__ import annotations
import numpy as np
from bmipy import Bmi

from .troute_model import Model

_COUNT_SUFFIX = "__count"

_VAR_NAME_UNITS_MAP = {
    'land_surface_water_source__volume_flow_rate': ['streamflow_cms', 'm3 s-1'],
}

_OUTPUT_VAR_NAMES = []

_INPUT_VAR_NAMES = [
    "land_surface_water_source__id",
    "land_surface_water_source__volume_flow_rate",
    "upstream_id",
]


class BmiTroute(Bmi):
    _model: Model
    _values: dict[str, np.ndarray]

    def __init__(self):
        super().__init__()
        self._values = {
            "land_surface_water_source__id": np.zeros(0, dtype=np.intc),
            "land_surface_water_source__id" + _COUNT_SUFFIX: np.zeros(1, dtype=np.int64),
            "land_surface_water_source__volume_flow_rate": np.zeros(0, dtype=float),
            "land_surface_water_source__volume_flow_rate" + _COUNT_SUFFIX: np.zeros(1, dtype=np.int64),
            "upstream_id": np.zeros(0, dtype=int),
        }
        self._var_loc = "node"
        self._var_grid_id = 0
        self._time_units = "s"
        self._start_time = 0.0
        self._end_time = np.finfo("d").max

    def initialize(self, bmi_cfg_file):
        self._model = Model(bmi_cfg_file)

    def update(self):
        self._model.update(self._values)
    
    def update_until(self, time):
        while self._model._time < time:
            self.update()

    def set_value(self, var_name: str, src):
        """
        Set model values
        
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        # special case for changing data size 
        if var_name.endswith(_COUNT_SUFFIX):
            source_var_name = var_name[:-len(_COUNT_SUFFIX)]
            source = self._values.get(source_var_name)
            self._values[source_var_name] = np.resize(source, int(src[0]))
        self._values[var_name][:] = src

    def get_value(self, var_name: str):
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
        return np.copy(self._values[var_name])

    def get_value_ptr(self, var_name: str):
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
        return self._model.time

    def get_time_step(self):
        return self._model.dt

    def get_time_units(self):
        return self._time_units

    def finalize(self):
        """Finalize model."""
        if self._model is not None:
            self._model.run(self._values)
            self._model = None

    # BMI functions that are not being used yet...
    def update_frac(self, time_frac: float):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        time_step = self.get_time_step()
        self._model.dt = int(time_frac * time_step)
        if self._model.dt > 0:
            self.update()
        self._model.dt = time_step

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
        return str(self.get_value(var_name).dtype)

    def get_var_units(self, var_name: str):
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
        return _VAR_NAME_UNITS_MAP[var_name][1]

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
        return self.get_value(var_name).nbytes

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
        raise NotImplementedError("get_var_grid")

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
        raise NotImplementedError("get_grid_rank")

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
        raise NotImplementedError("get_grid_size")

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
        return "T-Route"

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(_INPUT_VAR_NAMES)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(_OUTPUT_VAR_NAMES)

    def get_input_var_names(self):
        """Get names of input variables."""
        return _INPUT_VAR_NAMES

    def get_output_var_names(self):
        """Get names of output variables."""
        return _OUTPUT_VAR_NAMES

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        raise NotImplementedError("get_grid_shape")

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        raise NotImplementedError("get_grid_spacing")

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        raise NotImplementedError("get_grid_origin")

    def get_grid_type(self, grid_id):
        """Type of grid."""
        raise NotImplementedError("get_grid_type")

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
