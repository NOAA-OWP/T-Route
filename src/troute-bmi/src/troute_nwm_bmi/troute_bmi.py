"""Basic Model Interface implementation for NGEN t-route."""
from __future__ import annotations
import pickle
import typing
import numpy as np
from bmipy import Bmi
import logging

from ewts.helper import getenv_any
from ewts.logger import configure_existing_logger

from .troute_model import Model, BmiVars

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

LOG = logging.getLogger("TROUTE")

_VAR_NAME_UNITS_MAP = {
    BmiVars.CATCHMENT_VALUE: ['streamflow_cms', 'm3 s-1'],
    BmiVars.NEXUS_VALUE: ['streamflow_cms', 'm3 s-1'],
    BmiVars.CHANNEL_WATER_RATE: ['streamflow_cms', 'm3 s-1'],
    BmiVars.CHANNEL_WATER_SPEED: ['streamflow_ms', 'm s-1'],
    BmiVars.CHANNEL_WATER_DEPTH: ['streamflow_m', 'm'],
    BmiVars.LAKE_WATER_INCOMING: ['waterbody_cms', 'm3 s-1'],
    BmiVars.LAKE_WATER_OUTGOING: ['waterbody_cms', 'm3 s-1'],
    BmiVars.LAKE_WATER_ELEVATION: ['waterbody_m', 'm'],
}

_OUTPUT_VAR_NAMES = [
    BmiVars.CHANNEL_WATER_ID,
    BmiVars.CHANNEL_WATER_RATE,
    BmiVars.CHANNEL_WATER_SPEED,
    BmiVars.CHANNEL_WATER_DEPTH,
    BmiVars.LAKE_WATER_ID,
    BmiVars.LAKE_WATER_INCOMING,
    BmiVars.LAKE_WATER_OUTGOING,
    BmiVars.LAKE_WATER_ELEVATION,
]

_INPUT_VAR_NAMES = [
    BmiVars.CATCHMENT_ID,
    BmiVars.CATCHMENT_VALUE,
    BmiVars.NEXUS_ID,
    BmiVars.NEXUS_VALUE,
    BmiVars.UPSTREAM_ID,
    BmiVars.NGEN_DT,
]


class BmiTroute(Bmi):
    _model: Model

    def __init__(self):
        val = getenv_any("EWTS_USE_NGEN_BRIDGE", "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            configure_existing_logger(LOG)

        super().__init__()
        self._values: dict[str, NDArray] = {
            BmiVars.NGEN_DT: np.array([-1], dtype=np.intc),
            BmiVars.NEXUS_ID: np.zeros(0, dtype=np.intc),
            BmiVars.NEXUS_VALUE: np.zeros(0, dtype=np.double),
            BmiVars.CATCHMENT_ID: np.zeros(0, dtype=np.intc),
            BmiVars.CATCHMENT_VALUE: np.zeros(0, dtype=np.double),
            BmiVars.UPSTREAM_ID: np.zeros(0, dtype=int),
            BmiVars.CHANNEL_WATER_ID: np.zeros(0, dtype=np.int64),
            BmiVars.CHANNEL_WATER_RATE: np.zeros(0, dtype=np.float32),
            BmiVars.CHANNEL_WATER_SPEED: np.zeros(0, dtype=np.float32),
            BmiVars.CHANNEL_WATER_DEPTH: np.zeros(0, dtype=np.float32),
            BmiVars.LAKE_WATER_ID: np.zeros(0, dtype=np.int64),
            BmiVars.LAKE_WATER_INCOMING: np.zeros(0, dtype=np.float32),
            BmiVars.LAKE_WATER_OUTGOING: np.zeros(0, dtype=np.float32),
            BmiVars.LAKE_WATER_ELEVATION: np.zeros(0, dtype=np.float32),
        }
        self._free_serialized()

    def initialize(self, bmi_cfg_file):
        self._model = Model(bmi_cfg_file, self.get_start_time())

    def update(self):
        self._model.run(self._values)
        # clear current flow values
        for inputs_var in [BmiVars.NEXUS_VALUE, BmiVars.CATCHMENT_VALUE]:
            dtype = self._values[inputs_var].dtype
            self._values[inputs_var] = np.zeros(0, dtype=dtype)

    def update_until(self, time):
        if self._model.time < time:
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
        if var_name == "serialization_create":
            self._serialize()
        elif var_name == "serialization_state":
            self._deserialize(src)
        elif var_name == "serialization_free":
            self._free_serialized()
        elif var_name == "reset_time":
            self._model.reset_time()
        else:
            _log_set_value(var_name, src)
            var = self._values[var_name]
            if len(src) == len(var):
                var[:] = src
            else:
                self._values[var_name] = np.array(src, dtype=var.dtype, copy=True)

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
        return np.copy(self.get_value_ptr(var_name))

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
        if var_name == "serialization_state":
            return self._serialized
        if var_name == "serialization_size" or var_name == "serialization_create":
            return self._serialized_size
        return self._values[var_name]

    def get_start_time(self):
        """Start time of model."""
        return 0.0

    def get_end_time(self):
        """End time of model."""
        return float(self._model.ngen_dt(self._values) * (self._model.nts - 1))

    def get_current_time(self):
        return self._model.time

    def get_time_step(self):
        return self._model.ngen_dt(self._values)

    def get_time_units(self):
        return "s"

    def finalize(self):
        """Finalize model."""
        if self._model is not None:
            self._model.log_times()
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

    def get_var_type(self, var_name: str):
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
        if var_name == "serialization_free":
            return np.dtype(np.intc).name
        if var_name == "reset_time":
            return np.dtype(np.double).name
        return self.get_value_ptr(var_name).dtype.name

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

    def get_var_nbytes(self, var_name: str):
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
        if var_name == "serialization_state":
            return int(self._serialized_size[0])
        if var_name == "serialization_free" or var_name == "reset_time":
            return np.dtype(self.get_var_type(var_name)).itemsize
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        # needs to go through `get_var_type` for names without stored values
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return "node"

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

    def _serialize(self):
        data = {
            "values": self._values,
            "model": self._model.create_state(),
        }
        # HIGHEST_PROTOCOL recommended for pickling pandas DataFrames
        serialized = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        self._serialized = np.array(
            bytearray(serialized), dtype=self._serialized.dtype
        )
        self._serialized_size[0] = len(self._serialized)

    def _deserialize(self, data):
        deserialized = pickle.loads(bytes(data))
        self._values = deserialized["values"]
        self._model.load_state(deserialized["model"])
        self._free_serialized()

    def _free_serialized(self):
        self._serialized = np.zeros(0, dtype=np.uint8)
        self._serialized_size = np.zeros(1, dtype=np.uint64)


def _log_set_value(var_name: str, array: np.ndarray):
    size = array.size
    if size > 5:
        array = array[:4]
        end = ", ..."
    else:
        end = ""
    msg = f"Setting {var_name} with {size} elements: [{', '.join(map(str, array))}{end}]"
    LOG.debug(msg)
