cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

from troute.network.reach cimport compute_type, _Reach
"""
Externally defined symbols
"""

cdef extern from "hybrid_structs.c":
  void init_hybrid_reach(_Reach* reach, long lake_number,
                            float dam_length, float area, float max_depth,
                            float orifice_area, float orifice_coefficient, float orifice_elevation,
                            float weir_coefficient, float weir_elevation, float weir_length,
                            float initial_fractional_depth, float water_elevation,
                            int reservoir_type, char *reservoir_parameter_file, char *start_date,
                            char *usgs_timeslice_path, char *usace_timeslice_path,
                            int observation_lookback_hours,
                            int observation_update_time_interval_seconds
  )
  void free_hybrid_reach(_Reach* reach)
  void route(_Reach* reach, float routing_period, float inflow, float lateral_inflow, float* outflow,  float* water_elevation) nogil

cdef void run_hybrid_c(_Reach* reach, float inflow, float lateral_inflow, float routing_period, float* outflow, float* water_elevation) nogil:
    route(reach, inflow, lateral_inflow, routing_period, outflow, water_elevation)

cdef class MC_Hybrid(Reach):
  """
    MC_Reservoir is a subclass of MC_Reach_Base_Class
  """

  def __init__(self, long id, long lake_number, long[::1] upstream_ids, args, reservoir_type,
              reservoir_parameter_file, start_date, usgs_timeslice_path, usace_timeslice_path,
              observation_lookback_hours, observation_update_time_interval_seconds):

    """
      Construct the kernel based on passed parameters,
      which only constructs the parent class

      Params:
        id: long
          unique identity of the reach this reservoir represents
        lake_number: int TODO (long?)
          WRF_Hydro lake number of this reservoir
        upstream_ids: array[long]
          buffer/array of upstream identifiers which contribute flow to this reservoir
        args: list
          the hybrid parameters ordered as follows:
            area = args[0]
            max_depth = args[1]
            orifice_area = args[2]
            orifice_coefficient = args[3]
            orifice_elevation  =  args[4]
            weir_coefficient = args[5]
            weir_elevation = args[6]
            weir_length = args[7]
            initial_fractional_depth  = args[8]
            water_elevation = args[10]
            reservoir_type
            reservoir_parameter_file
            start_date
            usgs_timeslice_path
            usace_timeslice_path
            observation_lookback_hours
            observation_update_time_interval_seconds
    """
    super().__init__(id, upstream_ids, compute_type.RESERVOIR_HYBRID)
    # Note Some issues with __calloc__:
    # The python type isn't guaranteed to be properly constructed, so cannot depend on super class being constructured.
    # Thus I don't think we can put these C init functions in __calloc__, at least not in all cases.
    # init the backing struct, pass a dam_length of 10.0 for now

    #Setting default dam_length to 10
    dam_length = 10.0
    area = args[0]
    max_depth = args[1]
    orifice_area = args[2]
    orifice_coefficient = args[3]
    orifice_elevation  =  args[4]
    weir_coefficient = args[5]
    weir_elevation = args[6]
    weir_length = args[7]
    initial_fractional_depth = args[8]
    water_elevation = args[10]

    #Check lengths of input strings to ensure that they do not exceed buffer size
    if (len(reservoir_parameter_file) > 256):
       raise ValueError("reservoir_parameter_file path is too large. Length must be less than or equal to 256 characters.")
   
    if (len(start_date) > 19):
       raise ValueError("start_date is too large. Length must be less than or equal to 19 characters.")

    if (len(usgs_timeslice_path) > 256):
       raise ValueError("usgs_timeslice_path is too large. Length must be less than or equal to 256 characters.")
   
    if (len(usace_timeslice_path) > 256):
       raise ValueError("usace_timeslice_path is too large. Length must be less than or equal to 256 characters.")

    init_hybrid_reach(&self._reach, lake_number,
                         dam_length, area, max_depth, orifice_area,
                         orifice_coefficient, orifice_elevation,
                         weir_coefficient, weir_elevation,
                         weir_length, initial_fractional_depth,
                         water_elevation, reservoir_type,
                         reservoir_parameter_file.encode('utf-8'),
                         start_date.encode('utf-8'),
                         usgs_timeslice_path.encode('utf-8'),
                         usace_timeslice_path.encode('utf-8'),
                         observation_lookback_hours,
                         observation_update_time_interval_seconds)
                         

  def __dealloc__(self):
    """
    """
    free_hybrid_reach(&self._reach)

  cpdef (float,float) run(self, float inflow, float lateral_inflow, float routing_period):
    """
      Run the hybrid routing function
      Params:
        inflow: float
          inflow into the reservoir
        lateral_inflow: float
          lateral flows into the reservoir
        routing_period: float
          amount of time to simulatie reservoir operation for, outflow if valid until this time
      Return:
        outflow: float
          flow rate out of the reservoir valid for routing_period seconds
        water_elevation:
          reservoir water surface elevation after routing_period seconds
    """
    cdef float outflow = 0.0
    cdef float water_elevation = 0.0
    with nogil:
      route(&self._reach, inflow, lateral_inflow, routing_period, &outflow, &water_elevation)
      #printf("outflow: %f\n", outflow)
      return outflow, water_elevation

  @property
  def water_elevation(self):
    """
      Reservoir water surface elevation
    """
    return self._reach.reach.hybrid.water_elevation

  @property
  def lake_area(self):
    """
      Surface area of the reservoir
    """
    return self._reach.reach.hybrid.area

  @property
  def weir_elevation(self):
    """
      Elevation, in meters, of the bottom of the weir
    """
    return self._reach.reach.hybrid.weir_elevation

  @property
  def weir_coefficient(self):
    """
      Weir coefficient
    """
    return self._reach.reach.hybrid.weir_coefficient

  @property
  def weir_length(self):
    """
      Length of the weir, in meters
    """
    return self._reach.reach.hybrid.weir_length

  @property
  def dam_length(self):
    """
      Length of the dam, in meters
    """
    return self._reach.reach.hybrid.dam_length

  @property
  def orifice_elevation(self):
    """
      Elevation, in meters, of the orifice flow component
    """
    return self._reach.reach.hybrid.orifice_elevation

  @property
  def orifice_area(self):
    """
      Area of the orifice flow component, in square meters
    """
    return self._reach.reach.hybrid.orifice_area

  @property
  def max_depth(self):
    """
      Maximum water elevaiton, in meters, before overflow occurs
    """
    return self._reach.reach.hybrid.max_depth

  @property
  def lake_number(self):
    """
      WRF Hydro lake identifier
    """
    return self._reach.reach.hybrid.lake_number

  @property
  def initial_fractional_depth(self):
    """
      Initial water surface elevation, as a percentage of total capacity,
      to use if initial water elevation is unknown.
    """
    return self._reach.reach.hybrid.initial_fractional_depth

  @property
  def reservoir_type(self):
    """
      Reservoir Type as in 1 for Level Pool, 2 for Persistence-Hybrid-USGS,
      3 for Persistence-Hybrid-USACE, or 4 for RFC
    """
    return self._reach.reach.hybrid.reservoir_type

  @property
  def reservoir_parameter_file(self):
    """
      Reservoir Parameter File with path
    """
    return self._reach.reach.hybrid.reservoir_parameter_file

  @property
  def start_date(self):
    """
      Start date and time of simulation
    """
    return self._reach.reach.hybrid.start_date

  @property
  def usgs_timeslice_path(self):
    """
      Path for USGS timeslice files
    """
    return self._reach.reach.hybrid.usgs_timeslice_path

  @property
  def usace_timeslice_path(self):
    """
      Path for USACE timeslice files
    """
    return self._reach.reach.hybrid.usace_timeslice_path

  @property
  def observation_lookback_hours(self):
    """
      Number of hours to look back in time for a timeslice file
    """
    return self._reach.reach.hybrid.observation_lookback_hours

  @property
  def observation_update_time_interval_seconds(self):
    """
      Time interval in seconds between updates to observations
      from a new timeslice file
    """
    return self._reach.reach.hybrid.observation_update_time_interval_seconds
