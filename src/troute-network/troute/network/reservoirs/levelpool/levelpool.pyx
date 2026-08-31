cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

from troute.network.reach cimport compute_type, _Reach
"""
Externally defined symbols
"""

cdef extern from "levelpool_structs.c":
  void init_levelpool_reach(_Reach* reach, long lake_number,
                            float dam_length, float area, float max_depth,
                            float orifice_area, float orifice_coefficient, float orifice_elevation,
                            float weir_coefficient, float weir_elevation, float weir_length,
                            float initial_fractional_depth, float water_elevation, int wbody_type_code
  )
  void free_levelpool_reach(_Reach* reach)

  void route(_Reach* reach, float routing_period, float inflow, float lateral_inflow, float* outflow,  float* water_elevation) nogil

  void update_elevation(_Reach* reach, float updated_elevation, float* water_elevation) nogil

cdef void run_lp_c(_Reach* reach, float inflow, float lateral_inflow, float routing_period, float* outflow,  float* water_elevation) nogil:
    route(reach, inflow, lateral_inflow, routing_period, outflow, water_elevation)
    
cdef void update_lp_c(_Reach* reach, float updated_elevation, float* water_elevation) nogil:
    update_elevation(reach, updated_elevation, water_elevation)

cdef class MC_Levelpool(Reach):
  """
    MC_Reservoir is a subclass of MC_Reach_Base_Class
  """

  def __init__(self, long id, long lake_number, long[::1] upstream_ids, args, wbody_type_code):
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
          the levelpool parameters ordered as follows:
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
    """
    super().__init__(id, upstream_ids, compute_type.RESERVOIR_LP)
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
    orifice_elevation = args[4]
    weir_coefficient = args[5]
    weir_elevation = args[6]
    weir_length = args[7]
    initial_fractional_depth = args[8]
    water_elevation = args[10]

    init_levelpool_reach(&self._reach, lake_number,
                         dam_length, area, max_depth,
                         orifice_area, orifice_coefficient, orifice_elevation,
                         weir_coefficient, weir_elevation, weir_length,
                         initial_fractional_depth, water_elevation, wbody_type_code)

  def __dealloc__(self):
    """
      Release pointers and resources used to construct a levelpool reach
    """
    free_levelpool_reach(&self._reach)

  cpdef (float,float) run(self, float inflow, float lateral_inflow, float routing_period):
    """
      Run the levelpool routing function

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
  
  cpdef (float) assimilate_elevation(self, float updated_elevation):
    """
      Update the water elevation state variable

      Params:
        updated_elevation: float
          water elevation after data assimilation has been performed
      
      Return:
        water_elevation: float
          water elevation after data assimilation has been performed
    """
    cdef float water_elevation = 0.0
    with nogil:
      update_elevation(&self._reach, updated_elevation, &water_elevation)
      return water_elevation

  @property
  def water_elevation(self):
    """
      Reservoir water surface elevation
    """
    return self._reach.reach.lp.water_elevation

  @property
  def lake_area(self):
    """
      Surface area of the reservoir
    """
    return self._reach.reach.lp.area

  @property
  def weir_elevation(self):
    """
      Elevation, in meters, of the bottom of the weir
    """
    return self._reach.reach.lp.weir_elevation

  @property
  def weir_coefficient(self):
    """
      Weir coefficient
    """
    return self._reach.reach.lp.weir_coefficient

  @property
  def weir_length(self):
    """
      Length of the weir, in meters
    """
    return self._reach.reach.lp.weir_length

  @property
  def dam_length(self):
    """
      Length of the dam, in meters
    """
    return self._reach.reach.lp.dam_length

  @property
  def orifice_elevation(self):
    """
      Elevation, in meters, of the orifice flow component
    """
    return self._reach.reach.lp.orifice_elevation

  @property
  def orifice_area(self):
    """
      Area of the orifice flow component, in square meters
    """
    return self._reach.reach.lp.orifice_area

  @property
  def max_depth(self):
    """
      Maximum water elevaiton, in meters, before overflow occurs
    """
    return self._reach.reach.lp.max_depth

  @property
  def lake_number(self):
    """
      WRF Hydro lake identifier
    """
    return self._reach.reach.lp.lake_number

  @property
  def initial_fractional_depth(self):
    """
      Initial water surface elevation, as a percentage of total capacity,
      to use if initial water elevation is unknown.
    """
    return self._reach.reach.lp.initial_fractional_depth
