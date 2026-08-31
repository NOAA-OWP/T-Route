cimport numpy as np
"""
Declaring C types for Level Pool Class variables and functions
"""
from troute.network.reach cimport Reach, compute_type

############ Other Reservoir Interface ############
cdef void run_lp_c(_Reach* reach, float inflow, float lateral_inflow, float routing_period, float* outflow,  float* water_elevation) nogil

cdef void update_lp_c(_Reach* reach, float updated_elevation, float* water_elevation) nogil

cdef extern from "levelpool_structs.h":
  ctypedef struct _MC_Levelpool:
    long lake_number
    float dam_length, area, max_depth
    float orifice_area, orifice_coefficient, orifice_elevation
    float weir_coefficient, weir_elevation, weir_length
    float initial_fractional_depth, water_elevation
    int wbody_type_code
  ctypedef struct _Reach:
    pass

cdef class MC_Levelpool(Reach):
  """
  C type for MC_Levelpool which is a resevoir subclass of a Reach
  """
  cpdef (float,float) run(self, float inflow, float lateral_inflow, float routing_period)
    
  cpdef (float) assimilate_elevation(self, float updated_elevation)
