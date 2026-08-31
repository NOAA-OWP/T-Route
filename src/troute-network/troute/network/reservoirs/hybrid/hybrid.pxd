cimport numpy as np
"""
Declaring C types for Hybrid Class variables and functions
"""
from troute.network.reach cimport Reach, compute_type

############ Other Reservoir Interface ############
cdef void run_hybrid_c(_Reach* reach, float inflow, float lateral_inflow, float routing_period, float* outflow, float* water_elevation) nogil

cdef extern from "hybrid_structs.h":
  ctypedef struct _MC_Hybrid:
    long lake_number
    float dam_length, area, max_depth
    float orifice_area, orifice_coefficient, orifice_elevation
    float weir_coefficient, weir_elevation, weir_length
    float initial_fractional_depth, water_elevation
    int reservoir_type
    char* reservoir_parameter_file
    char* start_date
    char* usgs_timeslice_path
    char* usace_timeslice_path
    int observation_lookback_hours
    int observation_update_time_interval_seconds
  ctypedef struct _Reach:
    pass

cdef class MC_Hybrid(Reach):
  """
  C type for MC_Hybrid which is a reservoir subclass of a Reach
  """
  cpdef (float,float) run(self, float inflow, float lateral_inflow, float routing_period)
