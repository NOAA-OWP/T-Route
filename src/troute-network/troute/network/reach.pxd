cimport numpy as np
"""
FIXME add some significant inline documentation
"""
cdef extern from "reach_structs.h":
  ctypedef struct _MC_Levelpool:
    long lake_number
    float dam_length, area, max_depth;
    float orifice_area, orifice_coefficient, orifice_elevation;
    float weir_coefficient, weir_elevation, weir_length;
    float initial_fractional_depth, water_elevation;
    int wbody_type_code
  ctypedef struct _MC_Hybrid:
    long lake_number
    float dam_length, area, max_depth;
    float orifice_area, orifice_coefficient, orifice_elevation;
    float weir_coefficient, weir_elevation, weir_length;
    float initial_fractional_depth, water_elevation;
    int reservoir_type;
    char[256] reservoir_parameter_file;
    char[19] start_date;
    char[256] usgs_timeslice_path;
    char[256] usace_timeslice_path;
    int observation_lookback_hours;
    int observation_update_time_interval_seconds;
  ctypedef struct _MC_RFC:
    long lake_number
    float dam_length, area, max_depth;
    float orifice_area, orifice_coefficient, orifice_elevation;
    float weir_coefficient, weir_elevation, weir_length;
    float initial_fractional_depth, water_elevation;
    int reservoir_type;
    char[256] reservoir_parameter_file;
    char[19] start_date;
    char[256] time_series_path;
    int forecast_lookback_hours;
  ctypedef struct _MC_Reach:
    int num_segments
  ctypedef struct _MC_Levelpool:
    pass
  ctypedef struct _MC_Hybrid:
    pass
  ctypedef struct _MC_RFC:
    pass
  ctypedef union _ReachUnion:
    _MC_Reach mc_reach;
    _MC_Levelpool lp;
    _MC_Hybrid hybrid;
    _MC_RFC rfc;
  ctypedef struct _Reach:
    _ReachUnion reach
    int _num_segments;
    long* _upstream_ids;
    int _num_upstream_ids;
    int type;
    long id;

ctypedef enum compute_type:
  MC_REACH, RESERVOIR_LP, RESERVOIR_HYBRID, RESERVOIR_RFC

#TODO implement junction or make multiple upstreams
cdef class Segment():
  """
    A Single routing segment
  """
  cdef readonly long id
  cdef long upstream_id
  cdef long downstream_id

cdef class Reach():
  #Keep a python list of ids only for pre/post processing and diagnostics
  cdef readonly long[::1] to_ids
  cdef int _num_upstream_ids
  cdef _Reach _reach
  cdef compute_type _type
  #cdef readonly np.ndarray upstream_ids
