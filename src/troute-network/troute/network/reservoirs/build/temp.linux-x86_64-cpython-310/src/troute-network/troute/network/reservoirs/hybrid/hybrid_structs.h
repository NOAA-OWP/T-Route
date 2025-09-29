#ifndef HYBRID_STRUCTS_H
#define HYBRID_STRUCTS_H
/*
    C Structures
*/
#include "../../reach_structs.h"

#define HYBRID_MAX_PATH_LENGTH 256
#define HYBRID_MAX_START_DATE_LENGTH 20 

typedef struct {
  int lake_number;
  float dam_length, area, max_depth;
  float orifice_area, orifice_coefficient, orifice_elevation;
  float weir_coefficient, weir_elevation, weir_length;
  float initial_fractional_depth, water_elevation;
  int reservoir_type;
  char* reservoir_parameter_file;
  char* start_date;
  char* usgs_timeslice_path;
  char* usace_timeslice_path;
  int observation_lookback_hours;
  int observation_update_time_interval_seconds;  

  //Handle to operate hybrid res
  void* handle;
} _MC_Hybrid;

#endif //HYBRID_STRUCTS_H
