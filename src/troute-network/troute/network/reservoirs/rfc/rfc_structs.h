#ifndef RFC_STRUCTS_H
#define RFC_STRUCTS_H
/*
    C Structures
*/
#include "../../reach_structs.h"

#define RFC_MAX_PATH_LENGTH 256
#define RFC_MAX_START_DATE_LENGTH 20

typedef struct {
  long lake_number;
  float dam_length, area, max_depth;
  float orifice_area, orifice_coefficient, orifice_elevation;
  float weir_coefficient, weir_elevation, weir_length;
  float initial_fractional_depth, water_elevation;
  int reservoir_type;
  char* reservoir_parameter_file;
  char* start_date;
  char* time_series_path;
  int forecast_lookback_hours;

  //Handle to operate rfc reservoir
  void* handle;
} _MC_RFC;

#endif //RFC_STRUCTS_H
