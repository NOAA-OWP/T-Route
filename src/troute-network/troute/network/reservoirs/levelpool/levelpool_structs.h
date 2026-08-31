#ifndef LEVELPOOL_STRUCTS_H
#define LEVELPOOL_STRUCTS_H
/*
    C Structures
*/
#include "../../reach_structs.h"
typedef struct {
  long lake_number;
  float dam_length, area, max_depth;
  float orifice_area, orifice_coefficient, orifice_elevation;
  float weir_coefficient, weir_elevation, weir_length;
  float initial_fractional_depth, water_elevation;
  int wbody_type_code;

  //Handle to operate levelpool reservoir
  void* handle;
} _MC_Levelpool;

#endif //LEVELPOOL_STRUCTS_H
