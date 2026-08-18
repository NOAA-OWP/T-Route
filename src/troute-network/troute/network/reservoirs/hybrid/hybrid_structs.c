#include <stdlib.h>
#include "logger.h"
#include "../../reach_structs.h"
#include <stdio.h>
/* Hybrid Reservoir Interface */
extern void* get_hybrid_handle();

extern void init_hybrid(void* handle, float *water_elevation, float *lake_area, float *weir_elevation,
                    float *weir_coefficient, float *weir_length, float *dam_length, float *orifice_elevation,
                    float *orifice_coefficient, float *orifice_area, float *max_depth, 
                    float *initial_fractional_depth, int *lake_number, int *reservoir_type, 
                    char *reservoir_parameter_file, char *start_date, char *usgs_timeslice_path, 
                    char *usace_timeslice_path, int *observation_lookback_hours,
                    int *observation_update_time_interval_seconds);

extern void run_hybrid(void* handle, float *inflow, float *lateral_inflow,
                    float *water_elevation, float *outflow, float *routing_period);

extern void free_hybrid(void* handle);

init_hybrid_reach(_Reach* reach, int lake_number,
                          float dam_length, float area, float max_depth,
                          float orifice_area, float orifice_coefficient, float orifice_elevation,
                          float weir_coefficient, float weir_elevation, float weir_length,
                          float initial_fractional_depth, float water_elevation,
                          int reservoir_type, char *reservoir_parameter_file_ptr, char *start_date_ptr,
                          char *usgs_timeslice_path_ptr, char *usace_timeslice_path_ptr, 
                          int observation_lookback_hours,
                          int observation_update_time_interval_seconds
)
{
  if( reach != NULL )
  {
    reach->reach.hybrid.lake_number = lake_number;
    reach->reach.hybrid.dam_length = dam_length;
    reach->reach.hybrid.area = area;
    reach->reach.hybrid.max_depth = max_depth;
    reach->reach.hybrid.orifice_area = orifice_area;
    reach->reach.hybrid.orifice_coefficient = orifice_coefficient;
    reach->reach.hybrid.orifice_elevation = orifice_elevation;
    reach->reach.hybrid.weir_coefficient = weir_coefficient;
    reach->reach.hybrid.weir_elevation = weir_elevation;
    reach->reach.hybrid.weir_length = weir_length;
    reach->reach.hybrid.initial_fractional_depth = initial_fractional_depth;
    reach->reach.hybrid.reservoir_type = reservoir_type;

    reach->reach.hybrid.reservoir_parameter_file = (char*) malloc(sizeof(char)*HYBRID_MAX_PATH_LENGTH);
    strcpy(reach->reach.hybrid.reservoir_parameter_file, reservoir_parameter_file_ptr);
    reach->reach.hybrid.start_date = (char*) malloc(sizeof(char)*HYBRID_MAX_START_DATE_LENGTH);
    strcpy(reach->reach.hybrid.start_date, start_date_ptr);
    reach->reach.hybrid.usgs_timeslice_path = (char*) malloc(sizeof(char)*HYBRID_MAX_PATH_LENGTH);
    strcpy(reach->reach.hybrid.usgs_timeslice_path, usgs_timeslice_path_ptr);
    reach->reach.hybrid.usace_timeslice_path = (char*) malloc(sizeof(char)*HYBRID_MAX_PATH_LENGTH);
    strcpy(reach->reach.hybrid.usace_timeslice_path, usace_timeslice_path_ptr);

    reach->reach.hybrid.observation_lookback_hours = observation_lookback_hours;
    reach->reach.hybrid.observation_update_time_interval_seconds = observation_update_time_interval_seconds;

    if(water_elevation < -900000000){
      //Equation below is used in wrf-hydro
      Log(WARNING, "HYBRID PERSISTENCE RESERVOIR USING COLDSTART WATER ELEVATION");
      reach->reach.hybrid.water_elevation = orifice_elevation + ((max_depth - orifice_elevation) * initial_fractional_depth);
    }
    else{
      reach->reach.hybrid.water_elevation = water_elevation;
    }

    reach->reach.hybrid.handle = get_hybrid_handle();
    init_hybrid(reach->reach.hybrid.handle, &reach->reach.hybrid.water_elevation, &reach->reach.hybrid.area,
                 &reach->reach.hybrid.weir_elevation, &reach->reach.hybrid.weir_coefficient, 
                 &reach->reach.hybrid.weir_length, &reach->reach.hybrid.dam_length, 
                 &reach->reach.hybrid.orifice_elevation, &reach->reach.hybrid.orifice_coefficient,
                 &reach->reach.hybrid.orifice_area, &reach->reach.hybrid.max_depth, 
                 &reach->reach.hybrid.initial_fractional_depth, &reach->reach.hybrid.lake_number, 
                 &reach->reach.hybrid.reservoir_type, reach->reach.hybrid.reservoir_parameter_file, 
                 reach->reach.hybrid.start_date, reach->reach.hybrid.usgs_timeslice_path, 
                 reach->reach.hybrid.usace_timeslice_path, &reach->reach.hybrid.observation_lookback_hours, 
                 &reach->reach.hybrid.observation_update_time_interval_seconds);
  }
}

void free_hybrid_reach(_Reach* reach)
{
  free_hybrid(reach->reach.hybrid.handle);
  free(reach->reach.hybrid.reservoir_parameter_file);
  free(reach->reach.hybrid.start_date);
  free(reach->reach.hybrid.usgs_timeslice_path);
  free(reach->reach.hybrid.usace_timeslice_path);
}

void route(_Reach* reach, float inflow, float lateral_inflow, float routing_period,
           float* outflow,  float* water_elevation)
{
  run_hybrid(reach->reach.hybrid.handle, &inflow, &lateral_inflow, &reach->reach.hybrid.water_elevation, outflow, &routing_period);
  *water_elevation = reach->reach.hybrid.water_elevation;
}
