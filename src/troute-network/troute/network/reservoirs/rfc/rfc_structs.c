#include <stdlib.h>
#include "logger.h"
#include "../../reach_structs.h"
#include <stdio.h>
/* RFC Reservoir Interface */
extern void* get_rfc_handle();

extern void init_rfc(void* handle, float *water_elevation, float *lake_area, float *weir_elevation,
                    float *weir_coefficient, float *weir_length, float *dam_length, float *orifice_elevation,
                    float *orifice_coefficient, float *orifice_area, float *max_depth, 
                    float *initial_fractional_depth, int *lake_number, int *reservoir_type, 
                    char *reservoir_parameter_file, char *start_date, char *time_series_path, 
                    int *forecast_lookback_hours);

extern void run_rfc(void* handle, float *inflow, float *lateral_inflow,
                    float *water_elevation, float *outflow, float *routing_period);

extern void free_rfc(void* handle);

void init_rfc_reach(_Reach* reach, int lake_number,
                          float dam_length, float area, float max_depth,
                          float orifice_area, float orifice_coefficient, float orifice_elevation,
                          float weir_coefficient, float weir_elevation, float weir_length,
                          float initial_fractional_depth, float water_elevation,
                          int reservoir_type, char *reservoir_parameter_file_ptr, char *start_date_ptr,
                          char *time_series_path_ptr, int forecast_lookback_hours
)
{
  if( reach != NULL )
  {
    reach->reach.rfc.lake_number = lake_number;
    reach->reach.rfc.dam_length = dam_length;
    reach->reach.rfc.area = area;
    reach->reach.rfc.max_depth = max_depth;
    reach->reach.rfc.orifice_area = orifice_area;
    reach->reach.rfc.orifice_coefficient = orifice_coefficient;
    reach->reach.rfc.orifice_elevation = orifice_elevation;
    reach->reach.rfc.weir_coefficient = weir_coefficient;
    reach->reach.rfc.weir_elevation = weir_elevation;
    reach->reach.rfc.weir_length = weir_length;
    reach->reach.rfc.initial_fractional_depth = initial_fractional_depth;
    reach->reach.rfc.reservoir_type = reservoir_type;

    reach->reach.rfc.reservoir_parameter_file = (char*) malloc(sizeof(char)*RFC_MAX_PATH_LENGTH);
    strcpy(reach->reach.rfc.reservoir_parameter_file, reservoir_parameter_file_ptr);
    reach->reach.rfc.start_date = (char*) malloc(sizeof(char)*RFC_MAX_START_DATE_LENGTH);
    strcpy(reach->reach.rfc.start_date, start_date_ptr);
    reach->reach.rfc.time_series_path = (char*) malloc(sizeof(char)*RFC_MAX_PATH_LENGTH);
    strcpy(reach->reach.rfc.time_series_path, time_series_path_ptr);

    reach->reach.rfc.forecast_lookback_hours = forecast_lookback_hours;

    if(water_elevation < -900000000){
      //Equation below is used in wrf-hydro
      Log(WARNING, "RFC RESERVOIR USING COLDSTART WATER ELEVATION");
      reach->reach.rfc.water_elevation = orifice_elevation + ((max_depth - orifice_elevation) * initial_fractional_depth);
    }
    else{
      reach->reach.rfc.water_elevation = water_elevation;
    }

    reach->reach.rfc.handle = get_rfc_handle();
    init_rfc(reach->reach.rfc.handle, &reach->reach.rfc.water_elevation, &reach->reach.rfc.area,
                 &reach->reach.rfc.weir_elevation, &reach->reach.rfc.weir_coefficient,
                 &reach->reach.rfc.weir_length, &reach->reach.rfc.dam_length, 
                 &reach->reach.rfc.orifice_elevation, &reach->reach.rfc.orifice_coefficient,
                 &reach->reach.rfc.orifice_area, &reach->reach.rfc.max_depth, 
                 &reach->reach.rfc.initial_fractional_depth, &reach->reach.rfc.lake_number,
                 &reach->reach.rfc.reservoir_type, reach->reach.rfc.reservoir_parameter_file,
                 reach->reach.rfc.start_date, reach->reach.rfc.time_series_path,
                 &reach->reach.rfc.forecast_lookback_hours);
  }
}

void free_rfc_reach(_Reach* reach)
{
  free_rfc(reach->reach.rfc.handle);
  free(reach->reach.rfc.reservoir_parameter_file);
  free(reach->reach.rfc.start_date);
  free(reach->reach.rfc.time_series_path);
}

void route(_Reach* reach, float inflow, float lateral_inflow, float routing_period,
           float* outflow,  float* water_elevation)
{
  run_rfc(reach->reach.rfc.handle, &inflow, &lateral_inflow, &reach->reach.rfc.water_elevation, outflow, &routing_period);
  *water_elevation = reach->reach.rfc.water_elevation;
}
