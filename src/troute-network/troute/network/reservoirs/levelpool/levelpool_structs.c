#include <stdlib.h>
#include "logger.h"
#include "../../reach_structs.h"
#include <stdio.h>
/* Level Pool Reservoir Interface */

/* define C compatible Fortran subroutines from bind_lp.f90 */
extern void* get_lp_handle();

extern void init_lp(void* handle, float *water_elevation, float *lake_area, 
                    float *weir_elevation, float *weir_coefficient, 
                    float *weir_length, float *dam_length, 
                    float *orifice_elevation, float *orifice_coefficient, 
                    float *orifice_area, float *max_depth, int *lake_number, 
                    int *wbody_type_code);

extern void run_lp(void* handle, float *inflow, float *lateral_inflow,
                   float *water_elevation, float *outflow, 
                   float *routing_period);

extern void free_lp(void* handle);

extern void assim(void* handle, float *updated_elevation, 
                  float *water_elevation);

/*
 * Function: init_levelpool_reach
 *-------------------------------
 * Initializes levelpool reservoir structure. The levelpool structure contains 
 * various reservoir attributes and the water elevation state, which are all
 * passed in as arguments
 *
 * Arguments
 * ---------
 * - reach                  (pointer):
 * 
 * - lake_number                (int): Unique waterbody ID code
 *
 * - dam_length               (float): Length of dam, meters
 * 
 * - area                     (float): Area of waterbody, meters
 *
 * - max_depth                (float): maximum waterbody depth, meters
 * 
 * - orifice_area             (float): reservoir orifice area, square-meters
 *
 * - orifice_coefficient      (float): orifice coefficient
 * 
 * - orifice_elevation        (float): elevation of reservoir orifice, meters
 * 
 * - weir_coefficient         (float): weir coefficient
 * 
 * - weir_elevation           (float): elevation of reservoir weir, meters
 *
 * - weir_length              (float): length of reservoir weir, meters
 *
 * - initial_fractional_depth (float): 
 *
 * - water_elevation          (float): water surface elevation, meters
 *
 * - wbody_type_code            (int): integer indicating the type of reservoir
 *                                      1: Levelpool
 *                                      2: USGS Hybrid
 *                                      3: USACE Hybrid
 *
 * Returns
 * -------
 *
 *
 * Notes
 *
 */
void init_levelpool_reach(_Reach* reach, int lake_number,
                     float dam_length, float area, float max_depth, 
                     float orifice_area, float orifice_coefficient, 
                     float orifice_elevation, float weir_coefficient, 
                     float weir_elevation, float weir_length, 
                     float initial_fractional_depth, float water_elevation, 
                     int wbody_type_code
)
{
  if( reach != NULL )
  {
    // specify structure members
    reach->reach.lp.lake_number = lake_number;
    reach->reach.lp.dam_length = dam_length;
    reach->reach.lp.area = area;
    reach->reach.lp.max_depth = max_depth;
    reach->reach.lp.orifice_area = orifice_area;
    reach->reach.lp.orifice_coefficient = orifice_coefficient;
    reach->reach.lp.orifice_elevation = orifice_elevation;
    reach->reach.lp.weir_coefficient = weir_coefficient;
    reach->reach.lp.weir_elevation = weir_elevation;
    reach->reach.lp.weir_length = weir_length;
    reach->reach.lp.initial_fractional_depth = initial_fractional_depth;
    reach->reach.lp.wbody_type_code = wbody_type_code;

    if(water_elevation < -900000000){
      //Equation below is used in wrf-hydro
      Log(WARNING, "LEVELPOOL USING COLDSTART WATER ELEVATION");
      reach->reach.lp.water_elevation = orifice_elevation 
          + ((max_depth - orifice_elevation) * initial_fractional_depth);
    }
    else{
      reach->reach.lp.water_elevation = water_elevation;
    }

    reach->reach.lp.handle = get_lp_handle();
    
    // call C compatible Fortran function to initialize reservoir object
    // init_lp is in bind_lp.f90
    init_lp(reach->reach.lp.handle, &reach->reach.lp.water_elevation, 
                &reach->reach.lp.area, &reach->reach.lp.weir_elevation, 
                &reach->reach.lp.weir_coefficient, &reach->reach.lp.weir_length,
                &reach->reach.lp.dam_length, &reach->reach.lp.orifice_elevation, 
                &reach->reach.lp.orifice_coefficient, &reach->reach.lp.orifice_area, 
                &reach->reach.lp.max_depth, &reach->reach.lp.lake_number, 
                &reach->reach.lp.wbody_type_code);
  }
}

void free_levelpool_reach(_Reach* reach)
{
  free_lp(reach->reach.lp.handle);
}

/*
 * Function: route
 * --------------------------
 * This function executes reservoir routing for a single timestep,
 * single reservoir. It calls run_lp, with is a subrouting in bind_lp.f90
 *
 * Arguments
 * ---------
 * reach                  (): 
 *
 * inflow            (float): reservoir inflow from upstream segments
 *
 * lateral_inflow    (float): lateral inflows to reservoir (cms)
 *
 * routing_period    (float): timestep (seconds)
 *
 * outflow           (float): reservoir outflow at end of routing period (cms)
 *
 * water_elevation   (float): (initial, then computed) water elevation
 *
 */
void route(_Reach* reach, float inflow, float lateral_inflow, float routing_period,
           float* outflow, float* water_elevation)
{
  run_lp(reach->reach.lp.handle, &inflow, &lateral_inflow, &reach->reach.lp.water_elevation, outflow, &routing_period);
  *water_elevation = reach->reach.lp.water_elevation;
}

/*
 * Function: update_elevation
 * --------------------------
 * This function updates the waterbody elevation state
 *
 * Arguments
 * ---------
 * reach                  ():
 *
 * updated_elevation (float):
 *
 * water_elevation   (float):
 *
 */
void update_elevation(_Reach* reach, float updated_elevation, float* water_elevation)
{
  assim(reach->reach.lp.handle, &updated_elevation, &reach->reach.lp.water_elevation);
  *water_elevation = reach->reach.lp.water_elevation;

}
