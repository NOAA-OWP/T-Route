! This module defines Fortran to C bindings for Level Pool type reservoirs

module reservoir_to_c_lp
    use module_levelpool
    implicit none

contains

        FUNCTION get_lp_handle() RESULT(handle) BIND(C, NAME='get_lp_handle')
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_LOC
            TYPE(C_PTR) :: handle
            TYPE(levelpool), POINTER :: lp_ptr   ! ptr to LP object
            ALLOCATE(lp_ptr) ! Allocate LP example object
            handle = C_LOC(lp_ptr) ! Get the C address of the LP object and assign to handle
        END FUNCTION get_lp_handle

        SUBROUTINE free_lp(handle) BIND(C, NAME='free_lp')
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER
            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            TYPE(levelpool), POINTER :: lp_ptr
            CALL C_F_POINTER(handle, lp_ptr)
            DEALLOCATE(lp_ptr)
        END SUBROUTINE free_lp

        SUBROUTINE init_lp(handle,  water_elevation,  &
        lake_area, weir_elevation, weir_coeffecient, &
        weir_length, dam_length, orifice_elevation, orifice_coefficient, &
        orifice_area, max_depth, lake_number, wbody_type_code) BIND(C, NAME='init_lp')
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER, C_INT64_T
            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            real, intent(inout) :: water_elevation           ! meters AMSL
            real, intent(in)    :: lake_area                 ! area of lake (km^2)
            real, intent(in)    :: weir_elevation            ! bottom of weir elevation (meters AMSL)
            real, intent(in)    :: weir_coeffecient          ! weir coefficient
            real, intent(in)    :: weir_length               ! weir length (meters)
            real, intent(in)    :: dam_length                ! dam length (meters)
            real, intent(in)    :: orifice_elevation         ! orifice elevation (meters AMSL)
            real, intent(in)    :: orifice_coefficient       ! orifice coefficient
            real, intent(in)    :: orifice_area              ! orifice area (meters^2)
            real, intent(in)    :: max_depth                 ! max depth of reservoir before overtop (meters)
            integer(C_INT64_T), intent(in) :: lake_number               ! lake number
            integer, intent(in) :: wbody_type_code           ! lake number
            type (levelpool), POINTER :: levelpool_ptr ! ptr to LP object
            CALL C_F_POINTER(handle, levelpool_ptr)
 
            call levelpool_ptr%init(water_elevation,  &
                                                   lake_area, weir_elevation, weir_coeffecient, &
                                                   weir_length, dam_length, orifice_elevation, orifice_coefficient, &
                                                   orifice_area, max_depth, lake_number, wbody_type_code )
        END SUBROUTINE init_lp

        SUBROUTINE run_lp(handle, inflow, lateral_inflow, water_elevation, outflow, routing_period) BIND(C, NAME='run_lp')

            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER, C_CHAR, C_LOC, C_NULL_CHAR

            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            real, intent(in)    :: inflow                   ! cubic meters per second (cms)
            real, intent(in)    :: lateral_inflow           ! cubic meters per second (cms)
            real, intent(inout) :: water_elevation          ! meters AMSL
            real, intent(out)   :: outflow                  ! cubic meters per second (cms)
            real, intent(in)    :: routing_period           ! seconds

            integer :: dynamic_reservoir_type   ! dynamic reservoir type sent to lake out files
            real  :: assimilated_value        ! value assimilated from observation or forecast
            character(len=256) :: assimilated_source_file ! source file of assimilated value

            type (levelpool), POINTER :: levelpool_ptr ! ptr to LP object

            CALL C_F_POINTER(handle, levelpool_ptr)

            call levelpool_ptr%run(inflow, &
                                   inflow, &
                                   lateral_inflow, &
                                   water_elevation, &
                                   outflow, &
                                   routing_period, &
                                   dynamic_reservoir_type, &
                                   assimilated_value, &
                                   assimilated_source_file)

            !Use something like below if needing to return assimilated_source_file
            !copy the tmp str to c_str and null terminate
            !assimilated_source_file = trim(str) // c_null_char
            !deallocate tmp str
            !deallocate(str)
            !character(kind = C_CHAR), intent(out) :: assimilated_source_file(256)
            !character(256), pointer :: str
            !allocate(str) !allocate the pointer to hold the tmp string

        END SUBROUTINE run_lp
        
        SUBROUTINE assim(handle, updated_elevation, water_elevation) BIND(C, NAME='assim')

            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER, C_CHAR, C_LOC, C_NULL_CHAR

            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            real, intent(in)    :: updated_elevation 
            real, intent(out)   :: water_elevation
            
            type (levelpool), POINTER :: levelpool_ptr ! ptr to LP object

            CALL C_F_POINTER(handle, levelpool_ptr)
            
            call levelpool_ptr%assim(updated_elevation, water_elevation)
            
        END SUBROUTINE assim
        
end module reservoir_to_c_lp
