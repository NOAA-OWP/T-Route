! This module defines Fortran to C bindings for Persistence Level Pool Hybrid type reservoirs

module reservoir_to_c_hybrid
    use module_persistence_levelpool_hybrid
    implicit none

contains

        FUNCTION get_hybrid_handle() RESULT(handle) BIND(C, NAME='get_hybrid_handle')
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_LOC
            TYPE(C_PTR) :: handle
            TYPE(persistence_levelpool_hybrid), POINTER :: hybrid_ptr   ! ptr to hybrid object
            ALLOCATE(hybrid_ptr) ! Allocate hybrid example object
            handle = C_LOC(hybrid_ptr) ! Get the C address of the hybrid object and assign to handle
        END FUNCTION get_hybrid_handle

        SUBROUTINE free_hybrid(handle) BIND(C, NAME='free_hybrid')
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER
            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            TYPE(persistence_levelpool_hybrid), POINTER :: hybrid_ptr
            CALL C_F_POINTER(handle, hybrid_ptr)
            DEALLOCATE(hybrid_ptr)
        END SUBROUTINE free_hybrid

        SUBROUTINE init_hybrid(handle,  water_elevation,  &
        lake_area, weir_elevation, weir_coeffecient, &
        weir_length, dam_length, orifice_elevation, orifice_coefficient, &
        orifice_area, lake_max_water_elevation, initial_fractional_depth, &
        lake_number, reservoir_type, reservoir_parameter_file, start_date, &
        usgs_timeslice_path, usace_timeslice_path, observation_lookback_hours, &
        observation_update_time_interval_seconds) BIND(C, NAME='init_hybrid')
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER, C_CHAR, c_null_char, C_INT64_T
            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            real,    intent(inout) :: water_elevation           ! meters AMSL
            real,    intent(in)    :: lake_area                 ! area of lake (km^2)
            real,    intent(in)    :: weir_elevation            ! bottom of weir elevation (meters AMSL)
            real,    intent(in)    :: weir_coeffecient          ! weir coefficient
            real,    intent(in)    :: weir_length               ! weir length (meters)
            real,    intent(in)    :: dam_length                ! dam length (meters)
            real,    intent(in)    :: orifice_elevation         ! orifice elevation (meters AMSL)
            real,    intent(in)    :: orifice_coefficient       ! orifice coefficient
            real,    intent(in)    :: orifice_area              ! orifice area (meters^2)
            real,    intent(in)    :: lake_max_water_elevation  ! max water elevation (meters)
            real,    intent(in)    :: initial_fractional_depth  ! initial fraction water depth
            integer(C_INT64_T), intent(in) :: lake_number               ! lake number
            integer, intent(in)    :: reservoir_type            ! reservoir type
            character(kind=c_char), dimension(*), intent(IN) :: reservoir_parameter_file
            character(kind=c_char), dimension(*), intent(IN) :: start_date
            character(kind=c_char), dimension(*), intent(IN) :: usgs_timeslice_path
            character(kind=c_char), dimension(*), intent(IN) :: usace_timeslice_path
            integer,            intent(in) :: observation_lookback_hours
            integer,            intent(in) :: observation_update_time_interval_seconds

            type (persistence_levelpool_hybrid), POINTER :: hybrid_ptr ! ptr to hybrid object

            character(len=256) :: reservoir_parameter_file_F
            character(len=256) :: start_date_F
            character(len=256) :: usgs_timeslice_path_F
            character(len=256) :: usace_timeslice_path_F

            integer :: char_index, nchars_reservoir_parameter_file 
            integer :: nchars_start_date, nchars_usgs_timeslice_path, nchars_usace_timeslice_path

            ! Use fortran calls to go find certain things like null_char instead of using this loop every time
            ! Look up more about converting char array to string

            do char_index = 1, 256 
                if (reservoir_parameter_file(char_index) == c_null_char) then 
                    exit
                else
                   reservoir_parameter_file_F(char_index:char_index) = reservoir_parameter_file(char_index)
                end if
            end do
            nchars_reservoir_parameter_file = char_index - 1  ! Exclude null character from Fortran string

            do char_index = 1, 256 
                if (start_date(char_index) == c_null_char) then 
                    exit
                else
                   start_date_F(char_index:char_index) = start_date(char_index)
                end if
            end do
            nchars_start_date = char_index - 1  ! Exclude null character from Fortran string

            do char_index = 1, 256 
                if (usgs_timeslice_path(char_index) == c_null_char) then 
                    exit
                else
                   usgs_timeslice_path_F(char_index:char_index) = usgs_timeslice_path(char_index)
                end if
            end do
            nchars_usgs_timeslice_path = char_index - 1  ! Exclude null character from Fortran string

            do char_index = 1, 256 
                if (usace_timeslice_path(char_index) == c_null_char) then 
                    exit
                else
                   usace_timeslice_path_F(char_index:char_index) = usace_timeslice_path(char_index)
                end if
            end do
            nchars_usace_timeslice_path = char_index - 1  ! Exclude null character from Fortran string

            CALL C_F_POINTER(handle, hybrid_ptr)

            call hybrid_ptr%init(water_elevation,  &
                                 lake_area, weir_elevation, weir_coeffecient, &
                                 weir_length, dam_length, orifice_elevation, orifice_coefficient, &
                                 orifice_area, lake_max_water_elevation, initial_fractional_depth, lake_number, &
                                 reservoir_type, reservoir_parameter_file_F(:nchars_reservoir_parameter_file), &
                                 start_date_F(:nchars_start_date), usgs_timeslice_path_F(:nchars_usgs_timeslice_path), &
                                 usace_timeslice_path_F(:nchars_usace_timeslice_path), observation_lookback_hours, &
                                 observation_update_time_interval_seconds)

        END SUBROUTINE init_hybrid


        SUBROUTINE run_hybrid(handle, inflow, lateral_inflow, water_elevation, outflow, routing_period) BIND(C, NAME='run_hybrid')

            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_F_POINTER, C_CHAR, C_LOC, C_NULL_CHAR

            TYPE(C_PTR), INTENT(IN), VALUE :: handle
            real, intent(in)    :: inflow                   ! cubic meters per second (cms)
            real, intent(in)    :: lateral_inflow           ! cubic meters per second (cms)
            real, intent(inout) :: water_elevation          ! meters AMSL 
            real, intent(out)   :: outflow                  ! cubic meters per second (cms)
            real, intent(in)    :: routing_period           ! seconds
            integer :: dynamic_reservoir_type               ! dynamic reservoir type sent to lake out files
            real  :: assimilated_value                      ! value assimilated from observation or forecast
            character(len=256) :: assimilated_source_file   ! source file of assimilated value

            type (persistence_levelpool_hybrid), POINTER :: hybrid_ptr ! ptr to hybrid object

            CALL C_F_POINTER(handle, hybrid_ptr)

            call hybrid_ptr%run(inflow, &
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

        END SUBROUTINE run_hybrid

end module reservoir_to_c_hybrid
