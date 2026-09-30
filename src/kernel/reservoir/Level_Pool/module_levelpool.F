! This module defines and instantiates objects
! for a level pool type reservoir. The level
! pool reservoir type inherits input and
! output types from the reservoir base
! module and calls instantiation of these into
! sub-objects. The level pool reservoir type
! also points to types for level pool properties
! and state and calls instantiation of these into
! sub-objects. This module also contains the
! subroutine to run level pool reservoir that is
! derived from the reservoir base type interface
! to run reservoir. Running level pool will
! then call the LEVELPOOL_PHYSICS subroutine, which
! processes the given inputs, properties, and
! state for a particular level pool reservoir and
! returns the output/outflow.

module module_levelpool

    use module_levelpool_properties, only: levelpool_properties_interface
    use module_levelpool_state, only: levelpool_state_interface
    use module_reservoir, only: reservoir, reservoir_input, &
                                     reservoir_output
    use, intrinsic :: iso_c_binding, only: C_INT64_T
    !use module_hydro_stop, only: HYDRO_stop

!#ifdef RESERVOIR_D
!    use module_reservoir_utilities, only: create_levelpool_diagnostic_log_file, &
!                                          log_levelpool_diagnostic_data
!#endif

    implicit none

    ! Extend/derive level pool type from the abstract base
    ! type for reservoirs.
    type, extends(reservoir) :: levelpool

        ! Define pointers to sub-types / sub-objects to and
        ! held by a level pool reservoir object.
        type (levelpool_properties_interface), pointer :: properties => null()
        type (levelpool_state_interface), pointer :: state => null()

        logical :: pointer_allocation_guard = .false.

    contains

        procedure :: init => levelpool_init
        procedure :: destroy => levelpool_destroy
        procedure :: run => run_levelpool_reservoir
        procedure :: assim => assimilate_water_elevation

    end type levelpool

contains

    !Level Pool Constructor
    subroutine levelpool_init(this, water_elevation,  &
        lake_area, weir_elevation, weir_coeffecient, &
        weir_length, dam_length, orifice_elevation, orifice_coefficient, &
        orifice_area, max_depth, lake_number, wbody_type_code)
        implicit none
        class(levelpool), intent(inout) :: this ! object being initialized
        double precision, intent(inout) :: water_elevation ! meters AMSL
        real, intent(in)    :: lake_area                 ! area of lake (km^2)
        real, intent(in)    :: weir_elevation            ! bottom of weir elevation (meters AMSL)
        real, intent(in)    :: weir_coeffecient          ! weir coefficient
        real, intent(in)    :: weir_length               ! weir length (meters)
        real, intent(in)    :: dam_length                ! dam length, a multiplier on weir length
        real, intent(in)    :: orifice_elevation         ! orifice elevation (meters AMSL)
        real, intent(in)    :: orifice_coefficient       ! orifice coefficient
        real, intent(in)    :: orifice_area              ! orifice area (meters^2)
        real, intent(in)    :: max_depth                 ! max depth of reservoir before overtop (meters)
        integer(C_INT64_T), intent(in) :: lake_number               ! lake number
        integer, intent(in) :: wbody_type_code
        character(len=15)   :: lake_number_string

!#ifdef RESERVOIR_D
        ! Create diagnostic log file only for development/debugging purposes
        !call create_levelpool_diagnostic_log_file(lake_number)
!#endif

        if (this%pointer_allocation_guard .eqv. .false. ) then
            ! try to allocate input
            allocate ( this%input )
            if ( .not. associated(this%input) ) then
                ! if the input structure could not be created, call hydro_stop.
                write(lake_number_string, "(I15)") lake_number
                !call hydro_stop("ERROR: Failure to allocate level pool input structure for reservoir " &
                !// trim(ADJUSTL(lake_number_string)) // ".")
            else
                ! initialize the input structure
                call this%input%init()
            end if

            ! try to allocate output
            allocate ( this%output )
            if ( .not. associated(this%output) ) then
                ! if the output structure could not be created, call hydro_stop.
                write(lake_number_string, "(I15)") lake_number
                !call hydro_stop("ERROR: Failure to allocate level pool output structure for reservoir " &
                !// trim(ADJUSTL(lake_number_string)) // ".")
            else
                ! initialize the output structure
                call this%output%init()
            end if

            ! try to allocate properties
            allocate ( this%properties )
            if ( .not. associated(this%properties) ) then
                ! if the properties structure could not be created, call hydro_stop.
                write(lake_number_string, "(I15)") lake_number
                !call hydro_stop("ERROR: Failure to allocate levelpool properties structure for reservoir " &
                !// trim(ADJUSTL(lake_number_string)) // ".")
            else
                ! initialize levelpool properties
                call this%properties%init( lake_area,  &
                    weir_elevation, weir_coeffecient, weir_length, dam_length, &
                    orifice_elevation, orifice_coefficient, &
                    orifice_area, max_depth, lake_number, wbody_type_code )
            end if
            this%pointer_allocation_guard = .true.

            ! try to allocate state
            allocate ( this%state )
            if ( .not. associated(this%state) ) then
                ! if the state structure could not be created, call hydro_stop.
                write(lake_number_string, "(I15)") lake_number
                !call hydro_stop("ERROR: Failure to allocate state properties structure for reservoir " &
                !// trim(ADJUSTL(lake_number_string)) // ".")
            else
                ! initialize levelpool state
                call this%state%init( water_elevation )
            end if
            this%pointer_allocation_guard = .true.
        end if

    end subroutine levelpool_init


    !Level Pool Destructor
    subroutine levelpool_destroy(this)
        implicit none
        class(levelpool), intent(inout) :: this ! object being destroyed
    end subroutine levelpool_destroy
    
    ! Subroutine for assimilating data-informed depth data into reservoir state
    subroutine assimilate_water_elevation(this, updated_elevation, water_elevation)
        implicit none
        class(levelpool), intent(inout) :: this
        double precision, intent(in)  :: updated_elevation ! meters
        double precision, intent(out) :: water_elevation   ! meters
        
        ! set reservoir depth state to update value
        water_elevation = updated_elevation
        this%state%water_elevation = water_elevation
                
    end subroutine assimilate_water_elevation


    ! Subroutine for running a level pool reservoir,
    ! which will then call the LEVELPOOL method/subroutine for processing the
    ! inputs and returning the output.
    subroutine run_levelpool_reservoir(this, previous_timestep_inflow, inflow, &
        lateral_inflow, water_elevation, outflow, routing_period, dynamic_reservoir_type, &
        assimilated_value, assimilated_source_file)
        implicit none
        class(levelpool), intent(inout) :: this
        real, intent(in)    :: previous_timestep_inflow ! cubic meters per second (cms)
        real, intent(in)    :: inflow                   ! cubic meters per second (cms)
        real, intent(in)    :: lateral_inflow           ! cubic meters per second (cms)
        double precision, intent(inout) :: water_elevation ! meters
        real, intent(out)   :: outflow                  ! cubic meters per second (cms)
        real, intent(in)    :: routing_period           ! seconds
        integer, intent(out):: dynamic_reservoir_type   ! dynamic reservoir type sent to lake out files
        real, intent(out)   :: assimilated_value        ! value assimilated from observation or forecast
        character(len=256), intent(out) :: assimilated_source_file ! source file of assimilated value

        ! Update input variables
        this%input%inflow = inflow
        this%input%lateral_inflow = lateral_inflow

        ! Update state variables
        this%state%water_elevation = water_elevation

        call LEVELPOOL_PHYSICS(this%properties%lake_number,                  &
                               previous_timestep_inflow,                     &
                               this%input%inflow,                            &
                               this%output%outflow,                          &
                               this%input%lateral_inflow,                    &
                               routing_period,                               &
                               this%state%water_elevation,                   &
                               this%properties%lake_area,                    &
                               this%properties%weir_elevation,               &
                               this%properties%max_depth,                    &
                               this%properties%weir_coeffecient,             &
                               this%properties%weir_length,                  &
                               this%properties%dam_length,                   &
                               this%properties%orifice_elevation,            &
                               this%properties%orifice_coefficient,          &
                               this%properties%orifice_area                  &
        )

        ! Update output variable returned from this subroutine
        outflow = this%output%outflow

        ! Set current inflow to previous_timestep_inflow
        this%input%previous_timestep_inflow = inflow

        ! Update water_elevation variable returned from this subroutine
        water_elevation = this%state%water_elevation

        ! The dynamic reservoir type is always set to 1 for level pool in this module because
        ! it cannot change reservoir types
        dynamic_reservoir_type = 1

        ! The assimilated value would always be a sentinel of -9999.0 because level pool does
        ! not assimilate any external observations or forecasts
        assimilated_value = -9999.0

        ! The assimilated source file would always be an empty string
        assimilated_source_file = ""

#ifdef RESERVOIR_D
        ! Log diagnostic data only for development/debugging purposes
        !call log_levelpool_diagnostic_data(this%properties%lake_number, inflow, water_elevation, outflow)
#endif

    end subroutine run_levelpool_reservoir

    ! ------------------------------------------------
    !   SUBROUTINE LEVELPOOL
    ! ------------------------------------------------

    subroutine LEVELPOOL_PHYSICS(ln,qi0,qi1,qo1,ql,dt,H,ar,we,maxh,wc,wl,dl,oe,oc,oa)

        !! ----------------------------  argument variables
        !! All elevations should be relative to a common base (often belev(k))

        double precision, intent(INOUT) :: H ! water elevation height (m)
        real, intent(IN)    :: dt      ! routing period [s]
        real, intent(IN)    :: qi0     ! inflow at previous timestep (cms)
        real, intent(IN)    :: qi1     ! inflow at current timestep (cms)
        real, intent(OUT)   :: qo1     ! outflow at current timestep
        real, intent(IN)    :: ql      ! lateral inflow
        real, intent(IN)    :: ar      ! area of reservoir (km^2)
        real, intent(IN)    :: we      ! bottom of weir elevation
        real, intent(IN)    :: wc      ! weir coeff.
        real, intent(IN)    :: wl      ! weir length (m)
        real, intent(IN)    :: dl      ! dam length, a multiplier on weir length
        real, intent(IN)    :: oe      ! orifice elevation
        real, intent(IN)    :: oc      ! orifice coeff.
        real, intent(IN)    :: oa      ! orifice area (m^2)
        real, intent(IN)    :: maxh    ! max depth of reservoir before overtop (m)
        integer(C_INT64_T), intent(IN) :: ln      ! lake number

        !!DJG Add lake option switch here...move up to namelist in future versions...
        integer :: LAKE_OPT            ! Lake model option (move to namelist later)
        double precision :: Htmp       ! Temporary assign of incoming lake el. (m)

        !! ----------------------------  local variables
        real :: sap                    ! local surface area values
        real :: discharge              ! storage discharge m^3/s
        real :: dh, dh1, dh2, dh3      ! Depth in weir, and height function for 3 order RK
        real :: It, Itdt_3, Itdt_2_3   ! inflow hydrographs
        real :: maxWeirDepth           !maximum capacity of weir
        real :: h_floor                ! lowest elevation any outlet discharges from (m)
        real :: dt_sub                 ! sub-step length (s)
        real :: qo_sum                 ! outflow accumulated over the sub-steps
        real :: q1, q3                 ! the stage discharges the storage change weights
        real :: q_eff                  ! the sub-step's outflow, once storage bounds it
        real :: head_scale             ! the head a sub-step is measured against (m)
        real :: dqdh                   ! how fast discharge answers elevation (m2/s)
        real :: lateral                ! lateral inflow, which runs for the whole step
        real :: span                   ! inflow change across the step (cms)
        real :: t0, available, drawdown
        real :: reach                  ! where one whole step would take the pool (m)
        real :: want                   ! sub-steps the stability rule asks for
        real :: net                    ! mean inflow less discharge at the starting elevation
        real :: threshold(3)           ! the elevations an outlet opens at
        integer :: k
        ! The elevation is carried in double between calls: in single precision a storage
        ! change under half an ulp, about 4 cms on a 300 km2 lake near 100 m, rounds away.
        double precision :: h_acc
        real :: h_now                  ! the elevation discharge is evaluated at (m)
        integer :: nsub, isub
        ! A sub-step stays stable while short against tau = sap / (dQ/dH), and is sized on
        ! dQ/dH since the net flux vanishes near equilibrium. SAFETY is its share of tau;
        ! NSUB_MAX caps the cost and the storage bound catches the rest.
        real, parameter :: SAFETY = 0.5
        real, parameter :: MIN_HEAD_SCALE = 0.01
        real, parameter :: PROBE_HEAD = 0.01
        integer, parameter :: NSUB_MAX = 256
        !real :: hdiff_vol, qdiff_vol   ! water balance check variables
        !! ----------------------------  subroutine body: from chow, mad mays. pg. 252
        !! -- determine from inflow hydrograph


        !!DJG Set hardwire for LAKE_OPT...move specification of this to namelist in
        !future versions...
        LAKE_OPT = 2
        Htmp = H   !temporary set of incoming lake water elevation...
        !hdiff_vol = 0.0
        !qdiff_vol = 0.0

        !!DJG IF-block for lake model option  1 - outflow=inflow, 2 - Chow et al level
        !pool, .....
        if (LAKE_OPT == 1) then     ! If-block for simple pass through scheme....

           qo1 = qi1                 ! Set outflow equal to inflow at current time
           H = Htmp                  ! Set new lake water elevation to incoming lake el.

        else if (LAKE_OPT == 2) then   ! If-block for Chow et al level pool scheme

           ! A non-finite elevation from a state file fails every comparison below and would
           ! publish zero for the rest of the run, so the pool restarts at its orifice invert.
           if (.not. (H >= -huge(H) .and. H <= huge(H))) then
              H = oe
           endif
           h_now = real(H)

           ! Lateral inflow runs for the whole step, so every stage carries it. t-route passes
           ! no lateral inflow and one inflow as both endpoints, so neither path is live there.
           lateral = ql
           span = qi1 - qi0
           maxWeirDepth =  maxh - we
           h_floor = min(oe, we, maxh)

           !assume vertically walled reservoir
           !remove this when moving to a variable head area volume
           sap = ar * 1.0E6

           ! A non-finite inflow says nothing about the step, so the pool keeps its
           ! elevation and publishes its own discharge.
           if (.not. (abs(qi0) <= huge(1.0) .and. abs(qi1) <= huge(1.0) .and. &
                      abs(ql) <= huge(1.0))) then
              qo1 = lp_discharge(h_now)
              return
           endif

           !-- size the sub-step from dQ/dH over a fixed PROBE_HEAD: an outlet is steepest just
           !-- above its threshold, which a deep-weir scale would average away. Probe where the
           !-- pool starts, where the step would take it, and every threshold in between.
           head_scale = max(maxWeirDepth, MIN_HEAD_SCALE)
           threshold = [oe, we, maxh]
           nsub = 1
           if (sap > 0.0) then
              net = 0.5 * (qi0 + qi1) + lateral - lp_discharge(h_now)
              reach = h_now + (net / sap) * dt
              dqdh = max(lp_sensitivity(h_now), lp_sensitivity(reach))
              do k = 1, size(threshold)
                 if (threshold(k) >= min(h_now, reach) .and. &
                     threshold(k) <= max(h_now, reach)) then
                    dqdh = max(dqdh, lp_sensitivity(threshold(k) + PROBE_HEAD))
                 endif
              end do
              ! A pool far from equilibrium still has to be stepped in small moves even
              ! where the sensitivity is mild.
              want = max(dt * dqdh / (SAFETY * sap), &
                         abs(net) * dt / (sap * SAFETY * head_scale))
              ! Clamp before converting: ceiling() past the 32 bit range returns the most
              ! negative integer, which would read as one sub-step.
              if (.not. (want < real(NSUB_MAX))) then
                 nsub = NSUB_MAX
              else if (want > 1.0) then
                 nsub = ceiling(want)
              else
                 nsub = 1
              endif
           endif
           dt_sub = dt / real(nsub)
           qo_sum = 0.0
           h_acc = H

           do isub = 1, nsub

              ! Each sub-step reads the inflow over its own span of the step.
              t0 = real(isub - 1) / real(nsub)
              It       = qi0 + span * t0 + lateral
              Itdt_3   = qi0 + span * (t0 + 1.0 / (3.0 * real(nsub))) + lateral
              Itdt_2_3 = qi0 + span * (t0 + 2.0 / (3.0 * real(nsub))) + lateral

              !-- determine Q(dh) from elevation-discharge relationship
              !-- and dh1
              q1 = lp_discharge(h_now)
              if (sap > 0) then
                 dh1 = ((It - q1)/sap)*dt_sub
              else
                 dh1 = 0.0
              endif

              !-- determine Q(H + dh1/3) from elevation-discharge relationship
              !-- dh2
              discharge = lp_discharge(h_now + dh1/3.)
              if (sap > 0.0) then
                 dh2 = ((Itdt_3 - discharge)/sap)*dt_sub
              else
                 dh2 = 0.0
              endif

              !-- determine Q(H + 2/3 dh2) from elevation-discharge relationship
              !-- dh3
              q3 = lp_discharge(h_now + dh2*0.667)
              if (sap > 0.0) then
                 dh3 = ((Itdt_2_3 - q3)/sap)*dt_sub
              else
                 dh3 = 0.0
              endif

              ! The outflow carries the Runge-Kutta weights of the storage change, so the
              ! volume released is the volume drawn down.
              q_eff = (q1/4.) + (0.75*q3)
              dh = (dh1/4.) + (0.75*dh3)

              ! A pool cannot release water it does not hold. A step at NSUB_MAX can still
              ! overdraw, and below the lowest outlet there is nothing to draw on.
              if (sap > 0.0 .and. h_now + dh < h_floor) then
                 available = (h_now - h_floor) * sap + ((It/4.) + (0.75*Itdt_2_3)) * dt_sub
                 if (available < 0.0) available = 0.0
                 drawdown = q_eff * dt_sub
                 if (drawdown > available) then
                    q_eff = available / dt_sub
                    dh = h_floor - h_now
                 endif
              endif

              h_acc = h_acc + dble(dh)
              h_now = real(h_acc)
              qo_sum = qo_sum + q_eff

           end do

           H = h_acc
           qo1  = qo_sum / real(nsub)  ! return the flow rate from reservoir

        !#ifdef HYDRO_D
        !#ifndef NCEP_WCOSS
        !   ! Water balance check
        !   qdiff_vol = (qi1+ql-qo1)*dt !m3
        !   hdiff_vol = (H-Htmp)*sap    !m3
        !22 format(f8.4,2x,f8.4,2x,f8.4,2x,f8.4,2x,f8.4,2x,f6.0,2x,f20.1,2x,f20.1)
        !   open (unit=67, &
        !     file='lake_massbalance_out.txt', status='unknown',position='append')
        !   write(67,22) Htmp, H, qi1, ql, qo1, dt, qdiff_vol, hdiff_vol
        !   close(67)
        !#endif
        !#endif

        !23 format('botof H dh orf wr Q',f8.4,2x,f8.4,2x,f8.3,2x,f8.3,2x,f8.2)
        !24 format('ofonl H dh sap Q ',f8.4,2x,f8.4,2x,f8.0,2x,f8.2)


        else   ! ELSE for LAKE_OPT....
        endif  ! ENDIF for LAKE_OPT....

        return

    contains

        ! ------------------------------------------------------------
        !   How fast discharge answers elevation near one elevation: a finite
        !   difference over PROBE_HEAD, since the relationship is piecewise and
        !   steepest at its branch boundaries.
        ! ------------------------------------------------------------
        real function lp_sensitivity(elev) result(dqdh)
            real, intent(IN) :: elev
            real :: lo, hi

            lo = elev - PROBE_HEAD
            if (lo < h_floor) lo = h_floor
            hi = elev + PROBE_HEAD
            if (hi - lo < tiny(1.0)) then
               dqdh = 0.0
            else
               dqdh = (lp_discharge(hi) - lp_discharge(lo)) / (hi - lo)
            endif
            if (dqdh < 0.0) dqdh = 0.0
        end function lp_sensitivity

        ! ------------------------------------------------------------
        !   The elevation-discharge relationship at one elevation. Every branch test
        !   and term reads the elevation passed in, and every head is floored at zero,
        !   so no term raises a negative number to 3/2 or takes the square root of one.
        ! ------------------------------------------------------------
        real function lp_discharge(elev) result(q)
            real, intent(IN) :: elev
            real :: weir_depth, orifice_head, overtop_head, weir_q, orifice_q

            weir_depth = elev - we
            if (weir_depth > maxWeirDepth) weir_depth = maxWeirDepth
            if (weir_depth < 0.0) weir_depth = 0.0

            orifice_head = elev - oe
            if (orifice_head < 0.0) orifice_head = 0.0

            overtop_head = elev - maxh
            if (overtop_head < 0.0) overtop_head = 0.0

            orifice_q = oc * oa * sqrt(2. * 9.81 * orifice_head)
            weir_q    = wc * wl * (weir_depth ** (3./2.))

            if (elev > maxh) then
               q = orifice_q + weir_q + (wc * (wl*dl) * (overtop_head ** (3./2.)))
            else if (elev > we) then
               q = orifice_q + weir_q
            else if (elev > oe) then
               q = orifice_q
            else
               q = 0.0   !in the dead pool
            endif
        end function lp_discharge

    ! ----------------------------------------------------------------
    end subroutine LEVELPOOL_PHYSICS
    ! ----------------------------------------------------------------

end module module_levelpool
