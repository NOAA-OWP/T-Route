!-------------------------------------------------------------------------------
module diffusive

  IMPLICIT NONE
  
!-----------------------------------------------------------------------------
! Description:
!   Numerically solve diffusive wave PDEs using Crank-Nicholson and Hermite 
!   Interpolation. 
!
! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
!
! Code Description:
!   Language: Fortran 90.
!   This code is written to JULES coding standards v1.
!-----------------------------------------------------------------------------
  
  ! Module constants
  double precision, parameter :: grav = 9.81
  double precision, parameter :: TOLERANCE = 1e-8
  
  ! Module variables
  integer :: nlinks
  integer :: mxncomp
  integer :: maxTableLength
  integer :: nel
  integer :: newtonRaphson
  integer :: applyNaturalSection
  integer :: nel_g
  integer :: nmstem_rch
  integer, dimension(:),   allocatable :: currentROutingDiffusive
  integer, dimension(:),   allocatable :: notSwitchRouting
  integer, dimension(:),   allocatable :: mstem_frj    
  integer, dimension(:,:), allocatable :: currentRoutingNormal
  integer, dimension(:,:), allocatable :: routingNotChanged
  integer, dimension(:,:), allocatable :: frnw_g

  double precision :: dtini, dxini, cfl, minDx, maxCelerity,  theta
  double precision :: C_llm, D_llm, D_ulm, DD_ulm, DD_llm, q_llm, so_llm
  double precision :: frus2, minNotSwitchRouting, minNotSwitchRouting2
  double precision :: dt_qtrib
  double precision :: z_g, bo_g, traps_g, tw_g, twcc_g, so_g, mann_g, manncc_g
  double precision :: dmyi, dmyj
  double precision, dimension(:),       allocatable :: eei, ffi, exi, fxi, qcx, diffusivity2, celerity2
  double precision, dimension(:),       allocatable :: ini_y, ini_q
  double precision, dimension(:),       allocatable :: lowerLimitCount, higherLimitCount
  double precision, dimension(:),       allocatable :: elevTable, areaTable, skkkTable
  double precision, dimension(:),       allocatable :: pereTable, rediTable
  double precision, dimension(:),       allocatable :: convTable, topwTable
  double precision, dimension(:),       allocatable :: nwi1Table, dPdATable
  double precision, dimension(:),       allocatable :: ncompElevTable, ncompAreaTable
  double precision, dimension(:),       allocatable :: currentSquareDepth
  double precision, dimension(:),       allocatable :: tarr_qtrib, varr_qtrib
  double precision, dimension(:),       allocatable :: area, depth, co, froud, courant
  double precision, dimension(:,:),     allocatable :: bo, dx
  double precision, dimension(:,:),     allocatable :: areap, qp, z, sk
  double precision, dimension(:,:),     allocatable :: celerity, diffusivity, qpx  
  double precision, dimension(:,:),     allocatable :: pere, oldQ, newQ, oldArea, newArea, oldY, newY  
  double precision, dimension(:,:),     allocatable :: volRemain  
  double precision, dimension(:,:),     allocatable :: lateralFlow
  double precision, dimension(:,:),     allocatable :: dimensionless_Cr, dimensionless_Fo, dimensionless_Fi
  double precision, dimension(:,:),     allocatable :: dimensionless_Di, dimensionless_Fc, dimensionless_D  
  double precision, dimension(:,:),     allocatable :: qtrib  
  double precision, dimension(:,:,:,:), allocatable :: xsec_tab
  
contains

  subroutine diffnw(timestep_ar_g, nts_ql_g, nts_ub_g, nts_db_g, ntss_ev_g,       &
                    nts_qtrib_g, mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g,  &
                    tw_ar_g, twcc_ar_g, mann_ar_g, manncc_ar_g, so_ar_g, dx_ar_g, &
                    iniq, frnw_col, frnw_ar_g, qlat_g, ubcd_g, dbcd_g, qtrib_g,   &
                    paradim, para_ar_g, q_ev_g, elv_ev_g)

    IMPLICIT NONE
          
  !-----------------------------------------------------------------------------
  ! Description:
  !   Compute diffusive routing on National Water Model channel network domain.
  !
  ! Method:
  !   A Crank Nicholson solution of the diffusive wave equations is solved with
  !   adaptive timestepping to maintain numerical stability. Major operations are
  !   as follows:
  !
  !     1. Initialize domain flow and depth. Depth is computed from initial flow
  !     2. For each timestep....
  !       2.1. Compute domain flow, upstream-to-downstream
  !       2.2. Compute domain depth, downstream-to-upstream
  !       2.3. Determinie time step duration needed for stability
  !       2.4. repeat until until final time is reached
  !     3. Record results in output arrays at user-specified time intervals
  !
  ! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
  !
  ! Development Team:
  !  - DongHa Kim
  !  - Adam N. Wlostowski
  !  - Nazmul Azim Beg
  !  - Ehab Meselhe
  !  - James Halgren
  !  - Jacob Hreha
  !
  ! Code Description:
  !   Language: Fortran 90.
  !   This code is written to JULES coding standards v1.
  !-----------------------------------------------------------------------------

  ! Subroutine arguments
    integer, intent(in) :: mxncomp_g 
    integer, intent(in) :: nrch_g
    integer, intent(in) :: nts_ql_g
    integer, intent(in) :: nts_ub_g
    integer, intent(in) :: nts_db_g
    integer, intent(in) :: ntss_ev_g
    integer, intent(in) :: nts_qtrib_g
    integer, intent(in) :: frnw_col
    integer, intent(in) :: paradim
    double precision, dimension(paradim ), intent(in) :: para_ar_g
    double precision, dimension(:)       , intent(in) :: timestep_ar_g(8)
    double precision, dimension(nts_db_g), intent(in) :: dbcd_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: z_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: bo_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: traps_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: tw_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: twcc_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: mann_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: manncc_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: dx_ar_g
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: iniq
    double precision, dimension(mxncomp_g,   nrch_g),   intent(in) :: so_ar_g
    double precision, dimension(nts_ub_g,    nrch_g),   intent(in) :: ubcd_g
    double precision, dimension(nts_qtrib_g, nrch_g),   intent(in) :: qtrib_g
    integer, dimension(nrch_g,      frnw_col), intent(in) :: frnw_ar_g
    double precision, dimension(nts_ql_g,  mxncomp_g, nrch_g), intent(in ) :: qlat_g
    double precision, dimension(ntss_ev_g, mxncomp_g, nrch_g), intent(out) :: q_ev_g
    double precision, dimension(ntss_ev_g, mxncomp_g, nrch_g), intent(out) :: elv_ev_g

  ! Local variables    
    integer :: ncomp
    integer :: i
    integer :: j
    integer :: k
    integer :: n
    integer :: ntim
    integer :: pp
    integer :: tableLength
    integer :: timestep
    integer :: kkk
    integer :: frj
    integer :: i1
    integer :: ts_ev
    integer :: nts_db_g2
    integer :: jm
    integer :: rch
    integer :: usrchj
    integer :: ts
    integer, dimension(:), allocatable :: dmy_frj
    double precision :: cour
    double precision :: da
    double precision :: dq
    double precision :: x
    double precision :: saveInterval, width
    double precision :: xt
    double precision :: maxCourant
    double precision :: dtini_given
    double precision :: linknb
    double precision :: frds
    double precision :: currentQ
    double precision :: areac
    double precision :: timesDepth
    double precision :: t
    double precision :: tfin
    double precision :: t0
    double precision :: area_0
    double precision :: width_0
    double precision :: errorY
    double precision :: hydR_0
    double precision :: q_sk_multi
    double precision :: maxCelDx
    double precision :: dmy1
    double precision :: dmy2
    double precision :: slope
    double precision :: y_norm
    double precision :: area_n
    double precision :: temp
    double precision :: dt_ql
    double precision :: dt_ub
    double precision :: dt_db
    double precision :: wdepth
    double precision :: q_usrch
    double precision :: tf0
    double precision :: sumdmy1
    double precision :: sumdmy2
    double precision, dimension(:), allocatable :: tarr_ql
    double precision, dimension(:), allocatable :: varr_ql
    double precision, dimension(:), allocatable :: tarr_ub
    double precision, dimension(:), allocatable :: varr_ub
    double precision, dimension(:), allocatable :: tarr_db
    double precision, dimension(:), allocatable :: varr_db
    double precision, dimension(:,:), allocatable :: leftBank
    double precision, dimension(:,:), allocatable :: rightBank
    double precision, dimension(:,:), allocatable :: skLeft
    double precision, dimension(:,:), allocatable :: skMain
    double precision, dimension(:,:), allocatable :: skRight

  !-----------------------------------------------------------------------------
  ! Time domain parameters
    dtini        = timestep_ar_g(1) ! initial timestep duration [sec]
    t0           = timestep_ar_g(2) ! simulation start time [hr]
    tfin         = timestep_ar_g(3) ! simulation end time [hr]
    saveInterval = timestep_ar_g(4) ! output recording interval [sec]
    dt_ql        = timestep_ar_g(5) ! lateral inflow data time step [sec]
    dt_ub        = timestep_ar_g(6) ! upstream boundary time step [sec]
    dt_db        = timestep_ar_g(7) ! downstream boundary time step [sec]
    dt_qtrib     = timestep_ar_g(8) ! tributary data time step [sec]
    dtini_given  = dtini            ! preserve user-input timestep duraction
    
  !-----------------------------------------------------------------------------
  ! miscellaneous parameters
    timesDepth = 4.0 ! water depth multiplier used in readXsection
    nel = 501 ! number of sub intervals in look-up tables

  !-----------------------------------------------------------------------------
  ! Network mapping (frnw_g) array size parameters
    mxncomp = mxncomp_g ! maximum number of nodes in a single reach
    nlinks  = nrch_g    ! number of reaches in the network

  !-----------------------------------------------------------------------------
  ! Some essential parameters for Diffusive Wave
    cfl    = para_ar_g(1)  ! maximum Courant number (default: 0.95)
    C_llm  = para_ar_g(2)  ! lower limit of celerity (default: 0.5)
    D_llm  = para_ar_g(3)  ! lower limit of diffusivity (default: 50)
    D_ulm  = para_ar_g(4)  ! upper limit of diffusivity (default: 1000)
    DD_llm = para_ar_g(5)  ! lower limit of dimensionless diffusivity (default -15)
    DD_ulm = para_ar_g(6)  ! upper limit of dimensionless diffusivity (default: -10.0)
    q_llm  = para_ar_g(8)  ! lower limit of discharge (default: 0.02831 cms)
    so_llm = para_ar_g(9)  ! lower limit of channel bed slope (default: 0.0001)
    theta  = para_ar_g(10) ! weight for computing 2nd derivative: 
                         ! 0: explicit, 1: implicit (default: 1.0)

    !* root-finding technique used to compute diffusive depth
    !* 0: Bisection to compute water level; 1: Newton Raphson (default: 1.0)
    newtonRaphson = int(para_ar_g(7))

  !-----------------------------------------------------------------------------
  ! consider moving variable allocation to a separate module
    allocate(area(mxncomp))
    allocate(bo(mxncomp,nlinks))
    allocate(pere(mxncomp,nlinks))
    allocate(areap(mxncomp,nlinks))
    allocate(qp(mxncomp,nlinks))
    allocate(z(mxncomp,nlinks))
    allocate(depth(mxncomp))
    allocate(sk(mxncomp,nlinks))
    allocate(co(mxncomp))
    allocate(dx(mxncomp,nlinks))
    allocate(volRemain(mxncomp-1,nlinks))
    allocate(froud(mxncomp))
    allocate(courant(mxncomp-1))
    allocate(oldQ(mxncomp, nlinks))
    allocate(newQ(mxncomp, nlinks))
    allocate(oldArea(mxncomp, nlinks))
    allocate(newArea(mxncomp, nlinks))
    allocate(oldY(mxncomp, nlinks))
    allocate(newY(mxncomp, nlinks))
    allocate(lateralFlow(mxncomp,nlinks))
    allocate(celerity(mxncomp, nlinks))
    allocate(diffusivity(mxncomp, nlinks))
    allocate(celerity2(mxncomp))
    allocate(diffusivity2(mxncomp))
    allocate(eei(mxncomp))
    allocate(ffi(mxncomp))
    allocate(exi(mxncomp))
    allocate(fxi(mxncomp))
    allocate(qpx(mxncomp, nlinks))
    allocate(qcx(mxncomp))
    allocate(dimensionless_Cr(mxncomp-1,nlinks))
    allocate(dimensionless_Fo(mxncomp-1,nlinks))
    allocate(dimensionless_Fi(mxncomp-1,nlinks))
    allocate(dimensionless_Di(mxncomp-1,nlinks))
    allocate(dimensionless_Fc(mxncomp-1,nlinks))
    allocate(dimensionless_D(mxncomp-1,nlinks))
    allocate(lowerLimitCount(nlinks))
    allocate(higherLimitCount(nlinks))
    allocate(currentRoutingNormal(mxncomp-1,nlinks))
    allocate(routingNotChanged(mxncomp-1,nlinks))
    allocate(elevTable(nel))
    allocate(areaTable(nel))
    allocate(pereTable(nel))
    allocate(rediTable(nel))
    allocate(convTable(nel))
    allocate(topwTable(nel))
    allocate(skkkTable(nel))
    allocate(nwi1Table(nel))
    allocate(dPdATable(nel))
    allocate(ncompElevTable(nel))
    allocate(ncompAreaTable(nel))
    allocate(xsec_tab(11, nel, mxncomp, nlinks))
    allocate(rightBank(mxncomp, nlinks), leftBank(mxncomp, nlinks))
    allocate(skLeft(mxncomp, nlinks), skMain(mxncomp, nlinks), skRight(mxncomp, nlinks))
    allocate(currentSquareDepth(nel))
    allocate(ini_y(nlinks))
    allocate(ini_q(nlinks))
    allocate(notSwitchRouting(nlinks))
    allocate(currentROutingDiffusive(nlinks))
    allocate(tarr_ql(nts_ql_g), varr_ql(nts_ql_g))
    allocate(tarr_ub(nts_ub_g), varr_ub(nts_ub_g))
    allocate(tarr_qtrib(nts_qtrib_g), varr_qtrib(nts_qtrib_g))
    allocate(dmy_frj(nlinks))
    allocate(frnw_g(nlinks,frnw_col))

  !-----------------------------------------------------------------------------
    frnw_g = frnw_ar_g ! network mapping matrix
    z      = z_ar_g    ! node elevation array
       
  !-----------------------------------------------------------------------------
  ! variable initializations

    routingNotChanged = 0
    applyNaturalSection = 1
    x = 0.0
    newQ = -999
    newY = -999
    dimensionless_Cr = -999
    dimensionless_Fo = -999
    dimensionless_Fi = -999
    dimensionless_Di = -999
    dimensionless_Fc = -999
    dimensionless_D = -999
    volRemain = -999
    t = t0*60.0     ! [min]
    q_sk_multi = 1.0
    oldQ = iniq
    newQ = oldQ
    qp = oldQ

  !-----------------------------------------------------------------------------
  ! Identify mainstem reaches and list their ids in an array

    ! Create a dummy array containing mainstem reaches
    nmstem_rch = 0
    do j = 1, nlinks
      if (frnw_g(j,3).ge.2) then ! mainstem reach identification
        nmstem_rch = nmstem_rch + 1
        dmy_frj(nmstem_rch) = j
      end if
    end do

    ! allocate and populate array for upstream reach ids
    allocate (mstem_frj(nmstem_rch))
    do jm=1, nmstem_rch
      mstem_frj(jm)= dmy_frj(jm)
    end do
    deallocate(dmy_frj)

  !-----------------------------------------------------------------------------
  ! create dx array from dx_ar_g and determine minimum dx.

    dx    = 0.
    minDx = 1e10

    do jm = 1, nmstem_rch !* mainstem reach only
      j = mstem_frj(jm)
      ncomp = frnw_g(j,1)
      do i = 1, ncomp-1
        dx(i,j) = dx_ar_g(i,j)
      end do
      minDx = min(minDx, minval(dx(1:ncomp-1,j)))
    end do
    
  !-----------------------------------------------------------------------------
  ! Build natural cross sections

    if (applyNaturalSection .eq. 1) then
      do jm = 1, nmstem_rch !* mainstem reach only
        j     = mstem_frj(jm)
        ncomp = frnw_g(j,1)
        do i = 1, ncomp
          leftBank(i,j)  = (twcc_ar_g(i,j) - tw_ar_g(i,j)) / 2.0
          rightBank(i,j) = (twcc_ar_g(i,j) - tw_ar_g(i,j)) / 2.0 + tw_ar_g(i,j)
        end do
      end do
    end if

    do jm = 1, nmstem_rch !* mainstem reach only
        j     = mstem_frj(jm)
        ncomp = frnw_g(j,1)
        if (applyNaturalSection .eq. 0) then
          ! no option for this, yet
          
        else
          do i=1,ncomp
            skLeft(i,j) = 1.0 / manncc_ar_g(i,j)
            skRight(i,j)= 1.0 / manncc_ar_g(i,j)
            skMain(i,j) = 1.0 / mann_ar_g(i,j)

            call readXsection(i, (1.0/skLeft(i,j)), (1.0/skMain(i,j)), &
                            (1.0/skRight(i,j)), leftBank(i,j),       &
                            rightBank(i,j), timesDepth, j, z_ar_g,   &
                            bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g)

          end do
        end if
    end do

  !-----------------------------------------------------------------------------
  ! Build time arrays for lateral flow, upstream boundary, donwstream boundary,
  ! and tributary flow

    ! time step series for lateral flow
    do n = 1, nts_ql_g
      tarr_ql(n) =    t0 * 60.0 + dt_ql * &
                      real(n-1,KIND(dt_ql))   / 60.0 ! [min]
    end do

    ! time step series for upstream boundary data
    do n = 1, nts_ub_g
      tarr_ub(n) =    t0 * 60.0 + dt_ub * &
                      real(n-1,KIND(dt_ub))   / 60.0 ! [min]
    end do

    ! time step series for tributary flow data
    do n=1, nts_qtrib_g
      tarr_qtrib(n) = t0 * 60.0 + dt_qtrib * &
                      real(n-1,KIND(dt_qtrib)) / 60.0 ! [min]
    end do
    
    ! time step series for downstream boundary data
    ! needed for coastal coupling
    ! **** COMING SOON ****

  !-----------------------------------------------------------------------------
  ! Initialize water surface elevation, channel area, and volume

    do jm = 1, nmstem_rch
      j     = mstem_frj(jm)  ! reach index
      ncomp = frnw_g(j, 1)   ! number of nodes in reach j    
      if (frnw_g(j, 2) < 0.0) then
      
        ! Initial depth at bottom node of tail water reach
        
        ! use tailwater downstream boundary observations
        ! needed for coastal coupling
        ! **** COMING SOON ****
            
        ! normal depth as TW boundary condition
        slope = (z(ncomp-1, j) - z(ncomp, j)) / dx(ncomp-1, j)
        if (slope .le. so_llm) slope = so_llm
        call normal_crit_y(ncomp, j, q_sk_multi, slope, oldQ(ncomp, j), &
                           oldY(ncomp, j), temp,  temp, temp)                                   
      else
      
        ! Initial depth at botton node of interror reach
        
        ! calculate initial depth as normal depth
        slope = (z(ncomp-1, j) - z(ncomp, j)) / dx(ncomp-1, j)
        if (slope .le. so_llm) slope = so_llm
        call normal_crit_y(ncomp, j, q_sk_multi, slope, oldQ(ncomp, j), &
                           oldY(ncomp, j), temp, temp, temp)       
      end if
              
      ! compute initial depth at interrior nodes
      newY(ncomp, j) = oldY(ncomp, j)
      call mesh_diffusive_backward(dtini_given, t0, t, tfin, saveInterval, &
                                   j, leftBank, rightBank)
      do i = 1,ncomp
      
        ! copy computed initial depths to initial depth array for first timestep
        oldY(i, j) = newY(i, j)
        
        ! Check that node elevation is not lower than bottom node.
        ! If node elevation is lower than bottom node elevation, correct.
        if (oldY(i, j) .lt. oldY(ncomp, nlinks)) oldY(i, j) = oldY(ncomp, nlinks)
        
        ! Initalize node area
        if (applyNaturalSection .eq. 0) then
          oldArea(i, j) = (oldY(i, j) - z(i, j)) * bo(i, j)
        else
          elevTable = xsec_tab(1, 1:nel, i, j)
          areaTable = xsec_tab(2, 1:nel, i, j)
          call r_interpol(elevTable, areaTable, nel, oldY(i, j), oldArea(i, j))
          if (oldArea(i, j) .eq. -9999) then
            print*, 'At j = ',j,', i = ',i, 'time =',t, &
                    'interpolation of (initial) oldArea(i,j) was not possible'
            stop
          end if   
        end if
      end do
      
      ! Initialize channel volume
      do i = 1, ncomp - 1
        volRemain(i, j) = (oldArea(i, j) + oldArea(i+1, j)) / 2.0 * dx(i, j) 
      end do  
    end do

  !-----------------------------------------------------------------------------
  ! Write tributary results to output arrays
  ! TODO: consider if this is necessary - output arrays are immediately trimmed
  ! to exclude triburay results (from MC) and pass-out only diffusive-calculated
  ! flow and depth on mainstem segments.
          
    ts_ev=1
    do while (t .le. tfin*60.0)
      if ( (mod( (t - t0 * 60.) * 60., saveInterval) .le. TOLERANCE) &
            .or. (t .eq. tfin * 60.) ) then
        do j=1, nlinks
          if (all(mstem_frj/=j)) then ! NOT a mainstem reach
            do n = 1, nts_qtrib_g
              varr_qtrib(n) = qtrib_g(n, j)
            end do
              q_ev_g(ts_ev, frnw_g(j, 1), j) = intp_y(nts_qtrib_g, tarr_qtrib, &
                                                      varr_qtrib, t)
              q_ev_g(ts_ev,            1, j) = q_ev_g(ts_ev, frnw_g(j, 1), j)
          end if
        end do
        ts_ev=ts_ev+1
      end if
      t = t + dtini/60. !* [min]
    end do
    
  !-----------------------------------------------------------------------------
  ! Initializations and re-initializations
    qpx                     = 0.
    width                   = 100.
    maxCelerity             = 1.0
    maxCelDx                = maxCelerity / minDx
    dimensionless_Fi        = 10.1
    dimensionless_Fc        = 10.1
    dimensionless_D         = 0.1
    currentROutingDiffusive = 1
    currentRoutingNormal    = 0
    routingNotChanged       = 0
    frus2                   = 9999.
    notSwitchRouting        = 0
    minNotSwitchRouting     = 10000
    minNotSwitchRouting2    = 000
    timestep                = 0
    ts_ev                   = 1
    t                       = t0 * 60.0

  !-----------------------------------------------------------------------------
  ! Ordered network routing computations

    do while ( t .lt. tfin * 60.)
      timestep = timestep + 1 ! advance timestep
      !+-------------------------------------------------------------------------
      !+                             PREDICTOR
      !+
      !+-------------------------------------------------------------------------
      do jm = 1, nmstem_rch   ! loop over mainstem reaches [upstream-to-downstream]
        j     = mstem_frj(jm) ! reach index
        ncomp = frnw_g(j,1)   ! number of nodes in reach j

        ! Calculate the duration of this timestep (dtini)
        ! Timestep duration is selected to maintain numerical stability
        if (j .eq. mstem_frj(1)) call calculateDT(t0, t, saveInterval, cfl, &
                                                tfin, maxCelDx, dtini_given)

        ! estimate lateral flow at current time t
        do i = 1, ncomp - 1
          do n = 1, nts_ql_g
            varr_ql(n) = qlat_g(n, i, j)
          end do
          lateralFlow(i, j) = intp_y(nts_ql_g, tarr_ql, varr_ql, t)
        end do

        !+++----------------------------------------------------------------
        !+ Hand over water from upstream to downstream properly according
        !+ to network connections, i.e., serial or branching.
        !+ Refer to p.52, RM1_MESH
        !+++----------------------------------------------------------------
        if (frnw_g(j, 3) > 0) then        ! reach j is not a headwater
          newQ(1, j) = 0.0
          do k = 1, frnw_g(j, 3)          ! loop over ustream connected reaches
            usrchj = frnw_g(j, 3 + k) 
            if (any(mstem_frj == usrchj)) then

              ! inflow from upstream mainstem reach
              q_usrch = newQ(frnw_g(usrchj, 1), usrchj)
              !print *, 'added flow from upstream mainstem'
            else

              ! inflow from upstream tributary reach
              do n = 1, nts_qtrib_g
                varr_qtrib(n) = qtrib_g(n, usrchj)
              end do
              tf0 = t +  dtini / 60.0
              q_usrch = intp_y(nts_qtrib_g, tarr_qtrib, varr_qtrib, tf0)
              !print *, 'added flow from tributary'
            end if

            ! add upstream flows to reach head
            newQ(1,j)= newQ(1,j) + q_usrch

          end do
          !print *, 'sum of upstream reach inflows:', newQ(1,j)
        else
        
          ! There are no links at the upstream of the reach (frnw_g(j,3)==0)
        end if

        ! Add lateral inflows to the reach head
        newQ(1,j) = newQ(1,j)+lateralFlow(1,j)*dx(1,j)
        !print *, 'sum of upstream reach inflows and lateral inflows:', newQ(1,j)
        
        call mesh_diffusive_forward(dtini_given, t0, t, tfin, saveInterval,j)

      end do  ! end of j loop for predictor
      
      !+-------------------------------------------------------------------------
      !+                             CORRECTOR
      !+
      !+-------------------------------------------------------------------------
      do jm = nmstem_rch, 1, -1 ! loop over mainstem reaches [downstream-to-upstream]
        j     = mstem_frj(jm)
        ncomp = frnw_g(j,1)
          
        !+++------------------------------------------------------------+
        !+ Downstream boundary condition for water elevation either at
        !+ a junction or TW.
        !+ Refer to p.53-1,RM1_MESH
        !+++------------------------------------------------------------+
         if (frnw_g(j, 2) .ge. 0.0) then 
         
            ! Downstream boundary at JUNCTION
            ! reach index j has a downstream connection (is NOT a tailwater reach)
              
            ! set bottom node WSEL equal WSEL in top node of downstream reach
            linknb         = frnw_g(j,2)
            newY(ncomp, j) = newY(1, int(linknb)) 
        else
        
          ! Downstream boundary at TAILWATER
          ! reach index j has NO downstream connection (it IS a tailwater reach)
            
          ! use downstream boundary data source to set bottom node WSEL value
          ! ***** COMING SOON *****

          ! Assume normal depth at tailwater downstream boundary
          slope = (z(ncomp-1,j)-z(ncomp,j))/dx(ncomp-1,j)
          if (slope .le. so_llm) slope = so_llm
          call normal_crit_y(ncomp, j, q_sk_multi, slope, newQ(ncomp,j), &
                             newY(ncomp,j), temp, newArea(ncomp,j), temp)
        end if

        ! Calculate WSEL at interrior reach nodes
        call mesh_diffusive_backward(dtini_given, t0, t, tfin, &
                                     saveInterval,j,leftBank, rightBank)
        
        ! Identify the maximum calculated celerity/dx ratio at this timestep
        ! maxCelDx is used to determine the duration of the next timestep
        if (j .eq. mstem_frj(1)) then
          maxCelDx = 0.
          maxCelerity = 0.
          do i  =1, nmstem_rch
            do kkk = 2, frnw_g(mstem_frj(i), 1)
              maxCelDx = max(maxCelDx, &
                             celerity(kkk, mstem_frj(i)) / &
                             dx(kkk-1, mstem_frj(i)))
              maxCelerity = max(maxCelerity, celerity(kkk,i))
            end do
          end do
        endif
      end do  ! end of corrector j loop

      !+-------------------------------------------------------------------------
      !+                             BOOK KEEPING
      !+
      !+-------------------------------------------------------------------------
      
      ! Calculate Froud and maximum Courant number
      do jm = 1, nmstem_rch
        j = mstem_frj(jm)
        ncomp = frnw_g(j,1)
        do i = 1, ncomp
          froud(i) = abs(newQ(i, j)) / sqrt(grav * newArea(i, j) ** 3.0 / bo(i, j))
          if (i .lt. ncomp) then
            courant(i) = (newQ(i, j) + newQ(i+1, j)) / (newArea(i, j) + newArea(i+1, j)) * dtini / dx(i, j)
          endif
        end do
        if (maxCourant .lt. maxval(courant(1:ncomp - 1))) then
          maxCourant = maxval(courant(1:ncomp-1))
        end if
      end do

      ! Calculate volume
      do jm = 1, nmstem_rch
        j = mstem_frj(jm)
        ncomp = frnw_g(j, 1)
        do i = 1, ncomp - 1
          volRemain(i, j) = (newArea(i, j) + newArea(i+1, j)) / 2.0 * dx(i, j)
        end do
      end do

      ! Advance model time
      t = t + dtini/60.
      
      !* after a warm up of 24hours, the model will not be forced to run in partial diffusive mode
      if ((t - t0 * 60.) .ge. 24. * 60.) minNotSwitchRouting2 = 100

      ! Calculate dimensionless numbers for each reach
      do jm = 1, nmstem_rch  !* mainstem reach only
        j = mstem_frj(jm)
        call calc_dimensionless_numbers(j)
      end do

      ! write results to output arrays
      if ( (mod((t - t0 * 60.) * 60., saveInterval) .le. TOLERANCE) .or. (t .eq. tfin * 60.)) then
        do jm = 1, nmstem_rch
          j     = mstem_frj(jm)
          ncomp = frnw_g(j, 1)
          do i = 1, ncomp
            q_ev_g  (ts_ev + 1, i, j) = newQ(i,j)
            elv_ev_g(ts_ev + 1, i, j) = newY(i,j)
          end do
              
          !* water elevation for tributary/mainstem upstream boundary at a junction point
          do k = 1, frnw_g(j, 3)
            usrchj = frnw_g(j, 3 + k)
            if (all(mstem_frj /= usrchj)) then
                  
              !* tributary upstream reach or mainstem upstream boundary reach
              wdepth = newY(1,j) - z(1,j)
              elv_ev_g(ts_ev+1, frnw_g(usrchj,1), usrchj) = newY(1,j)
              elv_ev_g(ts_ev+1, 1, usrchj) = wdepth + z(1, usrchj)!* test only
            endif
          end do
        end do
        
        ! Advance recording timestep
        ts_ev = ts_ev+1
      end if

      ! write initial conditions to output arrays
      if ( ( t .eq. t0 + dtini / 60 ) ) then
        do jm = 1, nmstem_rch  !* mainstem reach only
          j = mstem_frj(jm)
          ncomp = frnw_g(j, 1)
          do i = 1, ncomp
            q_ev_g  (1, i, j) = oldQ(i, j)
            elv_ev_g(1, i, j) = oldY(i, j)
          end do
              
          !* water elevation for tributary/mainstem upstream boundary at a junction point
          do k = 1, frnw_g(j,3) !* then number of upstream reaches
            usrchj = frnw_g(j, 3 + k) !* js corresponding to upstream reaches
            if (all(mstem_frj /= usrchj)) then
                  
              !* tributary upstream reach or mainstem upstream boundary reach
              wdepth = oldY(1, j) - z(1, j)
              elv_ev_g(1, frnw_g(usrchj,1), usrchj) = oldY(1,j)
              elv_ev_g(1, 1,                usrchj) = wdepth + z(1, usrchj)!* test only
            end if
          end do
        end do
      end if

      ! update of Y, Q and Area vectors
      oldY    = newY
      newY    = -999
      oldQ    = newQ
      newQ    = -999
      oldArea = newArea
      newArea = -999
      pere    = -999
      
    end do  ! end of time loop

    deallocate(frnw_g)
    deallocate(area, bo, pere, areap, qp, z,  depth, sk, co, dx) !dqp, dqc, dap, dac,
    deallocate(volRemain, froud, courant, oldQ, newQ, oldArea, newArea, oldY, newY)
    deallocate(lateralFlow, celerity, diffusivity, celerity2, diffusivity2)
    deallocate(eei, ffi, exi, fxi, qpx, qcx)
    deallocate(dimensionless_Cr, dimensionless_Fo, dimensionless_Fi)
    deallocate(dimensionless_Di, dimensionless_Fc, dimensionless_D)
    deallocate(lowerLimitCount, higherLimitCount, currentRoutingNormal, routingNotChanged)
    deallocate(elevTable, areaTable, pereTable, rediTable, convTable, topwTable)
    deallocate( skkkTable, nwi1Table, dPdATable, ncompElevTable, ncompAreaTable)
    deallocate(xsec_tab, rightBank, leftBank, skLeft, skMain, skRight)
    deallocate(currentSquareDepth, ini_y, ini_q, notSwitchRouting, currentROutingDiffusive )
    deallocate(tarr_ql, varr_ql, tarr_ub, varr_ub, tarr_qtrib, varr_qtrib)
    deallocate(mstem_frj)


  end subroutine diffnw
  
  subroutine calculateDT(initialTime, time, saveInterval, &
                         maxAllowCourantNo, tfin, max_C_dx, given_dt)
                         
    IMPLICIT NONE
  
  !-----------------------------------------------------------------------------
  ! Description:
  !   Calculate time step duration. In order to maintain numerical stability,
  !   the model timestep duration must be short enough to keep Courant numbers
  !   below a threshold level. Else, numerical instabilities will occur. 
  !
  ! Method:
  !   
  !
  ! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
  !
  ! Code Description:
  !   Language: Fortran 90.
  !   This code is written to JULES coding standards v1.
  !-----------------------------------------------------------------------------
  
  ! Subroutine arguments
    double precision, intent(in) :: initialTime       ! [sec]
    double precision, intent(in) :: time              ! [hrs]
    double precision, intent(in) :: saveInterval      ! [sec]
    double precision, intent(in) :: tfin              ! [hrs]
    double precision, intent(in) :: given_dt          ! [sec]
    double precision, intent(in) :: maxAllowCourantNo 
    double precision, intent(in) :: max_C_dx
      
  ! Local variables
    integer :: a
    integer :: b
    double precision :: dmy
    
  !-----------------------------------------------------------------------------    
  ! Calculate maximum timestep duration for numerical stability
  
    dtini = maxAllowCourantNo / max_C_dx
    a = floor( (time - initialTime * 60.) / &
               ( saveInterval / 60.))           
    b = floor(((time - initialTime * 60.) + dtini / 60.) / &
               ( saveInterval / 60.))           
    if (b .gt. a) then
      dtini = (a + 1) * (saveInterval) - (time - initialTime * 60.) * 60.
    end if

    ! if dtini extends beyond final time, then truncate it
    if (time + dtini / 60. .gt. tfin * 60.) dtini = (tfin * 60. - time) * 60.
    
  end subroutine  

  subroutine calc_dimensionless_numbers(j)
  
    IMPLICIT NONE

  !-----------------------------------------------------------------------------
  ! Description:
  !   Compute dimensionless parameters for deciding which depth computation
  !   schemes to use - normal depth or diffusive depth.
  !
  ! Method:
  !   
  !
  ! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
  !
  ! Code Description:
  !   Language: Fortran 90.
  !   This code is written to JULES coding standards v1.
  !-----------------------------------------------------------------------------

  ! Subrouting arguments
    integer, intent(in) :: j
    
  ! Local variables
    integer :: i
    integer :: ncomp
    double precision :: wl_us
    double precision :: depth_us
    double precision :: q_us
    double precision :: v_us
    double precision :: pere_us
    double precision :: r_us
    double precision :: sk_us
    double precision :: ch_us
    double precision :: wl_ds
    double precision :: depth_ds
    double precision :: q_ds
    double precision :: v_ds
    double precision :: pere_ds
    double precision :: r_ds
    double precision :: sk_ds
    double precision :: ch_ds
    double precision :: ch_star_avg
    double precision :: channel_length
    double precision :: avg_celerity
    double precision :: avg_velocity
    double precision :: avg_depth
    double precision :: maxValue
    double precision :: dimlessWaveLength
      
  !-----------------------------------------------------------------------------  
    maxValue = 1e7
    dimlessWaveLength= 4000.
    
    ncomp = frnw_g(j, 1)
    do i = 1, ncomp - 1

      ! upstream metrics
      wl_us        = newY(i, j)                       ! water level
      depth_us     = newArea(i,j) / bo(i,j)           ! depth (rectangular?)
      q_us         = newQ(i,j)                        ! flow
      v_us         = abs(newQ(i,j) / newArea(i,j) )   ! velocity
      pere_us      = pere(i,j)                        ! wetted perimiter
      r_us         = newArea(i,j) / pere(i,j)         ! hydraulic radius
      sk_us        = sk(i,j)                          ! ???
      ch_us        = sk(i,j) * r_us ** (1./6.)        ! ???

      ! downstream metrics
      wl_ds        = newY(i+1,j)                       ! water level
      depth_ds     = newArea(i+1,j) / bo(i+1,j)        ! depth (rectangular?)
      q_ds         = newQ(i+1,j)                       ! flow
      v_ds         = abs(newQ(i+1,j) / newArea(i+1,j)) ! velocity
      pere_ds      = pere(i+1,j)                       ! wetted perimiter
      r_ds         = newArea(i+1,j) / pere(i+1,j)      ! hydraulic radius
      sk_ds        = sk(i+1,j)                         ! ???
      ch_ds        = sk(i+1,j) * r_ds ** (1./6.)       ! ???


      ch_star_avg = ((ch_us + ch_ds) / 2.)  / sqrt(grav)
      avg_celerity = (celerity(i, j) + celerity(i + 1, j)) / 2.0
      avg_velocity = (v_us + v_ds) / 2.
      avg_depth = (depth_us + depth_ds) / 2.
      
      channel_length = dx(i,j)

      ! dimensionless Courant number
      dimensionless_Cr(i,j) = abs(avg_velocity / avg_celerity)
      if (dimensionless_Cr(i,j) .gt. maxValue) dimensionless_Cr(i,j) = maxValue

      ! ????
      dimensionless_Fo(i,j) = avg_velocity / sqrt(grav * avg_depth)
      if (dimensionless_Fo(i,j) .gt. maxValue) dimensionless_Fo(i,j) = maxValue

      ! ????
      dimensionless_Fi(i,j) = 2*dimensionless_Cr(i,j) / (ch_star_avg ** 2.) * &
                              dimlessWaveLength
      if (dimensionless_Fi(i,j) .gt. maxValue) dimensionless_Fi(i,j) = maxValue

      ! ????
      dimensionless_Fc(i,j) = dimensionless_Cr(i,j) * dimensionless_Fi(i,j)
      if (dimensionless_Fc(i,j) .gt. maxValue) dimensionless_Fc(i,j) = maxValue

      ! ????
      dimensionless_Di(i,j) = (dimensionless_Cr(i,j) / &
                               dimensionless_Fo(i,j)) ** 2.
      if (dimensionless_Di(i,j) .gt. maxValue) dimensionless_Di(i,j) = maxValue

      ! ????
      dimensionless_D(i,j)  = dimensionless_Di(i,j) / dimensionless_Fc(i,j)
      if (dimensionless_D(i,j) .gt. maxValue) dimensionless_D(i,j) = maxValue
        
    end do
      
  end subroutine calc_dimensionless_numbers
  

  subroutine mesh_diffusive_forward(dtini_given, t0, t, tfin, saveInterval,j)
	
    IMPLICIT NONE
    
  !-----------------------------------------------------------------------------
  ! Description:
  !   Compute discharge using diffusive equation that is numerically solved by
  !   Crank-Nicolson + Hermite Interpolation method.
  !
  ! Method:
  !   
  ! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
  !
  ! Code Description:
  !   Language: Fortran 90.
  !   This code is written to JULES coding standards v1.
  !-----------------------------------------------------------------------------

    ! Subroutine Arguments
      integer, intent(in) :: j
      double precision, intent(in) :: dtini_given
      double precision, intent(in) :: t0
      double precision, intent(in) :: t
      double precision, intent(in) :: tfin
      double precision, intent(in) :: saveInterval
    
    ! Local variables
      integer :: tableLength
      integer :: ll
      integer :: ncomp
      integer :: i
      integer :: pp
      double precision :: a1, a2, a3, a4
      double precision :: b1, b2, b3, b4
      double precision :: dd1, dd2, dd3, dd4
      double precision :: h1, h2, h3, h4
      double precision :: xt
      double precision :: allqlat
      double precision :: qy, qxy, qxxy, qxxxy
      double precision :: ppi, qqi, rri, ssi, sxi
      double precision :: mannings, Sb, width
      double precision :: cour, cour2
      double precision :: q_sk_multi
      double precision :: sfi
      double precision :: temp
      double precision :: alpha
      double precision :: y_norm_ds, y_crit_ds
      double precision :: S_ncomp
      double precision :: frds
      double precision :: area_0, width_0, hydR_0
      double precision :: errorY
      double precision :: currentQ
      double precision :: eei_ghost, ffi_ghost, exi_ghost
      double precision :: fxi_ghost, qp_ghost, qpx_ghost
    
    !-----------------------------------------------------------------------------
    !* change 20210228: All qlat to a river reach is applied to the u/s boundary
    !* Note: lateralFlow(1,j) is already added to the boundary
    
      eei = -999.
      ffi = -999.
      exi = -999.
      fxi = -999.
      eei(1) = 1.
      ffi(1) = 0.
      exi(1) = 0.
      fxi(1) = 0.
    
      ! sum of lateral inflows along the reach
      allqlat = sum(lateralFlow(2:ncomp - 1, j) * dx(2:ncomp - 1, j))

      !print *, '****** DIFFUSIVE FORWARD *****'
      !print *, '---------'
      ncomp = frnw_g(j, 1)
      do i = 2, ncomp
          
        cour = dtini / dx(i - 1, j)
        cour2= abs(celerity(i, j) ) * cour
        
        !print *, 'i:', i
        !print *, 'j:', j
        !print *, 'courant:',cour2
        !print *, '---------'
        
        a1 = 3.0 * cour2 ** 2.0 - 2.0 * cour2 ** 3.0
        a2 = 1 - a1
        a3 = ( cour2 ** 2.0 - cour2 ** 3.0 ) * dx(i-1,j)
        a4 = ( -1.0 * cour2 + 2.0 * cour2 ** 2.0 - cour2 ** 3.0 ) * dx(i-1,j)

        b1 = ( 6.0 * cour2 - 6.0 * cour2 ** 2.0 ) / ( -1.0 * dx(i-1,j) )
        b2 = - b1
        b3 = ( 2.0 * cour2 - 3.0 * cour2 ** 2.0 ) * ( -1.0 )
        b4 = ( -1.0 + 4.0 * cour2 - 3.0 * cour2 ** 2.0 ) * ( -1.0 )

        dd1 = ( 6.0 - 12.0 * cour2 ) / ( dx(i-1,j) ** 2.0 )
        dd2 = - dd1
        dd3 = ( 2.0 - 6.0 * cour2 ) / dx(i-1,j)
        dd4 = ( 4.0 - 6.0 * cour2 ) / dx(i-1,j)

        h1 = 12.0 / ( dx(i-1,j) ** 3.0 )
        h2 = - h1
        h3 = 6.0 / ( dx(i-1,j) ** 2.0 )
        h4 = h3

        if (i .eq. ncomp) then
          alpha = 1.0
        else
          alpha = dx(i,j) / dx(i-1,j)
        end if

        qy    = a1 * oldQ(i-1,j) + a2 * oldQ(i,j) + &
                a3 * qpx(i-1,j) + a4 * qpx(i,j)
        qxy   = b1 * oldQ(i-1,j) + b2 * oldQ(i,j) + &
                b3 * qpx(i-1,j) + b4 * qpx(i,j)
        qxxy  = dd1* oldQ(i-1,j) + dd2* oldQ(i,j) + &
                dd3* qpx(i-1,j) + dd4* qpx(i,j)
        qxxxy = h1 * oldQ(i-1,j) + h2 * oldQ(i,j) + &
                h3 * qpx(i-1,j) + h4 * qpx(i,j)

        ppi = - theta * diffusivity(i,j) * dtini / &
                ( dx(i-1,j) ** 2.0 ) * 2.0 / (alpha*(alpha + 1.0)) * alpha
        qqi = 1.0 - ppi * (alpha + 1.0) / alpha
        rri = ppi / alpha

        ssi = qy  + dtini * diffusivity(i,j) * ( 1.0 - theta ) * qxxy
        sxi = qxy + dtini * diffusivity(i,j) * ( 1.0 - theta ) * qxxxy

        eei(i) = -1.0 * rri / ( ppi * eei(i-1) + qqi )
        ffi(i) = ( ssi - ppi * ffi(i-1) ) / ( ppi * eei(i-1) + qqi )
        exi(i) = -1.0 * rri / ( ppi * exi(i-1) + qqi )
        fxi(i) = ( sxi - ppi * fxi(i-1) ) / ( ppi * exi(i-1) + qqi )
        
      end do
      
      ! Ghost point calculation
      cour  = dtini / dx(ncomp-1,j)
      cour2 = abs(celerity(ncomp-1,j)) * cour

      a1 = 3.0 * cour2 ** 2.0 - 2.0 * cour2 ** 3.0
      a2 = 1 - a1
      a3 = ( cour2 ** 2.0 - cour2 ** 3.0 ) * dx(ncomp-1,j)
      a4 = ( -1.0 * cour2 + 2.0 * cour2 ** 2.0 - cour2 ** 3.0 ) * dx(ncomp-1,j)

      b1 = ( 6.0 * cour2 - 6.0 * cour2 ** 2.0 ) / ( -1.0 * dx(ncomp-1,j) )
      b2 = - b1
      b3 = ( 2.0 * cour2 - 3.0 * cour2 ** 2.0 ) * ( -1.0 )
      b4 = ( -1.0 + 4.0 * cour2 - 3.0 * cour2 ** 2.0 ) * ( -1.0 )

      dd1 = ( 6.0 - 12.0 * cour2 ) / ( dx(ncomp-1,j) ** 2.0 )
      dd2 = - dd1
      dd3 = ( 2.0 - 6.0 * cour2 ) / dx(ncomp-1,j)
      dd4 = ( 4.0 - 6.0 * cour2 ) / dx(ncomp-1,j)

      h1 = 12.0 / ( dx(ncomp-1,j) ** 3.0 )
      h2 = - h1
      h3 = 6.0 / ( dx(ncomp-1,j) ** 2.0 )
      h4 = h3

      alpha = 1.0

      qy   = a1 * oldQ(ncomp,j) + a2 * oldQ(ncomp-1,j) + &
             a3 * qpx(ncomp,j) + a4 * qpx(ncomp-1,j)
      qxy  = b1 * oldQ(ncomp,j) + b2 * oldQ(ncomp-1,j) + &
             b3 * qpx(ncomp,j) + b4 * qpx(ncomp-1,j)
      qxxy = dd1* oldQ(ncomp,j) + dd2* oldQ(ncomp-1,j) + &
             dd3* qpx(ncomp,j) + dd4* qpx(ncomp-1,j)
      qxxxy= h1 * oldQ(ncomp,j) + h2 * oldQ(ncomp-1,j) + &
             h3 * qpx(ncomp,j) + h4 * qpx(ncomp-1,j)

      ppi = - theta * diffusivity(ncomp,j) * dtini / &
              ( dx(ncomp-1,j) ** 2.0 ) * 2.0 / (alpha*(alpha + 1.0)) * alpha
      qqi = 1.0 - ppi * (alpha + 1.0) / alpha
      rri = ppi / alpha

      ssi = qy  + dtini * diffusivity(ncomp-1,j) * ( 1.0 - theta ) * qxxy
      sxi = qxy + dtini * diffusivity(ncomp-1,j) * ( 1.0 - theta ) * qxxxy

      eei_ghost = -1.0 * rri / ( ppi * eei(ncomp) + qqi )
      ffi_ghost = ( ssi - ppi * ffi(ncomp) ) / ( ppi * eei(ncomp) + qqi )

      exi_ghost = -1.0 * rri / ( ppi * exi(ncomp) + qqi )
      fxi_ghost = ( sxi - ppi * fxi(ncomp) ) / ( ppi * exi(ncomp) + qqi )

      qp_ghost = oldQ(ncomp-1,j)
      qpx_ghost= 0.

      qp(ncomp,j) = eei(ncomp) * qp_ghost + ffi(ncomp)
      qpx(ncomp,j)= exi(ncomp) *qpx_ghost + fxi(ncomp)

      do i = ncomp-1,1,-1
        qp(i,j) = eei(i) * qp(i+1,j) + ffi(i)
        qpx(i,j)= exi(i) *qpx(i+1,j) + fxi(i)
      end do

      qp(1,j) = newQ(1,j)

      ! All qlat to a river reach is applied to the u/s boundary
      qp(1,j) = qp(1,j) + allqlat

      do i=1,ncomp
        if (abs(qp(i,j)) .lt. q_llm) then
          qp(i,j) = q_llm
        end if
      end do
      
      ! update newQ
      newQ(1:ncomp, j) = qp(1:ncomp, j)

    ! ============== DEBUG to find unstable flow calcls =================
    !        do i= ncomp, 1, -1
    !            if (abs(qp(i,j)) .gt. 2E4) then
    !                print *, 'j:', j
    !                print *, 'i:', i
    !                print *, 'ncomp', ncomp
    !                print *, 't:', t
    !                print *, 'calculated flow value:', qp(i,j)
    !                print *, 'previous upstream flow', oldQ(i-1,j)
    !                print *, 'diffusivity(i,j):', diffusivity(i,j)
    !                print *, 'eei(i):', eei(i)
    !                print *, 'qp(i+1,j):', qp(i+1,j)
    !                print *, 'ffi(i):', ffi(i)
    !                print *, 'qp_ghost:', qp_ghost
    !                print *, 'ffi(ncomp):', ffi(ncomp)
    !                print *, 'eei(ncomp):', eei(ncomp)
    !                print *, 'cour2:', cour2
    !                print *, 'qp(ncomp,j):', qp(ncomp,j)
    !                print *, 'allqlat:', allqlat
    !                print *, 'upstream inflow newQ(1,j):', newQ(1,j)
    !                stop
    !            end if
    !        end do
      
  end subroutine mesh_diffusive_forward
  
  subroutine mesh_diffusive_backward(dtini_given, t0, t, tfin, saveInterval,j, leftBank, rightBank)

    IMPLICIT NONE

  !-----------------------------------------------------------------------------
  ! Description:
  !   Compute water depth using diffusive momentum equation or normal depth
  !   with computed Q from the forward step.
  !
  ! Method:
  !   
  !
  ! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
  !
  ! Code Description:
  !   Language: Fortran 90.
  !   This code is written to JULES coding standards v1.
  !-----------------------------------------------------------------------------

  ! Subrouting arguments
    integer, intent(in) :: j
    double precision, intent(in) :: dtini_given
    double precision, intent(in) :: t0
    double precision, intent(in) :: t
    double precision, intent(in) :: tfin
    double precision, intent(in) :: saveInterval
      
  ! Subroutine variables
    integer :: depthCalOk(mxncomp)
    integer :: i, pp, ncomp  
    integer :: tableLength, jj, newMassBalance, iii  
    double precision :: a1, a2, a3, a4
    double precision :: b1, b2, b3, b4
    double precision :: dd1, dd2, dd3, dd4
    double precision :: h1, h2, h3, h4
    double precision :: xt
    double precision :: qy, qxy, qxxy, qxxxy
    double precision :: ppi, qqi, rri, ssi, sxi
    double precision :: mannings, Sb, width, slope
    double precision :: cour, cour2
    double precision :: q_sk_multi, sfi, temp, dkdh
    double precision :: y_norm, y_crit, area_n, area_c, chnWidth, vel
    double precision :: y_norm_ds, y_crit_ds, S_ncomp
    double precision :: frds, area_0width_0, hydR_0
    double precision :: errorY, currentQ, stg1, stg2
    double precision :: elevTable_1(nel),areaTable_1(nel)
    double precision :: rediTable_1(nel),convTable_1(nel),topwTable_1(nel)
    double precision :: skkkTable_1(nel), dKdATable_1(nel)
    double precision :: pereTable_1(nel),currentSquaredDepth_1(nel)                                                     
    double precision :: depthYi,tempDepthi_1
    double precision :: tempCo_1,temp_q_sk_multi_1
    double precision :: tempY_1,tempArea_1,tempRadi_1,tempbo_1,ffy
    double precision :: ffy_1, ffy_2, ffy_3
    double precision :: tempCo_2, tempCo_3, tempsfi_2, tempsfi_3
    double precision :: ffprime,tempDepthi_1_new,tempsfi_1,toll, dkda
    doubleprecision :: tempPere_1, tempsk_1, tempY_2, tempY_3, tempdKdA_1
    double precision, dimension(mxncomp, nlinks) :: leftBank, rightBank

  !-----------------------------------------------------------------------------
    ncomp = frnw_g(j, 1)
    S_ncomp = (-z(ncomp, j) + z(ncomp-1, j)) / dx(ncomp-1, j)
    depthCalOk(ncomp) = 1
    elevTable = xsec_tab(1, :,ncomp, j)
    areaTable = xsec_tab(2, :,ncomp, j)
    topwTable = xsec_tab(6, :,ncomp, j)
    
    ! Estimate channel area @ downstream boundary
    call r_interpol(elevTable, areaTable, nel, &
                    newY(ncomp, j), newArea(ncomp, j))
    if (newArea(ncomp,j) .eq. -9999) then
      print*, 'At j = ',j,', i = ',ncomp, 'time =',t, &
              'interpolation of newArea was not possible'
!     stop
    end if

    ! Estimate channel bottom width @ downstream boundary
    call r_interpol(elevTable, topwTable, nel, &
                    newY(ncomp, j), bo(ncomp, j))
    if (bo(ncomp, j) .eq. -9999) then
      print*, 'At j = ',j,', i = ',ncomp, 'time =',t, &
              'interpolation of bo was not possible'
!     stop
    end if

    do i = ncomp, 1, -1 ! loop through reach nodes [downstream-upstream]
      currentQ = qp(i, j)
      q_sk_multi=1.0

      elevTable = xsec_tab(1,:,i,j)
      convTable = xsec_tab(5,:,i,j)
      areaTable = xsec_tab(2,:,i,j)
      pereTable = xsec_tab(3,:,i,j)
      topwTable = xsec_tab(6,:,i,j)
      skkkTable = xsec_tab(11,:,i,j)
              
      xt=newY(i, j)
      
      ! Estimate co(i) (???? what is this?) by interpolation
      currentSquareDepth = (elevTable - z(i, j)) ** 2.
      call r_interpol(currentSquareDepth, convTable, nel, &
                      (newY(i, j)-z(i, j)) ** 2.0, co(i)) 
      if (co(i) .eq. -9999) then
        print*, 'At j = ',j,', i = ',i, 'time =',t, &
                'interpolation of conveyence was not possible, wl', &
                newY(i,j), 'z',z(i,j),'previous wl',newY(i+1,j), &
                'previous z',z(i+1,j), 'dimensionless_D(i,j)', &
                dimensionless_D(i,j)
  !      stop
      end if
      co(i) =q_sk_multi * co(i)

      ! Estimate channel area by interpolation
      call r_interpol(elevTable, areaTable, nel, &
                      xt, newArea(i,j))
      if (newArea(i,j) .eq. -9999) then
        print*, 'At j = ',j,', i = ',i, 'time =',t, &
                'interpolation of newArea was not possible'
!        stop
      end if
      
      ! Estimate wetted perimiter by interpolation
      call r_interpol(elevTable, pereTable, nel, &
                      xt, pere(i, j))
      
      ! Estimate width by interpolation
      call r_interpol(elevTable, topwTable, nel, &
                      xt, bo(i, j))
      
      ! Estimate sk(i,j) by interpolation (???? what is sk?)
      call r_interpol(elevTable, skkkTable, nel, &
                      xt, sk(i, j))

      sfi = qp(i, j) * abs(qp(i, j)) / (co(i) ** 2.0)
      chnWidth = rightBank(i, j)-leftBank(i, j)
      chnWidth = min(chnWidth, bo(i, j))

      ! ???? What exactly is happening, here ????
      if (depthCalOk(i) .eq. 1) then
      
        ! Calculate celerity, diffusivity and velocity
        celerity2(i)    = 5.0 / 3.0 * abs(sfi) ** 0.3 * &
                          abs(qp(i, j)) ** 0.4 / bo(i, j) ** 0.4 &
                          / (1. / (sk(i, j) * q_sk_multi)) ** 0.6
        diffusivity2(i) = abs(qp(i, j)) / 2.0 / bo(i, j) / abs(sfi)
        vel             = qp(i, j) / newArea(i, j)
        
        ! Check celerity value
        if (celerity2(i) .gt. 3.0 * vel) celerity2(i) = vel * 3.0
      else
        if (qp(i, j) .lt. 1) then
          celerity2(i) = C_llm
        else
          celerity2(i) = 1.0
        end if
        diffusivity2(i) =  diffusivity(i,j)
      end if
      
      newMassBalance = 0
      ! ???? What is happening here, the line skips whatever this is.
      if (newMassBalance .eq. 1) then
        if (i .gt. 1) then
          newArea(i - 1,j) = oldArea(i - 1, j) + oldArea(i, j) - &
                             newArea(i, j) - 2. * dtini / dx(i-1, j) &
                             *(qp(i, j) - qp(i-1, j))
          elevTable = xsec_tab(1,:,i-1,j)
          areaTable = xsec_tab(2,:,i-1,j)
          if (newArea(i-1, j) .le. 0) then
            slope = (z(i-1, j)-z(i, j))/dx(i-1, j)
            call normal_y(i-1, j, q_sk_multi, slope, qp(i-1,j), &
                          newY(i-1,j), temp, newArea(i-1,j), temp)
            currentRoutingNormal(i-1,j) = 1
          else
            call r_interpol(areaTable, elevTable, nel, &
                            newArea(i-1,j), newY(i-1,j))
            currentRoutingNormal(i-1,j) = 0
          end if
        end if
      else
        if (i .gt. 1) then
        
          ! If routing method is changed just a few time steps ago, 
          ! we maintain the same routing to avoid oscillation
          if ((routingNotChanged(i-1, j) .lt. minNotSwitchRouting2) &
               .and. (currentRoutingNormal(i-1, j) .lt. 3)) then
            if (currentRoutingNormal(i-1, j) .eq. 0) then
              newY(i-1,j) = newY(i, j) + sfi * dx(i-1, j)
            else if (currentRoutingNormal(i-1, j) .eq. 1) then
              slope = (z(i-1, j) - z(i, j)) / dx(i-1, j)
              if (slope .le. so_llm) slope = so_llm
              q_sk_multi = 1.0
              
              ! applying normal depth to all the nodes
              call normal_crit_y(i-1, j, q_sk_multi, slope, qp(i-1,j), &
                                 newY(i-1,j), temp, newArea(i-1,j), temp)
            end if
          else
          
            !! If DSP: D is below 1.0, we switch to partial diffusive routing
            if (dimensionless_D(i-1, j) .lt. DD_ulm) then
            
              ! ====================================================
              ! normal depth calculation
              ! ====================================================
              !print *, 'NORMAL DEPTH CALCULATION'
              
              slope = (z(i-1, j) - z(i, j)) / dx(i-1, j)
              if (slope .le. so_llm) slope = so_llm
              q_sk_multi = 1.0
                  
              ! using normal depth to calculate node depth
              call normal_crit_y(i-1, j, q_sk_multi, slope, qp(i-1, j), &
                                 newY(i-1,j), temp, newArea(i-1,j), temp)
              
              ! Book-keeping: changing from full diffusive to partial diffusive
              if (currentRoutingNormal(i-1, j) .ne. 1 ) &
                  routingNotChanged(i-1, j) = 0
              currentRoutingNormal(i-1, j) = 1
                    
            !! If DSP: D is not below 1.0, we switch to full diffusive routing
            else if ((dimensionless_D(i-1, j) .ge. DD_ulm) &
                     .and. (dimensionless_D(i-1, j) .lt. DD_llm)) then
                
              ! ====================================================
              ! hybrid - diffusive/normal - depth calculation
              ! ====================================================
              !print *, 'HYBRID DEPTH CALCULATION'
              
              slope = (z(i-1, j) - z(i, j)) / dx(i-1, j)
              if (slope .le. so_llm) slope = so_llm
              q_sk_multi = 1.0

              ! applying normal depth to all the nodes
              call normal_crit_y(i-1, j, q_sk_multi, slope, qp(i-1,j), &
                                 stg1, temp, newArea(i-1,j), temp)

              ! weighted average transitional depth calculation
              stg2 = newY(i,j) + sfi * dx(i-1, j)
              newY(i-1, j) = (stg2 * (dimensionless_D(i-1, j) - DD_ulm) + &
                              stg1 * (DD_llm - dimensionless_D(i-1, j))) &
                              / (DD_llm - DD_ulm)

              ! Book-keeping: changing from full diffusive to partial diffusive
              if (currentRoutingNormal(i-1,j) .ne. 3 ) &
                  routingNotChanged(i-1,j) = 0
              currentRoutingNormal(i-1,j) = 3
                    
            else
            
              ! =====================================
              ! pure diffusive depth calculation
              ! =====================================
              !print *, 'PURE DIFFUSIVE DEPTH CALCULATION'

              slope = (z(i-1, j) - z(i, j)) / dx(i-1, j) 
              ! <------- why are we not imposing the slope limit, here?
              !if (slope .le. so_llm) slope = so_llm
              
              depthYi = newY(i,j) - z(i,j)
              tempDepthi_1 = oldY(i-1,j)-z(i-1,j)
              elevTable_1  = xsec_tab(1,  :, i-1, j)
              areaTable_1  = xsec_tab(2,  :, i-1, j)
              pereTable_1  = xsec_tab(3,  :, i-1, j)
              rediTable_1  = xsec_tab(4,  :, i-1, j)
              convTable_1  = xsec_tab(5,  :, i-1, j)
              topwTable_1  = xsec_tab(6,  :, i-1, j)
              dKdATable_1  = xsec_tab(9,  :, i-1, j)
              skkkTable_1  = xsec_tab(11, :, i-1, j)
              currentSquaredDepth_1=(elevTable_1 - z(i-1, j)) ** 2.0
              toll = 1.0
              iii = 0
              ! Applying Newton\96Raphson method
              if (newtonRaphson .eq. 1) then
                do while (abs(toll) .gt. 0.001)
                  iii = iii + 1
                  tempY_1 = tempDepthi_1 + z(i-1, j)

                  call r_interpol(currentSquaredDepth_1, convTable_1, nel, &
                                  (tempDepthi_1) ** 2.0, tempCo_1)

                  temp_q_sk_multi_1=1.0
                  tempCo_1 = tempCo_1 * temp_q_sk_multi_1

                  call r_interpol(elevTable_1, areaTable_1, nel, &
                                  tempY_1, tempArea_1)
                  call r_interpol(elevTable_1, pereTable_1, nel, &
                                  tempY_1, tempPere_1)
                  call r_interpol(elevTable_1, rediTable_1, nel, &
                                  tempY_1,tempRadi_1)
                  call r_interpol(elevTable_1, topwTable_1, nel, &
                                  tempY_1,tempbo_1)
                  call r_interpol(elevTable_1, dKdATable_1, nel, &
                                  tempY_1,tempdKdA_1)

                  tempsfi_1 = qp(i-1, j) * abs(qp(i-1, j)) / (tempCo_1 ** 2.0)

                  ffy = tempDepthi_1 - depthYi + dx(i-1,j) * slope &
                        - 0.5 * dx(i-1, j) * (sfi + tempsfi_1)

                  dkda = tempdKdA_1 

                  ffprime = 1 + dx(i-1, j) * tempbo_1 *  qp(i-1, j) * &
                            abs(qp(i-1, j)) / (tempCo_1 ** 3.0) * dkda

                  tempDepthi_1_new = tempDepthi_1 - ffy / ffprime

                  tempDepthi_1_new = max(tempDepthi_1_new, 0.005)

                  toll = abs(tempDepthi_1_new - tempDepthi_1)

                  if(iii .gt. 30)then
                    print*, 'Warning: Depth iteration reached maximum trial at j=', &
                            j, 'i=', i-1 , 'and', i,'depths are', tempDepthi_1,     &
                            tempDepthi_1_new, 'slope=', slope, 'dx=', dx(i-1,j),    &
                            'depth at d/s',depthYi, 'Q-s are', qp(i-1,j), qp(i,j)
                    depthCalOk(i-1) = 0
                    EXIT
                  end if

                  tempDepthi_1 = tempDepthi_1_new
                  depthCalOk(i-1) = 1
                end do
              end if

              ! Applying mid point bisection
              if (newtonRaphson .eq. 0) then
                tempY_1 = elevTable_1(2)
                tempY_2 = depthYi * 3. + z(i-1, j)
                tempY_3 = (tempY_1 + tempY_2) / 2.
                do while (abs(toll) .gt. 0.001)
                  iii = iii +1

                  call r_interpol(currentSquaredDepth_1, convTable_1, nel, &
                                  (tempY_1-z(i-1,j)) ** 2.0, tempCo_1)
                  call r_interpol(currentSquaredDepth_1, convTable_1, nel, &
                                  (tempY_2-z(i-1,j)) ** 2.0, tempCo_2)
                  call r_interpol(currentSquaredDepth_1, convTable_1, nel, &
                                  (tempY_3-z(i-1,j)) ** 2.0, tempCo_3)

                  temp_q_sk_multi_1 = 1.0
                  tempCo_1 = tempCo_1 * temp_q_sk_multi_1
                  tempCo_2 = tempCo_2 * temp_q_sk_multi_1
                  tempCo_3 = tempCo_3 * temp_q_sk_multi_1

                  tempsfi_1 = qp(i-1, j) * abs(qp(i-1, j)) / ( tempCo_1 ** 2.0 )
                  tempsfi_2 = qp(i-1, j) * abs(qp(i-1, j)) / ( tempCo_2 ** 2.0 )
                  tempsfi_3 = qp(i-1, j) * abs(qp(i-1, j)) / ( tempCo_3 ** 2.0 )

                  ffy_1 = (tempY_1-z(i-1, j)) - depthYi + dx(i-1, j) * slope &
                           - 0.5 * dx(i-1, j) * (sfi + tempsfi_1)
                  ffy_2 = (tempY_2-z(i-1, j)) - depthYi + dx(i-1, j) * slope &
                           - 0.5 * dx(i-1, j) * (sfi + tempsfi_2)
                  ffy_3 = (tempY_3-z(i-1, j)) - depthYi + dx(i-1, j) * slope &
                           - 0.5 * dx(i-1, j) * (sfi + tempsfi_3)

                  if ((ffy_1 * ffy_2) .gt. 0.) then
                    tempY_2 = (tempY_2 - z(i-1, j)) * 2.0 + z(i-1, j)
                  else if ((ffy_1 * ffy_3) .le. 0.) then
                    tempY_2 = tempY_3
                  else if ((ffy_2 * ffy_3) .le. 0.) then
                    tempY_1 = tempY_3
                  end if
                  tempY_3 = (tempY_1 + tempY_2) / 2.0
                  toll = tempY_2 - tempY_1
                  tempDepthi_1 = tempY_3 - z(i-1, j)
                  depthCalOk(i-1) = 1
                end do
              end if
              
              newY(i-1,j) = tempDepthi_1 + z(i-1, j)

              if (newY(i-1, j) .gt. 10.0**5.) newY(i-1, j) = 10.0**5.
              
              ! Book-keeping: changing from partial diffusive to full diffusive
              if ( currentRoutingNormal(i-1, j) .ne. 0 ) routingNotChanged(i-1,j) = 0
              currentRoutingNormal(i-1,j) = 0
                
            end if    
          end if

          if (newY(i-1,j)-z(i-1,j) .le. 0.) then
            print *, ' newY(i-1,j)-z(i-1,j): ', newY(i-1,j)-z(i-1,j)
            print *, ' newY(i-1,j): ', newY(i-1,j)
            print *, 'z(i-1,j): ', z(i-1,j)
            print *, 'qp(i-1,j): ', qp(i-1,j)
            print*, 'depth is negative at time=,', t,'j= ', j,'i=',i-1, &
                    'newY =', (newY(jj,j),jj=1,ncomp)
            print*, 'dimensionless_D',(dimensionless_D(jj,j),jj=1,ncomp)
            print*, 'newQ',(newQ(jj,j),jj=1,ncomp)
            print*, 'Bed',(z(jj,j),jj=1,ncomp)
            print*, 'dx',(dx(jj,j),jj=1,ncomp-1)
          end if
        
        end if
      end if

      ! Book-keeping: Counting the number as for how many time steps
      ! the routing method is unchanged
      if (i.gt.1) then
        routingNotChanged(i-1, j) = routingNotChanged(i-1, j) + 1
      endif
            
    end do

    celerity(1:ncomp, j) =  sum(celerity2(1:ncomp)) / ncomp
    if (celerity(1, j) .lt. C_llm) celerity(1:ncomp,j) = C_llm
    diffusivity(1:ncomp, j) = sum(diffusivity2(1:ncomp)) / ncomp
    do i = 1, ncomp
      if (diffusivity(i, j) .gt. D_ulm) diffusivity(i, j) = D_ulm !!! Test
      if (diffusivity(i, j) .lt. D_llm) diffusivity(i, j) = D_llm !!! Test
    end do

  end subroutine mesh_diffusive_backward
  
    !**-----------------------------------------------------------------------------------------
    !*      Create lookup tables at each node storing computed values of channel geometries
    !*      such as area and conveyance and normal/critical depth for possible ranges of
    !*      water depth.
    !
    !**-----------------------------------------------------------------------------------------
    subroutine readXsection(k,lftBnkMann,rmanning_main,rgtBnkMann,leftBnkX_given,rghtBnkX_given,timesDepth,num_reach,&
                            z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g )
        implicit none
        save

        integer, intent(in) :: k, num_reach
        doubleprecision, intent(in) :: rmanning_main,lftBnkMann,rgtBnkMann,leftBnkX_given,rghtBnkX_given, timesDepth
        double precision, dimension(mxncomp, nlinks), intent(in) :: z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g
        doubleprecision, dimension(:), allocatable :: xcs, ycs
        doubleprecision, dimension(:,:), allocatable :: el1, a1, peri1, redi1
        doubleprecision, dimension(:), allocatable :: redi1All
        doubleprecision, dimension(:,:), allocatable :: conv1, tpW1, diffArea, newI1, diffPere
        doubleprecision, dimension(:), allocatable :: newdPdA, diffAreaAll, diffPereAll, newdKdA       ! change Nazmul 20210601
        doubleprecision, dimension(:), allocatable :: compoundSKK, elev
        integer, dimension(:), allocatable :: i_start, i_end, totalNodes
        doubleprecision, dimension(:,:), allocatable :: allXcs, allYcs
        integer :: i_area, i_find, i, j, jj, num
        doubleprecision :: el_min, el_max, el_range, el_incr, el_now, x1, y1, x2, y2, x_start, x_end, waterElev, leftBnkX,rghtBnkX
        doubleprecision :: f2m, cal_area, cal_peri, cal_topW,  diffAreaCenter
        doubleprecision :: compoundMann, el_min_1
        integer:: i1, i2
        doubleprecision :: leftBnkY, rghtBnkY,rmanning
        integer:: mainChanStrt, mainChanEnd,  kkk, startFound, endFound
        doubleprecision :: hbf

        allocate (el1(nel,3),a1(nel,3),peri1(nel,3),redi1(nel,3),redi1All(nel))
        allocate (conv1(nel,3), tpW1(nel,3), diffArea(nel,3), newI1(nel,3), diffPere(nel,3))
        allocate (newdPdA(nel), diffAreaAll(nel), diffPereAll(nel), newdKdA(nel))       ! change Nazmul 20210601
        allocate (compoundSKK(nel), elev(nel))
        allocate (i_start(nel), i_end(nel))
        allocate (totalNodes(3))


        leftBnkX=leftBnkX_given
        rghtBnkX=rghtBnkX_given
        startFound = 0
        endFound = 0
        !* channel geometry at a given segment
        z_g = z_ar_g(k, num_reach)
        bo_g= bo_ar_g(k, num_reach)
        traps_g=traps_ar_g(k, num_reach)
        tw_g= tw_ar_g(k, num_reach)
        twcc_g= twcc_ar_g(k, num_reach)
        hbf= (tw_g-bo_g)/(2.0*traps_g) !* bankfull depth
        maxTableLength=8
        f2m=1.0
        allocate (xcs(maxTableLength), ycs(maxTableLength))
        allocate (allXcs(maxTableLength,3), allYcs(maxTableLength,3))
        do i=1, maxTableLength
            !* channel x-section vertices at a given segment
            if (i==1) then
                x1=0.0; y1=z_g + timesDepth*hbf
            elseif (i==2) then
                x1=0.0; y1=z_g + hbf
            elseif (i==3) then
                x1=(twcc_g-tw_g)/2.0; y1= z_g + hbf
            elseif (i==4) then
                x1= xcs(3) + traps_g*hbf; y1= z_g
            elseif (i==5) then
                x1= xcs(4) + bo_g; y1= z_g
            elseif (i==6) then
                x1= xcs(5) + traps_g*hbf; y1= z_g + hbf
            elseif (i==7) then
                x1= twcc_g; y1= z_g + hbf
            elseif (i==8) then
                x1= xcs(7); y1= z_g + timesDepth*hbf
            endif

            xcs(i)=x1*f2m
            ycs(i)=y1*f2m
            if ((xcs(i) .ge. leftBnkX) .and. (startFound .eq. 0)) then
                mainChanStrt = i-1
                startFound = 1
            end if
            if ((xcs(i) .ge. rghtBnkX) .and. (endFound .eq. 0)) then
                mainChanEnd = i-1
                endFound = 1
            end if
        enddo
        mainChanStrt=3
        mainChanEnd=6
        num=i

        if (leftBnkX .lt. minval(xcs(2:num-1))) leftBnkX = minval(xcs(2:num-1))
        if (rghtBnkX .gt. maxval(xcs(2:num-1))) rghtBnkX = maxval(xcs(2:num-1))

        leftBnkY = ycs(mainChanStrt)+(leftBnkX-xcs(mainChanStrt))/&
          (xcs(mainChanStrt+1)-xcs(mainChanStrt))*(ycs(mainChanStrt+1)-ycs(mainChanStrt))
        rghtBnkY = ycs(mainChanEnd)+(rghtBnkX-xcs(mainChanEnd))/&
          (xcs(mainChanEnd+1)-xcs(mainChanEnd))*(ycs(mainChanEnd+1)-ycs(mainChanEnd))
        el_min=99999.
        el_max=-99999.
        do i=2,num-1
            if(ycs(i).lt.el_min)el_min=ycs(i)
            if(ycs(i).gt.el_max)el_max=ycs(i)
        enddo
        el_range=(el_max-el_min)*2.0 ! change Nazmul 20210601

        do i=1, 3
            allXcs(i+1,1)=xcs(i) !x1*f2m
            allYcs(i+1,1)=ycs(i) !y1*f2m
        enddo
        allXcs(1,1)=xcs(1)
        allYcs(1,1)=el_min+el_range+1.
        allXcs(mainChanStrt+2,1)=xcs(3)
        allYcs(mainChanStrt+2,1)=el_min+el_range+1.

        do i=3,4
            allXcs(i-1,2)=xcs(i) !x1*f2m
            allYcs(i-1,2)=ycs(i) !y1*f2m
        enddo

        do i=5,6
            allXcs(i,2)=xcs(i) !x1*f2m
            allYcs(i,2)=ycs(i) !y1*f2m
        enddo
        allXcs(1,2)=xcs(3)
        allYcs(1,2)=el_min+el_range+1.
        allXcs(7,2)=xcs(6)
        allYcs(7,2)=el_min+el_range+1.

        do i=6,8
            allXcs(i-4,3)=xcs(i) !x1*f2m
            allYcs(i-4,3)=ycs(i) !y1*f2m
        enddo
        allXcs(1,3) = allXcs(2,3)
        allYcs(1,3) = el_min+el_range+1.
        i=5
        allXcs(i,3) = allXcs(i-1,3)
        allYcs(i,3) = el_min+el_range+1.

        totalNodes(1) = 5
        totalNodes(2) = 7
        totalNodes(3) = 5

        allXcs(4,2) = (allXcs(3,2)+allXcs(5,2))/2.0
        allYcs(4,2) = allYcs(3,2) - 0.01

        el_min_1 = el_min
        el_min = allYcs(4,2)    ! change Nazmul 20210601 ! corrected

        elev(1) = el_min
        elev(2) = el_min + 0.01/4.
        elev(3) = el_min + 0.01/4.*2.
        elev(4) = el_min + 0.01/4.*3.
        elev(5) = el_min + 0.01

        el_incr=el_range/real(nel-6.0)

        do kkk = 6,nel
            elev(kkk) = elev(5)+el_incr * (kkk-5)
        end do

        xcs = 0.
        ycs = 0.
        newI1=0.0 !Hu changed
        do kkk=1,3
            num = totalNodes(kkk)
            xcs(1:num) = allXcs(1:num,kkk)
            ycs(1:num) = allYcs(1:num,kkk)
            if (kkk .eq. 1) rmanning = lftBnkMann
            if (kkk .eq. 2) rmanning = rmanning_main
            if (kkk .eq. 3) rmanning = rgtBnkMann
            do j=1,nel
                el_now = elev(j)
                if(abs(el_now - el_min) < TOLERANCE) then
                    el_now=el_now+0.00001
                end if
                i_start(1)=-999
                i_end(1)=-999
                i_area=0
                i_find=0
                do i=1,num-1
                    y1=ycs(i)
                    y2=ycs(i+1)
                    if(el_now.le.y1 .and. el_now.gt.y2 .and. i_find.eq.0)then
                        i_find=1
                        i_area=i_area+1
                        i_start(i_area)=i
                    endif
                    if(el_now.gt.y1 .and. el_now.le.y2 .and. i_find.eq.1)then
                        i_find=0
                        i_end(i_area)=i
                    endif
                enddo

                cal_area=0.
                cal_peri=0.
                cal_topW=0.

                do i=1,i_area
                    x1=xcs(i_start(i))
                    x2=xcs(i_start(i)+1)
                    y1=ycs(i_start(i))
                    y2=ycs(i_start(i)+1)
                    if(y1.eq.y2)then
                        x_start=x1
                    else
                        x_start=x1+(el_now-y1)/(y2-y1)*(x2-x1)
                    endif

                    x1=xcs(i_end(i))
                    x2=xcs(i_end(i)+1)
                    y1=ycs(i_end(i))
                    y2=ycs(i_end(i)+1)

                    if(y1.eq.y2)then
                      x_end=x1
                    else
                      x_end=x1+(el_now-y1)/(y2-y1)*(x2-x1)
                    endif

                    cal_topW=x_end-x_start+cal_topW

                    i1=i_start(i)
                    i2=i_end(i)

                    cal_area = cal_area    &
                             +cal_tri_area(el_now,x_start,xcs(i1+1),ycs(i1+1))    &
                             +cal_multi_area(el_now,xcs,ycs,maxTableLength,i1+1,i2)    &
                             +cal_tri_area(el_now,x_end,xcs(i2),ycs(i2))
                    cal_peri = cal_peri    &
                            +cal_dist(x_start,el_now,xcs(i1+1),ycs(i1+1))    &
                            +cal_perimeter(xcs,ycs,maxTableLength,i1+1,i2)    &
                            +cal_dist(x_end,el_now,xcs(i2),ycs(i2))
                    if(i1.eq.1)cal_peri=cal_peri    &
                             -cal_dist(x_start,el_now,xcs(i1+1),ycs(i1+1))
                    if(i2.eq.(num-1))cal_peri=cal_peri    &
                             -cal_dist(x_end,el_now,xcs(i2),ycs(i2))

                enddo

                el1(j,kkk)=el_now
                a1(j,kkk)=cal_area
                peri1(j,kkk)=cal_peri
                redi1(j,kkk)=a1(j,kkk)/peri1(j,kkk)

                conv1(j,kkk)=1./rmanning*a1(j,kkk)*(redi1(j,kkk))**(2./3.)
                if (peri1(j,kkk) .le. TOLERANCE) then
                    redi1(j,kkk) =0.0; conv1(j,kkk)=0.0
                endif
                tpW1(j,kkk)=cal_topW

                if(j.eq.1) then !* Dongha added
                    diffArea(j,kkk)=a1(j,kkk) !* Dongha added
                    diffPere(j,kkk)=peri1(j,kkk) !* Dongha added
                else
                    if (el_now .le. minval(ycs(1:num))) then
                      diffArea(j,kkk)=a1(j,kkk)
                      diffPere(j,kkk)=peri1(j,kkk)
                    else
                      diffArea(j,kkk)=a1(j,kkk)-a1(j-1,kkk)
                      diffPere(j,kkk)=peri1(j,kkk)-peri1(j-1,kkk)
                    endif
                endif

                waterElev=el1(j,kkk)
                do jj=2,j
                  diffAreaCenter=el1(jj,kkk)-(el1(jj,kkk)-el1(jj-1,kkk))*0.5
                  newI1(j,kkk)=newI1(j,kkk)+diffArea(jj,kkk)*(waterElev-diffAreaCenter)
                enddo
            end do
        end do

        do j = 1,nel
            el_now=el1(j,1)
            if (j .eq. 1) then
                newdPdA(j) = sum(peri1(j,:)) / sum(a1(j,:))
                newdKdA(j) = sum(conv1(j,:)) / sum(a1(j,:))     ! change Nazmul 20210601
            else
                newdPdA(j)= (sum(peri1(j,:)) - sum(peri1(j-1,:))) / (sum(a1(j,:)) - sum(a1(j-1,:)))
                newdKdA(j)= (sum(conv1(j,:)) - sum(conv1(j-1,:))) / (sum(a1(j,:)) - sum(a1(j-1,:)))
            end if

            compoundMann = sqrt((abs(peri1(j,1))*lftBnkMann ** 2. + abs(peri1(j,2))*rmanning_main ** 2.+&
             abs(peri1(j,3))*rgtBnkMann ** 2.) / (abs(peri1(j,1))+abs(peri1(j,2))+abs(peri1(j,3))))
            compoundSKK(j) = 1. / compoundMann

            redi1All(j)=sum(a1(j,:)) /sum(peri1(j,:))
            xsec_tab(1,j,k,num_reach) = el1(j,1)
            xsec_tab(2,j,k,num_reach) = sum(a1(j,:))
            xsec_tab(3,j,k,num_reach) = sum(peri1(j,:))
            xsec_tab(4,j,k,num_reach) = redi1All(j)
            xsec_tab(5,j,k,num_reach) = sum(conv1(j,:))
            xsec_tab(6,j,k,num_reach) = abs(tpW1(j,1))+abs(tpW1(j,2))+abs(tpW1(j,3))
            xsec_tab(7,j,k,num_reach) = sum(newI1(j,:))
            xsec_tab(8,j,k,num_reach) = newdPdA(j)
            xsec_tab(9,j,k,num_reach) = newdKdA(j)
            xsec_tab(11,j,k,num_reach) = compoundSKK(j)
        end do
        z(k,num_reach)= el_min

        deallocate (el1, a1, peri1, redi1, redi1All)
        deallocate (conv1, tpW1, diffArea, newI1, diffPere)
        deallocate (newdPdA, diffAreaAll, diffPereAll, newdKdA)       ! change Nazmul 20210601
        deallocate (compoundSKK, elev)
        deallocate (i_start, i_end)
        deallocate (totalNodes)
        deallocate (xcs, ycs)
        deallocate (allXcs, allYcs)

        contains
            !**----------------------------------------
            !*      calculate area of triangle
            !**----------------------------------------
            double precision function cal_tri_area(el,x0,x1,y1)
                implicit none
                  doubleprecision, intent(in) :: el,x0,x1,y1
                  cal_tri_area=abs(0.5*(x1-x0)*(el-y1))
                  return
            end function cal_tri_area
            !**----------------------------------------
            !*      calculate area of trapezoid
            !**----------------------------------------
            double precision function cal_trap_area(el,x1,y1,x2,y2)
                implicit none
                doubleprecision, intent(in) :: el,x1,y1,x2,y2
                cal_trap_area=abs(0.5*(x2-x1)*(el-y1+el-y2))
                return
            end function cal_trap_area
            !**--------------------------------------------
            !*     calculate sum of areas of trapezoids
            !**--------------------------------------------
            double precision function cal_multi_area(el,xx,yy,n,i1,i2)
                implicit none
                integer, intent(in) :: n,i1,i2
                doubleprecision, intent(in) :: el
                doubleprecision, intent(in) :: xx(n),yy(n)
                integer :: i
                doubleprecision :: area, x1, x2, y1, y2
                area=0
                do i=i1,i2-1
                    x1=xx(i)
                    y1=yy(i)
                    x2=xx(i+1)
                    y2=yy(i+1)
                    area=area+cal_trap_area(el,x1,y1,x2,y2)
                enddo
                cal_multi_area=area
                return
            endfunction cal_multi_area
            !**--------------------------------------------
            !*     calculate distance of two vertices
            !**--------------------------------------------
            double precision function cal_dist(x1,y1,x2,y2)
                implicit none
                doubleprecision, intent(in) :: x1,y1,x2,y2
                cal_dist=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+1.e-32)
                return
            end function cal_dist
            !**--------------------------------------------
            !*     calculate wetted perimeter
            !**--------------------------------------------
            double precision function cal_perimeter(xx,yy,n,i1,i2)
                implicit none
                integer, intent(in) :: n,i1,i2
                doubleprecision, intent(in) :: xx(n),yy(n)
                integer :: i
                doubleprecision :: p, x1, x2, y1, y2
                p=0.
                do i=i1,i2-1
                    x1=xx(i)
                    y1=yy(i)
                    x2=xx(i+1)
                    y2=yy(i+1)
                    p=p+cal_dist(x1,y1,x2,y2)
                enddo
                cal_perimeter=p
                return
            endfunction cal_perimeter
    end subroutine readXsection
    !*---------------------------------------------------
    !*      interpolation with given arrays
    !
    !*---------------------------------------------------
    subroutine r_interpol(x,y,kk,xrt,yt)
        implicit none
        integer, intent(in) :: kk
        doubleprecision, intent(in) :: xrt, x(kk), y(kk)
        doubleprecision, intent(out) :: yt
        integer :: k

        if (xrt.le.maxval(x) .and. xrt.ge.minval(x)) then
            do k=1,kk-1
                if((x(k)-xrt)*(x(k+1)-xrt).le.0)then

                    yt=(xrt-x(k))/(x(k+1)-x(k))*(y(k+1)-y(k))+y(k)

                    EXIT
                endif
            end do
        else if (xrt.ge.maxval(x)) then
!            print*, xrt, ' the given x data point is larger than the upper limit of the set of x data points'
!            print*, 'the upper limit: ', maxval(x)
            yt=(xrt-x(kk-1))/(x(kk)-x(kk-1))*(y(kk)-y(kk-1))+y(kk-1) ! extrapolation

        else
!            print*, xrt, ' the given x data point is less than lower limit of the range of known x data point, '
!            print*, 'so linear interpolation cannot be performed.'
            yt = -9999.0
!            print*, 'The proper range of x is that: ', 'the upper limit: ', maxval(x),&
!			 ' and lower limit: ', minval(x)
!            print*, 'kk', kk
!            print*, 't', 'i', dmyi, 'j', dmyj
!            print*, 'x', (x(k), k=1, kk)
!            print*, 'y', (y(k), k=1, kk)
        end if
    end subroutine r_interpol
    !*----------------------------------------------------------------
    !*     compute normal/critical depth/area using lookup tables
    !
    !*----------------------------------------------------------------
    subroutine normal_crit_y(i, j, q_sk_multi, So, dsc, y_norm, y_crit, area_n, area_c)
        implicit none
        integer, intent(in) :: i, j
        doubleprecision, intent(in) :: q_sk_multi, So, dsc
        doubleprecision, intent(out) :: y_norm, y_crit, area_n, area_c
        doubleprecision :: area_0, width_0, errorY, pere_0, hydR_0, skk_0
        integer :: trapnm_app, recnm_app, iter

        elevTable = xsec_tab(1,:,i,j)
        areaTable = xsec_tab(2,:,i,j)
        pereTable = xsec_tab(3,:,i,j)
        convTable = xsec_tab(5,:,i,j)
        topwTable = xsec_tab(6,:,i,j)
        call r_interpol(convTable,areaTable,nel,dsc/sqrt(So),area_n)
        call r_interpol(convTable,elevTable,nel,dsc/sqrt(So),y_norm)
        call r_interpol(elevTable,areaTable,nel,oldY(i,j),area_0) ! initial estimate
        call r_interpol(elevTable,topwTable,nel,oldY(i,j),width_0) ! initial estimate

        area_c=area_0
        errorY = 100.
        !pause
        do while (errorY .gt. 0.0001)
            area_c = (dsc * dsc * width_0 / grav) ** (1./3.)
            errorY = abs(area_c - area_0)
            call r_interpol(areaTable,topwTable,nel,area_c, width_0)
            area_0 = area_c
        end do

        call r_interpol(areaTable,elevTable,nel,area_c,y_crit)
        if (y_norm .eq. -9999) then
            print*, 'At j = ',j,', i = ',i, 'interpolation of y_norm in calculating normal area was not possible, Q', &
            dsc,'slope',So
!            stop
        end if
    end subroutine normal_crit_y
    !*--------------------------------------------------
    !*     compute normal depth using lookup tables
    !
    !*--------------------------------------------------
    subroutine normal_y(i, j, q_sk_multi, So, dsc, y_norm, y_crit, area_n, area_c)
        implicit none
        integer, intent(in) :: i, j
        doubleprecision, intent(in) :: q_sk_multi, So, dsc
        doubleprecision, intent(out) :: y_norm, y_crit, area_n, area_c
        doubleprecision :: area_0, width_0, errorY, hydR_0,skk_0!, fro
        integer :: trapnm_app, recnm_app, iter

        elevTable = xsec_tab(1,:,i,j)
        areaTable = xsec_tab(2,:,i,j)
        rediTable = xsec_tab(4,:,i,j)
        topwTable = xsec_tab(6,:,i,j)
        skkkTable = xsec_tab(11,:,i,j)

        call r_interpol(elevTable,areaTable,nel,oldY(i,j),area_0) ! initial estimate

        errorY = 100.
        do while (errorY .gt. 0.00001)
            call r_interpol(areaTable,rediTable,nel,area_0,hydR_0)
            call r_interpol(areaTable,skkkTable,nel,area_0,skk_0)
            area_n = dsc/skk_0/q_sk_multi/ hydR_0 ** (2./3.) / sqrt(So)
            errorY = abs(area_n - area_0) / area_n
            area_0 = area_n
        enddo
        call r_interpol(areaTable,elevTable,nel,area_0,y_norm)
        y_crit = -9999.
        area_c = -9999.
    end subroutine normal_y
    !*--------------------------------------------------
    !*                 Linear Interpolation
    !
    !*--------------------------------------------------
    double precision function LInterpol(x1,y1,x2,y2,x)
        implicit none
        doubleprecision, intent(in) :: x1, y1, x2, y2, x
        !* interpolate y for the given x
        LInterpol= (y2-y1)/(x2-x1)*(x-x1)+y1
    end function LInterpol
    !*--------------------------------------------
    !           Interpolate any value
    !
    !*--------------------------------------------
    double precision function intp_y(nrow, xarr, yarr, x)
        implicit none
        integer, intent(in) :: nrow
        doubleprecision, dimension(nrow), intent(in) :: xarr, yarr
        doubleprecision, intent(in) :: x
        integer :: irow
        doubleprecision :: x1, y1, x2, y2, y

        irow= locate(xarr, x)
        if (irow.eq.0) irow= 1
        if (irow.eq.nrow) irow= nrow-1
        x1= xarr(irow); y1= yarr(irow)
        x2= xarr(irow+1); y2= yarr(irow+1)
        y= LInterpol(x1,y1,x2,y2,x)
        intp_y = y

    end function intp_y
    !*-----------------------------------------------------------------------------
    !               Locate function in f90, p.1045,NR f90
    !
    !   klo=max(min(locate(xa,x),n-1),1) In the Fortran 77 version of splint,
    !   there is in-line code to find the location in the table by bisection. Here
    !   we prefer an explicit call to locate, which performs the bisection. On
    !   some massively multiprocessor (MMP) machines, one might substitute a different,
    !   more parallel algorithm (see next note).
    !*-----------------------------------------------------------------------------
    integer function locate(xx,x)
        implicit none
        doubleprecision, dimension(:), intent(in) :: xx
        doubleprecision, intent(in) :: x
        !* Given an array xx(1:N), and given a value x, returns a value j such that x is between
        !* xx(j) and xx(j + 1). xx must be monotonic, either increasing or decreasing.
        !* j = 0 or j = N is returned to indicate that x is out of range.
        integer :: n,jl,jm,ju
        logical :: ascnd

        n=size(xx)
        ascnd = (xx(n) >= xx(1))  !* True if ascending order of table, false otherwise.
        jl=0    !* Initialize lower
        ju=n+1  !* and upper limits.
        do
            if (ju-jl <= 1) exit    !* Repeat until this condition is satisfied.
            jm=(ju+jl)/2            !* Compute a midpoint,
            if (ascnd .eqv. (x >= xx(jm))) then
                jl=jm               !* and replace either the lower limit
            else
                ju=jm               !* or the upper limit, as appropriate.
            end if
        end do

        if (x == xx(1)) then        !* Then set the output, being careful with the endpoints.
            locate=1
        else if (x == xx(n)) then
            locate=n-1
        else
            locate=jl
        end if
    end function locate
end module diffusive