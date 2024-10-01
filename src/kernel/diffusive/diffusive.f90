!-------------------------------------------------------------------------------
module diffusive
  use precis
  IMPLICIT NONE
  
!-----------------------------------------------------------------------------
! Description:
!   Numerically solve diffusive wave PDEs using Crank-Nicolson and Hermite
!   Interpolation. 
!
! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
!
! Code Description:
!   Language: Fortran 90.
!   This code is written to JULES coding standards v1.
!-----------------------------------------------------------------------------
  
  ! Module constants
  real(prec), parameter :: grav = 9.81
  real(prec), parameter :: TOLERANCE = 1e-8
  
  ! Module variables
  integer :: nlinks
  integer :: mxncomp
  integer :: maxTableLength
  integer :: nel
  integer :: newtonRaphson
  integer :: applyNaturalSection
  integer :: nel_g
  integer :: nmstem_rch
  integer :: mxnbathy
  integer :: nts_da
  integer, dimension(:),   allocatable :: currentROutingDiffusive
  integer, dimension(:),   allocatable :: notSwitchRouting
  integer, dimension(:),   allocatable :: mstem_frj    
  integer, dimension(:),   allocatable :: usgs_da_reach
  integer, dimension(:,:), allocatable :: currentRoutingNormal
  integer, dimension(:,:), allocatable :: routingNotChanged
  integer, dimension(:,:), allocatable :: frnw_g
  integer, dimension(:,:), allocatable :: size_bathy 
  real(prec) :: dtini, dxini, cfl, minDx, maxCelerity,  theta
  real(prec) :: C_llm, D_llm, D_ulm, DD_ulm, DD_llm, q_llm, so_llm
  real(prec) :: frus2, minNotSwitchRouting, minNotSwitchRouting2
  real(prec) :: dt_qtrib
  real(prec) :: z_g, bo_g, traps_g, tw_g, twcc_g, so_g, mann_g, manncc_g
  real(prec) :: dmyi, dmyj
  real(prec) :: dtini_min
  real(prec), dimension(:),       allocatable :: eei, ffi, exi, fxi, qcx, diffusivity2, celerity2
  real(prec), dimension(:),       allocatable :: ini_y, ini_q
  real(prec), dimension(:),       allocatable :: lowerLimitCount, higherLimitCount
  real(prec), dimension(:),       allocatable :: elevTable, areaTable, skkkTable
  real(prec), dimension(:),       allocatable :: pereTable, rediTable
  real(prec), dimension(:),       allocatable :: convTable, topwTable
  real(prec), dimension(:),       allocatable :: nwi1Table, dPdATable
  real(prec), dimension(:),       allocatable :: ncompElevTable, ncompAreaTable
  real(prec), dimension(:),       allocatable :: currentSquareDepth
  real(prec), dimension(:),       allocatable :: tarr_qtrib, varr_qtrib
  real(prec), dimension(:),       allocatable :: tarr_da, varr_da
  real(prec), dimension(:),       allocatable :: area, depth, co, froud, courant
  real(prec), dimension(:,:),     allocatable :: bo, dx
  real(prec), dimension(:,:),     allocatable :: areap, qp, z, sk
  real(prec), dimension(:,:),     allocatable :: celerity, diffusivity, qpx  
  real(prec), dimension(:,:),     allocatable :: pere, oldQ, newQ, oldArea, newArea, oldY, newY  
  real(prec), dimension(:,:),     allocatable :: volRemain  
  real(prec), dimension(:,:),     allocatable :: lateralFlow
  real(prec), dimension(:,:),     allocatable :: dimensionless_Cr, dimensionless_Fo, dimensionless_Fi
  real(prec), dimension(:,:),     allocatable :: dimensionless_Di, dimensionless_Fc, dimensionless_D  
  real(prec), dimension(:,:),     allocatable :: qtrib 
  real(prec), dimension(:,:),     allocatable :: usgs_da
  real(prec), dimension(:,:,:),   allocatable :: x_bathy, z_bathy, mann_bathy 
  real(prec), dimension(:,:,:,:), allocatable :: xsec_tab
  
contains

  subroutine diffnw(timestep_ar_g, nts_ql_g, nts_ub_g, nts_db_g, nts_ev_g, nts_qtrib_g, nts_da_g,      &
                    mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g, mann_ar_g,      &
                    manncc_ar_g, so_ar_g, dx_ar_g,                                                      &
                    iniq, frnw_col, frnw_ar_g, qlat_g, ubcd_g, dbcd_g, qtrib_g,                         &
                    paradim, para_ar_g, mxnbathy_g, x_bathy_g, z_bathy_g, mann_bathy_g, size_bathy_g,   &
                    usgs_da_g, usgs_da_reach_g, rdx_ar_g, cwnrow_g, cwncol_g, crosswalk_g, z_thalweg_g, &
                    q_ev_g, elv_ev_g, depth_ev_g)                                     
                    

    IMPLICIT NONE
          
  !-----------------------------------------------------------------------------
  ! Description:
  !   Compute diffusive routing on National Water Model channel network domain.
  !
  ! Method:
  !   A Crank-Nicolson solution of the diffusive wave equations is solved with
  !   adaptive timestepping to maintain numerical stability. Major operations are
  !   as follows:
  !
  !     1. Initialize domain flow and depth. Depth is computed from initial flow
  !     2. For each timestep....
  !       2.1. Compute domain flow, upstream-to-downstream
  !       2.2. Compute domain depth, downstream-to-upstream
  !       2.3. Determine time step duration needed for stability
  !       2.4. repeat until until final time is reached
  !     3. Record results in output arrays at user-specified time intervals
  !
  ! Current Code Owner: NOAA-OWP, Inland Hydraulics Team
  !
  ! Development Team:
  !  - Dong Ha Kim
  !  - Nazmul Azim Beg
  !  - Ehab Meselhe
  !  - Adam N. Wlostowski
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
    integer, intent(in) :: nts_ev_g
    integer, intent(in) :: nts_qtrib_g
    integer, intent(in) :: nts_da_g
    integer, intent(in) :: frnw_col
    integer, intent(in) :: mxnbathy_g
    integer, intent(in) :: paradim
    integer, intent(in) :: cwnrow_g
    integer, intent(in) :: cwncol_g
    integer, dimension(nrch_g), intent(in) :: usgs_da_reach_g
    integer, dimension(nrch_g, frnw_col),  intent(in) :: frnw_ar_g
    integer, dimension(mxncomp_g, nrch_g), intent(in) :: size_bathy_g
    real(prec), dimension(paradim ), intent(in) :: para_ar_g
    real(prec), dimension(:)       , intent(in) :: timestep_ar_g(10)
    real(prec), dimension(nts_db_g), intent(in) :: dbcd_g    
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: z_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: bo_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: traps_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: tw_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: twcc_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: mann_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: manncc_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: dx_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: rdx_ar_g        
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: iniq
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: so_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g),            intent(in) :: z_thalweg_g
    real(prec), dimension(nts_ub_g,    nrch_g),            intent(in) :: ubcd_g
    real(prec), dimension(nts_qtrib_g, nrch_g),            intent(in) :: qtrib_g 
    real(prec), dimension(nts_da_g,    nrch_g),            intent(in) :: usgs_da_g 
    real(prec), dimension(cwnrow_g, cwncol_g),             intent(in) :: crosswalk_g    
    real(prec), dimension(nts_ql_g,  mxncomp_g, nrch_g),   intent(in ) :: qlat_g 
    real(prec), dimension(mxnbathy_g, mxncomp_g, nrch_g),  intent(in ) :: x_bathy_g
    real(prec), dimension(mxnbathy_g, mxncomp_g, nrch_g),  intent(in ) :: z_bathy_g
    real(prec), dimension(mxnbathy_g, mxncomp_g, nrch_g),  intent(in ) :: mann_bathy_g
    real(prec), dimension(nts_ev_g, mxncomp_g, nrch_g),   intent(out) :: q_ev_g
    real(prec), dimension(nts_ev_g, mxncomp_g, nrch_g),   intent(out) :: elv_ev_g
    real(prec), dimension(nts_ev_g, mxncomp_g, nrch_g),   intent(out) :: depth_ev_g

  ! Local variables    
    integer :: ncomp
    integer :: i
    integer :: j
    integer :: k
    integer :: n
    integer :: timestep
    integer :: kkk
    integer :: ts_ev
    integer :: jm
    integer :: usrchj
    integer :: linknb
    integer :: xcolID
    integer :: ycolID
    integer :: iel
    integer :: dsbc_option
    integer :: ts, cwrow 
    integer :: ri, rj, oi, oj
    integer :: nlnk, lnk
    integer :: nusrch
    integer :: n_decimal_places 
    integer, dimension(:), allocatable   :: dmy_frj
    integer, dimension(:,:), allocatable :: flag_lfrac
    real(prec) :: x
    real(prec) :: saveInterval, width
    !real(prec) :: maxCourant
    real(prec) :: dtini_given, dtini_divisor
    real(prec) :: timesDepth
    real(prec) :: t
    real(prec) :: tfin
    real(prec) :: t0
    real(prec) :: q_sk_multi
    real(prec) :: maxCelDx
    real(prec) :: slope
    real(prec) :: y_norm
    real(prec) :: temp
    real(prec) :: dt_ql
    real(prec) :: dt_ub
    real(prec) :: dt_db
    real(prec) :: dt_da
    real(prec) :: wdepth
    real(prec) :: q_usrch
    real(prec) :: tf0
    real(prec) :: convey
    real(prec) :: equiv_one
    real(prec) :: slopeQ, intcQ, slopeE, intcE, slopeD, intcD
    real(prec) :: lfrac, dst_lnk, dst_top, dst_btm
    real(prec) :: mindepth_nstab
    real(prec), dimension(:), allocatable :: tarr_ql
    real(prec), dimension(:), allocatable :: varr_ql
    real(prec), dimension(:), allocatable :: tarr_ub
    real(prec), dimension(:), allocatable :: varr_ub
    real(prec), dimension(:), allocatable :: tarr_db
    real(prec), dimension(:), allocatable :: varr_db
    real(prec), dimension(:), allocatable :: dbcd ! temporary
    real(prec), dimension(:,:), allocatable :: leftBank
    real(prec), dimension(:,:), allocatable :: rightBank
    real(prec), dimension(:,:), allocatable :: skLeft
    real(prec), dimension(:,:), allocatable :: skMain
    real(prec), dimension(:,:), allocatable :: skRight
    real(prec), dimension(:,:), allocatable :: used_lfrac    
    real(prec), dimension(:,:,:), allocatable :: temp_q_ev_g
    real(prec), dimension(:,:,:), allocatable :: temp_elv_ev_g    

  !-----------------------------------------------------------------------------
  ! Time domain parameters
    dtini         = timestep_ar_g(1)    ! initial timestep duration [sec]
    t0            = timestep_ar_g(2)    ! simulation start time [hr]
    tfin          = timestep_ar_g(3)    ! simulation end time [hr]
    saveInterval  = timestep_ar_g(4)    ! output recording interval [sec]
    dt_ql         = timestep_ar_g(5)    ! lateral inflow data time step [sec]
    dt_ub         = timestep_ar_g(6)    ! upstream boundary time step [sec]
    dt_db         = timestep_ar_g(7)    ! downstream boundary time step [sec]
    dt_qtrib      = timestep_ar_g(8)    ! tributary data time step [sec]
    dt_da         = timestep_ar_g(9)    ! data assimilation data time step [sec]
    dtini_given   = dtini               ! preserve user-input timestep duration
    dtini_divisor = timestep_ar_g(10)   ! numeric value that divide dtini in the next line
    dtini_min     = dtini/dtini_divisor ! minimum increment in internal time step 

  !-----------------------------------------------------------------------------
  ! miscellaneous parameters
    timesDepth = 4.0 ! water depth multiplier used in readXsection
    nel        = 501 ! number of sub intervals in look-up tables
    nts_da     = nts_da_g ! make DA time steps global

  !-----------------------------------------------------------------------------
  ! Network mapping (frnw_g) array size parameters
    mxncomp = mxncomp_g ! maximum number of nodes in a single reach
    nlinks  = nrch_g    ! number of reaches in the network

  !-----------------------------------------------------------------------------
  ! Some essential parameters for Diffusive Wave
    cfl         = para_ar_g(1)  ! maximum Courant number (default: 0.95)
    C_llm       = para_ar_g(2)  ! lower limit of celerity (default: 0.5)
    D_llm       = para_ar_g(3)  ! lower limit of diffusivity (default: 50)
    D_ulm       = para_ar_g(4)  ! upper limit of diffusivity (default: 1000)
    DD_llm      = para_ar_g(5)  ! lower limit of dimensionless diffusivity (default -15)
    DD_ulm      = para_ar_g(6)  ! upper limit of dimensionless diffusivity (default: -10.0)
    q_llm       = para_ar_g(8)  ! lower limit of discharge (default: 0.02831 cms)
    so_llm      = para_ar_g(9)  ! lower limit of channel bed slope (default: 0.00001)
    theta       = para_ar_g(10) ! weight for computing 2nd derivative: 
                                ! 0: explicit, 1: implicit (default: 1.0)
    dsbc_option = para_ar_g(11) ! downstream water depth boundary condition option 1:given water depth data, 2:normal depth   
    mindepth_nstab = 0.1        ! minimum water depth for numerical stability when estimated/observed downstream boundary depth data is used.
  !-----------------------------------------------------------------------------
  ! Some parameters for using natural cross section bathymetry data
    mxnbathy = mxnbathy_g  ! maximum size of bathymetry data points

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
    allocate(tarr_ql(nts_ql_g+1), varr_ql(nts_ql_g+1))
    allocate(tarr_ub(nts_ub_g), varr_ub(nts_ub_g))
    allocate(tarr_qtrib(nts_qtrib_g), varr_qtrib(nts_qtrib_g))
    allocate(tarr_db(nts_db_g), varr_db(nts_db_g))
    allocate(tarr_da(nts_da))
    allocate(dmy_frj(nlinks))
    allocate(frnw_g(nlinks,frnw_col))
    allocate(x_bathy(mxnbathy, mxncomp, nlinks), z_bathy(mxnbathy, mxncomp, nlinks))
    allocate(mann_bathy(mxnbathy, mxncomp, nlinks))
    allocate(size_bathy(mxncomp, nlinks))
    allocate(usgs_da_reach(nlinks))
    allocate(usgs_da(nts_da, nlinks))
    allocate(dbcd(nts_db_g))
    allocate(temp_q_ev_g(nts_ev_g, mxncomp_g, nrch_g), temp_elv_ev_g(nts_ev_g, mxncomp_g, nrch_g)) 
    
  !--------------------------------------------------------------------------------------------
    frnw_g        = frnw_ar_g ! network mapping matrix
    z             = z_ar_g    ! node elevation array
    usgs_da_reach = usgs_da_reach_g ! contains indices of reaches where usgs data are available
    usgs_da       = usgs_da_g       ! contains usgs data at a related reach 
        
  !-----------------------------------------------------------------------------
  ! variable initializations  
    routingNotChanged   = 0
    applyNaturalSection = 1
    x                   = 0.0
    newQ                = -999
    newY                = -999
    t                   = t0*60.0     ! [min]
    q_sk_multi          = 1.0
    oldQ                = iniq    
    newQ                = oldQ
    qp                  = oldQ
    dimensionless_Cr    = -999
    dimensionless_Fo    = -999
    dimensionless_Fi    = -999
    dimensionless_Di    = -999
    dimensionless_Fc    = -999
    dimensionless_D     = -999
    dbcd                = dbcd_g   
    q_ev_g              = 0.0
    elv_ev_g            = 0.0
    depth_ev_g          = 0.0
    temp_q_ev_g         = 0.0
    temp_elv_ev_g       = 0.0    

  !-----------------------------------------------------------------------------
  ! Identify mainstem reaches and list their ids in an array

    nmstem_rch = 0
    do j = 1, nlinks
      !if (frnw_g(j,3) >= 1) then  ! mainstem reach identification
      nusrch = frnw_g(j,3)         ! the number of upstream reaches
      if (frnw_g(j, 3+nusrch+1)==555) then ! 555 indicate diffusive reach while -555 already routed reach
        nmstem_rch          = nmstem_rch + 1
        dmy_frj(nmstem_rch) = j
      end if
    end do

    ! allocate and populate array for upstream reach ids
    allocate (mstem_frj(nmstem_rch))
    do jm = 1, nmstem_rch
      mstem_frj(jm) = dmy_frj(jm)
    end do
    deallocate(dmy_frj)
    
  !-----------------------------------------------------------------------------
  ! create dx array from dx_ar_g and determine minimum dx.
    
    dx    = 0.
    minDx = 1e10

    do jm = 1, nmstem_rch !* mainstem reach only
      j = mstem_frj(jm)
      ncomp = frnw_g(j, 1)
      do i = 1, ncomp-1
        dx(i, j) = dx_ar_g(i, j)
      end do
      minDx = min(minDx, minval(dx(1:ncomp-1, j)))
    end do
    
  !-----------------------------------------------------------------------------
  ! Build natural / synthetic cross sections and related hydraulic lookup table
  if (mxnbathy == 0) then
    applyNaturalSection = 0
    print*, 'Applying synthetic channel cross section...'
  else
    applyNaturalSection = 1
    print*, 'Applying natural channel cross section...'
  end if 
    
  if (applyNaturalSection == 1) then
  
    ! use bathymetry data 
    x_bathy    = x_bathy_g
    z_bathy    = z_bathy_g
    mann_bathy = mann_bathy_g
    size_bathy = size_bathy_g
    
    do jm = 1, nmstem_rch !* mainstem reach only
      j = mstem_frj(jm)
      do i = 1, frnw_g(j, 1)
        call readXsection_natural_mann_vertices(i, j, timesDepth)
      end do
    end do
  
  else
    ! use RouteLink.nc data
    do jm = 1, nmstem_rch !* mainstem reach only
      j     = mstem_frj(jm)
      ncomp = frnw_g(j,1)
      do i = 1, ncomp
        leftBank(i,j)  = (twcc_ar_g(i,j) - tw_ar_g(i,j)) / 2.0
        rightBank(i,j) = (twcc_ar_g(i,j) - tw_ar_g(i,j)) / 2.0 + tw_ar_g(i,j)
      end do
    end do
    
    do jm = 1, nmstem_rch !* mainstem reach only
        j     = mstem_frj(jm)
        ncomp = frnw_g(j,1)
        
        do i=1,ncomp
          skLeft(i,j) = 1.0 / manncc_ar_g(i,j)
          skRight(i,j)= 1.0 / manncc_ar_g(i,j)
          skMain(i,j) = 1.0 / mann_ar_g(i,j)

          call readXsection(i, (1.0/skLeft(i,j)), (1.0/skMain(i,j)), &
                            (1.0/skRight(i,j)), leftBank(i,j),       &
                            rightBank(i,j), timesDepth, j, z_ar_g,   &
                            bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g, mxncomp_g, nrch_g)
        end do
    end do  
  end if  
      
  !-----------------------------------------------------------------------------
  ! Add uniform flow column to the hydraulic lookup table in order to avoid the 
  ! use of the trial-and-error iteration for solving normal depth
  do jm = 1, nmstem_rch !* mainstem reach only
    j = mstem_frj(jm)
    do i = 1, frnw_g(j,1)
      do iel = 1, nel
        convey = xsec_tab(5, iel, i, j)
        if (i < frnw_g(j, 1)) then
          slope = (z(i, j) - z(i+1, j)) / dx(i, j)
        else
          slope = (z(i-1, j) - z(i, j)) / dx(i-1, j)
        endif

        if (slope .le. so_llm) slope = so_llm

        xsec_tab(10, iel, i, j) = convey * slope**0.50
      end do
    end do
  end do

  !-----------------------------------------------------------------------------
  ! Build time arrays for lateral flow, upstream boundary, downstream boundary,
  ! and tributary flow

    ! time step series for lateral flow. 
    ! ** ql from t-route starts from the first time step. For example, when t0 and tfin = 0 and 2 hrs with dt=300 sec,
    ! ** ql values from t-route starts at 5, 10, 15, ..., 120 min.
    do n = 1, nts_ql_g
      tarr_ql(n+1) =    t0 * 60.0 + dt_ql * &
                      real(n, KIND(dt_ql)) / 60.0 ! [min]
    end do
    tarr_ql(1) = t0 * 60   ! initial time step [min]

    ! time step series for upstream boundary data
    do n = 1, nts_ub_g
      tarr_ub(n) =    t0 * 60.0 + dt_ub * &
                      real(n-1,KIND(dt_ub)) / 60.0 ! [min]
    end do

    ! time step series for tributary flow data
    ! ** qtrib from t-route starts from the initial time step . For example, when t0 and tfin = 0 and 2 hrs with dt=300 sec,
    ! ** qtrib values from t-route starts at 0, 5, 10, 15, ..., 120 min.
    do n=1, nts_qtrib_g
      tarr_qtrib(n) = t0 * 60.0 + dt_qtrib * &
                      real(n-1,KIND(dt_qtrib)) / 60.0 ! [min]
    end do
    
    ! use tailwater downstream boundary observations
    ! needed for coastal coupling
    ! **** COMING SOON ****
    ! time step series for downstream boundary data
    do n=1, nts_db_g
        tarr_db(n) = t0 * 60.0 + dt_db * &
                      real(n-1,KIND(dt_db)) / 60.0 ! [min]                 
    end do
    
    ! time step series for data assimilation for discharge at bottom node of a related reach
    do n=1, nts_da
        tarr_da(n) = t0 * 60.0 + dt_da * &
                      real(n-1,KIND(dt_da)) / 60.0 ! [min]
    end do

  !-----------------------------------------------------------------------------
  ! Initialize water surface elevation, channel area, and volume
    do j = 1, nlinks 
      ncomp = frnw_g(j, 1)   ! number of nodes in reach j  
      do i=1, ncomp
        oldQ(i,j) = max(oldQ(i,j), q_llm)
        qp(i,j) = oldQ(i,j)
      enddo
    enddo

    do jm = nmstem_rch, 1, -1
      j     = mstem_frj(jm)  ! reach index
      ncomp = frnw_g(j, 1)   ! number of nodes in reach j       
      if (frnw_g(j, 2) < 0) then 
      
        ! Initial depth at bottom node of tail water reach        
        if (dsbc_option == 1) then
        ! use tailwater downstream boundary observations
        ! needed for coastal coupling
        ! **** COMING SOON **** 
          do n = 1, nts_db_g
            varr_db(n) = dbcd(n) + z(ncomp, j) !* when dbcd is water depth [m], channel bottom elev is added.
          end do
          t              = t0 * 60.0
          oldY(ncomp, j) = intp_y(nts_db_g, tarr_db, varr_db, t)
          newY(ncomp, j) = oldY(ncomp, j)  
          if ((newY(ncomp, j) - z(ncomp, j)).lt.mindepth_nstab) then
            newY(ncomp, j) = mindepth_nstab + z(ncomp, j)
          end if          
        else if (dsbc_option == 2) then
        ! normal depth as TW boundary condition
          xcolID         = 10
          ycolID         = 1
          oldY(ncomp, j) = intp_xsec_tab(ncomp, j, nel, xcolID, ycolID, oldQ(ncomp,j)) ! normal elevation not depth
          newY(ncomp, j) = oldY(ncomp, j)  
        endif
      else
        ! Initial depth at bottom node of interior reach is equal to the depth at top node of the downstream reach
        linknb         = frnw_g(j, 2)
        newY(ncomp, j) = newY(1, linknb)        
      end if              
     
      ! compute newY(i, j) for i=1, ncomp-1 with the given newY(ncomp, j)
      ! ** At initial time, oldY(i,j) values at i < ncomp, used in subroutine rtsafe, are not defined.
      ! ** So, let's assume the depth values are all nodes equal to depth at the bottom node.
      oldY(ncomp, j) = newY(ncomp, j)
      wdepth         = newY(ncomp, j) - z(ncomp, j)
      do i = 1, ncomp -1
        oldY(i,j) = wdepth + z(i, j)      
      end do
      
      call mesh_diffusive_backward(dtini_given, t0, t, tfin, saveInterval, j)
      
      do i = 1,ncomp      
        ! copy computed initial depths to initial depth array for first timestep
        oldY(i, j) = newY(i, j)
        
        ! Check that node elevation is not lower than bottom node.
        ! If node elevation is lower than bottom node elevation, correct.
        !if (oldY(i, j) .lt. oldY(ncomp, nlinks)) oldY(i, j) = oldY(ncomp, nlinks)
      end do      
    end do

  !-----------------------------------------------------------------------------
  ! Write tributary results to output arrays
  ! TODO: consider if this is necessary - output arrays are immediately trimmed
  ! to exclude tributary results (from MC) and pass-out only diffusive-calculated
  ! flow and depth on mainstem segments.
          
    ts_ev = 1
    do while (t <= tfin * 60.0)
      if ( (mod( (t - t0 * 60.) * 60., saveInterval) <= TOLERANCE) &
            .or. (t == tfin * 60.) ) then
        do j = 1, nlinks
          if (all(mstem_frj /= j)) then ! NOT a mainstem reach
            do n = 1, nts_qtrib_g
              varr_qtrib(n) = qtrib_g(n, j)
            end do
            q_ev_g(ts_ev, frnw_g(j, 1), j) = intp_y(nts_qtrib_g, tarr_qtrib, &
                                                      varr_qtrib, t)
            q_ev_g(ts_ev,            1, j) = q_ev_g(ts_ev, frnw_g(j, 1), j)            
          end if
        end do
        ts_ev = ts_ev + 1
      end if
      t = t + dtini / 60. !* [min]
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
    timestep                = 0
    ts_ev                   = 1
    t                       = t0 * 60.0

  !-----------------------------------------------------------------------------
  ! Ordered network routing computations
    do while ( t < tfin * 60.)
      timestep = timestep + 1 ! advance timestep
      !+-------------------------------------------------------------------------
      !+                             PREDICTOR
      !+
      !+    Applies only to mainstem
      !+-------------------------------------------------------------------------
      do jm = 1, nmstem_rch   ! loop over mainstem reaches [upstream-to-downstream]
        j     = mstem_frj(jm) ! reach index
        ncomp = frnw_g(j,1)   ! number of nodes in reach j
        
        ! Calculate the duration of this timestep (dtini)
        ! Timestep duration is selected to maintain numerical stability
        if (j == mstem_frj(1)) then 
          call calculateDT(t0, t, saveInterval, cfl, tfin, maxCelDx, dtini_given)
        end if        

        ! estimate lateral flow at current time t
        do i = 1, ncomp - 1
          do n = 1, nts_ql_g
            varr_ql(n+1) = qlat_g(n, i, j)
          end do
          varr_ql(1)        = qlat_g(1, i, j)
          lateralFlow(i, j) = intp_y(nts_ql_g+1, tarr_ql, varr_ql, t)
        end do

        !+++----------------------------------------------------------------
        !+ Hand over water from upstream to downstream properly according
        !+ to network connections, i.e., serial or branching.
        !+ Refer to p.52, RM1_MESH
        !+++----------------------------------------------------------------
        if (frnw_g(j, 3) > 0) then        ! number of upstream reaches, so reach j is not a headwater reach's index
          newQ(1, j) = 0.0
          do k = 1, frnw_g(j, 3)          ! loop over upstream connected reaches
            usrchj = frnw_g(j, 3 + k) 
            if (any(mstem_frj == usrchj)) then

              ! inflow from upstream mainstem reach's bottom node
              q_usrch = newQ(frnw_g(usrchj, 1), usrchj)
            else

              ! inflow from upstream tributary reach
              do n = 1, nts_qtrib_g
                varr_qtrib(n) = qtrib_g(n, usrchj)
              end do
              tf0 = t !+  dtini / 60
              q_usrch = intp_y(nts_qtrib_g, tarr_qtrib, varr_qtrib, tf0)
            end if

            ! add upstream flows to reach head
            newQ(1,j)= newQ(1,j) + q_usrch           
          end do        
        else
        
          ! no stream reaches at the upstream of the reach (frnw_g(j,3)==0) such as a headwater reach among tributaries
          newQ(1,j)= 0.0          
        end if

        ! Add lateral inflows to the reach head
        newQ(1, j) = newQ(1, j) + lateralFlow(1, j) * dx(1, j)

        call mesh_diffusive_forward(dtini_given, t0, t, tfin, saveInterval, j)

      end do  ! end of j loop for predictor
      
      !+-------------------------------------------------------------------------
      !+                             CORRECTOR
      !+
      !+    Applies only to mainstem
      !+-------------------------------------------------------------------------
      do jm = nmstem_rch, 1, -1 ! loop over mainstem reaches [downstream-to-upstream]
        j     = mstem_frj(jm)
        ncomp = frnw_g(j,1)
          
        !+++------------------------------------------------------------+
        !+ Downstream boundary condition for water elevation either at
        !+ a junction or TW.
        !+ Refer to p.53-1,RM1_MESH
        !+++------------------------------------------------------------+
         if (frnw_g(j, 2) >= 0.0) then          

            ! reach of index j has a reach in the downstream 
            ! set bottom node WSEL equal WSEL in top node of downstream reach
            linknb         = frnw_g(j,2)
            newY(ncomp, j) = newY(1, linknb) 
        else
        
          ! reach of index j has NO downstream connection, so it is a tailwater reach
          if (dsbc_option == 1) then  
          ! use downstream boundary data source to set bottom node WSEL value       
            newY(ncomp, j) = intp_y(nts_db_g, tarr_db, varr_db, t+dtini/60.)
            ! to prevent too small water depth at boundary
            if ((newY(ncomp, j) - z(ncomp, j)).lt.mindepth_nstab) then
              newY(ncomp, j) = mindepth_nstab + z(ncomp, j)
            end if 
            xcolID  = 1
            ycolID  = 2
            newArea(ncomp, j) = intp_xsec_tab(ncomp, j, nel, xcolID, ycolID, newY(ncomp,j)) ! area of normal elevation
          else if (dsbc_option == 2) then  
          ! Assume normal depth at tailwater downstream boundary
            xcolID  = 10
            ycolID  = 1
            newY(ncomp,j) = intp_xsec_tab(ncomp, j, nel, xcolID, ycolID, abs(newQ(ncomp,j))) ! normal elevation not depth
            xcolID  = 1
            ycolID  = 2
            newArea(ncomp,j) = intp_xsec_tab(ncomp, j, nel, xcolID, ycolID, newY(ncomp,j)) ! area of normal elevation
          endif
        end if

        ! Calculate WSEL at interior reach nodes
        call mesh_diffusive_backward(dtini_given, t0, t, tfin, saveInterval, j)
        
        ! Identify the maximum calculated celerity/dx ratio at this timestep
        ! maxCelDx is used to determine the duration of the next timestep
        if (jm == 1) then
          maxCelDx    = 0.
          do i = 1, nmstem_rch
            do kkk = 1, frnw_g(mstem_frj(i), 1)-1
              maxCelDx    = max(maxCelDx, celerity(kkk, mstem_frj(i)) / dx(kkk, mstem_frj(i)))
            end do
          end do
        endif
      end do  ! end of corrector j loop
    
      !+-------------------------------------------------------------------------
      !+                             BOOK KEEPING
      !+
      !+-------------------------------------------------------------------------
      
      ! Calculate Froude and maximum Courant number
      do jm = 1, nmstem_rch
        j     = mstem_frj(jm)
        ncomp = frnw_g(j,1)
        do i = 1, ncomp
!          froud(i) = abs(newQ(i, j)) / sqrt(grav * newArea(i, j) ** 3.0 / bo(i, j))
          if (i < ncomp) then
            courant(i) = (newQ(i, j) + newQ(i+1, j)) / (newArea(i, j) + newArea(i+1, j)) * dtini / dx(i, j)
          endif
        end do
        !if (maxCourant < maxval(courant(1:ncomp - 1))) then
        !  maxCourant = maxval(courant(1:ncomp-1))
        !end if
      end do

      ! Advance model time [minute]
      t = t + dtini/60.
      
      ! Calculate dimensionless numbers for each reach
      !do jm = 1, nmstem_rch  !* mainstem reach only
      !  j = mstem_frj(jm)
      !  call calc_dimensionless_numbers(j)
      !end do

      ! diffusive wave simulation time print
      if (mod(t,5.)==0.) then
        print*, "diffusive simulation time in minute=", t
      endif

      ! write results to output arrays
      if ( (mod((t - t0 * 60.) * 60., saveInterval) <= TOLERANCE) .or. (t == tfin * 60.)) then
         do jm = 1, nmstem_rch
          j     = mstem_frj(jm)
          ncomp = frnw_g(j, 1)
          do i = 1, ncomp
            q_ev_g  (ts_ev + 1, i, j)   = newQ(i, j)
            elv_ev_g(ts_ev + 1, i, j)   = newY(i, j)
            depth_ev_g(ts_ev + 1, i, j) = elv_ev_g(ts_ev + 1, i, j) - z(i, j)
          end do
          
          !* water elevation for tributaries flowing into the mainstem in the middle or at the upper end
          do k = 1, frnw_g(j, 3)
            usrchj = frnw_g(j, 3 + k)
            if (all(mstem_frj /= usrchj)) then
              wdepth                                         = newY(1, j) - z(1, j)
              elv_ev_g(ts_ev+1, frnw_g(usrchj, 1), usrchj)   = newY(1, j)
              depth_ev_g(ts_ev+1, frnw_g(usrchj, 1), usrchj) = wdepth
            endif
          end do
        end do
        
        ! Advance recording timestep
        ts_ev = ts_ev+1
      end if

      ! write initial conditions to output arrays
      if ( ( t == t0 + dtini / 60. ) ) then
        do jm = 1, nmstem_rch  !* mainstem reach only
          j     = mstem_frj(jm)
          ncomp = frnw_g(j, 1)
          do i = 1, ncomp
            q_ev_g  (1, i, j)   = oldQ(i, j)
            elv_ev_g(1, i, j)   = oldY(i, j)
            depth_ev_g(1, i, j) = elv_ev_g(1, i, j) - z(i, j)
          end do
              
          !* water elevation for tributaries flowing into the mainstem in the middle or at the upper end
          do k = 1, frnw_g(j, 3) !* then number of upstream reaches
            usrchj = frnw_g(j, 3 + k) !* js corresponding to upstream reaches
            if (all(mstem_frj /= usrchj)) then                  
              wdepth                                   = oldY(1, j) - z(1, j)
              elv_ev_g(1, frnw_g(usrchj,1), usrchj)    = oldY(1,j)
              depth_ev_g(1, frnw_g(usrchj, 1), usrchj) = wdepth
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
    
    !---------------------------------------------------------------------------
    ! map routing result from refactored hydrofabric to unrefactored hydrofabric
    if (1==2) then
      if (cwnrow_g > 0) then    
        allocate(used_lfrac(mxncomp, nlinks))
        allocate(flag_lfrac(mxncomp, nlinks))   
        equiv_one = 0.99    
        temp_q_ev_g   = q_ev_g
        temp_elv_ev_g = elv_ev_g
        
        q_ev_g = 0.0
        elv_ev_g = 0.0

        do ts=1, nts_ev_g
          used_lfrac = 0.0
          flag_lfrac = 0
          do cwrow = 1, cwnrow_g
            ri     = int(crosswalk_g(cwrow, 1))
            rj     = int(crosswalk_g(cwrow, 2))
            nlnk   = int(crosswalk_g(cwrow, 3))
            ! flow slope and y-interceptor
            slopeQ = (temp_q_ev_g(ts, ri+1, rj) - temp_q_ev_g(ts, ri, rj)) / rdx_ar_g(ri,rj)
            intcQ  = temp_q_ev_g(ts, ri, rj)
            ! elevation slope and y-interceptor
            slopeE = (temp_elv_ev_g(ts, ri+1, rj) - temp_elv_ev_g(ts, ri, rj)) / rdx_ar_g(ri,rj)
            intcE  = temp_elv_ev_g(ts, ri, rj)
            ! depth slope and y-interceptor
            slopeD = ((temp_elv_ev_g(ts, ri+1, rj)-z(ri+1, rj)) - (temp_elv_ev_g(ts, ri, rj)-z(ri, rj))) / rdx_ar_g(ri,rj)
            intcD  = temp_elv_ev_g(ts, ri, rj) - z(ri, rj)

            dst_lnk = 0.0
            do lnk = 1, nlnk
              oi = int(crosswalk_g(cwrow, 4 + 3*(lnk-1)))
              oj = int(crosswalk_g(cwrow, 5 + 3*(lnk-1)))
              lfrac = crosswalk_g(cwrow,  6 + 3*(lnk-1))
              
              dst_top = dst_lnk
              dst_lnk = dst_lnk + dx_ar_g(oi, oj)*lfrac
              dst_btm = dst_lnk

              used_lfrac(oi, oj) = used_lfrac(oi, oj) + lfrac          
              if  (used_lfrac(oi, oj) < equiv_one) then
                flag_lfrac(oi, oj) = flag_lfrac(oi, oj) + 1
              end if

              if ((used_lfrac(oi, oj) >= equiv_one).and.(flag_lfrac(oi, oj) == 0)) then
                 q_ev_g(ts, oi, oj)     = intcQ + slopeQ*dst_top
                 q_ev_g(ts, oi+1, oj)   = intcQ + slopeQ*dst_btm
                 ! linear interpolation by elevation
                 !elv_ev_g(ts, oi, oj)   = intcE + slopeE*dst_top                 
                 !elv_ev_g(ts, oi+1, oj) = intcE + slopeE*dst_btm    
                 ! linear interpolation by depth
                 elv_ev_g(ts, oi, oj)   = intcD + slopeD*dst_top + z_thalweg_g(oi, oj)
                 elv_ev_g(ts, oi+1, oj) = intcD + slopeD*dst_btm + z_thalweg_g(oi+1, oj)
              else if ((used_lfrac(oi, oj) < equiv_one).and.(flag_lfrac(oi, oj) == 1)) then
                 q_ev_g(ts, oi, oj)   = intcQ + slopeQ*dst_top
                 ! linear interpolation by elevation
                 !elv_ev_g(ts, oi, oj) = intcE + slopeE*dst_top
                 ! linear interpolation by depth
                 elv_ev_g(ts, oi, oj) = intcD + slopeD*dst_top + z_thalweg_g(oi, oj)
              else if ((used_lfrac(oi, oj) >= equiv_one).and.(flag_lfrac(oi, oj) >= 1)) then
                 q_ev_g(ts, oi+1, oj)   = intcQ + slopeQ*dst_btm
                 ! linear interpolation by elevation
                 !elv_ev_g(ts, oi+1, oj) = intcE + slopeE*dst_btm   
                 ! linear interpolation by depth
                 elv_ev_g(ts, oi+1, oj) = intcD + slopeD*dst_btm + z_thalweg_g(oi+1, oj)
                 flag_lfrac(oi, oj)     = 0
              endif
            end do
          end do
        end do
        deallocate(used_lfrac)
        deallocate(flag_lfrac)   
      endif
    endif    
   
    deallocate(frnw_g)
    deallocate(area, bo, pere, areap, qp, z,  depth, sk, co, dx) 
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
    deallocate(tarr_ql, varr_ql, tarr_ub, varr_ub, tarr_qtrib, varr_qtrib, tarr_da)
    deallocate(mstem_frj)
    deallocate(x_bathy, z_bathy, mann_bathy, size_bathy)
    deallocate(usgs_da_reach, usgs_da)

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
    real(prec), intent(in) :: initialTime       ! [sec]
    real(prec), intent(in) :: time              ! [hrs]
    real(prec), intent(in) :: saveInterval      ! [sec]
    real(prec), intent(in) :: tfin              ! [hrs]
    real(prec), intent(in) :: given_dt          ! [sec]
    real(prec), intent(in) :: maxAllowCourantNo ! = cfl
    real(prec), intent(in) :: max_C_dx          ! = maxCelDx
      
  ! Local variables
    integer :: a
    integer :: b
    
  !-----------------------------------------------------------------------------    
  ! Calculate maximum timestep duration for numerical stability
  
    dtini = maxAllowCourantNo / max_C_dx
    a     = floor( (time - initialTime * 60.) / &
                 ( saveInterval / 60.))           
    b     = floor(((time - initialTime * 60.) + dtini / 60.) / &
                 ( saveInterval / 60.))           
    if (b > a) then
      dtini = (a + 1) * (saveInterval) - (time - initialTime * 60.) * 60.
    end if

    ! if dtini extends beyond final time, then truncate it
    if (time + dtini / 60. > tfin * 60.) dtini = (tfin * 60. - time) * 60.

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
    real(prec) :: wl_us
    real(prec) :: depth_us
    real(prec) :: q_us
    real(prec) :: v_us
    real(prec) :: pere_us
    real(prec) :: r_us
    real(prec) :: sk_us
    real(prec) :: ch_us
    real(prec) :: wl_ds
    real(prec) :: depth_ds
    real(prec) :: q_ds
    real(prec) :: v_ds
    real(prec) :: pere_ds
    real(prec) :: r_ds
    real(prec) :: sk_ds
    real(prec) :: ch_ds
    real(prec) :: ch_star_avg
    real(prec) :: channel_length
    real(prec) :: avg_celerity
    real(prec) :: avg_velocity
    real(prec) :: avg_depth
    real(prec) :: maxValue
    real(prec) :: dimlessWaveLength
      
  !-----------------------------------------------------------------------------  
    maxValue          = 1e7
    dimlessWaveLength = 4000.
    
    ncomp = frnw_g(j, 1)
    do i = 1, ncomp - 1

      ! upstream metrics
      wl_us    = newY(i, j)                          ! water level
      depth_us = newArea(i, j) / bo(i, j)            ! depth (rectangular?)
      q_us     = newQ(i, j)                          ! flow
      v_us     = abs(newQ(i, j) / newArea(i, j) )    ! velocity
      pere_us  = pere(i, j)                          ! wetted perimeter
      r_us     = newArea(i, j) / pere(i, j)          ! hydraulic radius
      sk_us    = sk(i, j)                            ! 1/Mannings' N 
      ch_us    = sk(i, j) * r_us ** (1. / 6.)        ! parameter for Chezy's constant

      ! downstream metrics
      wl_ds    = newY(i+1, j)                        ! water level
      depth_ds = newArea(i+1, j) / bo(i+1, j)        ! depth (rectangular?)
      q_ds     = newQ(i+1, j)                        ! flow
      v_ds     = abs(newQ(i+1, j) / newArea(i+1, j)) ! velocity
      pere_ds  = pere(i+1, j)                        ! wetted perimeter
      r_ds     = newArea(i+1, j) / pere(i+1, j)      ! hydraulic radius
      sk_ds    = sk(i+1, j)                          ! 1/Mannings' N 
      ch_ds    = sk(i+1, j) * r_ds ** (1./6.)        ! parameter for Chezy's constant


      ch_star_avg  = ((ch_us + ch_ds) / 2.)  / sqrt(grav)
      avg_celerity = (celerity(i, j) + celerity(i + 1, j)) / 2.0
      avg_velocity = (v_us + v_ds) / 2.
      avg_depth    = (depth_us + depth_ds) / 2.
      
      channel_length = dx(i, j)

      ! dimensionless Courant number
      dimensionless_Cr(i, j) = abs(avg_velocity / avg_celerity)
      if (dimensionless_Cr(i, j) > maxValue) dimensionless_Cr(i, j) = maxValue

      ! dimensionless Froude number
      dimensionless_Fo(i, j) = avg_velocity / sqrt(grav * avg_depth)
      if (dimensionless_Fo(i, j) > maxValue) dimensionless_Fo(i, j) = maxValue

      ! dimensionless Friction parameter (influence of friction effects on river flow)
      dimensionless_Fi(i, j) = 2. * dimensionless_Cr(i,j) / (ch_star_avg ** 2.) * &
                               dimlessWaveLength
      if (dimensionless_Fi(i, j) > maxValue) dimensionless_Fi(i, j) = maxValue

      ! dimensionless Friction parameter (influence of Fi and Courant number)
      dimensionless_Fc(i, j) = dimensionless_Cr(i, j) * dimensionless_Fi(i, j)
      if (dimensionless_Fc(i, j) > maxValue) dimensionless_Fc(i, j) = maxValue

      ! dimensionless ratio of Courant to Froude numbers or surface wave to measured wave celerity
      dimensionless_Di(i, j) = (dimensionless_Cr(i, j) / &
                                dimensionless_Fo(i, j)) ** 2.
      if (dimensionless_Di(i,j) > maxValue) dimensionless_Di(i,j) = maxValue

      ! dimensionless diffusion coefficient (ratio of wave diffusion to wave advection)
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
      real(prec), intent(in) :: dtini_given
      real(prec), intent(in) :: t0
      real(prec), intent(in) :: t
      real(prec), intent(in) :: tfin
      real(prec), intent(in) :: saveInterval
    
    ! Local variables
      integer :: ncomp
      integer :: i, irow, flag_da, n
      real(prec) :: a1, a2, a3, a4
      real(prec) :: b1, b2, b3, b4
      real(prec) :: dd1, dd2, dd3, dd4
      real(prec) :: h1, h2, h3, h4
      real(prec) :: allqlat
      real(prec) :: qy, qxy, qxxy, qxxxy
      real(prec) :: ppi, qqi, rri, ssi, sxi
      real(prec) :: cour, cour2
      real(prec) :: alpha
      real(prec) :: currentQ
      real(prec) :: eei_ghost, ffi_ghost, exi_ghost
      real(prec) :: fxi_ghost, qp_ghost, qpx_ghost
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

      ncomp  = frnw_g(j, 1)
    
      ! sum of lateral inflows along the reach
      !allqlat = sum(lateralFlow(2:ncomp - 1, j) * dx(2:ncomp - 1, j))
      allqlat = 0.0
      do i = 2, ncomp - 1
        allqlat = allqlat + lateralFlow(i, j) * dx(i, j)
      end do

      !print *, '****** DIFFUSIVE FORWARD *****'
      !print *, '---------'
      ncomp = frnw_g(j, 1)
      
      do i = 2, ncomp          
        
        cour  = dtini / dx(i - 1, j)
        cour2 = abs(celerity(i, j) ) * cour
               
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
      cour  = dtini / dx(ncomp-1, j)
      cour2 = abs(celerity(ncomp-1, j)) * cour

      a1 = 3.0 * cour2 ** 2.0 - 2.0 * cour2 ** 3.0
      a2 = 1 - a1
      a3 = ( cour2 ** 2.0 - cour2 ** 3.0 ) * dx(ncomp-1, j)
      a4 = ( -1.0 * cour2 + 2.0 * cour2 ** 2.0 - cour2 ** 3.0 ) * dx(ncomp-1, j)

      b1 = ( 6.0 * cour2 - 6.0 * cour2 ** 2.0 ) / ( -1.0 * dx(ncomp-1, j) )
      b2 = - b1
      b3 = ( 2.0 * cour2 - 3.0 * cour2 ** 2.0 ) * ( -1.0 )
      b4 = ( -1.0 + 4.0 * cour2 - 3.0 * cour2 ** 2.0 ) * ( -1.0 )

      dd1 = ( 6.0 - 12.0 * cour2 ) / ( dx(ncomp-1, j) ** 2.0 )
      dd2 = - dd1
      dd3 = ( 2.0 - 6.0 * cour2 ) / dx(ncomp-1, j)
      dd4 = ( 4.0 - 6.0 * cour2 ) / dx(ncomp-1, j)

      h1 = 12.0 / ( dx(ncomp-1, j) ** 3.0 )
      h2 = - h1
      h3 = 6.0 / ( dx(ncomp-1, j) ** 2.0 )
      h4 = h3

      alpha = 1.0

      qy    = a1 * oldQ(ncomp, j) + a2 * oldQ(ncomp-1, j) + &
              a3 * qpx(ncomp, j) + a4 * qpx(ncomp-1, j)
      qxy   = b1 * oldQ(ncomp, j) + b2 * oldQ(ncomp-1, j) + &
              b3 * qpx(ncomp, j) + b4 * qpx(ncomp-1, j)
      qxxy  = dd1 * oldQ(ncomp, j) + dd2 * oldQ(ncomp-1, j) + &
              dd3 * qpx(ncomp, j) + dd4 * qpx(ncomp-1, j)
      qxxxy = h1 * oldQ(ncomp, j) + h2 * oldQ(ncomp-1, j) + &
              h3 * qpx(ncomp, j) + h4 * qpx(ncomp-1, j)

      ppi = - theta * diffusivity(ncomp, j) * dtini / &
              ( dx(ncomp-1, j) ** 2.0 ) * 2.0 / (alpha*(alpha + 1.0)) * alpha
      qqi = 1.0 - ppi * (alpha + 1.0) / alpha
      rri = ppi / alpha

      ssi = qy  + dtini * diffusivity(ncomp-1, j) * ( 1.0 - theta ) * qxxy
      sxi = qxy + dtini * diffusivity(ncomp-1, j) * ( 1.0 - theta ) * qxxxy

      eei_ghost = -1.0 * rri / ( ppi * eei(ncomp) + qqi )
      ffi_ghost = ( ssi - ppi * ffi(ncomp) ) / ( ppi * eei(ncomp) + qqi )

      exi_ghost = -1.0 * rri / ( ppi * exi(ncomp) + qqi )
      fxi_ghost = ( sxi - ppi * fxi(ncomp) ) / ( ppi * exi(ncomp) + qqi )

      qp_ghost  = oldQ(ncomp-1, j)
      qpx_ghost = 0.
      
      ! when a reach has usgs streamflow data at its location, apply DA
      !if (usgs_da_reach(j) /= 0) then
        
      !  allocate(varr_da(nts_da))
      !  do n = 1, nts_da
      !      varr_da(n) = usgs_da(n, j)
      !  end do
      !  qp(ncomp,j) = intp_y(nts_da, tarr_da, varr_da, t + dtini/60.)
      !  flag_da = 1
        ! check usgs_da value is in good quality
      !  irow = locate(tarr_da, t + dtini/60.)
      !  if (irow == nts_da) then
      !    irow = irow-1
      !  endif        
      !  if ((varr_da(irow)<= -4443.999).or.(varr_da(irow+1)<= -4443.999)) then
          ! when usgs data is missing or in poor quality
          qp(ncomp,j)  = eei(ncomp) * qp_ghost + ffi(ncomp)
          flag_da = 0
      !  endif
      !  deallocate(varr_da)
     ! else
        
        qp(ncomp,j)  = eei(ncomp) * qp_ghost + ffi(ncomp)
        flag_da = 0
     ! endif
      
      qpx(ncomp,j) = exi(ncomp) *qpx_ghost + fxi(ncomp)

      do i = ncomp-1, 1, -1
        qp(i, j)  = eei(i) * qp(i+1, j) + ffi(i)
        qpx(i, j) = exi(i) * qpx(i+1, j) + fxi(i)
      end do

      ! when a reach hasn't been applied to DA 
     ! if ((usgs_da_reach(j) == 0).or.(flag_da == 0)) then
       qp(1, j) = newQ(1, j)
       qp(1, j) = qp(1, j) + allqlat
     ! endif
  
      do i = 1, ncomp
        if (abs(qp(i, j)) < q_llm) then
          qp(i, j) = q_llm    
        end if
      end do

      ! update newQ
      do i = 1, ncomp
        newQ(i, j) = qp(i, j)
      end do

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
  
  subroutine mesh_diffusive_backward(dtini_given, t0, t, tfin, saveInterval, j) ! leftBank, rightBank

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

  ! Subroutine arguments
    integer, intent(in) :: j
    real(prec), intent(in) :: dtini_given
    real(prec), intent(in) :: t0
    real(prec), intent(in) :: t
    real(prec), intent(in) :: tfin
    real(prec), intent(in) :: saveInterval
      
  ! Subroutine variables
    integer :: depthCalOk(mxncomp)
    integer :: i, ncomp  
    integer :: jj
    real(prec) :: xt
    real(prec) :: q_sk_multi, sfi
    real(prec) :: vel, currentQ
    real(prec) :: S_ncomp                             
    real(prec) :: tempDepthi_1
    real(prec) :: Q_cur, Q_ds, z_cur, z_ds, y_cur, y_ds
    real(prec) :: C_ulm

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
      print*, 'At j = ',j,'i = ',ncomp, 'time =',t, 'newY(ncomp, j)=', newY(ncomp, j), &
            'newArea(ncomp, j)=', newArea(ncomp,j), 'interpolation of newArea was not possible'
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
        ! test                
        print*, t, i, j, newY(i,j), newY(i, j)-z(i, j), co(i)  
        print*, 'At j = ',j,', i = ',i, 'time =',t, &
                'interpolation of conveyance was not possible, wl', &
                newY(i,j), 'z',z(i,j),'previous wl',newY(i+1,j) !, &
                !'previous z',z(i+1,j), 'dimensionless_D(i,j)', &
                !dimensionless_D(i,j)
  !      stop
         pause !test
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
      
      ! Estimate wetted perimeter by interpolation
      call r_interpol(elevTable, pereTable, nel, &
                      xt, pere(i, j))
      
      ! Estimate width by interpolation
      call r_interpol(elevTable, topwTable, nel, &
                      xt, bo(i, j))
      
      ! Estimate sk(i,j) by interpolation (???? what is sk?)
      call r_interpol(elevTable, skkkTable, nel, &
                      xt, sk(i, j))

      sfi = qp(i, j) * abs(qp(i, j)) / (co(i) ** 2.0)
!      chnWidth = rightBank(i, j)-leftBank(i, j)      
!      chnWidth = min(chnWidth, bo(i, j))

      ! ???? What exactly is happening, here ????
      !if (depthCalOk(i) .eq. 1) then
      
        ! Calculate celerity, diffusivity and velocity
        celerity2(i)    = 5.0 / 3.0 * abs(sfi) ** 0.3 * &
                          abs(qp(i, j)) ** 0.4 / bo(i, j) ** 0.4 &
                          / (1. / (sk(i, j) * q_sk_multi)) ** 0.6

        ! put upper limit on computed celerity to make sure internal temporal increment not being too small.
        if (i.gt.1) then
          C_ulm = cfl*dx(i-1,j)/dtini_min
        else 
          C_ulm = cfl*dx(i,j)/dtini_min
        endif

        if (celerity2(i).gt.C_ulm) then
          celerity2(i) = C_ulm
        endif
        
        diffusivity2(i) = abs(qp(i, j)) / 2.0 / bo(i, j) / abs(sfi)
        vel             = qp(i, j) / newArea(i, j)
        
        ! Check celerity value
        !if (celerity2(i) .gt. 3.0 * vel) celerity2(i) = vel * 3.0
      !else
      !  if (qp(i, j) .lt. 1) then
      !    celerity2(i) = C_llm
      !  else
      !    celerity2(i) = 1.0
      !  end if
      !  diffusivity2(i) =  diffusivity(i,j)
      !end if
      
      if (i .gt. 1) then       
        ! ====================================================
        ! Combination of Newton-Raphson and Bisection
        ! ====================================================                    
        Q_cur        = qp(i-1, j)
        Q_ds         = qp(i, j)
        z_cur        = z(i-1, j)
        z_ds         = z(i, j)
        y_ds         = newY(i, j) - z(i, j)
        y_ds         = max(y_ds, 0.005)       
        y_cur        = rtsafe(i-1, j, Q_cur, Q_ds, z_cur, z_ds, y_ds)
        tempDepthi_1 = y_cur              
        newY(i-1,j)  = tempDepthi_1 + z(i-1, j)

        if (newY(i-1, j) > 10.0**5.) newY(i-1, j) = 10.0**5.
              
        if (newY(i - 1, j) - z(i-1, j) <= 0.) then
            print *, ' newY(i-1,j)-z(i-1,j): ', newY(i-1,j)-z(i-1,j)
            print *, ' newY(i-1,j): ', newY(i-1,j)
            print *, 'z(i-1,j): ', z(i-1,j)
            print *, 'qp(i-1,j): ', qp(i-1,j)
            print*, 'depth is negative at time=,', t,'j= ', j,'i=',i-1, &
                    'newY =', (newY(jj,j),jj=1,ncomp)
            !print*, 'dimensionless_D',(dimensionless_D(jj,j),jj=1,ncomp)
            print*, 'newQ',(newQ(jj,j),jj=1,ncomp)
            print*, 'Bed',(z(jj,j),jj=1,ncomp)
            print*, 'dx',(dx(jj,j),jj=1,ncomp-1)
        end if     
      end if
           
    end do

    celerity(1:ncomp, j) =  sum(celerity2(1:ncomp)) / ncomp
    
    if (celerity(1, j) < C_llm) celerity(1:ncomp,j) = C_llm
    
    diffusivity(1:ncomp, j) = sum(diffusivity2(1:ncomp)) / ncomp
    
    do i = 1, ncomp
      if (diffusivity(i, j) > D_ulm) diffusivity(i, j) = D_ulm 
      if (diffusivity(i, j) < D_llm) diffusivity(i, j) = D_llm 
    end do

  end subroutine mesh_diffusive_backward

  function rtsafe(i, j, Q_cur, Q_ds, z_cur, z_ds, y_ds)
    
    implicit none
        
  !----------------------------------------------------------------------------------------------------------
  ! Description:
  !   Compute water depth using diffusive momentum equation using a combination of Newton-Raphson and Bisection
  !
  ! Method:     
  !  p.1189, Numerical Recipes in F90  
  !  Using a combination of newton-raphson and bisection, find the root of a function bracketed
  !  between x1 and x2. the root, returned as the function value rtsafe, will be refined until
  !  its accuracy is known within ±xacc.
  !  - funcd is a user-supplied subroutine that returns both the function value and the first   
  !    derivative of the function.
  !  - parameter: maxit is the maximum allowed number of iterations.
  !----------------------------------------------------------------------------------------------------------

    ! Subroutine arguments
    integer,          intent(in) :: i, j
    real(prec), intent(in) :: Q_cur, Q_ds, z_cur, z_ds, y_ds
  
    ! Subroutine local variable
    integer,          parameter :: maxit = 40
    real(prec), parameter :: xacc = 1e-4
    integer                     :: iter
    integer                     :: xcolID, ycolID
    real(prec)            :: x1, x2, df, dxx, dxold, f, fh, fl, temp, xh, xl
    real(prec)            :: y_norm, y_ulm_multi, y_llm_multi, elv_norm, y_old
    real(prec)            :: rtsafe    
    
    y_ulm_multi = 2.0
    y_llm_multi = 0.1
    
    xcolID   = 10
    ycolID   = 1
    elv_norm = intp_xsec_tab(i, j, nel, xcolID, ycolID, abs(Q_cur)) ! normal elevation not depth
    y_norm   = elv_norm - z(i, j)
    y_old    = oldY(i, j) - z(i, j)    
   
    ! option 1 for initial point
    !x1       = y_norm * y_llm_multi
    !x2       = y_norm * y_ulm_multi
    ! option 2 for initial point
    x1       = 0.5 * (y_norm + y_old) * y_llm_multi
    x2       = 0.5 * (y_norm + y_old) * y_ulm_multi
       
    call funcd_diffdepth(i, j, Q_cur, Q_ds, z_cur, z_ds, x1, y_ds, fl, df)

    call funcd_diffdepth(i, j, Q_cur, Q_ds, z_cur, z_ds, x2, y_ds, fh, df)    
   
    if ((fl > 0.0 .and. fh > 0.0) .or. (fl < 0.0 .and. fh < 0.0)) then
      rtsafe = y_norm
      return
    endif

    if (fl == 0.0) then
      rtsafe = x1
      return
    elseif (fh == 0.0) then
      rtsafe = x2
      return
    elseif (fl < 0.0) then ! orient the search so that f(xl) < 0.
      xl = x1
      xh = x2
    else
      xh = x1
      xl = x2
    end if

    rtsafe = 0.50 * (x1 + x2)      ! initialize the guess for root
    dxold  = abs(x2 - x1)          ! the “stepsize before last,”
    dxx    = dxold                 ! and the last step.
    
    call funcd_diffdepth(i, j, Q_cur, Q_ds, z_cur, z_ds, rtsafe, y_ds, f, df)

    do iter = 1, maxit             ! loop over allowed iterations.
      if (((rtsafe - xh) * df - f) * ((rtsafe - xl) * df - f) > 0.0 .or. &
                                    abs(2.0 * f) > abs(dxold * df) ) then
      ! bisect if newton out of range, or not decreasing fast enough.
        dxold  = dxx
        dxx    = 0.50 * (xh - xl)
        rtsafe = xl + dxx
        if (xl == rtsafe) return       
      else                         ! newton step acceptable. take it.
        dxold  = dxx
        dxx    = f / df
        temp   = rtsafe
        rtsafe = rtsafe - dxx
        if (temp == rtsafe) return      
      end if
      
      if (abs(dxx) < xacc) return 

      ! one new function evaluation per iteration.
      call funcd_diffdepth(i, j, Q_cur, Q_ds, z_cur, z_ds, rtsafe, y_ds, f, df)

      if (f < 0.0) then            ! maintain the bracket on the root.
        xl = rtsafe
      else
        xh = rtsafe
      end if
    end do

    ! when root is not converged:
    rtsafe = y_norm
    
  end function rtsafe

  subroutine funcd_diffdepth(i, j, Q_cur, Q_ds, z_cur, z_ds, y_cur, y_ds, f, df)
    
    implicit none
    
    !-------------------------------------------------------------------------
    ! Description:
    !   Compute diffusive momentum function value and the first derivative
    !
    ! Method:     
    !   Analytical function and its analytical derivative
    !-------------------------------------------------------------------------

    ! subroutine arguments
    integer,          intent(in)  :: i, j
    real(prec), intent(in)  :: Q_cur, Q_ds, z_cur, z_ds, y_cur, y_ds
    real(prec), intent(out) :: f, df
    
    ! subroutine local variables
    integer          :: xcolID, ycolID
    real(prec) :: elv_cur, elv_ds, conv_cur, conv_ds, sf_cur, sf_ds, slope
    real(prec) :: dKdA_cur, topw_cur

    xcolID  = 1
    ! f(y_cur): function of y at the current node
    ! - energy slope at downstream node
    ycolID  = 5
    elv_ds  = y_ds + z_ds
    conv_ds = intp_xsec_tab(i + 1, j, nel, xcolID, ycolID, elv_ds)
    sf_ds   = abs(Q_ds) * Q_ds / conv_ds**2.0    
    ! - energy slope at current node
    elv_cur  = y_cur + z_cur
    conv_cur = intp_xsec_tab(i, j, nel, xcolID, ycolID, elv_cur)
    sf_cur   = abs(Q_cur) * Q_cur / conv_cur**2.0
    ! - f(y_cur)
    slope    = (z(i, j) - z(i+1, j)) / dx(i, j)
    slope    = max(slope, so_llm)
    f        = y_cur - y_ds + slope * dx(i, j) - 0.50 * (sf_cur + sf_ds) * dx(i, j)

    ! df/dy at y at current node
    ! - dK/dA
    ycolID   = 9
    dKdA_cur = intp_xsec_tab(i, j, nel, xcolID, ycolID, elv_cur)
    ! - top width
    ycolID   = 6
    topw_cur = intp_xsec_tab(i, j, nel, xcolID, ycolID, elv_cur)
    df = 1.0 + (abs(Q_cur) * Q_cur / conv_cur**3.0) * dx(i, j) * topw_cur * dKdA_cur

  end subroutine funcd_diffdepth

  real(prec) function intp_xsec_tab(i, j, nrow, xcolID, ycolID, x)
    
    implicit none
    
    !-------------------------------------------------------------------
    ! Description:
    !   Interpolate given columns of hydraulic lookup table
    !
    ! Method:     
    !   linear interpolation between selected adjacent data points
    !-------------------------------------------------------------------

    ! subroutine arguments
    integer         , intent(in) :: i, j, nrow, xcolID, ycolID
    real(prec), intent(in) :: x
    
    ! subroutine local variables
    integer                           :: irow
    real(prec)                  :: x1, y1, x2, y2, y
    real(prec), dimension(nrow) :: xarr, yarr

    xarr = xsec_tab(xcolID, 1:nrow, i, j)
    yarr = xsec_tab(ycolID, 1:nrow, i, j)

    irow = locate(xarr, x)

    if (irow == 0)    irow = 1
    if (irow == nrow) irow = nrow-1

    x1            =   xarr(irow)
    y1            =   yarr(irow)
    x2            =   xarr(irow+1)
    y2            =   yarr(irow+1)
    y             = LInterpol(x1, y1, x2, y2, x)
    intp_xsec_tab = y
  end function intp_xsec_tab

   !**-----------------------------------------------------------------------------------------
    !*      Create lookup tables at each node storing computed values of channel geometries
    !*      such as area and conveyance and normal/critical depth for possible ranges of
    !*      water depth.
    !
    !**-----------------------------------------------------------------------------------------
  subroutine readXsection_natural_mann_vertices(idx_node, idx_reach, timesDepth)
    
    implicit none

    !-------------------------------------------------------------------------------------------------
    ! Description:
    !   Build hydraulic lookup table containing cross sections' area, wetted perimeter, hydraulic
    !   radius, top water surface width, conveyance,  derivative of conveyance w.r.t. area,
    !   1 / composite Mannings' N all with respect to incrementally increasing water elevation values.
    !
    ! Method:     
    !   All hydraulic properties of each cross section are computed for a possible range of 
    !   water elevation value that starts from the lowest channel elevation point in bathymetry data
    !-------------------------------------------------------------------------------------------------
    
    ! subroutine arguments
    integer,          intent(in) :: idx_node, idx_reach 
    real(prec), intent(in) :: timesDepth
    
    ! subroutine local variables
    integer          :: i_area, i_find, num
    integer          :: i1, i2
    integer          :: ic, iel, ii, ii2, iv, iel_start, iel_incr_start, iel_decr_start, ndmy
    real(prec) :: el_min, el_max, el_range, el_incr, el_now, x1, y1, x2, y2, x_start, x_end
    real(prec) :: f2m, cal_area, cal_peri, cal_topW
    real(prec) :: mN_start, mN_end, cal_equiv_mann
    real(prec) :: pos_slope, incr_rate, max_value  
    integer,          dimension(:), allocatable :: i_start, i_end
    real(prec), dimension(:), allocatable :: x_bathy_leftzero
    real(prec), dimension(:), allocatable :: xcs, ycs, manncs
    real(prec), dimension(:), allocatable :: el1, a1, peri1, redi1, equiv_mann
    real(prec), dimension(:), allocatable :: redi1All
    real(prec), dimension(:), allocatable :: conv1, tpW1
    real(prec), dimension(:), allocatable :: newdKdA
    real(prec), dimension(:), allocatable :: compoundSKK, elev, dmyarr

    allocate(el1(nel), a1(nel), peri1(nel), redi1(nel), redi1All(nel))
    allocate(equiv_mann(nel), conv1(nel), tpW1(nel))
    allocate(newdKdA(nel))
    allocate(compoundSKK(nel), elev(nel))
    allocate(i_start(nel), i_end(nel))

    f2m            =   1.0
    maxTableLength = size_bathy(idx_node, idx_reach) + 2 ! 2 is added to count for a vertex on each infinite vertical wall on either side.

    allocate(xcs(maxTableLength), ycs(maxTableLength), manncs(maxTableLength))
    allocate(x_bathy_leftzero(maxTableLength) )

    ! As x_bathy data take negative values for the left of the streamline (where x=0) while positive for the right when looking from
    ! upstream to downstream. This subroutine takes zero at left-most x data point, so an adjustment is required.

    do ic = 1, size_bathy(idx_node, idx_reach)
      x_bathy_leftzero(ic) = - x_bathy(1, idx_node, idx_reach) + x_bathy(ic, idx_node, idx_reach)
    end do
        
    do ic = 2, size_bathy(idx_node, idx_reach) + 1
      x1         = x_bathy_leftzero(ic-1)
      y1         = z_bathy(ic-1, idx_node, idx_reach)
      xcs(ic)    = x1 * f2m
      ycs(ic)    = y1 * f2m
      manncs(ic) = mann_bathy(ic-1, idx_node, idx_reach)
      ! avoid too large (egregiously) manning's N value
      if  (manncs(ic).gt.0.15) then !0.15 is typical value for Floodplain trees
        manncs(ic) = 0.15
      endif 
    end do

    num = maxTableLength

    ! max. and min elevation
    el_min = 99999.
    el_max = -99999.

    do ic = 2, num - 1
      if (ycs(ic) < el_min) el_min = ycs(ic)
      if (ycs(ic) > el_max) el_max = ycs(ic)
    enddo

    el_range = (el_max - el_min) * timesDepth
    el_incr  = el_range / real(nel - 1.0)

    ! vertex on each infinite vertical wall on each side of x-section
    xcs(1)        = xcs(2)
    ycs(1)        = el_min + el_range + 1.0
    xcs(num)      = xcs(num-1)
    ycs(num)      = el_min + el_range + 1.0
    manncs(1)     = 0.0 ! to make perimeter * manningN equal to zero along vertical walls
    manncs(num-1) = 0.0 ! to make perimeter * manningN equal to zero along vertical walls
    manncs(num)   = 0.0 ! to make perimeter * manningN equal to zero along vertical walls

    do iel = 1, nel
      el_now = el_min + real(iel - 1) * el_incr

      if (abs(el_now - el_min) < TOLERANCE) then
        el_now = el_now + 0.00001
      end if

      i_start(1) = -999
      i_end(1)   = -999
      i_area     = 0
      i_find     = 0

      ! find starting and ending vertices of multiple sub-xsections under the current elevation (=el_now).
      do ic = 1, num - 1
        y1 = ycs(ic)
        y2 = ycs(ic+1)

        if ((el_now <= y1) .and. (el_now > y2) .and. (i_find == 0)) then
          i_find          = 1
          i_area          = i_area + 1
          i_start(i_area) = ic
        endif

        if ((el_now > y1) .and. (el_now <= y2) .and. (i_find == 1)) then
          i_find        = 0
          i_end(i_area) = ic
        endif
      end do

      cal_area       = 0.0
      cal_peri       = 0.0
      cal_topW       = 0.0
      cal_equiv_mann = 0.0

      do ic = 1, i_area ! i_area counts selected sub-x sections (each with start and end vertices) under el_now.
        x1 = xcs( i_start(ic) )
        x2 = xcs( i_start(ic) + 1 )
        y1 = ycs( i_start(ic) )
        y2 = ycs( i_start(ic) + 1 )

        if (y1 == y2) then
          x_start = x1
        else
          x_start = x1 + (el_now - y1) / (y2 - y1) * (x2 - x1)
        endif

        x1 = xcs(i_end(ic))
        x2 = xcs(i_end(ic) + 1)
        y1 = ycs(i_end(ic))
        y2 = ycs(i_end(ic) + 1)

        if (y1 == y2) then
          x_end = x1
        else
          x_end = x1 + (el_now - y1) / (y2 - y1) * (x2 - x1)
        endif

        cal_topW = x_end - x_start + cal_topW

        i1 = i_start(ic)
        i2 = i_end(ic)
        ! area
        cal_area       =   cal_area    &
                       +   cal_tri_area(el_now, x_start, xcs(i1 + 1), ycs(i1 + 1))    &
                       +   cal_multi_area(el_now, xcs, ycs, maxTableLength, i1 + 1, i2) &
                       +   cal_tri_area(el_now, x_end, xcs(i2), ycs(i2))
        ! wetted parameter
        cal_peri       =   cal_peri    &
                       +   cal_dist(x_start, el_now, xcs(i1 + 1), ycs(i1 + 1))    &
                       +   cal_perimeter(xcs, ycs, maxTableLength, i1 + 1, i2)    &
                       +   cal_dist(x_end, el_now, xcs(i2), ycs(i2))
        ! nominator value for computing composite or equivalent manning's N
        mN_start       =   manncs(i1)
        mN_end         =   manncs(i2)
        cal_equiv_mann =   cal_equiv_mann &
                       +   cal_dist_x_mann(x_start, el_now, xcs(i1 + 1), ycs(i1 + 1), mN_start) &
                       +   cal_peri_x_mann(xcs, ycs, manncs, maxTableLength, i1 + 1, i2) &
                       +   cal_dist_x_mann(x_end, el_now, xcs(i2), ycs(i2), mN_end)

        if (i1 == 1) cal_peri         = cal_peri - cal_dist(x_start, el_now, xcs(i1 + 1), ycs(i1 + 1))                                           
        if (i2 == (num - 1)) cal_peri = cal_peri - cal_dist(x_end, el_now, xcs(i2), ycs(i2))                                                
      enddo

      el1(iel)        = el_now
      a1(iel)         = cal_area
      peri1(iel)      = cal_peri
      tpW1(iel)       = cal_topW
      redi1(iel)      = a1(iel) / peri1(iel)
      equiv_mann(iel) = (cal_equiv_mann / cal_peri)**(2.0 / 3.0)
      conv1(iel)      = (1.0 / equiv_mann(iel)) * a1(iel) * (redi1(iel)**(2.0 / 3.0))

      if (peri1(iel) <= TOLERANCE) then
        redi1(iel) = 0.0
        conv1(iel) = 0.0
      endif

      if (iel == 1) then
        newdKdA(iel) = conv1(iel) / a1(iel)
      else
        newdKdA(iel) = (conv1(iel) - conv1(iel-1)) / (a1(iel) - a1(iel-1))
      end if

      compoundSKK(iel) = 1.0 / equiv_mann(iel)   
    enddo 

    ! smooth conveyance curve (a function of elevation) so as to have monotonically increasing curve
    iel_start = 2
    incr_rate = 0.01
    do iel = iel_start, nel
      if (conv1(iel) <= conv1(iel-1)) then
        ! -- find j* such that conv1(j*) >> conv1(j-1)
        ii = iel
        
        do while ((conv1(ii) < conv1(iel-1)).and.(ii < nel))
          ii = ii + 1
        end do
        
        iel_incr_start = ii
        ! when there is no data point greater than conv1(iel-1), then use artificially increased one.
        if ((iel_incr_start.ge.nel).and.(conv1(iel_incr_start) < conv1(iel-1))) then
          conv1(iel_incr_start) = (1.0 + incr_rate) * conv1(iel-1)
        endif

        pos_slope      = (conv1(iel_incr_start) - conv1(iel-1)) / (el1(iel_incr_start) - el1(iel-1))

        do ii = iel, iel_incr_start - 1
          conv1(ii) = conv1(iel-1) + pos_slope * (el1(ii) - el1(iel-1))
        end do

        ! update dKdA accordingly
        do ii = iel, iel_incr_start - 1
          if (ii == 1) then
            newdKdA(ii) = conv1(ii) / a1(ii)
          else
            newdKdA(ii) = (conv1(ii) - conv1(ii-1)) / (a1(ii) - a1(ii-1))
          end if
        enddo

        iel_start = iel_incr_start
      endif
    enddo

    ! smooth dKdA curve (a function of elevation) so as to have monotonically increasing curve
    iel_start = 2
    incr_rate = 0.01
    do iel = iel_start, nel 
      if (newdKdA(iel) <= newdKdA(iel-1)) then
        ! -- find j* such that conv1(j*) >> conv1(j-1)
        ii = iel
        
        do while ((newdKdA(ii) < newdKdA(iel-1)).and.(ii < nel))
          ii = ii + 1
        end do
        
        iel_incr_start = ii
        ! when there is no data point greater than newdKdA(iel-1), then use artificially increased one.        
        if ((iel_incr_start.ge.nel).and.(newdKdA(iel_incr_start) < newdKdA(iel-1))) then
          newdKdA(iel_incr_start) =  (1.0 + incr_rate) * newdKdA(iel-1)
        endif
        
        pos_slope = (newdKdA(iel_incr_start) - newdKdA(iel-1)) / (el1(iel_incr_start) - el1(iel-1))     
        
        do ii = iel, iel_incr_start - 1
          newdKdA(ii) = newdKdA(iel-1) + pos_slope * (el1(ii) - el1(iel-1))
        enddo

        iel_start = iel_incr_start
      endif 
    enddo

    ! finally build lookup table
    do iel = 1,  nel
      xsec_tab(1, iel, idx_node, idx_reach)     =   el1(iel)
      xsec_tab(2, iel, idx_node, idx_reach)     =   a1(iel)
      xsec_tab(3, iel, idx_node, idx_reach)     =   peri1(iel)
      xsec_tab(4, iel, idx_node, idx_reach)     =   redi1(iel)
      xsec_tab(5, iel, idx_node, idx_reach)     =   conv1(iel)
      xsec_tab(6, iel, idx_node, idx_reach)     =   tpW1(iel)
      !xsec_tab(7,iel,idx_node,idx_reach) = sum(newI1(iel,:))  !* <- not used
      !xsec_tab(8,iel,idx_node,idx_reach) = newdPdA(iel)       !* <- not used
      xsec_tab(9, iel, idx_node, idx_reach)     =   newdKdA(iel)
      xsec_tab(11,iel, idx_node, idx_reach)     =   compoundSKK(iel)
    end do

    z(idx_node, idx_reach)  =   el_min

    deallocate(el1, a1, peri1, redi1, redi1All)
    deallocate(conv1, tpW1, equiv_mann)
    deallocate(newdKdA)
    deallocate(compoundSKK, elev)
    deallocate(i_start, i_end)
    deallocate(xcs, ycs, manncs)
    deallocate(x_bathy_leftzero)

    contains
      real(prec) function cal_dist_x_mann(x1, y1, x2, y2, mN)
                
        implicit none
        
        !----------------------------------------------------- 
        ! Description:           	 
        !   calculate distance * manning's N of two vertices
        !-----------------------------------------------------                
        
        ! function arguments
        real(prec), intent(in) :: x1, y1, x2, y2, mN
        ! function local variable
        real(prec) :: dist

        dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + 1.e-32)
        cal_dist_x_mann = dist * mN**1.50

      end function cal_dist_x_mann

      real(prec) function cal_peri_x_mann(xx, yy, mN, n, i1, i2)
        
        implicit none

        !------------------------------------------------------------------------------ 
        ! Description:           	 
        !   calculate wetted perimeter * manning's N of multiple pairs of two vertices
        !------------------------------------------------------------------------------  
        
        ! function arguments
        integer,          intent(in) :: n, i1, i2
        real(prec), intent(in) :: xx(n), yy(n), mN(n)
        ! function local variables
        integer          :: i
        real(prec) :: x1, x2, y1, y2, mN1, pxmN

        pxmN = 0.0

        do i = i1, i2 - 1
          x1      =   xx(i)
          y1      =   yy(i)
          x2      =   xx(i + 1)
          y2      =   yy(i + 1)
          mN1     =   mN(i)
          pxmN    =   pxmN + cal_dist(x1, y1, x2, y2) * mN1**1.50
        enddo

        cal_peri_x_mann = pxmN

      endfunction cal_peri_x_mann

  end subroutine readXsection_natural_mann_vertices

  subroutine readXsection(k,lftBnkMann,rmanning_main,rgtBnkMann,leftBnkX_given,rghtBnkX_given,timesDepth,num_reach,&
                            z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g, lmxncomp, lnlinks)
    implicit none

    !------------------------------------------------------------------------------------------------- 
    ! Description:           	 
    !   Create lookup tables at each node storing computed values of channel geometries
    !   such as area and conveyance for possible ranges of water elevation.
    ! 
    ! Method:     
    !   All hydraulic properties of each cross section are computed for a possible range of 
    !   water elevation value that starts from the lowest channel elevation point in bathymetry data
    !------------------------------------------------------------------------------------------------- 

    ! subroutine arguments
    integer,                                        intent(in) :: k, num_reach, lmxncomp, lnlinks
    real(prec),                               intent(in) :: rmanning_main,lftBnkMann,rgtBnkMann
    real(prec),                               intent(in) :: leftBnkX_given,rghtBnkX_given, timesDepth
    real(prec), dimension(lmxncomp, lnlinks), intent(in) :: z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g
    
    ! subroutine local variables
    integer          :: i_area, i_find, i, j, jj, num  
    integer          :: i1, i2
    integer          :: mainChanStrt, mainChanEnd, kkk, startFound, endFound 
    real(prec) :: el_min, el_max, el_range, el_incr, el_now
    real(prec) :: x1, y1, x2, y2, x_start, x_end
    real(prec) :: waterElev, leftBnkX,rghtBnkX
    real(prec) :: f2m, cal_area, cal_peri, cal_topW,  diffAreaCenter
    real(prec) :: compoundMann, el_min_1
    real(prec) :: leftBnkY, rghtBnkY,rmanning
    real(prec) :: hbf
    integer, dimension(:),            allocatable :: i_start, i_end, totalNodes  
    real(prec), dimension(:),   allocatable :: xcs, ycs
    real(prec), dimension(:,:), allocatable :: el1, a1, peri1, redi1
    real(prec), dimension(:),   allocatable :: redi1All
    real(prec), dimension(:,:), allocatable :: conv1, tpW1, diffArea, newI1, diffPere
    real(prec), dimension(:),   allocatable :: newdPdA, diffAreaAll, diffPereAll, newdKdA       
    real(prec), dimension(:),   allocatable :: compoundSKK, elev
    real(prec), dimension(:,:), allocatable :: allXcs, allYcs
    
    allocate(el1(nel,3), a1(nel,3), peri1(nel,3), redi1(nel,3), redi1All(nel))
    allocate(conv1(nel,3), tpW1(nel,3), diffArea(nel,3), newI1(nel,3), diffPere(nel,3))
    allocate(newdPdA(nel), diffAreaAll(nel), diffPereAll(nel), newdKdA(nel))       ! change Nazmul 20210601
    allocate(compoundSKK(nel), elev(nel))
    allocate(i_start(nel), i_end(nel))
    allocate(totalNodes(3))

    f2m = 1.0 ! conversion from feet to meter (actually using meter so no conversion necessary for now)
    
    leftBnkX   = leftBnkX_given
    rghtBnkX   = rghtBnkX_given
    startFound = 0
    endFound   = 0
    
    ! channel geometry at a given segment
    z_g     = z_ar_g(k, num_reach)
    bo_g    = bo_ar_g(k, num_reach)
    traps_g = traps_ar_g(k, num_reach)
    tw_g    = tw_ar_g(k, num_reach)
    twcc_g  = twcc_ar_g(k, num_reach)
    hbf     = (tw_g - bo_g)/(2.0 * traps_g) !* bankfull depth

    maxTableLength = 8
    allocate(xcs(maxTableLength), ycs(maxTableLength))
    allocate(allXcs(maxTableLength,3), allYcs(maxTableLength,3))
    
    do i = 1, maxTableLength
      ! channel x-section vertices at a given segment
      if (i == 1) then
        x1 = 0.0 
        y1 = z_g + timesDepth * hbf
      elseif (i == 2) then
        x1 = 0.0 
        y1 = z_g + hbf
      elseif (i == 3) then
        x1 = (twcc_g - tw_g) / 2.0 
        y1 = z_g + hbf
      elseif (i == 4) then
        x1 = xcs(3) + traps_g * hbf 
        y1 = z_g
      elseif (i == 5) then
        x1 = xcs(4) + bo_g
        y1 = z_g
      elseif (i == 6) then
        x1 = xcs(5) + traps_g * hbf 
        y1= z_g + hbf
      elseif (i == 7) then
        x1 = twcc_g 
        y1 = z_g + hbf
      elseif (i == 8) then
        x1 = xcs(7) 
        y1 = z_g + timesDepth * hbf
      endif

      xcs(i) = x1 * f2m
      ycs(i) = y1 * f2m
      
      if ((xcs(i) >= leftBnkX) .and. (startFound == 0)) then
        mainChanStrt = i - 1
        startFound   = 1
      end if
      
      if ((xcs(i) >= rghtBnkX) .and. (endFound == 0)) then
        mainChanEnd = i - 1
        endFound    = 1
      end if
    enddo
        
    mainChanStrt = 3
    mainChanEnd  = 6
    num          = i

    if (leftBnkX < minval(xcs(2:num-1))) leftBnkX = minval(xcs(2:num-1))
    if (rghtBnkX > maxval(xcs(2:num-1))) rghtBnkX = maxval(xcs(2:num-1))

    leftBnkY = ycs(mainChanStrt) + (leftBnkX-xcs(mainChanStrt)) / &
               (xcs(mainChanStrt+1) - xcs(mainChanStrt)) *        & 
               (ycs(mainChanStrt+1) - ycs(mainChanStrt))
    
    rghtBnkY = ycs(mainChanEnd) + (rghtBnkX - xcs(mainChanEnd)) / &
               (xcs(mainChanEnd+1) - xcs(mainChanEnd)) *          & 
               (ycs(mainChanEnd+1) - ycs(mainChanEnd))
    
    el_min = 99999.
    el_max = -99999.
    do i = 2, num-1
      if (ycs(i) < el_min) el_min = ycs(i)
      if (ycs(i) > el_max) el_max = ycs(i)
    enddo
        
    el_range = (el_max - el_min) * 2.0 ! change Nazmul 20210601

    do i = 1, 3
      allXcs(i+1, 1) = xcs(i)
      allYcs(i+1, 1) = ycs(i)
    enddo
    
    allXcs(1, 1)              = xcs(1)
    allYcs(1, 1)              = el_min + el_range + 1.
    allXcs(mainChanStrt+2, 1) = xcs(3)
    allYcs(mainChanStrt+2, 1) = el_min + el_range + 1.

    do i = 3, 4
      allXcs(i-1, 2) = xcs(i) 
      allYcs(i-1, 2) = ycs(i) 
    enddo

    do i= 5, 6
      allXcs(i, 2) = xcs(i) 
      allYcs(i, 2) = ycs(i) 
    enddo
        
    allXcs(1, 2) = xcs(3)
    allYcs(1, 2) = el_min + el_range + 1.
    allXcs(7, 2) = xcs(6)
    allYcs(7, 2) = el_min + el_range + 1.

    do i = 6, 8
      allXcs(i-4, 3) = xcs(i) 
      allYcs(i-4, 3) = ycs(i) 
    enddo
    
    allXcs(1, 3)  = allXcs(2, 3)
    allYcs(1, 3)  = el_min + el_range + 1.
    i             = 5
    allXcs(i, 3)  = allXcs(i-1, 3)
    allYcs(i, 3)  = el_min + el_range + 1.

    totalNodes(1) = 5
    totalNodes(2) = 7
    totalNodes(3) = 5

    allXcs(4, 2) = (allXcs(3, 2) + allXcs(5, 2)) / 2.0
    allYcs(4, 2) = allYcs(3, 2) - 0.01

    el_min_1 = el_min
    el_min   = allYcs(4,2)    
    elev(1)  = el_min
    elev(2)  = el_min + 0.01/4.
    elev(3)  = el_min + 0.01/4.*2.
    elev(4)  = el_min + 0.01/4.*3.
    elev(5)  = el_min + 0.01

    el_incr = el_range / real(nel - 6.0)

    do kkk = 6, nel
      elev(kkk) = elev(5) + el_incr * real(kkk - 5)
    end do

    xcs   = 0.
    ycs   = 0.
    newI1 = 0.0
    
    do kkk = 1, 3
      num        = totalNodes(kkk)
      xcs(1:num) = allXcs(1:num, kkk)
      ycs(1:num) = allYcs(1:num, kkk)
      
      if (kkk == 1) rmanning = lftBnkMann
      if (kkk == 2) rmanning = rmanning_main
      if (kkk == 3) rmanning = rgtBnkMann
      
      do j = 1, nel
        el_now = elev(j)
        
        if (abs(el_now - el_min) < TOLERANCE) then
          el_now=el_now+0.00001
        end if
                
        i_start(1) = -999
        i_end(1)   = -999
        i_area     = 0
        i_find     = 0
        
        do i = 1, num - 1
          y1 = ycs(i)
          y2 = ycs(i+1)
          
          if ((el_now <= y1) .and. (el_now > y2) .and. (i_find == 0)) then
            i_find          = 1
            i_area          = i_area + 1
            i_start(i_area) = i
          endif
          
          if ((el_now > y1) .and. (el_now <= y2) .and. (i_find == 1)) then
            i_find        = 0
            i_end(i_area) = i
          endif
        enddo

        cal_area = 0.
        cal_peri = 0.
        cal_topW = 0.

        do i = 1, i_area
          x1=xcs(i_start(i))
          x2=xcs(i_start(i)+1)
          y1=ycs(i_start(i))
          y2=ycs(i_start(i)+1)
          if (y1 == y2) then
            x_start = x1
          else
            x_start = x1 + (el_now - y1) / (y2 - y1) * (x2 - x1)
          endif

          x1 = xcs(i_end(i))
          x2 = xcs(i_end(i) + 1)
          y1 = ycs(i_end(i))
          y2 = ycs(i_end(i) + 1)

          if (y1 == y2) then
            x_end = x1
          else
            x_end = x1 + (el_now - y1) / (y2 - y1) * (x2 - x1)
          endif

          cal_topW = x_end - x_start + cal_topW

          i1       = i_start(i)
          i2       = i_end(i) 
          cal_area = cal_area                                                   &
                   + cal_tri_area(el_now, x_start, xcs(i1+1), ycs(i1+1))        &
                   + cal_multi_area(el_now, xcs, ycs, maxTableLength, i1+1, i2) &
                   + cal_tri_area(el_now, x_end, xcs(i2), ycs(i2))
          
          cal_peri = cal_peri                                                   &
                   + cal_dist(x_start, el_now, xcs(i1+1), ycs(i1+1))            &
                   + cal_perimeter(xcs, ycs, maxTableLength, i1+1, i2)          &
                   + cal_dist(x_end, el_now, xcs(i2), ycs(i2))
                    
          if (i1 == 1)       cal_peri = cal_peri - cal_dist(x_start, el_now, xcs(i1+1), ycs(i1+1))
          if (i2 == (num-1)) cal_peri = cal_peri - cal_dist(x_end, el_now, xcs(i2), ycs(i2))

        enddo

        el1(j, kkk)   = el_now
        a1(j, kkk)    = cal_area
        peri1(j, kkk) = cal_peri
        redi1(j, kkk) = a1(j, kkk) / peri1(j, kkk)
        conv1(j,kkk)  = 1. / rmanning * a1(j, kkk) * (redi1(j, kkk))**(2. / 3.)
        
        if (peri1(j, kkk) <= TOLERANCE) then
          redi1(j, kkk) = 0.0
          conv1(j, kkk) = 0.0
        endif
        
        tpW1(j, kkk) = cal_topW

        if (j == 1) then 
          diffArea(j, kkk) = a1(j, kkk) 
          diffPere(j, kkk) = peri1(j, kkk) 
        else
          if (el_now <= minval(ycs(1:num))) then
            diffArea(j, kkk) = a1(j, kkk)
            diffPere(j, kkk) = peri1(j, kkk)
          else
            diffArea(j, kkk) = a1(j, kkk) - a1(j-1, kkk)
            diffPere(j, kkk) = peri1(j, kkk) - peri1(j-1, kkk)
          endif
        endif

        waterElev=el1(j,kkk)
        
        do jj = 2, j
          diffAreaCenter = el1(jj, kkk) - (el1(jj, kkk) - el1(jj-1, kkk))*0.5
          newI1(j, kkk)  = newI1(j, kkk) + diffArea(jj, kkk) * (waterElev - diffAreaCenter)
        enddo
      end do
    end do

    do j = 1, nel
      el_now = el1(j, 1)
      
      if (j == 1) then
        newdPdA(j) = sum(peri1(j,:)) / sum(a1(j,:))
        newdKdA(j) = sum(conv1(j,:)) / sum(a1(j,:))    
      else
        newdPdA(j) = (sum(peri1(j,:)) - sum(peri1(j-1,:))) / (sum(a1(j,:)) - sum(a1(j-1,:)))
        newdKdA(j) = (sum(conv1(j,:)) - sum(conv1(j-1,:))) / (sum(a1(j,:)) - sum(a1(j-1,:)))
      end if

      compoundMann   = sqrt((abs(peri1(j,1)) * lftBnkMann** 2. + abs(peri1(j, 2)) * rmanning_main**2. + &
                       abs(peri1(j, 3)) * rgtBnkMann**2.) /                                             & 
                       (abs(peri1(j, 1)) + abs(peri1(j, 2)) + abs(peri1(j, 3))))
      compoundSKK(j) = 1. / compoundMann
      redi1All(j)    = sum(a1(j, :)) / sum(peri1(j, :))
            
      xsec_tab(1, j, k, num_reach) = el1(j, 1)
      xsec_tab(2, j, k, num_reach) = sum(a1(j, :))
      xsec_tab(3, j, k, num_reach) = sum(peri1(j, :))
      xsec_tab(4, j, k, num_reach) = redi1All(j)
      xsec_tab(5, j, k, num_reach) = sum(conv1(j, :))
      xsec_tab(6, j, k, num_reach) = abs(tpW1(j, 1)) + abs(tpW1(j, 2)) + abs(tpW1(j, 3))
      xsec_tab(7, j, k, num_reach) = sum(newI1(j, :))
      xsec_tab(8, j, k, num_reach) = newdPdA(j)
      xsec_tab(9, j, k, num_reach) = newdKdA(j)
      xsec_tab(11,j, k, num_reach) = compoundSKK(j)
    end do

    z(k, num_reach) = el_min

    deallocate(el1, a1, peri1, redi1, redi1All)
    deallocate(conv1, tpW1, diffArea, newI1, diffPere)
    deallocate(newdPdA, diffAreaAll, diffPereAll, newdKdA)       
    deallocate(compoundSKK, elev)
    deallocate(i_start, i_end)
    deallocate(totalNodes)
    deallocate(xcs, ycs)
    deallocate(allXcs, allYcs)

  end subroutine readXsection

  real(prec) function cal_tri_area(el, x0, x1, y1)
      
      implicit none
      
      !----------------------------------------
      ! Description:
      !   calculate area of triangle
      !----------------------------------------
      
      ! function arguments
      real(prec), intent(in) :: el, x0, x1, y1

      cal_tri_area = abs(0.5 * (x1 - x0) * (el - y1))
    
    end function cal_tri_area

    real(prec) function cal_trap_area(el, x1, y1, x2, y2)
      
      implicit none

      !----------------------------------------
      ! Description:
      !   calculate area of trapezoid
      !----------------------------------------
                
      real(prec), intent(in) :: el, x1, y1, x2, y2

      cal_trap_area = abs(0.5 * (x2 - x1) * (el - y1 + el - y2))

    end function cal_trap_area

    real(prec) function cal_multi_area(el, xx, yy, n, i1, i2)
    
      implicit none

      !------------------------------------------
      ! Description:
      !   calculate sum of areas of trapezoids
      !------------------------------------------                
      
      ! function arguments
      integer,          intent(in) :: n, i1, i2
      real(prec), intent(in) :: el
      real(prec), intent(in) :: xx(n), yy(n)
      ! function local variables
      integer          :: i
      real(prec) :: area, x1, x2, y1, y2

      area = 0.0

      do i = i1, i2 - 1
        x1   = xx(i)
        y1   = yy(i)
        x2   = xx(i+1)
        y2   = yy(i+1)
        area = area + cal_trap_area(el, x1, y1, x2, y2)
      enddo

      cal_multi_area = area
    
    endfunction cal_multi_area

    real(prec) function cal_dist(x1, y1, x2, y2)
      
      implicit none

      !------------------------------------------
      ! Description:
      !   calculate distance of two vertices
      !------------------------------------------  
      
      ! function arguments
      real(prec), intent(in) :: x1, y1, x2, y2

      cal_dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + 1.e-32)
    
    end function cal_dist

    real(prec) function cal_perimeter(xx,yy,n,i1,i2)
      
      implicit none
        
      !------------------------------------------
      ! Description:
      !   calculate wetted perimeter
      !------------------------------------------ 
      
      ! function arguments
      integer,          intent(in) :: n, i1, i2
      real(prec), intent(in) :: xx(n), yy(n)
      ! function local variables
      integer          :: i
      real(prec) :: p, x1, x2, y1, y2

      p = 0.
      
      do i = i1, i2 - 1
        x1 = xx(i)
        y1 = yy(i)
        x2 = xx(i + 1)
        y2 = yy(i + 1)
        p  = p + cal_dist(x1, y1, x2, y2)
      enddo

      cal_perimeter=p

    end function cal_perimeter

    subroutine r_interpol(x, y, kk, xrt, yt)
      
      implicit none
      !---------------------------------------------------------------------------
      ! Description:
      !   Estimate y for a given along x and y arrays using linear interpolation
      !---------------------------------------------------------------------------         
      ! subroutine arguments
      integer,          intent(in)  :: kk
      real(prec), intent(in)  :: xrt, x(kk), y(kk)
      real(prec), intent(out) :: yt
      ! subroutine local variables
      integer :: k

      if ((xrt <= maxval(x)) .and. (xrt >= minval(x))) then
      
        do k = 1, kk-1
          if (((x(k) - xrt) * (x(k+1) - xrt)) <= 0.0) then
            yt = (xrt - x(k)) / (x(k+1) - x(k)) * (y(k+1) - y(k)) + y(k)
            EXIT
          endif
        end do
      
      else if (xrt >= maxval(x)) then
!       print*, xrt, ' the given x data point is larger than the upper limit of the set of x data points'
!       print*, 'the upper limit: ', maxval(x)
        yt = (xrt - x(kk-1)) / (x(kk) - x(kk-1)) * (y(kk) - y(kk-1)) + y(kk-1) ! extrapolation

      else
!       print*, xrt, ' the given x data point is less than lower limit of the range of known x data point, '
!       print*, 'so linear interpolation cannot be performed.'
!        yt = -9999.0
        yt = minval(y)
!       print*, 'The proper range of x is that: ', 'the upper limit: ', maxval(x),&
!               ' and lower limit: ', minval(x)
!       print*, 'kk', kk
!       print*, 't', 'i', dmyi, 'j', dmyj
!       print*, 'x', (x(k), k=1, kk)
!       print*, 'y', (y(k), k=1, kk)
      end if
    
    end subroutine r_interpol

    subroutine normal_crit_y(i, j, q_sk_multi, So, dsc, y_norm, y_crit, area_n, area_c)
      
      implicit none
        
      !-------------------------------------------------------------------------------------------------
      ! Description:
      !   Estimate normal and critical depth and related areas
      !
      ! Method:
      !   normal depth by linearly interpolating elevation & conveyance columns w.r.t. given conveyance
      !   normal depth area by linearly interpolation elevation & area w.r.t. given normal depth
      !   critical depth by iterative method and the computed depth leads to the estimating of the area	
      !--------------------------------------------------------------------------------------------------

      ! subroutine arguments
      integer,          intent(in)  :: i, j
      real(prec), intent(in)  :: q_sk_multi, So, dsc
      real(prec), intent(out) :: y_norm, y_crit, area_n, area_c
      
      ! subroutine local variables 
      real(prec) :: area_0, width_0, errorY

      elevTable = xsec_tab(1, :, i, j)
      areaTable = xsec_tab(2, :, i, j)
      pereTable = xsec_tab(3, :, i, j)
      convTable = xsec_tab(5, :, i, j)
      topwTable = xsec_tab(6, :, i, j)
      
      call r_interpol(convTable, areaTable, nel, dsc / sqrt(So), area_n)
      call r_interpol(convTable, elevTable, nel, dsc / sqrt(So), y_norm)
      call r_interpol(elevTable, areaTable, nel, oldY(i, j), area_0)  ! initial estimate for critical depth
      call r_interpol(elevTable, topwTable, nel, oldY(i, j), width_0) ! initial estimate for critical depth

      area_c = area_0
      errorY = 100.
       
      do while (errorY > 0.0001)
        area_c = (dsc * dsc * width_0 / grav) ** (1./3.)
        errorY = abs(area_c - area_0)
        
        call r_interpol(areaTable, topwTable, nel, area_c, width_0)
        
        area_0 = area_c
      end do

      call r_interpol(areaTable,elevTable,nel,area_c,y_crit)
      
      if (y_norm .eq. -9999) then
        print*, 'At j = ',j,', i = ',i, 'interpolation of y_norm in calculating normal area was not possible, Q', &
                dsc,'slope',So
!       stop
      end if
    end subroutine normal_crit_y

    real(prec) function LInterpol(x1, y1, x2, y2, x)
      
      implicit none
      
      !-------------------------------------------------------------------------------------
      ! Description:
      !   Estimate y for a given x applying linear interpolation to two given (x, y) points
      !-------------------------------------------------------------------------------------          
      
      ! function arguments
      real(prec), intent(in) :: x1, y1, x2, y2, x
      
      if (abs(x2-x1).lt.0.0001) then
       ! to prevent absurdly small value in the denominator
       LInterpol = 0.5*(y1 + y2)
      else
       LInterpol = (y2 - y1) / (x2 - x1) * (x - x1) + y1
      endif
      
    end function LInterpol

    real(prec) function intp_y(nrow, xarr, yarr, x)
      
      implicit none
      !-------------------------------------------------------------------------------------
      ! Description:
      !   Estimate y for a given x applying linear interpolation to two x and y arrays
      !-------------------------------------------------------------------------------------         

      ! function arguments
      integer,                           intent(in) :: nrow
      real(prec),                  intent(in) :: x      
      real(prec), dimension(nrow), intent(in) :: xarr, yarr
      ! function local variables
      integer          :: irow
      real(prec) :: x1, y1, x2, y2, y

      irow = locate(xarr, x)
      
      if (irow == 0)    irow = 1
      if (irow == nrow) irow = nrow - 1
      
      x1     = xarr(irow)
      y1     = yarr(irow)
      x2     = xarr(irow+1)
      y2     = yarr(irow+1)
      y      = LInterpol(x1, y1, x2, y2, x)
      intp_y = y

    end function intp_y

    integer function locate(xx, x)
      
      implicit none
      
      !------------------------------------------------------------------------------------
      ! Description:               
      !   Locate function in p.1045,Numerical Recipes in Fortran f90
      !      
      ! Method:
      !   klo=max(min(locate(xa,x),n-1),1) In the Fortran 77 version of splint,
      !   there is in-line code to find the location in the table by bisection. Here
      !   we prefer an explicit call to locate, which performs the bisection. On
      !   some massively multiprocessor (MMP) machines, one might substitute a different,
      !   more parallel algorithm (see next note).
      !  Given an array xx(1:N), and given a value x, returns a value j such that x is between
      !  xx(j) and xx(j + 1). xx must be monotonic, either increasing or decreasing.
      !  j = 0 or j = N is returned to indicate that x is out of range.
      !------------------------------------------------------------------------------------
      
      ! function arguments
      real(prec),               intent(in) :: x      
      real(prec), dimension(:), intent(in) :: xx

      ! function local variables
      integer :: n, jl, jm, ju
      logical :: ascnd

      n     = size(xx)
      ascnd = (xx(n) >= xx(1))    ! True if ascending order of table, false otherwise.
      jl    = 0                   ! Initialize lower
      ju    = n + 1               ! and upper limits.
      
      do
        if ((ju - jl) <= 1) exit  ! Repeat until this condition is satisfied.
        
        jm = (ju + jl) / 2        ! Compute a midpoint,
        
        if (ascnd .eqv. (x >= xx(jm))) then
          jl = jm                 ! and replace either the lower limit
        else
          ju = jm                 ! or the upper limit, as appropriate.
        end if
      end do

      if (x == xx(1)) then        ! Then set the output, being careful with the endpoints.
        locate = 1
      else if (x == xx(n)) then
        locate = n - 1
      else
        locate = jl
      end if

    end function locate

end module diffusive
