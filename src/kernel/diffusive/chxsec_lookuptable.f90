!-------------------------------------------------------------------------------
module lookuptable
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
  integer          :: nlinks
  integer          :: mxncomp
  integer          :: maxTableLength
  integer          :: nel
  real(prec) :: so_llm
  real(prec) :: z_g, bo_g, traps_g, tw_g, twcc_g, so_g, mann_g, manncc_g
  integer, dimension(:,:),            allocatable :: size_bathy 
  real(prec), dimension(:,:),   allocatable :: z
  real(prec), dimension(:,:,:), allocatable :: x_bathy, z_bathy, mann_bathy 
  real(prec), dimension(:,:,:,:), allocatable :: xsec_tab
  
contains

  subroutine chxsec_lookuptable_calc(mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g,    &
                                mann_ar_g, manncc_ar_g, dx_ar_g, so_lowerlimit_g,  frnw_col, frnw_ar_g, & 
                                mxnbathy_g, x_bathy_g, z_bathy_g, mann_bathy_g, size_bathy_g,           & 
                                nrow_chxsec_lookuptable, chxsec_lookuptable, z_adj) 
    IMPLICIT NONE
          
  !-----------------------------------------------------------------------------
  ! Description:
  !   Hydraulic properties of channel cross sections are computed for either  
  !   trapezoidal-rectangular compound channel shapes or irregular shapes provided by 
  !   topobathy data.
  
  ! Method:
  !   For the synthetic cross sections, the process involves dividing a cross section
  !   into three sub-sections. The hydraulic properties are then computed for each
  !   sub-section individually, and finally, these properties are combined to represent
  !   the overall properties of the entire cross section.   
  !   For irregularly shaped channel cross sections, the process first identifies sub-sections 
  !   that are close to convex shapes. Then, hydraulic properties are computed for each sub-section 
  !   individually by incrementally increasing the water elevation, starting from the bottom of 
  !   the sub-section. Finally, these properties are combined to represent the overall properties 
  !   the entire cross section.
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
    integer, intent(in) :: frnw_col
    integer, intent(in) :: mxnbathy_g
    integer, intent(in) :: nrow_chxsec_lookuptable
    real(prec), intent(in) :: so_lowerlimit_g
    integer, dimension(nrch_g, frnw_col), intent(in) :: frnw_ar_g
    integer, dimension(mxncomp_g, nrch_g), intent(in) :: size_bathy_g   
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: z_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: bo_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: traps_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: tw_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: twcc_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: mann_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: manncc_ar_g
    real(prec), dimension(mxncomp_g,   nrch_g), intent(in) :: dx_ar_g
    real(prec), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in) :: x_bathy_g
    real(prec), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in) :: z_bathy_g
    real(prec), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in) :: mann_bathy_g
    real(prec), dimension(11, nrow_chxsec_lookuptable, mxncomp_g, nrch_g), intent(out) :: chxsec_lookuptable
    real(prec), dimension(mxncomp_g, nrch_g), intent(out) :: z_adj
 
  ! Local variables    
    integer :: ncomp
    integer :: i
    integer :: j
    integer :: k
    integer :: n
    integer :: jm 
    integer :: iel
    integer :: nusrch
    integer :: applyNaturalSection
    integer :: nmstem_rch
    integer :: mxnbathy
    integer, dimension(:), allocatable   :: dmy_frj
    integer, dimension(:),   allocatable :: mstem_frj
    integer, dimension(:,:), allocatable :: frnw_g
    real(prec) :: timesDepth
    !real(prec) :: t   
    real(prec) :: slope
    real(prec) :: convey
    real(prec), dimension(:,:), allocatable :: leftBank
    real(prec), dimension(:,:), allocatable :: rightBank
    real(prec), dimension(:,:), allocatable :: skLeft
    real(prec), dimension(:,:), allocatable :: skMain
    real(prec), dimension(:,:), allocatable :: skRight
    real(prec), dimension(:,:), allocatable :: dx

  !-----------------------------------------------------------------------------
  ! channel network data  
    mxncomp = mxncomp_g       ! maximum number of nodes in a single reach
    nlinks  = nrch_g          ! number of reaches in the network
    so_llm  = so_lowerlimit_g ! lower limit of channel bed slope

  !-----------------------------------------------------------------------------
  ! Some parameters for using natural cross section bathymetry data
    mxnbathy            = mxnbathy_g ! maximum size of bathymetry data points
    !z                  = z_ar_g     ! node elevation array
    applyNaturalSection = 1          ! 0: synthetic channel xsec;  1: topobathy data
    timesDepth          = 4.0 ! water depth multiplier used in readXsection
    nel                 = nrow_chxsec_lookuptable ! number of rows in the hydraulic value lookup tables for ch.xsec   

  !-----------------------------------------------------------------------------
    allocate(dx(mxncomp,nlinks))
    allocate(rightBank(mxncomp, nlinks), leftBank(mxncomp, nlinks))
    allocate(skLeft(mxncomp, nlinks), skMain(mxncomp, nlinks), skRight(mxncomp, nlinks))
    allocate(dmy_frj(nlinks))
    allocate(frnw_g(nlinks,frnw_col))
    allocate(x_bathy(mxnbathy, mxncomp, nlinks), z_bathy(mxnbathy, mxncomp, nlinks))
    allocate(mann_bathy(mxnbathy, mxncomp, nlinks))
    allocate(size_bathy(mxncomp, nlinks))
    allocate(z(mxncomp, nlinks))
    allocate(xsec_tab(11, nel, mxncomp, nlinks))
  
  !-----------------------------------------------------------------------------
  ! channel network mapping matrix  
    frnw_g  = frnw_ar_g      
      
  !-----------------------------------------------------------------------------
  ! Identify mainstem reaches and list their ids in an array
    ! Create a dummy array containing mainstem reaches where diffusive wave is applied.
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
    do jm = 1, nmstem_rch !* mainstem reach only
      j = mstem_frj(jm)
      ncomp = frnw_g(j, 1)
      do i = 1, ncomp-1
        dx(i, j) = dx_ar_g(i, j)
      end do
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
    
    ! pass computed channel xsec lookup table and channel bottom elevation adjusted at compute nodes
    chxsec_lookuptable = xsec_tab
    z_adj = z

    deallocate(frnw_g)
    deallocate (z,  dx) 
    deallocate(xsec_tab, rightBank, leftBank, skLeft, skMain, skRight)
    deallocate(mstem_frj)
    deallocate(x_bathy, z_bathy, mann_bathy, size_bathy)

  end subroutine chxsec_lookuptable_calc
  
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
end module lookuptable
