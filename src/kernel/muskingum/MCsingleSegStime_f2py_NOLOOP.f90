module muskingcunge_module

    use precis
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

contains

subroutine muskingcungenwm(dt, qup, quc, qdp, ql, dx, bw, tw, twcc,&
    n, ncc, cs, s0, velp, depthp, qdc, velc, depthc, ck, cn, X)

    !* exactly follows SUBMUSKINGCUNGE in NWM:
    !* 1) qup and quc for a reach in upstream limit take zero values all the time
    !* 2) initial value of depth of time t of each reach is equal to the value at time t-1
    !* 3) qup as well as quc at time t for a downstream reach in a serial network takes
    !*    exactly the same value qdp at time t (or qdc at time t-1) for the upstream reach

    implicit none

    real(prec), intent(in) :: dt
    real(prec), intent(in) :: qup, quc, qdp, ql
    real(prec), intent(in) :: dx, bw, tw, twcc, n, ncc, cs, s0
    real(prec), intent(in) :: velp
    real(prec), intent(in) :: depthp
    real(prec), intent(out) :: qdc, velc, depthc
    real(prec), intent(out) :: ck, cn, X
    real(prec) :: z, sqrt_s0, sqrt_1pz2
    ! Hoist more loop-invariants out of secant2_h. sqrt_s0/n and
    ! sqrt_s0/ncc save 2 divisions per call; bw_plus_2bfdz and
    ! two_sqrt_1pz2 save another ~3 mul/adds per call. Across ~200
    ! secant2_h calls/kernel * ~11k segments * 1728 ts, this removes
    ! ~10B ops on the Tier A workload.
    real(prec) :: sqrt_s0_over_n, sqrt_s0_over_ncc
    real(prec) :: two_sqrt_1pz2, bw_plus_2bfdz
    real(prec) :: bfd, C1, C2, C3, C4

    !Uncomment next line for old initialization
    !real(prec) :: WPC, AREAC

    integer :: iter
    integer :: maxiter, tries
    real(prec) :: mindepth, aerror, rerror
    real(prec) :: R, twl, h_1, h, h_0, Qj, Qj_0

    ! qdc = 0.0
    ! velc = velp
    ! depthc = depthp

    !* parameters of Secant method
    maxiter  = 100
    mindepth = 0.01_prec

    aerror = 0.01_prec
    rerror = 1.0_prec
    tries = 0

    if(cs .eq. 0.0_prec) then
        z = 1.0_prec
    else
        z = 1.0_prec/cs          !channel side distance (m)
    endif

    if(bw .gt. tw) then   !effectively infinite deep bankful
        bfd = bw/0.00001_prec
    elseif (bw .eq. tw) then
        bfd =  bw/(2.0_prec*z)  !bankfull depth is effectively
    else
        bfd =  (tw - bw)/(2.0_prec*z)  !bankfull depth (m)
    endif

    !print *, bfd
    ! Guard against invalid channel parameters (non-positive or NaN). NaN
    ! inputs would otherwise propagate through sqrt(s0) into NaN velocity,
    ! and `.le. 0.0_prec` alone is false for NaN under IEEE 754, so an
    ! explicit ieee_is_nan check is required. Fail loud rather than
    ! silently producing zeroed or NaN-tainted output.
    if (n  .le. 0.0_prec .or. ieee_is_nan(n)  .or. &
        s0 .le. 0.0_prec .or. ieee_is_nan(s0) .or. &
        z  .le. 0.0_prec .or. ieee_is_nan(z)  .or. &
        bw .le. 0.0_prec .or. ieee_is_nan(bw)) then
        write(*,*) "ERROR: muskingcungenwm received invalid channel parameter " // &
            "(NaN or non-positive). n=", n, " s0=", s0, " z=", z, " bw=", bw
        error stop "muskingcungenwm: invalid channel parameter (NaN or non-positive)"
    end if

    ! Loop-invariant transcendentals, hoisted out of secant2_h (called up
    ! to ~200x per kernel call) so they are computed once, not per call.
    sqrt_s0   = sqrt(s0)
    sqrt_1pz2 = sqrt(1.0_prec + z*z)
    ! Step 9: additional loop-invariants hoisted out of secant2_h's
    ! Ck formula. These are constants across the Secant iteration.
    sqrt_s0_over_n   = sqrt_s0 / n
    sqrt_s0_over_ncc = sqrt_s0 / ncc
    two_sqrt_1pz2    = 2.0_prec * sqrt_1pz2
    bw_plus_2bfdz    = bw + 2.0_prec * bfd * z

    depthc = max(depthp, 0.0_prec)
    h     = (depthc * 1.33_prec) + mindepth !1.50 of  depthc
    h_0   = (depthc * 0.67_prec)            !0.50 of depthc

    if(ql .gt. 0.0_prec .or. qup .gt. 0.0_prec .or. quc .gt. 0.0_prec &
        .or. qdp .gt. 0.0_prec .or. qdc .gt. 0.0_prec) then  !only solve if there's water to flux
110 continue

        !Uncomment next two lines for old initialization
        !WPC = 0.0_prec
        !AREAC = 0.0_prec

        iter = 0

        do while (rerror .gt. 0.01_prec .and. aerror .ge. mindepth .and. iter .le. maxiter)

            !Uncomment next four lines for old initialization
            !call secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx, &
            !    qdp, ql, qup, quc, h_0, 1, WPC, Qj_0, C1, C2, C3, C4)
            !call secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx, &
            !    qdp, ql, qup, quc, h, 2, WPC, Qj, C1, C2, C3, C4)

            !Uncomment next four lines for new initialization
            call secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx, &
                qdp, ql, qup, quc, h_0, 1, Qj_0, C1, C2, C3, C4, X, &
                sqrt_s0, sqrt_1pz2, sqrt_s0_over_n, sqrt_s0_over_ncc, &
                two_sqrt_1pz2, bw_plus_2bfdz)
            call secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx, &
                qdp, ql, qup, quc, h, 2, Qj, C1, C2, C3, C4, X, &
                sqrt_s0, sqrt_1pz2, sqrt_s0_over_n, sqrt_s0_over_ncc, &
                two_sqrt_1pz2, bw_plus_2bfdz)

            if(Qj_0-Qj .ne. 0.0_prec) then
                h_1 = h - ((Qj * (h_0 - h))/(Qj_0 - Qj)) !update h, 3rd estimate

                if(h_1 .lt. 0.0_prec) then
                    h_1 = h
                endif
            else
                h_1 = h
            endif

            if(h .gt. 0.0_prec) then
                rerror = abs((h_1 - h)/h) !relative error is new estimate and 2nd estimate
                aerror = abs(h_1 -h)      !absolute error
            else
                rerror = 0.0_prec
                aerror = 0.9_prec
            endif

            h_0  = max(0.0_prec,h)
            h    = max(0.0_prec,h_1)
            iter = iter + 1
                        !write(41,"(3i5,2x,8f15.4)") k, i, iter, dmy1, Qj_0, dmy2, Qj, h_0, h, rerror, aerror
                        !write(42,*) k, i, iter, dmy1, Qj_0, dmy2, Qj, h_0, h, rerror, aerror
            if( h .lt. mindepth) then  ! exit loop if depth is very small
                goto 111
            endif
        end do !*do while (rerror .gt. 0.01 .and. ....
111    continue

        if(iter .ge. maxiter) then
            tries = tries + 1

            if(tries .le. 4) then  ! expand the search space
                h     =  h * 1.33_prec
                h_0   =  h_0 * 0.67_prec
                maxiter = maxiter + 25 !and increase the number of allowable iterations
                goto 110
            endif
                    !print*, "Musk Cunge WARNING: Failure to converge"
                    !print*, 'RouteLink index:', idx + linkls_s(my_id+1) - 1
                    !print*, "id,err,iters,tries",PC*ncc))/(WP+WPC))) * &
                    !        (AREA+AREAC) * (R**(2./3.)) * sqrt(s0)) idx, rerror, iter, tries
                    !print*, "Ck,X,dt,Km",Ck,X,dt,Km
                    !print*, "s0,dx,h",s0,dx,h
                    !print*, "qup,quc,qdp,ql", qup,quc,qdp,ql
                    !print*, "bfd,bw,tw,twl", bfd,bw,tw,twl
                    !print*, "Qmc,Qmn", (C1*qup)+(C2*quc)+(C3*qdp) + C4,((1/(((WP*n)+(WPC*ncc))/(WP+WPC))) * &
                    !        (AREA+AREAC) * (R**(2./3.)) * sqrt(s0))
        endif

        !*yw added for test
        !*DY and LKR Added to update for channel loss
        if(((C1*qup)+(C2*quc)+(C3*qdp) + C4) .lt. 0.0_prec) then
            if( (C4 .lt. 0.0_prec) .and. (abs(C4) .gt. (C1*qup)+(C2*quc)+(C3*qdp)) )  then ! channel loss greater than water in chan
                qdc = 0.0_prec
                !qdc = -111.1
            else
                qdc = MAX( ( (C1*qup)+(C2*quc) + C4),((C1*qup)+(C3*qdp) + C4) )
                !qdc = -222.2
            endif
        else
            qdc = ((C1*qup)+(C2*quc)+(C3*qdp) + C4) !-- pg 295 Bedient huber
            !write(*,*)"C1", C1, "qup", qup, "C2", C2, "quc", quc, "C3", C3, "qdp", qdp, "C4", C4
            !qdc = -333.3
        endif

        ! Step 9b: the hydraulic_geometry call here was wasted -- only twl is
        ! used by the velocity calc below; its R is immediately overwritten
        ! by the legacy formula. Compute twl inline (one line) and skip the
        ! subroutine call. Bit-identical (twl computation matches exactly).
        twl = bw + 2.0_prec * z * h
        !TODO: The following line allows the system to reproduce the current
        !velocity calculation, however the hydraulic radius provided is not
        !taking into account the flood-plan flow, nor is the velocity
        !accouting for the variation in Manning n.
        R = (h*(bw + twl) / 2.0_prec) / (bw + 2.0_prec*(((twl - bw) / 2.0_prec)**2.0_prec + h**2.0_prec)**0.5_prec)
        velc = (1.0_prec/n) * (R **(2.0_prec/3.0_prec)) * sqrt_s0  !*average velocity in m/s
        depthc = h
    else   !*no flow to route
        qdc = 0.0_prec
        cn = 0.0_prec
        ck = 0.0_prec
        !qdc = -444.4
        velc = 0.0_prec
        depthc = 0.0_prec
    end if !*if(ql .gt. 0.0 .or. ...

    ! *************************************************************
    ! call courant subroutine here
    ! *************************************************************
    call courant(h, bfd, bw, twcc, ncc, s0, n, z, dx, dt, ck, cn)
    !print*, "deep down", depthc

end subroutine muskingcungenwm

!**---------------------------------------------------**!
!*                                                     *!
!*                 SECANT2 SUBROUTINE                  *!
!*                                                     *!
!**---------------------------------------------------**!
!Uncomment this function signature for old initialization
!subroutine secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx, &
!    qdp, ql, qup, quc, h, interval, WPC, Qj, C1, C2, C3, C4)

!Uncomment this function signature for new initialization
subroutine secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx, &
    qdp, ql, qup, quc, h, interval, Qj, C1, C2, C3, C4, X, &
    sqrt_s0, sqrt_1pz2, &
    sqrt_s0_over_n, sqrt_s0_over_ncc, two_sqrt_1pz2, bw_plus_2bfdz)

    implicit none

    real(prec), intent(in) :: z, bw, bfd, twcc, s0, n, ncc
    real(prec), intent(in) :: dt, dx
    real(prec), intent(in) :: qdp, ql, qup, quc
    real(prec), intent(in) :: h
    ! sqrt(s0) and sqrt(1+z*z) precomputed once by the caller -- see
    ! muskingcungenwm; this routine runs up to ~200x per kernel call.
    real(prec), intent(in) :: sqrt_s0, sqrt_1pz2
    ! Step 9 additional hoisted invariants (rs_route-inspired). All
    ! constant across the Secant iteration; pre-computed by the caller.
    real(prec), intent(in) :: sqrt_s0_over_n, sqrt_s0_over_ncc
    real(prec), intent(in) :: two_sqrt_1pz2, bw_plus_2bfdz
    real(prec), intent(out) :: Qj, C1, C2, C3, C4, X
    integer,    intent(in) :: interval

    real(prec) :: twl, AREA, WP, R, r_23, s3
    real(prec) :: Ck, Cn, Km, D
    integer    :: upper_interval, lower_interval

    !Uncomment for old initialization
    !real(prec), intent(out) :: WPC
    !real(prec) :: AREAC
    !Uncomment for new initialization
    real(prec) :: WPC, AREAC

    twl = 0.0_prec
    WP = 0.0_prec

    !Uncomment next line for old initialization
    !AREA = 0.0_prec
    !Uncomment next two lines for new initialization
    WPC = 0.0_prec
    AREA = 0.0_prec
    AREAC = 0.0_prec

    R = 0.0_prec
    Ck = 0.0_prec
    Cn = 0.0_prec

    Km = 0.0_prec
    X = 0.0_prec
    D = 0.0_prec

    !--upper interval -----------
    upper_interval = 1
    !--lower interval -----------
    lower_interval = 2


    call hydraulic_geometry(h, bfd, bw, twcc, z, sqrt_1pz2, &
        twl, R, AREA, AREAC, WP, WPC)

    ! Strength-reduce the hydraulic-radius powers: compute R**(2/3) once,
    ! then R**(5/3) = R**(2/3) * R -- replaces a pow() with a multiply.
    r_23 = R**(2.0_prec/3.0_prec)

    !**kinematic celerity, c
    if( (h .gt. bfd) .and. (twcc .gt. 0.0_prec) .and. (ncc .gt. 0.0_prec) ) then
    !*water outside of defined channel weight the celerity by the contributing area, and
    !*assume that the mannings of the spills is 2x the manning of the channel
        Ck = max(0.0_prec,(sqrt_s0_over_n &
            * ((5.0_prec/3.0_prec)*r_23 &
            - ((2.0_prec/3.0_prec)*(r_23*R) &
            * (two_sqrt_1pz2/bw_plus_2bfdz))) &
            * AREA &
            + (sqrt_s0_over_ncc*(5.0_prec/3.0_prec) &
            * (h-bfd)**(2.0_prec/3.0_prec))*AREAC) &
            / (AREA+AREAC))
    else
        if(h .gt. 0.0_prec) then !avoid divide by zero
            Ck = max(0.0_prec, sqrt_s0_over_n &
                * ((5.0_prec/3.0_prec)*r_23 &
                - ((2.0_prec/3.0_prec)*(r_23*R) &
                * (two_sqrt_1pz2/(bw+2.0_prec*h*z)))))
        else
            Ck = 0.0_prec
        endif
    endif

    !**MC parameter, K
    if(Ck .gt. 0.0_prec) then
        Km = max(dt,dx/Ck)
    else
        Km = dt
    endif

    !**MC parameter, X
    if( (h .gt. bfd) .and. (twcc .gt. 0.0_prec) .and. (ncc .gt. 0.0_prec) .and. (Ck .gt. 0.0_prec) ) then !water outside of defined channel
        !H0
        if (interval .eq. upper_interval) then
            X = min(0.5_prec,max(0.0_prec,0.5_prec*(1.0_prec-(Qj/(2.0_prec*twcc*s0*Ck*dx)))))
        endif
        if (interval .eq. lower_interval) then
        !H
            X = min(0.5_prec,max(0.25_prec,0.5_prec*(1.0_prec-(((C1*qup)+(C2*quc)+(C3*qdp) + C4)/(2.0_prec*twcc*s0*Ck*dx)))))
        endif
    else
        if(Ck .gt. 0.0_prec) then
            !H0
            if (interval .eq. upper_interval) then
                X = min(0.5_prec,max(0.0_prec,0.5_prec*(1.0_prec-(Qj/(2.0_prec*twl*s0*Ck*dx)))))
            endif
            !H
            if (interval .eq. lower_interval) then
                X = min(0.5_prec,max(0.25_prec,0.5_prec*(1.0_prec-(((C1*qup)+(C2*quc)+(C3*qdp) + C4)/(2.0_prec*twl*s0*Ck*dx)))))
            endif
        else
            X = 0.5_prec
        endif
    endif

    !write(45,"(3i5,2x,4f10.3)") gk, gi, idx, h, Ck, Km, X
    D = (Km*(1.0_prec - X) + dt/2.0_prec)              !--seconds
    if(D .eq. 0.0_prec) then
        !print *, "FATAL ERROR: D is 0 in MUSKINGCUNGE", Km, X, dt,D
        !call hydro_stop("In MUSKINGCUNGE() - D is 0.")
    endif

    C1 =  (Km*X + dt/2.0_prec)/D
    C2 =  (dt/2.0_prec - Km*X)/D
    C3 =  (Km*(1.0_prec-X)-dt/2.0_prec)/D
    C4 =  (ql*dt)/D

    ! Step 9: cache the C1/C2/C3-weighted upstream sum used both in the
    ! C4 channel-loss adjustment and in the Qj residual below. Order of
    ! ops is preserved -- pure CSE, bit-exact to the original.
    s3 = (C1*qup) + (C2*quc) + (C3*qdp)

    !H
    if (interval .eq. lower_interval) then
        if( (C4 .lt. 0.0_prec) .and. (abs(C4) .gt. s3))  then
            C4 = -s3
        endif
    endif
    !!Uncomment to show WP/WPC behavior above bankfull
    !if (interval .eq. upper_interval) then
    !    print *,"secant1 --", "WP:", WP, "WPC:", WPC
    !else
    !    print *,"secant2 --", "WP:", WP, "WPC:", WPC
    !endif

    if((WP+WPC) .gt. 0.0_prec) then  !avoid divide by zero
        Qj =  (s3 + C4) - ((1.0_prec/(((WP*n)+(WPC*ncc))/(WP+WPC))) * &
                (AREA+AREAC) * r_23 * sqrt_s0) !f(x)
    else
        Qj = 0.0_prec
    endif

end subroutine secant2_h


!**---------------------------------------------------**!
!*                                                     *!
!*                 COURANT SUBROUTINE                  *!
!*                                                     *!
!**---------------------------------------------------**!
subroutine courant(h, bfd, bw, twcc, ncc, s0, n, z, dx, dt, ck, cn)

    implicit none

    real(prec), intent(in) :: h, bfd, bw, twcc, z
    real(prec), intent(in) :: ncc, s0, n, dx, dt
    real(prec), intent(out) :: ck
    real(prec), intent(out) :: cn
    real(prec) :: h_gt_bf, h_lt_bf, AREA, AREAC, WP, WPC, R, sqrt_1pz2
    real(prec) :: twl !UNUSED -- needed only for hydraulic_geometry call

    sqrt_1pz2 = sqrt(1.0_prec + z*z)

    call hydraulic_geometry(h, bfd, bw, twcc, z, sqrt_1pz2, &
        twl, R, AREA, AREAC, WP, WPC, h_lt_bf, h_gt_bf)

    ck = max(0.0_prec,((sqrt(s0)/n) &
        * ((5.0_prec/3.0_prec)*R**(2.0_prec/3.0_prec) &
        - ((2.0_prec/3.0_prec)*R**(5.0_prec/3.0_prec) &
        * (2.0_prec*sqrt_1pz2/(bw+2.0_prec*h_lt_bf*z)))) &
        * AREA &
        + ((sqrt(s0)/(ncc))*(5.0_prec/3.0_prec) &
        * (h_gt_bf)**(2.0_prec/3.0_prec))*AREAC) &
        / (AREA+AREAC))

    cn = ck * (dt/dx)

end subroutine courant

!**---------------------------------------------------**!
!*                                                     *!
!*           Hydraulic Geometry SUBROUTINE             *!
!*                                                     *!
!**---------------------------------------------------**!
subroutine hydraulic_geometry(h, bfd, bw, twcc, z, sqrt_1pz2, &
    twl, R, AREA, AREAC, WP, WPC, h_lt_bf, h_gt_bf)

    implicit none

    real(prec), intent(in) :: h, bfd, bw, twcc, z
    real(prec), intent(in) :: sqrt_1pz2  ! sqrt(1+z*z), precomputed by caller
    real(prec), intent(out), optional :: twl, R, AREA, AREAC, WP, WPC
    real(prec) :: twl_loc, R_loc, AREA_loc, AREAC_loc, WP_loc, WPC_loc
    real(prec), intent(out), optional :: h_gt_bf, h_lt_bf
    real(prec) :: h_gt_bf_loc, h_lt_bf_loc

    twl_loc = bw + 2.0_prec*z*h

    ! Step 9 (rs_route-inspired): split into in-channel vs floodplain
    ! paths. The in-channel case (h <= bfd, or h > bfd with twcc <= 0,
    ! which the NWM 3.0 exception treats as in-channel) is the common
    ! one in practice -- most channel cells run below bankfull. The
    ! else-branch below skips the AREAC/WPC/h_gt_bf computations that
    ! would always be 0 anyway. Bit-identical to the original branchy
    ! form for both cases.
    if (h .le. bfd .or. twcc .le. 0.0_prec) then
        ! Water entirely within the channel cross-section.
        h_lt_bf_loc = h
        h_gt_bf_loc = 0.0_prec
        AREA_loc    = (bw + h * z) * h
        WP_loc      = bw + 2.0_prec * h * sqrt_1pz2
        AREAC_loc   = 0.0_prec
        WPC_loc     = 0.0_prec
        R_loc       = AREA_loc / WP_loc
    else
        ! Floodplain (compound channel) case.
        h_gt_bf_loc = h - bfd
        h_lt_bf_loc = bfd
        AREA_loc    = (bw + bfd * z) * bfd
        WP_loc      = bw + 2.0_prec * bfd * sqrt_1pz2
        AREAC_loc   = twcc * h_gt_bf_loc
        WPC_loc     = twcc + 2.0_prec * h_gt_bf_loc
        R_loc       = (AREA_loc + AREAC_loc) / (WP_loc + WPC_loc)
    endif
    !R = (h*(bw + twl) / 2.0_prec) / (bw + 2.0_prec*(((twl - bw) / 2.0_prec)**2.0_prec + h**2.0_prec)**0.5_prec)
    if (present(twl)) then
        twl = twl_loc
    endif
    if (present(R)) then
        R = R_loc
    endif
    if (present(AREA)) then
        AREA = AREA_loc
    endif
    if (present(AREAC)) then
        AREAC = AREAC_loc
    endif
    if (present(WP)) then
        WP = WP_loc
    endif
    if (present(WPC)) then
        WPC = WPC_loc
    endif
    if (present(h_gt_bf)) then
        h_gt_bf = h_gt_bf_loc
    endif
    if (present(h_lt_bf)) then
        h_lt_bf = h_lt_bf_loc
    endif

end subroutine hydraulic_geometry


end module muskingcunge_module
