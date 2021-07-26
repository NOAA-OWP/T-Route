import traceback

debuglevel = 0
COMPILE = True
if COMPILE:
    try:
        import subprocess

        fortran_compile_call = []
        fortran_compile_call.append(r"f2py3")
        fortran_compile_call.append(r"-c")
        fortran_compile_call.append(r"varPrecision.f90")
        fortran_compile_call.append(r"MUSKINGCUNGE.f90")
        fortran_compile_call.append(r"-m")
        fortran_compile_call.append(r"mc_wrf_hydro")

        if debuglevel <= -1:
            print(fortran_compile_call)
        if debuglevel <= -2:
            subprocess.run(fortran_compile_call)
        else:
            subprocess.run(
                fortran_compile_call,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        from mc_wrf_hydro import submuskingcunge_wrf_module

    except Exception as e:
        print(e)
        if debuglevel <= -1:
            traceback.print_exc()
else:
    from mc_wrf_hydro import submuskingcunge_wrf_module

debuglevel = 0
if COMPILE:
    try:
        import subprocess

        fortran_compile_call = []
        fortran_compile_call.append(r"f2py3")
        fortran_compile_call.append(r"-c")
        fortran_compile_call.append(r"varPrecision.f90")
        fortran_compile_call.append(r"MCsingleSegStime_f2py_NOLOOP.f90")
        fortran_compile_call.append(r"-m")
        fortran_compile_call.append(r"mc_sseg_stime")
        # fortran_compile_call.append(r"--opt='-fdefault-real-8'")

        if debuglevel <= -1:
            print(fortran_compile_call)
        if debuglevel <= -2:
            subprocess.run(fortran_compile_call)
        else:
            subprocess.run(
                fortran_compile_call,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        from mc_sseg_stime import muskingcunge_module

        # import mc_sseg_stime as muskingcunge_module
    except Exception as e:
        print(e)
        if debuglevel <= -1:
            traceback.print_exc()
else:
    from mc_sseg_stime import muskingcunge_module

    # import mc_sseg_stime as muskingcunge_module

# # Method 1
# Python: time loop; segment loop; constant channel variables are passed to Fortran
# Fortran: Take constant variable values and then run MC for a single segment


def compute_mc_up2down_ReachbySegment():
    """HANDLE LOOPING,
    Then call single segment routine for each segment"""
    pass


def singlesegment_wrf(
    dt,  # dt
    qup=None,  # qup
    quc=None,  # quc
    qdp=None,  # qdp
    qlat=None,  # ql
    dx=None,  # dx
    bw=None,  # bw
    tw=None,  # tw
    twcc=None,  # twcc
    n_manning=None,  #
    n_manning_cc=None,  # ncc
    cs=None,  # cs
    s0=None,  # s0
    depthp=None,  # depth at previous time step
):

    # call Fortran routine
    return submuskingcunge_wrf_module.submuskingcunge(
        qup,
        quc,
        qdp,
        qlat,
        dt,
        s0,
        dx,
        n_manning,
        cs,
        bw,
        tw,
        twcc,
        n_manning_cc,
        depthp,
    )
    # return qdc, vel, depth


def singlesegment(
    dt,  # dt
    qup=None,  # qup
    quc=None,  # quc
    qdp=None,  # qdp
    qlat=None,  # ql
    dx=None,  # dx
    bw=None,  # bw
    tw=None,  # tw
    twcc=None,  # twcc
    n_manning=None,  #
    n_manning_cc=None,  # ncc
    cs=None,  # cs
    s0=None,  # s0
    velp=None,  # DUMMY -- dropped from the computation
    depthp=None,  # depth at previous time step
):

    # call Fortran routine
    rv = muskingcunge_module.muskingcungenwm(
        dt,
        qup,
        quc,
        qdp,
        qlat,
        dx,
        bw,
        tw,
        twcc,
        n_manning,
        n_manning_cc,
        cs,
        s0,
        0,
        depthp,
    )
    # return qdc, vel, depth

    return rv[:3]


def main():
    """
    No Inputs:
    Uses seceral sets of hard-coded test values to show the behavior
    of the Muskingum Cunge routing calculation module under a low-flow
    and a high-flow condition.
    """
    print(
        "First test low-flow, showing expected result depending on precision of calculation."
    )
    dt = 60.0  # Time step
    dx = 1800.0  # segment length
    bw = 112.0  # Trapezoidal bottom width
    tw = 448.0  # Channel top width (at bankfull)
    twcc = 623.5999755859375  # Flood plain width
    n_manning = 0.02800000086426735  # manning roughness of channel
    n_manning_cc = 0.03136000037193298  # manning roughness of floodplain
    cs = 1.399999976158142  # channel trapezoidal sideslope
    s0 = 0.0017999999690800905  # downstream segment bed slope
    qlat = 40.0  # Lateral inflow in this time step

    # precision = "double"  # the fortran precis module must be edited
    precision = "single"

    if precision is "single":

        """
        single precision results from standard procedure
        ORIG..ORIG
        k,   i,   q,    vel,    depth
        0 0 0.0891760066151619 0.06941037625074387 0.010050795041024685
        0 1 0.06863606721162796 0.07104580104351044 0.009998836554586887
        0 2 0.04598825052380562 0.06900732219219208 0.009981763549149036
        0 3 0.21487340331077576 0.07048019021749496 0.010033470578491688
        1 0 0.3173820674419403 0.15248453617095947 0.03275299444794655
        1 1 0.2014089822769165 0.09322302788496017 0.015030190348625183
        1 2 0.13257014751434326 0.061184611171483994 0.008333344012498856
        1 3 0.7570106983184814 0.12373604625463486 0.02334451675415039
        """
        # Providing the truncated inputs produces the exact same answer...
        qup = 0.04598825
        quc = 0.04598825
        qdp = (
            0.21487340
            # Flow at the current segment in the previous timestep
        )
        depthp = 0.0100334705
        velp = 0.0704801953

        # qup = 0.04598825052380562  # Flow from the upstream neighbor in the previous timestep
        # quc = 0.04598825052380562  # Flow from the upstream neighbor in the current timestep
        # qdp = (
        # 0.21487340331077576
        # # Flow at the current segment in the previous timestep
        # )
        # depthp = 0.010033470578491688  # Depth at the current segment in the previous timestep')
        # velp = 0.07048019021749496  # Velocity in the current segment in the previous timestep NOT USED AS AN INPUT!!!

    elif precision is "double":
        """
        double precision results from standard procedure
        k,   i,   q,    vel,    depth
        0 0 0.08917601071480002 0.06941037462636288 0.010050795648053421
        0 1 0.06863607847374494 0.07104581370597118 0.009998837063033011
        0 2 0.04598825885217007 0.06900733569152935 0.009981763967449955
        0 3 0.21487345087737053 0.07048020184743511 0.010033471026476835
        1 0 0.3173821233250696 0.15248451094165638 0.03275299244073234
        1 1 0.20140900207783252 0.09322304279007165 0.015030191144899728
        1 2 0.13257016086743934 0.061184627249756214 0.00833334535644666
        1 3 0.7570107902354513 0.12373606306742324 0.02334451646521419
        """
        qup = 0.04598825885217007  # Flow from the upstream neighbor in the previous timestep
        quc = 0.04598825885217007  # Flow from the upstream neighbor in the current timestep
        qdp = (
            0.21487345087737053  # Flow at the current segment in the previous timestep
        )
        depthp = 0.010033471026476835  # Depth at the current segment in the previous timestep')
        velp = 0.07048020184743511  # Velocity in the current segment in the previous timestep NOT USED AS AN INPUT!!!

    qdc_expected_sngl = 0.7570106983184814
    velc_expected_sngl = 0.12373604625463486
    depthc_expected_sngl = 0.02334451675415039

    qdc_expected_dbl = 0.7570107902354513
    velc_expected_dbl = 0.12373606306742324
    depthc_expected_dbl = 0.02334451646521419

    # run M-C model
    qdc, velc, depthc = singlesegment(
        # precision=precision,
        dt=dt,
        qup=qup,
        quc=quc,
        qdp=qdp,
        qlat=qlat,
        dx=dx,
        bw=bw,
        tw=tw,
        twcc=twcc,
        n_manning=n_manning,
        n_manning_cc=n_manning_cc,
        cs=cs,
        s0=s0,
        depthp=depthp,
    )
    print(
        "real double precision expected q: {} vel: {} depth: {}".format(
            qdc_expected_dbl, velc_expected_dbl, depthc_expected_dbl
        )
    )
    print(
        "real single precision expected q: {} vel: {} depth: {}".format(
            qdc_expected_sngl, velc_expected_sngl, depthc_expected_sngl
        )
    )

    print(
        "real {} precision computed via updated method q: {} vel: {} depth: {}".format(
            precision, qdc, velc, depthc
        )
    )
    # run M-C model
    qdc, velc, depthc = singlesegment_wrf(
        # precision=precision,
        dt=dt,
        qup=qup,
        quc=quc,
        qdp=qdp,
        qlat=qlat,
        dx=dx,
        bw=bw,
        tw=tw,
        twcc=twcc,
        n_manning=n_manning,
        n_manning_cc=n_manning_cc,
        cs=cs,
        s0=s0,
        depthp=depthp,
    )
    print(
        "real {} precision computed via WRF-Hydro method q: {} vel: {} depth: {}".format(
            precision, qdc, velc, depthc
        )
    )

    print("\nSecond set of inputs activating compound channel")
    dt = 60.0  # Time step
    dx = 1800.0  # segment length
    bw = 112.0  # Trapezoidal bottom width
    tw = 248.0  # Channel top width (at bankfull)
    twcc = 623.5999755859375  # Flood plain width
    n_manning = 0.02800000086426735  # manning roughness of channel
    n_manning_cc = 0.03136000037193298  # manning roughness of floodplain
    cs = 0.42  # channel trapezoidal sideslope
    s0 = 0.007999999690800905  # downstream segment bed slope
    qlat = 40.0  # Lateral inflow in this time step

    qup = 45009
    quc = 50098
    qdp = 50014
    depthp = 30

    # run M-C model
    qdc, velc, depthc = singlesegment(
        # precision=precision,
        dt=dt,
        qup=qup,
        quc=quc,
        qdp=qdp,
        qlat=qlat,
        dx=dx,
        bw=bw,
        tw=tw,
        twcc=twcc,
        n_manning=n_manning,
        n_manning_cc=n_manning_cc,
        cs=cs,
        s0=s0,
        depthp=depthp,
    )

    print(
        "real {} precision computed via updated method q: {} vel: {} depth: {}".format(
            precision, qdc, velc, depthc
        )
    )
    # run M-C model
    qdc, velc, depthc = singlesegment_wrf(
        # precision=precision,
        dt=dt,
        qup=qup,
        quc=quc,
        qdp=qdp,
        qlat=qlat,
        dx=dx,
        bw=bw,
        tw=tw,
        twcc=twcc,
        n_manning=n_manning,
        n_manning_cc=n_manning_cc,
        cs=cs,
        s0=s0,
        depthp=depthp,
    )
    print(
        "real {} precision computed via WRF-Hydro method q: {} vel: {} depth: {}".format(
            precision, qdc, velc, depthc
        )
    )


if __name__ == "__main__":
    main()
