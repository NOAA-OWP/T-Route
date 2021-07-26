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
        fortran_compile_call.append(r"module_levelpool.f90")
        fortran_compile_call.append(r"-m")
        fortran_compile_call.append(r"levelpools")
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
        from levelpools import *

        # import mc_sseg_stime as muskingcunge_module
    except Exception as e:
        print(e)
        if debuglevel <= -1:
            traceback.print_exc()
else:
    from levelpools import *

    # import mc_sseg_stime as muskingcunge_module


def reservoirs_calc(
    dt=None,
    qi0=None,
    qi1=None,
    ql=None,
    ar=None,
    we=None,
    maxh=None,
    wc=None,
    wl=None,
    dl=None,
    oe=None,
    oc=None,
    oa=None,
    h0=None,
):
    """This test function demonstrates a simple connection between
    the module_levelpool routing function and python via f2py
    inputs are described by the fortran header:
     real(prec), intent(IN)    :: qi0     ! inflow at previous timestep (cms)
     real(prec), intent(IN)    :: qi1     ! inflow at current timestep (cms)
     real(prec), intent(IN)    :: ql      ! lateral inflow
     real(prec), intent(IN)    :: ar      ! area of reservoir (km^2)
     real(prec), intent(IN)    :: we      ! bottom of weir elevation
     real(prec), intent(IN)    :: wc      ! weir coeff.
     real(prec), intent(IN)    :: wl      ! weir length (m)
     real(prec), intent(IN)    :: dl      ! dam length(m)
     real(prec), intent(IN)    :: oe      ! orifice elevation
     real(prec), intent(IN)    :: oc      ! orifice coeff.
     real(prec), intent(IN)    :: oa      ! orifice area (m^2)
     real(prec), intent(IN)    :: maxh    ! max depth of reservoir before overtop (m)
     real(prec), intent(IN)    :: H0      ! water elevation height (m)


    outputs are, in order the new water depth and new outflow.
     real(prec), intent(OUT)   :: H1      ! water elevation height (m)
     real(prec), intent(OUT)   :: qo1     ! outflow at current timestep
    """

    # call Fortran routine
    return module_levelpool.levelpool_physics(
        dt,
        qi0,
        qi1,
        ql,
        ar,
        we,
        maxh,
        wc,
        wl,
        dl,
        oe,
        oc,
        oa,
        h0,
    )
    # return hc, qdc


def main():

    # dummy execution of reservoirs_calc()

    dt = 300
    qi0 = 500
    qi1 = 600
    ql = 400
    ar = 22
    we = 20
    maxh = 30
    wc = 1.6
    wl = 20
    dl = 200
    oe = 1
    oc = 0.6
    oa = 0.2
    h0 = 22

    print(
        reservoirs_calc(
            dt,
            qi0,
            qi1,
            ql,
            ar,
            we,
            maxh,
            wc,
            wl,
            dl,
            oe,
            oc,
            oa,
            h0,
        )
    )


if __name__ == "__main__":
    main()
