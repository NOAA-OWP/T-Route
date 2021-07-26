import numpy as np
import mc_sseg_stime_noIC as mc

# Muskingum-Cunge applied to each segment in entire timesteps as it run through multiple segments.
def mc_tlp_over_seg(
    connections=None,
    supernetwork_data=None,
    reach=None,
    last_segment_reach=None,
    nts=None,
    dt_mc=None,
    latflow=None,
    flowdepthvel=None,
    debuglevel=0,
):

    ## Store last segment numbers for upstream reaches for a current head_segment reach
    usrch_list = list(reach["upstream_reaches"])
    if reach["upstream_reaches"] == {0}:
        usrchnb = 0
    else:
        usrchnb = len(reach["upstream_reaches"])
    ## channel data inputs
    seg_list = list(reach["segments_list"])
    seg_list = seg_list[::-1]  # to reversed order
    ncomp = len(reach["segments_list"])
    ## cofficients for artificial lateral flow
    bply = nts / 2.0
    aply = -0.2

    for seg in range(0, ncomp):
        segID = seg_list[seg]
        mc.var.dt = dt_mc
        mc.var.dx = connections[segID]["data"][supernetwork_data["length_col"]]
        mc.var.bw = connections[segID]["data"][supernetwork_data["bottomwidth_col"]]
        mc.var.tw = connections[segID]["data"][supernetwork_data["topwidth_col"]]
        mc.var.twcc = connections[segID]["data"][supernetwork_data["topwidthcc_col"]]
        mc.var.n = connections[segID]["data"][supernetwork_data["manningn_col"]]
        mc.var.ncc = connections[segID]["data"][supernetwork_data["manningncc_col"]]
        mc.var.cs = connections[segID]["data"][supernetwork_data["ChSlp_col"]]
        mc.var.so = connections[segID]["data"][supernetwork_data["slope_col"]]
        mc.var.ncomp = ncomp

        for ts in range(0, nts):
            if latflow == None:
                ## case 1: create artificial lateral flow
                mc.var.ql = (
                    aply * (ts + 1 - bply) ** 2.0
                    - aply * bply ** 2.0
                    + (seg + 1) * 10.0
                )
            else:
                ## case 2: take given lateral flow as argument
                mc.var.ql = latflow[segID]["lateralflow"][ts]

            flowdepthvel[segID]["qlat"][ts] = mc.var.ql
            # initial values of MC parameters when time is zero and beyond
            if ts == 0:
                mc.var.qup = 0.0
                mc.var.quc = 0.0
                mc.var.qdp = 0.0
                mc.var.qdc = 0.0
                mc.var.vel = 0.0
                mc.var.depth = -999.0
            else:
                if seg > 0:
                    mc.var.qup = flowdepthvel[seg_list[seg - 1]]["flow"][ts - 1]
                    mc.var.quc = mc.var.qup
                elif reach["upstream_reaches"] != {0}:
                    ## when it is first segment of head_segment's reach, compute previous flow from
                    ## the last segments of upstream reaches by adding the flows if the reaches are
                    ## more than 1.
                    mc.var.qus_prev = 0.0
                    for usrch in range(0, usrchnb):
                        uslsegID = last_segment_reach[usrch_list[usrch]]
                        mc.var.qus_prev = (
                            mc.var.qus_prev + flowdepthvel[uslsegID]["flow"][ts - 1]
                        )

                    mc.var.qup = mc.var.qus_prev
                    mc.var.quc = mc.var.qup
                elif reach["upstream_reaches"] == {0}:
                    mc.var.qup = 0.0
                    mc.var.quc = 0.0
                mc.var.qdp = flowdepthvel[segID]["flow"][ts - 1]
                mc.var.qdc = 0.0
                mc.var.vel = flowdepthvel[segID]["vel"][ts - 1]
                mc.var.depth = flowdepthvel[segID]["depth"][ts - 1]

            #             print(f"ts {ts} head_segment {head_segment} segINDEX {seg} segID {segID} ql {mc.var.ql} \
            #                   INI:qup quc qdp qdc vel depth {mc.var.qup} {mc.var.quc} {mc.var.qdp} {mc.var.qdc} \
            #                   {mc.var.vel} {mc.var.depth}")

            if debuglevel <= -2:
                """
                rv=[qdc=0.005649629049003124, depthc=0.010014507919549942, velc=0.04874464496970177]
                muskingcunge(60.0,0.0,0.0,0.0,1.5,282.0,1.2110410928726196,2.018401861190796,6.055205345153809,0.05999999865889549,0.11999999731779099,0.8560649752616882,0.008999999612569809,0.0,0.0)
                """
                print(
                    segID,
                    mc.var.dt,
                    mc.var.qup,
                    mc.var.quc,
                    mc.var.qdp,
                    mc.var.ql,
                    mc.var.dx,
                    mc.var.bw,
                    mc.var.tw,
                    mc.var.twcc,
                    mc.var.n,
                    mc.var.ncc,
                    mc.var.cs,
                    mc.var.so,
                    mc.var.vel,
                    mc.var.depth,
                    mc.var.ncomp,
                )

            ## call Fortran routines
            mc.muskingcunge_module.main()
            if debuglevel <= -2:
                print(
                    mc.var.qdc,
                    mc.var.vel,
                    mc.var.depth,
                )

            #             print(f"ts {ts} head_segment {head_segment} segINDEX {seg} segID {segID} ql {mc.var.ql} \
            #             FNL:qup quc qdp qdc vel depth {mc.var.qup} {mc.var.quc} {mc.var.qdp} {mc.var.qdc} \
            #             {mc.var.vel} {mc.var.depth}")

            # store computed values for curent time k
            # output keeping
            flowdepthvel[segID]["flow"][ts] = mc.var.qdc
            flowdepthvel[segID]["vel"][ts] = mc.var.vel
            flowdepthvel[segID]["depth"][ts] = mc.var.depth
    return flowdepthvel
