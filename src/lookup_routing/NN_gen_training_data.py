import numpy as np
import sys

sys.path.append(r"../fortran_routing/mc_pylink_v00/MC_singleSeg_singleTS")
# import mc_sseg_stime_NOLOOP as mc
from mc_sseg_stime import muskingcunge_module as mc
import itertools
import NN_normalization
import nwm_single_segment


def main(
    depthp_min,
    depthp_max,
    qlat_min,
    qlat_max,
    qdp_min,
    qdp_max,
    quc_min,
    quc_max,
    qup_min,
    qup_max,
    s0_min,
    s0_max,
    cs_min,
    cs_max,
    tw_min,
    tw_max,
    bw_min,
    bw_max,
    AL,
):

    AL = [5]

    # output lists if multiple runs were perfermed
    mean_errors_list = []
    max_errors_list = []
    mean_rel_errors_list = []
    max_rel_errors_list = []

    for size in AL:
        # these are the min a max ranges for each input variable. Based on array length specified this will slice these ranges up to be used in our combinations of test data.
        depthp_min = 0.010
        depthp_max = 0.011
        qlat_min = 35
        qlat_max = 45
        qdp_min = 0.01
        qdp_max = 1
        quc_min = 0.01
        quc_max = 1
        qup_min = 0.01
        qup_max = 1
        s0_min = 0.00001
        s0_max = 0.002
        cs_min = 0.085
        cs_max = 2.254
        tw_min = 150
        tw_max = 500
        bw_min = 112
        bw_max = 150

        # nwm_single_segment.singlesegment():
        array_length = size

        dt = 60  # Time step
        dx = 1800  # segment length
        # bw = np.linspace(0.135, 230.035, array_length, endpoint=True) # Trapezoidal bottom width
        bw = np.linspace(
            bw_min, bw_max, array_length, endpoint=True
        )  # Trapezoidal bottom width
        tw = np.linspace(
            tw_min, tw_max, array_length, endpoint=True
        )  # Channel top width (at bankfull)
        # twcc = np.linspace(0.67, 1150.17, array_length, endpoint=True) # Flood plain width
        # twcc = tw*  # Flood plain width tw*3
        n_manning = 0.028  # manning roughness of channel
        n_manning_cc = 0.028  # manning roughness of floodplain
        cs = np.linspace(
            cs_min, cs_max, array_length, endpoint=True
        )  # channel trapezoidal sideslope
        s0 = np.linspace(
            s0_min, s0_max, array_length, endpoint=True
        )  # Lateral inflow in this time step
        qup = np.linspace(
            qup_min, qup_max, array_length, endpoint=True
        )  # Flow from the upstream neighbor in the previous timestep
        # quc = np.linspace(10, 1000, array_length, endpoint=True) # Flow from the upstream neighbor in the current timestep
        quc = np.linspace(
            quc_min, quc_max, array_length, endpoint=True
        )  # Flow from the upstream neighbor in the current timestep
        # qdp = np.linspace(10, 1000, array_length, endpoint=True) # Flow at the current segment in the previous timestep
        qdp = np.linspace(
            qdp_min, qdp_max, array_length, endpoint=True
        )  # Flow at the current segment in the previous timestep
        qlat = np.linspace(
            qlat_min, qlat_max, array_length, endpoint=True
        )  # lat inflow into current segment in the current timestep
        velp = 0.5  # Velocity in the current segment in the previous timestep NOT USED AS AN INPUT!!!
        depthp = np.linspace(depthp_min, depthp_max, array_length, endpoint=True)  # D

        # this is a normalization function that is used to scale the input variables between 0-1 to make the network train more efficiently. If not normalized some values may influence the NN too much
        def normalize(val, max, min, target_max=1, target_min=0):
            return (val - min) / (max - min) * (target_max - target_min) + target_min

        # MC function that allows us to calculate the expected outputs used as our Y output. Can be used to generate real calculations from the MC code.

        # return qdc, vel, depth

        # for loops used to find every combination of our variables possible based on our original array size between the mins and maxs of each variable.
        Y = []
        M = []

        # for qup_i in range(0,(array_length)-1):
        #     for quc_j in range(0,(array_length)-1):
        #         for qlat_k in range(0,(array_length)-1):
        #             for qdp_l in range(0,(array_length)-1):
        #                 for bw_n in range(0,(array_length)-1):
        #                     for tw_o in range(0,(array_length)-1):
        #                         for cs_p in range(0,(array_length)-1):
        #                             for s0_q in range(0,(array_length)-1):
        # for depthp_s in range(0,(array_length)-1):
        temp = list(itertools.product(qup, quc, qlat, qdp, bw, tw, cs, s0, depthp))
        for i in temp:
            M.append(
                [
                    # dt,
                    normalize(i[0], qup_max, qup_min),
                    normalize(i[1], quc_max, quc_min),
                    normalize(i[2], qlat_max, qlat_min),
                    normalize(i[3], qdp_max, qdp_min),
                    # dx,
                    normalize(i[4], bw_max, bw_min),
                    normalize(i[5], tw_max, tw_min),
                    # normalize(tw[tw_o]*3,tw_max,tw_min),
                    # n_manning,
                    # n_manning_cc,
                    normalize(i[6], cs_max, cs_min),
                    normalize(i[7], s0_max, s0_min),
                    # velp,
                    normalize(i[8], depthp_max, depthp_min),
                ]
            )
            # M.append([dt, qup[qup_i], quc[quc_j], qlat[qlat_k],qdp[qdp_l],dx,  bw[bw_n], tw[tw_o], tw[tw_o]*3,n_manning, n_manning_cc, cs[cs_p], s0[s0_q], velp, depthp[depthp_s]])
            # M.append([
            #     # dt,
            #     normalize(qup[qup_i],qup_max,qup_min),
            #     normalize(quc[quc_j],quc_max,quc_min),
            #     normalize(qlat[qlat_k],qlat_max,qlat_min),
            #     normalize(qdp[qdp_l],qdp_max,qdp_min),
            #     # dx,
            #     normalize(bw[bw_n],bw_max,bw_min),
            #     normalize(tw[tw_o],tw_max,tw_min),
            #     # normalize(tw[tw_o]*3,tw_max,tw_min),
            #     # n_manning,
            #     # n_manning_cc,
            #     normalize(cs[cs_p],cs_max,cs_min),
            #     normalize(s0[s0_q], s0_max, s0_min),
            #     # velp,
            #     normalize(depthp[depthp_s],depthp_max,depthp_min)])

            S = nwm_single_segment.singlesegment(
                dt=dt,
                qup=i[0],
                quc=i[1],
                qlat=i[2],
                qdp=i[3],
                dx=dx,
                bw=i[4],
                tw=i[5],
                twcc=i[5] * 3,
                n_manning=n_manning,
                n_manning_cc=n_manning_cc,
                cs=i[6],
                s0=i[7],
                velp=velp,
                depthp=i[8],
            )
            Y.append((S[0]))
            if len(Y) % 100000 == 0:
                print(len(Y))
        # this section just adds a test sample with the expected output of .75 to the end of our data in case we would like to compare it
        dt = 60.0  # diffxxxxx
        dx = 1800.0  # diffxxxxx
        bw = 112.0  # small diffxxxxx
        tw = 448.0  # small diffxxxxx
        twcc = 623.5999755859375  # no differencexxxxx
        n_manning = 0.02800000086426735  # diffxxxxxxx
        n_manning_cc = 0.03136000037193298  # no differencexxxxxxx
        cs = 1.399999976158142  # tiny diffxxxxx
        s0 = 0.0017999999690800905  # big diffxxxxxxxxx
        qlat = 40.0  # diffxxxx
        qup = 0.04598825052380562  # almost 1 to 1 with qucxxxx
        quc = 0.04598825052380562  # xxxxxx
        qdp = 0.21487340331077576  # same as qup qucxxxxx
        velp = 0.070480190217494964  # no differencedepthp = 0.010033470578491688) # large diff
        depthp = 0.010033470578491688

        M.append(
            [
                normalize(qup, qup_max, qup_min),
                normalize(quc, quc_max, quc_min),
                normalize(qlat, qlat_max, qlat_min),
                normalize(qdp, qdp_max, qdp_min),
                # dx,
                normalize(bw, bw_max, bw_min),
                normalize(tw, tw_max, tw_min),
                # normalize(tw[tw_o]*3,tw_max,tw_min),
                # n_manning,
                # n_manning_cc,
                normalize(cs, cs_max, cs_min),
                normalize(s0, s0_max, s0_min),
                # velp,
                normalize(depthp, depthp_max, depthp_min),
            ]
        )
        Y.append(0.7570106983184814)
        Y_max = max(Y)
        Y_min = min(Y)
        for i in range(0, len(Y)):
            Y[i] = NN_normalization.normalize(Y[i], Y_max, Y_min)
        # M = np.array(M)
        # for i in range(0,len(M),1):
        #     S = nwm_single_segment.singlesegment(*M[i])
        #     Y.append(S[0])
        Y = np.array(Y)
        M = np.array(M)

        return (
            M,
            Y,
            mean_errors_list,
            max_errors_list,
            mean_rel_errors_list,
            max_rel_errors_list,
            Y_max,
            Y_min,
        )
