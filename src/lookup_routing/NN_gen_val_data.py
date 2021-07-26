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
    num_samp_val,
    Y_max,
    Y_min,
):

    dt = 60  # Time step
    dx = 1800  # segment length
    # bw = np.linspace(0.135, 230.035, array_length, endpoint=True) # Trapezoidal bottom width
    bw = np.random.uniform(bw_min, bw_max, num_samp_val)  # Trapezoidal bottom width
    tw = np.random.uniform(
        tw_min, tw_max, num_samp_val
    )  # Channel top width (at bankfull)
    # twcc = np.linspace(0.67, 1150.17, array_length, endpoint=True) # Flood plain width
    # twcc = tw*  # Flood plain width tw*3
    n_manning = 0.028  # manning roughness of channel
    n_manning_cc = 0.028  # manning roughness of floodplain
    cs = np.random.uniform(
        cs_min, cs_max, num_samp_val
    )  # channel trapezoidal sideslope
    s0 = np.random.uniform(
        s0_min, s0_max, num_samp_val
    )  # Lateral inflow in this time step
    qup = np.random.uniform(
        qup_min, qup_max, num_samp_val
    )  # Flow from the upstream neighbor in the previous timestep
    # quc = np.linsprandom.uniformace(10, 1000, array_length, endpoint=True) # Flow from the upstream neighbor in the current timestep
    quc = np.random.uniform(
        quc_min, quc_max, num_samp_val
    )  # Flow from the upstream neighbor in the current timestep
    # qdp = np.linspace(10, 1000, array_length, endpoint=True) # Flow at the current segment in the previous timestep
    qdp = np.random.uniform(
        qdp_min, qdp_max, num_samp_val
    )  # Flow at the current segment in the previous timestep
    qlat = np.random.uniform(
        qlat_min, qlat_max, num_samp_val
    )  # lat inflow into current segment in the current timestep
    velp = 0.5  # Velocity in the current segment in the previous timestep NOT USED AS AN INPUT!!!
    depthp = np.random.uniform(depthp_min, depthp_max, num_samp_val)  # D

    VAL_x = []
    VAL_y = []
    for i in range(num_samp_val):
        VAL_x.append(
            [
                NN_normalization.normalize(qup[i], qup_max, qup_min),
                NN_normalization.normalize(quc[i], quc_max, quc_min),
                NN_normalization.normalize(qlat[i], qlat_max, qlat_min),
                NN_normalization.normalize(qdp[i], qdp_max, qdp_min),
                # dx,
                NN_normalization.normalize(bw[i], bw_max, bw_min),
                NN_normalization.normalize(tw[i], tw_max, tw_min),
                # NN_normalization.normalize(tw[tw_o]*3,tw_max,tw_min),
                # n_manning,
                # n_manning_cc,
                NN_normalization.normalize(cs[i], cs_max, cs_min),
                NN_normalization.normalize(s0[i], s0_max, s0_min),
                # velp,
                NN_normalization.normalize(depthp[i], depthp_max, depthp_min),
            ]
        )
        S = nwm_single_segment.singlesegment(
            dt=dt,
            qup=qup[i],
            quc=quc[i],
            qlat=qlat[i],
            qdp=qdp[i],
            dx=dx,
            bw=bw[i],
            tw=tw[i],
            twcc=tw[i] * 3,
            n_manning=n_manning,
            n_manning_cc=n_manning_cc,
            cs=cs[i],
            s0=s0[i],
            velp=velp,
            depthp=depthp[i],
        )
        VAL_y.append(S[0])

    for i in range(0, len(VAL_y)):
        VAL_y[i] = NN_normalization.normalize(VAL_y[i], Y_max, Y_min)
    VAL_x = np.array(VAL_x)
    VAL_y = np.array(VAL_y)
    return (VAL_x, VAL_y)
