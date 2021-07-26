# import required modules
from __future__ import division
import constants
import numpy as np
from scipy.optimize import fmin
from scipy import stats


def Bernoulli_Energy(WSE, V, hl=0, gravity=constants.GRAVITY):
    """compute flow using Manning's equation"""
    return (WSE) + (V ** 2.0) / (2 * gravity) - hl


def y_direct(B, n, S0, Q, k=constants.MANNING_M):
    """function to compute error in depth calculation for a guessed depth compared to a calculated depth for a given flow.
    Uses scipy.optimize fmin"""
    y_optimum = fmin(
        flow_min, Q / B / 3, args=(n, S0, Q, B, k), full_output=True, disp=False
    )
    return float(y_optimum[0])


def flow_min(y, n, S0, Q, B, k=constants.MANNING_M):
    """computes the error term comparing the Manning's computed depth with the given Q"""
    epsilon = np.abs(Manning_Q(y, n, S0, B, k) - Q)
    return epsilon


def Manning_Slope(n, Q, A, Rw, k=constants.MANNING_M):
    # print(f'n * Q / (k * A * (Rw ** (2/3)))) ** 2.0 {n} * {Q} / ({k} * {A} * ({Rw} ** (2/3)))) ** 2.0')
    return (n * Q / (k * A * (Rw ** (2.0 / 3.0)))) ** 2.0


def Manning_Q(y, n, S0, B, k=constants.MANNING_M):
    """compute flow using Manning's equation"""
    return (k / n) * (S0 ** 0.5) * ((B * y) ** (5 / 3)) / ((B + y) ** (2 / 3))


# Generate_Hydrograph from Maryam Asgari Lamjiri and Kelly Flint
# NWC Summer institute coordinators 2019
def Generate_Hydrograph(Num_TimeStep, Steady_time, width, Skewness, QMax):
    # Steady_time += 359
    XX = np.linspace(-5, 5, Num_TimeStep)
    Steady_t = -5 + Steady_time / Num_TimeStep * 10
    t = (XX - Steady_t) / width
    P = 2.0 / width * stats.norm.pdf(t) * stats.norm.cdf(Skewness * t)
    Q = (QMax - 100) * P / np.max(P) + 100
    # plt.plot(np.arange(0,Num_TimeStep),Q)
    return Q


def main():
    pass


if __name__ == "__main__":
    main()
