# import required modules
from __future__ import division
import helpers
import constants
from reach import Reach
import sys
import numpy as np
import pandas as pd
from scipy.optimize import fmin
from scipy import stats
from math import ceil

# from plotly.offline import plot
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
import csv
import os


class DummyReach(Reach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_initial_state(self):
        """Compute a dummy initial state"""
        # print(self.upstream_flow_ts)
        # print(self.downstream_stage_ts)
        for section in self.sections:
            self.add_normal_depth_time_step(section, self.upstream_flow_ts[0])

    def compute_next_time_step_state(
        self,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
    ):
        """the Dummy Reach simply copies the current channel state to the next time step
        flow
        """
        for section in self.sections:
            # print(j)
            self.add_normal_depth_time_step(section, upstream_flow_next)


def main():
    input_type = "simple"
    input_vars = {}
    input_vars["n_sections"] = 11
    input_vars["n_timesteps"] = 10
    input_vars["station_downstream"] = 10000
    input_vars["station_upstream"] = 11000
    input_vars["bottom_width_downstream"] = 100
    input_vars["bottom_width_upstream"] = 1000
    input_vars["bottom_z_downstream"] = 0
    input_vars["bottom_z_upstream"] = 100
    input_vars["dx_ds_boundary"] = 1000
    input_vars["S0_ds_boundary"] = 0.0001
    input_vars["manning_n_ds_all"] = 0.035
    input_vars["loss_coeff_all"] = 0.03
    input_vars["hydrograph_steady_time"] = 0
    input_vars["hydrograph_event_width"] = 7
    input_vars["hydrograph_skewness"] = 4
    input_vars["hydrograph_qpeak"] = 5000
    reach = DummyReach(input_type=input_type, input_vars=input_vars)
    # reach = SimpleFlowTrace()
    # reach = SteadyReach()
    # reach = MuskCReach()
    # reach = MESHDReach()

    reach.compute_initial_state()
    reach.compute_time_steps(verbose=True)


if __name__ == "__main__":
    main()
