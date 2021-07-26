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


class SteadyReach(Reach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_initial_state(self):
        """Compute a steady initial state (this uses the same math as the next-
        time-step-state, only we simply assume we are using the first timestep
        of the boundary time-series.)
        """
        # print(self.upstream_flow_ts)
        # print(self.downstream_stage_ts)
        self.add_steady_time_step(
            0, 0, self.sections, self.downstream_stage_ts[0], self.upstream_flow_ts[0]
        )

    def compute_next_time_step_state(
        self,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
    ):
        """the Steady Reach performs the Standard Step method to compute a steady profile
        for each flow and downstream stage in the series.
        """
        self.add_steady_time_step(
            j_current, j_next, self.sections, downstream_stage_next, upstream_flow_next
        )

    def add_steady_time_step(
        self, j_current, j_next, sections, downstream_stage_next, upstream_flow_next
    ):
        # TODO: Ask Nick -- should this be inside the Steady Timestep class?

        for i, section in enumerate(sections):
            """Step through using the standard step method
            See tutorial here: https://www.youtube.com/watch?v=P4MhwS03Kl0
            """
            # print(f'dssn {downstream_stage_next}')
            if i == 0:  # Add known downstream boundary
                section.time_steps.append(
                    self.TimeStep(
                        new_flow=upstream_flow_next, new_depth=downstream_stage_next
                    )
                )
                # print(f'stsd_jcurr {section.time_steps[j_current].depth}')
                # print(f'sdstsd_jcurr {section_ds.time_steps[j_current].depth}')
                # print(f's0tsd_jcurr {sections[0].time_steps[j_current].depth}')
                # stage.append(downstream_stage_next)
                # WS.append(section.bottom_z + stage[0])
                # A.append(section.get_area_depth(stage[0]))
                continue

            section_ds = sections[i - 1]
            # section
            # print(f'jnext {j_next}')
            # print(f'jcurr {j_current}')
            # print(f'stsd_jcurr {section.time_steps[j_current].depth}')
            # print(f'sdstsd_jcurr {section_ds.time_steps[j_current].depth}')
            # print(f's0tsd_jcurr {sections[0].time_steps[j_current].depth}')
            # print(f'stsd_jnext {section.time_steps[j_next].depth}')
            # print(f'sdstsd_jnext {section_ds.time_steps[j_next].depth}')
            # print(f's0tsd_jnext {sections[0].time_steps[j_next].depth}')
            # Use normal depth as an seed estimate convergence solution for standard depth
            y_guess = helpers.y_direct(
                section.bottom_width,
                section.manning_n_ds,
                section.bed_slope_ds,
                upstream_flow_next,
            )
            y_standard = self.y_standard_step(
                j_next, section_ds, section, upstream_flow_next, y_guess
            )

            section.time_steps.append(
                self.TimeStep(new_flow=upstream_flow_next, new_depth=y_standard)
            )

    def y_standard_step(
        self,
        j,
        section_ds,
        section_us,
        Q,
        y_guess=1.0,
        k=constants.MANNING_M,
        gravity=constants.GRAVITY,
    ):
        """function to compute error in depth calculation for a guessed depth compared to a calculated depth for a given flow.
        Uses scipy.optimize fmin"""
        y_ds = section_ds.time_steps[j].depth
        z_us = section_us.bottom_z
        z_ds = section_ds.bottom_z
        n_us = section_us.manning_n_ds
        n_ds = section_ds.manning_n_ds

        loss_coeff = (
            section_us.loss_coeff_ds
        )  # The loss coefficient is a reach property so we use only one of them from the upstream reach
        dx = (
            section_us.dx_ds
        )  # The downstream distance is a reach property so we use only one of them from the upstream reach

        # y_standard = fmin(self.stage_standard_min, y_guess, args=(y_ds, z_us, z_ds, V_us, V_ds, hl_us2ds, gravity), full_output=True, disp=True)
        y_standard = fmin(
            self.stage_standard_min,
            y_guess,
            args=(
                section_us,
                section_ds,
                loss_coeff,
                dx,
                Q,
                y_ds,
                z_us,
                z_ds,
                n_us,
                n_ds,
                gravity,
                k,
            ),
            full_output=True,
            disp=False,
        )

        y_us = float(y_standard[0])

        # print(f'{dx}')
        # print(f'y_us: {y_us}   y_ds: {y_ds}')
        # print(f'z_us: {z_us}   z_ds: {z_ds}')
        # print(f'n_us: {n_us}   n_ds: {n_ds}')
        # print(f'{loss_coeff}')

        Area_us = section_us.get_area_depth(y_us)
        Area_ds = section_ds.get_area_depth(y_ds)
        # print(f'Area_us: {Area_us}   Area_ds: {Area_ds}')
        V_us = Q / Area_us
        V_ds = Q / Area_ds
        # print(f'V_us: {V_us}   V_ds: {V_ds}')
        V_head_us = V_us ** 2.0 / (2 * gravity)
        V_head_ds = V_ds ** 2.0 / (2 * gravity)
        # print(f'V_head_us: {V_head_us}   V_head_ds: {V_head_ds}')
        Pw_us = section_us.get_wetted_perimeter_depth(y_us)
        Pw_ds = section_ds.get_wetted_perimeter_depth(y_ds)
        # print(f'Pw_us: {Pw_us}   Pw_ds: {Pw_ds}')
        Rw_us = Area_us / Pw_us
        Rw_ds = Area_ds / Pw_ds
        # print(f'Rw_us: {Rw_us}   Rw_ds: {Rw_ds}')
        Area_mean = (Area_ds + Area_us) / 2.0
        n_mean = (n_ds + n_us) / 2.0
        Rw_mean = (Rw_ds + Rw_us) / 2.0
        Sf_ds = helpers.Manning_Slope(Q=Q, A=Area_ds, Rw=Rw_ds, n=n_ds, k=k)
        Sf_us = helpers.Manning_Slope(Q=Q, A=Area_us, Rw=Rw_us, n=n_us, k=k)
        Sf_mean = helpers.Manning_Slope(Q=Q, A=Area_mean, Rw=Rw_mean, n=n_mean, k=k)
        # Sf_mean = (Sf_ds + Sf_us) / 2.0
        # print (f'Q {Q}')
        # print(f'Sf_ds = {Sf_ds} Sf_us = {Sf_us} Sf_mean = {Sf_mean}')
        hl_us2ds = Sf_mean * dx + loss_coeff * abs(V_head_us - V_head_ds)
        # print('hl_us2ds = {hl_us2ds}')
        # print('y_us, y_guess, y_ds, z_us, z_ds, Q, V_us, V_ds, hl_us2ds, constants.MANNING_M')
        # print(y_us, y_guess, y_ds, z_us, z_ds, Q, V_us, V_ds, hl_us2ds, constants.MANNING_M)
        WSE_ds = y_ds + z_ds
        WSE_us = y_us + z_us
        E_ds = helpers.Bernoulli_Energy(WSE_ds, V_ds, 0, gravity)
        E_us = helpers.Bernoulli_Energy(WSE_us, V_us, hl_us2ds, gravity)
        # print(f'E_ds: {E_ds}')
        # print(f'E_us: {E_us}')

        return float(y_standard[0])

    def stage_standard_min(
        self,
        y_us,
        section_us,
        section_ds,
        loss_coeff,
        dx,
        Q,
        y_ds,
        z_us,
        z_ds,
        n_us,
        n_ds,
        gravity=constants.GRAVITY,
        k=constants.MANNING_M,
    ):
        """computes the error term comparing the Manning's computed depth with the given Q"""
        Area_us = section_us.get_area_depth(y_us)
        Area_ds = section_ds.get_area_depth(y_ds)
        # print(f'{Area_us} {Area_ds}')
        V_us = Q / Area_us
        V_ds = Q / Area_ds
        # print(f'{V_us} {V_ds}')
        V_head_us = V_us ** 2.0 / (2 * gravity)
        V_head_ds = V_ds ** 2.0 / (2 * gravity)
        # print(f'{V_head_us} {V_head_ds}')
        Pw_us = section_us.get_wetted_perimeter_depth(y_us)
        Pw_ds = section_ds.get_wetted_perimeter_depth(y_ds)
        # print(f'{Pw_us} {Pw_ds}')
        Rw_us = Area_us / Pw_us
        Rw_ds = Area_ds / Pw_ds
        # print(f'{Rw_us} {Rw_ds}')

        # print (f'Q {Q}')

        Area_mean = (Area_ds + Area_us) / 2.0
        n_mean = (n_ds + n_us) / 2.0
        Rw_mean = (Rw_ds + Rw_us) / 2.0
        Sf_ds = helpers.Manning_Slope(Q=Q, A=Area_ds, Rw=Rw_ds, n=n_ds, k=k)
        Sf_us = helpers.Manning_Slope(Q=Q, A=Area_us, Rw=Rw_us, n=n_us, k=k)
        # Sf_mean = helpers.Manning_Slope(Q = Q, A = Area_mean, Rw = Rw_mean, n = n_mean, k = k)
        Sf_mean = (Sf_ds + Sf_us) / 2.0
        # print(f'Sf_ds = {Sf_ds} Sf_us = {Sf_us} Sf_mean = {Sf_mean}')

        hl_us2ds = Sf_mean * dx + loss_coeff * abs(V_head_us - V_head_ds)
        # print('hl_us2ds')
        # print(hl_us2ds)

        # print(y_ds, z_ds)
        # print(y_us, z_us)
        WSE_ds = y_ds + z_ds
        WSE_us = y_us + z_us
        E_ds = helpers.Bernoulli_Energy(WSE_ds, V_ds, 0, gravity)
        E_us = helpers.Bernoulli_Energy(WSE_us, V_us, hl_us2ds, gravity)
        # print(f'E_ds: {E_ds}')
        # print(f'E_us: {E_us}')
        epsilon = np.abs(E_us - E_ds)
        # epsilon = np.abs(helpers.Bernoulli_Energy(WSE_ds, V_ds, 0, gravity) - helpers.Bernoulli_Energy(WSE_us, V_us, hl_us2ds, gravity))
        return epsilon

    class TimeStep(Reach.TimeStep):
        def __init__(self, *args, **kwargs):
            # super(Reach.TimeStep, self).__init__(*args, **kwargs)
            super().__init__(*args, **kwargs)


def main():
    input_type = "simple"
    input_vars = {}
    input_vars["n_sections"] = 11
    input_vars["n_timesteps"] = 22
    input_vars["station_downstream"] = 0
    input_vars["station_upstream"] = 1000000
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
    # reach = DummyReach()
    # reach = SimpleFlowTrace()
    reach = SteadyReach(input_type=input_type, input_vars=input_vars)
    # reach = MuskCReach()
    # reach = MESHDReach()

    # reach.input_and_initialize()
    reach.compute_initial_state()
    reach.compute_time_steps(verbose=True)


if __name__ == "__main__":
    main()
