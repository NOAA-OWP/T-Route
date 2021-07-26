# import required modules
from __future__ import division
import helpers
import constants
import meshconstants
import pandas as pd
from reach import Reach
import csv
import os

"""This python module implements  (and to a degree, extends) the
   MESH code first developed by Dr. Ehab Meselhe.
   A number of helpful references are available to explain the
   derivation of the method and may be requested by contacting
   the authors or Dr. Meselhe.
   We gratefully acknowlege his contribution to this work."""


class MESHpyReach(Reach):
    """USE Global Declarations here to manage these values"""

    # TODO Determine why these values do not persist when declared in __init__
    debug = False
    dx_tolerance = meshconstants.DX_TOLERANCE
    depth_tolerance = meshconstants.DEPTH_TOLERANCE
    celerity_epsilon = meshconstants.CELERITY_EPSILON
    area_tolerance = meshconstants.AREA_TOLERANCE
    crmax = 0.0
    crmin = 100.0
    yy = 0.0
    qq = 0.0
    phi = meshconstants.PHI  # source term treatment (0:explicit, 1:implicit)
    theta = meshconstants.THETA  # ?
    # thetas is the weighting coefficient for the source term
    # (to determine if we want to compute the source term in an
    # implicit or an explicit manner.)
    thetas = meshconstants.THETAS  # ?
    thesinv = meshconstants.THESINV  # ?
    # TODO: THIS THESINV is probably extranneous and should be removed
    alfa2 = meshconstants.ALFA2  # emp parameter for artificial diffusion (lit)
    alfa4 = meshconstants.ALFA4  # maximum value for artificial diffusion

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def input_and_initialize_meshpyfile(self, filetype=None, input_path=None):
        with open(input_path, newline="") as f:

            # Put the first chunk of each line into a lsit
            data = list(map(lambda x: x.split(" ")[0], f.read().split("\n")))
            # print(data)

            # TODO: Get rid of this kludge to cast the input numbers into the right datatype
            for i, item in enumerate(data[0:23]):
                data[i] = float(item)

            data[3] = int(data[3])
            data[4] = int(data[4])

            # Assign all the input values into the variables
            (
                dtini,
                dxini,
                tfin,
                n_sections,
                ntim,
                self.phi,
                self.theta,
                self.thetas,
                self.thetasinv,
                self.alfa2,
                self.alfa4,
                f,
                skk,
                self.yy,
                self.qq,
                cfl,
                time_step_optimization,
                yw,
                bw,
                w,
                option,
                yn,
                qn,
                igate,
                bed_elevation_path,
                upstream_path,
                downstream_path,
                channel_width_path,
                output_path,
                option_dsbc,
                null,
            ) = data

            # print(data)

        self.I_UPSTREAM = 0
        self.I_DOWNSTREAM = n_sections - 1
        self.manning_m = f

        # print(f'yy {self.yy}')

        root = os.path.dirname(input_path)
        # print(root)
        # Read in bed elevation and bottom width input
        with open(os.path.join(root, bed_elevation_path), newline="") as file1:
            with open(os.path.join(root, channel_width_path), newline="") as file2:
                read_data1 = list(csv.reader(file1, delimiter=" "))
                read_data2 = list(csv.reader(file2, delimiter=" "))
                for i in range(
                    n_sections
                ):  # use usual convention of i as spatial dimension
                    z = float(read_data1[i][1])
                    b0 = float(read_data2[i][1])
                    # print(f'i: {i} -- b0 {b0}, z {z}')
                    self.sections.append(
                        self.RectangleSection(
                            station=i,
                            bottom_z=z,
                            bottom_width=b0,
                            dx_ds=dxini,
                            manning_n_ds=1 / skk,
                        )
                    )  # dx is a static value in the test cases
                us_section = None
                for section in self.sections:
                    if us_section:
                        us_section.ds_section = section
                    section.us_section = us_section
                    us_section = section
        # Read hydrograph input Upstream and Downstream
        with open(os.path.join(root, upstream_path), newline="") as file3:
            with open(os.path.join(root, downstream_path), newline="") as file4:
                read_data3 = list(csv.reader(file3, delimiter=" "))
                read_data4 = list(csv.reader(file4, delimiter=" "))
                for j in range(ntim):  # use usual convention of j as time dimension
                    self.upstream_flow_ts.append(float(read_data3[j][1]))
                    self.downstream_stage_ts.append(float(read_data4[j][1]))
                    self.time_list.append(dtini * j)

    def compute_initial_state(
        self, verbose=False, write_output=False, output_path=None
    ):
        """Initial state computed in initialization function."""
        q_init = self.qq
        y_init = self.yy
        j_init = 0
        t_init = self.time_list[j_init]
        for i, section in enumerate(self.sections):
            # print(f'yy, qq {self.yy} {self.qq}')
            if i == self.I_UPSTREAM:
                q_init = self.upstream_flow_ts[j_init]
            elif i == self.I_DOWNSTREAM:
                y_init = self.downstream_stage_ts[j_init]
            section.time_steps.append(
                self.TimeStep(
                    new_time=t_init,
                    new_flow=q_init,
                    new_depth=y_init,
                    new_water_z=section.bottom_z + y_init,
                    new_area=section.get_area_depth(y_init),
                )
            )
            if section.ds_section:
                section.bed_slope_ds = (
                    section.bottom_z - section.ds_section.bottom_z
                ) / section.dx_ds
            else:
                section.bed_slope_ds = 0.00001
        # print(self.upstream_flow_ts)
        # print(self.downstream_stage_ts)

    def compute_time_steps_mesh(
        self, verbose=False, write_output=False, output_path=None
    ):
        """This function can operate with
        1) Nt and dt (number of time steps and size of time step) and a pointer to boundary information
        2) List of times and a pointer to boundary information
        3) an initial time, a list of time deltas, and a corresponding list of boundary conditions
         but since they really all boil down to the last situation, we'll just
         make it work for #3 and then have other translator methods that create these."""
        # print (output_path)

        # print(self.time_list)
        for j, t in enumerate(self.time_list):
            if verbose:
                print(f"timestep {j} of {len(self.time_list)}, t = {t}")
            if j + 1 < len(self.time_list):
                j_current = j
                j_next = j + 1
                self.compute_next_time_step_state(
                    j_current=j_current,
                    j_next=j_next,
                    upstream_flow_current=self.upstream_flow_ts[j_current],
                    upstream_flow_next=self.upstream_flow_ts[j_next],
                    downstream_stage_current=self.downstream_stage_ts[j_current],
                    downstream_stage_next=self.downstream_stage_ts[j_next],
                )
                if write_output:
                    pass
                    self.write_state_timestep_mesh(
                        self.sections, j_current, "mesh_", output_path, self.debug
                    )
            else:
                if write_output:
                    self.write_state_timestep_mesh(
                        self.sections, j_next, "mesh_", output_path, self.debug
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

        # There is the possibility that the Theta for the Predictor and
        # Corrector steps could be numerically different -- but we don't need to
        # take advantage of that here, so theta and theta_s are set to be
        # equal.

        # print(f'yy {self.yy}')
        self.compute_sections(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
            predictor_step=True,
        )
        # TODO: Determine the need for thes AND thetas
        # print(f'self.thes; self.thetas: {self.thes} {self.thetas}')
        # self.thes = self.thetas
        self.matrix_pc(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
            predictor_step=True,
        )
        self.compute_predictor(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
        )
        self.thetas = self.thetasinv
        output_path = r"../test/output/MESHpyReach_out.txt"
        if self.debug:
            self.write_state_timestep_mesh(
                self.sections, j_current, "mesh_predictor", output_path
            )
        self.compute_sections(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
            predictor_step=False,
        )
        self.matrix_pc(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
            predictor_step=False,
        )
        self.compute_corrector(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
        )
        self.mesh_final_update(
            section_arr=self.sections,
            j_current=j_current,
            j_next=j_next,
            upstream_flow_current=upstream_flow_current,
            upstream_flow_next=upstream_flow_next,
            downstream_stage_current=downstream_stage_current,
            downstream_stage_next=downstream_stage_next,
        )

        if self.debug:
            self.write_state_timestep_mesh(
                self.sections, j_current, "mesh_corrector", output_path
            )

    def dsbc():
        # c----------------------------------------------------------------------
        # c     Downstream boundary condition
        #       subroutine dsbc(n)
        # c----------------------------------------------------------------------
        pass

    def compute_sections(
        self,
        section_arr,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
        predictor_step=False,
    ):

        for i, section in enumerate(section_arr):
            section_j = section.time_steps[j_current]
            if (
                predictor_step
            ):  # If we are on the second step, applying the predictors, use the areap and qp
                if i == self.I_DOWNSTREAM:
                    # TODO: Make sure we are handling the DS Boundary correctly
                    section_j.depth = downstream_stage_current
                self.debug = False
                if self.debug and i == self.I_DOWNSTREAM:
                    print(f"predictor step {predictor_step} i {i}")
                    print(f"depth {section_j.depth} depthp {section_j.depthp}")
                self.debug = False
                area = section_j.flow_area
                flow = section_j.flow
                depth = section_j.depth
            elif not predictor_step:
                self.debug = False
                if self.debug and i == self.I_DOWNSTREAM:
                    print(f"corrector step i {i}")
                    print(f"depth {section_j.depth} depthp {section_j.depthp}")
                self.debug = False
                area = section_j.areap  # computed via 'compute_predictor' method
                flow = section_j.qp  # computed via 'compute_predictor' method
                depth = section_j.depthp

            if self.debug:
                print(f"flow_area {section_j.flow_area}")
                print(f"i, z: {i} {section_j.water_z}")
                print(f"depth {section_j.depth}")
                print(f"depthp {section_j.depthp}")
                print(f"yy {self.yy}")
                print(f"bw {section.bottom_width}")
                print(f"current stage {downstream_stage_current}")
                print(f"next stage {downstream_stage_next}")
            section_j.ci1 = section.get_ci1_depth(depth)
            section_j.wetted_perimeter = section.get_wetted_perimeter_depth(depth)
            section_j.hy = area / section.get_wetted_perimeter_depth(depth)
            # print(f'manning_n_ds {section.manning_n_ds}')
            section_j.conveyance_ds = (
                1 / section.manning_n_ds * area * section_j.hy ** (2.0 / 3.0)
            )
            self.debug = False
            if self.debug:
                if i == self.I_UPSTREAM or i == self.I_DOWNSTREAM:
                    print(
                        f"ci1 {section_j.ci1: 9g} Rw {section_j.hy: 9g} conveyance {section_j.conveyance_ds: 9g}"
                    )
                self.debug = False
            if predictor_step:
                if i == self.I_UPSTREAM:
                    pass
                else:
                    section_US = section.us_section
                    section_US_j = section_US.time_steps[j_current]
                    # TODO: USE DBDX in Ci2 computation
                    section_j.dbdx_ds = (
                        section_US.get_dbdx_ds_depth(section_US_j.depth, depth)
                        / section.dx_ds
                    )
                    section_j.ci2_ds = section_US.get_ci2_depth_depth_ds(
                        section_US_j.depth, depth
                    )
                    if self.debug:
                        print(
                            "Upstream n, flow, conveyance_ds {} {} {}",
                            section_US.manning_n_ds,
                            section_US_j.flow,
                            section_US_j.conveyance_ds,
                        )
                        print(
                            "This section n, flow, conveyance_ds {} {} {}",
                            section.manning_n_ds,
                            section_j.flow,
                            section_j.conveyance_ds,
                        )
                    # TODO: Determine if we need to make the 0.5 to compute fs a variable
                    # TODO: Move manning_m into conveyance calculation
                    # TODO: consider suggestion to combine Manning N and Manning M
                    section_j.friction_slope_ds = (
                        0.5
                        * self.manning_m
                        * (
                            section_US_j.flow
                            * abs(section_US_j.flow)
                            / (section_US_j.conveyance_ds ** 2.0)
                            + section_j.flow
                            * abs(section_j.flow)
                            / (section_j.conveyance_ds ** 2.0)
                        )
                    )
                    section_j.as0_ds = (
                        (section_US_j.flow_area + section_j.flow_area)
                        / 2.0
                        * (section_US.bed_slope_ds - section_j.friction_slope_ds)
                    )
                    section_j.gs0_ds = self.gravity * (
                        section_US.bed_slope_ds - section_j.friction_slope_ds
                    )
            if not predictor_step:
                if i == self.I_DOWNSTREAM:
                    pass
                else:
                    section_DS = section.ds_section
                    section_DS_j = section_DS.time_steps[j_current]
                    section_j.dbdx_ds = (
                        section.get_dbdx_ds_depth(depth, section_DS_j.depthp)
                        / section.dx_ds
                    )
                    section_j.ci2_ds = section.get_ci2_depth_depth_ds(
                        depth, section_DS_j.depthp
                    )
                    # TODO: Verify that depthp is calculated in time.
                    self.debug = False
                    if self.debug:
                        if i == self.I_DOWNSTREAM or i == self.I_UPSTREAM:
                            # print ('depthp{: 9g} depthp_DS{: 9g} {: 9g}'\
                            #      .format(section_j.ci2_ds, section_j.dbdx_ds
                            #      , section_j.friction_slope_ds))
                            self.debug = False
                    section_j.friction_slope_ds = (
                        0.5
                        * self.manning_m
                        * (
                            flow * abs(flow) / (section_j.conveyance_ds ** 2.0)
                            + section_DS_j.qp
                            * abs(section_DS_j.qp)
                            / (section_DS_j.conveyance_ds ** 2.0)
                        )
                    )
                    section_j.as0_ds = (
                        (section_j.areap + section_DS_j.areap)
                        / 2.0
                        * (section.bed_slope_ds - section_j.friction_slope_ds)
                    )
                    section_j.gs0_ds = self.gravity * (
                        section.bed_slope_ds - section_j.friction_slope_ds
                    )
                self.debug = False
                if self.debug:
                    if i == self.I_DOWNSTREAM or i == self.I_UPSTREAM:
                        print(
                            "ci2 {: 9g} dbdx {: 9g} fs{: 9g}".format(
                                section_j.ci2_ds,
                                section_j.dbdx_ds,
                                section_j.friction_slope_ds,
                            )
                        )
                    self.debug = False

    def matrix_pc(
        self,
        section_arr,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
        predictor_step=False,
    ):

        for i, section in enumerate(section_arr):
            section_j = section.time_steps[j_current]
            if predictor_step:
                area = section_j.flow_area
                flow = section_j.flow
            else:
                # If we are on the second step, applying the predictors,
                # use the areap and qp
                area = section_j.areap
                flow = section_j.qp
            # print(f'area, flow {area}, {flow}')
            section_j.velocity = flow / area
            section_j.celerity = section.get_celerity_area(
                area, self.gravity, debug=self.debug
            )
            # print(section_j.celerity)
            if section_j.velocity == section_j.celerity:
                section_j.celerity = section_j.celerity + self.celerity_epsilon
            # print(section_j.celerity_epsilon)

            # c     This is the matrix L (left eigenvector matrix - eq 13)
            e11 = 1.0
            e12 = -1.0 / (section_j.velocity - section_j.celerity)
            e21 = 1.0
            e22 = -1.0 / (section_j.velocity + section_j.celerity)
            if self.debug:
                section_j.e11 = e11
                section_j.e12 = e12
                section_j.e21 = e21
                section_j.e22 = e22

            # c       L^{-1} (inverse of Left eigenvector matrix)
            f11 = -(section_j.velocity - section_j.celerity) / (
                2.0 * section_j.celerity
            )
            f12 = (section_j.velocity + section_j.celerity) / (2.0 * section_j.celerity)
            f21 = -(section_j.velocity ** 2.0 - section_j.celerity ** 2.0) / (
                2.0 * section_j.celerity
            )
            f22 = (section_j.velocity ** 2.0 - section_j.celerity ** 2.0) / (
                2.0 * section_j.celerity
            )
            if self.debug:
                section_j.f11 = f11
                section_j.f12 = f12
                section_j.f21 = f21
                section_j.f22 = f22

            # c       Diagonal wave matrix D (eq 12)
            section_j.d11 = abs(section_j.velocity + section_j.celerity)
            section_j.d22 = abs(section_j.velocity - section_j.celerity)

            # c       Equation 11 (L^{-1} D L)
            a11 = e11 * f11 * section_j.d11 + e21 * f12 * section_j.d22
            a12 = e12 * f11 * section_j.d11 + e22 * f12 * section_j.d22
            a21 = e11 * f21 * section_j.d11 + e21 * f22 * section_j.d22
            a22 = e12 * f21 * section_j.d11 + e22 * f22 * section_j.d22
            if self.debug:
                section_j.a11 = a11
                section_j.a12 = a12
                section_j.a21 = a21
                section_j.a22 = a22

            dt = self.time_list[j_next] - self.time_list[j_current]
            # c     Calculating dK/dA (eq 15)
            dkda = section.get_dkda_area(area)
            if self.debug:
                section_j.dkda = dkda
            # print(dkda)

            # c     Matrix S (eq 14)
            st11 = 0.0
            st12 = 0.0
            if not section_j.gs0_ds:
                # TODO: FIND THIS
                # It is possible that the entire MatrixP loop should be shifted right.
                section_j.gs0_ds = 0
                section_j.dbdx_ds = 0
            st21 = section.get_st21_area(
                area,
                flow,
                section_j.gs0_ds,
                section_j.conveyance_ds,
                dkda,
                section_j.dbdx_ds,
                self.gravity,
                self.manning_m,
            )
            # TODO: Determine if st22 is a section-property-dependent value
            # and if so, move it to a method within the RectangleSection class
            st22 = -2.0 * self.manning_m * area * flow / section_j.conveyance_ds ** 2.0

            if self.debug:
                section_j.st11 = st11
                section_j.st12 = st12
                section_j.st21 = st21
                section_j.st22 = st22

            section_US = section.us_section
            if predictor_step:
                # TODO: Confirm the order of these statements with Ehab
                # The following if statements run in reverse order in matrixp
                # vs. matrixc
                # TODO: combine the two inner statements if order is important
                # TODO: Combine all three statments if order is not important
                thetassign = -1.0
                if section.dx_ds < self.dx_tolerance:
                    section_j.sigma_ds = dt
                else:
                    section_j.sigma_ds = dt / section.dx_ds
                    self.crmax = max(
                        self.crmax,
                        section_j.sigma_ds * max(section_j.d11, section_j.d22),
                    )
                    self.crmin = min(
                        self.crmin,
                        section_j.sigma_ds * max(section_j.d11, section_j.d22),
                    )

                # c     LHS of eq 7
                # print(self.phi, self.theta, section_j.sigma_ds, thetassign, dt)
                # print(f'{a11: 9f}, {a12: 9f}, {a21: 9f}, {a22: 9f}, {st11: 9f}, {st12: 9f}, {st21: 9f}, {st22: 9f}')
                section_j.b11 = (
                    0.5
                    - self.phi
                    - self.theta * section_j.sigma_ds * a11
                    + thetassign * 0.5 * self.thetas * st11 * dt
                )
                section_j.b12 = (
                    -self.theta * section_j.sigma_ds * a12
                    + thetassign * 0.5 * self.thetas * st12 * dt
                )
                section_j.b21 = (
                    -self.theta * section_j.sigma_ds * a21
                    + thetassign * 0.5 * self.thetas * st21 * dt
                )
                section_j.b22 = (
                    0.5
                    - self.phi
                    - self.theta * section_j.sigma_ds * a22
                    + thetassign * 0.5 * self.thetas * st22 * dt
                )

                if i == self.I_UPSTREAM:
                    section_j.sigma_ds = dt
                elif section_US.dx_ds < self.dx_tolerance:
                    section_j.sigma_ds = dt
                else:
                    section_j.sigma_ds = dt / section_US.dx_ds

            elif not predictor_step:
                thetassign = 1.0
                # TODO: Confirm that these non-parallel if statements are congruent
                if i == self.I_UPSTREAM:
                    section_j.sigma_ds = dt
                elif section.dx_ds < self.dx_tolerance:
                    section_j.sigma_ds = dt
                else:
                    section_j.sigma_ds = dt / section_US.dx_ds

                # c     LHS of eq 7
                section_j.b11 = (
                    0.5
                    - self.phi
                    - self.theta * section_j.sigma_ds * a11
                    + thetassign * 0.5 * self.thetas * st11 * dt
                )
                section_j.b12 = (
                    -self.theta * section_j.sigma_ds * a12
                    + thetassign * 0.5 * self.thetas * st12 * dt
                )
                section_j.b21 = (
                    -self.theta * section_j.sigma_ds * a21
                    + thetassign * 0.5 * self.thetas * st21 * dt
                )
                section_j.b22 = (
                    0.5
                    - self.phi
                    - self.theta * section_j.sigma_ds * a22
                    + thetassign * 0.5 * self.thetas * st22 * dt
                )

                if section.dx_ds == 0.0:
                    section_j.sigma_ds = dt
                else:
                    section_j.sigma_ds = dt / section.dx_ds
                    self.crmax = max(
                        self.crmax,
                        section_j.sigma_ds * max(section_j.d11, section_j.d22),
                    )
                    self.crmin = min(
                        self.crmin,
                        section_j.sigma_ds * max(section_j.d11, section_j.d22),
                    )

            # TODO: activate this code and delete the g-matrix lines following
            # to bring consistency with new version of code.
            # if predictor_step:
            #     thetassign = -1.0
            # elif not predictor_step: # This reads more clearly that the equivalent simple 'else'
            #     thetassign = 1.0
            # g11 = (0.5 + self.phi + self.theta * section_j.sigma_ds * a11
            #               + thetassign * 0.5 * self.thetas * st11 * dt)
            # g12 = (self.theta * section_j.sigma_ds * a12 + thetassign * 0.5
            #               * self.thetas * st12 * dt)
            # g21 = (self.theta * section_j.sigma_ds * a21 + thetassign * 0.5
            #               * self.thetas * st21 * dt)
            # g22 = (0.5 + self.phi + self.theta * section_j.sigma_ds * a22
            #               + thetassign * 0.5 * self.thetas * st22 * dt)
            if predictor_step:
                thetassign = -1.0
                g11 = (
                    0.5
                    + self.phi
                    + self.theta * section_j.sigma_ds * a11
                    + thetassign * 0.5 * self.thetas * st11 * dt
                )
                g12 = (
                    self.theta * section_j.sigma_ds * a12
                    + thetassign * 0.5 * self.thetas * st12 * dt
                )
                g21 = (
                    self.theta * section_j.sigma_ds * a21
                    + thetassign * 0.5 * self.thetas * st21 * dt
                )
                g22 = (
                    0.5
                    + self.phi
                    + self.theta * section_j.sigma_ds * a22
                    + thetassign * 0.5 * self.thetas * st22 * dt
                )
            elif (
                not predictor_step
            ):  # This reads more clearly that the equivalent simple 'else'
                thetassign = 1.0
                g11 = (
                    0.5
                    + self.phi
                    + self.theta * section_j.sigma_ds * a11
                    + thetassign * 0.5 * self.thetas * st11
                )
                g12 = (
                    self.theta * section_j.sigma_ds * a12
                    + thetassign * 0.5 * self.thetas * st12
                )
                g21 = (
                    self.theta * section_j.sigma_ds * a21
                    + thetassign * 0.5 * self.thetas * st21
                )
                g22 = (
                    0.5
                    + self.phi
                    + self.theta * section_j.sigma_ds * a22
                    + thetassign * 0.5 * self.thetas * st22
                )

            if self.debug:
                section_j.g11 = g11
                section_j.g12 = g12
                section_j.g21 = g21
                section_j.g22 = g22

            section_j.g11inv = g22 / (g11 * g22 - g12 * g21)
            section_j.g12inv = -g12 / (g11 * g22 - g12 * g21)
            section_j.g21inv = -g21 / (g11 * g22 - g12 * g21)
            section_j.g22inv = g11 / (g11 * g22 - g12 * g21)

            section_j.f1 = flow
            section_j.f2 = flow ** 2.0 / area + self.gravity * section_j.ci1

            #      if(i.ge.2.and.i.lt.n_sections) then
            if i < self.I_DOWNSTREAM and i > self.I_UPSTREAM:
                section_DS = section.ds_section
                section_DS_j = section_DS.time_steps[j_current]
                section_US = section.us_section
                section_US_j = section_US.time_steps[j_current]
                if (
                    predictor_step
                ):  # If we are on the second step, applying the predictors, use the areap and qp
                    area_DS = section_DS_j.flow_area
                    flow_DS = section_DS_j.flow
                    area_US = section_US_j.flow_area
                    flow_US = section_US_j.flow
                else:
                    area_DS = section_DS_j.areap
                    flow_DS = section_DS_j.qp
                    area_US = section_US_j.areap
                    flow_US = section_US_j.qp

                dip1 = section_DS.get_depth_area(area_DS)
                di = 2 * section.get_depth_area(area)
                dim1 = section_US.get_depth_area(area_US)
                if abs(dip1 + di + dim1) < self.depth_tolerance:
                    section_j.eps2 = self.depth_tolerance
                else:
                    section_j.eps2 = (
                        self.alfa2 * abs(dip1 - di + dim1) / (dip1 + di + dim1)
                    )
                # Assign values for eps2 at boundaries
                if i == self.I_DOWNSTREAM - 1:
                    section_DS_j.eps2 = section_j.eps2
                elif i == self.I_UPSTREAM + 1:
                    section_US_j.eps2 = section_j.eps2

        # c
        # TODO: WHAT IS THIS STEP --- something to do with the boundary?
        #      do 20 i=2,n_sections-1
        #        if(ityp(i).ne.1) then
        #          eps2(i)=eps2(i-1)
        #          eps2(i+1)=eps2(i+2)
        #        endif
        # 20    continue
        # c

        if predictor_step:
            for i, section in enumerate(section_arr):
                if i == self.I_UPSTREAM or i == self.I_DOWNSTREAM:
                    continue
                section_j = section.time_steps[j_current]
                section_DS = section.ds_section
                section_DS_j = section_DS.time_steps[j_current]
                section_j.eps2 = max(section_DS_j.eps2, section_j.eps2)
                # print('0.0, alfa4, eps2, velocity, celerity {} {} {} {} {}'.format(0.0, self.alfa4, section_j.eps2
                #                 , section_j.velocity, section_j.celerity))
                section_j.eps4 = max(
                    0.0,
                    self.alfa4
                    - section_j.eps2 / (section_j.velocity + section_j.celerity),
                )
            #      eps4(i+1)=max(0.,alfa4-eps2(i+1)/(u(i+1)+c(i+1)))
            # c      write(*,*)'corr',i,eps2(i)

        elif not predictor_step:
            for k, section in enumerate(reversed(section_arr)):
                i = self.I_DOWNSTREAM - k
                # print(k)
                if i == self.I_DOWNSTREAM:
                    continue
                section_j = section.time_steps[j_current]
                section_DS = section.ds_section
                section_DS_j = section_DS.time_steps[j_current]
                section_DS_j.eps2 = max(section_DS_j.eps2, section_j.eps2)
                # print('0.0, alfa, eps2, velocity, celerity {} {} {} {} {}'.format(0.0, self.alfa4, section_j.eps2
                #                 , section_j.velocity, section_j.celerity))
                section_DS_j.eps4 = max(
                    0.0,
                    self.alfa4
                    - section_DS_j.eps2
                    / (section_DS_j.velocity + section_DS_j.celerity),
                )
        # c      write(*,*)'corr',i,eps2(i)

        #      d1(1)=0.0
        #      d2(1)=0.0
        #      d1(I_UPSTREAM)=0.0
        #      d2(I_UPSTREAM)=0.0

        for i, section in enumerate(section_arr):
            section_j = section.time_steps[
                j_current
            ]  # TODO: Implement direct iterator for timestep
            if i == self.I_UPSTREAM:
                section_j.d1 = 0.0
                section_j.d2 = 0.0
            elif i == self.I_DOWNSTREAM:
                section_j.d1 = 0.0
                section_j.d2 = 0.0
            else:
                if (
                    predictor_step
                ):  # If we are on the second step, applying the predictors, use the areap and qp
                    section_DS = section.ds_section
                    section_DS_j = section_DS.time_steps[j_current]
                    area = section_j.flow_area
                    flow = section_j.flow
                    area_DS = section_DS_j.flow_area
                    flow_DS = section_DS_j.flow
                else:
                    section_US = section.us_section
                    section_US_j = section_US.time_steps[j_current]
                    area = section_j.areap
                    flow = section_j.qp
                    area_US = section_US_j.areap
                    flow_US = section_US_j.qp
                ei = max(
                    abs(section_j.velocity + section_j.celerity),
                    abs(section_j.velocity - section_j.celerity),
                )
                if predictor_step:
                    ei1 = max(
                        abs(section_DS_j.velocity + section_DS_j.celerity),
                        abs(section_DS_j.velocity - section_DS_j.celerity),
                    )
                elif not predictor_step:
                    ei1 = max(
                        abs(section_US_j.velocity + section_US_j.celerity),
                        abs(section_US_j.velocity - section_US_j.celerity),
                    )
                eia = (ei + ei1) / 2.0
                if self.debug:
                    section_j.eia = eia
                #      if(ityp(i-1).ne.1) then
                #        d1(i)=0.0
                #        d2(i)=0.0
                #      elseif(i.eq.2.or.i.eq.(n_sections-1)) then
                if predictor_step:
                    area_difference = area_DS - area
                    flow_difference = flow_DS - flow
                elif (
                    not predictor_step
                ):  # This if could technically be combined with the above if statement
                    # but it reads more clearly to have it here.
                    area_difference = area - area_US
                    flow_difference = flow - flow_US
                section_j.d1 = section_j.eps2 * eia * area_difference
                section_j.d2 = section_j.eps2 * eia * flow_difference
                if i > self.I_UPSTREAM + 1 and i < self.I_DOWNSTREAM - 1:
                    # print(i, self.I_DOWNSTREAM, self.I_UPSTREAM)
                    #        d1(i)=eps2(i)*eia*(areap(i)-areap(i-1))
                    #        d2(i)=eps2(i)*eia*(qp(i)-qp(i-1))
                    # TODO: Determine if the reduction for the next-to-boundary sections
                    # is unnecessary (i.e., Could the if statement be replaced by simply
                    # using the next lines and letting the equation reduce automatically
                    # based on the value in eps4?
                    if predictor_step:
                        section_US = section.us_section
                        section_US_j = section_US.time_steps[j_current]
                        area_US = section_US_j.flow_area
                        flow_US = section_US_j.flow
                        section_2DS = section.ds_section.ds_section
                        section_2DS_j = section_DS.time_steps[j_current]
                        area_2DS = section_2DS_j.flow_area
                        flow_2DS = section_2DS_j.flow
                        area3diff = area_2DS - 3.0 * area_DS + 3.0 * area - area_US
                        flow3diff = flow_2DS - 3.0 * flow_DS + 3.0 * flow - flow_US
                    elif not predictor_step:
                        section_DS = section.ds_section
                        section_DS_j = section_DS.time_steps[j_current]
                        area_DS = section_DS_j.areap
                        flow_DS = section_DS_j.qp
                        section_2US = section.us_section.us_section
                        section_2US_j = section_2US.time_steps[j_current]
                        area_2US = section_2US_j.areap
                        flow_2US = section_2US_j.qp
                        area3diff = area_DS - 3.0 * area + 3.0 * area_US - area_2US
                        flow3diff = flow_DS - 3.0 * flow + 3.0 * flow_US - flow_2US
                    section_j.d1 = section_j.d1 - section_j.eps4 * area3diff
                    section_j.d2 = section_j.d2 - section_j.eps4 * flow3diff
                    pass
            #      else
            #      d1(i)=eps2(i)*eia*(areap(i)-areap(i-1))-eps4(i)*
            #    1        (areap(i+1)-3*areap(i)+3*areap(i-1)-areap(i-2))
            #      d2(i)=eps2(i)*eia*(qp(i)-qp(i-1))-eps4(i)*(qp(i+1)-3*qp(i)+
            #    1      3*qp(i-1)-qp(i-2))
            #      endif
            # c      write(*,*)'corr',i,d1(i),d2(i),eps2(i),flow_area(i+1),flow_area(i)

    def compute_predictor(
        self,
        section_arr,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
    ):
        """docstring here"""
        dt = self.time_list[j_next] - self.time_list[j_current]
        for i, section in enumerate(section_arr):
            # Use the flow time series for the upstream boundary
            section_j = section.time_steps[j_current]
            section_j.sigma_ds = dt / section.dx_ds
            if i == self.I_UPSTREAM:
                # Set the upstream boundary flow
                # This is set as a delta -- a correction factor -- to work in the
                # predictor sweep.
                # So we set the Delta Q predictor, which is set from the input
                # time series for the upstream-most point.
                # TODO: Confirm that DAP(1) is never used... Why not? ASK EHAB
                section_j.delta_area_predictor = 0.0
                section_j.delta_flow_predictor = (
                    upstream_flow_next - upstream_flow_current
                )
            else:
                section_US = section.us_section
                section_US_j = section_US.time_steps[j_current]
                section_j.rhs1 = (
                    -1.0
                    * section_j.sigma_ds
                    * (section_j.f1 - section_US_j.f1 - section_j.d1 + section_US_j.d1)
                )
                section_j.rhs2 = -1.0 * section_j.sigma_ds * (
                    section_j.f2 - section_US_j.f2 - section_j.d2 + section_US_j.d2
                ) + dt * self.gravity * (section_j.ci2_ds + section_j.as0_ds)
                c11 = (
                    section_j.g11inv * section_US_j.b11
                    + section_j.g12inv * section_US_j.b21
                )
                c12 = (
                    section_j.g11inv * section_US_j.b12
                    + section_j.g12inv * section_US_j.b22
                )
                c21 = (
                    section_j.g21inv * section_US_j.b11
                    + section_j.g22inv * section_US_j.b21
                )
                c22 = (
                    section_j.g21inv * section_US_j.b12
                    + section_j.g22inv * section_US_j.b22
                )

                if self.debug:
                    section_j.c11 = c11
                    section_j.c12 = c12
                    section_j.c21 = c21
                    section_j.c22 = c22

                section_j.delta_area_predictor = (
                    section_j.g11inv * section_j.rhs1
                    + section_j.g12inv * section_j.rhs2
                    - c11 * section_US_j.delta_area_predictor
                    - c12 * section_US_j.delta_flow_predictor
                )
                section_j.delta_flow_predictor = (
                    section_j.g21inv * section_j.rhs1
                    + section_j.g22inv * section_j.rhs2
                    - c21 * section_US_j.delta_area_predictor
                    - c22 * section_US_j.delta_flow_predictor
                )
                if i == self.I_DOWNSTREAM:
                    section_j.delta_area_predictor = 0.0
                    # TODO: Ask Ehab why these values are touched in the predictor step
                    # section_j.delta_area_corrector = 0.0
                    # section_j.delta_flow_corrector = section_j.delta_flow_predictor

            # Update via predictor
            if self.debug and (i == self.I_UPSTREAM or i == self.I_DOWNSTREAM):
                print(
                    f"area {section_j.flow_area}, dap {section_j.delta_area_predictor}"
                )
            section_j.areap = section_j.flow_area + section_j.delta_area_predictor
            section_j.depthp = section.get_depth_area(section_j.areap)
            if self.debug and (i == self.I_UPSTREAM or i == self.I_DOWNSTREAM):
                print(
                    f"flow {section_j.flow_area}, dqp {section_j.delta_flow_predictor}"
                )
            section_j.qp = section_j.flow + section_j.delta_flow_predictor

    def compute_corrector(
        self,
        section_arr,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
    ):
        """docstring here"""
        dt = self.time_list[j_next] - self.time_list[j_current]
        for k, section in enumerate(reversed(section_arr)):
            # for i, section in enumerate((self.sections)):
            i = self.I_DOWNSTREAM - k
            # TODO: CONFIRM That the reversed array is necessary in this case (I don't think it is...)
            # Use the flow time series for the upstream boundary
            section_j = section.time_steps[j_current]
            if i == self.I_DOWNSTREAM:
                # Set the upstream boundary flow
                # This is set as a delta -- a correction factor -- to work in the
                # predictor sweep.
                # So we set the Delta Q predictor, which is set from the input
                # time series for the upstream-most point.
                section_j.delta_area_corrector = 0.0
                section_j.delta_flow_corrector = section_j.delta_flow_predictor
            else:
                section_DS = section.ds_section
                section_DS_j = section_DS.time_steps[j_current]
                section_j.sigma_ds = dt / section.dx_ds
                section_j.rhs1 = (
                    -1.0
                    * section_j.sigma_ds
                    * (section_DS_j.f1 - section_j.f1 - section_DS_j.d1 + section_j.d1)
                )
                section_j.rhs2 = -1.0 * section_j.sigma_ds * (
                    section_DS_j.f2 - section_j.f2 - section_DS_j.d2 + section_j.d2
                ) + dt * self.gravity * (section_j.ci2_ds + section_j.as0_ds)
                c11 = (
                    section_j.g11inv * section_DS_j.b11
                    + section_j.g12inv * section_DS_j.b21
                )
                c12 = (
                    section_j.g11inv * section_DS_j.b12
                    + section_j.g12inv * section_DS_j.b22
                )
                c21 = (
                    section_j.g21inv * section_DS_j.b11
                    + section_j.g22inv * section_DS_j.b21
                )
                c22 = (
                    section_j.g21inv * section_DS_j.b12
                    + section_j.g22inv * section_DS_j.b22
                )

                if self.debug:
                    # predictor_step = False
                    # if not predictor_step:
                    section_j.c11 = c11
                    section_j.c12 = c12
                    section_j.c21 = c21
                    section_j.c22 = c22

                section_j.delta_area_corrector = (
                    section_j.g11inv * section_j.rhs1
                    + section_j.g12inv * section_j.rhs2
                    - c11 * section_DS_j.delta_area_corrector
                    - c12 * section_DS_j.delta_flow_corrector
                )
                section_j.delta_flow_corrector = (
                    section_j.g21inv * section_j.rhs1
                    + section_j.g22inv * section_j.rhs2
                    - c21 * section_DS_j.delta_area_corrector
                    - c22 * section_DS_j.delta_flow_corrector
                )
                if i == self.I_UPSTREAM:
                    section_j.delta_flow_corrector = section_j.delta_flow_predictor

    def mesh_final_update(
        self,
        section_arr,
        j_current,
        j_next,
        upstream_flow_current,
        upstream_flow_next,
        downstream_stage_current,
        downstream_stage_next,
        predictor_step=False,
    ):

        for i, section in enumerate(section_arr):
            section_j = section.time_steps[j_current]
            da_bar = (
                section_j.delta_area_predictor + section_j.delta_area_corrector
            ) / 2.0
            dq_bar = (
                section_j.delta_flow_predictor + section_j.delta_flow_corrector
            ) / 2.0
            if (da_bar + section_j.flow_area) > self.area_tolerance:
                next_flow_area = section_j.flow_area + da_bar
            else:
                next_flow_area = self.area_tolerance
            next_depth = section.get_depth_area(next_flow_area)
            next_flow = section_j.flow + dq_bar
            self.debug = False
            section.time_steps.append(
                self.TimeStep(
                    new_time=self.time_list[j_next],
                    new_flow=next_flow,
                    new_depth=next_depth,
                    new_water_z=section.bottom_z + next_depth,
                    new_area=next_flow_area,
                )
            )
            section_jnext = section.time_steps[j_next]
            # if self.debug:
            if self.debug and (i == self.I_UPSTREAM or i == self.I_UPSTREAM + 1):
                print(
                    "current depth area flow {: 10g} {: 9g} {: 9g}".format(
                        section_j.depth, section_j.flow_area, section_j.flow
                    )
                )
                print(f"              da    dq             {da_bar: 9g} {dq_bar: 9g}")
                print(
                    "next    depth area flow {: 10g} {: 9g} {: 9g}".format(
                        next_depth, next_flow_area, next_flow
                    )
                )
                print(
                    "next    depth area flow {: 10g} {: 9g} {: 9g}\n(in new array)".format(
                        section_jnext.depth, section_jnext.flow_area, section_jnext.flow
                    )
                )
            self.debug = False

    class RectangleSection(Reach.RectangleSection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_ci1_depth(self, depth):
            return self.bottom_width * (depth ** 2.0) / 2.0

        def get_dbdx_ds_depth(self, depth, depth_ds):
            # FOR RectangleSection -- the depth is trivial
            return (self.ds_section.bottom_width - self.bottom_width) / (self.dx_ds)

        def get_ci2_depth_depth_ds(self, depth, depth_ds):
            # print((depth_ds ** 2.0) + (depth ** 2.0))
            # print(self.ds_section.bottom_width - self.bottom_width)
            # print(self.dx_ds)
            return (
                ((depth_ds ** 2.0) + (depth ** 2.0))
                * (self.ds_section.bottom_width - self.bottom_width)
                / (self.dx_ds * 4.0)
            )

        def get_dkda_area(self, area):
            wetted_perimeter = self.get_wetted_perimeter_area(area)
            return (
                1
                / self.manning_n_ds
                * (
                    (5.0 / 3.0 * area ** (2.0 / 3.0) * wetted_perimeter)
                    - (area ** (5.0 / 3.0) * 2.0 / self.bottom_width)
                )
                / (wetted_perimeter ** 2.0)
            )
            # TODO: CORRECTED
            # / (wetted_perimeter ** (5.0 / 3.0)))

        def get_st21_area(
            self, area, flow, gs0, conveyance, dkda, dbdx, gravity, manning_m
        ):
            return (
                gravity * area / (self.bottom_width ** 2.0) * dbdx
                + gs0
                + manning_m
                * 2.0
                * gravity
                * area
                * flow
                * abs(flow)
                / (conveyance ** 3.0)
                * dkda
            )

    def write_state_timestep_mesh(
        self, section_arr, j_current, prefix="mesh", output_path=None, verbose=False
    ):
        elevation = [
            section.bottom_z + section.time_steps[j_current].water_z
            for section in section_arr
        ]
        depth = [section.time_steps[j_current].depth for section in section_arr]
        flow = [section.time_steps[j_current].flow for section in section_arr]
        flow_area = [section.time_steps[j_current].flow_area for section in section_arr]

        if verbose:
            depthp = [section.time_steps[j_current].depthp for section in section_arr]
            qp = [section.time_steps[j_current].qp for section in section_arr]
            areap = [section.time_steps[j_current].areap for section in section_arr]

            # Section/SecPred
            ci1 = [section.time_steps[j_current].ci1 for section in section_arr]
            hy = [section.time_steps[j_current].hy for section in section_arr]
            conveyance_ds = [
                section.time_steps[j_current].conveyance_ds for section in section_arr
            ]
            ci2_ds = [section.time_steps[j_current].ci2_ds for section in section_arr]
            dbdx_ds = [section.time_steps[j_current].dbdx_ds for section in section_arr]
            friction_slope_ds = [
                section.time_steps[j_current].friction_slope_ds
                for section in section_arr
            ]
            gs0_ds = [section.time_steps[j_current].gs0_ds for section in section_arr]
            as0_ds = [section.time_steps[j_current].as0_ds for section in section_arr]

            # MatrixC/MatrixP
            velocity = [
                section.time_steps[j_current].velocity for section in section_arr
            ]
            celerity = [
                section.time_steps[j_current].celerity for section in section_arr
            ]
            e11 = [section.time_steps[j_current].e11 for section in section_arr]
            e12 = [section.time_steps[j_current].e12 for section in section_arr]
            e21 = [section.time_steps[j_current].e21 for section in section_arr]
            e22 = [section.time_steps[j_current].e22 for section in section_arr]
            f11 = [section.time_steps[j_current].f11 for section in section_arr]
            f12 = [section.time_steps[j_current].f12 for section in section_arr]
            f21 = [section.time_steps[j_current].f21 for section in section_arr]
            f22 = [section.time_steps[j_current].f22 for section in section_arr]
            d11 = [section.time_steps[j_current].d11 for section in section_arr]
            d22 = [section.time_steps[j_current].d22 for section in section_arr]
            a11 = [section.time_steps[j_current].a11 for section in section_arr]
            a12 = [section.time_steps[j_current].a12 for section in section_arr]
            a21 = [section.time_steps[j_current].a21 for section in section_arr]
            a22 = [section.time_steps[j_current].a22 for section in section_arr]
            wetted_perimeter = [
                section.time_steps[j_current].wetted_perimeter
                for section in section_arr
            ]
            dkda = [section.time_steps[j_current].dkda for section in section_arr]

            st11 = [section.time_steps[j_current].st11 for section in section_arr]
            st12 = [section.time_steps[j_current].st12 for section in section_arr]
            st21 = [section.time_steps[j_current].st21 for section in section_arr]
            st22 = [section.time_steps[j_current].st22 for section in section_arr]
            b11 = [section.time_steps[j_current].b11 for section in section_arr]
            b12 = [section.time_steps[j_current].b12 for section in section_arr]
            b21 = [section.time_steps[j_current].b21 for section in section_arr]
            b22 = [section.time_steps[j_current].b22 for section in section_arr]
            g11 = [section.time_steps[j_current].g11 for section in section_arr]
            g12 = [section.time_steps[j_current].g12 for section in section_arr]
            g21 = [section.time_steps[j_current].g21 for section in section_arr]
            g22 = [section.time_steps[j_current].g22 for section in section_arr]
            g11inv = [section.time_steps[j_current].g11inv for section in section_arr]
            g12inv = [section.time_steps[j_current].g12inv for section in section_arr]
            g21inv = [section.time_steps[j_current].g21inv for section in section_arr]
            g22inv = [section.time_steps[j_current].g22inv for section in section_arr]
            f1 = [section.time_steps[j_current].f1 for section in section_arr]
            f2 = [section.time_steps[j_current].f2 for section in section_arr]

            eps2 = [section.time_steps[j_current].eps2 for section in section_arr]
            eps4 = [section.time_steps[j_current].eps4 for section in section_arr]
            eia = [section.time_steps[j_current].eia for section in section_arr]

            d1 = [section.time_steps[j_current].d1 for section in section_arr]
            d2 = [section.time_steps[j_current].d2 for section in section_arr]
            c11 = [section.time_steps[j_current].c11 for section in section_arr]
            c12 = [section.time_steps[j_current].c12 for section in section_arr]
            c21 = [section.time_steps[j_current].c21 for section in section_arr]
            c22 = [section.time_steps[j_current].c22 for section in section_arr]
            rhs1 = [section.time_steps[j_current].rhs1 for section in section_arr]
            rhs2 = [section.time_steps[j_current].rhs2 for section in section_arr]
            dap = [
                section.time_steps[j_current].delta_area_predictor
                for section in section_arr
            ]
            dqp = [
                section.time_steps[j_current].delta_flow_predictor
                for section in section_arr
            ]
            dac = [
                section.time_steps[j_current].delta_area_corrector
                for section in section_arr
            ]
            dqc = [
                section.time_steps[j_current].delta_flow_corrector
                for section in section_arr
            ]
        if output_path:
            # pd.DataFrame(zip(elevations, depths, flows, areas)).to_csv(output_path)
            if verbose:
                print(f"output to: {output_path}")
            folder, file = os.path.split(output_path)
            # TODO: wrap these into some useful chunking for different verbosity levels
            # e.g., main_output, Predictor, corrector, etc.
            pd.DataFrame(elevation).to_csv(
                os.path.join(folder, f"{prefix}_elevation_{j_current:04}_{file}")
            )
            pd.DataFrame(depth).to_csv(
                os.path.join(folder, f"{prefix}_depth_{j_current:04}_{file}")
            )
            pd.DataFrame(flow).to_csv(
                os.path.join(folder, f"{prefix}_flow_{j_current:04}_{file}")
            )
            pd.DataFrame(flow_area).to_csv(
                os.path.join(folder, f"{prefix}_area_{j_current:04}_{file}")
            )
            if verbose:
                pd.DataFrame(areap).to_csv(
                    os.path.join(folder, f"{prefix}_areap_{j_current:04}_{file}")
                )
                pd.DataFrame(qp).to_csv(
                    os.path.join(folder, f"{prefix}_qp_{j_current:04}_{file}")
                )
                pd.DataFrame(depthp).to_csv(
                    os.path.join(folder, f"{prefix}_depthp_{j_current:04}_{file}")
                )

                pd.DataFrame(ci1).to_csv(
                    os.path.join(folder, f"{prefix}_ci1_{j_current:04}_{file}")
                )
                pd.DataFrame(hy).to_csv(
                    os.path.join(folder, f"{prefix}_hy_{j_current:04}_{file}")
                )
                pd.DataFrame(conveyance_ds).to_csv(
                    os.path.join(
                        folder, f"{prefix}_converyance_ds_{j_current:04}_{file}"
                    )
                )
                pd.DataFrame(ci2_ds).to_csv(
                    os.path.join(folder, f"{prefix}_ci2_ds_{j_current:04}_{file}")
                )
                pd.DataFrame(dbdx_ds).to_csv(
                    os.path.join(folder, f"{prefix}_dbdx_ds_{j_current:04}_{file}")
                )
                pd.DataFrame(friction_slope_ds).to_csv(
                    os.path.join(
                        folder, f"{prefix}_friction_slope_ds_{j_current:04}_{file}"
                    )
                )
                pd.DataFrame(gs0_ds).to_csv(
                    os.path.join(folder, f"{prefix}_gs0_ds_{j_current:04}_{file}")
                )
                pd.DataFrame(as0_ds).to_csv(
                    os.path.join(folder, f"{prefix}_as0_ds_{j_current:04}_{file}")
                )

                pd.DataFrame(velocity).to_csv(
                    os.path.join(folder, f"{prefix}_velocity_{j_current:04}_{file}")
                )
                pd.DataFrame(celerity).to_csv(
                    os.path.join(folder, f"{prefix}_celerity_{j_current:04}_{file}")
                )
                pd.DataFrame(e11).to_csv(
                    os.path.join(folder, f"{prefix}_e11_{j_current:04}_{file}")
                )
                pd.DataFrame(e12).to_csv(
                    os.path.join(folder, f"{prefix}_e12_{j_current:04}_{file}")
                )
                pd.DataFrame(e21).to_csv(
                    os.path.join(folder, f"{prefix}_e21_{j_current:04}_{file}")
                )
                pd.DataFrame(e22).to_csv(
                    os.path.join(folder, f"{prefix}_e22_{j_current:04}_{file}")
                )
                pd.DataFrame(f11).to_csv(
                    os.path.join(folder, f"{prefix}_f11_{j_current:04}_{file}")
                )
                pd.DataFrame(f12).to_csv(
                    os.path.join(folder, f"{prefix}_f12_{j_current:04}_{file}")
                )
                pd.DataFrame(f21).to_csv(
                    os.path.join(folder, f"{prefix}_f21_{j_current:04}_{file}")
                )
                pd.DataFrame(f22).to_csv(
                    os.path.join(folder, f"{prefix}_f22_{j_current:04}_{file}")
                )
                pd.DataFrame(d11).to_csv(
                    os.path.join(folder, f"{prefix}_d11_{j_current:04}_{file}")
                )
                pd.DataFrame(d22).to_csv(
                    os.path.join(folder, f"{prefix}_d22_{j_current:04}_{file}")
                )
                pd.DataFrame(a11).to_csv(
                    os.path.join(folder, f"{prefix}_a11_{j_current:04}_{file}")
                )
                pd.DataFrame(a12).to_csv(
                    os.path.join(folder, f"{prefix}_a12_{j_current:04}_{file}")
                )
                pd.DataFrame(a21).to_csv(
                    os.path.join(folder, f"{prefix}_a21_{j_current:04}_{file}")
                )
                pd.DataFrame(a22).to_csv(
                    os.path.join(folder, f"{prefix}_a22_{j_current:04}_{file}")
                )
                pd.DataFrame(wetted_perimeter).to_csv(
                    os.path.join(
                        folder, f"{prefix}_wetted_perimeter_{j_current:04}_{file}"
                    )
                )
                pd.DataFrame(dkda).to_csv(
                    os.path.join(folder, f"{prefix}_dkda_{j_current:04}_{file}")
                )

                pd.DataFrame(st11).to_csv(
                    os.path.join(folder, f"{prefix}_st11_{j_current:04}_{file}")
                )
                pd.DataFrame(st12).to_csv(
                    os.path.join(folder, f"{prefix}_st12_{j_current:04}_{file}")
                )
                pd.DataFrame(st21).to_csv(
                    os.path.join(folder, f"{prefix}_st21_{j_current:04}_{file}")
                )
                pd.DataFrame(st22).to_csv(
                    os.path.join(folder, f"{prefix}_st22_{j_current:04}_{file}")
                )
                pd.DataFrame(b11).to_csv(
                    os.path.join(folder, f"{prefix}_b11_{j_current:04}_{file}")
                )
                pd.DataFrame(b12).to_csv(
                    os.path.join(folder, f"{prefix}_b12_{j_current:04}_{file}")
                )
                pd.DataFrame(b21).to_csv(
                    os.path.join(folder, f"{prefix}_b21_{j_current:04}_{file}")
                )
                pd.DataFrame(b22).to_csv(
                    os.path.join(folder, f"{prefix}_b22_{j_current:04}_{file}")
                )
                pd.DataFrame(g11).to_csv(
                    os.path.join(folder, f"{prefix}_g11_{j_current:04}_{file}")
                )
                pd.DataFrame(g12).to_csv(
                    os.path.join(folder, f"{prefix}_g12_{j_current:04}_{file}")
                )
                pd.DataFrame(g21).to_csv(
                    os.path.join(folder, f"{prefix}_g21_{j_current:04}_{file}")
                )
                pd.DataFrame(g22).to_csv(
                    os.path.join(folder, f"{prefix}_g22_{j_current:04}_{file}")
                )
                pd.DataFrame(g11inv).to_csv(
                    os.path.join(folder, f"{prefix}_g11inv_{j_current:04}_{file}")
                )
                pd.DataFrame(g12inv).to_csv(
                    os.path.join(folder, f"{prefix}_g12inv_{j_current:04}_{file}")
                )
                pd.DataFrame(g21inv).to_csv(
                    os.path.join(folder, f"{prefix}_g21inv_{j_current:04}_{file}")
                )
                pd.DataFrame(g22inv).to_csv(
                    os.path.join(folder, f"{prefix}_g22inv_{j_current:04}_{file}")
                )
                pd.DataFrame(f1).to_csv(
                    os.path.join(folder, f"{prefix}_f1_{j_current:04}_{file}")
                )
                pd.DataFrame(f2).to_csv(
                    os.path.join(folder, f"{prefix}_f2_{j_current:04}_{file}")
                )
                # pd.DataFrame(zip(flows, areas)).to_csv(output_path)
                pd.DataFrame(eps2).to_csv(
                    os.path.join(folder, f"{prefix}_eps2_{j_current:04}_{file}")
                )
                pd.DataFrame(eps4).to_csv(
                    os.path.join(folder, f"{prefix}_eps4_{j_current:04}_{file}")
                )
                pd.DataFrame(eia).to_csv(
                    os.path.join(folder, f"{prefix}_eia_{j_current:04}_{file}")
                )

                pd.DataFrame(d1).to_csv(
                    os.path.join(folder, f"{prefix}_d1_{j_current:04}_{file}")
                )
                pd.DataFrame(d2).to_csv(
                    os.path.join(folder, f"{prefix}_d2_{j_current:04}_{file}")
                )
                pd.DataFrame(c11).to_csv(
                    os.path.join(folder, f"{prefix}_c11_{j_current:04}_{file}")
                )
                pd.DataFrame(c12).to_csv(
                    os.path.join(folder, f"{prefix}_c12_{j_current:04}_{file}")
                )
                pd.DataFrame(c21).to_csv(
                    os.path.join(folder, f"{prefix}_c21_{j_current:04}_{file}")
                )
                pd.DataFrame(c22).to_csv(
                    os.path.join(folder, f"{prefix}_c22_{j_current:04}_{file}")
                )
                pd.DataFrame(rhs1).to_csv(
                    os.path.join(folder, f"{prefix}_rhs1_{j_current:04}_{file}")
                )
                pd.DataFrame(rhs2).to_csv(
                    os.path.join(folder, f"{prefix}_rhs2_{j_current:04}_{file}")
                )
                pd.DataFrame(dap).to_csv(
                    os.path.join(folder, f"{prefix}_dap_{j_current:04}_{file}")
                )
                pd.DataFrame(dqp).to_csv(
                    os.path.join(folder, f"{prefix}_dqp_{j_current:04}_{file}")
                )
                pd.DataFrame(dac).to_csv(
                    os.path.join(folder, f"{prefix}_dac_{j_current:04}_{file}")
                )
                pd.DataFrame(dqc).to_csv(
                    os.path.join(folder, f"{prefix}_dqc_{j_current:04}_{file}")
                )
        else:
            print(f"{elevation}")
            print(f"{depth}")
            print(f"{flow}")
            print(f"{area}")

    class TimeStep(Reach.TimeStep):
        """MESH-specific time-step values"""

        def __init__(self, new_water_z=0.0, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Per-time-step at-a-section properties
            self.delta_flow_corrector = None
            self.delta_flow_predictor = None
            self.delta_area_corrector = None
            self.delta_area_predictor = None
            self.water_z = new_water_z
            self.areap = None
            self.qp = None
            self.depthp = None
            self.ci1 = None
            self.hy = None  # Hydraulic Radius (used to compute co)

            # Per-time-step downstream reach properties
            self.conveyance_ds = None
            self.ci2_ds = None
            # self.friction_slope_ds = 0 # Derived from parent Class, Reach.TimeStep
            self.as0_ds = None
            self.gs0_ds = None
            self.sigma_ds = None  # Sigma is related to the courant parameter: CFL = celerity * sigma
            # self.cour = 0
            self.dbdx_ds = None
            self.velocity = None
            self.celerity = None

            self.wetted_perimeter = None

            self.f1 = None
            self.f2 = None
            self.d1 = None
            self.d2 = None
            self.d11 = None
            self.d22 = None
            self.b11 = None
            self.b12 = None
            self.b21 = None
            self.b22 = None
            self.g11inv = None
            self.g12inv = None
            self.g21inv = None
            self.g22inv = None
            self.eps2 = None
            self.eps4 = None
            self.rhs1 = None
            self.rhs2 = None

            # FROM HERE DOWN are unnecessary
            # TODO: These could be in a debug statement
            self.a11 = None
            self.a12 = None
            self.a21 = None
            self.a22 = None
            self.g11 = None
            self.g12 = None
            self.g21 = None
            self.g22 = None
            self.e11 = None
            self.e12 = None
            self.e21 = None
            self.e22 = None
            self.f11 = None
            self.f12 = None
            self.f21 = None
            self.f22 = None
            self.st11 = None
            self.st12 = None
            self.st21 = None
            self.st22 = None

            self.eia = None

            self.dkda = None

            self.c11 = None
            self.c12 = None
            self.c21 = None
            self.c22 = None


def main():

    input_type = "file"
    input_vars = {}
    input_vars["filetype"] = "mesh.py"

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_folder = os.path.join(root, r"test")
    output_folder = os.path.join(test_folder, r"output", r"text")
    input_folder = os.path.join(test_folder, r"input", r"text")
    input_path = os.path.join(input_folder, r"input.txt")
    output_path = os.path.join(output_folder, r"out.txt")

    input_vars[r"input_path"] = input_path
    # print(input_path)

    # if len(sys.argv) > 1:
    #     input_path = sys.argv[1]
    # else:
    #     input_path = os.path.join(This_Example_Path,"input.txt")
    #     print(input_path)
    #     #input_path = "./input.txt"

    # reach = DummyReach()
    # reach = SimpleFlowTrace()
    # reach = SteadyReach(input_type = input_type, input_vars = input_vars)
    # input_and_initialize(sections, input_path, input_opt)
    reach = MESHpyReach(input_type=input_type, input_vars=input_vars)
    # reach = MuskCReach()
    # reach = MESHDReach()

    reach.compute_initial_state(write_output=False, output_path=output_path)
    reach.debug = True
    reach.compute_time_steps_mesh(
        verbose=True, write_output=False, output_path=output_path
    )
    reach.output_dump_all(output_path=output_path, verbose=True)


if __name__ == "__main__":
    main()
