# import required modules
from __future__ import division
import helpers
import constants
import meshconstants
from reach import Reach
import csv
import os


class MESHpyDUMMYReach(Reach):
    """USE Global Declarations here to manage these values"""

    # TODO Determine why these values do not persist when declared in __init__
    debug = False
    dx_tolerance = meshconstants.DX_TOLERANCE
    depth_tolerance = meshconstants.DEPTH_TOLERANCE
    celerity_epsilon = meshconstants.CELERITY_EPSILON
    area_tolerance = meshconstants.AREA_TOLERANCE
    crmax = 0.0
    crmin = 100.0
    predictor_step = True
    yy = 0.0
    qq = 0.0
    phi = meshconstants.PHI  # source term treatment (0:explicit, 1:implicit)
    theta = meshconstants.THETA  # ?
    thetas = meshconstants.THETAS  # ?
    thesinv = meshconstants.THESINV  # ?
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
                    y0 = (
                        z + self.yy
                    )  # TODO: This seems like a clunky definition of the initial water surface
                    #      Elevation and I think we can do better.
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
        for i, section in enumerate(self.sections):
            # print(f'yy, qq {self.yy} {self.qq}')
            section.time_steps.append(
                self.TimeStep(
                    new_time=self.time_list[0],
                    new_flow=self.qq,
                    new_depth=self.yy,
                    new_water_z=section.bottom_z + self.yy,
                    new_area=section.get_area_depth(self.yy),
                )
            )
            # self.sections[self.I_UPSTREAM].time_steps.append(self.TimeStep(new_flow = q_upstream))
            # self.sections[self.I_DOWNSTREAM].time_steps.append(self.TimeStep(new_depth = y_downstream))
        # print(self.upstream_flow_ts)
        # print(self.downstream_stage_ts)
        pass

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
            if 1 == 1:
                section_j.depth = section.get_depth_area(section_j.flow_area)
                area = section_j.flow_area
                flow = section_j.flow
                depth = section_j.depth

        dt = self.time_list[j_next] - self.time_list[j_current]
        for i, section in enumerate(section_arr):
            # Use the flow time series for the upstream boundary
            section_j = section.time_steps[j_current]
            if 1 == 1:

                section_j.delta_flow_predictor = 25.0

                section_j.delta_area_predictor = 10.0

        dt = self.time_list[j_next] - self.time_list[j_current]
        for i, section in enumerate(reversed(self.sections)):
            section_j = section.time_steps[j_current]
            if 1 == 1:
                section_j.delta_flow_corrector = section_j.delta_flow_predictor
                section_j.delta_area_corrector = 0.0
            da = (section_j.delta_area_predictor + section_j.delta_area_corrector) / 2.0
            dq = (section_j.delta_flow_predictor + section_j.delta_flow_corrector) / 2.0
            self.debug = True
            # if self.debug:
            if self.debug and i == self.I_UPSTREAM:
                print(
                    "current depth area flow {: 10g} {: 6g} {: 6g}".format(
                        section_j.depth, section_j.flow_area, section_j.flow
                    )
                )
                print(f"              da    dq             {da: 6g} {dq: 6g}")
            if (da + section_j.flow_area) > self.area_tolerance:
                next_flow_area = section_j.flow_area + da
            else:
                next_flow_area = self.area_tolerance
            next_depth = section.get_depth_area(next_flow_area)
            next_flow = section_j.flow + dq
            if self.debug and i == self.I_UPSTREAM:
                print(
                    "next    depth area flow {: 10g} {: 6g} {: 6g}".format(
                        next_depth, next_flow_area, next_flow
                    )
                )
            section.time_steps.append(
                self.TimeStep(
                    new_time=self.time_list[0],
                    new_flow=next_flow,
                    new_depth=next_depth,
                    new_water_z=section.bottom_z + next_depth,
                    new_area=next_flow_area,
                )
            )
            section_jnext = section.time_steps[j_next]
            if self.debug and i == self.I_UPSTREAM:
                print(
                    "next    depth area flow {: 10g} {: 6g} {: 6g}\n(in new array)".format(
                        section_jnext.depth, section_jnext.flow_area, section_jnext.flow
                    )
                )
            self.debug = False

    class RectangleSection(Reach.RectangleSection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class TimeStep(Reach.TimeStep):
        """MESH-specific time-step values"""

        def __init__(self, new_water_z=0.0, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Per-time-step at-a-section properties
            self.delta_flow_corrector = 0.0
            self.delta_flow_predictor = 0.0
            self.delta_area_corrector = 0.0
            self.delta_area_predictor = 0.0
            self.water_z = new_water_z
            self.areap = 0.0
            self.qp = 0.0
            self.depthp = 0.0


def main():

    input_type = "file"
    input_vars = {}
    input_vars["filetype"] = "mesh.py"

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_folder = os.path.join(root, r"test")
    output_folder = os.path.join(test_folder, r"output", r"text")
    input_folder = os.path.join(test_folder, r"input", r"text")
    input_path = os.path.join(input_folder, r"input_simple.txt")
    output_path = os.path.join(output_folder, r"Dummy_out.txt")

    input_vars[r"input_path"] = input_path
    # print(input_path)

    # reach = DummyReach()
    # reach = SimpleFlowTrace()
    # reach = SteadyReach(input_type = input_type, input_vars = input_vars)
    reach = MESHpyDUMMYReach(input_type=input_type, input_vars=input_vars)
    # reach = MuskCReach()
    # reach = MESHDReach()

    reach.compute_initial_state(write_output=True, output_path=output_path)
    reach.debug = False
    reach.compute_time_steps(verbose=True, write_output=True, output_path=output_path)
    reach.output_dump_all(output_path=output_path, verbose=True)


if __name__ == "__main__":
    main()
