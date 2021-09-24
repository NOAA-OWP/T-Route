from nwm_routing.__main__ import _run_everything_v02  # , _input_handler_v02
# from nwm_routing.__main__ import _run_everything_v03  # , _input_handler_v03
# from troute.routing.compute import compute_nhd_routing_v02
from troute.nhd_io import read_custom_input
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
# import subprocess
# import os
# result = os.system("python -m nwm_routing -V2 -f ../../test/input/yaml/CustomInput.yaml")
#
# version = "-V2"
# subp_c = ["python", "-m", "nwm_routing", version, "-f", test_file]
# result = subprocess.call(subp_c)
# 
# import pickle
# with open("test_data.pickle","rb") as f:
#     input_tuple = pickle.load(f)
# results = compute_nhd_routing_v02(*input_tuple)


def _produce_result_set(results):
    num_networks = len(results)
    id_list_j = 0
    values_list_j = 1
    file_base = "custom_test_standard_result"
    test_folder = Path("../../test/output/Pocono_TEST1/standard_test")

    for net_i in range(num_networks):
        ext_str_id = f".net{net_i:05d}_id.out" 
        ext_str_values = f".net{net_i:05d}_values.out" 
        with open(test_folder / (file_base + ext_str_id), "wb") as fid:
            np.save(fid, results[net_i][id_list_j], allow_pickle = False)
        with open(test_folder / (file_base + ext_str_values), "wb") as fid:
            np.save(fid, results[net_i][values_list_j], allow_pickle = False)



def test_croton_ny_levelpool():

    final_flow = run_everything_v02_test("../../test/input/yaml/Croton_NY_levelpool.yaml")

    assert pytest.approx(final_flow) == 1.415740


def test_croton_ny_hybrid_usgs():

    final_flow = run_everything_v02_test("../../test/input/yaml/Croton_NY_hybrid_usgs.yaml")

    assert pytest.approx(final_flow) == 1.084535


#def test_croton_ny_hybrid_usgs_AnA():
#    '''
#    This hybrid usgs AnA test does not run correctly whenever the
#    above hybrid usgs test is run first. This is because the Fortran
#    singleton timeslice reader is initialized in memory from the 
#    above test call with an input timeslice directory set, and that
#    timeslice reader persists in memory for the below call.
#    This test call when run by itself initializes the timeslice reader
#    to a different input timeslice directory, but when run after
#    the above test call, the timeslice reader will still search for 
#    timeslice files in the directory specified in the above test call.
#    '''

#    final_flow = run_everything_v02_test("../../test/input/yaml/Croton_NY_hybrid_usgs_AnA.yaml")

#    assert pytest.approx(final_flow) == 1.3


def test_croton_ny_rfc():

    final_flow = run_everything_v02_test("../../test/input/yaml/Croton_NY_rfc.yaml")

    assert pytest.approx(final_flow) == 8.681314


#def test_croton_ny_levelpool_diffusive():
#    # TODO: Updates needed to verify parity for Croton_NY_levelpool_diffusive.yaml

#    final_flow = run_everything_v02_test("../../test/input/yaml/Croton_NY_levelpool_diffusive.yaml")

#    assert pytest.approx(final_flow) == 0.0


def test_CustomInput_yaml():

    final_flow = run_everything_v02_test("../../test/input/yaml/CustomInput.yaml")

    assert pytest.approx(final_flow) == 0.870077


def test_florence_benchmark():

    final_flow = run_everything_v02_test("../../test/input/yaml/Florence_Benchmark.yaml")

    assert pytest.approx(final_flow) == 0.0043528992


#def test_florence_benchmark_da():

#    final_flow = run_everything_v02_test("../../test/input/yaml/Florence_Benchmark_da.yaml")

#    assert pytest.approx(final_flow) == 0.0


def run_everything_v02_test(custom_input_file):
    """
    Integration test for python_routing_v02 called by individual configurations 
    """

    (
        supernetwork_parameters,
        waterbody_parameters,
        forcing_parameters,
        restart_parameters,
        output_parameters,
        run_parameters,
        parity_parameters,
        data_assimilation_parameters,
        diffusive_parameters,
        coastal_parameters,
    ) = read_custom_input(custom_input_file)

    results = _run_everything_v02(
        supernetwork_parameters,
        waterbody_parameters,
        forcing_parameters,
        restart_parameters,
        output_parameters,
        run_parameters,
        parity_parameters,
        data_assimilation_parameters,
        diffusive_parameters,
        coastal_parameters,
    )

    nts = run_parameters["nts"] 
    dt = run_parameters["dt"] 

    compare_node = parity_parameters["parity_check_compare_node"]

    # construct a dataframe of simulated flows
    fdv_columns = pd.MultiIndex.from_product([range(nts), ["q", "v", "d"]])
    flowveldepth = pd.concat(
        [pd.DataFrame(r[1], index=r[0], columns=fdv_columns) for r in results],
        copy=False,
    )
    flowveldepth = flowveldepth.sort_index()

    flows = flowveldepth.loc[:, (slice(None), "q")]
    flows = flows.T.reset_index(level=[0, 1])
    flows.rename(columns={"level_0": "Timestep", "level_1": "Parameter"}, inplace=True)
    flows["Time (d)"] = ((flows.Timestep + 1) * dt) / (24 * 60 * 60)
    flows = flows.set_index("Time (d)")

    depths = flowveldepth.loc[:, (slice(None), "d")]
    depths = depths.T.reset_index(level=[0, 1])
    depths.rename(columns={"level_0": "Timestep", "level_1": "Parameter"}, inplace=True)
    depths["Time (d)"] = ((depths.Timestep + 1) * dt) / (24 * 60 * 60)
    depths = depths.set_index("Time (d)")

    # Construct dataframe for results from set compare_node.
    # TODO: Consider testing reservoir water elevation also 
    trt = pd.DataFrame(
        flows.loc[:, compare_node].values,
        columns=["flow, t-route (cms)"],
    )

    return trt.iloc[-1][0]


def test_run_everything_v02_custom_input():
    custom_input_file = "../../test/input/yaml/CustomInput.yaml"
    (
        supernetwork_parameters,
        waterbody_parameters,
        forcing_parameters,
        restart_parameters,
        output_parameters,
        run_parameters,
        parity_parameters,
        data_assimilation_parameters,
        diffusive_parameters,
        coastal_parameters,
    ) = read_custom_input(custom_input_file)

    run_parameters["nts"] = 12

    results = _run_everything_v02(
            supernetwork_parameters,
            waterbody_parameters,
            forcing_parameters,
            restart_parameters,
            output_parameters,
            run_parameters,
            parity_parameters,
            data_assimilation_parameters,
            diffusive_parameters,
            coastal_parameters,
    )

    num_networks = len(results)
    id_list_j = 0
    values_list_j = 1
    file_base = "custom_test_standard_result"
    test_folder = Path("../../test/output/Pocono_TEST1/standard_test")

    results_standard = []
    for net_i in range(num_networks):
        ext_str_id = f".net{net_i:05d}_id.out" 
        ext_str_values = f".net{net_i:05d}_values.out" 
        with open(test_folder / (file_base + ext_str_id), "rb") as fid:
            ids = np.load(fid)
        with open(test_folder / (file_base + ext_str_values), "rb") as fid:
            values = np.load(fid)
        results_standard.append((ids,values))

        assert np.array_equal(results_standard[net_i][id_list_j], results[net_i][id_list_j])
        assert np.allclose(results_standard[net_i][values_list_j], results[net_i][values_list_j])


