import pytest
import mc_sseg_stime_NOLOOP_demo as demo
from test_suite_parameters import generate_conus_MC_parameters


input_tup = (
    "single",  # calculation precision (NOT USED)
    300.0,  # Time step
    1800.0,  # segment length
    112.0,  # Trapezoidal bottom width
    448.0,  # Channel top width (at bankfull)
    623.5999755859375,  # Flood plain width
    0.02800000086426735,  # manning roughness of channel
    0.03136000037193298,  # manning roughness of floodplain
    1.399999976158142,  # channel trapezoidal sideslope
    0.0017999999690800905,  # downstream segment bed slope
    40.0,  # Lateral inflow in this time step
    4509,
    5098,
    5017,
    30,
)

# qdc1, qdc2, velc1, velc2, depthc1, depthc2 = demo.compare_methods(*input_tup)

input_tup_random = generate_conus_MC_parameters(1, 16)
out = demo.compare_methods("single", *(next(input_tup_random)))
# print(out)
qdc1, qdc2, velc1, velc2, depthc1, depthc2 = out


def test_MC_kernel_q():
# TODO: Take advantage of ranges above to build a state-space 
# exploration of potential inputs and confirm it parity across all inputs

    assert qdc1 == qdc2


def test_MC_kernel_vel():
# TODO: Take advantage of ranges above to build a state-space 
# exploration of potential inputs and confirm it parity across all inputs

    assert velc1 == velc2
 

def test_MC_kernel_depth():
# TODO: Take advantage of ranges above to build a state-space 
# exploration of potential inputs and confirm it parity across all inputs

   assert depthc1 == depthc2


