import numpy as np
import pytest
from troute.routing.fast_reach.simple_da import simple_da_with_decay_py

obs_nogap = [10, 11, 14, 18, 30, 32, 26, 20, 14, 12, 11, 10, 10, 10, 10]
obs_gap1 = [None, None, None, 18, 30, 32, 26, 20, 14, 12, 11, 10, 10, 10, 10]
obs_gap2 = [10, None, None, 18, 30, 32, 26, 20, 14, 12, 11, 10, 10, 10, 10]
obs_gap3 = [10, 11, 14, None, None, None, 26, 20, 14, 12, 11, 10, 10, 10, 10]
obs_gap4 = [10, 11, 14, 18, 30, 32, 26, None, None, None, 11, 10, 10, 10, 10]
obs_gap5 = [10, 11, 14, 18, 30, 32, 26, 20, 14, 12, 11, None, None, None, None]
all_obs = [obs_nogap, obs_gap1, obs_gap2, obs_gap3, obs_gap4, obs_gap5]
modeled_low = [8, 9, 12, 16, 28, 30, 24, 18, 12, 10, 9, 8, 8, 8, 8]
modeled_high = [12, 13, 16, 20, 32, 34, 28, 22, 16, 14, 13, 12, 12, 12, 12]
modeled_shift_late = [10, 10, 10, 11, 14, 18, 30, 32, 26, 20, 14, 12, 11, 10, 10]
modeled_shift_early = [11, 14, 18, 30, 32, 26, 20, 14, 12, 11, 10, 10, 10, 10, 10]
all_modeled = [modeled_low, modeled_high, modeled_shift_late, modeled_shift_early]
lastobs = {"obs": 9.5, "time": 0}  # Most recent observation at simulation start
lastobs_old = {"obs": 9.5, "time": 60}  # Most recent observation 1 hour ago
lastobs_wrongtime = {
    "obs": 9.5,
    "time": -60,
}  # Most recent observation 1 hour in the future (not really possible)
lastobs_NaN = {
    "obs": np.NaN,
    "time": np.datetime64("NaT"),
}  # No valid recent observation
all_lastobs = [lastobs, lastobs_old, lastobs_NaN]

decay_coeff = 120


def test_simple_da():
    m_i = 2
    lo = lastobs_old["obs"]
    lt = lastobs_old["time"]
    m = modeled_low[m_i]

    expected = 10.483673095703125
    adjusted = simple_da_with_decay_py(lo, m, lt, decay_coeff)
    assert adjusted == pytest.approx(expected, 2.3e-06)
