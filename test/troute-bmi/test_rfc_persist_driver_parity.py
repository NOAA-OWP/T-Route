"""Both drivers must walk the same window starts on a contiguous schedule.

t-route runs under ngen through the BMI, so the BMI path is the product-critical one and
it has to agree with the CLI. They express a window's t0 differently: the CLI advances
arithmetically (``network.t0 += dt * nts``, ``AbstractNetwork.new_t0``) while the BMI
parses it from the window's first forcing column (``_build_run_sets``). The persistence
horizon is packed as ``persist_until - window_t0``, so a disagreement there is a
disagreement about when RFC assimilation stops.

Scope, deliberately narrow: this drives the BMI's own partitioner and models the CLI's
arithmetic beside it. It is a parity check on a CONTIGUOUS schedule, not a comparison of
the two real entry points. Both drivers generate their forcing labels from t0 by the
same arithmetic in production -- the BMI in ``_construct_qlats``, the CLI when it
enumerates expected filenames -- so the parse is a round trip rather than a reading of
independent timestamps, and a schedule where they could disagree is one neither driver
builds.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
import troute_nwm_bmi.troute_model as tm

_T0 = datetime(2019, 5, 29, 0, 0)
_DT = 300
_QTS = 12


def _model(nts_cols, max_loop_size, dt=_DT, qts=_QTS):
    """A Model carrying just enough state for _build_run_sets."""
    model = tm.Model.__new__(tm.Model)
    model._config = {
        "compute_parameters": {
            "forcing_parameters": {
                "qts_subdivisions": qts, "dt": dt, "nts": nts_cols * qts,
                "max_loop_size": max_loop_size,
            }
        }
    }
    model._network = SimpleNamespace(t0=_T0)
    model.dt = dt
    return model


def _qlats(n_cols):
    columns = [(_T0 + timedelta(hours=i)).strftime("%Y%m%d%H%M") for i in range(n_cols)]
    return pd.DataFrame(0.5, index=pd.Index([101, 102], name="feature_id"), columns=columns)


def _cli_window_t0s(run_sets):
    """What the CLI would produce: t0 advanced by dt * nts after each window."""
    t0, out = _T0, []
    for run in run_sets:
        out.append(t0)
        t0 += timedelta(seconds=_DT * run["nts"])
    return out


@pytest.mark.parametrize("max_loop_size", [6, 24, 48])
def test_both_drivers_walk_the_same_window_starts(max_loop_size):
    """Parsed-from-forcing and advanced-arithmetically must give the same sequence."""
    run_sets = list(_model(96, max_loop_size)._build_run_sets(_qlats(96)))
    assert [run["t0"] for run in run_sets] == _cli_window_t0s(run_sets)


@pytest.mark.parametrize("max_loop_size", [6, 24, 48])
def test_both_drivers_see_the_same_horizon_remaining(max_loop_size):
    """And therefore the same seconds left of the persistence horizon per window.

    This is the value the packer hands the kernel, so equal sequences here mean the two
    drivers stop assimilating at the same instant.
    """
    persist_until = pd.Timestamp(_T0) + timedelta(days=2)
    run_sets = list(_model(96, max_loop_size)._build_run_sets(_qlats(96)))

    bmi = [(persist_until - pd.Timestamp(run["t0"])).total_seconds() for run in run_sets]
    cli = [(persist_until - pd.Timestamp(t0)).total_seconds() for t0 in _cli_window_t0s(run_sets)]
    assert bmi == cli
    # Strictly decreasing, and negative once past the deadline, which is what stops the
    # kernel re-arming: no window-local clock satisfies a negative budget.
    assert bmi == sorted(bmi, reverse=True)
    assert bmi[0] == 2 * 86400
    assert any(remaining <= 0 for remaining in bmi), "the run must cross the horizon"
