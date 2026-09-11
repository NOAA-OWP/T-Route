import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from nwm_routing.__main__ import new_nwm_q0
from nwm_routing.nwm_route import nwm_route
from nwm_routing.preprocess import nwm_forcing_preprocess
from test import find_cwd, temporarily_change_dir


def _route(
    return_courant: bool,
    nhd_test_network: Dict[str, Any],
    nhd_built_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
) -> pd.DataFrame:
    """Run the main routing computation once; return (q0, run_results)."""
    path = nhd_test_network["path"]

    connections = nhd_built_test_network["connections"]
    rconn = nhd_built_test_network["rconn"]
    wbody_conn = nhd_built_test_network["wbody_conn"]
    reaches_bytw = nhd_built_test_network["reaches_bytw"]
    independent_networks = nhd_built_test_network["independent_networks"]
    param_df = nhd_built_test_network["param_df"]
    unrefactored_topobathy_df = nhd_built_test_network["unrefactored_topobathy_df"]
    refactored_reaches = nhd_built_test_network["refactored_reaches"]
    refactored_diffusive_domain = nhd_built_test_network["refactored_diffusive_domain"]
    topobathy_df = nhd_built_test_network["topobathy_df"]
    diffusive_network_data = nhd_built_test_network["diffusive_network_data"]
    waterbody_type_specified = nhd_built_test_network["waterbody_type_specified"]
    waterbody_types_df = nhd_built_test_network["waterbody_types_df"]

    forcing_parameters = nhd_test_network["forcing_parameters"]
    compute_parameters = nhd_test_network["compute_parameters"]
    waterbody_parameters = nhd_test_network["waterbody_parameters"]

        
    parallel_compute_method = compute_parameters.get("parallel_compute_method", None)
    subnetwork_target_size = compute_parameters.get("subnetwork_target_size", 1)
    cpu_pool = compute_parameters.get("cpu_pool", None)
    compute_kernel = compute_parameters.get("compute_kernel", "V02-caching")
    assume_short_ts = compute_parameters.get("assume_short_ts", False)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)

    run_sets = [nhd_qlat_data]
    t0 = run_sets[0].get("t0")
    dt = forcing_parameters.get('dt')
    nts = run_sets[0].get("nts")

    t0 = warmstart_nhd_test["t0"]
    q0 = warmstart_nhd_test["q0"]
    lastobs_df = warmstart_nhd_test["lastobs_df"]
    waterbodies_df = warmstart_nhd_test["waterbodies_df"]
    da_parameter_dict = warmstart_nhd_test["da_parameter_dict"]

    subnetwork_list = [None, None, None]

    hybrid_parameters = nhd_test_network["hybrid_parameters"]
    compute_parameters = nhd_test_network["compute_parameters"]
    data_assimilation_parameters = nhd_test_network["data_assimilation_parameters"]
    segment_index = nhd_built_test_network["param_df"].index
    link_gage_df = nhd_built_test_network["link_gage_df"]
    usgs_lake_gage_crosswalk = nhd_built_test_network["usgs_lake_gage_crosswalk"]
    usace_lake_gage_crosswalk = nhd_built_test_network["usace_lake_gage_crosswalk"]
    link_lake_crosswalk = nhd_built_test_network["link_lake_crosswalk"]

    run_sets = [nhd_qlat_data]
    da_sets = [{"usgs_timeslice_files": []}]

    break_network_at_waterbodies = nhd_built_test_network["break_network_at_waterbodies"]

    cpu_pool = compute_parameters.get("cpu_pool", None)

    with temporarily_change_dir(path):
        (
            qlats, 
            usgs_df, 
            reservoir_usgs_df, 
            reservoir_usgs_param_df,
            reservoir_usace_df,
            reservoir_usace_param_df,
            coastal_boundary_depth_df
        ) = nwm_forcing_preprocess(
            run_sets[0],
            forcing_parameters,
            hybrid_parameters,
            da_sets[0] if data_assimilation_parameters else {},
            data_assimilation_parameters,
            break_network_at_waterbodies,
            segment_index,
            link_gage_df,
            usgs_lake_gage_crosswalk, 
            usace_lake_gage_crosswalk,
            link_lake_crosswalk,
            lastobs_df.index,
            cpu_pool,
            t0,
        )

        run_results, subnetwork_list = nwm_route(
            connections,
            rconn,
            wbody_conn,
            reaches_bytw,
            parallel_compute_method,
            compute_kernel,
            subnetwork_target_size,
            cpu_pool,
            t0,
            dt,
            nts,
            qts_subdivisions,
            independent_networks,
            param_df,
            q0,
            qlats,
            pd.DataFrame(), #empty dataframe for ET .. not supported here
            0.0, # SSOUT not supported in run
            usgs_df,
            lastobs_df,
            reservoir_usgs_df,
            reservoir_usgs_param_df,
            reservoir_usace_df,
            reservoir_usace_param_df,
            pd.DataFrame(), #empty dataframe for USBR data...not needed unless running via BMI
            pd.DataFrame(), #empty dataframe for USBR data...not needed unless running via BMI
            pd.DataFrame(), #empty dataframe for RFC data...not needed unless running via BMI
            pd.DataFrame(), #empty dataframe for RFC param data...not needed unless running via BMI
            pd.DataFrame(), #empty dataframe for great lakes data...
            pd.DataFrame(), #empty dataframe for great lakes param data...
            pd.DataFrame(), #empty dataframe for great lakes climatology data...
            da_parameter_dict,
            assume_short_ts,
            return_courant,
            waterbodies_df,
            waterbody_parameters,
            waterbody_types_df,
            waterbody_type_specified,
            diffusive_network_data,
            topobathy_df,
            refactored_diffusive_domain,
            refactored_reaches,
            subnetwork_list,
            coastal_boundary_depth_df,
            unrefactored_topobathy_df,
            firstRun=False,
            logFileName ='troute_run_log.txt'             
        )

        q0 = new_nwm_q0(run_results)

    return q0, run_results


def _route_q0(*args, **kwargs) -> pd.DataFrame:
    """Back-compat wrapper: the q0 alone, for callers that do not need the raw results."""
    return _route(*args, **kwargs)[0]


@pytest.mark.xfail(
    reason=(
        "NHD q0 baseline is contested, and NEITHER side is currently defensible. The "
        "committed golden has h0 mean 1.26 m / max 923.7 m -- 924 m of water in a "
        "channel is not physical. The current run gives h0 mean 0.003 m / max 0.79 m, "
        "which is 3 mm of water in a river and no better. Same 10907 segment ids, "
        "98.3% of rows differ.\n\n"
        "It is NOT a regression from the scaling-DA or diffusive work: development "
        "cannot produce a comparison at all, because its V3/V4 path dies before "
        "routing on the 3-column q0 the kernel rejects (fixed here, see "
        "build_channel_initial_state). This branch is the first runnable state of "
        "this test, so there is no prior value to diff against.\n\n"
        "It traces to the reservoir-handling refactor (waterbody reach-split topology "
        "in compute.py plus the vfp/reservoir work already on development), which is a "
        "reservoir-baseline question, not a DA one. Regenerating the golden here would "
        "bless one implausible number over another; it was tried once and reverted.\n\n"
        "CHRTOUT CANNOT ADJUDICATE THIS, checked 2026-08-03. The obvious next step -- "
        "score both sides against the NWM CHRTOUT streamflow this domain ships -- does "
        "not work, because the three id spaces are disjoint and the fixtures carry no "
        "crosswalk: CHRTOUT has 11248 features keyed by feature_id, the HYDRO_RST "
        "restart has 11141 positional `links` and no ids at all, and the run routes "
        "10907 segments. Compared anyway, BOTH sides sit ~98% below CHRTOUT (golden "
        "PBIAS -97.4%, corr 0.09; current -98.4%, corr -0.05), which measures the "
        "domain mismatch, not the routing.\n\n"
        "Two leads for whoever picks this up. (1) The restart itself is non-physical: "
        "hlink has mean -2.38 m, i.e. negative depth on input, so the absurd h0 on BOTH "
        "sides is garbage-in, not a routing bug. (2) The disagreement is domain-wide, "
        "not lake-localized -- all 30 link_lake_crosswalk segments differ, but so do "
        "10692 of 10877 non-lake segments -- and the sharpest signal is that max|dqd0| "
        "is 19.3055 against a golden max qd0 of 19.3062, i.e. the single segment "
        "carrying the domain's peak flow in the golden drops to ~0 in the current run. "
        "Start there, not with a bulk diff.\n\n"
        "xfail rather than skip so a baseline fix shows up as XPASS instead of staying "
        "invisible."
    ),
    strict=False,
)
def test_nwm_route_execution(
    nhd_test_network: Dict[str, Any],
    nhd_built_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
    nhd_validation_files: Dict[str, Any],
    expected_q0: pd.DataFrame,
):
    """Test the main routing computation"""
    q0 = _route_q0(
        nhd_test_network["compute_parameters"].get("return_courant", False),
        nhd_test_network,
        nhd_built_test_network,
        warmstart_nhd_test,
        nhd_qlat_data,
    )

    pd.testing.assert_frame_equal(
       q0,
       expected_q0,
       check_dtype=False,
       check_exact=False,
       rtol=1e-5
   )


def test_return_courant_does_not_change_routed_flows(
    nhd_test_network: Dict[str, Any],
    nhd_built_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
):
    """``return_courant`` asks the kernel to also report cn/ck/X. It must not alter routing.

    Compares the two runs against each other rather than against a stored golden, so this
    stays valid independently of the NHD golden's own state.

    SCOPE, because this test is weaker than it looks: it does NOT gate the out_buf sizing
    bug it was written for. ``out_buf`` was allocated 3 columns wide while
    ``compute_reach_kernel`` writes cn/ck/X into columns 3..5 under ``nogil`` and
    ``boundscheck(False)``. For a C-contiguous (n, 3) buffer ``out_buf[i, 3:6]`` aliases
    ``out_buf[i+1, 0:3]``, but the kernel overwrites row i+1's flow/velocity/depth on the
    very next iteration, before the read-back loop runs -- so the values this test compares
    come out identical either way, and it passes with the bug present. The genuine defect is
    the LAST segment of the longest reach writing 12 bytes past the end of the allocation,
    which is undefined behavior and not reliably observable from Python.
    ``test_out_buf_covers_every_column_the_kernel_writes`` is the real gate.

    Keep this one anyway: it is a cheap, direct check that switching the flag does not
    perturb routed flows, which is the user-visible contract.
    """
    args = (nhd_test_network, nhd_built_test_network, warmstart_nhd_test, nhd_qlat_data)

    pd.testing.assert_frame_equal(_route_q0(True, *args), _route_q0(False, *args))


def test_out_buf_covers_every_column_the_kernel_writes():
    """``out_buf`` must be at least as wide as the highest column ``compute_reach_kernel`` writes.

    This is the actual regression gate for the out-of-bounds write, and it reads the kernel
    source rather than running it, because the bug is not observable from Python: the aliased
    writes are repaired by the next loop iteration, and the one genuinely out-of-bounds write
    (last segment of the longest reach) is undefined behavior that usually lands on slack
    bytes. ``compute_reach_kernel`` is ``nogil`` and ``boundscheck(False)``, so neither Cython
    nor the compiler will catch a short buffer -- the invariant has to be asserted somewhere,
    and the only cheap place is against the source itself.
    """
    import re
    from pathlib import Path

    pyx = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src/troute-routing/troute/routing/fast_reach/mc_reach.pyx"
        if candidate.exists():
            pyx = candidate
            break
    assert pyx is not None, "could not locate mc_reach.pyx from the test file"
    src = pyx.read_text()

    # Highest column index written through the kernel's output_buf parameter.
    written = [int(m) for m in re.findall(r"output_buf\[\s*\w+\s*,\s*(\d+)\s*\]", src)]
    assert written, "found no output_buf writes -- has the kernel been renamed?"
    needed = max(written) + 1

    # Columns the caller actually allocates.
    alloc = re.search(r"out_buf\s*=\s*np\.full\(\s*\(\s*max_buff_size\s*,\s*(\d+)\s*\)", src)
    assert alloc, "could not find the out_buf allocation"
    allocated = int(alloc.group(1))

    assert allocated >= needed, (
        f"out_buf is allocated {allocated} columns but compute_reach_kernel writes up to "
        f"column {needed - 1}. Under nogil + boundscheck(False) the surplus writes alias the "
        f"next row and, on the longest reach, run past the end of the allocation."
    )


def test_return_courant_populates_the_courant_slot(
    nhd_test_network: Dict[str, Any],
    nhd_built_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
):
    """With ``return_courant``, r[2] must actually carry cn/ck/X -- not the placeholder 0.

    The kernel has always computed the Courant diagnostics, but ``compute_network_structured``
    dropped them and returned a literal ``0`` for r[2], while ``output.py`` builds
    ``pd.DataFrame(r[2], index=r[0], columns=courant_columns)`` from it -- so requesting
    Courant metrics could never have produced a usable frame. This asserts the shape and
    ordering that consumer expects: (n_segments, nts*3), timestep-major, cn/ck/X within each
    timestep.
    """
    args = (nhd_test_network, nhd_built_test_network, warmstart_nhd_test, nhd_qlat_data)
    _, results = _route(True, *args)
    _, results_off = _route(False, *args)
    assert results, "no run results -- the assertions below would pass vacuously"

    nts = len(nhd_qlat_data["qlat"].columns) if "qlat" in nhd_qlat_data else None

    for r, r_off in zip(results, results_off):
        courant = r[2]
        assert not np.isscalar(courant), "r[2] is still the placeholder scalar"
        assert courant.shape[0] == r[0].shape[0], "one Courant row per segment"
        assert courant.shape[1] % 3 == 0, "columns must be whole (cn, ck, X) triples"
        # flowveldepth is nts*4 wide over the same timesteps, so nts*3 must line up with it
        assert courant.shape[1] // 3 == r[1].shape[1] // 4, "timestep count disagrees with flowveldepth"
        assert np.isfinite(courant).all(), "Courant diagnostics contain non-finite values"
        # ck is a wave celerity in m/s: strictly positive wherever the reach is wet.
        ck = courant[:, 1::3]
        assert (ck >= 0).all(), "negative wave celerity"
        assert ck.max() > 0, "all celerities zero -- the columns were never written"
        # and the placeholder must still be returned when the flag is off
        assert np.isscalar(r_off[2]) and r_off[2] == 0
