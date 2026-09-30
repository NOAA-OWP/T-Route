"""A level pool whose surface is small against its outflow must stay finite and conserve.

One step of overtopping discharge moves a 1,100 m2 pool meters, beyond one explicit
Runge-Kutta step; a pool that fails publishes NaN, then zero, and the run still exits 0.
"""

from __future__ import annotations

from array import array

import numpy as np
import pytest

from troute.network.reservoirs.levelpool.levelpool import MC_Levelpool
from troute.routing.compute import NWM_DAM_LENGTH_MULTIPLIER

# A low-head dam's level-pool parameters (waterbody 1285260065444388). Its surface is
# 1,124 m2, which is what makes one step of overtopping discharge move it so far.
LAKE_NUMBER = 1285260065444388
AREA_KM2 = 0.0011
MAX_ELEVATION = 16.578815460205078
WEIR_ELEVATION = 16.426416397094727
ORIFICE_ELEVATION = 15.283415794372559
WEIR_COEFFICIENT = 0.4
WEIR_LENGTH = 33.52799987792969
ORIFICE_COEFFICIENT = 0.1
ORIFICE_AREA = 0.5
INITIAL_FRACTIONAL_DEPTH = 0.9

DT = 300.0
INFLOW = 1.7631          # enough to keep this pool overtopping
START_ELEVATION = 16.5502
STEPS = 48


def _pool(water_elevation: float) -> MC_Levelpool:
    args = [
        AREA_KM2, MAX_ELEVATION, ORIFICE_AREA, ORIFICE_COEFFICIENT, ORIFICE_ELEVATION,
        WEIR_COEFFICIENT, WEIR_ELEVATION, WEIR_LENGTH, INITIAL_FRACTIONAL_DEPTH, 0.0,
        water_elevation,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    return MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)


def _route(steps: int = STEPS, inflow: float = INFLOW) -> tuple[list[float], list[float]]:
    pool = _pool(START_ELEVATION)
    outflows, elevations = [], []
    for _ in range(steps):
        outflow, elevation = pool.run(inflow, 0.0, int(DT))
        outflows.append(float(outflow))
        elevations.append(float(elevation))
    return outflows, elevations


def test_the_pool_stays_finite_while_it_overtops() -> None:
    """An overtopping step must not drive the water surface elevation NaN."""
    outflows, elevations = _route()
    assert np.all(np.isfinite(elevations)), (
        f"elevation went non-finite at step {int(np.argmax(~np.isfinite(elevations)))}"
    )
    assert np.all(np.isfinite(outflows))


def test_the_pool_keeps_passing_water() -> None:
    """A pool taking 1.76 cms for four hours cannot release nothing.

    Zero outflow with a live inflow is the signature of the failure this pins: the routine
    publishes 0 for a NaN elevation, so the run looks healthy and the water is gone.
    """
    outflows, _ = _route()
    assert min(outflows) > 0.0
    assert np.mean(outflows) == pytest.approx(INFLOW, rel=0.25)


def test_the_pool_conserves_its_water() -> None:
    """Inflow minus outflow over the run has to be the change in storage."""
    outflows, elevations = _route()
    sap = AREA_KM2 * 1.0e6
    delivered = sum(outflows) * DT
    entered = INFLOW * DT * len(outflows)
    stored = (elevations[-1] - START_ELEVATION) * sap
    assert entered - delivered - stored == pytest.approx(0.0, abs=0.001 * entered)


def _discharge(elev: float) -> float:
    """The elevation-discharge relationship, as the kernel defines it."""
    weir_depth = max(min(elev - WEIR_ELEVATION, MAX_ELEVATION - WEIR_ELEVATION), 0.0)
    orifice_head = max(elev - ORIFICE_ELEVATION, 0.0)
    q = (ORIFICE_COEFFICIENT * ORIFICE_AREA * np.sqrt(2 * 9.81 * orifice_head)
         + WEIR_COEFFICIENT * WEIR_LENGTH * weir_depth ** 1.5)
    if elev > MAX_ELEVATION:
        q += WEIR_COEFFICIENT * (WEIR_LENGTH * 10.0) * (elev - MAX_ELEVATION) ** 1.5
    return float(q)


def test_a_stable_pool_publishes_its_own_discharge() -> None:
    """A pool large enough for one stable step must route on its own outlet.

    The published flow is the step's mean, so it lies between the discharge at each end
    of the step, give or take the flow one float32 ulp of elevation is worth (ulp * sap/dt).
    """
    area_km2, start = 0.1, WEIR_ELEVATION + 0.1
    args = [
        area_km2, MAX_ELEVATION, ORIFICE_AREA, ORIFICE_COEFFICIENT, ORIFICE_ELEVATION,
        WEIR_COEFFICIENT, WEIR_ELEVATION, WEIR_LENGTH, INITIAL_FRACTIONAL_DEPTH, 0.0,
        start,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    outflow, elevation = pool.run(INFLOW, 0.0, int(DT))

    lo, hi = sorted((_discharge(start), _discharge(float(elevation))))
    band = (area_km2 * 1.0e6 / DT) * float(np.spacing(np.float32(start)))
    assert lo - band <= float(outflow) <= hi + band, (
        f"{float(outflow)} outside [{lo - band}, {hi + band}]"
    )
    # The pool is large enough that one step cannot move it far.
    assert float(elevation) == pytest.approx(start, abs=5e-3)


def test_a_dead_pool_releases_nothing() -> None:
    """Below the orifice invert every term is off, and the guards must not invent flow."""
    pool = _pool(ORIFICE_ELEVATION - 1.0)
    outflow, elevation = pool.run(0.0, 0.0, int(DT))
    assert outflow == pytest.approx(0.0)
    assert np.isfinite(elevation)


def test_a_pool_at_equilibrium_stays_there() -> None:
    """At equilibrium inflow and discharge cancel, so a step sized on their difference
    alone is taken whole while discharge still answers elevation steeply. A pool a
    millimeter off equilibrium must stay near it.
    """
    equilibrium = 16.755184
    pool = _pool(equilibrium - 0.001)
    _, elevation = pool.run(11.0, 0.0, int(DT))
    assert abs(float(elevation) - equilibrium) < 0.01


def test_a_pool_cannot_release_water_it_does_not_hold() -> None:
    """Below the orifice invert there is nothing left to draw on.

    A deep weir makes the step selector permissive, so a shallow pool under one gets a
    single step, which an orifice sized to empty it would overdraw.
    """
    area_km2, orifice_elevation = 0.0011, 0.0
    start = 0.01
    available = (start - orifice_elevation) * area_km2 * 1.0e6
    args = [
        area_km2, 110.0, 1.0, 0.6, orifice_elevation,
        WEIR_COEFFICIENT, 10.0, 33.5, INITIAL_FRACTIONAL_DEPTH, 0.0, start,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    outflow, elevation = pool.run(0.0, 0.0, int(DT))
    assert float(outflow) * DT <= available * 1.001, (
        f"released {float(outflow) * DT:.2f} m3 of {available:.2f} available"
    )
    assert float(elevation) >= orifice_elevation - 1e-4


def test_a_pool_past_the_sub_step_cap_still_stops_at_the_invert() -> None:
    """The cap bounds the cost, so it has to fail into a state that is still physical."""
    args = [
        1.0e-5, MAX_ELEVATION, ORIFICE_AREA, ORIFICE_COEFFICIENT, ORIFICE_ELEVATION,
        WEIR_COEFFICIENT, WEIR_ELEVATION, WEIR_LENGTH, INITIAL_FRACTIONAL_DEPTH, 0.0, 18.0,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    outflow, elevation = pool.run(0.0, 0.0, int(DT))
    assert np.isfinite(outflow)
    assert np.isfinite(elevation)
    assert float(elevation) >= ORIFICE_ELEVATION - 1e-4


def test_the_binding_collapses_the_inflow_endpoints() -> None:
    """``bind_lp.f90`` passes one inflow as both endpoints of the step, so the sub-steps'
    inflow interpolation is inert under t-route. If the binding ever passes distinct
    endpoints, this fails and the interpolation needs its own test.
    """
    both = _pool(16.55)
    both.run(10.0, 0.0, int(DT))
    _, ramped = both.run(30.0, 0.0, int(DT))

    once = _pool(float(_pool(16.55).run(10.0, 0.0, int(DT))[1]))
    _, flat = once.run(30.0, 0.0, int(DT))

    assert float(ramped) == pytest.approx(float(flat), abs=1e-6)


def test_a_shallow_pool_under_a_deep_weir_settles() -> None:
    """An outlet is steepest just above its own threshold. A probe scaled to a deep weir
    reads the orifice as gentle, and a pool a millimeter off equilibrium then alternates
    between two elevations, pulsing discharge.
    """
    args = [
        0.0011, 20.0, 1.0, 0.6, 0.0,
        WEIR_COEFFICIENT, 10.0, 33.5, INITIAL_FRACTIONAL_DEPTH, 0.0, 0.011,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    elevations = [float(pool.run(0.265766815, 0.0, int(DT))[1]) for _ in range(12)]
    assert np.all(np.isfinite(elevations))
    # Settled: the last few steps stop moving.
    assert max(elevations[-4:]) - min(elevations[-4:]) < 0.005, elevations


def test_a_pool_at_a_high_elevation_still_conserves() -> None:
    """Single precision resolves a quarter of a millimeter near 3000 m, so sub-steps
    below that would round away while their discharge is published. Accumulated in
    single precision, this 100 m2 pool loses about 4% of its inflow over the run.
    """
    area_km2 = 1.0e-4
    sap = area_km2 * 1.0e6
    start, inflow = 3000.02, 0.1
    args = [
        area_km2, 3110.0, 1.0, 0.6, 3000.0,
        WEIR_COEFFICIENT, 3010.0, 33.5, INITIAL_FRACTIONAL_DEPTH, 0.0, start,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    steps = 200
    outflows, elevation = [], start
    for _ in range(steps):
        outflow, elevation = pool.run(inflow, 0.0, int(DT))
        outflows.append(float(outflow))
    entered = inflow * DT * steps
    delivered = sum(outflows) * DT
    stored = (float(elevation) - start) * sap
    assert entered - delivered - stored == pytest.approx(0.0, abs=0.001 * entered)


def test_a_long_run_of_small_steps_loses_only_what_the_state_cannot_hold() -> None:
    """A storage change below the single-precision elevation's resolution cannot persist,
    so the imbalance is bounded by `ulp(H) * sap` per call. Reconciling the flux against
    the rounded elevation would close it and send a large lake's storage downstream.
    """
    sap = 0.0011 * 1.0e6
    start, inflow, steps = 3000.01, 0.01, 1000
    args = [
        0.0011, 3110.0, 1.0, 0.6, 3000.0,
        WEIR_COEFFICIENT, 3010.0, 33.5, INITIAL_FRACTIONAL_DEPTH, 0.0, start,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    delivered, elevation = 0.0, start
    for _ in range(steps):
        outflow, elevation = pool.run(inflow, 0.0, int(DT))
        delivered += float(outflow) * DT
    entered = inflow * DT * steps
    stored = (float(elevation) - start) * sap
    budget = steps * float(np.spacing(np.float32(start))) * sap
    assert abs(entered - delivered - stored) <= budget, (
        f"imbalance {entered - delivered - stored:.1f} m3 exceeds the "
        f"{budget:.1f} m3 the elevation's resolution allows"
    )


def test_an_outlet_opening_mid_step_is_resolved() -> None:
    """A threshold crossed mid-step is invisible to probes at either end. Starting below
    the orifice invert and ending above it, both read a flat curve, and a step taken
    whole releases a third more than a converged reference.
    """
    args = [
        0.0011, 30.0, 1.0, 0.6, 10.0,
        WEIR_COEFFICIENT, 20.0, 33.5, INITIAL_FRACTIONAL_DEPTH, 0.0, 9.0,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    coarse = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    outflow, elevation = coarse.run(10.0, 0.0, int(DT))

    pieces = 100
    fine = MC_Levelpool(0, LAKE_NUMBER, array("l", []), list(args), 1)
    reference, fine_out = 9.0, 0.0
    for _ in range(pieces):
        o, reference = fine.run(10.0, 0.0, int(DT / pieces))
        fine_out += float(o) * (DT / pieces)

    assert float(elevation) == pytest.approx(float(reference), abs=0.02)
    assert float(outflow) * DT == pytest.approx(fine_out, rel=0.05)


def test_a_pool_that_starts_non_finite_recovers() -> None:
    """A NaN elevation from a state file fails every comparison in the routine, so an
    unguarded pool publishes zero for the rest of the run. It restarts at its orifice
    invert, as a cold start seeds it.
    """
    pool = _pool(float("nan"))
    outflow, elevation = pool.run(INFLOW, 0.0, int(DT))
    assert np.isfinite(elevation)
    assert np.isfinite(outflow)
    # One step of inflow from the invert leaves it below the weir crest.
    assert ORIFICE_ELEVATION < float(elevation) < WEIR_ELEVATION


def test_a_non_finite_inflow_leaves_the_pool_where_it_was() -> None:
    """A NaN inflow says nothing about the step. Carried into the elevation, it would
    reach the non-finite guard on the next call and discard the pool's storage."""
    pool = _pool(START_ELEVATION)
    outflow, elevation = pool.run(float("nan"), 0.0, int(DT))
    assert float(elevation) == pytest.approx(START_ELEVATION)
    assert float(outflow) == pytest.approx(_discharge(START_ELEVATION), rel=1e-5)


def test_an_orifice_above_the_weir_crest_creates_no_water() -> None:
    """The storage bound floors a pool at its lowest outlet, here the weir crest.
    Flooring it at the orifice would lift a pool between the two by meters of water it
    never received."""
    area_km2, start, inflow = 1.0, 187.0, 5.0
    args = [
        area_km2, 200.0, 1.0, 0.1, 190.0,
        WEIR_COEFFICIENT, 185.0, 10.0, INITIAL_FRACTIONAL_DEPTH, 0.0, start,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    outflow, elevation = pool.run(inflow, 0.0, int(DT))
    stored = (float(elevation) - start) * area_km2 * 1.0e6
    assert inflow * DT - float(outflow) * DT - stored == pytest.approx(0.0, abs=20.0)


def test_a_large_pool_keeps_a_small_net_inflow() -> None:
    """A day of 1 cms into a closed 300 km2 pool raises it 0.29 mm. Carried in single
    precision near 100 m the elevation cannot move by less than 7.6e-6 m a step, so
    every step's storage change would round away and the pool would stay put."""
    area_km2, start, inflow, steps = 300.0, 100.0, 1.0, 288
    args = [
        area_km2, 103.0, 1.0, 0.1, 101.0,
        WEIR_COEFFICIENT, 102.0, 10.0, INITIAL_FRACTIONAL_DEPTH, 0.0, start,
        NWM_DAM_LENGTH_MULTIPLIER,
    ]
    pool = MC_Levelpool(0, LAKE_NUMBER, array("l", []), args, 1)
    elevation = start
    for _ in range(steps):
        outflow, elevation = pool.run(inflow, 0.0, int(DT))
        assert outflow == 0.0
    stored = (float(elevation) - start) * area_km2 * 1.0e6
    assert stored == pytest.approx(inflow * DT * steps, rel=1e-6)
