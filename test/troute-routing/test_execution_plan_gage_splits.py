"""The cached execution plan must split at gages regardless of data availability.

The plan is a topological decomposition built once and reused for every forcing
window. Building it from the window's observation frame rather than the network's
gage set meant a first window with no observations produced a plan with no gage
splits at all, and nothing rebuilt it when the feed recovered: the whole run then
routed without splitting at gages, so the in-kernel override could not land at the
gaged nexus.

In normal operation the two sets are identical, because the observation frame is
built by left-joining the gage crosswalk and a gage with no data still gets an
all-NaN row. They diverge only when a window yields no observations at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from troute.routing.compute import AssimilationData, ExecutionPlan, NetworkTopology, ReachData, WaterbodyData


# A short chain with one gage in the middle:  4 -> 3 -> 2(gage) -> 1(tailwater)
CONNECTIONS = {4: [3], 3: [2], 2: [1], 1: []}
GAGE_SEGMENT = 2


def _assimilation(usgs_index, gage_segments):
    """AssimilationData carrying only what the plan decomposition reads."""
    empty = pd.DataFrame()
    usgs = (
        pd.DataFrame(index=pd.Index(usgs_index, dtype="int64"), columns=[0], data=1.0)
        if usgs_index
        else empty
    )
    return AssimilationData(
        reservoir_usgs_df=empty, reservoir_usgs_param_df=empty,
        reservoir_usace_param_df=empty, reservoir_usace_df=empty,
        reservoir_usbr_df=empty, reservoir_usbr_param_df=empty,
        reservoir_rfc_df=empty, reservoir_rfc_param_df=empty,
        great_lakes_df=empty, great_lakes_param_df=empty,
        great_lakes_climatology_df=empty,
        usgs_df=usgs, lastobs_df=empty,
        gage_segments=set(gage_segments),
    )


def _reaches(assimilation) -> list[list[int]]:
    """Decompose the chain into routing paths under the given assimilation data."""
    reverse = {2: [3], 3: [4], 1: [2], 4: []}
    topology = NetworkTopology(
        connections=CONNECTIONS, reverse_connections=reverse,
        paths_by_tailwater={}, connections_by_tw={1: CONNECTIONS},
    )
    plan = ExecutionPlan.__new__(ExecutionPlan)
    paths = plan._clean_compute_jobs(
        {0: {1: set(CONNECTIONS)}}, topology,
        WaterbodyData(pd.DataFrame(), pd.DataFrame()), assimilation,
    )
    return [list(r) for r in paths[0][1]]


def _splits_at_gage(reaches) -> bool:
    """True when some reach ends at the gage, i.e. the plan split there."""
    return any(r and r[-1] == GAGE_SEGMENT for r in reaches)


def test_splits_at_gage_when_observations_are_present():
    assert _splits_at_gage(_reaches(_assimilation([GAGE_SEGMENT], [GAGE_SEGMENT])))


def test_splits_at_gage_when_this_window_has_no_observations():
    """The regression: an empty observation frame must not cost the gage split.

    Reproduces the outage case. Without the static gage set the decomposition sees
    an empty frame, takes the no-gages branch, and the plan it caches for the whole
    run never splits at the gage.
    """
    assert _splits_at_gage(_reaches(_assimilation([], [GAGE_SEGMENT])))


def test_no_gages_configured_means_no_gage_split():
    """A network with no gages at all must still decompose, just without splits."""
    reaches = _reaches(_assimilation([], []))
    assert reaches, "decomposition must still produce reaches"
    assert not _splits_at_gage(reaches)


def test_falls_back_to_the_observation_frame():
    """Callers that predate gage_segments keep working off the observation frame."""
    assert _splits_at_gage(_reaches(_assimilation([GAGE_SEGMENT], [])))
