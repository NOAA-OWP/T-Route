"""Per-gage tree *setup* for the simple-scaling streamflow data assimilation.

This package (in ``troute-network``) builds the per-gage upstream trees the DA
distributes corrections over: BFS upstream from each USGS gage through the network
topology, stopping at other gages and waterbodies (Edge Case 1). It mirrors
``DataAssimilation.py`` -- the DA *setup* lives in troute-network.

The DA *correction kernel* that consumes these trees lives in troute-routing,
beside the other routing/DA kernels:
:func:`troute.routing.fast_reach.scaling_da.apply_scaling_da`.
"""

from __future__ import annotations

from troute.scaling_da.build_trees import build_gage_trees, build_gage_trees_from_mappings
from troute.scaling_da.gage_tree import GageTree, build_one_gage_tree
from troute.scaling_da.preprocess import (
    ScalingDASetup,
    build_scaling_da_setup,
    timeslice_station_roster,
)

__all__ = [
    "GageTree",
    "ScalingDASetup",
    "build_gage_trees",
    "build_gage_trees_from_mappings",
    "build_one_gage_tree",
    "build_scaling_da_setup",
    "timeslice_station_roster",
]
