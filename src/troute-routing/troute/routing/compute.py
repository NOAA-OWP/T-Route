"""
Routing terminology
-------------------

reach
    A directed river segment used as the fundamental routing unit.
network
    The full directed graph representing all routing topology, potentially
    consisting of multiple disconnected trees (a forest).
tree
    An individual drainage network with a single downstream tailwater/root.
tailwater
    The downstream root node of a tree that defines the outlet boundary condition.
partition
    A connected subset of a tree grouped for dependency management and execution.
routing_level
    A hierarchical rank within a tree where level 0 is the tailwater/root and
    increasing levels move upstream.
routing_path
    A contiguous sequence of reaches uninterrupted by confluences, waterbodies,
    or points of interest.
computation_job
    A set of routing paths passed together to the routing kernel for execution.
computation_batch
    A collection of computation jobs that share the same routing level and can
    be executed concurrently.
execution_plan
    An ordered mapping of computation batches and their dependency relationships
    used to orchestrate network execution.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from functools import cached_property, partial
from typing import Any, Callable, Literal, Self, Sequence, TYPE_CHECKING, Iterable, TypedDict, Union, cast, get_args
from joblib import delayed, Parallel
from datetime import datetime
import pandas as pd
import numpy as np

import troute.nhd_network as nhd_network
from troute.routing.fast_reach.mc_reach import compute_network_structured
import troute.routing.diffusive_utils_v02 as diff_utils
# Compiled f2py/Cython extension with no stubs -- pyright cannot see the
# submodule symbol even though it imports fine at runtime.
from troute.routing.fast_reach import diffusive  # pyright: ignore[reportAttributeAccessIssue]

import logging
LOG = logging.getLogger("TROUTE")

if TYPE_CHECKING:
    from typing import Annotated
    from numpy.typing import NDArray

    # Typed numpy array aliases. NDArray encodes the element dtype (what the MC
    # kernel's typed memoryviews enforce) but not shape; use these throughout
    # the module in annotations in place of the untyped `np.ndarray`.
    Float32Array = NDArray[np.float32]
    Float64Array = NDArray[np.float64]
    Int32Array = NDArray[np.int32]
    Int64Array = NDArray[np.int64]
    IntpArray = NDArray[np.intp]
    BoolArray = NDArray[np.bool_]
    ObjectArray = NDArray[np.object_]

    # Domain type aliases. Type-check-only: with `from __future__ import
    # annotations` every use is a string annotation, so they need not exist at
    # runtime (do not reference them as values, e.g. in cast()/isinstance()).
    # A single routing link
    ReachId = int
    # The outlet of a RoutingPath or tree
    TailwaterId = ReachId
    # A connected subset of a tree grouped for dependency management and execution.
    Partition = set[ReachId]
    # A contiguous sequence of reaches uninterrupted by confluences, waterbodies, or points of interest.
    RoutingPath = list[ReachId]
    # A list of routing paths ordered such that upstream paths are run before any of their downstream dependent paths
    OrderedRoutingPaths = list[RoutingPath]
    # A hierarchical rank within a tree where level 0 is the tailwater/root and increasing levels move upstream.
    RoutingLevel = int
    # The connection between a reach and downstream reach(es)
    Adjacency = dict[ReachId, list[ReachId]]
    # Network connectivity (reach -> downstream reaches) for the full routing graph or a subgraph
    DownstreamGraph = Adjacency
    # The connection between a reach and upstream reach(es)
    UpstreamAdjacency = dict[ReachId, list[ReachId]]
    # Network connectivity (reach -> upstream reaches) for the full routing graph or a subgraph
    UpstreamGraph = UpstreamAdjacency


PARALLEL_COMPUTE_METHODS = Literal[
    "by-subnetwork-jit-clustered",
    "by-subnetwork-jit",
    "by-network",
    "serial",
    "bmi"
]
_compute_func_map = {
    "V02-structured": compute_network_structured,
}


# Sentinel empties reused by _prep_reservoir_da_dataframes for the all-DA-disabled
# case (the CONUS benchmark path). The function is called once per cluster prep
# (~1200 times per CONUS run); each else-branch was constructing fresh
# pd.DataFrame() and pd.DataFrame().to_numpy().reshape(0,) sentinels (~100 µs each,
# 31 per call) -- ~4 s of wasted BlockManager allocation across the run.
# These constants are never mutated: they are empty (0-size), so the
# to_numpy(dtype=...) views taken of them downstream have nothing to mutate,
# and the kernel treats its array inputs as read-only.
_EMPTY_F64 = np.empty(0, dtype=np.float64)
_EMPTY_DF = pd.DataFrame()
_EMPTY_GL_DF = pd.DataFrame(columns=["lake_id", "time", "Discharge"])
_EMPTY_LIST: list = []
_QLAT_LOC_MAP = {"top": 0, "middle": 1, "bottom": 2}

### OBJECT DEFINITIONS ###


class BoundaryCondition(TypedDict):
    """A storage container for flow values between reaches in different RoutingLevels."""

    # None until the producing routing level publishes results via
    # BoundaryConditionStore.update (and again after clear_data).
    results: Float32Array | None
    position_index: int


def _concat_unique_sorted(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-job frames, dropping duplicated indices (shared
    offnetwork-upstream ghost rows) and sorting for binary-search use."""
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs)
    return merged.loc[~merged.index.duplicated()].sort_index()


@dataclass
class ComputeConfig:
    """Parameters to control routing compute call."""

    nts: int
    dt: float
    qts_subdivisions: int
    t0: datetime
    ssout: float
    data_assimilation_parameters: dict[str, Any]
    da_parameter_dict: dict[str, Any]
    waterbody_type_specified: bool
    assume_short_ts: bool
    return_courant: bool
    from_files: bool
    cpu_pool: int
    parallel_compute_method: PARALLEL_COMPUTE_METHODS
    qlat_add_loc: str
    compute_func_name: str
    subnetwork_target_size: int
    backend: Literal["loky", "threading", "multiprocessing"] = "loky"

    def __post_init__(self):
        # Control serial execution by forcing single worker instead of a code change.
        if self.parallel_compute_method == "serial":
            self.cpu_pool = 1

    @property
    def compute_function(self) -> Callable:
        """Callable routing function."""
        try:
            return _compute_func_map[self.compute_func_name]
        except KeyError:
            opts = ", ".join(_compute_func_map)
            raise ValueError(
                f"compute kernel should be one of {opts} but got "
                f"{self.compute_func_name!r}"
            ) from None

    @property
    def qlat_add_loc_c(self) -> int:
        """Integer representing lateral flow addition location."""
        return _QLAT_LOC_MAP[self.qlat_add_loc]

    @property
    def t0_str(self) -> str:
        """String representation of t0 datetime."""
        return self.t0.strftime("%Y-%m-%d_%H:%M:%S")

    @property
    def da_decay_coefficient(self) -> float:
        """Data Assimilation decay coefficient for streamflow nudging."""
        return self.da_parameter_dict.get("da_decay_coefficient", 0.0)


@dataclass
class ComputationJob:
    """A set of routing paths passed together to the routing kernel for execution."""

    connections: UpstreamGraph
    routing_paths: OrderedRoutingPaths
    waterbodies_df: pd.DataFrame
    waterbodies_types_df: pd.DataFrame
    river_df: pd.DataFrame
    tailwaters: list[TailwaterId]
    offnetwork_upstreams: set[ReachId]

    @classmethod
    def from_multijob(cls, jobs: list[ComputationJob]) -> ComputationJob:
        """Merge multiple computation jobs into a single job, sorting data indices for binary search compatibility."""
        merged_connections: UpstreamGraph = {}
        for job in jobs:
            merged_connections.update(job.connections)
        merged_routing_paths = list(
            chain.from_iterable(job.routing_paths for job in jobs)
        )
        waterbodies_dfs = [
            job.waterbodies_df for job in jobs if not job.waterbodies_df.empty
        ]
        merged_waterbodies_df = _concat_unique_sorted(waterbodies_dfs)
        waterbodies_types_dfs = [
            job.waterbodies_types_df
            for job in jobs
            if not job.waterbodies_types_df.empty
        ]
        merged_waterbodies_types_df = _concat_unique_sorted(waterbodies_types_dfs)
        river_dfs = [job.river_df for job in jobs if not job.river_df.empty]
        merged_river_df = _concat_unique_sorted(river_dfs)
        merged_tailwaters = list(chain.from_iterable(job.tailwaters for job in jobs))
        merged_offnetwork_upstreams: set[ReachId] = set().union(
            *(job.offnetwork_upstreams for job in jobs)
        )
        return cls(
            connections=merged_connections,
            routing_paths=merged_routing_paths,
            waterbodies_df=merged_waterbodies_df,
            waterbodies_types_df=merged_waterbodies_types_df,
            river_df=merged_river_df,
            tailwaters=merged_tailwaters,
            offnetwork_upstreams=merged_offnetwork_upstreams,
        )

    @cached_property
    def waterbody_reaches(self) -> list[ReachId]:
        """Reach IDs for all waterbodies in this job."""
        return list(self.waterbodies_df.index.to_numpy())

    @cached_property
    def waterbody_set(self) -> set[ReachId]:
        return set(self.waterbody_reaches)

    @cached_property
    def river_reaches(self) -> Int64Array:
        """Reach IDs for all river segments in this job."""
        # dtype=int64 matches the kernel's `const long[:] data_idx` memoryview
        # (C long is 64-bit on the LP64 targets); a mismatch would raise a
        # Cython buffer-dtype error at the call.
        return self.river_df.index.to_numpy(dtype="int64")

    @cached_property
    def reach_types(self) -> list[tuple[Any, int]]:
        """Routing paths paired with their reach-type flag (1 = waterbody, 0 = river, etc)."""
        return _build_reach_type_list(self.routing_paths, self.waterbody_set)

    @cached_property
    def waterbody_types(self) -> Int32Array:
        """Waterbody type codes."""
        return self.waterbodies_types_df.to_numpy(dtype="int32")

    @cached_property
    def river_fields(self) -> ObjectArray:
        """Column names of the river parameter DataFrame."""
        return self.river_df.columns.to_numpy()

    @cached_property
    def river_values(self) -> Float32Array:
        """River parameter values as a 2-D array."""
        # dtype=float32 matches the kernel's `const float[:,:] data_values`
        # memoryview (river_df is already float32 via param_df.astype, so this
        # is a no-copy view; the cast just makes the contract explicit here).
        return self.river_df.to_numpy(dtype="float32")

    # NOTE: no cached property for waterbodies_df.values -- build_compute_package
    # mutates the frame's h0 column on every run set (the state-transfer step),
    # so a cached values array would freeze the first run set's elevations.

    @cached_property
    def river_index(self) -> pd.Index:
        return pd.Index(self.river_reaches)

    @cached_property
    def lake_mask(self) -> BoolArray | None:
        if len(self.waterbody_reaches) > 0:
            _lake_arr = np.fromiter(self.waterbody_set, dtype=np.int64, count=len(self.waterbody_set))
            return np.isin(self.river_index, _lake_arr)

    @cached_property
    def tailwater_results_indices(self) -> IntpArray:
        """The index of tailwaters in the results tuple returned by the kernel for this job.

        The kernel applies fill_index_mask to data_idx before returning, which strips
        every offnetwork_upstream row. Since offnetwork_upstreams is known at plan-build
        time, we reconstruct the same filtered index here and get_indexer the tailwaters
        against it in one vectorized call — giving precomputed positions instead of the
        O(n) .tolist().index() scan that was previously done at routing time on every
        call to update_boundary_conditions.
        """
        # kernel will remove these from results array
        filtered_index = self.river_df.index[~self.river_df.index.isin(self.offnetwork_upstreams)]
        # get_indexer accepts list-likes at runtime; the stub over-narrows to Index.
        positions = filtered_index.get_indexer(self.tailwaters)  # pyright: ignore[reportArgumentType]
        # Tailwaters are this job's outlets, so always in the stripped result index;
        # a -1 would silently take the wrong result row, so fail loud.
        if (positions < 0).any():
            missing = [tw for tw, p in zip(self.tailwaters, positions) if p < 0]
            raise KeyError(f"tailwaters not in job result index: {missing[:5]}")
        return positions

@dataclass
class NetworkTopology:
    """Representation of the full directed graph representing all routing topology, potentially consisting of multiple disconnected trees (a forest) with tools to traverse and subset."""

    connections: DownstreamGraph
    reverse_connections: UpstreamGraph
    paths_by_tailwater: dict[TailwaterId, OrderedRoutingPaths]
    connections_by_tw: dict[TailwaterId, DownstreamGraph]

    @property
    def tailwaters(self) -> list[TailwaterId]:
        """Get all network tailwaters."""
        return list(self.paths_by_tailwater.keys())


@dataclass
class ReachData:
    """Concise package of river channel data."""

    dataframe: pd.DataFrame

    def __post_init__(self) -> None:
        # Precompute the column subset AND the index as a set once. generate_view()
        # and the per-job reach intersection in ExecutionPlan._build_compute_job
        # both run once per subnetwork (tens of thousands at CONUS scale); a
        # label-based .loc / pandas Index.intersection on the full reach frame each
        # time made plan construction O(jobs x frame) and dominated CONUS wall time.
        # A precomputed column view + positional take, and a set intersection, are
        # O(len(reaches)) per call.
        cols = ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"]
        self._view = None if self.dataframe.empty else self.dataframe[cols]
        self._index_set: set[ReachId] = set(self.dataframe.index)

    def generate_view(self, reaches: list[ReachId]) -> pd.DataFrame:
        """Subset data to a set of reaches."""
        if self._view is None:
            return pd.DataFrame()
        # get_indexer accepts list-likes at runtime; the stub over-narrows to Index.
        positions = self._view.index.get_indexer(reaches)  # pyright: ignore[reportArgumentType]
        if (positions < 0).any():
            missing = [r for r, p in zip(reaches, positions) if p < 0]
            raise KeyError(f"reaches not in ReachData: {missing[:5]}")
        return self._view.take(positions)


@dataclass
class WaterbodyData:
    """Concise package of waterbody data."""

    dataframe: pd.DataFrame
    types: pd.DataFrame

    def __post_init__(self) -> None:
        # Precompute the column subset AND the index as a set once. generate_view()
        # and the per-job lake intersection in ExecutionPlan._build_compute_job both
        # run once per subnetwork (tens of thousands at CONUS scale); a label-based
        # .loc / pandas Index.intersection on the full frame each time made plan
        # construction O(jobs x frame). A precomputed view + positional take, and a
        # set intersection, are O(len(reaches)) per call.
        cols = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
                "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0"]
        self._view = None if self.dataframe.empty else self.dataframe[cols]
        self._index_set: set[ReachId] = set(self.dataframe.index)

    def generate_view(
        self, reaches: list[ReachId]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Subset data to a set of reaches."""
        if self._view is None:
            return pd.DataFrame(), pd.DataFrame()
        # Positional take instead of label .loc (see ReachData.generate_view). The
        # dataframe and types frames are looked up independently in case their
        # indexes differ in order; a -1 (missing label) raises rather than silently
        # taking the last row, preserving the old .loc KeyError behaviour.
        # get_indexer accepts list-likes at runtime; the stub over-narrows to Index.
        positions = self._view.index.get_indexer(reaches)  # pyright: ignore[reportArgumentType]
        types_positions = self.types.index.get_indexer(reaches)  # pyright: ignore[reportArgumentType]
        if (positions < 0).any() or (types_positions < 0).any():
            missing = [r for r, p in zip(reaches, positions) if p < 0]
            missing += [r for r, p in zip(reaches, types_positions) if p < 0]
            raise KeyError(f"waterbodies not in WaterbodyData: {missing[:5]}")
        return self._view.take(positions), self.types.take(types_positions)


@dataclass
class ForcingData:
    """Concise package of forcing data."""

    qlats: pd.DataFrame
    q0: pd.DataFrame
    eloss: pd.DataFrame
    wbody_init: pd.DataFrame

    @cached_property
    def qlat_vals(self) -> Float32Array:
        return self.qlats.to_numpy()

    @cached_property
    def qlat_idx(self) -> pd.Index:
        return self.qlats.index

    @cached_property
    def qlat_cols(self) -> pd.Index:
        return self.qlats.columns

    @cached_property
    def q0_vals(self) -> Float32Array:
        return self.q0.to_numpy()

    @cached_property
    def q0_idx(self) -> pd.Index:
        return self.q0.index

    @cached_property
    def q0_cols(self) -> pd.Index:
        return self.q0.columns

    @cached_property
    def eloss_vals(self) -> Float64Array:
        return self.eloss.to_numpy()

    @cached_property
    def eloss_idx(self) -> pd.Index:
        return self.eloss.index

    @cached_property
    def eloss_cols(self) -> pd.Index:
        return self.eloss.columns



@dataclass
class AssimilationData:
    """Concise package of assimilation data."""

    reservoir_usgs_df: pd.DataFrame
    reservoir_usgs_param_df: pd.DataFrame
    reservoir_usace_param_df: pd.DataFrame
    reservoir_usace_df: pd.DataFrame
    reservoir_usbr_df: pd.DataFrame
    reservoir_usbr_param_df: pd.DataFrame
    reservoir_rfc_df: pd.DataFrame
    reservoir_rfc_param_df: pd.DataFrame
    great_lakes_df: pd.DataFrame
    great_lakes_param_df: pd.DataFrame
    great_lakes_climatology_df: pd.DataFrame
    usgs_df: pd.DataFrame
    lastobs_df: pd.DataFrame


@dataclass
class ComputeInputs:
    """Concise arguments for compute_network_structured."""

    nsteps: int
    dt: float
    qts_subdivisions: int
    reaches_wTypes: list[tuple[list[int], int]]
    upstream_connections: dict[int, list[int]]
    data_idx: Int64Array
    data_cols: ObjectArray
    data_values: Float32Array
    initial_conditions: Float32Array
    qlat_values: Float32Array
    eloss_values: Float32Array
    ssout: float
    lake_numbers_col: list[int]
    wbody_cols: Float64Array
    data_assimilation_parameters: dict[str, Any]
    reservoir_types: Int32Array
    reservoir_type_specified: bool
    model_start_time: str
    usgs_values: Float32Array
    usgs_positions: Int32Array
    usgs_positions_reach: Int32Array
    usgs_positions_gage: Int32Array
    lastobs_values_init: Float32Array
    time_since_lastobs_init: Float32Array
    da_decay_coefficient: float
    reservoir_usgs_obs: Float32Array
    reservoir_usgs_wbody_idx: Int64Array
    reservoir_usgs_time: Float32Array
    reservoir_usgs_update_time: Float32Array
    reservoir_usgs_prev_persisted_flow: Float32Array
    reservoir_usgs_persistence_update_time: Float32Array
    reservoir_usgs_persistence_index: Float32Array
    reservoir_usace_obs: Float32Array
    reservoir_usace_wbody_idx: Int64Array
    reservoir_usace_time: Float32Array
    reservoir_usace_update_time: Float32Array
    reservoir_usace_prev_persisted_flow: Float32Array
    reservoir_usace_persistence_update_time: Float32Array
    reservoir_usace_persistence_index: Float32Array
    reservoir_usbr_obs: Float32Array
    reservoir_usbr_wbody_idx: Int64Array
    reservoir_usbr_time: Float32Array
    reservoir_usbr_update_time: Float32Array
    reservoir_usbr_prev_persisted_flow: Float32Array
    reservoir_usbr_persistence_update_time: Float32Array
    reservoir_usbr_persistence_index: Float32Array
    reservoir_rfc_obs: Float32Array
    reservoir_rfc_wbody_idx: Int64Array
    reservoir_rfc_totalCounts: Int32Array
    reservoir_rfc_file: list[str]
    reservoir_rfc_use_forecast: Int32Array
    reservoir_rfc_timeseries_idx: Int32Array
    reservoir_rfc_update_time: Float32Array
    reservoir_rfc_da_timestep: Int32Array
    reservoir_rfc_persist_days: Int32Array
    great_lakes_idx: Int32Array
    great_lakes_times: Int32Array
    great_lakes_discharge: Float32Array
    great_lakes_param_idx: Int32Array
    great_lakes_param_prev_assim_flow: Float32Array
    great_lakes_param_prev_assim_times: Int32Array
    great_lakes_param_update_times: Int32Array
    great_lakes_climatology: Float32Array
    upstream_results: dict[int, Any] = field(default_factory=dict)
    assume_short_ts: bool = False
    return_courant: bool = False
    da_check_gage: int = -1
    from_files: bool = True
    qlat_add_loc: int = 1


@dataclass
class BoundaryConditionStore:
    """Boundary condition between networks."""

    bcs: dict[ReachId, BoundaryCondition]

    def generate_view(self, reaches: Iterable[ReachId]) -> dict[ReachId, BoundaryCondition]:
        """Return boundary conditions for the specified upstream reaches."""
        # TODO: add a view cache
        return {reach_id: self.bcs[reach_id] for reach_id in reaches}

    def update(self, reach_id: ReachId, results: Float32Array) -> None:
        """Store routing results for a tailwater to be used as an upstream boundary condition at the next routing level."""
        self.bcs[reach_id]["results"] = results

    def clear_data(self) -> None:
        """Release stored results arrays to free memory after all routing levels complete."""
        for i in self.bcs.values():
            i["results"] = None


class ExecutionPlan:
    """An ordered mapping of computation batches and their dependency relationships used to orchestrate network execution."""

    def __init__(
        self,
        parallel_compute_method: PARALLEL_COMPUTE_METHODS,
        topology: NetworkTopology,
        reach_data: ReachData,
        waterbody_data: WaterbodyData,
        assimilation_data: AssimilationData,
        subnetwork_target_size: int,
    ):
        # Validation
        parallel_options = get_args(PARALLEL_COMPUTE_METHODS)
        if parallel_compute_method not in parallel_options:
            opts = " ".join(parallel_options)
            raise ValueError(
                f"parallel compute method should be one of {opts} but got {parallel_compute_method}"
            )

        # Initialization
        self.batches: dict[RoutingLevel, list[ComputationJob]] = {}
        if parallel_compute_method == "by-subnetwork-jit-clustered":
            self._init_clustered_partition_plan(
                topology,
                reach_data,
                waterbody_data,
                assimilation_data,
                subnetwork_target_size,
            )
        elif parallel_compute_method == "by-subnetwork-jit":
            self._init_partitioned_plan(
                topology,
                reach_data,
                waterbody_data,
                assimilation_data,
                subnetwork_target_size,
            )
        elif parallel_compute_method in ["serial", "by-network", "bmi"]:
            self._init_treewise_plan(topology, reach_data, waterbody_data)
        self._init_boundary_conditions()

    def _build_compute_job(
        self,
        reaches: OrderedRoutingPaths,
        graph: UpstreamGraph,
        waterbody_data: WaterbodyData,
        reach_data: ReachData,
        tailwaters: list[TailwaterId],
    ) -> ComputationJob:
        """Build a ComputationJob for a set of ordered routing paths, including data for any off-network upstream reaches."""
        # Flatten ordered chains to reach ids
        flat_reaches = set(chain.from_iterable(reaches))

        # Get offnetwork upstreams
        all_upstreams = set().union(*(graph.get(seg, ()) for seg in flat_reaches))
        offnetwork_upstreams = all_upstreams - flat_reaches

        # Get data for offnetwork upstreams too. Keep flat_reaches a set -- its
        # only consumers are the two intersections below.
        flat_reaches |= offnetwork_upstreams

        # subset waterbodies -- intersect the precomputed index set rather than
        # calling pandas Index.intersection(list) per job, which coerced the list
        # to an Index and dominated CONUS plan-build wall (~22k subnetworks).
        lake_reaches = sorted(waterbody_data._index_set & flat_reaches)
        waterbodies_df, waterbodies_types_df = waterbody_data.generate_view(
            lake_reaches
        )

        # subset reaches
        river_reaches = sorted(reach_data._index_set & flat_reaches)
        river_df = reach_data.generate_view(river_reaches)

        # Extend river_df with NaN rows for lake reaches so that data_idx
        # (= river_df.index) includes waterbody IDs. The routing kernel uses
        # binary_find(data_idx, upstream_reach) for ALL upstream lookups,
        # including cases where the upstream is a reservoir. The old code
        # achieved this via param_df_sub.reindex(...+lake_segs).sort_index().
        if lake_reaches:
            extended_index = np.sort(np.concatenate([river_df.index.to_numpy(), lake_reaches]))
            river_df = river_df.reindex(extended_index)

        # Create subnetwork instance
        return ComputationJob(
            graph,
            reaches,
            waterbodies_df,
            waterbodies_types_df,
            river_df,
            tailwaters,
            offnetwork_upstreams,
        )

    def _init_treewise_plan(
        self,
        topology: NetworkTopology,
        reach_data: ReachData,
        waterbody_data: WaterbodyData,
    ) -> None:
        """Build a single-level execution plan with one computation job per tree."""
        self.batches = {0: []}
        for i in topology.tailwaters:
            job = self._build_compute_job(
                topology.paths_by_tailwater[i],
                topology.connections_by_tw[i],
                waterbody_data,
                reach_data,
                [i],
            )
            self.batches[0].append(job)

    def _init_partitioned_plan(
        self,
        topology: NetworkTopology,
        reach_data: ReachData,
        waterbody_data: WaterbodyData,
        assimilation_data: AssimilationData,
        subnetwork_target_size: int,
    ) -> None:
        """Build a multi-level execution plan by partitioning each tree into subnetworks by routing level."""
        # Break whole networks into partitions
        partitions_by_tailwater: dict[
            TailwaterId, dict[RoutingLevel, dict[TailwaterId, Partition]]
        ] = nhd_network.build_subnetworks(
            topology.connections, topology.reverse_connections, subnetwork_target_size
        )

        # Dissolve partitions on routing level
        partitions_by_routing_level = self._reorganize_partitions(
            partitions_by_tailwater
        )

        # Break network at points of interest (gages, waterbodies, and/or junctions)
        # and order such that any upstream deps run first.
        computable_routing_paths = self._clean_compute_jobs(
            partitions_by_routing_level, topology, waterbody_data, assimilation_data
        )

        for routing_level, partitions in computable_routing_paths.items():
            self.batches[routing_level] = []
            for tw, partition in partitions.items():
                sub_connections = {
                    k: topology.reverse_connections[k]
                    for k in partitions_by_routing_level[routing_level][tw]
                }
                subnetwork = self._build_compute_job(
                    partition, sub_connections, waterbody_data, reach_data, [tw]
                )
                self.batches[routing_level].append(subnetwork)

    def _reorganize_partitions(
        self,
        partitions_by_tailwater: dict[
            TailwaterId, dict[RoutingLevel, dict[TailwaterId, Partition]]
        ],
    ) -> dict[RoutingLevel, dict[TailwaterId, Partition]]:
        """Dissolve partitions on routing level."""
        partitions_by_level = defaultdict(dict)
        for tmp_partitions_by_level in partitions_by_tailwater.values():
            for level, partition in tmp_partitions_by_level.items():
                partitions_by_level[level].update(partition)
        return dict(partitions_by_level)

    def _clean_compute_jobs(
        self,
        partitions_by_level: dict[RoutingLevel, dict[TailwaterId, Partition]],
        topology: NetworkTopology,
        waterbody_data: WaterbodyData,
        assimilation_data: AssimilationData,
    ) -> dict[RoutingLevel, dict[TailwaterId, OrderedRoutingPaths]]:
        """Decompose each partition into ordered routing paths split at gages, waterbodies, and/or junctions."""
        computable_routing_paths = defaultdict(dict)
        for level, partitions in partitions_by_level.items():
            for partition_tailwater, partition in partitions.items():
                rconn_subn = {
                    k: topology.reverse_connections[k]
                    for k in partition
                    if k in topology.reverse_connections
                }
                if (
                    not waterbody_data.dataframe.empty
                    and not assimilation_data.usgs_df.empty
                ):
                    path_func = partial(
                        nhd_network.split_at_gages_waterbodies_and_junctions,
                        set(assimilation_data.usgs_df.index.to_numpy()),
                        set(waterbody_data.dataframe.index.to_numpy()),
                        rconn_subn,
                    )

                elif (
                    waterbody_data.dataframe.empty
                    and not assimilation_data.usgs_df.empty
                ):
                    path_func = partial(
                        nhd_network.split_at_gages_and_junctions,
                        set(assimilation_data.usgs_df.index.to_numpy()),
                        rconn_subn,
                    )

                elif (
                    not waterbody_data.dataframe.empty
                    and assimilation_data.usgs_df.empty
                ):
                    path_func = partial(
                        nhd_network.split_at_waterbodies_and_junctions,
                        set(waterbody_data.dataframe.index.to_numpy()),
                        rconn_subn,
                    )

                else:
                    path_func = None
                computable_routing_paths[level][partition_tailwater] = (
                    nhd_network.dfs_decomposition(rconn_subn, path_func)
                )
        return dict(computable_routing_paths)

    def _init_clustered_partition_plan(
        self,
        topology: NetworkTopology,
        reach_data: ReachData,
        waterbody_data: WaterbodyData,
        assimilation_data: AssimilationData,
        subnetwork_target_size: int,
    ) -> None:
        """Build a partitioned execution plan and cluster small adjacent jobs to reduce kernel-call overhead."""
        cluster_threshold = 0.65  # When a job has a total segment count 65% of the target size, compute it
        # Otherwise, keep adding reaches.

        self._init_partitioned_plan(
            topology,
            reach_data,
            waterbody_data,
            assimilation_data,
            subnetwork_target_size,
        )
        for routing_level, batch in self.batches.items():
            new_batch = []
            jobs = []
            reach_count = 0
            reach_limit = subnetwork_target_size * cluster_threshold
            for job in batch:
                jobs.append(job)
                reach_count += len(list(chain.from_iterable(job.routing_paths)))
                if reach_count > reach_limit:
                    new_batch.append(ComputationJob.from_multijob(jobs))
                    jobs = []
                    reach_count = 0
            if len(jobs) > 0:
                new_batch.append(ComputationJob.from_multijob(jobs))
            self.batches[routing_level] = new_batch

    def _init_boundary_conditions(self) -> None:
        """Pre-allocate boundary condition slots for every off-network upstream reach in the plan."""
        boundary_conditions: dict[ReachId, BoundaryCondition] = {}
        claimed_level: dict[ReachId, RoutingLevel] = {}
        for routing_level, batch in self.batches.items():
            for job in batch:
                for upstream_reach in job.offnetwork_upstreams:
                    if upstream_reach in claimed_level:
                        raise NotImplementedError(
                            f"off-network upstream reach {upstream_reach} is "
                            f"consumed by computation jobs at routing levels "
                            f"{claimed_level[upstream_reach]} and {routing_level}. "
                            "The boundary condition store holds a single "
                            "position_index per reach, so a second consumer would "
                            "fill its boundary row at the wrong position (silent "
                            "numeric corruption). This arises on networks with "
                            "flow divergences; store positions per job to "
                            "support them."
                        )
                    claimed_level[upstream_reach] = routing_level
                    # get_loc returns int for a unique scalar label (slice/mask
                    # only for duplicate/monotonic-range labels, impossible here).
                    position_index = cast(
                        int, job.river_df.index.get_loc(upstream_reach)
                    )
                    boundary_conditions[upstream_reach] = {
                        "results": None,
                        "position_index": position_index,
                    }
        self.boundary_conditions = BoundaryConditionStore(boundary_conditions)

    def update_boundary_conditions(
        self, results: list[tuple], routing_level: RoutingLevel
    ) -> None:
        """Propagate tailwater results from a completed routing level into the boundary condition store."""
        for job_ind, job in enumerate(self.batches[routing_level]):
            for tailwater, tw_result_ind in zip(job.tailwaters, job.tailwater_results_indices):
                tw_results = results[job_ind][1][tw_result_ind]
                self.boundary_conditions.update(tailwater, tw_results)

    def export_job_list(self, path: str = "parallelization.parquet") -> None:
        """Useful for debugging."""
        out_dict = {"reach_id": [], "batch": [], "job_num": []}
        for routing_level, batch in self.batches.items():
            for job_ind, job in enumerate(batch):
                for routing_path in job.routing_paths:
                    for reach in routing_path:
                        out_dict["reach_id"].append(reach)
                        out_dict["batch"].append(routing_level)
                        out_dict["job_num"].append(job_ind)
        pd.DataFrame(out_dict).to_parquet(path)


def _format_qlat_start_time(qlat_start_time):
    if not isinstance(qlat_start_time,datetime):
        try:
            return datetime.strptime(qlat_start_time, '%Y-%m-%d %H:%M:%S')
        except:  # TODO: make sure this doesn't introduce a silent error
            return datetime.now()

    else:
        return qlat_start_time


def _build_reach_type_list(reach_list, wbodies_segs):

    # No waterbody break segments (e.g. NHF stubs waterbodies): every reach is
    # type 0, so we can skip the per-reach set work entirely.
    if not wbodies_segs:
        return [(reaches, 0) for reaches in reach_list]

    # set.isdisjoint() short-circuits at the first shared element and builds no
    # intermediate set, unlike `set(reaches) & wbodies_segs`.
    reach_type_list = [
        0 if wbodies_segs.isdisjoint(reaches) else 1 for reaches in reach_list
    ]

    return list(zip(reach_list, reach_type_list))


def _reindex_via_take(values_arr, positions, fill_value=np.nan):
    # Equivalent to df.reindex(<index whose .get_indexer is positions>).to_numpy()
    # given values_arr = df.to_numpy(). pd.api.extensions.take dispatches
    # to the pandas Cython take primitive (_take_nd_ndarray), which handles the
    # -1-means-missing semantics natively via allow_fill=True. ``values_arr`` must
    # be the cached ndarray view of the source DataFrame -- calling df.to_numpy()
    # per cluster (especially on a multi-block source, or on a freshly column-
    # sliced df[cols]) would re-trigger pandas' internal block consolidation
    # on the 1.1 M-row source frame each call.
    return pd.api.extensions.take(
        values_arr, positions, allow_fill=True, fill_value=fill_value
    )


def _assert_channel_rows_present(label, positions, extended_index_arr, lake_segs_set):
    """Restore the fast-fail behavior of the original .loc[]-based code.

    The legacy ``qlats.loc[common_segs]`` / ``param_df.loc[common_segs]`` raised
    KeyError on missing channel rows. The extended-index ``take`` path collapses
    that into a silent NaN-fill via ``allow_fill=True`` (positions == -1). Here
    we re-validate that every -1 in ``positions`` corresponds to a lake-segment
    extension row -- the only legitimate source of missing entries. A missing
    channel row signals either a malformed forcing file or a flowpath/reservoir
    ID namespace collision and should surface immediately, not propagate as NaN.
    """
    missing_mask = (positions == -1)
    if not missing_mask.any():
        return
    missing = extended_index_arr[missing_mask]
    if not lake_segs_set:
        bad = [int(s) for s in missing]
    else:
        bad = [int(s) for s in missing if int(s) not in lake_segs_set]
    if bad:
        raise KeyError(
            f"{label}: {len(bad)} channel segment(s) missing from source frame "
            f"(e.g. {bad[:5]}). Lake-seg extension rows are expected to be "
            f"missing (filled per the per-call fill_value); channel rows are not."
        )


def _prep_da_dataframes(
    usgs_df,
    lastobs_df,
    param_df_sub_idx,
    exclude_segments=None,
    ):
    """
    Produce, based on the segments in the param_df_sub_idx (which is a subset
    representing a subnetwork of the larger collection of all segments),
    a subset of the relevant usgs gage observation time series
    and the relevant last-valid gage observation from any
    prior model execution.
    
    exclude_segments (list): segments to exclude from param_df_sub when searching for gages
                             This catches and excludes offnetwork upstreams segments from being
                             realized as locations for DA substitution. Else, by-subnetwork
                             parallel executions fail.

    Cases to consider:
    USGS_DF, LAST_OBS
    Yes, Yes: Analysis and Assimilation; Last_Obs used to fill gaps in the front of the time series
    No, Yes: Forecasting mode;
    Yes, No; Cold-start case;
    No, No: Open-Loop;

    For both cases where USGS_DF is present, there is a sub-case where the length of the observed
    time series is as long as the simulation.

    """
    
    subnet_segs = param_df_sub_idx
    # segments in the subnetwork ONLY, no offnetwork upstreams included
    if exclude_segments:
        subnet_segs = param_df_sub_idx.difference(set(exclude_segments))
    
    # NOTE: Uncomment to easily test no observations...
    # usgs_df = pd.DataFrame()
    if not usgs_df.empty and not lastobs_df.empty:
        # index values for last obs are not correct, but line up correctly with usgs values. Switched
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                       )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        usgs_segs = (usgs_df.index.
                     intersection(subnet_segs).
                     reindex(lastobs_segs)[0].
                     to_list()
                    )
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
    elif usgs_df.empty and not lastobs_df.empty:
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                       )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        # Create a completely empty list of gages -- the .shape[1] attribute
        # will be == 0, and that will trigger a reference to the lastobs.
        # in the compute kernel below.
        usgs_df_sub = pd.DataFrame(index=lastobs_df_sub.index,columns=[])
        usgs_segs = lastobs_segs
        da_positions_list_byseg = param_df_sub_idx.get_indexer(lastobs_segs)
    elif not usgs_df.empty and lastobs_df.empty:
        usgs_segs = list(usgs_df.index.intersection(subnet_segs))
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
        lastobs_df_sub = pd.DataFrame(index=usgs_df_sub.index,columns=["discharge","time","model_discharge"])
    else:
        usgs_df_sub = pd.DataFrame()
        lastobs_df_sub = pd.DataFrame()
        da_positions_list_byseg = []

    return usgs_df_sub, lastobs_df_sub, da_positions_list_byseg


def _prep_da_positions_byreach(reach_list, gage_index):
    """
    produce a list of indexes of the reach_list identifying reaches with gages
    and a corresponding list of indexes of the gage_list of the gages in
    the order they are found in the reach_list.
    """
    # Empty gage_index (DA disabled) is the dominant case in CONUS runs.
    # The original double loop did 12.2 M ``seg in gage_index`` checks on an
    # empty RangeIndex per CONUS run (~5 s under cProfile / ~1.5 s clean),
    # all of which return False. Short-circuit when there's nothing to find.
    if len(gage_index) == 0:
        return [], gage_index.get_indexer([])

    reach_key = []
    reach_gage = []
    for i, r in enumerate(reach_list):
        for s in r:
            if s in gage_index:
                reach_key.append(i)
                reach_gage.append(s)
    gage_reach_i = gage_index.get_indexer(reach_gage)

    return reach_key, gage_reach_i

def _prep_reservoir_da_dataframes(reservoir_usgs_df,
                                  reservoir_usgs_param_df,
                                  reservoir_usace_df,
                                  reservoir_usace_param_df,
                                  reservoir_usbr_df,
                                  reservoir_usbr_param_df,
                                  reservoir_rfc_df,
                                  reservoir_rfc_param_df,
                                  great_lakes_df,
                                  great_lakes_param_df,
                                  great_lakes_climatology_df,
                                  waterbody_types_df_sub,
                                  t0, 
                                  from_files,
                                  exclude_segments=None):
    '''
    Helper function to build reservoir DA data arrays for routing computations

    Arguments
    ---------
    reservoir_usgs_df        (DataFrame): gage flow observations at USGS-type reservoirs
    reservoir_usgs_param_df  (DataFrame): USGS reservoir DA state parameters
    reservoir_usace_df       (DataFrame): gage flow observations at USACE-type reservoirs
    reservoir_usace_param_df (DataFrame): USACE reservoir DA state parameters
    reservoir_usbr_df       (DataFrame): gage flow observations at USBR-type reservoirs
    reservoir_usbr_param_df (DataFrame): USBR reservoir DA state parameters
    reservoir_rfc_df         (DataFrame): gage flow observations and forecasts at RFC-type reservoirs
    reservoir_rfc_param_df   (DataFrame): RFC reservoir DA state parameters
    waterbody_types_df_sub   (DataFrame): type-codes for waterbodies in sub domain
    t0                        (datetime): model initialization time

    Returns
    -------
    * there are many returns, because we are passing explicit arrays to mc_reach cython code
    reservoir_usgs_df_sub                 (DataFrame): gage flow observations for USGS-type reservoirs in sub domain
    reservoir_usgs_df_time                  (ndarray): time in seconds from model initialization time
    reservoir_usgs_update_time              (ndarray): update time (sec) to search for new observation at USGS reservoirs
    reservoir_usgs_prev_persisted_flow      (ndarray): previously persisted outflow rates at USGS reservoirs
    reservoir_usgs_persistence_update_time  (ndarray): update time (sec) of persisted value at USGS reservoirs
    reservoir_usgs_persistence_index        (ndarray): index denoting elapsed persistence epochs at USGS reservoirs
    reservoir_usace_df_sub                (DataFrame): gage flow observations for USACE-type reservoirs in sub domain
    reservoir_usace_df_time                 (ndarray): time in seconds from model initialization time
    reservoir_usace_update_time             (ndarray): update time (sec) to search for new observation at USACE reservoirs
    reservoir_usace_prev_persisted_flow     (ndarray): previously persisted outflow rates at USACE reservoirs
    reservoir_usace_persistence_update_time (ndarray): update time (sec) of persisted value at USACE reservoirs
    reservoir_usace_persistence_index       (ndarray): index denoting elapsed persistence epochs at USACE reservoirs
    reservoir_usbr_df_sub                (DataFrame): gage flow observations for USBR-type reservoirs in sub domain
    reservoir_usbr_df_time                 (ndarray): time in seconds from model initialization time
    reservoir_usbr_update_time             (ndarray): update time (sec) to search for new observation at USBR reservoirs
    reservoir_usbr_prev_persisted_flow     (ndarray): previously persisted outflow rates at USBR reservoirs
    reservoir_usbr_persistence_update_time (ndarray): update time (sec) of persisted value at USBR reservoirs
    reservoir_usbr_persistence_index       (ndarray): index denoting elapsed persistence epochs at USBR reservoirs
    '''
    if not reservoir_usgs_df.empty:
        usgs_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==2
                                ].index
        if exclude_segments:
            usgs_wbodies_sub = list(set(usgs_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usgs_df_sub = reservoir_usgs_df.loc[usgs_wbodies_sub]
        reservoir_usgs_df_time = []
        for timestamp in reservoir_usgs_df.columns:
            reservoir_usgs_df_time.append((timestamp - t0).total_seconds())
        reservoir_usgs_df_time = np.array(reservoir_usgs_df_time)
        reservoir_usgs_update_time = reservoir_usgs_param_df['update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_prev_persisted_flow = reservoir_usgs_param_df['prev_persisted_outflow'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_update_time = reservoir_usgs_param_df['persistence_update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_index = reservoir_usgs_param_df['persistence_index'].loc[usgs_wbodies_sub].to_numpy()
    else:
        # Reuse module-level empties; the original per-cluster
        # pd.DataFrame().to_numpy().reshape(0,) idiom allocates a fresh
        # BlockManager each call (~100 µs × 31 calls/iter × 1200 iters).
        reservoir_usgs_df_sub = _EMPTY_DF
        reservoir_usgs_df_time = _EMPTY_F64
        reservoir_usgs_update_time = _EMPTY_F64
        reservoir_usgs_prev_persisted_flow = _EMPTY_F64
        reservoir_usgs_persistence_update_time = _EMPTY_F64
        reservoir_usgs_persistence_index = _EMPTY_F64
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 2] = 1

    # select USACE reservoir DA data waterbodies in sub-domain
    if not reservoir_usace_df.empty:
        usace_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==3
                                ].index
        if exclude_segments:
            usace_wbodies_sub = list(set(usace_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usace_df_sub = reservoir_usace_df.loc[usace_wbodies_sub]
        reservoir_usace_df_time = []
        for timestamp in reservoir_usace_df.columns:
            reservoir_usace_df_time.append((timestamp - t0).total_seconds())
        reservoir_usace_df_time = np.array(reservoir_usace_df_time)
        reservoir_usace_update_time = reservoir_usace_param_df['update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_prev_persisted_flow = reservoir_usace_param_df['prev_persisted_outflow'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_update_time = reservoir_usace_param_df['persistence_update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_index = reservoir_usace_param_df['persistence_index'].loc[usace_wbodies_sub].to_numpy()
    else:
        reservoir_usace_df_sub = _EMPTY_DF
        reservoir_usace_df_time = _EMPTY_F64
        reservoir_usace_update_time = _EMPTY_F64
        reservoir_usace_prev_persisted_flow = _EMPTY_F64
        reservoir_usace_persistence_update_time = _EMPTY_F64
        reservoir_usace_persistence_index = _EMPTY_F64
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 3] = 1

    # select USBR reservoir DA data waterbodies in sub-domain
    if not reservoir_usbr_df.empty:
        usbr_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==7
                                ].index
        if exclude_segments:
            usbr_wbodies_sub = list(set(usbr_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usbr_df_sub = reservoir_usbr_df.loc[usbr_wbodies_sub]
        reservoir_usbr_df_time = []
        for timestamp in reservoir_usbr_df.columns:
            reservoir_usbr_df_time.append((timestamp - t0).total_seconds())
        reservoir_usbr_df_time = np.array(reservoir_usbr_df_time)
        reservoir_usbr_update_time = reservoir_usbr_param_df['update_time'].loc[usbr_wbodies_sub].to_numpy()
        reservoir_usbr_prev_persisted_flow = reservoir_usbr_param_df['prev_persisted_outflow'].loc[usbr_wbodies_sub].to_numpy()
        reservoir_usbr_persistence_update_time = reservoir_usbr_param_df['persistence_update_time'].loc[usbr_wbodies_sub].to_numpy()
        reservoir_usbr_persistence_index = reservoir_usbr_param_df['persistence_index'].loc[usbr_wbodies_sub].to_numpy()
    else:
        reservoir_usbr_df_sub = _EMPTY_DF
        reservoir_usbr_df_time = _EMPTY_F64
        reservoir_usbr_update_time = _EMPTY_F64
        reservoir_usbr_prev_persisted_flow = _EMPTY_F64
        reservoir_usbr_persistence_update_time = _EMPTY_F64
        reservoir_usbr_persistence_index = _EMPTY_F64
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 7] = 1
    
    # RFC reservoirs
    if not reservoir_rfc_df.empty:
        rfc_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==4
            ].index
        if exclude_segments:
            rfc_wbodies_sub = list(set(rfc_wbodies_sub).difference(set(exclude_segments)))
        reservoir_rfc_df_sub = reservoir_rfc_df.loc[rfc_wbodies_sub]
        reservoir_rfc_totalCounts = reservoir_rfc_param_df['totalCounts'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_file = reservoir_rfc_param_df['file'].loc[rfc_wbodies_sub].to_list()
        reservoir_rfc_use_forecast = reservoir_rfc_param_df['use_rfc'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_timeseries_idx = reservoir_rfc_param_df['timeseries_idx'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_update_time = reservoir_rfc_param_df['update_time'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_da_timestep = reservoir_rfc_param_df['da_timestep'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_persist_days = reservoir_rfc_param_df['rfc_persist_days'].loc[rfc_wbodies_sub].to_numpy()
    else:
        reservoir_rfc_df_sub = _EMPTY_DF
        reservoir_rfc_totalCounts = _EMPTY_F64
        reservoir_rfc_file = _EMPTY_LIST
        reservoir_rfc_use_forecast = _EMPTY_F64
        reservoir_rfc_timeseries_idx = _EMPTY_F64
        reservoir_rfc_update_time = _EMPTY_F64
        reservoir_rfc_da_timestep = _EMPTY_F64
        reservoir_rfc_persist_days = _EMPTY_F64
        if not from_files:
            if not waterbody_types_df_sub.empty:
                waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 4] = 1
    
    # Great Lakes
    if not great_lakes_df.empty:
        gl_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==6
            ].index
        if exclude_segments:
            gl_wbodies_sub = list(set(gl_wbodies_sub).difference(set(exclude_segments)))
        gl_df_sub = great_lakes_df[great_lakes_df['lake_id'].isin(gl_wbodies_sub)]
        gl_climatology_df_sub = great_lakes_climatology_df.loc[gl_wbodies_sub]
        gl_param_df_sub = great_lakes_param_df[great_lakes_param_df['lake_id'].isin(gl_wbodies_sub)]
        gl_parm_lake_id_sub = gl_param_df_sub.lake_id.to_numpy()
        gl_param_flows_sub = gl_param_df_sub.previous_assimilated_outflows.to_numpy()
        gl_param_time_sub = gl_param_df_sub.previous_assimilated_time.to_numpy()
        gl_param_update_time_sub = gl_param_df_sub.update_time.to_numpy()
    else:
        gl_df_sub = _EMPTY_GL_DF
        gl_climatology_df_sub = _EMPTY_DF
        gl_parm_lake_id_sub = _EMPTY_F64
        gl_param_flows_sub = _EMPTY_F64
        gl_param_time_sub = _EMPTY_F64
        gl_param_update_time_sub = _EMPTY_F64
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 6] = 1

    return (
        reservoir_usgs_df_sub, reservoir_usgs_df_time, reservoir_usgs_update_time, reservoir_usgs_prev_persisted_flow, reservoir_usgs_persistence_update_time, reservoir_usgs_persistence_index,
        reservoir_usace_df_sub, reservoir_usace_df_time, reservoir_usace_update_time, reservoir_usace_prev_persisted_flow, reservoir_usace_persistence_update_time, reservoir_usace_persistence_index,
        reservoir_usbr_df_sub, reservoir_usbr_df_time, reservoir_usbr_update_time, reservoir_usbr_prev_persisted_flow, reservoir_usbr_persistence_update_time, reservoir_usbr_persistence_index,
        reservoir_rfc_df_sub, reservoir_rfc_totalCounts, reservoir_rfc_file, reservoir_rfc_use_forecast, reservoir_rfc_timeseries_idx, reservoir_rfc_update_time, reservoir_rfc_da_timestep, reservoir_rfc_persist_days,
        gl_df_sub, gl_parm_lake_id_sub, gl_param_flows_sub, gl_param_time_sub, gl_param_update_time_sub, gl_climatology_df_sub,
        waterbody_types_df_sub
        )


def compute_log_mc(
    fileName,
    connections,
    rconn,
    wbody_conn,
    reaches_bytw,
    compute_func_name,
    parallel_compute_method,
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
    usgs_df,
    lastobs_df,
    reservoir_usgs_df,
    reservoir_usgs_param_df,
    reservoir_usace_df,
    reservoir_usace_param_df,
    reservoir_usbr_df,
    reservoir_usbr_param_df,
    reservoir_rfc_df,
    reservoir_rfc_param_df,
    assume_short_ts,
    waterbodies_df,
    data_assimilation_parameters,
    waterbody_types_df,
    waterbody_type_specified,
):
    
    # TODO: do something with param_df, reservoir_XXX_param_df, or delete them as args

    # append parameters and some statistics to log file
    with open(fileName, 'a') as preRunLog:

        preRunLog.write("*******************\n") 
        preRunLog.write("Compute Parameters:\n") 
        preRunLog.write("*******************\n") 
        preRunLog.write("\n")   
        preRunLog.write("General Compute Parameters:\n")
        preRunLog.write("\n")  
        preRunLog.write("Parallel Compute Method: "+parallel_compute_method+'\n')
        preRunLog.write("Compute Kernel Name: "+compute_func_name+'\n')
        preRunLog.write("Assume Short Timescale: "+str(assume_short_ts)+'\n')
        preRunLog.write("Subnetwork Target Size: "+str(subnetwork_target_size)+'\n')
        preRunLog.write("CPU Pool: "+str(cpu_pool)+'\n')
        #preRunLog.write("\n")
        #preRunLog.write("Restart Parameters:\n")  
        #preRunLog.write("\n")
        preRunLog.write("Start_datetime: "+str(t0)+'\n')
        preRunLog.write("Coldstart: "+str(((q0==0).all()).all())+'\n')
        preRunLog.write("\n")
        preRunLog.write("Forcing Parameters:\n")
        preRunLog.write("\n")           
        preRunLog.write("qts subdivisions: "+str(qts_subdivisions)+'\n')
        preRunLog.write("dt [sec]: "+str(dt)+'\n')
        preRunLog.write("nts: "+str(nts)+'\n')
        preRunLog.write("\n")
        preRunLog.write("Data Assimilation Parameters:\n")
        preRunLog.write("\n")
    
        if ('usgs_timeslices_folder' in data_assimilation_parameters.keys()):           
            preRunLog.write("usgs timeslice folder: "+str(data_assimilation_parameters['usgs_timeslices_folder'])+'\n')
        if ('usace_timeslices_folder' in data_assimilation_parameters.keys()):    
            preRunLog.write("usace timeslice folder: "+str(data_assimilation_parameters['usace_timeslices_folder'])+'\n')
        preRunLog.write("-----\n")
        preRunLog.write("Streamflow DA\n")
        if ('streamflow_da' in data_assimilation_parameters.keys()):  
            outPutStr = "Streamflow nudging: "+str(data_assimilation_parameters['streamflow_da']['streamflow_nudging'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            outPutStr = "Diffusive streamflow nudging: "+str(data_assimilation_parameters['streamflow_da']['diffusive_streamflow_nudging'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            preRunLog.write("Lastobs file: "+str(data_assimilation_parameters['streamflow_da']['lastobs_file'])+'\n')
            preRunLog.write("-----\n")
            preRunLog.write("Reservoir DA\n")
            outPutStr = "Reservoir persistence USGS: "+str(data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usgs'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            outPutStr = "Reservoir persistence USACE: "+str(data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usace'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            outPutStr = "Reservoir persistence USBR: "+str(data_assimilation_parameters['reservoir_da']['reservoir_persistence_da']['reservoir_persistence_usbr'])
            preRunLog.write(outPutStr+'\n')
            LOG.info(outPutStr)
            preRunLog.write("Reservoir RFC forecasts: "+str(data_assimilation_parameters['reservoir_da']['reservoir_rfc_da']['reservoir_rfc_forecasts'])+'\n')

        preRunLog.write("\n")                   
        preRunLog.write("****************************\n") 
        preRunLog.write("Network Topology Parameters:\n") 
        preRunLog.write("****************************\n") 
        preRunLog.write("\n")   
        preRunLog.write("General network:\n")
        preRunLog.write("Number of downstream connections: "+str(len(connections))+'\n')
        preRunLog.write("Number of upstream connections: "+str(len(rconn))+'\n')
        preRunLog.write("Number of waterbody connections: "+str(len(wbody_conn))+'\n')
        preRunLog.write("Number of reaches by tailwater: "+str(len(reaches_bytw))+'\n')
        preRunLog.write("Number of independent networks: "+str(len(independent_networks))+'\n')
        preRunLog.write("Number of waterbodies: "+str(len(waterbodies_df.index))+'\n')
        preRunLog.write("Waterbody type specified: "+str(waterbody_type_specified)+'\n')
        nH20_types = len(waterbody_types_df.value_counts())
        if (nH20_types>0):
            for nH20 in range(nH20_types):
                preRunLog.write("Type: "+str(waterbody_types_df.value_counts().index[nH20][0]))
                preRunLog.write("   Number of waterbodies: "+str(waterbody_types_df.value_counts().to_numpy()[nH20])+'\n')
        preRunLog.write("-----\n")
        preRunLog.write("Gages and relations with waterbodies:\n")
        preRunLog.write("Number of USGS gages in network: "+str(len(usgs_df.index))+'\n')
        preRunLog.write("Number of USGS gage time bins in network: "+str(len(usgs_df.columns))+'\n')
        preRunLog.write("Lastobs files, number of gages: "+str(len(lastobs_df.index))+'\n')
        preRunLog.write("Number of USGS gages in waterbodies: "+str(len(reservoir_usgs_df.index))+'\n')
        preRunLog.write("Number of USACE gages in waterbodies: "+str(len(reservoir_usace_df.index))+'\n')
        preRunLog.write("Number of USBR gages in waterbodies: "+str(len(reservoir_usbr_df.index))+'\n')
        preRunLog.write("Number of RFC gages in waterbodies: "+str(len(reservoir_rfc_df.index))+'\n')
        preRunLog.write("\n")        

    preRunLog.close()


def compute_log_diff(
    fileName,
    diffusive_network_data,
    topobathy_df,
    refactored_diffusive_domain,
    refactored_reaches,                
    coastal_boundary_depth_df,
    unrefactored_topobathy_df,                
):

    # TODO: do something with refactored_diffusive_domain, refactored_reaches, unrefactored_topobathy_df, or delete args

    # append parameters and some statistics to log file
    with open(fileName, 'a') as preRunLog:

        preRunLog.write("*******************\n") 
        preRunLog.write("Diffusive Routing :\n") 
        preRunLog.write("*******************\n") 
        nTw = len(diffusive_network_data)
        preRunLog.write("\n")   
        outPutStr = "Number of diffusive tailwaters: "+str(nTw)
        preRunLog.write(outPutStr+'\n')
        LOG.info(outPutStr)
        preRunLog.write("-----\n")  

        twList = [key for key in diffusive_network_data]

        for i_nTw in range(nTw):  
            outPutStr = "Tailwater number and ID: "+str(i_nTw+1)+"   "+str(twList[i_nTw])
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)
            diffNw = diffusive_network_data[twList[i_nTw]]
            nMainSegs = len(diffNw['mainstem_segs'])
            firstSeg = diffNw['mainstem_segs'][0]
            lastSeg = diffNw['mainstem_segs'][-1]
            preRunLog.write("Number of mainstem segments: "+str(nMainSegs)+"\n")
            preRunLog.write("First and last segment ID: "+str(firstSeg)+"   "+str(lastSeg)+"\n")
            nTribSegs = len(diffNw['tributary_segments'])
            preRunLog.write("Number of tributary segments: "+str(nTribSegs)+"\n")
            connGraphLength = len(diffNw['connections'])
            revConnGraphLength = len(diffNw['rconn'])
            preRunLog.write("Connections in network: "+str(connGraphLength)+"\n")
            preRunLog.write("Reverse connections in network: "+str(revConnGraphLength)+"\n")
            paramDf_Columns = [column for column in diffNw['param_df'].columns]
            preRunLog.write("Diffusive parameters:\n")
    
            for paramDf_Col in paramDf_Columns:
                preRunLog.write(str(paramDf_Col)+"  ")
            preRunLog.write("\n")
            preRunLog.write("-----\n")

        if (not topobathy_df.empty):    
            topoIDs = topobathy_df.index
            topoTraces = len(topoIDs)
            topoTracesUnique = len(set(topoIDs))
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "Number of topobathy profiles: "+str(topoTraces)
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)
            preRunLog.write("Number of segment IDs with topobathy profiles: "+str(topoTracesUnique)+"\n")
            preRunLog.write("-----\n")
        else:
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "No topobathy profiles."
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)
            preRunLog.write("-----\n")            

        if (not coastal_boundary_depth_df.empty):    
            coastalIDs = coastal_boundary_depth_df.index
            coastalTraces = len(coastalIDs)
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "Number of segments with coastal boundary condition: "+str(coastalTraces)
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)            
            preRunLog.write("-----\n")
        else:
            preRunLog.write("\n")
            preRunLog.write("-----\n")
            outPutStr = "No coastal boundary condition."
            preRunLog.write(outPutStr+"\n")
            LOG.info(outPutStr)  
            preRunLog.write("-----\n")   

        preRunLog.write("\n")


def build_compute_package(
    job: ComputationJob,
    forcing: ForcingData,
    assimilation_data: AssimilationData,
    config: ComputeConfig,
    interorder_boundaries: dict[ReachId, BoundaryCondition],
) -> ComputeInputs:
    """Assemble a ComputeInputs package for a computation job by subsetting all forcing and DA data to the job's reaches."""
    # Build qlats
    qlat_pos = forcing.qlat_idx.get_indexer(job.river_index)
    _assert_channel_rows_present(
        "qlats", qlat_pos, job.river_index, job.waterbody_set,
    )
    if job.lake_mask is not None:
        # Force lake-row NaN-fill regardless of whether the
        # source qlats happens to contain those IDs.
        qlat_pos = np.where(job.lake_mask, -1, qlat_pos)
    qlat_sub = pd.DataFrame(
        _reindex_via_take(forcing.qlat_vals, qlat_pos),
        index=job.river_reaches,
        columns=forcing.qlat_cols,
    )
    # Build q0
    q0_pos = forcing.q0_idx.get_indexer(job.river_index)
    _assert_channel_rows_present(
        "q0", q0_pos, job.river_index, job.waterbody_set,
    )
    if job.lake_mask is not None:
        q0_pos = np.where(job.lake_mask, -1, q0_pos)
    q0_sub = pd.DataFrame(
        _reindex_via_take(forcing.q0_vals, q0_pos),
        index=job.river_index,
        columns=forcing.q0_cols,
    )

    # Build eloss
    # eloss_df is intentionally lenient: the legacy
    # ``eloss_df.reindex(...).fillna(0.0)`` swallowed missing
    # channel rows the same way (0-fill, no error). The
    # ``fill_value=0.0`` below preserves that semantic for both
    # missing channel rows and lake extension rows.
    eloss_pos = forcing.eloss_idx.get_indexer(job.river_index)
    eloss_sub = pd.DataFrame(
        _reindex_via_take(forcing.eloss_vals, eloss_pos, fill_value=0.0),
        index=job.river_index,
        columns=forcing.eloss_cols,
    )

    # Update h0. This in-place update of the job's waterbody frame is the
    # run-set state-transfer mechanism: the execution plan (and its per-job
    # frames) is reused across run sets, and each run set re-applies the
    # network's current waterbody elevations here before packaging.
    job.waterbodies_df.update(forcing.wbody_init)

    # Build streamflow DA dataframes
    usgs_df_sub, lastobs_df_sub, da_positions_list_byseg = _prep_da_dataframes(
        assimilation_data.usgs_df, assimilation_data.lastobs_df, job.river_reaches
    )
    da_positions_list_byreach, da_positions_list_bygage = _prep_da_positions_byreach(
        job.routing_paths, lastobs_df_sub.index
    )

    # prepare reservoir DA data
    (
        reservoir_usgs_df_sub,
        reservoir_usgs_df_time,
        reservoir_usgs_update_time,
        reservoir_usgs_prev_persisted_flow,
        reservoir_usgs_persistence_update_time,
        reservoir_usgs_persistence_index,
        reservoir_usace_df_sub,
        reservoir_usace_df_time,
        reservoir_usace_update_time,
        reservoir_usace_prev_persisted_flow,
        reservoir_usace_persistence_update_time,
        reservoir_usace_persistence_index,
        reservoir_usbr_df_sub,
        reservoir_usbr_df_time,
        reservoir_usbr_update_time,
        reservoir_usbr_prev_persisted_flow,
        reservoir_usbr_persistence_update_time,
        reservoir_usbr_persistence_index,
        reservoir_rfc_df_sub,
        reservoir_rfc_totalCounts,
        reservoir_rfc_file,
        reservoir_rfc_use_forecast,
        reservoir_rfc_timeseries_idx,
        reservoir_rfc_update_time,
        reservoir_rfc_da_timestep,
        reservoir_rfc_persist_days,
        gl_df_sub,
        gl_parm_lake_id_sub,
        gl_param_flows_sub,
        gl_param_time_sub,
        gl_param_update_time_sub,
        gl_climatology_df_sub,
        waterbody_types_df_sub,
    ) = _prep_reservoir_da_dataframes(
        assimilation_data.reservoir_usgs_df,
        assimilation_data.reservoir_usgs_param_df,
        assimilation_data.reservoir_usace_df,
        assimilation_data.reservoir_usace_param_df,
        assimilation_data.reservoir_usbr_df,
        assimilation_data.reservoir_usbr_param_df,
        assimilation_data.reservoir_rfc_df,
        assimilation_data.reservoir_rfc_param_df,
        assimilation_data.great_lakes_df,
        assimilation_data.great_lakes_param_df,
        assimilation_data.great_lakes_climatology_df,
        job.waterbodies_types_df,
        config.t0,
        config.from_files,
    )

    return ComputeInputs(
        nsteps=config.nts,
        dt=config.dt,
        qts_subdivisions=config.qts_subdivisions,
        reaches_wTypes=job.reach_types,
        upstream_connections=job.connections,
        data_idx=job.river_reaches,
        data_cols=job.river_fields,
        data_values=job.river_values,
        # to_numpy(dtype=...) does not force a copy (copy
        # defaults to False), so it returns a view when the
        # source already matches the requested dtype (q0 is
        # float32 after build_channel_initial_state, qlats is
        # float32 after np.stack of CHRTOUT data); for eloss_df
        # (built as pd.DataFrame(0.0, ...) which is float64)
        # the float64 -> float32 cast still forces a copy.
        initial_conditions=q0_sub.to_numpy(dtype="float32"),
        qlat_values=qlat_sub.to_numpy(dtype="float32"),
        eloss_values=eloss_sub.to_numpy(dtype="float32"),
        ssout=config.ssout,
        lake_numbers_col=job.waterbody_reaches,
        wbody_cols=job.waterbodies_df.to_numpy(dtype="float64"),  # kernel: const double[:,:]
        data_assimilation_parameters=config.data_assimilation_parameters,
        reservoir_types=job.waterbody_types,
        reservoir_type_specified=config.waterbody_type_specified,
        model_start_time=config.t0_str,
        usgs_values=usgs_df_sub.to_numpy(dtype="float32"),
        usgs_positions=np.array(da_positions_list_byseg, dtype="int32"),
        usgs_positions_reach=np.array(da_positions_list_byreach, dtype="int32"),
        usgs_positions_gage=np.array(da_positions_list_bygage, dtype="int32"),
        lastobs_values_init=lastobs_df_sub.get(
            "lastobs_discharge",
            pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32"),
        ).to_numpy(dtype="float32"),
        time_since_lastobs_init=lastobs_df_sub.get(
            "time_since_lastobs",
            pd.Series(index=lastobs_df_sub.index, name="Null", dtype="float32"),
        ).to_numpy(dtype="float32"),
        da_decay_coefficient=config.da_decay_coefficient,
        # USGS Hybrid Reservoir DA data
        reservoir_usgs_obs=reservoir_usgs_df_sub.to_numpy(dtype="float32"),
        reservoir_usgs_wbody_idx=reservoir_usgs_df_sub.index.to_numpy(dtype="int64"),
        reservoir_usgs_time=reservoir_usgs_df_time.astype("float32"),
        reservoir_usgs_update_time=reservoir_usgs_update_time.astype("float32"),
        reservoir_usgs_prev_persisted_flow=reservoir_usgs_prev_persisted_flow.astype(
            "float32"
        ),
        reservoir_usgs_persistence_update_time=reservoir_usgs_persistence_update_time.astype(
            "float32"
        ),
        reservoir_usgs_persistence_index=reservoir_usgs_persistence_index.astype(
            "float32"
        ),
        # USACE Hybrid Reservoir DA data
        reservoir_usace_obs=reservoir_usace_df_sub.to_numpy(dtype="float32"),
        reservoir_usace_wbody_idx=reservoir_usace_df_sub.index.to_numpy(dtype="int64"),
        reservoir_usace_time=reservoir_usace_df_time.astype("float32"),
        reservoir_usace_update_time=reservoir_usace_update_time.astype("float32"),
        reservoir_usace_prev_persisted_flow=reservoir_usace_prev_persisted_flow.astype(
            "float32"
        ),
        reservoir_usace_persistence_update_time=reservoir_usace_persistence_update_time.astype(
            "float32"
        ),
        reservoir_usace_persistence_index=reservoir_usace_persistence_index.astype(
            "float32"
        ),
        # USBR Hybrid Reservoir DA data
        reservoir_usbr_obs=reservoir_usbr_df_sub.to_numpy(dtype="float32"),
        reservoir_usbr_wbody_idx=reservoir_usbr_df_sub.index.to_numpy(dtype="int64"),
        reservoir_usbr_time=reservoir_usbr_df_time.astype("float32"),
        reservoir_usbr_update_time=reservoir_usbr_update_time.astype("float32"),
        reservoir_usbr_prev_persisted_flow=reservoir_usbr_prev_persisted_flow.astype(
            "float32"
        ),
        reservoir_usbr_persistence_update_time=reservoir_usbr_persistence_update_time.astype(
            "float32"
        ),
        reservoir_usbr_persistence_index=reservoir_usbr_persistence_index.astype(
            "float32"
        ),
        # RFC Reservoir DA data
        reservoir_rfc_obs=reservoir_rfc_df_sub.to_numpy(dtype="float32"),
        reservoir_rfc_wbody_idx=reservoir_rfc_df_sub.index.to_numpy(dtype="int64"),
        reservoir_rfc_totalCounts=reservoir_rfc_totalCounts.astype("int32"),
        reservoir_rfc_file=reservoir_rfc_file,
        reservoir_rfc_use_forecast=reservoir_rfc_use_forecast.astype("int32"),
        reservoir_rfc_timeseries_idx=reservoir_rfc_timeseries_idx.astype("int32"),
        reservoir_rfc_update_time=reservoir_rfc_update_time.astype("float32"),
        reservoir_rfc_da_timestep=reservoir_rfc_da_timestep.astype("int32"),
        reservoir_rfc_persist_days=reservoir_rfc_persist_days.astype("int32"),
        # Great Lakes DA data
        great_lakes_idx=gl_df_sub.lake_id.to_numpy(dtype="int32"),
        great_lakes_times=gl_df_sub.time.to_numpy(dtype="int32"),
        great_lakes_discharge=gl_df_sub.Discharge.to_numpy(dtype="float32"),
        great_lakes_param_idx=gl_parm_lake_id_sub.astype("int32"),
        great_lakes_param_prev_assim_flow=gl_param_flows_sub.astype("float32"),
        great_lakes_param_prev_assim_times=gl_param_time_sub.astype("int32"),
        great_lakes_param_update_times=gl_param_update_time_sub.astype("int32"),
        great_lakes_climatology=gl_climatology_df_sub.to_numpy(dtype="float32"),
        upstream_results=interorder_boundaries,
        assume_short_ts=config.assume_short_ts,
        return_courant=config.return_courant,
        from_files=config.from_files,
        qlat_add_loc=config.qlat_add_loc_c
    )


def compute_routing(
    config: ComputeConfig,
    topology: NetworkTopology,
    reach_data: ReachData,
    waterbody_data: WaterbodyData,
    forcing_data: ForcingData,
    assimilation_data: AssimilationData,
    execution_plan: Union[ExecutionPlan, None],
) -> tuple[RoutingResultsCollection, ExecutionPlan]:
    """Execute all computation batches in dependency order and return results alongside the reusable execution plan."""
    if execution_plan is None:
        execution_plan = ExecutionPlan(
            config.parallel_compute_method,
            topology,
            reach_data,
            waterbody_data,
            assimilation_data,
            config.subnetwork_target_size,
        )

    # Execute routing
    results = []
    with Parallel(n_jobs=config.cpu_pool, backend=config.backend) as parallel:
        # Iterate over groups that depend on upstream data (routing levels)
        for routing_level in sorted(execution_plan.batches.keys(), reverse=True):
            computation_batch = execution_plan.batches[routing_level]
            jobs = []
            # Iterate over fully parallel groups
            for compute_job in computation_batch:
                bcs = execution_plan.boundary_conditions.generate_view(
                    compute_job.offnetwork_upstreams
                )
                package = build_compute_package(
                    compute_job, forcing_data, assimilation_data, config, bcs
                )

                jobs.append(delayed(config.compute_function)(**vars(package)))

            # Compute and collect results. joblib's stub returns a
            # list/generator union; with the default backend and no
            # return_as override this is always a list of result tuples.
            level_results = cast("list[tuple]", parallel(jobs))
            results.extend(level_results)
            if routing_level > 0:
                execution_plan.update_boundary_conditions(level_results, routing_level)

    # Clear boundary condition data to save some memory
    execution_plan.boundary_conditions.clear_data()

    # Format and return
    return RoutingResultsCollection(results), execution_plan


def compute_nhd_routing_v02(
    connections: DownstreamGraph,
    rconn: UpstreamGraph,
    wbody_conn: dict,
    reaches_bytw: dict[TailwaterId, OrderedRoutingPaths],
    compute_func_name: str,
    parallel_compute_method: PARALLEL_COMPUTE_METHODS,
    subnetwork_target_size: int,
    cpu_pool: int,
    t0: datetime,
    dt: float,
    nts: int,
    qts_subdivisions: int,
    independent_networks: dict[TailwaterId, DownstreamGraph],
    param_df: pd.DataFrame,
    q0: pd.DataFrame,
    qlats: pd.DataFrame,
    eloss_df: pd.DataFrame,
    ssout: float,
    usgs_df: pd.DataFrame,
    lastobs_df: pd.DataFrame,
    reservoir_usgs_df: pd.DataFrame,
    reservoir_usgs_param_df: pd.DataFrame,
    reservoir_usace_df: pd.DataFrame,
    reservoir_usace_param_df: pd.DataFrame,
    reservoir_usbr_df: pd.DataFrame,
    reservoir_usbr_param_df: pd.DataFrame,
    reservoir_rfc_df: pd.DataFrame,
    reservoir_rfc_param_df: pd.DataFrame,
    great_lakes_df: pd.DataFrame,
    great_lakes_param_df: pd.DataFrame,
    great_lakes_climatology_df: pd.DataFrame,
    da_parameter_dict: dict[str, Any],
    assume_short_ts: bool,
    return_courant: bool,
    waterbodies_df: pd.DataFrame,
    data_assimilation_parameters: dict[str, Any],
    waterbody_types_df: pd.DataFrame,
    waterbody_type_specified: bool,
    subnetwork_list: Union[ExecutionPlan, list],
    flowveldepth_interorder: dict = {},
    from_files: bool = True,
    qlat_add_loc: Literal["top", "middle", "bottom"] = "middle",
) -> tuple[RoutingResultsCollection, ExecutionPlan]:
    """Build typed routing objects from legacy flat arguments and delegate to compute_routing."""
    if flowveldepth_interorder:
        raise NotImplementedError(
            "flowveldepth_interorder (inter-domain boundary exchange for coupled "
            "BMI runs) is not supported by the refactored compute path. The "
            "legacy implementation injected these values as upstream boundary "
            "conditions and wrote tailwater results back into this dict; "
            "silently ignoring them would route without the upstream inflow. "
            "Wire the dict through ExecutionPlan.boundary_conditions to "
            "support coupled runs."
        )
    param_df["dt"] = dt
    param_df = param_df.astype("float32")
    config = ComputeConfig(
        nts=nts,
        dt=dt,
        qts_subdivisions=qts_subdivisions,
        t0=t0,
        ssout=ssout,
        data_assimilation_parameters=data_assimilation_parameters,
        da_parameter_dict=da_parameter_dict,
        waterbody_type_specified=waterbody_type_specified,
        assume_short_ts=assume_short_ts,
        return_courant=return_courant,
        from_files=from_files,
        cpu_pool=cpu_pool,
        parallel_compute_method=parallel_compute_method,
        qlat_add_loc=qlat_add_loc,
        compute_func_name=compute_func_name,
        subnetwork_target_size=subnetwork_target_size
    )
    topology = NetworkTopology(
        connections = connections,
        reverse_connections=rconn,
        paths_by_tailwater=reaches_bytw,
        connections_by_tw=independent_networks,
    )
    reach_data = ReachData(param_df)
    waterbody_data = WaterbodyData(waterbodies_df, waterbody_types_df)
    wbody_init = waterbodies_df[["h0", "qd0"]]
    forcing_data = ForcingData(qlats, q0, eloss_df, wbody_init)
    assimilation_data = AssimilationData(
        reservoir_usgs_df = reservoir_usgs_df,
        reservoir_usgs_param_df = reservoir_usgs_param_df,
        reservoir_usace_param_df = reservoir_usace_param_df,
        reservoir_usace_df = reservoir_usace_df,
        reservoir_usbr_df = reservoir_usbr_df,
        reservoir_usbr_param_df = reservoir_usbr_param_df,
        reservoir_rfc_df = reservoir_rfc_df,
        reservoir_rfc_param_df = reservoir_rfc_param_df,
        great_lakes_df = great_lakes_df,
        great_lakes_param_df = great_lakes_param_df,
        great_lakes_climatology_df = great_lakes_climatology_df,
        usgs_df = usgs_df,
        lastobs_df = lastobs_df,
    )

    if isinstance(subnetwork_list, ExecutionPlan):
        execution_plan = subnetwork_list
    else:
        # Legacy callers pass a [None, None, None] sentinel (or another
        # placeholder list) on the first run set -> build a fresh plan.
        execution_plan = None
    results, execution_plan = compute_routing(config, topology, reach_data, waterbody_data, forcing_data, assimilation_data, execution_plan)
    return results, execution_plan


def compute_diffusive_routing(
    results,
    diffusive_network_data,
    cpu_pool,
    t0,
    dt,
    nts,
    q0,
    qlats,
    qts_subdivisions,
    usgs_df,
    lastobs_df,
    da_parameter_dict,
    waterbodies_df,
    topobathy,
    refactored_diffusive_domain,
    refactored_reaches,
    coastal_boundary_depth_df, 
    unrefactored_topobathy,
    ):

    results_diffusive = []
    for tw in diffusive_network_data: # <------- TODO - by-network parallel loop, here.
        trib_segs = None
        trib_flow = None
        # extract junction inflows from results array
        for j, i in enumerate(results):
            x = np.in1d(i[0], diffusive_network_data[tw]['tributary_segments'])
            if sum(x) > 0:
                if j == 0:
                    trib_segs = i[0][x]
                    trib_flow = i[1][x, ::3]
                else:
                    # trib_segs and trib_flow are always set together; the
                    # second check gives the type checker that invariant.
                    if trib_segs is None or trib_flow is None:
                        trib_segs = i[0][x]
                        trib_flow = i[1][x, ::3]
                    else:
                        trib_segs = np.append(trib_segs, i[0][x])
                        trib_flow = np.append(trib_flow, i[1][x, ::3], axis = 0)  

        # create DataFrame of junction inflow data            
        junction_inflows = pd.DataFrame(data = trib_flow, index = trib_segs)

        if not topobathy.empty:
            # create topobathy data for diffusive mainstem segments related to this given tw segment        
            if refactored_diffusive_domain:
                topobathy_bytw               = topobathy.loc[refactored_diffusive_domain[tw]['rlinks']] 
                # TODO: missing topobathy data in one of diffuisve domains, so inactivate the next line for now. 
                #unrefactored_topobathy_bytw  = unrefactored_topobathy.loc[diffusive_network_data[tw]['mainstem_segs']]
                unrefactored_topobathy_bytw = pd.DataFrame()
            else:
                topobathy_bytw               = topobathy.loc[diffusive_network_data[tw]['mainstem_segs']] 
                unrefactored_topobathy_bytw = pd.DataFrame()
            
        else:
            topobathy_bytw = pd.DataFrame()
            unrefactored_topobathy_bytw = pd.DataFrame()

        # diffusive streamflow DA activation switch
        #if da_parameter_dict['diffusive_streamflow_nudging']==True:
        if 'diffusive_streamflow_nudging' in da_parameter_dict:
            diffusive_usgs_df = usgs_df
        else:
            diffusive_usgs_df = pd.DataFrame()

        # tw in refactored hydrofabric
        if refactored_diffusive_domain:
            refactored_tw = refactored_diffusive_domain[tw]['refac_tw']
            refactored_diffusive_domain_bytw = refactored_diffusive_domain[tw]
            refactored_reaches_byrftw        = refactored_reaches[refactored_tw]
        else:
            refactored_diffusive_domain_bytw = None
            refactored_reaches_byrftw        = None
  
        # coastal boundary depth input data at TW
        if tw in coastal_boundary_depth_df.index:
            coastal_boundary_depth_bytw_df = coastal_boundary_depth_df.loc[tw].to_frame().T
        else:
            coastal_boundary_depth_bytw_df = pd.DataFrame()

        # temporary: column names of qlats from HYfeature are currently timestamps. To be consistent with qlats from NHD
        # the column names need to be changed to intergers from zero incrementing by 1
        diffusive_qlats = qlats.copy()
        diffusive_qlats.columns = range(diffusive_qlats.shape[1])  

        # build diffusive inputs
        diffusive_inputs = diff_utils.diffusive_input_data_v02(
            tw,
            diffusive_network_data[tw]['connections'],
            diffusive_network_data[tw]['rconn'],
            diffusive_network_data[tw]['reaches'],
            diffusive_network_data[tw]['mainstem_segs'],
            diffusive_network_data[tw]['tributary_segments'],
            None, # place holder for diffusive parameters
            diffusive_network_data[tw]['param_df'],
            diffusive_qlats,
            q0,
            junction_inflows,
            qts_subdivisions,
            t0,
            nts,
            dt,
            waterbodies_df,
            topobathy_bytw,
            diffusive_usgs_df,
            refactored_diffusive_domain_bytw,
            refactored_reaches_byrftw, 
            coastal_boundary_depth_bytw_df,
            unrefactored_topobathy_bytw,
        )

        # run the simulation
        out_q, out_elv, out_depth = diffusive.compute_diffusive(diffusive_inputs)

        # unpack results
        rch_list, dat_all = diff_utils.unpack_output(
            diffusive_inputs['pynw'], 
            diffusive_inputs['ordered_reaches'], 
            out_q, 
            out_depth, #out_elv
        )
        
        # mask segments for which we already have MC solution
        x = np.in1d(rch_list, diffusive_network_data[tw]['tributary_segments'])
        
        results_diffusive.append(
            (
                rch_list[~x], dat_all[~x,3:], 0,
                # place-holder for streamflow DA parameters
                (np.asarray([]), np.asarray([]), np.asarray([])),
                # place-holder for reservoir DA parameters
                (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])),
                (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])),
                # place holder for reservoir inflows
                np.zeros(dat_all[~x,3::3].shape),
                # place-holder for rfc DA parameters
                (np.asarray([]), np.asarray([]), np.asarray([])),
                # place-holder for nudge values
                (np.empty(shape=(0, nts + 1), dtype='float32')),
                # place-holder for great lakes DA values/parameters
                (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])),
            )
        )

    return results_diffusive


class _RoutingResultsParser:
    def __init__(self, raw_results: tuple | list):
        self._raw = raw_results

    def __getitem__(self, index: int):
        return self._raw[index]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    @property
    def ids(self) -> IntpArray:
        """Segment IDs as 1D array"""
        return self._raw[0]

    def _append(self, a: NDArray[Any], b: NDArray[Any]):
        axis = len(a.shape) - 1
        return np.concatenate([a, b], axis=axis)

    def append(self, other: Self):
        copy: list[Any] = list(self)
        # skip first element as that is the ID
        for i in range(1, len(self)):
            copy[i] = self._append(self[i], other[i])
        return self.__class__(copy)

    @classmethod
    def merge(cls, to_merge: Sequence[Self]):
        size = len(to_merge[0])
        data: list[Any] = [None] * size
        for i in range(size):
            data[i] = np.concatenate([r[i] for r in to_merge], axis=0)
        return cls(data)

    def align_ids(self, source: Self):
        if self.ids.size and not np.array_equal(self.ids, source.ids):
            copy: list[Any] = [None] * len(self)
            sorter = np.argsort(source.ids)
            for i, item in enumerate(self):
                copy[i] = item[sorter]
            return self.__class__(copy)
        return self

    def _set_index(self, value, index: int):
        if isinstance(self._raw, tuple):
            self._raw = list(self._raw)
        self._raw[index] = value


class RoutingResultsCollection:
    def __init__(self, results: Iterable[Any]):
        # Each element is a raw kernel result tuple, or an existing
        # RoutingResults (itself indexable), e.g. from append_timesteps.
        self.results = [RoutingResults(r) for r in results]

    def __getitem__(self, index: int):
        return self.results[index]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def flow_velocity_depth(self, nts: int, drop_ql: bool = False):
        columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d", "ql"]]
        ).to_flat_index()
        dfs = []
        for result in self.results:
            df = pd.DataFrame(
                result.flow,
                index=result.ids,
                columns=columns,
            )
            dfs.append(df)
        # copy=False is valid in pandas 2.x (a no-op under copy-on-write) but
        # absent from pandas-stubs' concat signature.
        flowveldepth = pd.concat(dfs, copy=False)  # pyright: ignore[reportCallIssue]
        if drop_ql:
            flowveldepth = flowveldepth.drop(columns=[
                col for col in flowveldepth.columns if col[1] == "ql"
            ])
        return flowveldepth

    def waterbodies(self, nts: int):
        columns = pd.MultiIndex.from_product(
            [range(nts), ["i"]]
        ).to_flat_index()
        dfs = []
        for result in self.results:
            df = pd.DataFrame(
                result.upstream,
                index=result.ids,
                columns=columns,
            )
            dfs.append(df)
        # copy=False: see flow_velocity_depth (pandas-stubs gap).
        return pd.concat(dfs, copy=False)  # pyright: ignore[reportCallIssue]

    def courant(self, nts: int):
        columns = pd.MultiIndex.from_product(
            [range(nts), ["cn", "ck", "X"]]
        ).to_flat_index()
        dfs = []
        for result in self.results:
            df = pd.DataFrame(
                result.courant,
                index=result.ids,
                columns=columns,
            )
            dfs.append(df)
        # copy=False: see flow_velocity_depth (pandas-stubs gap).
        return pd.concat(dfs, copy=False)  # pyright: ignore[reportCallIssue]

    def nudge(self):
        return np.concatenate(
            [result.nudge for result in self.results]
        )

    def usgs_position_ids(self):
        return np.concatenate(
            [result.usgs_reservoir.ids for result in self.results]
        )

    def merged_results(self) -> RoutingResults:
        """Merge the separate results into one single results."""
        if len(self.results) > 1:
            merged = RoutingResults([None] * len(self.results[0]))
            merged.ids = np.concatenate([r.ids for r in self.results])
            merged.flow = np.concatenate([r.flow for r in self.results], axis=0)
            merged.courant = 0 # fix when this is no longer a placeholder
            merged.lastobs = RoutingLastObs.merge([r.lastobs for r in self.results])
            merged.usgs_reservoir = RoutingReservoir.merge([r.usgs_reservoir for r in self.results])
            merged.usace_reservoir = RoutingReservoir.merge([r.usace_reservoir for r in self.results])
            merged.usbr_reservoir = RoutingReservoir.merge([r.usbr_reservoir for r in self.results])
            merged.upstream = np.concatenate([r.upstream for r in self.results], axis=0)
            merged.rfc_reservoir = RoutingRfc.merge([r.rfc_reservoir for r in self.results])
            merged.nudge = np.concatenate([r.nudge for r in self.results], axis=0)
            merged.great_lakes = RoutingGreatLakes.merge([r.great_lakes for r in self.results])
            return merged
        return self.results[0]

    def append_timesteps(self, other: RoutingResultsCollection):
        a = self.merged_results()
        b = other.merged_results()
        b = b.align_ids(a)
        return RoutingResultsCollection([a.append(b)])


class RoutingResults(_RoutingResultsParser):
    # Parser instances are only ever aligned/appended with their own kind
    # (see append_timesteps), so narrowing the base's parameter is safe in
    # practice even though it is not LSP-clean.
    def align_ids(self, source: RoutingResults):  # pyright: ignore[reportIncompatibleMethodOverride]
        if not np.array_equal(self.ids, source.ids):
            self = self.__class__(list(self))
            sorter = np.argsort(source.ids)
            self.ids = self.ids[sorter]
            self.flow = self.flow[sorter]
            if self.upstream.size > 0:
                self.upstream = self.upstream[sorter]
            if self.nudge.size > 0:
                self.nudge = self.nudge[sorter]
        self.usgs_reservoir = self.usgs_reservoir.align_ids(source.usgs_reservoir)
        self.usace_reservoir = self.usace_reservoir.align_ids(source.usace_reservoir)
        self.usbr_reservoir = self.usbr_reservoir.align_ids(source.usbr_reservoir)
        self.rfc_reservoir = self.rfc_reservoir.align_ids(source.rfc_reservoir)
        self.great_lakes = self.great_lakes.align_ids(source.great_lakes)
        return self

    def append(self, other: RoutingResults):  # pyright: ignore[reportIncompatibleMethodOverride]
        appended = self.__class__(list(self))
        appended.flow = self._append(self.flow, other.flow)
        appended.lastobs = self.lastobs.append(other.lastobs)
        appended.usgs_reservoir = self.usgs_reservoir.append(other.usgs_reservoir)
        appended.usace_reservoir = self.usace_reservoir.append(other.usace_reservoir)
        appended.usbr_reservoir = self.usbr_reservoir.append(other.usbr_reservoir)
        appended.upstream = self._append(self.upstream, other.upstream)
        appended.rfc_reservoir = self.rfc_reservoir.append(other.rfc_reservoir)
        # remove leading timestep from other's nudge
        appended.nudge = self._append(self.nudge, other.nudge[:, 1:])
        appended.great_lakes = self.great_lakes.append(other.great_lakes)
        return appended

    @property
    def ids(self) -> IntpArray:
        """Catchment IDs as 1D array"""
        return self._raw[0]
    @ids.setter
    def ids(self, value):
        self._set_index(value, 0)

    @property
    def flow(self) -> Float32Array:
        """Flow velocity depth 2D array: (num_ids, nts * 4)"""
        return self._raw[1]
    @flow.setter
    def flow(self, value):
        self._set_index(value, 1)

    @property
    def courant(self) -> Literal[0]:
        """Is currently a placeholder, so the value will always be 0."""
        return self._raw[2]
    @courant.setter
    def courant(self, value):
        self._set_index(value, 2)

    @property
    def lastobs(self):
        return RoutingLastObs(self._raw[3])
    @lastobs.setter
    def lastobs(self, value):
        self._set_index(list(value), 3)

    @property
    def usgs_reservoir(self):
        return RoutingReservoir(self._raw[4])
    @usgs_reservoir.setter
    def usgs_reservoir(self, value):
        self._set_index(list(value), 4)

    @property
    def usace_reservoir(self):
        return RoutingReservoir(self._raw[5])
    @usace_reservoir.setter
    def usace_reservoir(self, value):
        self._set_index(list(value), 5)

    @property
    def usbr_reservoir(self):
        return RoutingReservoir(self._raw[6])
    @usbr_reservoir.setter
    def usbr_reservoir(self, value):
        self._set_index(list(value), 6)

    @property
    def upstream(self) -> Float32Array:
        """Upstream 2D array: (num_ids, nts)"""
        return self._raw[7]
    @upstream.setter
    def upstream(self, value):
        self._set_index(value, 7)

    @property
    def rfc_reservoir(self):
        return RoutingRfc(self._raw[8])
    @rfc_reservoir.setter
    def rfc_reservoir(self, value):
        self._set_index(value, 8)

    @property
    def nudge(self) -> Float32Array:
        """Nudge 2D array: (num_ids, nts + 1)"""
        return self._raw[9]
    @nudge.setter
    def nudge(self, value):
        self._set_index(value, 9)

    @property
    def great_lakes(self):
        return RoutingGreatLakes(self._raw[10])
    @great_lakes.setter
    def great_lakes(self, value):
        self._set_index(list(value), 10)


class RoutingLastObs(_RoutingResultsParser):
    @property
    def times(self) -> Float32Array:
        return self._raw[1]

    @property
    def values(self) -> Float32Array:
        return self._raw[2]


class RoutingReservoir(_RoutingResultsParser):
    @property
    def update_times(self) -> Float32Array:
        return self._raw[1]

    @property
    def persisted_outflow(self) -> Float32Array:
        return self._raw[2]

    @property
    def persistence_index(self) -> Float32Array:
        return self._raw[3]

    @property
    def persistence_update_time(self) -> Float32Array:
        return self._raw[4]


class RoutingRfc(_RoutingResultsParser):
    @property
    def update_times(self) -> Float32Array:
        return self._raw[1]

    @property
    def timeseries(self) -> IntpArray:
        return self._raw[2]


class RoutingGreatLakes(_RoutingResultsParser):
    @property
    def outflows(self) -> Float32Array:
        return self._raw[1]

    @property
    def timestamps(self) -> IntpArray:
        return self._raw[2]

    @property
    def update_times(self) -> IntpArray:
        return self._raw[3]
