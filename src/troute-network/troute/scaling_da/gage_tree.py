"""Precompute per-gage upstream trees for the streamflow DA correction.

For each USGS gage we BFS upstream from its NHF segment through reverse
connectivity, halting any branch that hits another gage segment or a
flowpath inside a waterbody. The resulting tree is encoded as flat NumPy
arrays (segment order, per-segment drainage area, per-segment theta, and the
parent index and confluence flag each node needs) suitable for tight
per-timestep DA loops in :mod:`troute.routing.fast_reach.scaling_da`.

This is the structure the compiled spread kernel consumes;
:func:`build_one_gage_tree` is topology-agnostic (it takes a reverse
connectivity mapping, an area lookup, and a stop set), so the t-route network
builder in a later stage supplies those from the real network instead of the
proof-of-concept's Ohio loader.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

LOG = logging.getLogger("TROUTE")

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


__all__ = ["GageTree", "build_one_gage_tree"]


@dataclass(frozen=True)
class GageTree:
    """Precomputed per-gage upstream-tree structure for area-scaling DA.

    ``seg_order`` lists the fp_ids in the gage's upstream tree (BFS order,
    gage at position 0), bounded by the stop rule (other gages, waterbody
    interiors). ``seg_parent_idx`` and ``step_is_junction`` carry the
    parent/junction metadata the white paper's distribution needs: the
    correction scales by drainage area (Eq. 2) along linear steps and splits
    between branches by modeled flow (Edge Case 2) at confluences.

    Attributes
    ----------
    gage_fp : int
        Flowpath ``fp_id`` of the gage segment.
    gage_area_sqkm : float
        ``total_da_sqkm`` of the gage segment (the proposal's ``A_o``).
    seg_order : numpy.ndarray of int64
        Tree-ordered fp_ids; ``seg_order[0] == gage_fp``.
    seg_areas_sqkm : numpy.ndarray of float64
        ``total_da_sqkm`` per tree segment (the proposal's ``A_s``).
    theta : float
        The tree's region theta, resolved once from the gage's VPU at build time.
        A SCALAR, not a per-segment array: the linear step telescopes to the
        closed form dQ_o*(A_s/A_o)^theta only for a constant exponent, so one
        value per gage is the finest resolution the method admits. Holding it here
        makes that a structural invariant instead of a runtime check that could
        fail deep inside a run.
    seg_positions : numpy.ndarray of int64
        Position of each tree segment within the *global* fp_id index
        used by the model output (filled in by :meth:`with_positions`
        once the global ordering is known).
    seg_parent_idx : numpy.ndarray of int64
        For each segment, the index (within ``seg_order``) of its downstream
        (parent) segment in the tree. The gage's entry is ``-1``. BFS order
        guarantees ``seg_parent_idx[j] < j``.
    step_is_junction : numpy.ndarray of bool
        For each segment, whether its parent is a *physical* confluence: two or
        more upstream branches in the full network, counted before the stop rule
        prunes any of them. True selects the flow-ratio split (Edge Case 2);
        False selects the area-scaling step (Eq. 2). So a junction with one
        tributary cut off by an upstream gage or reservoir still flow-ratio-splits
        the surviving tributary. The gage's entry is False.
    """

    gage_fp: int
    gage_area_sqkm: float
    seg_order: NDArray[np.int64]
    seg_areas_sqkm: NDArray[np.float64]
    theta: float
    seg_positions: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    seg_parent_idx: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    step_is_junction: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=np.bool_))
    # Branches the Edge Case 1 stop rule cut off, per tree node, as CSR. They are not tree
    # members; their only role is to appear in their confluence's flow denominator so a
    # surviving sibling gets its true proportional share and the stopped branch's share is
    # left unallocated for the gage that owns it.
    pruned_ptr: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    pruned_segs: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    pruned_positions: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    # Positions carry their OWN pointer: a stop segment can be absent from the flow frame
    # (a waterbody interior outside the routed columns), so the position CSR can be shorter
    # than the segment CSR and the two must not share an index.
    pruned_pos_ptr: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def n_segments(self) -> int:
        """int: Number of segments in the tree (length of ``seg_order``)."""
        return int(self.seg_order.size)

    def with_positions(self, fp_to_position: Mapping[int, int]) -> GageTree:
        """Return a copy with ``seg_positions`` filled from a global mapping.

        Parameters
        ----------
        fp_to_position : Mapping[int, int]
            Mapping ``fp_id -> column position`` in the global modeled-flow frame
            (e.g. ``{fp: i for i, fp in enumerate(q_model.columns)}``). Every
            segment in ``seg_order`` must be present.

        Returns
        -------
        GageTree
            A copy of this tree with ``seg_positions`` set to each segment's
            position in the global ordering; all other fields are shared.

        Raises
        ------
        KeyError
            If a segment in ``seg_order`` is absent from ``fp_to_position``.
        """
        positions = np.fromiter(
            (fp_to_position[int(fp)] for fp in self.seg_order),
            dtype=np.int64,
            count=self.n_segments,
        )
        # A pruned branch absent from the flow frame is dropped, LOUDLY: its
        # flow then leaves the confluence denominator and surviving siblings
        # inherit a share Edge Case 1 says must stay unallocated.
        ptr, cols, dropped = self.pruned_ptr, [], []
        new_ptr = np.zeros(self.n_segments + 1, dtype=np.int64)
        if ptr.size == self.n_segments + 1:
            for i in range(self.n_segments):
                for k in range(int(ptr[i]), int(ptr[i + 1])):
                    pos = fp_to_position.get(int(self.pruned_segs[k]))
                    if pos is not None:
                        cols.append(int(pos))
                    else:
                        dropped.append(int(self.pruned_segs[k]))
                new_ptr[i + 1] = len(cols)
        if dropped:
            LOG.warning(
                "gage tree %s: %d pruned branch segment(s) absent from the routed "
                "flow frame (%s); their flow cannot enter the confluence "
                "denominator, so surviving siblings at those junctions receive a "
                "LARGER share than Edge Case 1 intends.",
                self.gage_fp, len(dropped), dropped[:5],
            )
        return GageTree(
            gage_fp=self.gage_fp,
            gage_area_sqkm=self.gage_area_sqkm,
            seg_order=self.seg_order,
            seg_areas_sqkm=self.seg_areas_sqkm,
            theta=self.theta,
            seg_positions=positions,
            seg_parent_idx=self.seg_parent_idx,
            step_is_junction=self.step_is_junction,
            pruned_ptr=self.pruned_ptr,
            pruned_segs=self.pruned_segs,
            pruned_positions=np.asarray(cols, dtype=np.int64),
            pruned_pos_ptr=new_ptr,
        )



def build_one_gage_tree(
    gage_fp: int,
    rconn: Mapping[int, list[int]],
    area_sqkm: Mapping[int, float],
    stop_segs: frozenset[int],
    *,
    theta: float,
) -> GageTree:
    """Build the upstream tree for a single gage.

    Parameters
    ----------
    gage_fp : int
        Flowpath ``fp_id`` of the gage segment.
    rconn : Mapping[int, list[int]]
        Reverse connectivity: ``fp_id -> list[upstream fp_ids]``.
    area_sqkm : Mapping[int, float]
        ``fp_id -> drainage area`` (km^2) per segment.
    stop_segs : frozenset[int]
        Set of segments at which BFS halts (other gages + waterbody
        interiors). The gage itself must *not* appear here even if it
        belongs to the global gage set.
    theta : float
        Region theta applied to every segment in the tree. A single global
        default is the usual choice.

    Returns
    -------
    GageTree
        The gage's upstream tree, with ``seg_positions`` left empty (fill it via
        :meth:`GageTree.with_positions` once the global column ordering is known).

    Raises
    ------
    ValueError
        If ``gage_fp`` is in ``stop_segs`` (the caller must exclude the gage's own
        segment from the stop set).
    """
    if gage_fp in stop_segs:
        msg = f"gage_fp {gage_fp} cannot be in stop_segs (exclude it from caller)"
        raise ValueError(msg)

    # BFS upstream through reverse connectivity, halting at stop segments.
    # Track each node's parent index and how many upstream branches each node
    # has in the tree (to flag confluences for the flow-ratio split).
    seg_order: list[int] = [gage_fp]
    parent_of_idx: list[int] = [-1]  # parent index per position; gage = -1
    order_index: dict[int, int] = {gage_fp: 0}
    seen: set[int] = {gage_fp}
    queue: deque[int] = deque([gage_fp])
    # Branches the stop rule cuts off, recorded against the confluence they feed:
    # not tree members, but their flow must appear in the split denominator or a
    # surviving sibling inherits their share on lagged rising limbs.
    pruned_of_idx: dict[int, list[int]] = {}
    while queue:
        node = queue.popleft()
        node_idx = order_index[node]
        for u in rconn.get(node, ()):
            if u in stop_segs:
                pruned_of_idx.setdefault(node_idx, []).append(int(u))
                continue
            if u in seen:
                continue
            seen.add(u)
            order_index[u] = len(seg_order)
            seg_order.append(u)
            parent_of_idx.append(node_idx)
            queue.append(u)

    n = len(seg_order)
    seg_order_arr = np.asarray(seg_order, dtype=np.int64)
    seg_areas_arr = np.fromiter(
        (float(area_sqkm.get(int(fp), np.nan)) for fp in seg_order_arr),
        dtype=np.float64,
        count=n,
    )
    seg_parent_idx_arr = np.asarray(parent_of_idx, dtype=np.int64)
    # A step into a child is a confluence step when its parent is a *physical*
    # confluence: two or more upstream branches in the full network (``rconn``),
    # counted before the stop rule prunes any of them. This is the Edge Case 1 +
    # 2 interaction: if a tributary is cut off by an upstream gage or reservoir,
    # the surviving tributary at that junction still takes the flow-ratio split
    # (Q_branch/Q_parent), not area-scaling.
    step_is_junction_arr = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        parent_fp = int(seg_order_arr[seg_parent_idx_arr[i]])
        step_is_junction_arr[i] = len(rconn.get(parent_fp, ())) >= 2

    # CSR over tree nodes: pruned_ptr[i]:pruned_ptr[i+1] indexes pruned_segs, the branches
    # cut off at node i. Flat rather than a list of lists so the compiled kernel can read it.
    pruned_ptr = np.zeros(n + 1, dtype=np.int64)
    flat: list[int] = []
    for i in range(n):
        flat.extend(pruned_of_idx.get(i, ()))
        pruned_ptr[i + 1] = len(flat)
    pruned_segs_arr = np.asarray(flat, dtype=np.int64)

    return GageTree(
        gage_fp=int(gage_fp),
        gage_area_sqkm=float(seg_areas_arr[0]),
        seg_order=seg_order_arr,
        seg_areas_sqkm=seg_areas_arr,
        theta=float(theta),
        seg_parent_idx=seg_parent_idx_arr,
        step_is_junction=step_is_junction_arr,
        pruned_ptr=pruned_ptr,
        pruned_segs=pruned_segs_arr,
    )
