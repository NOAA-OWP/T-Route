"""Per-timestep application of the simple-scaling DA correction.

The simple-scaling DA *kernel*: it lives in ``troute.routing.fast_reach`` beside
the nudging kernel (:mod:`simple_da`) and the routing kernels, mirroring t-route's
split of data assimilation -- the per-gage *tree* setup is built in
``troute.scaling_da`` (troute-network), and the *correction kernel* is here
(troute-routing). This module assembles the per-tree buffers and hands them to
the compiled spread kernel (``scaling_da_kernel``). The pure-NumPy reference the
kernel was ported from, and the 2000-case fuzz that proved bit-level parity,
were retired once the port was proven; both are in git history
(``scaling_da_numpy_reference.py``, ``test_scaling_da_cython_equiv.py``).

Trees are precomputed once by
:func:`troute.scaling_da.gage_tree.build_one_gage_tree`; this module's
:func:`apply_scaling_da` consumes them and produces both the
corrected discharge time series and the per-segment delta-Q diagnostic.

The gage correction ``dQ_o`` is distributed upstream by the white paper's
two prescriptions, combined as the white paper writes them:

- Along a linear step (one upstream branch), the correction scales by
  drainage area, ``dQ(s) = dQ(parent) * (A_s / A_parent)^theta_s`` (Eq. 2);
  along a chain this telescopes to ``dQ_o * (A_s / A_o)^theta``.
- At a confluence (two or more upstream branches), the parent correction
  splits between branches in proportion to modeled flow,
  ``dQ_branch = dQ(parent) * Q_branch / Q_parent`` (Edge Case 2).

Walking the tree in BFS order (each node's parent precedes it) accumulates a
per-node, per-timestep multiplier ``M`` with ``dQ(s, t) = dQ_o(t) * M(s, t)``.
The flow-ratio step reads the modeled flow at each node, so the correction at
confluences follows the model's flow partition rather than drainage area.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# t-route is always installed compiled, so a missing extension is a broken
# install and the ImportError should propagate here, not degrade to a slow
# NumPy fallback at runtime.
from troute.routing.fast_reach.scaling_da_kernel import spread_trees

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from troute.scaling_da.gage_tree import GageTree


__all__ = ["apply_scaling_da"]

LOG = logging.getLogger("TROUTE")


def _compute_dq_o(
    obs: NDArray[np.float64],
    q_gage: NDArray[np.float64],
    times_min: NDArray[np.float64],
    decay_tau_min: float | None,
    last_dq: float = 0.0,
    last_t_min: float = -np.inf,
) -> tuple[NDArray[np.float64], float, float]:
    """Per-timestep gage-level ``dQ_o``, with optional exponential decay.

    Parameters
    ----------
    obs : numpy.ndarray
        Observed discharge at the gage, one value per timestep. NaN marks a
        timestep with no observation.
    q_gage : numpy.ndarray
        Modeled (background) discharge at the gage, aligned to ``obs``.
    times_min : numpy.ndarray
        Time of each timestep in minutes from a common reference. Only used when
        ``decay_tau_min`` is set; must share the reference across calls for the
        cross-loop seed to be meaningful.
    decay_tau_min : float or None
        Exponential-decay time constant (minutes) for stale observations, or
        ``None`` to disable decay (gaps then contribute zero).
    last_dq, last_t_min : float, optional
        Seed the persistence from a prior call (the last fresh residual and its
        time), so decay carries continuously across forcing-loop boundaries.
        Default to ``0.0`` / ``-inf`` (no prior).

    Returns
    -------
    dq_o : numpy.ndarray
        Per-timestep gage correction.
    last_dq : float
        The last fresh residual seen, for threading into the next call.
    last_t_min : float
        The time (min) of that last fresh residual.

    Notes
    -----
    With ``decay_tau_min=None``, ``dQ_o(t) = obs(t) - q_gage(t)`` where ``obs`` is
    finite, else 0. With a positive ``decay_tau_min``, the last fresh residual
    persists with ``exp(-(t - t_o) / tau)`` decay until the next valid observation
    -- equivalent to t-route nudging,
    ``Q_nudged(t) = Q_model(t) + (Q_obs(t_o) - Q_model(t_o)) * exp(-(t - t_o)/tau)``.
    """
    valid = np.isfinite(obs)
    resid = np.where(valid, obs - q_gage, 0.0)
    if decay_tau_min is None:
        return resid, last_dq, last_t_min

    # Vectorized forward-fill-with-decay (was a per-timestep Python loop): carry
    # the most recent fresh residual forward, decayed by exp(-gap/tau), until the
    # next valid obs. ``ff`` is the index of the last valid sample at or before t
    # (-1 before any), found by a running max over the valid positions.
    n_t = obs.size
    idx = np.where(valid, np.arange(n_t), -1)
    ff = np.maximum.accumulate(idx)
    have = ff >= 0
    # value/time to decay from: the last fresh residual, or the cross-loop seed.
    val = np.where(have, resid[ff], last_dq)
    t_ref_arr = np.where(have, times_min[ff], last_t_min)
    with np.errstate(over="ignore", invalid="ignore"):
        decayed = val * np.exp(-(times_min - t_ref_arr) / decay_tau_min)
    # No reference yet (before the first valid obs and no finite seed) -> 0.
    usable = have | np.isfinite(last_t_min)
    dq_o = np.where(valid, resid, np.where(usable, decayed, 0.0))
    if have.any():
        li = int(np.flatnonzero(valid)[-1])
        last_dq = float(resid[li])
        last_t_min = float(times_min[li])
    return dq_o, last_dq, last_t_min


def _stack_dq(
    rows: list[NDArray[np.float64]], edge_decay: float = 1.0
) -> NDArray[np.float64]:
    """Stack per-tree innovation rows, padding shorter rows with DECAYED values.

    Rows can differ in length when some sites carry a halo and others do not (or
    a chunked call gives each tree the overlap its own maximum shift reads into).
    The compiled kernel takes ONE innovation length for the whole batch, so a
    padded read lands inside the array and gets no edge decay of its own; padding
    with the raw last value therefore turned a short row's tail into an undecayed
    constant, and the batch composition changed the result. Padding with
    ``last * edge_decay**k`` instead makes every padded read -- and, composed
    with the kernel's decay past the batch end -- every read beyond it, equal
    what decaying from that row's true end gives.
    """
    n = max(r.shape[0] for r in rows)
    if all(r.shape[0] == n for r in rows):
        return np.stack(rows)
    out = []
    for r in rows:
        k = n - r.shape[0]
        if k == 0:
            out.append(r)
        else:
            pad = r[-1] * edge_decay ** np.arange(1, k + 1, dtype=np.float64)
            out.append(np.concatenate([r, pad]))
    return np.stack(out)



def _resolve_site_dq_o(
    site: str,
    gage_pos: int,
    q_model_arr: NDArray[np.float64],
    times_min: NDArray[np.float64],
    obs_decay_tau_min: float | None,
    q_obs: pd.DataFrame | None,
    dq_o_by_site: Mapping[str, NDArray[np.float64]] | None,
    decay_state: dict[str, tuple[float, float]] | None,
) -> NDArray[np.float64] | None:
    """Per-site gage correction ``dq_o(t)`` from either DA path.

    Parameters
    ----------
    site : str
        ``site_no`` of the gage.
    gage_pos : int
        Column position of the gage segment in ``q_model_arr``.
    q_model_arr : numpy.ndarray
        Modeled-flow array, shape ``(n_timesteps, n_segments)``.
    times_min : numpy.ndarray
        Per-timestep time (min) from a common reference, for the decay path.
    obs_decay_tau_min : float or None
        Exponential-decay time constant (minutes), or ``None`` to disable.
    q_obs : pandas.DataFrame or None
        Observation frame (time x ``site_no``), or ``None`` in the in-kernel path.
    dq_o_by_site : Mapping[str, numpy.ndarray] or None
        In-kernel override: ``site_no -> per-timestep applied delta`` (the kernel
        ``nudge``). When given, the obs/decay path is skipped.
    decay_state : dict[str, tuple[float, float]] or None
        Per-site ``(last_dq, last_t_min)`` persistence, threaded across forcing
        loops and **updated in place**; ``None`` disables cross-loop decay.

    Returns
    -------
    numpy.ndarray or None
        Per-timestep ``dq_o``, or ``None`` when the site has no usable observation.

    Notes
    -----
    Two sources. The **in-kernel override** (Stage A): when ``dq_o_by_site`` is
    given, ``dq_o`` is the kernel-recorded applied delta (decay already handled
    in-kernel), and ``q_model`` is expected to carry the reconstructed background at
    the gage so the flow-ratio split reads the modeled flow at the tree root. The
    **output-only** path: ``obs - Q_model(gage)`` with optional exponential decay.
    """
    if dq_o_by_site is not None:
        return np.asarray(dq_o_by_site[site], dtype=np.float64)
    if q_obs is None:  # guarded by the caller; defensive
        return None
    obs = q_obs[site].to_numpy(dtype=np.float64)
    q_gage = q_model_arr[:, gage_pos]
    ld, lt = decay_state.get(site, (0.0, -np.inf)) if decay_state is not None else (0.0, -np.inf)
    dq_o, ld, lt = _compute_dq_o(obs, q_gage, times_min, obs_decay_tau_min, ld, lt)
    if decay_state is not None:
        decay_state[site] = (ld, lt)
    return dq_o


def _prepare_site_tree(
    site: str,
    tree: GageTree,
    gage_to_fp: Mapping[str, int],
    fp_to_pos: Mapping[int, int],
    q_obs: pd.DataFrame | None,
    dq_o_by_site: Mapping[str, NDArray[np.float64]] | None,
) -> GageTree | None:
    """Validate a site and map its tree to ``q_model`` column positions.

    Parameters
    ----------
    site : str
        ``site_no`` of the gage.
    tree : GageTree
        The gage's tree (positions not yet filled).
    gage_to_fp : Mapping[str, int]
        ``site_no -> gage segment id``.
    fp_to_pos : Mapping[int, int]
        ``segment id -> column position`` in the modeled-flow frame.
    q_obs : pandas.DataFrame or None
        Observation frame (time x ``site_no``), or ``None`` in the in-kernel path.
    dq_o_by_site : Mapping[str, numpy.ndarray] or None
        In-kernel override mapping; when given, ``q_obs`` is not required.

    Returns
    -------
    GageTree or None
        The position-filled tree, or ``None`` to skip the site (unknown gage, no
        usable observation/override, a ``gage_fp`` mismatch, or a tree segment
        absent from ``q_model``). The mismatch and missing-segment cases are
        logged as warnings.
    """
    if site not in gage_to_fp:
        return None
    if dq_o_by_site is None and (q_obs is None or site not in q_obs.columns):
        return None
    if dq_o_by_site is not None and site not in dq_o_by_site:
        return None
    if tree.gage_fp != int(gage_to_fp[site]):
        LOG.warning(
            "Site %s: tree gage_fp %s != gage_to_fp %s; skipping",
            site,
            tree.gage_fp,
            gage_to_fp[site],
        )
        return None
    try:
        return tree.with_positions(fp_to_pos)
    except KeyError as e:
        LOG.warning("Site %s: tree segment %s not in q_model columns; skipping", site, e)
        return None


def apply_scaling_da(
    q_model: pd.DataFrame,
    q_obs: pd.DataFrame | None,
    gage_to_fp: Mapping[str, int],
    trees: Mapping[str, GageTree],
    *,
    obs_decay_tau_min: float | None = None,
    min_flow_cms: float = 1e-6,
    decay_state: dict[str, tuple[float, float]] | None = None,
    time_ref: pd.Timestamp | None = None,
    dq_o_by_site: Mapping[str, NDArray[np.float64]] | None = None,
    lag_by_site: Mapping[str, tuple[NDArray[np.float64], NDArray[np.int64]]] | None = None,
    edge_decay: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the simple-scaling DA correction to *q_model* using *q_obs*.

    Implements the white paper's scheme: replace the modeled discharge at
    each USGS gage with its observation, then distribute the gage correction
    ``dQ_o`` upstream. Along a linear step the correction scales by drainage
    area (Eq. 2, ``(A_s/A_parent)^theta``); at a confluence it splits between
    branches in proportion to modeled flow (Edge Case 2,
    ``Q_branch/Q_parent``). The per-node multiplier is accumulated in BFS
    order, so ``dQ(s, t) = dQ_o(t) * M(s, t)``.

    Parameters
    ----------
    q_model : pandas.DataFrame
        Modeled discharge, indexed by time with integer ``fp_id`` columns.
        Must contain every ``fp_id`` that appears in any tree's ``seg_order``;
        the flow-ratio split at confluences reads the modeled flow at each
        node.
    q_obs : pandas.DataFrame or None
        USGS observations, indexed by time (aligned to ``q_model.index``)
        with ``site_no`` columns. NaN at a timestep means "no observation".
        May be ``None`` when ``dq_o_by_site`` supplies the corrections instead.
    gage_to_fp : Mapping[str, int]
        ``site_no -> fp_id`` for the gages being assimilated.
    trees : Mapping[str, GageTree]
        Precomputed per-gage trees. Sites missing from ``q_obs`` or
        ``gage_to_fp`` are skipped with a warning.
    obs_decay_tau_min : float, optional
        Exponential-decay time constant in minutes for stale
        observations. When ``None`` (default), missing observations
        contribute zero correction. Otherwise, the last fresh dQ persists
        with ``exp(-dt/tau)`` decay until the next valid observation.
    min_flow_cms : float, optional
        Floor (m^3/s) on the parent's modeled flow in the confluence
        flow-ratio split. When the parent flow is at or below this floor
        the branch receives no correction at that timestep, guarding the
        ``Q_branch/Q_parent`` division at near-dry junctions. Default
        ``1e-6``.
    dq_o_by_site : Mapping[str, numpy.ndarray], optional
        Stage A (in-kernel) override. When given, the per-gage correction
        ``dQ_o(t)`` is taken directly from this mapping (``site -> length-nt
        array``) instead of being computed as ``obs - Q_model(gage)``. Used after
        the MC kernel has already overridden the gage in-kernel: the recorded
        ``nudge`` (= applied analysis increment, decay included) is the authoritative
        per-timestep delta, and ``q_model`` must carry the reconstructed background
        ``Q_analyzed - nudge`` at each gage segment so the flow-ratio split sees the
        modeled (not overridden) flow at the tree root. ``q_obs``,
        ``obs_decay_tau_min`` and ``decay_state`` are then unused (the gage residual
        and decay are not recomputed). The corrected gage value lands back at
        ``Q_analyzed`` (background + nudge), a no-op against the kernel's override.
    lag_by_site : Mapping[str, tuple[numpy.ndarray, numpy.ndarray]], optional
        Travel-time lag, site -> (in_window, tshift), each aligned to that
        tree's seg_order. in_window is 1.0 inside the smoother window and
        0.0 beyond it; tshift is the advective displacement in whole
        timesteps. Omit a site (or pass None) to leave it un-lagged.
    decay_state : dict, optional
        Per-site ``site_no -> (last_dq, last_t_min)`` decay persistence, threaded
        across forcing loops and **updated in place**. ``None`` (default) disables
        cross-loop decay. Unused in the ``dq_o_by_site`` path.
    time_ref : pandas.Timestamp, optional
        Common reference time for the decay clock; required for the cross-loop
        ``decay_state`` seed to be meaningful. Defaults to ``q_model.index[0]``.

    Returns
    -------
    q_corrected : pandas.DataFrame
        Same shape as ``q_model``. Negative values clamped at 0.
    dq_diagnostic : pandas.DataFrame
        Same shape as ``q_model``, recording the delta-Q applied at each
        segment per timestep. Zero where no correction was applied.

    Raises
    ------
    ValueError
        If ``obs_decay_tau_min`` is non-positive, if any tree is malformed, or
        if ``lag_by_site`` carries a shift shape the compiled kernel's contract
        cannot express (2-D or negative).

    Notes
    -----
    Per-tree corrections are summed into the output; the trees are mutually
    disjoint by construction, and an overlap (two trees sharing a segment) is
    logged as a warning. The confluence split divides each branch flow by
    ``max(Q_parent, sum of branch flows)``, so it is non-expansive and a
    correction never amplifies across a junction.
    """
    if obs_decay_tau_min is not None and not obs_decay_tau_min > 0:
        msg = f"obs_decay_tau_min must be a positive number of minutes, got {obs_decay_tau_min!r}"
        raise ValueError(msg)

    if q_obs is not None and not q_model.index.equals(q_obs.index):
        q_obs = q_obs.reindex(q_model.index)

    fp_to_pos: dict[int, int] = {int(fp): i for i, fp in enumerate(q_model.columns)}
    # to_numpy() on a homogeneous-float frame returns the F-ordered internal block
    # (C-contiguous only when there is a single timestep); the kernel binds q_model
    # as double[:, ::1], so force C-contiguity here. copy() below is already C-order.
    q_model_arr = np.ascontiguousarray(q_model.to_numpy(dtype=np.float64))
    q_corrected = q_model_arr.copy()
    dq_diagnostic = np.zeros_like(q_corrected)
    # Track which columns have already received a correction, to enforce the
    # tree-disjoint invariant (no segment corrected by two DA trees).
    corrected_mask = np.zeros(q_model_arr.shape[1], dtype=bool)

    ref = q_model.index[0] if time_ref is None else pd.Timestamp(time_ref)
    times_min = (pd.DatetimeIndex(q_model.index) - ref).total_seconds().to_numpy() / 60.0

    skipped: list[str] = []

    def _accept_site(site: str, tree: GageTree) -> tuple[GageTree, NDArray[np.float64]] | None:
        """Validate a site and resolve its ``dq_o``; warn on tree overlap.

        Returns ``(tree_p, dq_o)`` for a site that contributes a correction,
        else ``None`` (skipped or all-zero ``dq_o``). Marks ``corrected_mask``
        and emits the tree-disjoint warning, order-preserving over ``trees``.
        """
        tree_p = _prepare_site_tree(site, tree, gage_to_fp, fp_to_pos, q_obs, dq_o_by_site)
        if tree_p is None:
            skipped.append(site)
            return None
        seg_pos = tree_p.seg_positions
        dq_o = _resolve_site_dq_o(
            site,
            int(seg_pos[0]),
            q_model_arr,
            times_min,
            obs_decay_tau_min,
            q_obs,
            dq_o_by_site,
            decay_state,
        )
        if dq_o is None or not np.any(dq_o):
            return None
        overlap = corrected_mask[seg_pos]
        if overlap.any():
            LOG.warning(
                "Site %s: %d segment(s) already corrected by another DA tree; "
                "trees are not disjoint and corrections will double-count",
                site,
                int(overlap.sum()),
            )
        corrected_mask[seg_pos] = True
        return tree_p, dq_o

    # The compiled kernel takes ONE non-negative shift per segment and indexes
    # dQ_o forward only; a 2-D (per-timestep) or negative (backward) shift is
    # not expressible in its contract.
    _numpy_only_lag = bool(lag_by_site) and any(
        np.asarray(v[1]).ndim == 2 or np.asarray(v[1]).min(initial=0) < 0
        for v in lag_by_site.values()
    )
    if _numpy_only_lag:
        # _build_lag never produces these shapes; a caller contract violation,
        # not a mode -- there is no fallback path.
        msg = (
            "apply_scaling_da: lag_by_site must carry one non-negative shift "
            "per segment (1-D); a per-timestep or backward shift is not "
            "expressible by the compiled spread kernel."
        )
        raise ValueError(msg)

    # Batched compiled path: flatten every surviving tree into CSR-style
    # buffers and spread them in one nogil pass (replaces ~22k per-tree calls).
    parents: list[NDArray[np.int64]] = []
    is_juncs: list[NDArray[np.uint8]] = []
    areas_f: list[NDArray[np.float64]] = []
    thetas_f: list[NDArray[np.float64]] = []
    qcols: list[NDArray[np.int64]] = []
    dq_o_rows: list[NDArray[np.float64]] = []
    windows: list[NDArray[np.float64]] = []
    tshifts: list[NDArray[np.int64]] = []
    pruned_counts: list[NDArray[np.int64]] = []
    pruned_cols: list[NDArray[np.int64]] = []
    offsets: list[int] = [0]
    nt = q_model_arr.shape[0]
    for site, tree in trees.items():
        accepted = _accept_site(site, tree)
        if accepted is None:
            continue
        tree_p, dq_o = accepted
        # The compiled kernel takes one theta per SEGMENT; the tree carries one
        # scalar. Broadcast here rather than changing the kernel signature: the
        # uniformity the kernel assumes is now guaranteed by construction, so the
        # buffer is a repeat of a single value by definition.
        thetas = np.full(tree_p.n_segments, tree_p.theta, dtype=np.float64)
        parents.append(np.ascontiguousarray(tree_p.seg_parent_idx, dtype=np.int64))
        is_juncs.append(np.ascontiguousarray(tree_p.step_is_junction, dtype=np.uint8))
        areas_f.append(np.ascontiguousarray(tree_p.seg_areas_sqkm, dtype=np.float64))
        thetas_f.append(np.ascontiguousarray(thetas, dtype=np.float64))
        qcols.append(np.ascontiguousarray(tree_p.seg_positions, dtype=np.int64))
        # Per-node count of Edge-Case-1 branches; the flat CSR offsets are built once
        # below by cumulative sum, so a tree with none contributes zeros and no columns.
        pp = getattr(tree_p, "pruned_pos_ptr", None)
        pc = getattr(tree_p, "pruned_positions", None)
        if pp is not None and pp.size == tree_p.n_segments + 1 and pc is not None:
            pruned_counts.append(np.diff(np.asarray(pp, dtype=np.int64)))
            pruned_cols.append(np.asarray(pc, dtype=np.int64))
        else:
            pruned_counts.append(np.zeros(tree_p.n_segments, dtype=np.int64))
        # Travel-time lag, per flat segment. Same length contract as dq_o: the
        # kernel indexes these with bounds checking off, so a short array is a
        # silent out-of-bounds read. Absent site -> inert (w=1, shift=0).
        n_s = tree_p.n_segments
        lag = lag_by_site.get(site) if lag_by_site else None
        if lag is None:
            windows.append(np.ones(n_s, dtype=np.float64))
            tshifts.append(np.zeros(n_s, dtype=np.int64))
        else:
            tp, ts_ = lag
            tp = np.ascontiguousarray(tp, dtype=np.float64).ravel()
            ts_ = np.ascontiguousarray(ts_, dtype=np.int64).ravel()
            if tp.shape[0] != n_s or ts_.shape[0] != n_s:
                msg = (
                    f"site {site}: lag_by_site arrays have {tp.shape[0]}/"
                    f"{ts_.shape[0]} entries but the tree has {n_s} segments; "
                    "the compiled spread kernel requires one in_window and one "
                    "shift per segment."
                )
                raise ValueError(msg)
            # The kernel runs with boundscheck=False AND wraparound=False, and it
            # only guards the upper end (`ts >= nt`). A negative shift would index
            # dq_o backwards past the start of the buffer -- an out-of-bounds READ,
            # not an IndexError. A non-finite in_window would write NaN into discharge.
            # Neither is reachable from _build_lag, but this is the trust boundary
            # for any caller supplying lag_by_site, so enforce it here.
            if (ts_ < 0).any():
                msg = (
                    f"site {site}: lag_by_site shift must be non-negative (got min "
                    f"{int(ts_.min())}); the compiled kernel indexes dQ_o forward "
                    "only and does not bounds-check."
                )
                raise ValueError(msg)
            if not np.isfinite(tp).all():
                msg = (
                    f"site {site}: lag_by_site in_window must be finite; a NaN or inf "
                    "weight would be written straight into the corrected discharge."
                )
                raise ValueError(msg)
            windows.append(tp)
            tshifts.append(ts_)
        # Shape contract for the bounds-check-free kernel: spread_trees reads
        # dq_o[it, t] for t in range(nt) with boundscheck off, so a row shorter
        # than nt would read out of bounds and silently corrupt the correction
        # (the retired NumPy reference broadcast a length-1 dq_o over all
        # timesteps; the compiled kernel cannot, so enforce the length here).
        # Broadcast a scalar dq_o to preserve that behavior; reject any other
        # short length.
        dq_o = np.ascontiguousarray(dq_o, dtype=np.float64).ravel()
        if dq_o.shape[0] == 1 and nt > 1:
            dq_o = np.broadcast_to(dq_o, (nt,))
        # LONGER than nt is legitimate: a chunked call passes the overlap its
        # forward shifts read into, and the kernel takes the innovation length
        # from the array. SHORTER is still an out-of-bounds read.
        elif dq_o.shape[0] < nt:
            msg = (
                f"site {site}: dq_o has {dq_o.shape[0]} timesteps but q_model has "
                f"{nt}; the compiled spread kernel needs at least nts values per "
                "tree (or a broadcastable scalar)."
            )
            raise ValueError(msg)
        dq_o_rows.append(dq_o)
        offsets.append(offsets[-1] + tree_p.n_segments)

    if dq_o_rows:
        tree_off = np.asarray(offsets, dtype=np.int64)
        max_seg = int(np.max(np.diff(tree_off)))
        qbuf = np.empty((max_seg, nt), dtype=np.float64)
        mult = np.empty((max_seg, nt), dtype=np.float64)
        bsum = np.empty((max_seg, nt), dtype=np.float64)
        pruned_off = np.zeros(int(tree_off[-1]) + 1, dtype=np.int64)
        np.cumsum(np.concatenate(pruned_counts), out=pruned_off[1:])
        pruned_col = (np.concatenate(pruned_cols) if pruned_cols
                      else np.empty(0, dtype=np.int64))
        spread_trees(
            q_model_arr,
            tree_off,
            np.concatenate(parents),
            np.concatenate(is_juncs),
            np.concatenate(areas_f),
            np.concatenate(thetas_f),
            np.concatenate(qcols),
            np.ascontiguousarray(_stack_dq(dq_o_rows, edge_decay), dtype=np.float64),
            float(min_flow_cms),
            float(edge_decay),
            np.ascontiguousarray(np.concatenate(windows, axis=0)),
            np.ascontiguousarray(np.concatenate(tshifts, axis=0)),
            np.ascontiguousarray(pruned_off),
            np.ascontiguousarray(pruned_col),
            q_corrected,
            dq_diagnostic,
            qbuf,
            mult,
            bsum,
        )

    if skipped:
        LOG.warning("apply_scaling_da: skipped %d sites: %s", len(skipped), skipped[:5])

    np.clip(q_corrected, 0.0, None, out=q_corrected)

    q_corrected_df = pd.DataFrame(q_corrected, index=q_model.index, columns=q_model.columns)
    dq_df = pd.DataFrame(dq_diagnostic, index=q_model.index, columns=q_model.columns)
    return q_corrected_df, dq_df
