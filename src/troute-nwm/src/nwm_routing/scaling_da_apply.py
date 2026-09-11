"""In-run application of the simple-scaling streamflow DA.

Wired into the NHF (-V5) run loop and the BMI driver. Before each ``nwm_route``
call, :meth:`ScalingDA.build_usgs_df` injects the accepted gage observations into
the Muskingum-Cunge nudging override, carrying the correction DOWNSTREAM through
routing. After routing, :meth:`ScalingDA.apply_in_kernel` reads the kernel-recorded
innovation (``nudge``), reconstructs the gage background, and spreads the
correction UPSTREAM (area-scaling along reaches, flow-ratio split at confluences).
Stale-obs decay within a window follows ``da_decay_coefficient``; decay state is
NOT carried across forcing windows.

Everything here is in ``up_node_id`` space, and the gage crosswalk comes from
``network.gages`` -- the same set the execution plan splits reaches at, so the
injection always lands on a reach boundary.

Observations come from the shared ``usgs_timeslices_folder`` through
``nhd_io.get_obs_from_timeslices`` (same files, QC gate, and interpolation as
nudging and reservoir persistence), or synthetically as ``synthetic_obs_factor``
times a frozen no-DA baseline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da import build_scaling_da_setup
from troute.scaling_da import (
    # Deliberate re-export; tests and harnesses import the roster from here.
    timeslice_station_roster as timeslice_station_roster,  # noqa: PLC0414
)
from troute.scaling_da.preprocess import TIMESLICE_GLOB_SUFFIX as _TIMESLICE_GLOB_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from troute.AbstractNetwork import AbstractNetwork
    from troute.scaling_da import GageTree

    # One kernel result block, read POSITIONALLY: r[0] segment ids, r[1] the
    # (n_seg, nts*4) q/v/d/ql array, r[3][0] gage segs, r[9] the nudge array.
    RunResult = Sequence[Any]
    RunResults = Sequence[RunResult]

LOG = logging.getLogger("TROUTE")


def merge_injected_obs(
    injected: pd.DataFrame, existing: pd.DataFrame | None
) -> pd.DataFrame:
    """Overlay the scaling DA's rows on the existing observation frame.

    The frame is shared: the diversion DA reads its gage's row from it, so the
    injection must overlay its own rows and leave the rest intact. Surviving rows
    are reindexed onto the injected frame's columns because the kernel reads the
    frame positionally -- a column union of two grids would shift observations.
    """
    if injected.empty:
        # Nothing injected this call; do not wipe the diversion/nudging rows.
        return existing if existing is not None else injected
    if existing is None or existing.empty:
        return injected
    kept = existing.drop(index=injected.index, errors="ignore")
    if kept.empty:
        return injected
    aligned = kept.reindex(columns=injected.columns)
    # A producer on a different time grid would lose every value in this reindex
    # and silently assimilate nothing; say so.
    emptied = kept.notna().any(axis=1) & ~aligned.notna().any(axis=1)
    if emptied.any():
        LOG.warning(
            "merge_injected_obs: %d observation row(s) lost every value when "
            "reindexed onto the injected frame's time grid (e.g. %s); those rows "
            "will assimilate nothing this window.",
            int(emptied.sum()),
            list(kept.index[emptied][:5]),
        )
    return pd.concat([injected, aligned]).sort_index()


def should_seed_state(
    scaling_da: ScalingDA | None, window_index: int, n_windows: int
) -> bool:
    """Should this window's upstream spread run BEFORE the warmstate snapshot?

    True only on the FINAL window. Muskingum-Cunge's C3*qdp recurrence carries only short previous-discharge
    memory, so a
    dQ written into q0 is a one-shot pulse that inflates the source gage's
    background and debits the next window's innovation (measured: held-out bias
    -23.8% with one seeding boundary vs -29.1% with eleven). Seeding only at the
    hand-off keeps the cycling analysis bit-identical to the diagnostic arm while
    a forecast still inherits the correction.
    """
    return scaling_da is not None and window_index == n_windows - 1


class ScalingDA:
    """Consume the network's prebuilt static setup; apply the correction per loop."""

    # Class defaults so a partially built instance (tests construct via __new__)
    # behaves: with no dx nothing is beyond the propagation limit, and with no
    # traced travel time every segment sits at shift 0, which is numerically
    # identical to an un-shifted spread.
    max_reach_km: float = 200.0
    innovation_spread_h: float = 0.0
    travel_time_lag: bool = False
    lag_window_h: float = 48.0
    min_flow: float = 1e-6
    da_decay_min: float = 120.0
    _dx = None

    def __init__(
        self,
        network: AbstractNetwork,
        params: Mapping[str, Any],
        da_params: Mapping[str, Any] | None = None,
        cpu_pool: int | None = None,
    ) -> None:
        # Propagation limit as network DISTANCE, not travel time. Every route to
        # a defensible celerity was measured and rejected: a constant cannot
        # describe a continental network, and the router's own Courant number
        # reports the kinematic celerity, which runs fast where an event is
        # passing and slower than any usable constant at ambient flow on the same
        # reaches. Reach length is known exactly, so the limit uses that.
        self.max_reach_km = float(params.get("max_reach_km", 200.0))
        if not (math.isfinite(self.max_reach_km) and self.max_reach_km > 0):
            raise ValueError(
                "streamflow_scaling_parameters.max_reach_km must be a finite "
                f"positive number, got {self.max_reach_km!r}"
            )
        # Width of the temporal spread applied to the innovation. Muskingum-Cunge
        # routes a diffusion wave, so an error seen at the gage at one instant was
        # carried by water that passed an upstream segment over a RANGE of
        # instants; the correction is spread over that range rather than moved to
        # one estimated arrival.
        self.innovation_spread_h = float(params.get("innovation_spread_h", 0.0))
        if not (math.isfinite(self.innovation_spread_h)
                and self.innovation_spread_h >= 0):
            raise ValueError(
                "streamflow_scaling_parameters.innovation_spread_h must be a "
                f"finite non-negative number, got {self.innovation_spread_h!r}"
            )
        # Travel-time shift: apply a gage innovation at an upstream segment as
        # dQ_o(t + tau), where tau is traced BACKWARD along the characteristic
        # from the router's own Courant number. One switch, because there is one
        # defensible answer: every celerity that could be PICKED was measured and
        # rejected, cross-correlating the routed hydrographs resolved a tenth of
        # the segments and scored the same as no timing at all, and applying the
        # shift backward instead is 2*tau late by construction.
        self.travel_time_lag = bool(params.get("travel_time_lag", False))
        self.lag_window_h = float(params.get("lag_window_h", 48.0))
        if not (math.isfinite(self.lag_window_h) and self.lag_window_h > 0):
            raise ValueError(
                "streamflow_scaling_parameters.lag_window_h must be a finite "
                f"positive number, got {self.lag_window_h!r}"
            )
        try:
            self._dx = network.dataframe["dx"].astype(float)
        except (AttributeError, KeyError) as e:
            raise ValueError(
                "the scaling DA needs reach lengths for the travel-time lag, and the "
                f"network dataframe has no usable 'dx' column ({e})."
            ) from e
        # The edge closure (past the window end) is innovation PERSISTENCE, so it
        # decays on the same clock as every other stale observation here rather than
        # holding a constant increment forward indefinitely.
        self.da_decay_min = float((da_params or {}).get("da_decay_coefficient", 120.0))
        if not self.da_decay_min > 0:
            raise ValueError(
                f"data_assimilation_parameters.da_decay_coefficient must be positive, "
                f"got {self.da_decay_min!r}"
            )
        self.min_flow = float(params.get("min_flow_cms", 1e-6))

        # Spread time-chunking: None = auto (only above the memory budget),
        # 0 = off, N = fixed. Bit-identical either way; see _resolve_spread_chunk.
        chunk = params.get("spread_chunk_timesteps")
        self.spread_chunk_timesteps = None if chunk is None else int(chunk)
        if self.spread_chunk_timesteps is not None and self.spread_chunk_timesteps < 0:
            raise ValueError(
                f"streamflow_scaling_parameters.spread_chunk_timesteps must be >= 0 (0 disables "
                f"chunking) or unset for auto; got {chunk!r}"
            )
        self.synthetic_factor = params.get("synthetic_obs_factor", None)
        # Must be a FROZEN no-DA output dir; factor*live_model compounds under feedback.
        self.synthetic_obs_baseline = params.get("synthetic_obs_baseline", None)
        da_params = da_params or {}
        self._qc_threshold = da_params.get("qc_threshold", 1)
        self._interpolation_limit = da_params.get("interpolation_limit_min", 59)
        self._cpu_pool = cpu_pool or 1
        self._obs_cache: tuple[tuple[Any, ...], pd.DataFrame] | None = None
        self._synth_base = None  # cached frozen synthetic baseline (site -> Series)

        # Static setup (crosswalk, source set, trees): built by NHF during network
        # construction when the DA is enabled; the fallback is the SAME function,
        # for callers constructing ScalingDA against a network that did not
        # pre-build (tests, harnesses, stubs). One implementation either way.
        setup = getattr(network, "scaling_da_setup", None)
        if setup is None:
            setup = build_scaling_da_setup(network, params, da_params, cpu_pool)
        self.trees = dict(setup.trees)
        self.gage_seg = dict(setup.gage_seg)
        self.gage_fp = dict(setup.gage_fp)
        # Full crosswalk (incl. held-out/co-located sites) for evaluation harnesses.
        self._all_gage_seg = dict(setup.all_gage_seg)
        self._da_sites = list(setup.da_sites)  # fixed injected set (plan determinism)
        self._obs_sites = set(setup.obs_sites)
        self._waterbody = setup.waterbody
        self._ts_folder: Path | None = setup.ts_folder
        self.theta_default = setup.theta_default
        self.theta_by_vpu = dict(setup.theta_by_vpu)
        LOG.info(
            "scaling DA: in-kernel downstream mode -- %d gages injected",
            len(self._da_sites),
        )

    # -- per-loop helpers --------------------------------------------------
    def _assemble_q_model(
        self, run_results: RunResults, nts: int, dt: float, t0: Any
    ) -> pd.DataFrame:
        # q lives at flat columns 0::4 (q,v,d,ql interleaved per timestep)
        ids = np.concatenate([np.asarray(r[0]) for r in run_results])
        q = np.concatenate([np.asarray(r[1])[:, 0::4] for r in run_results], axis=0)
        index = pd.date_range(
            pd.Timestamp(t0) + pd.Timedelta(seconds=dt),
            periods=nts,
            freq=pd.Timedelta(seconds=dt),
            name="time",
        )
        return pd.DataFrame(q.T.astype(np.float64), index=index, columns=[int(i) for i in ids])

    def _reach_dx(self, seg: int) -> float | None:
        """One reach's length in meters, or None when it is unusable.

        A length of 0 or less is as unusable as NaN: it would yield a NEGATIVE
        shift, which the NumPy spread reads as a wrap-around index rather than
        an error.
        """
        if self._dx is None:
            return None
        dx = self._dx.get(seg)
        if dx is None or not np.isfinite(dx) or dx <= 0:
            return None
        return float(dx)

    def _tree_distance_km(self, tree: GageTree) -> tuple[NDArray[np.float64], int]:
        """Distance to the gage along the network, in km, per segment in seg_order.

        ``d(s) = d(parent) + dx(parent)``, one forward pass along the BFS parent
        chain since parents precede children. Returns ``(distance, n_unusable)``.

        Distance replaces travel time here because ``dx`` is known exactly while
        travel time is not: every route to a defensible celerity was measured and
        rejected (a constant cannot describe a continental network; the router's
        own Courant number reports the KINEMATIC celerity, which runs fast during
        an event and slower than any usable constant at ambient flow on the same
        reaches; channel geometry needs a fitted reference depth). Localizing on
        network distance is the standard alternative and needs nothing estimated.

        The PARENT's reach is charged, not the segment's own: the correction sits
        on segment j's output, so the water it describes still has to traverse
        the parent, then the parent's parent, down to the gage.
        """
        segs = np.asarray(tree.seg_order).astype("int64").ravel()
        parent = np.asarray(tree.seg_parent_idx).astype("int64").ravel()
        dist = np.zeros(segs.shape[0], dtype=np.float64)
        unusable = 0
        if self._dx is None:
            # No reach lengths at all: every segment sits at distance 0, so the
            # limit never bites. A partially built instance, not a data defect.
            return dist, 0
        for j in range(1, segs.shape[0]):
            p = int(parent[j])
            dx = self._reach_dx(int(segs[p]))
            if dx is None:
                # A reach with no usable length in a network that HAS lengths is
                # a data defect, and fails CLOSED: distance 0 would hold its whole
                # subtree inside the limit and correct segments arbitrarily far
                # from the gage.
                unusable += 1
                dist[j] = self.max_reach_km + 1.0
                continue
            dist[j] = dist[p] + dx / 1000.0
        return dist, unusable

    def _assemble_cn(
        self, run_results: RunResults, nts: int
    ) -> tuple[NDArray[np.float64], dict[int, int]] | None:
        """Per-segment, per-timestep Courant number, ``([nts, N_seg], colpos)``.

        ``r[2]`` is the kernel's Courant block, ``[n_seg, nts*3]`` with cn/ck/X
        interleaved (cn at ``0::3``), initial-condition column already dropped,
        so the same grid and mask as ``r[1]``. It is the integer ``0``
        placeholder unless the driver asked for it, which is what ``None`` here
        reports.
        """
        ids, cols, cks = [], [], []
        for r in run_results:
            cur = r[2]
            # PARTIAL by design: diffusive results carry the scalar 0
            # placeholder, so a hybrid run has cn for its MC reaches and none for
            # the rest. Skipping those results rather than abandoning the whole
            # field leaves their segments out of colpos, and each falls back per
            # reach.
            if not isinstance(cur, np.ndarray) or cur.ndim != 2:
                continue
            block = np.asarray(cur)[:, 0::3].astype(np.float64)
            if block.shape[1] != nts:
                continue
            ids.append(np.asarray(r[0]).astype("int64").ravel())
            cols.append(block)
            # ck (column 1) is the wave celerity in m/s directly. Not used for
            # the trace, which needs no dx because 1/cn is already the transit in
            # timesteps, but logged: it is the one number that says whether a
            # traced travel time is physically plausible.
            cks.append(np.asarray(cur)[:, 1::3].astype(np.float64))
        if not ids:
            return None
        seg_ids = np.concatenate(ids)
        cn = np.concatenate(cols, axis=0).T  # [nts, N_seg]
        ck = np.concatenate(cks, axis=0)
        alive = np.isfinite(ck) & (ck > 0)
        live = ck[alive]
        if live.size:
            # The trace moves at min(ck, dx/dt) = ck/max(cn, 1); log both so a
            # traced tau is sanity-checked against the EFFECTIVE speed.
            eff = live / np.maximum(cn.T[alive], 1.0)
            LOG.info(
                "scaling DA: router wave celerity ck over %d live segment-steps -- "
                "median %.3f m/s, p10 %.3f, p90 %.3f, max %.2f; effective "
                "(K clamped at one timestep, min(ck, dx/dt)): median %.3f m/s, "
                "p90 %.3f",
                live.size, float(np.median(live)), float(np.percentile(live, 10)),
                float(np.percentile(live, 90)), float(live.max()),
                float(np.median(eff)), float(np.percentile(eff, 90)),
            )
        return cn, {int(s): i for i, s in enumerate(seg_ids)}

    def _ensure_trace(self, run_results: RunResults, nts: int, dt: float) -> None:
        """Trace every tree's travel time, once, from the earliest routed window.

        Called before the innovation-dependent early returns in
        :meth:`apply_in_kernel` so the traced span is the run's opening rather
        than "the first window that happened to carry an observation", which
        moves with the partition.
        """
        if not self.travel_time_lag or self._trace_cached(self.trees):
            return
        cn = self._assemble_cn(run_results, nts)
        if cn is None:
            return
        self._ensure_trace_from_cn(cn[0], cn[1], dt)

    def _ensure_trace_from_cn(
        self, cn: NDArray[np.float64], colpos: Mapping[int, int], dt: float
    ) -> None:
        """Fill the tau cache for every tree from an assembled Courant field."""
        if self._trace_cached(self.trees):
            return
        n_lag = min(cn.shape[0], self._lag_window_steps(dt))
        if n_lag < self._lag_window_steps(dt):
            LOG.warning(
                "scaling DA: the first window supplies %d timestep(s), less than "
                "the %d of lag_window_h, so the trace is capped at the SHORTER "
                "span. That span is part of the result, so this run is not "
                "comparable with one whose first window covers lag_window_h. It "
                "happens when the windows are given explicitly "
                "(qlat_forcing_sets), which bypasses the enlargement, or when the "
                "run is shorter than lag_window_h.",
                n_lag, self._lag_window_steps(dt),
            )
        tau_cache = getattr(self, "_tau_cache", None)
        if tau_cache is None:
            tau_cache = self._tau_cache = {}
        for site, tree in self.trees.items():
            if (site, n_lag) not in tau_cache:
                tau_cache[(site, n_lag)] = self._tree_tau_trace(
                    tree, n_lag, cn[:n_lag], colpos
                )
        self._tau_span = n_lag
        # tau is in timesteps OF THIS dt; checkpoint restore checks it.
        self._tau_dt = float(dt)

    def _tree_fingerprint(self) -> tuple[tuple[str, str], ...]:
        """Per-site digest of the full seg_order + parents. tau is positional,
        so a same-root/same-size tree with a different interior (hydrofabric
        revision, changed stop set) must not accept a restored tau array."""
        out = []
        for site, t in self.trees.items():
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(t.seg_order, dtype=np.int64).tobytes())
            h.update(np.ascontiguousarray(t.seg_parent_idx, dtype=np.int64).tobytes())
            out.append((site, h.hexdigest()))
        return tuple(sorted(out))

    def trace_checkpoint(self) -> dict[str, Any] | None:
        """Traced tau plus its identity (dt, lag_window_h, tree fingerprint).

        The trace is result-determining state: a resumed run that retraces
        from its own first window shifts corrections. None if nothing traced.
        """
        span = getattr(self, "_tau_span", None)
        if span is None:
            return None
        return {
            "cache": dict(getattr(self, "_tau_cache", {}) or {}),
            "span": span,
            "dt": getattr(self, "_tau_dt", None),
            "lag_window_h": float(self.lag_window_h),
            "trees": self._tree_fingerprint(),
        }

    def restore_trace_checkpoint(
        self, ckpt: Mapping[str, Any] | None, dt: float
    ) -> None:
        """Install a serialized trace, or explicitly invalidate the cache.

        Every path clears first (a stale own-trace must not survive a load);
        a missing entry or a mismatched identity warns and retraces rather
        than reinterpreting tau measured under another dt/span/tree set.
        """
        for attr in ("_tau_cache", "_tau_span", "_tau_dt"):
            if hasattr(self, attr):
                delattr(self, attr)
        if not self.travel_time_lag:
            return
        if not ckpt:
            LOG.warning(
                "scaling DA: the loaded state carries no traced travel time, so "
                "this run traces its own from its first routed window. That span "
                "differs from the original run's opening, so a resumed run is "
                "not comparable with an uninterrupted one."
            )
            return
        same = (
            ckpt.get("dt") == float(dt)
            and ckpt.get("lag_window_h") == float(self.lag_window_h)
            and ckpt.get("trees") == self._tree_fingerprint()
        )
        if not same:
            LOG.warning(
                "scaling DA: the loaded travel-time trace was measured under a "
                "different identity (dt %s vs %s, lag_window_h %s vs %s, or "
                "another tree set) and is DISCARDED; this run retraces from its "
                "own first window.",
                ckpt.get("dt"), float(dt),
                ckpt.get("lag_window_h"), float(self.lag_window_h),
            )
            return
        self._tau_cache = dict(ckpt["cache"])
        self._tau_span = ckpt["span"]
        self._tau_dt = ckpt["dt"]

    def _trace_cached(self, trees: Mapping[str, GageTree]) -> bool:
        """True when every tree's traced tau is already cached for this run.

        The trace is measured once over a fixed span, so after the window that
        fills the cache the Courant field is dead weight: see the call site.
        """
        span = getattr(self, "_tau_span", None)
        if span is None:
            return False
        cache = getattr(self, "_tau_cache", None) or {}
        return all((site, span) in cache for site in trees)

    def _lag_window_steps(self, dt: float) -> int:
        """Timesteps of routed flow the lag is measured over, run-wide.

        Fixed, and deliberately NOT the forcing window: the correlation span
        decides the answer, so letting it follow max_loop_size would let a
        memory knob change discharge. Both drivers enlarge a short window to
        cover this, so the measured span is the same data under every partition.
        """
        return max(4, round(self.lag_window_h * 3600.0 / float(dt)))

    def _tree_tau_trace(
        self,
        tree: GageTree,
        nt: int,
        cn: NDArray[np.float64],
        colpos: Mapping[int, int],
    ) -> tuple[NDArray[np.float64], dict[str, int]]:
        """Travel time to the gage in TIMESTEPS, by a BACKWARD particle trace.

        Walk back in time from the gage, accumulating the distance the wave
        covered in each past timestep until it exceeds the reach's length, then
        carry on up the parent chain. What is being traced is the KINEMATIC WAVE,
        not the water: an innovation is a discharge perturbation, and by the
        Kleitz-Seddon law a perturbation moves at ``c = dQ/dA = beta * V``, which
        for a hydraulically wide Manning channel is ``(5/3) V``. Tracing with the
        water velocity would over-estimate travel time by that factor.

        ``ck`` is not assumed: the kernel computes it for the compound
        trapezoidal section, including the wetted-perimeter correction and the
        overbank blend, and exports it alongside the Courant number. Since
        ``cn = ck*dt/dx``, the distance covered in one timestep is exactly
        ``cn*dx``, so the trace works in units of reach length and never needs
        ``dx`` itself:

            steps back until   sum_k min(cn(reach, t-k), 1)  >=  1.0

        The ``min`` follows the MODEL, not the physics: the MC solver clamps
        K at ``max(dt, dx/Ck)`` and the discrete reach lag IS K, so the routed
        perturbation crosses at most one segment per timestep while the
        exported cn stays unclamped; tracing raw cn under-estimated tau by
        exactly the factor cn wherever cn > 1 (most event flow at NHF reach
        lengths).

        Every read is at an already-routed time (backward is well posed) and
        the span ``nt`` is FIXED (``lag_window_h``), so the resolved set cannot
        depend on the window partition. Unresolvable reaches return ``inf``.

        Returns ``(tau, counts)``; ``counts`` splits unresolved by cause,
        since only one of them is fixable by configuration:
        ``inherited`` (parent unresolved), ``no_cn`` (no Courant export, e.g.
        the diffusive domain), ``dry`` (no live sample ever), ``short``
        (record ended; a longer ``lag_window_h`` would resolve it),
        ``lower_bound`` (crossed only on the oldest sample).
        """
        segs = np.asarray(tree.seg_order).astype("int64").ravel()
        parent = np.asarray(tree.seg_parent_idx).astype("int64").ravel()
        tau = np.zeros(segs.shape[0], dtype=np.float64)
        counts = {"inherited": 0, "no_cn": 0, "dry": 0, "short": 0, "lower_bound": 0}
        for j in range(1, segs.shape[0]):
            p = int(parent[j])
            if not np.isfinite(tau[p]):
                tau[j] = np.inf
                counts["inherited"] += 1
                continue
            # The PARENT's reach is what this segment's water must cross.
            col = colpos.get(int(segs[p]))
            if col is None:
                tau[j] = np.inf
                counts["no_cn"] += 1
                continue
            # Start where the parent's own trace ended, and keep walking back.
            t0 = nt - 1 - tau[p]
            covered = 0.0
            steps = 0.0
            k = int(np.floor(t0))
            while k >= 0 and covered < 1.0:
                # min(cn, 1): the solver clamps K at one timestep, so the
                # routed wave never crosses more than one segment per step;
                # the exported cn is unclamped physics (see docstring).
                c = cn[k, col]
                if np.isfinite(c) and c > 0.0:
                    covered += min(c, 1.0)
                steps += 1.0
                k -= 1
            if covered < 1.0:
                # Record ended first: zero coverage = dry/reservoir reach,
                # partial = record too short for this travel time.
                tau[j] = np.inf
                counts["dry" if covered == 0.0 else "short"] += 1
                continue
            # Linear interpolation within the final step, so the trace resolves
            # a partial traverse rather than rounding up a whole timestep.
            over = covered - 1.0
            c_last = cn[max(k + 1, 0), col]
            frac = (
                (over / min(c_last, 1.0))
                if (np.isfinite(c_last) and c_last > 0)
                else 0.0
            )
            if k < 0:
                # The walk ran to the START of the record and crossed the reach
                # only on the oldest sample it had. That tau is a lower bound,
                # not a measurement -- the record ran out at the same moment the
                # trace finished -- and applying it as an exact shift is the
                # silent failure this estimator exists to avoid: one such
                # segment carried a 53 h timing error against a 14 h truth in
                # the OSSE. Unresolved, like any other segment the record cannot
                # speak for. Note this is a property of the walk, not a
                # threshold on tau: nothing here is tuned.
                tau[j] = np.inf
                counts["lower_bound"] += 1
                continue
            tau[j] = tau[p] + max(steps - frac, 0.0)
        return tau, counts

    def _build_lag(
        self,
        trees: Mapping[str, GageTree],
        dt: float,
        nt: int = 0,
        cn: NDArray[np.float64] | None = None,
        cn_colpos: Mapping[int, int] | None = None,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.int64]]]:
        """site -> (in_reach, tshift) per tree, in ``seg_order``.

        ``in_reach`` is 1.0 within ``max_reach_km`` of the gage along the network
        and 0.0 beyond it.

        ``tshift`` is all zeros unless ``travel_time_lag`` is on, in which case
        it is one non-negative shift per segment, ``[n_seg]``, traced backward
        along the characteristic from the routed Courant number. The trace runs
        upstream from the gage so every celerity read is at a time already
        routed; see :meth:`_tree_tau_trace`.

        The distance part depends only on topology and reach length, so it is
        cached per run. The travel time is traced once, over a fixed span taken
        from the first window, and cached too, so that a memory knob cannot move
        it.
        """
        cache = getattr(self, "_lag_cache", None)
        if cache is None:
            cache = self._lag_cache = {}
        tau_cache = getattr(self, "_tau_cache", None)
        if tau_cache is None:
            tau_cache = self._tau_cache = {}
        by_trace = (
            self.travel_time_lag
            and nt > 0
            # cn is only needed to FILL the cache; once filled the trace runs
            # without it, and the driver stops assembling it.
            and (cn is not None or self._trace_cached(trees))
        )
        if by_trace and cn is not None and not self._trace_cached(self.trees):
            # Trace EVERY tree the run has, not just the ones with an innovation
            # this window. Filling the cache on first appearance made a gage that
            # first fires in window 3 read window 3's span, which is a different
            # slice of the record under a different max_loop_size: the same
            # partition dependence the fixed span exists to remove, hidden behind
            # "it happened not to matter on this domain, where all 19 gages fire
            # in window 1". Tracing them together also lets the driver stop
            # exporting the Courant field once this is done.
            #
            # apply_in_kernel normally fills this through _ensure_trace, before
            # its innovation-dependent early returns; this branch covers a direct
            # call (tests, and any future caller that assembles cn itself).
            self._ensure_trace_from_cn(cn, cn_colpos or {}, dt)
        out, unusable, dropped, total = {}, 0, 0, 0
        hist: list[float] = []
        tau_hist: list[float] = []
        why: dict[str, int] = {}
        for site, tree in trees.items():
            hit = cache.get(site)
            if hit is None:
                dist, bad = self._tree_distance_km(tree)
                hit = cache[site] = (
                    (dist <= self.max_reach_km).astype(np.float64),
                    np.zeros(dist.shape[0], dtype=np.int64),
                    bad,
                    dist,
                )
            inside, shift, bad, dist = hit
            if by_trace:
                # Traced ONCE per run over a fixed span taken from the start of
                # the first window, then cached, because the trace is
                # record-dependent: the walk stops at the start of the
                # record, so a longer window resolves MORE segments (measured
                # unresolved counts 14,023 / 12,841 / 9,596 / 12,813 across four
                # windows) and recomputing per window moves tau as well (median
                # 12.9 / 12.1 / 14.4 / 13.6 h). Both would let max_loop_size, a
                # memory knob, change discharge. Slicing a fixed span from a
                # fixed start makes every partition read exactly the same data,
                # and caps the longest travel time the trace may resolve at
                # lag_window_h.
                # Keyed by the span as well as the site: tau is in TIMESTEPS, so
                # a cache entry is only valid for the dt and lag_window_h it was
                # traced under. Filled for every tree at once, above.
                hit_tau = tau_cache.get((site, int(getattr(self, "_tau_span", -1))))
                if hit_tau is None:
                    # Only reachable if the tree set grew after construction
                    # (the build never does this). An untimed fallback here
                    # would silently run a different estimator for one gage.
                    msg = (
                        f"scaling DA: no traced travel time for site {site!r} "
                        "and no Courant field to trace one from. The tree set "
                        "is static and every tree is traced together on the "
                        "first routed window, so this state should be "
                        "unreachable; if it is reached, the run cannot honor "
                        "travel_time_lag for this gage."
                    )
                    raise RuntimeError(msg)
                tau1d, tau_counts = hit_tau
                for reason, n in tau_counts.items():
                    why[reason] = why.get(reason, 0) + n
                # A reach the router cannot speak for carries inf, which drops it
                # and its subtree from the correction rather than guessing a
                # speed for it.
                reachable = np.isfinite(tau1d)
                inside = inside * reachable.astype(np.float64)
                # ONE non-negative lag per segment, constant over the run: it is
                # traced over a fixed span, so it has no time argument to vary
                # over, and dQ_o(t + tau) is the only direction that makes the
                # upstream field reproduce what the gage observed. Kept 1-D
                # deliberately -- a 1-D non-negative shift is what the COMPILED
                # kernel takes, and repeating it to [n_seg, nt] to satisfy the
                # 2-D interface both materialised the array and forced the NumPy
                # path for every window of a CONUS run.
                shift = np.rint(np.where(reachable, tau1d, 0.0)).astype(np.int64)
                tau_hist.extend(
                    (tau1d[reachable & (inside > 0)] * float(dt) / 3600.0).tolist()
                )
            unusable += bad
            dropped += int(inside.size - inside.sum())
            total += int(inside.size)
            hist.extend(dist[inside > 0].tolist())
            out[site] = (inside, shift)
        if by_trace:
            th = np.asarray(tau_hist) if tau_hist else np.zeros(1)
            LOG.info(
                "scaling DA: travel time traced along the characteristic from the "
                "routed Courant number, applied as dQ_o(t + tau) -- over the "
                "RESOLVED segments only: median %.1f h, p90 %.1f h, max %.1f h; "
                "%d segment(s) unresolved and excluded (%s; only 'short' "
                "resolves under a longer lag_window_h).",
                float(np.median(th)), float(np.percentile(th, 90)),
                float(th.max()), sum(why.values()),
                ", ".join(f"{k}: {v}" for k, v in why.items() if v) or "none",
            )
        if unusable:
            LOG.warning(
                "scaling DA: %d tree reach(es) have no usable length; they are "
                "held outside the propagation limit rather than treated as "
                "zero distance.",
                unusable,
            )
        d = np.asarray(hist) if hist else np.zeros(1)
        LOG.info(
            "scaling DA: propagation limit %.0f km along the network -- %d of %d "
            "segment(s) beyond it (not corrected); distance within the limit: "
            "median %.1f km, p90 %.1f km, max %.1f km",
            self.max_reach_km, dropped, total,
            float(np.median(d)), float(np.percentile(d, 90)), float(d.max()),
        )
        return out

    def _smooth_innovation(
        self, dq_o: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        """FORWARD moving average of the gage innovation over the spread window.

        What an upstream segment needs at time t is the innovation the gage will
        report once this water arrives, ``dQ_o(t + tau)``: the segment's present
        state is corrected, not its past. Averaging over ``[t, t + T]`` is the
        estimator of that when tau is treated as unknown on the interval.

        The window is one-sided for that reason. A centred window would average
        in ``dQ_o(t - T/2)``, water that had already passed the gage, which is a
        negative travel time.

        Two properties of the edges, both deliberate and neither obvious:

        - The tail is padded with the LAST value, not with zeros, so the final
          timestep always comes out equal to its own raw innovation. That is
          what the forecast hand-off snapshots, so the seeded state is the same
          for every T. Zero padding would conserve volume at the tail instead
          but would shrink the seeded correction by a factor of the window.
        - Volume is therefore preserved only in the interior. An innovation
          within T of the START of the supplied series loses part of its mass
          (there are no earlier output steps to carry it) and one at the very
          END is amplified by the padding. Interior steps of a run are covered
          by the halo; the run's own first and last T hours are not.

        Width 0 leaves the innovation untouched and every property above becomes
        trivial.
        """
        win = round(self.innovation_spread_h * 3600.0 / float(dt)) + 1
        if win < 2:
            return dq_o
        n = dq_o.shape[0]
        # Padded at the tail only, with the last value, so a window running past
        # the end of the supplied innovation falls back to persistence rather
        # than to a shortened average that quietly reweights the edge.
        padded = np.concatenate([dq_o, np.full(win - 1, dq_o[-1])])
        csum = np.concatenate([[0.0], np.cumsum(padded)])
        return (csum[win:win + n] - csum[:n]) / win

    # Manning wide-channel exponent: Q ~ h^(5/3), so h scales as the discharge
    # ratio to the 3/5. Used to carry a discharge correction into DEPTH.
    _DEPTH_EXPONENT: float = 0.6
    # Bound on the depth ratio. A correction that would multiply or divide depth
    # by more than this is not carried: past it the wide-channel approximation
    # is doing more work than the correction it is transporting.
    _DEPTH_RATIO_CAP: float = 2.0

    def _log_depth_transfer(self) -> None:
        """Report how far the depth transform actually moved the state.

        The transform is an approximation of the kernel's own geometry, so the
        size of the move is what says whether that approximation matters: a
        fraction of a percent is noise against MC's coefficients, and a large
        move means the wide-channel relation is carrying more than it should.
        """
        moves = getattr(self, "_depth_moves", np.empty(0))
        if not moves.size:
            return
        dh = moves ** self._DEPTH_EXPONENT
        LOG.info(
            "scaling DA: correction carried into depth on %d segment-step(s); "
            "depth ratio median %.4f, p1 %.4f, p99 %.4f, largest change %.2f%%",
            dh.size, float(np.median(dh)), float(np.percentile(dh, 1)),
            float(np.percentile(dh, 99)), 100.0 * float(np.abs(dh - 1.0).max()),
        )

    def _scatter_back(self, run_results: RunResults, q_corr: pd.DataFrame) -> None:
        """Write the corrected discharge back, and carry it into depth.

        Depth matters because it is STATE, not just output. ``new_q0`` snapshots
        ``h0`` from this array to seed the next chunk, and Muskingum-Cunge
        derives its wave celerity and X weighting from depth, so a corrected
        discharge paired with the uncorrected depth hands the forecast an
        internally inconsistent initial condition: the right flow routed on the
        wrong geometry.

        The transform is the wide-channel Manning relation, ``Q ~ h^(5/3)``, so
        ``h_new = h_old * (Q_new/Q_old)^(3/5)``. That is the dQ-to-dy transform
        the source proposal anticipates for hydraulic routing, applied here to
        the MC warm state where the same inconsistency arises. It is an
        APPROXIMATION: the kernel solves a trapezoidal section with compound
        overbank geometry, and inverting that properly means calling its own
        depth solver.
        """
        pos = {int(c): k for k, c in enumerate(q_corr.columns)}
        corr = q_corr.to_numpy()  # [time, seg]
        # Column window this call covers. A chunked spread writes one slice at a
        # time and MUST come through here too: writing discharge alone left the
        # depth uncorrected on that slice, so a chunked run handed the forecast a
        # different h0 than an unchunked one while both reported the same
        # discharge, and the equivalence test only compared discharge.
        c0 = int(getattr(self, "_scatter_c0", 0))
        c1 = c0 + corr.shape[0]
        for r in run_results:
            ids_r = np.asarray(r[0])
            idx = [pos[int(i)] for i in ids_r]
            new_q = corr[:, idx].T
            old_q = r[1][:, 4 * c0:4 * c1:4]
            depth = r[1][:, 4 * c0 + 2:4 * c1:4]
            # Only where both discharges are usable: a ratio taken against a dry
            # or near-zero background is not a depth signal.
            live = (old_q > self.min_flow) & (new_q > self.min_flow) & (depth > 0.0)
            if live.any():
                ratio = np.ones_like(depth)
                np.divide(new_q, old_q, out=ratio, where=live)
                np.clip(
                    ratio, 1.0 / self._DEPTH_RATIO_CAP, self._DEPTH_RATIO_CAP, out=ratio
                )
                new_depth = np.where(
                    live, depth * ratio ** self._DEPTH_EXPONENT, depth
                )
                moved = ratio[live]
                if moved.size:
                    self._depth_moves = np.concatenate(
                        [getattr(self, "_depth_moves", np.empty(0)), moved.ravel()]
                    )
                r[1][:, 4 * c0 + 2:4 * c1:4] = new_depth
            # write corrected q back (float32 cast)
            r[1][:, 4 * c0:4 * c1:4] = new_q

    # -- Stage A: in-kernel downstream propagation -------------------------
    def build_usgs_df(
        self, t0: Any, dt: float, nts: int, da_run: Mapping[str, Any] | None = None
    ) -> pd.DataFrame:
        """The ``usgs_df`` injected into the MC nudging override (downstream leg).

        index = gage ``up_node_id``; columns = the kernel's positional grid
        ``[t0, t0+dt, ..., t0+nts*dt]`` (column 0 seeds the IC). The injected gage
        set is FIXED across loops for execution-plan determinism; NaN where a gage
        has no obs (kernel persists/decays).
        """
        if not self._da_sites:
            return pd.DataFrame()
        full_index = pd.date_range(
            pd.Timestamp(t0), periods=nts + 1, freq=pd.Timedelta(seconds=dt), name="time"
        )
        obs = self._loop_observations(full_index, da_run)
        if obs is None or obs.shape[1] == 0:
            return pd.DataFrame()
        rows, segs = [], []
        for site in obs.columns:
            seg = self.gage_seg.get(site)
            if seg is None:
                continue
            rows.append(obs[site].to_numpy())
            segs.append(int(seg))
        if not rows:
            return pd.DataFrame()
        usgs = pd.DataFrame(
            np.asarray(rows, dtype=np.float64),
            index=pd.Index(segs, name="link", dtype="int64"),
            columns=full_index,
        )
        return usgs[~usgs.index.duplicated(keep="first")].sort_index()

    def _loop_observations(
        self, index: pd.DatetimeIndex, da_run: Mapping[str, Any] | None = None
    ) -> pd.DataFrame | None:
        """Obs (time x site) over the fixed DA-site set, aligned to *index*."""
        sites = self._da_sites
        if self.synthetic_factor is not None:
            base = self._synthetic_baseline()
            if not base:
                return None
            cols = {
                s: base[s].reindex(index).to_numpy() * float(self.synthetic_factor)
                for s in sites
                if s in base
            }
            return pd.DataFrame(cols, index=index) if cols else None
        if self._ts_folder is not None and self._obs_sites:
            want = [s for s in sites if s in self._obs_sites]
            if not want:
                return None
            df = self._read_timeslices(want, index, da_run)
            return None if df is None else df.reindex(index)
        return None

    def _read_timeslices(
        self, want: list[str], index: pd.DatetimeIndex, da_run: Mapping[str, Any] | None
    ) -> pd.DataFrame | None:
        """Site-keyed observations for this window, via t-route's shared reader.

        ``get_obs_from_timeslices`` with an identity crosswalk gives site-keyed
        rows under the same QC and interpolation every other consumer uses. The
        file list is ``build_da_sets``'s per-window list; a driver that never
        built da_sets falls back to the whole directory.
        """
        from troute import nhd_io

        folder = self._ts_folder
        if folder is None:  # _loop_observations gates on this; keep the narrow local
            return None
        # A PRESENT key is authoritative even when empty (build_da_sets chose no
        # file); only an ABSENT key falls back to directory discovery.
        if da_run is not None and "usgs_timeslice_files" in da_run:
            paths = [folder / f for f in da_run["usgs_timeslice_files"]]
        else:
            paths = sorted(folder.glob(f"*.{_TIMESLICE_GLOB_SUFFIX}"))
        paths = [p for p in paths if p.exists()]
        if not paths:
            return None
        step = index[1] - index[0] if len(index) > 1 else pd.Timedelta(seconds=300)
        # The reader resamples in whole minutes; a dt it cannot express would drop
        # or misalign observations against the kernel grid, so refuse it.
        secs = int(step.total_seconds())
        if secs < 60 or secs % 60:
            raise ValueError(
                f"scaling_da cannot read TimeSlice observations at dt={secs}s: the "
                "shared reader resamples in whole minutes. Use a dt that is a "
                "multiple of 60 s."
            )
        # Cache: the BMI driver replays one da_run across every window of an
        # update_until call. The key covers the files, the step, the site list,
        # and the resample-grid PHASE (not t0 itself -- window boundaries share
        # the phase, and keying on t0 would defeat the cache for exactly that
        # replay; an off-phase t0 is a genuine miss).
        phase = int(pd.Timestamp(index[0]).value) % (secs * 1_000_000_000)
        key = (tuple(str(p) for p in paths), step, phase, tuple(want))
        if self._obs_cache is not None and self._obs_cache[0] == key:
            return self._obs_cache[1]
        obs = nhd_io.get_obs_from_timeslices(
            crosswalk_df=pd.DataFrame({"gages": want, "site": want}),
            crosswalk_gage_field="gages",
            crosswalk_dest_field="site",
            timeslice_files=paths,
            qc_threshold=self._qc_threshold,
            interpolation_limit=self._interpolation_limit,
            frequency_secs=secs,
            t0=index[0],
            cpu_pool=self._cpu_pool,
        )
        if obs.empty:
            return None
        df = obs.transpose()  # reader returns destination x time
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize(None) if idx.tz is not None else idx
        df = df[~df.index.duplicated(keep="first")].reindex(columns=want)
        # Cache what is RETURNED, so hit and miss hand back the same column set.
        self._obs_cache = (key, df)
        return df

    def _synthetic_baseline(self):
        """Frozen open-loop baseline for synthetic runs: ``site -> flow Series``
        read once from a no-DA output dir (feature_id = fp_id)."""
        if self._synth_base is not None:
            return self._synth_base
        if not self.synthetic_obs_baseline:
            # The config validator enforces this pairing for config-driven runs;
            # this guards direct construction. A synthetic run with no baseline
            # would inject nothing and exit 0.
            msg = (
                "scaling DA: synthetic_obs_factor is set but "
                "synthetic_obs_baseline (a no-DA output dir) is not; refusing "
                "to run a silent no-assimilation synthetic run."
            )
            raise ValueError(msg)
        import xarray as xr

        frames = []
        for f in sorted(Path(self.synthetic_obs_baseline).glob("troute_output_*.nc")):
            ds = xr.open_dataset(f)
            cols = [int(x) for x in ds["feature_id"].to_numpy()]
            frames.append(
                pd.DataFrame(
                    ds["flow"].transpose("time", "feature_id").to_numpy(),
                    index=pd.DatetimeIndex(ds["time"].to_numpy()),
                    columns=cols,
                )
            )
            ds.close()
        if not frames:
            msg = (
                f"scaling DA: synthetic_obs_baseline "
                f"{self.synthetic_obs_baseline!r} holds no troute_output_*.nc "
                "files; a synthetic run with no baseline observations would "
                "assimilate nothing and exit 0."
            )
            raise ValueError(msg)
        q = pd.concat(frames).sort_index()
        q = q[~q.index.duplicated()]
        self._synth_base = {
            s: q[self.gage_fp[s]] for s in self._da_sites if self.gage_fp.get(s) in q.columns
        }
        return self._synth_base

    @staticmethod
    def gather_innovation(run_results: RunResults) -> dict[int, NDArray[np.float64]]:
        """``gage segment -> per-timestep applied delta`` from the kernel's nudge.

        Exposed on its own so the driver can read window k+1's innovation and hand it
        to window k as the halo its forward window needs. Column 0 is the initial
        condition, so it is dropped to align with the output grid.
        """
        out: dict[int, NDArray[np.float64]] = {}
        for r in run_results:
            gage_ids = np.asarray(r[3][0]).astype("int64").ravel()
            if gage_ids.size == 0:
                continue
            nud = np.asarray(r[9])
            for k, seg in enumerate(gage_ids):
                out[int(seg)] = nud[k, 1:].astype(np.float64)
        return out

    def apply_in_kernel(
        self,
        run_results: RunResults,
        nts: int,
        dt: float,
        t0: Any,
        halo: Mapping[int, NDArray[np.float64]] | None = None,
        seed_untimed: bool = False,
    ) -> None:
        """Post-routing pass: spread the kernel-recorded innovation UPSTREAM.

        The gage and the reach below it are already corrected in-kernel. The
        kernel's ``nudge`` (r[9]) is the applied per-timestep delta; the gage
        background is reconstructed as ``Q_analyzed - nudge`` so the confluence
        flow-ratio split reads modeled flow at the tree root.

        ``halo`` maps gage segment to the NEXT window's innovation; the forward
        window reads it past this window's end.

        ``seed_untimed`` marks the state hand-off window (the one whose
        ``new_q0`` a forecast inherits). With the lag on, its FINAL timestep is
        rewritten with the untimed spread: the lagged read there falls past the
        analysis edge and decays to ~0 (measured: a lagged hand-off scores at
        the no-DA baseline), so the seed uses the current innovation while
        earlier timesteps keep the traced timing.
        """
        if not self.trees:
            return
        import os as _os
        if _os.environ.get("SCALING_DA_LEANMEM"):
            # UNSUPPORTED DIAGNOSTIC (profiling only; the supported memory control
            # is the spread_chunk_timesteps config field): release the idle loky
            # routing pool before the memory-heavy spread. Bit-identical.
            try:
                from joblib.externals.loky import get_reusable_executor

                get_reusable_executor().shutdown(wait=True)
            except Exception as _e:  # noqa: BLE001 -- best-effort memory relief
                LOG.debug("scaling DA: worker-pool release skipped (%s)", _e)
        _t = time.perf_counter()
        self._depth_moves = np.empty(0)
        # 0. Trace the travel time from the FIRST routed window, before anything
        # that can return early. Every early return below is keyed on this
        # window's OBSERVATIONS -- no nudges, all-zero innovations -- and which
        # window first carries an observation depends on where the boundaries
        # fell. Tracing on the first window that has a Courant field instead
        # makes the span the run's own opening, under every partition.
        self._ensure_trace(run_results, nts, dt)
        # 1. map kernel nudge -> gage segment.
        nudge_by_seg = self.gather_innovation(run_results)
        if not nudge_by_seg:
            LOG.info("scaling DA: in-kernel -- no gage nudges this loop; skipping spread.")
            return
        # 2. assemble q_model and gather candidate gages. Work on the numpy array +
        # a position map: per-column pandas writes on a CONUS-wide frame copy it.
        q_model = self._assemble_q_model(run_results, nts, dt, t0)
        nt = len(q_model.index)
        arr = q_model.to_numpy(dtype=np.float64, copy=True)  # [nts, N_seg]
        colpos = {int(c): i for i, c in enumerate(q_model.columns)}
        cand = {}  # site -> (seg, nudge) for gages with a non-zero applied delta
        halo_by_site: dict[str, NDArray[np.float64]] = {}
        for site in self.trees:
            seg = self.gage_seg.get(site)
            if seg is None or seg not in nudge_by_seg or seg not in colpos:
                continue
            nud = nudge_by_seg[seg]
            if nud.shape[0] != nt:
                continue
            # NOTE: a site with an all-zero OWN innovation stays a candidate. Its
            # halo can still be nonzero, and the lag reads it at t inside THIS
            # window -- gating on the window's own values made inclusion depend
            # on max_loop_size. Whether anything spreads is decided below, from
            # the concatenated array.
            cand[site] = (seg, nud)
            # The halo is the NEXT window's innovation at the same gage, so the
            # tail of this window reads real observations rather than falling
            # back to persistence.
            if halo is not None and seg in halo:
                halo_by_site[site] = np.asarray(halo[seg], dtype=np.float64)
        if not cand:
            LOG.info("scaling DA: in-kernel -- no candidate gages this loop.")
            return
        # 3. per site: splice the halo onto this window's innovation and include
        # the site if the result is nonzero anywhere. Inclusion is read off the
        # concatenated array, not off this window's own values: a site can have a
        # zero innovation here and a nonzero halo that the lag reads at a
        # timestep inside this window.
        dq_o_by_site = {}
        # column position -> what the gage's own output owes back once the spread
        # has added the smoothed value at the root. Empty when the spread width
        # is 0, where the raw and smoothed innovations are the same array.
        obs_debt: dict[int, NDArray[np.float64]] = {}
        _nhalo = 0
        for site, (seg, nud) in cand.items():
            # Spread in time before distributing in space. The average looks
            # FORWARD only, so the next window's innovation is appended and
            # nothing before this window is needed; only the own-window portion
            # is kept. Without the halo the tail of every window would fall back
            # to persistence and the result would depend on where the window
            # boundaries were drawn.
            h = halo_by_site.get(site)
            if h is not None and h.size:
                _nhalo = max(_nhalo, int(h.size))
                ext = self._smooth_innovation(np.concatenate([nud, h]), dt)
            else:
                ext = self._smooth_innovation(nud, dt)
            # `full` is this window's own steps, for the background subtraction,
            # which has to stay window-shaped. `ext` keeps the halo on the end: a
            # FORWARD shift reads dQ_o past this window, and slicing the halo off
            # here left every forward-shifted segment reading the clipped,
            # edge-decayed last value instead of the next window's real
            # innovation.
            full = ext[:nt]
            if not np.any(ext):
                continue  # nothing to spread at any reachable timestep
            # Reconstruct the gage background so the confluence flow-ratio split
            # reads modeled flow at the tree root. The RAW nudge is what the
            # kernel applied, so the raw nudge is what has to come off; anything
            # else leaves an inflated denominator and every upstream branch gets
            # a proportionally SMALLER share (background 10 with raw 10 and
            # smoothed 2 delivered 1.11 upstream instead of 2.00).
            #
            # The spread then adds the smoothed value back at the root, which
            # would leave the gage at background + smoothed and overwrite the
            # observation the in-kernel override placed there, so the difference
            # is restored on the gage column afterwards (see _restore_gage_obs).
            # Identical to subtracting `full` when the spread width is 0.
            arr[:, colpos[seg]] -= nud
            debt = nud - full
            if debt.any():
                obs_debt[colpos[seg]] = debt
            dq_o_by_site[site] = ext
        if not dq_o_by_site:
            LOG.info(
                "scaling DA: in-kernel -- innovations all zero everywhere the "
                "spread can read; downstream only."
            )
            return
        trees = {s: self.trees[s] for s in dq_o_by_site}
        gmap = {s: self.gage_seg[s] for s in trees}
        _need_cn = self.travel_time_lag
        # The trace reads the Courant field ONCE, on the window that fills the
        # cache; every later window would assemble an [nts, N_seg] float64 array
        # only to look up a tau it already has. Skipping it costs nothing in
        # results (the cache is what is used either way) and is most of the
        # trace's per-window overhead: 3.4 s against 1.7 s per window on the Ohio
        # subset, on an array that grows with the domain.
        if _need_cn and self._trace_cached(trees):
            _need_cn = False
        _cn = self._assemble_cn(run_results, nt) if _need_cn else None
        if _need_cn and _cn is None:
            # Fail closed: lag requested but untraceable (no MC Courant field);
            # applying untimed instead would be a different estimator.
            msg = (
                "travel_time_lag is enabled but the kernel exported no Courant "
                "field, so no travel time can be traced (the drivers force "
                "return_courant on while the trace is uncached, so this means "
                "no Muskingum-Cunge reach produced one -- e.g. an all-diffusive "
                "domain). Set travel_time_lag: false for this run."
            )
            raise RuntimeError(msg)
        lag = self._build_lag(
            trees, dt, nt, None if _cn is None else _cn[0],
            None if _cn is None else _cn[1],
        )
        # Every shift is now non-negative and 1-D (dQ_o(t + tau), one lag per
        # segment), so nothing reads before this window's start: the halo from
        # the NEXT window is the only cross-boundary read left, and the backward
        # history that used to be prepended here is gone with the backward
        # direction. Chunking is therefore always available.
        _chunk = self._resolve_spread_chunk(nt, arr.shape[1])
        # A chunk's shifts read FORWARD, so each chunk's innovation slice is
        # extended by the largest shift in play; slicing to the chunk alone would
        # make t+shift cross the boundary and read the wrong end.
        _maxshift = max((int(s.max()) for _, s in lag.values()), default=0)
        if _nhalo:
            LOG.info(
                "scaling DA: halo of %d step(s) from the next window feeds the lag's "
                "tail; %d step(s) of it are actually reached.",
                _nhalo, min(_nhalo, _maxshift),
            )
        _edge = float(np.exp(-float(dt) / (60.0 * self.da_decay_min)))
        if _chunk > 0:
            # Chunk the spread so its [nt x N] transients stay [chunk x N].
            # Timesteps are independent on this path, so the result is
            # bit-identical to the single call.
            for c0 in range(0, nt, _chunk):
                c1 = min(c0 + _chunk, nt)
                qmb = pd.DataFrame(
                    arr[c0:c1], index=q_model.index[c0:c1], columns=q_model.columns
                )
                # +_maxshift: the overlap the lag needs to read into.
                dqo = {s: dq_o_by_site[s][c0:min(c1 + _maxshift, dq_o_by_site[s].shape[0])]
                       for s in dq_o_by_site}
                qc, _ = apply_scaling_da(
                    qmb, None, gmap, trees,
                    min_flow_cms=self.min_flow, dq_o_by_site=dqo,
                    lag_by_site=lag, edge_decay=_edge,
                )
                corr = qc.to_numpy()  # [chunk_time, seg]
                for cp, d in obs_debt.items():
                    corr[:, cp] += d[c0:c1]
                # Same write path as the unchunked call, so the correction lands
                # in DEPTH as well as discharge on every chunk.
                self._scatter_c0 = c0
                self._scatter_back(
                    run_results,
                    pd.DataFrame(corr, index=q_model.index[c0:c1],
                                 columns=q_model.columns),
                )
            self._scatter_c0 = 0
        else:
            q_model_bg = pd.DataFrame(arr, index=q_model.index, columns=q_model.columns)
            q_corr, _dq = apply_scaling_da(
                q_model_bg,
                None,
                gmap,
                trees,
                min_flow_cms=self.min_flow,
                dq_o_by_site=dq_o_by_site,
                lag_by_site=lag,
                edge_decay=_edge,
            )
            if obs_debt:
                cvals = q_corr.to_numpy(dtype=np.float64, copy=True)
                for cp, d in obs_debt.items():
                    cvals[:, cp] += d
                q_corr = pd.DataFrame(cvals, index=q_corr.index, columns=q_corr.columns)
            self._scatter_back(run_results, q_corr)
        if seed_untimed and self.travel_time_lag and lag:
            # Rewrite the hand-off instant with the untimed spread: full
            # in-reach mask (no resolved-tau requirement -- the untimed arm
            # corrects the whole tree) and shift 0, so the seeded state is
            # exactly the untimed arm's. See the docstring.
            cache = getattr(self, "_lag_cache", {}) or {}
            lag0 = {
                site: (hit[0], np.zeros(hit[0].shape[0], dtype=np.int64))
                for site, hit in ((s, cache.get(s)) for s in trees)
                if hit is not None
            }
            qm_last = pd.DataFrame(
                arr[nt - 1 : nt], index=q_model.index[nt - 1 : nt],
                columns=q_model.columns,
            )
            qc_last, _ = apply_scaling_da(
                qm_last, None, gmap, trees,
                min_flow_cms=self.min_flow,
                dq_o_by_site={s: dq_o_by_site[s][nt - 1 : nt] for s in dq_o_by_site},
                lag_by_site=lag0, edge_decay=_edge,
            )
            last_vals = qc_last.to_numpy(dtype=np.float64, copy=True)
            for cp, d in obs_debt.items():
                last_vals[:, cp] += d[nt - 1 : nt]
            self._scatter_c0 = nt - 1
            self._scatter_back(
                run_results,
                pd.DataFrame(last_vals, index=q_model.index[nt - 1 : nt],
                             columns=q_model.columns),
            )
            self._scatter_c0 = 0
            LOG.info(
                "scaling DA: hand-off instant re-seeded UNTIMED across %d tree(s); "
                "the analysis record before it keeps the traced timing.",
                len(lag0),
            )
        self._log_depth_transfer()
        # Mean |innovation|: a corrected state should feed back a SMALLER
        # innovation next window; growth means divergent feedback.
        # THIS window's own steps, never the halo appended for the lag's tail.
        per_site = {
            s: float(np.mean(np.abs(dq_o_by_site[s][:nt]))) for s in dq_o_by_site
        }
        LOG.info(
            "scaling DA: upstream spread on %d gage trees, "
            "mean |innovation| %.4f cms [%.2fs]",
            len(trees),
            np.mean(list(per_site.values())),
            time.perf_counter() - _t,
        )
        # Per-site breakdown, so arms can be A/B'd on a matched site set.
        # ponytail: capped at 50 sites -- readable on Ohio, CONUS has thousands.
        if len(per_site) <= 50:
            LOG.info(
                "scaling DA: per-site |innovation| %s",
                " ".join(f"{s}={v:.4f}" for s, v in sorted(per_site.items())),
            )

    # ~0.5 GB of float64 per spread transient before auto-chunking kicks in;
    # CONUS-sized windows exceed it (with a measured swap cliff), Ohio never does.
    _SPREAD_CHUNK_BUDGET_ELEMS = 64_000_000

    def _resolve_spread_chunk(self, nt: int, n_seg: int) -> int:
        """Chunk size (timesteps) for this window's spread; 0 = no chunking.

        Priority: config field (0 = off, N = fixed), then the SCALING_DA_CHUNK env
        var (kept for the benchmark harnesses), then auto -- chunk only above the
        budget, and never under 'aligned'.
        """
        import os

        cfg = getattr(self, "spread_chunk_timesteps", None)
        if cfg is not None:
            return int(cfg)
        env = int(os.environ.get("SCALING_DA_CHUNK", "0") or "0")
        if env > 0:
            return env
        if nt * n_seg <= self._SPREAD_CHUNK_BUDGET_ELEMS:
            return 0
        chunk = max(1, self._SPREAD_CHUNK_BUDGET_ELEMS // max(n_seg, 1))
        LOG.info(
            "scaling DA: auto-chunking the upstream spread to %d timestep(s) per "
            "pass (%d segments x %d steps ~ %.1f GB per transient); bit-identical. "
            "Set scaling_da.spread_chunk_timesteps to override.",
            chunk, n_seg, nt, nt * n_seg * 8 / 1e9,
        )
        return chunk


def build_scaling_da(
    network: AbstractNetwork,
    supernetwork_parameters: Mapping[str, Any] | None,
    data_assimilation_parameters: Mapping[str, Any] | None,
    cpu_pool: int | None = None,
) -> ScalingDA | None:
    """Return a configured ScalingDA if enabled in the config, else None.

    The whole ``data_assimilation_parameters`` block is passed through: the
    observation folder, qc_threshold, and interpolation limit live on the parent.
    """
    da_params = data_assimilation_parameters or {}
    sda = da_params.get("streamflow_da") or {}
    if not sda.get("streamflow_scaling"):
        return None
    params = sda.get("streamflow_scaling_parameters") or {}
    return ScalingDA(network, params, da_params, cpu_pool)


def network_gage_segments(network: AbstractNetwork) -> set[int]:
    """The static gage split set handed to ``nwm_route``.

    Both drivers derive it through this one function so the cached execution plan
    never depends on a window's observations and the -V5/BMI plans split
    identically. ``network.gages`` is ``{"gages": {up_node_id: site_no}}``.
    """
    gages = (getattr(network, "gages", None) or {}).get("gages", {}) or {}
    return set(gages)
