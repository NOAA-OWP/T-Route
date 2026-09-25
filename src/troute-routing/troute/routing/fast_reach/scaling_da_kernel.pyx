# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
"""Batched ``nogil`` Cython port of the simple-scaling DA upstream spread.

Output-identical replacement for ``_tree_dq_nodes`` (flow_ratio), run once over
flat CSR buffers for every surviving tree (~22k per-tree Python calls at CONUS
become one nogil pass); the NumPy reference remains the equivalence oracle.
Named differently from ``scaling_da.py`` because a same-named .pyx would shadow
it at import.
"""

from libc.math cimport pow
from libc.stdint cimport int64_t


cpdef void spread_trees(
    const double[:, ::1] q_model,        # [nt, N_seg] modeled flow (background at roots)
    const int64_t[::1] tree_off,         # [n_tree + 1] CSR tree boundaries
    const int64_t[::1] parent,           # [total_seg] LOCAL parent index within its tree (root = -1)
    const unsigned char[::1] is_junc,    # [total_seg] confluence-step flag
    const double[::1] area,              # [total_seg] drainage area (km^2)
    const double[::1] theta,             # [total_seg] per-segment theta (uniform per tree for flow_ratio)
    const int64_t[::1] qcol,             # [total_seg] -> column in q_model / q_corrected
    const double[:, ::1] dq_o,           # [n_tree, nt] per-tree gage correction dQ_o(t)
    const double min_flow_cms,
    const double edge_decay,           # per-step decay of the persisted edge innovation
    const double[::1] in_window,         # [total_seg] 1.0 inside the horizon, 0.0 beyond
    const int64_t[::1] tshift,           # [total_seg] travel-time lag in whole timesteps
    const int64_t[::1] pruned_off,       # [total_seg + 1] CSR into pruned_col, per tree node
    const int64_t[::1] pruned_col,       # [n_pruned] q_model columns of Edge-Case-1 branches
    double[:, ::1] q_corrected,          # [nt, N_seg] in/out, += applied
    double[:, ::1] dq_diag,              # [nt, N_seg] in/out, += applied (diagnostic)
    double[:, ::1] qbuf,                 # scratch [max_tree_seg, nt]
    double[:, ::1] mult,                 # scratch [max_tree_seg, nt]
    double[:, ::1] bsum,                 # scratch [max_tree_seg, nt]
) noexcept nogil:
    """Spread each tree's gage correction upstream; ``dQ(s,t) = dQ_o(t)*M(s,t)``.

    Per tree: pass 0 gathers the tree's flow columns into ``qbuf``, pass 1 sums
    surviving branch flows at each confluence, pass 2 telescopes the multiplier
    (junctions: ``Q_j / max(Q_parent, sum branches)``, the non-expansive clamp;
    linear steps: ``(A_j/A_parent)**theta``), accumulating ``dQ_o(t)*M`` into
    ``q_corrected``/``dq_diag``.

    CSR layout: tree ``it`` owns flat rows ``[tree_off[it], tree_off[it+1])``,
    gage at local 0; ``parent`` is the LOCAL index and precedes its children.
    Scratch is caller-allocated and fully rewritten per tree.
    """
    cdef Py_ssize_t n_tree = tree_off.shape[0] - 1
    cdef Py_ssize_t nt = q_model.shape[0]
    cdef Py_ssize_t it, s0, n_seg, a, p, t, col, col0, ts, over
    cdef Py_ssize_t nt_dq = dq_o.shape[1]
    cdef double factor, factor_lin, denom, qp, bs, m, applied, w
    cdef int64_t sh

    for it in range(n_tree):
        s0 = tree_off[it]
        n_seg = tree_off[it + 1] - s0

        # Pass 0: gather this tree's modeled-flow columns (== reference q_nodes)
        # and zero the branch-sum accumulator. Rewriting/zeroing every tree is
        # what keeps per-tree state per-tree (trap 1); reads below use only local
        # indices 0..n_seg-1 just written here.
        for a in range(n_seg):
            col = qcol[s0 + a]
            for t in range(nt):
                qbuf[a, t] = q_model[t, col]
                bsum[a, t] = 0.0

        # Pass 1: per-confluence sum of surviving branch flows, at the parent.
        # parent[s0 + a] is a LOCAL index within this tree (trap 2), indexing the
        # per-tree scratch directly.
        for a in range(1, n_seg):
            if is_junc[s0 + a]:
                p = parent[s0 + a]
                for t in range(nt):
                    bsum[p, t] += qbuf[a, t]

        # Edge Case 1 removed some branches from the tree, but Edge Case 2 splits the
        # parent increment across ALL of them, so their flow still belongs in the
        # denominator. Without it the clamp hands a surviving sibling the stopped
        # branch's share whenever routing lag drops the parent flow below the surviving
        # branch alone. They receive nothing themselves -- they are not tree members.
        for a in range(n_seg):
            for k in range(pruned_off[s0 + a], pruned_off[s0 + a + 1]):
                c = pruned_col[k]
                for t in range(nt):
                    bsum[a, t] += q_model[t, c]

        # Root (local 0): M = 1, applied = dQ_o (mirrors mult[:,0]=1 and the
        # caller's += of dq_nodes[:,0] at the gage column).
        col0 = qcol[s0]
        w = in_window[s0]
        sh = tshift[s0]
        for t in range(nt):
            mult[0, t] = 1.0
            ts = t + sh
            over = 0
            if ts >= nt_dq:
                over = ts - (nt_dq - 1)   # edge: persist the latest increment...
                ts = nt_dq - 1
            applied = dq_o[it, ts] * w
            if over > 0:
                applied = applied * pow(edge_decay, <double>over)   # ...decayed
            q_corrected[t, col0] += applied
            dq_diag[t, col0] += applied

        # Pass 2: telescoping per-node multiplier.
        for a in range(1, n_seg):
            p = parent[s0 + a]
            col = qcol[s0 + a]
            w = in_window[s0 + a]
            sh = tshift[s0 + a]
            if is_junc[s0 + a]:
                # Confluence split, clamped up to the branch sum (non-expansive,
                # trap 3); factor 0 at a dry junction (denom <= min_flow_cms,
                # trap 4).
                for t in range(nt):
                    qp = qbuf[p, t]
                    bs = bsum[p, t]
                    # NaN-propagating max, matching the reference
                    # np.maximum(q_parent, branch_sum): a plain ``qp if qp > bs``
                    # is np.fmax (suppresses a NaN parent, letting the split fall
                    # back to the finite branch sum and manufacture a correction
                    # the reference never applies). ``qp != qp`` forces denom=NaN
                    # when qp is NaN; a NaN bs already routes to ``else bs``. Then
                    # ``denom > min_flow_cms`` is False and factor is 0, exactly
                    # as np.maximum + the ok-mask do in _tree_dq_nodes.
                    denom = qp if (qp > bs or qp != qp) else bs
                    if denom > min_flow_cms:
                        factor = qbuf[a, t] / denom
                        # Bound the share into [0, 1]. A negative branch flow would give a
                        # negative multiplier and a branch above the denominator would
                        # amplify; both are input defects, not physics. No-op on valid input.
                        if factor < 0.0:
                            factor = 0.0
                        elif factor > 1.0:
                            factor = 1.0
                    else:
                        factor = 0.0
                    m = mult[p, t] * factor
                    mult[a, t] = m
                    ts = t + sh
                    over = 0
                    if ts >= nt_dq:
                        over = ts - (nt_dq - 1)
                        ts = nt_dq - 1
                    applied = dq_o[it, ts] * m * w
                    if over > 0:
                        applied = applied * pow(edge_decay, <double>over)
                    q_corrected[t, col] += applied
                    dq_diag[t, col] += applied
            else:
                # Linear step: area-scaling (Eq. 2) is constant in time -> hoist
                # it out of the t-loop (trap 5). M still telescopes through
                # time-varying junction ancestors via mult[p, t].
                if area[s0 + p] > 0.0:
                    factor_lin = pow(area[s0 + a] / area[s0 + p], theta[s0 + a])
                else:
                    factor_lin = 0.0
                # Same bound as the junction share, for the same reason: a non-monotone or
                # NaN area, or a negative theta, would otherwise amplify the correction as
                # it moves away from the observation. No-op on valid input.
                if factor_lin != factor_lin or factor_lin < 0.0:
                    factor_lin = 0.0
                elif factor_lin > 1.0:
                    factor_lin = 1.0
                for t in range(nt):
                    m = mult[p, t] * factor_lin
                    mult[a, t] = m
                    ts = t + sh
                    over = 0
                    if ts >= nt_dq:
                        over = ts - (nt_dq - 1)
                        ts = nt_dq - 1
                    applied = dq_o[it, ts] * m * w
                    if over > 0:
                        applied = applied * pow(edge_decay, <double>over)
                    q_corrected[t, col] += applied
                    dq_diag[t, col] += applied
