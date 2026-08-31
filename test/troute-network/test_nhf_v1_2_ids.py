"""Regression tests for NHF >= 1.2 large/non-dense integer IDs.

NHF 1.1.4 used small, dense, sequential ids (fp_id 1..5.5M). NHF 1.2.0 switched
to large ids (~1e15) stored as floats. They are integral and below 2^53 so they
round-trip through float64 exactly, but their *magnitude* broke code that used
raw ids to size arrays:

* ``NHF._build_div_weighting_matrix`` did ``np.bincount(div_id)``, allocating a
  ``max(div_id)+1`` array (~10 PiB for a 1e15 id) -> MemoryError. Fixed by
  factorizing div_id to dense group codes first.

These tests build a minimal network with large ids and assert the network-build
math completes and is correct, guarding against a reintroduction of any
max(id)-sized allocation.
"""
import numpy as np
import pandas as pd
import pytest

from troute.NHF import NHF

# A large, sparse id space like NHF >= 1.2 (well above the 1.1.4 dense range,
# large enough that np.bincount(id) would attempt a multi-petabyte allocation).
BIG = 1_000_000_000_000_000  # 1e15


def _bare_nhf_with_dataframe(dataframe):
    """An NHF instance with only the state _build_div_weighting_matrix needs."""
    net = NHF.__new__(NHF)
    net._dataframe = dataframe
    return net


def test_build_div_weighting_matrix_handles_large_ids():
    """_build_div_weighting_matrix must not allocate a max(div_id)-sized array.

    With NHF 1.2 ids (~1e15) the old np.bincount(div_id) raised MemoryError; the
    factorized version completes and produces the same weights.
    """
    div_id = BIG + 7              # one divide/flowpath
    n1, n2, term = BIG + 100, BIG + 200, BIG + 300   # routing link node ids
    vfp = BIG + 50               # one virtual flowpath

    # Routing links for the single flowpath: n1 -> n2 -> term
    dataframe = pd.DataFrame(
        {"downstream": [n2, term], "fp_id": [div_id, div_id]},
        index=pd.Index([n1, n2], name="up_node_id"),
    )
    virtual_flowpaths = pd.DataFrame({
        "virtual_fp_id": [vfp],
        "percentage_area_contribution": [1.0],
        "dn_virtual_nex_id": [n2],
    })
    reference_flowpaths = pd.DataFrame({"virtual_fp_id": [vfp], "div_id": [div_id]})

    net = _bare_nhf_with_dataframe(dataframe)
    # Must not raise MemoryError (the regression) and must populate the vectors.
    net._build_div_weighting_matrix(virtual_flowpaths, reference_flowpaths, {})

    assert net.vfp_nex_ids.tolist() == [n1]          # vfp drains to the upstream link
    assert net.vfp_divs.tolist() == [div_id]
    # Single vfp covers 100% of its divide -> weight 1.0
    assert net.weights.shape == (1, 1)
    np.testing.assert_allclose(net.weights, [[1.0]])
    # zero_nodes = links that never receive lateral flow (n2 here)
    assert net.zero_nodes == [n2]


def test_div_weighting_distributes_remainder_across_large_id_divide():
    """The 'percentages don't sum to 100' remainder branch (the bincount user)
    must split the shortfall evenly across a divide's vfps, with large ids."""
    div_id = BIG + 11
    # two vfps in one divide, contributing 30% + 50% = 80% -> 20% remainder,
    # split evenly -> each vfp weight = own% + 10%.
    n1, n2, n3, term = BIG + 100, BIG + 200, BIG + 300, BIG + 400
    v1, v2 = BIG + 50, BIG + 60

    dataframe = pd.DataFrame(
        {"downstream": [n2, n3, term], "fp_id": [div_id, div_id, div_id]},
        index=pd.Index([n1, n2, n3], name="up_node_id"),
    )
    virtual_flowpaths = pd.DataFrame({
        "virtual_fp_id": [v1, v2],
        "percentage_area_contribution": [0.30, 0.50],
        "dn_virtual_nex_id": [n2, n3],
    })
    reference_flowpaths = pd.DataFrame({
        "virtual_fp_id": [v1, v2],
        "div_id": [div_id, div_id],
    })

    net = _bare_nhf_with_dataframe(dataframe)
    net._build_div_weighting_matrix(virtual_flowpaths, reference_flowpaths, {})

    # 0.30 + 0.10 and 0.50 + 0.10
    np.testing.assert_allclose(sorted(net.weights.ravel()), [0.40, 0.60])
    # weights for a single divide sum to 1.0 (100% of runoff routed)
    np.testing.assert_allclose(net.weights.sum(), 1.0)
