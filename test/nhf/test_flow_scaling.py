"""Tests for distribute_catchment_discharge — catchment discharge distribution to links."""
import pytest
import pandas as pd
import numpy as np

from troute.nhf_discretize import distribute_catchment_discharge


def _make_links(records):
    """Helper: build links_df from list of dicts with link_id, fp_id, div_id, up_node_id."""
    df = pd.DataFrame(records).set_index('link_id')
    if 'up_node_id' not in df.columns:
        df['up_node_id'] = None
    return df


def _make_nodes(records):
    """Helper: build nodes_df from list of dicts."""
    if not records:
        return pd.DataFrame(columns=['node_id', 'dn_link_id', 'fp_id', 'is_terminal_nexus'])
    return pd.DataFrame(records)


def _make_ref_fps(records):
    """Helper: build reference_flowpaths from list of dicts."""
    return pd.DataFrame(records)


class TestDistributeCatchmentDischarge:
    """Tests for distribute_catchment_discharge function."""

    @pytest.fixture
    def single_divide_with_vfps(self):
        """
        Single divide (div_id=1) with fp_id=10, discretized into 3 links.
        Two VFPs (vfp_30: 20%, vfp_31: 30%) with terminal nexuses mapped to
        specific links. Remainder (50%) falls back to qlat (no downstream fp).

        Link chain (downstream → upstream):
            link_1001 (dn_node=500) -- node_901(tnex) -- link_1002 -- node_902(tnex) -- link_1003
        """
        div_lateralflows_df = pd.DataFrame(
            {'202001010000': [100.0], '202001010100': [110.0]},
            index=[1],
        )
        div_lateralflows_df.index.name = 'div_id'

        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1],
                'percentage_area_contribution': [0.2, 0.3],
                'dn_virtual_nex_id': [901, 902],
            },
            index=[30, 31],
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': 901},
            {'link_id': 1002, 'fp_id': 10, 'div_id': 1, 'up_node_id': 902},
            {'link_id': 1003, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
        ])

        nodes_df = _make_nodes([
            {'node_id': 901, 'dn_link_id': 1001, 'fp_id': 10, 'is_terminal_nexus': True},
            {'node_id': 902, 'dn_link_id': 1002, 'fp_id': 10, 'is_terminal_nexus': True},
        ])

        reference_flowpaths = _make_ref_fps([
            {'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1},
        ])

        # No downstream flowpath → remainder falls back to qlat
        fp_to_dn_nex = {10: 500}
        nex_to_dn_fp = {}  # 500 has no downstream fp

        return (div_lateralflows_df, vfp_dataframe, links_df, nodes_df,
                reference_flowpaths, fp_to_dn_nex, nex_to_dn_fp)

    def test_vfp_goes_to_upstream_inflow(self, single_divide_with_vfps):
        """VFP discharge should go to upstream_inflow_df at the target link."""
        div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp = single_divide_with_vfps

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp
        )

        # vfp_30 (20%) → node_901 → link_1001 in upstream_inflow
        assert upstream_inflow.loc[1001, '202001010000'] == pytest.approx(20.0)
        # vfp_31 (30%) → node_902 → link_1002 in upstream_inflow
        assert upstream_inflow.loc[1002, '202001010000'] == pytest.approx(30.0)
        # VFP flow should NOT be in routing_qlats
        remainder_per_link = 100.0 * 0.5 / 3
        assert routing_qlats.loc[1001, '202001010000'] == pytest.approx(remainder_per_link)
        assert routing_qlats.loc[1002, '202001010000'] == pytest.approx(remainder_per_link)

    def test_remainder_fallback_to_qlat(self, single_divide_with_vfps):
        """Remainder goes to qlat when there's no downstream flowpath (terminal)."""
        div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp = single_divide_with_vfps

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp
        )

        # Remainder 50% / 3 links = 16.667 per link in routing_qlats
        remainder_per_link = 100.0 * 0.5 / 3
        assert routing_qlats.loc[1003, '202001010000'] == pytest.approx(remainder_per_link)

    def test_mass_balance(self, single_divide_with_vfps):
        """Total routing_qlats + upstream_inflow must equal total divide qlat."""
        div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp = single_divide_with_vfps

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp
        )

        total_input = div_qlats['202001010000'].sum()
        total_output = routing_qlats['202001010000'].sum() + upstream_inflow['202001010000'].sum()
        assert total_output == pytest.approx(total_input)

    def test_multiple_timestamps(self, single_divide_with_vfps):
        """Distribution should work identically across multiple timestamps."""
        div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp = single_divide_with_vfps

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp
        )

        # t=1: div_qlat=110
        assert upstream_inflow.loc[1001, '202001010100'] == pytest.approx(110 * 0.2)
        assert upstream_inflow.loc[1002, '202001010100'] == pytest.approx(110 * 0.3)

    def test_flow_scaling_df_preserves_vfp_values(self, single_divide_with_vfps):
        """flow_scaling_df should contain per-VFP qlats indexed by virtual_fp_id."""
        div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp = single_divide_with_vfps

        _, flow_scaling_df, _ = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp
        )

        assert set(flow_scaling_df.index) == {30, 31}
        assert flow_scaling_df.loc[30, '202001010000'] == pytest.approx(20.0)   # 100 * 0.2
        assert flow_scaling_df.loc[31, '202001010000'] == pytest.approx(30.0)   # 100 * 0.3
        assert flow_scaling_df.loc[30, '202001010100'] == pytest.approx(22.0)   # 110 * 0.2
        assert flow_scaling_df.loc[31, '202001010100'] == pytest.approx(33.0)   # 110 * 0.3

    def test_routing_qlats_indexed_by_link_id(self, single_divide_with_vfps):
        """routing_qlats should be indexed by link_id, not virtual_fp_id."""
        div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp = single_divide_with_vfps

        routing_qlats, _, _ = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_dn_nex, nex_dn_fp
        )

        assert set(routing_qlats.index) == {1001, 1002, 1003}

    def test_divide_without_vfps(self):
        """A divide with no VFPs puts all remainder in qlat (terminal flowpath)."""
        div_qlats = pd.DataFrame(
            {'202001010000': [120.0]},
            index=[1],
        )
        div_qlats.index.name = 'div_id'

        vfp_df = pd.DataFrame(
            {'div_id': pd.Series(dtype=int),
             'percentage_area_contribution': pd.Series(dtype=float),
             'dn_virtual_nex_id': pd.Series(dtype=float)},
        )
        vfp_df.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 2001, 'fp_id': 20, 'div_id': 1, 'up_node_id': 800},
            {'link_id': 2002, 'fp_id': 20, 'div_id': 1, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([])
        ref_fps = _make_ref_fps([{'fp_id': 20, 'virtual_fp_id': np.nan, 'div_id': 1}])

        # Terminal flowpath: no downstream
        fp_to_dn_nex = {20: 500}
        nex_to_dn_fp = {}

        routing_qlats, flow_scaling_df, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        # All remainder → qlat (120 / 2 links = 60 each)
        assert routing_qlats.loc[2001, '202001010000'] == pytest.approx(60.0)
        assert routing_qlats.loc[2002, '202001010000'] == pytest.approx(60.0)
        assert upstream_inflow['202001010000'].sum() == pytest.approx(0.0)
        assert flow_scaling_df.empty

    def test_remainder_to_upstream_inflow_with_downstream_fp(self):
        """Remainder goes to upstream_inflow at first link of downstream flowpath."""
        div_qlats = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1],
        )
        div_qlats.index.name = 'div_id'

        # VFP covers 40%, remainder 60%
        vfp_df = pd.DataFrame(
            {
                'div_id': [1],
                'percentage_area_contribution': [0.4],
                'dn_virtual_nex_id': [901],
            },
            index=[30],
        )
        vfp_df.index.name = 'virtual_fp_id'

        # fp_id=10 (upstream), fp_id=20 (downstream)
        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': 901},
            {'link_id': 1002, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
            {'link_id': 2001, 'fp_id': 20, 'div_id': 2, 'up_node_id': 800},
            {'link_id': 2002, 'fp_id': 20, 'div_id': 2, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([
            {'node_id': 901, 'dn_link_id': 1001, 'fp_id': 10, 'is_terminal_nexus': True},
        ])
        ref_fps = _make_ref_fps([
            {'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1},
            {'fp_id': 20, 'virtual_fp_id': np.nan, 'div_id': 2},
        ])

        # fp_id=10 → dn_nex=500 → dn_fp=20
        fp_to_dn_nex = {10: 500, 20: 600}
        nex_to_dn_fp = {500: 20}

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        # VFP 40% → upstream_inflow at link_1001 (via terminal nexus 901)
        assert upstream_inflow.loc[1001, '202001010000'] == pytest.approx(40.0)
        # Remainder 60% → upstream_inflow at first link of fp_id=20 (link_2002, up_node_id=None)
        assert upstream_inflow.loc[2002, '202001010000'] == pytest.approx(60.0)
        # No qlat fallback
        assert routing_qlats['202001010000'].sum() == pytest.approx(0.0)
        # Mass balance
        total = routing_qlats['202001010000'].sum() + upstream_inflow['202001010000'].sum()
        assert total == pytest.approx(100.0)

    def test_multiple_divides(self):
        """Each divide distributes independently to its own flowpath's links."""
        div_qlats = pd.DataFrame(
            {'202001010000': [100.0, 200.0]},
            index=[1, 2],
        )
        div_qlats.index.name = 'div_id'

        # div_1 has 1 VFP (40%), div_2 has no VFPs
        vfp_df = pd.DataFrame(
            {
                'div_id': [1],
                'percentage_area_contribution': [0.4],
                'dn_virtual_nex_id': [901],
            },
            index=[30],
        )
        vfp_df.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': 901},
            {'link_id': 1002, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
            {'link_id': 2001, 'fp_id': 20, 'div_id': 2, 'up_node_id': 800},
            {'link_id': 2002, 'fp_id': 20, 'div_id': 2, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([
            {'node_id': 901, 'dn_link_id': 1001, 'fp_id': 10, 'is_terminal_nexus': True},
        ])
        ref_fps = _make_ref_fps([
            {'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1},
            {'fp_id': 20, 'virtual_fp_id': np.nan, 'div_id': 2},
        ])

        # Terminal flowpaths: no downstream for either
        fp_to_dn_nex = {10: 500, 20: 600}
        nex_to_dn_fp = {}

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        # div_1: VFP 40% → upstream_inflow at link_1001
        assert upstream_inflow.loc[1001, '202001010000'] == pytest.approx(40.0)
        # div_1: remainder 60% / 2 links = 30 each (qlat fallback)
        assert routing_qlats.loc[1001, '202001010000'] == pytest.approx(30.0)
        assert routing_qlats.loc[1002, '202001010000'] == pytest.approx(30.0)

        # div_2: no VFPs, 200 / 2 = 100 each (qlat fallback)
        assert routing_qlats.loc[2001, '202001010000'] == pytest.approx(100.0)
        assert routing_qlats.loc[2002, '202001010000'] == pytest.approx(100.0)

        # mass balance
        total = routing_qlats['202001010000'].sum() + upstream_inflow['202001010000'].sum()
        assert total == pytest.approx(300.0)

    def test_multiple_vfps_to_same_terminal_nexus(self):
        """Multiple VFPs draining to the same terminal nexus accumulate in upstream_inflow."""
        div_qlats = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1],
        )
        div_qlats.index.name = 'div_id'

        vfp_df = pd.DataFrame(
            {
                'div_id': [1, 1, 1],
                'percentage_area_contribution': [0.3, 0.2, 0.1],
                'dn_virtual_nex_id': [901, 901, 901],
            },
            index=[30, 31, 32],
        )
        vfp_df.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': 901},
            {'link_id': 1002, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([
            {'node_id': 901, 'dn_link_id': 1001, 'fp_id': 10, 'is_terminal_nexus': True},
        ])
        ref_fps = _make_ref_fps([{'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1}])

        fp_to_dn_nex = {10: 500}
        nex_to_dn_fp = {}

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        # All 3 VFPs (60% total) → upstream_inflow at link_1001
        assert upstream_inflow.loc[1001, '202001010000'] == pytest.approx(60.0)
        # Remainder 40% / 2 links = 20 each (qlat fallback)
        assert routing_qlats.loc[1001, '202001010000'] == pytest.approx(20.0)
        assert routing_qlats.loc[1002, '202001010000'] == pytest.approx(20.0)
        total = routing_qlats['202001010000'].sum() + upstream_inflow['202001010000'].sum()
        assert total == pytest.approx(100.0)

    def test_vfp_without_mapped_terminal_nexus(self):
        """VFP whose dn_virtual_nex_id is not a terminal nexus node falls back to qlat."""
        div_qlats = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1],
        )
        div_qlats.index.name = 'div_id'

        vfp_df = pd.DataFrame(
            {
                'div_id': [1],
                'percentage_area_contribution': [0.4],
                'dn_virtual_nex_id': [999],
            },
            index=[30],
        )
        vfp_df.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': 800},
            {'link_id': 1002, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([])  # no terminal nexus nodes
        ref_fps = _make_ref_fps([{'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1}])

        fp_to_dn_nex = {10: 500}
        nex_to_dn_fp = {}

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        # VFP 40% has no mapped terminal nexus → qlat fallback (spread across 2 links: 20 each)
        # Remainder 60% / 2 = 30 each (qlat fallback, terminal fp)
        assert routing_qlats.loc[1001, '202001010000'] == pytest.approx(50.0)
        assert routing_qlats.loc[1002, '202001010000'] == pytest.approx(50.0)
        assert upstream_inflow['202001010000'].sum() == pytest.approx(0.0)

    def test_single_link_flowpath(self):
        """Flowpath with a single link: VFP fallback + remainder fallback."""
        div_qlats = pd.DataFrame(
            {'202001010000': [80.0]},
            index=[1],
        )
        div_qlats.index.name = 'div_id'

        vfp_df = pd.DataFrame(
            {
                'div_id': [1],
                'percentage_area_contribution': [0.5],
                'dn_virtual_nex_id': [901],
            },
            index=[30],
        )
        vfp_df.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([])  # single link → no internal nodes
        ref_fps = _make_ref_fps([{'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1}])

        fp_to_dn_nex = {10: 500}
        nex_to_dn_fp = {}

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        # 901 not in nodes as terminal nexus → VFP qlat fallback to all links
        # VFP: 80 * 0.5 = 40 → link_1001 (qlat)
        # Remainder: 80 * 0.5 = 40 → link_1001 (qlat, terminal fp)
        assert routing_qlats.loc[1001, '202001010000'] == pytest.approx(80.0)
        assert upstream_inflow['202001010000'].sum() == pytest.approx(0.0)

    def test_total_flow_conserved_multiple_divides_timestamps(self):
        """Mass balance holds across multiple divides and timestamps."""
        div_qlats = pd.DataFrame(
            {
                '202001010000': [100.0, 200.0, 50.0],
                '202001010100': [110.0, 180.0, 60.0],
            },
            index=[1, 2, 3],
        )
        div_qlats.index.name = 'div_id'

        vfp_df = pd.DataFrame(
            {
                'div_id': [1, 2],
                'percentage_area_contribution': [0.6, 0.3],
                'dn_virtual_nex_id': [901, 902],
            },
            index=[30, 31],
        )
        vfp_df.index.name = 'virtual_fp_id'

        links_df = _make_links([
            {'link_id': 1001, 'fp_id': 10, 'div_id': 1, 'up_node_id': 901},
            {'link_id': 1002, 'fp_id': 10, 'div_id': 1, 'up_node_id': None},
            {'link_id': 2001, 'fp_id': 20, 'div_id': 2, 'up_node_id': None},
            {'link_id': 3001, 'fp_id': 30, 'div_id': 3, 'up_node_id': 800},
            {'link_id': 3002, 'fp_id': 30, 'div_id': 3, 'up_node_id': None},
        ])
        nodes_df = _make_nodes([
            {'node_id': 901, 'dn_link_id': 1001, 'fp_id': 10, 'is_terminal_nexus': True},
            {'node_id': 902, 'dn_link_id': 2001, 'fp_id': 20, 'is_terminal_nexus': True},
        ])
        ref_fps = _make_ref_fps([
            {'fp_id': 10, 'virtual_fp_id': np.nan, 'div_id': 1},
            {'fp_id': 20, 'virtual_fp_id': np.nan, 'div_id': 2},
            {'fp_id': 30, 'virtual_fp_id': np.nan, 'div_id': 3},
        ])

        fp_to_dn_nex = {10: 500, 20: 600, 30: 700}
        nex_to_dn_fp = {}  # all terminal

        routing_qlats, _, upstream_inflow = distribute_catchment_discharge(
            div_qlats, vfp_df, links_df, nodes_df, ref_fps, fp_to_dn_nex, nex_to_dn_fp
        )

        for col in div_qlats.columns:
            total_in = div_qlats[col].sum()
            total_out = routing_qlats[col].sum() + upstream_inflow[col].sum()
            assert total_out == pytest.approx(total_in), f"Mass balance violated at {col}"
