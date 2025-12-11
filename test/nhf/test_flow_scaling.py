"""A testing file to ensure flow-scaling works as intended"""
import pytest
import pandas as pd
import numpy as np

from troute.NHF import distribute_qlateral_to_virtual_flowpaths


class TestDistributeQlateralToVirtualFlowpaths:
    """Tests for distribute_qlateral_to_virtual_flowpaths function."""

    @pytest.fixture
    def simple_network(self):
        """
        Simple network with 2 divides, 6 virtual flowpaths:
        
        div_1 (qlat=100) has 3 virtual flowpaths forming a tree:
            vfp_30 (non-routing) --> nex_102 --> vfp_20 (routing) --> nex_101 --> vfp_10 (routing) --> nex_100 (outlet)
            
            - vfp_30: 20% area, routing=False, dn_nex=102, up_nex=None (headwater)
            - vfp_20: 30% area, routing=True,  dn_nex=101, up_nex=102
            - vfp_10: 50% area, routing=True,  dn_nex=100, up_nex=101
            
        div_2 (qlat=200) has 3 virtual flowpaths forming a tree:
            vfp_60 (non-routing) --> nex_203 --> vfp_50 (non-routing) --> nex_202 --> vfp_40 (routing) --> nex_200 (outlet)
            
            - vfp_60: 10% area, routing=False, dn_nex=203, up_nex=None (headwater)
            - vfp_50: 40% area, routing=False, dn_nex=202, up_nex=203
            - vfp_40: 50% area, routing=True,  dn_nex=200, up_nex=202
            
        Note: vfp_60 drains to vfp_50 (non-routing), which then drains to vfp_40 (routing).
              The function should map both vfp_60 and vfp_50 flows to vfp_40.
        """
        # Lateral flows by divide
        div_lateralflows_df = pd.DataFrame(
            {
                '202001010000': [100.0, 200.0],
                '202001010100': [110.0, 220.0],
            },
            index=[1, 2]  # div_id
        )
        div_lateralflows_df.index.name = 'div_id'

        # Virtual flowpath dataframe
        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1, 1, 2, 2, 2],
                'percentage_area_contribution': [0.5, 0.3, 0.2, 0.5, 0.4, 0.1],
                'dn_virtual_nex_id': [100, 101, 102, 200, 202, 203],
                'up_virtual_nex_id': [101, 102, np.nan, 202, 203, np.nan],  # nan = headwater
                'routing_segment': [True, True, False, True, False, False],
            },
            index=[10, 20, 30, 40, 50, 60]  # virtual_fp_id
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        # Flowpath dict: maps up_virtual_nex_id -> virtual_fp_id
        # "Which flowpath receives flow from this nexus?"
        # nex_102 flows into vfp_20 (up_nex=102)
        # nex_101 flows into vfp_10 (up_nex=101)
        # nex_202 flows into vfp_40 (up_nex=202)
        # nex_203 flows into vfp_50 (up_nex=203)
        flowpath_dict = {
            102: 20,  # nexus 102 -> vfp_20
            101: 10,  # nexus 101 -> vfp_10
            202: 40,  # nexus 202 -> vfp_40
            203: 50,  # nexus 203 -> vfp_50
        }

        return div_lateralflows_df, vfp_dataframe, flowpath_dict

    @pytest.fixture
    def simple_network_no_chained_non_routing(self):
        """
        Simple network where non-routing segments always drain directly to routing segments.
        
        div_1 (qlat=100) has 3 virtual flowpaths forming a tree:
            vfp_30 (non-routing) --> nex_102 --> vfp_20 (routing) --> nex_101 --> vfp_10 (routing) --> nex_100 (outlet)
            
            - vfp_30: 20% area, routing=False, dn_nex=102, up_nex=None (headwater)
            - vfp_20: 30% area, routing=True,  dn_nex=101, up_nex=102
            - vfp_10: 50% area, routing=True,  dn_nex=100, up_nex=101
            
        div_2 (qlat=200) has 3 virtual flowpaths forming a tree with branching:
            vfp_60 (non-routing) --> nex_202 --+
                                               +--> vfp_40 (routing) --> nex_200 (outlet)
            vfp_50 (non-routing) --> nex_202 --+
            
            - vfp_60: 10% area, routing=False, dn_nex=202, up_nex=None (headwater)
            - vfp_50: 40% area, routing=False, dn_nex=202, up_nex=None (headwater)
            - vfp_40: 50% area, routing=True,  dn_nex=200, up_nex=202
        """
        # Lateral flows by divide
        div_lateralflows_df = pd.DataFrame(
            {
                '202001010000': [100.0, 200.0],
                '202001010100': [110.0, 220.0],
            },
            index=[1, 2]  # div_id
        )
        div_lateralflows_df.index.name = 'div_id'

        # Virtual flowpath dataframe
        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1, 1, 2, 2, 2],
                'percentage_area_contribution': [0.5, 0.3, 0.2, 0.5, 0.4, 0.1],
                'dn_virtual_nex_id': [100, 101, 102, 200, 202, 202],
                'up_virtual_nex_id': [101, 102, np.nan, 202, np.nan, np.nan],  # nan = headwater
                'routing_segment': [True, True, False, True, False, False],
            },
            index=[10, 20, 30, 40, 50, 60]  # virtual_fp_id
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        # Flowpath dict: maps up_virtual_nex_id -> virtual_fp_id
        flowpath_dict = {
            102: 20,  # nexus 102 -> vfp_20
            101: 10,  # nexus 101 -> vfp_10
            202: 40,  # nexus 202 -> vfp_40
        }

        return div_lateralflows_df, vfp_dataframe, flowpath_dict

    def test_routing_qlats_area_distribution(self, simple_network_no_chained_non_routing):
        """Test that routing segments get correct area-weighted qlat."""
        div_lateralflows_df, vfp_dataframe, flowpath_dict = simple_network_no_chained_non_routing

        routing_qlats, _ = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # vfp_10 (div_1, 50%, routing=True): 100 * 0.5 = 50
        # No non-routing segments drain directly to vfp_10
        assert routing_qlats.loc[10, '202001010000'] == pytest.approx(50.0)
        assert routing_qlats.loc[10, '202001010100'] == pytest.approx(55.0)

    def test_non_routing_flows_added_to_downstream(self, simple_network_no_chained_non_routing):
        """Test that non-routing segment flows are added to downstream routing segment."""
        div_lateralflows_df, vfp_dataframe, flowpath_dict = simple_network_no_chained_non_routing

        routing_qlats, _ = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # vfp_20 (routing=True) should have:
        #   - its own flow: 100 * 0.3 = 30
        #   - vfp_30's flow (routing=False, dn_nex=102 -> vfp_20): 100 * 0.2 = 20
        #   - Total: 50
        assert routing_qlats.loc[20, '202001010000'] == pytest.approx(50.0)
        assert routing_qlats.loc[20, '202001010100'] == pytest.approx(55.0)  # 110 * 0.5

        # vfp_40 (routing=True) should have:
        #   - its own flow: 200 * 0.5 = 100
        #   - vfp_50's flow (routing=False, dn_nex=202 -> vfp_40): 200 * 0.4 = 80
        #   - vfp_60's flow (routing=False, dn_nex=202 -> vfp_40): 200 * 0.1 = 20
        #   - Total: 200
        assert routing_qlats.loc[40, '202001010000'] == pytest.approx(200.0)
        assert routing_qlats.loc[40, '202001010100'] == pytest.approx(220.0)

    def test_flow_scaling_df_preserves_original_values(self, simple_network_no_chained_non_routing):
        """Test that flow_scaling_df has non-routing segments with their original qlat values."""
        div_lateralflows_df, vfp_dataframe, flowpath_dict = simple_network_no_chained_non_routing

        _, flow_scaling_df = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # Should have vfp_30, vfp_50, vfp_60 (non-routing segments)
        assert set(flow_scaling_df.index) == {30, 50, 60}

        # vfp_30 (routing=False): 100 * 0.2 = 20
        assert flow_scaling_df.loc[30, '202001010000'] == pytest.approx(20.0)
        assert flow_scaling_df.loc[30, '202001010100'] == pytest.approx(22.0)  # 110 * 0.2

        # vfp_50 (routing=False): 200 * 0.4 = 80
        assert flow_scaling_df.loc[50, '202001010000'] == pytest.approx(80.0)
        assert flow_scaling_df.loc[50, '202001010100'] == pytest.approx(88.0)  # 220 * 0.4

        # vfp_60 (routing=False): 200 * 0.1 = 20
        assert flow_scaling_df.loc[60, '202001010000'] == pytest.approx(20.0)
        assert flow_scaling_df.loc[60, '202001010100'] == pytest.approx(22.0)  # 220 * 0.1

    def test_routing_qlats_only_contains_routing_segment_ids(self, simple_network_no_chained_non_routing):
        """Test that routing_qlats index only contains routing segment IDs."""
        div_lateralflows_df, vfp_dataframe, flowpath_dict = simple_network_no_chained_non_routing

        routing_qlats, _ = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # Should only have routing segments: vfp_10, vfp_20, vfp_40
        # vfp_30, vfp_50, vfp_60 are routing=False
        assert set(routing_qlats.index) == {10, 20, 40}
        assert 30 not in routing_qlats.index
        assert 50 not in routing_qlats.index
        assert 60 not in routing_qlats.index

    def test_total_flow_conserved(self, simple_network_no_chained_non_routing):
        """Test that total flow is conserved (input = output)."""
        div_lateralflows_df, vfp_dataframe, flowpath_dict = simple_network_no_chained_non_routing

        routing_qlats, _ = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # Total input: div_1 (100) + div_2 (200) = 300
        total_input = div_lateralflows_df['202001010000'].sum()

        # Total output in routing_qlats (which includes non-routing contributions)
        total_output = routing_qlats['202001010000'].sum()

        assert total_output == pytest.approx(total_input)

    def test_multiple_non_routing_to_same_downstream(self):
        """Test multiple non-routing segments draining to the same downstream routing segment."""
        div_lateralflows_df = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1]
        )
        div_lateralflows_df.index.name = 'div_id'

        # 1 routing, 3 non-routing all draining to same downstream via nex_101
        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1, 1, 1],
                'percentage_area_contribution': [0.4, 0.3, 0.2, 0.1],
                'dn_virtual_nex_id': [100, 101, 101, 101],
                'up_virtual_nex_id': [101, np.nan, np.nan, np.nan],
                'routing_segment': [True, False, False, False],
            },
            index=[10, 20, 30, 40]
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        flowpath_dict = {101: 10}  # all non-routing drain to vfp_10

        routing_qlats, flow_scaling_df = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # vfp_10 should have all flow:
        #   - own: 100 * 0.4 = 40
        #   - vfp_20: 100 * 0.3 = 30
        #   - vfp_30: 100 * 0.2 = 20
        #   - vfp_40: 100 * 0.1 = 10
        #   - Total: 100
        assert routing_qlats.loc[10, '202001010000'] == pytest.approx(100.0)

        # flow_scaling_df has the 3 non-routing segments
        assert set(flow_scaling_df.index) == {20, 30, 40}

    def test_all_routing_segments_no_non_routing(self):
        """Test case where all segments are routing (no non-routing)."""
        div_lateralflows_df = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1]
        )
        div_lateralflows_df.index.name = 'div_id'

        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1],
                'percentage_area_contribution': [0.6, 0.4],
                'dn_virtual_nex_id': [100, 101],
                'up_virtual_nex_id': [101, np.nan],
                'routing_segment': [True, True],
            },
            index=[10, 20]
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        flowpath_dict = {101: 10}

        routing_qlats, flow_scaling_df = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # Each routing segment gets its own area-weighted qlat
        assert routing_qlats.loc[10, '202001010000'] == pytest.approx(60.0)
        assert routing_qlats.loc[20, '202001010000'] == pytest.approx(40.0)

        # No non-routing segments
        assert flow_scaling_df.empty

    def test_all_non_routing_segments(self):
        """Test case where all segments are non-routing (edge case)."""
        div_lateralflows_df = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1]
        )
        div_lateralflows_df.index.name = 'div_id'

        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1],
                'percentage_area_contribution': [0.6, 0.4],
                'dn_virtual_nex_id': [101, 101],
                'up_virtual_nex_id': [np.nan, np.nan],
                'routing_segment': [False, False],
            },
            index=[10, 20]
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        # Both drain to downstream fp 99 (which is outside this dataframe)
        flowpath_dict = {101: 99}

        routing_qlats, flow_scaling_df = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # routing_qlats should have fp 99 with all the flow aggregated
        # (even though 99 isn't in vfp_dataframe, it receives the flow)
        assert routing_qlats.loc[99, '202001010000'] == pytest.approx(100.0)

        # flow_scaling_df should have original non-routing segments
        assert set(flow_scaling_df.index) == {10, 20}
        assert flow_scaling_df.loc[10, '202001010000'] == pytest.approx(60.0)
        assert flow_scaling_df.loc[20, '202001010000'] == pytest.approx(40.0)

    def test_multiple_divides_multiple_timestamps(self):
        """Test with multiple divides and multiple timestamps."""
        div_lateralflows_df = pd.DataFrame(
            {
                '202001010000': [100.0, 200.0, 50.0],
                '202001010100': [110.0, 180.0, 60.0],
                '202001010200': [90.0, 220.0, 55.0],
            },
            index=[1, 2, 3]  # div_id
        )
        div_lateralflows_df.index.name = 'div_id'

        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1, 2, 2, 3],
                'percentage_area_contribution': [0.7, 0.3, 0.5, 0.5, 1.0],
                'dn_virtual_nex_id': [100, 101, 102, 103, 101],
                'up_virtual_nex_id': [101, np.nan, np.nan, np.nan, np.nan],
                'routing_segment': [True, False, True, True, False],
            },
            index=[10, 20, 30, 40, 50]
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        # vfp_20 (div_1, non-routing) and vfp_50 (div_3, non-routing) both drain to nex_101 -> vfp_10
        flowpath_dict = {101: 10}

        routing_qlats, flow_scaling_df = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # vfp_10 at t=0:
        #   - own: 100 * 0.7 = 70
        #   - vfp_20: 100 * 0.3 = 30
        #   - vfp_50: 50 * 1.0 = 50
        #   - Total: 150
        assert routing_qlats.loc[10, '202001010000'] == pytest.approx(150.0)

        # vfp_10 at t=1:
        #   - own: 110 * 0.7 = 77
        #   - vfp_20: 110 * 0.3 = 33
        #   - vfp_50: 60 * 1.0 = 60
        #   - Total: 170
        assert routing_qlats.loc[10, '202001010100'] == pytest.approx(170.0)

        # vfp_30 and vfp_40 (both routing, no non-routing drains to them)
        assert routing_qlats.loc[30, '202001010000'] == pytest.approx(100.0)  # 200 * 0.5
        assert routing_qlats.loc[40, '202001010000'] == pytest.approx(100.0)  # 200 * 0.5

        # flow_scaling_df
        assert set(flow_scaling_df.index) == {20, 50}

    def test_non_routing_mapped_to_correct_downstream_explicit(self):
        """
        Explicitly test that non-routing segments are mapped to correct downstream
        routing segment via flowpath_dict with clear verification.
        
        Network (single divide, dendritic tree with two branches):
        
            vfp_100 (non-routing) --dn_nex=901--> vfp_200 (routing) --dn_nex=900--> outlet
            vfp_102 (non-routing) --dn_nex=901--/
            
            vfp_101 (non-routing) --dn_nex=902--> vfp_201 (routing) --dn_nex=900--> outlet
        """
        div_lateralflows_df = pd.DataFrame(
            {'202001010000': [100.0]},
            index=[1]
        )
        div_lateralflows_df.index.name = 'div_id'

        vfp_dataframe = pd.DataFrame(
            {
                'div_id': [1, 1, 1, 1, 1],
                'percentage_area_contribution': [0.1, 0.2, 0.3, 0.15, 0.25],
                'dn_virtual_nex_id': [901, 902, 901, 900, 900],
                'up_virtual_nex_id': [np.nan, np.nan, np.nan, 901, 902],
                'routing_segment': [False, False, False, True, True],
            },
            index=[100, 101, 102, 200, 201]
        )
        vfp_dataframe.index.name = 'virtual_fp_id'

        # vfp_200 receives from nex_901, vfp_201 receives from nex_902
        flowpath_dict = {
            901: 200,
            902: 201,
        }

        routing_qlats, flow_scaling_df = distribute_qlateral_to_virtual_flowpaths(
            div_lateralflows_df, vfp_dataframe, flowpath_dict
        )

        # vfp_200 should have:
        #   - its own: 100 * 0.15 = 15
        #   - vfp_100's (dn_nex=901): 100 * 0.1 = 10
        #   - vfp_102's (dn_nex=901): 100 * 0.3 = 30
        #   - Total: 55
        assert routing_qlats.loc[200, '202001010000'] == pytest.approx(55.0)

        # vfp_201 should have:
        #   - its own: 100 * 0.25 = 25
        #   - vfp_101's (dn_nex=902): 100 * 0.2 = 20
        #   - Total: 45
        assert routing_qlats.loc[201, '202001010000'] == pytest.approx(45.0)

        # flow_scaling_df should have non-routing with original values
        assert set(flow_scaling_df.index) == {100, 101, 102}
        assert flow_scaling_df.loc[100, '202001010000'] == pytest.approx(10.0)
        assert flow_scaling_df.loc[101, '202001010000'] == pytest.approx(20.0)
        assert flow_scaling_df.loc[102, '202001010000'] == pytest.approx(30.0)

        # Total flow conserved
        assert routing_qlats['202001010000'].sum() == pytest.approx(100.0)
