"""
Tests for NHF network connection building utilities.

These tests verify that the network connection logic correctly:
1. Builds flowpath-to-flowpath connections through nexus joins
2. Handles terminal nexuses correctly
3. Identifies headwaters and tailwaters
4. Validates connection integrity
"""

import pandas as pd

from troute.nhf_topology import (
    build_downstream_connections,
    build_upstream_terminal,
    find_headwaters,
    find_tailwaters,
    get_terminal_nexus_ids,
    validate_connections,
)


class TestBuildDownstreamConnections:
    """Tests for build_downstream_connections function."""
    
    def test_simple_linear_network(self):
        """Test a simple linear network: fp1 -> fp2 -> fp3 (terminal)."""
        # Flowpath 1 drains to nexus 100, flowpath 2 receives from nexus 100
        # Flowpath 2 drains to nexus 101, flowpath 3 receives from nexus 101
        # Flowpath 3 drains to nexus 102 (terminal)
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 101, 102],
            'up_virtual_nex_id': [pd.NA, 100, 101],
            'routing_segment': [True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {102}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        assert connections[1] == [2], "fp1 should flow to fp2"
        assert connections[2] == [3], "fp2 should flow to fp3"
        assert connections[3] == [], "fp3 is terminal, should have no downstream"
    
    def test_confluence_network(self):
        """Test a network with confluence: fp1 and fp2 both flow into fp3."""
        # Both fp1 and fp2 drain to nexus 100
        # fp3 receives from nexus 100
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 100, 101],
            'up_virtual_nex_id': [pd.NA, pd.NA, 100],
            'routing_segment': [True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {101}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        assert connections[1] == [3], "fp1 should flow to fp3"
        assert connections[2] == [3], "fp2 should flow to fp3"
        assert connections[3] == [], "fp3 is terminal"
    
    def test_divergence_network(self):
        """Test a network with divergence: fp1 flows into both fp2 and fp3."""
        # fp1 drains to nexus 100
        # Both fp2 and fp3 receive from nexus 100
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 101, 102],
            'up_virtual_nex_id': [pd.NA, 100, 100],
            'routing_segment': [True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {101, 102}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        # fp1 should flow to both fp2 and fp3
        assert set(connections[1]) == {2, 3}, "fp1 should flow to both fp2 and fp3"
        assert connections[2] == [], "fp2 is terminal"
        assert connections[3] == [], "fp3 is terminal"
    
    def test_no_terminal_nexus_ids_provided(self):
        """Test behavior when no terminal nexus IDs are explicitly provided."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2],
            'dn_virtual_nex_id': [100, 101],
            'up_virtual_nex_id': [pd.NA, 100],
            'routing_segment': [True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        
        connections = build_downstream_connections(routing_vfp, all_vfp)
        
        assert connections[1] == [2]
        # fp2's downstream nexus (101) has no flowpath receiving from it
        assert connections[2] == [], "No flowpath receives from nexus 101"
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [],
            'dn_virtual_nex_id': [],
            'up_virtual_nex_id': [],
            'routing_segment': []
        })
        routing_vfp = all_vfp[all_vfp['routing_segment'] == True]
        
        connections = build_downstream_connections(routing_vfp, all_vfp)
        
        assert connections == {}
    
    def test_single_flowpath(self):
        """Test a network with only one flowpath."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1],
            'dn_virtual_nex_id': [100],
            'up_virtual_nex_id': [pd.NA],
            'routing_segment': [True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {100}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        assert connections == {1: []}
    
    def test_connections_use_flowpath_ids_not_nexus_ids(self):
        """
        Critical test: verify connections map to flowpath IDs, not nexus IDs.
        
        This catches the bug where nexus IDs accidentally end up in the values.
        """
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [100, 200, 300],  # Flowpath IDs
            'dn_virtual_nex_id': [1000, 2000, 3000],  # Nexus IDs (different range)
            'up_virtual_nex_id': [pd.NA, 1000, 2000],
            'routing_segment': [True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {3000}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        # Values should be flowpath IDs, not nexus IDs
        assert connections[100] == [200], "Should use flowpath ID 200, not nexus ID"
        assert connections[200] == [300], "Should use flowpath ID 300, not nexus ID"
        
        # Validate no nexus IDs leaked into values
        is_valid, orphaned = validate_connections(connections)
        assert is_valid, f"Connections should be valid, but found orphaned IDs: {orphaned}"
    
    def test_routing_segment_filtering(self):
        """Test that only routing segments appear in connections."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3, 4],
            'dn_virtual_nex_id': [100, 101, 102, 103],
            'up_virtual_nex_id': [pd.NA, 100, 101, 102],
            'routing_segment': [True, True, False, True]  # fp3 is NOT a routing segment
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {103}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        # fp3 should NOT be in connections keys (it's not a routing segment)
        assert 3 not in connections, "Non-routing segment should not be in connections"
        # fp1 should connect directly to fp2 (fp3 is skipped)
        assert connections[1] == [2], "fp1 should flow to fp2"
        # fp2's downstream nexus (101) only connects to fp3 which is non-routing
        # so fp2 has no routing downstream connections
        assert connections[2] == [], "fp2 has no routing downstream (fp3 is non-routing)"
        assert connections[4] == [], "fp4 is terminal"
    
    def test_non_routing_segments_not_in_downstream_values(self):
        """Test that non-routing segments don't appear as downstream values."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 101, 102],
            'up_virtual_nex_id': [pd.NA, 100, 100],  # Both fp2 and fp3 receive from nex 100
            'routing_segment': [True, True, False]  # fp3 is NOT a routing segment
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_nexus_ids = {101, 102}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids)
        
        # fp1 should only connect to fp2, not fp3 (since fp3 is non-routing)
        assert connections[1] == [2], "fp1 should only flow to routing segment fp2"
        assert 3 not in connections[1], "Non-routing fp3 should not be in downstream list"


class TestGetTerminalNexusIds:
    """Tests for get_terminal_nexus_ids function."""
    
    def test_single_terminal(self):
        """Test extraction of a single terminal nexus."""
        vnex = pd.DataFrame({
            'virtual_nex_id': [100, 101, 102],
            'dn_virtual_fp_id': [1, 2, pd.NA]  # 102 is terminal
        })
        
        terminals = get_terminal_nexus_ids(vnex)
        
        assert terminals == {102}
    
    def test_multiple_terminals(self):
        """Test extraction of multiple terminal nexuses."""
        vnex = pd.DataFrame({
            'virtual_nex_id': [100, 101, 102, 103],
            'dn_virtual_fp_id': [1, pd.NA, 2, pd.NA]  # 101 and 103 are terminals
        })
        
        terminals = get_terminal_nexus_ids(vnex)
        
        assert terminals == {101, 103}
    
    def test_no_terminals(self):
        """Test when there are no terminal nexuses."""
        vnex = pd.DataFrame({
            'virtual_nex_id': [100, 101],
            'dn_virtual_fp_id': [1, 2]
        })
        
        terminals = get_terminal_nexus_ids(vnex)
        
        assert terminals == set()


class TestBuildUpstreamTerminal:
    """Tests for build_upstream_terminal function."""
    
    def test_single_upstream_to_terminal(self):
        """Test a single flowpath draining to a terminal nexus."""
        vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 101, 102],
        })
        terminal_nexus_ids = {102}
        
        upstream_terminal = build_upstream_terminal(vfp, terminal_nexus_ids)
        
        assert upstream_terminal == {102: {3}}
    
    def test_multiple_upstream_to_same_terminal(self):
        """Test multiple flowpaths draining to the same terminal nexus."""
        vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 101, 101],  # fp2 and fp3 both drain to terminal 101
        })
        terminal_nexus_ids = {101}
        
        upstream_terminal = build_upstream_terminal(vfp, terminal_nexus_ids)
        
        assert upstream_terminal == {101: {2, 3}}
    
    def test_multiple_terminal_nexuses(self):
        """Test with multiple terminal nexuses."""
        vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3, 4],
            'dn_virtual_nex_id': [100, 101, 102, 102],
        })
        terminal_nexus_ids = {101, 102}
        
        upstream_terminal = build_upstream_terminal(vfp, terminal_nexus_ids)
        
        assert upstream_terminal == {101: {2}, 102: {3, 4}}
    
    def test_no_flowpaths_drain_to_terminal(self):
        """Test when no flowpaths drain to terminal nexuses."""
        vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2],
            'dn_virtual_nex_id': [100, 101],
        })
        terminal_nexus_ids = {999}  # No flowpath drains to this
        
        upstream_terminal = build_upstream_terminal(vfp, terminal_nexus_ids)
        
        assert upstream_terminal == {}
    
    def test_empty_terminal_set(self):
        """Test with empty terminal nexus set."""
        vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2],
            'dn_virtual_nex_id': [100, 101],
        })
        
        upstream_terminal = build_upstream_terminal(vfp, set())
        
        assert upstream_terminal == {}


class TestValidateConnections:
    """Tests for validate_connections function."""
    
    def test_valid_connections(self):
        """Test validation of valid connections."""
        connections = {1: [2], 2: [3], 3: []}
        
        is_valid, orphaned = validate_connections(connections)
        
        assert is_valid is True
        assert orphaned == set()
    
    def test_invalid_connections_with_orphaned_ids(self):
        """Test detection of orphaned IDs (IDs in values but not in keys)."""
        connections = {1: [2], 2: [999], 3: []}  # 999 doesn't exist as a key
        
        is_valid, orphaned = validate_connections(connections)
        
        assert is_valid is False
        assert orphaned == {999}
    
    def test_multiple_orphaned_ids(self):
        """Test detection of multiple orphaned IDs."""
        connections = {1: [2, 888], 2: [999]}  # 888 and 999 don't exist
        
        is_valid, orphaned = validate_connections(connections)
        
        assert is_valid is False
        assert orphaned == {888, 999}
    
    def test_empty_connections(self):
        """Test validation of empty connections dict."""
        connections = {}
        
        is_valid, orphaned = validate_connections(connections)
        
        assert is_valid is True
        assert orphaned == set()
    
    def test_all_terminal_flowpaths(self):
        """Test validation when all flowpaths are terminal."""
        connections = {1: [], 2: [], 3: []}
        
        is_valid, orphaned = validate_connections(connections)
        
        assert is_valid is True
        assert orphaned == set()


class TestFindHeadwaters:
    """Tests for find_headwaters function."""
    
    def test_linear_network_headwater(self):
        """Test finding headwater in a linear network."""
        connections = {1: [2], 2: [3], 3: []}
        
        headwaters = find_headwaters(connections)
        
        assert headwaters == {1}
    
    def test_confluence_network_headwaters(self):
        """Test finding multiple headwaters in a confluence network."""
        connections = {1: [3], 2: [3], 3: []}
        
        headwaters = find_headwaters(connections)
        
        assert headwaters == {1, 2}
    
    def test_all_headwaters(self):
        """Test when all flowpaths are headwaters (disconnected)."""
        connections = {1: [], 2: [], 3: []}
        
        headwaters = find_headwaters(connections)
        
        assert headwaters == {1, 2, 3}
    
    def test_empty_network(self):
        """Test finding headwaters in empty network."""
        connections = {}
        
        headwaters = find_headwaters(connections)
        
        assert headwaters == set()


class TestFindTailwaters:
    """Tests for find_tailwaters function."""
    
    def test_linear_network_tailwater(self):
        """Test finding tailwater in a linear network."""
        connections = {1: [2], 2: [3], 3: []}
        
        tailwaters = find_tailwaters(connections)
        
        assert tailwaters == {3}
    
    def test_divergence_network_tailwaters(self):
        """Test finding multiple tailwaters in a divergence network."""
        connections = {1: [2, 3], 2: [], 3: []}
        
        tailwaters = find_tailwaters(connections)
        
        assert tailwaters == {2, 3}
    
    def test_all_tailwaters(self):
        """Test when all flowpaths are tailwaters (disconnected)."""
        connections = {1: [], 2: [], 3: []}
        
        tailwaters = find_tailwaters(connections)
        
        assert tailwaters == {1, 2, 3}
    
    def test_empty_network(self):
        """Test finding tailwaters in empty network."""
        connections = {}
        
        tailwaters = find_tailwaters(connections)
        
        assert tailwaters == set()


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_network_workflow(self):
        """
        Test the complete workflow of building and validating a network.
        
        Network topology:
            fp1 (headwater)
             |
             v
            fp2
             |
             v
            fp3 (tailwater)
        """
        # Create virtual flowpaths
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3],
            'dn_virtual_nex_id': [100, 101, 102],
            'up_virtual_nex_id': [pd.NA, 100, 101],
            'routing_segment': [True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        
        # Create virtual nexus
        vnex = pd.DataFrame({
            'virtual_nex_id': [100, 101, 102],
            'dn_virtual_fp_id': [2, 3, pd.NA]
        })
        
        # Build network
        terminal_ids = get_terminal_nexus_ids(vnex)
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_ids)
        upstream_terminal = build_upstream_terminal(all_vfp, terminal_ids)
        
        # Validate
        is_valid, orphaned = validate_connections(connections)
        assert is_valid, f"Network should be valid, orphaned: {orphaned}"
        
        # Check topology
        headwaters = find_headwaters(connections)
        tailwaters = find_tailwaters(connections)
        
        assert headwaters == {1}
        assert tailwaters == {3}
        assert connections == {1: [2], 2: [3], 3: []}
        assert upstream_terminal == {102: {3}}
    
    def test_complex_network_with_confluence_and_divergence(self):
        """
        Test a more complex network with both confluence and divergence.
        
        Network topology:
            fp1     fp2
             \\     /
              \\   /
               v v
               fp3
              /   \\
             v     v
            fp4   fp5
        """
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3, 4, 5],
            'dn_virtual_nex_id': [100, 100, 101, 102, 103],  # fp1 and fp2 both drain to nex 100
            'up_virtual_nex_id': [pd.NA, pd.NA, 100, 101, 101],  # fp4 and fp5 both receive from nex 101
            'routing_segment': [True, True, True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        
        vnex = pd.DataFrame({
            'virtual_nex_id': [100, 101, 102, 103],
            'dn_virtual_fp_id': [3, 4, pd.NA, pd.NA]  # Note: simplified, 101 feeds both 4 and 5
        })
        
        terminal_ids = get_terminal_nexus_ids(vnex)
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_ids)
        
        # Validate
        is_valid, orphaned = validate_connections(connections)
        assert is_valid, f"Network should be valid, orphaned: {orphaned}"
        
        # Check topology
        assert connections[1] == [3], "fp1 flows to fp3"
        assert connections[2] == [3], "fp2 flows to fp3"
        assert set(connections[3]) == {4, 5}, "fp3 flows to fp4 and fp5"
        assert connections[4] == []
        assert connections[5] == []
        
        headwaters = find_headwaters(connections)
        tailwaters = find_tailwaters(connections)
        
        assert headwaters == {1, 2}
        assert tailwaters == {4, 5}
    
    def test_mixed_routing_and_non_routing_segments(self):
        """
        Test network with mix of routing and non-routing segments.
        
        Network topology (R = routing, N = non-routing):
            fp1(R) -> fp2(N) -> fp3(R) -> fp4(R)
        
        Expected connections (only routing segments):
            fp1 -> [] (fp2 is non-routing)
            fp3 -> [fp4]
            fp4 -> []
        """
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2, 3, 4],
            'dn_virtual_nex_id': [100, 101, 102, 103],
            'up_virtual_nex_id': [pd.NA, 100, 101, 102],
            'routing_segment': [True, False, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_ids = {103}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_ids)
        
        # Only routing segments should be keys
        assert set(connections.keys()) == {1, 3, 4}
        # fp1 has no routing downstream (fp2 is non-routing)
        assert connections[1] == []
        # fp3 connects to fp4
        assert connections[3] == [4]
        # fp4 is terminal
        assert connections[4] == []


class TestEdgeCases:
    """Tests for edge cases and potential error conditions."""
    
    def test_nan_downstream_nexus(self):
        """Test handling of NaN downstream nexus IDs."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [1, 2],
            'dn_virtual_nex_id': [pd.NA, 100],
            'up_virtual_nex_id': [pd.NA, pd.NA],
            'routing_segment': [True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        
        connections = build_downstream_connections(routing_vfp, all_vfp)
        
        assert connections[1] == [], "NaN dn_nex should result in empty downstream list"
    
    def test_self_referential_nexus_ids(self):
        """
        Test when flowpath IDs and nexus IDs happen to overlap.
        
        This is the scenario that caused the original bug - when 
        virtual_fp_id == dn_virtual_nex_id, the old code would incorrectly
        map flowpaths to themselves.
        """
        # Flowpath 100 drains to nexus 100 (same number!)
        # Flowpath 200 receives from nexus 100
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [100, 200],
            'dn_virtual_nex_id': [100, 101],  # Note: fp 100 drains to nex 100
            'up_virtual_nex_id': [pd.NA, 100],
            'routing_segment': [True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_ids = {101}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_ids)
        
        # fp 100 should flow to fp 200, NOT to itself
        assert connections[100] == [200], "fp 100 should connect to fp 200, not itself"
        assert connections[200] == []
    
    def test_large_id_values(self):
        """Test handling of large ID values (like real hydrofabric IDs)."""
        all_vfp = pd.DataFrame({
            'virtual_fp_id': [3493573, 3493574, 3493575],
            'dn_virtual_nex_id': [3493573, 3493574, 3493575],
            'up_virtual_nex_id': [pd.NA, 3493573, 3493574],
            'routing_segment': [True, True, True]
        })
        routing_vfp = all_vfp[all_vfp['routing_segment']]
        terminal_ids = {3493575}
        
        connections = build_downstream_connections(routing_vfp, all_vfp, terminal_ids)
        
        is_valid, orphaned = validate_connections(connections)
        assert is_valid, f"Should handle large IDs, orphaned: {orphaned}"
        
        assert connections[3493573] == [3493574]
        assert connections[3493574] == [3493575]
        assert connections[3493575] == []
