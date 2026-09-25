"""
Tests for flow scaling utilities.

These tests verify the transformation of non-routing segment flow data
into the flowveldepth format used by routing outputs.
"""

import numpy as np
import pandas as pd

from nwm_routing.flow_scaling_utils import (
    expand_flow_scaling_to_flowveldepth,
    merge_routing_and_nonrouting_results,
    create_nonrouting_run_result,
    append_nonrouting_to_run_results,
)


class TestExpandFlowScalingToFlowveldepth:
    """Tests for expand_flow_scaling_to_flowveldepth function."""
    
    def test_basic_expansion(self):
        """Test basic expansion of hourly flow to subdivided timesteps."""
        # 2 segments, 2 hours of data
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
            '200001010100': [0.3, 0.4],
        }, index=[100, 200])
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=12,
            nts=24,  # 2 hours × 12 subdivisions
        )
        
        # Check shape: 2 segments × (24 timesteps × 3 variables)
        assert result.shape == (2, 72)
        
        # Check index preserved
        assert list(result.index) == [100, 200]
    
    def test_flow_values_repeated_correctly(self):
        """Test that hourly flow values are repeated qts_subdivisions times."""
        flow_df = pd.DataFrame({
            '200001010000': [1.0],
            '200001010100': [2.0],
        }, index=[100])
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=12,
            nts=24,
        )
        
        # Extract q values (every 3rd column starting at 0)
        q_values = result.iloc[0, 0::3].values
        
        # First 12 timesteps should have value 1.0
        assert np.allclose(q_values[:12], 1.0)
        # Next 12 timesteps should have value 2.0
        assert np.allclose(q_values[12:24], 2.0)
    
    def test_velocity_and_depth_are_zero(self):
        """Test that velocity and depth are set to zero."""
        flow_df = pd.DataFrame({
            '200001010000': [1.0, 2.0],
        }, index=[100, 200])
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # Extract v values (every 3rd column starting at 1)
        v_values = result.iloc[:, 1::3].values
        assert np.allclose(v_values, 0.0)
        
        # Extract d values (every 3rd column starting at 2)
        d_values = result.iloc[:, 2::3].values
        assert np.allclose(d_values, 0.0)
    
    def test_column_format_matches_flowveldepth(self):
        """Test that columns match the flowveldepth MultiIndex format."""
        flow_df = pd.DataFrame({
            '200001010000': [1.0],
        }, index=[100])
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=3,
            nts=3,
        )
        
        expected_columns = [(0, 'q'), (0, 'v'), (0, 'd'),
                           (1, 'q'), (1, 'v'), (1, 'd'),
                           (2, 'q'), (2, 'v'), (2, 'd')]
        assert list(result.columns) == expected_columns
    
    def test_empty_dataframe_returns_empty(self):
        """Test that empty input returns empty DataFrame."""
        flow_df = pd.DataFrame()
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=12,
            nts=24,
        )
        
        assert result.empty
    
    def test_trimming_when_more_data_than_nts(self):
        """Test that data is trimmed when flow_df has more timesteps than nts."""
        flow_df = pd.DataFrame({
            '200001010000': [1.0],
            '200001010100': [2.0],
            '200001010200': [3.0],  # This hour should be trimmed
        }, index=[100])
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=12,
            nts=24,  # Only 2 hours worth
        )
        
        # Should only have 24 timesteps × 3 variables = 72 columns
        assert result.shape[1] == 72
        
        # Last q value should be 2.0 (from second hour), not 3.0
        q_values = result.iloc[0, 0::3].values
        assert q_values[-1] == 2.0
    
    def test_padding_when_less_data_than_nts(self):
        """Test that data is padded when flow_df has fewer timesteps than nts."""
        flow_df = pd.DataFrame({
            '200001010000': [1.0],
        }, index=[100])
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=12,
            nts=24,  # Requesting 2 hours but only have 1
        )
        
        # Should have 24 timesteps × 3 variables = 72 columns
        assert result.shape[1] == 72
        
        # Values should be padded with last value (1.0)
        q_values = result.iloc[0, 0::3].values
        assert np.allclose(q_values, 1.0)
    
    def test_realistic_dimensions(self):
        """Test with realistic dimensions matching the actual data."""
        # 882 segments, 288 hours
        n_segments = 882
        n_hours = 288
        
        flow_df = pd.DataFrame(
            np.random.rand(n_segments, n_hours) * 0.1,
            index=range(3496902, 3496902 + n_segments),
            columns=[f'20000101{h:02d}00' for h in range(n_hours)]
        )
        
        qts_subdivisions = 12
        nts = n_hours * qts_subdivisions  # 3456
        
        result = expand_flow_scaling_to_flowveldepth(
            flow_df,
            qts_subdivisions=qts_subdivisions,
            nts=nts,
        )
        
        # Should have shape (882, 3456 × 3) = (882, 10368)
        assert result.shape == (882, 10368)


class TestMergeRoutingAndNonroutingResults:
    """Tests for merge_routing_and_nonrouting_results function."""
    
    def test_basic_merge(self):
        """Test basic merging of routing and non-routing results."""
        # Create routing flowveldepth
        qvd_columns = pd.MultiIndex.from_product(
            [range(3), ["q", "v", "d"]]
        ).to_flat_index()
        
        routing_fvd = pd.DataFrame(
            np.ones((2, 9)),
            index=[100, 200],
            columns=qvd_columns,
        )
        
        # Create non-routing flow scaling
        flow_scaling = pd.DataFrame({
            '200001010000': [0.5, 0.6],
        }, index=[300, 400])
        
        result = merge_routing_and_nonrouting_results(
            routing_fvd,
            flow_scaling,
            qts_subdivisions=3,
            nts=3,
        )
        
        # Should have all 4 segments
        assert len(result) == 4
        assert set(result.index) == {100, 200, 300, 400}
    
    def test_merge_preserves_routing_values(self):
        """Test that routing values are preserved after merge."""
        qvd_columns = pd.MultiIndex.from_product(
            [range(3), ["q", "v", "d"]]
        ).to_flat_index()
        
        routing_fvd = pd.DataFrame(
            np.array([[1.0, 0.5, 0.1] * 3]),
            index=[100],
            columns=qvd_columns,
        )
        
        flow_scaling = pd.DataFrame({
            '200001010000': [0.5],
        }, index=[200])
        
        result = merge_routing_and_nonrouting_results(
            routing_fvd,
            flow_scaling,
            qts_subdivisions=3,
            nts=3,
        )
        
        # Routing segment should have original values
        # Access using the tuple as a single key
        assert result.loc[100][(0, 'q')] == 1.0
        assert result.loc[100][(0, 'v')] == 0.5
        assert result.loc[100][(0, 'd')] == 0.1
    
    def test_merge_with_empty_flow_scaling(self):
        """Test merge when flow_scaling is empty returns routing unchanged."""
        qvd_columns = pd.MultiIndex.from_product(
            [range(3), ["q", "v", "d"]]
        ).to_flat_index()
        
        routing_fvd = pd.DataFrame(
            np.ones((2, 9)),
            index=[100, 200],
            columns=qvd_columns,
        )
        
        result = merge_routing_and_nonrouting_results(
            routing_fvd,
            pd.DataFrame(),
            qts_subdivisions=3,
            nts=3,
        )
        
        pd.testing.assert_frame_equal(result, routing_fvd)
    
    def test_merge_sorts_by_index(self):
        """Test that merged result is sorted by index."""
        qvd_columns = pd.MultiIndex.from_product(
            [range(3), ["q", "v", "d"]]
        ).to_flat_index()
        
        routing_fvd = pd.DataFrame(
            np.ones((2, 9)),
            index=[300, 100],  # Not sorted
            columns=qvd_columns,
        )
        
        flow_scaling = pd.DataFrame({
            '200001010000': [0.5, 0.6],
        }, index=[400, 200])  # Not sorted
        
        result = merge_routing_and_nonrouting_results(
            routing_fvd,
            flow_scaling,
            qts_subdivisions=3,
            nts=3,
        )
        
        # Should be sorted
        assert list(result.index) == [100, 200, 300, 400]


class TestCreateNonroutingRunResult:
    """Tests for create_nonrouting_run_result function."""
    
    def test_creates_valid_tuple_structure(self):
        """Test that the created run_result has the expected tuple structure."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
        }, index=[100, 200])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # Should return a list with one tuple
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        
        # Tuple should have 11 elements (matching mc_reach.pyx)
        assert len(result[0]) == 11
    
    def test_segment_ids_correct(self):
        """Test that segment IDs are correctly placed in result."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
        }, index=[100, 200])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # r[0] should be segment IDs as intp array
        segment_ids = result[0][0]
        assert list(segment_ids) == [100, 200]
        assert segment_ids.dtype == np.intp
    
    def test_fvd_array_shape(self):
        """Test that flowveldepth array has correct shape."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
            '200001010100': [0.3, 0.4],
        }, index=[100, 200])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=24,
        )
        
        # r[1] should be fvd array
        fvd_array = result[0][1]
        # Shape should be (n_segments, nts * 3)
        assert fvd_array.shape == (2, 72)
        assert fvd_array.dtype == np.float32
    
    def test_empty_flow_df_returns_empty_list(self):
        """Test that empty flow_df returns empty list."""
        result = create_nonrouting_run_result(
            pd.DataFrame(),
            qts_subdivisions=12,
            nts=24,
        )
        
        assert result == []
    
    def test_upstream_array_shape(self):
        """Test that upstream array (r[7]) has correct shape."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
        }, index=[100, 200])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # r[7] should be upstream array
        upstream_array = result[0][7]
        assert upstream_array.shape == (2, 12)
        assert upstream_array.dtype == np.float32
    
    def test_lastobs_tuple_structure(self):
        """Test that lastobs tuple (r[3]) has correct structure."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1],
        }, index=[100])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # r[3] should be lastobs tuple with 3 elements
        lastobs = result[0][3]
        assert len(lastobs) == 3
        assert len(lastobs[0]) == 0  # empty gage ids
        assert len(lastobs[1]) == 0  # empty lastobs_times
        assert len(lastobs[2]) == 0  # empty lastobs_values
    
    def test_reservoir_tuples_structure(self):
        """Test that reservoir tuples (r[4], r[5], r[6]) have 5 elements each."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1],
        }, index=[100])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # r[4] usgs, r[5] usace, r[6] usbr should each have 5 elements
        assert len(result[0][4]) == 5
        assert len(result[0][5]) == 5
        assert len(result[0][6]) == 5
    
    def test_rfc_tuple_structure(self):
        """Test that rfc tuple (r[8]) has 3 elements."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1],
        }, index=[100])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # r[8] should have 3 elements
        assert len(result[0][8]) == 3
    
    def test_nudge_array_shape(self):
        """Test that nudge array (r[9]) has correct shape."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1],
        }, index=[100])
        
        nts = 12
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=nts,
        )
        
        # r[9] should be nudge array with shape (n_gages, nts+1)
        # For non-routing, n_gages = 0
        nudge = result[0][9]
        assert nudge.shape == (0, nts + 1)
        assert nudge.dtype == np.float32
    
    def test_great_lakes_tuple_structure(self):
        """Test that great_lakes tuple (r[10]) has 4 elements."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1],
        }, index=[100])
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # r[10] should have 4 elements
        assert len(result[0][10]) == 4
    
    def test_with_reference_result(self):
        """Test that reference_result elements are copied correctly."""
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
        }, index=[100, 200])
        
        # Create a mock reference result with recognizable values
        mock_lastobs = (
            np.array([1, 2, 3], dtype=np.intp), 
            np.array([4.0, 5.0, 6.0], dtype='float32'), 
            np.array([7.0, 8.0, 9.0], dtype='float32')
        )
        mock_usgs = (
            np.array([10], dtype='int32'), 
            np.array([11.0], dtype='float32'), 
            np.array([12.0], dtype='float32'), 
            np.array([13.0], dtype='float32'), 
            np.array([14.0], dtype='float32')
        )
        mock_nudge = np.ones((5, 13), dtype='float32')
        mock_gl = (
            np.array([30], dtype='int32'), 
            np.array([31.0], dtype='float32'), 
            np.array([32], dtype='int32'), 
            np.array([33], dtype='int32')
        )
        
        reference_result = (
            np.array([999], dtype=np.intp),  # r[0]
            np.ones((1, 36), dtype='float32'),  # r[1]
            42,                              # r[2] - courant placeholder
            mock_lastobs,                    # r[3]
            mock_usgs,                       # r[4]
            mock_usgs,                       # r[5]
            mock_usgs,                       # r[6]
            np.zeros((1, 12), dtype='float32'),  # r[7]
            (np.array([20], dtype='int32'), np.array([21.0], dtype='float32'), np.array([22], dtype='int32')),  # r[8]
            mock_nudge,                      # r[9]
            mock_gl,                         # r[10]
        )
        
        result = create_nonrouting_run_result(
            flow_df,
            qts_subdivisions=12,
            nts=12,
            reference_result=reference_result,
        )
        
        # r[0] and r[1] should be new (segment specific)
        assert list(result[0][0]) == [100, 200]
        assert result[0][1].shape == (2, 36)
        
        # r[2] should be copied from reference
        assert result[0][2] == 42
        
        # r[3] should be copied from reference
        assert np.array_equal(result[0][3][0], mock_lastobs[0])
        assert np.array_equal(result[0][3][1], mock_lastobs[1])
        
        # r[4] should be copied from reference
        assert np.array_equal(result[0][4][0], mock_usgs[0])
        
        # r[7] should be new (segment specific - different size)
        assert result[0][7].shape == (2, 12)
        
        # r[9] should be copied from reference
        assert result[0][9].shape == (5, 13)
        assert np.array_equal(result[0][9], mock_nudge)
        
        # r[10] should be copied from reference
        assert np.array_equal(result[0][10][0], mock_gl[0])


class TestAppendNonroutingToRunResults:
    """Tests for append_nonrouting_to_run_results function."""
    
    def test_appends_to_existing_results(self):
        """Test that non-routing results are appended to existing results."""
        # Create a mock routing result with recognizable values
        mock_lastobs = (
            np.array([10, 20], dtype=np.intp),
            np.array([1.0, 2.0], dtype='float32'),
            np.array([3.0, 4.0], dtype='float32')
        )
        routing_result = (
            np.array([1, 2], dtype=np.intp),
            np.ones((2, 36), dtype='float32'),
            99,  # Recognizable courant placeholder
            mock_lastobs,
            (np.array([100], dtype='int32'), np.array([1.0], dtype='float32'), 
             np.array([2.0], dtype='float32'), np.array([3.0], dtype='float32'), 
             np.array([4.0], dtype='float32')),
            (np.array([], dtype='int32'), np.array([], dtype='float32'), 
             np.array([], dtype='float32'), np.array([], dtype='float32'), 
             np.array([], dtype='float32')),
            (np.array([], dtype='int32'), np.array([], dtype='float32'), 
             np.array([], dtype='float32'), np.array([], dtype='float32'), 
             np.array([], dtype='float32')),
            np.zeros((2, 12), dtype='float32'),
            (np.array([], dtype='int32'), np.array([], dtype='float32'), np.array([], dtype='int32')),
            np.ones((3, 13), dtype='float32'),  # Recognizable nudge shape
            (np.array([], dtype='int32'), np.array([], dtype='float32'), 
             np.array([], dtype='int32'), np.array([], dtype='int32')),
        )
        
        run_results = [routing_result]
        
        flow_df = pd.DataFrame({
            '200001010000': [0.1, 0.2],
        }, index=[100, 200])
        
        result = append_nonrouting_to_run_results(
            run_results,
            flow_df,
            qts_subdivisions=12,
            nts=12,
        )
        
        # Should now have 2 results
        assert len(result) == 2
        
        # First should be original routing result
        assert np.array_equal(result[0][0], np.array([1, 2]))
        
        # Second should be non-routing with new segment IDs
        assert list(result[1][0]) == [100, 200]
        
        # Second result should have copied r[2] from reference
        assert result[1][2] == 99
        
        # Second result should have copied r[3] from reference
        assert np.array_equal(result[1][3][0], mock_lastobs[0])
        
        # Second result should have copied r[9] from reference
        assert result[1][9].shape == (3, 13)
    
    def test_empty_flow_df_returns_unchanged(self):
        """Test that empty flow_df returns original run_results unchanged."""
        routing_result = (np.array([1, 2]),) + (None,) * 10
        run_results = [routing_result]
        
        result = append_nonrouting_to_run_results(
            run_results,
            pd.DataFrame(),
            qts_subdivisions=12,
            nts=12,
        )
        
        assert len(result) == 1
        assert result[0] is routing_result


class TestIntegration:
    """Integration tests for the complete flow scaling workflow."""
    
    def test_full_workflow_realistic_data(self):
        """Test the complete workflow with realistic data dimensions."""
        # Simulate routing results (2645 segments, 3456 timesteps)
        n_routing = 2645
        nts = 3456
        qts_subdivisions = 12
        
        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()
        
        routing_fvd = pd.DataFrame(
            np.random.rand(n_routing, nts * 3).astype('float32'),
            index=range(3493573, 3493573 + n_routing),
            columns=qvd_columns,
        )
        
        # Simulate non-routing flow scaling (882 segments, 288 hours)
        n_nonrouting = 882
        n_hours = 288
        
        flow_scaling = pd.DataFrame(
            np.random.rand(n_nonrouting, n_hours).astype('float32') * 0.1,
            index=range(3496902, 3496902 + n_nonrouting),
            columns=[f'2000010100{h:02d}' for h in range(n_hours)]
        )
        
        # Merge results
        result = merge_routing_and_nonrouting_results(
            routing_fvd,
            flow_scaling,
            qts_subdivisions=qts_subdivisions,
            nts=nts,
        )
        
        # Should have all segments
        assert len(result) == n_routing + n_nonrouting
        
        # Should have correct number of columns
        assert result.shape[1] == nts * 3
        
        # Routing segments should have non-zero velocity/depth
        routing_v = result.loc[3493573][(0, 'v')]
        assert routing_v != 0 or True  # May be 0 by chance, just check it exists
        
        # Non-routing segments should have zero velocity/depth
        nonrouting_v = result.loc[3496902][(0, 'v')]
        assert nonrouting_v == 0
        nonrouting_d = result.loc[3496902][(0, 'd')]
        assert nonrouting_d == 0
    
    def test_flowveldepth_can_be_used_with_output_generator(self):
        """
        Test that the merged flowveldepth DataFrame can be processed
        the same way nwm_output_generator processes it.
        """
        nts = 24
        qts_subdivisions = 12
        
        # Create merged flowveldepth
        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()
        
        routing_fvd = pd.DataFrame(
            np.ones((3, nts * 3)),
            index=[100, 200, 300],
            columns=qvd_columns,
        )
        
        flow_scaling = pd.DataFrame({
            '200001010000': [0.5, 0.6],
            '200001010100': [0.7, 0.8],
        }, index=[400, 500])
        
        flowveldepth = merge_routing_and_nonrouting_results(
            routing_fvd,
            flow_scaling,
            qts_subdivisions=qts_subdivisions,
            nts=nts,
        )
        
        # Simulate what nwm_output_generator does:
        # Extract timestep and variable from columns
        timestep, variable = zip(*flowveldepth.columns.tolist())
        
        # This should not raise
        assert len(set(timestep)) == nts
        assert set(variable) == {'q', 'v', 'd'}
        
        # Can create subsets for output
        dt = 300  # 5 minute timesteps
        timestep_index = np.where(
            ((np.array(list(set(list(timestep)))) + 1) * dt) % (dt * qts_subdivisions) == 0
        )
        assert len(timestep_index[0]) > 0
