"""
Utility functions for handling non-routing segment flow scaling data.

These functions transform the flow scaling data (qlat for non-routing segments)
into the flowveldepth format used by the routing output, allowing seamless
integration of routing and non-routing segment results.
"""

import numpy as np
import pandas as pd


def expand_flow_scaling_to_flowveldepth(
    flow_scaling_df: pd.DataFrame,
    qts_subdivisions: int,
    nts: int,
) -> pd.DataFrame:
    """
    Expand flow scaling DataFrame to match flowveldepth format.
    
    The flow scaling DataFrame contains hourly qlat values for non-routing segments.
    This function expands it to match the flowveldepth format by:
    1. Repeating each hourly value `qts_subdivisions` times
    2. Adding velocity (v=0) and depth (d=0) columns
    3. Creating MultiIndex columns matching flowveldepth format
    
    Parameters
    ----------
    flow_scaling_df : pd.DataFrame
        DataFrame with non-routing segment qlat values.
        Index: virtual_fp_id
        Columns: timestamps (e.g., '200001010000')
        Values: lateral flow values (already distributed by area)
    qts_subdivisions : int
        Number of routing timesteps per forcing timestep (e.g., 12)
    nts : int
        Total number of routing timesteps
        
    Returns
    -------
    pd.DataFrame
        DataFrame in flowveldepth format:
        Index: virtual_fp_id
        Columns: MultiIndex of (timestep, variable) where variable is 'q', 'v', 'd'
        
    Examples
    --------
    >>> flow_df = pd.DataFrame({
    ...     '200001010000': [0.1, 0.2],
    ...     '200001010100': [0.15, 0.25]
    ... }, index=[100, 200])
    >>> result = expand_flow_scaling_to_flowveldepth(flow_df, qts_subdivisions=12, nts=24)
    >>> result.shape
    (2, 72)  # 2 segments × (24 timesteps × 3 variables)
    """
    if flow_scaling_df.empty:
        return pd.DataFrame()
    
    n_segments = len(flow_scaling_df)
    n_forcing_timesteps = len(flow_scaling_df.columns)
    
    # Get the flow values as numpy array for efficient manipulation
    flow_values = flow_scaling_df.values  # shape: (n_segments, n_forcing_timesteps)
    
    # Repeat each hourly value qts_subdivisions times along axis 1
    # This expands from (n_segments, n_hours) to (n_segments, n_hours * qts_subdivisions)
    expanded_flow = np.repeat(flow_values, qts_subdivisions, axis=1)
    
    # Trim or pad to match nts if necessary
    if expanded_flow.shape[1] > nts:
        expanded_flow = expanded_flow[:, :nts]
    elif expanded_flow.shape[1] < nts:
        # Pad with last value if we don't have enough timesteps
        padding = np.tile(expanded_flow[:, -1:], (1, nts - expanded_flow.shape[1]))
        expanded_flow = np.concatenate([expanded_flow, padding], axis=1)
    
    # Create the full flowveldepth array with q, v, d
    # Shape: (n_segments, nts, 3) where 3 = [q, v, d]
    fvd_3d = np.zeros((n_segments, nts, 3), dtype='float32')
    fvd_3d[:, :, 0] = expanded_flow  # q = flow values
    fvd_3d[:, :, 1] = 0.0            # v = 0 (no velocity for non-routing)
    fvd_3d[:, :, 2] = 0.0            # d = 0 (no depth for non-routing)
    
    # Reshape to 2D: (n_segments, nts * 3)
    fvd_2d = fvd_3d.reshape(n_segments, -1)
    
    # Create MultiIndex columns matching flowveldepth format
    qvd_columns = pd.MultiIndex.from_product(
        [range(nts), ["q", "v", "d"]]
    ).to_flat_index()
    
    # Create DataFrame
    result = pd.DataFrame(
        fvd_2d,
        index=flow_scaling_df.index,
        columns=qvd_columns,
        dtype='float32'
    )
    
    return result


def merge_routing_and_nonrouting_results(
    flowveldepth: pd.DataFrame,
    flow_scaling_df: pd.DataFrame,
    qts_subdivisions: int,
    nts: int,
) -> pd.DataFrame:
    """
    Merge routing segment results with non-routing segment flow scaling data.
    
    Parameters
    ----------
    flowveldepth : pd.DataFrame
        Routing results DataFrame with flowveldepth format.
        Index: routing segment virtual_fp_ids
        Columns: (timestep, variable) tuples
    flow_scaling_df : pd.DataFrame
        Non-routing segment qlat values.
        Index: non-routing segment virtual_fp_ids
        Columns: timestamps
    qts_subdivisions : int
        Number of routing timesteps per forcing timestep
    nts : int
        Total number of routing timesteps
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all segments (routing + non-routing)
        in flowveldepth format, sorted by index.
    """
    if flow_scaling_df.empty:
        return flowveldepth
    
    # Expand non-routing flow data to flowveldepth format
    nonrouting_fvd = expand_flow_scaling_to_flowveldepth(
        flow_scaling_df,
        qts_subdivisions,
        nts,
    )
    
    # Concatenate routing and non-routing results
    combined = pd.concat([flowveldepth, nonrouting_fvd], axis=0)
    
    # Sort by index for consistent ordering
    combined = combined.sort_index()
    
    return combined


def create_nonrouting_run_result(
    flow_scaling_df: pd.DataFrame,
    qts_subdivisions: int,
    nts: int,
    reference_result: tuple = None,
) -> list[tuple]:
    """
    Create a run_result entry for non-routing segments that matches
    the format returned by mc_reach.pyx compute_network_structured.
    
    The routing compute functions return results as a list of tuples.
    This function creates a similar tuple for non-routing segments.
    
    Parameters
    ----------
    flow_scaling_df : pd.DataFrame
        Non-routing segment qlat values.
    qts_subdivisions : int
        Number of routing timesteps per forcing timestep
    nts : int
        Total number of routing timesteps
    reference_result : tuple, optional
        An existing routing result tuple to use as template for elements
        r[2] through r[10]. If provided, these elements are copied directly
        to ensure exact compatibility. If None, empty placeholders are created.
        
    Returns
    -------
    list[tuple]
        List containing a single tuple matching mc_reach.pyx return format:
        (
            r[0]:  data_idx (segment IDs),
            r[1]:  flowveldepth array (n_segments, nts*3),
            r[2]:  courant placeholder (0),
            r[3]:  lastobs tuple (gage_ids, lastobs_times, lastobs_values),
            r[4]:  usgs tuple (5 elements),
            r[5]:  usace tuple (5 elements),
            r[6]:  usbr tuple (5 elements),
            r[7]:  upstream_array (n_segments, nts),
            r[8]:  rfc tuple (3 elements),
            r[9]:  nudge array (n_gages, nts+1),
            r[10]: great_lakes tuple (4 elements),
        )
    """
    if flow_scaling_df.empty:
        return []
    
    # Expand to flowveldepth format
    nonrouting_fvd = expand_flow_scaling_to_flowveldepth(
        flow_scaling_df,
        qts_subdivisions,
        nts,
    )
    
    # Extract components for run_result tuple format
    # r[0]: segment IDs as intp array (matching mc_reach.pyx line 844)
    segment_ids = np.asarray(nonrouting_fvd.index.values, dtype=np.intp)
    
    # r[1]: flowveldepth array - shape (n_segments, nts*3)
    # mc_reach.pyx reshapes from (n_segments, nts, 3) to (n_segments, nts*3)
    fvd_array = nonrouting_fvd.values.astype('float32')
    
    n_segments = len(segment_ids)
    
    # r[7]: upstream_array - shape (n_segments, nts)
    # This must be segment-specific (different size than routing segments)
    upstream_array = np.zeros((n_segments, nts), dtype='float32')
    
    if reference_result is not None:
        # Use the existing structure from routing results for r[2]-r[6], r[8]-r[10]
        # This ensures exact compatibility with the output generator
        run_result = (
            segment_ids,              # r[0]: data_idx (new - segment specific)
            fvd_array,                # r[1]: flowveldepth array (new - segment specific)
            reference_result[2],      # r[2]: courant placeholder (copy)
            reference_result[3],      # r[3]: lastobs tuple (copy)
            reference_result[4],      # r[4]: usgs tuple (copy)
            reference_result[5],      # r[5]: usace tuple (copy)
            reference_result[6],      # r[6]: usbr tuple (copy)
            upstream_array,           # r[7]: upstream array (new - segment specific)
            reference_result[8],      # r[8]: rfc tuple (copy)
            reference_result[9],      # r[9]: nudge array (copy)
            reference_result[10],     # r[10]: great lakes tuple (copy)
        )
    else:
        # Fallback: create empty placeholders if no reference provided
        # r[9]: nudge array - shape (n_gages, nts+1) 
        # For non-routing segments, there are no gages, so empty 2D array
        nudge_array = np.zeros((0, nts + 1), dtype='float32')
        
        run_result = (
            segment_ids,                                                    # r[0]: data_idx
            fvd_array,                                                      # r[1]: flowveldepth
            0,                                                              # r[2]: courant placeholder
            (                                                               # r[3]: lastobs tuple
                np.array([], dtype=np.intp),                                # gage segment ids
                np.array([], dtype='float32'),                              # lastobs_times
                np.array([], dtype='float32'),                              # lastobs_values
            ),
            (                                                               # r[4]: usgs tuple (5 elements)
                np.array([], dtype='int32'),                                # usgs_idx
                np.array([], dtype='float32'),                              # usgs_update_time
                np.array([], dtype='float32'),                              # usgs_prev_persisted_outflow
                np.array([], dtype='float32'),                              # usgs_prev_persistence_index
                np.array([], dtype='float32'),                              # usgs_persistence_update_time
            ),
            (                                                               # r[5]: usace tuple (5 elements)
                np.array([], dtype='int32'),
                np.array([], dtype='float32'),
                np.array([], dtype='float32'),
                np.array([], dtype='float32'),
                np.array([], dtype='float32'),
            ),
            (                                                               # r[6]: usbr tuple (5 elements)
                np.array([], dtype='int32'),
                np.array([], dtype='float32'),
                np.array([], dtype='float32'),
                np.array([], dtype='float32'),
                np.array([], dtype='float32'),
            ),
            upstream_array,                                                 # r[7]: upstream_array
            (                                                               # r[8]: rfc tuple (3 elements)
                np.array([], dtype='int32'),                                # rfc_idx
                np.array([], dtype='float32'),                              # rfc_update_time
                np.array([], dtype='int32'),                                # rfc_timeseries_idx
            ),
            nudge_array,                                                    # r[9]: nudge
            (                                                               # r[10]: great_lakes tuple (4 elements)
                np.array([], dtype='int32'),                                # gl_param_idx
                np.array([], dtype='float32'),                              # gl_prev_assim_outflow
                np.array([], dtype='int32'),                                # gl_prev_assim_timestamp
                np.array([], dtype='int32'),                                # gl_update_time
            ),
        )
    
    return [run_result]


def append_nonrouting_to_run_results(
    run_results: list,
    flow_scaling_df: pd.DataFrame,
    qts_subdivisions: int,
    nts: int,
) -> list:
    """
    Append non-routing segment results to the existing run_results list.
    
    This is the main entry point for integrating non-routing segment data
    with the routing results before passing to nwm_output_generator.
    
    Parameters
    ----------
    run_results : list
        List of run result tuples from routing computation.
    flow_scaling_df : pd.DataFrame
        Non-routing segment qlat values.
    qts_subdivisions : int
        Number of routing timesteps per forcing timestep.
    nts : int
        Total number of routing timesteps.
        
    Returns
    -------
    list
        Extended run_results list including non-routing segments.
        
    Usage
    -----
    In nhf_routing.py, after getting run_results from nwm_route:
    
    ```python
    from flow_scaling_utils import append_nonrouting_to_run_results
    
    # After routing computation
    run_results = run_results[0]
    
    # Append non-routing segments
    if hasattr(network, '_flow_scaling_segment_df') and not network._flow_scaling_segment_df.empty:
        run_results = append_nonrouting_to_run_results(
            run_results,
            network._flow_scaling_segment_df,
            qts_subdivisions,
            nts,
        )
    ```
    """
    if flow_scaling_df.empty:
        return run_results
    
    # Use the first routing result as a reference for structure compatibility
    reference_result = run_results[0] if run_results else None
    
    nonrouting_results = create_nonrouting_run_result(
        flow_scaling_df,
        qts_subdivisions,
        nts,
        reference_result=reference_result,
    )
    
    return run_results + nonrouting_results
