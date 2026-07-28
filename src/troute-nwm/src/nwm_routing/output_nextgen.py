import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple
import logging

LOG = logging.getLogger("")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column offsets within r[1] (flowpath and reservoir results)
_Q_COL = 0 # streamflow / reservoir outflow
_V_COL = 1 # velocity (flowpath) / unused (reservoir)
_D_COL = 2 # depth (flowpath) / water surface elevation (reservoir)

_VAR_ATTRS = {
    'streamflow': ('m3 s-1', 'Streamflow'),
    'velocity': ('m s-1', 'Flow velocity'),
    'depth': ('m', 'Flow depth'),
    'nudge': ('m3 s-1', 'Data assimilation nudge applied to streamflow'),
    'inflow': ('m3 s-1', 'Reservoir inflow'),
    'outflow': ('m3 s-1', 'Reservoir outflow'),
    'water_sfc_elev': ('m', 'Reservoir water surface elevation'),
    'reservoir_assimilated_value': ('m3 s-1', 'Reservoir data assimilation value'),
}

# ---------------------------------------------------------------------------
# OutputWriter
# ---------------------------------------------------------------------------
class OutputWriter:
    """
    Writes t-route model output to a single pre-allocated NetCDF file.
    
    Handles three feature types in the same file under separate dimensions:
      - feature_id  : individual reach outputs (streamflow, velocity, depth, nudge)
      - nexus_id    : nexus aggregation points (streamflow, nudge only — additive)
      - lake_id     : reservoir outputs (inflow, outflow, water_sfc_elev)
    """
    
    def __init__(self, cfg: Dict, output_cfg: Dict, all_network_ids: np.ndarray,
                 total_sim_seconds: float, start_time: datetime, dt: float,
                 rconn: Dict[int, List[int]], q0: pd.DataFrame,
                 waterbody_df: Optional[pd.DataFrame] = None,
                 waterbody_types_df: Optional[pd.DataFrame] = None,):
        
        self._output_cfg = output_cfg
        self._dt = float(dt)
        self._start_time = start_time
        self._output_interval = output_cfg['output_interval']
        self._total_steps = int(total_sim_seconds // self._output_interval) + 1
        self._stream_vars = output_cfg['variables']['stream']
        self._reservoir_vars = output_cfg['variables']['reservoir']
        
        # Reservoir domain (excluded from flowpaths)
        reservoir_ids: Set[int] = (set(waterbody_df.index.tolist()) if waterbody_df is not None else set())
        self._write_reservoirs = (
            bool(self._reservoir_vars)
            and waterbody_df is not None
            and not waterbody_df.empty
        )
        
        # Flowpath index
        self._wb_ids, self._wb_id_to_pos = self._build_flowpath_index(all_network_ids, reservoir_ids)
        
        # Nexus index
        self._nex_ids, self._nex_id_to_pos, self._nex_upstream, self._nex_upstream_ids = self._build_nexus_index(all_network_ids, rconn)
        
        # Reservoir index
        if self._write_reservoirs:
            self._lake_ids = np.sort(waterbody_df.index.to_numpy().astype(np.int64))
            self._lake_id_to_pos = {int(lid): i for i, lid in enumerate(self._lake_ids)}
            self._lake_ids_array = self._lake_ids
        else:
            self._lake_ids = np.array([], dtype=np.int64)
            self._lake_id_to_pos = {}
            self._lake_ids_array = np.array([], dtype=np.int64)
        
        # Initialize output arrays
        num_ids = len(self._wb_ids)
        self._buf_flow = np.full(num_ids, np.nan, dtype=np.float32)
        self._buf_vel = np.full(num_ids, np.nan, dtype=np.float32)
        self._buf_depth = np.full(num_ids, np.nan, dtype=np.float32)
        
        # Flag for nexus aggregation
        self._need_nex_flow = bool(self._nex_upstream_ids) and 'streamflow' in self._stream_vars
        
        # Open and pre-allocate NetCDF
        self._nc = self._create_netcdf(output_cfg['output_path'], start_time, cfg)
        
        # Write initial conditions to file
        self._write_initial_conditions(q0)
        
        LOG.info(
            "OutputWriter ready: %s  "
            "(%d flowpath, %d nexus, %d reservoir, %d timesteps)",
            output_cfg['output_path'], num_ids, len(self._nex_ids),
            len(self._lake_ids), self._total_steps,
        )
    
    # ------------------------------------------------------------------
    # Write initial conditions
    # ------------------------------------------------------------------
    def _write_initial_conditions(self, q0: pd.DataFrame):
        """Extracts t=0 state from q0 and writes to the first NetCDF index."""
        self._nc.variables['time'][0] = 0 # 0 minutes elapsed
        
        # Flowpaths
        if len(self._wb_ids):
            # Intersect q0 with valid flowpaths
            valid_fps = q0.index.intersection(self._wb_ids)
            q0_fp = q0.loc[valid_fps]
            
            # Map q0 columns to netcdf variables (qu0 -> streamflow, qd0 -> velocity, h0 -> depth)
            if 'streamflow' in self._stream_vars and 'qu0' in q0_fp.columns:
                flow_buf = np.full(len(self._wb_ids), np.nan, dtype=np.float32)
                pos = [self._wb_id_to_pos[fid] for fid in valid_fps]
                flow_buf[pos] = q0_fp['qu0'].values
                self._nc.variables['streamflow'][0, :] = flow_buf
            
            if 'velocity' in self._stream_vars and 'qd0' in q0_fp.columns:
                flow_buf = np.full(len(self._wb_ids), np.nan, dtype=np.float32)
                pos = [self._wb_id_to_pos[fid] for fid in valid_fps]
                flow_buf[pos] = q0_fp['qd0'].values
                self._nc.variables['velocity'][0, :] = flow_buf
                
            if 'depth' in self._stream_vars and 'h0' in q0_fp.columns:
                depth_buf = np.full(len(self._wb_ids), np.nan, dtype=np.float32)
                pos = [self._wb_id_to_pos[fid] for fid in valid_fps]
                depth_buf[pos] = q0_fp['h0'].values
                self._nc.variables['depth'][0, :] = depth_buf
    
    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------
    def _build_flowpath_index(self, all_network_ids: np.ndarray,
                               reservoir_ids: Set[int]
                               ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Build the sorted flowpath ID array and its position lookup dict.
        
        Reservoir segment IDs are excluded from the flowpath domain so they
        don't appear as regular reaches in the output.  If a subset_file is
        configured with a 'feature_ids' key, only those IDs are written;
        if none match the network the full domain is used as a fallback.
        """
        subset_ids = self._load_subset_ids('feature_ids')
        all_fp = set(all_network_ids.tolist()) - reservoir_ids
        
        if subset_ids is not None:
            valid = subset_ids & all_fp
            if not valid:
                LOG.warning("subset_file feature_ids had no matches; "
                            "writing all flowpath IDs")
                valid = all_fp
        else:
            valid = all_fp
            
        ids = np.sort(np.array(list(valid), dtype=np.int64))
        return ids, {int(fid): i for i, fid in enumerate(ids)}
    
    def _build_nexus_index(self, all_network_ids: np.ndarray,
                            rconn: Dict[int, List[int]]
                            ) -> Tuple[np.ndarray, Dict[int, int],
                                       Dict[int, List[int]], Set[int]]:
        """
        Build the nexus index from the 'nexuses' key in the subset_file.
        
        For each requested nexus ID, rconn is queried to find its immediate
        upstream flowpath IDs.
        
        Returns empty structures when no nexus subset is configured.
        """
        nex_ids = self._load_subset_ids('nexuses')
        if not nex_ids:
            empty: np.ndarray = np.array([], dtype=np.int64)
            return empty, {}, {}, frozenset()
        
        all_set = set(all_network_ids.tolist())
        nex_upstream = {}
        for nid in sorted(nex_ids):
            upstream = [u for u in rconn.get(nid, []) if u in all_set]
            if not upstream:
                LOG.warning("Nexus %d has no resolvable upstream reaches", nid)
            nex_upstream[nid] = upstream
            
        ids = np.sort(np.array(list(nex_ids), dtype=np.int64))
        id_to_pos = {int(nid): i for i, nid in enumerate(ids)}
        upstream_ids = set(chain.from_iterable(nex_upstream.values()))
        return ids, id_to_pos, nex_upstream, upstream_ids
    
    def _load_subset_ids(self, key: str) -> Optional[Set[int]]:
        """
        Read feature_ids or nexuses from the subset_file YAML.
        Returns None if no subset file is configured or the key is absent.
        TODO: Add reservoir filtering too?
        """
        subset_file = self._output_cfg.get('subset_file')
        if not subset_file:
            return None
        with open(subset_file) as fh:
            data = yaml.safe_load(fh)
        raw = data.get(key, []) if isinstance(data, dict) else []
        return _parse_int_ids(raw) if raw else None
    
    # ------------------------------------------------------------------
    # NetCDF creation
    # ------------------------------------------------------------------
    def _create_netcdf(self, output_path: str, start_time: datetime, full_config: Optional[Dict] = None) -> nc.Dataset:
        """
        Create and pre-allocate the output NetCDF file.
        
        All dimensions, coordinate variables, and data variables are created
        here before the simulation loop begins, so each write_step call only
        needs to fill pre-existing variable slices rather than modify file
        structure.  The three feature types share the same file but use
        separate dimensions (feature_id, nexus_id, lake_id).
        """
        ds = nc.Dataset(output_path, 'w', format='NETCDF4')
        
        # time dimension
        ds.createDimension('time', self._total_steps)
        
        # ID dimensions
        if len(self._wb_ids):
            ds.createDimension('feature_id', len(self._wb_ids))
        if len(self._nex_ids):
            ds.createDimension('nexus_id', len(self._nex_ids))
        if len(self._lake_ids):
            ds.createDimension('lake_id', len(self._lake_ids))
            
        # time
        vt = ds.createVariable('time', 'i4', ('time',), fill_value=-9999)
        vt.units = 'minutes since {}'.format(start_time.strftime('%Y-%m-%d %H:%M:%S'))
        vt.calendar = 'standard'
        vt.long_name = 'valid output time'
        
        # coordinate variables
        if len(self._wb_ids):
            v = ds.createVariable('feature_id', 'i4', ('feature_id',))
            v.long_name = 'Flowpath integer ID'
            v[:] = self._wb_ids.astype(np.int32)
            
        if len(self._nex_ids):
            v = ds.createVariable('nexus_id', 'i4', ('nexus_id',))
            v.long_name = 'Nexus integer ID'
            v[:] = self._nex_ids.astype(np.int32)
            
        if len(self._lake_ids):
            v = ds.createVariable('lake_id', 'i4', ('lake_id',))
            v.long_name = 'Reservoir (lake) integer ID'
            v[:] = self._lake_ids.astype(np.int32)
            
        # data variables
        comp = dict(zlib=True, complevel=2, fill_value=np.nan)
        
        if len(self._wb_ids):
            for var in self._stream_vars:
                units, long_name = _VAR_ATTRS.get(var, ('', var))
                v = ds.createVariable(var, 'f4', ('time', 'feature_id'),
                                      chunksizes=(1, len(self._wb_ids)), **comp)
                v.units = units
                v.long_name = long_name
                
        if len(self._nex_ids):
            for var in self._stream_vars:
                if var =='streamflow':
                    units, long_name = _VAR_ATTRS.get(var, ('', var))
                    v = ds.createVariable('nex_' + var, 'f4', ('time', 'nexus_id'),
                                          chunksizes=(1, len(self._nex_ids)), **comp)
                    v.units = units
                    v.long_name = long_name + ' (nexus aggregation)'
                    
        if len(self._lake_ids):
            for var in self._reservoir_vars:
                units, long_name = _VAR_ATTRS.get(var, ('', var))
                v = ds.createVariable(var, 'f4', ('time', 'lake_id'),
                                      chunksizes=(1, len(self._lake_ids)), **comp)
                v.units = units
                v.long_name = long_name
                
        ds.Conventions = 'CF-1.8'
        
        if full_config:
            ds.model_configuration = yaml.dump(full_config, default_flow_style=False)
        
        return ds

    # ------------------------------------------------------------------
    # Write data to file
    # ------------------------------------------------------------------
    def write_step(self, run_results: List[Tuple],
                   current_chunk_start_time: datetime):
        """
        Called once per routing chunk.  Identifies all output-aligned timesteps
        within this chunk, then loops run_results to extract data
        for all of them before writing.
        """
        if not run_results:
            return
        fvd_sample = next((r[1] for r in run_results if r[1] is not None), None)
        if fvd_sample is None:
            return

        nts_chunk = fvd_sample.shape[1] // 3

        # Find which internal step indices align with output_interval
        output_steps = []
        for i in range(1, nts_chunk + 1):
            step_time = current_chunk_start_time + timedelta(seconds=i * self._dt)
            elapsed   = (step_time - self._start_time).total_seconds()
            if abs(elapsed % self._output_interval) < 0.1:
                nc_idx = int(round(elapsed / self._output_interval))
                if 0 <= nc_idx < self._total_steps:
                    output_steps.append((i - 1, nc_idx, step_time))

        if not output_steps:
            return

        # Write time coordinate for each output step
        for step_index, nc_idx, step_time in output_steps:
            elapsed_min = int((step_time - self._start_time).total_seconds() / 60)
            self._nc.variables['time'][nc_idx] = elapsed_min

        # Single pass over run_results, extracting all output timesteps at once
        self._extract_and_write(run_results, output_steps)

    def _extract_and_write(self, run_results: List[Tuple],
                            output_steps: List[Tuple]):
        """
        Single pass over run_results. For each chunk, extracts all needed q/v/d columns at once, 
        then places them into per-timestep output buffers.
        """
        n_steps = len(output_steps)
        num_wb = len(self._wb_ids)
        wb_sorted = self._wb_ids

        # Pre-allocate per-timestep output buffers for all steps at once
        flow_out = np.full((n_steps, num_wb), np.nan, dtype=np.float32)
        vel_out = np.full((n_steps, num_wb), np.nan, dtype=np.float32)
        depth_out = np.full((n_steps, num_wb), np.nan, dtype=np.float32)

        # Reservoir and nudge remain per-step dicts
        nudge_per_step  = [{} for _ in range(n_steps)]
        res_out_per_step = res_elev_per_step = res_in_per_step = None
        if self._write_reservoirs:
            nlake = len(self._lake_ids)
            res_out_per_step = np.full((n_steps, nlake), np.nan, dtype=np.float32)
            res_elev_per_step = np.full((n_steps, nlake), np.nan, dtype=np.float32)
            res_in_per_step = np.full((n_steps, nlake), np.nan, dtype=np.float32)

        # Extract step_index -> column offset mapping
        step_q_cols = [s * 3 + _Q_COL for s, _, _ in output_steps]
        step_v_cols = [s * 3 + _V_COL for s, _, _ in output_steps]
        step_d_cols = [s * 3 + _D_COL for s, _, _ in output_steps]

        for r in run_results:
            seg_ids = r[0]
            fvd = r[1]
            if seg_ids is None or len(seg_ids) == 0 or fvd is None:
                continue
            
            # Split reservoir vs flowpath once for this chunk
            if self._write_reservoirs:
                res_mask = np.isin(seg_ids, self._lake_ids_array)
                fp_mask = ~res_mask
            else:
                res_mask = np.zeros(len(seg_ids), dtype=bool)
                fp_mask = np.ones(len(seg_ids), dtype=bool)
            
            # flowpaths
            if fp_mask.any():
                fp_ids = seg_ids[fp_mask]
                pos = np.searchsorted(wb_sorted, fp_ids)
                pos = np.clip(pos, 0, len(wb_sorted) - 1)
                in_wb = wb_sorted[pos] == fp_ids
                if in_wb.any():
                    out_pos = pos[in_wb]
                    # Extract all needed columns at once, then scatter
                    fvd_wb = fvd[fp_mask][in_wb]
                    for k, (qc, vc, dc) in enumerate(zip(step_q_cols, step_v_cols, step_d_cols)):
                        flow_out[k, out_pos] = fvd_wb[:, qc]
                        vel_out[k, out_pos] = fvd_wb[:, vc]
                        depth_out[k, out_pos] = fvd_wb[:, dc]
            
            # reservoirs
            if self._write_reservoirs and res_mask.any():
                res_ids = seg_ids[res_mask]
                res_pos = np.array([self._lake_id_to_pos.get(int(s), -1)
                                    for s in res_ids], dtype=np.intp)
                valid = res_pos >= 0
                if valid.any():
                    vpos = res_pos[valid]
                    fvd_res = fvd[res_mask][valid]
                    for k, (qc, dc) in enumerate(zip(step_q_cols, step_d_cols)):
                        res_out_per_step[k, vpos] = fvd_res[:, qc]
                        res_elev_per_step[k, vpos] = fvd_res[:, dc]

                # inflow from r[6]
                if r[6] is not None:
                    inflow_data = r[6]
                    inflow_res = inflow_data[res_mask][valid]
                    for k, (step_index, _, _) in enumerate(output_steps):
                        res_in_per_step[k, vpos] = inflow_res[:, step_index]
            
            # nudge
            if 'nudge' in self._stream_vars:
                for k, (step_index, _, _) in enumerate(output_steps):
                    _collect_nudge(r, step_index, nudge_per_step[k])
        
        # Write all output timesteps to NetCDF
        nc_vars = self._nc.variables
        for k, (_, nc_idx, _) in enumerate(output_steps):
            self._write_flowpath_frame(nc_vars, nc_idx,
                                       flow_out[k], vel_out[k], depth_out[k],
                                       nudge_per_step[k])
            if len(self._nex_ids):
                self._write_nexus_frame(nc_vars, nc_idx,
                                        flow_out[k])
            if self._write_reservoirs:
                self._write_reservoir_frame(nc_vars, nc_idx,
                                            res_out_per_step[k],
                                            res_elev_per_step[k],
                                            res_in_per_step[k])

    def _write_flowpath_frame(self, nc_vars, nc_idx,
                               flow, vel, depth, nudge_dict):
        """
        Write one timestep of flowpath output to the NetCDF file.
        
        flow, vel, and depth are pre-allocated numpy arrays already indexed
        to the flowpath output domain (length = n_feature_ids), filled by
        _extract_and_write. nudge is handled separately because it is sparse
        so it is passed as a dict and scattered into a NaN buffer here.
        """
        buf_map = {'streamflow': flow, 'velocity': vel, 'depth': depth}
        for var in self._stream_vars:
            if var in buf_map:
                nc_vars[var][nc_idx, :] = buf_map[var]
            elif var == 'nudge':
                buf = np.full(len(self._wb_ids), np.nan, dtype=np.float32)
                for sid, val in nudge_dict.items():
                    pos = self._wb_id_to_pos.get(sid)
                    if pos is not None:
                        buf[pos] = val
                nc_vars[var][nc_idx, :] = buf

    def _write_nexus_frame(self, nc_vars, nc_idx, flow_arr):
        """
        Write one timestep of nexus output to the NetCDF file. Only works
        for streamflow variable, which is a sum of upstream flows.
        """
        for var in self._stream_vars:
            if var != "streamflow":
                continue
            buf = np.full(len(self._nex_ids), np.nan, dtype=np.float32)
            for nid, pos in self._nex_id_to_pos.items():
                upstream = self._nex_upstream.get(nid, [])
                if not upstream:
                    continue
                vals = []
                for u in upstream:
                    wb_pos = self._wb_id_to_pos.get(u)
                    if wb_pos is not None:
                        v = flow_arr[wb_pos]
                        if not np.isnan(v):
                            vals.append(v)
                if vals:
                    buf[pos] = sum(vals)
                    
            nc_vars["nex_" + var][nc_idx, :] = buf

    def _write_reservoir_frame(self, nc_vars, nc_idx,
                                outflow, elev, inflow):
        """
        Write one timestep of reservoir output to the NetCDF file.
        
        outflow and elev come from r[1] columns 0 and 2 respectively,
        following the same layout as flowpath flow and depth.  inflow
        comes from r[6].  All three are pre-indexed numpy arrays of length
        n_lake_ids, filled by _extract_and_write.
        """
        # TODO: add reservoir assimilated values below...
        src_map = {'inflow': inflow, 'outflow': outflow, 'water_sfc_elev': elev, 'reservoir_assimilated_value': None}
        for var in self._reservoir_vars:
            arr = src_map.get(var)
            if arr is not None:
                nc_vars[var][nc_idx, :] = arr
            else:
                nc_vars[var][nc_idx, :] = np.full(
                    len(self._lake_ids), np.nan, dtype=np.float32)

    def close(self):
        if self._nc:
            self._nc.sync()
            self._nc.close()
            self._nc = None

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _parse_int_ids(raw: list) -> Set[int]:
    """
    Convert a raw yaml list to a set of integers. Strips prefix-dash strings
    like 'nex-12345' or 'wb-12345' by taking everything after the last dash.
    """
    out = set()
    for item in raw:
        if isinstance(item, int):
            out.add(item)
            continue
        s = str(item).strip()
        if '-' in s:
            s = s.rsplit('-', 1)[-1]
        try:
            out.add(int(s))
        except ValueError:
            LOG.warning("Could not parse ID %r as integer — skipping", item)
    return out

def _collect_nudge(r: Tuple, step_index: int, seg_nudge: Dict[int, float]) -> None:
    """
    Pull nudge values for one timestep from r[8] into seg_nudge.
    """
    nudge_arr = r[8]
    if not isinstance(nudge_arr, np.ndarray) or nudge_arr.shape[0]==0:
        return
    if step_index >= nudge_arr.shape[1]:
        return
    gauge_ids = r[3][0]
    col = nudge_arr[:, step_index]
    for gid, val in zip(gauge_ids, col):
        seg_nudge[int(gid)] = float(val)