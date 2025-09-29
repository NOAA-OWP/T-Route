# cython: language_level=3, boundscheck=True, wraparound=False, profile=True

import numpy as np
from itertools import chain
from operator import itemgetter
from array import array
from numpy cimport ndarray  # TODO: Do we need to import numpy and ndarray separately?
from libc.math cimport isnan, NAN
cimport numpy as np  # TODO: We are cimporting and importing numpy into the same symbol, 'np'. Problem?
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
#Note may get slightly better performance using cython mem module (pulls from python's heap)
#from cpython.mem cimport PyMem_Malloc, PyMem_Free
from troute.network.musking.mc_reach cimport MC_Segment, MC_Reach, _MC_Segment, get_mc_segment

from troute.network.reach cimport Reach, _Reach, compute_type
from troute.network.reservoirs.levelpool.levelpool cimport MC_Levelpool, run_lp_c, update_lp_c
from troute.network.reservoirs.rfc.rfc cimport MC_RFC, run_rfc_c
from troute.routing.fast_reach.reservoir_hybrid_da import reservoir_hybrid_da
from troute.routing.fast_reach.reservoir_RFC_da import reservoir_RFC_da
from troute.routing.fast_reach.reservoir_GL_da import great_lakes_da
from cython.parallel import prange

#import cProfile
#pr = cProfile.Profile()
#NJF For whatever reason, when cimporting muskingcunge from reach, the linker breaks in weird ways
#the mc_reach.so will have an undefined symbol _muskingcunge, and reach.so will have a ____pyx_f_5reach_muskingcunge
#if you cimport reach, then call explicitly reach.muskingcung, then mc_reach.so maps to the correct module symbol
#____pyx_f_5reach_muskingcunge
#from reach cimport muskingcunge, QVD
cimport troute.routing.fast_reach.reach as reach
from troute.routing.fast_reach.simple_da cimport obs_persist_shift, simple_da_with_decay, simple_da

@cython.boundscheck(False)
cpdef object binary_find(object arr, object els):
    """
    Find elements in els in arr.
    Args:
        arr: Array to search. Must be sorted
        els:
    Returns:
    """
    cdef long hi = len(arr)
    cdef object idxs = []

    cdef Py_ssize_t L, R, m
    cdef long cand, el
    for el in els:
        L = 0
        R = hi - 1
        m = 0
        while L <= R:
            m = (L + R) // 2
            cand = arr[m]
            if cand < el:
                L = m + 1
            elif cand > el:
                R = m - 1
            else:
                break
        if arr[m] == el:
            idxs.append(m)
        else:
            raise ValueError(f"element {el} not found in {np.asarray(arr)}")
    return idxs


@cython.boundscheck(False)
cdef void compute_reach_kernel(float qup, float quc, int nreach, const float[:,:] input_buf, float[:, :] output_buf, bint assume_short_ts, bint return_courant=False) nogil:
    """
    Kernel to compute reach.
    Input buffer is array matching following description:
    axis 0 is reach
    axis 1 is inputs in th following order:
        qlat, dt, dx, bw, tw, twcc, n, ncc, cs, s0, qdp, velp, depthp
        qup and quc are initial conditions.
    Output buffer matches the same dimsions as input buffer in axis 0
    Input is nxm (n reaches by m variables)
    Ouput is nx3 (n reaches by 3 return values)
        0: current flow, 1: current depth, 2: current velocity
    """
    cdef reach.QVD rv
    cdef reach.QVD *out = &rv

    cdef:
        float dt, qlat, dx, bw, tw, twcc, n, ncc, cs, s0, qdp, velp, depthp
        int i

    for i in range(nreach):
        qlat = input_buf[i, 0] # n x 1
        dt = input_buf[i, 1] # n x 1
        dx = input_buf[i, 2] # n x 1
        bw = input_buf[i, 3]
        tw = input_buf[i, 4]
        twcc =input_buf[i, 5]
        n = input_buf[i, 6]
        ncc = input_buf[i, 7]
        cs = input_buf[i, 8]
        s0 = input_buf[i, 9]
        qdp = input_buf[i, 10]
        velp = input_buf[i, 11]
        depthp = input_buf[i, 12]

        reach.muskingcunge(
                    dt,
                    qup,
                    quc,
                    qdp,
                    qlat,
                    dx,
                    bw,
                    tw,
                    twcc,
                    n,
                    ncc,
                    cs,
                    s0,
                    velp,
                    depthp,
                    out)

#        output_buf[i, 0] = quc = out.qdc # this will ignore short TS assumption at seg-to-set scale?
        output_buf[i, 0] = out.qdc
        output_buf[i, 1] = out.velc
        output_buf[i, 2] = out.depthc

        if return_courant:
            output_buf[i, 3] = out.cn
            output_buf[i, 4] = out.ck
            output_buf[i, 5] = out.X

        qup = qdp

        if assume_short_ts:
            quc = qup
        else:
            quc = out.qdc

cdef void fill_buffer_column(const Py_ssize_t[:] srows,
    const Py_ssize_t scol,
    const Py_ssize_t[:] drows,
    const Py_ssize_t dcol,
    const float[:, :] src, float[:, ::1] out) nogil except *:

    cdef Py_ssize_t i
    for i in range(srows.shape[0]):
        out[drows[i], dcol] = src[srows[i], scol]

cpdef object column_mapper(object src_cols):
    """Map source columns to columns expected by algorithm"""
    cdef object index = {}
    cdef object i_label
    for i_label in enumerate(src_cols):
        index[i_label[1]] = i_label[0]

    cdef object rv = []
    cdef object label
    #qlat, dt, dx, bw, tw, twcc, n, ncc, cs, s0, qdp, velp, depthp
    for label in ['dt', 'dx', 'bw', 'tw', 'twcc', 'n', 'ncc', 'cs', 's0']:
        rv.append(index[label])
    return rv

cpdef object compute_network_structured(
    int nsteps,
    float dt,
    int qts_subdivisions,
    list reaches_wTypes, # a list of tuples
    dict upstream_connections,
    const long[:] data_idx,
    object[:] data_cols,
    const float[:,:] data_values,
    const float[:,:] initial_conditions,
    const float[:,:] qlat_values,
    const float[:,:] eloss_values,
    float ssout,
    list lake_numbers_col,
    const double[:,:] wbody_cols,
    dict data_assimilation_parameters,
    const int[:,:] reservoir_types,
    bint reservoir_type_specified,
    str model_start_time,
    const float[:,:] usgs_values,
    const int[:] usgs_positions,
    const int[:] usgs_positions_reach,
    const int[:] usgs_positions_gage,
    const float[:] lastobs_values_init,
    const float[:] time_since_lastobs_init,
    const double da_decay_coefficient,
    const float[:,:] reservoir_usgs_obs,
    const int[:] reservoir_usgs_wbody_idx,
    const float[:] reservoir_usgs_time,
    const float[:] reservoir_usgs_update_time,
    const float[:] reservoir_usgs_prev_persisted_flow,
    const float[:] reservoir_usgs_persistence_update_time,
    const float[:] reservoir_usgs_persistence_index,
    const float[:,:] reservoir_usace_obs,
    const int[:] reservoir_usace_wbody_idx,
    const float[:] reservoir_usace_time,
    const float[:] reservoir_usace_update_time,
    const float[:] reservoir_usace_prev_persisted_flow,
    const float[:] reservoir_usace_persistence_update_time,
    const float[:] reservoir_usace_persistence_index,
    const float[:,:] reservoir_rfc_obs,
    const int[:] reservoir_rfc_wbody_idx,
    const int[:] reservoir_rfc_totalCounts,
    list reservoir_rfc_file,
    const int[:] reservoir_rfc_use_forecast,
    const int[:] reservoir_rfc_timeseries_idx,
    const float[:] reservoir_rfc_update_time,
    const int[:] reservoir_rfc_da_timestep,
    const int[:] reservoir_rfc_persist_days,
    const int[:] great_lakes_idx,
    const int[:] great_lakes_times,
    const float[:] great_lakes_discharge,
    const int[:] great_lakes_param_idx,
    const float[:] great_lakes_param_prev_assim_flow,
    const int[:] great_lakes_param_prev_assim_times,
    const int[:] great_lakes_param_update_times,
    const float[:,:] great_lakes_climatology,
    dict upstream_results={},
    bint assume_short_ts=False,
    bint return_courant=False,
    int da_check_gage = -1,
    bint from_files=True,
    ):
    
    """
    Compute network
    Args:
        nsteps (int): number of time steps
        reaches_wTypes (list): List of tuples: (reach, reach_type), where reach_type is 0 for Muskingum Cunge reach and 1 is a reservoir
        upstream_connections (dict): Network
        data_idx (ndarray): a 1D sorted index for data_values
        data_values (ndarray): a 2D array of data inputs (nodes x variables)
        qlats (ndarray): a 2D array of qlat values (nodes x nsteps). The index must be shared with data_values
        initial_conditions (ndarray): an n x 3 array of initial conditions. n = nodes, column 1 = qu0, column 2 = qd0, column 3 = h0
        assume_short_ts (bool): Assume short time steps (quc = qup)
    Notes:
        Array dimensions are checked as a precondition to this method.
        This version creates python objects for segments and reaches,
        but then uses only the C structures and access for efficiency
    """
    # Check shapes
    if qlat_values.shape[0] != data_idx.shape[0]:
        raise ValueError(f"Number of rows in Qlat is incorrect: expected ({data_idx.shape[0]}), got ({qlat_values.shape[0]})")
    
    if qlat_values.shape[1] < nsteps/qts_subdivisions:
        raise ValueError(f"Number of columns (timesteps) in Qlat is incorrect: expected at most ({data_idx.shape[0]}), got ({qlat_values.shape[1]}). The number of columns in Qlat must be equal to or less than the number of routing timesteps")
    
    if data_values.shape[0] != data_idx.shape[0] or data_values.shape[1] != data_cols.shape[0]:
        raise ValueError(f"data_values shape mismatch")
    #define and initialize the final output array, add one extra time step for initial conditions
    cdef int qvd_ts_w = 3  # There are 3 values per timestep (corresponding to 3 columns per timestep)
    cdef np.ndarray[float, ndim=3] flowveldepth_nd = np.zeros((data_idx.shape[0], nsteps+1, qvd_ts_w), dtype='float32')
    #Make ndarrays from the mem views for convience of indexing...may be a better method
    cdef np.ndarray[float, ndim=2] data_array = np.asarray(data_values)
    cdef np.ndarray[float, ndim=2] init_array = np.asarray(initial_conditions)
    cdef np.ndarray[float, ndim=2] qlat_array = np.asarray(qlat_values)
    cdef np.ndarray[float, ndim=2] eloss_array = np.asarray(eloss_values)
    cdef np.ndarray[double, ndim=2] wbody_parameters = np.asarray(wbody_cols)
    ###### Declare/type variables #####
    # Source columns
    cdef Py_ssize_t[:] scols = np.array(column_mapper(data_cols), dtype=np.intp)
    cdef Py_ssize_t max_buff_size = 0
    #lists to hold reach definitions, i.e. list of ids
    cdef list reach
    cdef list upstream_reach
    #lists to hold segment ids
    cdef list segment_ids
    cdef list upstream_ids
    #flow accumulation variables
    cdef float upstream_flows, previous_upstream_flows
    #starting timestep, shifted by 1 to account for initial conditions
    cdef int timestep = 1
    cdef float routing_period = dt  # TODO: harmonize the dt with the value from the param_df dt (see line 153) #cdef float routing_period = data_values[0][0]
    #buffers to pass to compute_reach_kernel
    cdef float[:,:] buf_view
    cdef float[:,:] out_buf
    cdef float[:] lateral_flows
    # list of reach objects to operate on
    cdef list reach_objects = []
    cdef list segment_objects

    cdef long sid
    cdef _MC_Segment segment
    #pr.enable()
    #Preprocess the raw reaches, creating MC_Reach/MC_Segments

    for reach, reach_type in reaches_wTypes:
        upstream_reach = upstream_connections.get(reach[0], ())
        upstream_ids = binary_find(data_idx, upstream_reach)
        #Check if reach_type is 1 for reservoir
        if (reach_type == 1):
            my_id = binary_find(data_idx, reach)
            wbody_index = binary_find(lake_numbers_col,reach)[0]
            #Reservoirs should be singleton list reaches, TODO enforce that here?

            # write initial reservoir flows to flowveldepth array
            flowveldepth_nd[my_id, 0, 0] = wbody_parameters[wbody_index, 9] # TODO ref dataframe column label list, rather than hard-coded number

            #Check if reservoir_type is not specified, then initialize default Level Pool reservoir
            if (not reservoir_type_specified):

                # Initialize levelpool reservoir object
                lp_obj =  MC_Levelpool(
                    my_id[0],                        # index position of waterbody reach  
                    lake_numbers_col[wbody_index],   # lake number 
                    array('l',upstream_ids),         # upstream segment IDs
                    wbody_parameters[wbody_index],   # water body parameters
                    reservoir_types[wbody_index][0], # waterbody type code
                )
                reach_objects.append(lp_obj)

            else:
                # Check whether to use the fortran or python RFC DA module:
                if from_files:
                    # If reservoir_type is 1, 2, or 3, then initialize Levelpool reservoir
                    # reservoir_type 1 is a straight levelpool reservoir.
                    # reservoir_types 2 and 3 are USGS and USACE Hybrid reservoirs, respectively.
                    if (reservoir_types[wbody_index][0] >= 1 and reservoir_types[wbody_index][0] <= 3):
                                            
                        # Initialize levelpool reservoir object
                        lp_obj =  MC_Levelpool(
                            my_id[0],                        # index position of waterbody reach  
                            lake_numbers_col[wbody_index],   # lake number 
                            array('l',upstream_ids),         # upstream segment IDs
                            wbody_parameters[wbody_index],   # water body parameters
                            reservoir_types[wbody_index][0], # waterbody type code
                        )
                        reach_objects.append(lp_obj)

                    #If reservoir_type is 4, then initialize RFC forecast reservoir
                    elif (reservoir_types[wbody_index][0] == 4 or reservoir_types[wbody_index][0] == 5):
                        
                        # Initialize rfc reservoir object
                        rfc_obj = MC_RFC(
                            my_id[0], 
                            lake_numbers_col[wbody_index],
                            array('l',upstream_ids),
                            wbody_parameters[wbody_index],
                            reservoir_types[wbody_index][0],
                            data_assimilation_parameters["reservoir_da"]["reservoir_parameter_file"],
                            model_start_time,
                            data_assimilation_parameters["reservoir_da"]["reservoir_rfc_da"]["reservoir_rfc_forecasts_time_series_path"],
                            data_assimilation_parameters["reservoir_da"]["reservoir_rfc_da"]["reservoir_rfc_forecasts_lookback_hours"],
                        )
                        reach_objects.append(rfc_obj)
                
                else:
                    # Initialize levelpool reservoir object
                    lp_obj =  MC_Levelpool(
                        my_id[0],                        # index position of waterbody reach  
                        lake_numbers_col[wbody_index],   # lake number 
                        array('l',upstream_ids),         # upstream segment IDs
                        wbody_parameters[wbody_index],   # water body parameters
                        reservoir_types[wbody_index][0], # waterbody type code
                    )
                    reach_objects.append(lp_obj)

        else:
            segment_ids = binary_find(data_idx, reach)
            #Set the initial condtions before running loop
            flowveldepth_nd[segment_ids, 0] = init_array[segment_ids]
            segment_objects = []
            #Find the max reach size, used to create buffer for compute_reach_kernel
            if len(segment_ids) > max_buff_size:
                max_buff_size=len(segment_ids)

            for sid in segment_ids:
                #Initialize parameters  from the data_array, and set the initial initial_conditions
                #These aren't actually used (the initial contions) in the kernel as they are extracted from the
                #flowdepthvel array, but they could be used I suppose.  Note that velp isn't used anywhere, so
                #it is inialized to 0.0
                segment_objects.append(
                MC_Segment(sid, *data_array[sid, scols], init_array[sid, 0], 0.0, init_array[sid, 2])
            )
            reach_objects.append(
                #tuple of MC_Reach and reach_type
                MC_Reach(segment_objects, array('l',upstream_ids))
                )

    # replace initial conditions with gage observations, wherever available
    cdef int gages_size = usgs_positions.shape[0]
    cdef int gage_maxtimestep = usgs_values.shape[1]
    cdef int gage_i, usgs_position_i
    cdef float a, da_decay_minutes, da_weighted_shift, replacement_val  # , original_val, lastobs_val,
    cdef float [:] lastobs_values, lastobs_times
    cdef (float, float, float, float) da_buf
    cdef int[:] reach_has_gage = np.full(len(reaches_wTypes), np.iinfo(np.int32).min, dtype="int32")
    cdef float[:,:] nudge = np.zeros((gages_size, nsteps + 1), dtype="float32")

    lastobs_times = np.full(gages_size, NAN, dtype="float32")
    lastobs_values = np.full(gages_size, NAN, dtype="float32")
    if gages_size:
        # if da_check_gage > 0:
        #     print(f"gage_i     usgs_positions[gage_i]  usgs_positions_reach[gage_i]  usgs_positions_gage[gage_i]   list(usgs_positions)")
        for gage_i in range(gages_size):
            lastobs_values[gage_i] = lastobs_values_init[gage_i]
            lastobs_times[gage_i] = time_since_lastobs_init[gage_i]
            reach_has_gage[usgs_positions_reach[gage_i]] = usgs_positions_gage[gage_i]
            # if da_check_gage > 0:
            #     print(f"{gage_i} {usgs_positions[gage_i]} {usgs_positions_reach[gage_i]} {usgs_positions_gage[gage_i]} {list(usgs_positions)}")

    if gages_size and gage_maxtimestep > 0:
        for gage_i in range(gages_size):
            usgs_position_i = usgs_positions[gage_i]
            # Handle the instance where there are no values, only gage positions
            # TODO: Compare performance with math.isnan (imported for nogil...)
            if not np.isnan(usgs_values[gage_i, 0]):
                flowveldepth_nd[usgs_position_i, 0, 0] = usgs_values[gage_i, 0]

    
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------

    # reservoir id index arrays
    cdef np.ndarray[int, ndim=1] usgs_idx  = np.asarray(reservoir_usgs_wbody_idx)
    cdef np.ndarray[int, ndim=1] usace_idx = np.asarray(reservoir_usace_wbody_idx)
    cdef np.ndarray[int, ndim=1] rfc_idx = np.asarray(reservoir_rfc_wbody_idx)
    
    # reservoir update time arrays
    cdef np.ndarray[float, ndim=1] usgs_update_time  = np.asarray(reservoir_usgs_update_time)
    cdef np.ndarray[float, ndim=1] usace_update_time = np.asarray(reservoir_usace_update_time)
    cdef np.ndarray[float, ndim=1] rfc_update_time = np.asarray(reservoir_rfc_update_time)
    
    # reservoir persisted outflow arrays
    cdef np.ndarray[float, ndim=1] usgs_prev_persisted_ouflow  = np.asarray(reservoir_usgs_prev_persisted_flow)
    cdef np.ndarray[float, ndim=1] usace_prev_persisted_ouflow = np.asarray(reservoir_usace_prev_persisted_flow)  
    
    # reservoir persistence index update time arrays
    cdef np.ndarray[float, ndim=1] usgs_persistence_update_time  = np.asarray(reservoir_usgs_persistence_update_time)
    cdef np.ndarray[float, ndim=1] usace_persistence_update_time = np.asarray(reservoir_usace_persistence_update_time)
    
    # reservoir persisted outflow period index
    cdef np.ndarray[float, ndim=1] usgs_prev_persistence_index  = np.asarray(reservoir_usgs_persistence_index)
    cdef np.ndarray[float, ndim=1] usace_prev_persistence_index = np.asarray(reservoir_usace_persistence_index)
    cdef np.ndarray[int, ndim=1] rfc_timeseries_idx             = np.asarray(reservoir_rfc_timeseries_idx)

    # great lakes arrays
    cdef np.ndarray[int, ndim=1] gl_idx = np.asarray(great_lakes_idx)
    cdef np.ndarray[float, ndim=1] gl_obs = np.asarray(great_lakes_discharge)
    cdef np.ndarray[int, ndim=1] gl_times = np.asarray(great_lakes_times)
    cdef np.ndarray[int, ndim=1] gl_param_idx = np.asarray(great_lakes_param_idx)
    cdef np.ndarray[int, ndim=1] gl_update_time = np.asarray(great_lakes_param_update_times)
    cdef np.ndarray[float, ndim=1] gl_prev_assim_ouflow = np.asarray(great_lakes_param_prev_assim_flow)
    cdef np.ndarray[int, ndim=1] gl_prev_assim_timestamp = np.asarray(great_lakes_param_prev_assim_times)
    cdef np.ndarray[float, ndim=2] gl_climatology = np.asarray(great_lakes_climatology) 


    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    
    cdef np.ndarray fill_index_mask = np.ones_like(data_idx, dtype=bool)
    cdef Py_ssize_t fill_index
    cdef long upstream_tw_id
    cdef dict tmp
    cdef int idx
    cdef float val

    for upstream_tw_id in upstream_results:
        tmp = upstream_results[upstream_tw_id]
        fill_index = tmp["position_index"]
        fill_index_mask[fill_index] = False
        for idx, val in enumerate(tmp["results"]):
            flowveldepth_nd[fill_index, (idx//qvd_ts_w) + 1, idx%qvd_ts_w] = val
            if data_idx[fill_index]  in lake_numbers_col:
                res_idx = binary_find(lake_numbers_col, [data_idx[fill_index]])
                flowveldepth_nd[fill_index, 0, 0] = wbody_parameters[res_idx, 9] # TODO ref dataframe column label
            else:
                flowveldepth_nd[fill_index, 0, 0] = init_array[fill_index, 0] # initial flow condition
                flowveldepth_nd[fill_index, 0, 2] = init_array[fill_index, 2] # initial depth condition

    #Init buffers
    lateral_flows = np.zeros( max_buff_size, dtype='float32' )
    buf_view = np.zeros( (max_buff_size, 13), dtype='float32')
    out_buf = np.full( (max_buff_size, 3), -1, dtype='float32')

    cdef int num_reaches = len(reach_objects)
    #Dynamically allocate a C array of reach structs
    cdef _Reach* reach_structs = <_Reach*>malloc(sizeof(_Reach)*num_reaches)
    #Populate the above array with the structs contained in each reach object
    for i in range(num_reaches):
        reach_structs[i] = (<Reach>reach_objects[i])._reach

    #reach iterator
    cdef _Reach* r
    #create a memory view of the ndarray
    cdef float[:,:,::1] flowveldepth = flowveldepth_nd
    cdef np.ndarray[float, ndim=3] upstream_array = np.empty((data_idx.shape[0], nsteps+1, 1), dtype='float32')
    cdef float reservoir_outflow, reservoir_water_elevation
    cdef int id = 0
    
    
    while timestep < nsteps+1:
        for i in range(num_reaches):
            r = &reach_structs[i]
            #Need to get quc and qup
            upstream_flows = 0.0
            previous_upstream_flows = 0.0

            for _i in range(r._num_upstream_ids):#Explicit loop reduces some overhead
                id = r._upstream_ids[_i]
                upstream_flows += flowveldepth[id, timestep, 0]
                previous_upstream_flows += flowveldepth[id, timestep-1, 0]

            if assume_short_ts:
                upstream_flows = previous_upstream_flows

            if r.type == compute_type.RESERVOIR_LP: 
                
                # Great Lake waterbody: doesn't actually route anything, default outflows
                # are from climatology.
                if r.reach.lp.wbody_type_code == 6:
                    # find index location of waterbody in great_lakes_df 
                    # and great_lakes_param_df
                    res_idx                    = np.where(gl_idx == r.reach.lp.lake_number)
                    wbody_gage_obs             = gl_obs[res_idx[0]]
                    wbody_gage_time            = gl_times[res_idx[0]]
                    res_param_idx              = np.where(gl_param_idx == r.reach.lp.lake_number)
                    param_prev_assim_flow      = gl_prev_assim_ouflow[res_param_idx[0][0]]
                    param_prev_assim_timestamp = gl_prev_assim_timestamp[res_param_idx[0][0]]
                    param_update_time          = gl_update_time[res_param_idx[0][0]]
                    climatology                = gl_climatology[res_param_idx[0][0],:]

                    (new_outflow,
                    new_assimilated_outflow, 
                    new_assimilated_timestamp, 
                    new_update_time
                    ) = great_lakes_da(
                        wbody_gage_obs,             # gage observations (cms)
                        wbody_gage_time,            # timestamps of gage observations (datetime str)
                        param_prev_assim_flow,      # last used observation (cms)
                        param_prev_assim_timestamp, # timestamp of last used observation (datetime str)
                        param_update_time,          # timestamp to determine whether to look for new observation (datetime str)
                        model_start_time,           # model initilaization time (datetime str)
                        dt * timestep,              # model time (sec)
                        climatology,                # climatology outflows (cms)
                    )

                    gl_update_time[res_param_idx[0][0]] = new_update_time
                    gl_prev_assim_ouflow[res_param_idx[0][0]] = new_assimilated_outflow
                    gl_prev_assim_timestamp[res_param_idx[0][0]] = new_assimilated_timestamp

                    # populate flowveldepth array with levelpool or hybrid DA results 
                    flowveldepth[r.id, timestep, 0] = new_outflow
                    flowveldepth[r.id, timestep, 1] = 0.0
                    flowveldepth[r.id, timestep, 2] = 0.0
                    upstream_array[r.id, timestep, 0] = upstream_flows

                else:
                    # water elevation before levelpool calculation
                    initial_water_elevation = r.reach.lp.water_elevation
                    
                    # levelpool reservoir storage/outflow calculation
                    run_lp_c(r, upstream_flows, 0.0, routing_period, &reservoir_outflow, &reservoir_water_elevation)
                    
                    # USGS reservoir hybrid DA inputs
                    if r.reach.lp.wbody_type_code == 2:
                        # find index location of waterbody in reservoir_usgs_obs 
                        # and reservoir_usgs_time
                        res_idx = np.where(usgs_idx == r.reach.lp.lake_number)
                        wbody_gage_obs          = reservoir_usgs_obs[res_idx[0][0],:]
                        wbody_gage_time         = reservoir_usgs_time
                        prev_persisted_outflow  = usgs_prev_persisted_ouflow[res_idx[0][0]]
                        persistence_update_time = usgs_persistence_update_time[res_idx[0][0]] 
                        persistence_index       = usgs_prev_persistence_index[res_idx[0][0]]
                        update_time             = usgs_update_time[res_idx[0][0]] 
                    
                    # USACE reservoir hybrid DA inputs
                    if r.reach.lp.wbody_type_code == 3:
                        # find index location of waterbody in reservoir_usgs_obs 
                        # and reservoir_usgs_time
                        res_idx = np.where(usace_idx == r.reach.lp.lake_number)
                        wbody_gage_obs          = reservoir_usace_obs[res_idx[0][0],:]
                        wbody_gage_time         = reservoir_usace_time
                        prev_persisted_outflow  = usace_prev_persisted_ouflow[res_idx[0][0]]
                        persistence_update_time = usace_persistence_update_time[res_idx[0][0]] 
                        persistence_index       = usace_prev_persistence_index[res_idx[0][0]]
                        update_time             = usace_update_time[res_idx[0][0]] 
                        
                    # Execute reservoir DA - both USGS(2) and USACE(3) types
                    if r.reach.lp.wbody_type_code == 2 or r.reach.lp.wbody_type_code == 3:
                        
                        #print('***********************************************************')
                        #print('calling reservoir DA code for lake_id:', r.reach.lp.lake_number) 
                        #print('before DA, simulated outflow = ', reservoir_outflow)
                        #print('before DA, simulated water elevation = ', r.reach.lp.water_elevation)
                        
                        (new_outflow,
                        new_persisted_outflow,
                        new_water_elevation, 
                        new_update_time, 
                        new_persistence_index, 
                        new_persistence_update_time
                        ) = reservoir_hybrid_da(
                            r.reach.lp.lake_number,       # lake identification number
                            wbody_gage_obs,               # gage observation values (cms)
                            wbody_gage_time,              # gage observation times (sec)
                            dt * timestep,                # model time (sec)
                            prev_persisted_outflow,       # previously persisted outflow (cms)
                            persistence_update_time,
                            persistence_index,            # number of sequentially persisted update cycles
                            reservoir_outflow,            # levelpool simulated outflow (cms)
                            upstream_flows,               # waterbody inflow (cms)
                            dt,                           # model timestep (sec)
                            r.reach.lp.area,              # waterbody surface area (km2)
                            r.reach.lp.max_depth,         # max waterbody depth (m)
                            r.reach.lp.orifice_elevation, # orifice elevation (m)
                            initial_water_elevation,      # water surface el., previous timestep (m)
                            48.0,                         # gage lookback hours (hrs)
                            update_time                   # waterbody update time (sec)
                        )
                        
                        #print('After DA, outflow = ', new_outflow)
                        #print('After DA, water elevation =', new_water_elevation)
                        
                        # update levelpool water elevation state
                        update_lp_c(r, new_water_elevation, &reservoir_water_elevation)
                        
                        # change reservoir_outflow
                        reservoir_outflow = new_outflow
                        
                        #print('confirming DA elevation replacement:', reservoir_water_elevation)
                        #print('===========================================================')
                        
                    # update USGS DA reservoir state arrays
                    if r.reach.lp.wbody_type_code == 2:
                        usgs_update_time[res_idx[0][0]]              = new_update_time
                        usgs_prev_persisted_ouflow[res_idx[0][0]]    = new_persisted_outflow
                        usgs_prev_persistence_index[res_idx[0][0]]   = new_persistence_index
                        usgs_persistence_update_time[res_idx[0][0]]  = new_persistence_update_time
                        
                    # update USACE DA reservoir state arrays
                    if r.reach.lp.wbody_type_code == 3:
                        usace_update_time[res_idx[0][0]]             = new_update_time
                        usace_prev_persisted_ouflow[res_idx[0][0]]   = new_persisted_outflow
                        usace_prev_persistence_index[res_idx[0][0]]  = new_persistence_index
                        usace_persistence_update_time[res_idx[0][0]] = new_persistence_update_time


                    # RFC reservoir hybrid DA inputs
                    if r.reach.lp.wbody_type_code == 4:
                        # find index location of waterbody in reservoir_rfc_obs 
                        # and reservoir_rfc_time
                        res_idx            = np.where(rfc_idx == r.reach.lp.lake_number)
                        wbody_gage_obs     = reservoir_rfc_obs[res_idx[0][0],:]
                        totalCounts        = reservoir_rfc_totalCounts[res_idx[0][0]]
                        rfc_file           = reservoir_rfc_file[res_idx[0][0]]
                        use_RFC            = reservoir_rfc_use_forecast[res_idx[0][0]]
                        current_timeseries_idx = rfc_timeseries_idx[res_idx[0][0]]
                        update_time        = rfc_update_time[res_idx[0][0]]
                        rfc_timestep       = reservoir_rfc_da_timestep[res_idx[0][0]]
                        rfc_persist_days   = reservoir_rfc_persist_days[res_idx[0][0]]

                    # Execute RFC reservoir DA - both RFC(4) and Glacially Dammed Lake(5) types
                    if r.reach.lp.wbody_type_code == 4 or r.reach.lp.wbody_type_code == 5:
                        
                        #print('***********************************************************')
                        #print('calling reservoir DA code for lake_id:', r.reach.lp.lake_number) 
                        #print('before DA, simulated outflow = ', reservoir_outflow)
                        #print('before DA, simulated water elevation = ', r.reach.lp.water_elevation)
                        
                        (
                            new_outflow, 
                            new_water_elevation, 
                            new_update_time,
                            new_timeseries_idx,
                            dynamic_reservoir_type, 
                            assimilated_value, 
                            assimilated_source_file,
                        ) = reservoir_RFC_da(
                            use_RFC,                            # boolean whether to use RFC values or not
                            wbody_gage_obs,              # gage observation values (cms)
                            current_timeseries_idx,                     # index of for current time series observation
                            totalCounts,                       # total number of observations in RFC timeseries
                            routing_period,                          # routing period (sec)
                            dt * timestep,                               # model time (sec)
                            update_time,                        # time to advance to next time series index
                            rfc_timestep,                       # frequency of DA observations (sec)
                            rfc_persist_days*24*60*60, # max seconds RFC forecasts will be used/persisted (days -> seconds)
                            r.reach.lp.wbody_type_code,                           # reservoir type
                            upstream_flows,                                   # waterbody inflow (cms)
                            initial_water_elevation,                  # water surface el., previous timestep (m)
                            reservoir_outflow,                  # levelpool simulated outflow (cms)
                            reservoir_water_elevation,                # levelpool simulated water elevation (m)
                            r.reach.lp.area*1.0e6,          # waterbody surface area (km2 -> m2)
                            r.reach.lp.max_depth,                # max waterbody depth (m)
                            rfc_file,                # RFC file name
                        )

                        #print('After DA, outflow = ', new_outflow)
                        #print('After DA, water elevation =', new_water_elevation)
                        
                        # update levelpool water elevation state
                        update_lp_c(r, new_water_elevation, &reservoir_water_elevation)
                        
                        # change reservoir_outflow
                        reservoir_outflow = new_outflow
                        
                        #print('confirming DA elevation replacement:', reservoir_water_elevation)
                        #print('===========================================================')
                        
                        # update RFC DA reservoir state arrays
                        rfc_update_time[res_idx[0][0]]    = new_update_time
                        rfc_timeseries_idx[res_idx[0][0]] = new_timeseries_idx
                        
                    
                    # populate flowveldepth array with levelpool or hybrid DA results 
                    flowveldepth[r.id, timestep, 0] = reservoir_outflow
                    flowveldepth[r.id, timestep, 1] = 0.0
                    flowveldepth[r.id, timestep, 2] = reservoir_water_elevation
                    upstream_array[r.id, timestep, 0] = upstream_flows

            elif r.type == compute_type.RESERVOIR_RFC:
                run_rfc_c(r, upstream_flows, 0.0, routing_period, &reservoir_outflow, &reservoir_water_elevation)
                flowveldepth[r.id, timestep, 0] = reservoir_outflow
                flowveldepth[r.id, timestep, 1] = 0.0
                flowveldepth[r.id, timestep, 2] = reservoir_water_elevation
                upstream_array[r.id, timestep, 0] = upstream_flows
            
            else:
                #Create compute reach kernel input buffer
                for _i in range(r.reach.mc_reach.num_segments):
                    segment = get_mc_segment(r, _i)#r._segments[_i]
                    buf_view[_i, 0] = qlat_array[ segment.id, <int>((timestep-1)/qts_subdivisions)]
                    buf_view[_i, 1] = segment.dt
                    buf_view[_i, 2] = segment.dx
                    buf_view[_i, 3] = segment.bw
                    buf_view[_i, 4] = segment.tw
                    buf_view[_i, 5] = segment.twcc
                    buf_view[_i, 6] = segment.n
                    buf_view[_i, 7] = segment.ncc
                    buf_view[_i, 8] = segment.cs
                    buf_view[_i, 9] = segment.s0
                    buf_view[_i, 10] = flowveldepth[segment.id, timestep-1, 0]
                    buf_view[_i, 11] = 0.0 #flowveldepth[segment.id, timestep-1, 1]
                    buf_view[_i, 12] = flowveldepth[segment.id, timestep-1, 2]

                compute_reach_kernel(previous_upstream_flows, upstream_flows,
                                     r.reach.mc_reach.num_segments, buf_view,
                                     out_buf,
                                     assume_short_ts)

                #Copy the output out
                for _i in range(r.reach.mc_reach.num_segments):
                    segment = get_mc_segment(r, _i)

                    # Setting flow based on the output of MC - SSOUT - ELOSS
                    flowveldepth[segment.id, timestep, 0] = out_buf[_i, 0] - ssout - eloss_array[segment.id, <int>((timestep-1)/qts_subdivisions)]

                    if reach_has_gage[i] == da_check_gage:
                        printf("segment.id: %ld\t", segment.id)
                        printf("segment.id: %d\t", usgs_positions[reach_has_gage[i]])
                    flowveldepth[segment.id, timestep, 1] = out_buf[_i, 1]
                    flowveldepth[segment.id, timestep, 2] = out_buf[_i, 2]

                    # Ensuring flow does not go negative
                    if flowveldepth[segment.id, timestep, 0] <= 0.0:
                        flowveldepth[segment.id, timestep, 0] = 0.0

            # For each reach,
            # at the end of flow calculation, Check if there is something to assimilate
            # by evaluating whether the reach_has_gage array has a value different from
            # the initialized value, np.iinfo(np.int32).min (the minimum possible integer).

            # TODO: If it were possible to invert the time and reach loops
            # (should be possible for the MC), then this check could be
            # performed fewer times -- consider implementing such a change.

            if reach_has_gage[i] > -1:
            # We only enter this process for reaches where the
            # gage actually exists.
            # If assimilation is active for this reach, we touch the
            # exactly one gage which is relevant for the reach ...
                gage_i = reach_has_gage[i]
                usgs_position_i = usgs_positions[gage_i]
                da_buf = simple_da(
                    timestep,
                    routing_period,
                    da_decay_coefficient,
                    gage_maxtimestep,
                    NAN if timestep >= gage_maxtimestep else usgs_values[gage_i,timestep],
                    flowveldepth[usgs_position_i, timestep, 0],
                    lastobs_times[gage_i],
                    lastobs_values[gage_i],
                    gage_i == da_check_gage,
                )
                if gage_i == da_check_gage:
                    printf("ts: %d\t", timestep)
                    printf("gmxt: %d\t", gage_maxtimestep)
                    printf("gage: %d\t", gage_i)
                    printf("old: %g\t", flowveldepth[usgs_position_i, timestep, 0])
                    printf("exp_gage_val: %g\t", 
                    NAN if timestep >= gage_maxtimestep else usgs_values[gage_i,timestep],)

                flowveldepth[usgs_position_i, timestep, 0] = da_buf[0]

                if gage_i == da_check_gage:
                    printf("new: %g\t", flowveldepth[usgs_position_i, timestep, 0])
                    printf("repl: %g\t", da_buf[0])
                    printf("nudg: %g\n", da_buf[1])

                nudge[gage_i, timestep] = da_buf[1]
                lastobs_times[gage_i] = da_buf[2]
                lastobs_values[gage_i] = da_buf[3]

        # TODO: Address remaining TODOs (feels existential...), Extra commented material, etc.

        timestep += 1

    #pr.disable()
    #pr.print_stats(sort='time')
    #IMPORTANT, free the dynamic array created
    free(reach_structs)
    #slice off the initial condition timestep and return
    output = np.asarray(flowveldepth[:,1:,:], dtype='float32')
    #do the same for the upstream_array
    output_upstream = np.asarray(upstream_array[:,1:,:], dtype='float32')
    #return np.asarray(data_idx, dtype=np.intp), np.asarray(flowveldepth.base.reshape(flowveldepth.shape[0], -1), dtype='float32')
    return (
        np.asarray(data_idx, dtype=np.intp)[fill_index_mask], 
        output.reshape(output.shape[0], -1)[fill_index_mask], 
        0, 
        (
            np.asarray([data_idx[usgs_position_i] for usgs_position_i in usgs_positions]), 
            np.asarray(lastobs_times), 
            np.asarray(lastobs_values)
        ), 
        (
            usgs_idx, 
            usgs_update_time-((timestep-1)*dt), 
            usgs_prev_persisted_ouflow, 
            usgs_prev_persistence_index, 
            usgs_persistence_update_time-((timestep-1)*dt)
        ), 
        (
            usace_idx, usace_update_time-((timestep-1)*dt), 
            usace_prev_persisted_ouflow, 
            usace_prev_persistence_index, 
            usace_persistence_update_time-((timestep-1)*dt)
        ), 
        output_upstream.reshape(output.shape[0], -1)[fill_index_mask], 
        (
            rfc_idx, rfc_update_time-((timestep-1)*dt), 
            rfc_timeseries_idx
        ),
        np.asarray(nudge),
        (
            gl_param_idx,
            gl_prev_assim_ouflow,
            gl_prev_assim_timestamp,
            gl_update_time
        )
    )
