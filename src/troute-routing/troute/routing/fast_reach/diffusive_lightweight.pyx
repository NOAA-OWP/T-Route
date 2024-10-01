import cython
import numpy as np
cimport numpy as np

from .fortran_wrappers cimport c_compute_diffusive_couplingtimestep

@cython.boundscheck(False)
cdef void diffusive_couplingtimestep(
                                    float[::1] timestep_ar_g,
                                    int nts_ql_g,
                                    int nts_db_g,
                                    int nts_qtrib_g,
                                    int nts_da_g,
                                    int nts_ev_g,
                                    int mxncomp_g,
                                    int nrch_g,
                                    float[::1,:] dx_ar_g,
                                    float[::1,:] iniq,
                                    float[::1,:] inidepth,
                                    float[::1,:] iniqpx,
                                    int frnw_col, 
                                    int[::1,:] frnw_ar_g, 
                                    float[::1,:,:] qlat_g, 
                                    float[::1] dbcd_g,      
                                    float[::1,:] qtrib_g, 
                                    int paradim, 
                                    float[::1] para_ar_g, 
                                    float[::1,:] usgs_da_g, 
                                    int[::1] usgs_da_reach_g,
                                    int nrow_chxsec_lookuptable, 
                                    float[::1,:,:,:] chxsec_lookuptable, 
                                    float[::1,:] z_adj, 
                                    float t_start, 
                                    float t_end,             
                                    float[:,:,:] out_q_next_out_time, 
                                    float[:,:,:] out_elv_next_out_time, 
                                    float[:,:,:] out_depth_next_out_time,
                                    float[:,:] out_qpx_next_out_time,
):

    cdef:
        float[::1,:,:] q_next_out_time     = np.empty([nts_ev_g, mxncomp_g, nrch_g], dtype = np.float32, order = 'F')
        float[::1,:,:] elv_next_out_time   = np.empty([nts_ev_g, mxncomp_g, nrch_g], dtype = np.float32, order = 'F')
        float[::1,:,:] depth_next_out_time = np.empty([nts_ev_g, mxncomp_g, nrch_g], dtype = np.float32, order = 'F')
        float[::1,:] qpx_next_out_time     = np.empty([mxncomp_g, nrch_g], dtype = np.float32, order = 'F')

    c_compute_diffusive_couplingtimestep(&timestep_ar_g[0], 
                                         &nts_ql_g, 
                                         &nts_db_g, 
                                         &nts_qtrib_g, 
                                         &nts_da_g,
                                         &nts_ev_g,
                                         &mxncomp_g, 
                                         &nrch_g, 
                                         &dx_ar_g[0,0], 
                                         &iniq[0,0], 
                                         &inidepth[0,0], 
                                         &iniqpx[0,0],
                                         &frnw_col, 
                                         &frnw_ar_g[0,0], 
                                         &qlat_g[0,0,0], 
                                         &dbcd_g[0],      
                                         &qtrib_g[0,0], 
                                         &paradim, 
                                         &para_ar_g[0], 
                                         &usgs_da_g[0,0], 
                                         &usgs_da_reach_g[0],
                                         &nrow_chxsec_lookuptable, 
                                         &chxsec_lookuptable[0,0,0,0], 
                                         &z_adj[0,0], 
                                         &t_start, 
                                         &t_end,             
                                         &q_next_out_time[0,0,0], 
                                         &elv_next_out_time[0,0,0], 
                                         &depth_next_out_time[0,0,0],
                                         &qpx_next_out_time[0,0],
                                         )

    # copy data from Fortran to Python memory view
    out_q_next_out_time[:,:,:]     = q_next_out_time[::1,:,:]
    out_elv_next_out_time[:,:,:]   = elv_next_out_time[::1,:,:]
    out_depth_next_out_time[:,:,:] = depth_next_out_time[::1,:,:]
    out_qpx_next_out_time[:,:]     = qpx_next_out_time[::1,:]

cpdef object compute_diffusive_couplingtimestep(
    dict diff_inputs,
    out_chxsec_lookuptable, 
    out_z_adj,
    couplingtime_start,
    couplingtime_end,
):

    # unpack/declare diffusive input variables
    cdef:
        float[::1] timestep_ar_g = np.asfortranarray(diff_inputs['timestep_ar_g'].astype(np.float32))
        int nts_ql_g = diff_inputs["nts_ql_g"]
        int nts_db_g = diff_inputs["nts_db_g"]
        int nts_qtrib_g = diff_inputs['nts_qtrib_g']
        int nts_da_g = diff_inputs["nts_da_g"]   
        int nts_ev_g = diff_inputs["nts_ev_g"]       
        int mxncomp_g = diff_inputs["mxncomp_g"]
        int nrch_g = diff_inputs["nrch_g"]
        float[::1,:] dx_ar_g = np.asfortranarray(diff_inputs["dx_ar_g"].astype(np.float32))
        float[::1,:] iniq = np.asfortranarray(diff_inputs["iniq"].astype(np.float32))
        float[::1,:] inidepth = np.asfortranarray(diff_inputs["inidepth"].astype(np.float32))
        float[::1,:] iniqpx = np.asfortranarray(diff_inputs["iniqpx"].astype(np.float32))
        int frnw_col = diff_inputs["frnw_col"]
        int[::1,:] frnw_ar_g = np.asfortranarray(diff_inputs["frnw_g"])
        float[::1,:,:] qlat_g = np.asfortranarray(diff_inputs["qlat_g"].astype(np.float32))
        float[::1] dbcd_g = np.asfortranarray(diff_inputs["dbcd_g"].astype(np.float32))
        float[::1,:] qtrib_g = np.asfortranarray(diff_inputs["qtrib_g"].astype(np.float32))
        int paradim = diff_inputs['paradim']
        float[::1] para_ar_g = np.asfortranarray(diff_inputs["para_ar_g"].astype(np.float32))
        float[::1,:] usgs_da_g = np.asfortranarray(diff_inputs["usgs_da_g"].astype(np.float32))   
        int[::1] usgs_da_reach_g = np.asfortranarray(diff_inputs["usgs_da_reach_g"]) 
        int nrow_chxsec_lookuptable = diff_inputs["nrow_chxsec_lookuptable"]        
        float[::1,:,:,:] chxsec_lookuptable = np.asfortranarray(out_chxsec_lookuptable) 
        float[::1,:] z_adj = np.asfortranarray(out_z_adj) 
        float t_start = float(couplingtime_start)
        float t_end = float(couplingtime_end)
        float[:,:,:] out_q_next_out_time = np.empty([nts_ev_g, mxncomp_g, nrch_g], dtype = np.float32)
        float[:,:,:] out_elv_next_out_time = np.empty([nts_ev_g, mxncomp_g, nrch_g], dtype = np.float32)
        float[:,:,:] out_depth_next_out_time = np.empty([nts_ev_g, mxncomp_g, nrch_g], dtype = np.float32)
        float[:,:] out_qpx_next_out_time = np.empty([mxncomp_g, nrch_g], dtype = np.float32)

    # call diffusive compute kernel
    diffusive_couplingtimestep(
                                timestep_ar_g,
                                nts_ql_g,
                                nts_db_g,
                                nts_qtrib_g,
                                nts_da_g,
                                nts_ev_g,
                                mxncomp_g,
                                nrch_g,
                                dx_ar_g,
                                iniq,
                                inidepth,
                                iniqpx,
                                frnw_col, 
                                frnw_ar_g, 
                                qlat_g, 
                                dbcd_g,      
                                qtrib_g, 
                                paradim, 
                                para_ar_g, 
                                usgs_da_g, 
                                usgs_da_reach_g,
                                nrow_chxsec_lookuptable, 
                                chxsec_lookuptable, 
                                z_adj, 
                                t_start, 
                                t_end,             
                                out_q_next_out_time, 
                                out_elv_next_out_time, 
                                out_depth_next_out_time,
                                out_qpx_next_out_time,
                                )

    return (
            np.asarray(out_q_next_out_time), 
            np.asarray(out_elv_next_out_time), 
            np.asarray(out_depth_next_out_time),
            np.asarray(out_qpx_next_out_time),
            )            