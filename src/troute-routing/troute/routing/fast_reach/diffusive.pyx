import cython
import numpy as np
cimport numpy as np

from .fortran_wrappers cimport c_diffnw

@cython.boundscheck(False)
cdef void diffnw(
        float[::1] timestep_ar_g,
        int nts_ql_g,
        int nts_ub_g,
        int nts_db_g,
        int nts_ev_g,
        int nts_qtrib_g,
        int nts_da_g,
        int mxncomp_g,
        int nrch_g,
        float[::1,:] z_ar_g,
        float[::1,:] bo_ar_g,
        float[::1,:] traps_ar_g,
        float[::1,:] tw_ar_g,
        float[::1,:] twcc_ar_g,
        float[::1,:] mann_ar_g,
        float[::1,:] manncc_ar_g,
        float[::1,:] so_ar_g,
        float[::1,:] dx_ar_g,
        float[::1,:] iniq,
        int frnw_col,
        int[::1,:] frnw_g,
        float[::1,:,:] qlat_g,
        float[::1,:] ubcd_g,
        float[::1] dbcd_g,
        float[::1,:] qtrib_g,
        int paradim,
        float[::1] para_ar_g,
        int mxnbathy_g,
        float[::1,:,:] x_bathy_g,
        float[::1,:,:] z_bathy_g,
        float[::1,:,:] mann_bathy_g,
        int[::1,:] size_bathy_g,  
        float[::1,:] usgs_da_g,
        int[::1] usgs_da_reach_g,
        float[::1,:] rdx_ar_g,
        int cwnrow_g,
        int cwncol_g,
        float[::1,:] crosswalk_g,
        float[::1,:] z_thalweg_g,
        float[:,:,:] out_q,
        float[:,:,:] out_elv,
        float[:,:,:] out_depth,
):

    cdef:
        float[::1,:,:] q_ev_g     = np.empty([nts_ev_g,mxncomp_g,nrch_g], dtype = np.float32, order = 'F')
        float[::1,:,:] elv_ev_g   = np.empty([nts_ev_g,mxncomp_g,nrch_g], dtype = np.float32, order = 'F')
        float[::1,:,:] depth_ev_g = np.empty([nts_ev_g,mxncomp_g,nrch_g], dtype = np.float32, order = 'F')
    
    c_diffnw(
        &timestep_ar_g[0],
        &nts_ql_g,
        &nts_ub_g,
        &nts_db_g,
        &nts_ev_g,
        &nts_qtrib_g,
        &nts_da_g,
        &mxncomp_g,
        &nrch_g,
        &z_ar_g[0,0],
        &bo_ar_g[0,0],
        &traps_ar_g[0,0],
        &tw_ar_g[0,0],
        &twcc_ar_g[0,0],
        &mann_ar_g[0,0],
        &manncc_ar_g[0,0],
        &so_ar_g[0,0],
        &dx_ar_g[0,0],
        &iniq[0,0],
        &frnw_col,
        &frnw_g[0,0],
        &qlat_g[0,0,0],
        &ubcd_g[0,0],
        &dbcd_g[0],
        &qtrib_g[0,0],
        &paradim,
        &para_ar_g[0],
        &mxnbathy_g,
        &x_bathy_g[0,0,0],
        &z_bathy_g[0,0,0],
        &mann_bathy_g[0,0,0],
        &size_bathy_g[0,0],        
        &usgs_da_g[0,0],
        &usgs_da_reach_g[0], 
        &rdx_ar_g[0,0],
        &cwnrow_g,
        &cwncol_g,
        &crosswalk_g[0,0],  
        &z_thalweg_g[0,0],
        &q_ev_g[0,0,0],
        &elv_ev_g[0,0,0],
        &depth_ev_g[0,0,0]
    )
    
    # copy data from Fortran to Python memory view
    out_q[:,:,:]     = q_ev_g[::1,:,:]
    out_elv[:,:,:]   = elv_ev_g[::1,:,:]
    out_depth[:,:,:] = depth_ev_g[::1,:,:]


cpdef object compute_diffusive(
    dict diff_inputs
    ):

    # unpack/declare diffusive input variables
    cdef:
        float[::1] timestep_ar_g = np.asfortranarray(diff_inputs['timestep_ar_g'].astype(np.float32))
        int nts_ql_g = diff_inputs["nts_ql_g"]
        int nts_ub_g = diff_inputs["nts_ub_g"]
        int nts_db_g = diff_inputs["nts_db_g"]
        int nts_ev_g = diff_inputs["nts_ev_g"] 
        int nts_qtrib_g = diff_inputs['nts_qtrib_g']
        int nts_da_g = diff_inputs["nts_da_g"]       
        int mxncomp_g = diff_inputs["mxncomp_g"]
        int nrch_g = diff_inputs["nrch_g"]
        float[::1,:] z_ar_g = np.asfortranarray(diff_inputs["z_ar_g"].astype(np.float32))
        float[::1,:] bo_ar_g = np.asfortranarray(diff_inputs["bo_ar_g"].astype(np.float32))
        float[::1,:] traps_ar_g = np.asfortranarray(diff_inputs["traps_ar_g"].astype(np.float32))
        float[::1,:] tw_ar_g = np.asfortranarray(diff_inputs["tw_ar_g"].astype(np.float32))
        float[::1,:] twcc_ar_g = np.asfortranarray(diff_inputs["twcc_ar_g"].astype(np.float32))
        float[::1,:] mann_ar_g = np.asfortranarray(diff_inputs["mann_ar_g"].astype(np.float32))
        float[::1,:] manncc_ar_g = np.asfortranarray(diff_inputs["manncc_ar_g"].astype(np.float32))
        float[::1,:] so_ar_g = np.asfortranarray(diff_inputs["so_ar_g"].astype(np.float32))
        float[::1,:] dx_ar_g = np.asfortranarray(diff_inputs["dx_ar_g"].astype(np.float32))
        float[::1,:] iniq = np.asfortranarray(diff_inputs["iniq"].astype(np.float32))
        int frnw_col = diff_inputs["frnw_col"]
        int[::1,:] frnw_g = np.asfortranarray(diff_inputs["frnw_g"])
        float[::1,:,:] qlat_g = np.asfortranarray(diff_inputs["qlat_g"].astype(np.float32))
        float[::1,:] ubcd_g = np.asfortranarray(diff_inputs["ubcd_g"].astype(np.float32))
        float[::1] dbcd_g = np.asfortranarray(diff_inputs["dbcd_g"].astype(np.float32))
        float[::1,:] qtrib_g = np.asfortranarray(diff_inputs["qtrib_g"].astype(np.float32))
        int paradim = diff_inputs['paradim']
        float[::1] para_ar_g = np.asfortranarray(diff_inputs["para_ar_g"].astype(np.float32))
        int mxnbathy_g = diff_inputs['mxnbathy_g']
        float[::1,:,:] x_bathy_g = np.asfortranarray(diff_inputs["x_bathy_g"].astype(np.float32))
        float[::1,:,:] z_bathy_g = np.asfortranarray(diff_inputs["z_bathy_g"].astype(np.float32))
        float[::1,:,:] mann_bathy_g = np.asfortranarray(diff_inputs["mann_bathy_g"].astype(np.float32))
        int[::1,:] size_bathy_g = np.asfortranarray(diff_inputs["size_bathy_g"])    
        float[::1,:] usgs_da_g = np.asfortranarray(diff_inputs["usgs_da_g"].astype(np.float32))   
        int[::1] usgs_da_reach_g = np.asfortranarray(diff_inputs["usgs_da_reach_g"]) 
        float[::1,:] rdx_ar_g = np.asfortranarray(diff_inputs["rdx_ar_g"].astype(np.float32))
        int cwnrow_g = diff_inputs["cwnrow_g"]
        int cwncol_g = diff_inputs["cwncol_g"]
        float[::1,:] crosswalk_g = np.asfortranarray(diff_inputs["crosswalk_g"].astype(np.float32)) 
        float[::1,:] z_thalweg_g = np.asfortranarray(diff_inputs["z_thalweg_g"].astype(np.float32))
        float[:,:,:] out_q = np.empty([nts_ev_g,mxncomp_g,nrch_g], dtype = np.float32)
        float[:,:,:] out_elv = np.empty([nts_ev_g,mxncomp_g,nrch_g], dtype = np.float32)
        float[:,:,:] out_depth = np.empty([nts_ev_g,mxncomp_g,nrch_g], dtype = np.float32)

    # call diffusive compute kernel
    diffnw(
        timestep_ar_g,
        nts_ql_g,
        nts_ub_g,
        nts_db_g,
        nts_ev_g,
        nts_qtrib_g,
        nts_da_g,
        mxncomp_g,
        nrch_g,
        z_ar_g,
        bo_ar_g,
        traps_ar_g,
        tw_ar_g,
        twcc_ar_g,
        mann_ar_g,
        manncc_ar_g,
        so_ar_g,
        dx_ar_g,
        iniq,
        frnw_col,
        frnw_g,
        qlat_g,
        ubcd_g,
        dbcd_g,
        qtrib_g,
        paradim,
        para_ar_g,
        mxnbathy_g,
        x_bathy_g,
        z_bathy_g,
        mann_bathy_g,
        size_bathy_g,
        usgs_da_g,
        usgs_da_reach_g,
        rdx_ar_g,
        cwnrow_g,
        cwncol_g,
        crosswalk_g,
        z_thalweg_g,
        out_q,
        out_elv,
        out_depth
    )
    return np.asarray(out_q), np.asarray(out_elv), np.asarray(out_depth)