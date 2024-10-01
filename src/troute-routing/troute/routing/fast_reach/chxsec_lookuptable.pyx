import cython
import numpy as np
cimport numpy as np

from .fortran_wrappers cimport c_chxsec_lookuptable_calc

@cython.boundscheck(False)
cdef void chxsec_lookuptable(
                            int mxncomp_g,
                            int nrch_g,
                            float[::1,:] z_ar_g,
                            float[::1,:] bo_ar_g,
                            float[::1,:] traps_ar_g,
                            float[::1,:] tw_ar_g,
                            float[::1,:] twcc_ar_g,
                            float[::1,:] mann_ar_g,
                            float[::1,:] manncc_ar_g,
                            float[::1,:] dx_ar_g,
                            float so_lowerlimit_g,
                            int frnw_col,
                            int[::1,:] frnw_g,
                            int mxnbathy_g,
                            float[::1,:,:] x_bathy_g,
                            float[::1,:,:] z_bathy_g,
                            float[::1,:,:] mann_bathy_g,
                            int[::1,:] size_bathy_g, 
                            int nrow_chxsec_lookuptable,
                            float[:,:,:,:] out_chxsec_lookuptable,
                            float[:,:] out_z_adj,
):

    cdef:
        float[::1,:,:,:] xsec_tab = np.empty([11, nrow_chxsec_lookuptable, mxncomp_g, nrch_g], dtype = np.float32, order = 'F')
        float[::1,:]     z_adj    = np.empty([mxncomp_g, nrch_g], dtype = np.float32, order = 'F')
    
    c_chxsec_lookuptable_calc(
                            &mxncomp_g,
                            &nrch_g,
                            &z_ar_g[0,0],
                            &bo_ar_g[0,0],
                            &traps_ar_g[0,0],
                            &tw_ar_g[0,0],
                            &twcc_ar_g[0,0],
                            &mann_ar_g[0,0],
                            &manncc_ar_g[0,0],
                            &dx_ar_g[0,0],
                            &so_lowerlimit_g,
                            &frnw_col,
                            &frnw_g[0,0],
                            &mxnbathy_g,
                            &x_bathy_g[0,0,0],
                            &z_bathy_g[0,0,0],
                            &mann_bathy_g[0,0,0],
                            &size_bathy_g[0,0],        
                            &nrow_chxsec_lookuptable,
                            &xsec_tab[0,0,0,0],
                            &z_adj[0,0],
    )
    
    # copy data from Fortran to Python memory view
    out_chxsec_lookuptable[:,:,:,:] = xsec_tab[::1,:,:,:]
    out_z_adj[:,:]                  = z_adj[::1,:]

cpdef object compute_chxsec_lookuptable(
    dict diff_inputs
    ):

    # unpack/declare diffusive input variables
    cdef:
        int mxncomp_g = diff_inputs["mxncomp_g"]
        int nrch_g = diff_inputs["nrch_g"]
        float[::1,:] z_ar_g = np.asfortranarray(diff_inputs["z_ar_g"].astype(np.float32))
        float[::1,:] bo_ar_g = np.asfortranarray(diff_inputs["bo_ar_g"].astype(np.float32))
        float[::1,:] traps_ar_g = np.asfortranarray(diff_inputs["traps_ar_g"].astype(np.float32))
        float[::1,:] tw_ar_g = np.asfortranarray(diff_inputs["tw_ar_g"].astype(np.float32))
        float[::1,:] twcc_ar_g = np.asfortranarray(diff_inputs["twcc_ar_g"].astype(np.float32))
        float[::1,:] mann_ar_g = np.asfortranarray(diff_inputs["mann_ar_g"].astype(np.float32))
        float[::1,:] manncc_ar_g = np.asfortranarray(diff_inputs["manncc_ar_g"].astype(np.float32))
        float[::1,:] dx_ar_g = np.asfortranarray(diff_inputs["dx_ar_g"].astype(np.float32))
        float so_lowerlimit_g =  diff_inputs['para_ar_g'][8].astype(np.float32)  
        int frnw_col = diff_inputs["frnw_col"]
        int[::1,:] frnw_g = np.asfortranarray(diff_inputs["frnw_g"])
        int mxnbathy_g = diff_inputs['mxnbathy_g']
        float[::1,:,:] x_bathy_g = np.asfortranarray(diff_inputs["x_bathy_g"].astype(np.float32))
        float[::1,:,:] z_bathy_g = np.asfortranarray(diff_inputs["z_bathy_g"].astype(np.float32))
        float[::1,:,:] mann_bathy_g = np.asfortranarray(diff_inputs["mann_bathy_g"].astype(np.float32))
        int[::1,:] size_bathy_g = np.asfortranarray(diff_inputs["size_bathy_g"])             
        int nrow_chxsec_lookuptable = diff_inputs["nrow_chxsec_lookuptable"] 
        float[:,:,:,:] out_chxsec_lookuptable = np.empty([11, nrow_chxsec_lookuptable, mxncomp_g, nrch_g], dtype = np.float32)
        float[:,:]     out_z_adj = np.empty([mxncomp_g, nrch_g], dtype = np.float32)

    # call fortran channel cross-section look up table creation subroutine 
    chxsec_lookuptable(                      
                        mxncomp_g,
                        nrch_g,
                        z_ar_g,
                        bo_ar_g,
                        traps_ar_g,
                        tw_ar_g,
                        twcc_ar_g,
                        mann_ar_g,
                        manncc_ar_g,                        
                        dx_ar_g,
                        so_lowerlimit_g,
                        frnw_col,
                        frnw_g,
                        mxnbathy_g,
                        x_bathy_g,
                        z_bathy_g,
                        mann_bathy_g,
                        size_bathy_g,
                        nrow_chxsec_lookuptable,
                        out_chxsec_lookuptable,
                        out_z_adj,
    )
    return np.asarray(out_chxsec_lookuptable), np.asarray(out_z_adj)