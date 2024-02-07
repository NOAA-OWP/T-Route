cdef extern from "pyResLevelPool.h":
    void c_levelpool_physics(float *dt,
                              float *qi0,
                              float *qi1,
                              float *ql,
                              float *ar,
                              float *we,
                              float *maxh,
                              float *wc,
                              float *wl,
                              float *dl,
                              float *oe,
                              float *oc,
                              float *oa,
                              float *H0,
                              float *H1,
                              float *qo1) nogil;

cdef extern from "pyMCsingleSegStime_NoLoop.h":
    void c_muskingcungenwm(float *dt,
                                  float *qup,
                                  float *quc,
                                  float *qdp,
                                  float *ql,
                                  float *dx,
                                  float *bw,
                                  float *tw,
                                  float *twcc,
                                  float *n,
                                  float *ncc,
                                  float *cs,
                                  float *s0,
                                  float *velp,
                                  float *depthp,
                                  float *qdc,
                                  float *velc,
                                  float *depthc,
                                  float *ck,
                                  float *cn,
                                  float *X) nogil;
    
cdef extern from "pydiffusive.h":
    void c_diffnw(double *timestep_ar_g,
                     int *nts_ql_g,
                     int *nts_ub_g,
                     int *nts_db_g,
                     int *ntss_ev_g,
                     int *nts_qtrib_g,
                     int *nts_da_g,
                     int *mxncomp_g,
                     int *nrch_g,
                     double *z_ar_g,
                     double *bo_ar_g,
                     double *traps_ar_g,
                     double *tw_ar_g,
                     double *twcc_ar_g,
                     double *mann_ar_g,
                     double *manncc_ar_g,
                     double *so_ar_g,
                     double *dx_ar_g,
                     double *iniq,
                     int *frnw_col,
                     int *frnw_g,
                     double *qlat_g,
                     double *ubcd_g,
                     double *dbcd_g,
                     double *qtrib_g,
                     int *paradim,
                     double *para_ar_g,
                     int *mxnbathy_g,
                     double *x_bathy_g,
                     double *z_bathy_g,
                     double *mann_bathy_g,
                     int *size_bathy_g,
                     double *usgs_da_g,
                     int *usgs_da_reach_g,
                     double *rdx_ar_g,
                     int *cwnrow_g,
                     int *cwncol_g,
                     double *crosswalk_g, 
                     double *z_thalweg_g,
                     double *q_ev_g,
                     double *elv_ev_g,
                     double *depth_ev_g) nogil;
    
cdef extern from "pydiffusive_cnt.h":
    void c_diffnw_cnt(double *dtini_g,
                     double *t0_g,
                     double *tfin_g,
                     double *saveinterval_ev_g,
                     double *dt_ql_g,
                     double *dt_ub_g,
                     double *dt_db_g,
                     int *nts_ql_g,
                     int *nts_ub_g,
                     int *nts_db_g,
                     int *mxncomp_g,
                     int *nrch_g,
                     double *z_ar_g,
                     double *bo_ar_g,
                     double *traps_ar_g,
                     double *tw_ar_g,
                     double *twcc_ar_g,
                     double *mann_ar_g,
                     double *manncc_ar_g,
                     double *so_ar_g,
                     double *dx_ar_g,
                     double *iniq,
                     int *nhincr_m_g,
                     int *nhincr_f_g,
                     double *ufhlt_m_g,
                     double *ufqlt_m_g,
                     double *ufhlt_f_g,
                     double *ufqlt_f_g,
                     int *frnw_col,
                     double *dfrnw_g,
                     double *qlat_g,
                     double *ubcd_g,
                     double *dbcd_g,
                     double *cfl_g,
                     double *theta_g,
                     int *tzeq_flag_g,
                     int *y_opt_g,
                     double *so_llm_g,
                     int *ntss_ev_g,
                     double *q_ev_g,
                     double *elv_ev_g) nogil;

cdef extern from "pydiffusive_cnx.h":
    void c_diffnw_cnx(double *dtini_g,
                     double *t0_g,
                     double *tfin_g,
                     double *saveinterval_ev_g,
                     double *dt_ql_g,
                     double *dt_ub_g,
                     double *dt_db_g,
                     int *nts_ql_g,
                     int *nts_ub_g,
                     int *nts_db_g,
                     int *mxncomp_g,
                     int *nrch_g,
                     double *z_ar_g,
                     double *bo_ar_g,
                     double *traps_ar_g,
                     double *tw_ar_g,
                     double *twcc_ar_g,
                     double *mann_ar_g,
                     double *manncc_ar_g,
                     double *so_ar_g,
                     double *dx_ar_g,
                     double *iniq,
                     int *nhincr_m_g,
                     int *nhincr_f_g,
                     double *ufhlt_m_g,
                     double *ufqlt_m_g,
                     double *ufhlt_f_g,
                     double *ufqlt_f_g,
                     int *frnw_col,
                     double *dfrnw_g,
                     double *qlat_g,
                     double *ubcd_g,
                     double *dbcd_g,
                     double *cfl_g,
                     double *theta_g,
                     int *tzeq_flag_g,
                     int *y_opt_g,
                     double *so_llm_g,
                     int *ntss_ev_g,
                     double *q_ev_g,
                     double *elv_ev_g) nogil;

