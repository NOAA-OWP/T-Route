module diffusive_lightweight_interface

use, intrinsic :: iso_c_binding, only: c_double, c_float, c_int
use diffusive_lightweight, only: compute_diffusive_couplingtimestep

implicit none
contains
subroutine c_compute_diffusive_couplingtimestep(timestep_ar_g, nts_ql_g, nts_db_g, nts_qtrib_g, nts_da_g, nts_ev_g,       &
                                                mxncomp_g, nrch_g, dx_ar_g, iniq, inidepth, iniqpx, frnw_col, frnw_ar_g,  &
                                                qlat_g, dbcd_g, qtrib_g, paradim, para_ar_g, usgs_da_g, usgs_da_reach_g,  &
                                                nrow_chxsec_lookuptable, chxsec_lookuptable, z_adj, t_start, t_end,       & 
                                                q_next_out_time, elv_next_out_time, depth_next_out_time, qpx_next_out_time) bind(c)      

    integer(c_int), intent(in) :: mxncomp_g 
    integer(c_int), intent(in) :: nrch_g
    integer(c_int), intent(in) :: nts_ql_g
    integer(c_int), intent(in) :: nts_db_g
    integer(c_int), intent(in) :: nts_qtrib_g
    integer(c_int), intent(in) :: nts_da_g
    integer(c_int), intent(in) :: nts_ev_g    
    integer(c_int), intent(in) :: frnw_col
    integer(c_int), intent(in) :: paradim
    integer(c_int), intent(in) ::  nrow_chxsec_lookuptable
    integer(c_int), dimension(nrch_g),            intent(in) :: usgs_da_reach_g
    integer(c_int), dimension(nrch_g, frnw_col),  intent(in) :: frnw_ar_g
    real(c_float), intent(in) :: t_start                                       
    real(c_float), intent(in) :: t_end                                             
    real(c_float), dimension(paradim ), intent(in) :: para_ar_g
    real(c_float), dimension(:)       , intent(in) :: timestep_ar_g(10)
    real(c_float), dimension(nts_db_g), intent(in) :: dbcd_g
    real(c_float), dimension(mxncomp_g,   nrch_g),          intent(in) :: dx_ar_g  
    real(c_float), dimension(mxncomp_g,   nrch_g),          intent(in) :: iniq
    real(c_float), dimension(mxncomp_g,   nrch_g),          intent(in) :: inidepth
    real(c_float), dimension(mxncomp_g,   nrch_g),          intent(in) :: iniqpx
    real(c_float), dimension(nts_qtrib_g, nrch_g),          intent(in) :: qtrib_g 
    real(c_float), dimension(nts_da_g,    nrch_g),          intent(in) :: usgs_da_g
    real(c_float), dimension(nts_ql_g,  mxncomp_g, nrch_g), intent(in) :: qlat_g
    real(c_float), dimension(mxncomp_g, nrch_g),            intent(in) :: z_adj
    real(c_float), dimension(11, nrow_chxsec_lookuptable, mxncomp_g, nrch_g), intent(in) :: chxsec_lookuptable
    real(c_float), dimension(nts_ev_g, mxncomp_g, nrch_g), intent(out) :: q_next_out_time
    real(c_float), dimension(nts_ev_g, mxncomp_g, nrch_g), intent(out) :: elv_next_out_time
    real(c_float), dimension(nts_ev_g, mxncomp_g, nrch_g), intent(out) :: depth_next_out_time
    real(c_float), dimension(mxncomp_g, nrch_g),           intent(out) :: qpx_next_out_time
          
    call compute_diffusive_couplingtimestep(timestep_ar_g, nts_ql_g, nts_db_g, nts_qtrib_g, nts_da_g, nts_ev_g,       &
                                            mxncomp_g, nrch_g, dx_ar_g, iniq, inidepth, iniqpx, frnw_col, frnw_ar_g,  &
                                            qlat_g, dbcd_g, qtrib_g, paradim, para_ar_g, usgs_da_g, usgs_da_reach_g,  &
                                            nrow_chxsec_lookuptable, chxsec_lookuptable, z_adj, t_start, t_end,       & 
                                            q_next_out_time, elv_next_out_time, depth_next_out_time, qpx_next_out_time)                            
    
end subroutine c_compute_diffusive_couplingtimestep
end module diffusive_lightweight_interface