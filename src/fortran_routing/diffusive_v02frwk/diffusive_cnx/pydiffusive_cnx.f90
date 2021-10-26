module diffusive_interface

use, intrinsic :: iso_c_binding
use diffusive, only: diffnw

implicit none
contains
subroutine c_diffnw_cnx(timestep_ar_g, nts_ql_g, nts_ub_g, nts_db_g, ntss_ev_g, &
                    mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g, &
                    mann_ar_g, manncc_ar_g, so_ar_g, dx_ar_g, iniq, &
                    frnw_col, frnw_g, qlat_g, ubcd_g, dbcd_g, &
  		    paradim, para_ar_g, q_ev_g, elv_ev_g) bind(c)

    real(c_double), dimension(:), intent(in) :: timestep_ar_g(7)
    integer(c_int), intent(in) :: nts_ql_g, nts_ub_g, nts_db_g
    integer(c_int), intent(in) :: mxncomp_g, nrch_g
    real(c_double), dimension(mxncomp_g, nrch_g), intent(in) :: z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g
    real(c_double), dimension(mxncomp_g, nrch_g), intent(in) :: mann_ar_g, manncc_ar_g
    real(c_double), dimension(mxncomp_g, nrch_g), intent(inout) :: so_ar_g
    real(c_double), dimension(mxncomp_g, nrch_g), intent(in) :: dx_ar_g, iniq
    integer(c_int), intent(in) :: frnw_col
    real(c_double), dimension(nrch_g, frnw_col), intent(in) :: frnw_g
    real(c_double), dimension(nts_ql_g, mxncomp_g, nrch_g), intent(in) :: qlat_g
    real(c_double), dimension(nts_ub_g, nrch_g), intent(in) :: ubcd_g
    real(c_double), dimension(nts_db_g), intent(in) :: dbcd_g
    integer(c_int), intent(in) :: paradim	
    real(c_double), dimension(paradim), intent(in):: para_ar_g
    integer(c_int), intent(in) :: ntss_ev_g
    real(c_double), dimension(ntss_ev_g, mxncomp_g, nrch_g), intent(out) :: q_ev_g, elv_ev_g
    
    call diffnw(timestep_ar_g, nts_ql_g, nts_ub_g, nts_db_g, ntss_ev_g, &
                mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g, &
                mann_ar_g, manncc_ar_g, so_ar_g, dx_ar_g, iniq, &
                frnw_col, frnw_g, qlat_g, ubcd_g, dbcd_g, &
  		paradim, para_ar_g, q_ev_g, elv_ev_g)
    
end subroutine c_diffnw_cnx
end module diffusive_interface
