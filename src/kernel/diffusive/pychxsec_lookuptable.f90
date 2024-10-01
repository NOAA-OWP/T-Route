module lookuptable_interface

use, intrinsic :: iso_c_binding, only: c_float, c_float, c_int
use lookuptable, only: chxsec_lookuptable_calc

implicit none
contains

subroutine c_chxsec_lookuptable_calc(mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g,    &
                               mann_ar_g, manncc_ar_g, dx_ar_g, so_lowerlimit_g,  frnw_col, frnw_ar_g, &                   
                               mxnbathy_g, x_bathy_g, z_bathy_g, mann_bathy_g, size_bathy_g,           &
                               nrow_chxsec_lookuptable, chxsec_lookuptable, z_adj)  bind(c)    

    integer(c_int), intent(in) :: mxncomp_g
    integer(c_int), intent(in) :: nrch_g
    integer(c_int), intent(in) :: frnw_col
    integer(c_int), intent(in) :: mxnbathy_g
    integer(c_int), intent(in) :: nrow_chxsec_lookuptable
    real(c_float), intent(in) :: so_lowerlimit_g
    integer(c_int), dimension(nrch_g, frnw_col), intent(in) :: frnw_ar_g
    integer(c_int), dimension(mxncomp_g, nrch_g), intent(in) :: size_bathy_g
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: z_ar_g
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: bo_ar_g 
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: traps_ar_g
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: tw_ar_g 
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: twcc_ar_g
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: mann_ar_g 
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: manncc_ar_g
    real(c_float), dimension(mxncomp_g, nrch_g), intent(in) :: dx_ar_g
    real(c_float), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in)  :: x_bathy_g
    real(c_float), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in)  :: z_bathy_g
    real(c_float), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in)  :: mann_bathy_g  
    real(c_float), dimension(11, nrow_chxsec_lookuptable, mxncomp_g, nrch_g), intent(out) :: chxsec_lookuptable
    real(c_float), dimension(mxncomp_g, nrch_g), intent(out) :: z_adj   

    call chxsec_lookuptable_calc(mxncomp_g, nrch_g, z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g,   &
                               mann_ar_g, manncc_ar_g, dx_ar_g, so_lowerlimit_g,  frnw_col, frnw_ar_g, &                   
                               mxnbathy_g, x_bathy_g, z_bathy_g, mann_bathy_g, size_bathy_g,           &
                               nrow_chxsec_lookuptable, chxsec_lookuptable, z_adj)                                     
                    
end subroutine c_chxsec_lookuptable_calc
end module lookuptable_interface