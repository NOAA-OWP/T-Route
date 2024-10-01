module precis
    implicit none

    !!NOTE: kind(1.0) and selected_real_kind(4) produce equivalent results
    integer, parameter :: prec = kind(1.0)
    !integer, parameter :: prec = selected_real_kind(4)

    !!NOTE: kind(1.d0) and selected_real_kind(8) produce equivalent results
    integer, parameter :: prec_db = kind(1.0d0)
    !integer, parameter :: prec = selected_real_kind(8)


    !integer, parameter :: prec = selected_real_kind(15)

end module precis