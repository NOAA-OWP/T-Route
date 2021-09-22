from libc.math cimport exp, isnan, NAN
from libc.stdio cimport printf

cpdef float simple_da_with_decay_py(
    const float last_valid_obs,
    const float model_val,
    const float minutes_since_last_valid,
    const float decay_coeff,
):
    """
    pass-through for using pytest with `simple_da_with_decay`
    """
    return simple_da_with_decay(
        last_valid_obs,
        model_val,
        minutes_since_last_valid,
        decay_coeff,
    )


cdef (float, float, float, float) simple_da(
    const float timestep,
    const float routing_period,
    const float decay_coeff,
    const float gage_maxtimestep,
    const float target_val,
    const float model_val,
    float lastobs_time,
    float lastobs_val,
    bint da_check_gage = 0,
) nogil:
    """
    wrapper function to compute all DA elements
    """
    cdef float replacement_val, nudge_val, da_weighted_shift, da_decay_minutes,
    # cdef float lastobs_timestep, lastobs_value,

    # TODO: It is possible to remove the following branching logic if
    # we just loop over the timesteps during DA and post-DA, if that
    # is a major performance optimization. On the flip side, it would
    # probably introduce unwanted code complexity.
    if isnan(target_val):
        if da_check_gage:
            printf("THIS IS A NAN\t")
    if ((timestep <= gage_maxtimestep) and not (isnan(target_val))):
        if da_check_gage:
            printf("replace\t")
        replacement_val = target_val
        nudge_val = target_val - model_val
        # add/update lastobs_timestep
        lastobs_time = (timestep) * routing_period
        lastobs_val = target_val
    if ((isnan(target_val)) and (isnan(lastobs_val))):
        replacement_val = model_val
        nudge_val = 0.0
        lastobs_val = NAN
        lastobs_time = NAN
    else:
        if da_check_gage:
            printf("So we are fixing that...\t")
        if da_check_gage:
            printf("decay; target: %g\t", target_val)
        da_decay_minutes = ((timestep) * routing_period - lastobs_time) / 60 # seconds to minutes
        da_weighted_shift = obs_persist_shift(lastobs_val, model_val, da_decay_minutes, decay_coeff)
        nudge_val = da_weighted_shift
        # TODO: we need to export these values
        # replacement_val = simple_da_with_decay(lastobs_val, model_val, da_decay_minutes, decay_coeff)
        replacement_val = model_val + da_weighted_shift

        if da_check_gage:
            printf("a: %g\t", decay_coeff)
            printf("ts: %g\t", timestep)
            printf("dt: %g\t", routing_period)
            printf("min: %g\t", da_decay_minutes)
            printf("lo_t: %f\t", lastobs_time)
            printf("lov: %g\t", lastobs_val)
            printf("ndg: %g\t", da_weighted_shift)
            printf("orig: %g\t", model_val)
            printf("new: %g\t", replacement_val)
            printf("\n")

    return replacement_val, nudge_val, lastobs_time, lastobs_val,


cdef float simple_da_with_decay(
    const float last_valid_obs,
    const float model_val,
    const float minutes_since_last_valid,
    const float decay_coeff,
) nogil:
    """
    pass-through for computing final value instead of just 'nudge'.
    """
    return model_val + obs_persist_shift(
        last_valid_obs,
        model_val,
        minutes_since_last_valid,
        decay_coeff,
    )


cdef float obs_persist_shift(
    const float last_valid_obs,
    const float model_val,
    const float minutes_since_last_valid,
    const float decay_coeff,
) nogil:
    """
    Given a modeled value, last valid observation,
    time since that observation, and an exponential
    decay_coefficient, compute the 'nudge' value
    """

    cdef float da_weight, da_shift, da_weighted_shift
    da_weight = exp(minutes_since_last_valid/-decay_coeff)  # TODO: This could be pre-calculated knowing when obs finish relative to simulation time
    # TODO: we need to be able to export these values to compute the 'Nudge'
    # One possibility would be to return only the nudge from this function...
    da_shift = last_valid_obs - model_val
    da_weighted_shift = da_shift * da_weight
    # printf("t: %g\ta: %g\t%g %g\tlo: %g\torig: %g --> new:%g\n", minutes_since_last_valid, decay_coeff, da_shift, da_weighted_shift, last_valid_obs, model_val, model_val + da_weighted_shift)
    return da_weighted_shift
