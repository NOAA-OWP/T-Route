import numpy as np
from datetime import datetime, timedelta
import logging
from troute_ewts import MODULE_NAME
LOG = logging.getLogger(MODULE_NAME)

def great_lakes_da(
    gage_obs,
    gage_time,
    previous_assimilated_outflow,
    previous_assimilated_time,
    update_time,
    t0,
    now,
    climatology_outflows,
    update_time_interval = 3600,
    persistence_limit = 11,
):

    """
    Perform persistence reservoir data assimilation for the Great Lakes.
    
    Arguments
    ---------
    - lake_number                          (int): unique lake identification
                                                  number
    - gage_obs                (memoryview slice): array of gage observations for
                                                  at the gage associated with this
                                                  waterbody
    - gage_time               (memoryview slice): array of observation times
                                                  (secs) relative to the model
                                                  initialization time (t0).
    - t0                                   (str): Initialization time (t0).
    - now                                (float): Current time, seconds since in-
                                                  itialization time (t0).
    - previous_persisted_outflow (numpy.float32): Persisted outflow value from 
                                                  last timestep.                                
    - obs_lookback_hours                   (int): Maximum allowable lookback time
                                                  when searching for new outflow
                                                  observations
    - update_time                (numpy.float32): time to search for new observ-
                                                  ations, seconds since model 
                                                  initialization time (t0)
    - update_time_interval               (float): Time interval from current to
                                                  next update time (secs)
    
    Returns
    -------
    - outflow             (float): Persisted reservoir outflow rate (m3/sec)
    """

    # LOG.debug('Great Lakes data assimilation for lake_id: %s at time %s from run start' % (lake_number, now))

    # set new values as old values to start:
    new_assimilated_outflow = previous_assimilated_outflow
    new_assimilated_time = previous_assimilated_time
    new_update_time = update_time

    # determine which climatology value to use based on model time
    t0_datetime = datetime.strptime(t0, '%Y-%m-%d_%H:%M:%S')
    now_datetime = t0_datetime + timedelta(seconds=now)
    month_idx = now_datetime.month - 1 # subtract 1 for python indexing
    climatology_outflow = climatology_outflows[month_idx]

    if np.isnan(previous_assimilated_outflow):
        previous_assimilated_outflow = climatology_outflow

    if now >= update_time:
        # LOG.debug(
        #     'Looking for observation to assimilate...'
        # )

        # initialize variable to store assimilated observations. We initialize
        # as np.nan, so that if no good quality observations are found, we can
        # easily catch it.
        obs = np.nan

        # identify location of gage_time that is nearest to, but not greater 
        # than the update time
        t_idxs = np.nonzero(((now - gage_time) >= 0))[0]
        if len(t_idxs)>0:
            t_idx = t_idxs[-1]

            # record good observation to obs
            obs = gage_obs[t_idx]

            # determine how many seconds prior to the update_time the 
            # observation was taken
            t_obs = gage_time[t_idx]
            gage_lookback_seconds = now - t_obs
            
        if np.isnan(obs):
            '''
            - If no good observation is found, then we do not update the 
                update time. Consequently we will continue to search for a good
                observation at each model timestep, before updating the update
                time. 
            '''
            # LOG.debug(
            #     'No good observation found, persisting previously assimilated flow'
            # )
            outflow = previous_assimilated_outflow

        elif gage_lookback_seconds > (persistence_limit*60*60*24):
            '''
            If a good observation was found, but the time difference between
            the current model time and the observation timestamp is greater than
            the persistence limit, default to climatology.
            '''
            outflow = climatology_outflow

        else:
            '''
            A good observation is found and it is within the persistence limit.
            ''' 
            outflow = obs
            new_assimilated_outflow = obs
            new_assimilated_time = t_obs
            new_update_time = update_time + update_time_interval

    else:
        outflow = previous_assimilated_outflow

        if (now - previous_assimilated_time) > (persistence_limit*60*60*24):
            '''
            If the time difference between the current model time and the 
            observation timestamp is greater than
            the persistence limit, default to climatology.
            '''
            outflow = climatology_outflow
    
    return outflow, new_assimilated_outflow, new_assimilated_time, new_update_time
