import numpy as np
import logging
LOG = logging.getLogger("TROUTE")

def _modify_for_projected_storage(
    inflow, 
    outflow_assess,
    initial_storage,
    max_storage,
    lake_number,
    current_time,
    routing_period,
    min_storage = 0,
):
    '''
    
    '''
    
    # initialize return values
    outflow = outflow_assess
    max_storage_reached = False
    
    if outflow_assess < 0:
        LOG.warning('WARNING: Calculations return a negative outflow for reservoir %s', lake_number)
        LOG.warning('at %s seconds after model start time.', current_time)
        outflow = 0
        
    projected_storage = initial_storage + (inflow - outflow_assess) * routing_period
    if projected_storage > max_storage:
        
        max_storage_reached = True
        LOG.warning('WARNING: Modified release to prevent maximum storage exceedance for reservoir %s', lake_number)
        LOG.warning('at %s seconds after model start time.', current_time)
        LOG.warning('simulated storage would be %s m3', projected_storage)
        LOG.warning('maximum waterbody storage is %s m3', max_storage)
        
        
    if projected_storage < min_storage and projected_storage > 0:
        
        outflow = (projected_storage - min_storage) / routing_period + outflow_assess
        LOG.warning('WARNING: Modified release to maintain minimum storage for reservoir %s', lake_number)
        LOG.warning('at %s seconds after model start time.', current_time)
        
    if projected_storage <= 0:
        
        outflow = inflow
        LOG.warning('WARNING: Modified release to prevent storage deficit for reservoir %s', lake_number)
        LOG.warning('at %s seconds after model start time.', current_time)
        
    if outflow < 0: outflow = 0
        
    
    return outflow, max_storage_reached

def reservoir_hybrid_da(
    lake_number,
    gage_obs,
    gage_time,
    now,
    previous_persisted_outflow,
    persistence_update_time,
    persistence_index,
    levelpool_outflow,
    inflow,
    routing_period,
    lake_area,
    max_depth,
    orifice_elevation,
    initial_water_elevation,
    obs_lookback_hours,
    update_time,
    update_time_interval = 3600,
    persistence_update_time_interval = 86400,
):

    """
    Perform hybrid reservoir data assimilation.
    
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
                                                  
    - now                                (float): Current time, seconds since in-
                                                  itialization time (t0).
                                                  
    - previous_persisted_outflow (numpy.float32): Persisted outflow value from 
                                                  last timestep. 
                                                  
    - levelpool_outflow                  (float): Reservoir outflow rate (m3/s)
                                                  calculated by levelpool module
                                                  
    - inflow                             (float): Reservoir inflow from upstream
                                                  stream reaches (m3/s)
                                                  
    - routing_period                     (float): Model timestep (sec) used in
                                                  the reservoir and stream routing
                                                  modules
                                                  
    - lake_area                          (float): waterbody surface area (km2)
    
    - max_depth                          (float): maximum waterbody depth (m)
    
    - orifice_elevation                  (float): waterbody orifice elevation
                                                  (m above datum)                                   
    - initial_water_elevation            (float): water surface elevetion from 
                                                  previous timestep
                                                  
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
    
    - new_water_elevation (float): modified reservoir water surface 
                                   elevation (meters above datum).
    """

    LOG.debug('Hybrid data assimilation for lake_id: %s at time %s from run start' % (lake_number, now))
    
    # set persistence limit at persistence_update_time cycles
    persistence_limit = 11
    
    # initialize new_persistence_index as persistence_index and 
    # new_persistence_update_time as persistence_update_time. 
    new_persistence_index = persistence_index
    new_persistence_update_time = persistence_update_time
    
    # initialize new_update_time as update time. If the update time needs to be
    # updated, then this variable will be reset later.
    new_update_time = update_time
    
    # initial waterbody storage (from the previous timestep) - m3
    initial_storage = (initial_water_elevation - orifice_elevation) \
        * (lake_area * 1e6)
    
    # maximum storage - m3
    maximum_storage = (max_depth - orifice_elevation) \
        * (lake_area * 1e6)
        
    if now >= update_time:
        LOG.debug(
            'Looking for observation to assimilate...'
        )
    
        # initialize variable to store assimilated observations. We initialize
        # as np.nan, so that if no good quality observations are found, we can
        # easily catch it.
        obs = np.nan
        
        # identify TimeSlice time (gage_time) index nearest to, but not greater 
        # than the update_time
        t_diff = update_time - gage_time
        t_idx = np.where(t_diff >=  0, t_diff, np.inf).argmin()
        
        # look backwards from the nearest update_time for the first available 
        # observation,
        # NOTE: QA/QC has already happened upstream upon loading and formatting 
        # TimeSlice observations, so all poor values have already been replaced 
        # with nans
        for i in range(t_idx, -1, -1):
            
            # check if gage observation is good quality (not nan)
            if np.isnan(gage_obs[i]) == False:
                
                # record good observation to obs
                obs = gage_obs[i]
                
                # determine how many seconds prior to the update_time the 
                # observation was taken
                t_obs = gage_time[i]
                gage_lookback_seconds = update_time - t_obs
                
                # reset the observation update time
                new_update_time = update_time + update_time_interval
                
                break
                
        if np.isnan(obs): # no good observation was found
        
            '''
            OUTSTANDING QUESTIONS:
            ----------------------
            - should we presist the previously persisted outflow if an 
              observation was found but was outside of the obs_lookback_hours
            
            - What to do with the persistence weight? Persistence weights are 
              intended to control what fraction of the outflow value is from
              the levelpool simulation, versus the assimilated observation.
              However, these values are all ones in the reservoir parameter. Is
              this still of utility. It is not considered in this code, yet. 
            
            - If no good observation is found, then we do not update the 
              update time. Consequently we will continue to search for a good
              observation at each model timestep, before updating the update
              time. 

            '''
            LOG.debug(
                'No good observation found, persisting previously assimilated flow'
            )
            persisted_outflow = previous_persisted_outflow
            
            if now >= persistence_update_time:
                new_persistence_index = persistence_index + 1
                new_persistence_update_time = persistence_update_time \
                    + persistence_update_time_interval

        
        else: # good observation found    
        
            # check that observation is not taken from beyond the
            # allowable lookback window
            if gage_lookback_seconds > obs_lookback_hours*60*60:
                LOG.debug('good observation found, but is outside of lookback window')
                LOG.debug(
                    'observation at %s seconds from update time', 
                    gage_lookback_seconds
                )
                persisted_outflow = previous_persisted_outflow
                if now >= persistence_update_time:
                    new_persistence_index = persistence_index + 1
                    new_persistence_update_time = persistence_update_time \
                        + persistence_update_time_interval
                
            else:
                LOG.debug('good observation found!: %s cms', obs)
                LOG.debug(
                    'observation at %s seconds from update time', 
                    gage_lookback_seconds
                )
                # the new persisted outflow is the discovered gage observation
                persisted_outflow = obs
                # reset persistence index and update persistence update time
                new_persistence_index = 1
                new_persistence_update_time = persistence_update_time \
                    + persistence_update_time_interval
    
    elif now >= persistence_update_time:
     
        # increment the persistence index
        new_persistence_index = persistence_index + 1
        new_persistence_update_time = persistence_update_time \
            + persistence_update_time_interval
            
        # persist previously persisted outflow value
        if persistence_index <= persistence_limit:
            LOG.debug(
                'Persisting previously assimilated outflow'
            )
            persisted_outflow = previous_persisted_outflow
            
        # persistence limit reached - use levelpool outflow
        if persistence_index > persistence_limit:
            LOG.debug(
                'Persistence limit reached, defaulting to levelpool outflow'
            )
            persisted_outflow = levelpool_outflow
            new_persistence_index = 0
        
    else:
        LOG.debug(
            'Persisting previously assimilated outflow'
        )
        persisted_outflow = previous_persisted_outflow
    
    # set reservoir outflow
    if np.isnan(persisted_outflow):
        LOG.debug(
            'Previously persisted outflow is nan, defaulting to levelpool outflow'
        )
        # levelpool outflow
        outflow = levelpool_outflow
        new_persistence_index = 0
    
    else:
        # data assimilated outflow
        outflow = persisted_outflow
        
    # check that adjusted outflow does not violate storage limitations
    outflow, max_storage_reached = _modify_for_projected_storage(
        inflow,
        outflow,
        initial_storage,
        maximum_storage,
        lake_number,
        now,
        routing_period,
    )
    
    # if storage limits are violated, reset outflow to levepool
    if max_storage_reached and outflow < levelpool_outflow:
        outflow = levelpool_outflow
        
    # compute the new water surface elevation
    delta_storage = (inflow - outflow) * routing_period
    new_storage = initial_storage + delta_storage
    new_water_elevation = initial_water_elevation \
        + (delta_storage / (lake_area*1e6))

    return outflow, persisted_outflow, new_water_elevation, new_update_time, new_persistence_index, new_persistence_update_time

    
        
                    
    
