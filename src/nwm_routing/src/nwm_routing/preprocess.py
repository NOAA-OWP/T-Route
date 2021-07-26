import time
import pandas as pd
from collections import defaultdict
import troute.nhd_network_utilities_v02 as nnu
import troute.nhd_network as nhd_network
import troute.nhd_io as nhd_io


def nwm_network_preprocess(
    supernetwork_parameters,
    waterbody_parameters,
    showtiming=False,
    verbose=False,
    debuglevel=0,
):

    if verbose:
        print("creating supernetwork connections set")
    if showtiming:
        start_time = time.time()

    # STEP 1: Build basic network connections graph,
    # read network parameters, identify waterbodies and gages, if any.
    connections, param_df, wbodies, gages = nnu.build_connections(
        supernetwork_parameters,
    )

    break_network_at_waterbodies = waterbody_parameters.get(
        "break_network_at_waterbodies", False
    )
    break_network_at_gages = supernetwork_parameters.get(
        "break_network_at_gages", False
    )

    if (
        not wbodies
    ):  # Turn off any further reservoir processing if the network contains no waterbodies
        break_network_at_waterbodies = False

    if break_network_at_waterbodies:
        connections = nhd_network.replace_waterbodies_connections(connections, wbodies)

    if verbose:
        print("supernetwork connections set complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    ################################
    ## STEP 3a: Read waterbody parameter file
    # waterbodies_values = supernetwork_values[12]
    # waterbodies_segments = supernetwork_values[13]
    # connections_tailwaters = supernetwork_values[4]

    if break_network_at_waterbodies:
        # Read waterbody parameters
        waterbodies_df = nhd_io.read_waterbody_df(
            waterbody_parameters, {"level_pool": wbodies.values()}
        )

        # Remove duplicate lake_ids and rows
        waterbodies_df = (
            waterbodies_df.reset_index()
            .drop_duplicates(subset="lake_id")
            .set_index("lake_id")
        )

        # Declare empty dataframe
        waterbody_types_df = pd.DataFrame()

        # Check if hybrid-usgs, hybrid-usace, or rfc type reservoirs are set to true
        wbtype = "hybrid_and_rfc"
        wb_params_hybrid_and_rfc = waterbody_parameters.get(
            wbtype, defaultdict(list)
        )  # TODO: Convert these to `get` statments

        wbtype = "level_pool"
        wb_params_level_pool = waterbody_parameters.get(
            wbtype, defaultdict(list)
        )  # TODO: Convert these to `get` statments

        waterbody_type_specified = False

        # NOTE: What are we accomplishing with this logic here?
        if (
            wb_params_hybrid_and_rfc["reservoir_persistence_usgs"]
            or wb_params_hybrid_and_rfc["reservoir_persistence_usace"]
            or wb_params_hybrid_and_rfc["reservoir_rfc_forecasts"]
        ):

            waterbody_type_specified = True

            waterbody_types_df = nhd_io.read_reservoir_parameter_file(
                wb_params_hybrid_and_rfc["reservoir_parameter_file"],
                wb_params_level_pool["level_pool_waterbody_id"],
                wbodies.values(),
            )

            # Remove duplicate lake_ids and rows
            waterbody_types_df = (
                waterbody_types_df.reset_index()
                .drop_duplicates(subset="lake_id")
                .set_index("lake_id")
            )

    else:
        # Declare empty dataframes
        waterbody_types_df = pd.DataFrame()
        waterbodies_df = pd.DataFrame()

    # STEP 2: Identify Independent Networks and Reaches by Network
    if showtiming:
        start_time = time.time()
    if verbose:
        print("organizing connections into reaches ...")

    network_break_segments = set()
    if break_network_at_waterbodies:
        network_break_segments = network_break_segments.union(wbodies.values())
    if break_network_at_gages:
        network_break_segments = network_break_segments.union(gages.keys())

    independent_networks, reaches_bytw, rconn = nnu.organize_independent_networks(
        connections,
        network_break_segments,
    )
    if verbose:
        print("reach organization complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    return (
        connections,
        param_df,
        wbodies,
        waterbodies_df,
        waterbody_types_df,
        break_network_at_waterbodies,  # Could this be inferred from the wbodies or waterbodies_df  # Could this be inferred from the wbodies or waterbodies_df? Consider making this name less about the network and more about the reservoir simulation.
        waterbody_type_specified,  # Seems like this could be inferred from waterbody_types_df...
        independent_networks,
        reaches_bytw,
        rconn,
    )


def nwm_initial_warmstate_preprocess(
    break_network_at_waterbodies,
    restart_parameters,
    segment_index,
    waterbodies_df,
    segment_list=None,
    wbodies_list=None,
    showtiming=False,
    verbose=False,
    debuglevel=0,
):

    if break_network_at_waterbodies:
        ## STEP 3c: Handle Waterbody Initial States
        # TODO: move step 3c into function in nnu, like other functions wrapped in main()
        if showtiming:
            start_time = time.time()
        if verbose:
            print("setting waterbody initial states ...")

        if restart_parameters.get("wrf_hydro_waterbody_restart_file", None):
            waterbodies_initial_states_df = nhd_io.get_reservoir_restart_from_wrf_hydro(
                restart_parameters["wrf_hydro_waterbody_restart_file"],
                restart_parameters["wrf_hydro_waterbody_ID_crosswalk_file"],
                restart_parameters["wrf_hydro_waterbody_ID_crosswalk_file_field_name"],
                restart_parameters["wrf_hydro_waterbody_crosswalk_filter_file"],
                restart_parameters[
                    "wrf_hydro_waterbody_crosswalk_filter_file_field_name"
                ],
            )
        else:
            # TODO: Consider adding option to read cold state from route-link file
            waterbodies_initial_ds_flow_const = 0.0
            waterbodies_initial_depth_const = -1.0
            # Set initial states from cold-state
            waterbodies_initial_states_df = pd.DataFrame(
                0,
                index=waterbodies_df.index,
                columns=[
                    "qd0",
                    "h0",
                ],
                dtype="float32",
            )
            # TODO: This assignment could probably by done in the above call
            waterbodies_initial_states_df["qd0"] = waterbodies_initial_ds_flow_const
            waterbodies_initial_states_df["h0"] = waterbodies_initial_depth_const
            waterbodies_initial_states_df["index"] = range(
                len(waterbodies_initial_states_df)
            )

        if verbose:
            print("waterbody initial states complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
            start_time = time.time()

    # STEP 4: Handle Channel Initial States
    if showtiming:
        start_time = time.time()
    if verbose:
        print("setting channel initial states ...")

    q0 = nnu.build_channel_initial_state(restart_parameters, segment_index)

    if verbose:
        print("channel initial states complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))
        start_time = time.time()

    # TODO: Does this need to live outside the if-block for waterbodies above? If not, let's move it up there to keep things together.
    waterbodies_df = pd.merge(
        waterbodies_df, waterbodies_initial_states_df, on="lake_id"
    )

    last_obs_file = restart_parameters.get("wrf_hydro_last_obs_file", None)
    last_obs_df = pd.DataFrame()

    return waterbodies_df, q0, last_obs_df
    # TODO: This returns a full dataframe (waterbodies_df) with the
    # merged initial states for waterbodies, but only the
    # initial state values (q0; not merged with the channel properties)
    # for the channels --
    # That is because that is how they are used downstream. Need to
    # trace that back and decide if there is one of those two ways
    # that is optimal and make both returns that way.


def nwm_forcing_preprocess(
    run,
    forcing_parameters,
    data_assimilation_parameters,
    break_network_at_waterbodies,
    segment_index,
    showtiming=False,
    verbose=False,
    debuglevel=0,
):

    nts = forcing_parameters.get("nts", None)
    dt = forcing_parameters.get("dt", None)
    qts_subdivisions = forcing_parameters.get("qts_subdivisions", None)
    qlat_input_folder = forcing_parameters.get("qlat_input_folder", None)
    qlat_file_index_col = forcing_parameters.get("qlat_file_index_col", None)
    qlat_file_value_col = forcing_parameters.get("qlat_file_value_col", None)

    # TODO: find a better way to deal with these defaults and overrides.
    run["nts"] = run.get("nts", nts)
    run["dt"] = run.get("dt", dt)
    run["qts_subdivisions"] = run.get("qts_subdivisions", qts_subdivisions)
    run["qlat_input_folder"] = run.get("qlat_input_folder", qlat_input_folder)
    run["qlat_file_index_col"] = run.get("qlat_file_index_col", qlat_file_index_col)
    run["qlat_file_value_col"] = run.get("qlat_file_value_col", qlat_file_value_col)

    # STEP 5: Read (or set) QLateral Inputs
    if showtiming:
        start_time = time.time()
    if verbose:
        print("creating qlateral array ...")

    qlats_df = nnu.build_qlateral_array(
        run,
        segment_index,
    )

    if verbose:
        print("qlateral array complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    # STEP 6
    data_assimilation_csv = data_assimilation_parameters.get(
        "data_assimilation_csv", None
    )
    data_assimilation_folder = data_assimilation_parameters.get(
        "data_assimilation_timeslices_folder", None
    )
    if data_assimilation_csv or data_assimilation_folder:

        if data_assimilation_folder and data_assimilation_csv:
            print(
                "Please select data_assimilation_parameters_folder + data_assimilation_filter or data_assimilation_csv not both."
            )

        if showtiming:
            start_time = time.time()
        if verbose:
            print("creating usgs time_slice data array ...")

        usgs_df, _ = nnu.build_data_assimilation(data_assimilation_parameters)

        if verbose:
            print("usgs array complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))

    else:
        usgs_df = pd.DataFrame()

    # STEP 7
    coastal_boundary_elev = forcing_parameters.get("coastal_boundary_elev_data", None)
    coastal_ncdf = forcing_parameters.get("coastal_ncdf", None)

    if coastal_boundary_elev:
        print("creating coastal dataframe ...")
        coastal_df = nhd_io.build_coastal_dataframe(coastal_boundary_elev)

    if coastal_ncdf:
        print("creating coastal ncdf dataframe ...")
        coastal_ncdf_df = nhd_io.build_coastal_ncdf_dataframe(coastal_ncdf)

    # TODO: disentangle the implicit (run) and explicit (qlats_df, usgs_df) returns
    return qlats_df, usgs_df
