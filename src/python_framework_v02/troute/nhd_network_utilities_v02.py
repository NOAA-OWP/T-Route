import json
import pathlib
import pandas as pd
from functools import partial
import troute.nhd_io as nhd_io
import troute.nhd_network as nhd_network


def set_supernetwork_parameters(
    supernetwork="", geo_input_folder=None, verbose=True, debuglevel=0
):
    # TODO: consider managing path concatenation outside this function (and lose the os dependency)

    # The following datasets are extracts from the feature datasets available
    # from https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/web/data_tools/
    # the CONUS_ge5 and Brazos_LowerColorado_ge5 datasets are included
    # in the github test folder

    supernetwork_options = {
        "Pocono_TEST1",
        "Pocono_TEST2",
        "LowerColorado_Conchos_FULL_RES",
        "Brazos_LowerColorado_ge5",
        "Brazos_LowerColorado_FULL_RES",
        "Brazos_LowerColorado_Named_Streams",
        "CONUS_ge5",
        "Mainstems_CONUS",
        "CONUS_Named_Streams",
        "CONUS_FULL_RES_v20",
        "CapeFear_FULL_RES",
        "Florence_FULL_RES",
        "custom",
    }

    if supernetwork not in supernetwork_options:
        print(
            "Note: please call function with supernetworks set to one of the following:"
        )
        for s in supernetwork_options:
            print(f"'{s}'")
        raise ValueError

    elif supernetwork == "Pocono_TEST1":
        return {
            "geo_file_path": pathlib.Path(
                geo_input_folder, "PoconoSampleData1", "PoconoSampleRouteLink1.shp"
            ).resolve(),
            "columns": {
                "key": "link",
                "downstream": "to",
                "dx": "Length",
                "n": "n",
                "ncc": "nCC",
                "s0": "So",
                "bw": "BtmWdth",
                "waterbody": "NHDWaterbo",
                "tw": "TopWdth",
                "twcc": "TopWdthCC",
                "alt": "alt",
                "musk": "MusK",
                "musx": "MusX",
                "cs": "ChSlp",
            },
            "waterbody_null_code": -9999,
            "title_string": "Pocono Test Example",
            "driver_string": "ESRI Shapefile",
            "terminal_code": 0,
            "layer_string": 0,
            "waterbody_parameter_file_type": "Level_Pool",
            "waterbody_parameter_file_path": pathlib.Path(
                geo_input_folder, "NWM_2.1_Sample_Datasets", "LAKEPARM_CONUS.nc"
            ).resolve(),
            "waterbody_parameter_columns": {
                "waterbody_area": "LkArea",  # area of reservoir
                "weir_elevation": "WeirE",
                "waterbody_max_elevation": "LkMxE",
                "outfall_weir_coefficient": "WeirC",
                "outfall_weir_length": "WeirL",
                "overall_dam_length": "DamL",
                "orifice_elevation": "OrificeE",
                "orifice_coefficient": "OrificeC",
                "orifice_area": "OrificeA",
            },
        }
    elif supernetwork == "Pocono_TEST2":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "Pocono Test 2 Example",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "PoconoRouteLink_TEST2_nwm_mc.txt",
                ).resolve(),
                "mask_driver_string": "csv",
                "mask_layer_string": "",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

        # return {
        #'geo_file_path' : pathlib.Path(geo_input_folder
    # , r'PoconoSampleData2'
    # , r'PoconoRouteLink_testsamp1_nwm_mc.shp').resolve()
    # , 'key_col' : 18
    # , 'downstream_col' : 23
    # , 'length_col' : 5
    # , 'manningn_col' : 20
    # , 'manningncc_col' : 21
    # , 'slope_col' : 10
    # , 'bottomwidth_col' : 2
    # , 'topwidth_col' : 11
    # , 'topwidthcc_col' : 12
    # , 'MusK_col' : 6
    # , 'MusX_col' : 7
    # , 'ChSlp_col' : 3
    # , 'terminal_code' : 0
    # , 'title_string' : 'Pocono Test 2 Example'
    # , 'driver_string' : 'ESRI Shapefile'
    # , 'layer_string' : 0
    # }

    elif supernetwork == "LowerColorado_Conchos_FULL_RES":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "NHD 2.0 Conchos Basin of the LowerColorado River",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "LowerColorado_Conchos_FULL_RES.txt",
                ).resolve(),
                "mask_driver_string": "csv",
                "mask_layer_string": "",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "Brazos_LowerColorado_ge5":
        return {
            "geo_file_path": pathlib.Path(
                geo_input_folder, "Channels", "NHD_BrazosLowerColorado_Channels.shp"
            ).resolve(),
            "columns": {
                "key": "featureID",
                "downstream": "to",
                "waterbody": "NHDWaterbo",
                "dx": "Length",
                "n": "n",
                "s0": "So",
                "bw": "BtmWdth",
                "musk": "MusK",
                "musx": "MusX",
                "cs": "ChSlp",
            },
            "waterbody_null_code": -9999,
            "title_string": "NHD Subset including Brazos + Lower Colorado\nNHD stream orders 5 and greater",
            "driver_string": "ESRI Shapefile",
            "terminal_code": 0,
            "layer_string": 0,
        }

    elif supernetwork == "Brazos_LowerColorado_FULL_RES":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "NHD 2.0 Brazos and LowerColorado Basins",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "Brazos_LowerColorado_FULL_RES.txt",
                ).resolve(),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "Brazos_LowerColorado_Named_Streams":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "NHD 2.0 GNIS labeled streams in the Brazos and LowerColorado Basins",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "Brazos_LowerColorado_Named_Streams.csv",
                ).resolve(),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "CONUS_ge5":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "NHD CONUS Order 5 and Greater",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder, "Channels", "masks", "CONUS_ge5.txt"
                ).resolve(),
                "mask_driver_string": "csv",
                "mask_layer_string": "",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "Mainstems_CONUS":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "CONUS 'Mainstems' (Channels below gages and AHPS prediction points)",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder, r"Channels", r"masks", r"conus_Mainstem_links.txt"
                ).resolve(),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

        # return {
        #     "geo_file_path": pathlib.Path(
        #         geo_input_folder, r"Channels", r"conus_routeLink_subset.nc"
        #     ).resolve(),
        #     "key_col": 0,
        #     "downstream_col": 2,
        #     "length_col": 10,
        #     "manningn_col": 11,
        #     "manningncc_col": 20,
        #     "slope_col": 12,
        #     "bottomwidth_col": 14,
        #     "topwidth_col": 22,
        #     "topwidthcc_col": 21,
        #     "waterbody_col": 15,
        #     "waterbody_null_code": -9999,
        #     "MusK_col": 8,
        #     "MusX_col": 9,
        #     "ChSlp_col": 13,
        #     "terminal_code": 0,
        #     "title_string": 'CONUS "Mainstem"',
        #     "driver_string": "NetCDF",
        #     "layer_string": 0,
        # }

    elif supernetwork == "CONUS_Named_Streams":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "CONUS NWM v2.0 only GNIS labeled streams",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "nwm_reaches_conus_v21_wgnis_name.csv",
                ).resolve(),
                "mask_driver_string": "csv",
                "mask_layer_string": "",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "CapeFear_FULL_RES":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "Cape Fear River Basin, NC",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "CapeFear_FULL_RES.txt",
                ).resolve(),
                "mask_driver_string": "csv",
                "mask_layer_string": "",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "Florence_FULL_RES":
        rv = set_supernetwork_parameters(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        rv.update(
            {
                "title_string": "Hurricane Florence Domain, near Durham NC",  # overwrites other title...
                "mask_file_path": pathlib.Path(
                    geo_input_folder,
                    "Channels",
                    "masks",
                    "Florence_FULL_RES.txt",
                ).resolve(),
                "mask_driver_string": "csv",
                "mask_layer_string": "",
                "mask_key": 0,
                "mask_name": 1,  # TODO: Not used yet.
            }
        )
        return rv

    elif supernetwork == "CONUS_FULL_RES_v20":

        ROUTELINK = "RouteLink_NHDPLUS"
        ModelVer = "nwm.v2.0.4"
        ext = "nc"
        sep = "."

        return {
            "geo_file_path": pathlib.Path(
                geo_input_folder, "Channels", sep.join([ROUTELINK, ModelVer, ext])
            ).resolve(),
            "data_link": f"https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/{ModelVer}/parm/domain/{ROUTELINK}{sep}{ext}",
            "columns": {
                "key": "link",
                "downstream": "to",
                "dx": "Length",
                "n": "n",
                "ncc": "nCC",
                "s0": "So",
                "bw": "BtmWdth",
                "tw": "TopWdth",
                "twcc": "TopWdthCC",
                "alt": "alt",
                "waterbody": "NHDWaterbodyComID",
                "musk": "MusK",
                "musx": "MusX",
                "cs": "ChSlp",
            },
            "waterbody_parameter_file_type": "Level_Pool",
            "waterbody_parameter_file_path": pathlib.Path(
                geo_input_folder, "NWM_2.1_Sample_Datasets", "LAKEPARM_CONUS.nc"
            ).resolve(),
            "waterbody_parameter_columns": {
                "waterbody_area": "LkArea",
                "weir_elevation": "WeirE",
                "waterbody_max_elevation": "LkMxE",
                "outfall_weir_coefficient": "WeirC",
                "outfall_weir_length": "WeirL",
                "overall_dam_length": "DamL",
                "orifice_elevation": "OrificeE",
                "oriface_coefficient": "OrificeC",
                "oriface_are": "OrifaceA",
            },
            "waterbody_null_code": -9999,
            "title_string": "CONUS Full Resolution NWM v2.0",
            "driver_string": "NetCDF",
            "terminal_code": 0,
            "layer_string": 0,
        }

    elif supernetwork == "custom":
        custominput = pathlib.Path(geo_input_folder).resolve()
        with open(custominput, "r") as json_file:
            return json.load(json_file)
            # TODO: add error trapping for potentially missing files


def reverse_dict(d):
    """
    Reverse a 1-1 mapping
    Values must be hashable!
    """
    return {v: k for k, v in d.items()}


def build_connections(supernetwork_parameters):
    cols = supernetwork_parameters["columns"]
    terminal_code = supernetwork_parameters.get("terminal_code", 0)

    param_df = nhd_io.read(pathlib.Path(supernetwork_parameters["geo_file_path"]))

    param_df = param_df[list(cols.values())]
    param_df = param_df.set_index(cols["key"])

    if "mask_file_path" in supernetwork_parameters:
        data_mask = nhd_io.read_mask(
            pathlib.Path(supernetwork_parameters["mask_file_path"]),
            layer_string=supernetwork_parameters["mask_layer_string"],
        )
        param_df = param_df.filter(
            data_mask.iloc[:, supernetwork_parameters["mask_key"]], axis=0
        )

    param_df = param_df.rename(columns=reverse_dict(cols))
    # Rename parameter columns to standard names: from route-link names
    #        key: "link"
    #        downstream: "to"
    #        dx: "Length"
    #        n: "n"  # TODO: rename to `manningn`
    #        ncc: "nCC"  # TODO: rename to `mannningncc`
    #        s0: "So"  # TODO: rename to `bedslope`
    #        bw: "BtmWdth"  # TODO: rename to `bottomwidth`
    #        waterbody: "NHDWaterbodyComID"
    #        gages: "gages"
    #        tw: "TopWdth"  # TODO: rename to `topwidth`
    #        twcc: "TopWdthCC"  # TODO: rename to `topwidthcc`
    #        alt: "alt"
    #        musk: "MusK"
    #        musx: "MusX"
    #        cs: "ChSlp"  # TODO: rename to `sideslope`
    param_df = param_df.sort_index()

    param_df = param_df.rename(columns=reverse_dict(cols))

    wbodies = {}
    if "waterbody" in cols:
        wbodies = build_waterbodies(
            param_df[["waterbody"]], supernetwork_parameters, "waterbody"
        )
        param_df = param_df.drop("waterbody", axis=1)

    gages = {}
    if "gages" in cols:
        gages = build_gages(param_df[["gages"]])
        param_df = param_df.drop("gages", axis=1)

    connections = nhd_network.extract_connections(param_df, "downstream")
    param_df = param_df.drop("downstream", axis=1)

    param_df = param_df.astype("float32")

    # datasub = data[['dt', 'bw', 'tw', 'twcc', 'dx', 'n', 'ncc', 'cs', 's0']]
    return connections, param_df, wbodies, gages


def build_gages(
    segment_gage_df,
):
    gage_list = list(map(bytes.strip, segment_gage_df.gages.values))
    gage_mask = list(map(bytes.isdigit, gage_list))
    gages = segment_gage_df.loc[gage_mask, "gages"].to_dict()

    return gages


def build_waterbodies(
    segment_reservoir_df,
    supernetwork_parameters,
    waterbody_crosswalk_column="waterbody",
):
    """
    segment_reservoir_list
    supernetwork_parameters
    waterbody_crosswalk_column
    """
    wbodies = nhd_network.extract_waterbodies(
        segment_reservoir_df,
        waterbody_crosswalk_column,
        supernetwork_parameters["waterbody_null_code"],
    )

    # TODO: Add function to read LAKEPARM.nc here
    # TODO: return the lakeparam_df

    return wbodies


def organize_independent_networks(connections, wbodies=None):

    rconn = nhd_network.reverse_network(connections)
    independent_networks = nhd_network.reachable_network(rconn)
    reaches_bytw = {}
    for tw, net in independent_networks.items():
        if wbodies:
            path_func = partial(
                nhd_network.split_at_waterbodies_and_junctions, wbodies, net
            )
        else:
            path_func = partial(nhd_network.split_at_junction, net)

        reaches_bytw[tw] = nhd_network.dfs_decomposition(net, path_func)

    return independent_networks, reaches_bytw, rconn


def build_channel_initial_state(restart_parameters, segment_index=pd.Index([])):

    channel_restart_file = restart_parameters.get("channel_restart_file", None)

    wrf_hydro_channel_restart_file = restart_parameters.get(
        "wrf_hydro_channel_restart_file", None
    )

    if channel_restart_file:
        q0 = nhd_io.get_channel_restart_from_csv(channel_restart_file)

    elif wrf_hydro_channel_restart_file:

        q0 = nhd_io.get_channel_restart_from_wrf_hydro(
            restart_parameters["wrf_hydro_channel_restart_file"],
            restart_parameters["wrf_hydro_channel_ID_crosswalk_file"],
            restart_parameters["wrf_hydro_channel_ID_crosswalk_file_field_name"],
            restart_parameters["wrf_hydro_channel_restart_upstream_flow_field_name"],
            restart_parameters["wrf_hydro_channel_restart_downstream_flow_field_name"],
            restart_parameters["wrf_hydro_channel_restart_depth_flow_field_name"],
        )
    else:
        # Set cold initial state
        # assume to be zero
        # 0, index=connections.keys(), columns=["qu0", "qd0", "h0",], dtype="float32"
        q0 = pd.DataFrame(
            0,
            index=segment_index,
            columns=["qu0", "qd0", "h0"],
            dtype="float32",
        )
    # TODO: If needed for performance improvement consider filtering mask file on read.
    if not segment_index.empty:
        q0 = q0[q0.index.isin(segment_index)]

    return q0


def build_qlateral_array(
    forcing_parameters,
    segment_index=pd.Index([]),
    ts_iterator=None,
    file_run_size=None,
):
    # TODO: set default/optional arguments

    qts_subdivisions = forcing_parameters.get("qts_subdivisions", 1)
    nts = forcing_parameters.get("nts", 1)
    qlat_input_folder = forcing_parameters.get("qlat_input_folder", None)
    qlat_input_file = forcing_parameters.get("qlat_input_file", None)
    if qlat_input_folder:
        qlat_input_folder = pathlib.Path(qlat_input_folder)
        if "qlat_files" in forcing_parameters:
            qlat_files = forcing_parameters.get("qlat_files")
            qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
        elif "qlat_file_pattern_filter" in forcing_parameters:
            qlat_file_pattern_filter = forcing_parameters.get(
                "qlat_file_pattern_filter", "*CHRT_OUT*"
            )
            qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))

        qlat_file_index_col = forcing_parameters.get(
            "qlat_file_index_col", "feature_id"
        )
        qlat_file_value_col = forcing_parameters.get("qlat_file_value_col", "q_lateral")

        qlat_df = nhd_io.get_ql_from_wrf_hydro_mf(
            qlat_files=qlat_files,
            # ts_iterator=ts_iterator,
            # file_run_size=file_run_size,
            index_col=qlat_file_index_col,
            value_col=qlat_file_value_col,
        )

        qlat_df = qlat_df[qlat_df.index.isin(segment_index)]

    # TODO: These four lines seem extraneous
    #    df_length = len(qlat_df.columns)
    #    for x in range(df_length, 144):
    #        qlat_df[str(x)] = 0
    #        qlat_df = qlat_df.astype("float32")

    elif qlat_input_file:
        qlat_df = nhd_io.get_ql_from_csv(qlat_input_file)

    else:
        qlat_const = forcing_parameters.get("qlat_const", 0)
        qlat_df = pd.DataFrame(
            qlat_const,
            index=segment_index,
            columns=range(nts // qts_subdivisions),
            dtype="float32",
        )

    # TODO: Make a more sophisticated date-based filter
    max_col = 1 + nts // qts_subdivisions
    if len(qlat_df.columns) > max_col:
        qlat_df.drop(qlat_df.columns[max_col:], axis=1, inplace=True)

    if not segment_index.empty:
        qlat_df = qlat_df[qlat_df.index.isin(segment_index)]

    return qlat_df


def build_data_assimilation(data_assimilation_parameters):
    data_assimilation_csv = data_assimilation_parameters.get(
        "data_assimilation_csv", None
    )
    data_assimilation_folder = data_assimilation_parameters.get(
        "data_assimilation_timeslices_folder", None
    )
    # TODO: Fix the Logic here according to the following.

    # If there are any observations for data assimilation, there
    # needs to be a complete set in the first time set or else
    # there must be a "LastObs". If there is a complete set in
    # the first time step, the LastObs is optional. If there are
    # no observations for assimilation, there can be a LastObs
    # with an empty usgs dataframe.

    last_obs_file = data_assimilation_parameters.get("wrf_hydro_last_obs_file", None)
    last_obs_type = data_assimilation_parameters.get("wrf_last_obs_type", "error-based")
    last_obs_crosswalk_file = data_assimilation_parameters.get(
        "wrf_hydro_da_channel_ID_crosswalk_file", None
    )

    last_obs_df = pd.DataFrame()

    if last_obs_file:
        last_obs_df = nhd_io.build_last_obs_df(
            last_obs_file,
            last_obs_crosswalk_file,
            last_obs_type,
        )

    if data_assimilation_csv:
        usgs_df = build_data_assimilation_csv(data_assimilation_parameters)
    elif data_assimilation_folder:
        usgs_df = build_data_assimilation_folder(data_assimilation_parameters)
    return usgs_df, last_obs_df


def build_data_assimilation_csv(data_assimilation_parameters):

    usgs_df = nhd_io.get_usgs_from_time_slices_csv(
        data_assimilation_parameters["wrf_hydro_da_channel_ID_crosswalk_file"],
        data_assimilation_parameters["data_assimilation_csv"],
    )

    return usgs_df


def build_data_assimilation_folder(data_assimilation_parameters):

    if data_assimilation_parameters:
        usgs_timeslices_folder = pathlib.Path(
            data_assimilation_parameters["data_assimilation_timeslices_folder"],
        ).resolve()

        usgs_df = nhd_io.get_usgs_from_time_slices_folder(
            data_assimilation_parameters["wrf_hydro_da_channel_ID_crosswalk_file"],
            usgs_timeslices_folder,
            data_assimilation_parameters["data_assimilation_filter"],
        )

    return usgs_df
