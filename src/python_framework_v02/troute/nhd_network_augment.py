import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from functools import partial
from itertools import chain
import sys
import pathlib
import argparse
import json
from tqdm import tqdm
import time

# FIXME remove this!!!
root = pathlib.Path("../../../").resolve()
sys.path.append(str(pathlib.Path(root, "src", "python_framework_v01").resolve()))

import nhd_network_utilities_v02 as nnu
import nhd_network
import nhd_io
import network_dl


def _handle_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--threshold_length",
        help="threshold segment length (meters)",
        dest="threshold",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--network",
        help="Choose from among the pre-programmed supernetworks (Pocono_TEST1, Pocono_TEST2, LowerColorado_Conchos_FULL_RES, Brazos_LowerColorado_ge5, Brazos_LowerColorado_FULL_RES, Brazos_LowerColorado_Named_Streams, CONUS_ge5, Mainstems_CONUS, CONUS_Named_Streams, CONUS_FULL_RES_v20, CapeFear_FULL_RES)",
        dest="supernetwork",
        choices=[
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
        ],
        default="CapeFear_FULL_RES",
    )
    parser.add_argument(
        "-p",
        "--prune",
        help="prune short headwater reaches, 1 if yes, 0 if no",
        dest="prune",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--snap",
        help="snap junctions adjacent to short reaches, 1 if yes, 0 if no",
        dest="snap",
        action="store_true",
    )
    parser.add_argument(
        "-return_original",
        "--return_original",
        help="return an unmodified RouteLink.nc file for the specified domain",
        dest="return_original",
        action="store_true",
    )

    return parser.parse_args()


def get_network_data(network_name):

    # Create directory path variable for test/input/geo, where NHD data and masks are stored
    test_folder = pathlib.Path(root, r"test").resolve()
    geo_input_folder = pathlib.Path(test_folder, r"input", r"geo").resolve()

    # Load network meta data for the Cape Fear Basin
    supernetwork = network_name
    network_data = nnu.set_supernetwork_data(
        supernetwork=supernetwork, geo_input_folder=geo_input_folder
    )

    # if the NHDPlus RouteLink file does not exist, download it.
    if not network_data["geo_file_path"].is_file:
        filename = network_data["geo_file_path"].name
        network_dl.download(network_data["geo_file_path"], network_data["data_link"])

    # read-in NHD data, retain copies for viz- and full network analysis purposes
    RouteLink = nhd_io.read(network_data["geo_file_path"])

    # select only the necessary columns of geospatial data, set the DataFrame index
    cols = [v for c, v in network_data["columns"].items()]
    # GET THE STRAHLER ORDER DATA TOO!
    cols.append("order")

    data = nhd_io.read(network_data["geo_file_path"])
    data = data[cols]
    data = data.set_index(network_data["columns"]["key"])

    # mask NHDNetwork to isolate test network - full resolution Cape Fear basin, NC
    if "mask_file_path" in network_data:
        data_mask = nhd_io.read_mask(
            network_data["mask_file_path"],
            layer_string=network_data["mask_layer_string"],
        )
        data = data.filter(data_mask.iloc[:, network_data["mask_key"]], axis=0)

    # sort index
    data = data.sort_index()

    # replace downstreams
    data = nhd_io.replace_downstreams(data, network_data["columns"]["downstream"], 0)

    return data, RouteLink, network_data


def network_connections(data, network_data):

    """
    Extract upstream and downstream connections between segments in network
    Args:
        data (DataFrame): Network parameter dataset, prepared
        network_data (dict): network metadata
    Returns:
        conn (dict): downstream connections
        rconn (dict): upstream connections
    """

    # extract downstream connections
    conn = nhd_network.extract_connections(data, network_data["columns"]["downstream"])

    # extract upstream connections
    rconn = nhd_network.reverse_network(conn)

    return conn, rconn


def build_reaches(rconn):

    # isolate independent subnetworks
    subnets = nhd_network.reachable_network(rconn)

    # identify the segments in each subnetwork
    subreachable = nhd_network.reachable(rconn)

    # break each subnetwork into reaches
    subreaches = {}
    for tw, net in subnets.items():
        path_func = partial(nhd_network.split_at_junction, net)
        subreaches[tw] = nhd_network.dfs_decomposition(net, path_func)

    return subreachable, subreaches, subnets


def prune_headwaters(data, threshold, network_data):

    # initialize list to store pruned reaches
    hw_prune_list = []

    iter_count = 1
    while 1 == 1:

        # STEP 1: Find headwater reaches:
        # --------------------------------------#

        # build connections and reverse connections
        connections, rconn = network_connections(
            data.drop(index=chain.from_iterable(hw_prune_list)), network_data
        )

        # identify headwater segments
        hws = connections.keys() - chain.from_iterable(connections.values())

        # build reaches
        subreachable, subreaches, subnets = build_reaches(rconn)

        # find headwater reaches
        hw_reaches = []
        for sublists in list(subreaches.values()):
            for rch in sublists:
                for val in rch:
                    if val in hws:
                        hw_reaches.append(rch)
                        pass

        # STEP 2: identify short headwater reaches
        # --------------------------------------#
        # find headwater reaches shorter than threshold
        short_hw_reaches = []
        for rch in hw_reaches:
            if data.loc[rch, "Length"].sum() < threshold:
                short_hw_reaches.append(rch)

        # CHECK: Are there any short headwter reaches?
        # --------------------------------------#
        if len(short_hw_reaches) == 0:
            print("no more short headwaters to prune")

            # if no more reaches, exit while loop
            break

        # STEP 3: trim short headwater reaches
        # --------------------------------------#
        hw_junctions = {}
        for rch in short_hw_reaches:
            hw_junctions[rch[-1]] = connections[rch[-1]]

        touched = set()
        for i, (tw, jun) in enumerate(hw_junctions.items()):
            touched.add(i)

            if list(hw_junctions.values()).count(jun) > 1:
                # two short headwaters draining to the same junction

                # record reach1 as list of segments
                reach1 = short_hw_reaches[i]

                for y, (tw_y, jun_y) in enumerate(hw_junctions.items()):
                    if jun_y == jun and tw_y != tw and y not in touched:
                        # the correlary headwater reach draining to the same junction has been found
                        reach2 = short_hw_reaches[y]

                        # trim the shorter of two reaches
                        if (
                            data.loc[reach1, "Length"].sum()
                            <= data.loc[reach2, "Length"].sum()
                        ):
                            # trim reach1
                            hw_prune_list.append(reach1)
                        else:
                            # trim reach2
                            hw_prune_list.append(reach2)

            if list(hw_junctions.values()).count(jun) == 1:
                hw_prune_list.append(short_hw_reaches[i])

        print("completed", iter_count, "iterations of headwater pruning")
        iter_count += 1

    data_pruned = data.drop(index=chain.from_iterable(hw_prune_list))

    return data_pruned


def snap_junctions(data, threshold, network_data):

    """
    This function snaps junctions on oposite ends of a short reach
    by forcing the lowest order upstream tributary to drain to the reach tail.
    Short reaches are defined by a user-appointed threshold length.

    For example, consider a short reach (*) stranded between two junctions:

          \  /
           \/ | 4th
       2nd  \ |
             \| /
             *|/
              |
              |

    The algoritm would select the lowest order tributary segment to the reach head,
    which in this case is a second-order drainage, and change the downstream connection
    to the reach tail. This produces the following "snapped" network config:

          \  /
           \/ |
            \ | /
             \|/
              |
              |

    Inputs
    --------------
    - data (DataFrame): NHD RouteLink data, must contain Strahler order attribute
        - this input can be either the native NHD data OR the data with pruned headwaters

    Outputs
    --------------
    - snapped_data (DataFrame): NHD RouteLink data containing no short reaches

    """
    # create a copy of the native data for snapping
    data_snapped = data.copy()

    # snap junctions
    iter_num = 1
    while 1 == 1:

        # evaluate connections
        connections, rconn = network_connections(data_snapped, network_data)

        # build reaches
        subreachable, subreaches, subnets = build_reaches(rconn)

        # build list of headwater segments
        hws = connections.keys() - chain.from_iterable(connections.values())

        print("finding non-headwater reaches")
        # create a list of short reaches stranded between junctions
        short_reaches = ()
        for sublists in tqdm(list(subreaches.values())):

            for rch in sublists:
                head = rch[0]

                # if reach is not a headwater
                if rconn[head] and data.loc[rch, "Length"].sum() < threshold:
                    short_reaches += tuple(rch)

        if iter_num > 1:
            print(
                "After iteration",
                iter_num - 1,
                ",",
                len(short_reaches),
                "short reaches remain",
            )
        else:
            print(
                "Prior to itteration",
                iter_num,
                ",",
                len(short_reaches),
                "short reaches exist in the network",
            )

        # check that short reaches exist, if none - terminate process
        if len(short_reaches) == 0:
            break

        # for each short reach, snap lower order upstream trib to downstream drainage destination
        for i, rch in enumerate(short_reaches):

            # identify reach tail (downstream-most) and head (upstream-most) segments

            if type(rch) is tuple:
                tail = rch[-1]
                head = rch[0]
            else:
                tail = rch
                head = rch

            # identify segments that drain to reach head
            us_conn = rconn[head]

            # select the upstream segment to snap
            o = data_snapped.loc[us_conn, "order"].tolist()  # Strahler order
            if min(o) != max(o):
                # select the segment with lowest Strahler order
                rch_to_move = data_snapped.loc[us_conn, "order"].idxmin()
            else:
                # if all segments are same Strahler order, select the shortest
                rch_to_move = data_snapped.loc[us_conn, "Length"].idxmin()

            # snap destination is the segment that the reach tail drains to
            snap_destination = connections[tail]

            # if reach tail doesn't have a downstrem connection (it is a network tailwater)
            # then the snap desination is the ocean
            if len(snap_destination) == 0:
                snap_destination = data_snapped.loc[tail, "to"]

            # update RouteLink data with new tributary destination info
            data_snapped.loc[rch_to_move, "to"] = snap_destination

        iter_num += 1

    return data_snapped


def len_weighted_av(df, var, weight):

    """
    Calculate a weighted average
    Args:
        df (DataFrame): DataFrame containing variables to be averaged and used as weights
        var (str): name of the variable to be averaged
        weight (str): name of the variable to be used as a weight
    Returns:
        x (float32): weighted average
    """

    x = (df[weight] * df[var]).sum() / df[weight].sum()

    return x


def merge_parameters(to_merge):

    """
    length-weighted averaging of channel routing parameters across merged segments
    Args:
        to_merge (DataFrame): DataFrame containing routing parameters for segments to be merged together
    Returns:
        replace (DataFrame): weighted average
    """

    data_replace = to_merge.tail(1)
    data_replace._is_copy = None

    idx = to_merge.tail(1).index

    data_replace.loc[idx, "Length"] = to_merge.Length.sum()
    data_replace.loc[idx, "n"] = len_weighted_av(to_merge, "n", "Length")
    data_replace.loc[idx, "nCC"] = len_weighted_av(to_merge, "nCC", "Length")
    data_replace.loc[idx, "So"] = len_weighted_av(to_merge, "So", "Length")
    data_replace.loc[idx, "BtmWdth"] = len_weighted_av(to_merge, "BtmWdth", "Length")
    data_replace.loc[idx, "TopWdth"] = len_weighted_av(to_merge, "TopWdth", "Length")
    data_replace.loc[idx, "TopWdthCC"] = len_weighted_av(
        to_merge, "TopWdthCC", "Length"
    )
    data_replace.loc[idx, "MusK"] = len_weighted_av(to_merge, "MusK", "Length")
    data_replace.loc[idx, "MusX"] = len_weighted_av(to_merge, "MusX", "Length")
    data_replace.loc[idx, "ChSlp"] = len_weighted_av(to_merge, "ChSlp", "Length")

    return data_replace


def correct_reach_connections(data_merged):

    """
    Update downstream connections ("to") for segments in a merged reach.
    Only updates *in-reach* connections.
    Args:
        data_merged (DataFrame): Routing parameters for segments in merged reach
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
    """

    for i, idx in enumerate(data_merged.index.values[0:-1]):
        data_merged.loc[idx, "to"] = data_merged.index.values[i + 1]

    return data_merged


def upstream_merge(data_merged, chop):

    """
    Merge a short reach tail segment with upstream neighbor
    Args:
        data_merged (DataFrame): Routing parameters for segments in merged reach
        chop (list): list of merged-out segments
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
        chop (list): updated list of merged-out segments
    """

    # grab the two segments that need to be merged - simply the last two segments of the reach
    to_merge = data_merged.tail(2)

    # calculate new parameter values
    data_replace = merge_parameters(to_merge)

    # paste new parameters in to data_merged
    data_merged.loc[to_merge.tail(1).index] = data_replace

    # remove merged segments from data_merged
    data_merged = data_merged.drop(to_merge.head(1).index)

    # update "chop" list with merged-out segment IDs
    chop.append(to_merge.head(1).index.values[0])

    return data_merged, chop


def downstream_merge(data_merged, chop, thresh):

    """
    Merge short segments with their downstream neighbors
    Args:
        data_merged (DataFrame): Routing parameters for segments in merged reach
        chop (list): list of merged-out segments
        thresh (int): theshold reach length (meters)
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
        chop (list): updated list of merged-out segments
    """

    # find the upstream-most short segment and it's downstream connection
    idx_us = data_merged.loc[data_merged.Length < thresh].head(1).index.values[0]

    pos_idx_us = data_merged.index.get_loc(idx_us)
    idx_to = data_merged.iloc[pos_idx_us + 1].name

    # grab segments to be merged
    to_merge = data_merged.loc[[idx_us, idx_to]]

    # calculate new parameter values
    data_replace = merge_parameters(to_merge)

    # paste new parameters in to data_merged
    data_merged.loc[to_merge.tail(1).index] = data_replace

    # remove merged segments from data_merged
    data_merged = data_merged.drop(to_merge.head(1).index)

    # update "chop" list with merged-out segment IDs
    chop.append(to_merge.head(1).index.values[0])

    return data_merged, chop


def merge_all(rch, data, chop):

    """
    Merge all segments in a reach
    Args:
        rch (list): Segment indices in the reach to be merged
        data (DataFrame): Routing parameters for network containing the reach to be merged
        chop (list): list of merged-out segments
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
        chop (list): updated list of merged-out segments
    """

    # subset the model parameter data for this reach
    data_merged = data.loc[rch].copy()

    # grab the two segments that need to be merged - in this case, merge all segments!
    to_merge = data_merged.copy()

    # calculate new parameter values
    data_replace = merge_parameters(to_merge)

    # paste new parameters in to data_merged
    data_merged.loc[to_merge.tail(1).index] = data_replace

    # remove merged segments from data_merged - in this case, all but the last
    data_merged = data_merged.drop(data_merged.iloc[:-1, :].index)

    # update "chop" list with merged-out segment IDs
    chop.extend(list(to_merge.iloc[:-1, :].index))

    return data_merged, chop


def update_network_data(data, rch, data_merged, chop, rconn):

    """
    Update the network routing parameter data with merged segment data
    Args:
        data (DataFrame): Routing parameters for network to be updated
        rch (list): Segment indices in the reach to be merged
        data_merged (DataFrame): Routing parameters for merged reach
    Returns:
        data (DataFrame): Updated network routing parameters
    """

    # drop the segments that disapeared with merger
    data = data.drop(chop)

    # adjust the segment data for those that remain
    data.loc[data_merged.index] = data_merged

    # update out of reach connections - these will change in the first segment was merged out
    upstreams = rconn[rch[0]]  # upstream connection of the OLD reach head

    if bool(upstreams):

        data.loc[upstreams, "to"] = data_merged.head(1).index.values[
            0
        ]  # index of NEW reach head

    return data


def qlat_destination_compute(
    data_native, data_merged, merged_segments, pruned_segments, network_data
):

    # build a list of all segments that need crosswalking
    if bool(list(pruned_segments)):
        segments = merged_segments + list(pruned_segments)

    else:
        segments = merged_segments

    # compute connections using native network data
    conn = nhd_network.extract_connections(
        data_native, network_data["columns"]["downstream"]
    )
    rconn = nhd_network.reverse_network(conn)

    # initialize a dictionary to store qlat destination nodes for pruned/merged segments
    qlat_destinations = {}

    for idx in segments:

        # find the segment to recieve qlats from the pruned or merged segment
        if conn[idx]:
            ds_idx = conn[idx]
            while bool(ds_idx[0] in data_merged.index) == False:
                ds_idx = conn[ds_idx[0]]

        elif rconn[idx]:
            ds_idx = rconn[idx]
            while bool(ds_idx[0] in data_merged.index) == False:
                us_idx = conn[ds_idx[0]]

        else:
            ds_idx = []

        # update the qlat destination dict
        qlat_destinations[str(idx)] = str(ds_idx)

    return qlat_destinations


def segment_merge(data_native, data, network_data, thresh, pruned_segments):

    # create a copy of the pruned network dataset, which will be updated with merged data
    data_merged = data.copy()

    # initialize list to store merged segment IDs
    merged_segments = []

    # build connections and reverse connections
    conn, rconn = network_connections(data, network_data)

    # organize network into reaches
    subreachable, subreaches, subnets = build_reaches(rconn)

    # loop through each reach in the network
    for twi, (tw, rchs) in enumerate(subreaches.items(), 1):

        for rch in rchs:

            rch_len = data.loc[rch].Length.sum()

            ##################################################
            # orphaned short single segment reaches
            ##################################################
            # if reach length is shorter than threshold and composed of a single segment
            if rch_len < thresh and len(data.loc[rch]) == 1:
                continue  # do nothing

            ##################################################
            # multi segment reaches - combine into a single segment reach
            ##################################################
            # if reach length is shorter than threshold and composed more than one segment
            if rch_len < thresh and len(data.loc[rch]) > 1:

                # merge ALL reach segments into one
                chop = []
                reach_merged, chop = merge_all(rch, data, chop)

                # update network with merged reach data
                data_merged = update_network_data(
                    data_merged, rch, reach_merged, chop, rconn
                )

                # update merged_segments list with merged-out segments
                merged_segments.extend(chop)

            ##################################################
            # multi segment reaches longer than threshold with some segments shorter than threshold
            ##################################################
            # if reach length is longer than threshold and smallest segment length is less than threshold
            if rch_len > thresh and data.loc[rch].Length.min() < thresh:

                # initialize data_merged - this DataFrame will be subsequently revised
                reach_merged = data.loc[rch]

                # initialize list of segments chopped from this reach
                chop_reach = []

                # so long as the shortest segment is shorter than the threshold...
                while reach_merged.Length.min() < thresh:

                    # if shortest segment is the last segment in the reach - conduct an upstream merge.
                    if (
                        reach_merged.Length.idxmin()
                        == reach_merged.tail(1).index.values[0]
                        and reach_merged.Length.min() < thresh
                    ):

                        # upstream merge
                        chop = []
                        reach_merged, chop = upstream_merge(reach_merged, chop)

                        # update chop_reach list with merged-out segments
                        chop_reach.extend(chop)

                    # if shortest segment is NOT the last segment in the reach - conduct a downstream merge
                    if (
                        reach_merged.Length.idxmin()
                        != reach_merged.tail(1).index.values[0]
                        and reach_merged.Length.min() < thresh
                    ):

                        # downstream merge
                        chop = []
                        reach_merged, chop = downstream_merge(
                            reach_merged, chop, thresh
                        )

                        # update chop_reach list with merged-out segments
                        chop_reach.extend(chop)

                # correct segment connections within reach
                reach_merged = correct_reach_connections(reach_merged)

                # update the greater network data set
                data_merged = update_network_data(
                    data_merged, rch, reach_merged, chop_reach, rconn
                )

                # update merged_segments list with merged-out segments
                merged_segments.extend(chop_reach)

    # create a qlateral destinations dictionary
    qlat_destinations = qlat_destination_compute(
        data_native, data_merged, merged_segments, pruned_segments, network_data
    )

    return data_merged, qlat_destinations


def main():

    # unpack command line arguments
    args = _handle_args()
    supernetwork = args.supernetwork
    threshold = args.threshold
    prune = args.prune
    snap = args.snap
    return_original = args.return_original

    # get network data
    print("Extracting and organizing supernetwork data")
    data, RouteLink, network_data = get_network_data(supernetwork)
    RouteLink = RouteLink.set_index(network_data["columns"]["key"])

    pruned_segs = []
    # prune headwaters
    if prune and snap:
        dirname = (
            "RouteLink_" + supernetwork + "_" + str(threshold) + "m_prune_snap_merge"
        )
        filename = (
            "RouteLink_"
            + supernetwork
            + "_"
            + str(threshold)
            + "m_prune_snap_merge.shp"
        )
        filename_cw = (
            "CrossWalk_"
            + supernetwork
            + "_"
            + str(threshold)
            + "m_prune_snap_merge.json"
        )

        print("Prune, snap, then merge:")
        print("pruning headwaters...")
        data_pruned = prune_headwaters(data, threshold, network_data)

        # identify pruned segments
        pruned_segs = list(np.setdiff1d(data.index, data_pruned.index))

        print("snapping junctions...")
        data_snapped = snap_junctions(data_pruned, threshold, network_data)

        print("merging segments...")
        data_merged, qlat_destinations = segment_merge(
            data, data_snapped, network_data, threshold, pruned_segs
        )

    if snap and not prune:
        dirname = "RouteLink_" + supernetwork + "_" + str(threshold) + "m_snap_merge"
        filename = (
            "RouteLink_" + supernetwork + "_" + str(threshold) + "m_snap_merge.shp"
        )
        filename_cw = (
            "CrossWalk_" + supernetwork + "_" + str(threshold) + "m_snap_merge.json"
        )

        print("Snap and merge:")
        print("snapping junctions...")
        data_snapped = snap_junctions(data, threshold, network_data)

        print("merging segments...")
        data_merged, qlat_destinations = segment_merge(
            data, data_snapped, network_data, threshold, pruned_segs
        )

    if not snap and prune:
        dirname = "RouteLink_" + supernetwork + "_" + str(threshold) + "m_prune_merge"
        filename = (
            "RouteLink_" + supernetwork + "_" + str(threshold) + "m_prune_merge.shp"
        )
        filename_cw = (
            "CrossWalk_" + supernetwork + "_" + str(threshold) + "m_prune_merge.json"
        )

        print("Prune and merge:")
        print("snapping junctions...")
        data_snapped = snap_junctions(data, threshold, network_data)

        print("merging segments...")
        data_merged, qlat_destinations = segment_merge(
            data, data_snapped, network_data, threshold, pruned_segs
        )

    if not snap and not prune:
        dirname = "RouteLink_" + supernetwork + "_" + str(threshold) + "m_merge"
        filename = "RouteLink_" + supernetwork + "_" + str(threshold) + "m_merge.nc"
        filename_cw = (
            "CrossWalk_" + supernetwork + "_" + str(threshold) + "m_merge.json"
        )
        print("Just merge:")
        print("merging segments...")
        data_merged, qlat_destinations = segment_merge(
            data, data, network_data, threshold, pruned_segs
        )

    # update RouteLink data
    RouteLink_edit = RouteLink.loc[data_merged.index.values]

    for (columnName, columnData) in data_merged.iteritems():
        RouteLink_edit.loc[:, columnName] = columnData

    for idx in RouteLink_edit.index:
        if RouteLink_edit.loc[idx, "to"] < 0:
            RouteLink_edit.loc[idx, "to"] = 0

    # convert RouteLink to geodataframe
    RouteLink_edit = gpd.GeoDataFrame(
        RouteLink_edit,
        geometry=gpd.points_from_xy(RouteLink_edit.lon, RouteLink_edit.lat),
    )

    # export merged data
    print(
        "exporting RouteLink file:",
        filename,
        "to",
        pathlib.Path(root, "test", "input", "geo", "Channels").resolve(),
    )

    dir_path = pathlib.Path(root, "test", "input", "geo", "Channels", dirname).resolve()
    if not pathlib.Path.is_dir(dir_path):
        pathlib.Path.mkdir(dir_path)

    # save RouteLink data as shapefile
    RouteLink_edit = RouteLink_edit.drop(columns=["time", "gages"])
    RouteLink_edit.to_file(pathlib.Path(dir_path, filename).resolve())

    # save cross walk as json
    print(
        "exporting CrossWalk file:",
        filename_cw,
        "to",
        pathlib.Path(root, "test", "input", "geo", "Channels").resolve(),
    )
    with open(pathlib.Path(dir_path, filename_cw).resolve(), "w") as outfile:
        json.dump(qlat_destinations, outfile)

    # export original data
    if return_original:
        dirname = "RouteLink_" + supernetwork
        filename = "RouteLink_" + supernetwork + ".shp"
        print(
            "exporting unmodified RouteLink file:",
            filename,
            "to",
            pathlib.Path(root, "test", "input", "geo", "Channels").resolve(),
        )

        dir_path = pathlib.Path(
            root, "test", "input", "geo", "Channels", dirname
        ).resolve()
        if not pathlib.Path.is_dir(dir_path):
            pathlib.Path.mkdir(dir_path)

        RouteLink_domain = RouteLink.loc[data.index.values]
        RouteLink_domain = gpd.GeoDataFrame(
            RouteLink_domain,
            geometry=gpd.points_from_xy(RouteLink_domain.lon, RouteLink_domain.lat),
        )

        RouteLink_domain = RouteLink_domain.drop(columns=["time", "gages"])
        RouteLink_domain.to_file(pathlib.Path(dir_path, filename).resolve())

        print("Number of segments in original RouteLink:", len(RouteLink_domain))
    print("Number of segments in modified RouteLink:", len(RouteLink_edit))


if __name__ == "__main__":
    main()
