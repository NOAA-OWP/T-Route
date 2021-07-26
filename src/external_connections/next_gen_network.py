#!/usr/bin/env python
# coding: utf-8
# example usage: python next_gen_network.py -n ./data/coarse/flowpath_data.geojson -s cat-71 cat-42 cat-34 cat-26 cat-68 cat-67 cat-27 cat-52 -i ./data

"""This program reads in and modifies lateral flows from the next generation water modeling framework and feeds them into mc_reach.compute_network. 
This program presupposes that the mc_reach has been compiled prior to running."""
import os
import sys
import time
import numpy as np
import argparse
import pathlib
import pandas as pd
from functools import partial
from itertools import chain, islice
import next_gen_io
import json


def _handle_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n", "--supernetwork", help="Input catchment network", dest="supernetwork"
    )
    parser.add_argument(
        "-s",
        "--subset",
        help="Subset of catchments to process",
        dest="subset",
        nargs="+",
    )
    parser.add_argument(
        "-i", "--input", help="path to lateral flow inputs", dest="input"
    )
    return parser.parse_args()


root = pathlib.Path(".").resolve()
# FIXME remove these eventually
sys.path.append(r"../python_framework_v02")
sys.path.append(r"../python_routing_v02")
sys.path.append(r"../../test/input/next_gen")

test_folder = pathlib.Path(root, "test")

import troute.nhd_network as nhd_network
import troute.nhd_io as nhd_io
import mc_reach


def main():

    args = _handle_args()

    next_gen_input_folder = test_folder.joinpath("input", "next_gen")
    if args.input:
        next_gen_input_folder = pathlib.Path(args.input)

    # The following 2 values are currently hard coded for this test domain
    nts = 720  # number of timestep = 1140 * 60(model timestep) = 86400 = day
    dt_mc = 300.0  # time interval for MC

    # Currently tested on the Sugar Creek domain
    ngen_network_df = nhd_io.read_geopandas(args.supernetwork)
    if args.subset:
        ngen_network_df = ngen_network_df[
            ngen_network_df["realized_catchment"].isin(args.subset)
        ]

    # Create dictionary mapping each connection ID
    ngen_network_dict = dict(zip(ngen_network_df.id, ngen_network_df.toid))
    # ngen_network_dict = dict(zip(ngen_network_df.ID, ngen_network_df.toID))

    def node_key_func(x):
        return int(x[3:])

    # Extract the ID integer values
    waterbody_connections = {
        node_key_func(k): node_key_func(v) for k, v in ngen_network_dict.items()
    }

    # Convert dictionary connections to data frame and make ID column the index
    waterbody_df = pd.DataFrame.from_dict(
        waterbody_connections, orient="index", columns=["to"]
    )
    # Sort ID index column
    waterbody_df = waterbody_df.sort_index()

    waterbody_df = nhd_io.replace_downstreams(waterbody_df, "to", 0)

    connections = nhd_network.extract_connections(waterbody_df, "to")

    # Read and convert catchment lateral flows to format that can be processed by compute_network
    qlats = next_gen_io.read_catchment_lateral_flows(next_gen_input_folder)
    print(qlats)
    rconn = nhd_network.reverse_network(connections)

    subnets = nhd_network.reachable_network(rconn, check_disjoint=False)

    # read the routelink file
    nhd_routelink = nhd_io.read_netcdf("data/RouteLink_NHDPLUS.nc")
    nhd_routelink["dt"] = 300.0

    nhd_routelink.set_index("link", inplace=True)

    routelink_cols = {
        "downstream": "to",
        "dx": "Length",
        "n": "n",
        "ncc": "nCC",
        "s0": "So",
        "bw": "BtmWdth",
        "tw": "TopWdth",
        "twcc": "TopWdthCC",
        "waterbody": "NHDWaterbodyComID",
        "musk": "MusK",
        "musx": "MusX",
        "cs": "ChSlp",
    }

    routelink_cols = dict([(value, key) for key, value in routelink_cols.items()])

    nhd_routelink.rename(columns=routelink_cols, inplace=True)

    with open(next_gen_input_folder / "coarse/crosswalk.json") as f:
        crosswalk_data = json.load(f)
    waterbody_df["comid"] = waterbody_df.apply(
        lambda x: crosswalk_data["cat-" + str(x.name)]["outlet_COMID"], axis=1
    )

    waterbody_df = waterbody_df.join(nhd_routelink, on="comid", how="left")

    del nhd_routelink

    # initial conditions, assume to be zero
    # TO DO: Allow optional reading of initial conditions from WRF
    q0 = pd.DataFrame(
        0, index=waterbody_df.index, columns=["qu0", "qd0", "h0"], dtype="float32"
    )

    # Set types as float32
    waterbody_df = waterbody_df.astype(
        {
            "dt": "float32",
            "bw": "float32",
            "tw": "float32",
            "twcc": "float32",
            "dx": "float32",
            "n": "float32",
            "ncc": "float32",
            "cs": "float32",
            "s0": "float32",
        }
    )

    subreaches = {}

    for tw, net in subnets.items():
        path_func = partial(nhd_network.split_at_junction, net)
        subreaches[tw] = nhd_network.dfs_decomposition(net, path_func)

    results = []
    for twi, (tw, reach) in enumerate(subreaches.items(), 1):
        r = list(chain.from_iterable(reach))
        data_sub = waterbody_df.loc[
            r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
        ].sort_index()
        # data_sub = waterbody_df.loc[r, ['dt', 'bw', 'tw', 'twcc', 'dx', 'n', 'ncc', 'cs', 's0']]
        qlat_sub = qlats.loc[r].sort_index()
        q0_sub = q0.loc[r].sort_index()

        results.append(
            mc_reach.compute_network(
                nts,
                reach,
                subnets[tw],
                data_sub.index.values,
                data_sub.columns.values,
                data_sub.values,
                qlat_sub.values,
                q0_sub.values,
            )
        )

    fdv_columns = pd.MultiIndex.from_product(
        [range(nts), ["q", "v", "d"]]
    ).to_flat_index()
    flowveldepth = pd.concat(
        [pd.DataFrame(d, index=i, columns=fdv_columns) for i, d in results], copy=False
    )
    flowveldepth = flowveldepth.sort_index()
    outfile_base_name = (args.supernetwork).split(".")[0]
    flowveldepth.to_csv(f"{outfile_base_name}_mc_results.csv")
    print(flowveldepth)


if __name__ == "__main__":
    main()
