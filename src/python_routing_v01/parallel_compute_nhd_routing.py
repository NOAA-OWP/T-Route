## Basic imports
import sys
import os
import time

## Parallel execution
import multiprocessing

# WARNING: These global declarations cause the parallel implementation to
# crash when executed on Windows
connections = None
networks = None
flowdepthvel = None
num_processes = None

from sys import platform

if platform == "linux" or platform == "linux2":
    pass
elif platform == "darwin":
    pass
elif platform == "win32":
    print(
        "The parallel version of compute_nhd_routing.py will not execute as currently"
    )
    print("written due to the lack of a fork() capability in the windows OS.")
    print("For parallel execution, please us a *nix OS.")
    print("\nexiting...")
    sys.exit()
    # Some useful references:
    # https://stackoverflow.com/questions/985281/what-is-the-closest-thing-windows-has-to-fork/985525#985525
    # https://stackoverflow.com/questions/8220108/how-do-i-check-the-operating-system-in-python
    # https://stackoverflow.com/questions/6596617/python-multiprocess-diff-between-windows-and-linux

ENV_IS_CL = False
if ENV_IS_CL:
    root = "/content/wrf_hydro_nwm_public/trunk/NDHMS/dynamic_channel_routing/"
elif not ENV_IS_CL:
    sys.setrecursionlimit(4000)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root, r"src", r"python_framework"))
fortran_source_dir = os.path.join(
    root, r"src", r"fortran_routing", r"mc_pylink_v00", r"MC_singleRch_singleTS"
)
sys.path.append(fortran_source_dir)
from mc_singleRch_SingleTStep import compute_mc_reach_up2down

# import mc_sc_stime as mc
# print(fortran_source_dir)

## network and reach utilities
import nhd_network_utilities as nnu
import nhd_reach_utilities as nru

## Muskingum Cunge
import numpy as np


def compute_network(
    terminal_segment=None,
    network=None,
    supernetwork_data=None,
    nts=None
    # , connections = None
    ,
    verbose=False,
    debuglevel=0,
):

    global connections
    global flowdepthvel

    if verbose:
        print(
            f"Executing simulation on network {terminal_segment} beginning with streams of order {network['maximum_reach_seqorder']}"
        )

    ordered_reaches = {}
    reach_flowdepthvel = {}
    for head_segment, reach in network["reaches"].items():
        if reach["seqorder"] not in ordered_reaches:
            # ordered_reaches.update({reach['seqorder']:[]}) #TODO: Should this be a set/dictionary?
            ordered_reaches.update(
                {reach["seqorder"]: {}}
            )  # TODO: Should this be a set/dictionary?
        # ordered_reaches[reach['seqorder']].append([head_segment
        # , reach
        # ])
        ordered_reaches[reach["seqorder"]].update({head_segment: reach})

        # initialize flowdepthvel dict
        reach_flowdepthvel.update({head_segment: {}})
        reach_flowdepthvel[head_segment].update(
            {
                seg: {
                    "flow": {"prev": 0, "curr": 0},
                    "depth": {"prev": -999, "curr": 0},
                    "vel": {"prev": 0, "curr": 0},
                    "qlat": {"prev": 0, "curr": 0},
                }
                for seg in reach["segments"]
            }
        )
        if 1 == 0:  # TODO: Implement this alternate method -- a more shallow dictionary
            # creates a much faster search in this highly utilized object.
            flow_prev = flow_curr = 0
            depth_prev = depth_curr = 0
            vel_prev = vel_curr = 0
            qlat_prev = qlat_curr = 0
            reach_flowdepthvel[head_segment].update(
                {
                    seg: [
                        flow_prev,
                        flow_curr,
                        depth_prev,
                        depth_curr,
                        vel_prev,
                        vel_curr,
                        qlat_prev,
                        qlat_curr,
                    ]
                    for seg in reach["segments"]
                }
            )
        # ordered_reaches[order][head_segment].update({'reach_connections':{key:connection for key, connection in connections.items() if key in reach['segments']}})
        # ordered_reaches[reach['seqorder']][head_segment].update({'reach_connections':{key:connections[key] for key in connections.keys() & reach['segments']}})
        ordered_reaches[reach["seqorder"]][head_segment].update(
            {"reach_connections": {key: connections[key] for key in reach["segments"]}}
        )

    for ts in range(0, nts):
        # print(f'timestep: {ts}\n')

        for order in range(network["maximum_reach_seqorder"], -1, -1):
            for head_segment, reach in ordered_reaches[order].items():
                # print(f'{{{head_segment}}}:{reach}')

                # TODO: Add a flag here to switch between methods
                compute_method = "byreach"  # Other options: 'bysegment'
                if compute_method == "byreach":
                    # upstream flow per reach
                    qup_tmp = 0
                    # import pdb; pdb.set_trace()
                    if reach["upstream_reaches"] == {
                        supernetwork_data["terminal_code"]
                    }:  # Headwaters
                        qup_tmp = 0.0  # no flows
                    else:  # Loop over upstream reaches
                        # for us in reach['upstream_reaches']:
                        for us in reach["upstream_reaches"]:
                            # qup_tmp += flowdepthvel[network['reaches'][us]['reach_tail']]['flow']['curr']
                            # import pdb; pdb.set_trace()
                            qup_tmp += reach_flowdepthvel[us][
                                network["reaches"][us]["reach_tail"]
                            ]["flow"]["curr"]

                    for current_segment in reach["segments"]:
                        # add some flow
                        reach_flowdepthvel[head_segment][current_segment]["qlat"][
                            "curr"
                        ] = (
                            ts + 1
                        ) * 10.0  # lateral flow per segment

                        reach_flowdepthvel[head_segment][current_segment]["flow"][
                            "prev"
                        ] = reach_flowdepthvel[head_segment][current_segment]["flow"][
                            "curr"
                        ]
                        reach_flowdepthvel[head_segment][current_segment]["depth"][
                            "prev"
                        ] = reach_flowdepthvel[head_segment][current_segment]["depth"][
                            "curr"
                        ]
                        reach_flowdepthvel[head_segment][current_segment]["vel"][
                            "prev"
                        ] = reach_flowdepthvel[head_segment][current_segment]["vel"][
                            "curr"
                        ]
                        reach_flowdepthvel[head_segment][current_segment]["qlat"][
                            "prev"
                        ] = reach_flowdepthvel[head_segment][current_segment]["qlat"][
                            "curr"
                        ]

                    reach_flowdepthvel.update(
                        compute_mc_reach_up2down(
                            head_segment=head_segment,
                            reach=reach
                            # , network = network
                            ,
                            reach_connections=ordered_reaches[order][head_segment][
                                "reach_connections"
                            ],
                            reach_flowdepthvel=reach_flowdepthvel[head_segment],
                            upstream_inflow=qup_tmp,
                            supernetwork_data=supernetwork_data,
                            ts=ts,
                            verbose=verbose,
                            debuglevel=debuglevel,
                        )
                    )
                    # print(f'timestep: {ts} {flowdepthvel}')
                    # print(f'{head_segment} {flowdepthvel[head_segment]}')


def compute_network_parallel_cluster(
    networks=None,
    supernetwork_data=None,
    nts=None
    # , connections = None
    ,
    verbose=False,
    debuglevel=0,
):

    global connections
    global flowdepthvel
    global num_processes

    overall_max = -1
    for terminal_segment, network in networks.items():
        overall_max = max(network["maximum_reach_seqorder"], overall_max)

    if debuglevel <= -1:
        print(
            f"Executing simulation for all large network beginning with maximum order {overall_max}"
        )

    ordered_reaches = {}
    reach_flowdepthvel = {}
    # ordered_reaches.update({overall_max:[]})
    # import pdb; pdb.set_trace()
    for terminal_segment, network in networks.items():
        for head_segment, reach in network["reaches"].items():
            if reach["seqorder"] not in ordered_reaches:
                ordered_reaches.update(
                    {reach["seqorder"]: {}}
                )  # TODO: Should this be a set/dictionary?
            ordered_reaches[reach["seqorder"]].update(
                {head_segment: {"network": terminal_segment, "reach": reach}}
            )

            # initialize flowdepthvel dict
            reach_flowdepthvel.update({head_segment: {}})
            reach_flowdepthvel[head_segment].update(
                {
                    seg: {
                        "flow": {"prev": 0, "curr": 0},
                        "depth": {"prev": -999, "curr": 0},
                        "vel": {"prev": 0, "curr": 0},
                        "qlat": {"prev": 0, "curr": 0},
                    }
                    for seg in reach["segments"]
                }
            )
            ordered_reaches[reach["seqorder"]][head_segment].update(
                {
                    "reach_connections": {
                        key: connections[key] for key in reach["segments"]
                    }
                }
            )

    # exit()
    with multiprocessing.Pool() as netpool:

        num_processes = netpool._processes

        #        for order in range(overall_max,-1,-1):
        #            for head_segment, network_reach in ordered_reaches[order].items():
        #                #print(f'{{{head_segment}}}:{reach}')
        #                reach = network_reach['reach']
        #
        #                #TODO: Add a flag here to switch between methods
        #                compute_method = 'byreach' # Other options: 'bysegment'
        #                if compute_method == 'byreach':
        #                    pass

        for ts in range(0, nts):
            # print(f'timestep: {ts}\n')

            for order in range(overall_max, -1, -1):
                parallel_arglist = []
                # print(order)
                # import pdb; pdb.set_trace()
                for head_segment, network_reach in ordered_reaches[order].items():
                    # print(f'{{{head_segment}}}:{reach}')
                    reach = network_reach["reach"]
                    network = networks[network_reach["network"]]

                    # TODO: Add a flag here to switch between methods
                    compute_method = "byreach"  # Other options: 'bysegment'
                    if compute_method == "byreach":
                        # upstream flow per reach
                        qup_tmp = 0
                        # import pdb; pdb.set_trace()
                        if reach["upstream_reaches"] == {
                            supernetwork_data["terminal_code"]
                        }:  # Headwaters
                            qup_tmp = 0.0  # no flows
                        else:  # Loop over upstream reaches
                            # for us in reach['upstream_reaches']:
                            for us in reach["upstream_reaches"]:
                                # qup_tmp += flowdepthvel[network['reaches'][us]['reach_tail']]['flow']['curr']
                                # import pdb; pdb.set_trace()
                                qup_tmp += reach_flowdepthvel[us][
                                    network["reaches"][us]["reach_tail"]
                                ]["flow"]["curr"]

                        # reach_connections = {key:connection for key, connection in connections.items() if key in reach['segments']}
                        for current_segment in reach["segments"]:
                            # add some flow
                            reach_flowdepthvel[head_segment][current_segment]["qlat"][
                                "curr"
                            ] = (
                                ts + 1
                            ) * 10.0  # lateral flow per segment

                            reach_flowdepthvel[head_segment][current_segment]["flow"][
                                "prev"
                            ] = reach_flowdepthvel[head_segment][current_segment][
                                "flow"
                            ][
                                "curr"
                            ]
                            reach_flowdepthvel[head_segment][current_segment]["depth"][
                                "prev"
                            ] = reach_flowdepthvel[head_segment][current_segment][
                                "depth"
                            ][
                                "curr"
                            ]
                            reach_flowdepthvel[head_segment][current_segment]["vel"][
                                "prev"
                            ] = reach_flowdepthvel[head_segment][current_segment][
                                "vel"
                            ][
                                "curr"
                            ]
                            reach_flowdepthvel[head_segment][current_segment]["qlat"][
                                "prev"
                            ] = reach_flowdepthvel[head_segment][current_segment][
                                "qlat"
                            ][
                                "curr"
                            ]
                        # import pdb; pdb.set_trace()
                        parallel_arglist.append(
                            (
                                head_segment,
                                ordered_reaches[order][head_segment]["reach"],
                                ordered_reaches[order][head_segment][
                                    "reach_connections"
                                ],
                                reach_flowdepthvel[head_segment],
                                qup_tmp,
                                supernetwork_data,
                                ts,
                            )
                        )

                # print(f'Time: {ts} Execution Args for order {order}: {parallel_arglist}')
                if debuglevel <= -1:
                    print(
                        f"Time: {ts} Executing simulation for {len(parallel_arglist)} large network reaches of order {order}"
                    )
                results = netpool.starmap(compute_mc_reach_up2down, parallel_arglist)
                for result in results:
                    # print(result)
                    reach_flowdepthvel.update(result)


# ### Pseudocode
#
# ```
# Call Compute Network
#     for each reach in the network
#         Call compute reach
#             For each segment in the reach
#                 Import the mc object/module
#                 Call prepare segment
#                     Populate the Mc segment array with the individual segment properties
#                     Obtain and set any lateral inflows
#                     obtain and set the upstream and downstream (requrires logic to deal with junctions or boundaries at headwaters)
#                 With the populated arrays, execute MC for the reach
# ```
#


def read_segments():
    pass


def prepare_segments():
    pass


def handle_junctions():
    pass


def get_upstream_inflow():
    pass


def get_lateral_inflow():
    pass


def compute_junction_downstream():
    pass


def main():

    global connections
    global networks
    global flowdepthvel

    verbose = True
    debuglevel = 0
    showtiming = True

    test_folder = os.path.join(root, r"test")
    geo_input_folder = os.path.join(test_folder, r"input", r"geo")

    # TODO: Make these commandline args
    supernetwork = "Pocono_TEST1"
    """##NHD Subset (Brazos/Lower Colorado)"""
    # supernetwork = 'Brazos_LowerColorado_ge5'
    """##NWM CONUS Mainstems"""
    # supernetwork = 'Mainstems_CONUS'
    """These are large -- be careful"""
    # supernetwork = 'CONUS_FULL_RES_v20'
    # supernetwork = 'CONUS_Named_Streams' #create a subset of the full resolution by reading the GNIS field
    # supernetwork = 'CONUS_Named_combined' #process the Named streams through the Full-Res paths to join the many hanging reaches

    if verbose:
        print("creating supernetwork connections set")
    if showtiming:
        start_time = time.time()
    # STEP 1
    supernetwork_data, supernetwork_values = nnu.set_networks(
        supernetwork=supernetwork,
        geo_input_folder=geo_input_folder,
        verbose=False
        # , verbose = verbose
        ,
        debuglevel=debuglevel,
    )
    if verbose:
        print("supernetwork connections set complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    # STEP 2
    if showtiming:
        start_time = time.time()
    if verbose:
        print("organizing connections into networks and reaches ...")
    networks = nru.compose_networks(
        supernetwork_values,
        verbose=False
        # , verbose = verbose
        ,
        debuglevel=debuglevel,
        showtiming=showtiming,
    )
    if verbose:
        print("reach organization complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    # STEP 3
    if showtiming:
        start_time = time.time()
    executiontype = "parallel"  # 'serial'

    connections = supernetwork_values[0]

    # number_of_time_steps = 10 #
    number_of_time_steps = 50  #
    # number_of_time_steps = 1440 # number of timestep = 1140 * 60(model timestep) = 86400 = day

    # initialize flowdepthvel dict
    flowdepthvel = {
        connection: {
            "flow": np.zeros(number_of_time_steps + 1),
            "depth": np.zeros(number_of_time_steps + 1),
            "vel": np.zeros(number_of_time_steps + 1),
            "qlat": np.zeros(number_of_time_steps + 1),
        }
        for connection in connections
    }

    if executiontype == "serial":
        if verbose:
            print("executing serial computation on ordered reaches ...")

        for terminal_segment, network in networks.items():
            if showtiming:
                network_start_time = time.time()
            compute_network(
                terminal_segment=terminal_segment,
                network=network,
                supernetwork_data=supernetwork_data,
                nts=number_of_time_steps
                # , connections = connections
                ,
                verbose=False
                # , verbose = verbose
                ,
                debuglevel=debuglevel,
            )

            if verbose:
                print(f"{terminal_segment} completed")
            if showtiming:
                print("... in %s seconds." % (time.time() - network_start_time))

    elif executiontype == "parallel":

        # parallel_split = -1 # -1 turns off the splitting and runs everything through the lumped execution
        parallel_split = 10000  # -1 turns off the splitting and runs everything through the lumped execution

        # STEP 3a -- Large Networks
        # TODO: fix this messaging -- we are less specifically concerned with whether these are large networks and more interested in the general idea of grouping.
        if verbose:
            print(
                f"executing computation on ordered reaches for networks of order greater than {parallel_split} ..."
            )

        parallel_network_cluster = {
            terminal_segment: network
            for terminal_segment, network in networks.items()
            if network["maximum_reach_seqorder"] > parallel_split
        }
        # print(networks)
        compute_network_parallel_cluster(
            networks=parallel_network_cluster,
            supernetwork_data=supernetwork_data,
            nts=number_of_time_steps
            # , connections = connections
            # , verbose = False
            ,
            verbose=verbose,
            debuglevel=debuglevel,
        )
        if verbose:
            print(
                f"ordered reach computation complete for networks of order greater than {parallel_split}"
            )
        if verbose:
            print(
                f"calculation completed for the following networks (as labelled by their terminal segments)\n{list(parallel_network_cluster.keys())} completed"
            )
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
        if showtiming:
            print(f"... with {num_processes} cores")

        ##STEP 3b -- Small Networks
        # if parallel_split >= 0: print(r'DO NOT RUN WITH `parallel_split` >= 0')
        parallel_network_separate = {
            terminal_segment: network
            for terminal_segment, network in networks.items()
            if network["maximum_reach_seqorder"] <= parallel_split
        }

        if verbose:
            print(f"executing parallel computation on ordered reaches .... ")
        # for terminal_segment, network in networks.items():
        #    print(terminal_segment, network)
        # print(tuple(([x for x in networks.keys()][i], [x for x in networks.values()][i]) for i in range(len(networks))))
        nslist = (
            [
                terminal_segment,
                network,
                supernetwork_data,  # TODO: This should probably be global...
                number_of_time_steps,
                False,
                debuglevel,
            ]
            for terminal_segment, network in parallel_network_separate.items()
        )
        with multiprocessing.Pool() as pool:
            results = pool.starmap(compute_network, nslist)
        if verbose:
            print(
                f"calculation completed for the following networks (as labelled by their terminal segments)\n{list(parallel_network_separate.keys())} completed"
            )

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))


if __name__ == "__main__":
    main()
