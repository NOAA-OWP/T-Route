import networkbuilder


def recursive_reach_read_new(
    segment,
    order_iter,
    connections,
    network,
    reach_breaking_segments=set(),
    network_breaking_segments=set(),
    terminal_code=0,
    verbose=False,
    debuglevel=0,
):
    """
    This function improves on the `recursive_reach_read` function by removing
    some of the duplication in the tests.
    """

    csegment = segment
    reach = {}
    reach.update({"reach_tail": csegment})
    reach.update({"downstream_reach": connections[csegment]["downstream"]})
    segmentset = set()
    segmentlist = []  # Ordered Segment List bottom to top

    CONTINUE = True
    while CONTINUE:
        # TODO: Change the flag to use csegment with the junction list, headwaters, and network upstreams
        # TODO: ... such a change would remove the i == 0 tests and other
        # TODO: ... and other complications from this loop
        usegments = connections[csegment]["upstreams"]
        for i, usegment in enumerate(usegments):

            if usegment in network_breaking_segments:  # NETWORK
                if i == 0:
                    network["total_segment_count"] += 1
                    if debuglevel <= -3:
                        print("NEW NETWORK UPSTREAM: RECORD AND STOP")
                    if debuglevel <= -3:
                        print(f"csegment {csegment} --> usegment {usegment}")
                    if debuglevel <= -3:
                        print(f"receiving segment found at {csegment}")
                    if debuglevel <= -3:
                        print(
                            f"segs at csegment {csegment}: {network['total_segment_count']}"
                        )
                segmentset.add(csegment)
                if i == 0:
                    segmentlist.append(csegment)  # Ordered Segment List bottom to top
                reach.update({"reach_head": csegment})
                reach.update({"seqorder": order_iter})
                if order_iter == 0:
                    network.update(
                        {"terminal_reach": csegment}
                    )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                network.update(
                    {
                        "maximum_reach_seqorder": max(
                            network["maximum_reach_seqorder"], order_iter
                        )
                    }
                )
                reach.update({"segments": segmentset})
                reach.update(
                    {"segments_list": segmentlist}
                )  # Ordered Segment List bottom to top
                network["all_segments"].update(segmentset)
                network["reaches"].update({csegment: reach})
                network["receiving_reaches"].add(csegment)
                CONTINUE = False

            elif usegment in reach_breaking_segments:  # JUNCTION
                if i == 0:
                    network["total_segment_count"] += 1
                    if debuglevel <= -3:
                        print(
                            f"junction found at {csegment} with upstreams {usegments}"
                        )
                    if debuglevel <= -3:
                        print(
                            f"segs at csegment {csegment}: {network['total_segment_count']}"
                        )
                segmentset.add(csegment)
                if i == 0:
                    segmentlist.append(csegment)  # Ordered Segment List
                reach.update({"reach_head": csegment})
                reach.update({"seqorder": order_iter})
                if order_iter == 0:
                    network.update(
                        {"terminal_reach": csegment}
                    )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                network.update(
                    {
                        "maximum_reach_seqorder": max(
                            network["maximum_reach_seqorder"], order_iter
                        )
                    }
                )
                reach.update({"segments": segmentset})
                reach.update(
                    {"segments_list": segmentlist}
                )  # Ordered Segment List bottom to top
                network["all_segments"].update(segmentset)
                network["reaches"].update({csegment: reach})
                if i == 0:
                    network["total_junction_count"] += 1  # the Terminal Segment
                network["junctions"].add(csegment)
                if debuglevel <= -3:
                    print("JUNCTION UPSTREAM: CALL RECURSION")
                if debuglevel <= -3:
                    print(f"csegment {csegment} --> usegment {usegment}")
                recursive_reach_read_new(
                    usegment,
                    order_iter + 1,
                    connections,
                    network,
                    reach_breaking_segments,
                    network_breaking_segments,
                    terminal_code=terminal_code,
                    verbose=verbose,
                    debuglevel=debuglevel,
                )
                CONTINUE = False

            elif i == 0:
                if usegment == terminal_code:  # HEADWATERS
                    if debuglevel <= -3:
                        print("HEADWATER UPSTREAM: RECORD AND STOP")
                    if debuglevel <= -3:
                        print(f"csegment {csegment} --> usegment {usegment}")
                    if debuglevel <= -3:
                        print(f"headwater found at {csegment}")
                    network["total_segment_count"] += 1
                    if debuglevel <= -3:
                        print(
                            f"segs at csegment {csegment}: {network['total_segment_count']}"
                        )
                    segmentset.add(csegment)
                    segmentlist.append(csegment)  # Ordered Segment List bottom to top
                    reach.update({"reach_head": csegment})
                    reach.update({"seqorder": order_iter})
                    if order_iter == 0:
                        network.update(
                            {"terminal_reach": csegment}
                        )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                    network.update(
                        {
                            "maximum_reach_seqorder": max(
                                network["maximum_reach_seqorder"], order_iter
                            )
                        }
                    )
                    reach.update({"segments": segmentset})
                    reach.update(
                        {"segments_list": segmentlist}
                    )  # Ordered Segment List bottom to top
                    network["all_segments"].update(segmentset)
                    network["reaches"].update({csegment: reach})
                    network["headwater_reaches"].add(csegment)
                    CONTINUE = False

                else:
                    if debuglevel <= -3:
                        print("REGULAR SEGMENT UPSTREAM: proceeding upstream")
                    if debuglevel <= -3:
                        print(f"csegment {csegment} --> usegment {usegment}")
                    network["total_segment_count"] += 1
                    if debuglevel <= -3:
                        print(
                            f"segs at csegment {csegment}: {network['total_segment_count']}"
                        )
                    # the terminal code will indicate a headwater
                    if debuglevel <= -4:
                        print(usegments)
                    segmentset.add(csegment)
                    segmentlist.append(csegment)  # Ordered Segment List
                    # (csegment,) = usegments
                    # usegments = connections[csegment]['upstreams']
                csegment = usegment


def recursive_reach_read(
    segment,
    order_iter,
    connections,
    network,
    reach_breaking_segments=set(),
    network_breaking_segments=set(),
    terminal_code=0,
    verbose=False,
    debuglevel=0,
):
    """
    This function now uses only single segment
    The DEPRECATED function below used the list of upstream segments as an input.
    """

    csegment = segment
    reach = {}
    reach.update({"reach_tail": csegment})
    reach.update({"downstream_reach": connections[csegment]["downstream"]})
    segmentset = set()
    segmentlist = []  # Ordered Segment List bottom to top
    while True:
        usegments = connections[csegment]["upstreams"]
        if len(usegments) > 1:
            if debuglevel <= -3:
                print(f"junction found at {csegment} with upstreams {usegments}")
            network["total_segment_count"] += 1
            if debuglevel <= -3:
                print(f"segs at csegment {csegment}: {network['total_segment_count']}")
            segmentset.add(csegment)
            segmentlist.append(csegment)  # Ordered Segment List
            reach.update({"reach_head": csegment})
            reach.update({"seqorder": order_iter})
            if order_iter == 0:
                network.update(
                    {"terminal_reach": csegment}
                )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
            network.update(
                {
                    "maximum_reach_seqorder": max(
                        network["maximum_reach_seqorder"], order_iter
                    )
                }
            )
            reach.update({"segments": segmentset})
            reach.update(
                {"segments_list": segmentlist}
            )  # Ordered Segment List bottom to top
            network["all_segments"].update(segmentset)
            network["reaches"].update({csegment: reach})
            network["total_junction_count"] += 1  # the Terminal Segment
            network["junctions"].add(csegment)
            for usegment in usegments:
                # import pdb; pdb.set_trace()
                if usegment in network_breaking_segments:  # NETWORK
                    if debuglevel <= -3:
                        print("NEW NETWORK UPSTREAM: RECORD AND STOP")
                    if debuglevel <= -3:
                        print(f"csegment {csegment} --> usegment {usegment}")
                    network["receiving_reaches"].add(csegment)
                elif usegment in reach_breaking_segments:  # JUNCTION
                    if debuglevel <= -3:
                        print("JUNCTION UPSTREAM: CALL RECURSION")
                    if debuglevel <= -3:
                        print(f"csegment {csegment} --> usegment {usegment}")
                    recursive_reach_read(
                        usegment,
                        order_iter + 1,
                        connections,
                        network,
                        reach_breaking_segments,
                        network_breaking_segments,
                        terminal_code=terminal_code,
                        verbose=verbose,
                        debuglevel=debuglevel,
                    )
            break
        else:
            (usegment,) = usegments
            if usegment in network_breaking_segments:  # NETWORK
                if debuglevel <= -3:
                    print("NEW NETWORK UPSTREAM: RECORD AND STOP")
                if debuglevel <= -3:
                    print(f"csegment {csegment} --> usegment {usegment}")
                if debuglevel <= -3:
                    print(f"receiving segment found at {csegment}")
                network["total_segment_count"] += 1
                if debuglevel <= -3:
                    print(
                        f"segs at csegment {csegment}: {network['total_segment_count']}"
                    )
                segmentset.add(csegment)
                segmentlist.append(csegment)  # Ordered Segment List bottom to top
                reach.update({"reach_head": csegment})
                reach.update({"seqorder": order_iter})
                if order_iter == 0:
                    network.update(
                        {"terminal_reach": csegment}
                    )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                network.update(
                    {
                        "maximum_reach_seqorder": max(
                            network["maximum_reach_seqorder"], order_iter
                        )
                    }
                )
                reach.update({"segments": segmentset})
                reach.update(
                    {"segments_list": segmentlist}
                )  # Ordered Segment List bottom to top
                network["all_segments"].update(segmentset)
                network["reaches"].update({csegment: reach})
                network["receiving_reaches"].add(csegment)
                break
            if usegment == terminal_code:  # HEADWATERS
                if debuglevel <= -3:
                    print("HEADWATER UPSTREAM: RECORD AND STOP")
                if debuglevel <= -3:
                    print(f"csegment {csegment} --> usegment {usegment}")
                if debuglevel <= -3:
                    print(f"headwater found at {csegment}")
                network["total_segment_count"] += 1
                if debuglevel <= -3:
                    print(
                        f"segs at csegment {csegment}: {network['total_segment_count']}"
                    )
                segmentset.add(csegment)
                segmentlist.append(csegment)  # Ordered Segment List bottom to top
                reach.update({"reach_head": csegment})
                reach.update({"seqorder": order_iter})
                if order_iter == 0:
                    network.update(
                        {"terminal_reach": csegment}
                    )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                network.update(
                    {
                        "maximum_reach_seqorder": max(
                            network["maximum_reach_seqorder"], order_iter
                        )
                    }
                )
                reach.update({"segments": segmentset})
                reach.update(
                    {"segments_list": segmentlist}
                )  # Ordered Segment List bottom to top
                network["all_segments"].update(segmentset)
                network["reaches"].update({csegment: reach})
                network["headwater_reaches"].add(csegment)
                break
            else:
                if debuglevel <= -3:
                    print("REGULAR SEGMENT UPSTREAM: proceeding upstream")
                if debuglevel <= -3:
                    print(f"csegment {csegment} --> usegment {usegment}")
                network["total_segment_count"] += 1
                if debuglevel <= -3:
                    print(
                        f"segs at csegment {csegment}: {network['total_segment_count']}"
                    )
                # the terminal code will indicate a headwater
                if debuglevel <= -4:
                    print(usegments)
                segmentset.add(csegment)
                segmentlist.append(csegment)  # Ordered Segment List
                # (csegment,) = usegments
                # usegments = connections[csegment]['upstreams']
            csegment = usegment


def DEPRECATED_recursive_junction_read(
    segments,
    order_iter,
    connections,
    network,
    terminal_code=0,
    verbose=False,
    debuglevel=0,
):

    for segment in segments:
        csegment = segment
        if 1 == 1:
            # try:
            reach = {}
            reach.update({"reach_tail": csegment})
            reach.update({"downstream_reach": connections[csegment]["downstream"]})
            segmentset = set()
            segmentlist = []  # Ordered Segment List bottom to top
            usegments = connections[segment]["upstreams"]
            while True:
                if usegments == {terminal_code}:  # HEADWATERS
                    if debuglevel <= -3:
                        print(f"headwater found at {csegment}")
                    network["total_segment_count"] += 1
                    if debuglevel <= -3:
                        print(
                            f"segs at csegment {csegment}: {network['total_segment_count']}"
                        )
                    segmentset.add(csegment)
                    segmentlist.append(csegment)  # Ordered Segment List bottom to top
                    reach.update({"reach_head": csegment})
                    reach.update({"seqorder": order_iter})
                    if order_iter == 0:
                        network.update(
                            {"terminal_reach": csegment}
                        )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                    network.update(
                        {
                            "maximum_reach_seqorder": max(
                                network["maximum_reach_seqorder"], order_iter
                            )
                        }
                    )
                    reach.update({"segments": segmentset})
                    reach.update(
                        {"segments_list": segmentlist}
                    )  # Ordered Segment List bottom to top
                    network["all_segments"].update(segmentset)
                    network["reaches"].update({csegment: reach})
                    network["headwater_reaches"].add(csegment)
                    break
                elif len(usegments) >= 2:  # JUNCTIONS
                    if debuglevel <= -3:
                        print(
                            f"junction found at {csegment} with upstreams {usegments}"
                        )
                    network["total_segment_count"] += 1
                    if debuglevel <= -3:
                        print(
                            f"segs at csegment {csegment}: {network['total_segment_count']}"
                        )
                    segmentset.add(csegment)
                    segmentlist.append(csegment)  # Ordered Segment List
                    reach.update({"reach_head": csegment})
                    reach.update({"seqorder": order_iter})
                    if order_iter == 0:
                        network.update(
                            {"terminal_reach": csegment}
                        )  # ; import pdb; pdb.set_trace() #TODO: FIX THIS; SEEMS FRAGILE
                    network.update(
                        {
                            "maximum_reach_seqorder": max(
                                network["maximum_reach_seqorder"], order_iter
                            )
                        }
                    )
                    reach.update({"segments": segmentset})
                    reach.update(
                        {"segments_list": segmentlist}
                    )  # Ordered Segment List bottom to top
                    network["all_segments"].update(segmentset)
                    network["reaches"].update({csegment: reach})
                    network["total_junction_count"] += 1  # the Terminal Segment
                    network["junctions"].add(csegment)
                    DEPRECATED_recursive_junction_read(
                        usegments,
                        order_iter + 1,
                        connections,
                        network,
                        terminal_code=terminal_code,
                        verbose=verbose,
                        debuglevel=debuglevel,
                    )
                    break
                network["total_segment_count"] += 1
                if debuglevel <= -3:
                    print(
                        f"segs at csegment {csegment}: {network['total_segment_count']}"
                    )
                # the terminal code will indicate a headwater
                if debuglevel <= -4:
                    print(usegments)
                segmentset.add(csegment)
                segmentlist.append(csegment)  # Ordered Segment List
                (csegment,) = usegments
                usegments = connections[csegment]["upstreams"]


# TODO: re-implement as parallel process on terminal segments
# TODO: NOTE: this will be a little more complicated now that the global `connections`
# TODO: NOTE: object is not available from the calling function. Probably requires a class.
def network_trace(
    terminal_segment=None,
    order_iter=0,
    connections=None,
    reach_breaking_segments=None,
    network_breaking_segments=None,
    terminal_code=0,
    verbose=False,
    debuglevel=0,
):

    network = {}
    us_length_total = 0

    if debuglevel <= -2:
        print(f"\ntraversing upstream on network {terminal_segment}:")
    # try:
    if 1 == 1:
        network.update({"total_segment_count": 0})
        network.update({"total_junction_count": 0})
        network.update({"network_seqorder": 0})
        network.update({"maximum_reach_seqorder": 0})
        network.update({"junctions": set()})
        network.update({"headwater_reaches": set()})
        network.update({"all_segments": set()})
        network.update(
            {"receiving_reaches": set()}
        )  # Reaches that are downstream of another network
        network.update({"tailwater_reach": set()})
        network.update({"reaches": {}})
        recursive_reach_read_new(
            segment=terminal_segment,
            order_iter=order_iter,
            connections=connections,
            network=network,
            reach_breaking_segments=reach_breaking_segments,
            network_breaking_segments=network_breaking_segments,
            terminal_code=terminal_code,
            verbose=verbose,
            debuglevel=debuglevel,
        )
        if debuglevel <= -1:
            print(f"junctions: {network['total_junction_count']}")
        if debuglevel <= -1:
            print(f"segments: {network['total_segment_count']}")
    # except Exception as exc:
    #     print(exc)
    # TODO: compute upstream length as a surrogate for the routing computation
    return {terminal_segment: network, "upstream_length": us_length_total}


def compose_networks(
    supernetwork_values=None,
    terminal_code=0,
    break_network_at_waterbodies=False,
    debuglevel=0,
    verbose=False,
    showtiming=False,
):

    terminal_segments = supernetwork_values[4]
    circular_segments = supernetwork_values[6]
    confluence_segment_set = supernetwork_values[11]
    terminal_segments_super = terminal_segments - circular_segments
    waterbody_outlet_set = supernetwork_values[14]
    waterbody_upstreams_set = supernetwork_values[15]
    waterbody_breaking_segments = set()
    if break_network_at_waterbodies:
        waterbody_breaking_segments = waterbody_outlet_set.union(
            waterbody_upstreams_set
        )
        terminal_segments_super = terminal_segments_super.union(
            waterbody_breaking_segments
        )

    connections = supernetwork_values[0]

    networks = {terminal_segment: {} for terminal_segment in terminal_segments_super}

    if verbose:
        print("verbose output")
    if verbose:
        print(f"number of Independent Networks to be analyzed is {len(networks)}")
    if verbose:
        print(f"debuglevel is {debuglevel}")

    init_order = 0
    for terminal_segment, network in networks.items():
        network.update(
            network_trace(
                terminal_segment=terminal_segment,
                order_iter=init_order,
                connections=connections,
                terminal_code=terminal_code,
                reach_breaking_segments=confluence_segment_set,
                network_breaking_segments=waterbody_breaking_segments,
                verbose=verbose,
                debuglevel=debuglevel,
            )[terminal_segment]
        )

        upstream_reaches = network["headwater_reaches"]
        if break_network_at_waterbodies:
            upstream_reaches = upstream_reaches | network["receiving_reaches"]

        up_reaches = networkbuilder.get_up_connections(
            network["reaches"],
            terminal_code,
            upstream_reaches,
            {network["terminal_reach"]},
            r"upstream_reaches",
            r"downstream_reach",
            verbose=False
            # , verbose = verbose
            ,
            debuglevel=debuglevel,
        )

        if debuglevel <= -2:
            if debuglevel <= -2:
                print(f"terminal_segment: {terminal_segment}")
            if debuglevel <= -3:
                for k, v in network.items():
                    if type(v) is dict:
                        print(f"{k}:")
                        for k1, v1 in v.items():
                            print(f"\t{k1}: {v1}")
                    else:
                        print(f"{k}: {v}")
    if debuglevel <= -1:
        print(f"Number of networks in the Supernetwork: {len(networks.items())}")

    return networks


def order_networks(connections, networks, tailwaters):
    """Assumes that every tailwater is a network -- but there may
    be many networks that are upstream networks and therefore
    not tailwaters."""

    curr_seqorder = 0
    recursive_order_networks(connections, networks, tailwaters, curr_seqorder)


def recursive_order_networks(connections, networks, tailwaters, curr_seqorder):
    for tailwater in tailwaters:
        network = networks.get(tailwater, None)
        if network:
            network["network_seqorder"] = curr_seqorder
            for rr in network["receiving_reaches"]:
                recursive_order_networks(
                    connections,
                    networks,
                    connections[rr]["upstreams"],
                    curr_seqorder + 1,
                )
