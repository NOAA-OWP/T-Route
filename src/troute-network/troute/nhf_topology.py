from collections import defaultdict
from itertools import chain
from typing import Optional

import pandas as pd
from troute.nhd_network import reverse_network


def build_downstream_connections(
    routing_flowpaths: pd.DataFrame,
    all_flowpaths: pd.DataFrame,
    terminal_nexus_ids: Optional[set[int]] = None,
) -> dict[int, list[int]]:
    """
    Build downstream connectivity dictionary mapping flowpath IDs to their downstream flowpath IDs.

    The NHF data model uses nexus points as connection nodes between flowpaths:
    - Flowpath A has dn_virtual_nex_id = X (A drains TO nexus X)
    - Flowpath B has up_virtual_nex_id = X (B receives FROM nexus X)
    - Therefore A -> B (A flows into B)

    Parameters
    ----------
    routing_flowpaths : pd.DataFrame
        DataFrame containing ONLY routing segment flowpaths (routing_segment == True).
        These are the flowpaths that will appear as keys in the connections dict.
        Must have columns: virtual_fp_id, dn_virtual_nex_id, up_virtual_nex_id
    all_flowpaths : pd.DataFrame
        DataFrame containing ALL virtual flowpaths (both routing and non-routing).
        Used to build the nexus-to-downstream-flowpath mapping.
        Must have columns: virtual_fp_id, dn_virtual_nex_id, up_virtual_nex_id
    terminal_nexus_ids : set[int], optional
        Set of nexus IDs that are terminal (no downstream flowpath).
        If provided, flowpaths draining to these nexuses will have empty downstream lists.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping each routing virtual_fp_id to a list of downstream virtual_fp_ids.
        Terminal flowpaths will have empty lists [].

    Examples
    --------
    >>> all_vfp = pd.DataFrame({
    ...     'virtual_fp_id': [1, 2, 3],
    ...     'dn_virtual_nex_id': [100, 101, 102],
    ...     'up_virtual_nex_id': [None, 100, 101],
    ...     'routing_segment': [True, True, True]
    ... })
    >>> routing_vfp = all_vfp[all_vfp['routing_segment']]
    >>> connections = build_downstream_connections(routing_vfp, all_vfp, terminal_nexus_ids={102})
    >>> connections
    {1: [2], 2: [3], 3: []}
    """
    if terminal_nexus_ids is None:
        terminal_nexus_ids = set()

    # Get the set of routing flowpath IDs for filtering downstream connections
    routing_fp_ids = set(routing_flowpaths['virtual_fp_id'].tolist())

    # Map: nexus_id -> list of ROUTING flowpaths that have this nexus as their UPSTREAM nexus
    # We use all_flowpaths to find the mapping, but filter to only routing segments
    routing_vfp = all_flowpaths[all_flowpaths['virtual_fp_id'].isin(routing_fp_ids)]
    nex_to_downstream_fps = (
        routing_vfp[routing_vfp['up_virtual_nex_id'].notna()]
        .groupby('up_virtual_nex_id')['virtual_fp_id']
        .apply(list)
        .to_dict()
    )

    # Build connections: for each routing flowpath, find downstream routing flowpath(s)
    connections = {}
    for _, row in routing_flowpaths.iterrows():
        fp_id = row['virtual_fp_id']
        dn_nex = row['dn_virtual_nex_id']

        if pd.isna(dn_nex):
            # No downstream nexus
            connections[fp_id] = []
        elif dn_nex in terminal_nexus_ids:
            # Downstream nexus is terminal (no further flowpaths)
            connections[fp_id] = []
        else:
            # Find routing flowpaths whose up_virtual_nex_id == this flowpath's dn_virtual_nex_id
            downstream_fps = nex_to_downstream_fps.get(dn_nex, [])
            connections[fp_id] = downstream_fps

    return connections


def build_upstream_terminal(
    virtual_flowpaths: pd.DataFrame,
    terminal_nexus_ids: set[int],
) -> dict[int, set[int]]:
    """
    Build mapping of terminal nexus IDs to their upstream flowpath IDs.

    This identifies which flowpaths drain into terminal nexuses (network outlets).

    Parameters
    ----------
    virtual_flowpaths : pd.DataFrame
        DataFrame containing virtual flowpath information with columns:
        - virtual_fp_id: unique flowpath identifier
        - dn_virtual_nex_id: downstream nexus ID for each flowpath
    terminal_nexus_ids : set[int]
        Set of nexus IDs that are terminal (no downstream flowpath).

    Returns
    -------
    dict[int, set[int]]
        Dictionary mapping each terminal nexus ID to a set of upstream flowpath IDs.

    Examples
    --------
    >>> vfp = pd.DataFrame({
    ...     'virtual_fp_id': [1, 2, 3],
    ...     'dn_virtual_nex_id': [100, 101, 101],  # fp2 and fp3 both drain to terminal nex 101
    ... })
    >>> upstream_terminal = build_upstream_terminal(vfp, terminal_nexus_ids={101})
    >>> upstream_terminal
    {101: {2, 3}}
    """
    upstream_terminal = {}

    for _, row in virtual_flowpaths.iterrows():
        dn_nex = row['dn_virtual_nex_id']
        if pd.notna(dn_nex) and dn_nex in terminal_nexus_ids:
            fp_id = row['virtual_fp_id']
            upstream_terminal.setdefault(dn_nex, set()).add(fp_id)

    return upstream_terminal


def get_terminal_nexus_ids(virtual_nexus: pd.DataFrame) -> set[int]:
    """
    Extract terminal nexus IDs from the virtual nexus DataFrame.

    Terminal nexuses are those with no downstream flowpath (dn_virtual_fp_id is NaN).

    Parameters
    ----------
    virtual_nexus : pd.DataFrame
        DataFrame containing virtual nexus information with columns:
        - virtual_nex_id: unique nexus identifier
        - dn_virtual_fp_id: downstream flowpath ID (NaN for terminals)

    Returns
    -------
    set[int]
        Set of terminal nexus IDs.
    """
    terminals = virtual_nexus[pd.isna(virtual_nexus["dn_virtual_fp_id"])]
    return set(terminals["virtual_nex_id"].tolist())


def validate_connections(connections: dict[int, list[int]]) -> tuple[bool, set[int]]:
    """
    Validate that all downstream flowpath IDs in the connections dictionary
    exist as keys (i.e., they are valid flowpath IDs).

    This catches the case where nexus IDs are accidentally used instead of
    flowpath IDs in the connections values.

    Parameters
    ----------
    connections : dict[int, list[int]]
        Dictionary mapping flowpath IDs to lists of downstream flowpath IDs.

    Returns
    -------
    tuple[bool, set[int]]
        - bool: True if all downstream IDs are valid, False otherwise
        - set[int]: Set of orphaned IDs (downstream IDs not found as keys)

    Examples
    --------
    >>> connections = {1: [2], 2: [3], 3: []}
    >>> is_valid, orphaned = validate_connections(connections)
    >>> is_valid
    True
    >>> orphaned
    set()

    >>> bad_connections = {1: [2], 2: [999], 3: []}  # 999 doesn't exist as key
    >>> is_valid, orphaned = validate_connections(bad_connections)
    >>> is_valid
    False
    >>> orphaned
    {999}
    """
    all_downstream_fps = set(chain.from_iterable(connections.values()))
    all_fp_ids = set(connections.keys())

    orphaned = all_downstream_fps - all_fp_ids

    return len(orphaned) == 0, orphaned


def find_headwaters(connections: dict[int, list[int]]) -> set[int]:
    """
    Find headwater flowpaths in the network.

    Headwaters are flowpaths that are never referenced as downstream of any other
    flowpath (i.e., they appear as keys but never in any values list).

    Parameters
    ----------
    connections : dict[int, list[int]]
        Dictionary mapping flowpath IDs to lists of downstream flowpath IDs.

    Returns
    -------
    set[int]
        Set of headwater flowpath IDs.
    """
    all_downstream_fps = set(chain.from_iterable(connections.values()))
    all_fp_ids = set(connections.keys())

    return all_fp_ids - all_downstream_fps


def find_tailwaters(connections: dict[int, list[int]]) -> set[int]:
    """
    Find tailwater flowpaths in the network.

    Tailwaters are flowpaths with no downstream connections (empty list).

    Parameters
    ----------
    connections : dict[int, list[int]]
        Dictionary mapping flowpath IDs to lists of downstream flowpath IDs.

    Returns
    -------
    set[int]
        Set of tailwater flowpath IDs.
    """
    return {fp_id for fp_id, downstream in connections.items() if not downstream}


def build_link_connections(
    links_df: pd.DataFrame,
    nexus: pd.DataFrame,
) -> dict[int, list[int]]:
    """Build downstream connectivity for links.

    For adjacent links within the same flowpath:
        link_upstream.dn_node_id == link_downstream.up_node_id

    For cross-flowpath connections:
        link's dn_node_id is the flowpath's dn_nex_id ->
        nexus.dn_fp_id gives the downstream flowpath ->
        find the most-upstream link of that flowpath (up_node_id is None)

    Parameters
    ----------
    links_df : pd.DataFrame
        Link DataFrame indexed by link_id with columns:
        fp_id, dn_node_id, up_node_id
    nexus : pd.DataFrame
        Regular nexus DataFrame with columns: nex_id, dn_fp_id

    Returns
    -------
    connections : dict[int, list[int]]
        Mapping of link_id -> [downstream_link_ids]

    """
    if links_df.empty:
        return {}

    # Pre extract arrays because np is faster
    link_ids = links_df.index.to_numpy()
    up_nodes = links_df["up_node_id"].to_numpy()
    dn_nodes = links_df["dn_node_id"].to_numpy()
    fp_ids = links_df["fp_id"].to_numpy()

    # Build lookup: up_node_id -> link_id (for within-flowpath connections)
    up_node_to_link = {
        int(up): int(lid)
        for up, lid in zip(up_nodes, link_ids)
        if pd.notna(up)
    }

    # Build lookup: fp_id -> most-upstream link_id (up_node_id is None)
    fp_to_upstream_link = {
        int(fp): int(lid)
        for fp, up, lid in zip(fp_ids, up_nodes, link_ids)
        if pd.isna(up)
    }

    # Build lookup: nex_id -> dn_fp_id (cross-flowpath via regular nexus)
    nex_to_dn_fp = {}
    if not nexus.empty:
        nex_ids = nexus["nex_id"].to_numpy()
        dn_fps = nexus["dn_fp_id"].to_numpy()

        nex_to_dn_fp = {
            int(nex): int(fp)
            for nex, fp in zip(nex_ids, dn_fps)
            if pd.notna(fp)
        }

    connections = {}
    for link_id, dn_node in zip(link_ids, dn_nodes):
        dn_node = int(dn_node)

        # Same-flowpath connection
        ds = up_node_to_link.get(dn_node)
        if ds is not None:
            connections[int(link_id)] = [ds]
            continue

        # Cross-flowpath via nexus
        dn_fp = nex_to_dn_fp.get(dn_node)
        if dn_fp is not None:
            connections[int(link_id)] = [fp_to_upstream_link.get(dn_fp)] if dn_fp in fp_to_upstream_link else []
            continue

        # Terminal
        connections[int(link_id)] = []

    return connections


def tailwaters(N):
    """
    Find network tailwaters

    Arguments
    ---------
    N (dict, int: [int]): Network connections graph

    Returns
    -------
    (iterable): tailwater segments

    Notes
    -----
    - If reverse connections graph is handed as input, then function
      will return network headwaters.

    """
    tw = chain.from_iterable(N.values()) - N.keys()
    for m, n in N.items():
        if not n:
            tw.add(m)
    return tw


def reservoir_shore(connections, waterbody_nodes):
    wbody_set = set(waterbody_nodes)
    not_in = lambda x: x not in wbody_set

    shore = set()
    for node in wbody_set:
        shore.update(filter(not_in, connections[node]))
    return list(shore)


def reservoir_boundary(connections, waterbodies, n):
    if n not in waterbodies and n in connections:
        return any(x in waterbodies for x in connections[n])
    return False


def reverse_surjective_mapping(d):
    rd = defaultdict(list)
    for src, dst in d.items():
        rd[dst].append(src)
    rd.default_factory = None
    return rd


def separate_waterbodies(connections, waterbodies):
    waterbody_nodes = {}
    for wb, nodes in reverse_surjective_mapping(waterbodies).items():
        waterbody_nodes[wb] = net = {}
        for n in nodes:
            if n in connections:
                net[n] = list(filter(waterbodies.__contains__, connections[n]))
    return waterbody_nodes


def replace_waterbodies_connections(connections, waterbodies):
    """
    Use a single node to represent waterbodies. The node id is the
    waterbody id. Create a cross walk dictionary that relates lake_ids
    to the terminal segments within the waterbody footprint.

    Arguments
    ---------
    - connections (dict):
    - waterbodies (dict): dictionary relating segment linkIDs to the
                          waterbody lake_id that they lie in

    Returns
    -------
    - new_conn  (dict): connections dictionary with waterbodies represented by single nodes.
                        Waterbody node ids are lake_ids
    - link_lake (dict): cross walk dictionary where keys area lake_ids and values are lists
                        of waterbody tailwater nodes (i.e. the nodes connected to the
                        waterbody outlet).
    """
    new_conn = {}
    link_lake = {}
    waterbody_nets = separate_waterbodies(connections, waterbodies)
    rconn = reverse_network(connections)

    for n in connections:
        if n in waterbodies:
            wbody_code = waterbodies[n]
            if wbody_code in new_conn:
                continue

            # get all nodes from waterbody
            wbody_nodes = [k for k, v in waterbodies.items() if v == wbody_code]
            outgoing = reservoir_shore(connections, wbody_nodes)
            new_conn[wbody_code] = outgoing

            if len(outgoing) >= 1:
                if outgoing[0] in waterbodies:
                    new_conn[wbody_code] = [waterbodies.get(outgoing[0])]
                link_lake[wbody_code] = list(set(rconn[outgoing[0]]).intersection(set(wbody_nodes)))[0]
            else:
                subset_dict = {key: value for key, value in connections.items() if key in wbody_nodes}
                link_lake[wbody_code] = list(tailwaters(subset_dict))[0]

        elif reservoir_boundary(connections, waterbodies, n):
            # one of the children of n is a member of a waterbody
            # replace that child with waterbody code.
            new_conn[n] = []

            for child in connections[n]:
                if child in waterbodies:
                    new_conn[n].append(waterbodies[child])
                else:
                    new_conn[n].append(child)
        else:
            # copy to new network unchanged
            new_conn[n] = connections[n]

    return new_conn, link_lake
