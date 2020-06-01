from collections import defaultdict, Counter, deque
from itertools import chain
import itertools


def nodes(N):
    yield from N.keys() | (v for v in chain.from_iterable(N.values()) if v not in N)


def edges(N):
    for i, v in N.items():
        for j in v:
            yield (i, j)


def in_degrees(N):
    """
    Compute indegree of nodes in N.

    Args:
        N (dict): Network

    Returns:

    """
    degs = Counter(chain.from_iterable(N.values()))
    degs.update(dict.fromkeys(headwaters(N), 0))
    return degs


def out_degrees(N):
    """
    Compute outdegree of nodes in N

    Args:
        N (dict): Network

    Returns:

    """
    return in_degrees(reverse_network(N))


def extract_network(rows, target_col, terminal_code=0):
    """Extract connection network from dataframe.

    Arguments:
        rows (DataFrame): Dataframe indexed by key_col.
        key_col (str): Source of each edge
        target_col (str): Target of edge

    Returns:
        (dict)
    """
    network = {}
    for src, dst in rows[target_col].items():
        if src not in network:
            network[src] = []

        if dst != terminal_code:
            network[src].append(dst)
    return network


def reverse_network(N):
    rg = defaultdict(list)
    for src, dst in N.items():
        if not dst:
            rg[src]

        for n in dst:
            rg[n].append(src)

    rg.default_factory = None
    return rg


def junctions(N):
    c = Counter(chain.from_iterable(N.values()))
    return {k for k, v in c.items() if v > 1}


def headwaters(N):
    yield from N.keys() - chain.from_iterable(N.values())


def tailwaters(N):
    yield from chain.from_iterable(N.values()) - N.keys()
    yield from (m for m, n in N.items() if not n)


def dfs_decomposition(N):
    """
    Decompose N into a list of simple segments.
    The order of these segments are suitable to be parallelized as we guarantee that for any segment,
    the predecessor segments appear before it in the list.

    This is accomplished by a depth first search on the reversed graph and
    finding the path from node to its nearest junction.

    Arguments:
        N (Dict[obj: List[obj]]): The graph

    Returns:
        [List]: List of paths to be processed in order.
    """
    RN = reverse_network(N)

    paths = []
    visited = set()
    for h in headwaters(RN):
        stack = [(h, iter(RN[h]))]
        #visited = set()
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    # Check to see if we are at a leaf
                    if child in RN:
                        stack.append((child, iter(RN[child])))
                    else:
                        # At a leaf, process the stack
                        path = [child]
                        for n, _ in reversed(stack):
                            if len(RN[n]) == 1:
                                path.append(n)
                            else:
                                break
                        paths.append(path)
                        # optimization (clear tail of stack)
                        if len(path) > 1:
                            del stack[-(len(path) - 1) :]
                    visited.add(child)
            except StopIteration:
                node, _ = stack.pop()

                path = [node]
                # process between junction nodes
                if len(RN[node]) == 0:
                    paths.append(path)
                elif len(RN[node]) > 1:
                    for n, _ in reversed(stack):
                        if len(RN[n]) == 1:
                            path.append(n)
                        else:
                            break
                    paths.append(path)
                    if len(path) > 1:
                        del stack[-(len(path) - 1) :]

    return paths


def segment_deps(segments, connections):
    """Build a dependency graph of segments

    Arguments:
        segments (list): List of paths
        connections {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # index segements
    index = {d[0]: i for i, d in enumerate(segments)}
    deps = defaultdict(list)
    for i, s in enumerate(segments):
        cand = s[-1]
        if cand in connections:
            if connections[cand]:
                # There is a node downstream
                deps[i].append(index[connections[cand][0]])
    return deps


def kahn_toposort(N):
    degrees = in_degrees(N)
    zero_degree = set(k for k, v in degrees.items() if v == 0)

    _deg_pop = zero_degree.pop
    _deg_add = zero_degree.add
    _network_get = N.get
    while zero_degree:
        n = _deg_pop()
        for j in _network_get(n, ()):
            degrees[j] = c = degrees[j] - 1
            if c == 0:
                _deg_add(j)
        yield n

    try:
        next(degrees.elements())
        raise Exception("Cycle exists!")
    except StopIteration:
        pass


def kahn_toposort_edges(N):
    sorted_nodes = kahn_toposort(N)
    for n in sorted_nodes:
        for m in N.get(n, ()):
            yield (n, m)
