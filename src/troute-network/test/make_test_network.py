import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
import warnings

def process_branch(edges, start, depth):
    if(depth == 0): return [] #stop condition
    ids = []
    try:
        nexus = edges.loc[ start ].id
    except KeyError:
        #found a headwater
        return ids
    flowpaths = edges.loc[ [nexus] ]
    
    for _, fp in flowpaths.iterrows():
        ids.append(fp.id) #mark this flowpath
        got = process_branch(edges, fp.id, depth-1)
        ids.extend( got )

    return ids #in this branch

def walk_upstream(edges, start, n=None):
    if n is None:
        n = edges.size() #maximum we would have to walk
    ids = [start]
    edges = edges.set_index('toid')
    ids.extend( process_branch(edges, start, n-1 ) )

    return ids

def make_network_from_segment(flowpaths, edges, attributes, depth, segment):
    ids = walk_upstream(edges, segment, depth)
    flowpaths.set_index('id', inplace=True)
    attributes.set_index('id', inplace=True)
    sub_flowpaths = flowpaths.loc[ ids ]
    sub_attributes = attributes.loc[ ids ]
    sub_edges = edges[ edges.id.isin(ids) | edges.toid.isin(ids) ]
    sub_edges.to_file("hy_network.gpkg", layer="flowpath_edge_list", driver="GPKG")
    sub_flowpaths.to_file("hy_network.gpkg", layer='flowpaths', driver="GPKG")
    sub_attributes.to_file("hy_network.gpkg", layer='flowpath_attributes', driver="GPKG")
    #make json version
    sub_flowpaths = pd.merge(sub_flowpaths, sub_attributes.drop('geometry', axis=1), on='id')
    sub_flowpaths.drop('geometry', axis=1).to_json("flowpath_attributes.json", orient='index', indent=2)
    sub_edges.drop('geometry', axis=1).to_json("flowpath_edge_list.json", orient='records', indent=2)

def make_network_from_geopkg(file_path, depth, segment=None):
    flowpaths = gpd.read_file(file_path, layer="flowpaths")
    attributes = gpd.read_file(file_path, layer="flowpath_attributes")
    edges = gpd.read_file(file_path, layer="flowpath_edge_list")
    if segment is None:
        segment = flowpaths[flowpaths['toid'].str.startswith('tnex')].iloc[0]['id']
    make_network_from_segment(flowpaths, edges, attributes, depth, segment)

if __name__ == "__main__":
    """
        use this script to make a test network.  Pass it a geopkg and the rough size of the network
        the network in tests/data was created using nextgen_09.geopkg with a depth of 10
        by default, it picks the first terminal nexus in the list and walks upstream of that, which
        is why the test network only has 5 links.
    """
    warnings.warn(
        "This script is deprecated. Please use updated network creation methods.",
        DeprecationWarning,
        stacklevel=2,
    )
    file_path = Path(sys.argv[1])
    #used to make simple test network
    make_network_from_geopkg(file_path, int(sys.argv[2]), segment=None)
