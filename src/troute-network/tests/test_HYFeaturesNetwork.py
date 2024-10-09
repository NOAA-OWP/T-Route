import pytest
import numpy as np
import yaml
from pathlib import Path
import pandas as pd
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from troute.routing.compute import compute_nhd_routing_v02
#set the workdir relative to this test config
#and use that to look for test data
_workdir=Path(__file__).parent

"""
Fixtures for setting up various components for testing
"""

file_type = ['gpkg']   # list of geospatial file types e.g., ['json']


def get_topology(topolgy_parameter):
    yaml_file =  _workdir.joinpath("data/topology.yml")
    with open(yaml_file, "r") as ymlfile:
         topology = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return topology[topolgy_parameter]


def supernetwork_parameters(type):
    if( type == 'gpkg'):
       geo_file =  _workdir.joinpath("data/hy_network.gpkg")
       edge_list = None
    elif( type == 'json'):
        geo_file =  _workdir.joinpath("data/flowpath_attributes.json")
        edge_list = _workdir.joinpath("data/flowpath_edge_list.json")
    params = {
        "title_string":"HY_Features Test",
        "geo_file_path":geo_file,
        "flowpath_edge_list":edge_list,
        "columns": {
            #link????
            "key": "id",
            "downstream": "toid",
            "dx": "lengthkm",
            "n": "n",  # TODO: rename to `manningn`
            "ncc": "nCC",  # TODO: rename to `mannningncc`
            "s0": "slope_percent",  # TODO: rename to `bedslope`
            "bw": "BtmWdth",  # TODO: rename to `bottomwidth`
            #waterbody: "NHDWaterbodyComID",
            "tw": "TopWdth",  # TODO: rename to `topwidth`
            "twcc": "TopWdthCC",  # TODO: rename to `topwidthcc`
            #alt: "alt",
            "musk": "MusK",
            "musx": "MusX",
            "cs": "ChSlp"  # TODO: rename to `sideslope`
        },
        "waterbody_null_code": -9999,
        "terminal_code": 0,
        "waterbody_null_code": -9999,
        "driver_string": "NetCDF",
        "layer_string": 0
    }
    return params

@pytest.fixture
def waterbody_parameters():
    return {}

@pytest.fixture
def forcing_parameters():
    return {
        "nexus_input_folder": _workdir.joinpath("data"),
        "nexus_file_pattern_filter": "nex-*"
    }

@pytest.fixture
def restart_parameters():
    return {}

@pytest.fixture
def network(request, waterbody_parameters, restart_parameters, forcing_parameters):
    type = request.param
    return HYFeaturesNetwork(supernetwork_parameters(type), waterbody_parameters, restart_parameters, forcing_parameters, verbose=True, showtiming=False)


@pytest.mark.parametrize("network",file_type, indirect=True )
def test_qlateral(network):
    # check wb-2, wb-3, and wb-4
    for nxid in [2,3,4]:
        df_qlat = pd.read_csv(_workdir.joinpath('data/nex-%d_output.csv'%nxid),
                              index_col=0, names=['date','flow'])
        assert(np.all(network.qlateral.loc[nxid,:].values-df_qlat['flow'].values==0))
    # wb-1 and wb-5 (headwaters) should have no inflow
    assert(np.all(network.qlateral.loc[1,:]==0))
    assert(np.all(network.qlateral.loc[5,:]==0))


@pytest.mark.parametrize("network",file_type, indirect=True )
def test_downstream_flowpath_dict(network):
    assert(network.downstream_flowpath_dict == get_topology('downstream_flowpath_dict'))


@pytest.mark.parametrize("network",file_type, indirect=True)
def test_connections(network):
    assert(network.connections == get_topology('connections'))


@pytest.mark.parametrize("network",file_type, indirect=True)
def test_reverse_network(network):
    assert(network.reverse_network==get_topology('reverse_network'))


@pytest.mark.parametrize("network",file_type, indirect=True)
def test_reaches_by_tailwater(network):
    assert(network.reaches_by_tailwater==get_topology('reaches_by_tailwater'))


@pytest.mark.parametrize("network",file_type, indirect=True)
def test_independent_networks(network):
    assert(network.independent_networks==get_topology('independent_networks'))


@pytest.mark.parametrize("network",file_type, indirect=True)
def test_waterbody_connection(network):
    # TO-DO: 11/01/2022- water-body is not implemented in HYFeature network yet
    pass


@pytest.mark.parametrize("network",file_type, indirect=True)
def test_init(network):
    #This isn't really nessicary since init is done by the fixture
    #errors in init will be caught and shown in the test
    #this test just gives a clear indication that inititalization
    #is working, instead of infering it from non failure of the fixture creation
    pass


#@pytest.skip #compute_nhd_routing_v02 has some additonal args that need to be considered
@pytest.mark.parametrize("network",file_type, indirect=True )
def test_routable(network, waterbody_parameters):
        #This is really more of an integration test
        #but it is here to ensure that the data structure works as intended
        #when passed off to the router
        compute_func = "V02-structured"

        results = compute_nhd_routing_v02(
        network.connections,
        network.reverse_network,
        network.waterbody_connections,
        network.reaches_by_tailwater,
        compute_func,
        "by-network",
        1,
        # The default here might be the whole network or some percentage...
        None,
        network.qlateral.iloc[0].index[0], #t0???
        300, #dt
        40, #nts
        12, #qts_subs
        network.independent_networks,
        network.dataframe,
        network.q0,
        network.qlateral,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        True,
        False,
        network.waterbody_dataframe,
        waterbody_parameters,  # TODO: Can we remove the dependence on this input? It's like passing argv down into the compute kernel -- seems like we can strip out the specifically needed items.
        network.waterbody_types_dataframe,
        not network.waterbody_types_dataframe.index.empty,
        {},
    )
