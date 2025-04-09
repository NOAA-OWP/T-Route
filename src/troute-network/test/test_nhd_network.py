import pytest
from troute import nhd_network


def test_reverse_dict(test_columns, reverse_test_columns):
    result = nhd_network.reverse_dict(test_columns)
    assert result == reverse_test_columns


def test_build_connections(test_param_df, test_terminal_code, expected_connections):
    # There can be an externally determined terminal code -- that's this first value
    terminal_codes = set()
    terminal_codes.add(test_terminal_code)

    # ... but there may also be off-domain nodes that are not explicitly identified
    # but which are terminal (i.e., off-domain) as a result of a mask or some other
    # an interior domain truncation that results in a
    # otherwise valid node value being pointed to, but which is masked out or
    # being intentionally separated into another domain.
    terminal_codes = terminal_codes | set(
        test_param_df[~test_param_df["downstream"].isin(test_param_df.index)][
            "downstream"
        ].values
    )

    connections = nhd_network.extract_connections(
        test_param_df, "downstream", terminal_codes
    )
    assert connections == expected_connections


def test_reverse_network(expected_connections, expected_rconn):
    connections = expected_connections
    rconn = nhd_network.reverse_network(connections)
    assert expected_rconn == rconn
    rrconn = nhd_network.reverse_network(rconn)
    assert rrconn == connections


@pytest.mark.skip(reason="This test is deprecated")
def test_extract_waterbodies(
    test_param_df, test_waterbody_null_code, expected_wbody_connections
):
    wbody_connections = nhd_network.extract_waterbody_connections(
        test_param_df, "waterbody", test_waterbody_null_code
    )
    assert wbody_connections == expected_wbody_connections
