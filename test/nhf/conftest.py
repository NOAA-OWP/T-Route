"""Shared fixtures for the NHF integration tests.

These tests each need a subset hydrofabric, generated forcing and, where data
assimilation is exercised, timeslice files. Building that costs minutes and hits
the network, so it is done once per session and reused, and every test skips
cleanly when the data is absent.

Case configuration lives in fixtures rather than module-level globals so that
importing a test module does no work, a case can be parameterized, and the
build step is ordered by pytest rather than by a hand-called ``setup()``.
"""

from __future__ import annotations

import pytest

from .utils.integration_helpers import delete_outputs, skip_if_not_built


def pytest_addoption(parser):
    parser.addoption(
        "--refresh-nhf-data",
        action="store_true",
        default=False,
        help="Rebuild subset domains, forcing and timeslices even if they exist.",
    )


@pytest.fixture(scope="session")
def refresh_nhf_data(request) -> bool:
    """Whether generated inputs should be rebuilt rather than reused."""
    return request.config.getoption("--refresh-nhf-data")


@pytest.fixture
def built_case(request):
    """Skip the calling test unless its case has been built.

    Usage::

        def test_x(built_case):
            cfg = built_case(CFG)

    Returns the config so the test reads as a single statement, and clears any
    outputs left by a previous run so assertions cannot pass on stale files.

    The YAML is rewritten from the in-code Config on every run. Prep only writes
    it when it is missing, so editing a Config here otherwise left the old file on
    disk and the run silently used settings the test no longer describes. That is
    not a visible failure, it is a test that quietly measures the wrong thing. The
    expensive inputs, domain and forcing and timeslices, are still reused.
    """

    def _use(cfg, *, clear=()):
        skip_if_not_built(cfg)
        cfg.write_yaml()
        delete_outputs(cfg.output_dir)
        for extra in clear:
            delete_outputs(extra)
        return cfg

    return _use
