"""Configuration contract for diversion data assimilation.

The report accompanying the Old River work documents three usable shapes, and the
integration cases in ``test/nhf/old_river`` exercise all three. They are pinned here
because the difference between them is which source supplies the diversion gage's
discharge, and getting that wrong either silently disables the transfer or breaks a
supported mode outright.
"""

from __future__ import annotations

import logging

import pytest

from troute.config.compute_parameters import DataAssimilationParameters

DONOR_FP_ID = 1270479816524705
DIVERSION_GAGE = "07381482"


@pytest.fixture
def crosswalk() -> dict:
    return {DONOR_FP_ID: DIVERSION_GAGE}


def _params(crosswalk, *, nudging: bool, persist: bool = False) -> dict:
    return {
        "streamflow_da": {"streamflow_nudging": nudging},
        "diversion_da": {
            "diversion_gage_crosswalk": crosswalk,
            "persist_historical_median": persist,
        },
    }


class TestSupportedModes:
    def test_nudging_supplies_observations(self, crosswalk):
        cfg = DataAssimilationParameters(**_params(crosswalk, nudging=True))
        assert cfg.diversion_da.diversion_gage_crosswalk == crosswalk

    def test_historical_median_only_is_supported(self, crosswalk):
        """Nudging off with climatology on is a documented forecast-mode shape.

        The climatological fill runs outside the nudging branch and writes the
        diversion gage's row at its own routing link, so the receiving river still
        gets the water. Rejecting this would break the mode the report describes.
        """
        cfg = DataAssimilationParameters(
            **_params(crosswalk, nudging=False, persist=True)
        )
        assert cfg.diversion_da.persist_historical_median is True

    def test_median_fill_alongside_real_observations(self, crosswalk):
        """Both sources together: observations win, climatology fills the gaps."""
        cfg = DataAssimilationParameters(
            **_params(crosswalk, nudging=True, persist=True)
        )
        assert cfg.streamflow_da.streamflow_nudging is True
        assert cfg.diversion_da.persist_historical_median is True


class TestGuards:
    def test_no_observation_source_warns(self, crosswalk, caplog):
        """With neither source the gage row is never populated, so nothing diverts."""
        with caplog.at_level(logging.WARNING, logger="TROUTE"):
            DataAssimilationParameters(**_params(crosswalk, nudging=False))
        assert "no flow will be diverted" in caplog.text

    def test_no_warning_when_diversion_absent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="TROUTE"):
            DataAssimilationParameters(streamflow_da={"streamflow_nudging": False})
        assert "diverted" not in caplog.text

    def test_crosswalk_is_typed(self, crosswalk):
        """A free-form dict let a misspelled key validate and silently do nothing."""
        cfg = DataAssimilationParameters(**_params(crosswalk, nudging=True))
        assert not hasattr(cfg.diversion_da, "diversion_gage_crosswlak")
        # ids stay integral; NHF flowpath ids exceed 32-bit range
        (fp_id,) = cfg.diversion_da.diversion_gage_crosswalk
        assert isinstance(fp_id, int) and fp_id > 2**32
