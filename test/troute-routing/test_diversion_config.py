"""Configuration contract for diversion data assimilation.

The diversion gage's discharge comes from ``usgs_timeslices_folder``, read by the
diversion itself, and is held for ``diversion_persist_days`` past the record's end.
The shapes are pinned here because getting the source wrong silently disables the
transfer.
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


def _params(crosswalk, *, nudging: bool) -> dict:
    return {
        "streamflow_da": {"streamflow_nudging": nudging},
        "diversion_da": {"diversion_gage_crosswalk": crosswalk},
    }


class TestSupportedModes:
    def test_nudging_supplies_observations(self, crosswalk):
        cfg = DataAssimilationParameters(**_params(crosswalk, nudging=True))
        assert cfg.diversion_da.diversion_gage_crosswalk == crosswalk

    def test_timeslice_folder_alone_is_a_source(self, crosswalk, tmp_path):
        """The diversion reads its gage from the TimeSlices itself; nudging is
        slated for removal and must not be required."""
        params = _params(crosswalk, nudging=False)
        params["usgs_timeslices_folder"] = str(tmp_path)
        cfg = DataAssimilationParameters(**params)
        assert cfg.diversion_da.diversion_persist_days == 11


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


class TestPersistenceHorizon:
    """Requirement 2.2.3.15: hold the last observation, flat, for a configurable
    number of days, the way the other DAs express persistence."""

    def test_defaults_to_the_rfc_horizon(self, crosswalk):
        cfg = DataAssimilationParameters(**_params(crosswalk, nudging=True))
        assert cfg.diversion_da.diversion_persist_days == 11

    def test_parses(self, crosswalk):
        params = _params(crosswalk, nudging=True)
        params["diversion_da"]["diversion_persist_days"] = 45
        cfg = DataAssimilationParameters(**params)
        assert cfg.diversion_da.diversion_persist_days == 45

    def test_rejects_a_negative_horizon(self, crosswalk):
        params = _params(crosswalk, nudging=True)
        params["diversion_da"]["diversion_persist_days"] = -1
        with pytest.raises(ValueError):
            DataAssimilationParameters(**params)

    def test_timeslice_folder_is_a_source_without_nudging(self, crosswalk, caplog, tmp_path):
        """The diversion reads its gage from the TimeSlices itself."""
        params = _params(crosswalk, nudging=False)
        params["usgs_timeslices_folder"] = str(tmp_path)
        with caplog.at_level(logging.WARNING, logger="TROUTE"):
            DataAssimilationParameters(**params)
        assert "no flow will be diverted" not in caplog.text

    def test_two_donors_on_one_gage_is_rejected(self):
        """Two donors would subtract the transfer twice and add it once."""
        with pytest.raises(ValueError, match="same gage"):
            DataAssimilationParameters(
                **_params({DONOR_FP_ID: DIVERSION_GAGE, DONOR_FP_ID + 1: DIVERSION_GAGE},
                          nudging=True)
            )
