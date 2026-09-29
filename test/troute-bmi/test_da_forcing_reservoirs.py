"""Reservoir observations cross the BMI DA forcing path under their own source.

Each test writes TimeSlices for real stations, runs the DA forcing model, hands its BMI
arrays to the troute model, and reads the frames DataAssimilation rebuilds from them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import netCDF4
import numpy as np
import pandas as pd
import pytest

from troute.config.compute_parameters import DataAssimilationParameters
from troute.DataAssimilation import PersistenceDA

_T0 = pd.Timestamp("2026-09-01 00:00")
# One station per source, with releases that tell the two apart.
_USACE = ("LD10_Dardanelle", 21, [150.0, 0.0, 310.0])
# USBR closes at t0: a zero release is an observation.
_USBR = ("usbr-445", 22, [11.5, 0.0, 13.5])
_HOURS = [_T0 + pd.Timedelta(hours=h) for h in (-1, 0, 1)]
_SRC = Path(__file__).resolve().parents[2] / "src"
_RFC_FIXTURES = Path(__file__).resolve().parents[1] / "BMI" / "rfc_timeseries"


@pytest.fixture
def forcing_modules() -> tuple[ModuleType, ModuleType]:
    """model_DAforcing and bmi_DAforcing sit beside the BMI entry points in src/."""
    sys.path.insert(0, str(_SRC))
    try:
        import bmi_DAforcing
        import model_DAforcing
    finally:
        sys.path.remove(str(_SRC))
    return model_DAforcing, bmi_DAforcing


def _slices(folder: Path, family: str, site: str, releases: list[float]) -> None:
    """Hourly TimeSlices for one station from an hour before t0, named as retrieved."""
    folder.mkdir()
    for hour, q in enumerate(releases, start=-1):
        stamp = (_T0 + pd.Timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
        with netCDF4.Dataset(folder / f"{stamp}.60min.{family}TimeSlice.ncdf", "w") as nc:
            nc.createDimension("stationIdInd", 1)
            nc.createDimension("stationIdStringLength", 15)
            nc.createDimension("timeStringLength", 19)
            ids = nc.createVariable("stationId", "S1", ("stationIdInd", "stationIdStringLength"))
            ids[:] = np.array([list(site.rjust(15))], dtype="S1")
            times = nc.createVariable("time", "S1", ("stationIdInd", "timeStringLength"))
            times[:] = np.array([list(stamp)], dtype="S1")
            nc.createVariable("discharge", "f4", ("stationIdInd",))[:] = [q]
            nc.createVariable("discharge_quality", "i2", ("stationIdInd",))[:] = [100]


def _compute(t0: pd.Timestamp) -> dict[str, Any]:
    """A one-hour window from *t0*, with no restart files."""
    return {
        "cpu_pool": 1,
        "forcing_parameters": {"dt": 300, "nts": 12},
        "restart_parameters": {"start_datetime": t0.to_pydatetime(),
                               "lite_channel_restart_file": None,
                               "lite_waterbody_restart_file": None},
    }


def _forcing(modules: tuple[ModuleType, ModuleType], tmp_path: Path,
             monkeypatch: pytest.MonkeyPatch, persistence: dict[str, bool]) -> Any:
    """An initialized bmi_DAforcing whose config holds the given persistence switches."""
    model_da, bmi_da = modules
    folders = {}
    for family, (site, _, releases) in (("usace", _USACE), ("usbr", _USBR)):
        folders[family] = tmp_path / family
        _slices(folders[family], family, site, releases)
    compute = _compute(_T0)
    # Through the config schema, which is what the model reads: with RFC DA off it keeps
    # only the switch, and a section left out comes back as None.
    da = DataAssimilationParameters(
        qc_threshold=1,
        timeslice_lookback_hours=1,
        usace_timeslices_folder=folders["usace"],
        usbr_timeslices_folder=folders["usbr"],
        reservoir_da={
            "reservoir_persistence_da": {f"reservoir_persistence_{k}": v
                                         for k, v in persistence.items()},
            "reservoir_rfc_da": {"reservoir_rfc_forecasts": False},
        },
    ).model_dump()
    monkeypatch.setattr(model_da, "_read_config_file",
                        lambda _: (compute, compute["forcing_parameters"], da, {}))
    forcing = bmi_da.bmi_DAforcing()
    forcing.initialize(bmi_cfg_file="config.yaml")
    return forcing


def _rebuilt(forcing: Any, families: tuple[str, ...]) -> PersistenceDA:
    """The frames DataAssimilation builds from what a driver hands over: every BMI array
    of the given sources, set on the troute model under the same names."""
    names = ["dateNull"]
    for f in families:
        names += [f"datesSecondsArray_reservoir_{f}", f"nDates_reservoir_{f}",
                  f"stationArray_reservoir_{f}", f"stationStringLengthArray_reservoir_{f}",
                  f"nStations_reservoir_{f}", f"{f}_reservoir_Array"]
    values = {name: forcing.get_value(name) for name in names}
    network = SimpleNamespace(
        t0=_T0,
        link_lake_crosswalk={},
        **{f"{f}_lake_gage_crosswalk": pd.DataFrame(
            {f"{f}_gage_id": [site]}, index=pd.Index([lake], name=f"{f}_lake_id"))
           for f, (site, lake, _) in (("usace", _USACE), ("usbr", _USBR))},
    )
    da = PersistenceDA.__new__(PersistenceDA)
    da._data_assimilation_parameters = {"reservoir_da": {"reservoir_persistence_da": {
        f"reservoir_persistence_{f}": True for f in families}}}
    da._run_parameters = {}
    da._usgs_df = pd.DataFrame()
    da._usbr_df = pd.DataFrame()
    PersistenceDA.__init__(da, network, False, values)
    return da


def test_usace_observations_survive_the_bmi_arrays(
        forcing_modules: tuple[ModuleType, ModuleType], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    forcing = _forcing(forcing_modules, tmp_path, monkeypatch, {"usace": True})
    _, lake, releases = _USACE
    frame = _rebuilt(forcing, ("usace",))._reservoir_usace_df
    assert frame.index.tolist() == [lake]
    assert frame.loc[lake, _T0] == releases[1]


def test_usace_and_usbr_read_into_their_own_frames(
        forcing_modules: tuple[ModuleType, ModuleType], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    model = _forcing(forcing_modules, tmp_path, monkeypatch,
                     {"usace": True, "usbr": True})._model
    assert model._reservoir_usace_df.index.tolist() == [_USACE[0]]
    assert model._reservoir_usbr_df.index.tolist() == [_USBR[0]]


def test_usbr_observations_cross_the_bmi_arrays_beside_usace(
        forcing_modules: tuple[ModuleType, ModuleType], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    forcing = _forcing(forcing_modules, tmp_path, monkeypatch, {"usace": True, "usbr": True})
    da = _rebuilt(forcing, ("usace", "usbr"))
    for frame, (_, lake, releases) in ((da._reservoir_usace_df, _USACE),
                                       (da._reservoir_usbr_df, _USBR)):
        assert frame.index.tolist() == [lake]
        assert frame.loc[lake, _HOURS].tolist() == releases


def test_rfc_forecasts_are_read_when_rfc_da_is_on(
        forcing_modules: tuple[ModuleType, ModuleType],
        monkeypatch: pytest.MonkeyPatch) -> None:
    model_da, _ = forcing_modules
    compute = _compute(pd.Timestamp("2021-10-21 12:00"))
    da = DataAssimilationParameters(reservoir_da={"reservoir_rfc_da": {
        "reservoir_rfc_forecasts": True,
        "reservoir_rfc_forecasts_time_series_path": _RFC_FIXTURES,
        "reservoir_rfc_forecasts_offset_hours": 0,
    }}).model_dump()
    monkeypatch.setattr(model_da, "_read_config_file",
                        lambda _: (compute, compute["forcing_parameters"], da, {}))
    model = model_da.DAforcing_model("config.yaml")
    assert set(model._rfc_timeseries_df["stationId"]) == {"KNFC1"}
