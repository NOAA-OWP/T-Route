"""Helpers for the Old River diversion notebook.

Reading run output, forcing and observations, comparing runs at the gages, building the
overdraw case, and keeping the driver's logging out of the cells.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from nwm_routing.nhf_routing import nhf_routing

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

CFS_PER_CMS = 35.3147
LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s]: "
    "%(message)s"
)
COLOR_OBSERVED = "#1a1a1a"
COLOR_WITH = "#2166ac"
COLOR_WITHOUT = "#d6604d"
COLOR_THIRD = "#4dac26"


def configure_logging(log_file: Path) -> None:
    """Send every logger, the driver's included, to one file.

    The driver configures logging with ``basicConfig``, which does nothing once the root
    logger has a handler, so the handler installed here receives every run's records.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file, filemode="w", level=logging.INFO, format=LOG_FORMAT, force=True
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def run(config: str, log_file: Path) -> None:
    """Run one configuration from the current directory, its timing summary going to the log."""
    with log_file.open("a") as log, contextlib.redirect_stdout(log):
        nhf_routing(["-f", config])


def load_output(output_dir: str | Path) -> xr.Dataset:
    """Return a run's hourly output files concatenated in time."""
    files = sorted(Path(output_dir).glob("*.nc"))
    if not files:
        msg = f"no output in {output_dir}; run the configuration first"
        raise FileNotFoundError(msg)
    return xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in files], dim="time", data_vars="all"
    )


def flow_at(output_dir: str | Path, fp_id: int) -> pd.Series[float]:
    """Return the hourly flow at a flowpath's outlet."""
    return load_output(output_dir)["flow"].sel(feature_id=fp_id).to_series()


def _text(value: object) -> str:
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def timeslice_series(folder: str | Path, site_no: str) -> pd.Series[float]:
    """Return the gage's discharge on the hour, read back from the TimeSlice files."""
    values: dict[pd.Timestamp, float] = {}
    for path in sorted(Path(folder).glob("*_??:00:00.15min.usgsTimeSlice.ncdf")):
        with xr.open_dataset(path) as ds:
            ids = [_text(s) for s in ds["stationId"].to_numpy()]
            if site_no not in ids:
                continue
            i = ids.index(site_no)
            stamp = _text(ds["time"].to_numpy()[i])
            values[pd.Timestamp(stamp.replace("_", " "))] = float(ds["discharge"].to_numpy()[i])
    return pd.Series(values).sort_index()


def qlat_series(forcing_dir: str | Path, fp_ids: Sequence[int]) -> pd.DataFrame:
    """Return the hourly lateral inflow of the given flowpaths, one column each."""
    rows: dict[pd.Timestamp, NDArray[np.float64]] = {}
    for path in sorted(Path(forcing_dir).glob("*.CHRTOUT_DOMAIN1.csv")):
        column = pd.read_csv(path, index_col=0).iloc[:, 0]
        rows[pd.Timestamp(str(column.name))] = column.reindex(list(fp_ids)).to_numpy(dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(fp_ids)).sort_index()


def _figure() -> tuple[Figure, Axes]:
    """Return one figure with one axes; pyplot's overloads type the pair too loosely."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    return cast("Figure", fig), cast("Axes", ax)


def style(ax: Axes, title: str, ylabel: str = "Discharge (cms)") -> None:
    """Apply the notebook's axis styling and draw the legend."""
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_facecolor("whitesmoke")
    ax.grid(color="white", linewidth=0.8)
    ax.legend(framealpha=0.9, fontsize=9)


def _reference_sites(
    ref_data: Path, sites: Sequence[str]
) -> Iterator[tuple[str, int, xr.Dataset]]:
    """Yield (site number, flowpath id, reference slice) for each requested gage."""
    ds = xr.open_dataset(ref_data)
    for gage_idx, site_no in enumerate(ds["site_no"].to_numpy()):
        if str(site_no) in sites:
            sub = ds.isel(gage=gage_idx)
            yield str(site_no), int(sub["fp_id"].item()), sub


def compare_runs(
    runs: Sequence[tuple[str | Path, str]],
    sites: Sequence[str],
    suffix: str,
    *,
    ref_data: Path,
    out_dir: Path,
) -> None:
    """Plot the observed record against two runs at each gage, from their first common hour.

    The first run is drawn solid, the second dashed on top, so coincident curves show both.
    """
    (out_a, label_a), (out_b, label_b) = runs
    ds_a, ds_b = load_output(out_a), load_output(out_b)
    first_t = max(ds_a["time"].to_numpy()[0], ds_b["time"].to_numpy()[0])
    styles = ((ds_a, label_a, COLOR_WITH, (), 2), (ds_b, label_b, COLOR_WITHOUT, (5, 4), 3))
    for site_no, fp_id, sub in _reference_sites(ref_data, sites):
        times = sub["time"].to_numpy()
        shown = times >= first_t
        fig, ax = _figure()
        ax.plot(
            times[shown], sub["usgs_q"].to_numpy()[shown] / CFS_PER_CMS,
            label=f"USGS {site_no}", color=COLOR_OBSERVED, linewidth=3.5, alpha=0.6, zorder=1,
        )
        for out, label, color, dashes, z in styles:
            series = out["flow"].sel(feature_id=fp_id).to_series()
            series = series[(series.index >= first_t) & (series.index <= times[-1])]
            ax.plot(
                series.index, series.to_numpy(),
                label=label, color=color, linewidth=1.6, dashes=dashes, zorder=z,
            )
        style(ax, f"USGS {site_no}, flowpath {fp_id}")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"diversion_{site_no}{suffix}.png", dpi=150)


def compare_retro(
    sites: Sequence[str], *, ref_data: Path, out_dir: Path
) -> dict[str, tuple[float, float]]:
    """Plot the retrospective against the observed record at each gage.

    Returns the (retrospective, observed) median discharge per site, in cms.
    """
    medians: dict[str, tuple[float, float]] = {}
    for site_no, fp_id, sub in _reference_sites(ref_data, sites):
        times = sub["time"].to_numpy()
        retro = sub["retrospective_q"].to_numpy()
        usgs = sub["usgs_q"].to_numpy() / CFS_PER_CMS
        medians[site_no] = (float(np.nanmedian(retro)), float(np.nanmedian(usgs)))
        fig, ax = _figure()
        ax.plot(
            times, retro, label="NWM v3.0 retrospective",
            color=COLOR_OBSERVED, linewidth=1.5, linestyle="--",
        )
        ax.plot(times, usgs, label=f"USGS {site_no}", color=COLOR_OBSERVED, linewidth=2)
        style(ax, f"USGS {site_no}, flowpath {fp_id}")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"diversion_{site_no}_retro.png", dpi=150)
    return medians


def best_lag(series: pd.Series[float], reference: pd.Series[float], max_hours: int = 72) -> int:
    """Return the hours by which *series* trails *reference*: the best-correlated shift."""
    scores = {lag: float(series.shift(-lag).corr(reference)) for lag in range(max_hours + 1)}
    return max(scores, key=lambda lag: scores[lag])


@dataclass(frozen=True)
class Scaling:
    """A gage's record multiplied by *factor* from *start* on."""

    site_no: str
    factor: float
    start: pd.Timestamp


def scale_observation(
    src: Path, dst: Path, scaling: Scaling, window: tuple[pd.Timestamp, pd.Timestamp]
) -> None:
    """Copy the TimeSlice files inside *window* to *dst*, applying *scaling*."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    for path in sorted(src.glob("*.ncdf")):
        stamp = pd.Timestamp(path.name[:19].replace("_", " "))
        if not window[0] <= stamp <= window[1]:
            continue
        with xr.open_dataset(path) as ds:
            out = ds.load()
        if stamp >= scaling.start:
            ids = [_text(s) for s in out["stationId"].to_numpy()]
            i = ids.index(scaling.site_no)
            out["discharge"][i] = out["discharge"][i] * scaling.factor
        out.to_netcdf(dst / path.name)


def variant_config(
    base: Path, out: Path, *, nts: int, usgs_folder: Path, output_dir: Path
) -> None:
    """Write *base* with a new run length, observation folder and output directory."""
    cfg = yaml.safe_load(base.read_text())
    compute = cfg["compute_parameters"]
    compute["forcing_parameters"]["nts"] = nts
    compute["data_assimilation_parameters"]["usgs_timeslices_folder"] = str(usgs_folder)
    cfg["output_parameters"]["stream_output"]["stream_output_directory"] = str(output_dir)
    output_dir.mkdir(exist_ok=True)
    for old in output_dir.glob("*.nc"):
        old.unlink()
    out.write_text(yaml.safe_dump(cfg))


@contextlib.contextmanager
def captured_warnings(
    logger_name: str = "TROUTE", needle: str = "diversion DA"
) -> Generator[list[str], None, None]:
    """Collect the warnings a logger emits containing *needle* while the block runs."""
    lines: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if needle in message:
                lines.append(message)

    handler = _Catch(logging.WARNING)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)
