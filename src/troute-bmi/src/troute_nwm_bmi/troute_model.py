"""Basic Model Interface backing model for NGEN t-route."""
from __future__ import annotations
import math
from functools import partial
from tempfile import NamedTemporaryFile
import psutil
from joblib import effective_n_jobs
import time
import typing
from typing import Iterator, TypedDict
import yaml
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import timedelta, datetime
from troute.config import Config

from troute.NHDNetwork import NHDNetwork
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from troute.NHF import NHF
from troute.DataAssimilation import DataAssimilation

import troute.hyfeature_network_utilities as hnu
from troute.job_memory import job_memory_headroom
from troute.window_plan import plan_windows, resolve_window

import nwm_routing.nwm_route as nwm_routing
from nwm_routing.output import nwm_output_generator
from nwm_routing.scaling_da_apply import (
    build_scaling_da,
    network_gage_segments,
)

import logging
LOG = logging.getLogger("TROUTE")

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def _write_merged_netcdf(files: list[Path], dest: Path) -> None:
    """Concatenate *files* along time and write the result to *dest*."""
    with xr.open_mfdataset(
        files,
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ) as ds:
        # Materialize before writing. The lazy dask graph reads the same files the
        # writer holds open, which is where an integrated ngen run hung.
        ds.load().to_netcdf(dest)


def _merge_into_first(files: list[Path], write: typing.Callable[[Path], None]) -> None:
    """Replace ``files[0]`` with the merge of every file in *files*.

    The temp file is created in the DESTINATION directory (a cross-filesystem
    rename fails with EXDEV), and parts are deleted only AFTER the atomic
    replace -- deleting first turns any rename failure into total output loss.
    """
    out_path = files[0].resolve()
    with NamedTemporaryFile(
        suffix=out_path.suffix, dir=out_path.parent, delete=False
    ) as tmp:
        combo_path = Path(tmp.name)
    try:
        write(combo_path)
        combo_path.chmod(out_path.stat().st_mode)
        combo_path.replace(out_path)
    except Exception:
        combo_path.unlink(missing_ok=True)
        raise
    for f in files[1:]:
        f.unlink()


class BmiVars:
    CATCHMENT_ID = "catchment_water_source__id"
    CATCHMENT_VALUE = "catchment_water_source__volume_flow_rate"
    NEXUS_ID = "land_surface_water_source__id"
    NEXUS_VALUE = "land_surface_water_source__volume_flow_rate"
    NGEN_DT = "ngen_dt"
    UPSTREAM_ID = "upstream_id"


class RunSet(TypedDict):
    """One forcing window handed to ``nwm_route``."""

    nts: int
    qlats: pd.DataFrame
    t0: datetime
    final_timestamp: datetime


# Peak RSS has two terms, and which dominates depends on the domain. Swept on one
# machine over two domains (benchmark/RESULTS.md 5b): read as ONE per-element constant
# Ohio gives 409 B and CONUS 31 B, off by 13x. The per-column term is forcing I/O,
# frame assembly and output, and it is 94% of a small domain's slope, which is why
# fitting Ohio alone put CONUS at 152 GB a window.
PER_COLUMN_BYTES = 52_000_000
BASE_PER_ELEMENT_BYTES = 28

# Same sweep with the DA on. It lands mostly per COLUMN; the old model had it as
# +34 B per element and nothing per column. Measured with travel_time_lag OFF, and the
# drivers return Courant only for the lag's trace, so this carries no Courant block.
SCALING_PER_COLUMN_BYTES = 68_000_000
SCALING_PER_ELEMENT_DELTA = 13

# Never swept: declared widths. Courant is 3-wide float32, the lag adds _assemble_cn's
# [nt, N_seg] float64 trace. They land on the widest window, since the span sizes it.
COURANT_PER_ELEMENT_BYTES = 17
LAG_TRACE_PER_ELEMENT_BYTES = 11

# Measured flat in the pool size: CONUS tree PSS 27.8 GB against 24.6 GB main-process
# at cpu_pool 8. Workers route their own clusters and share the parent's pages.
POOL_OVERHEAD = 1.15


def per_column_bytes(scaling: bool) -> int:
    """Bytes per forcing column that do NOT scale with the domain.

    Forcing I/O, frame assembly and output, plus the DA's observation handling when it
    is on. This is ~94% of a small domain's slope and ~13% of CONUS's, which is exactly
    why a model fitted on one domain cannot be trusted on the other.
    """
    return PER_COLUMN_BYTES + (SCALING_PER_COLUMN_BYTES if scaling else 0)


def per_element_bytes(courant: bool, scaling: bool, lag: bool = False) -> int:
    """Bytes per link-timestep, the term that DOES scale with the domain.

    flowveldepth is 4-wide float32 and upstream_array 1-wide, so 20 B declared against
    28 B measured. Additive, because the arrays are: a lag run allocates the DA's
    frames AND a Courant block AND the trace, and folding them together under-read the
    one window the lag makes widest.
    """
    total = BASE_PER_ELEMENT_BYTES
    if scaling:
        total += SCALING_PER_ELEMENT_DELTA
    if courant or lag:
        total += COURANT_PER_ELEMENT_BYTES
    if lag:
        total += LAG_TRACE_PER_ELEMENT_BYTES
    return total


class Model:
    dt: int

    # Warmstate WITH the upstream DA correction, recorded per window for
    # create_state(). Class-level so __new__-built instances have it.
    _seeded_q0 = None

    def __init__(self, config_file: str, start_time: float):
        self._time = start_time
        self._timings = {
            "forcing_time": 0.0,
            "route_time": 0.0,
            "output_time": 0.0,
            "network_creation_time": 0.0,
        }

        with open(config_file) as reader:
            data = yaml.load(reader, Loader=yaml.SafeLoader)
        self._config: dict = Config.with_strict_mode(**data).model_dump()
        # Anything already in the output directory belongs to another run (an earlier
        # AnA cycle, typically) and must survive this one's merge.
        _stream = (self.output_parameters or {}).get("stream_output")
        self._preexisting_output: frozenset[str] = frozenset(
            f.name
            for f in Path(_stream["stream_output_directory"]).glob("troute_output_*")
        ) if isinstance(_stream, dict) and _stream.get("stream_output_directory") else frozenset()
        # Same for lake output: its merge also replaces files[0] and deletes the
        # rest, so a later AnA cycle would destroy the earlier cycle's lake file.
        _lake = (self.output_parameters or {}).get("lakeout_output")
        self._preexisting_lakeout: frozenset[str] = frozenset(
            f.name for f in Path(_lake).glob("troute_lakeout_*.nc")
        ) if isinstance(_lake, Path) else frozenset()

        self.dt = int(self.forcing_parameters["dt"])

        LOG.info("Creating network of type " + self.supernetwork_parameters.get("network_type"))
        network_start_time = time.time()
        if self.supernetwork_parameters["network_type"] == "HYFeaturesNetwork":
            self._network = HYFeaturesNetwork(
                supernetwork_parameters=self.supernetwork_parameters,
                waterbody_parameters=self.waterbody_parameters,
                data_assimilation_parameters=self.data_assimilation_parameters,
                restart_parameters=self.restart_parameters,
                compute_parameters=self.compute_parameters,
                forcing_parameters=self.forcing_parameters,
                hybrid_parameters=self.hybrid_parameters,
                preprocessing_parameters=self.preprocessing_parameters,
                output_parameters=self.output_parameters,
                verbose=self.verbose,
                showtiming=self.show_timing,
            )
        elif self.supernetwork_parameters["network_type"] == "NHDNetwork":
            self._network = NHDNetwork(
                supernetwork_parameters=self.supernetwork_parameters,
                waterbody_parameters=self.waterbody_parameters,
                restart_parameters=self.restart_parameters,
                forcing_parameters=self.forcing_parameters,
                compute_parameters=self.compute_parameters,
                data_assimilation_parameters=self.data_assimilation_parameters,
                hybrid_parameters=self.hybrid_parameters,
                output_parameters=self.output_parameters,
                verbose=self.verbose,
                showtiming=self.show_timing,
            )
        elif self.supernetwork_parameters["network_type"] == "NHF":
            self._network = NHF(
                supernetwork_parameters=self.supernetwork_parameters,
                waterbody_parameters=self.waterbody_parameters,
                data_assimilation_parameters=self.data_assimilation_parameters,
                restart_parameters=self.restart_parameters,
                compute_parameters=self.compute_parameters,
                forcing_parameters=self.forcing_parameters,
                hybrid_parameters=self.hybrid_parameters,
                preprocessing_parameters=self.preprocessing_parameters,
                output_parameters=self.output_parameters,
                verbose=self.verbose,
                showtiming=self.show_timing,
                from_files=True,
                bmi_parameters=self.bmi_parameters,
            )
        else:
            raise Exception("Supernetwork network type must be HYFeaturesNetwork, NHDNetwork, or NHF")
        if not self._is_nhf():
            self._network.assemble_coastal_coupling_data()
        self._orig_t0 = self._network.t0
        self._timings["network_creation_time"] = time.time() - network_start_time

        # Data data assimilation
        LOG.debug("Creating DataAssimilation object")
        forcing_start_time = time.time()
        da_run = {}
        if self.data_assimilation_parameters:
            run_set = {
                "nts": self.nts,
                "final_timestamp": self.t0 + timedelta(seconds=self.nts * self.dt)
            }
            da_sets = hnu.build_da_sets(self.data_assimilation_parameters, [run_set], self._network.t0)
            if da_sets:
                da_run = da_sets[0]
        self._data_assimilation = DataAssimilation(
            network=self._network,
            data_assimilation_parameters=self.data_assimilation_parameters,
            # Not an empty dict: the observation readers need dt and nts. With them
            # missing, file-based nudging computed its resampling frequency from
            # dt=None, and the climatological diversion fallback fell back to
            # dt=300/nts=0 and built a single column, which the kernel (indexing from
            # timestep 1) never reads.
            run_parameters={
                "dt": self.dt,
                "nts": self.nts,
                "cpu_pool": self.cpu_pool,
            },
            waterbody_parameters=self.waterbody_parameters,
            from_files=True,
            value_dict=None,
            da_run=da_run,
        )
        self._timings["forcing_time"] = time.time() - forcing_start_time

        # Build the scaling DA if enabled (NHF only), mirroring the -V5 driver.
        self._scaling_da = None
        if self._is_nhf():
            self._scaling_da = build_scaling_da(
                self._network, self.supernetwork_parameters, self.data_assimilation_parameters,
                cpu_pool=self.cpu_pool
            )

        # Pass empty subnetwork list to nwm_route. These objects will be calculated/populated
        # on first iteration of for loop only. For additional loops this will be passed
        # to function from inital loop.
        self._subnetwork = [None, None, None]

    def _warn_cross_update_cadence(self) -> None:
        """Say once that the caller's update cadence is part of the answer.

        Every guard in _build_run_sets covers windows WITHIN one update. Nothing
        crosses between update() calls, so with a DA span the same forcing split into
        different update lengths differs near every boundary. Carrying the halo would
        withhold an update's output until the next arrives, so this is disclosed
        rather than fixed.
        """
        if getattr(self, "_warned_cadence", False) or not getattr(self, "_has_routed", False):
            return
        if getattr(self, "_scaling_da", None) is None or not self._span_columns():
            return
        self._warned_cadence = True
        LOG.warning(
            "more than one update in a scaling DA run with a span of %d forcing "
            "timestep(s): each update closes its own final window, so the same forcing "
            "split into different update lengths gives different discharge near every "
            "boundary. Keep the cadence fixed across runs you mean to compare.",
            self._span_columns(),
        )

    def run(self, bmi_values: dict[str, NDArray]):
        self._warn_cross_update_cadence()
        self._has_routed = True
        is_nhf = self._is_nhf()
        qts_subdivisions = self.qts_subdivisions

        LOG.debug("Assembling forcing dataframe")
        forcing_start_time = time.time()
        qlats = self._construct_qlats(bmi_values)
        LOG.debug(str(qlats))
        self._timings["forcing_time"] += time.time() - forcing_start_time

        # Build param_df
        param_df = self._network.dataframe
        if is_nhf:
            qlat_add_loc = "bottom"
        else:
            qlat_add_loc = "middle"

        # full_results = None
        # Materialized: the hand-off logic needs to know the window count.
        run_sets = list(self._build_run_sets(qlats))
        # One TimeSlice list for the whole update: a per-window list makes the
        # injected observations depend on how max_loop_size partitions it.
        scaling_da_run = (
            self._update_da_run(run_sets) if self._scaling_da is not None else None
        )
        # The travel-time lag's deferral, WITHIN this update call only: a
        # non-final window's spread waits for the next window's innovation (its
        # halo). Local by construction, so nothing is ever pending across
        # create_state()/load_state() boundaries; each update's final window
        # closes with decayed persistence, exactly like the run's final window
        # on the -V5 driver. The halo's price: two windows of run_results stay
        # resident, and an exception mid-update loses the pending window's
        # unwritten output -- recovery is checkpoint-restart (load_state), the
        # same contract a mid-update failure already had before the deferral.
        pending = None
        for run in run_sets:
            LOG.debug("Starting routing function")
            route_start_time = time.time()
            # Inject gage obs into the MC nudging override BEFORE usgs_df is read
            # below (nwm_route gets the local binding, so injecting later leaves the
            # kernel on the previous window's frame). The injected gage set is fixed,
            # keeping the cached execution plan valid across runs.
            if self._scaling_da is not None:
                from nwm_routing.scaling_da_apply import merge_injected_obs

                # One list spanning this update: t0 advances every update_until
                # call, and a per-window list would tie the observations to
                # max_loop_size.
                self._data_assimilation._usgs_df = merge_injected_obs(  # pyright: ignore[reportPrivateUsage]
                    self._scaling_da.build_usgs_df(
                        run["t0"], self.dt, run["nts"], scaling_da_run
                    ),
                    self._data_assimilation.usgs_df,
                )

            usgs_df = self._data_assimilation.usgs_df
            if not usgs_df.empty:
                # Trims run-spanning nudging/diversion frames; no-op for the
                # injected frame, whose columns already start at t0.
                usgs_df = usgs_df.loc[:,run["t0"]:]

            run_results, self._subnetwork = nwm_routing.nwm_route(
                downstream_connections=self._network.connections,
                upstream_connections=self._network.reverse_network,
                waterbodies_in_connections=self._network.waterbody_connections,
                reaches_bytw=self._network._reaches_by_tw,
                parallel_compute_method=self.compute_parameters.get("parallel_compute_method", "serial"),
                compute_kernel=self.compute_parameters.get("compute_kernel"),
                subnetwork_target_size=self.compute_parameters.get('subnetwork_target_size'),
                cpu_pool=self.cpu_pool,
                t0=run["t0"],
                dt=self.dt,
                nts=run["nts"],
                qts_subdivisions=qts_subdivisions,
                independent_networks=self._network.independent_networks,
                param_df=param_df,
                q0=self._network.q0,
                qlats=run.get("qlats", qlats),
                eloss_df=self._network._eloss if self._network._eloss is not None else pd.DataFrame(0.0, index=qlats.index, columns=qlats.columns),
                ssout=self.forcing_parameters.get("ssout"),
                # The window-sliced frame computed above, not the full one. Passing
                # the unsliced frame restarted nudging and the donor subtraction from
                # observation column zero on every BMI update.
                usgs_df=usgs_df,
                lastobs_df=self._data_assimilation.lastobs_df,
                reservoir_usgs_df=self._data_assimilation.reservoir_usgs_df,
                reservoir_usgs_param_df=self._data_assimilation.reservoir_usgs_param_df,
                reservoir_usace_df=self._data_assimilation.reservoir_usace_df,
                reservoir_usace_param_df=self._data_assimilation.reservoir_usace_param_df,
                reservoir_usbr_df=self._data_assimilation.reservoir_usbr_df,
                reservoir_usbr_param_df=self._data_assimilation.reservoir_usbr_param_df,
                reservoir_rfc_df=self._data_assimilation.reservoir_rfc_df,
                reservoir_rfc_param_df=self._data_assimilation.reservoir_rfc_param_df,
                great_lakes_df=self._data_assimilation.great_lakes_df,
                great_lakes_param_df=self._data_assimilation.great_lakes_param_df,
                great_lakes_climatology_df=self._network.great_lakes_climatology_df,
                da_parameter_dict=self._data_assimilation.assimilation_parameters,
                assume_short_ts=self.compute_parameters.get('assume_short_ts', False),
                # Forced on for ROUTING when a traced travel-time shift is
                # configured: the trace reads r[2] for each reach's transit. The
                # OUTPUT call below keeps reading the config value, so this does
                # not start writing Courant output files. Mirrors the -V5 driver.
                # Dropped again once the trace has filled its cache: after that
                # the kernel would fill an [n_seg, nts*3] block per window that
                # nothing reads. Mirrors the -V5 driver.
                # Two requests, not one: the user's is permanent and feeds the
                # output writer, the trace's is dropped once its cache is full.
                return_courant=self.compute_parameters.get('return_courant', False) or (
                    getattr(getattr(self, "_scaling_da", None), "travel_time_lag", False)
                    and not self._scaling_da._trace_cached(self._scaling_da.trees)
                ),
                waterbodies_df=self._network._waterbody_df,
                data_assimilation_parameters=self.waterbody_parameters,
                waterbody_types_df=self._network._waterbody_types_df,
                waterbody_type_specified=self._network.waterbody_type_specified,
                diffusive_network_data=self._network.diffusive_network_data,
                topobathy_df=self._network.topobathy_df,
                refactored_diffusive_domain=self._network.refactored_diffusive_domain,
                refactored_reaches=self._network.refactored_reaches,
                subnetwork_list=self._subnetwork,
                coastal_boundary_depth_df=self._network.coastal_boundary_depth_df,
                unrefactored_topobathy_df=self._network.unrefactored_topobathy_df,
                qlat_add_loc=qlat_add_loc,
                # Without this the parameter defaults to an empty dict and the
                # diversion is silently disabled under BMI, which is the path ngen
                # drives. NHF only: other network types do not resolve this map.
                diversion_da=getattr(self._network, "diversion_da", {}) or {},
                # Same static split points as the -V5 driver: the plan is cached
                # across updates, so it must not depend on this window's data. Same
                # helper as the -V5 driver so both plans split at an identical set.
                gage_segments=network_gage_segments(self._network),
            )

            # The warmstate that CONTINUES the run is taken BEFORE the upstream
            # spread: a seeded q0 would debit the next window's innovation by the
            # amount injected (measured: -23.8% held-out bias with one seeding
            # boundary vs -29.1% with eleven).
            self._network.new_q0(run_results)
            self._network.update_waterbody_water_elevation()

            # update reservoir parameters and lastobs_df
            self._data_assimilation.update_after_compute(run_results, self.dt * run["nts"])

            self._timings["route_time"] += time.time() - route_start_time

            LOG.debug("Generating output")
            output_start_time = time.time()

            def _write_output(_res, _t0, _nts):
                nwm_output_generator(
                    run={"t0": _t0, "dt": self.dt, "nts": _nts},
                    results=_res,
                    supernetwork_parameters=self.supernetwork_parameters,
                    output_parameters=self.output_parameters,
                    parity_parameters=self.parity_parameters,
                    restart_parameters=self.restart_parameters,
                    parity_set={},
                    qts_subdivisions=qts_subdivisions,
                    return_courant=self.compute_parameters.get("return_courant", False),
                    cpu_pool=self.cpu_pool,
                    waterbodies_df=self._network.waterbody_dataframe,
                    waterbody_types_df=self._network.waterbody_types_dataframe,
                    duplicate_ids_df=getattr(self._network, "_duplicate_ids_df", pd.DataFrame()),
                    data_assimilation_parameters=self.data_assimilation_parameters,
                    lastobs_df=self._data_assimilation.lastobs_df,
                    link_gage_df=self._network.link_gage_df,
                    link_lake_crosswalk=self._network.link_lake_crosswalk,
                    nexus_dict=self._network.nexus_dict,
                    poi_crosswalk=self._network.poi_nex_dict or {},
                    fp_outlet_crosswalk=self._network.fp_outlet_crosswalk if is_nhf else None,
                )

            if self._scaling_da is None:
                _write_output(run_results, self.t0, run["nts"])
            else:
                if pending is not None:
                    # Flush the previous window now that its halo exists. Its
                    # spread is output-only: the routing chain above already
                    # continued from the uncorrected warmstate.
                    p_res, p_nts, p_run_t0, p_out_t0 = pending
                    self._scaling_da.apply_in_kernel(
                        p_res, p_nts, self.dt, p_run_t0,
                        halo=self._scaling_da.gather_innovation(run_results),
                    )
                    _write_output(p_res, p_out_t0, p_nts)
                    pending = None
                if run is run_sets[-1]:
                    # The update's final window: no future innovation exists, so
                    # the spread closes with decayed persistence -- the same
                    # semantics as the run's final window on the -V5 driver.
                    self._scaling_da.apply_in_kernel(
                        run_results, run["nts"], self.dt, run["t0"],
                        seed_untimed=True,
                    )
                    # Record (without installing) the seeded hand-off q0 for
                    # create_state(). Under ngen every window is the last of its
                    # own update() call, so this runs on precisely the windows a
                    # checkpoint can observe -- an index-based final-window test
                    # across updates would seed every update (the -29.1% case).
                    cycling_q0 = self._network._q0  # pyright: ignore[reportPrivateUsage]
                    seeded = self._network.new_q0(run_results).copy()
                    self._network._q0 = cycling_q0  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

                    # Depth is deliberately NOT corrected: MC uses h0 only as a
                    # solver seed, and there is no defensible dQ -> dh mapping at
                    # this layer. Log the injected volume so the increment is
                    # auditable.
                    if cycling_q0 is not None and "qd0" in seeded.columns:
                        shared = seeded.index.intersection(cycling_q0.index)
                        dq = (
                            seeded.loc[shared, "qd0"].to_numpy(dtype="float64")
                            - cycling_q0.loc[shared, "qd0"].to_numpy(dtype="float64")
                        )
                        n_changed = int((dq != 0.0).sum())
                        if n_changed:
                            LOG.info(
                                "scaling DA: analysis hand-off state seeded -- %d of %d "
                                "segments corrected, net dQ %+.4f cms (volume %+.1f m3 over "
                                "one dt); h0 is the uncorrected routed depth (MC uses it as "
                                "a solver seed only).",
                                n_changed, len(shared), float(dq.sum()),
                                float(dq.sum()) * float(self.dt),
                            )
                    self._seeded_q0 = seeded
                    _write_output(run_results, self.t0, run["nts"])
                else:
                    # Held for the next window's halo; run["t0"] feeds the
                    # model-time arithmetic and self.t0 the output writer.
                    pending = (run_results, run["nts"], run["t0"], self.t0)

            self._network.new_t0(self.dt, run["nts"])

            self._timings["output_time"] += time.time() - output_start_time
            del run_results  # free space for the next run (pending keeps its ref)

        # update time as (ngen dt in seconds) * (number of steps processed)
        self._time += self.ngen_dt(bmi_values) * qlats.shape[1]
        self._merge_run_results()

    def log_times(self):
        if self.show_timing and all(self._timings.values()):
            process_time = sum(self._timings.values())
            def sec_and_per(title, key: str):
                seconds = round(self._timings[key], 2)
                percent = round(self._timings[key] / process_time * 100, 2)
                LOG.info(f"{title}: {seconds} secs, {percent} %")
            LOG.debug(f"Processes complete in {process_time} seconds.")
            LOG.info('************ TIMING SUMMARY ************')
            LOG.info('----------------------------------------')
            sec_and_per("Network graph construction", 'network_creation_time')
            sec_and_per("Forcing array construction", "forcing_time")
            sec_and_per("Routing computations", "route_time")
            sec_and_per("Output writing", "output_time")
            LOG.info(f"Total execution time: {round(process_time, 2)} secs")

    def create_state(self):
        """Create a dictionary of data that can be serialized using `pickle.dumps`."""
        return {
            "time": self._time,
            # BOTH warmstates: "q0" is the cycling background (a resumed analysis
            # must start from it, or its next innovation is debited), "seeded_q0"
            # the corrected hand-off a forecast must inherit. load_state chooses.
            "q0": self._network._q0,  # pyright: ignore[reportPrivateUsage]
            "seeded_q0": self._seeded_q0,
            "t0": self._network._t0,
            # updated data stored on DataAssimilation
            "last_obs": self._data_assimilation._last_obs_df,
            "usgs": self._data_assimilation._reservoir_usgs_param_df,
            "usace": self._data_assimilation._reservoir_usace_param_df,
            # USBR persistence state is updated every window alongside USGS and
            # USACE (_set_persistence_reservoir_da_params), so omitting it made a
            # restarted run diverge from an uninterrupted one at type-7 reservoirs.
            "usbr": self._data_assimilation._reservoir_usbr_param_df,
            "rfc": self._data_assimilation._reservoir_rfc_param_df,
            "gl": self._data_assimilation._great_lakes_param_df,
            # Which reservoir DA types produced the frames above. Provenance: it is
            # what distinguishes a deliberately empty frame from a starved one when
            # reading a checkpoint back.
            "reservoir_da_enabled": self._reservoir_da_enabled(),
            # Result-determining state: without it a resumed run retraces from
            # its own first window and shifts corrections.
            "scaling_tau": (
                self._scaling_da.trace_checkpoint()
                if getattr(self, "_scaling_da", None) is not None
                else None
            ),
        }

    def _reservoir_da_enabled(self) -> dict[str, bool]:
        """Which reservoir DA types this run has switched on, read from its config.

        Whether a frame is empty cannot answer this: a type that is ON but has seen
        no observations yet is empty too. The flags are spelled here exactly as
        DataAssimilation reads them, so the two cannot drift apart silently.
        """
        da = self.data_assimilation_parameters or {}
        reservoir = da.get("reservoir_da") or {}
        persistence = reservoir.get("reservoir_persistence_da") or {}
        rfc = reservoir.get("reservoir_rfc_da") or {}
        return {
            "usgs": bool(persistence.get("reservoir_persistence_usgs", False)),
            "usace": bool(persistence.get("reservoir_persistence_usace", False)),
            "usbr": bool(persistence.get("reservoir_persistence_usbr", False)),
            "rfc": bool(rfc.get("reservoir_rfc_forecasts", False)),
            "gl": bool(persistence.get("reservoir_persistence_greatLake", False)),
        }

    def _compatible_lastobs(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop lastobs rows this run does not assimilate.

        Scaling's roster is a subset of nudging's, and the checkpoint records no
        producer mode, so a nudging frame raises "not in index" downstream.
        """
        da = getattr(self, "_scaling_da", None)
        if da is None or frame is None or frame.empty:
            return frame
        roster = {int(seg) for seg in getattr(da, "gage_seg", {}).values()}
        if not roster:
            return frame
        keep = frame.index.isin(roster)
        if keep.all():
            return frame
        LOG.warning(
            "load_state: dropped %d lastobs row(s) the scaling DA does not "
            "assimilate (holdout, reservoir-routed or co-located gages); the "
            "checkpoint was written by a run with a wider roster.",
            int((~keep).sum()),
        )
        return frame[keep]

    def _owns_lastobs(self) -> bool:
        """Whether this run owns the lastobs frame, so a restore must keep it.

        Both streamflow arms do; asking only about nudging made a scaling run
        discard its own checkpoint.
        """
        sda = (self.data_assimilation_parameters or {}).get("streamflow_da") or {}
        return bool(sda.get("streamflow_nudging", False)
                    or sda.get("streamflow_scaling", False))

    @staticmethod
    def _restore_da_frame(
        saved: pd.DataFrame | None,
        live: pd.DataFrame,
        label: str,
        *,
        advanced: bool,
        live_on: bool | None = None,
    ) -> pd.DataFrame:
        """Install a saved DA frame unless it would erase live state.

        A state written with this type off carries an empty frame; installing it
        over a live one leaves observations without parameters. ``live_on`` None
        leaves the emptiness rules in charge.
        """
        if live_on is False:
            # A type this run does not run must not adopt the checkpoint's rows.
            # Adopting them lets a frame ride through a run that never updates it
            # and be re-serialized as if this run had produced it, so the NEXT run
            # to switch the type on inherits values that are stale by however long
            # it stayed off.
            if saved is not None and not saved.empty:
                LOG.warning(
                    "load_state: %s is switched off in this run, so the "
                    "checkpoint's rows are dropped rather than carried through to "
                    "the state this run writes.",
                    label,
                )
            return live
        if saved is not None and (not saved.empty or live.empty):
            return saved
        if advanced and not live.empty:
            # Keeping live state is only right while it still describes the
            # checkpoint's time. Past that, neither frame does, and the kernel reads
            # update_time and timeseries_idx straight from this one. Whether the
            # checkpoint recorded the type as off does not change that; switching a
            # type on mid-cycle is unaffected, since that happens on a fresh model.
            msg = (
                f"load_state: the checkpoint carries no {label} but this model has "
                "already routed, so its live state is past the checkpoint's time. "
                "Restore into a fresh model, or use a checkpoint written with this "
                "DA type on."
            )
            raise ValueError(msg)
        LOG.warning(
            "load_state: checkpoint carries no %s, so this run keeps what it "
            "built; no DA state carries over.",
            label,
        )
        return live

    def load_state(self, data: dict):
        # Whether the live DA frames still describe the checkpoint's time. Not
        # `self._time`, which load_state overwrites and reset_time zeroes.
        advanced = bool(getattr(self, "_has_routed", False))
        da = self._data_assimilation
        restore = partial(self._restore_da_frame, advanced=advanced)
        # Absent from checkpoints written before the flags were recorded, which
        # leaves those restores on the emptiness rules alone.
        live_on = self._reservoir_da_enabled()

        def restore_reservoir(
            key: str, saved: pd.DataFrame | None, live: pd.DataFrame, label: str
        ) -> pd.DataFrame:
            return self._restore_da_frame(
                saved, live, label, advanced=advanced, live_on=live_on[key],
            )
        # Resolve every frame BEFORE mutating anything: a refused restore must
        # leave the model as it was, or an identical retry accepts it.
        # lastobs gets the reservoir treatment too: time_since_lastobs is relative,
        # so carrying a stale frame hands the next run day-old obs as current.
        resolved = {
            "last_obs": restore(
                data["last_obs"], da._last_obs_df, "last observations",
                live_on=self._owns_lastobs()),
            "usgs": restore_reservoir(
                "usgs", data["usgs"], da._reservoir_usgs_param_df,
                "USGS reservoir DA parameters"),
            "usace": restore_reservoir(
                "usace", data["usace"], da._reservoir_usace_param_df,
                "USACE reservoir DA parameters"),
            "rfc": restore_reservoir(
                "rfc", data["rfc"], da._reservoir_rfc_param_df,
                "RFC reservoir DA parameters"),
            "gl": restore_reservoir(
                "gl", data["gl"], da._great_lakes_param_df,
                "Great Lakes DA parameters"),
        }
        # .get for backward compatibility with state files written before USBR
        # persistence state was included.
        if "usbr" in data:
            resolved["usbr"] = restore_reservoir(
                "usbr", data["usbr"], da._reservoir_usbr_param_df,
                "USBR reservoir DA parameters")

        # Nothing below here may raise.
        self._has_routed = False
        self._time = data["time"]
        # Install the warmstate this model will need: cycling background when it
        # will keep assimilating (seeded would debit the next innovation), seeded
        # hand-off for a forecast. Pre-split state files carry only "q0".
        seeded = data.get("seeded_q0")
        if seeded is not None and getattr(self, "_scaling_da", None) is None:
            self._network._q0 = seeded
        else:
            self._network._q0 = data["q0"]
        # RETAIN the loaded hand-off so load -> create round-trips reproduce the
        # state (a re-serialized checkpoint must not drop the forecast hand-off);
        # Model.run overwrites it every routed window, so it cannot go stale.
        self._seeded_q0 = seeded
        self._network._t0 = data["t0"]
        da._last_obs_df = self._compatible_lastobs(resolved["last_obs"])
        da._reservoir_usgs_param_df = resolved["usgs"]
        da._reservoir_usace_param_df = resolved["usace"]
        if "usbr" in resolved:
            da._reservoir_usbr_param_df = resolved["usbr"]
        da._reservoir_rfc_param_df = resolved["rfc"]
        da._great_lakes_param_df = resolved["gl"]
        if getattr(self, "_scaling_da", None) is not None:
            # Restore-or-invalidate, never keep: a stale own-trace surviving a
            # load is the divergence this entry exists to prevent.
            self._scaling_da.restore_trace_checkpoint(
                data.get("scaling_tau"), dt=float(self.dt)
            )
        self._network.update_waterbody_water_elevation()

    def reset_time(self):
        self._time = 0.0
        self._network.t0 = self._orig_t0

    @property
    def nts(self) -> int:
        return self.forcing_parameters["nts"]

    @property
    def cpu_pool(self) -> int:
        return self.compute_parameters["cpu_pool"]

    @property
    def bmi_parameters(self) -> dict:
        return self._config.get("bmi_parameters", {})

    @property
    def log_parameters(self) -> dict:
        return self._config.get("log_parameters", {})

    @property
    def compute_parameters(self) -> dict:
        return self._config.get("compute_parameters", {})

    @property
    def network_topology_parameters(self) -> dict:
        return self._config.get("network_topology_parameters", {})

    @property
    def output_parameters(self) -> dict:
        return self._config.get("output_parameters", {})

    @property
    def preprocessing_parameters(self) -> dict:
        return self.network_topology_parameters.get("preprocessing_parameters", {})

    @property
    def waterbody_parameters(self) -> dict:
        return self.network_topology_parameters.get("waterbody_parameters", {})

    @property
    def supernetwork_parameters(self) -> dict:
        return self.network_topology_parameters.get("supernetwork_parameters", {})

    @property
    def forcing_parameters(self) -> dict:
        return self.compute_parameters.get("forcing_parameters", {})

    @property
    def restart_parameters(self) -> dict:
        return self.compute_parameters.get("restart_parameters", {})

    @property
    def hybrid_parameters(self) -> dict:
        return self.compute_parameters.get("hybrid_parameters", {})

    @property
    def data_assimilation_parameters(self) -> dict:
        return self.compute_parameters.get("data_assimilation_parameters", {})

    @property
    def parity_parameters(self) -> dict:
        return self.output_parameters.get("wrf_hydro_parity_check", {})

    @property
    def show_timing(self):
        return bool(self.log_parameters.get("showtiming"))

    @property
    def verbose(self):
        log_level = self.log_parameters.get("log_level")
        if isinstance(log_level, str):
            return log_level.upper() == "DEBUG"
        elif isinstance(log_level, (int, float)):
            return log_level == 10
        return False

    @property
    def time(self) -> float:
        return self._time

    @property
    def t0(self) -> datetime:
        return self._network.t0

    @property
    def qts_subdivisions(self) -> int:
        return self.forcing_parameters["qts_subdivisions"]

    def ngen_dt(self, bmi_values: dict[str, NDArray]) -> int:
        if len(bmi_values.get(BmiVars.NGEN_DT, [])) == 1:
            dt = bmi_values[BmiVars.NGEN_DT][0]
            if dt > 0:
                return int(dt)
        # backup if NGEN's delta time was not explicitly set
        return int(self.dt * self.qts_subdivisions)

    def _update_da_run(self, run_sets: list[RunSet]) -> dict | None:
        """The TimeSlice list spanning every window of this update call.

        Separate update calls still differ: real-time semantics, not a defect.
        """
        from nwm_routing.scaling_da_apply import span_da_runs

        return span_da_runs(self._window_da_run(run) for run in run_sets)

    def _window_da_run(self, run: RunSet) -> dict | None:
        """The TimeSlice file list covering one routing window.

        Same builder, same lookback padding and same filename convention the -V5
        driver and the nudging path use; only the window differs.
        """
        if not self.data_assimilation_parameters:
            return None
        da_sets = hnu.build_da_sets(
            self.data_assimilation_parameters,
            [{"final_timestamp": run["final_timestamp"]}],
            run["t0"],
        )
        return da_sets[0] if da_sets else None

    def _span_columns(self) -> int:
        """The DA's own horizon, in forcing columns. Zero when no span is active.

        The forward innovation window reads past the end of a window into the next
        one's innovation, and that halo is exactly ONE window deep, so a window
        shorter than this leaves its own tail uncovered and the result starts
        depending on the partition. getattr with the ScalingDA class defaults,
        since tests stub _scaling_da.
        """
        scaling_da = getattr(self, "_scaling_da", None)
        if scaling_da is None:
            return 0
        spread_h = float(getattr(scaling_da, "innovation_spread_h", 0.0))
        # SUM, not max: the lag reads the SMOOTHED innovation at t + tau, so the tail
        # of a window needs raw innovation out to tau_max + spread. The lag is measured
        # over a fixed span taken from the first window, so that window must hold it,
        # or the span follows max_loop_size and a memory knob changes discharge.
        if getattr(scaling_da, "travel_time_lag", False):
            spread_h += float(getattr(scaling_da, "lag_window_h", 48.0))
        col_s = float(self.qts_subdivisions) * float(self.dt)
        return math.ceil(spread_h * 3600.0 / col_s)

    def _build_run_sets(self, qlats: pd.DataFrame) -> Iterator[RunSet]:
        nts = len(qlats.columns)
        # Memory is a SAFETY CAP only, never the primary window control.
        # Bytes per link-routing-step, from the arrays the kernel actually allocates
        # rather than one flat constant: flowveldepth is 4-wide float32 (16 B) and
        # upstream_array 1-wide (4 B) always; the Courant block adds 3-wide float32
        # (12 B) only when it is returned; the scaling DA holds q_model, its copy and
        # the corrected frame in float64 (24 B) only when it is active. Workers each
        # hold their own job's arrays, so the pool multiplies the transient.
        _scaling_now = getattr(self, "_scaling_da", None) is not None
        # Courant follows the LAG, not the DA: the drivers return it for the
        # travel-time trace. Tying it to the DA missed the trace on lag-on runs.
        _lag = _scaling_now and bool(
            getattr(self._scaling_da, "travel_time_lag", False)
        )
        _courant = bool(self.compute_parameters.get("return_courant", False))
        per_element = per_element_bytes(_courant, _scaling_now, _lag)
        # Workers hold per-CLUSTER payloads, not domain copies, so the pool is a small
        # constant and not a multiplier: benchmark/RESULTS.md measures CONUS tree PSS at
        # cpu_pool 8 as 27.8 GB against 24.6 GB main-process, and Tier A at parity. A
        # linear factor sized VPU01 into 2-column windows. effective_n_jobs because
        # joblib reads -1 as every core, so it is a pool and not serial.
        workers = max(1, effective_n_jobs(self.compute_parameters.get("cpu_pool") or 1))
        pool_overhead = POOL_OVERHEAD if workers > 1 else 1.0
        # Two terms: one that scales with the domain and one that does not.
        column_bytes = int(
            (per_column_bytes(_scaling_now)
             + per_element * qlats.shape[0] * self.qts_subdivisions)
            * pool_overhead
        )
        # No intercept term: available_memory is read HERE, after the network, plan and
        # forcing are already resident, so the baseline is excluded by construction.
        # only account for 90% of the memory this process may actually use
        _budget, _budget_source = job_memory_headroom(
            int(psutil.virtual_memory().available),
            int(psutil.Process().memory_info().rss),
        )
        if _budget_source != "host":
            # Named because an operator seeing a small window needs to know which.
            LOG.info(
                "memory budget taken from the %s: %.1f GB. The host reports more free "
                "than this job may use.", _budget_source, _budget / 1e9,
            )
        available_memory = _budget * 0.9
        if available_memory <= 0:
            # A cgroup at or over its limit. This used to be a ZeroDivisionError, and
            # "one column fits" would guess at a budget the kernel already refused.
            raise MemoryError(
                "this process has no memory budget left: its cgroup is at or over its "
                "limit. Raise the job's memory reservation, or free memory inside it."
            )
        # FLOOR of the byte budget: ceil(nts / ceil(required / available)) rounds the
        # wrong way, giving a 4-column window against a budget of 3.4, 18% over.
        # Clamped to the update, which the old ceil form was as a side effect: below,
        # loop_size < cfg_loop is read as "the update is shorter than the window".
        mem_loop_size = max(1, min(nts, int(available_memory // column_bytes)))
        mem_divisions = math.ceil(nts / mem_loop_size)

        # max_loop_size is the PRIMARY control (as in the -V5 driver): every
        # per-window DA operation lands on this partition, so a RAM-derived split
        # made results depend on machine load. Under active scaling DA the
        # partition is part of the RESULT (decay resets at window boundaries), so
        # a RAM cap below the configured window is a hard error there; without
        # the DA it stays a warning.
        scaling_active = getattr(self, "_scaling_da", None) is not None
        cfg_loop = self.forcing_parameters.get("max_loop_size") or 0
        # The DA's own span, in forcing columns. At zero span the spread is
        # output-only and per timestep, so a single-window update may be served
        # below max_loop_size without measurement.
        span_cols = self._span_columns()
        if scaling_active:
            if cfg_loop > 0 and span_cols > cfg_loop:
                LOG.info(
                    "scaling DA: max_loop_size enlarged %d -> %d forcing columns "
                    "so every non-final window covers innovation_spread_h.",
                    int(cfg_loop), span_cols,
                )
        # Only a DA with a span makes the partition part of the result. Without
        # one, a short update is served like any no-DA run: the NWM Standard AnA
        # is 3 forcing columns against a much longer window, and erroring there
        # would make the shipped operational config unrunnable.
        partition_matters = scaling_active and span_cols > 0
        auto = cfg_loop <= 0
        # The same call the CLI makes, output cadence included, so one config gives
        # one width on either driver. Only the memory cap below is this driver's own,
        # and it never sizes UP: the estimate is calibrated on one domain.
        _stream = self.output_parameters.get("stream_output") or {}
        _sot_h = _stream.get("stream_output_time") if isinstance(_stream, dict) else None
        _out_cols = (
            math.ceil(float(_sot_h) * 3600.0 / (self.qts_subdivisions * float(self.dt)))
            if _sot_h and float(_sot_h) > 0
            else 0
        )
        cfg_loop = resolve_window(cfg_loop, span_cols, _out_cols)

        if partition_matters and nts < span_cols:
            # BEFORE the memory guard: freeing memory cannot make a 24-column update
            # cover a 48-column span, and that message would send them chasing it.
            raise ValueError(
                f"this update supplies {nts} forcing timestep(s), fewer than the "
                f"scaling DA's span of {span_cols} (innovation_spread_h, plus "
                "lag_window_h when travel_time_lag is on). No window covers the span, "
                "so the DA's reach would be set by the update cadence. Memory is NOT "
                "the limit. Drive longer updates, or reduce the DA span; at "
                "innovation_spread_h 0 with the lag off the span is zero and this "
                "constraint disappears."
            )

        loop_size = min(int(cfg_loop), mem_loop_size)
        if loop_size < int(cfg_loop):
            # mem_divisions <= 1 means the cap is this update's own forcing:
            # one window, nothing partitioned. A RAM-driven split is many
            # windows, and a partition set by machine load stays fatal.
            # An explicit window is refused even at zero span, since someone pinned
            # it. Auto pinned nothing, so it refuses only below the span.
            if mem_divisions > 1 and scaling_active and (
                not auto or loop_size < span_cols
            ):
                remedy = (
                    f"Free memory, or reduce the DA span ({span_cols} timesteps), which "
                    "is what sets the floor under automatic sizing."
                ) if auto else "Free memory, or lower max_loop_size to a value that fits."
                raise MemoryError(
                    f"available memory caps the run window at {loop_size} forcing "
                    f"timesteps, below the {int(cfg_loop)} this run needs. A "
                    "RAM-derived split would make the window partition, and so the "
                    f"discharge, depend on current machine load. {remedy}"
                )
            # A window longer than the update cannot bound memory, so refusing over it
            # stops a run for nothing. The wall below is the one that cannot be dodged.
            if partition_matters and loop_size >= span_cols and mem_divisions == 1:
                # mem_divisions == 1 pins the cause to the update. Without it this
                # blamed update length on memory-capped runs too.
                LOG.warning(
                    "%s is %d but this update supplies %d forcing timestep(s), so the "
                    "scaling DA operates over %d. Discharge depends on the update "
                    "length, not on the window: the same run split into different "
                    "updates differs near every boundary.",
                    "automatic max_loop_size" if auto else "max_loop_size",
                    int(cfg_loop), nts, loop_size,
                )
            elif partition_matters and loop_size >= span_cols:
                # Auto accepts a machine-load partition where an explicit window
                # refuses, so on this path memory IS in the result.
                LOG.warning(
                    "available memory, not the configuration, set this run's window to "
                    "%d forcing timestep(s) under an active DA span. The partition is "
                    "part of the result, so this run is not reproducible on a "
                    "differently loaded machine. Pin max_loop_size to fix the window.",
                    loop_size,
                )
            if partition_matters and loop_size < span_cols:
                # Memory was never the limit here: the cap IS this update's own
                # forcing, so saying "free memory" sends the operator to the
                # wrong place entirely. Under auto the span is the only term left,
                # so naming max_loop_size would point at a knob nobody set.
                raise ValueError(
                    f"this update supplies {nts} forcing timestep(s), but the scaling "
                    f"DA needs windows of {int(cfg_loop)}"
                    + ("" if auto else " (max_loop_size, or the DA's span if larger)")
                    + f"; its span is {span_cols} (innovation_spread_h, plus "
                    "lag_window_h when travel_time_lag is on). That span is part of "
                    "the result, so it must not shrink silently to the update. Memory "
                    "is NOT the limit here. Drive longer updates, "
                    + ("" if auto else "lower max_loop_size to the update cadence, ")
                    + "or reduce the DA span; halving lag_window_h also halves the "
                    "longest resolvable travel time. At innovation_spread_h 0 with "
                    "the lag off the span is zero and this constraint disappears."
                )
            if mem_divisions > 1:
                LOG.warning(
                    "available memory caps the run window at %d forcing "
                    "timesteps, below the %d requested",
                    loop_size, int(cfg_loop),
                )
            else:
                # Every short update logs this, so it is not a warning: the run
                # is one window and nothing is partitioned.
                LOG.info(
                    "the run window is this update's own %d forcing timestep(s), "
                    "below the %d requested; nothing is partitioned.",
                    loop_size, int(cfg_loop),
                )

        # One partition rule for both drivers, see troute.window_plan. Filling and
        # folding disagreed with the CLI and pushed a window past the memory cap.
        bounds = plan_windows(nts, loop_size, span_cols)
        widest = max(stop - start for start, stop in bounds)
        if auto:
            # After the cap AND the split: an even split lands at or below the cap, so
            # neither the requested nor the capped width is what runs.
            LOG.info(
                "max_loop_size not set; automatic sizing chose %d forcing timestep(s) "
                "per window.", widest,
            )
        if widest > loop_size:
            # The one partition wider than was asked for, and max_loop_size is a
            # promise about memory, so not something to mention at INFO.
            LOG.warning(
                "no split holds every window at the scaling DA's span of %d forcing "
                "timestep(s), so all %d run as one window, wider than the %d asked "
                "for. Reduce the DA span to partition this update.",
                span_cols, nts, loop_size,
            )
        if widest > mem_loop_size:
            # Only reachable on the single-window fallback: no split holds every
            # window at the span, and the one window that does will not fit.
            raise MemoryError(
                f"covering the scaling DA's span of {span_cols} forcing timesteps "
                f"needs this {nts}-timestep update in one window, but available "
                f"memory caps it at {mem_loop_size}. Splitting would leave a window "
                "under the span, which changes discharge. Free memory, or reduce the "
                "DA span."
            )
        if len(bounds) == 1:
            yield {
                "nts": nts * self.qts_subdivisions,
                "qlats": qlats,
                "t0": self.t0,
                # From THIS update's forcing columns (like the multi-window branch),
                # never self.nts: that is the whole configured horizon, and with t0
                # advancing per update it enumerated TimeSlices far beyond the
                # window (O(horizon) I/O, no CLI parity, possible future leakage).
                "final_timestamp": datetime.strptime(str(qlats.columns[-1]), "%Y%m%d%H%M")
            }
        else:
            for start, stop in bounds:
                times = qlats.columns[start:stop]
                yield {
                    "nts": len(times) * self.qts_subdivisions,
                    "qlats": qlats[times],
                    # Columns are "%Y%m%d%H%M"; parsing with a trailing %S silently
                    # backtracks to the wrong time ("...1445" -> 14:04:05).
                    "t0": datetime.strptime(times[0], "%Y%m%d%H%M"),
                    "final_timestamp": datetime.strptime(times[-1], "%Y%m%d%H%M")
                }

    def _output_files(self, stream_params: dict) -> list[Path]:
        """This run's output parts, never another run's.

        The merge replaces files[0] and deletes the rest, so it must not reach files
        it did not write. An AnA cycle is a separate model over the same output
        directory: globbing the directory made each cycle swallow the previous
        cycle's merged result, leaving one file of overlapping timestamps and no way
        back. Anything already present when this model was built is off limits.
        """
        stream_type = stream_params.get("stream_output_type")
        foreign = getattr(self, "_preexisting_output", frozenset())
        return sorted(
            (f for f in Path(stream_params["stream_output_directory"]).glob(
                "troute_output_*" + stream_type) if f.name not in foreign),
            key=lambda f: f.stem,
        )

    def _merge_run_results(self):
        stream_params = self.output_parameters.get("stream_output")
        if isinstance(stream_params, dict):
            stream_type = stream_params.get("stream_output_type")
            files = self._output_files(stream_params)
            if len(files) > 1:
                start_time = time.time()

                def write_stream(dest: Path) -> None:
                    if stream_type == ".nc":
                        _write_merged_netcdf(files, dest)
                    elif stream_type == ".csv":
                        pd.concat(
                            (pd.read_csv(f) for f in files),
                            ignore_index=True
                        ).to_csv(dest, index=False)
                    elif stream_type == ".pkl":
                        pd.concat(
                            (pd.read_pickle(f) for f in files),
                            ignore_index=False
                        ).to_pickle(dest)
                    else:
                        err = f"Cannot merge output formats other than .nc, .csv, or .pkl. Format provided in the config file: {stream_type}"
                        LOG.error(err)
                        raise RuntimeError(err)

                _merge_into_first(files, write_stream)
                self._timings["output_time"] = time.time() - start_time
        wbdy_dir = self.output_parameters.get("lakeout_output", None)
        if isinstance(wbdy_dir, Path):
            foreign = getattr(self, "_preexisting_lakeout", frozenset())
            files = sorted(
                (f for f in Path(wbdy_dir).glob("troute_lakeout_*.nc")
                 if f.name not in foreign),
                key=lambda f: f.stem
            )
            if len(files) > 1:
                start_time = time.time()
                _merge_into_first(files, lambda dest: _write_merged_netcdf(files, dest))
                self._timings["output_time"] = time.time() - start_time

    def _is_nhf(self):
        return self.supernetwork_parameters["network_type"] == "NHF"

    def _construct_qlats(self, bmi_values: dict[str, NDArray]):
        dt = self.ngen_dt(bmi_values)
        step_time = self._network.t0
        # NHF uses catchment results whilst the other fabrics use accumulated nexus flows
        if self._is_nhf():
            water_source_ids = bmi_values[BmiVars.CATCHMENT_ID]
            water_source_values = bmi_values[BmiVars.CATCHMENT_VALUE]
        else:
            water_source_ids = bmi_values[BmiVars.NEXUS_ID]
            water_source_values = bmi_values[BmiVars.NEXUS_VALUE]
        LOG.debug(f"Qlat data constructed from {len(water_source_values)} values between {water_source_values.min()} and {water_source_values.max()}")
        num_ids = len(water_source_ids)
        # build the dataframe data
        # the flow rate data should be organized as one large array broken into chunks per timestep with sources aligned with the IDs
        df_data = {}
        index = 0
        while index < len(water_source_values):
            next_index = index + num_ids
            timeslice = water_source_values[index:next_index]
            timestamp = step_time.strftime("%Y%m%d%H%M")
            df_data[timestamp] = timeslice
            step_time += timedelta(seconds=dt)
            index = next_index
        ## use a DataFrame to view the inputs grouped by timestep
        qlats = pd.DataFrame(data=df_data, index=water_source_ids)
        if self._is_nhf():
            self._network._build_qlateral_array_direct(qlats)
            return self._network._qlateral
        else:
            # Take flowpath ids entering NEXUS and replace NEXUS ids by the upstream flowpath ids
            qlats = qlats.rename(index=self._network.downstream_flowpath_dict)
            # create zero values for missing values
            missing = self._network.segment_index[~self._network.segment_index.isin(qlats.index)]
            zeros = pd.DataFrame(data=0.0, index=missing, columns=qlats.columns)
            return pd.concat([qlats, zeros]).sort_index()
