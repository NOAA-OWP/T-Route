"""Basic Model Interface backing model for NGEN t-route."""
from __future__ import annotations
import math
from tempfile import NamedTemporaryFile
import psutil
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

    def run(self, bmi_values: dict[str, NDArray]):
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

                # da_run is built per WINDOW: t0 advances every update_until call,
                # and the initialize-time list pinned to the original t0 stopped
                # covering later windows (silent no-obs assimilation).
                self._data_assimilation._usgs_df = merge_injected_obs(  # pyright: ignore[reportPrivateUsage]
                    self._scaling_da.build_usgs_df(
                        run["t0"], self.dt, run["nts"], self._window_da_run(run)
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
            # Result-determining state: without it a resumed run retraces from
            # its own first window and shifts corrections.
            "scaling_tau": (
                self._scaling_da.trace_checkpoint()
                if getattr(self, "_scaling_da", None) is not None
                else None
            ),
        }

    def load_state(self, data: dict):
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
        self._data_assimilation._last_obs_df = data["last_obs"]
        self._data_assimilation._reservoir_usgs_param_df = data["usgs"]
        self._data_assimilation._reservoir_usace_param_df = data["usace"]
        # .get for backward compatibility with state files written before USBR
        # persistence state was included.
        if "usbr" in data:
            self._data_assimilation._reservoir_usbr_param_df = data["usbr"]
        self._data_assimilation._reservoir_rfc_param_df = data["rfc"]
        self._data_assimilation._great_lakes_param_df = data["gl"]
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

    def _build_run_sets(self, qlats: pd.DataFrame) -> Iterator[RunSet]:
        nts = len(qlats.columns)
        # Memory is a SAFETY CAP only, never the primary window control.
        required_bytes = qlats.shape[0] \
            * qlats.shape[1] \
            * self.qts_subdivisions \
            * 100  # bytes per link-timestep, measured: a 96 h single-window
        # run over this 103,559-link domain peaks at 7.86 GB whole-process RSS
        # (70.7 B per element); the old 200 blocked runs the machine performs,
        # which is now fatal under active scaling DA (hard error, not shrink).
        system_memory = psutil.virtual_memory()
        available_memory = system_memory.available * 0.9  # only account for 90% of the currently available memory
        mem_divisions = math.ceil(required_bytes / available_memory)
        mem_loop_size = math.ceil(nts / mem_divisions)

        # max_loop_size is the PRIMARY control (as in the -V5 driver): every
        # per-window DA operation lands on this partition, so a RAM-derived split
        # made results depend on machine load. Under active scaling DA the
        # partition is part of the RESULT (decay resets at window boundaries), so
        # a RAM cap below the configured window is a hard error there; without
        # the DA it stays a warning.
        scaling_active = getattr(self, "_scaling_da", None) is not None
        cfg_loop = self.forcing_parameters.get("max_loop_size") or 0
        if scaling_active and cfg_loop > 0:
            # The forward innovation window reads past the end of a window into
            # the next one's innovation, and that halo is exactly ONE window
            # deep, so a window shorter than innovation_spread_h leaves its own
            # tail uncovered and the result starts depending on the partition.
            # max_loop_size counts qlat COLUMNS here, so convert via the column
            # cadence and enlarge. The final window of each update is exempt
            # (edge closure). getattr with the ScalingDA class default: tests
            # stub _scaling_da.
            col_s = float(self.qts_subdivisions) * float(self.dt)
            spread_h = float(getattr(self._scaling_da, "innovation_spread_h", 0.0))
            # The travel-time lag is measured over a fixed span taken from the
            # first window, so that window has to contain it. Otherwise the span
            # follows max_loop_size and a memory knob changes discharge.
            # SUM, not max: the lag reads the SMOOTHED innovation at t + tau, so
            # the tail of a window needs raw innovation out to tau_max + spread.
            if getattr(self._scaling_da, "travel_time_lag", False):
                spread_h += float(getattr(self._scaling_da, "lag_window_h", 48.0))
            need = max(int(cfg_loop), math.ceil(spread_h * 3600.0 / col_s))
            if need > cfg_loop:
                LOG.info(
                    "scaling DA: max_loop_size enlarged %d -> %d forcing columns "
                    "so every non-final window covers innovation_spread_h.",
                    int(cfg_loop), need,
                )
                cfg_loop = need
        if cfg_loop > 0:
            loop_size = min(int(cfg_loop), mem_loop_size)
            if loop_size < int(cfg_loop):
                if scaling_active:
                    # Two very different causes land here and must not be
                    # reported as one. mem_divisions == 1 means memory was never
                    # the limit: the cap IS this update's own forcing, because
                    # mem_loop_size is then just nts. A caller cannot fix that
                    # with a bigger machine, only by feeding longer updates or
                    # asking for a shorter span, so saying "free memory" sends
                    # them to the wrong place entirely.
                    if mem_divisions <= 1:
                        raise ValueError(
                            f"this update supplies {nts} forcing timestep(s), but the "
                            f"scaling DA requires windows of {int(cfg_loop)} -- the "
                            "larger of the configured max_loop_size and the DA's own "
                            "span (innovation_spread_h, plus lag_window_h when "
                            "travel_time_lag is on). The window partition is part of "
                            "the result under active assimilation, so it must not "
                            "shrink silently to the update. Memory is NOT the limit "
                            "here. Either drive longer updates, lower max_loop_size "
                            "to the update cadence, or reduce the DA span "
                            "(lag_window_h / innovation_spread_h); halving "
                            "lag_window_h also halves the longest resolvable travel "
                            "time."
                        )
                    raise MemoryError(
                        f"available memory caps the run window at {loop_size} forcing "
                        f"timesteps, below the configured max_loop_size of "
                        f"{int(cfg_loop)}. The scaling DA's window boundaries are part "
                        "of the result, so continuing would produce discharge that "
                        "depends on current machine load. Free memory, or lower "
                        "max_loop_size to a value that fits."
                    )
                LOG.warning(
                    "available memory caps the run window at %d forcing timesteps, "
                    "below the configured max_loop_size of %d; results with "
                    "assimilation active depend on the window partition",
                    loop_size, int(cfg_loop),
                )
        else:
            loop_size = mem_loop_size
            if mem_divisions > 1:
                if scaling_active:
                    raise MemoryError(
                        f"no forcing_parameters.max_loop_size is configured and the "
                        f"run does not fit in available memory (would be split into "
                        f"{mem_divisions} RAM-sized windows). The scaling DA's window "
                        "boundaries are part of the result, so a RAM-derived "
                        "partition is not reproducible. Set max_loop_size."
                    )
                LOG.warning(
                    "no forcing_parameters.max_loop_size configured; splitting the "
                    "run into %d windows sized by available memory. With "
                    "assimilation active the results depend on this partition, so "
                    "set max_loop_size for reproducible windows.", mem_divisions,
                )

        if loop_size >= nts:
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
            # construct run sets on the resolved window size
            step = 0
            while step < nts:
                next_step = step + loop_size
                times = qlats.columns[step:next_step]
                yield {
                    "nts": len(times) * self.qts_subdivisions,
                    "qlats": qlats[times],
                    # Columns are "%Y%m%d%H%M"; parsing with a trailing %S silently
                    # backtracks to the wrong time ("...1445" -> 14:04:05).
                    "t0": datetime.strptime(times[0], "%Y%m%d%H%M"),
                    "final_timestamp": datetime.strptime(times[-1], "%Y%m%d%H%M")
                }
                step = next_step

    def _merge_run_results(self):
        stream_params = self.output_parameters.get("stream_output")
        if isinstance(stream_params, dict):
            stream_type = stream_params.get("stream_output_type")
            files = sorted(
                Path(stream_params["stream_output_directory"]).glob(
                    "troute_output_*" + stream_type
                ),
                key=lambda f: f.stem
            )
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
            files = sorted(
                Path(wbdy_dir).glob("troute_lakeout_*.nc"),
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
