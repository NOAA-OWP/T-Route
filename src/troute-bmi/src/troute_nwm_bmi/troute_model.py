"""Basic Model Interface backing model for NGEN t-route."""
from __future__ import annotations
import math
import psutil
import time
import typing
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import timedelta, datetime
from troute.config import Config

from troute.NHDNetwork import NHDNetwork
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from troute.NHF import NHF
from troute.DataAssimilation import DataAssimilation

import troute.hyfeature_network_utilities as hnu

import nwm_routing.nwm_route as nwm_routing
from nwm_routing.output import nwm_output_generator, remap_outputs

import logging
LOG = logging.getLogger("TROUTE")

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


class BmiVars:
    CATCHMENT_ID = "catchment_water_source__id"
    CATCHMENT_VALUE = "catchment_water_source__volume_flow_rate"
    NEXUS_ID = "land_surface_water_source__id"
    NEXUS_VALUE = "land_surface_water_source__volume_flow_rate"
    NGEN_DT = "ngen_dt"
    UPSTREAM_ID = "upstream_id"

    CHANNEL_WATER_ID = "channel_water__id"
    CHANNEL_WATER_RATE = "channel_exit_water_x-section__volume_flow_rate"
    CHANNEL_WATER_SPEED = "channel_water_flow__speed"
    CHANNEL_WATER_DEPTH = "channel_water__mean_depth"

    LAKE_WATER_ID = "lake_water__id"
    LAKE_WATER_INCOMING = "lake_water~incoming__volume_flow_rate"
    LAKE_WATER_OUTGOING = "lake_water~outgoing__volume_flow_rate"
    LAKE_WATER_ELEVATION = "lake_surface__elevation"


class Model:
    dt: int

    def __init__(self, config_file: str, start_time: float):
        self._time = start_time

        with open(config_file) as reader:
            data = yaml.load(reader, Loader=yaml.SafeLoader)
        self._config: dict = Config.with_strict_mode(**data).dict()

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
        network_creation_time = time.time() - network_start_time

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
            run_parameters={},
            waterbody_parameters=self.waterbody_parameters,
            from_files=True,
            value_dict=None,
            da_run=da_run,
        )
        forcing_time = time.time() - forcing_start_time

        # Pass empty subnetwork list to nwm_route. These objects will be calculated/populated
        # on first iteration of for loop only. For additional loops this will be passed
        # to function from inital loop.
        self._subnetwork = [None, None, None]

        self._timings = {
            "forcing_time": forcing_time,
            "route_time": 0.0,
            "output_time": 0.0,
            "network_creation_time": network_creation_time,
        }

    def run(self, bmi_values: dict[str, NDArray]):
        is_nhf = self._is_nhf()
        qts_subdivisions = self.qts_subdivisions
        output_params = {
            "t0": self.t0,
            "dt": self.dt
        }

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

        LOG.debug("Starting routing function")
        route_start_time = time.time()
        full_results = None
        nts = 0
        for run in self._build_run_sets(qlats):
            nts += run["nts"]
            usgs_df = self._data_assimilation.usgs_df
            if not usgs_df.empty:
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
                usgs_df=self._data_assimilation.usgs_df,
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
                return_courant=self.compute_parameters.get('return_courant', False),
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
            )

            # create initial conditions for next loop iteration
            self._network.new_q0(run_results)
            self._network.update_waterbody_water_elevation()

            # update reservoir parameters and lastobs_df
            self._data_assimilation.update_after_compute(run_results, self.dt * run["nts"])

            if full_results is None:
                full_results = run_results
            else:
                full_results = full_results.append_timesteps(run_results)

        self._timings["route_time"] = time.time() - route_start_time

        LOG.debug("Generating output")
        output_start_time = time.time()
        output_params["nts"] = nts
        nwm_output_generator(
            run=output_params,
            results=full_results,
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

        self._network.new_t0(self.dt, nts)

        # # compute BMI outputs
        # def _update_values(name: str, values: pd.Series | pd.Index):
        #     dtype = bmi_values[name].dtype
        #     array = bmi_values[name] = values.to_numpy(dtype=dtype, copy=True)
        #     return array
        # qvd_columns = pd.MultiIndex.from_product(
        #     [range(nts), ["q", "v", "d", "ql"]]
        # ).to_flat_index()

        # flowveldepth = pd.concat(
        #     [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in full_results],
        #     copy=False,
        # )
        # flowveldepth = flowveldepth.drop(columns=[
        #     col for col in flowveldepth.columns if col[1] == "ql"
        # ])
        # if is_nhf:
        #     flowveldepth = remap_outputs(flowveldepth, self._network.fp_outlet_crosswalk)
        # _update_values(BmiVars.CHANNEL_WATER_RATE, flowveldepth.iloc[:,-3])
        # _update_values(BmiVars.CHANNEL_WATER_SPEED, flowveldepth.iloc[:,-2])
        # _update_values(BmiVars.CHANNEL_WATER_DEPTH, flowveldepth.iloc[:,-1])
        # _update_values(BmiVars.CHANNEL_WATER_ID, flowveldepth.index)

        # i_columns = pd.MultiIndex.from_product(
        #     [range(int(nts)), ["i"]]
        # ).to_flat_index()
        # if is_nhf or sum(len(w) for r in full_results for w in r[6]) == 0:
        #     # Waterbodies are not implemented in NHF yet.
        #     wbdy = pd.DataFrame(columns=i_columns)
        # else:
        #     wbdy = pd.concat(
        #         [pd.DataFrame(r[6], index=r[0], columns=i_columns) for r in full_results],
        #         copy=False,
        #     )

        # wbdy_id = _update_values(BmiVars.LAKE_WATER_ID, self._network.waterbody_dataframe.index)
        # _update_values(BmiVars.LAKE_WATER_INCOMING, wbdy.loc[wbdy_id].iloc[:,-1])
        # _update_values(BmiVars.LAKE_WATER_OUTGOING, flowveldepth.loc[wbdy_id].iloc[:,-3])
        # _update_values(BmiVars.LAKE_WATER_ELEVATION, flowveldepth.loc[wbdy_id].iloc[:,-1])

        self._timings["output_time"] = time.time() - output_start_time

        # update time as (ngen dt in seconds) * (number of steps processed)
        self._time += self.ngen_dt(bmi_values) * qlats.shape[1]

    def log_times(self):
        if self.show_timing:
            self._log_times()

    def create_state(self):
        """Create a dictionary of data that can be serialized using `pickle.dumps`."""
        # save current subnetwork and convert defaultdicts to dicts
        subnetwork = list(self._subnetwork)
        for i, value in enumerate(subnetwork):
            if isinstance(value, dict):
                subnetwork[i] = dict(value)
        return {
            "time": self._time,
            "subnetwork": subnetwork,
            # updated data stored on AbstractNetwork
            "q0": self._network._q0,
            "t0": self._network._t0,
            # updated data stored on DataAssimilation
            "last_obs": self._data_assimilation._last_obs_df,
            "usgs": self._data_assimilation._reservoir_usgs_param_df,
            "usace": self._data_assimilation._reservoir_usace_param_df,
            "rfc": self._data_assimilation._reservoir_rfc_param_df,
            "gl": self._data_assimilation._great_lakes_param_df,
        }

    def load_state(self, data: dict):
        self._time = data["time"]
        self._subnetwork = data["subnetwork"]
        self._network._q0 = data["q0"]
        self._network._t0 = data["t0"]
        self._data_assimilation._last_obs_df = data["last_obs"]
        self._data_assimilation._reservoir_usgs_param_df = data["usgs"]
        self._data_assimilation._reservoir_usace_param_df = data["usace"]
        self._data_assimilation._reservoir_rfc_param_df = data["rfc"]
        self._data_assimilation._great_lakes_param_df = data["gl"]
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

    def _build_run_sets(self, qlats: pd.DataFrame):
        nts = len(qlats.columns)
        # estimate required memory
        required_bytes = qlats.shape[0] \
            * qlats.shape[1] \
            * self.qts_subdivisions \
            * 200  # 200 based size of large arrays made during compute plus some padding
        # determing loop size based on system available memory
        system_memory = psutil.virtual_memory()
        available_memory = system_memory.available * 0.9  # only account for 90% of the currently available memory
        divisions = math.ceil(required_bytes / available_memory)
        if divisions <= 1:
            yield {
                "nts": nts * self.qts_subdivisions,
                "qlats": qlats,
                "t0": self.t0,
                "final_timestamp": self.t0 + timedelta(seconds=self.nts * self.dt)
            }
        else:
            LOG.info(f"T-Route detected a high probability of exceeding system memory. The processing will be broken into {divisions} chunks.")
            loop_size = math.ceil(nts / divisions)
            # construct run sets based on the loop size determined by available rmemory
            step = 0
            while step < nts:
                next_step = step + loop_size
                times = qlats.columns[step:next_step]
                yield {
                    "nts": len(times) * self.qts_subdivisions,
                    "qlats": qlats[times],
                    "t0": datetime.strptime(times[0], "%Y%m%d%H%M%S"),
                    "final_timestamp": datetime.strptime(times[-1], "%Y%m%d%H%M%S")
                }
                step = next_step

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

    def _log_times(self):
        def sec_and_per(title, key: str):
            seconds = round(self._timings[key], 2)
            percent = round(self._timings[key] / process_time * 100, 2)
            LOG.info(f"{title}: {seconds} secs, {percent} %")
        process_time = sum(self._timings.values())
        LOG.debug(f"Processes complete in {process_time} seconds.")
        LOG.info('************ TIMING SUMMARY ************')
        LOG.info('----------------------------------------')
        sec_and_per("Network graph construction", 'network_creation_time')
        sec_and_per("Forcing array construction", "forcing_time")
        sec_and_per("Routing computations", "route_time")
        sec_and_per("Output writing", "output_time")
        LOG.info(f"Total execution time: {round(process_time, 2)} secs")
