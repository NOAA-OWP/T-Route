import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

@dataclass
class DataAssimilationParameters:

    reservoir_persistence_usgs: bool = False
    reservoir_persistence_usace: bool = False
    reservoir_persistence_usbr: bool = False
    reservoir_persistence_greatLake: bool = False
    reservoir_rfc_forecasts: bool = False
    diffusive_streamflow_nudging: bool = False
    streamflow_nudging: bool = False
    timeslice_lookback_hours: Optional[int] = None
    usgs_timeslices_folder: Optional[str] = None
    usace_timeslices_folder: Optional[str] = None
    usbr_timeslices_folder: Optional[str] = None
    canada_timeslices_folder: Optional[str] = None
    LakeOntario_outflow: Optional[str] = None
    reservoir_rfc_forecasts_time_series_path: Optional[str] = None
    reservoir_rfc_forecasts_lookback_hours: Optional[int] = None
    reservoir_rfc_forecasts_offset_hours: Optional[int] = None
    reservoir_rfc_forecast_persist_days: Optional[int] = None
    diversion_gage_crosswalk: Optional[dict] = None
    persist_historical_median: bool = False

    def to_dict(self) -> dict:
        da: dict = {
            "streamflow_da": {
                "streamflow_nudging": self.streamflow_nudging,
                "diffusive_streamflow_nudging": self.diffusive_streamflow_nudging,
            },
            "reservoir_da": {
                "reservoir_persistence_da": {
                    "reservoir_persistence_usgs": self.reservoir_persistence_usgs,
                    "reservoir_persistence_usace": self.reservoir_persistence_usace,
                    "reservoir_persistence_usbr": self.reservoir_persistence_usbr,
                    "reservoir_persistence_greatLake": self.reservoir_persistence_greatLake,
                },
                "reservoir_rfc_da": {
                    "reservoir_rfc_forecasts": self.reservoir_rfc_forecasts,
                },
            },
        }
        if self.timeslice_lookback_hours is not None:
            da["timeslice_lookback_hours"] = self.timeslice_lookback_hours
        if self.usgs_timeslices_folder is not None:
            da["usgs_timeslices_folder"] = self.usgs_timeslices_folder
        if self.usace_timeslices_folder is not None:
            da["usace_timeslices_folder"] = self.usace_timeslices_folder
        if self.usbr_timeslices_folder is not None:
            da["usbr_timeslices_folder"] = self.usbr_timeslices_folder
        if self.canada_timeslices_folder is not None:
            da["canada_timeslices_folder"] = self.canada_timeslices_folder
        if self.LakeOntario_outflow is not None:
            da["LakeOntario_outflow"] = self.LakeOntario_outflow
        if self.diversion_gage_crosswalk is not None or self.persist_historical_median:
            da["diversion_da"] = {
                "persist_historical_median": self.persist_historical_median,
            }
            if self.diversion_gage_crosswalk is not None:
                da["diversion_da"]["diversion_gage_crosswalk"] = self.diversion_gage_crosswalk
        rfc_da = da["reservoir_da"]["reservoir_rfc_da"]
        if self.reservoir_rfc_forecasts_time_series_path is not None:
            rfc_da["reservoir_rfc_forecasts_time_series_path"] = self.reservoir_rfc_forecasts_time_series_path
        if self.reservoir_rfc_forecasts_lookback_hours is not None:
            rfc_da["reservoir_rfc_forecasts_lookback_hours"] = self.reservoir_rfc_forecasts_lookback_hours
        if self.reservoir_rfc_forecasts_offset_hours is not None:
            rfc_da["reservoir_rfc_forecasts_offset_hours"] = self.reservoir_rfc_forecasts_offset_hours
        if self.reservoir_rfc_forecast_persist_days is not None:
            rfc_da["reservoir_rfc_forecast_persist_days"] = self.reservoir_rfc_forecast_persist_days
        return da

@dataclass
class Config:
    """Configuration for a t-route simulation run."""

    root_dir: Path
    start_time: str
    end_time: str

    config_file_name: str = "config.yaml"
    forcing_dir_name: str = "channel_forcing"
    output_dir_name: str = "output"
    domain_dir_name: str = "domain"
    domain_file_name: str = "nhf.gpkg"
    dt: int = 300
    qlat_file_pattern: str = "*.CHRTOUT_DOMAIN1.csv"
    data_assimilation_parameters: DataAssimilationParameters = field(default_factory=DataAssimilationParameters)
    max_loop_size: int = 288
    lakeout_output: Optional[str] = None
    restart_dir_name: Optional[str] = None

    def __post_init__(self):
        """Create the expected directory structure."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.channel_forcing_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.domain_dir.mkdir(parents=True, exist_ok=True)
        if self.lakeout_output:
            self.lakeout_dir.mkdir(parents=True, exist_ok=True)
        if self.restart_dir_name:
            (self.root_dir / self.restart_dir_name).mkdir(parents=True, exist_ok=True)
        if self.usgs_timeslices_dir:
            self.usgs_timeslices_dir.mkdir(parents=True, exist_ok=True)
        if self.usace_timeslices_dir:
            self.usace_timeslices_dir.mkdir(parents=True, exist_ok=True)
        if self.usbr_timeslices_dir:
            self.usbr_timeslices_dir.mkdir(parents=True, exist_ok=True)
        if self.rfc_timeslices_dir:
            self.rfc_timeslices_dir.mkdir(parents=True, exist_ok=True)

    @property
    def nts(self) -> int:
        """Number of timesteps derived from start_time, end_time, and dt."""
        start_dt = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M")
        sim_time = (end_dt - start_dt).total_seconds()
        return int(sim_time / self.dt)

    @property
    def config_path(self) -> Path:
        """Absolute path to the output YAML config file."""
        return self.root_dir / self.config_file_name

    @property
    def channel_forcing_dir(self) -> Path:
        """Absolute path to the channel forcing input directory."""
        return self.root_dir / self.forcing_dir_name

    @property
    def output_dir(self) -> Path:
        """Absolute path to the simulation output directory."""
        return self.root_dir / self.output_dir_name

    @property
    def domain_dir(self) -> Path:
        """Absolute path to the domain directory."""
        return self.root_dir / self.domain_dir_name

    @property
    def domain_path(self) -> Path:
        """Absolute path to the hydrofabric GeoPackage file."""
        return self.domain_dir / self.domain_file_name

    @property
    def usgs_timeslices_dir(self) -> Optional[Path]:
        """Absolute path to the USGS timeslices DA folder."""
        da = self.data_assimilation_parameters
        if da.usgs_timeslices_folder is None:
            return None
        return self.root_dir / da.usgs_timeslices_folder

    @property
    def usace_timeslices_dir(self) -> Optional[Path]:
        """Absolute path to the USACE timeslices DA folder."""
        da = self.data_assimilation_parameters
        if da.usace_timeslices_folder is None:
            return None
        return self.root_dir / da.usace_timeslices_folder

    @property
    def usbr_timeslices_dir(self) -> Optional[Path]:
        """Absolute path to the USBR timeslices DA folder."""
        da = self.data_assimilation_parameters
        if da.usbr_timeslices_folder is None:
            return None
        return self.root_dir / da.usbr_timeslices_folder

    @property
    def rfc_timeslices_dir(self) -> Optional[Path]:
        """Absolute path to the RFC forecast DA folder."""
        da = self.data_assimilation_parameters
        if da.reservoir_rfc_forecasts_time_series_path is None:
            return None
        return self.root_dir / da.reservoir_rfc_forecasts_time_series_path

    @property
    def canada_timeslices_dir(self) -> Optional[Path]:
        """Absolute path to the Canada (WSC) timeslices DA folder."""
        da = self.data_assimilation_parameters
        if da.canada_timeslices_folder is None:
            return None
        return self.root_dir / da.canada_timeslices_folder

    @property
    def lake_ontario_outflow_path(self) -> Optional[Path]:
        """Absolute path to the Lake Ontario outflow CSV file."""
        da = self.data_assimilation_parameters
        if da.LakeOntario_outflow is None:
            return None
        return self.root_dir / da.LakeOntario_outflow
    
    @property
    def lakeout_dir(self) -> Optional[Path]:
        """Absolute path to the files with detailed lake outputs."""
        if self.lakeout_output is None:
            return None
        return self.root_dir / self.lakeout_output

    @property
    def reference_data_path(self) -> Path:
        """Absolute path to the gage reference data NetCDF file.

        Matches the file name and subdirectory convention used by RunContext
        in generate_diagnostics.
        """
        return self.root_dir / Path(self.config_file_name).stem / "gage_reference_data.nc"

    def write_yaml(self) -> None:
        """Write the t-route YAML configuration file to config_path."""
        # TODO: some day this could be a set of nested dataclasses
        config = {
            "log_parameters": {
                "showtiming": True,
                "log_level": "DEBUG",
            },
            "network_topology_parameters": {
                "supernetwork_parameters": {
                    "geo_file_path": str(self.domain_path.relative_to(self.root_dir)),
                    "network_type": "NHF",
                },
                "waterbody_parameters": {
                    "break_network_at_waterbodies": True,
                },
            },
            "compute_parameters": {
                "parallel_compute_method": "by-subnetwork-jit-clustered",
                "compute_kernel": "V02-structured",
                "assume_short_ts": True,
                "subnetwork_target_size": 10000,
                "cpu_pool": 1,
                "restart_parameters": {
                    "start_datetime": self.start_time,
                },
                "forcing_parameters": {
                    "dt": 300,
                    "qlat_input_folder": str(self.channel_forcing_dir.relative_to(
                        self.root_dir)
                    ),
                    "qlat_file_pattern_filter": self.qlat_file_pattern,
                    "nts": self.nts,
                    "max_loop_size": self.max_loop_size,
                },
                "data_assimilation_parameters": self.data_assimilation_parameters.to_dict(),
            },
            "output_parameters": {
                "stream_output": {
                    "stream_output_directory": str(self.output_dir.relative_to(
                        self.root_dir)
                    ),
                    "stream_output_time": -1,
                    "stream_output_type": ".nc",
                    "stream_output_internal_frequency": 60,
                },
            },
        }

        if self.lakeout_output:
            config["output_parameters"]["lakeout_output"] = self.lakeout_output

        if self.restart_dir_name:
            config["compute_parameters"]["restart_parameters"]["lite_channel_restart_file"] = str(Path(self.restart_dir_name) / "restart.pkl")

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)


def main():
    """Parse CLI arguments and write a t-route config YAML."""
    parser = argparse.ArgumentParser(description="Generate a t-route config YAML.")
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    args = parser.parse_args()

    cfg = Config(
        root_dir=args.root_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        dt=args.dt,
    )
    cfg.write_yaml()


if __name__ == "__main__":
    main()
