import configparser
import os
from pathlib import Path

from icefabric_manage import build
from pyiceberg.catalog import load_catalog


class Settings:
    """The global settings object for Troute-rnr.

    Notes
    -----
    This will assume you are running the code in the troute-rnr/ dir
    """

    def __init__(self):
        module_dir = Path.cwd()
        project_root = module_dir.parents[1]

        config_file = module_dir / "settings.ini"
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.headers = {
            "Accept": self.config["HEADERS"]["accept"],
            "User-Agent": self.config["HEADERS"]["user_agent"],
        }

        self.unsupported_wfo = [
            "AFC",  # Anchorage, AK
            "AFG",  # Fairbanks, AK
            "AJK",  # Juneau, AK
            "GUM",  # Guam
            "HFO",  # Hawaii
            "SJU",  # Puerto Rico
        ]

        self.STAGES = set(self.config["STAGES"]["stages"].split(","))

        self.BASE_URL = self.config["DEFAULT"]["BASE_URL"]
        self.rate_limit = self.config.getint("DEFAULT", "rate_limit")
        self.reach_limit = self.config.getint("DEFAULT", "reach_limit")

        self.rabbitmq_username = self.config["RABBITMQ"]["username"]
        self.rabbitmq_password = self.config["RABBITMQ"]["password"]
        self.rabbitmq_host = self.config["RABBITMQ"]["host"]
        self.rabbitmq_port = self.config.getint("RABBITMQ", "port")

        self.flooded_data_queue = self.config["QUEUES"]["flooded_data"]
        self.error_queue = self.config["QUEUES"]["error"]

        self.log_path = self.config["PATHS"]["log_path"]
        self.user = self.config["DEFAULT"]["user"]

        self.data_dir = (project_root / "data").resolve()
        self.catalog_settings = {
            "type": "sql",
            "uri": f"sqlite:///{str(self.data_dir / 'warehouse/pyiceberg_catalog.db')}",  # Note the three slashes for absolute path
            "warehouse": f"file://{str(self.data_dir.resolve())}/warehouse",  # Use resolved absolute path
        }

        self.catalog = load_catalog("hydrofabric", **self.catalog_settings)
        build(self.catalog, Path(f"{self.data_dir.resolve()}/parquet"))

        self.base_config_path = module_dir / "base_files/base_config.yaml"
        self.tmp_config = module_dir / "base_files/tmp_config.yaml"
        self.tmp_geopackage = module_dir / "base_files/tmp_domain.gpkg"
        self.tmp_flow_files_path = module_dir / "base_files/tmp_flow/"
        self.tmp_flow_files_path.mkdir(exist_ok=True)
        self.restart_path = module_dir / "base_files/tmp_restart_flow/"
        self.restart_path.mkdir(exist_ok=True)

        self.output_files_path = project_root / "data/output/"
        self.output_files_path.mkdir(exist_ok=True)

        if os.getenv("RABBITMQ_HOST"):
            self.rabbitmq_host = os.getenv("RABBITMQ_HOST")
        if os.getenv("RABBITMQ_USERNAME"):
            self.rabbitmq_username = os.getenv("RABBITMQ_USERNAME")
        if os.getenv("RABBITMQ_PASSWORD"):
            self.rabbitmq_password = os.getenv("RABBITMQ_PASSWORD")

        self.pika_url = f"amqp://{self.rabbitmq_username}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}/"

        if os.getenv("PIKA_URL"):
            self.pika_url = os.getenv("PIKA_URL")
