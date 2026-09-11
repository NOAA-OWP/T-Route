"""The entrypoint to RnR"""

import argparse
import json
import logging
import os
import shutil
import socket
import ssl
import sys
from urllib.parse import urlparse

import geopandas as gpd
import httpx
import pika
import polars as pl
from nwm_routing.__main__ import main_v04 as t_route
from pydantic import ValidationError
from troute_rnr import format, read
from troute_rnr.gpkg import get_rnr_segment
from troute_rnr.logging import log_function_debug
from troute_rnr.settings import Settings
from troute_rnr.utils import get

log = logging.getLogger(__name__)

GLOBAL_LOG_LEVEL = logging.INFO


@log_function_debug()
def reset_logging():
    """T-Route sets the logging level to INFO. This resets to WARNING"""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

    global GLOBAL_LOG_LEVEL
    logging.getLogger().setLevel(GLOBAL_LOG_LEVEL)


class MessageCounter:
    """Helper class to track processed messages"""

    def __init__(self, max_messages: int | None):
        self.count = 0
        self.max_messages = max_messages
        self.channel = None

    def increment(self):
        """A counter method"""
        self.count += 1
        log.info(f"Processed message {self.count}/{self.max_messages}")
        if self.count >= self.max_messages:
            log.info(f"Reached maximum messages ({self.max_messages}). Stopping consumer...")
            if self.channel:
                self.channel.stop_consuming()


@log_function_debug()
def run(
    ch: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
    hml_message_counter: MessageCounter | None,
    settings: Settings,
    layers: dict[str, pl.LazyFrame],
) -> None:
    """The main function for the T-Route replace and route module

    Parameters
    ----------
    ch: pika.channel.Channel
        The Queue we're reading from
    method: pika.spec.Basic.Deliver,
        The delivery object for message acknowledgement
    properties: pika.spec.BasicProperties,
        Pika properties
    body: bytes
        The message content body
    hml_message_counter: MessageCounter | None
        the number of forecasts to run for
    settings: Settings
        the settings of RnR
    """
    hml = json.loads(body.decode())
    log.info(f"Reading forecast for {hml['rdf']}, issued at {hml['issuance_time']}")
    sites_response = get(hml["rdf"], headers=settings.headers).json()
    try:
        sites = format.format_xml(sites_response["productText"])
        for site in sites:
            try:
                site_data = read.read_site_data(site, settings)
                if site_data is None:
                    continue
            except ValidationError:
                #  ValidationError: Pydantic validation error for the ingested forecast
                continue
            except httpx.HTTPStatusError:
                #  HTTPStatusError: There was no forecast/record within NWPS for the site given
                continue
            inputs = read.read_rfc_flows(site_data, settings)
            if inputs is not None:
                try:
                    rnr_layers = get_rnr_segment(layers, inputs.reach.id)
                    for table, layer in rnr_layers.items():
                        gpd.GeoDataFrame(layer).to_file(
                            settings.tmp_geopackage, layer=table, driver="GPKG"
                        )  # Writes the rnr geopackage to disk
                    yaml_file_path, tmp_flow_files_path = format.format_config(inputs, settings, layers)
                except IndexError:
                    log.error(
                        "Cannot find river segments downstream of the RFC point. RnR not available; skipping"
                    )
                    continue

                log.info("Configs are built. Running T-Route")
                try:
                    t_route(["-f", str(yaml_file_path)])
                    format.format_output_nc(site_data, inputs, yaml_file_path, settings)
                except IndexError:
                    log.error(f"T-Route inflow formatting error for {inputs.lid}. Skipping Routing")
                except TypeError:
                    log.error("Error with YAML file when running t-route")
                except Exception as e:  # noqa: BLE001
                    # Catching all T-route exceptions in this line
                    log.error(f"T-route failed: {e}")
                log.info("Closing tmp files")
                yaml_file_path.unlink()
                settings.tmp_geopackage.unlink()
                shutil.rmtree(tmp_flow_files_path)
                shutil.rmtree(settings.restart_path / inputs.lid)
                reset_logging()

    except KeyError:
        log.error(f"Sites not found. Status: {sites_response}")
        pass
    finally:
        # Acknowledging message since all HML files are read
        ch.basic_ack(delivery_tag=method.delivery_tag)

        if hml_message_counter.max_messages is not None:
            hml_message_counter.increment()


@log_function_debug()
def consume(
    settings: Settings,
    layers: dict[str, pl.LazyFrame],
    hml_message_counter: MessageCounter | None = None,
    is_iac: bool = False,
) -> None:
    """
    The message consumer interfacing with RabbitMQ

    Parameters
    ----------
    settings : Settings
        Configuration object containing RabbitMQ connection parameters and flooding information
    hml_message_counter: MessageCounter | None
        The number of forecasts to process (usually a debug setting or for demonstrations)

    Raises
    ------
    pika.exceptions.AMQPConnectionError
        If connection to RabbitMQ server fails
    socket.timeout
        If connection attempt times out
    """
    try:
        if is_iac:
            import boto3

            secret_arn = os.getenv("RABBITMQ_SECRET_ARN")
            region = os.getenv("AWS_REGION", "us-east-1")
            rabbit_mq_endpoint = os.getenv("RABBITMQ_ENDPOINT")

            if secret_arn is None:
                raise ValueError("Cannot find RABBITMQ_SECRET_ARN")
            if rabbit_mq_endpoint is None:
                raise ValueError("Cannot find RABBITMQ_ENDPOINT")

            client = boto3.client("secretsmanager", region_name=region)
            secret_value = client.get_secret_value(SecretId=secret_arn)
            secret = json.loads(secret_value["SecretString"])
            user = secret["username"]
            pwd = secret["password"]
            creds = pika.PlainCredentials(user, pwd)
            url = urlparse(rabbit_mq_endpoint)
            context = ssl.create_default_context()
            vhost = url.path.strip("/") if url.path.strip("/") else "/"
            conn = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=url.hostname,
                    port=url.port,
                    virtual_host=vhost,
                    credentials=creds,
                    ssl_options=pika.SSLOptions(context),
                    heartbeat=30,
                    blocked_connection_timeout=300,
                )
            )
            channel = conn.channel()
        else:
            # Set connection parameters with timeout
            connection_params = pika.URLParameters(settings.pika_url)
            connection_params.socket_timeout = 10  # 10-second timeout for connection

            # Attempt connection with timeout
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
        channel.queue_declare(queue=settings.flooded_data_queue, durable=True)
        log.info(f" [*] Waiting for messages from Queue on URL: {settings.pika_url}.")
        log.info(" [*] Be sure HML files are populated in the Message Queue")
        channel.basic_qos(prefetch_count=1)
        hml_message_counter.channel = channel

        channel.basic_consume(
            queue=settings.flooded_data_queue,
            on_message_callback=lambda ch, method, properties, body: run(
                ch, method, properties, body, hml_message_counter, settings, layers
            ),
        )
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            log.error("\n [*] Stopping consumer due to keyboard interrupt...")
            channel.stop_consuming()
        log.info(f" [*] Processed {hml_message_counter.max_messages} forecasts. Closing connection.")
        connection.close()
    except (pika.exceptions.AMQPConnectionError, socket.timeout) as e:
        log.error(f" [!] Failed to connect to RabbitMQ: {e}")
        log.error(" [!] Service is not running - check RabbitMQ connection from Hydrovis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-route RnR function")
    parser.add_argument(
        "--num-hml-files", type=int, help="The number of hml files to be read from the message queue"
    )
    parser.add_argument("--iac", action="store_true", help="If true this code is to be run as IaC")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    GLOBAL_LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(GLOBAL_LOG_LEVEL)

    logging.basicConfig(
        level=GLOBAL_LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    log = logging.getLogger(__name__)

    if args.num_hml_files is None:
        log.info(" [*] Running T-Route for all messages in the queue")
    else:
        log.info(f" [*] Running T-Route for {args.num_hml_files} forecasts")
    hml_message_counter = MessageCounter(args.num_hml_files)
    settings = Settings()
    if args.iac:
        bucket_name = os.getenv("APP_BUCKET_NAME")
        troute_output_path = os.getenv("APP_OUTPUT_S3_KEY")
        if not troute_output_path:
            raise FileNotFoundError("APP_OUTPUT_S3_KEY environment variable not set")

        hydrofabric_path = os.getenv("HYDROFABRIC_S3_KEY")
        if not hydrofabric_path:
            raise FileNotFoundError("HYDROFABRIC_S3_KEY environment variable not set")
        layers = {
            "network": pl.scan_parquet(f"s3://{bucket_name}/{hydrofabric_path}/network.parquet"),
            "flowpaths": pl.scan_parquet(f"s3://{bucket_name}/{hydrofabric_path}/flowpaths.parquet"),
            "lakes": pl.scan_parquet(f"s3://{bucket_name}/{hydrofabric_path}/lakes.parquet"),
            "hydrolocations": pl.scan_parquet(
                f"s3://{bucket_name}/{hydrofabric_path}/hydrolocations.parquet"
            ),
            "divides": pl.scan_parquet(f"s3://{bucket_name}/{hydrofabric_path}/divides.parquet"),
            "nexus": pl.scan_parquet(f"s3://{bucket_name}/{hydrofabric_path}/nexus.parquet"),
            "flowpath_attr": pl.scan_parquet(
                f"s3://{bucket_name}/{hydrofabric_path}/flowpath-attributes.parquet"
            ),
            "pois": pl.scan_parquet(f"s3://{bucket_name}/{hydrofabric_path}/pois.parquet"),
        }
        settings.bucket_name = bucket_name
        settings.troute_output_path = troute_output_path
    else:
        layers = {
            "network": pl.scan_parquet(settings.data_dir / "parquet/network.parquet"),
            "flowpaths": pl.scan_parquet(settings.data_dir / "parquet/flowpaths.parquet"),
            "lakes": pl.scan_parquet(settings.data_dir / "parquet/lakes.parquet"),
            "hydrolocations": pl.scan_parquet(settings.data_dir / "parquet/hydrolocations.parquet"),
            "divides": pl.scan_parquet(settings.data_dir / "parquet/divides.parquet"),
            "nexus": pl.scan_parquet(settings.data_dir / "parquet/nexus.parquet"),
            "flowpath_attr": pl.scan_parquet(settings.data_dir / "parquet/flowpath-attributes.parquet"),
            "pois": pl.scan_parquet(settings.data_dir / "parquet/pois.parquet"),
        }
        settings.bucket_name = None
        settings.troute_output_path = None
    consume(settings, layers, hml_message_counter=hml_message_counter, is_iac=args.iac)
