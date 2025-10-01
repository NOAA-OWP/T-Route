import argparse
import json
import logging
import shutil
import socket

import httpx
import pika
from icefabric_tools import rnr
from nwm_routing.__main__ import main_v04 as t_route
from pydantic import ValidationError
from troute_rnr import format, read
from troute_rnr.settings import Settings
from troute_rnr.utils import get

settings = Settings()


def reset_logging():
    """T-Route sets the logging level to INFO. This resets to WARNING"""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)


class MessageCounter:
    """Helper class to track processed messages"""

    def __init__(self, max_messages: int | None):
        self.count = 0
        self.max_messages = max_messages
        self.channel = None

    def increment(self):
        """A counter method"""
        self.count += 1
        print(f"Processed message {self.count}/{self.max_messages}")
        if self.count >= self.max_messages:
            print(f"Reached maximum messages ({self.max_messages}). Stopping consumer...")
            if self.channel:
                self.channel.stop_consuming()


def run(
    ch: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
    hml_message_counter: MessageCounter | None,
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
    """
    hml = json.loads(body.decode())
    print(f"Reading forecast for {hml['rdf']}, issued at {hml['issuance_time']}")
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
                    rnr.get_rnr_segment(
                        settings.catalog, inputs.reach.id, settings.tmp_geopackage
                    )  # Writes the rnr geopackage to disk
                    yaml_file_path, tmp_flow_files_path = format.format_config(inputs, settings)
                except IndexError:
                    print(
                        "Cannot find river segments downstream of the RFC point. RnR not available; skipping"
                    )
                    continue

                print("Configs are built. Running T-Route")
                try:
                    t_route(["-f", str(yaml_file_path)])
                    format.format_output_nc(site_data, inputs, yaml_file_path)
                except IndexError:
                    print(f"T-Route inflow formatting error for {inputs.lid}. Skipping Routing")
                print("Closing tmp files")
                yaml_file_path.unlink()
                settings.tmp_geopackage.unlink()
                shutil.rmtree(tmp_flow_files_path)
                shutil.rmtree(settings.restart_path / inputs.lid)
                reset_logging()

    except KeyError:
        print(f"Sites not found. Status: {sites_response['status']}")
        pass
    finally:
        # Acknowledging message since all HML files are read
        ch.basic_ack(delivery_tag=method.delivery_tag)

        if hml_message_counter.max_messages is not None:
            hml_message_counter.increment()


def consume(settings: Settings, hml_message_counter: MessageCounter | None = None) -> None:
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
        # Set connection parameters with timeout
        connection_params = pika.URLParameters(settings.pika_url)
        connection_params.socket_timeout = 10  # 10-second timeout for connection

        # Attempt connection with timeout
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=settings.flooded_data_queue, durable=True)
        print(f" [*] Waiting for messages from Queue on URL: {settings.pika_url}.")
        print(" [*] Be sure HML files are populated in the Message Queue")
        channel.basic_qos(prefetch_count=1)
        hml_message_counter.channel = channel

        channel.basic_consume(
            queue=settings.flooded_data_queue,
            on_message_callback=lambda ch, method, properties, body: run(
                ch, method, properties, body, hml_message_counter
            ),
        )
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            print("\n [*] Stopping consumer due to keyboard interrupt...")
            channel.stop_consuming()
        print(f" [*] Processed {hml_message_counter.max_messages} forecasts. Closing connection.")
        connection.close()
    except (pika.exceptions.AMQPConnectionError, socket.timeout) as e:
        print(f" [!] Failed to connect to RabbitMQ: {e}")
        print(" [!] Service is not running - check RabbitMQ connection from Hydrovis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-route RnR function")
    parser.add_argument(
        "--num-hml-files", type=int, help="The number of hml files to be read from the message queue"
    )
    args = parser.parse_args()
    if args.num_hml_files is None:
        print(" [*] Running T-Route for all messages in the queue")
    else:
        print(f" [*] Running T-Route for {args.num_hml_files} forecasts")
    hml_message_counter = MessageCounter(args.num_hml_files)
    consume(Settings(), hml_message_counter=hml_message_counter)
