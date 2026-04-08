from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import click

from src.config import get_settings
from src.db import get_connection_pool
from src.db.repositories import upsert_topic_metadata
from src.monitoring import run_health_checks
from src.services.topic_metadata_loader import load_topic_metadata_from_artifacts

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Entry point for backend services."""


@cli.command()
def consumer() -> None:
    """Run Kafka consumer worker."""
    from src.workers.kafka_consumer import run_consumer

    run_consumer()


@cli.command()
def inference_worker() -> None:
    """Run batch inference worker."""
    from src.workers.batch_inference import run_worker

    run_worker()


@cli.command()
def newsdata_producer() -> None:
    """Run NewsData ingestion producer."""
    from src.ingestion.newsdata_producer import main as run_newsdata

    run_newsdata()


@cli.command("sync-topics")
@click.option(
    "--artifacts-path",
    type=click.Path(path_type=Path),
    help="Override path to artifacts (defaults to Settings.artifacts_path)",
)
def sync_topics(artifacts_path: Path | None) -> None:
    """Seed topic_metadata table using topics.json from artifacts."""
    settings = get_settings()
    path = artifacts_path or settings.artifacts_path
    records = list(load_topic_metadata_from_artifacts(path))
    if not records:
        raise click.ClickException(
            f"No topic metadata found in artifacts directory: {path}"
        )

    pool = get_connection_pool()
    with pool.connection() as conn:
        upsert_topic_metadata(conn, records)
        conn.commit()
    click.echo(f"Inserted/updated {len(records)} topic metadata records")


@cli.command("health")
def health_command() -> None:
    """Run synchronous health checks and print the result."""
    checks = run_health_checks(get_settings())
    click.echo(json.dumps(checks, ensure_ascii=False, indent=2))
    if any(item["status"] != "ok" for item in checks):
        raise SystemExit(1)


def _start_service(name: str, target) -> None:
    def runner() -> None:
        try:
            target()
        except Exception:  # noqa: BLE001
            logger.exception("Service crashed", extra={"service": name})

    thread = threading.Thread(target=runner, name=f"{name}-thread", daemon=True)
    thread.start()


@cli.command("all")
def all_services() -> None:
    """Run consumer, inference worker, and NewsData producer in one process."""
    from src.workers.kafka_consumer import run_consumer
    from src.workers.batch_inference import run_worker
    from src.ingestion.newsdata_producer import main as run_newsdata

    services = [
        ("consumer", run_consumer),
        ("inference_worker", run_worker),
        ("newsdata_producer", run_newsdata),
    ]

    for name, target in services:
        _start_service(name, target)

    click.echo("Started services: consumer, inference_worker, newsdata_producer")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Combined runner interrupted, shutting down")


if __name__ == "__main__":
    print(os.getcwd())
    cli()
