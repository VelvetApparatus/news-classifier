from __future__ import annotations

import json
import logging
from uuid import uuid4

from kafka import KafkaConsumer
from kafka.structs import OffsetAndMetadata

from src.config import get_settings
from src.db import get_connection_pool
from src.db.repositories import insert_incoming_news
from src.utils.logging import setup_logging
from src.workers.utils import build_external_id, parse_published_at

logger = logging.getLogger(__name__)


def process_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    text = payload.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("Payload is missing text field")

    news_id = uuid4()
    external_id = build_external_id(payload)
    title = payload.get("title")
    source = payload.get("source")
    published_at = payload.get("published_at")
    url = payload.get("url")
    published_at_dt = None

    if isinstance(published_at, str):
        published_at_dt = parse_published_at(published_at)

    payload_to_store = dict(payload)
    payload_to_store["id"] = str(news_id)
    payload_to_store["external_id"] = external_id

    return {
        "news_id": news_id,
        "external_id": external_id,
        "title": title,
        "source": source,
        "published_at": published_at_dt,
        "raw_text": text,
        "url": url,
        "payload": payload_to_store,
    }


def run_consumer() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    pool = get_connection_pool()
    consumer = KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers_list,
        group_id=settings.kafka_group_id,
        enable_auto_commit=True,
        value_deserializer=lambda v: v,
        auto_offset_reset=settings.kafka_auto_offset_reset,
    )

    logger.info(
        "Kafka consumer started",
        extra={
            "topic": settings.kafka_topic,
            "bootstrap": settings.kafka_bootstrap_servers,
            "group": settings.kafka_group_id,
        },
    )

    try:
        while True:
            records = consumer.poll(timeout_ms=settings.kafka_consumer_poll_timeout_ms)
            if not records:
                continue

            for tp, messages in records.items():
                for message in messages:
                    try:
                        payload = json.loads(message.value.decode("utf-8"))
                    except Exception:
                        logger.exception(
                            "Failed to decode message as JSON",
                            extra={"partition": tp.partition, "offset": message.offset},
                        )
                        # consumer.commit(
                        #     {tp: OffsetAndMetadata(
                        #
                        #         message.offset + 1, None,
                        #     )}
                        # )
                        continue

                    try:
                        prepared = process_payload(payload)
                    except Exception as exc:
                        logger.error(
                            "Invalid payload, skipping message",
                            extra={
                                "partition": tp.partition,
                                "offset": message.offset,
                                "error": str(exc),
                            },
                        )
                        # consumer.commit(
                        #     {tp: OffsetAndMetadata(message.offset + 1, None)}
                        # )
                        continue

                    try:
                        with pool.connection() as conn:
                            inserted = insert_incoming_news(
                                conn,
                                news_id=prepared["news_id"],
                                external_id=prepared["external_id"],
                                title=prepared["title"],
                                source=prepared["source"],
                                published_at=prepared["published_at"],
                                raw_text=prepared["raw_text"],
                                url=prepared["url"],
                                payload=prepared["payload"],
                            )
                            # conn.commit()
                    except Exception:
                        logger.exception(
                            "Failed to insert incoming news",
                            extra={
                                "partition": tp.partition,
                                "offset": message.offset,
                                "external_id": prepared["external_id"],
                            },
                        )
                        continue

                    # consumer.commit(
                    #     {tp: OffsetAndMetadata(message.offset + 1, None)}
                    # )

                    if inserted:
                        logger.info(
                            "News stored from Kafka message",
                            extra={
                                "partition": tp.partition,
                                "offset": message.offset,
                                "news_id": str(prepared["news_id"]),
                                "external_id": prepared["external_id"],
                            },
                        )
                    else:
                        logger.info(
                            "Duplicate news skipped by external_id",
                            extra={
                                "partition": tp.partition,
                                "offset": message.offset,
                                "external_id": prepared["external_id"],
                            },
                        )
    except KeyboardInterrupt:
        logger.info("Kafka consumer interrupted, shutting down")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_consumer()
