from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import requests
from kafka import KafkaProducer

from src.config import get_settings
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

NEWSDATA_ENDPOINT = "https://newsdata.io/api/1/latest"


class NewsDataProducer:
    def __init__(self, api_key: str, bootstrap_servers: list[str], topic: str):
        if not api_key:
            raise ValueError("NewsData API key must be provided")

        self.api_key = api_key
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            linger_ms=10,
        )

    def fetch(self) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                NEWSDATA_ENDPOINT,
                params={
                    "apikey": self.api_key,
                    "language": "ru",
                    "country": "ru",
                },
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("NewsData API request failed", exc_info=exc)
            return []

        try:
            payload = response.json()
        except ValueError:
            logger.error("Failed to decode NewsData response as JSON")
            return []

        results = payload.get("results") or []
        logger.info("Fetched NewsData batch", extra={"count": len(results)})
        return results

    @staticmethod
    def normalize(article: dict[str, Any]) -> dict[str, Any] | None:
        text = article.get("content") or article.get("description")
        if not text:
            return None

        published_at = article.get("pubDate")
        title = article.get("title")
        source = article.get("source_id") or article.get("source")
        url = article.get("link")

        if url:
            external_id = url
        else:
            digest_source = (title or "") + (published_at or "") + text
            external_id = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()

        return {
            "external_id": external_id,
            "title": title,
            "source": source,
            "published_at": published_at,
            "text": text,
            "url": url,
            "payload": article,
        }

    def send(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self.producer.send(self.topic, value=message)
        self.producer.flush()
        logger.info("Published messages to Kafka", extra={"count": len(messages)})

    def run_once(self) -> None:
        articles = self.fetch()
        if not articles:
            return

        normalized: list[dict[str, Any]] = []
        for article in articles:
            normalized_article = self.normalize(article)
            if normalized_article:
                normalized.append(normalized_article)

        if not normalized:
            logger.info("No articles to send after normalization")
            return

        self.send(normalized)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    producer = NewsDataProducer(
        api_key=settings.newsdata_api_key,
        bootstrap_servers=settings.kafka_bootstrap_servers_list,
        topic=settings.kafka_topic,
    )

    poll_interval = settings.newsdata_poll_interval_seconds
    logger.info(
        "Starting NewsData producer",
        extra={"poll_interval_seconds": poll_interval, "topic": settings.kafka_topic},
    )

    try:
        while True:
            producer.run_once()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("NewsData producer interrupted, shutting down")
    finally:
        producer.producer.close()


if __name__ == "__main__":
    main()
