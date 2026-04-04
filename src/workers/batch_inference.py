from __future__ import annotations

import logging
import time
from typing import Sequence
from uuid import uuid4

from psycopg_pool import ConnectionPool

from src.config import get_settings
from src.db import get_connection_pool
from src.db.repositories import (
    get_unprocessed_news_batch,
    mark_news_failed,
    mark_news_processed,
    mark_news_processing,
    save_clustering_reports,
)
from src.services.model_service import ModelService
from src.services.topic_metadata_cache import TopicMetadataCache, UNKNOWN_TOPIC
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def fetch_batch(pool: ConnectionPool, limit: int) -> list[dict]:
    with pool.connection() as conn:
        with conn.transaction():
            batch = get_unprocessed_news_batch(conn, limit=limit)
            mark_news_processing(conn, [row["id"] for row in batch])
    if batch:
        logger.info("Fetched batch for inference", extra={"size": len(batch)})
    return batch


def should_mark_unknown(score: float | None, threshold: float | None) -> bool:
    if threshold is None or score is None:
        return False
    return score > threshold


def process_batch(
    pool: ConnectionPool,
    model_service: ModelService,
    metadata_cache: TopicMetadataCache,
    batch: Sequence[dict],
    unknown_threshold: float | None,
) -> None:
    texts = [item["raw_text"] for item in batch]
    news_ids = [item["id"] for item in batch]

    try:
        predictions = model_service.predict_batch(texts)
    except Exception as exc:
        logger.exception("Model inference failed for batch")
        with pool.connection() as conn:
            mark_news_failed(conn, news_ids, str(exc))
            conn.commit()
        return

    if len(predictions) != len(batch):
        raise RuntimeError("Predictions count does not match batch size")

    reports = []
    for item, prediction in zip(batch, predictions):
        cluster_id = prediction.get("cluster_id")
        score = prediction.get("score")
        resolved_cluster_id = cluster_id
        label = "unknown"
        top_words: list[str] = []

        if cluster_id is not None:
            meta = metadata_cache.get(cluster_id)
        else:
            meta = UNKNOWN_TOPIC

        if should_mark_unknown(score, unknown_threshold) or meta is UNKNOWN_TOPIC:
            resolved_cluster_id = None
            label = "unknown"
            top_words = []
        else:
            label = meta["label"]
            top_words = meta["top_words"]

        reports.append(
            {
                "id": uuid4(),
                "news_id": item["id"],
                "cluster_id": resolved_cluster_id,
                "cluster_label": label,
                "top_words_snapshot": top_words,
                "score": score,
            }
        )

    with pool.connection() as conn:
        save_clustering_reports(conn, reports)
        mark_news_processed(conn, news_ids)
        conn.commit()

    logger.info(
        "Batch inference completed",
        extra={
            "processed": len(batch),
            "unknown": sum(1 for report in reports if report["cluster_id"] is None),
        },
    )


def run_worker() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    pool = get_connection_pool()
    model_service = ModelService(settings.artifacts_path)
    metadata_cache = TopicMetadataCache(
        pool,
        reload_interval_seconds=settings.metadata_reload_interval_seconds,
    )
    metadata_cache.start()

    logger.info("Batch inference worker started", extra={"batch_size": settings.batch_size})

    try:
        while True:
            batch = fetch_batch(pool, settings.batch_size)
            if not batch:
                time.sleep(settings.batch_interval_seconds)
                continue

            try:
                process_batch(
                    pool,
                    model_service,
                    metadata_cache,
                    batch,
                    settings.unknown_threshold,
                )
            except Exception:
                logger.exception("Failed to process batch")
                with pool.connection() as conn:
                    mark_news_failed(
                        conn,
                        [item["id"] for item in batch],
                        "unexpected error during batch processing",
                    )
                    conn.commit()
            time.sleep(settings.batch_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Batch inference worker interrupted, shutting down")
    finally:
        metadata_cache.stop()


if __name__ == "__main__":
    run_worker()
