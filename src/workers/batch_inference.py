from __future__ import annotations

import logging
import time
from typing import Sequence
from uuid import uuid4

from psycopg_pool import ConnectionPool

from src.config import Settings, get_settings
from src.db import get_connection_pool
from src.db.repositories import (
    get_unprocessed_news_batch,
    mark_news_failed,
    mark_news_processed,
    mark_news_processing,
    save_clustering_reports,
)
from src.monitoring import setup_monitoring
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
    return score < threshold


def process_batch(
    pool: ConnectionPool,
    model_service: ModelService,
    metadata_cache: TopicMetadataCache,
    batch: Sequence[dict],
    unknown_threshold: float | None,
    metrics=None,
    health=None,
) -> None:

    texts = [item["title"] for item in batch]
    news_ids = [item["id"] for item in batch]

    try:
        inference_started = time.perf_counter()
        predictions = model_service.predict_batch(texts)
        inference_duration = time.perf_counter() - inference_started
        if metrics:
            metrics.observe_model_inference(inference_duration)
        if health:
            health.report_ok(
                "model",
                f"inference_success batch={len(batch)} duration={inference_duration:.3f}s",
            )
    except Exception as exc:
        logger.exception("Model inference failed for batch")
        with pool.connection() as conn:
            mark_news_failed(conn, news_ids, str(exc))
            conn.commit()
        if metrics:
            metrics.inc_batch_errors()
        if health:
            health.report_error("model", str(exc))
        return

    if len(predictions) != len(batch):
        raise RuntimeError("Predictions count does not match batch size")

    reports = []
    for item, prediction in zip(batch, predictions):
        print(prediction)
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
            if meta is UNKNOWN_TOPIC and health:
                health.report_error(
                    "topic_metadata", f"metadata missing for cluster_id={cluster_id}"
                )
        else:
            label = meta["label"]
            top_words = meta["top_words"]
            if health:
                health.report_ok("topic_metadata", f"cluster {cluster_id} resolved")

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

    unknown_count = sum(1 for report in reports if report["cluster_id"] is None)
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
    if metrics:
        metrics.inc_batch_processed(len(batch), unknown_count)
    if health:
        health.report_ok(
            "batch",
            f"processed={len(batch)} unknown={unknown_count}",
        )


def run_worker(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    metrics, health, monitoring_server = setup_monitoring(
        service_name=settings.service_name or "batch_inference",
        host=settings.monitoring_host,
        port=settings.monitoring_port,
        enabled=settings.monitoring_enabled,
    )
    pool = get_connection_pool()
    model_service = ModelService(
        settings.artifacts_path,
        huggingface_cache_dir=settings.hf_cache_path,
        huggingface_local_dir=settings.hf_local_model_path,
        huggingface_allow_download=settings.huggingface_allow_download,
        huggingface_download_retries=settings.huggingface_download_retries,
    )
    metadata_cache = TopicMetadataCache(
        pool,
        reload_interval_seconds=settings.metadata_reload_interval_seconds,
        artifacts_path=settings.artifacts_path,
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
                batch_started = time.perf_counter()
                process_batch(
                    pool,
                    model_service,
                    metadata_cache,
                    batch,
                    settings.unknown_threshold,
                    metrics=metrics,
                    health=health,
                )
                duration = time.perf_counter() - batch_started
                if metrics:
                    metrics.observe_batch_duration(duration)
                if health:
                    health.report_ok(
                        "worker",
                        f"batch processed size={len(batch)} duration={duration:.3f}s",
                    )
            except Exception:
                logger.exception("Failed to process batch")
                if metrics:
                    metrics.inc_batch_errors()
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
        if monitoring_server:
            monitoring_server.stop()


if __name__ == "__main__":
    run_worker()
