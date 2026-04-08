from __future__ import annotations

import logging
import threading
from typing import Any

from pathlib import Path

from psycopg_pool import ConnectionPool

from src.db.repositories import get_all_active_topic_metadata, upsert_topic_metadata
from src.services.topic_metadata_loader import load_topic_metadata_from_artifacts

logger = logging.getLogger(__name__)

UNKNOWN_TOPIC = {"label": "unknown", "top_words": []}


class TopicMetadataCache:
    def __init__(
        self,
        pool: ConnectionPool,
        reload_interval_seconds: int = 300,
        artifacts_path: Path | None = None,
    ):
        self._pool = pool
        self._reload_interval = reload_interval_seconds
        self._cache: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._artifacts_path = Path(artifacts_path) if artifacts_path else None

    def start(self) -> None:
        self.reload()
        if self._thread is None:
            self._thread = threading.Thread(target=self._auto_reload, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _auto_reload(self) -> None:
        while not self._stop_event.wait(self._reload_interval):
            try:
                self.reload()
            except Exception:
                logger.exception("Failed to reload topic metadata cache")

    def reload(self) -> None:
        with self._pool.connection() as conn:
            metadata = get_all_active_topic_metadata(conn)

            # if not metadata and self._artifacts_path:
            #     logger.warning(
            #         "Topic metadata table is empty, attempting to seed from artifacts",
            #         extra={"artifacts_path": str(self._artifacts_path)},
            #     )
            #     records = list(
            #         load_topic_metadata_from_artifacts(self._artifacts_path)
            #     )
            #     if records:
            #         upsert_topic_metadata(conn, records)
            #         conn.commit()
            #         metadata = get_all_active_topic_metadata(conn)
            #     else:
            #         logger.error(
            #             "Failed to seed topic metadata: no records were loaded",
            #         )

        with self._lock:
            self._cache = metadata
        logger.info(
            "Topic metadata cache reloaded",
            extra={
                "topics_len": len(metadata),
                "topics": metadata,

            },
        )

    def get(self, cluster_id: int) -> dict[str, Any]:
        with self._lock:
            return self._cache.get(cluster_id, UNKNOWN_TOPIC)

    def get_all(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return dict(self._cache)
