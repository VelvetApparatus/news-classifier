from __future__ import annotations

import logging
from typing import Optional

from psycopg_pool import ConnectionPool

from src.config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[ConnectionPool] = None


def get_connection_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        logger.info("Initializing PostgreSQL connection pool", extra={"dsn": settings.postgres_dsn})
        _pool = ConnectionPool(
            conninfo=settings.postgres_dsn,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
        )
    return _pool
