from __future__ import annotations

from datetime import datetime
from typing import Iterable
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Json

INCOMING_NEWS_STATUS_NEW = "new"
INCOMING_NEWS_STATUS_PROCESSING = "processing"
INCOMING_NEWS_STATUS_PROCESSED = "processed"
INCOMING_NEWS_STATUS_FAILED = "failed"


def insert_incoming_news(
    conn: Connection,
    *,
    news_id: UUID,
    external_id: str,
    title: str | None,
    source: str | None,
    published_at: datetime | None,
    raw_text: str,
    url: str | None,
    payload: dict,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO incoming_news (
                id,
                external_id,
                title,
                source,
                published_at,
                raw_text,
                url,
                payload_json,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id) DO NOTHING
            """,
            (
                news_id,
                external_id,
                title,
                source,
                published_at,
                raw_text,
                url,
                Json(payload),
                INCOMING_NEWS_STATUS_NEW,
            ),
        )
        return cur.rowcount > 0


def get_unprocessed_news_batch(
    conn: Connection,
    *,
    limit: int,
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, external_id, title, source, published_at, raw_text, url, payload_json
            FROM incoming_news
            WHERE status = %s
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT %s
            """,
            (INCOMING_NEWS_STATUS_NEW, limit),
        )
        return cur.fetchall()


def mark_news_processing(conn: Connection, news_ids: Iterable[UUID]) -> int:
    ids = list(news_ids)
    if not ids:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE incoming_news
            SET status = %s,
                updated_at = NOW(),
                error_message = NULL
            WHERE id = ANY(%s)
              AND status = %s
            """,
            (INCOMING_NEWS_STATUS_PROCESSING, ids, INCOMING_NEWS_STATUS_NEW),
        )
        return cur.rowcount


def mark_news_processed(conn: Connection, news_ids: Iterable[UUID]) -> int:
    ids = list(news_ids)
    if not ids:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE incoming_news
            SET status = %s,
                updated_at = NOW(),
                error_message = NULL
            WHERE id = ANY(%s)
            """,
            (INCOMING_NEWS_STATUS_PROCESSED, ids),
        )
        return cur.rowcount


def mark_news_failed(conn: Connection, news_ids: Iterable[UUID], error_message: str) -> int:
    ids = list(news_ids)
    if not ids:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE incoming_news
            SET status = %s,
                updated_at = NOW(),
                error_message = %s
            WHERE id = ANY(%s)
            """,
            (INCOMING_NEWS_STATUS_FAILED, error_message[:1000], ids),
        )
        return cur.rowcount
