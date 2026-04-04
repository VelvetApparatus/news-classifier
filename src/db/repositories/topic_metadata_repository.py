from __future__ import annotations

from typing import Iterable

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Json


def get_all_active_topic_metadata(conn: Connection) -> dict[int, dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT cluster_id, label, top_words
            FROM topic_metadata
            WHERE is_active = TRUE
            """
        )
        rows = cur.fetchall()
    return {
        int(row["cluster_id"]): {
            "label": row["label"],
            "top_words": row["top_words"] or [],
        }
        for row in rows
    }


def upsert_topic_metadata(
    conn: Connection,
    metadata: Iterable[tuple[int, str, list[str]]],
) -> int:
    records = list(metadata)
    if not records:
        return 0

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO topic_metadata (cluster_id, label, top_words, is_active, updated_at)
            VALUES (%s, %s, %s, TRUE, NOW())
            ON CONFLICT (cluster_id) DO UPDATE
            SET label = EXCLUDED.label,
                top_words = EXCLUDED.top_words,
                is_active = EXCLUDED.is_active,
                updated_at = NOW()
            """,
            [
                (cluster_id, label, Json(top_words))
                for cluster_id, label, top_words in records
            ],
        )
        return cur.rowcount
