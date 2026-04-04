from __future__ import annotations

from typing import Iterable

from psycopg import Connection
from psycopg.types.json import Json


def save_clustering_reports(conn: Connection, reports: Iterable[dict]) -> int:
    records = list(reports)
    if not records:
        return 0

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO clustering_reports (
                id,
                news_id,
                cluster_id,
                cluster_label,
                top_words_snapshot,
                score,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """,
            [
                (
                    report["id"],
                    report["news_id"],
                    report.get("cluster_id"),
                    report["cluster_label"],
                    Json(report.get("top_words_snapshot", [])),
                    report.get("score"),
                )
                for report in records
            ],
        )
        return cur.rowcount
