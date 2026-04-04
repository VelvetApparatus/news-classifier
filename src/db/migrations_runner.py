from __future__ import annotations

import logging
from pathlib import Path

import psycopg

from src.config import get_settings

logger = logging.getLogger(__name__)


def apply_migrations() -> None:
    settings = get_settings()
    migrations_dir = Path(__file__).resolve().parent
    migration_files = sorted(f for f in migrations_dir.glob("*.sql"))

    if not migration_files:
        logger.info("No migrations found", extra={"dir": str(migrations_dir)})
        return

    with psycopg.connect(settings.postgres_dsn) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        applied_versions = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_migrations")
        }

        for file_path in migration_files:
            version = file_path.stem
            if version in applied_versions:
                logger.info("Migration already applied", extra={"version": version})
                continue

            sql = file_path.read_text(encoding="utf-8")
            logger.info("Applying migration", extra={"version": version, "path": str(file_path)})
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )

    logger.info("Migrations applied")


def main() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    apply_migrations()


if __name__ == "__main__":
    main()
