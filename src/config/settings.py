from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_dsn: str = Field(default="postgresql://postgres:postgres@localhost:5432/news")
    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=10, ge=1)

    service_name: str = Field(default="app")
    monitoring_enabled: bool = Field(default=False)
    monitoring_host: str = Field(default="0.0.0.0")
    monitoring_port: int | None = Field(default=None, ge=1)

    huggingface_cache_dir: str = Field(default="meta/hf-cache")
    huggingface_local_dir: str | None = Field(default=None)
    huggingface_allow_download: bool = Field(default=True)
    huggingface_download_retries: int = Field(default=3, ge=1)

    newsdata_api_key: str = Field(default="")
    newsdata_poll_interval_seconds: int = Field(default=60, alias="POLL_INTERVAL_SEC", ge=5)

    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic: str = Field(default="news")
    kafka_group_id: str = Field(default="news-consumer")
    kafka_consumer_poll_timeout_ms: int = Field(default=1000, ge=100)
    kafka_auto_offset_reset: str = Field(default="earliest")

    batch_size: int = Field(default=32, ge=1)
    batch_interval_seconds: int = Field(default=30, ge=1)

    model_artifacts_path: str = Field(default="meta")
    unknown_threshold: float | None = Field(default=None)

    metadata_reload_interval_seconds: int = Field(default=300, ge=5)

    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def kafka_bootstrap_servers_list(self) -> list[str]:
        return [s.strip() for s in self.kafka_bootstrap_servers.split(",") if s.strip()]

    @property
    def artifacts_path(self) -> Path:
        return Path(self.model_artifacts_path)

    @property
    def hf_cache_path(self) -> Path:
        path = Path(self.huggingface_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def hf_local_model_path(self) -> Path:
        if self.huggingface_local_dir:
            path = Path(self.huggingface_local_dir)
        else:
            path = self.artifacts_path / "hf-model"
        path.mkdir(parents=True, exist_ok=True)
        return path
