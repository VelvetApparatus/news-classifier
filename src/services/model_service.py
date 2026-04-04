from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.models.rubert import (
    BertClusteringConfig,
    RuBERTEmbedder,
    TextClusterer,
    load_clusterer,
)

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, artifacts_path: str | Path):
        self._artifacts_path = Path(artifacts_path)
        self._embedder: RuBERTEmbedder | None = None
        self._clusterer: TextClusterer | None = None
        self._load()

    def _load(self) -> None:
        bert_config_path = self._artifacts_path / "bert_config.json"
        clusterer_path = self._artifacts_path / "clusterer.pkl"

        if not bert_config_path.exists():
            raise FileNotFoundError(f"Bert config not found at {bert_config_path}")

        if not clusterer_path.exists():
            raise FileNotFoundError(f"Clusterer artifact not found at {clusterer_path}")

        with bert_config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

        bert_config = BertClusteringConfig(**config_data)
        if bert_config.device == "cuda" and not torch.cuda.is_available():
            bert_config.device = "cpu"
        elif bert_config.device == "mps" and not torch.backends.mps.is_available():
            bert_config.device = "cpu"
        self._embedder = RuBERTEmbedder(bert_config)
        self._clusterer = load_clusterer(str(clusterer_path))
        logger.info(
            "Model service initialized",
            extra={"artifacts": str(self._artifacts_path)},
        )

    def predict_text(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        if self._embedder is None or self._clusterer is None:
            raise RuntimeError("Model service is not initialized")

        embeddings = self._embedder.encode(texts)
        labels, scores = self._clusterer.predict_with_scores(embeddings)

        return [
            {
                "cluster_id": int(cluster_id),
                "score": float(score),
            }
            for cluster_id, score in zip(labels, scores)
        ]
