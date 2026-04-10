from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import hdbscan
import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def save_clusterer(clusterer: TextClusterer, path: str) -> None:
    joblib.dump(clusterer, path)


def load_clusterer(path: str) -> TextClusterer:
    return joblib.load(path)


class TextClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.model = None
        self.reducer = PCA(n_components=config.pca_reducer_size, random_state=42)

        if config.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=20,
                max_iter=500,
            )
        else:
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                metric=self.config.metric,
                prediction_data=True
            )

    def _fit_reduce(self, X: np.ndarray) -> np.ndarray:
        reducer = getattr(self, "reducer", None)
        if reducer is None:
            return X
        return reducer.fit_transform(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        reducer = getattr(self, "reducer", None)
        if reducer is None:
            return X
        return reducer.transform(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        embeddings_reduced = self._fit_reduce(X)
        labels = self.model.fit_predict(embeddings_reduced)
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self._transform(X)

        if self.config.method == "kmeans":
            return self.model.predict(X_reduced)

        elif self.config.method == "hdbscan":
            labels, _ = hdbscan.approximate_predict(self.model, X_reduced)
            return labels

        raise ValueError(f"Unknown clustering method: {self.config.method}")

    def  predict_with_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_reduced = self._transform(X)

        if self.config.method == "kmeans":
            labels = self.model.predict(X_reduced)
            distances = self.model.transform(X_reduced)
            min_distances = distances.min(axis=1)
            confidences = np.exp(-min_distances)
            return labels, confidences

        elif self.config.method == "hdbscan":
            labels, strengths = hdbscan.approximate_predict(self.model, X_reduced)
            strengths = np.clip(strengths, 0.0, 1.0)
            return labels, strengths

        raise ValueError(f"Unknown clustering method: {self.config.method}")


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: [batch, seq_len, hidden]
    attention_mask:    [batch, seq_len]
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pooling(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    BERT CLS token representation.
    """
    return last_hidden_state[:, 0]


@dataclass
class BertClusteringConfig:
    model_name: str = "DeepPavlov/rubert-base-cased"
    max_length: int = 128
    batch_size: int = 32
    pooling: Literal["mean", "cls"] = "mean"
    normalize_embeddings: bool = True
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


class RuBERTEmbedder:
    def __init__(
        self,
        config: BertClusteringConfig,
        *,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        allow_download: bool = True,
        download_retries: int = 3,
    ):
        self.config = config
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._local_dir = Path(local_dir) if local_dir else None
        self._allow_download = allow_download
        self._download_retries = max(1, download_retries)
        self._local_ready = False

        if self._local_dir:
            self._prepare_local_weights()

        self.tokenizer = self._load_component("tokenizer", AutoTokenizer)
        self.model = self._load_component("model", AutoModel)
        self.model.to(config.device)
        self.model.eval()

    def _prepare_local_weights(self) -> None:
        if not self._local_dir:
            return
        if self._is_local_ready():
            self._local_ready = True
            logger.info("Using existing local RuBERT weights at %s", self._local_dir)
            return
        if not self._allow_download:
            logger.warning(
                "Local RuBERT directory %s is empty and downloads disabled",
                self._local_dir,
            )
            return

        for attempt in range(1, self._download_retries + 1):
            try:
                logger.info(
                    "Downloading RuBERT snapshot to %s (attempt %s/%s)",
                    self._local_dir,
                    attempt,
                    self._download_retries,
                )
                snapshot_download(
                    repo_id=self.config.model_name,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None,
                    local_dir=str(self._local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                self._local_ready = True
                logger.info("RuBERT snapshot downloaded to %s", self._local_dir)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to download RuBERT snapshot: %s (attempt %s/%s)",
                    exc,
                    attempt,
                    self._download_retries,
                )
                time.sleep(min(2**attempt, 10))

        logger.warning(
            "Unable to download RuBERT snapshot; falling back to cache/network"
        )

    def _is_local_ready(self) -> bool:
        if not self._local_dir:
            return False
        return (self._local_dir / "config.json").exists()

    def _iter_sources(self) -> list[tuple[str, bool, str]]:
        sources: list[tuple[str, bool, str]] = []
        if self._local_dir and self._is_local_ready():
            sources.append((str(self._local_dir), True, "local_dir"))
        sources.append((self.config.model_name, True, "cache"))
        if self._allow_download:
            sources.append((self.config.model_name, False, "remote"))
        return sources

    def _load_component(self, component: str, loader) -> Any:
        last_exc: Exception | None = None
        for source, local_only, origin in self._iter_sources():
            retries = 1 if local_only else self._download_retries
            for attempt in range(1, retries + 1):
                try:
                    kwargs = {"local_files_only": local_only}
                    if self._cache_dir and source == self.config.model_name:
                        kwargs["cache_dir"] = str(self._cache_dir)
                    return loader.from_pretrained(source, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "Failed to load %s from %s (attempt %s/%s): %s",
                        component,
                        origin,
                        attempt,
                        retries,
                        exc,
                    )
                    if local_only:
                        break
                    time.sleep(min(2**attempt, 10))
            # loop next source
        if last_exc:
            raise RuntimeError(
                f"Unable to load {component} for RuBERT from available sources"
            ) from last_exc
        raise RuntimeError(f"No sources available to load {component}")

    def encode_and_save(self, texts: list[str], save_path: str | Path) -> np.ndarray:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = self.encode(texts)
        np.save(save_path, embeddings)

        return embeddings

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Encoding"):
            batch_texts = texts[i:i + self.config.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )

            encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state  # [B, L, H]

            if self.config.pooling == "mean":
                batch_embeddings = mean_pooling(last_hidden_state, encoded["attention_mask"])
            elif self.config.pooling == "cls":
                batch_embeddings = cls_pooling(last_hidden_state)
            else:
                raise ValueError(f"Unknown pooling: {self.config.pooling}")

            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        if self.config.normalize_embeddings:
            embeddings = normalize(embeddings, norm="l2")

        return embeddings


@dataclass
class ClusteringConfig:
    method: Literal["kmeans", "hdbscan"] = "kmeans"
    n_clusters: int = 10
    random_state: int = 42
    pca_reducer_size: int = 50

    # HDBSCAN params
    min_cluster_size: int = 20
    min_samples: Optional[int] = None
    metric: str = "euclidean"
