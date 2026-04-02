from __future__ import annotations


from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import hdbscan




class TextClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.model = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:

        pca = PCA(n_components=50, random_state=42)
        embeddings_reduced = pca.fit_transform(X)
        if self.config.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=20,
                max_iter=500,
            )
            labels = self.model.fit_predict(embeddings_reduced)

        elif self.config.method == "hdbscan":
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                metric=self.config.metric,
                prediction_data=True
            )
            labels = self.model.fit_predict(embeddings_reduced)

        else:
            raise ValueError(f"Unknown clustering method: {self.config.method}")

        return labels

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
    def __init__(self, config: BertClusteringConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(config.device)
        self.model.eval()

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

    # HDBSCAN params
    min_cluster_size: int = 20
    min_samples: Optional[int] = None
    metric: str = "euclidean"