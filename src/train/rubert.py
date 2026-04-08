from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib

from src.metrics.coherence import calculate_coherence
from src.metrics.diversity import topic_diversity

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.models.rubert import BertClusteringConfig, ClusteringConfig, RuBERTEmbedder, TextClusterer
from src.processors.data_preprocess import preprocess_texts


def get_top_words_per_cluster(
        df: pd.DataFrame,
        labels: np.ndarray,
        text_col: str = "lemma_text",
        top_n: int = 15,
        min_df: int = 3,
        max_df: float = 0.8
) -> dict[int, list[tuple[str, float]]]:
    """
    Возвращает топ-слова для каждого кластера.
    Для HDBSCAN label = -1 — шум, его можно пропустить.
    """
    result = {}
    df_local = df.copy()
    df_local["cluster"] = labels

    for cluster_id in sorted(df_local["cluster"].unique()):
        if cluster_id == -1:
            continue

        cluster_texts = df_local.loc[df_local["cluster"] == cluster_id, text_col].dropna().astype(str).tolist()
        if len(cluster_texts) < 2:
            result[cluster_id] = []
            continue

        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b\w+\b"
        )
        X = vectorizer.fit_transform(cluster_texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        vocab = np.array(vectorizer.get_feature_names_out())

        top_idx = scores.argsort()[::-1][:top_n]
        result[cluster_id] = list(zip(vocab[top_idx], scores[top_idx]))

    return result


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique_labels = set(labels)
    if -1 in unique_labels:
        mask = labels != -1
        X_eval = X[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X
        labels_eval = labels

    if len(set(labels_eval)) < 2:
        return None

    return float(silhouette_score(X_eval, labels_eval))


def topic_diversity(topics: dict[int, list[tuple[str, float]]], top_n: int = 10) -> float:
    """
    Topic Diversity = unique_words / total_top_words
    """
    words = []
    for _, topic_words in topics.items():
        words.extend([w for w, _ in topic_words[:top_n]])

    if not words:
        return 0.0

    return len(set(words)) / len(words)


def run_bert_topic_pipeline(
        df: pd.DataFrame,
        raw_text_col: str = "text",
        topic_text_col: str = "text",
        bert_config: Optional[BertClusteringConfig] = None,
        clustering_config: Optional[ClusteringConfig] = None,
        top_n_words: int = 15,
        artifacts_dir: str | Path = "artifacts/rubert",
        save_artifacts: bool = True,
):
    bert_config = bert_config or BertClusteringConfig()
    clustering_config = clustering_config or ClusteringConfig()

    artifacts_dir = Path(artifacts_dir)
    if save_artifacts:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 0) Подготовка данных
    # -----------------------------
    df = df.copy()

    if "content" not in df.columns:
        title = df["title"].fillna("") if "title" in df.columns else ""
        text = df["text"].fillna("") if "text" in df.columns else ""
        if isinstance(title, str):
            df["content"] = text
        else:
            df["content"] = (title + " " + text).str.strip()

    df = df.dropna(subset=["content"]).copy()
    df["content"] = df["content"].astype(str)
    df = df[df["content"].str.len() > 20].copy()

    df["text"] = preprocess_texts(df["content"], needs_lower=False)
    df = df[df["text"].str.len() > 0].copy()

    df = df.dropna(subset=[raw_text_col, topic_text_col]).copy()
    df[raw_text_col] = df[raw_text_col].astype(str)
    df[topic_text_col] = df[topic_text_col].astype(str)

    df = df[df[raw_text_col].str.len() > 0].copy()
    df = df[df[topic_text_col].str.len() > 0].copy()

    texts = df[raw_text_col].tolist()

    # -----------------------------
    # 1) Embeddings
    # -----------------------------
    embedder = RuBERTEmbedder(bert_config)
    embeddings = embedder.encode(texts)

    # -----------------------------
    # 2) Clustering
    # -----------------------------
    clusterer = TextClusterer(clustering_config)
    labels = clusterer.fit_predict(embeddings)

    # -----------------------------
    # 3) Topics
    # -----------------------------

    print("predicted")
    topics_with_scores = get_top_words_per_cluster(
        df=df,
        labels=labels,
        text_col=topic_text_col,
        top_n=top_n_words
    )

    # для json и печати удобнее отдельно хранить только слова
    topics = {
        int(cluster_id): [word for word, _ in words]
        for cluster_id, words in topics_with_scores.items()
    }

    # -----------------------------
    # 4) Metrics
    # -----------------------------
    sil = compute_silhouette(embeddings, labels)
    td = topic_diversity(topics_with_scores, top_n=min(10, top_n_words))
    coherence = calculate_coherence(topics, df, topic_text_col)

    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float((labels == -1).mean()) if np.any(labels == -1) else 0.0

    metrics = {
        "model": "rubert_clustering",
        "bert_model_name": bert_config.model_name,
        "pooling": bert_config.pooling,
        "normalize_embeddings": bert_config.normalize_embeddings,
        "max_length": bert_config.max_length,
        "batch_size": bert_config.batch_size,
        "clustering_method": clustering_config.method,
        "n_samples": int(df.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "silhouette_score": sil,
        "coherence": coherence,
        "topic_diversity": td,
        "n_clusters_found": int(n_clusters_found),
        "noise_fraction": noise_fraction,
    }

    # дополнительные параметры кластеризации
    if clustering_config.method == "kmeans":
        metrics["n_clusters_requested"] = clustering_config.n_clusters
    elif clustering_config.method == "hdbscan":
        metrics["min_cluster_size"] = clustering_config.min_cluster_size
        metrics["min_samples"] = clustering_config.min_samples
        metrics["metric"] = clustering_config.metric

    # -----------------------------
    # 5) Result dataframe
    # -----------------------------
    result_df = df.copy()
    result_df["cluster"] = labels

    # -----------------------------
    # 6) Save artifacts
    # -----------------------------
    if save_artifacts:
        result_df.to_csv(artifacts_dir / "clustered_news.csv", index=False)
        np.save(artifacts_dir / "embeddings.npy", embeddings)
        np.save(artifacts_dir / "labels.npy", labels)

        # сохраняем именно обертку clusterer
        joblib.dump(clusterer, artifacts_dir / "clusterer.pkl")

        with open(artifacts_dir / "topics.json", "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)

        with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        with open(artifacts_dir / "bert_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(bert_config), f, ensure_ascii=False, indent=2)

        with open(artifacts_dir / "clustering_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(clustering_config), f, ensure_ascii=False, indent=2)

    return {
        "data": result_df,
        "embeddings": embeddings,
        "labels": labels,
        "topics": topics,
        "topics_with_scores": topics_with_scores,
        "metrics": metrics,
        "clusterer": clusterer,
        "embedder": embedder,
        "artifacts_dir": str(artifacts_dir),
    }


def run():
    data_path = "data/corpus/ru-news-lemmatized.csv"
    artifacts_dir = "artifacts/rubert-cls-kmeans-v13"

    df = pd.read_csv(data_path)

    # пример: если хочешь брать raw_text из text, а top words из lemmatized
    result = run_bert_topic_pipeline(
        df=df,
        raw_text_col="text",
        topic_text_col="lemmatized",
        bert_config=BertClusteringConfig(
            model_name="DeepPavlov/rubert-base-cased",
            max_length=128,
            batch_size=16,
            pooling="mean",
            normalize_embeddings=True,
        ),
        clustering_config=ClusteringConfig(
            method="kmeans",
            n_clusters=6,
            random_state=42,
            min_cluster_size=5,
            min_samples=5,
        ),
        top_n_words=10,
        artifacts_dir=artifacts_dir,
        save_artifacts=True,
    )

    print("RuBERT topic pipeline finished.")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    print("\nTopics:")
    for cluster_id, words in result["topics"].items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")

    print(f"\nArtifacts saved to: {result['artifacts_dir']}")
