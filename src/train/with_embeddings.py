from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.metrics.coherence import calculate_coherence
from src.models.rubert import BertClusteringConfig, ClusteringConfig, TextClusterer
from src.processors.data_preprocess import preprocess_texts


def get_top_words_per_cluster(
    df: pd.DataFrame,
    labels: np.ndarray,
    text_col: str = "lemma_text",
    top_n: int = 15,
    min_df: int = 3,
    max_df: float = 0.8,
) -> dict[int, list[tuple[str, float]]]:
    result = {}
    df_local = df.copy()
    df_local["cluster"] = labels

    for cluster_id in sorted(df_local["cluster"].unique()):
        if cluster_id == -1:
            continue

        cluster_texts = (
            df_local.loc[df_local["cluster"] == cluster_id, text_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        if len(cluster_texts) < 2:
            result[int(cluster_id)] = []
            continue

        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        X = vectorizer.fit_transform(cluster_texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        vocab = np.array(vectorizer.get_feature_names_out())

        top_idx = scores.argsort()[::-1][:top_n]
        result[int(cluster_id)] = [(str(vocab[i]), float(scores[i])) for i in top_idx]

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


def compute_topic_diversity(
    topics: dict[int, list[tuple[str, float]]],
    top_n: int = 10,
) -> float:
    words = []

    for _, topic_words in topics.items():
        words.extend([w for w, _ in topic_words[:top_n]])

    if not words:
        return 0.0

    return len(set(words)) / len(words)


def run_bert_topic_pipeline_from_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_text_col: str = "text",
    bert_config: Optional[BertClusteringConfig] = None,
    clustering_config: Optional[ClusteringConfig] = None,
    top_n_words: int = 15,
    artifacts_dir: str | Path = "artifacts/rubert_from_embeddings",
    save_artifacts: bool = True,
):
    bert_config = bert_config or BertClusteringConfig()
    clustering_config = clustering_config or ClusteringConfig()

    artifacts_dir = Path(artifacts_dir)
    if save_artifacts:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

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

    if "text" not in df.columns:
        df["text"] = preprocess_texts(df["content"], needs_lower=False)

    # df = df.dropna(subset=[topic_text_col]).copy()
    df[topic_text_col] = df[topic_text_col].fillna("").astype(str)
    # df = df[df[topic_text_col].str.len() > 0].copy()
    # texts = df[topic_text_col].fillna("").astype(str).tolist()

    if len(df) != len(embeddings):
        raise ValueError(
            f"Mismatch between dataframe rows ({len(df)}) and embeddings ({len(embeddings)}). "
            "Embeddings must correspond exactly to the filtered dataframe."
        )

    clusterer = TextClusterer(clustering_config)
    labels = clusterer.fit_predict(embeddings)

    topics_with_scores = get_top_words_per_cluster(
        df=df,
        labels=labels,
        text_col=topic_text_col,
        top_n=top_n_words,
    )

    topics = {
        int(cluster_id): [word for word, _ in words]
        for cluster_id, words in topics_with_scores.items()
    }

    silhouette_input = (
        clusterer.X_used_for_clustering
        if getattr(clusterer, "X_used_for_clustering", None) is not None
        else embeddings
    )

    sil = compute_silhouette(silhouette_input, labels)
    td = compute_topic_diversity(topics_with_scores, top_n=min(10, top_n_words))
    coherence = calculate_coherence(topics, df, topic_text_col)

    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float((labels == -1).mean()) if np.any(labels == -1) else 0.0

    metrics = {
        "model": "rubert_clustering_from_embeddings",
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

    if clustering_config.method == "kmeans":
        metrics["n_clusters_requested"] = clustering_config.n_clusters
    elif clustering_config.method == "hdbscan":
        metrics["min_cluster_size"] = clustering_config.min_cluster_size
        metrics["min_samples"] = clustering_config.min_samples
        metrics["metric"] = clustering_config.metric

    if getattr(clustering_config, "use_pca", None) is not None:
        metrics["use_pca"] = clustering_config.use_pca
    if getattr(clustering_config, "pca_reducer_size", None) is not None:
        metrics["pca_reducer_size"] = clustering_config.pca_reducer_size

    result_df = df.copy()
    result_df["cluster"] = labels

    if save_artifacts:
        result_df.to_csv(artifacts_dir / "clustered_news.csv", index=False)
        np.save(artifacts_dir / "embeddings.npy", embeddings)
        np.save(artifacts_dir / "labels.npy", labels)

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
        "artifacts_dir": str(artifacts_dir),
    }



def prepare_dataframe(
    df: pd.DataFrame,
    raw_text_col: str = "text",
    topic_text_col: str = "lemmatized",
) -> pd.DataFrame:
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

    if raw_text_col not in df.columns:
        raise ValueError(f"Column '{raw_text_col}' not found in dataframe")

    if topic_text_col not in df.columns:
        raise ValueError(f"Column '{topic_text_col}' not found in dataframe")

    df = df.dropna(subset=[raw_text_col, topic_text_col]).copy()
    df[raw_text_col] = df[raw_text_col].astype(str)
    df[topic_text_col] = df[topic_text_col].astype(str)

    df = df[df[raw_text_col].str.len() > 0].copy()
    df = df[df[topic_text_col].str.len() > 0].copy()

    df = df.reset_index(drop=True)

    return df

def run():
    # data_path = "data/corpus/joined.csv"
    data_path = "data/corpus/joined-lemmatizedv2.csv"
    embeddings_path = "data/corpus/embeddings.npy"
    artifacts_dir = "artifacts/rubert-mean-kmeans-from-embeddings"

    df = pd.read_csv(data_path, low_memory=False)
    # df["text"] = df["text"].fillna("").astype(str)
    # df = pd.read_csv("data/corpus/joined.csv")

    # df = df["text"].fillna("").astype(str).tolist()

    # print(df.head(3))



    # print("==============")


    # print(df.tail(3))

    # prepare_dataframe(df, raw_text_col="text", topic_text_col="text")

    embeddings = np.load(embeddings_path)

    result = run_bert_topic_pipeline_from_embeddings(
        df=df,
        embeddings=embeddings,
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

    print("RuBERT topic pipeline from embeddings finished.")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    print("\nTopics:")
    for cluster_id, words in result["topics"].items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")

    print(f"\nArtifacts saved to: {result['artifacts_dir']}")


if __name__ == "__main__":
    run()