import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.processors.data_preprocess import preprocess_texts
from src.metrics.diversity import topic_diversity
from src.metrics.coherence import calculate_coherence
from src.metrics.silhouette_score import calculate_silhouette_score

ARTIFACTS_DIR = Path("artifacts/baseline")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_top_words_per_cluster(
        vectorizer: TfidfVectorizer,
        kmeans: KMeans,
        top_n: int = 10,
):
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    centers = kmeans.cluster_centers_
    for cluster_id in range(centers.shape[0]):
        top_indices = centers[cluster_id].argsort()[::-1][:top_n]
        topics[int(cluster_id)] = [feature_names[i] for i in top_indices]

    return topics


def run():
    data_path = "data/corpus/ru-news.csv"
    df = pd.read_csv(data_path)

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

    df["clean_text"] = preprocess_texts(df["content"])
    df = df[df["clean_text"].str.len() > 0].copy()

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.5
    )
    X = vectorizer.fit_transform(df["clean_text"])

    n_clusters = 8
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )
    labels = kmeans.fit_predict(X)

    df["cluster"] = labels

    topics = extract_top_words_per_cluster(vectorizer, kmeans, top_n=10)
    sil_score = calculate_silhouette_score(X, labels)
    coherence = calculate_coherence(topics, df,  "clean_text")
    diversity = topic_diversity(topics)


    df.to_csv(ARTIFACTS_DIR / "clustered_news.csv", index=False)
    joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(kmeans, ARTIFACTS_DIR / "kmeans_model.pkl")

    with open(ARTIFACTS_DIR / "topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tfidf_kmeans_baseline",
        "n_clusters": n_clusters,
        "n_samples": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "silhouette_score": sil_score,
        "coherence": coherence,
        "diversity": diversity,
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Baseline training finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Topics:")
    for cluster_id, words in topics.items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")
