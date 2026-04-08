from src.models.rubert import RuBERTEmbedder, TextClusterer


def predict_text(
    text: str,
    embedder: RuBERTEmbedder,
    clusterer: TextClusterer,
    cluster_labels: dict[int, str],
    topics: dict[int, list[str]],
) -> dict:
    embedding = embedder.encode([text])
    labels, scores = clusterer.predict_with_scores(embedding)

    cluster_id = int(labels[0])
    score = float(scores[0])

    return {
        "cluster_id": cluster_id,
        "topic_label": cluster_labels[cluster_id],
        "top_words": topics[cluster_id],
        "score": score,
    }


def predict_batch(
    texts: list[str],
    embedder: RuBERTEmbedder,
    clusterer: TextClusterer,
    topic_metadata: dict[int, dict],
    unknown_threshold: float | None = None,
) -> list[dict]:
    if not texts:
        return []

    embeddings = embedder.encode(texts)
    labels, scores = clusterer.predict_with_scores(embeddings)

    results = []

    for text, cluster_id, score in zip(texts, labels, scores):
        cluster_id = int(cluster_id)
        score = float(score)

        # if unknown_threshold is not None and score < unknown_threshold:
        #     results.append({
        #         "text": text,
        #         "cluster_id": None,
        #         "topic_label": "unknown",
        #         "top_words": [],
        #         "score": score,
        #     })
        #     continue

        meta = topic_metadata.get(cluster_id, {
            "label": f"cluster_{cluster_id}",
            "top_words": []
        })

        results.append({
            "text": text,
            "cluster_id": cluster_id,
            "topic_label": meta["label"],
            "top_words": meta["top_words"],
            "score": score,
        })

    return results
