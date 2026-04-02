from sklearn.metrics import silhouette_score

def calculate_silhouette_score(
        X, labels,
):
    return float(silhouette_score(X, labels))