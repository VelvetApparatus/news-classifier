import pandas as pd

from src.models.rubert import RuBERTEmbedder, BertClusteringConfig




def run():
    df = pd.read_csv("data/corpus/joined.csv")

    texts = df["text"].fillna("").astype(str).tolist()
    embedder = RuBERTEmbedder(
        BertClusteringConfig(
            model_name="DeepPavlov/rubert-base-cased",
            max_length=128,
            batch_size=32,
            pooling="mean",
            normalize_embeddings=True,
        )
    )

    embeddings = embedder.encode_and_save(
        texts=texts,
        save_path="data/corpus/embeddings.npy"
    )

    print("Embeddings saved:", embeddings.shape)



if __name__ == "__main__":
    run()