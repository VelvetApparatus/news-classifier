from src.processors.data_preprocess import preprocess_texts
from src.processors.lemmatization import lemmatize_text
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def lemm_csv(
        csv_read_path: str,
        csv_write_path: str,
        target_col: str,
        lemm_col: str
):

    df = pd.read_csv(csv_read_path)

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

    df[target_col] = preprocess_texts(df["content"])
    df = df[df[target_col].str.len() > 0].copy()
    df.drop(columns=["content", ], inplace=True)


    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in the CSV file.")

    df[lemm_col] = df[target_col].progress_apply(lemmatize_text)
    df.to_csv(csv_write_path, index=False)



if __name__ == "__main__":
    lemm_csv(
        csv_read_path="data/corpus/ru-news.csv",
        csv_write_path="data/corpus/ru-news-lemmatized.csv",
        target_col="text",
        lemm_col="lemmatized"
    )

