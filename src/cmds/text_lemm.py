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

    df[target_col] = preprocess_texts(df[target_col])
    df.fillna("", inplace=True)
    # df = df[df[target_col].str.len() > 0].copy()

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in the CSV file.")

    df[lemm_col] = df[target_col].progress_apply(lemmatize_text)
    df.to_csv(csv_write_path, index=False)


if __name__ == "__main__":
    lemm_csv(
        csv_read_path="data/corpus/joined.csv",
        csv_write_path="data/corpus/joined-lemmatizedv2.csv",
        target_col="text",
        lemm_col="lemmatized"
    )
