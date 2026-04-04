import pandas as pd


def joincorpus():
    lenta_path = "data/corpus/lenta-ru-news.csv"
    twenty_path = "data/corpus/ru-news.csv"
    output_path = "data/corpus/joined.csv"

    lenta_df = pd.read_csv(lenta_path)

    # url, title, text, topic, tags, date
    lenta_df["date"] = pd.to_datetime(lenta_df["date"], errors="coerce")
    lenta_df = lenta_df[lenta_df["date"].dt.year > 2014]
    lenta_df = lenta_df.drop(columns=["url", "tags"])

    twenty_df = pd.read_csv(twenty_path)

    # source, title, text, publication_date, rubric, subrubric, tags
    twenty_df = twenty_df.drop(columns=["source", "rubric", "subrubric", "tags"])
    twenty_df = twenty_df.rename(columns={"publication_date": "date"})

    twenty_df["date"] = pd.to_datetime(twenty_df["date"], errors="coerce")

    new_df = pd.concat([lenta_df, twenty_df], ignore_index=True)
    new_df.to_csv(output_path, index=False)

    print("Joined corpus")
    print("Info: length:", len(new_df))