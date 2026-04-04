from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def save_articles_per_year_histogram(
    dates: Iterable[str],
    output_path: str = "articles_per_year_histogram.png",
) -> Path:
    """Сохраняет столбчатую диаграмму количества статей по годам."""
    parsed_dates = pd.to_datetime(list(dates), errors="coerce")
    years = pd.Series(parsed_dates).dropna().dt.year

    if years.empty:
        raise ValueError("В dates нет корректных значений даты для построения графика")

    counts = years.value_counts().sort_index()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Количество статей по годам")
    plt.xlabel("Год")
    plt.ylabel("Количество статей")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    return output


def save_article_lengths_histogram(
    article_texts: Iterable[str],
    output_path: str = "article_lengths_histogram.png",
    percentile_clip: float = 0.99,
    bins: int = 30,
) -> Path:
    """Сохраняет гистограмму длин статей.

    Длина измеряется в символах.
    Для улучшения читаемости обрезает длинный хвост по percentile_clip.
    """
    lengths = pd.Series([len((text or "").strip()) for text in article_texts], dtype="int64")

    if lengths.empty:
        raise ValueError("article_texts пуст, строить нечего")

    clip_value = int(lengths.quantile(percentile_clip))
    clipped_lengths = lengths[lengths <= clip_value]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(clipped_lengths, bins=bins, edgecolor="black")
    plt.title(f"Распределение длин статей (до {int(percentile_clip * 100)}-го перцентиля)")
    plt.xlabel("Длина статьи (символы)")
    plt.ylabel("Количество статей")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    return output


if __name__ == "__main__":
    df = pd.read_csv("data/corpus/joined.csv", low_memory=False)

    if "text" in df.columns:
        save_article_lengths_histogram(
            df["text"].fillna(""),
            output_path="article_lengths_histogram.png",
            percentile_clip=0.99,
            bins=40,
        )

    if "date" in df.columns:
        save_articles_per_year_histogram(
            df["date"],
            output_path="articles_per_year_histogram.png",
        )