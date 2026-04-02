


def topic_diversity(topics: dict) -> float:
    all_words = []
    for words in topics.values():
        all_words.extend(words)

    unique_words = set(all_words)
    return len(unique_words) / len(all_words)
