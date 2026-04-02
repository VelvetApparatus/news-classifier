from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


def calculate_coherence(topics, df, target_column):
    texts = [str(text).split() for text in df[target_column].fillna("")]
    dictionary = Dictionary(texts)

    topic_list = list(topics.values())

    if not topic_list:
        raise ValueError("Topics are empty.")

    cm = CoherenceModel(
        topics=topic_list,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    )

    return cm.get_coherence()