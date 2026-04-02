import pymorphy3
from functools import lru_cache

morph = pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=200000)
def lemmatize_token(token: str) -> str:
    return morph.parse(token)[0].normal_form


def lemmatize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    tokens = text.split()
    lemmas = [lemmatize_token(token) for token in tokens]
    return " ".join(lemmas)