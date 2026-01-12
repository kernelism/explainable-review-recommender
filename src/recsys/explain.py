from __future__ import annotations
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def top_phrases(user_doc: str, item_doc: str, top_n: int = 3) -> List[str]:
    # Lightweight, deterministic explanation: overlap phrases via TF-IDF on (user, item) docs
    corpus = [user_doc, item_doc]
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=20000,
        stop_words="english"
    )
    X = vec.fit_transform(corpus)  # shape (2, vocab)
    u = X[0].toarray().ravel()
    it = X[1].toarray().ravel()
    scores = u * it  # overlap importance
    if scores.sum() == 0:
        return []
    idx = np.argsort(-scores)[:top_n]
    feats = np.array(vec.get_feature_names_out())
    phrases = [feats[i] for i in idx if scores[i] > 0]
    # small cleanup
    phrases = [re.sub(r"\s+", " ", p).strip() for p in phrases]
    return phrases[:top_n]
