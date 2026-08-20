"""TF-IDF based lexical similarity."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_tokens, split_sentences


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def document_similarity(doc_a: str, doc_b: str) -> float:
    """Cosine similarity between two documents using TF-IDF vectors."""
    tokens_a = preprocess_tokens(doc_a)
    tokens_b = preprocess_tokens(doc_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=lambda t: t.split(),
        preprocessor=None,
        token_pattern=None,
    )
    matrix = vectorizer.fit_transform([" ".join(tokens_a), " ".join(tokens_b)])
    vecs = matrix.toarray()
    return _cosine(vecs[0], vecs[1])


def sentence_similarities(
    doc_a: str, doc_b: str
) -> Tuple[List[float], List[float], Dict[str, float]]:
    """Compare every sentence in A against every sentence in B.

    Returns (row_scores, col_scores, pair_scores) where row_scores[i] is the best
    similarity for sentence i of A, col_scores[j] for sentence j of B, and
    pair_scores maps "i-j" -> score.
    """
    sentences_a = split_sentences(doc_a)
    sentences_b = split_sentences(doc_b)
    if not sentences_a or not sentences_b:
        return [], [], {}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=lambda t: t.split(),
        preprocessor=None,
        token_pattern=None,
    )
    corpus = sentences_a + sentences_b
    matrix = vectorizer.fit_transform(corpus).toarray()
    n_a = len(sentences_a)

    pair_scores: Dict[str, float] = {}
    for i in range(n_a):
        for j in range(len(sentences_b)):
            pair_scores[f"{i}-{j}"] = _cosine(matrix[i], matrix[n_a + j])

    row_scores = [
        max(pair_scores[f"{i}-{j}"] for j in range(len(sentences_b)))
        for i in range(n_a)
    ]
    col_scores = [
        max(pair_scores[f"{i}-{j}"] for i in range(n_a))
        for j in range(len(sentences_b))
    ]
    return row_scores, col_scores, pair_scores