"""Semantic similarity backends.

Two backends share a common interface:

- ``SbertSimilarity``: sentence-transformers (SBERT) encoder. Strong, recommended
  default; needs ``sentence-transformers`` installed and downloads a model on
  first use.
- ``WordEmbeddingSimilarity``: TF-IDF / SIF weighted averages of pre-trained word
  vectors (Word2Vec / GloVe). Lighter fallback when no transformers available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import preprocess_tokens, split_sentences


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SbertSimilarity:
    """Semantic similarity using a sentence-transformer (SBERT) model."""

    name = "SBERT"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def document_similarity(self, doc_a: str, doc_b: str) -> float:
        from sentence_transformers import util

        embeddings = self.model.encode(
            [doc_a, doc_b], normalize_embeddings=True, show_progress_bar=False
        )
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())

    def sentence_similarities(
        self, doc_a: str, doc_b: str
    ) -> Tuple[List[float], List[float], Dict[str, float]]:
        from sentence_transformers import util

        sentences_a = split_sentences(doc_a)
        sentences_b = split_sentences(doc_b)
        if not sentences_a or not sentences_b:
            return [], [], {}

        emb_a = self.model.encode(
            sentences_a, normalize_embeddings=True, show_progress_bar=False
        )
        emb_b = self.model.encode(
            sentences_b, normalize_embeddings=True, show_progress_bar=False
        )
        matrix = util.cos_sim(emb_a, emb_b).cpu().numpy()

        pair_scores: Dict[str, float] = {}
        for i in range(len(sentences_a)):
            for j in range(len(sentences_b)):
                pair_scores[f"{i}-{j}"] = float(matrix[i, j])

        row_scores = [
            max(pair_scores[f"{i}-{j}"] for j in range(len(sentences_b)))
            for i in range(len(sentences_a))
        ]
        col_scores = [
            max(pair_scores[f"{i}-{j}"] for i in range(len(sentences_a)))
            for j in range(len(sentences_b))
        ]
        return row_scores, col_scores, pair_scores


@dataclass
class WordEmbeddingModel:
    """Thin wrapper around gensim KeyedVectors with lazy loading."""

    path: str
    _kv = None

    @property
    def kv(self):
        if self._kv is None:
            from gensim.models import KeyedVectors

            if self.path.endswith(".kv"):
                self._kv = KeyedVectors.load(self.path)
            elif self.path.endswith(".bin"):
                self._kv = KeyedVectors.load_word2vec_format(self.path, binary=True)
            else:
                self._kv = KeyedVectors.load_word2vec_format(self.path, binary=False)
        return self._kv

    def vector(self, token: str) -> Optional[np.ndarray]:
        try:
            return self.kv[token]
        except KeyError:
            return None

    def counts(self, token: str) -> Optional[int]:
        try:
            return self.kv.get_vecattr(token, "count")
        except Exception:
            return None


class WordEmbeddingSimilarity:
    """Compare documents via SIF-weighted averages of word vectors.

    Uses smooth inverse frequency (SIF) weighting, which counters the bias of
    common words, followed by removal of the dominant principal components of the
    embedding space (the 'all-but-the-top' idea) to reduce spurious similarity.
    """

    name = "Word2Vec/GloVe"

    def __init__(self, model: WordEmbeddingModel, num_pcs: int = 12):
        self.model = model
        self.num_pcs = num_pcs
        self._mu: Optional[np.ndarray] = None
        self._pcs: Optional[np.ndarray] = None
        self._total_counts: Optional[int] = None

    def _stats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        if self._pcs is None:
            kv = self.model.kv
            rng = np.random.RandomState(0)
            idx = rng.choice(len(kv.index_to_key), min(200000, len(kv.index_to_key)), replace=False)
            words = [kv.index_to_key[i] for i in idx]
            m = np.stack([kv[w] for w in words]).astype(np.float64)
            mu = m.mean(axis=0)
            centered = m - mu
            cov = centered.T @ centered / len(centered)
            _, vecs = np.linalg.eigh(cov)
            self._pcs = vecs[:, np.argsort(_.real)[::-1]]
            self._mu = mu
            total = 0
            for w in kv.index_to_key:
                c = self.model.counts(w)
                if c is not None:
                    total += c
            self._total_counts = total if total else 1
        return self._mu, self._pcs, self._total_counts

    def _doc_vector(self, text: str) -> Optional[np.ndarray]:
        tokens = preprocess_tokens(text)
        if not tokens:
            return None
        mu, pcs, total = self._stats()
        a = 1e-3
        vecs: List[np.ndarray] = []
        weights: List[float] = []
        for token in tokens:
            vec = self.model.vector(token)
            if vec is not None:
                vecs.append(vec.astype(np.float64) - mu)
                c = self.model.counts(token)
                p = (c / total) if c else 0.0
                weights.append(a / (a + p))
        if not vecs:
            return None
        arr = np.stack(vecs) * np.array(weights)[:, None]
        v = arr.mean(axis=0)
        for i in range(min(self.num_pcs, pcs.shape[1])):
            v = v - np.dot(v, pcs[:, i]) * pcs[:, i]
        return v

    def document_similarity(self, doc_a: str, doc_b: str) -> float:
        va = self._doc_vector(doc_a)
        vb = self._doc_vector(doc_b)
        if va is None or vb is None:
            return 0.0
        return _cosine(va, vb)

    def sentence_similarities(
        self, doc_a: str, doc_b: str
    ) -> Tuple[List[float], List[float], Dict[str, float]]:
        sentences_a = split_sentences(doc_a)
        sentences_b = split_sentences(doc_b)
        if not sentences_a or not sentences_b:
            return [], [], {}

        vecs_a = [self._doc_vector(s) for s in sentences_a]
        vecs_b = [self._doc_vector(s) for s in sentences_b]

        pair_scores: Dict[str, float] = {}
        for i, va in enumerate(vecs_a):
            if va is None:
                continue
            for j, vb in enumerate(vecs_b):
                if vb is None:
                    continue
                pair_scores[f"{i}-{j}"] = _cosine(va, vb)

        row_scores = [
            max([pair_scores.get(f"{i}-{j}", 0.0) for j in range(len(sentences_b))])
            for i in range(len(sentences_a))
        ]
        col_scores = [
            max([pair_scores.get(f"{i}-{j}", 0.0) for i in range(len(sentences_a))])
            for j in range(len(sentences_b))
        ]
        return row_scores, col_scores, pair_scores


def load_embeddings(path: str) -> WordEmbeddingModel:
    return WordEmbeddingModel(path=path)


def build_semantic(
    backend: str = "sbert",
    model_name: Optional[str] = None,
    path: Optional[str] = None,
):
    """Build a semantic similarity backend.

    Args:
        backend: "sbert" (sentence-transformers) or "word" (Word2Vec/GloVe).
        model_name: SBERT model name (default all-MiniLM-L6-v2).
        path: path to a word-embedding model file for the "word" backend.
    """
    if backend == "sbert":
        return SbertSimilarity(model_name or "all-MiniLM-L6-v2")
    if backend == "word":
        if path is None:
            raise ValueError("path is required for the word-embedding backend")
        return WordEmbeddingSimilarity(load_embeddings(path))
    raise ValueError(f"unknown backend: {backend}")