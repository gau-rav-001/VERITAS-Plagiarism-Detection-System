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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import preprocess_tokens, split_sentences


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


_SBERT_MODEL_CACHE: Dict[str, Any] = {}
_SBERT_EMBEDDING_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def clear_embedding_cache():
    """Clear in-memory SBERT embedding cache."""
    _SBERT_EMBEDDING_CACHE.clear()


def chunk_sentences(
    sentences: List[str], window_size: int = 3, stride: int = 2
) -> List[str]:
    """Group sentences into sliding windows to capture local discourse without exceeding token limits.

    Args:
        sentences: List of extracted sentence strings.
        window_size: Number of consecutive sentences per chunk (default 3).
        stride: Step size between sliding windows (default 2 for 33% overlap).

    Returns:
        List of windowed sentence chunk strings.
    """
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
    chunks: List[str] = []
    for i in range(0, len(sentences), stride):
        win = sentences[i : i + window_size]
        if win:
            chunks.append(" ".join(win))
        if i + window_size >= len(sentences):
            break
    return chunks


class SbertSimilarity:
    """Semantic similarity using Sentence-BERT with model & embedding caching and window chunking."""

    name = "SBERT"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.model_name in _SBERT_MODEL_CACHE:
                self._model = _SBERT_MODEL_CACHE[self.model_name]
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                _SBERT_MODEL_CACHE[self.model_name] = self._model
        return self._model

    def encode_texts(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True
    ) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors with caching and batching."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, text in enumerate(texts):
            key = (self.model_name, text)
            if key in _SBERT_EMBEDDING_CACHE:
                results[idx] = _SBERT_EMBEDDING_CACHE[key]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            encoded_batches = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            for m_idx, emb in zip(missing_indices, encoded_batches):
                emb_arr = np.asarray(emb, dtype=np.float32)
                _SBERT_EMBEDDING_CACHE[(self.model_name, texts[m_idx])] = emb_arr
                results[m_idx] = emb_arr

        return np.stack([r for r in results if r is not None])

    def document_similarity(self, doc_a: str, doc_b: str) -> float:
        """Compute document similarity handling both short texts and long documents via sentence windows."""
        if not doc_a.strip() or not doc_b.strip():
            return 0.0

        sentences_a = split_sentences(doc_a)
        sentences_b = split_sentences(doc_b)

        # For short snippets (3 or fewer sentences), encode directly
        if len(sentences_a) <= 3 and len(sentences_b) <= 3:
            embeddings = self.encode_texts([doc_a, doc_b])
            return float(np.clip(np.dot(embeddings[0], embeddings[1]), 0.0, 1.0))

        # For long documents: chunk by sentence windows to prevent 256/512 token truncation
        chunks_a = chunk_sentences(sentences_a, window_size=3, stride=2)
        chunks_b = chunk_sentences(sentences_b, window_size=3, stride=2)

        emb_a = self.encode_texts(chunks_a, batch_size=32)
        emb_b = self.encode_texts(chunks_b, batch_size=32)

        if len(emb_a) == 0 or len(emb_b) == 0:
            return 0.0

        # Matrix of chunk similarities (num_chunks_a, num_chunks_b)
        sim_mat = emb_a @ emb_b.T

        # Mean pooled normalized document embeddings
        mean_a = np.mean(emb_a, axis=0)
        norm_a = np.linalg.norm(mean_a)
        if norm_a > 0:
            mean_a = mean_a / norm_a

        mean_b = np.mean(emb_b, axis=0)
        norm_b = np.linalg.norm(mean_b)
        if norm_b > 0:
            mean_b = mean_b / norm_b

        pooled_sim = float(np.dot(mean_a, mean_b))

        # Asymmetric suspect chunk coverage (what fraction of suspect chunks find strong alignment)
        max_suspect_sims = np.max(sim_mat, axis=0)
        suspect_coverage = float(np.mean(max_suspect_sims))

        # Max coverage allows catching partial plagiarism in long documents
        score = max(pooled_sim, suspect_coverage)
        return float(np.clip(score, 0.0, 1.0))

    def sentence_similarities(
        self, doc_a: str, doc_b: str
    ) -> Tuple[List[float], List[float], Dict[str, float]]:
        """Compute pairwise sentence similarity matrix with cached batch encoding."""
        sentences_a = split_sentences(doc_a)
        sentences_b = split_sentences(doc_b)
        if not sentences_a or not sentences_b:
            return [], [], {}

        emb_a = self.encode_texts(sentences_a, batch_size=32)
        emb_b = self.encode_texts(sentences_b, batch_size=32)
        matrix = emb_a @ emb_b.T

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
        if self._kv is None:
            try:
                from gensim.models import KeyedVectors
            except ImportError as err:
                raise ImportError(
                    "gensim is required for Word2Vec/GloVe KeyedVectors. "
                    "Install gensim or select SBERT (Sentence-BERT) as the similarity backend."
                ) from err

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