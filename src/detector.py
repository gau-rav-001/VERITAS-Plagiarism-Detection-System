"""Orchestrates TF-IDF and embedding-based similarity into a single verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .embedding_similarity import WordEmbeddingSimilarity  # noqa: F401
from .preprocessing import split_sentences
from .tfidf_similarity import document_similarity as tfidf_document_similarity
from .tfidf_similarity import sentence_similarities as tfidf_sentence_similarities

# Similarity thresholds for the verdict.
VERDICT_LEVELS = [
    (0.85, "Severe Plagiarism"),
    (0.70, "High Plagiarism"),
    (0.50, "Moderate Similarity"),
    (0.30, "Low Similarity"),
    (0.00, "Original"),
]


@dataclass
class SentenceMatch:
    """A pair of sentences (one from A, one from B) flagged as similar."""

    index_a: int
    index_b: int
    score: float
    sentence_a: str
    sentence_b: str


@dataclass
class DetectionResult:
    tfidf_score: float
    semantic_score: float
    combined_score: float
    verdict: str
    verdict_threshold: float
    tfidf_weight: float
    sentence_matches: List[SentenceMatch] = field(default_factory=list)
    matched_rows: List[int] = field(default_factory=list)
    matched_cols: List[int] = field(default_factory=list)


def detect(
    doc_a: str,
    doc_b: str,
    embedding=None,
    tfidf_weight: float = 0.6,
    match_threshold: float = 0.75,
    combination: str = "max",
) -> DetectionResult:
    """Compare two documents and produce a combined plagiarism verdict.

    Args:
        doc_a: First document (typically the original / reference).
        doc_b: Second document (typically the submission / suspect).
        embedding: Optional semantic backend (SbertSimilarity /
            WordEmbeddingSimilarity).
        tfidf_weight: Weight given to the lexical TF-IDF score (0..1); the
            remainder is given to the semantic score. Only used when
            ``combination="weighted"``.
        match_threshold: Minimum similarity for a sentence pair to be flagged.
        combination: "max" reports the strongest signal of the two methods;
            "weighted" blends them with ``tfidf_weight``.
    """
    tfidf_score = tfidf_document_similarity(doc_a, doc_b)

    semantic_score = 0.0
    if embedding is not None:
        semantic_score = embedding.document_similarity(doc_a, doc_b)

    # Combine the two signals.
    if embedding is None:
        combined = tfidf_score
    elif combination == "max":
        combined = max(tfidf_score, semantic_score)
    elif combination == "weighted":
        combined = tfidf_weight * tfidf_score + (1.0 - tfidf_weight) * semantic_score
    else:
        raise ValueError(f"unknown combination: {combination}")

    threshold, verdict = next(
        (t, v) for t, v in VERDICT_LEVELS if combined >= t
    )

    sentence_matches: List[SentenceMatch] = []
    sentences_a = split_sentences(doc_a)
    sentences_b = split_sentences(doc_b)

    if sentences_a and sentences_b:
        # Merge lexical and semantic pairwise scores.
        _, _, pair_tfidf = tfidf_sentence_similarities(doc_a, doc_b)
        pair_semantic: Dict[str, float] = {}
        if embedding is not None:
            _, _, pair_semantic = embedding.sentence_similarities(doc_a, doc_b)

        pair_scores: Dict[str, float] = {}
        for i in range(len(sentences_a)):
            for j in range(len(sentences_b)):
                key = f"{i}-{j}"
                s_t = pair_tfidf.get(key, 0.0)
                if embedding is not None and combination == "max":
                    pair_scores[key] = max(s_t, pair_semantic.get(key, 0.0))
                elif embedding is not None:
                    pair_scores[key] = (
                        tfidf_weight * s_t + (1.0 - tfidf_weight) * pair_semantic.get(key, 0.0)
                    )
                else:
                    pair_scores[key] = s_t

        for i in range(len(sentences_a)):
            for j in range(len(sentences_b)):
                score = pair_scores[f"{i}-{j}"]
                if score >= match_threshold:
                    sentence_matches.append(
                        SentenceMatch(
                            index_a=i,
                            index_b=j,
                            score=score,
                            sentence_a=sentences_a[i],
                            sentence_b=sentences_b[j],
                        )
                    )

        sentence_matches.sort(key=lambda m: m.score, reverse=True)
        matched_rows = sorted({m.index_a for m in sentence_matches})
        matched_cols = sorted({m.index_b for m in sentence_matches})
    else:
        matched_rows, matched_cols = [], []

    return DetectionResult(
        tfidf_score=tfidf_score,
        semantic_score=semantic_score,
        combined_score=combined,
        verdict=verdict,
        verdict_threshold=threshold,
        tfidf_weight=tfidf_weight,
        sentence_matches=sentence_matches,
        matched_rows=matched_rows,
        matched_cols=matched_cols,
    )