"""Orchestrates TF-IDF, N-grams, LCS, and embedding-based similarity into a single verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .embedding_similarity import WordEmbeddingSimilarity  # noqa: F401
from .nlp_analysis import (
    find_synonym_pairs,
    longest_common_subsequence,
    ngram_overlap_stats,
    pos_structural_similarity,
)
from .preprocessing import preprocess_tokens, split_sentences, tokenize
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
    match_type: str = "Verbatim"
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    synonyms: List[Tuple[str, str]] = field(default_factory=list)
    source_label: str = ""
    confidence: str = "High"


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
    sentences_a: List[str] = field(default_factory=list)
    sentences_b: List[str] = field(default_factory=list)
    similarity_matrix: List[List[float]] = field(default_factory=list)
    bigram_containment: float = 0.0
    trigram_containment: float = 0.0
    lcs_ratio: float = 0.0
    pos_similarity: float = 0.0
    verbatim_count: int = 0
    paraphrase_count: int = 0
    original_count: int = 0
    calibrated_score: float = 0.0


def calculate_calibrated_score(
    tfidf_score: float,
    bigram_containment: float,
    trigram_containment: float,
    lcs_ratio: float,
    pos_similarity: float,
    semantic_score: Optional[float] = None,
) -> float:
    """Calculate the calibrated final plagiarism score.

    Documented Mathematical Weights & Rationale:
    --------------------------------------------
    1. Lexical Composite Sub-Score (S_lex):
       - TF-IDF unigram similarity (weight 0.35): Captures topical and domain-specific vocabulary overlap.
       - Bigram containment (weight 0.25): Measures 2-word phrase retention.
       - Trigram containment (weight 0.15): Measures verbatim phrase retention (Turnitin metric).
       - Longest Common Subsequence (LCS) ratio (weight 0.25): Measures continuous ordered word sequence copying.
       Formula: S_lex = 0.35 * tfidf + 0.25 * bigram + 0.15 * trigram + 0.25 * lcs

    2. Syntactic POS Alignment Signal (S_pos):
       - Natural English grammar baseline is ~0.45 across arbitrary unrelated documents due to subject-verb-object syntax.
       - Normalized Signal: S_pos = max(0.0, min(1.0, (pos_similarity - 0.45) / 0.55)).

    3. Multi-Engine Fusion:
       - When SBERT semantic embedding score (S_sem) is available:
         * Balanced multi-modal blend: 0.40 * S_lex + 0.50 * S_sem + 0.10 * S_pos
         * Adaptive verbatim dominance: If lexical copying is high, verbatim evidence dominates:
           0.80 * S_lex + 0.20 * S_sem
         * Adaptive paraphrase dominance: If semantic similarity is high but lexical words were swapped:
           0.80 * S_sem + 0.20 * S_lex
         * Final score = max(blend, dom_lex, dom_sem)
       - When semantic embedding is unavailable (pure lexical mode):
         * 0.90 * S_lex + 0.10 * S_pos

    Returns:
        Calibrated score bounded in [0.0, 1.0].
    """
    s_lex = (
        0.35 * tfidf_score
        + 0.25 * bigram_containment
        + 0.15 * trigram_containment
        + 0.25 * lcs_ratio
    )
    pos_signal = max(0.0, min(1.0, (pos_similarity - 0.45) / 0.55))

    if semantic_score is not None:
        blend = 0.40 * s_lex + 0.50 * semantic_score + 0.10 * pos_signal
        dom_lex = 0.80 * s_lex + 0.20 * semantic_score
        dom_sem = 0.80 * semantic_score + 0.20 * s_lex
        score = max(blend, dom_lex, dom_sem)
    else:
        score = 0.90 * s_lex + 0.10 * pos_signal

    return float(max(0.0, min(1.0, score)))


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
        combination: "max" / "calibrated" reports calibrated multi-engine fusion;
            "weighted" blends them with ``tfidf_weight``.
    """
    # Guard against empty/whitespace inputs
    if not doc_a.strip() or not doc_b.strip():
        threshold, verdict = VERDICT_LEVELS[-1]
        return DetectionResult(
            tfidf_score=0.0,
            semantic_score=0.0,
            combined_score=0.0,
            verdict=verdict,
            verdict_threshold=threshold,
            tfidf_weight=tfidf_weight,
            sentences_a=split_sentences(doc_a) if doc_a.strip() else [],
            sentences_b=split_sentences(doc_b) if doc_b.strip() else [],
            sentence_matches=[],
            matched_rows=[],
            matched_cols=[],
            similarity_matrix=[],
            bigram_containment=0.0,
            trigram_containment=0.0,
            lcs_ratio=0.0,
            pos_similarity=0.0,
            verbatim_count=0,
            paraphrase_count=0,
            original_count=0,
            calibrated_score=0.0,
        )

    tfidf_score = tfidf_document_similarity(doc_a, doc_b)

    semantic_score = 0.0
    if embedding is not None:
        semantic_score = embedding.document_similarity(doc_a, doc_b)

    sentences_a = split_sentences(doc_a)
    sentences_b = split_sentences(doc_b)

    tokens_a = preprocess_tokens(doc_a)
    tokens_b = preprocess_tokens(doc_b)

    # Global N-gram and LCS statistics
    bigram_stats = ngram_overlap_stats(tokens_a, tokens_b, n=2)
    trigram_stats = ngram_overlap_stats(tokens_a, tokens_b, n=3)
    _, lcs_ratio = longest_common_subsequence(tokens_a, tokens_b)
    pos_sim = pos_structural_similarity(doc_a, doc_b)

    calibrated = calculate_calibrated_score(
        tfidf_score=tfidf_score,
        bigram_containment=bigram_stats["containment"],
        trigram_containment=trigram_stats["containment"],
        lcs_ratio=lcs_ratio,
        pos_similarity=pos_sim,
        semantic_score=semantic_score if embedding is not None else None,
    )

    # Combine the signals
    if combination in ("max", "calibrated"):
        combined = calibrated
    elif combination == "weighted":
        combined = tfidf_weight * tfidf_score + (1.0 - tfidf_weight) * semantic_score
    else:
        raise ValueError(f"unknown combination: {combination}")

    threshold, verdict = next(
        (t, v) for t, v in VERDICT_LEVELS if combined >= t
    )

    sentence_matches: List[SentenceMatch] = []
    matrix_scores: List[List[float]] = []

    if sentences_a and sentences_b:
        # Merge lexical and semantic pairwise scores.
        _, _, pair_tfidf = tfidf_sentence_similarities(doc_a, doc_b)
        pair_semantic: Dict[str, float] = {}
        if embedding is not None:
            _, _, pair_semantic = embedding.sentence_similarities(doc_a, doc_b)

        pair_scores: Dict[str, float] = {}
        for i in range(len(sentences_a)):
            row = []
            for j in range(len(sentences_b)):
                key = f"{i}-{j}"
                s_t = pair_tfidf.get(key, 0.0)
                s_sem = pair_semantic.get(key, 0.0)

                if embedding is not None and combination in ("max", "calibrated"):
                    score = max(s_t, s_sem)
                elif embedding is not None:
                    score = tfidf_weight * s_t + (1.0 - tfidf_weight) * s_sem
                else:
                    score = s_t

                pair_scores[key] = score
                row.append(round(score, 3))
            matrix_scores.append(row)

        for i in range(len(sentences_a)):
            for j in range(len(sentences_b)):
                key = f"{i}-{j}"
                score = pair_scores[key]
                if score >= match_threshold:
                    s_t = pair_tfidf.get(key, 0.0)
                    s_sem = pair_semantic.get(key, 0.0)

                    tok_a = preprocess_tokens(sentences_a[i])
                    tok_b = preprocess_tokens(sentences_b[j])
                    syns = find_synonym_pairs(tok_a, tok_b)

                    # Classify match type
                    if s_t >= 0.80:
                        m_type = "Verbatim Match"
                    elif s_sem >= 0.70:
                        m_type = "Paraphrased Match"
                    else:
                        m_type = "Partial Match"

                    # Calculate confidence level
                    if score >= 0.85:
                        conf = "High"
                    elif score >= 0.70:
                        conf = "Moderate"
                    else:
                        conf = "Tentative"

                    sentence_matches.append(
                        SentenceMatch(
                            index_a=i,
                            index_b=j,
                            score=score,
                            sentence_a=sentences_a[i],
                            sentence_b=sentences_b[j],
                            match_type=m_type,
                            lexical_score=s_t,
                            semantic_score=s_sem,
                            synonyms=syns,
                            source_label=f"Ref Sentence #{i + 1}",
                            confidence=conf,
                        )
                    )

        sentence_matches.sort(key=lambda m: m.score, reverse=True)
        matched_rows = sorted({m.index_a for m in sentence_matches})
        matched_cols = sorted({m.index_b for m in sentence_matches})
    else:
        matched_rows, matched_cols = [], []

    verbatim_count = sum(1 for m in sentence_matches if m.match_type == "Verbatim Match")
    paraphrase_count = sum(1 for m in sentence_matches if m.match_type == "Paraphrased Match")
    matched_suspect_sentences = len(matched_cols)
    total_suspect_sentences = len(sentences_b)
    original_count = max(0, total_suspect_sentences - matched_suspect_sentences)

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
        sentences_a=sentences_a,
        sentences_b=sentences_b,
        similarity_matrix=matrix_scores,
        bigram_containment=bigram_stats["containment"],
        trigram_containment=trigram_stats["containment"],
        lcs_ratio=lcs_ratio,
        pos_similarity=pos_sim,
        verbatim_count=verbatim_count,
        paraphrase_count=paraphrase_count,
        original_count=original_count,
        calibrated_score=calibrated,
    )