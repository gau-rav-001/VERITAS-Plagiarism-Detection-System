"""Advanced NLP algorithms for plagiarism detection:

1. N-Gram Jaccard & Containment analysis (Unigrams, Bigrams, Trigrams).
2. Longest Common Subsequence (LCS) ratio for contiguous verbatim copying.
3. WordNet synonym substitution mapping.
4. Part-of-Speech (POS) syntactic structure similarity.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import nltk
from nltk.corpus import wordnet

# Ensure required NLTK resources
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from .preprocessing import preprocess_tokens, tokenize


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Generate n-grams as tuples of tokens."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap_stats(tokens_a: List[str], tokens_b: List[str], n: int = 2) -> Dict[str, float]:
    """Calculate Jaccard and Containment score for n-grams.

    - Jaccard = |A ∩ B| / |A ∪ B|
    - Containment = |A ∩ B| / |B| (What fraction of suspect B comes from reference A)
    """
    ng_a = get_ngrams(tokens_a, n)
    ng_b = get_ngrams(tokens_b, n)

    if not ng_a or not ng_b:
        return {"jaccard": 0.0, "containment": 0.0, "common_count": 0}

    intersection = ng_a.intersection(ng_b)
    union = ng_a.union(ng_b)

    jaccard = len(intersection) / len(union) if union else 0.0
    containment = len(intersection) / len(ng_b) if ng_b else 0.0

    return {
        "jaccard": float(jaccard),
        "containment": float(containment),
        "common_count": len(intersection),
    }


def longest_common_subsequence(tokens_a: List[str], tokens_b: List[str]) -> Tuple[int, float]:
    """Compute the Longest Common Subsequence (LCS) length and ratio between token lists.

    Returns:
        (lcs_length, lcs_ratio) where ratio is normalized by max(len(a), len(b)).
    """
    m, n = len(tokens_a), len(tokens_b)
    if m == 0 or n == 0:
        return 0, 0.0

    # Space-optimized DP
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    lcs_len = dp[n]
    ratio = (2.0 * lcs_len) / (m + n) if (m + n) > 0 else 0.0
    return lcs_len, float(ratio)


def find_synonym_pairs(tokens_a: List[str], tokens_b: List[str]) -> List[Tuple[str, str]]:
    """Find words in B that are synonymous replacements of distinct words in A."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    synonyms_found: List[Tuple[str, str]] = []
    seen = set()

    # Compare non-identical tokens
    diff_b = [w for w in tokens_b if w not in set_a]
    diff_a = [w for w in tokens_a if w not in set_b]

    for wb in diff_b:
        synsets_b = wordnet.synsets(wb)
        if not synsets_b:
            continue
        lemmas_b = {lem.name().lower() for s in synsets_b for lem in s.lemmas()}

        for wa in diff_a:
            if (wa, wb) in seen:
                continue
            if wa in lemmas_b:
                synonyms_found.append((wa, wb))
                seen.add((wa, wb))
                break

    return synonyms_found


def pos_structural_similarity(text_a: str, text_b: str) -> float:
    """Compare Part-of-Speech tag sequence similarity between two texts.

    Helps catch syntactic imitation where words are substituted but grammatical
    templates are identical.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    try:
        tags_a = [tag for _, tag in nltk.pos_tag(tokens_a)]
        tags_b = [tag for _, tag in nltk.pos_tag(tokens_b)]

        _, ratio = longest_common_subsequence(tags_a, tags_b)
        return ratio
    except Exception:
        return 0.0
