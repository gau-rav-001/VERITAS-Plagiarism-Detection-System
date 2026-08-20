"""Text preprocessing utilities: cleaning, tokenization, stopword removal, lemmatization."""

from __future__ import annotations

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Keep words that are at least this long and purely alphabetical.
_WORD_RE = re.compile(r"^[a-z][a-z']*$")


def clean_text(text: str) -> str:
    """Lowercase text and strip non-alphanumeric noise (keeps hyphens/possessives)."""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize into cleaned, lowercased tokens (no stopword removal, no lemmatization)."""
    cleaned = clean_text(text)
    return word_tokenize(cleaned)


def _lemmatize(token: str) -> str:
    """Lemmatize trying verb POS first, then noun POS."""
    verb = _LEMMATIZER.lemmatize(token, pos="v")
    if verb != token:
        return verb
    return _LEMMATIZER.lemmatize(token)


def preprocess_tokens(text: str) -> List[str]:
    """Full pipeline: clean, tokenize, remove stopwords/short words, lemmatize."""
    tokens: List[str] = []
    for token in tokenize(text):
        if token in _STOPWORDS:
            continue
        if len(token) < 3:
            continue
        tokens.append(_lemmatize(token))
    return tokens


def preprocess_document(text: str) -> str:
    """Return the cleaned text (sentence-preserving) for display purposes."""
    cleaned = clean_text(text)
    return " ".join(cleaned.split())


def split_sentences(text: str) -> List[str]:
    """Split raw text into individual sentences (original casing preserved)."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def preprocess_sentences(text: str) -> List[str]:
    """Return cleaned, joined sentence strings for downstream TF-IDF."""
    sentences = split_sentences(text)
    return [" ".join(preprocess_tokens(s)) for s in sentences]