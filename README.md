# Plagiarism Detection using NLP

A hybrid **document-to-document plagiarism detector** built with Python, Streamlit,
scikit-learn, NLTK and gensim. It combines two complementary signals:

1. **Lexical similarity (TF-IDF + cosine similarity)** – captures exact word overlap.
2. **Semantic similarity** – captures meaning even when the text is paraphrased
   using different words. Two backends are supported:
   - **SBERT** (sentence-transformers, recommended) – a transformer encoder that
     gives state-of-the-art semantic similarity.
   - **Word2Vec / GloVe** – SIF-weighted averages of pre-trained word vectors
     (a lighter, dependency-free option).

The two scores are combined into a single similarity score and mapped to a verdict
(Original / Low / Moderate / High / Severe).

## Features

- Paste or upload two documents (`.txt`, `.md`) and compare them.
- Sample document pairs included for quick demos (identical copy, paraphrase, unrelated).
- Sentence-level match detection with highlighted similar sentence pairs.
- Two combination modes: **Max** (report the strongest signal) and **Weighted blend**.
- Adjustable match threshold; graceful TF-IDF-only fallback if no semantic model loads.

## Project structure

```
plagiarism-detector/
├── app.py                        # Streamlit web app
├── requirements.txt
├── src/
│   ├── preprocessing.py          # cleaning, tokenization, lemmatization
│   ├── tfidf_similarity.py       # TF-IDF + cosine similarity
│   ├── embedding_similarity.py   # SBERT + Word2Vec/GloVe semantic backends
│   └── detector.py               # orchestrates scoring + verdict
├── scripts/
│   └── download_embeddings.py    # downloads a Word2Vec/GloVe model (optional)
├── data/
│   ├── samples/                  # sample documents for testing
│   └── embeddings/               # downloaded word vectors (git-ignored)
└── tests/
    └── test_detector.py          # unit tests
```

## Setup

```bash
pip install -r requirements.txt
```

NLTK data (tokenizer, stopwords, WordNet) is downloaded automatically on first run.

### Semantic models

- **SBERT (default):** the `all-MiniLM-L6-v2` model (~90 MB) downloads automatically
  from Hugging Face on the first run.
- **Word2Vec/GloVe (optional):** download once with

  ```bash
  python scripts/download_embeddings.py
  ```

  (default `glove-wiki-gigaword-50`, ~66 MB; alternative `glove-twitter-25`).

Without any semantic model the app still works in TF-IDF-only mode.

## Run

```bash
streamlit run app.py
```

Open the printed URL (default `http://localhost:8501`).

## Usage

1. Put the **reference** document in "Document A" and the **suspect** document in
   "Document B" (paste or upload, or pick a sample pair from the sidebar).
2. Pick the semantic backend and combination mode in the sidebar if desired.
3. Click **Detect Plagiarism**.

The results show the combined score, the verdict, the individual TF-IDF and
semantic scores, and a list of similar sentence pairs.

## How it works

- **Preprocessing** (`src/preprocessing.py`): lowercase, strip punctuation, remove
  stopwords and short tokens, lemmatize words (verb-first, then noun).
- **TF-IDF** (`src/tfidf_similarity.py`): both documents are vectorized with
  `TfidfVectorizer` and compared with cosine similarity.
- **Semantic** (`src/embedding_similarity.py`):
  - *SBERT* encodes documents/sentences into embeddings and compares them with
    cosine similarity.
  - *Word2Vec/GloVe* represents each text as a SIF-weighted average of word vectors
    with the dominant principal components removed (reduces GloVe's known
    anisotropy/hubness problem).
- **Detector** (`src/detector.py`): combines the two scores with either
  `max(lexical, semantic)` (default) or `tfidf_weight * lexical + (1 - tfidf_weight) * semantic`,
  then thresholds the result into a verdict. Sentence pairs above the match
  threshold are reported.

## Sample results (built-in sample pair 1 vs the variants)

| Document B        | TF-IDF | Semantic (SBERT) | Verdict         |
|-------------------|--------|------------------|-----------------|
| Identical copy    | 0.98   | 1.00             | Severe          |
| Paraphrase        | 0.17   | 0.74             | High            |
| Unrelated topic   | 0.01   | 0.06             | Original        |

## Tests

```bash
python tests/test_detector.py
# or
python -m unittest tests.test_detector -v
```

Tests that need a semantic model are skipped automatically when one is unavailable.