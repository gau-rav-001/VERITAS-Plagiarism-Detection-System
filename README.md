# NLP Plagiarism Detection System

A multi-signal **plagiarism verification and detection engine** built with Python, Streamlit, scikit-learn, NLTK, Sentence-BERT, and Plotly for academic document comparison and repository auditing.

It combines multi-tiered lexical analysis and deep semantic representations:

1. **Lexical Signals**:
   - **TF-IDF & Cosine Similarity**: Captures vocabulary overlap and document-level distribution.
   - **N-Gram Jaccard & Containment**: Evaluates Bi-gram and Tri-gram phrase containment (Turnitin-style).
   - **Longest Common Subsequence (LCS)**: Measures contiguous verbatim token sequence matches.
2. **Syntactic & Paraphrase Signals**:
   - **Part-of-Speech (POS) Syntactic Alignment**: Detects grammatical structure copying where sentences follow identical templates.
   - **WordNet Synonym Substitution Mapping**: Identifies synonym-swapped words between matched sentence pairs.
3. **Deep Semantic Signals**:
   - **SBERT (Sentence-BERT - `all-MiniLM-L6-v2`)**: Contextual transformer encoder that flags meaning-preserving paraphrases.
   - **Word2Vec / GloVe**: SIF-weighted word vector averaging with principal component removal.

The signals are fused into an overall similarity score mapped to a standardized verdict (**Original / Low / Moderate / High / Severe**).

---

## ✨ Features

- **Multi-Format Document Ingestion**: Upload `.txt`, `.md`, `.pdf`, and `.docx` (Word) documents directly.
- **Side-by-Side Synchronized Text Highlighting**: Turnitin-style visual comparison:
  - 🔴 **Red**: Verbatim exact match ($> 80\%$ lexical similarity).
  - 🟡 **Yellow**: Paraphrased match captured by semantic embeddings.
- **Interactive Similarity Heatmap**: Plotly interactive matrix mapping Document A sentences against Document B sentences with hover previews.
- **Visual Analytics**: Interactive gauge meter for overall risk level and donut chart showing the composition of Verbatim vs. Paraphrased vs. Original content.
- **Corpus / Repository Scanner (1-to-Many Mode)**: Scan a student submission against an entire folder or uploaded archive of past assignments/papers to automatically pinpoint the original source.
- **Exportable Plagiarism Verification Report**: Generate and download a standalone `.html` report with full metadata, verdict badges, and matched evidence tables.

---

## 📁 Project Structure

```
NLP-Plagiarism-Detection-System/
├── app.py                     # Streamlit web application with Plotly analytics
├── requirements.txt           # Environment dependencies
├── README.md                  # Project documentation
├── src/
│   ├── preprocessing.py       # Cleaning, tokenization, lemmatization, sentence splitter
│   ├── tfidf_similarity.py    # TF-IDF vectorization & cosine similarity
│   ├── nlp_analysis.py        # N-grams, LCS, WordNet synonyms, POS syntactic matching
│   ├── embedding_similarity.py# SBERT & Word2Vec/GloVe semantic backends
│   ├── detector.py            # Signal fusion, verdict mapping, sentence alignment
│   ├── file_reader.py         # Multi-format document parser (.pdf, .docx, .txt, .md)
│   └── report_generator.py    # Self-contained HTML inspection report generator
├── scripts/
│   └── download_embeddings.py # Pre-trained GloVe vector downloader (optional)
├── data/
│   ├── samples/               # Test document sample pairs
│   └── embeddings/            # Downloaded word vectors (git-ignored)
└── tests/
    └── test_detector.py       # Comprehensive unit test suite (17 tests)
```

---

## 🚀 Setup & Execution

### 1. Activate Virtual Environment
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
Open the browser URL (default: `http://localhost:8501`).

---

## 🧪 Testing

Run the full test suite with:

```bash
python -m unittest tests.test_detector -v
```

---

## 📊 Sample Results

| Suspect Document Type | TF-IDF | Semantic (SBERT) | N-Gram Containment | Verdict |
|---|---|---|---|---|
| **Identical copy** | 0.98 | 1.00 | 95%+ | **Severe Plagiarism** |
| **Paraphrased text** | 0.17 | 0.74 | 20% | **High Plagiarism** |
| **Unrelated topic** | 0.01 | 0.06 | 0% | **Original** |