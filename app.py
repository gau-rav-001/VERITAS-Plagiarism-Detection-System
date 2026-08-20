"""Plagiarism Detector - Streamlit web app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import glob
import os

import streamlit as st

from src.detector import detect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")

VERDICT_COLORS = {
    "Severe Plagiarism": "#d32f2f",
    "High Plagiarism": "#f57c00",
    "Moderate Similarity": "#fbc02d",
    "Low Similarity": "#7cb342",
    "Original": "#388e3c",
}


def available_embeddings() -> list[str]:
    patterns = ["*.kv", "*.bin", "*.txt", "*.vec"]
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(EMBEDDINGS_DIR, pattern)))
    return sorted(found)


@st.cache_resource(show_spinner="Loading semantic model...")
def load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    from src.embedding_similarity import build_semantic

    return build_semantic(backend="sbert", model_name=model_name)


@st.cache_resource(show_spinner="Loading word embeddings...")
def load_word_embeddings(path: str):
    from src.embedding_similarity import build_semantic

    return build_semantic(backend="word", path=path)


def sample_names() -> list[str]:
    names = glob.glob(os.path.join(SAMPLES_DIR, "*.txt"))
    return [os.path.basename(n) for n in sorted(names)]


def read_sample(name: str) -> str:
    with open(os.path.join(SAMPLES_DIR, name), encoding="utf-8") as fh:
        return fh.read()


st.set_page_config(page_title="Plagiarism Detector", layout="wide")

st.title("Plagiarism Detection using NLP")
st.caption(
    "Hybrid lexical (TF-IDF) + semantic (word embeddings) similarity engine "
    "for document-to-document comparison."
)

embedding_files = available_embeddings()
with st.sidebar:
    st.header("Settings")

    sbert_available = True
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        sbert_available = False

    backend = st.selectbox(
        "Semantic model",
        options=["SBERT (recommended)", "Word2Vec/GloVe (baseline)", "None (TF-IDF only)"],
        help="SBERT is a transformer encoder with excellent accuracy. The "
        "Word2Vec/GloVe baseline uses averaged word vectors; it is lighter but "
        "over-estimates similarity between unrelated topics. Select 'None' for "
        "pure lexical detection.",
        disabled=not sbert_available,
    )
    if not sbert_available:
        backend = "Word2Vec/GloVe" if embedding_files else "None (TF-IDF only)"

    combination = st.selectbox(
        "Combine signals",
        options=["Max (strongest signal)", "Weighted blend"],
        help="Max reports the higher of the lexical and semantic scores. "
        "Weighted blends them with the weight below.",
    )
    tfidf_weight = 0.6
    if combination == "Weighted blend":
        tfidf_weight = st.slider(
            "TF-IDF weight (rest = semantic)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
    match_threshold = st.slider(
        "Sentence match threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )

    st.divider()
    st.markdown("**Load a sample pair:**")
    sample = st.selectbox("Sample", ["(none)"] + sample_names())
    if sample != "(none)":
        st.session_state["doc_a"] = read_sample(sample)

embedding = None
if backend == "SBERT (recommended)":
    embedding = load_sbert()
    st.sidebar.success("SBERT model ready")
elif backend == "Word2Vec/GloVe (baseline)":
    if not embedding_files:
        st.sidebar.error(
            "No word-embedding model found. Run "
            "`python scripts/download_embeddings.py` once."
        )
    else:
        embedding = load_word_embeddings(embedding_files[0])
        st.sidebar.success(f"Embeddings ready: {os.path.basename(embedding_files[0])}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Document A (reference)")
    doc_a = st.text_area(
        "Document A",
        value=st.session_state.get("doc_a", ""),
        height=300,
        label_visibility="collapsed",
        placeholder="Paste the original / reference document here...",
    )
    uploaded_a = st.file_uploader("or upload", type=["txt", "md"], key="upload_a")
    if uploaded_a is not None:
        doc_a = uploaded_a.read().decode("utf-8", errors="ignore")
        st.session_state["doc_a"] = doc_a

with col2:
    st.subheader("Document B (suspect)")
    doc_b = st.text_area(
        "Document B",
        value=st.session_state.get("doc_b", ""),
        height=300,
        label_visibility="collapsed",
        placeholder="Paste the submission / suspect document here...",
    )
    uploaded_b = st.file_uploader("or upload", type=["txt", "md"], key="upload_b")
    if uploaded_b is not None:
        doc_b = uploaded_b.read().decode("utf-8", errors="ignore")
        st.session_state["doc_b"] = doc_b

if st.button("Detect Plagiarism", type="primary"):
    if not doc_a.strip() or not doc_b.strip():
        st.error("Please provide both documents.")
        st.stop()

    result = detect(
        doc_a,
        doc_b,
        embedding=embedding,
        tfidf_weight=tfidf_weight,
        match_threshold=match_threshold,
        combination="max" if combination == "Max (strongest signal)" else "weighted",
    )

    st.markdown("---")
    st.subheader("Results")

    score_col, verdict_col = st.columns([1, 1])
    with score_col:
        st.metric("Combined similarity score", f"{result.combined_score * 100:.1f}%")
    with verdict_col:
        color = VERDICT_COLORS.get(result.verdict, "#333")
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background:{color};"
            f"color:#fff;text-align:center;font-size:20px;font-weight:bold;'>"
            f"{result.verdict}</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Lexical (TF-IDF)", f"{result.tfidf_score * 100:.1f}%")
    c2.metric(
        "Semantic",
        f"{result.semantic_score * 100:.1f}%",
        help="0.0% if no semantic model is loaded",
    )
    c3.metric("Flagged sentence pairs", f"{len(result.sentence_matches)}")

    if result.sentence_matches:
        st.subheader("Similar sentence pairs")
        for match in result.sentence_matches[:10]:
            st.markdown(
                f"**{match.score * 100:.0f}% match**  "
                f"(Doc A # {match.index_a + 1} vs Doc B # {match.index_b + 1})"
            )
            with st.container(border=True):
                st.markdown(f"A: {match.sentence_a}")
                st.markdown(f"B: {match.sentence_b}")
    else:
        st.info("No sentence pairs exceeded the match threshold.")
else:
    st.info("Paste two documents and click **Detect Plagiarism**.")

st.markdown("---")
st.markdown(
    "**How it works:** text is cleaned, stopwords removed and words lemmatized. "
    "A TF-IDF vectorizer captures lexical overlap, while a semantic encoder "
    "(SBERT, or Word2Vec/GloVe) captures meaning even across paraphrases. "
    "The combined score maps to a verdict."
)