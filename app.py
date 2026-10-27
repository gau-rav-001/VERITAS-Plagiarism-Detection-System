"""VERITAS • Hybrid Lexical & Semantic NLP Plagiarism Detection System.

A streamlined, human-designed single-page web app featuring:
- Non-blocking 3D Interactive Neon Tubes Cursor Background (WebGL / threejs-components)
- Minimalist Dark Slate Design System with WCAG AA compliant contrast
- Lexical Engine (TF-IDF, N-Grams, LCS, POS) + Semantic Engine (Sentence-BERT, WordNet)
- Turnitin-Style Dual Highlighting, Plotly Analytics, Multi-Format Ingestion (PDF, DOCX, TXT, MD)
- Session state persistence (results survive reruns)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import glob
import os
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from src.detector import detect, DetectionResult
from src.file_reader import extract_text_from_file
from src.report_generator import generate_html_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")

VERDICT_PALETTE = {
    "Severe Plagiarism": {"bg": "rgba(220, 38, 38, 0.16)", "border": "#ef4444", "text": "#fca5a5", "badge": "#dc2626"},
    "High Plagiarism": {"bg": "rgba(234, 88, 12, 0.16)", "border": "#f97316", "text": "#fdba74", "badge": "#ea580c"},
    "Moderate Similarity": {"bg": "rgba(217, 119, 6, 0.16)", "border": "#f59e0b", "text": "#fde047", "badge": "#d97706"},
    "Low Similarity": {"bg": "rgba(22, 163, 74, 0.16)", "border": "#22c55e", "text": "#86efac", "badge": "#16a34a"},
    "Original": {"bg": "rgba(37, 99, 235, 0.16)", "border": "#3b82f6", "text": "#93c5fd", "badge": "#2563eb"},
}


def available_embeddings() -> list[str]:
    patterns = ["*.kv", "*.bin", "*.txt", "*.vec"]
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(EMBEDDINGS_DIR, pattern)))
    return sorted(found)


@st.cache_resource(show_spinner="Loading SBERT Transformer model (all-MiniLM-L6-v2)...")
def load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    from src.embedding_similarity import build_semantic

    return build_semantic(backend="sbert", model_name=model_name)


@st.cache_resource(show_spinner="Loading word embeddings...")
def load_word_embeddings(path: str):
    from src.embedding_similarity import build_semantic

    return build_semantic(backend="word", path=path)


@st.cache_data(show_spinner=False)
def get_cached_sbert_similarity(doc_a: str, doc_b: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Cache pairwise document SBERT similarity calculations for rapid re-runs."""
    model = load_sbert(model_name)
    return model.document_similarity(doc_a, doc_b)


def sample_names() -> list[str]:
    names = glob.glob(os.path.join(SAMPLES_DIR, "*.txt"))
    return [os.path.basename(n) for n in sorted(names)]


def read_sample(name: str) -> str:
    with open(os.path.join(SAMPLES_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def safe_extract_text(uploaded_file) -> tuple[str, str | None]:
    """Safely extracts text from uploaded file, returning (text, error_message)."""
    if uploaded_file is None:
        return "", None
    try:
        text = extract_text_from_file(uploaded_file, uploaded_file.name)
        if not text or not text.strip():
            return "", f"File '{uploaded_file.name}' was uploaded but contains no readable text."
        return text, None
    except Exception as err:
        return "", f"Error parsing '{uploaded_file.name}': {err}"


# ---------------------------------------------------------
# Page Configuration & Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="VERITAS • Plagiarism Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined, human-designed CSS (restrained glow, WCAG AA contrast, consistent border-radius)
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient Slate Canvas with subtle transparency for WebGL tubes */
    .stApp {
        background-color: transparent !important;
        background:
            radial-gradient(at 10% 10%, rgba(14, 165, 233, 0.05) 0px, transparent 40%),
            radial-gradient(at 90% 15%, rgba(99, 102, 241, 0.06) 0px, transparent 40%),
            #0a0e17 !important;
        color: #f8fafc;
    }

    /* Hide 0-height iframe container */
    div:has(> iframe[height="0"]) {
        height: 0px !important;
        min-height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        position: absolute;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(13, 18, 30, 0.88) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Header Nav */
    .nav-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 20px;
    }

    .brand-logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
    }

    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: #ffffff;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #38bdf8;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }

    .hero-title-clean {
        font-size: clamp(2.2rem, 5vw, 3.2rem);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 6px 0 8px 0;
    }

    .hero-title-gradient {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle-clean {
        font-size: 1.05rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 8px;
    }

    .hero-desc-clean {
        color: #94a3b8;
        font-size: 0.95rem;
        max-width: 720px;
        margin: 0 auto 16px auto;
        line-height: 1.55;
    }

    .interactive-hint-clean {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #94a3b8;
        font-size: 11.5px;
        font-weight: 500;
        margin-bottom: 20px;
    }

    /* Cards with solid dark backdrop for high contrast */
    .glass-card {
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.16);
    }

    .stitch-divider {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 24px 0;
    }

    /* Metric Box */
    .glass-metric {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 12px;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .glass-metric:hover {
        border-color: rgba(14, 165, 233, 0.35);
    }

    .glass-metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .glass-metric-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Document Highlighting Containers with WCAG AA Contrast */
    .doc-preview-glass {
        background: rgba(10, 14, 24, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 18px;
        height: 380px;
        overflow-y: auto;
        font-size: 14px;
        line-height: 1.7;
        color: #e2e8f0;
        word-wrap: break-word;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.35);
    }

    .highlight-verbatim {
        background: rgba(239, 68, 68, 0.18);
        border-bottom: 2px solid #ef4444;
        color: #fca5a5;
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 4px;
        display: inline;
    }

    .highlight-paraphrase {
        background: rgba(245, 158, 11, 0.18);
        border-bottom: 2px solid #f59e0b;
        color: #fde047;
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 4px;
        display: inline;
    }

    /* Highlight Legend Bar */
    .legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px 16px;
        margin-bottom: 14px;
        font-size: 12.5px;
        color: #cbd5e1;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .legend-badge-verbatim {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        background: #ef4444;
    }
    .legend-badge-paraphrase {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        background: #f59e0b;
    }
    .legend-badge-original {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        background: #475569;
    }

    /* Tabs Styling - Clean & Minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.7);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13.5px;
        padding: 9px 20px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.04);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.12) !important;
        border-bottom: 2px solid #38bdf8 !important;
        color: #ffffff !important;
    }

    /* Clean Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.2) !important;
    }
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Empty state card */
    .empty-state-card {
        text-align: center;
        padding: 40px 20px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        margin: 20px 0;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(8, 12, 24, 0.6);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.5);
    }
    </style>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Global 3D Interactive Tubes Background (WebGL + Three.js)
# Subtle, non-blocking background layer (opacity 0.38)
# ---------------------------------------------------------
components.html(
    """<script type="module">
      const pWin = window.parent;
      const pDoc = pWin.document;

      // 1. Ensure global canvas element in parent body
      let canvas = pDoc.getElementById('global-tubes-canvas');
      if (!canvas) {
        canvas = pDoc.createElement('canvas');
        canvas.id = 'global-tubes-canvas';
        canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;opacity:0.38;display:block;';
        pDoc.body.prepend(canvas);
      }

      // 2. Inject transparent canvas rules into parent document
      if (!pDoc.getElementById('veritas-tubes-fix-css')) {
        const style = pDoc.createElement('style');
        style.id = 'veritas-tubes-fix-css';
        style.textContent = `
          #global-tubes-canvas {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            pointer-events: none !important;
            z-index: 0 !important;
            display: block !important;
            opacity: 0.38 !important;
          }
          .stApp {
            background-color: transparent !important;
          }
        `;
        pDoc.head.appendChild(style);
      }

      // 3. Load TubesCursor with non-blocking event forwarding
      if (!pWin._veritasTubesLoaded) {
        pWin._veritasTubesLoaded = true;

        import('https://cdn.jsdelivr.net/npm/threejs-components@0.0.19/build/cursors/tubes1.min.js')
          .then(mod => {
            const TubesCursor = mod.default || mod;
            const app = TubesCursor(canvas, {
              tubes: {
                colors: ["#0284c7", "#6366f1", "#ec4899"],
                lights: {
                  intensity: 180,
                  colors: ["#38bdf8", "#818cf8", "#f43f5e", "#f59e0b"]
                }
              }
            });
            pWin._veritasTubesApp = app;

            // Forward pointer movements to canvas without interfering with form inputs
            const forwardPointer = (e) => {
              try {
                const evt = new PointerEvent(e.type, {
                  clientX: e.clientX,
                  clientY: e.clientY,
                  screenX: e.screenX,
                  screenY: e.screenY,
                  pageX: e.pageX,
                  pageY: e.pageY,
                  bubbles: true,
                  cancelable: true,
                  pointerType: e.pointerType || 'mouse'
                });
                document.body.dispatchEvent(evt);
              } catch(err) {}
            };

            pDoc.body.addEventListener('pointermove', forwardPointer, { passive: true });
            pDoc.body.addEventListener('mousemove', forwardPointer, { passive: true });

            pWin.addEventListener('resize', () => {
              window.dispatchEvent(new Event('resize'));
            });

            // Click canvas background to randomize colors
            const randCol = () => '#' + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0');
            pDoc.body.addEventListener('click', (e) => {
              const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : '';
              if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'BUTTON' && !e.target.closest('button')) {
                if (app && app.tubes) {
                  app.tubes.setColors([randCol(), randCol(), randCol()]);
                  app.tubes.setLightsColors([randCol(), randCol(), randCol(), randCol()]);
                }
              }
            });

            forwardPointer({ type: 'pointermove', clientX: pWin.innerWidth / 2, clientY: pWin.innerHeight / 2 });
          })
          .catch(err => console.warn('Tubes canvas background fallback:', err));
      }
    </script>""",
    height=0,
)

# ---------------------------------------------------------
# Native Navbar & Header
# ---------------------------------------------------------
st.markdown(
    """<div class="nav-header">
<div class="brand-logo-box">
<div class="brand-icon">🛡️</div>
<div>
<div class="brand-title">VERITAS</div>
<div style="font-size: 12px; color: #94a3b8; font-weight: 500;">
Hybrid Lexical &amp; Semantic Plagiarism Detection System
</div>
</div>
</div>
<div class="hero-badge">
<span>Semester 7 NLP &bull; Dual Engine</span>
</div>
</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Hero Section — Refined & Minimalist
# ---------------------------------------------------------
st.markdown(
    """<div style="text-align: center; padding: 18px 20px 8px 20px;">
<div class="hero-badge" style="margin: 0 auto 12px auto; display: inline-flex;">
<span>NLP POWERED &bull; DUAL-ENGINE INTELLIGENCE</span>
</div>
<div class="hero-title-clean">
<span class="hero-title-gradient">VERITAS</span> Plagiarism Detection
</div>
<div class="hero-subtitle-clean">
High-Precision Lexical Overlap &amp; Semantic Paraphrase Analysis
</div>
<p class="hero-desc-clean">
Combines statistical linguistics (TF-IDF, N-Grams, Longest Common Subsequence) with contextual transformer embeddings (Sentence-BERT, WordNet) to uncover both verbatim copying and subtle synonym swaps.
</p>
<div class="interactive-hint-clean">
✨ Interactive 3D tubes follow cursor movement &bull; Click background to shuffle palette
</div>
</div>""",
    unsafe_allow_html=True,
)

# Feature Spotlight Cards (4 Clean Columns)
feat_c1, feat_c2, feat_c3, feat_c4 = st.columns(4)

features = [
    (feat_c1, "📄", "Multi-Format Ingestion", "PDF, Word DOCX, TXT & Markdown with automatic text extraction.", "#38bdf8"),
    (feat_c2, "🎨", "Highlight Inspector", "Turnitin-style side-by-side color-coded verbatim & paraphrase highlights.", "#818cf8"),
    (feat_c3, "🔥", "Similarity Heatmap", "Interactive pairwise sentence similarity matrix with hover inspection.", "#f43f5e"),
    (feat_c4, "📚", "Corpus Scanner", "One-to-many batch scanning against entire document repositories.", "#34d399"),
]

for col, icon, title, desc, glow_color in features:
    with col:
        st.markdown(
            f"""<div class="glass-card" style="text-align: center; padding: 18px 14px; min-height: 145px;">
<div style="font-size: 26px; margin-bottom: 6px;">{icon}</div>
<div style="font-size: 13.5px; font-weight: 700; color: {glow_color}; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 5px;">
{title}
</div>
<p style="color: #94a3b8; font-size: 12px; line-height: 1.5; margin: 0;">
{desc}
</p>
</div>""",
            unsafe_allow_html=True,
        )

# Minimal Tech Stack Badges
st.markdown(
    """<div style="text-align: center; padding: 14px 0 6px 0;">
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;">
<span style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.25); color: #38bdf8; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">Python 3.11</span>
<span style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); color: #818cf8; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">Sentence-BERT</span>
<span style="background: rgba(236, 72, 153, 0.08); border: 1px solid rgba(236, 72, 153, 0.25); color: #f472b6; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">Scikit-learn</span>
<span style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); color: #38bdf8; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">Streamlit</span>
<span style="background: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.25); color: #fbbf24; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">Plotly.js</span>
<span style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); color: #34d399; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">WordNet NLTK</span>
</div>
</div>""",
    unsafe_allow_html=True,
)

st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Engine Settings
# ---------------------------------------------------------
embedding_files = available_embeddings()

with st.sidebar:
    st.markdown("### 🎛️ Engine Configuration")

    sbert_available = True
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        sbert_available = False

    backend = st.selectbox(
        "Semantic Model",
        options=["SBERT Transformer (Recommended)", "Word2Vec/GloVe (SIF-Weighted)", "None (Pure TF-IDF)"],
        help="SBERT captures contextual semantic meaning across paraphrases. Word2Vec/GloVe averages word vectors. 'None' disables embeddings.",
        disabled=not sbert_available,
    )
    if not sbert_available:
        backend = "Word2Vec/GloVe (SIF-Weighted)" if embedding_files else "None (Pure TF-IDF)"

    combination = st.selectbox(
        "Scoring Mode",
        options=["Max (Dominant Signal)", "Weighted Balance"],
        help="Max takes the maximum of lexical and semantic scores. Weighted blends both according to the ratio slider.",
    )
    tfidf_weight = 0.6
    if combination == "Weighted Balance":
        tfidf_weight = st.slider(
            "TF-IDF Lexical Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Remaining fraction is allocated to Semantic similarity.",
        )

    match_threshold = st.slider(
        "Sentence Match Sensitivity",
        min_value=0.40,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Pairwise score threshold above which sentence pairs are flagged.",
    )

    st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🧪 Load Example Test Cases")
    sample_choice = st.selectbox("Choose Benchmark Sample", ["(none)"] + sample_names())
    if sample_choice != "(none)":
        st.session_state["doc_a"] = read_sample(sample_choice)
        if "sample1_original" in sample_choice:
            st.session_state["doc_b"] = read_sample("sample1_plagiarized.txt")
        elif "sample2" in sample_choice:
            st.session_state["doc_b"] = read_sample("sample1_original.txt")
        elif "sample3" in sample_choice:
            st.session_state["doc_b"] = read_sample("sample3_paraphrased.txt")

# Load embedding model
embedding = None
backend_name = "None (Pure TF-IDF)"
if "SBERT" in backend:
    embedding = load_sbert()
    backend_name = "Sentence-BERT (all-MiniLM-L6-v2)"
    st.sidebar.success("⚡ SBERT Transformer Ready")
elif "Word2Vec" in backend:
    if not embedding_files:
        st.sidebar.error("No word vectors found. Run `python scripts/download_embeddings.py`.")
    else:
        embedding = load_word_embeddings(embedding_files[0])
        backend_name = f"Word2Vec ({os.path.basename(embedding_files[0])})"
        st.sidebar.success(f"⚡ Vectors: {os.path.basename(embedding_files[0])}")

# ---------------------------------------------------------
# Primary View Tabs: Plagiarism Studio vs Corpus Scanner
# ---------------------------------------------------------
tab_studio, tab_corpus = st.tabs([
    "⚡ Plagiarism Detection Studio",
    "📚 Corpus Repository Scanner",
])

# =========================================================
# TAB 1: Plagiarism Detection Studio
# =========================================================
with tab_studio:
    st.markdown(
        """<div style="padding: 6px 0 14px 0;">
<h3 style="color: #ffffff; margin: 0 0 4px 0; font-weight: 700; font-size: 1.25rem;">
📄 Document Comparison Studio
</h3>
<p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
Compare two documents to evaluate verbatim lexical duplication and contextual paraphrasing.
</p>
</div>""",
        unsafe_allow_html=True,
    )

    # Document Inputs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📘 Document A (Reference / Original)")
        uploaded_a = st.file_uploader(
            "Upload Reference Document",
            type=["txt", "md", "pdf", "docx"],
            key="upload_a",
            help="Supported formats: .txt, .md, .pdf, .docx • Max 20MB",
        )
        if uploaded_a is not None:
            text_a, err_a = safe_extract_text(uploaded_a)
            if err_a:
                st.error(err_a)
            else:
                st.session_state["doc_a"] = text_a

        st.caption("Accepted formats: `.txt`, `.md`, `.pdf`, `.docx` (Max 20MB)")
        doc_a = st.text_area(
            "Reference Text",
            value=st.session_state.get("doc_a", ""),
            height=220,
            placeholder="Paste or upload reference document text here...",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("##### 📕 Document B (Suspect / Submission)")
        uploaded_b = st.file_uploader(
            "Upload Suspect Document",
            type=["txt", "md", "pdf", "docx"],
            key="upload_b",
            help="Supported formats: .txt, .md, .pdf, .docx • Max 20MB",
        )
        if uploaded_b is not None:
            text_b, err_b = safe_extract_text(uploaded_b)
            if err_b:
                st.error(err_b)
            else:
                st.session_state["doc_b"] = text_b

        st.caption("Accepted formats: `.txt`, `.md`, `.pdf`, `.docx` (Max 20MB)")
        doc_b = st.text_area(
            "Suspect Text",
            value=st.session_state.get("doc_b", ""),
            height=220,
            placeholder="Paste or upload suspect document text here...",
            label_visibility="collapsed",
        )

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    detect_btn = st.button("🔍 Check for Plagiarism", type="primary", use_container_width=True)

    # Execute detection on button press and persist in session_state
    if detect_btn:
        if not doc_a.strip() or not doc_b.strip():
            st.error("⚠️ Please provide text for both Document A and Document B to run the analysis.")
        else:
            with st.spinner("Analyzing document semantics & vocabulary overlap (TF-IDF + SBERT)..."):
                result: DetectionResult = detect(
                    doc_a,
                    doc_b,
                    embedding=embedding,
                    tfidf_weight=tfidf_weight,
                    match_threshold=match_threshold,
                    combination="max" if "Max" in combination else "weighted",
                )
                st.session_state["detection_result"] = result
                st.session_state["analyzed_doc_a"] = doc_a
                st.session_state["analyzed_doc_b"] = doc_b

    # Render results if available in session_state (survives reruns)
    result = st.session_state.get("detection_result")

    if result is not None:
        st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)

        v_style = VERDICT_PALETTE.get(result.verdict, VERDICT_PALETTE["Original"])

        # -----------------------------------------------------
        # 1. OVERALL SCORE & VERDICT FIRST (Goals 1 & 3)
        # -----------------------------------------------------
        st.markdown("### 📊 Overall Similarity & Verdict")

        verdict_card_col, gauge_col, donut_col = st.columns([1.3, 1.1, 1.1])

        with verdict_card_col:
            st.markdown(
                f"""<div class="glass-card" style="height: 195px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid {v_style['border']}; background: {v_style['bg']}; text-align: center;">
<span style="font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; font-weight: 700; color: {v_style['text']};">
ANALYSIS VERDICT
</span>
<div style="font-size: 26px; font-weight: 800; color: #ffffff; margin: 8px 0 4px 0;">
{result.verdict}
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 800; color: {v_style['border']};">
{result.combined_score * 100:.1f}%
</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">
Flagged sentence pairs: <strong>{len(result.sentence_matches)}</strong>
</div>
</div>""",
                unsafe_allow_html=True,
            )

        with gauge_col:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(result.combined_score * 100, 1),
                    number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc", "family": "JetBrains Mono"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                        "bar": {"color": v_style["border"], "thickness": 0.28},
                        "bgcolor": "rgba(255,255,255,0.02)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "rgba(34, 197, 94, 0.12)"},
                            {"range": [30, 50], "color": "rgba(234, 179, 8, 0.12)"},
                            {"range": [50, 70], "color": "rgba(249, 115, 22, 0.12)"},
                            {"range": [70, 85], "color": "rgba(239, 68, 68, 0.15)"},
                            {"range": [85, 100], "color": "rgba(239, 68, 68, 0.25)"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=195,
                margin={"t": 15, "b": 10, "l": 20, "r": 20},
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8", "family": "Plus Jakarta Sans"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with donut_col:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Verbatim", "Paraphrased", "Original"],
                        values=[result.verbatim_count, result.paraphrase_count, result.original_count],
                        hole=0.6,
                        marker_colors=["#ef4444", "#f59e0b", "#3b82f6"],
                        textinfo="percent",
                        hoverinfo="label+value+percent",
                    )
                ]
            )
            fig_pie.update_layout(
                height=195,
                margin={"t": 10, "b": 10, "l": 10, "r": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cbd5e1", "family": "Plus Jakarta Sans"},
                showlegend=True,
                legend={"orientation": "h", "y": -0.15, "x": 0.05, "font": {"size": 11}},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # -----------------------------------------------------
        # 2. LEXICAL VS SEMANTIC DECOMPOSITION BREAKDOWN
        # -----------------------------------------------------
        st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔬 Multi-Signal Signal Decomposition")

        # Two primary engines
        lex_col, sem_col = st.columns(2)

        with lex_col:
            st.markdown(
                f"""<div class="glass-card" style="border-left: 3px solid #38bdf8;">
<div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #38bdf8; font-weight: 700;">
Lexical Engine Signal
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: #ffffff; margin: 4px 0;">
{result.tfidf_score * 100:.1f}%
</div>
<div style="font-size: 13px; color: #94a3b8;">
TF-IDF Cosine Vocabulary &amp; Word Overlap
</div>
</div>""",
                unsafe_allow_html=True,
            )

        with sem_col:
            st.markdown(
                f"""<div class="glass-card" style="border-left: 3px solid #818cf8;">
<div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #818cf8; font-weight: 700;">
Semantic Engine Signal
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: #ffffff; margin: 4px 0;">
{result.semantic_score * 100:.1f}%
</div>
<div style="font-size: 13px; color: #94a3b8;">
Sentence-BERT Dense Contextual Embeddings
</div>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # 4 Sub-metrics in clean 4-column row
        sub1, sub2, sub3, sub4 = st.columns(4)

        sub_metrics = [
            (sub1, f"{result.bigram_containment * 100:.1f}%", "2-Word Phrases (Bigram)"),
            (sub2, f"{result.trigram_containment * 100:.1f}%", "3-Word Phrases (Trigram)"),
            (sub3, f"{result.lcs_ratio * 100:.1f}%", "Longest Common Match (LCS)"),
            (sub4, f"{result.pos_similarity * 100:.1f}%", "Grammar Structure (POS)"),
        ]

        for col, val, label in sub_metrics:
            with col:
                st.markdown(
                    f"""<div class="glass-metric">
<div class="glass-metric-val">{val}</div>
<div class="glass-metric-lbl">{label}</div>
</div>""",
                    unsafe_allow_html=True,
                )

        # -----------------------------------------------------
        # 3. SIDE-BY-SIDE DUAL HIGHLIGHT COMPARISON WITH LEGEND
        # -----------------------------------------------------
        st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📑 Side-by-Side Highlight Comparison")

        # Explicit Legend Bar (Goal 3)
        st.markdown(
            """<div class="legend-bar">
<div style="font-weight: 600; color: #f8fafc;">Highlight Legend:</div>
<div class="legend-item">
<span class="legend-badge-verbatim"></span>
<span><strong>Verbatim Match (&ge; 80%)</strong> &bull; Direct lexical duplicate</span>
</div>
<div class="legend-item">
<span class="legend-badge-paraphrase"></span>
<span><strong>Paraphrase Match</strong> &bull; Rephrased with synonyms</span>
</div>
<div class="legend-item">
<span class="legend-badge-original"></span>
<span><strong>Original</strong> &bull; Unique content</span>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        match_map_a = {}
        match_map_b = {}
        for m in result.sentence_matches:
            if m.index_a not in match_map_a:
                match_map_a[m.index_a] = m
            if m.index_b not in match_map_b:
                match_map_b[m.index_b] = m

        preview_col1, preview_col2 = st.columns(2)

        with preview_col1:
            st.markdown(f"##### Document A &bull; Reference ({len(result.sentences_a)} sentences)")
            formatted_sents_a = []
            for i, sent in enumerate(result.sentences_a):
                if i in match_map_a:
                    m = match_map_a[i]
                    cls = "highlight-verbatim" if m.match_type == "Verbatim Match" else "highlight-paraphrase"
                    formatted_sents_a.append(
                        f"<span class='{cls}' title='Match #{i+1}: {m.score*100:.1f}%'>{sent}</span>"
                    )
                else:
                    formatted_sents_a.append(f"<span>{sent}</span>")

            st.markdown(
                f"<div class='doc-preview-glass'>{' '.join(formatted_sents_a)}</div>",
                unsafe_allow_html=True,
            )

        with preview_col2:
            st.markdown(f"##### Document B &bull; Suspect ({len(result.sentences_b)} sentences)")
            formatted_sents_b = []
            for j, sent in enumerate(result.sentences_b):
                if j in match_map_b:
                    m = match_map_b[j]
                    cls = "highlight-verbatim" if m.match_type == "Verbatim Match" else "highlight-paraphrase"
                    formatted_sents_b.append(
                        f"<span class='{cls}' title='Match with Reference sentence #{m.index_a+1}: {m.score*100:.1f}%'>{sent}</span>"
                    )
                else:
                    formatted_sents_b.append(f"<span>{sent}</span>")

            st.markdown(
                f"<div class='doc-preview-glass'>{' '.join(formatted_sents_b)}</div>",
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------
        # 4. PAIRWISE SENTENCE SIMILARITY HEATMAP
        # -----------------------------------------------------
        if result.similarity_matrix and len(result.sentences_a) > 1 and len(result.sentences_b) > 1:
            st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🔥 Pairwise Sentence Similarity Heatmap")
            st.caption("Hover over cells to inspect exact similarity between Reference and Suspect sentences.")

            y_labels = [f"Ref #{i+1}: {s[:28]}..." for i, s in enumerate(result.sentences_a)]
            x_labels = [f"Suspect #{j+1}: {s[:28]}..." for j, s in enumerate(result.sentences_b)]

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=result.similarity_matrix,
                    x=x_labels,
                    y=y_labels,
                    colorscale="Sunsetdark",
                    zmin=0.0,
                    zmax=1.0,
                    colorbar={
                        "title": {"text": "Similarity", "font": {"color": "#cbd5e1"}},
                        "tickfont": {"color": "#94a3b8", "family": "JetBrains Mono"},
                    },
                )
            )
            fig_heat.update_layout(
                height=340 + min(320, len(result.sentences_a) * 16),
                xaxis={"title": {"text": "Document B Sentences", "font": {"color": "#94a3b8"}}, "tickangle": -30, "tickfont": {"color": "#94a3b8", "size": 11}},
                yaxis={"title": {"text": "Document A Sentences", "font": {"color": "#94a3b8"}}, "autorange": "reversed", "tickfont": {"color": "#cbd5e1", "size": 11}},
                margin={"t": 30, "b": 60, "l": 140, "r": 30},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # -----------------------------------------------------
        # 5. FLAGGED SENTENCE ALIGNMENTS
        # -----------------------------------------------------
        st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Flagged Sentence Alignments ({len(result.sentence_matches)})")

        if result.sentence_matches:
            for idx, match in enumerate(result.sentence_matches):
                match_color = "#ef4444" if match.match_type == "Verbatim Match" else "#f59e0b"

                st.markdown(
                    f"""<div style="margin-top: 12px; margin-bottom: 6px;">
<span style="font-weight: 700; color: #ffffff;">Pair #{idx + 1}: {match.score * 100:.1f}% Match</span>
&nbsp;
<span style="background:{match_color}22; color:{match_color}; border: 1px solid {match_color}60; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700;">
{match.match_type}
</span>
&nbsp;
<small style="color:#94a3b8;">(Ref #{match.index_a + 1} &bull; Suspect #{match.index_b + 1})</small>
</div>""",
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**Reference (A):** {match.sentence_a}")
                    with c_b:
                        st.markdown(f"**Suspect (B):** {match.sentence_b}")

                    if match.synonyms:
                        syn_tags = [f"`{wa} ↔ {wb}`" for wa, wb in match.synonyms]
                        st.caption(f"🔄 **WordNet Synonyms detected:** {' '.join(syn_tags)}")
        else:
            st.info("✅ No sentence pairs exceeded the match threshold.")

        # -----------------------------------------------------
        # 6. DOWNLOAD VERIFICATION REPORT
        # -----------------------------------------------------
        st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
        html_report = generate_html_report(
            result,
            doc_a_name="Document A",
            doc_b_name="Document B",
            backend_name=backend_name,
        )
        st.download_button(
            label="📥 Export Full Verification Report (.html)",
            data=html_report,
            file_name="veritas_plagiarism_report.html",
            mime="text/html",
            use_container_width=True,
        )
    else:
        # Informative Empty State (Goal 4)
        st.markdown(
            """<div class="empty-state-card">
<div style="font-size: 32px; margin-bottom: 8px;">📑</div>
<div style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">
Ready for Document Comparison
</div>
<p style="color: #94a3b8; font-size: 13.5px; max-width: 540px; margin: 0 auto 12px auto; line-height: 1.5;">
Upload or paste two documents in the boxes above, or select an example benchmark case from the sidebar, then click <strong>Check for Plagiarism</strong>.
</p>
<div style="font-size: 12px; color: #64748b;">
Supports .txt, .md, .pdf, and Word .docx documents
</div>
</div>""",
            unsafe_allow_html=True,
        )

# =========================================================
# TAB 2: Corpus / Batch Repository Scanner
# =========================================================
with tab_corpus:
    st.markdown(
        """<div style="padding: 6px 0 14px 0;">
<h3 style="color: #ffffff; margin: 0 0 4px 0; font-weight: 700; font-size: 1.25rem;">
📚 Repository Scanner (One-to-Many)
</h3>
<p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
Scan a suspect submission against a corpus of reference files to locate original sources.
</p>
</div>""",
        unsafe_allow_html=True,
    )

    suspect_upload = st.file_uploader(
        "Upload Suspect Submission File",
        type=["txt", "md", "pdf", "docx"],
        key="corpus_suspect",
        help="Supported formats: .txt, .md, .pdf, .docx • Max 20MB",
    )
    suspect_text = ""
    if suspect_upload is not None:
        suspect_text, err_susp = safe_extract_text(suspect_upload)
        if err_susp:
            st.error(err_susp)
    else:
        suspect_text = st.text_area("Or Paste Suspect Submission Text", height=130, placeholder="Paste suspect text to scan against the repository...")

    # Reference documents source
    ref_source = st.radio(
        "Reference Corpus Source",
        options=["Use built-in benchmark repository (`data/samples`)", "Upload multiple reference documents"],
        horizontal=True,
    )

    ref_corpus: dict[str, str] = {}
    if "built-in" in ref_source:
        for fname in sample_names():
            ref_corpus[fname] = read_sample(fname)
        st.info(f"Loaded {len(ref_corpus)} benchmark reference documents from `data/samples/`.")
    else:
        multi_refs = st.file_uploader(
            "Upload Reference Documents",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            key="multi_refs",
            help="Upload multiple documents to scan against",
        )
        if multi_refs:
            for f in multi_refs:
                txt, err_ref = safe_extract_text(f)
                if not err_ref and txt:
                    ref_corpus[f.name] = txt
            st.info(f"Loaded {len(ref_corpus)} custom uploaded reference files.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    scan_corpus_btn = st.button("🔎 Scan Entire Corpus", type="primary")

    if scan_corpus_btn:
        if not suspect_text.strip():
            st.error("⚠️ Please provide text or upload a suspect document to begin scanning.")
        elif not ref_corpus:
            st.error("⚠️ No reference documents available in the corpus to compare against.")
        else:
            with st.spinner(f"Scanning suspect submission against {len(ref_corpus)} reference documents..."):
                scores = []
                for ref_name, ref_text in ref_corpus.items():
                    res = detect(
                        ref_text,
                        suspect_text,
                        embedding=embedding,
                        tfidf_weight=tfidf_weight,
                        match_threshold=match_threshold,
                        combination="max",
                    )
                    scores.append({
                        "Reference Document": ref_name,
                        "Similarity (%)": round(res.combined_score * 100, 1),
                        "Lexical Score (%)": round(res.tfidf_score * 100, 1),
                        "Semantic Score (%)": round(res.semantic_score * 100, 1),
                        "Verdict": res.verdict,
                        "Flagged Sentences": len(res.sentence_matches),
                    })

                scores.sort(key=lambda x: x["Similarity (%)"], reverse=True)
                st.session_state["corpus_scores"] = scores

    corpus_scores = st.session_state.get("corpus_scores")

    if corpus_scores:
        st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
        st.subheader("🏆 Plagiarism Attribution Leaderboard")

        top_refs = [x["Reference Document"] for x in corpus_scores[:8]]
        top_scores = [x["Similarity (%)"] for x in corpus_scores[:8]]
        top_colors = [
            "#ef4444" if s >= 70 else "#f59e0b" if s >= 50 else "#3b82f6" for s in top_scores
        ]

        fig_bar = go.Figure(
            go.Bar(
                x=top_scores,
                y=top_refs,
                orientation="h",
                marker_color=top_colors,
                text=[f"{s}%" for s in top_scores],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            height=280,
            xaxis={"title": {"text": "Similarity Score (%)", "font": {"color": "#94a3b8"}}, "range": [0, 100], "tickfont": {"color": "#94a3b8"}},
            yaxis={"autorange": "reversed", "tickfont": {"color": "#cbd5e1"}},
            margin={"t": 20, "b": 30, "l": 140, "r": 30},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(corpus_scores, use_container_width=True)
    elif not scan_corpus_btn:
        st.markdown(
            """<div class="empty-state-card">
<div style="font-size: 30px; margin-bottom: 6px;">📚</div>
<div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
Repository Scanner Ready
</div>
<p style="color: #94a3b8; font-size: 13px; max-width: 480px; margin: 0 auto; line-height: 1.5;">
Upload a suspect paper to rank matches against all documents in the corpus.
</p>
</div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Subtle Footer
# ---------------------------------------------------------
st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """<div style="text-align: center; color: #64748b; font-size: 12px; padding: 10px 0;">
VERITAS &bull; Natural Language Processing &bull; Semester 7 Micro Project
</div>""",
    unsafe_allow_html=True,
)