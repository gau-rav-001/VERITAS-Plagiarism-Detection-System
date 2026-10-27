"""Neon Tech Landing Page component with 3D Tubes Interactive Cursor Background.

Reference: Superdesign VERITAS | Ultimate Neon Tech Landing
Project ID: 38c5510c-fddf-4a0d-874e-0f4e87ecf4b8
Draft ID: 25e866d0-0866-4767-aeb8-30361007d656
"""

from __future__ import annotations


def get_neon_landing_html() -> str:
    """Returns the complete, responsive HTML for the Ultimate Neon Tech Landing page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VERITAS | Advanced NLP Plagiarism Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://code.iconify.design/iconify-icon/1.0.7/iconify-icon.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    * {
      box-sizing: border-box;
    }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #010409;
      color: #f8fafc;
      margin: 0;
      overflow-x: hidden;
      scroll-behavior: smooth;
    }
    .mono {
      font-family: 'JetBrains Mono', monospace;
    }
    .glass {
      background: rgba(8, 12, 24, 0.75);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.3);
    }
    .glass-hover:hover {
      background: rgba(30, 41, 59, 0.8);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .neon-text-teal {
      text-shadow: 0 0 15px rgba(0, 247, 255, 0.8), 0 0 40px rgba(0, 247, 255, 0.4), 0 0 80px rgba(0, 247, 255, 0.2);
    }
    .neon-text-purple {
      text-shadow: 0 0 20px rgba(124, 58, 237, 0.8), 0 0 45px rgba(124, 58, 237, 0.5), 0 0 90px rgba(124, 58, 237, 0.2);
    }
    .hero-title {
      font-size: clamp(4.5rem, 14vw, 10rem);
      line-height: 0.92;
      letter-spacing: 0.16em;
      text-shadow: 0 0 20px rgba(0, 247, 255, 0.4), 0 0 60px rgba(0, 247, 255, 0.2);
    }
    .glow-border-teal {
      box-shadow: 0 0 15px rgba(0, 217, 255, 0.15);
    }
    @keyframes pulse-slow {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.6; transform: scale(0.98); }
    }
    @keyframes fade-in-up {
      from { opacity: 0; transform: translateY(35px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .animate-pulse-slow {
      animation: pulse-slow 3s infinite ease-in-out;
    }
    .reveal {
      opacity: 0;
      animation: fade-in-up 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    .delay-1 { animation-delay: 0.15s; }
    .delay-2 { animation-delay: 0.3s; }
    .delay-3 { animation-delay: 0.45s; }
    
    #canvas-container {
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: auto;
      background-color: #010409;
    }
    #tubes-canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>
  <div class="min-h-screen relative">
    
    <!-- 3D INTERACTIVE BACKGROUND CANVAS -->
    <div id="canvas-container">
      <canvas id="tubes-canvas"></canvas>
    </div>

    <!-- CONTENT LAYER -->
    <div class="relative z-10">
      
      <!-- NAVBAR -->
      <nav class="fixed top-0 w-full z-50 px-8 py-5 flex justify-between items-center bg-[#020617]/60 backdrop-blur-md border-b border-white/5">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/25">
            <iconify-icon icon="lucide:shield-check" class="text-2xl text-white"></iconify-icon>
          </div>
          <span class="text-2xl font-extrabold tracking-tighter uppercase text-white">Veritas</span>
        </div>
        
        <div class="hidden md:flex gap-8 items-center text-sm font-medium text-slate-300">
          <a href="#engine" class="hover:text-cyan-400 transition-colors">Architecture</a>
          <a href="#features" class="hover:text-cyan-400 transition-colors">Features</a>
          <a href="#stack" class="hover:text-cyan-400 transition-colors">Tech Stack</a>
          <a href="#launch" onclick="notifyParent('launch');" class="px-5 py-2 rounded-full text-white border-white/20 hover:bg-white/10 transition-all bg-[#0f172a]/70 backdrop-blur-md border font-semibold hover:scale-105">
            ⚡ Launch Studio
          </a>
        </div>
      </nav>

      <!-- HERO SECTION -->
      <section class="min-h-[115vh] flex flex-col items-center justify-center px-6 pt-24 relative overflow-hidden text-center">
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[650px] h-[650px] bg-cyan-500/10 blur-[200px] rounded-full pointer-events-none"></div>
        
        <div class="max-w-6xl w-full space-y-10 relative z-10 reveal">
          <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-bold uppercase tracking-widest mb-2 shadow-[0_0_20px_rgba(6,182,212,0.15)]">
            <span class="flex h-2 w-2 rounded-full bg-cyan-400 animate-ping"></span>
            NLP-Powered Semantic Verification
          </div>
          
          <h1 class="hero-title font-black uppercase text-white select-none">
            VERI<span class="text-transparent bg-clip-text bg-gradient-to-r from-[#00f7ff] via-[#a855f7] to-[#ec4899]">TAS</span>
          </h1>
          
          <p class="text-xl md:text-3xl font-extrabold text-slate-100 max-w-4xl mx-auto neon-text-purple uppercase tracking-[0.25em] reveal delay-1">
            Dual-Engine Academic Plagiarism Detection
          </p>
          
          <p class="text-lg md:text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed reveal delay-2">
            A state-of-the-art hybrid system combining <span class="text-cyan-400 font-semibold">Lexical Overlap (TF-IDF + N-Grams)</span> and <span class="text-purple-400 font-semibold">Deep Semantic Embeddings (Sentence-BERT)</span> to expose paraphrasing that traditional tools miss.
          </p>

          <div class="flex flex-col sm:flex-row gap-6 justify-center pt-8 reveal delay-3">
            <button onclick="notifyParent('upload');" class="group relative px-10 py-5 bg-white text-black font-extrabold rounded-2xl overflow-hidden transition-all hover:scale-105 active:scale-95 shadow-[0_0_40px_rgba(255,255,255,0.25)] flex items-center justify-center gap-3 text-base">
              <iconify-icon icon="lucide:upload-cloud" class="text-2xl text-cyan-600"></iconify-icon>
              Upload Document
            </button>
            <button onclick="notifyParent('launch');" class="px-10 py-5 glass border-white/10 text-white font-extrabold uppercase tracking-widest text-base rounded-2xl hover:bg-white/10 hover:scale-105 transition-all flex items-center justify-center gap-3 shadow-[0_0_30px_rgba(0,247,255,0.1)]">
              Open Plagiarism Studio
              <iconify-icon icon="lucide:arrow-right" class="text-cyan-400"></iconify-icon>
            </button>
          </div>

          <div class="pt-8 flex flex-col items-center gap-2 text-slate-400 animate-pulse-slow pointer-events-none">
            <iconify-icon icon="lucide:mouse-pointer-2" class="text-2xl text-cyan-400"></iconify-icon>
            <span class="text-xs uppercase tracking-[0.25em] font-semibold">Click anywhere to randomize 3D neon space</span>
          </div>
        </div>
      </section>

      <!-- DUAL ENGINE SECTION -->
      <div class="w-full py-8 flex items-center justify-center">
        <div class="w-full max-w-lg h-[1px] bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"></div>
      </div>
      
      <section id="engine" class="py-28 px-6 relative overflow-hidden">
        <div class="max-w-6xl mx-auto relative z-10">
          <div class="text-center mb-20 reveal">
            <h2 class="text-4xl md:text-5xl font-black mb-4 uppercase tracking-[0.1em] text-white">The Dual-Engine Core</h2>
            <p class="text-slate-400 text-lg">Fusing classical linguistics with deep neural contextual transformers.</p>
          </div>

          <div class="grid md:grid-cols-2 gap-8">
            <!-- Lexical Engine Card -->
            <div class="p-10 rounded-3xl relative overflow-hidden group transition-all duration-300 bg-[#0f172a]/70 backdrop-blur-xl border border-white/10 hover:border-cyan-500/40 hover:shadow-[0_0_35px_rgba(6,182,212,0.15)]">
              <div class="absolute -right-8 -top-8 w-44 h-44 bg-cyan-500/10 blur-3xl rounded-full transition-all group-hover:scale-125"></div>
              
              <div class="w-16 h-16 rounded-2xl bg-cyan-500/20 flex items-center justify-center mb-8 border border-cyan-500/30">
                <iconify-icon icon="lucide:binary" class="text-3xl text-cyan-400"></iconify-icon>
              </div>

              <h3 class="text-2xl font-bold mb-3 text-white">Lexical Engine</h3>
              <p class="text-slate-400 mb-8 leading-relaxed text-sm md:text-base">
                Specializes in verbatim matching and identical sequence detection using statistical term-frequency weighting and sequential token alignment.
              </p>

              <div class="space-y-3 font-mono text-xs md:text-sm text-cyan-300">
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_#00f7ff]"></span>
                  TF-IDF Vectorization &amp; Cosine Metric
                </div>
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_#00f7ff]"></span>
                  N-Gram Containment (Bi/Tri-grams)
                </div>
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_#00f7ff]"></span>
                  Longest Common Subsequence (LCS)
                </div>
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_#00f7ff]"></span>
                  POS Syntactic Pattern Alignment
                </div>
              </div>
            </div>

            <!-- Semantic Engine Card -->
            <div class="p-10 rounded-3xl relative overflow-hidden group transition-all duration-300 bg-[#0f172a]/70 backdrop-blur-xl border border-white/10 hover:border-purple-500/40 hover:shadow-[0_0_35px_rgba(168,85,247,0.15)]">
              <div class="absolute -right-8 -top-8 w-44 h-44 bg-purple-500/10 blur-3xl rounded-full transition-all group-hover:scale-125"></div>
              
              <div class="w-16 h-16 rounded-2xl bg-purple-500/20 flex items-center justify-center mb-8 border border-purple-500/30">
                <iconify-icon icon="lucide:brain-circuit" class="text-3xl text-purple-400"></iconify-icon>
              </div>

              <h3 class="text-2xl font-bold mb-3 text-white">Semantic Engine</h3>
              <p class="text-slate-400 mb-8 leading-relaxed text-sm md:text-base">
                Deep contextual neural analysis that exposes smart paraphrasing, synonym substitutions, and concept theft even when every word is swapped.
              </p>

              <div class="space-y-3 font-mono text-xs md:text-sm text-purple-300">
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-purple-400 rounded-full shadow-[0_0_8px_#a855f7]"></span>
                  Sentence-BERT (all-MiniLM-L6-v2)
                </div>
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-purple-400 rounded-full shadow-[0_0_8px_#a855f7]"></span>
                  Contextual Cosine Distance
                </div>
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-purple-400 rounded-full shadow-[0_0_8px_#a855f7]"></span>
                  WordNet Synset Paraphrase Mapping
                </div>
                <div class="flex items-center gap-3">
                  <span class="w-2 h-2 bg-purple-400 rounded-full shadow-[0_0_8px_#a855f7]"></span>
                  GloVe SIF-Weighted Vector Fallback
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- FEATURES GRID -->
      <div class="w-full py-8 flex items-center justify-center">
        <div class="w-full max-w-lg h-[1px] bg-gradient-to-r from-transparent via-purple-500/50 to-transparent"></div>
      </div>

      <section id="features" class="py-24 px-6 relative overflow-hidden">
        <div class="max-w-6xl mx-auto relative z-10">
          <div class="text-center mb-16">
            <h2 class="text-4xl md:text-5xl font-black mb-3 uppercase tracking-[0.1em] text-white">Engine Capabilities</h2>
            <p class="text-slate-400 text-base">Comprehensive plagiarism auditing for modern academia.</p>
          </div>

          <div class="grid md:grid-cols-4 gap-6">
            <div class="p-8 bg-[#0f172a]/70 backdrop-blur-xl rounded-3xl border border-white/5 transition-all hover:border-cyan-500/40 hover:-translate-y-1">
              <div class="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center mb-6">
                <iconify-icon icon="lucide:file-up" class="text-3xl text-cyan-400"></iconify-icon>
              </div>
              <h4 class="font-bold text-lg mb-2 text-white">Multi-Format Ingestion</h4>
              <p class="text-sm text-slate-400 leading-relaxed">Direct support for PDF, Word DOCX, Markdown, and TXT files.</p>
            </div>

            <div class="p-8 bg-[#0f172a]/70 backdrop-blur-xl rounded-3xl border border-white/5 transition-all hover:border-purple-500/40 hover:-translate-y-1">
              <div class="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center mb-6">
                <iconify-icon icon="lucide:split" class="text-3xl text-purple-400"></iconify-icon>
              </div>
              <h4 class="font-bold text-lg mb-2 text-white">Dual Highlighting</h4>
              <p class="text-sm text-slate-400 leading-relaxed">Turnitin-style side-by-side view with color-coded verbatim and paraphrase tags.</p>
            </div>

            <div class="p-8 bg-[#0f172a]/70 backdrop-blur-xl rounded-3xl border border-white/5 transition-all hover:border-pink-500/40 hover:-translate-y-1">
              <div class="w-12 h-12 rounded-xl bg-pink-500/10 flex items-center justify-center mb-6">
                <iconify-icon icon="lucide:grid" class="text-3xl text-pink-400"></iconify-icon>
              </div>
              <h4 class="font-bold text-lg mb-2 text-white">Similarity Heatmap</h4>
              <p class="text-sm text-slate-400 leading-relaxed">Interactive matrix showing pairwise sentence similarities on hover.</p>
            </div>

            <div class="p-8 bg-[#0f172a]/70 backdrop-blur-xl rounded-3xl border border-white/5 transition-all hover:border-amber-500/40 hover:-translate-y-1">
              <div class="w-12 h-12 rounded-xl bg-amber-500/10 flex items-center justify-center mb-6">
                <iconify-icon icon="lucide:layers" class="text-3xl text-amber-400"></iconify-icon>
              </div>
              <h4 class="font-bold text-lg mb-2 text-white">Corpus Scanner</h4>
              <p class="text-sm text-slate-400 leading-relaxed">1-to-many batch scanner to locate sources across an entire assignment archive.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- TECH STACK -->
      <section id="stack" class="py-20 px-6 relative overflow-hidden border-t border-white/5">
        <div class="max-w-5xl mx-auto text-center">
          <h3 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-10">Powered By Academic NLP Stack</h3>
          <div class="flex flex-wrap justify-center items-center gap-x-12 gap-y-6 text-slate-400">
            <div class="flex items-center gap-2">
              <iconify-icon icon="logos:python" class="text-3xl"></iconify-icon>
              <span class="mono font-bold text-slate-300">Python 3.11</span>
            </div>
            <div class="flex items-center gap-2">
              <iconify-icon icon="logos:pytorch-icon" class="text-3xl"></iconify-icon>
              <span class="mono font-bold text-slate-300">Sentence-Transformers</span>
            </div>
            <div class="flex items-center gap-2">
              <iconify-icon icon="logos:scikit-learn" class="text-4xl"></iconify-icon>
              <span class="mono font-bold text-slate-300">Scikit-Learn</span>
            </div>
            <div class="flex items-center gap-2">
              <iconify-icon icon="logos:streamlit" class="text-3xl"></iconify-icon>
              <span class="mono font-bold text-slate-300">Streamlit</span>
            </div>
            <div class="flex items-center gap-2">
              <iconify-icon icon="logos:plotly" class="text-3xl"></iconify-icon>
              <span class="mono font-bold text-slate-300">Plotly JS</span>
            </div>
          </div>
        </div>
      </section>

      <!-- FOOTER -->
      <footer class="py-14 border-t border-white/5 px-6">
        <div class="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div>
            <div class="flex items-center gap-2 mb-2">
              <div class="w-7 h-7 bg-gradient-to-tr from-cyan-500 to-purple-600 rounded-lg flex items-center justify-center">
                <iconify-icon icon="lucide:shield-check" class="text-base text-white"></iconify-icon>
              </div>
              <span class="text-base font-extrabold tracking-tighter uppercase text-white">Veritas</span>
            </div>
            <p class="text-slate-400 text-xs max-w-sm">
              Natural Language Processing Semester 7 Capstone Project. Built for academic integrity verification.
            </p>
          </div>
          
          <div class="text-center md:text-right">
            <p class="text-xs text-slate-300 mb-1 font-semibold">Academic Year 2024–25 &bull; Department of Computer Engineering</p>
            <p class="text-[11px] text-slate-400">&copy; 2024 Veritas AI &bull; Natural Language Processing Lab</p>
          </div>
        </div>
      </footer>

    </div>
  </div>

  <!-- 3D INTERACTIVE SCRIPT (TubesCursor) -->
  <script type="module">
    import TubesCursor from 'https://cdn.jsdelivr.net/npm/threejs-components@0.0.19/build/cursors/tubes1.min.js';

    const canvas = document.getElementById('tubes-canvas');
    const container = document.getElementById('canvas-container');

    let app = null;
    try {
      app = TubesCursor(canvas, {
        tubes: {
          colors: ["#00d9ff", "#7c3aed", "#ff008a"],
          lights: {
            intensity: 220,
            colors: ["#00f7ff", "#a855f7", "#ff006e", "#ffbe0b"]
          }
        }
      });
    } catch (err) {
      console.warn("WebGL TubesCursor initialization fallback:", err);
    }

    // Handle click to randomize colors as per spec
    const randomColor = () => "#" + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0');
    
    if (container && app) {
      container.addEventListener('click', () => {
        const newTubesColors = [randomColor(), randomColor(), randomColor()];
        const newLightsColors = [randomColor(), randomColor(), randomColor(), randomColor()];
        if (app.tubes) {
          app.tubes.setColors(newTubesColors);
          app.tubes.setLightsColors(newLightsColors);
        }
      });
    }

    // Bridge actions to Streamlit parent frame if embedded
    window.notifyParent = function(action) {
      window.parent.postMessage({ type: 'streamlit:setComponentValue', value: action }, '*');
      // Also fallback to scrolling or activating tab in parent window if directly in DOM
      const parentDoc = window.parent.document;
      if (parentDoc) {
        const tabs = parentDoc.querySelectorAll('[data-baseweb="tab"]');
        if (tabs && tabs.length > 1) {
          tabs[1].click();
        }
      }
    };
  </script>
</body>
</html>
"""
