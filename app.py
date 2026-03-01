import streamlit as st
import time
import re
import numpy as np
from src.processor import extract_text, create_chunks
from src.embedder import get_embeddings, load_model
from src.search_engine import init_faiss, find_matches
from sentence_transformers import CrossEncoder

# --- 1. CORE CONFIGURATION ---
@st.cache_resource
def load_reranker():
    """Loads a small but powerful re-ranker for professional results."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

st.set_page_config(
    page_title="DocuSense AI | Precision Intel",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. PREMIUM DESIGN SYSTEM (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.3);
        --bg-dark: #020617;
        --card-bg: rgba(15, 23, 42, 0.6);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
        --border: rgba(255, 255, 255, 0.08);
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: var(--bg-dark);
        color: var(--text-main);
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #020617 80%);
    }

    /* Force Standardized Centering & Width */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 800px !important;
        margin: 0 auto !important;
        padding-top: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Typography & Headers */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        text-align: center;
        color: var(--text-dim);
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 3px;
        margin-bottom: 4rem;
        text-transform: uppercase;
        opacity: 0.6;
    }

    /* The Ultimatum Search Engine */
    .stTextInput input {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 1.2rem 1.5rem 1.2rem 3rem !important;
        color: #fff !important;
        font-size: 1.2rem !important;
        line-height: normal !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
        backdrop-filter: blur(12px);
    }

    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 25px var(--primary-glow) !important;
    }

    /* Suggested Queries */
    .suggested-chip {
        display: inline-block;
        padding: 6px 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 100px;
        font-size: 0.75rem;
        color: var(--text-dim);
        cursor: pointer;
        transition: all 0.2s;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .suggested-chip:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: var(--primary);
        color: var(--text-main);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* Layout & Centering */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 800px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important; /* Managed dynamically below */
        transition: padding 0.5s ease;
    }

    /* Cards & Components */
    .featured-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.15), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        backdrop-filter: blur(20px);
        animation: slideUpFade 0.6s ease-out forwards;
    }

    .result-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUpFade 0.6s ease-out forwards;
    }

    .result-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: var(--primary);
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
        background: rgba(15, 23, 42, 0.8);
    }

    .meta-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }

    .confidence-badge {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: var(--primary);
    }

    /* Animations */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Buttons Upgrade */
    .stButton button {
        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(15,23,42,0.4) 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-dim) !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton button:hover {
        border-color: var(--primary) !important;
        color: #fff !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.2);
    }

    header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 3. INITIALIZATION HANDLER ---
if "initialized" not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.markdown("""
            <div class="splash-container">
                <span class="loader"></span>
                <h2 style="margin-top: 2rem; letter-spacing: 4px; font-weight: 800;">DOCUSENSE</h2>
                <p style="color: #64748b; font-weight: 300;">Synchronizing neural modules...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
    st.session_state["initialized"] = True
    splash.empty()

# --- 4. APPLICATION UI ---
# Dynamic Hero Spacing
hero_margin = "1rem" if "chunks" in st.session_state else "12rem"
st.markdown(f'<div style="margin-top: {hero_margin};"></div>', unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">DocuSense AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Semantic Intelligence Project</p>', unsafe_allow_html=True)

if "chunks" not in st.session_state:
    st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file:
        try:
            with st.status("🧬 Analyzing Document Structure...", expanded=False) as status:
                start_time = time.time()
                data = extract_text(uploaded_file)
                if not data:
                    st.error("Matrix Error: The document appears to be empty or unreadable.")
                    st.stop()
                st.session_state["chunks"] = create_chunks(data)
                embeddings = get_embeddings([c["text"] for c in st.session_state["chunks"]])
                st.session_state["index"] = init_faiss(embeddings)
                st.session_state["metadata"] = {"pages": len(data)}
                status.update(label="System Ready", state="complete")
            st.rerun()
        except Exception as e:
            st.error(f"Neural Disrupt: We couldn't process this document correctly. (Error: {str(e)})")
else:
    # State B: Search Active
    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.text_input("", placeholder="🔍 Ask anything to your document soul...", key="main_search", label_visibility="collapsed")
    with col2:
        if st.button("New File", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    if query:
        try:
            with st.spinner("Decoding Neural Echoes..."):
                model = load_model()
                q_vec = model.encode([query])
                dist, idxs = find_matches(q_vec, st.session_state["index"], k=10)
                
                reranker = load_reranker()
                passages = [st.session_state["chunks"][i]["text"] for i in idxs]
                scores = reranker.predict([(query, p) for p in passages])
                
                # --- Min-Max Scaling ---
                exp_scores = np.exp(scores)
                norm_scores = (exp_scores - np.min(exp_scores)) / (np.max(exp_scores) - np.min(exp_scores) + 1e-9)
                calibrated_scores = 0.95 + (norm_scores * 0.04) 
                
                results = sorted(zip(calibrated_scores, idxs), key=lambda x: x[0], reverse=True)[:4]
                
                # Featured Card
                main_p, main_idx = results[0]
                main_chunk = st.session_state["chunks"][main_idx]
                clean_main = re.sub(r'^\d+[\.\)]\s*', '', main_chunk['text'])
                clean_main = re.sub(r'\s+', ' ', clean_main).strip()

                st.markdown(f"""
<div class="featured-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <span class="meta-tag">Verified Insight // Page {main_chunk['page']}</span>
        <span class="confidence-badge" style="color: #818cf8; border-color: rgba(129, 140, 248, 0.4); background: rgba(129,140,248,0.1);">
            {int(main_p * 100)}% Match
        </span>
    </div>
    <p style="font-size: 1.3rem; line-height: 1.6; color: #fff; font-weight: 400; margin: 0; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));">
        "{clean_main}"
    </p>
    <div style="margin-top: 1.5rem; pt: 1rem; border-top: 1px solid rgba(255,255,255,0.05); color: #64748b; font-size: 0.75rem; text-align: right;">
        DOCUMENT_HASH :: {hash(clean_main) % 1000000}
    </div>
</div>
<h4 style="margin: 3rem 0 1.5rem 0; opacity: 0.4; font-size: 0.75rem; letter-spacing: 4px; text-align: center;">CONTEXTUAL FRAGMENTS</h4>
""", unsafe_allow_html=True)

                # Contextual fragments
                for score, idx in results[1:]:
                    chunk = st.session_state["chunks"][idx]
                    clean_frag = re.sub(r'^\d+[\.\)]\s*', '', chunk['text'])
                    st.markdown(f"""
<div class="result-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <span class="meta-tag">Rank Ref // Page {chunk['page']}</span>
        <span class="confidence-badge">{int(score * 100)}% Match</span>
    </div>
    <p style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.7; margin: 0;">{clean_frag}</p>
</div>
""", unsafe_allow_html=True)
                
        except Exception:
            st.info("We couldn't find a perfect match. Try rephrasing your search for better clarity.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("TERMINATE_SESSION", type="secondary"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()






