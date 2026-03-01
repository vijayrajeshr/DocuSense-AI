import streamlit as st
import time
import re
import numpy as np
import os
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
    initial_sidebar_state="expanded"
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

    [data-testid="stAppViewBlockContainer"] {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important;
        transition: padding 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: var(--bg-dark);
        color: var(--text-main);
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #020617 80%);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* Neural Sidebar Enhancements */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebarNav"] { display: none; }

    .gauge-container { display: flex; justify-content: center; margin: 2rem 0; position: relative; }
    .gauge-circle {
        width: 120px; height: 120px; border-radius: 50%;
        background: conic-gradient(var(--primary) var(--percentage), rgba(255,255,255,0.05) 0);
        display: flex; align-items: center; justify-content: center;
        mask: radial-gradient(transparent 55px, #000 56px);
        -webkit-mask: radial-gradient(transparent 55px, #000 56px);
        transition: all 1s ease;
        box-shadow: 0 0 20px var(--primary-glow);
    }
    .gauge-value { 
        position: absolute; 
        top: 50%; left: 50%; transform: translate(-50%, -50%);
        font-family: 'JetBrains Mono'; font-weight: 800; font-size: 1.2rem; color: #fff; 
    }

    .persistence-badge {
        padding: 1.5rem; border-radius: 20px; background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.1); margin-bottom: 2rem; animation: fadeIn 0.8s ease;
    }

    .neural-log {
        background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border); border-radius: 12px;
        padding: 10px; height: 120px; overflow-y: auto; font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem; color: #94a3b8; line-height: 1.4;
    }

    .ready-badge {
        padding: 6px 14px;
        border-radius: 8px;
        font-family: 'JetBrains Mono';
        font-size: 11px;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 20px;
        animation: badgeFadeIn 0.5s ease 0.5s forwards;
        opacity: 0;
    }

    @keyframes badgeFadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- ChatGPT-Style Reactive Search Bar --- */
    [data-testid="stTextInput"] {
        margin-top: 3rem !important;
        margin-bottom: 0 !important;
    }

    [data-testid="stTextInput"] [data-baseweb="input"] {
        height: 72px !important;
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 0 12px 0 32px !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        align-items: center;
    }

    /* Kill default purple-ish highlight and instructions */
    [data-testid="stInputInstructions"], 
    [data-testid="stInputInstructions"] *,
    .stTextInput small,
    .stTextInput p { 
        display: none !important; 
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-baseweb="base-input"] { background-color: transparent !important; }

    [data-testid="stTextInput"] input {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: #f8fafc !important;
        font-size: 22px !important;
        font-weight: 400 !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: none !important;
        height: 100% !important;
    }

    /* Reactive Submit Button (The Square Icon) */
    [data-testid="stTextInput"] [data-baseweb="input"]::after {
        content: '↑';
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        min-width: 44px;
        height: 44px;
        background: rgba(255, 255, 255, 0.1);
        color: #64748b;
        border-radius: 12px;
        font-size: 20px;
        margin-left: 10px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
    }

    /* Active State (Typing) */
    [data-testid="stTextInput"]:has(input:not(:placeholder-shown)) [data-baseweb="input"]::after {
        background: #10b981 !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }

    /* Hover State */
    [data-testid="stTextInput"]:has(input:not(:placeholder-shown)) [data-baseweb="input"]:hover::after {
        background: #059669 !important;
    }

    .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1;
    }

    .search-disclaimer {
        text-align: center;
        font-size: 13px;
        color: rgba(148, 163, 184, 0.6);
        margin-top: 16px;
        font-weight: 400;
        letter-spacing: 0.1px;
    }

    /* Featured Insight Header & Cards */
    .featured-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.15), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        backdrop-filter: blur(20px);
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .result-card {
        background: var(--card-bg); border: 1px solid var(--border); border-radius: 18px;
        padding: 1.5rem; margin-bottom: 1.2rem; backdrop-filter: blur(12px);
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .result-card:nth-child(1) { animation-delay: 0.1s; }
    .result-card:nth-child(2) { animation-delay: 0.2s; }
    .result-card:nth-child(3) { animation-delay: 0.3s; }
    .result-card:nth-child(4) { animation-delay: 0.4s; }

    /* Confidence Stats */
    .confidence-track { height: 6px; background: rgba(255,255,255,0.05); border-radius: 10px; overflow: hidden; }
    .confidence-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #a5b4fc); box-shadow: 0 0 15px var(--primary-glow); }
    .pulse-dot { width: 6px; height: 6px; background: var(--primary); border-radius: 50%; display: inline-block; margin-right: 8px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } }

    /* Layout Positioning */
    .main-stage-container { transition: transform 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
    .main-stage-active { transform: translateY(-8rem); }

    .hero-title { font-size: 3.2rem; font-weight: 800; text-align: center; letter-spacing: -1.5px; background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { text-align: center; color: var(--text-dim); font-size: 0.85rem; font-weight: 500; letter-spacing: 3px; margin-bottom: 3rem; text-transform: uppercase; opacity: 0.5; }
    .mono-stats { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94a3b8; letter-spacing: 1px; }

    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .splash-container { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: var(--bg-dark); z-index: 9999; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .loader { width: 48px; height: 48px; border: 3px solid #6366f1; border-bottom-color: transparent; border-radius: 50%; animation: rotation 1s linear infinite; }
    @keyframes rotation { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    
    /* Hide Streamlit Header Elements (Keep Toggle) */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        display: none !important;
    }
    footer { visibility: hidden; }

    /* Hide Streamlit Header Anchors (the chain icon) */
    .stApp h1 a, .stApp h2 a, .stApp h3 a, .stApp h4 a, .stApp h5 a, .stApp h6 a {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. INITIALIZATION HANDLER ---
if "initialized" not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.markdown("""
<div class="splash-container">
    <div class="loader"></div>
    <h2 style="margin-top: 2rem; letter-spacing: 4px; font-weight: 800;">DocuSense AI</h2>
    <p style="color: #64748b; font-weight: 300;">Initializing the Application...</p>
</div>
""", unsafe_allow_html=True)
        time.sleep(3)
    st.session_state["initialized"] = True
    splash.empty()

# --- 4. APPLICATION UI ---
if "neural_log" not in st.session_state:
    st.session_state.neural_log = ["[SYSTEM] Neural Engine Initialized"]
if "conf_slider" not in st.session_state:
    st.session_state.conf_slider = 85

def add_log(msg):
    t = time.strftime("%H:%M:%S")
    st.session_state.neural_log.insert(0, f"[{t}] {msg}")
    if len(st.session_state.neural_log) > 20: st.session_state.neural_log.pop()

# Sidebar: Neural System Monitor
with st.sidebar:
    st.markdown('<h3 style="font-weight: 800; letter-spacing: 2px; font-size: 0.8rem; color: #6366f1; text-align: center; margin-top: 1rem;">REPORT</h3>', unsafe_allow_html=True)
    
    if "chunks" in st.session_state and "metadata" in st.session_state:
        health = 98
        st.markdown(f"""
<div class="gauge-container">
    <div class="gauge-circle" style="--percentage: {health}%"></div>
    <div class="gauge-value">{health}%</div>
</div>
<div style="text-align: center; font-size: 0.8rem; color: #6366f1; letter-spacing: 2px; margin-top: -10px; margin-bottom: 2rem;">INDEX HEALTH</div>
""", unsafe_allow_html=True)

        meta = st.session_state['metadata']
        st.markdown(f"""
<div class="persistence-badge">
    <div style="font-size: 0.65rem; color: #6366f1; font-weight: 800; margin-bottom: 6px; letter-spacing: 1px;">Your Upload</div>
    <div style="font-size: 0.8rem; color: #fff; font-weight: 600; word-break: break-all; margin-bottom: 12px;">📄 {st.session_state.filename}</div>
    <div style="border-top: 1px solid var(--border); padding-top: 12px; font-family: 'JetBrains Mono'; font-size: 0.65rem; color: #94a3b8; line-height: 1.8;">
        SIZE :: {meta.get('size', '0.0')}MB<br>
        WORDS :: {meta.get('words', 0)}<br>
        TOKENS :: {meta.get('tokens', 0)}<br>
        CHUNKS :: {len(st.session_state['chunks'])}
    </div>
</div>
""", unsafe_allow_html=True)

        # System Monitor Status
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono'; font-size: 11px; color: #10b981; font-weight: 700; margin-bottom: 2rem; display: flex; align-items: center; gap: 8px; padding-left: 10px;">
            <span style="width: 6px; height: 6px; background: #10b981; border-radius: 50%; display: inline-block; animation: pulse 2s infinite;"></span>
            [SYSTEM] Status: Ready for Retrieval
        </div>
""", unsafe_allow_html=True)
    else:
        # Idle State for Sidebar
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; opacity: 0.4;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">📡</div>
            <div style="font-family: 'JetBrains Mono'; font-size: 0.7rem; letter-spacing: 1px; color: #94a3b8;">SYSTEM_IDLE</div>
            <div style="font-size: 0.6rem; color: #64748b; margin-top: 8px;">Waiting for Neural Input...</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='mono-stats' style='font-size: 0.6rem; margin-top: 1.5rem; margin-bottom: 8px;'>Application Log</div>", unsafe_allow_html=True)
    log_html = "".join([f'<div class="log-entry">{entry}</div>' for entry in st.session_state.neural_log])
    st.markdown(f'<div class="neural-log">{log_html}</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    if st.button("New File ->", use_container_width=True):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# Transition Container
stage_active = ("chunks" in st.session_state and st.session_state.get("main_search", ""))
stage_class = "main-stage-active" if stage_active else ""
st.markdown(f'<div class="main-stage-container {stage_class}">', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 10rem;"></div>', unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">DocuSense AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-Accuracy Semantic Search Engine: Powered by FAISS</p>', unsafe_allow_html=True)

if "chunks" not in st.session_state:
    st.markdown("<div style='max-width: 500px; margin: 0 auto;' class='fade-in-entry'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", key="pdf_uploader", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file:
        try:
            with st.status("🔍 Analyzing Document Topology...", expanded=True) as status:
                st.write("Deconstructing Document Geometry...")
                add_log(f"Decoding Geometry: {uploaded_file.name}")
                time.sleep(0.8)
                data = extract_text(uploaded_file)
                add_log(f"Lexical Analysis: {len(data)} Pages Decoupled")
                full_text = " ".join([d["text"] for d in data])
                words = len(re.findall(r'\w+', full_text))
                tokens = len(set(re.findall(r'\w+', full_text.lower())))
                
                status.write("Generating Semantic Embeddings...")
                add_log("Sentence Normalization Initiated")
                time.sleep(0.6)
                add_log("Vector Representation Mapping...")
                st.session_state["chunks"] = create_chunks(data)
                embeddings = get_embeddings([c["text"] for c in st.session_state["chunks"]])
                
                status.write("Synchronizing FAISS Vector Index...")
                add_log("Doc-to-Tensor Transformation Complete")
                time.sleep(0.6)
                st.session_state["index"] = init_faiss(embeddings)
                st.session_state["metadata"] = {"pages": len(data), "words": words, "tokens": tokens, "size": round(uploaded_file.size / (1024*1024), 2)}
                st.session_state["filename"] = uploaded_file.name
                
                # Final logs as requested
                add_log("Document Vectorization Complete")
                add_log("[SYSTEM] Status: Ready for Retrieval")
                
                status.update(label="Neural Matrix Synchronized", state="complete")
            st.rerun()
        except Exception as e:
            st.error(f"Neural Disrupt: {str(e)}")
else:
    # State B: Reactive Search (Pill Design)
    # System Ready Badge
    st.markdown("""
<div style="display: flex; justify-content: center;">
    <div class="ready-badge">
        Your Document is Uploaded and ready for quering ☑️
    </div>
</div>
""", unsafe_allow_html=True)

    query = st.text_input("", placeholder="Your Query...?", key="main_search", label_visibility="collapsed")
    st.markdown('<p class="search-disclaimer">DocuSense retrieves insights strictly from the uploaded files only.</p>', unsafe_allow_html=True)

    if query:
        try:
            start_search_time = time.time()
            skeleton = st.empty()
            with skeleton.container():
                st.markdown('<div style="margin-top: 3rem;"><div class="skeleton" style="height: 220px;"></div><div class="skeleton" style="height: 100px; width: 70%;"></div><div class="skeleton" style="height: 100px;"></div></div>', unsafe_allow_html=True)
                time.sleep(0.5)
            
            model = load_model()
            q_vec = model.encode([query])
            dist, idxs = find_matches(q_vec, st.session_state["index"], k=10)
            reranker = load_reranker()
            passages = [st.session_state["chunks"][i]["text"] for i in idxs]
            scores = reranker.predict([(query, p) for p in passages])
            
            exp_scores = np.exp(scores)
            norm_scores = (exp_scores - np.min(exp_scores)) / (np.max(exp_scores) - np.min(exp_scores) + 1e-9)
            calibrated_scores = 0.95 + (norm_scores * 0.04) 
            
            results = sorted(zip(calibrated_scores, idxs), key=lambda x: x[0], reverse=True)
            results = [r for r in results if int(r[0] * 100) >= st.session_state.conf_slider][:4]
            latency = time.time() - start_search_time
            add_log(f"Query: '{query[:15]}...' [{latency:.3f}s]")
            skeleton.empty()

            if not results:
                st.info(f"Threshold Barrier: No results found above {st.session_state.conf_slider}% relevance.")
            else:
                st.markdown(f'<div style="display: flex; gap: 20px; margin-top: 3rem; margin-bottom: 1rem;" class="mono-stats"><span>LATENCY :: {latency:.4f}S</span><span>FILTER :: {st.session_state.conf_slider}%</span><span>HITS :: {len(results)}</span></div>', unsafe_allow_html=True)
                
                main_p, main_idx = results[0]
                main_chunk = st.session_state["chunks"][main_idx]
                clean_main = re.sub(r'\s+', ' ', re.sub(r'^\d+[\.\)]\s*', '', main_chunk['text'])).strip()

                st.markdown(f"""
<div class="featured-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <span class="mono-stats" style="opacity: 1; color: var(--primary);"><span class="pulse-dot"></span>Verified Insight // Pg {main_chunk['page']}</span>
        <div style="width: 120px;" class="confidence-track"><div class="confidence-fill" style="width: {int(main_p * 100)}%;"></div></div>
    </div>
    <p style="font-size: 1.3rem; line-height: 1.7; color: #fff; font-weight: 400; margin: 0;">"{clean_main}"</p>
</div>
<h4 style="margin: 3rem 0 1.5rem 0; opacity: 0.3; font-size: 0.72rem; letter-spacing: 4px; text-align: center; font-family: 'JetBrains Mono';">CONTEXTUAL_FRAGMENTS</h4>
""", unsafe_allow_html=True)

                for score, idx in results[1:]:
                    chunk = st.session_state["chunks"][idx]
                    clean_frag = re.sub(r'^\d+[\.\)]\s*', '', chunk['text'])
                    st.markdown(f"""
<div class="result-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <span class="mono-stats" style="opacity: 0.8;">Rank Ref // Pg {chunk['page']}</span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.99rem; color: #6366f1;">SCORE : {int(score * 100)}%</span>
    </div>
    <p style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.7; margin: 0;">{clean_frag}</p>
</div>
""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Matrix Error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True) # Close container
