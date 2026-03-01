import streamlit as st
from src.processor import extract_text, create_chunks
from src.embedder import load_model, get_embeddings
from src.search_engine import init_faiss, find_matches

# --- PAGE CONFIG ---
st.set_page_config(page_title="DocuSense-AI", page_icon="🧠", layout="centered")

st.title("🧠 DocuSense-AI")
st.caption("Private Semantic Document Search | Powered by FAISS")

# --- SIDEBAR & PERSONAL NOTE ---
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    st.divider()
    st.subheader("Personal Note")
    st.info("I built DocuSense-AI to move from traditional TF-IDF keyword matching to Neural Semantic Search using FAISS, ensuring data privacy by running 100% locally.")

# --- MAIN LOGIC ---
if uploaded_file:
    # Use session_state so we don't re-process the PDF on every click
    if "index" not in st.session_state:
        with st.spinner("Analyzing document with neural embeddings..."):
            # 1. Process
            raw_text = extract_text(uploaded_file)
            chunks = create_chunks(raw_text)
            
            # 2. Embed & Index
            embeddings = get_embeddings(chunks)
            index = init_faiss(embeddings)
            
            # 3. Store in session
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
        st.success("Document Indexed Successfully!")

    # --- CHAT INTERFACE ---
    user_query = st.text_input("Ask a question about your document:", placeholder="e.g., What are the key findings?")

    if user_query:
        model = load_model()
        query_vector = model.encode([user_query])
        
        # Search the FAISS index
        best_indices = find_matches(query_vector, st.session_state["index"])
        
        st.write("### Top Semantic Matches:")
        for i, idx in enumerate(best_indices):
            with st.expander(f"Result {i+1}"):
                st.write(st.session_state["chunks"][idx])
else:
    st.info("Please upload a PDF file in the sidebar to begin.")