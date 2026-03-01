from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_model():
    """Loads a local MiniLM model for semantic vector generation."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text_list):
    """Converts text chunks into numerical vectors (embeddings)."""
    model = load_model()
    return model.encode(text_list)