#Please go through the readme.md file before running.

import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
import tempfile


# --- Page Config ---
st.set_page_config(page_title="DocuSense AI - Local LLM", layout="wide")
st.title("DocuSense AI - Local - Llama3.2")

# --- 1. Initialize Models ---
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. Sidebar: File Upload ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_button = st.button("Process Document")

# --- 3. Processing Logic ---
if uploaded_file and process_button:
    with st.spinner("Analyzing document..."):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Load and split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Create Vector Store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("Document ready!")
        
        # Cleanup
        os.remove(tmp_path)

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is this document about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Setup RAG Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=st.session_state.retriever
                )
                response = qa_chain.invoke(prompt)
                full_response = response["result"]
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("Please upload and process a document first!")
        
