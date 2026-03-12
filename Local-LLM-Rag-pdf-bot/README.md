# DocuSense AI - Local LLM RAG PDF Bot

A simple Streamlit app for querying PDFs using a local LLM (Llama3.2) with RAG.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure Ollama is running with Llama3.2 and nomic-embed-text models.

## Usage

Run the app:
```
streamlit run app.py

```

Upload a PDF, click "Process Document", then ask questions about the content.