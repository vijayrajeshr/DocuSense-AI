import PyPDF2

def extract_text(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text

def create_chunks(text, chunk_size=500):
    """Splits text into smaller strings for better retrieval precision."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return [c for c in chunks if len(c) > 20]