import PyPDF2

def extract_text(pdf_file):
    """Extracts text with page mapping for context retention."""
    pages_data = []
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                cleaned = " ".join(content.split())
                pages_data.append({"page": i + 1, "text": cleaned})
    except Exception as e:
        print(f"Extraction error: {e}")
    return pages_data

def create_chunks(pages_data, chunk_size=600, overlap=120):
    """Splits pages into character-based overlapping chunks while preserving page metadata."""
    chunks = []
    for data in pages_data:
        text = data["text"]
        if not text:
            continue
            
        # Character-based sliding window
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to snap to the nearest space to avoid cutting words
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space != -1:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space
            
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "text": chunk_text.strip(),
                    "page": data["page"],
                    "char_count": len(chunk_text)
                })
            
            start = end - overlap
            if start < 0: start = 0
            if end >= len(text): break
                
    return chunks

