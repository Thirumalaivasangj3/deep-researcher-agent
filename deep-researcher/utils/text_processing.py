
import pdfplumber
import io

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file"""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                p = page.extract_text()
                if p:
                    text += p + "\n"
    except Exception as e:
        print(f"PDF read error: {e}")
    return text

def read_txt(file_bytes):
    """Read text from a TXT file"""
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except:
        return ""

def chunk_text(text, words_per_chunk=300, overlap=50):
    """Split text into overlapping word chunks"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + words_per_chunk]
        chunks.append(" ".join(chunk))
        i += words_per_chunk - overlap
    return chunks
