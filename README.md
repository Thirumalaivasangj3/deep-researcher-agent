# 🔍 Deep Researcher Agent

A local AI-powered research assistant **.  
It indexes documents (PDFs/TXT), generates embeddings locally, and allows users to query and retrieve relevant chunks with semantic search.

## 🚀 Features
- Upload multiple PDFs or text files
- Text extraction + chunking into 300-word passages
- Local embeddings using Sentence-Transformers (no external APIs)
- Query answering with cosine similarity
- Streamlit web interface for easy demo
- Saves and loads local index

## 🛠️ Tech Stack
- Python 3.9+
- Streamlit (UI)
- Sentence-Transformers (embeddings)
- scikit-learn (cosine similarity)
- pdfplumber (PDF parsing)

## ⚡ Quick Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/deep-researcher-agent.git
cd deep-researcher-agent

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
