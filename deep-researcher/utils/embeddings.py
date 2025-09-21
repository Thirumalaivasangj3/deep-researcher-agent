
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_model():
    """Load sentence transformer model (cached by Streamlit in app.py)"""
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(files, model):
    """Build embeddings index from uploaded files"""
    from utils.text_processing import extract_text_from_pdf, read_txt, chunk_text

    docs, metas = [], []
    for f in files:
        name = f.name
        b = f.read()
        if name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(b)
        else:
            text = read_txt(b)

        if not text.strip():
            continue

        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            docs.append(c)
            metas.append({"source": name, "chunk_id": idx})

    if not docs:
        return None, None, None

    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    return docs, metas, embeddings

def save_index(path_prefix, docs, metas, embeddings):
    """Save docs, metadata, and embeddings to disk"""
    with open(f"{path_prefix}_meta.pkl", "wb") as f:
        pickle.dump({"docs": docs, "metas": metas}, f)
    np.save(f"{path_prefix}_emb.npy", embeddings)

def load_index(path_prefix):
    """Load docs, metadata, and embeddings from disk"""
    with open(f"{path_prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    embeddings = np.load(f"{path_prefix}_emb.npy")
    return meta["docs"], meta["metas"], embeddings

def query_index(query, model, docs, metas, embeddings, top_k=5):
    """Search index for most relevant chunks to query"""
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for ix in top_idx:
        results.append({"score": float(sims[ix]), "text": docs[ix], "meta": metas[ix]})
    return results
