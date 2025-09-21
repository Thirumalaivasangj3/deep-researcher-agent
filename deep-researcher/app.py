
import streamlit as st
from utils.text_processing import extract_text_from_pdf, read_txt, chunk_text
from utils.embeddings import load_model, build_index, save_index, load_index, query_index

st.set_page_config(page_title="Deep Researcher Agent", layout="wide")

st.title("🔍 Deep Researcher Agent (Local Embeddings)")


model = load_model()


st.sidebar.header("📂 Indexing")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", accept_multiple_files=True, type=["pdf", "txt"])

if st.sidebar.button("Build index from uploaded files"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file.")
    else:
        with st.spinner("Building index... This may take a while."):
            docs, metas, embeddings = build_index(uploaded_files, model)
            if docs is None:
                st.error("No text extracted from uploaded files.")
            else:
                save_index("index/local_index", docs, metas, embeddings)
                st.success(f"Indexed {len(docs)} chunks from {len(uploaded_files)} files.")
                st.sidebar.write("Index saved in `index/` folder.")

st.sidebar.header("Or load existing index")
if st.sidebar.button("Load saved index"):
    try:
        docs, metas, embeddings = load_index("index/local_index")
        st.sidebar.success(f"Loaded index with {len(docs)} chunks.")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

# Query section
st.header("💡 Ask Questions")
query = st.text_input("Enter your question or keywords")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    try:
        if "docs" not in locals() or docs is None:
            try:
                docs, metas, embeddings = load_index("index/local_index")
            except:
                st.error("No index available. Please build or load one first.")
                st.stop()

        results = query_index(query, model, docs, metas, embeddings, top_k=top_k)

        st.markdown("### 📑 Results")
        for r in results:
            st.markdown(f"**Source:** {r['meta']['source']} — chunk #{r['meta']['chunk_id']} — score {r['score']:.3f}")
            st.write(r['text'][:1000] + ("..." if len(r['text']) > 1000 else ""))
            st.write("---")

    except Exception as e:
        st.error(f"Search error: {e}")
