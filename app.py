import os
import tempfile
from pathlib import Path

import streamlit as st

from rag_engine import RagPdfBot


APP_VERSION = "2026-05-10-local-embeddings"


st.set_page_config(
    page_title="RAG PDF Chatbot",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_bot(embedding_model: str, llm_model: str, chunk_size: int, chunk_overlap: int) -> RagPdfBot:
    return RagPdfBot(
        embedding_model_name=embedding_model,
        llm_model_name=llm_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def save_upload(uploaded_file) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_pdf_"))
    file_path = tmp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


with st.sidebar:
    st.title("PDF RAG")
    st.caption(f"Build: {APP_VERSION}")
    st.caption("Upload a PDF, index it with FAISS, then ask questions.")

    embedding_model = st.text_input(
        "Embedding model",
        value=os.getenv("EMBEDDING_MODEL", "local-hash-embeddings"),
        disabled=True,
    )
    llm_model = os.getenv("LLM_MODEL", "extractive")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, 20)
    top_k = st.slider("Retrieved chunks", 2, 10, 5)
    answer_style = st.selectbox(
        "Answer style",
        ["Balanced", "Concise", "Detailed"],
        index=0,
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    index_button = st.button("Index PDF", type="primary", disabled=uploaded_file is None)


st.title("RAG PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

bot = get_bot(embedding_model, llm_model, chunk_size, chunk_overlap)

if index_button and uploaded_file:
    with st.spinner("Reading, chunking, embedding, and indexing PDF..."):
        pdf_path = save_upload(uploaded_file)
        stats = bot.index_pdf(pdf_path)
        st.session_state.indexed_file = uploaded_file.name
        st.session_state.messages = []
    st.success(
        f"Indexed {st.session_state.indexed_file}: "
        f"{stats['pages']} pages, {stats['chunks']} chunks."
    )

if st.session_state.indexed_file:
    st.info(f"Current document: {st.session_state.indexed_file}")
else:
    st.warning("Upload and index a PDF to start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Page {source['page']}**  \n"
                        f"{source['preview']}"
                    )

question = st.chat_input("Ask a question about the PDF", disabled=not st.session_state.indexed_file)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = bot.answer(question, top_k=top_k, answer_style=answer_style)
        st.markdown(result.answer)
        with st.expander("Sources"):
            for source in result.sources:
                st.markdown(
                    f"**Page {source['page']}**  \n"
                    f"{source['preview']}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
        }
    )
