# PDF RAG Assistant

An interactive PDF question-answering app built with **Streamlit**, **LangChain**, and **FAISS**. Upload a PDF, index its content, ask questions, and get page-cited answers from the retrieved document text.

## Live Demo

Try the deployed app here:

**https://thageekiestone-pdf-rag-assistant-app-athfte.streamlit.app/**

## What It Does

- Upload a PDF directly from the browser.
- Extract and split PDF text into overlapping chunks.
- Generate lightweight local hashing embeddings with no external model download.
- Store document vectors in FAISS for fast similarity search.
- Use MMR retrieval to reduce duplicate context chunks.
- Return extractive answers from the most relevant PDF sentences.
- Show page references and source snippets for transparency.
- Run locally or deploy on Streamlit Cloud.

## Tech Stack

| Area | Tools |
| --- | --- |
| UI | Streamlit |
| RAG workflow | LangChain |
| Vector search | FAISS |
| PDF parsing | PyPDF |
| Embeddings | Local hashing embeddings |
| Deployment | Streamlit Cloud |

## How It Works

1. The user uploads a PDF.
2. `PyPDFLoader` extracts page text from the file.
3. LangChain splits the text into overlapping chunks.
4. Local hashing embeddings convert chunks into vectors.
5. FAISS indexes the vectors.
6. User questions are embedded and matched against the PDF index.
7. The app returns the most relevant sentences with page citations.

## Run Locally

Clone the repository:

```powershell
git clone https://github.com/ThaGeekiestOne/pdf-rag-assistant.git
cd pdf-rag-assistant
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL, upload a PDF, click **Index PDF**, and start asking questions.

## Project Structure

```text
pdf-rag-assistant/
|-- app.py                 # Streamlit user interface
|-- rag_engine.py          # PDF loading, chunking, FAISS retrieval, answer generation
|-- rag.py                 # Compatibility wrapper
|-- requirements.txt       # Python dependencies
|-- packages.txt           # Streamlit Cloud system dependency
|-- runtime.txt            # Python runtime hint
`-- .streamlit/config.toml # Streamlit deployment config
```

## Deployment Notes

This version is optimized for Streamlit Cloud reliability. It avoids large runtime LLM downloads and uses local embeddings plus extractive answers, which makes deployment faster and less memory-intensive.

## Future Improvements

- Add optional Hugging Face or API-based LLM answer generation.
- Persist FAISS indexes for repeated use.
- Support multiple PDFs at once.
- Add conversational memory.
- Add better ranking with rerankers or hybrid keyword/vector search.

## Author

Built by [ThaGeekiestOne](https://github.com/ThaGeekiestOne).
