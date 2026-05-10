# RAG PDF Chatbot

A PDF question-answering chatbot built with LangChain, FAISS, Hugging Face embeddings, and an open-source Hugging Face LLM.

## Features

- Upload and index a PDF from the browser.
- Split PDF text into overlapping chunks.
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store and search vectors with FAISS.
- Answer questions with `google/flan-t5-small` by default.
- Use MMR retrieval to reduce duplicate chunks and improve context quality.
- Format retrieved chunks with page labels for page-aware answers.
- Show source page snippets for each answer.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local Streamlit URL in your browser, upload a PDF, click **Index PDF**, and start asking questions.

## Model Configuration

You can change models from the sidebar or through environment variables:

```powershell
$env:EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
$env:LLM_MODEL="google/flan-t5-small"
streamlit run app.py
```

For stronger answers, try `google/flan-t5-base` or another compatible text-to-text model that your machine can run.

## Improving Answer Quality

The default model is intentionally small so it can run on most laptops. If answers feel weak, first try:

```powershell
$env:LLM_MODEL="google/flan-t5-base"
streamlit run app.py
```

Better answers usually come from improving retrieval and using a stronger instruction model. Actual fine-tuning is only useful when you have a dataset of example questions and ideal answers.
