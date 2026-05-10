# RAG PDF Chatbot

A PDF question-answering chatbot built with LangChain, FAISS, local hashing embeddings, and a deployment-friendly extractive answer engine.

## Features

- Upload and index a PDF from the browser.
- Split PDF text into overlapping chunks.
- Generate local hashing embeddings with no model download.
- Store and search vectors with FAISS.
- Answer questions from the most relevant retrieved PDF sentences.
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

Run the app:

```powershell
streamlit run app.py
```

## Improving Answer Quality

The deployed version uses local embeddings and extractive answers for reliability on Streamlit Cloud. Better answers usually come from improving chunk size, chunk overlap, retrieval count, and PDF text quality. Actual model fine-tuning is only useful when you have a dataset of example questions and ideal answers.
