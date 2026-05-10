from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict[str, Any]]


class RagPdfBot:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-small",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore: FAISS | None = None
        self._llm: HuggingFacePipeline | None = None

    def index_pdf(self, pdf_path: str | Path) -> dict[str, int]:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = self._split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

        return {"pages": len(pages), "chunks": len(chunks)}

    def answer(self, question: str, top_k: int = 5, answer_style: str = "Balanced") -> RagAnswer:
        if self.vectorstore is None:
            raise RuntimeError("No PDF has been indexed yet.")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": max(12, top_k * 4),
                "lambda_mult": 0.45,
            },
        )
        chain = RetrievalQA.from_chain_type(
            llm=self._get_llm(),
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "document_prompt": self._document_prompt(),
                "prompt": self._prompt(answer_style),
            },
        )
        result = chain.invoke({"query": question})

        return RagAnswer(
            answer=result["result"].strip(),
            sources=self._format_sources(result.get("source_documents", [])),
        )

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            page = int(chunk.metadata.get("page", 0)) + 1
            chunk.metadata["page_label"] = str(page)
        return chunks

    def _get_llm(self) -> HuggingFacePipeline:
        if self._llm is not None:
            return self._llm

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
        task = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.08,
            device=0 if torch.cuda.is_available() else -1,
        )
        self._llm = HuggingFacePipeline(pipeline=task)
        return self._llm

    @staticmethod
    def _prompt(answer_style: str) -> PromptTemplate:
        style_rules = {
            "Concise": "Answer in 2-4 direct sentences.",
            "Detailed": "Answer with a short explanation and bullet points when useful.",
            "Balanced": "Answer clearly in one short paragraph, using bullets only when helpful.",
        }
        style = style_rules.get(answer_style, style_rules["Balanced"])
        template = """You are a careful PDF question-answering assistant.
Use only the supplied PDF context. Do not use outside knowledge.
If the context does not contain enough information, say: "I do not know from the provided PDF."
When facts come from the PDF, include page references like (p. 3).
{style}

Context:
{context}

Question: {question}

Answer:"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"],
            partial_variables={"style": style},
        )

    @staticmethod
    def _document_prompt() -> PromptTemplate:
        return PromptTemplate(
            template="[Page {page_label}]\n{page_content}",
            input_variables=["page_content", "page_label"],
        )

    @staticmethod
    def _format_sources(documents: list[Document]) -> list[dict[str, Any]]:
        sources = []
        seen = set()
        for doc in documents:
            page = int(doc.metadata.get("page", 0)) + 1
            preview = " ".join(doc.page_content.split())
            key = (page, preview[:120])
            if key in seen:
                continue
            seen.add(key)
            sources.append({"page": page, "preview": preview[:500]})
        return sources
