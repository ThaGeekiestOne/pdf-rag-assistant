from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


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

        documents = self.vectorstore.max_marginal_relevance_search(
            question,
            k=top_k,
            fetch_k=max(12, top_k * 4),
            lambda_mult=0.45,
        )
        answer = self._generate_answer(
            question=question,
            documents=documents,
            answer_style=answer_style,
        )

        return RagAnswer(
            answer=answer,
            sources=self._format_sources(documents),
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

    def _generate_answer(
        self,
        question: str,
        documents: list[Document],
        answer_style: str,
    ) -> str:
        ranked_sentences = self._rank_sentences(question, documents)
        if not ranked_sentences:
            return "I do not know from the provided PDF."

        limits = {"Concise": 3, "Balanced": 5, "Detailed": 8}
        limit = limits.get(answer_style, limits["Balanced"])
        selected = ranked_sentences[:limit]

        if answer_style == "Detailed":
            return "\n".join(f"- {sentence} (p. {page})" for sentence, page, _ in selected)

        return " ".join(f"{sentence} (p. {page})" for sentence, page, _ in selected)

    @staticmethod
    def _rank_sentences(question: str, documents: list[Document]) -> list[tuple[str, int, int]]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
        terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(term) > 2 and term not in stop_words
        }

        scored = []
        seen = set()
        for doc in documents:
            page = int(doc.metadata.get("page", 0)) + 1
            sentences = re.split(r"(?<=[.!?])\s+|\n+", doc.page_content)
            for sentence in sentences:
                clean = " ".join(sentence.split())
                if len(clean) < 30 or clean in seen:
                    continue
                seen.add(clean)
                sentence_terms = set(re.findall(r"[a-zA-Z0-9]+", clean.lower()))
                score = len(terms & sentence_terms)
                if score:
                    scored.append((clean, page, score))

        return sorted(scored, key=lambda item: item[2], reverse=True)

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
