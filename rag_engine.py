from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict[str, Any]]


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class RagPdfBot:
    def __init__(
        self,
        embedding_model_name: str = "local-hash-embeddings",
        llm_model_name: str = "extractive",
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

        embeddings = LocalHashEmbeddings()
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
