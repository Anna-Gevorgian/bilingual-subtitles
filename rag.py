"""STEP 6 — RAG Enhancement.

Builds a FAISS vector store from the knowledge base (subtitle rules, Armenian
style guide, Armenian punctuation, and an optional domain glossary) and exposes
a tiny retriever. The retrieved text is injected into the LLM prompts for
Step 4 (English cleanup) and Step 5 (Armenian translation), so both calls are
conditioned on house-style context — a hard requirement of the spec.

FAISS is chosen over Chroma because it is in-memory, dependency-light, and
needs no server; for a small KB it is more than enough.
կարդում է .md / .txt
բաժանում մասերի
պահում vector database-ում (FAISS)
Եվ հետո տալիս է LLM-ին համապատասխան context
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from config import Config
from llm import get_embeddings

logger = logging.getLogger(__name__)


class Retriever:
    """Thin wrapper around a FAISS vector store."""

    def __init__(self, vectorstore, top_k: int):
        self._vs = vectorstore
        self._k = top_k

    def context_for(self, query: str) -> str:
        """Return the concatenated text of the top-k relevant chunks."""
        if self._vs is None:
            return ""
        docs = self._vs.as_retriever(
            search_kwargs={"k": self._k}
        ).invoke(query)
        return "\n\n---\n\n".join(d.page_content.strip() for d in docs)


def _load_documents(config: Config):
    from langchain_core.documents import Document

    kb_dir = Path(config.knowledge_base_dir)
    paths: List[Path] = []
    if kb_dir.is_dir():
        paths += sorted(p for p in kb_dir.glob("**/*")
                        if p.suffix.lower() in {".md", ".txt"})
    if config.glossary_path and Path(config.glossary_path).is_file():
        paths.append(Path(config.glossary_path))

    docs = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read KB file %s: %s", p, exc)
            continue
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": p.name}))
    return docs


def build_retriever(config: Config | None = None) -> Retriever:
    """Construct the FAISS-backed retriever from the knowledge base.

    If the KB is empty or embeddings are unavailable, returns a no-op retriever
    that yields empty context (the pipeline still runs; prompts just carry the
    static rules baked into refine.py).
    """
    config = config or Config()
    docs = _load_documents(config)
    if not docs:
        logger.warning("Knowledge base is empty; RAG context will be empty.")
        return Retriever(None, config.rag_top_k)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.rag_chunk_size,
        chunk_overlap=config.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("RAG: %d docs -> %d chunks; embedding...", len(docs), len(chunks))

    embeddings = get_embeddings(config)
    vs = FAISS.from_documents(chunks, embeddings)
    return Retriever(vs, config.rag_top_k)
