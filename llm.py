"""LLM / embeddings factory (LangChain).

Isolating model construction here means switching providers is a one-line
change. Defaults target OpenAI (the natural pairing with Whisper), but any
LangChain chat model works for the refinement/translation steps.

To switch to Anthropic, for example:
    pip install langchain-anthropic
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model="claude-...", temperature=cfg.llm_temperature)
"""
from __future__ import annotations

import os

from config import Config


def _require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or enter it in the UI "
            "before running the LLM/RAG steps."
        )


def get_chat_model(config: Config):
    """Return a LangChain chat model instance."""
    _require_openai_key()
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)


def get_embeddings(config: Config):
    """Return a LangChain embeddings instance for the RAG vector store.

    For a fully local/offline setup, swap this for a HuggingFace embedding:
        pip install langchain-huggingface
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    """
    _require_openai_key()
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.embedding_model)
