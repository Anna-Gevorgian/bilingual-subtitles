"""Central configuration for the subtitle pipeline.

Every tunable number lives here so the behavior of the whole pipeline can be
adjusted from one place (and overridden from the Streamlit UI).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo-relative default location of the RAG knowledge base.
_DEFAULT_KB = str(Path(__file__).resolve().parent / "knowledge_base")


@dataclass
class Config:
    # ---- Subtitle layout / quality limits (the "house style") -------------
    max_line_length: int = 42          # characters per line
    max_lines_per_cue: int = 2
    max_cps: float = 21.0              # reading speed ceiling (chars/second)
    min_cue_duration: float = 1.0      # seconds; short cues are extended if room
    max_cue_duration: float = 6.0      # seconds; forces a cue boundary
    min_gap: float = 0.04              # seconds between consecutive cues
    # When a translated cue can only end at a sentence-ish boundary, only break
    # mid-cue once the text is at least this long (avoids tiny dangling cues).
    sentence_break_min_chars: int = 28

    # ---- Source video constraints -----------------------------------------
    max_video_seconds: float = 180.0   # spec target: <= 3 minutes
    enforce_duration: bool = False     # True -> raise instead of warn if over

    # ---- ASR (Whisper) -----------------------------------------------------
    whisper_model: str = "base"        # tiny|base|small|medium|large-v3
    # Source language hint. None = auto-detect. The pipeline assumes the
    # *source* audio is English (Step 4 cleans English, Step 5 translates).
    source_language: str | None = "en"

    # ---- LLM (LangChain) ---------------------------------------------------
    # Set to whichever chat model you have access to. A stronger model
    # noticeably improves Armenian translation quality.
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2       # low: we want faithful, not creative
    embedding_model: str = "text-embedding-3-small"
    # Cues per LLM request. For <=3-min clips the whole file is usually one
    # batch, which maximizes cross-cue terminology consistency.
    llm_batch_size: int = 50

    # ---- RAG ---------------------------------------------------------------
    knowledge_base_dir: str = field(default_factory=lambda: _DEFAULT_KB)
    glossary_path: str | None = None   # optional extra domain glossary file
    rag_chunk_size: int = 600
    rag_chunk_overlap: int = 80
    rag_top_k: int = 4

    # ---- I/O ---------------------------------------------------------------
    output_dir: str = "output"
    keep_intermediate: bool = False    # keep extracted .wav etc.

    @property
    def cue_char_budget(self) -> int:
        """Soft upper bound on a cue's character count before forcing a break."""
        return self.max_line_length * self.max_lines_per_cue
