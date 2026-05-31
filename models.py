"""Shared data structures used across the pipeline.

Kept dependency-free (stdlib only) so every module can import these without
pulling in heavy packages (whisper, torch, langchain, faiss).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Word:
    """A single ASR token with its time span (seconds)."""
    text: str
    start: float
    end: float


@dataclass
class Cue:
    """One subtitle cue (an SRT block).

    `lines` holds the already-wrapped display lines (1 or 2 of them).
    Timestamps are in seconds; SRT serialization happens in srt_io.py.
    """
    start: float
    end: float
    lines: List[str] = field(default_factory=list)
    index: int = 0  # 1-based; assigned during finalization

    @property
    def text(self) -> str:
        """Single-line text (lines joined by space) — used as LLM input."""
        return " ".join(line.strip() for line in self.lines).strip()

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    @property
    def char_count(self) -> int:
        """Visible characters used for CPS (newlines excluded, spaces kept)."""
        return len(self.text)

    def cps(self) -> float:
        d = self.duration
        return self.char_count / d if d > 0 else float("inf")


# Severity levels for validation findings.
CRITICAL = "CRITICAL"
WARNING = "WARNING"


@dataclass
class Violation:
    track: str          # "english" or "armenian"
    cue_index: int      # 1-based cue index (0 == file-level)
    kind: str           # e.g. "timing_overlap", "line_too_long"
    severity: str       # CRITICAL | WARNING
    detail: str         # human-readable description

    @property
    def flagged_for_review(self) -> bool:
        return self.severity == CRITICAL
