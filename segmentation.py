"""STEP 3 — Subtitle Segmentation.

Turns a flat list of timed `Word`s into SRT cues that respect the house style:

  * <= max_lines_per_cue lines, each <= max_line_length characters
  * no overlapping timestamps (monotonic, with a small inter-cue gap)
  * breaks only at word boundaries, preferring sentence/clause punctuation
    (so we never cut mid-word and rarely cut mid-phrase)

Design notes
------------
Cue building is a greedy single pass. We keep appending words to the current
cue while the text still fits in two wrapped lines and the cue is not too long
in time. When the next word would overflow, we close the cue — backing up to
the last punctuation boundary if there is one, so the break lands at a natural
place. Because we only ever appended words that *did* fit, the closed cue is
always layout-valid.

Line wrapping (`wrap_balanced`) does a balanced 2-line split that minimizes the
longer line and prefers to break after punctuation — the standard subtitle look.
This module is stdlib-only and unit-tested.
"""
from __future__ import annotations

from typing import List, Optional

from config import Config
from models import Cue, Word

# Punctuation that marks a "good enough" place to end a line/cue.
_SENTENCE_END = (".", "!", "?", "…", "։", "՜", "՞")
_CLAUSE_END = _SENTENCE_END + (",", ";", ":", "—", "–", "՝")


# --------------------------------------------------------------------------- #
# Line wrapping
# --------------------------------------------------------------------------- #
def greedy_wrap(text: str, width: int) -> List[str]:
    """Classic greedy word wrap. Never splits a word."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def fits(text: str, width: int, max_lines: int) -> bool:
    """True if `text` wraps into at most `max_lines` lines of width `width`."""
    return len(greedy_wrap(text, width)) <= max_lines


def wrap_balanced(text: str, width: int, max_lines: int = 2) -> List[str]:
    """Wrap into <= max_lines lines, balancing length and favouring punctuation.

    If the text genuinely cannot fit in `max_lines` lines (e.g. an over-long
    translated cue), we return the greedy wrap, which may exceed `max_lines`.
    The validator will flag that case for human review rather than silently
    dropping text.
    """
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= width:
        return [text]

    tokens = text.split()
    # Try a 2-line balanced split (the common case).
    if max_lines == 2:
        best: Optional[tuple] = None  # (worse_metric, k)
        for k in range(1, len(tokens)):
            l1 = " ".join(tokens[:k])
            l2 = " ".join(tokens[k:])
            if len(l1) > width or len(l2) > width:
                continue
            longer = max(len(l1), len(l2))
            ends_punct = l1.rstrip().endswith(_CLAUSE_END)
            # Lower is better: short longer-line, prefer punctuation break,
            # then prefer a split near the middle.
            metric = (longer, 0 if ends_punct else 1, abs(len(l1) - len(l2)))
            if best is None or metric < best[0]:
                best = (metric, k)
        if best is not None:
            k = best[1]
            return [" ".join(tokens[:k]), " ".join(tokens[k:])]

    # Could not fit in max_lines (or max_lines > 2): fall back to greedy.
    return greedy_wrap(text, width)


# --------------------------------------------------------------------------- #
# Cue building
# --------------------------------------------------------------------------- #
def _join(words: List[Word]) -> str:
    return " ".join(w.text.strip() for w in words).strip()


def _last_punct_break(words: List[Word], width: int, max_lines: int
                      ) -> Optional[int]:
    """Largest interior index k (1..len-1) such that words[k-1] ends with
    clause punctuation AND words[:k] still fits the layout. Else None.
    """
    chosen = None
    for k in range(1, len(words)):
        if words[k - 1].text.rstrip().endswith(_CLAUSE_END):
            if fits(_join(words[:k]), width, max_lines):
                chosen = k
    return chosen


def _make_cue(words: List[Word], config: Config) -> Cue:
    text = _join(words)
    lines = wrap_balanced(text, config.max_line_length, config.max_lines_per_cue)
    return Cue(start=words[0].start, end=words[-1].end, lines=lines)


def build_cues(words: List[Word], config: Config | None = None) -> List[Cue]:
    """Group timed words into layout-valid, non-overlapping cues."""
    config = config or Config()
    width = config.max_line_length
    max_lines = config.max_lines_per_cue

    cues: List[Cue] = []
    buf: List[Word] = []
    i, n = 0, len(words)

    while i < n:
        w = words[i]

        if buf:
            candidate = buf + [w]
            overflow_layout = not fits(_join(candidate), width, max_lines)
            overflow_time = (w.end - buf[0].start) > config.max_cue_duration
            if overflow_layout or overflow_time:
                cut = _last_punct_break(buf, width, max_lines)
                if cut is not None and 0 < cut < len(buf):
                    cues.append(_make_cue(buf[:cut], config))
                    buf = buf[cut:]
                else:
                    cues.append(_make_cue(buf, config))
                    buf = []
                continue  # reconsider w against the (smaller/empty) buffer

        # Append w (always makes progress; a lone over-long word is accepted
        # and later flagged by the validator).
        buf.append(w)
        i += 1

        # Natural sentence-end break once the cue is reasonably full.
        if (w.text.rstrip().endswith(_SENTENCE_END)
                and len(_join(buf)) >= config.sentence_break_min_chars):
            cues.append(_make_cue(buf, config))
            buf = []

    if buf:
        cues.append(_make_cue(buf, config))

    return enforce_timing(cues, config)


def enforce_timing(cues: List[Cue], config: Config | None = None) -> List[Cue]:
    """Make timestamps strictly non-overlapping and assign 1-based indices.

    - Push each cue's start to at least prev.end + min_gap.
    - Keep end > start.
    - Gently extend very short cues into the gap before the next cue (better
      readability) without ever crossing into the next cue.
    """
    config = config or Config()
    gap = config.min_gap
    for idx, cue in enumerate(cues):
        if idx > 0:
            min_start = cues[idx - 1].end + gap
            if cue.start < min_start:
                cue.start = min_start
        if cue.end <= cue.start:
            cue.end = cue.start + 0.4  # minimal visible flash

    # Optional extension pass for short cues.
    for idx, cue in enumerate(cues):
        if cue.duration < config.min_cue_duration:
            ceiling = (cues[idx + 1].start - gap) if idx + 1 < len(cues) else None
            target = cue.start + config.min_cue_duration
            cue.end = min(target, ceiling) if ceiling is not None else target
            if cue.end <= cue.start:
                cue.end = cue.start + 0.4

    for idx, cue in enumerate(cues, start=1):
        cue.index = idx
    return cues


def rewrap_cue_text(text: str, config: Config | None = None) -> List[str]:
    """Re-wrap arbitrary (e.g. LLM-edited or translated) text into cue lines."""
    config = config or Config()
    return wrap_balanced(text, config.max_line_length, config.max_lines_per_cue)
