"""STEP 2 — Speech Recognition (ASR).

Transcribes audio with openai-whisper. We request **word-level timestamps**
because they are what makes Step 3 able to break cues at word boundaries with
accurate timing (instead of guessing splits inside a coarse Whisper segment).

Returns a flat list of `Word` objects. If, for some reason, word timestamps are
unavailable, we fall back to segment-level timing in `segments_to_words()`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from config import Config
from models import Word

logger = logging.getLogger(__name__)


def transcribe(audio_path: str, config: Config | None = None) -> Dict[str, Any]:
    """Run Whisper and return its raw result dict (segments + words).

    Whisper is imported lazily so the rest of the pipeline (and the unit tests
    for segmentation/validation) do not require torch to be installed.
    """
    config = config or Config()
    import whisper  # lazy import (heavy: pulls in torch)

    logger.info("Loading Whisper model '%s'", config.whisper_model)
    model = whisper.load_model(config.whisper_model)

    logger.info("Transcribing %s", audio_path)
    result = model.transcribe(
        audio_path,
        language=config.source_language,   # None -> auto-detect
        task="transcribe",                 # NOT "translate": Step 5 does HY
        word_timestamps=True,
        verbose=False,
    )
    n_seg = len(result.get("segments", []))
    logger.info("ASR done: %d segments, detected language '%s'",
                n_seg, result.get("language"))
    return result


def result_to_words(result: Dict[str, Any]) -> List[Word]:
    """Flatten a Whisper result into a single ordered list of `Word`.

    Prefers per-word timestamps; falls back to splitting segment text evenly
    across the segment's time span when words are missing.
    """
    words: List[Word] = []
    for seg in result.get("segments", []):
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                words.append(Word(text=text,
                                  start=float(w["start"]),
                                  end=float(w["end"])))
        else:
            words.extend(_split_segment(seg))
    # Guarantee monotonic, non-degenerate timing.
    return _sanitize(words)


def _split_segment(seg: Dict[str, Any]) -> List[Word]:
    """Fallback: distribute a segment's time proportionally over its words."""
    text = (seg.get("text") or "").strip()
    toks = text.split()
    if not toks:
        return []
    start, end = float(seg["start"]), float(seg["end"])
    span = max(end - start, 1e-3)
    total = sum(len(t) for t in toks)
    out, cursor = [], start
    for t in toks:
        frac = len(t) / total if total else 1.0 / len(toks)
        w_end = min(cursor + span * frac, end)
        out.append(Word(text=t, start=cursor, end=w_end))
        cursor = w_end
    return out


def _sanitize(words: List[Word]) -> List[Word]:
    """Clamp out-of-order / zero-length word spans so downstream code is safe."""
    cleaned: List[Word] = []
    prev_end = 0.0
    for w in words:
        start = max(w.start, prev_end)
        end = max(w.end, start + 1e-3)
        cleaned.append(Word(text=w.text, start=start, end=end))
        prev_end = end
    return cleaned
