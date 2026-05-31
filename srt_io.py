"""SRT serialization (pysrt).

Writes a list of `Cue`s to a UTF-8 .srt file. We convert float seconds to
millisecond ordinals so timestamps are exact and never drift.
"""
from __future__ import annotations

import logging
from typing import List

from models import Cue

logger = logging.getLogger(__name__)


def _subriptime(seconds: float):
    import pysrt
    ms = max(int(round(seconds * 1000)), 0)
    return pysrt.SubRipTime.from_ordinal(ms)


def write_srt(cues: List[Cue], path: str) -> str:
    """Serialize cues to `path` and return the path."""
    import pysrt

    subs = pysrt.SubRipFile()
    for i, cue in enumerate(cues, start=1):
        subs.append(pysrt.SubRipItem(
            index=i,
            start=_subriptime(cue.start),
            end=_subriptime(cue.end),
            text="\n".join(cue.lines),
        ))
    subs.save(path, encoding="utf-8")
    logger.info("Wrote %d cues -> %s", len(cues), path)
    return path
