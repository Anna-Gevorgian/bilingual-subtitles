"""STEP 7 — Validation.

Checks a finished cue list against the house style and produces a list of
`Violation`s plus a human-readable report. CRITICAL findings (overlaps, empty
text, degenerate timing) are flagged for human review; everything else is a
WARNING.

The validator is intentionally non-destructive: it never edits subtitles, it
only reports. This keeps the "never change meaning / never hallucinate"
guarantees intact — if a translated cue is too long, we surface it rather than
truncating it.
"""
from __future__ import annotations

import logging
from typing import List

from config import Config
from models import CRITICAL, WARNING, Cue, Violation

logger = logging.getLogger(__name__)


def validate(cues: List[Cue], track: str, config: Config | None = None
             ) -> List[Violation]:
    config = config or Config()
    out: List[Violation] = []

    if not cues:
        out.append(Violation(track, 0, "empty_track", CRITICAL,
                             "No subtitle cues were produced."))
        return out

    for i, cue in enumerate(cues):
        idx = cue.index or (i + 1)

        # Empty / missing text.
        if not cue.text.strip():
            out.append(Violation(track, idx, "empty_cue", CRITICAL,
                                 "Cue has no text."))

        # Degenerate timing.
        if cue.duration <= 0:
            out.append(Violation(track, idx, "bad_duration", CRITICAL,
                                 f"Non-positive duration ({cue.duration:.3f}s)."))

        # Overlap with previous cue.
        if i > 0 and cue.start < cues[i - 1].end - 1e-6:
            out.append(Violation(
                track, idx, "timing_overlap", CRITICAL,
                f"Starts at {cue.start:.3f}s before previous cue ends "
                f"({cues[i - 1].end:.3f}s)."))

        # Line count.
        if len(cue.lines) > config.max_lines_per_cue:
            out.append(Violation(
                track, idx, "too_many_lines", WARNING,
                f"{len(cue.lines)} lines (max {config.max_lines_per_cue})."))

        # Line length.
        for ln, line in enumerate(cue.lines, start=1):
            if len(line) > config.max_line_length:
                out.append(Violation(
                    track, idx, "line_too_long", WARNING,
                    f"Line {ln} is {len(line)} chars "
                    f"(max {config.max_line_length}): {line!r}"))

        # Reading speed.
        cps = cue.cps()
        if cps > config.max_cps:
            out.append(Violation(
                track, idx, "reading_speed", WARNING,
                f"{cps:.1f} CPS over {cue.duration:.2f}s "
                f"(max {config.max_cps:.0f})."))

    return out


def _fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_report(violations: List[Violation],
                 en_cues: List[Cue], hy_cues: List[Cue]) -> str:
    """Render a validation_report.txt body from collected violations."""
    lines: List[str] = []
    lines.append("SUBTITLE VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"English cues: {len(en_cues)}    Armenian cues: {len(hy_cues)}")

    crit = [v for v in violations if v.severity == CRITICAL]
    warn = [v for v in violations if v.severity == WARNING]
    lines.append(f"Issues: {len(violations)} total "
                 f"({len(crit)} CRITICAL, {len(warn)} WARNING)")
    lines.append("")

    if not violations:
        lines.append("No issues detected. All cues pass the house style.")
        return "\n".join(lines) + "\n"

    if crit:
        lines.append("CRITICAL — flagged for review")
        lines.append("-" * 60)
        for v in crit:
            lines.append(_line_for(v))
        lines.append("")

    if warn:
        lines.append("WARNINGS")
        lines.append("-" * 60)
        for v in warn:
            lines.append(_line_for(v))
        lines.append("")

    return "\n".join(lines) + "\n"


def _line_for(v: Violation) -> str:
    where = f"cue {v.cue_index}" if v.cue_index else "file"
    return f"[{v.track:8s}] {where:>9s}  {v.kind:<16s}  {v.detail}"
