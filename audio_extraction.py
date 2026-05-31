"""STEP 1 — Audio Extraction.

Extracts a mono 16 kHz PCM WAV from the input video using the ffmpeg CLI.
16 kHz mono is Whisper's native working format, so this avoids an internal
resample and keeps the file small.

We shell out to `ffmpeg` directly (rather than a wrapper library) to keep the
dependency surface small and the behavior transparent.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' was not found on PATH. Install ffmpeg "
            "(e.g. `apt install ffmpeg`, `brew install ffmpeg`)."
        )


def probe_duration(path: str) -> float | None:
    """Return media duration in seconds via ffprobe, or None if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def extract_audio(video_path: str, output_path: str | None = None,
                  config: Config | None = None) -> str:
    """Extract audio from `video_path` to a 16 kHz mono WAV.

    Returns the path to the written audio file.
    """
    config = config or Config()
    _require_binary("ffmpeg")

    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = probe_duration(str(src))
    if duration is not None and duration > config.max_video_seconds:
        msg = (f"Video is {duration:.1f}s, longer than the "
               f"{config.max_video_seconds:.0f}s target.")
        if config.enforce_duration:
            raise ValueError(msg + " (enforce_duration is on)")
        logger.warning(msg + " Continuing anyway.")

    out = Path(output_path) if output_path else src.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",                       # drop video
        "-acodec", "pcm_s16le",      # 16-bit PCM
        "-ar", "16000",             # 16 kHz
        "-ac", "1",                 # mono
        str(out),
    ]
    logger.info("Extracting audio: %s -> %s", src.name, out.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")
    if not out.is_file():
        raise RuntimeError("ffmpeg reported success but produced no output file.")
    return str(out)
