"""
Produces three files in config.output_dir:
    english_subtitles.srt
    armenian_subtitles.srt
    validation_report.txt
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from config import Config
from models import Cue, Violation

import audio_extraction
import asr
import segmentation
import rag
import llm
import refine
import validation
import srt_io

logger = logging.getLogger(__name__)

# step name -> coarse progress fraction, for UI feedback
ProgressCB = Callable[[str, float], None]


@dataclass
class PipelineResult:
    english_srt: str
    armenian_srt: str
    report_txt: str
    english_cues: List[Cue]
    armenian_cues: List[Cue]
    violations: List[Violation]
    detected_language: str | None


def _noop(_step: str, _frac: float) -> None:
    pass


def run_pipeline(video_path: str, config: Config | None = None,
                 progress: ProgressCB | None = None) -> PipelineResult:
    config = config or Config()
    progress = progress or _noop
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1 — audio (ffmpeg) video->audio
    progress("Extracting audio", 0.05)
    audio_path = audio_extraction.extract_audio(video_path, config=config)

    # STEP 2 — ASR (Whisper) speech->text 0.5s - 0.8s -> Hello, 0.8s - 1.1s -> world
    progress("Transcribing (Whisper)", 0.15)
    asr_result = asr.transcribe(audio_path, config=config)
    words = asr.result_to_words(asr_result)
    if not words:
        raise RuntimeError("ASR produced no words; cannot build subtitles.")

    # STEP 3 — segmentation (subtitle blocks) Clu 1: "Hello world"-> 0.5-1.1
    progress("Segmenting into cues", 0.40)
    base_cues = segmentation.build_cues(words, config)
    logger.info("Built %d base cues", len(base_cues))

    # STEP 6 — build RAG retriever before the LLM calls 
    progress("Building RAG index", 0.50)
    retriever = rag.build_retriever(config)
    chat_model = llm.get_chat_model(config)

    # STEP 4 — English refinement (RAG-conditioned) LLM does grammar fix, punctuation, readibility improvement, subtitle polishing
    progress("Refining English (LLM)", 0.60)
    en_cues = refine.refine_english(
        base_cues, chat_model, retriever, config,
        progress=lambda f: progress("Refining English (LLM)", 0.60 + 0.15 * f),
    )

    # STEP 5 — Armenian translation (RAG-conditioned) 
    progress("Translating to Armenian (LLM)", 0.78)
    hy_cues = refine.translate_armenian(
        en_cues, chat_model, retriever, config,
        progress=lambda f: progress("Translating to Armenian (LLM)", 0.78 + 0.15 * f),
    )

    # STEP 7 — validation -quality check, too long line, ovrelap, empty subtitle
    progress("Validating", 0.95)
    violations = (validation.validate(en_cues, "english", config)
                  + validation.validate(hy_cues, "armenian", config))

    # OUTPUT 
    en_path = srt_io.write_srt(en_cues, str(out_dir / "english_subtitles.srt"))
    hy_path = srt_io.write_srt(hy_cues, str(out_dir / "armenian_subtitles.srt"))
    report = validation.build_report(violations, en_cues, hy_cues)
    report_path = str(out_dir / "validation_report.txt")
    Path(report_path).write_text(report, encoding="utf-8")
    logger.info("Wrote validation report -> %s", report_path)

    if not config.keep_intermediate and audio_path != video_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    progress("Done", 1.0)
    return PipelineResult(
        english_srt=en_path,
        armenian_srt=hy_path,
        report_txt=report_path,
        english_cues=en_cues,
        armenian_cues=hy_cues,
        violations=violations,
        detected_language=asr_result.get("language"),
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Bilingual (EN+HY) subtitle pipeline")
    ap.add_argument("video", help="Path to input video (<= 3 min)")
    ap.add_argument("-o", "--output-dir", default="output")
    ap.add_argument("--whisper-model", default=None,
                    help="tiny|base|small|medium|large-v3")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--glossary", default=None, help="Optional glossary file")
    args = ap.parse_args()

    cfg = Config(output_dir=args.output_dir)
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.llm_model:
        cfg.llm_model = args.llm_model
    if args.glossary:
        cfg.glossary_path = args.glossary

    res = run_pipeline(args.video, cfg,
                       progress=lambda s, f: logger.info("[%3.0f%%] %s", f * 100, s))
    print("\nEnglish :", res.english_srt)
    print("Armenian:", res.armenian_srt)
    print("Report  :", res.report_txt)
    print(f"Issues  : {len(res.violations)} "
          f"({sum(1 for v in res.violations if v.flagged_for_review)} critical)")
