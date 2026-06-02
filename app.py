from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from config import Config
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Bilingual Subtitles (EN + HY)", layout="wide", page_icon="🎬")
st.markdown(
    """
    <style>
      /* Make the main content area use the full page width */
      .block-container {
          max-width: 100% !important;
          padding-left: 3rem;
          padding-right: 3rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎬 Bilingual Subtitle Generator")
st.caption("English + Armenian subtitles from short videos (≤ 3 min). "
           "Whisper ASR · RAG-conditioned LLM cleanup & translation · validation.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""))
    whisper_model = st.selectbox(
        "Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=1)
    llm_model = st.text_input("LLM model", value="gpt-4o-mini")
    glossary_file = st.file_uploader("Optional domain glossary (.txt/.md)",
                                     type=["txt", "md"])

uploaded = st.file_uploader(
    "Upload a video", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])

run = st.button("Generate subtitles", type="primary",
                disabled=uploaded is None)

if run and uploaded is not None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API key in the sidebar.")
        st.stop()

    work = Path(tempfile.mkdtemp(prefix="subs_"))
    video_path = work / uploaded.name
    video_path.write_bytes(uploaded.getbuffer())

    cfg = Config(output_dir=str(work / "out"),
                 whisper_model=whisper_model,
                 llm_model=llm_model)
    if glossary_file is not None:
        gpath = work / glossary_file.name
        gpath.write_bytes(glossary_file.getbuffer())
        cfg.glossary_path = str(gpath)

    bar = st.progress(0.0, text="Starting…")

    def on_progress(step: str, frac: float) -> None:
        bar.progress(min(frac, 1.0), text=step)

    try:
        with st.spinner("Processing…"):
            result = run_pipeline(str(video_path), cfg, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Pipeline failed: {exc}")
        st.stop()

    bar.empty()
    n_crit = sum(1 for v in result.violations if v.flagged_for_review)
    if n_crit:
        st.warning(f"Done with {n_crit} critical issue(s) flagged for review "
                   f"({len(result.violations)} total). See the report below.")
    else:
        st.success(f"Done. {len(result.violations)} minor issue(s). "
                   f"Detected source language: {result.detected_language}.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("⬇️ English .srt",
                           Path(result.english_srt).read_bytes(),
                           file_name="english_subtitles.srt")
    with col2:
        st.download_button("⬇️ Armenian .srt",
                           Path(result.armenian_srt).read_bytes(),
                           file_name="armenian_subtitles.srt")
    with col3:
        st.download_button("⬇️ Report .txt",
                           Path(result.report_txt).read_bytes(),
                           file_name="validation_report.txt")

    with st.expander("Validation report", expanded=bool(n_crit)):
        st.code(Path(result.report_txt).read_text(encoding="utf-8"))

    tab_en, tab_hy = st.tabs(["English preview", "Armenian preview"])
    with tab_en:
        st.code(Path(result.english_srt).read_text(encoding="utf-8"))
    with tab_hy:
        st.code(Path(result.armenian_srt).read_text(encoding="utf-8"))
