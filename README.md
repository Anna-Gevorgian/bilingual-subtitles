# Bilingual Subtitle Pipeline (English + Armenian)

Generates **English** and **Armenian** SRT subtitles from a short video
(≤ 3 minutes), end-to-end and automatically:

```
video ─▶ ffmpeg ─▶ Whisper ASR ─▶ segmentation ─▶ ┌─ RAG (FAISS) ─┐
                  (word timing)   (SRT cues)       │  ▼            │
                                                   ├▶ LLM #1: EN cleanup
                                                   └▶ LLM #2: HY translate
                                                              │
                                                   validation ─▶ 3 files
```

Outputs (in `output/` or a per-run temp dir from the UI):
`english_subtitles.srt`, `armenian_subtitles.srt`, `validation_report.txt`.

---

## Module map

| File | Step | Responsibility |
|------|------|----------------|
| `config.py` | — | Single source of truth for every tunable (line/CPS limits, model names, RAG params). |
| `models.py` | — | `Word`, `Cue`, `Violation` dataclasses. Stdlib-only so every module can share them. |
| `audio_extraction.py` | 1 | `ffmpeg` → mono 16 kHz WAV (Whisper's native format). Duration guard via `ffprobe`. |
| `asr.py` | 2 | Whisper transcription with **word-level timestamps**; flattens to `Word`s (with a segment-level fallback). |
| `segmentation.py` | 3 | Groups words into non-overlapping cues; balanced 2-line wrapping that breaks at word/punctuation boundaries. |
| `rag.py` | 6 | Loads the knowledge base, builds a **FAISS** index, returns top-k context per query. |
| `llm.py` | — | Chat-model / embeddings factory. Swap providers in one place. |
| `refine.py` | 4, 5 | The two LLM calls. English cleanup and Armenian translation, both RAG-conditioned, both preserving cue identity. |
| `validation.py` | 7 | Non-destructive checks (overlaps, line length, CPS, empties); renders the report. |
| `srt_io.py` | — | Serializes cues to UTF-8 SRT via `pysrt` (exact millisecond timing). |
| `pipeline.py` | 1–7 | Orchestrator. `run_pipeline(video, config)` + a CLI. |
| `app.py` | UI | Streamlit upload → run → download. |
| `knowledge_base/` | 6 | Editable RAG corpus: subtitle rules, Armenian style, Armenian punctuation. |

---

## How the hard constraints are enforced

These are enforced **structurally**, not just by asking the model nicely:

- **No hallucination / no meaning change.** Steps 4 and 5 send the model a JSON
  array of `{"id", "text"}` cues and require the *same ids* back. The model
  therefore cannot merge, split, drop, or reorder cues — timing stays aligned
  1:1 with the source audio. Any id the model omits falls back to the original
  text and is logged.
- **Both LLM calls are RAG-conditioned.** `pipeline.py` builds the retriever
  before either call and injects retrieved context into both prompts (a spec
  requirement).
- **42-char lines are guaranteed.** LLMs can't count characters reliably, so we
  re-wrap every cue *programmatically* (`segmentation.wrap_balanced`) after the
  model returns — for both tracks.
- **No silent truncation.** If a translated cue genuinely can't fit in 2×42
  (Armenian is sometimes longer than English), we keep the full text and let
  the validator flag it for review. Preserving meaning wins over the line limit;
  the report surfaces the conflict for a human. (See "Trade-offs".)

## Key design decisions

- **Word-level timestamps** drive segmentation, so cue boundaries land on real
  word edges with accurate timing instead of guessing inside coarse Whisper
  segments. The line-breaker prefers sentence/clause punctuation, producing
  natural, balanced two-liners.
- **FAISS over Chroma**: in-memory, no server, ideal for a small KB.
- **Structured output** (`with_structured_output`) instead of regex-scraping
  model text — schema-validated parsing, with a graceful fallback if a batch
  fails.
- **Stdlib-only core** (`segmentation`, `validation`) — unit-tested without any
  heavy dependency.

## Trade-offs / extension points

- *Over-long Armenian cues* are flagged, not shortened. A natural extension is
  an optional third LLM pass that compresses only the flagged cues (preserving
  meaning) and re-validates.
- *Terminology consistency* relies on the whole clip usually fitting in one LLM
  batch (`llm_batch_size`, default 50) plus the RAG glossary. For long inputs,
  add a glossary-extraction pre-pass.
- *CPS = 21* is applied to both tracks; Armenian may warrant a different ceiling.
- The Armenian KB is a **starter** — replace it with your authoritative style
  guide for best results.

---

## Install

```bash
# 1. System dependency (NOT pip):
sudo apt install ffmpeg        # Debian/Ubuntu
# brew install ffmpeg          # macOS

# 2. Python deps (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. API key for the LLM + embeddings
export OPENAI_API_KEY="sk-..."
```

## Run

**Streamlit UI** (upload + download):
```bash
streamlit run app.py
```
Enter your API key in the sidebar (or rely on `OPENAI_API_KEY`), pick a Whisper
model, upload a video, click *Generate subtitles*, download the three files.

**Command line:**
```bash
python pipeline.py path/to/clip.mp4 -o output \
    --whisper-model base --llm-model gpt-4o-mini \
    --glossary path/to/glossary.md      # optional
```

## Configuration

Edit `config.py` (or pass a `Config(...)` to `run_pipeline`). Common knobs:
`max_line_length`, `max_cps`, `min/max_cue_duration`, `whisper_model`,
`llm_model`, `embedding_model`, `llm_batch_size`, `rag_top_k`, `glossary_path`.

### Swapping providers
- **Anthropic LLM:** `pip install langchain-anthropic`, then edit
  `llm.get_chat_model` to return `ChatAnthropic(...)`.
- **Local/offline embeddings:** `pip install langchain-huggingface
  sentence-transformers`, then edit `llm.get_embeddings`.
- **Chroma instead of FAISS:** `pip install langchain-chroma` and swap the
  vector-store import in `rag.py`.

## A note on the optional glossary
Provide a `.md`/`.txt` file mapping source terms to required Armenian
renderings, one per line, e.g.:
```
endpoint -> վերջնակետ
packet   -> փաթեթ
```
Point `--glossary` (CLI) or the sidebar uploader (UI) at it; it is embedded into
the RAG store and retrieved for the translation step to keep terminology stable.

## What is tested
`segmentation` and `validation` are covered by unit + stress tests (random
inputs; asserts no word loss, no overlaps, line/​line-count limits, and that the
validator catches overlaps, empties, bad durations, over-long lines, and CPS).
SRT serialization is verified for exact millisecond timing and Armenian UTF-8.
The Whisper / RAG / LLM stages use standard, current LangChain 1.x APIs and run
against live services (they need the API key and model downloads).
