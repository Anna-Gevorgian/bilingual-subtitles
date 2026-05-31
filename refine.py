"""STEP 4 — English Subtitle Refinement   (LLM call #1)
STEP 5 — Armenian Translation + Optimization (LLM call #2)

Both steps share one robust mechanism: we hand the model a JSON array of
``{"id", "text"}`` cues and require it to return the **same ids** with new text.
That id mapping is what protects the pipeline's hard constraints:

  * The model cannot merge, split, drop, or reorder cues, so timing stays
    aligned 1:1 with the source — and "do not add / do not omit information"
    is enforced structurally, not just by instruction.
  * Any id the model forgets falls back to the original text (and is logged).

We use LangChain's structured output (``with_structured_output``) so parsing is
schema-validated rather than regex-scraped. After the model returns, line
lengths are re-wrapped *programmatically* (LLMs cannot reliably count
characters), so the 42-char rule is guaranteed regardless of the model.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List

from pydantic import BaseModel, Field

from config import Config
from models import Cue
from rag import Retriever
from segmentation import rewrap_cue_text

logger = logging.getLogger(__name__)


# --- structured output schema 
class _CueItem(BaseModel):
    id: int = Field(description="The cue id, copied unchanged from the input.")
    text: str = Field(description="The rewritten/translated single-line text.")


class _CueList(BaseModel):
    cues: List[_CueItem]


# --- prompts 
_ENGLISH_SYSTEM = (
    "You are a professional subtitle editor cleaning raw speech-to-text output.\n"
    "Apply the STYLE CONTEXT below. For every cue:\n"
    "- Remove disfluencies and fillers (uh, um, er, you know, I mean, and "
    "filler 'like'/'sort of').\n"
    "- Fix capitalization, punctuation, and obvious grammar.\n"
    "- Preserve the speaker's exact meaning and ALL information.\n"
    "- DO NOT add facts, words, or content that are not in the original.\n"
    "- DO NOT drop meaningful content. Keep names, numbers, and terms intact.\n"
    "- Keep each cue at roughly its original length.\n"
    "Return one entry per input id, with the SAME ids."
)

_ARMENIAN_SYSTEM = (
    "You are a professional English-to-Armenian (Eastern Armenian) subtitle "
    "translator. Apply the STYLE CONTEXT and any GLOSSARY below. For every cue:\n"
    "- Produce natural, fluent Armenian — translate meaning, NOT word by word.\n"
    "- Preserve meaning exactly: DO NOT add and DO NOT omit information.\n"
    "- Use correct Armenian punctuation (full stop \u0589; the question mark "
    "\u055e and exclamation \u055c go over the relevant vowel, not at the end).\n"
    "- Keep terminology consistent across ALL cues.\n"
    "- Prefer concise phrasing suitable for subtitles (short lines).\n"
    "Return one Armenian entry per input id, with the SAME ids."
)

_HUMAN = (
    "STYLE CONTEXT:\n{context}\n\n"
    "{instruction}\n"
    "Input cues (JSON):\n{cues_json}"
)


def _chunks(cues: List[Cue], size: int) -> List[List[Cue]]:
    return [cues[i:i + size] for i in range(0, len(cues), size)]


def _run_batch(chat_model, system: str, instruction: str, context: str,
               batch: List[Cue]) -> Dict[int, str]:
    """Invoke the model on one batch; return {cue_index: new_text}."""
    from langchain_core.prompts import ChatPromptTemplate

    payload = [{"id": c.index, "text": c.text} for c in batch]
    cues_json = json.dumps(payload, ensure_ascii=False)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", _HUMAN)]
    )
    structured = chat_model.with_structured_output(_CueList)
    chain = prompt | structured

    try:
        result: _CueList = chain.invoke({
            "context": context or "(no additional context retrieved)",
            "instruction": instruction,
            "cues_json": cues_json,
        })
        return {item.id: item.text for item in result.cues}
    except Exception as exc:  # noqa: BLE001 — keep the pipeline resilient
        logger.error("LLM batch failed (%s). Falling back to source text.", exc)
        return {}


def _apply(cues: List[Cue], chat_model, system: str, instruction: str,
           retriever: Retriever, query: str, config: Config,
           progress: Callable[[float], None] | None = None) -> List[Cue]:
    context = retriever.context_for(query)
    batches = _chunks(cues, config.llm_batch_size)
    mapping: Dict[int, str] = {}
    for bi, batch in enumerate(batches):
        mapping.update(_run_batch(chat_model, system, instruction, context, batch))
        if progress:
            progress((bi + 1) / len(batches))

    out: List[Cue] = []
    missing = 0
    for c in cues:
        new_text = mapping.get(c.index)
        if not new_text or not new_text.strip():
            new_text = c.text          # fall back to original
            missing += 1
        lines = rewrap_cue_text(new_text, config)
        out.append(Cue(start=c.start, end=c.end, lines=lines, index=c.index))
    if missing:
        logger.warning("%d/%d cues kept original text (model omitted them).",
                       missing, len(cues))
    return out


def refine_english(cues: List[Cue], chat_model, retriever: Retriever,
                   config: Config | None = None,
                   progress: Callable[[float], None] | None = None) -> List[Cue]:
    config = config or Config()
    logger.info("Step 4: refining %d English cues", len(cues))
    return _apply(
        cues, chat_model, _ENGLISH_SYSTEM,
        "Clean these subtitle cues. Return one entry per id.",
        retriever,
        "English subtitle formatting, grammar, punctuation and filler-word rules",
        config, progress,
    )


def translate_armenian(cues: List[Cue], chat_model, retriever: Retriever,
                       config: Config | None = None,
                       progress: Callable[[float], None] | None = None
                       ) -> List[Cue]:
    config = config or Config()
    logger.info("Step 5: translating %d cues to Armenian", len(cues))
    return _apply(
        cues, chat_model, _ARMENIAN_SYSTEM,
        "Translate these subtitle cues into Armenian. Return one entry per id.",
        retriever,
        "Armenian translation style, punctuation norms, and terminology glossary",
        config, progress,
    )
