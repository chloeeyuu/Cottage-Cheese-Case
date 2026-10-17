"""Aspect extraction: for every brand an answer mentions, pull out its pros and cons.

Runs over the already-collected raw answers (no new engine calls). Each item records a fixed aspect,
a polarity (pro / con) and a verbatim quote; quotes are checked against the answer text so a
hallucinated quote is flagged rather than counted.
"""
from __future__ import annotations

import json
import os
import re

from . import engines

ASPECTS = ["price", "protein", "calories_fat", "sodium", "ingredients_additives", "live_cultures",
           "taste", "texture", "availability", "lactose_digestion", "packaging_size", "cooking_use",
           "reputation_reviews", "other"]

SYSTEM = """You extract what an AI shopping assistant said about each {category} brand in one answer.
Brands (use exactly these names): {brands}. Store brands such as Kirkland, Good & Gather, Great Value,
365, Trader Joe's, Lucerne are all "Private label". Any other brand -> its own name.

For EVERY brand the answer mentions, list each distinct point the answer makes about it:
- aspect: one of {aspects}
- polarity: "pro" (a strength / reason to buy) or "con" (a weakness / reason not to buy).
  Neutral facts that are neither (e.g. "comes in 16 oz") -> skip them.
- quote: the shortest verbatim span from the answer (max 25 words) that supports the point.
  Copy it exactly; do not paraphrase.

Examples: "Good Culture is pricier" -> price/con. "Great Value was pasty and bland" -> texture/con and taste/con.
"Mulu adds whey protein powder, so it isn't plain cottage cheese" -> ingredients_additives/con.
"Friendship No Salt Added has 45 mg sodium" -> sodium/pro.
Only use what the answer says. Record the result by calling the record_aspects tool exactly once.
If no brand is mentioned, call the tool with an empty list."""

SCHEMA = {
    "name": "record_aspects",
    "description": "Record every pro and con the answer states about each brand.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {
                "brand": {"type": "string"},
                "aspect": {"type": "string", "enum": ASPECTS},
                "polarity": {"type": "string", "enum": ["pro", "con"]},
                "quote": {"type": "string"},
            }, "required": ["brand", "aspect", "polarity", "quote"]}},
        },
        "required": ["items"],
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[\s*_`’'\"]+", " ", s.lower()).strip()


def extract(question: str, answer: str, cfg: dict, spec: dict) -> dict:
    if os.environ.get("AIBENCH_MOCK") == "1":
        return {"items": [], "model": "mock"}
    system = SYSTEM.format(category=cfg["category"], brands=", ".join(cfg["brands"]), aspects=", ".join(ASPECTS))
    msg = f"SHOPPER QUESTION:\n{question}\n\nASSISTANT ANSWER:\n{answer}"
    for _ in range(2):
        res = engines.anthropic_call({**spec, "web_search": False}, [{"role": "user", "content": msg}],
                                     system=system, tools=[SCHEMA], tool_choice={"type": "auto"}, max_tokens=4000)
        out = res.get("tool_input")
        if out is None:
            m = re.search(r"\{.*\}", res.get("text", ""), re.S)
            try:
                out = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                out = None
        if isinstance(out, dict) and isinstance(out.get("items"), list):
            ans = _norm(answer)
            for it in out["items"]:
                it["quote_verified"] = bool(it.get("quote")) and _norm(it["quote"]) in ans
            out["model"] = res.get("model")
            return out
    raise RuntimeError("aspect extractor did not return items")
