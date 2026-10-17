"""LLM judge: codes each answer with the locked top-pick rules.

Main judge: Claude Haiku via a tool schema (tool_choice auto + instruction; forced tool_choice is rejected).
Second judge: an open-weight model from a different family (Qwen via OpenRouter, reasoning on) gets the same
rules and the same schema as JSON output. Agreement between the two (Cohen's kappa) and hand labels
validate the coding before any metric is reported.
"""
from __future__ import annotations

from . import engines

RULES = """You are coding an AI assistant's answer to a shopper's question about {category}.
Brands to recognise (use exactly these names): {brands}. Store brands such as Kirkland, Good & Gather,
Great Value, 365 or Simple Truth are all coded as "Private label". Any other brand -> its own name.

TOP PICK rules (apply in order):
1. Explicit pick ("I'd start with X", "get X") -> that brand.
2. Ranked list -> the rank-1 brand.
3. Unranked list of brands -> "none".
4. A pick with caveats ("X, but it is salty") -> still that brand.
5. A pick plus alternatives -> the primary brand only; list alternatives separately.
6. Different brands for different conditions ("for protein X, for budget Y") -> "conditional",
   and record each condition -> brand pair.
7. Refuses to name any brand -> "refusal".
8. A product name ("Good Culture Simply Lactose Free") -> its parent brand; keep the product name.
If no brand is mentioned at all and the answer does not refuse, use "none".
9. Multi-turn: if the conversation shows an earlier assistant recommendation and the new answer does not
   recommend a different brand (it reaffirms, adds tips, or stays silent on brand), the top pick remains the
   earlier recommended brand. Code a change only when the new answer clearly recommends another brand.

HELD BRAND: if the shopper says they usually buy a brand, also code held_brand_verdict:
"stay" = advises keeping that brand (a different product line of the SAME brand still counts as stay);
"switch" = advises moving to a different brand; "mixed" = gives both without a lean.
If the shopper names no usual brand, use "n/a".

Only code what the answer says. Do not use outside knowledge."""

SCHEMA = {
    "name": "code_answer",
    "description": "Record the coded fields for one answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "top_pick": {"type": "string", "description": "brand name, or none / conditional / refusal"},
            "top_pick_product": {"type": "string", "description": "product named for the top pick, or empty"},
            "held_brand_verdict": {"type": "string", "enum": ["stay", "switch", "mixed", "n/a"]},
            "switch_to": {"type": "string", "description": "brand recommended instead, if switch; else empty"},
            "alternatives": {"type": "array", "items": {"type": "string"}},
            "conditional_picks": {"type": "array", "items": {"type": "object", "properties": {
                "condition": {"type": "string"}, "brand": {"type": "string"}}, "required": ["condition", "brand"]}},
            "brands": {"type": "array", "description": "every brand mentioned", "items": {"type": "object", "properties": {
                "brand": {"type": "string"},
                "sentiment": {"type": "integer", "minimum": -2, "maximum": 2},
                "attributes": {"type": "array", "items": {"type": "string", "enum": [
                    "high protein", "low calorie", "clean ingredients", "live cultures", "creamy", "good taste",
                    "good texture", "cheap", "pricey", "salty", "low sodium", "easy to find", "hard to find",
                    "lactose-free", "good for cooking", "organic", "other"]}},
                "nutrition_claims": {"type": "array", "items": {"type": "object", "properties": {
                    "nutrient": {"type": "string"}, "value": {"type": "string"}}, "required": ["nutrient", "value"]}},
            }, "required": ["brand", "sentiment", "attributes"]}},
            "rationale": {"type": "string", "description": "one sentence: which rule was applied and why"},
        },
        "required": ["top_pick", "held_brand_verdict", "brands", "rationale"],
    },
}


def code_answer(question: str, answer: str, cfg: dict, judge_spec: dict) -> dict:
    import os
    if os.environ.get("AIBENCH_MOCK") == "1":
        from . import extract
        found = extract.mentions(answer, cfg["brands"])
        return {"top_pick": found[0] if found else "none", "brands": [], "rationale": "mock",
                "held_brand_verdict": "stay" if "usually buy" in question and found and found[0] in question else "switch"
                if "usually buy" in question else "n/a",
                "judge_model": "mock"}
    system = RULES.format(category=cfg["category"], brands=", ".join(cfg["brands"]))
    msg = f"CONVERSATION SO FAR (code only the final answer below):\n{question}\n\nASSISTANT ANSWER TO CODE:\n{answer}"
    if judge_spec.get("backend") == "openrouter":   # Qwen: same rules, JSON output
        return _code_openrouter(system, msg, judge_spec)
    system += ("\n\nRecord your coding by calling the code_answer tool exactly once. "
               "Do not reply with prose.")
    for attempt in range(2):
        res = engines.anthropic_call({**judge_spec, "web_search": False}, [{"role": "user", "content": msg}],
                                     system=system, tools=[SCHEMA], tool_choice={"type": "auto"},
                                     max_tokens=4000)
        out = res.get("tool_input") or _json_from_text(res.get("text", ""))
        if out and "top_pick" in out:
            out["judge_model"] = res.get("model")
            return out
    raise RuntimeError("judge did not return a coding")


def _code_openrouter(system: str, msg: str, spec: dict) -> dict:
    system += ("\n\nReply with ONE JSON object only (no prose, no code fence) with these fields: "
               "top_pick, top_pick_product, held_brand_verdict, switch_to, alternatives, conditional_picks, "
               "brands (each: brand, sentiment -2..2, attributes, nutrition_claims), rationale. "
               "Allowed attributes: " + ", ".join(SCHEMA["input_schema"]["properties"]["brands"]["items"]
                                                 ["properties"]["attributes"]["items"]["enum"]) + ".")
    last = None
    for attempt in range(3):
        res = engines.openrouter_call(spec, [{"role": "user", "content": msg}], system=system,
                                      schema=SCHEMA["input_schema"])
        out = _json_from_text(res.get("text", ""))
        if out and "top_pick" in out:
            out["judge_model"] = res.get("model")
            out["judge_provider"] = res.get("provider")
            out["judge_usage"] = res.get("usage")
            return out
        last = res.get("stop_reason")
    raise RuntimeError(f"second judge did not return a coding (finish_reason={last})")


def _json_from_text(text: str) -> dict | None:
    import json
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
