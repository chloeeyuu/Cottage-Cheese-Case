import json
import re
from pathlib import Path

import pandas as pd
import pytest

from aibench import build, extract, metrics

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "goodculture.yaml"
BANNED = ["good culture", "daisy", "breakstone", "knudsen", "kirkland", "lactaid", "mulu", "kemps",
          "darigold", "walmart", "costco", "kroger", "student", "mom", "year-old"]


def test_m1_has_81_brand_free_prompts():
    spec = build.load_yaml(ROOT / "prompts" / "core_need_grid.yaml")["prompts"]
    prompts = [p for need in spec.values() for stage in need.values() for p in stage]
    assert len(spec) == 9 and len(prompts) == 81
    assert not [p for p in prompts if any(b in p.lower() for b in BANNED)]


def test_call_budget():
    rows = build.build_plan(CFG, ["m1", "m3_seeded"])
    by = pd.Series([r["module"] for r in rows]).value_counts().to_dict()
    assert by == {"m1": 81 * 5 * 3, "m3_seeded": 58 * 3 * 2 * 3}
    assert len({r["task_id"] for r in rows}) == len(rows)


def test_probes_budget_one_engine():
    rows = [r for r in build.build_plan(CFG, ["probes"]) if r["engine"] == "anthropic"]
    by = pd.Series([r["module"] for r in rows]).value_counts().to_dict()
    assert by == {"pilot_need": 27, "pilot_loyalty": 36, "pilot_h2h": 6, "pilot_h2h_more": 30, "pilot_m2": 9,
                  "pilot_m5": 16, "pilot_mulu_price": 12, "pilot_region": 30}
    prompts = {r["messages"][0]["content"] for r in rows if r["module"] == "pilot_need"}
    assert not [p for p in prompts if any(b in p.lower() for b in BANNED)]


def test_seeded_template_is_claim_free_and_symmetric():
    rows = [r for r in build.build_plan(CFG, ["m3_seeded"])]
    turns = {r["messages"][1]["content"].replace(r["seeded_brand"], "{b}") for r in rows}
    assert turns == {"I'd recommend {b}. It's a solid choice."}
    assert {r["seeded_canonical"] for r in rows} == {"Good Culture", "Daisy", "Private label"}


def test_mentions_order_and_aliases():
    cfg = build.load_yaml(CFG)
    text = "Try Kirkland, or Good Culture; Daisy brand is fine too."
    assert extract.mentions(text, cfg["brands"]) == ["Private label", "Good Culture", "Daisy"]


def test_domain_classes():
    b = build.load_yaml(CFG)["brands"]
    assert extract.classify_domain("https://www.goodculture.com/p", b) == "owned:Good Culture"
    assert extract.classify_domain("https://www.walmart.com/ip/1", b) == "retailer"
    assert extract.classify_domain("https://my.clevelandclinic.org/x", b) == "health"
    assert extract.classify_domain("https://old.reddit.com/r/x", b) == "ugc"


def test_parse_anthropic_response():
    from aibench import engines
    data = {"model": "m", "usage": {"input_tokens": 10}, "content": [
        {"type": "server_tool_use", "name": "web_search", "input": {"query": "q"}},
        {"type": "web_search_tool_result", "content": [{"type": "web_search_result", "url": "https://a.com/1"}]},
        {"type": "text", "text": "Get Good Culture.", "citations": [{"type": "web_search_result_location",
                                                                     "url": "https://goodculture.com/p"}]},
        {"type": "text", "text": " It has 14g protein."}]}
    out = engines.parse_anthropic(data)
    assert out["text"] == "Get Good Culture. It has 14g protein."
    assert out["citations"] == ["https://goodculture.com/p"]
    assert out["search_results"] == ["https://a.com/1"]


def test_parse_gemini_response():
    from aibench import engines
    data = {"modelVersion": "gemini-x", "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7},
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Try Good Culture."}]},
                            "groundingMetadata": {"webSearchQueries": ["best cottage cheese"], "groundingChunks": [
                                {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/x",
                                         "title": "thekitchn.com"}}]}}]}
    out = engines.parse_gemini(data, "m")
    assert out["text"] == "Try Good Culture." and out["citations"] == ["https://thekitchn.com"]
    assert out["usage"]["server_tool_use"]["web_search_requests"] == 1


def test_parse_openai_response():
    from aibench import engines
    data = {"model": "gpt-x", "status": "completed", "usage": {"input_tokens": 9, "output_tokens": 4},
            "output": [{"type": "web_search_call", "status": "completed"},
                       {"type": "message", "content": [{"type": "output_text", "text": "Get Good Culture.",
                        "annotations": [{"type": "url_citation", "url": "https://thekitchn.com/a?utm_source=openai"}]}]}]}
    out = engines.parse_openai(data)
    assert out["text"] == "Get Good Culture." and out["citations"] == ["https://thekitchn.com/a"]
    assert out["usage"]["server_tool_use"]["web_search_requests"] == 1


def test_wilson_and_arr():
    lo, hi = metrics.wilson(18, 36)
    assert 0.33 < lo < 0.35 and 0.65 < hi < 0.67
    df = pd.DataFrame({"seeded_brand": ["Daisy"] * 4, "seeded_canonical": ["Daisy"] * 4,
                       "follow_up": ["price", "price", "placebo", "placebo"],
                       "top_pick": ["Daisy", "Private label", "Daisy", "Daisy"]})
    res = metrics.arr(df, min_n=1).set_index("follow_up")
    assert res.loc["price", "net_arr"] == pytest.approx(-0.5)


def test_extra_prompts_are_brand_free_and_counted():
    rows = [r for r in build.build_plan(CFG, ["extra"]) if r["engine"] == "anthropic"]
    spec = build.load_yaml(ROOT / "prompts" / "store_and_need_questions.yaml")
    assert len(rows) == len(spec["prompts"]) * spec["repeats"]
    brand_words = [b for b in BANNED if b not in ("walmart", "costco", "kroger")]
    assert not [r for r in rows if any(b in r["messages"][0]["content"].lower() for b in brand_words)]


def test_aspect_quote_check():
    from aibench import aspects
    ans = "Good Culture is **pricier**, about $0.31 per ounce."
    assert aspects._norm("Good Culture is pricier") in aspects._norm(ans)
    assert aspects._norm("Good Culture is cheap") not in aspects._norm(ans)


def test_every_collected_answer_is_in_the_plan():
    """Refactors must not change cache keys: each stored answer still maps to a planned call."""
    raw = ROOT / "data" / "raw" / "goodculture_answers.jsonl"
    if not raw.exists():
        pytest.skip("no collected data")
    planned = {r["task_id"] for r in build.build_plan(CFG)}
    stored = {json.loads(l)["task_id"] for l in open(raw, encoding="utf-8")}
    assert stored <= planned


def test_report_canonicalises_judge_spellings():
    from aibench import report
    cfg = build.load_yaml(CFG)
    rows = [{"module": "extra", "coded": {"top_pick": t}} for t in ("MULU", "Great Value", "Trader Joe's")]
    assert list(report.frame(rows, cfg).top_pick) == ["Mulu", "Private label", "Private label"]


def test_kappa():
    assert metrics.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(metrics.cohen_kappa(list("aabb"), list("abab"))) < 1e-9


def test_parse_openrouter_response():
    from aibench import engines
    data = {"model": "qwen/x", "provider": "HostA", "choices": [{"finish_reason": "stop",
            "message": {"content": '{"top_pick": "Daisy"}', "reasoning": "..."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "cost": 0.0001,
                      "completion_tokens_details": {"reasoning_tokens": 40}}}
    out = engines.parse_openrouter(data)
    assert out["text"] == '{"top_pick": "Daisy"}' and out["provider"] == "HostA"
    assert out["usage"]["reasoning_tokens"] == 40
