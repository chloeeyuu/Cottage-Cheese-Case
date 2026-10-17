"""Engine adapters. Each returns
{"text", "citations": [url], "search_results": [url], "model", "usage"}.

Engines: Anthropic, Gemini, OpenAI (all with web search). OpenRouter is used only for the second judge.
`AIBENCH_MOCK=1` runs the pipeline offline.
"""
from __future__ import annotations

import os
import random
import time

import httpx

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call(engine: str, spec: dict, messages: list[dict]) -> dict:
    if os.environ.get("AIBENCH_MOCK") == "1":
        return _mock(engine, messages)
    if engine == "anthropic":
        return anthropic_call(spec, messages)
    if engine == "gemini":
        return gemini_call(spec, messages)
    if engine == "openai":
        return openai_call(spec, messages)
    raise NotImplementedError(f"{engine} adapter not built yet (waiting on API access)")


def _post_with_retry(url: str, headers: dict, body: dict, tries: int = 5) -> dict:
    for attempt in range(tries):
        r = httpx.post(url, headers=headers, json=body, timeout=180)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 529) and attempt < tries - 1:
            time.sleep(min(60, 2 ** attempt * 5))
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    raise RuntimeError("unreachable")


def anthropic_call(spec: dict, messages: list[dict], system: str | None = None,
                   tools: list | None = None, tool_choice: dict | None = None,
                   max_tokens: int = 1500) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env)")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    body = {"model": spec["model"], "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    if tools is None and spec.get("web_search", True):
        tools = [{"type": spec.get("web_search_tool", "web_search_20250305"), "name": "web_search",
                  "max_uses": spec.get("max_searches", 5)}]
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    data = _post_with_retry(ANTHROPIC_URL, headers, body)
    return parse_anthropic(data)


def parse_anthropic(data: dict) -> dict:
    text, citations, results, tool_input = [], [], [], None
    for block in data.get("content", []):
        t = block.get("type")
        if t == "text":
            text.append(block.get("text", ""))
            for c in block.get("citations") or []:
                if c.get("url"):
                    citations.append(c["url"])
        elif t == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, list):
                results += [r["url"] for r in content if r.get("url")]
        elif t == "tool_use":
            tool_input = block.get("input")
    return {"text": "".join(text).strip(), "citations": list(dict.fromkeys(citations)),
            "search_results": list(dict.fromkeys(results)), "model": data.get("model"),
            "usage": data.get("usage", {}), "tool_input": tool_input, "stop_reason": data.get("stop_reason")}


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def gemini_call(spec: dict, messages: list[dict]) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages]
    body = {"contents": contents}
    if spec.get("web_search", True):
        body["tools"] = [{"google_search": {}}]
    headers = {"x-goog-api-key": key, "content-type": "application/json"}
    data = _post_with_retry(GEMINI_URL.format(model=spec["model"]), headers, body)
    return parse_gemini(data, spec["model"])


def parse_gemini(data: dict, model: str) -> dict:
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    gm = cand.get("groundingMetadata") or {}
    cites = []
    for ch in gm.get("groundingChunks", []):
        web = ch.get("web") or {}
        uri, title = web.get("uri", ""), web.get("title", "")
        # Grounding URIs are Google redirect links; the title carries the source domain.
        if "vertexaisearch" in uri and title and "." in title and " " not in title:
            cites.append("https://" + title)
        elif uri:
            cites.append(uri)
    um = data.get("usageMetadata", {})
    return {"text": text.strip(), "citations": list(dict.fromkeys(cites)), "search_results": [],
            "model": data.get("modelVersion", model),
            "usage": {"input_tokens": um.get("promptTokenCount", 0),
                      "output_tokens": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
                      "server_tool_use": {"web_search_requests": len(gm.get("webSearchQueries", []))}},
            "search_queries": gm.get("webSearchQueries", []), "stop_reason": cand.get("finishReason")}


OPENAI_URL = "https://api.openai.com/v1/responses"


def openai_call(spec: dict, messages: list[dict]) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env)")
    body = {"model": spec["model"], "input": [{"role": m["role"], "content": m["content"]} for m in messages]}
    if spec.get("web_search", True):
        body["tools"] = [{"type": spec.get("web_search_tool", "web_search")}]
    headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
    data = _post_with_retry(OPENAI_URL, headers, body)
    return parse_openai(data)


def parse_openai(data: dict) -> dict:
    text, cites, searches = [], [], 0
    for item in data.get("output", []):
        t = item.get("type", "")
        if t.startswith("web_search"):
            searches += 1
        elif t == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text.append(c.get("text", ""))
                    cites += [a["url"] for a in c.get("annotations", []) if a.get("type") == "url_citation" and a.get("url")]
    u = data.get("usage", {})
    clean = [u_.split("?utm_source=")[0] for u_ in cites]   # OpenAI appends utm tags to cited URLs
    return {"text": "".join(text).strip(), "citations": list(dict.fromkeys(clean)), "search_results": [],
            "model": data.get("model"),
            "usage": {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                      "server_tool_use": {"web_search_requests": searches}},
            "stop_reason": data.get("status")}


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_call(spec: dict, messages: list[dict], system: str | None = None, schema: dict | None = None,
                    max_tokens: int = 16000) -> dict:
    """OpenAI-format chat call via OpenRouter (used for the open-weight second judge, e.g. Qwen).

    spec: {model, reasoning: bool | "low"/"medium"/"high", provider: optional host name to pin}.
    If no host serves the model with structured output, falls back to a plain JSON instruction.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    body = {"model": spec["model"], "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "usage": {"include": True}}
    r = spec.get("reasoning")
    if r:
        body["reasoning"] = {"effort": r} if isinstance(r, str) else {"enabled": True}
    provider = {"require_parameters": True}
    if spec.get("provider"):
        provider |= {"order": [spec["provider"]], "allow_fallbacks": False}
    body["provider"] = provider
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "coding", "strict": False, "schema": schema}}
    headers = {"authorization": f"Bearer {key}", "content-type": "application/json",
               "x-title": "aibench"}
    try:
        data = _post_with_retry(OPENROUTER_URL, headers, body)
    except RuntimeError as exc:
        if schema and ("No endpoints" in str(exc) or str(exc).startswith("404")):
            body.pop("response_format")
            data = _post_with_retry(OPENROUTER_URL, headers, body)
        else:
            raise
    return parse_openrouter(data)


def parse_openrouter(data: dict) -> dict:
    if "error" in data:
        raise RuntimeError(f"openrouter error: {str(data['error'])[:300]}")
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    u = data.get("usage", {})
    return {"text": (msg.get("content") or "").strip(), "model": data.get("model"),
            "provider": data.get("provider"), "stop_reason": choice.get("finish_reason"),
            "usage": {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                      "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                      "cost": u.get("cost")}}


def _mock(engine, messages):
    rng = random.Random(str(messages) + engine)
    brand = rng.choice(["Good Culture", "Daisy", "Breakstone's", "Kirkland", "Lactaid"])
    return {"text": f"I'd recommend {brand}. It is a good option.",
            "citations": [rng.choice(["https://goodculture.com/x", "https://www.healthline.com/y"])],
            "search_results": [], "model": f"mock-{engine}", "usage": {}}
