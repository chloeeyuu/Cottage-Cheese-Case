"""aibench command line.

  python -m aibench.cli models                                          # list Anthropic model ids
  python -m aibench.cli plan    --config configs/goodculture.yaml       # run plan + call budget
  python -m aibench.cli collect --config ... --engines anthropic --repeats 1 --limit 10
  python -m aibench.cli score   --config ... --judge qwen               # LLM judge -> coded answers (--sample N)
  python -m aibench.cli agree   --config ...                            # kappa between the two judges
  python -m aibench.cli aspects --config ...                            # pros / cons per brand (optional)
  python -m aibench.cli summary --config ...                            # all result tables (see report.py)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import aspects, build, engines, judge, report


def _answers_path(config):
    """Every engine answer, append-only: data/raw/<study>_answers.jsonl"""
    return Path("data/raw") / f"{Path(config).stem}_answers.jsonl"


def _coded_path(config, judge):
    """One judge's coding of every answer: data/processed/<study>_judged_<judge>.jsonl"""
    return Path("data/processed") / f"{Path(config).stem}_judged_{judge}.jsonl"


def _aspects_path(config):
    return Path("data/processed") / f"{Path(config).stem}_aspects.jsonl"


def _read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if Path(path).exists() else []


def _plan(args):
    rows = build.build_plan(args.config, args.modules)
    if args.engines:
        rows = [r for r in rows if r["engine"] in args.engines]
    if args.repeats is not None:
        rows = [r for r in rows if r["repeat"] < args.repeats]
    return rows


def cmd_models(args):
    engines.load_env()
    if args.engine == "openai":
        r = httpx.get("https://api.openai.com/v1/models", timeout=30,
                      headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        r.raise_for_status()
        for m in sorted(r.json().get("data", []), key=lambda m: m["id"]):
            print(m["id"])
        return
    if args.engine == "openrouter":  # hosts serving the second-judge model, with price and parameters
        judges = build.load_yaml(args.config)["judges"]
        model = next(j["model"] for j in judges.values() if j.get("backend") == "openrouter")
        r = httpx.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        r.raise_for_status()
        for e in r.json()["data"]["endpoints"]:
            p = e.get("pricing", {})
            sp = e.get("supported_parameters", [])
            print(f"{e.get('provider_name')}: in ${float(p.get('prompt', 0)) * 1e6:.2f}/M, "
                  f"out ${float(p.get('completion', 0)) * 1e6:.2f}/M, "
                  f"json_schema={'response_format' in sp or 'structured_outputs' in sp}, reasoning={'reasoning' in sp}")
        return
    if args.engine == "gemini":
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", timeout=30,
                      headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        r.raise_for_status()
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].replace("models/", ""), "|", m.get("displayName"))
        return
    r = httpx.get("https://api.anthropic.com/v1/models", timeout=30,
                  headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    r.raise_for_status()
    for m in r.json().get("data", []):
        print(m["id"], "|", m.get("display_name"), "|", m.get("created_at"))


def cmd_plan(args):
    rows = _plan(args)
    counts = collections.Counter((r["module"], r["engine"]) for r in rows)
    print(f"{len(rows)} calls")
    for (m, e), n in sorted(counts.items()):
        print(f"  {m:12s} {e:10s} {n}")


def cmd_collect(args):
    engines.load_env()
    cfg = build.load_yaml(args.config)
    raw = _answers_path(args.config)
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in _read_jsonl(raw)}
    todo = [r for r in _plan(args) if r["task_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    for e in {r["engine"] for r in todo}:
        if str(cfg["engines"][e]["model"]).startswith("TBD") and os.environ.get("AIBENCH_MOCK") != "1":
            raise SystemExit(f"Set engines.{e}.model in {args.config} first (run: python -m aibench.cli models)")
    print(f"{len(done)} cached, {len(todo)} to collect")

    def work(r):
        spec = cfg["engines"][r["engine"]]
        msgs = list(r["messages"])
        res = engines.call(r["engine"], spec, msgs)
        turns = []
        for fu in r.get("followups", []):  # multi-turn: each follow-up is sent after the real answer
            turns.append({"text": res["text"], "citations": res["citations"]})
            msgs += [{"role": "assistant", "content": res["text"]}, {"role": "user", "content": fu}]
            usage = res.get("usage", {})
            res = engines.call(r["engine"], spec, msgs)
            res["usage"] = {k: (usage.get(k, 0) or 0) + (res.get("usage", {}).get(k, 0) or 0)
                            for k in ("input_tokens", "output_tokens")} | {
                "server_tool_use": {"web_search_requests":
                                    (usage.get("server_tool_use") or {}).get("web_search_requests", 0)
                                    + (res.get("usage", {}).get("server_tool_use") or {}).get("web_search_requests", 0)}}
        out = {**r, **res, "messages": msgs, "collected_at": datetime.now(timezone.utc).isoformat()}
        if turns:
            out["earlier_turns"] = turns
        return out

    ok = failed = 0
    with raw.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for fut in as_completed(futures):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            except Exception as exc:  # keep going; failed tasks are retried on the next run
                failed += 1
                print("  failed:", str(exc)[:200])
            if (ok + failed) % 10 == 0:
                print(f"  {ok + failed}/{len(todo)}")
    print(f"done: {ok} ok, {failed} failed -> {raw}")
    _usage(_read_jsonl(raw))


def _usage(rows):
    inp = sum(r.get("usage", {}).get("input_tokens", 0) for r in rows)
    out = sum(r.get("usage", {}).get("output_tokens", 0) for r in rows)
    searches = sum((r.get("usage", {}).get("server_tool_use") or {}).get("web_search_requests", 0) for r in rows)
    print(f"usage so far: {len(rows)} answers, {inp:,} input tokens, {out:,} output tokens, {searches} web searches")


def _stratified(rows, n, seed=0):
    """n rows spread over engine x module in proportion to their size (fixed seed -> same sample each run)."""
    import random
    groups = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["task_id"]):
        groups[(r["engine"], r["module"])].append(r)
    rng, out = random.Random(seed), []
    for key in sorted(groups):
        g = groups[key]
        out += rng.sample(g, min(len(g), max(1, round(n * len(g) / len(rows)))))
    return out


def cmd_score(args):
    engines.load_env()
    cfg = build.load_yaml(args.config)
    raw = _answers_path(args.config)
    args.judge = args.judge or cfg["primary_judge"]
    coded = _coded_path(args.config, args.judge)
    coded.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in _read_jsonl(coded)}
    pool_rows = _read_jsonl(raw)
    if args.sample:
        pool_rows = _stratified(pool_rows, args.sample)
    todo = [r for r in pool_rows if r["task_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"judge={args.judge}: {len(done)} coded, {len(todo)} to code -> {coded}")
    spec = cfg["judges"][args.judge]

    def work(r):
        msgs = r["messages"]
        context = msgs[-1]["content"] if len(msgs) == 1 else "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in msgs)
        out = judge.code_answer(context, r["text"], cfg, spec)
        out["top_pick"] = build.canonical(out.get("top_pick", "none"), cfg["brands"])
        return {k: r[k] for k in r if k not in ("text", "messages", "usage", "search_results")} | {"coded": out}

    ok = failed = 0
    with coded.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(work, r) for r in todo]):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            except Exception as exc:
                failed += 1
                print("  failed:", str(exc)[:200])
            if (ok + failed) % 25 == 0:
                print(f"  {ok + failed}/{len(todo)}")
    print(f"done: {ok} ok, {failed} failed -> {coded}")
    if spec.get("backend") == "openrouter":
        rows = _read_jsonl(coded)
        u = [r["coded"].get("judge_usage") or {} for r in rows]
        cost = sum(x.get("cost") or 0 for x in u)
        print(f"{len(rows)} coded by {args.judge}; reasoning tokens avg "
              f"{sum(x.get('reasoning_tokens') or 0 for x in u) / max(1, len(u)):.0f}; cost so far ${cost:.2f}; "
              f"hosts: {dict(collections.Counter(r['coded'].get('judge_provider') for r in rows))}")


def cmd_aspects(args):
    """Pros/cons per brand for every collected answer (reuses raw answers; no new engine calls)."""
    engines.load_env()
    cfg = build.load_yaml(args.config)
    raw = _answers_path(args.config)
    out = _aspects_path(args.config)
    done = {r["task_id"] for r in _read_jsonl(out)}
    todo = [r for r in _read_jsonl(raw) if r["task_id"] not in done and r.get("text")]
    if args.engines:
        todo = [r for r in todo if r["engine"] in args.engines]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to extract")
    spec = next(j for j in cfg["judges"].values() if j.get("backend") == "anthropic")

    def work(r):
        res = aspects.extract(r["messages"][-1]["content"], r["text"], cfg, spec)
        for it in res["items"]:
            it["brand"] = build.canonical(it.get("brand", ""), cfg["brands"])
        return {k: r.get(k) for k in ("task_id", "module", "engine", "need", "stage")} | res

    ok = failed = 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(work, r) for r in todo]):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            except Exception as exc:
                failed += 1
                print("  failed:", str(exc)[:200])
            if (ok + failed) % 50 == 0:
                print(f"  {ok + failed}/{len(todo)}")
    print(f"done: {ok} ok, {failed} failed -> {out}")
    report.aspect_tables(_read_jsonl(out), cfg)


def cmd_summary(args):
    cfg = build.load_yaml(args.config)
    which = args.judge or cfg["primary_judge"]
    coded = _coded_path(args.config, which)
    rows = _read_jsonl(coded)
    if not rows:
        raise SystemExit(f"nothing coded yet in {coded}")
    print(f"judge: {which} ({coded}, {len(rows)} answers)")
    report.summary(rows, cfg)
    report.aspect_tables(_read_jsonl(_aspects_path(args.config)), cfg)


def cmd_agree(args):
    """Agreement between the main judge and a second judge on the answers both coded."""
    cfg = build.load_yaml(args.config)
    first = cfg["primary_judge"]
    second = args.judge or next(j for j in cfg["judges"] if j != first)
    a = {r["task_id"]: r for r in _read_jsonl(_coded_path(args.config, first))}
    b = {r["task_id"]: r for r in _read_jsonl(_coded_path(args.config, second))}
    report.agreement(a, b, cfg, names=(first, second),
                     out=Path("data/processed") / f"{Path(args.config).stem}_judge_disagreements.csv")


def main():
    p = argparse.ArgumentParser(prog="aibench")
    sub = p.add_subparsers(required=True)
    mp = sub.add_parser("models")
    mp.add_argument("--engine", default="anthropic", choices=["anthropic", "gemini", "openai", "openrouter"])
    mp.add_argument("--config", default="configs/goodculture.yaml")
    mp.set_defaults(fn=cmd_models)
    for name, fn in [("plan", cmd_plan), ("collect", cmd_collect), ("score", cmd_score), ("summary", cmd_summary),
                     ("aspects", cmd_aspects), ("agree", cmd_agree)]:
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--modules", nargs="*")
        s.add_argument("--engines", nargs="*")
        s.add_argument("--repeats", type=int)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--judge", default=None,
                       help="a judge name from the config (qwen, haiku); defaults to primary_judge")
        s.add_argument("--sample", type=int, default=0, help="score a stratified sample of N answers")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
