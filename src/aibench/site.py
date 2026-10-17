"""Build the static dashboard: docs/index.html with every answer and result embedded (no server needed).

The page computes its charts in the browser from one compact row per answer, so every filter
(engine, need, module) works on the same data. Brand-specific text (the Findings narrative) lives in
the study's own file, configs/<study>_findings.json, so another brand study only swaps that file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import analysis, build, extract

TEMPLATE = Path(__file__).with_name("site_template.html")


def payload(cfg: dict, answers: list[dict], coded: dict, second: dict | None, gold: dict | None) -> dict:
    res = analysis.build_results(cfg, answers, coded, second, gold)
    df = analysis._frame(cfg, answers, coded)
    focal_owned = f"owned:{cfg['focal_brand']}"
    rows = []
    for r in df.itertuples():
        cls = [extract.classify_domain(u, cfg["brands"]) for u in r.cites]
        rows.append({
            "id": r.task_id, "e": r.engine, "m": r.module, "need": r.need, "stage": r.stage,
            "fu": r.follow_up, "seed": r.seeded_canonical, "held": r.held_canonical, "cond": r.condition,
            "rival": r.rival, "order": r.order, "gap": r.gap, "region": r.region, "pid": r.prompt_id,
            "pair": r.pair, "top": r.top, "verdict": r.verdict,
            "b": {b: [d.get("sentiment"), d.get("attributes") or []] for b, d in r.brands.items()},
            "c": [[re.sub(r"^www\.", "", re.sub(r"^https?://", "", u).split("/")[0]),
                   "owned_focal" if k == focal_owned else ("owned_other" if k.startswith("owned:") else k), u]
                  for u, k in zip(r.cites, cls)],
            "q": [m["content"] for m in r.conversation], "a": r.text})
    import pandas as pd
    sales = pd.read_csv(cfg["sales_baseline"])
    res["meta"]["sales_source"] = {"url": str(sales.source.iloc[0]), "period": str(sales.period.iloc[0])}
    return {"meta": res["meta"], "facts": res["facts"], "validation": res["validation"],
            "extra_prompts": res["extra"]["prompts"], "rows": rows,
            "brands": list(analysis.CORE), "oi_group": {b: s.get("oi_group") for b, s in cfg["brands"].items()}}


def write(cfg: dict, data: dict, findings: dict, out: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps({**data, "findings": findings}, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")  # keep "</script>" inside answers from closing the tag
    html = html.replace("/*__DATA__*/null", blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
