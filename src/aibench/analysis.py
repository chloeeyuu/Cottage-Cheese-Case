"""All study results as one JSON-ready dict: the single source for the write-up and the dashboard.

build(cfg, answers, coded, second=None) -> dict. Every rate is reported with its n. "Top pick" uses
rule A (a conditional answer credits no brand) and, where noted, rule B (credit the first brand named
in the conditional picks), which is closer to how the human labeller read the answers.
"""
from __future__ import annotations

import collections
import re

import pandas as pd

from . import build, extract, metrics

CORE = ["Good Culture", "Daisy", "Private label", "Mulu", "Friendship", "Breakstone's", "Lactaid", "Nancy's"]
ENGINES = {"openai": "ChatGPT", "gemini": "Gemini", "anthropic": "Claude"}
NON_BRAND = ("none", "conditional", "refusal")


def _frame(cfg, answers, coded):
    canon = lambda b: build.canonical(b or "none", cfg["brands"])
    raw = {r["task_id"]: r for r in answers}
    rows = []
    for r in coded.values():
        a, c = raw.get(r["task_id"]), r["coded"]
        if a is None:
            continue
        top = canon(c.get("top_pick"))
        top_b = top
        if top == "conditional" and c.get("conditional_picks"):
            top_b = canon(c["conditional_picks"][0].get("brand"))
        rows.append({**{k: a.get(k) for k in ("task_id", "module", "engine", "need", "stage", "follow_up",
                                               "seeded_canonical", "opener", "held_canonical", "condition",
                                               "order", "rival", "gap", "region", "prompt_id", "pair")},
                     "top": top, "top_b": top_b, "verdict": c.get("held_brand_verdict", "n/a"),
                     "brands": {canon(b.get("brand")): b for b in (c.get("brands") or [])},
                     "cites": a.get("citations") or [], "prompt": a["messages"][-1]["content"],
                     "conversation": a["messages"], "text": a.get("text", "")})
    return pd.DataFrame(rows)


def _shares(df, col="top"):
    n = len(df)
    vc = df[col].value_counts()
    return {"n": n, **{b: round(int(k) / n, 4) for b, k in vc.items()}}


def _crosstab(df, index, col="top"):
    out = {}
    for key, g in df.groupby(index, sort=False):
        out[str(key) if not isinstance(key, tuple) else " | ".join(map(str, key))] = _shares(g, col)
    return out


def build_results(cfg: dict, answers: list[dict], coded: dict, second: dict | None = None,
                  gold: dict | None = None) -> dict:
    df = _frame(cfg, answers, coded)
    sales = pd.read_csv(cfg["sales_baseline"])
    total = float(sales.loc[sales.brand == "_category_total", "dollar_sales_musd"].iloc[0])
    oi_group = {b: s.get("oi_group") for b, s in cfg["brands"].items()}
    dollar_share = {r.brand: r.dollar_sales_musd / total for r in sales.itertuples() if r.brand != "_category_total"}

    res: dict = {"meta": {"focal_brand": cfg["focal_brand"], "category": cfg["category"],
                          "answers": len(df), "engines": {e: ENGINES.get(e, e) for e in cfg["engines"]},
                          "models": {e: s["model"] for e, s in cfg["engines"].items()},
                          "primary_judge": cfg["primary_judge"],
                          "judges": {k: v["model"] for k, v in cfg["judges"].items()},
                          "dollar_share": {k: round(v, 4) for k, v in dollar_share.items()}}}

    # --- M1: unprompted top pick (brand + purchase stages) ---
    m1 = df[(df.module == "m1") & (df.stage != "category")]
    tps = {"all": _shares(m1), "all_rule_b": _shares(m1, "top_b"),
           "by_engine": _crosstab(m1, "engine"), "by_engine_rule_b": _crosstab(m1, "engine", "top_b"),
           "by_need": _crosstab(m1, "need"), "by_need_engine": _crosstab(m1, ["need", "engine"]),
           "by_need_rule_b": _crosstab(m1, "need", "top_b")}
    oi = {}
    for scope, sh in [("all", tps["all"])] + [(e, tps["by_engine"][e]) for e in tps["by_engine"]]:
        grp = collections.Counter()
        for b, v in sh.items():
            if b != "n" and oi_group.get(b):
                grp[oi_group[b]] += v
        oi[scope] = {g: round(grp[g] / s, 2) for g, s in dollar_share.items()}
    tps["over_index"] = oi
    tps["mention_rate"] = {b: round(m1.brands.apply(lambda d: b in d).mean(), 3) for b in CORE}
    tps["prompts"] = [{"need": need, "stage": stage, "prompt": p, "n": len(g),
                       "picks": g.top.value_counts().to_dict()}
                      for (need, stage, p), g in m1.groupby(["need", "stage", "prompt"], sort=False)]
    res["m1"] = tps

    # --- M3: retention after pushback, net of placebo ---
    m3 = df[df.module == "m3_seeded"].assign(ret=lambda d: d.top == d.seeded_canonical)
    arr = {}
    for scope, part in [("all", m3)] + [(e, g) for e, g in m3.groupby("engine")]:
        p = part.pivot_table(index="follow_up", columns="seeded_canonical", values="ret", aggfunc="mean")
        n = part.pivot_table(index="follow_up", columns="seeded_canonical", values="ret", aggfunc="count")
        arr[scope] = {"retention": p.round(3).to_dict(), "net": (p - p.loc["placebo"]).round(3).to_dict(),
                      "n": n.to_dict()}
    gc = m3[m3.seeded_canonical == cfg["focal_brand"]]
    arr["focal_destinations"] = {fu: g.top.value_counts().to_dict() for fu, g in gc.groupby("follow_up")}
    res["m3"] = arr

    # --- probes ---
    lo = df[df.module == "pilot_loyalty"]
    res["loyalty"] = {f"{b} | {c}": {"n": len(g), **g.verdict.value_counts().to_dict()}
                      for (b, c), g in lo.groupby(["held_canonical", "condition"])}
    h = df[df.module.isin(["pilot_h2h", "pilot_h2h_more"])].assign(rival=lambda d: d.rival.fillna("daisy"))
    res["head_to_head"] = {"by_rival": _crosstab(h, "rival"), "by_rival_engine": _crosstab(h, ["rival", "engine"]),
                           "by_rival_order": _crosstab(h, ["rival", "order"])}
    p5 = df[df.module.isin(["pilot_m5", "pilot_mulu_price"])]
    res["price"] = {"by_gap": _crosstab(p5, ["rival", "gap"]), "by_gap_engine": _crosstab(p5, ["rival", "gap", "engine"])}
    res["regions"] = _crosstab(df[df.module == "pilot_region"], ["need", "region"])
    res["two_step"] = _crosstab(df[df.module == "pilot_m2"], "pair")
    ex = df[df.module == "extra"]
    res["extra"] = {"by_prompt": _crosstab(ex, "prompt_id"), "by_prompt_engine": _crosstab(ex, ["prompt_id", "engine"]),
                    "prompts": {pid: g.prompt.iloc[0] for pid, g in ex.groupby("prompt_id")}}

    # --- how brands are described (all answers) ---
    rows = [{"engine": r.engine, "brand": b, "sent": d.get("sentiment"), "tags": d.get("attributes") or [],
             "claims": d.get("nutrition_claims") or []}
            for r in df.itertuples() for b, d in r.brands.items() if b in CORE]
    m = pd.DataFrame(rows)
    tags = sorted({t for ts in m.tags for t in ts})
    res["descriptions"] = {
        "mentions": m.brand.value_counts().to_dict(),
        "sentiment_mean": m.groupby("brand").sent.mean().round(2).to_dict(),
        "negative_share": m.groupby("brand").sent.apply(lambda s: round((s < 0).mean(), 3)).to_dict(),
        "tag_rate": {t: {b: round(g.tags.apply(lambda x, t=t: t in x).mean(), 3) for b, g in m.groupby("brand")}
                     for t in tags},
        "focal_pricey_by_engine": m[m.brand == cfg["focal_brand"]].groupby("engine").tags.apply(
            lambda s: round(s.apply(lambda x: "pricey" in x).mean(), 3)).to_dict()}

    # --- factual accuracy of the focal brand's nutrition claims vs its label ---
    res["facts"] = _facts(cfg, m)

    # --- citations ---
    cites = [{"engine": r.engine, "cls": extract.classify_domain(u, cfg["brands"]),
              "domain": re.sub(r"^www\.", "", re.sub(r"^https?://", "", u).split("/")[0])}
             for r in df.itertuples() for u in r.cites]
    c = pd.DataFrame(cites)
    focal_owned = f"owned:{cfg['focal_brand']}"
    c["group"] = c.cls.where(~c.cls.str.startswith("owned:"), "owned: other brand")
    c.loc[c.cls == focal_owned, "group"] = f"owned: {cfg['focal_brand']}"
    res["citations"] = {
        "answers_with_citation": df.groupby("engine").cites.apply(lambda s: round((s.str.len() > 0).mean(), 3)).to_dict(),
        "per_answer": df.groupby("engine").cites.apply(lambda s: round(s.str.len().mean(), 2)).to_dict(),
        "mix": {e: g.group.value_counts(normalize=True).round(3).to_dict() for e, g in c.groupby("engine")},
        "top_domains": c.domain.value_counts().head(25).to_dict(),
        "top_domains_by_engine": {e: g.domain.value_counts().head(10).to_dict() for e, g in c.groupby("engine")}}

    # --- validation ---
    val = {}
    if second:
        canon = lambda b: build.canonical(b or "none", cfg["brands"])
        ids = sorted(set(coded) & set(second))
        a = [canon(coded[t]["coded"].get("top_pick")) for t in ids]
        b = [canon(second[t]["coded"].get("top_pick")) for t in ids]
        val["judge_agreement"] = {"n": len(ids), "top_pick_agreement": round(sum(x == y for x, y in zip(a, b)) / len(ids), 3),
                                  "top_pick_kappa": round(metrics.cohen_kappa(a, b), 3)}
    if gold:
        val["hand_labels"] = gold
    res["validation"] = val

    # --- answer explorer (every answer, for the dashboard) ---
    res["answers"] = [{"id": r.task_id, "engine": r.engine, "module": r.module, "need": r.need, "stage": r.stage,
                       "follow_up": r.follow_up, "seeded": r.seeded_canonical, "prompt": r.prompt,
                       "turns": [m_["content"] for m_ in r.conversation], "text": r.text, "top": r.top,
                       "cites": r.cites[:8]} for r in df.itertuples()]
    return res


def _facts(cfg, m):
    labels = pd.read_csv(cfg["product_labels"])
    lab = labels[labels.brand == cfg["focal_brand"]].iloc[0]
    truth = {"calories": float(lab.calories), "sodium": float(lab.sodium_mg), "protein": float(lab.protein_g)}
    out = {"label": {**truth, "product": lab["product"], "source": lab.label_source}, "by_engine": {}}
    for e, g in m[m.brand == cfg["focal_brand"]].groupby("engine"):
        cnt = {k: collections.Counter() for k in truth}
        for claims in g.claims:
            for cl in claims:
                n, v = str(cl.get("nutrient", "")).lower(), str(cl.get("value", ""))
                x = re.search(r"(\d+(?:\.\d+)?)", v)
                if not x or re.search(r"\b1 cup|per cup|container|tub", v):
                    continue  # only per-serving (1/2 cup) figures are comparable with the label
                key = "calories" if "calor" in n else "sodium" if "sodium" in n else "protein" if n == "protein" else None
                if key:
                    cnt[key][float(x.group(1))] += 1
        out["by_engine"][e] = {k: {"n": sum(c.values()), "match_label": sum(v for val, v in c.items() if val == truth[k]),
                                   "values": {str(int(val) if val.is_integer() else val): v for val, v in c.most_common(4)}}
                               for k, c in cnt.items()}
    return out


def tables(cfg: dict, answers: list[dict], coded: dict, second: dict | None = None,
           second_name: str = "second") -> dict[str, pd.DataFrame]:
    """Flat, tidy tables for spreadsheets / pandas:
    answers   one row per answer (design fields, both judges' top picks, focal-brand flags, citation counts)
    mentions  one row per (answer, brand): sentiment, one 0/1 column per attribute tag, nutrition claims
    citations one row per (answer, cited URL): domain and source class
    """
    df = _frame(cfg, answers, coded)
    canon = lambda b: build.canonical(b or "none", cfg["brands"])
    focal = cfg["focal_brand"]
    owned = f"owned:{focal}"
    design = ["task_id", "module", "engine", "need", "stage", "follow_up", "seeded_canonical", "opener",
              "held_canonical", "condition", "rival", "order", "gap", "region", "prompt_id", "pair"]
    a = df[design].copy()
    a.insert(2, "engine_name", a.engine.map(ENGINES))
    a["question"] = df.prompt
    a["turns"] = df.conversation.apply(len)
    a["top_pick"] = df.top
    a["top_pick_rule_b"] = df.top_b
    a["held_brand_verdict"] = df.verdict
    if second:
        a[f"top_pick_{second_name}"] = [canon(second[t]["coded"].get("top_pick")) if t in second else None
                                       for t in df.task_id]
    a["brands_mentioned"] = df.brands.apply(lambda d: "; ".join(d))
    a["n_brands"] = df.brands.apply(len)
    a["focal_mentioned"] = df.brands.apply(lambda d: focal in d).astype(int)
    a["focal_sentiment"] = df.brands.apply(lambda d: (d.get(focal) or {}).get("sentiment"))
    a["focal_pricey"] = df.brands.apply(lambda d: int("pricey" in ((d.get(focal) or {}).get("attributes") or []))
                                        if focal in d else None)
    a["n_citations"] = df.cites.apply(len)
    a["focal_owned_cited"] = df.cites.apply(lambda cs: int(any(extract.classify_domain(u, cfg["brands"]) == owned
                                                               for u in cs)))
    a["answer_chars"] = df.text.str.len()
    a["answer"] = df.text

    tags = judge_tags()
    mrows = []
    for r in df.itertuples():
        for b, d in r.brands.items():
            t = set(d.get("attributes") or [])
            mrows.append({"task_id": r.task_id, "module": r.module, "engine": r.engine, "need": r.need,
                          "brand": b, "is_top_pick": int(b == r.top), "sentiment": d.get("sentiment"),
                          **{f"tag_{x.replace(' ', '_').replace('-', '_')}": int(x in t) for x in tags},
                          "nutrition_claims": "; ".join(f"{c.get('nutrient')}: {c.get('value')}"
                                                        for c in d.get("nutrition_claims") or [])})
    crows = [{"task_id": r.task_id, "module": r.module, "engine": r.engine, "url": u,
              "domain": re.sub(r"^www\.", "", re.sub(r"^https?://", "", u).split("/")[0]),
              "source_class": extract.classify_domain(u, cfg["brands"])}
             for r in df.itertuples() for u in r.cites]
    return {"answers": a, "mentions": pd.DataFrame(mrows), "citations": pd.DataFrame(crows)}


def judge_tags() -> list[str]:
    from .judge import SCHEMA
    return SCHEMA["input_schema"]["properties"]["brands"]["items"]["properties"]["attributes"]["items"]["enum"]
