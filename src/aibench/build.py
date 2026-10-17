"""Expand prompt templates + config into a run plan: one row per API call.

Every row gets a stable `task_id` (hash of module, engine, messages, repeat) so the collector
can cache results and a rerun skips anything already collected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_id(module: str, engine: str, messages: list[dict], repeat: int) -> str:
    key = json.dumps([module, engine, messages, repeat], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _row(module, engine, messages, repeat, **meta):
    return {"task_id": task_id(module, engine, messages, repeat), "module": module,
            "engine": engine, "repeat": repeat, "messages": messages, **meta}


def build_m1(cfg: dict, root: Path) -> list[dict]:
    spec = load_yaml(root / cfg["prompts"]["m1"])
    rows = []
    for need, stages in spec["prompts"].items():
        for stage, phrasings in stages.items():
            for i, text in enumerate(phrasings, start=1):
                prompt = text.format(category=cfg["category"])
                for engine in cfg["engines"]:
                    for r in range(cfg["repeats"]["m1"]):
                        rows.append(_row("m1", engine, [{"role": "user", "content": prompt}], r,
                                         need=need, stage=stage, phrasing=f"p{i}"))
    return rows


def build_m3_seeded(cfg: dict, root: Path) -> list[dict]:
    spec = load_yaml(root / cfg["prompts"]["m3"])
    seeded = spec["seeded"]
    all_openers = list(spec["openers"])
    rows = []
    for fu_name, fu in spec["follow_ups"].items():
        for opener in fu.get("openers", all_openers):
            opener_text = spec["openers"][opener].format(category=cfg["category"])
            for brand in seeded["brands"]:
                rival = _top_rival(cfg, brand)
                messages = [
                    {"role": "user", "content": opener_text},
                    {"role": "assistant", "content": seeded["assistant_turn_template"].format(brand=brand)},
                    {"role": "user", "content": fu["text"].format(top_rival=rival)},
                ]
                for engine in cfg["engines"]:
                    for r in range(seeded["repeats"]):
                        rows.append(_row("m3_seeded", engine, messages, r, opener=opener,
                                         follow_up=fu_name, fu_group=fu["group"], seeded_brand=brand,
                                         seeded_canonical=canonical(brand, cfg["brands"])))
    return rows


def canonical(name: str, brands: dict) -> str:
    """Map a surface name (e.g. 'Kirkland') to its brand key (e.g. 'Private label')."""
    low = name.lower()
    for brand, spec in brands.items():
        if low == brand.lower() or low in spec["aliases"]:
            return brand
    return name


def _top_rival(cfg: dict, brand: str) -> str:
    """Symmetric competitor challenge: GC is challenged with Daisy, everyone else with GC."""
    return "Daisy" if brand == cfg["focal_brand"] else cfg["focal_brand"]


def build_probes(cfg: dict, root: Path) -> list[dict]:
    """Targeted probes (loyalty, head-to-head, M2 two-step, M5 price, regions); see prompts/targeted_probes.yaml."""
    spec = load_yaml(root / cfg["prompts"]["probes"])
    cat = cfg["category"]
    items = []  # (section, prompt or messages, followups, meta)

    def add(sec, prompt, followups=None, **meta):
        items.append((sec, prompt, followups or [], meta))

    sec = spec["need_scan"]
    for need, text in sec["prompts"].items():
        add(sec, text.format(category=cat), need=need)
    sec = spec["loyalty"]
    for brand in sec["brands"]:
        for cond, text in sec["conditions"].items():
            add(sec, text.format(brand=brand, category=cat), held_brand=brand,
                held_canonical=canonical(brand, cfg["brands"]), condition=cond)
    sec = spec["head_to_head"]
    for order, text in sec["prompts"].items():
        add(sec, text.format(category=cat), order=order)
    sec = spec["h2h_more"]
    for name, (gc, rival) in sec["pairs"].items():
        tpl = sec["lactaid_template"] if name == "lactaid" else sec["template"]
        for order, a, b in [("gc_first", gc, rival), ("rival_first", rival, gc)]:
            add(sec, tpl.format(a=a, b=b, category=cat), rival=name, order=order)
    sec = spec["two_step"]
    for name, (t1, t2) in sec["pairs"].items():
        add(sec, t1.format(category=cat), [t2], pair=name)

    prod = spec["products"]
    def choice(sec, a, b):
        return sec["template"].format(category=cat, a=a, b=b)
    sec = spec["price_sweep"]
    for sc in sec["scenarios"]:
        gc = prod["gc"].replace("{price}", sc["gc_price"])
        rival = prod[sc["rival"]].replace("{price}", sc["rival_price"])
        for order, a, b in [("gc_first", gc, rival), ("rival_first", rival, gc)]:
            add(sec, choice(sec, a, b), rival=sc["rival"], gap=sc["gap"], gc_price=sc["gc_price"], order=order)
    sec = spec["mulu_price"]
    gc = prod["gc"].replace("{price}", sec["gc_price"])
    for gap, price in sec["mulu_prices"].items():
        mulu = prod["mulu"].replace("{price}", price)
        for order, a, b in [("gc_first", gc, mulu), ("rival_first", mulu, gc)]:
            add(sec, choice(sec, a, b), rival="mulu", gap=f"MULU {gap}", order=order)
    sec = spec["regions"]
    for region, area in sec["areas"].items():
        for qname, q in sec["questions"].items():
            add(sec, q.format(area=area, category=cat), region=region, need=qname)

    rows = []
    for sec, prompt, followups, meta in items:
        msgs = [{"role": "user", "content": prompt}]
        # the cache key covers the full conversation, including follow-up turns
        key = msgs + [{"role": "user", "content": t} for t in followups]
        for engine in cfg["engines"]:
            for r in range(sec["repeats"]):
                row = _row(sec["module"], engine, key, r, **meta)
                if followups:
                    row["messages"], row["followups"] = msgs, followups
                rows.append(row)
    return rows


def build_extra(cfg: dict, root: Path) -> list[dict]:
    """Generic single-turn prompts: one YAML entry per question, no code needed to add more."""
    spec = load_yaml(root / cfg["prompts"]["extra"])
    rows = []
    for p in spec["prompts"]:
        prompt = p["text"].format(category=cfg["category"])
        for engine in cfg["engines"]:
            for r in range(p.get("repeats", spec.get("repeats", 3))):
                rows.append(_row("extra", engine, [{"role": "user", "content": prompt}], r,
                                 prompt_id=p["id"], need=p.get("need", "other")))
    return rows


BUILDERS = {"m1": build_m1, "m3_seeded": build_m3_seeded, "probes": build_probes, "extra": build_extra}
# The M3 natural arm is not built yet (see plan).


def build_plan(config_path: str | Path, modules: list[str] | None = None) -> list[dict]:
    config_path = Path(config_path)
    root = config_path.parent.parent
    cfg = load_yaml(config_path)
    rows = []
    for name, fn in BUILDERS.items():
        if modules is None or name in modules:
            rows.extend(fn(cfg, root))
    return rows


def write_plan(rows: list[dict], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
