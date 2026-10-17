"""Result tables printed by `aibench summary` (and `aibench aspects`).

All brand names are passed through build.canonical so that judge spellings such as "MULU",
"Great Value" or "Trader Joe's" are counted under their brand key (Mulu, Private label).
"""
from __future__ import annotations

import pandas as pd

from . import build, extract, metrics

META = ("module", "engine", "need", "stage", "follow_up", "seeded_brand", "seeded_canonical", "held_canonical",
        "condition", "order", "pair", "gap", "rival", "region", "prompt_id")


def frame(rows: list[dict], cfg: dict) -> pd.DataFrame:
    """One row per coded answer: design metadata + canonical top pick + held-brand verdict + citations."""
    canon = lambda b: build.canonical(b or "none", cfg["brands"])
    return pd.DataFrame([{**{k: r.get(k) for k in META},
                          "top_pick": canon(r["coded"].get("top_pick")),
                          "verdict": r["coded"].get("held_brand_verdict", "n/a"),
                          "cites": r.get("citations") or []} for r in rows])


def _xtab(df, module, index, columns="top_pick", title=None):
    sub = df[df.module == module]
    if len(sub):
        print(f"\n{title or module}:")
        idx = [sub[i] for i in index] if isinstance(index, list) else sub[index]
        print(pd.crosstab(idx, sub[columns]).to_string())
    return sub


def summary(rows: list[dict], cfg: dict) -> None:
    pd.set_option("display.width", 200)
    df = frame(rows, cfg)

    m1 = df[df.module == "m1"]
    if len(m1):
        print("\nM1 top-pick mix (all answers):")
        print(m1.top_pick.value_counts(normalize=True).round(3).to_string())
        tps = metrics.top_pick_share(m1, ["need"], cfg.get("min_n", 20))
        print("\nM1 top-pick share by need state:")
        print(tps.pivot_table(index="brand", columns="need", values="rate").round(2).fillna(0).to_string())

    m3 = df[df.module == "m3_seeded"]
    if len(m3):
        print("\nM3 seeded ARR (net of placebo):")
        res = metrics.arr(m3, cfg.get("min_n", 20))
        print(res.pivot_table(index="follow_up", columns="seeded_brand", values="net_arr").round(2).to_string())

    _xtab(df, "pilot_need", "need", title="Need scan (early; superseded by M1)")
    pl = df[df.module == "pilot_loyalty"]
    if len(pl):
        print("\nLoyalty: share of answers telling the shopper to STAY with their brand:")
        stay = pl.assign(stay=pl.verdict == "stay").pivot_table(index="condition", columns="held_canonical",
                                                                 values="stay", aggfunc="mean")
        print(stay.round(2).to_string())
        if "baseline" in stay.index:
            print("\n  net of baseline (pressure effect):")
            print((stay - stay.loc["baseline"]).drop("baseline").round(2).to_string())
    _xtab(df, "pilot_h2h", "order", title="Head-to-head GC vs Daisy, by name order")
    _xtab(df, "pilot_h2h_more", ["rival", "order"], title="Head-to-head GC vs other rivals (M4)")
    _xtab(df, "pilot_m2", "pair", title="M2 two-step: turn-2 brand pick")
    _xtab(df, "pilot_m5", ["rival", "gap"], title="M5 price sweep (hypothetical prices)")
    _xtab(df, "pilot_mulu_price", "gap", title="M5 Mulu price sweep (hypothetical prices)")
    _xtab(df, "pilot_region", ["need", "region"], title="Regions")
    _xtab(df, "extra", ["need", "engine"], title="Extra prompts: top pick by need x engine")

    cites = [extract.classify_domain(u, cfg["brands"]) for c in df.cites for u in c]
    if cites:
        print("\nCitation mix (all answers):")
        print(pd.Series(cites).value_counts().to_string())


def aspect_tables(rows: list[dict], cfg: dict,
                  brands=("Good Culture", "Daisy", "Private label", "Mulu", "Friendship", "Breakstone's")) -> None:
    items = [it | {"engine": r["engine"]} for r in rows for it in r.get("items", [])]
    if not items:
        return
    df = pd.DataFrame(items)
    df["brand"] = df.brand.map(lambda b: build.canonical(b, cfg["brands"]))
    print(f"\n{len(df)} pro/con items; quotes verified verbatim: {df.quote_verified.mean():.0%}")
    df = df[df.quote_verified & df.brand.isin(brands)]
    pd.set_option("display.width", 200)
    for pol in ("pro", "con"):
        print(f"\n{pol.upper()} mentions by aspect x brand (verified quotes only):")
        print(pd.crosstab(df[df.polarity == pol].aspect, df[df.polarity == pol].brand).to_string())


def agreement(main: dict, second: dict, cfg: dict, names=("judge A", "judge B"), out=None) -> None:
    """Kappa between two judges on the answers both coded; writes the disagreements to CSV for review."""
    ids = sorted(set(main) & set(second))
    if not ids:
        raise SystemExit("no answers coded by both judges yet")
    canon = lambda b: build.canonical(b or "none", cfg["brands"])
    focal = cfg["focal_brand"]

    def tags(c, brand):
        for b in c.get("brands") or []:
            if canon(b.get("brand")) == brand:
                return set(b.get("attributes") or [])
        return None

    rows = []
    for t in ids:
        a, b = main[t]["coded"], second[t]["coded"]
        ta, tb = tags(a, focal), tags(b, focal)
        rows.append({"task_id": t, "module": main[t].get("module"), "engine": main[t].get("engine"),
                     "top_main": canon(a.get("top_pick")), "top_2nd": canon(b.get("top_pick")),
                     "verdict_main": a.get("held_brand_verdict", "n/a"),
                     "verdict_2nd": b.get("held_brand_verdict", "n/a"),
                     "focal_pricey_main": None if ta is None else "pricey" in ta,
                     "focal_pricey_2nd": None if tb is None else "pricey" in tb,
                     "why_main": a.get("rationale", ""), "why_2nd": b.get("rationale", "")})
    df = pd.DataFrame(rows)
    k = metrics.cohen_kappa
    print(f"\n{len(df)} answers coded by both judges ({names[0]} vs {names[1]})")
    print(f"TOP PICK   agreement {(df.top_main == df.top_2nd).mean():.0%}, "
          f"kappa {k(list(df.top_main), list(df.top_2nd)):.2f}   (target >= 0.80)")
    v = df[(df.verdict_main != "n/a") | (df.verdict_2nd != "n/a")]
    if len(v):
        print(f"VERDICT    agreement {(v.verdict_main == v.verdict_2nd).mean():.0%}, "
              f"kappa {k(list(v.verdict_main), list(v.verdict_2nd)):.2f}   (n={len(v)}, loyalty answers)")
    p = df.dropna(subset=["focal_pricey_main", "focal_pricey_2nd"])
    if len(p):
        print(f"{focal} tagged 'pricey'  agreement {(p.focal_pricey_main == p.focal_pricey_2nd).mean():.0%}, "
              f"kappa {k(list(p.focal_pricey_main), list(p.focal_pricey_2nd)):.2f}   (n={len(p)}; target >= 0.60)")
    print("\nTop-pick agreement by engine and module:")
    print(df.assign(agree=df.top_main == df.top_2nd).pivot_table(index="module", columns="engine",
                                                                  values="agree", aggfunc="mean").round(2).to_string())
    dis = df[df.top_main != df.top_2nd]
    if len(dis):
        print(f"\nMost common top-pick disagreements ({names[0]} -> {names[1]}):")
        print(dis.groupby(["top_main", "top_2nd"]).size().sort_values(ascending=False).head(10).to_string())
    if out is not None:
        dis.to_csv(out, index=False)
        print(f"\n{len(dis)} disagreements -> {out}")
