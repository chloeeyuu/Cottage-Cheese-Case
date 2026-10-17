"""Metric definitions. Every rate comes with its n and a Wilson 95% interval."""
from __future__ import annotations

import math

import pandas as pd

Z = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def rate(k: int, n: int, min_n: int = 20) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
            "interpretable": n >= min_n}


def top_pick_share(df: pd.DataFrame, by: list[str], min_n: int = 20) -> pd.DataFrame:
    """TPS per brand within groups. `df` has one row per answer with column `top_pick`
    (a brand, or 'none' / 'conditional' / 'refusal'). Rule A: conditional picks count in n
    but credit no brand."""
    out = []
    for keys, g in df.groupby(by) if by else [((), df)]:
        n = len(g)
        for brand, k in g["top_pick"].value_counts().items():
            if brand in ("none", "conditional", "refusal"):
                continue
            keys = keys if isinstance(keys, tuple) else (keys,)
            out.append({**dict(zip(by, keys)), "brand": brand, **rate(int(k), n, min_n)})
    return pd.DataFrame(out)


def over_index(tps: float, dollar_share: float) -> float:
    return tps / dollar_share if dollar_share else float("nan")


def arr(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Seeded-arm ARR. One row per thread: seeded_brand, follow_up, top_pick (after the follow-up).
    Net ARR = retention after a follow-up minus retention after the placebo, per brand."""
    target = df["seeded_canonical"] if "seeded_canonical" in df else df["seeded_brand"]
    df = df.assign(retained=df["top_pick"] == target)
    g = df.groupby(["seeded_brand", "follow_up"])["retained"].agg(["sum", "count"]).reset_index()
    rows = [{"seeded_brand": r.seeded_brand, "follow_up": r.follow_up, **rate(int(r["sum"]), int(r["count"]), min_n)}
            for _, r in g.iterrows()]
    res = pd.DataFrame(rows)
    placebo = res[res.follow_up == "placebo"].set_index("seeded_brand")["rate"]
    res["net_arr"] = res["rate"] - res["seeded_brand"].map(placebo)
    return res


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over the same items (nominal labels)."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
