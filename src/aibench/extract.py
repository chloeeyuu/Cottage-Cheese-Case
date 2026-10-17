"""Deterministic extraction: brand mentions (in order) and citation-domain classification.

The LLM judge (top pick, sentiment, attributes) lives in judge.py and is validated
against hand labels before any metric is reported.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

RETAILERS = {"walmart.com", "target.com", "kroger.com", "costco.com", "amazon.com", "instacart.com",
             "wholefoodsmarket.com", "safeway.com", "heb.com", "publix.com", "aldi.us", "sprouts.com"}
HEALTH = {"clevelandclinic.org", "mayoclinic.org", "healthline.com", "webmd.com", "nih.gov",
          "harvard.edu", "eatingwell.com", "verywellfit.com", "verywellhealth.com", "medicalnewstoday.com"}
UGC = {"reddit.com", "quora.com", "youtube.com", "tiktok.com", "instagram.com", "facebook.com"}


def mentions(text: str, brands: dict) -> list[str]:
    """Brands mentioned in the answer, ordered by first appearance. Longest aliases win ties."""
    low = text.lower()
    hits = []
    for brand, spec in brands.items():
        positions = [m.start() for a in spec["aliases"]
                     for m in re.finditer(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", low)]
        if positions:
            hits.append((min(positions), brand))
    return [b for _, b in sorted(hits)]


def domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def classify_domain(url: str, brands: dict) -> str:
    d = domain(url)
    for brand, spec in brands.items():
        if any(d == o or d.endswith("." + o) for o in spec.get("owned_domains", [])):
            return f"owned:{brand}"
    if any(d == r or d.endswith("." + r) for r in RETAILERS):
        return "retailer"
    if any(d == h or d.endswith("." + h) for h in HEALTH) or d.endswith(".gov") or d.endswith(".edu"):
        return "health"
    if any(d == u or d.endswith("." + u) for u in UGC):
        return "ugc"
    return "editorial_other"
