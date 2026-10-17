# Cottage-Cheese-Case: Good Culture in AI answer engines

A take-home case study. It asks whether Good Culture is the brand AI answer engines
(ChatGPT, Gemini, Claude) **tell shoppers to buy**, and whether that position would hold if a larger
dairy player pushed harder into the category.

> **Status:** data collected and coded. 1,596 answers (532 per engine), each coded by two LLM judges.
> Still to do: hand labels on judge disagreements, the write-up and the webpage.

## How it works

One YAML config drives the whole pipeline:

```
config -> prompt builder -> collector (cached) -> answers -> two LLM judges (+ optional pros/cons pass) -> report
```

To swap in another brand (e.g. Birkenstock), write a new `configs/<brand>.yaml` and its prompt files.
The code does not change.

Engines are called with web search on and no system prompt, using the model each free consumer tier runs:
`gpt-6-luna` (OpenAI), `gemini-3.5-flash-lite` (Google), `claude-sonnet-5-5` (Anthropic).

## Judging

Every answer is coded twice with the same rules (`src/aibench/judge.py`): top pick, held-brand verdict
(stay / switch), and each brand's sentiment and attributes.

| Judge | Model | Role |
|---|---|---|
| `qwen` | Qwen3.8 27B via OpenRouter (open weights, reasoning on, host pinned) | **Primary**: a different family from all three engines, so no engine is scored by its own maker |
| `haiku` | Claude Haiku 5.5 | Independent check |

Agreement on all 1,596 answers: top pick kappa 0.84 (87%), stay/switch kappa 0.71, "Good Culture is pricey"
kappa 0.91. Top-pick shares differ by at most 1.1 points between the two judges (Good Culture 32.7% vs 31.8%).
Most disagreements are "clear pick vs conditional pick" (e.g. "Good Culture is best; Good & Gather is the
budget option").

## What is asked

| Prompt file | Module(s) | Question | Answers (3 engines) |
|---|---|---|---|
| `m1_need_journey.yaml` | M1 | Who does AI recommend, unprompted, across 9 shopper needs x 3 funnel stages x 3 phrasings? | 486 |
| `m3_pushback_resilience.yaml` | M3 | After AI "recommends" GC / Daisy / Kirkland, does it hold under pushback (price, taste, protein, retailer, region, competitor)? Net of a placebo follow-up | 522 |
| `targeted_probes.yaml` | loyalty, head-to-head, M2, M5, regions | "I usually buy X, should I switch?"; GC vs each rival in both name orders; category question then "which brand?"; switch point at hypothetical prices; regional home advantage | 498 |
| `extra_questions.yaml` | extra | Gut health, store brand vs name brand, kids, best buy at each retailer | 90 |

Module labels in `targeted_probes.yaml` keep their original `pilot_*` names because the label is part of each
answer's cache key.

## Run it

```bash
pip install -e ".[dev]"
pytest                                                                    # design checks
python -m aibench.cli plan    --config configs/goodculture.yaml            # call budget per module x engine
python -m aibench.cli collect --config configs/goodculture.yaml --modules extra --engines anthropic gemini openai
python -m aibench.cli score   --config configs/goodculture.yaml --judge qwen    # code answers not yet coded
python -m aibench.cli score   --config configs/goodculture.yaml --judge haiku   # second judge
python -m aibench.cli agree   --config configs/goodculture.yaml                 # kappa + disagreements CSV
python -m aibench.cli summary --config configs/goodculture.yaml                 # all result tables
python -m aibench.cli aspects --config configs/goodculture.yaml            # optional: pros/cons per brand
```

Every step is cached, so a rerun skips anything already done. API keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
go in `.env` (see `.env.example`);
`AIBENCH_MOCK=1` runs everything offline.

To add a question, add one line to `prompts/extra_questions.yaml`. No code change is needed.

## Repo layout

```
configs/goodculture.yaml   brands, aliases, owned domains, engines, judges
prompts/
  m1_need_journey.yaml         M1: 9 needs x 3 stages x 3 phrasings, brand-free
  m3_pushback_resilience.yaml  M3: seeded recommendation + pushback follow-ups
  targeted_probes.yaml         loyalty, head-to-head, M2 two-step, M5 price, regions
  extra_questions.yaml         one-line questions; add more here
src/aibench/
  build.py                 prompt files -> run plan (one row per API call, stable task_id)
  engines.py               HTTPS adapters for OpenAI / Gemini / Anthropic, with web search
  judge.py                 judge rules + schema; Anthropic (tool call) and OpenRouter (JSON) backends
  aspects.py               optional pass: pros / cons per brand with verbatim, verified quotes
  extract.py               brand mentions, citation domain classes
  metrics.py               top-pick share, over-index, ARR, Wilson intervals, Cohen's kappa
  report.py                result tables and judge agreement
  cli.py                   the commands above
data/reference/            Circana category data, product labels and prices (with sources and dates)
data/raw/goodculture_answers.jsonl                  every engine answer with citations (append-only)
data/processed/goodculture_judged_qwen.jsonl        primary coding
data/processed/goodculture_judged_haiku.jsonl       second coding
data/processed/goodculture_judge_disagreements.csv  answers where the two judges differ on the top pick
tests/test_pipeline.py     design checks (counts, no brand leakage, cache keys, parsers, metrics)
```

## Cost and data sources

The brief asked for no spending. The author chose to pay for the API calls herself: roughly $50 in
total, estimated from token usage (Claude engine ~$23, OpenAI ~$12, Gemini free tier,
Haiku judge ~$6, Qwen judge ~$6).

- Circana category data via [Dairy Foods, May 2026](https://www.dairyfoods.com/articles/99101-cottage-cheese-sales-top-2-billion)
- Product labels and prices: `data/reference/product_labels.csv`; each row records its source URL and capture date

Outside-in only: public sources, and no contact with the company, retailers or competitors.
