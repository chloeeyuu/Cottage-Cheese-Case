# Good Culture Through AI Answer Engines

A benchmark of how ChatGPT, Gemini and Claude recommend cottage cheese brands: where Good Culture wins,
who it loses to, whether its position would hold against a larger dairy brand, and what shapes the answers.

**Main page:** [`docs/index.html`](docs/index.html), also served with GitHub Pages at
<https://chloeeyuu.github.io/Cottage-Cheese-Case/>. It contains the findings, a dashboard of nine measures,
the method, every prompt and response, and an appendix with tools, time spent and sources.

## At a glance

- 1,596 prompt-response pairs, 532 per engine, from the free-tier default models with web search on:
  `gpt-6-luna` (OpenAI), `gemini-3.5-flash-lite` (Google), `claude-sonnet-5-5` (Anthropic).
- Good Culture is the top pick in 46% of brand and purchase questions, more than four times its 11% share of
  dollar sales (Circana). It leads on taste and everyday use, and is weaker on price, mass retail and,
  on Gemini and Claude, protein.
- Every answer is coded by an independent judge model and cross-checked by a second model and by hand labels.

## How it works

One YAML config drives the whole pipeline. To study another brand, write a new config and question files.

```
config -> prompt builder -> collector (cached) -> answers -> judge models -> metrics -> static site
```

## Judging and validation

| Judge | Model | Role |
|---|---|---|
| `qwen` | Qwen3.8 27B via OpenRouter (open weights, reasoning on, host pinned) | Primary: from a different company than all three engines |
| `haiku` | Claude Haiku 5.5 | Independent cross-check on every answer |

The two judges agree on the top pick in 87% of answers (Cohen's κ 0.84). On 49 hand-labelled answers,
mostly cases where the judges disagree, the primary judge's top pick is right about 82% of the time after
re-weighting to the full data. "It depends" answers credit no brand, so top-pick shares are conservative.

## What is asked

| Prompt file | Test | Answers (3 engines) |
|---|---|---|
| `prompts/core_need_grid.yaml` | 9 consumer needs x 3 purchase stages x 3 phrasings, no brand named | 486 |
| `prompts/pushback_tests.yaml` | After a seeded recommendation (Good Culture, Daisy, Kirkland), does it survive a follow-up on price, taste, protein, store, region or a rival? | 522 |
| `prompts/supplementary_tests.yaml` | Loyalty, head-to-head in both name orders, price gaps, regions | 498 |
| `prompts/store_and_need_questions.yaml` | Best buy at five stores, plus gut health, kids and store brands | 90 |

Module labels inside the prompt files keep their original names (for example `pilot_h2h`) because the label
is part of each answer's cache key.

## Run it

```bash
pip install -e ".[dev]"
pytest                                                                      # design checks
python -m aibench.cli plan    --config configs/goodculture.yaml              # call budget per test x engine
python -m aibench.cli collect --config configs/goodculture.yaml --repeats 2  # engine answers (cached)
python -m aibench.cli score   --config configs/goodculture.yaml --judge qwen # primary judge
python -m aibench.cli score   --config configs/goodculture.yaml --judge haiku
python -m aibench.cli agree   --config configs/goodculture.yaml              # judge agreement + disagreements CSV
python -m aibench.cli gold-sheet --config configs/goodculture.yaml           # hand-label sheet
python -m aibench.cli gold-score --config configs/goodculture.yaml           # judges vs hand labels
python -m aibench.cli export  --config configs/goodculture.yaml              # answers, mentions, citations as CSV
python -m aibench.cli site    --config configs/goodculture.yaml              # builds docs/index.html
```

Every step is cached, so a rerun skips anything already done. API keys (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) go in `.env` (see `.env.example`).
`AIBENCH_MOCK=1` runs everything offline.

## Repository layout

```
configs/
  goodculture.yaml              brands, aliases, owned websites, engines, judges
  goodculture_findings.json     brand-specific text shown on the site
prompts/                        question files (see table above)
src/aibench/
  build.py                      prompt files -> run plan (one row per API call, stable task_id)
  engines.py                    HTTPS adapters for OpenAI, Gemini and Anthropic, with web search
  judge.py                      judging rules and schema; Anthropic and OpenRouter backends
  extract.py                    brand mentions, citation source classes
  metrics.py                    top-pick share, over-index, retention, Wilson intervals, Cohen's kappa
  analysis.py                   result tables and the nutrition fact check
  gold.py                       hand-label sheet and judge scoring
  aspects.py                    optional pass: pros and cons per brand with verified quotes
  report.py                     summary tables and judge agreement
  site.py, site_template.html   builds the static page
  cli.py                        the commands above
data/
  raw/goodculture_answers.jsonl                   every engine answer with citations (append-only)
  processed/goodculture_judged_qwen.jsonl         primary coding
  processed/goodculture_judged_haiku.jsonl        second coding
  processed/goodculture_judge_disagreements.csv   answers where the two judges differ
  processed/goodculture_table_*.csv               answers, mentions and citations as flat tables
  processed/goodculture_results.json              all computed results
  validation/goodculture_hand_labels.xlsx         hand-label sheet (49 answers)
  reference/circana_sales_by_brand.csv            category dollar sales by brand
  reference/product_labels.csv                    labels and prices, each with source URL and capture date
docs/index.html                 the main page
tests/test_pipeline.py          design checks (counts, no brand leakage, cache keys, parsers, metrics)
```

## Extending to other brands

The pipeline, judges, metrics and page are reusable. For a new brand (for example Birkenstock), the config,
question files, judge attribute list, fact-check fields, price test and findings text change. The Method tab
on the site lists each change.

## Cost and sources

API costs came to about $50 (Claude engine ~$23, OpenAI ~$12, Gemini free tier, each judge ~$6) and were
covered by Harvard University's education plan for students. All sources are listed in the site's Appendix.
The study is outside-in only: public sources, and no contact with the company, retailers or competitors.
