# PAPER DRAFT — 100 AI-PhD Math Exercises, Verified End-to-End (submission-ready skeleton)

## Title
From Video Claims to Checked Benchmarks: Verifying 100 AI-PhD Mathematics Exercises on
Synthetic, Public, and Live-Market Data

## Abstract
See README Sec. 0. We formalize 100 widely-taught ML-math claims as falsifiable numeric
predictions, verify 100/100 in a pinned container, ground the statistical subset on live-market
series anchored 2026-10-06, and report four fitted cross-cutting patterns. All artifacts reproduce
via `make all`.

## 1 Introduction
Motivation: AI papers assume fluency in shapes, spectra, calculus, backprop, probability,
estimation, information, optimization, numerics, and transformer arithmetic. Gap: learners lack a
single checked reference. Contribution: (i) harness, (ii) live grounding, (iii) patterns (cf. README).

## 2 Related work
Scaling laws (Kaplan 2020; Hoffmann et al. 2022; Schaeffer et al. 2025 robustness; Sardana et al.
2024 inference-optimal); PEFT/LoRA (Hu et al. 2022; LG-/La-LoRA 2025); optimization (AdamW;
D2L-AdamW dynamics; muP weight-decay scaling Wang et al. 2025); attention efficiency (KV-cache,
GQA, FlashAttention tiling); diffusion (Ho et al. 2020); RL (PPO/GRPO/GRPO-CARE 2025; GDRO 2026);
numerics (bf16 vs fp16, fp32 master); reproducibility (RepoCheck 2026; AIFaultBench 770 faults;
OpenRepro-Agent v1.26; GSLHub governed metrics); repo practice 2026 (Szyller: uv, ruff,
hydra, wandb, pre-commit, AGENTS.md).

## 3 Method: verification protocol (zero-to-hero)
For each exercise: (a) extract falsifiable number from transcript; (b) implement minimal
numpy/torch recomputation with seeded RNG; (c) assert with stated tolerance; (d) run
`pytest` + `run_all.py` + Docker; (e) never hand-edit results — only code, then re-execute.
Statistical claims additionally run on BTC/EUR/AAPL live snapshot. Search protocol: strictly
sequential single-query searches (429 backoff; DuckDuckGo-lite fallback), each tool with distinct
keywords — websearch (repo practices), duckduckgo (LoRA/scaling), searxng (attention, failed ->
fallback, recorded), openresearch web/openalex/HN/SO/news (transformers, LoRA, scaling, tooling),
paper-search unified+arxiv+semantic (Chinchilla, GRPO/diffusion, Adam), agent-reach web
(AdamW/bf16), wiki (Chinchilla/NSL), kaggle search+discussions (perplexity, clipping),
gitmcp pytorch docs (SDPA/einsum), gsd_websearch (RoPE), superpowers (repro skills),
webfetch lite-DuckDuckGo (20 tok/param), openresearch crypto/FX + agent stock (live anchors).

## 4 Results
4.1 Synthetic: 100/100 pass (table in README Sec. 3; raw JSON in benchmarks/results.json).
4.2 Live-market: Ex44/57-60 hold; BTC CI includes 0; fat tails (kurtosis>0); vol hierarchy
BTC>>EUR (benchmarks/live_market.json).
4.3 Patterns P1-P4 with fits (benchmarks/hidden_patterns.json).

## 5 Discussion (hidden patterns + implication)
As in README Sec. 5: quadratic context cost vs scale-invariant steps => prioritize context
efficiency. RoPE relativity + Chinchilla sqrt rule complete the scaling picture.

## 6 Limitations, ethics, reproducibility
As in README Sec. 6; no human subjects; public market data only; synthetic fixtures separated
from empirical claims; MIT code, CC-BY-4.0 text.

## 7 Search log (auditable)
2026-10-06 sequential: websearch deep x8 (repo practices) -> duckduckgo x8 (LoRA/Chinchilla) ->
searxng (network error, recorded, fallback used) -> openresearch web x5 (KV/Flash) ->
paper-search unified 6 papers (Chinchilla) -> agent-reach web 8 (AdamW/bf16) -> wiki 5
(Chinchilla) -> kaggle everything (perplexity notebooks) -> gitmcp pytorch docs (SDPA) ->
gsd_websearch (RoPE, empty, recorded) -> superpowers 5 (repro) -> HN 1 (Chinchilla limits) ->
SO 0 (recorded) -> arxiv 5 (GRPO/diffusion) -> kaggle discussions 5 (clipping/RL) ->
news 5 (2026 scaling) -> searxng suggestions (empty) -> webfetch lite-DD 10 (20 tok/param) ->
crypto 30d BTC + FX 22pts + AAPL quote (anchors) -> openalex 3 (LoRA) -> semantic 0 (recorded).

## References
Abridged in README Sec. 7; full .bib to be generated on acceptance. Data/code: this repo,
commit-pinned; `CITATION.cff` to be added with DOI on release.
