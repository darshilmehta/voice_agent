# Evals

Measuring the document agent against known answers. Today this is the **retrieval eval** (docs/DESIGN.md §10, phase 2 tuning and phase 9): does the right document and page come back, and does the agent stay quiet when the documents don't hold the answer? Conversation evals (`conversation_cases.jsonl`, §7) come later.

```text
evals/
├── README.md                      this file
├── canvas/
│   └── README.md                  the blind hold-out set v2 for the chart planner (the set itself:
│                                  backend/tests/integration/canvas_planner_holdout_v2.json)
└── retrieval/
    ├── manifest.json              every planted fact: document, real pages, evidence strings  (committed, text)
    └── questions.jsonl            194 questions with expected documents and pages              (committed)
scripts/eval/
├── build_corpus.py                generates the 5 documents and verifies the manifest against them (no ML)
├── build_questions.py             questions.jsonl from the authored bank, resolved through the manifest
├── question_bank.py               the authored questions (one line each)
└── corpus/                        document content: two financial models, five document builders
backend/app/evals/
├── retrieval_eval.py              the harness (CLI: python -m app.evals.retrieval_eval)
└── textnorm, manifest, questions, metrics, analysis, report   pure Python, unit-tested in CI
data/eval/                         generated, git-ignored: docs/ (the corpus), results/<timestamp>/
```

## 1. Build the corpus

```bash
uv run scripts/eval/build_corpus.py              # builds data/eval/docs/ in a few seconds, no models
```

Everything is deterministic and fictional (Valmora Industries, Zephyra Logistics, a made-up state scheme). Generated documents are never committed (`data/` is git-ignored); the manifest is.

| Document | Format | Pages | Planted facts | What it stresses |
|---|---|---:|---:|---|
| `valmora_annual_report_fy24.pdf` | PDF | 29 | 201 | narrative, multi-page P&L and balance sheet, segment and quarterly tables, footnotes, glossary, CIN/ISIN/policy numbers; FY23 and FY24 facts on separate pages and in shared tables; three segment pages from one template |
| `zephyra_investor_deck_q4fy24.pptx` | PPTX | 15 slides | 58 | a second company with the same metric names (revenue, EBITDA margin, PAT, employees, a Digital Services segment) |
| `valmora_travel_expense_policy.docx` | DOCX | 9 nominal | 56 | numbered clauses (4.2.1), eligibility by grade, limits by city tier, exceptions, a revision history with old values |
| `suryodaya_yojana_soochna.docx` | DOCX (Hindi) | 5 nominal | 42 | Devanagari text, dates, amounts, course and district tables |
| `valmora_group_health_policy_scan.pdf` | PDF, image only | 3 | 22 | OCR (RapidOCR): rasterised, skewed, noisy pages; policy number and coverage table |

### Page numbers are verified, not assumed

While laying pages out the generator *plants* facts: each is registered with the text it sits in and its evidence strings (and the registration asserts that the evidence really is in that text). Every PDF page is laid out in its own frame (`KeepInFrame(mode="error")`), so a page that overflows raises instead of pushing content to the next page. After writing the files the build **reads them back** (pypdf for PDFs, python-pptx for slides, python-docx for DOCX) and, for every fact:

- fails if its planted page's text does not contain the evidence, if the evidence is found nowhere, if a document has a different number of pages than planned, or a DOCX locator is not a heading;
- records every page whose text contains the evidence as a correct page (a headline figure that also appears in the MD&A and the P&L is right on all three pages), so a correct retrieval is never scored as a miss.

The scanned PDF has no text layer: its facts are verified on the vector PDF it was rasterised from, and the build checks the final file really is image-only.

DOCX has no pages in Docling (`page_count` is `None`, chunks carry no page), so DOCX facts are located by their **heading** and matched on text (`pages: []` in the questions). Their "9 pages" are explicit page breaks, nominal only.

The build compares the verified manifest with the committed `evals/retrieval/manifest.json` and fails if they differ (a layout change would silently move pages under the committed questions). After reviewing a deliberate change: `uv run scripts/eval/build_corpus.py --update-manifest`, then regenerate the questions (below). `--show-repeats` lists the facts stated on more pages than planted.

## 2. The question set

`evals/retrieval/questions.jsonl`, one object per line:

```json
{"id": "q046", "question": "What was Valmora's EBITDA margin in FY23?", "language": "en", "category": "wrong_year_trap",
 "expected": [{"document": "valmora_annual_report_fy24.pdf", "pages": [3, 16, 17, 18, 20], "facts": ["vmr.margin.fy23"]}],
 "answer_contains": ["19.8"], "should_abstain": false}
```

| Field | Meaning |
|---|---|
| `expected` | one entry per piece of evidence the answer needs (several = multi-hop or cross-document; all are required). `pages` = every page that states the fact (any one counts; empty for DOCX); `facts` = the manifest facts behind it (their evidence strings drive the fact-level metrics) |
| `answer_contains` | strings the answer must contain (`--answers` mode). `"a|b"` = either spelling; thousands separators and `21.0 %` vs `21.0%` don't matter |
| `should_abstain` | the documents do not hold the answer (`unanswerable` category, ~20%) |
| `query_en` | the English query the router (§3.4) would produce; required for Hindi and Hinglish, absent for English |
| `clean_question` | `asr_noise` only: what the user actually said, without the recognition error |
| `subtype` | `unanswerable` only: `near_miss_year`, `near_miss_segment`, `other_company`, `missing_detail`, `off_topic` |

| Category | n | What it tests |
|---|---:|---|
| `exact_fact` | 16 | numbers and facts stated in prose |
| `paraphrase` | 16 | no keyword overlap with the passage ("operating profit before depreciation" for "EBITDA") |
| `identifier` | 13 | CIN, ISIN, BSE code, policy numbers, clause numbers (4.2.1), auditor registration |
| `wrong_year_trap` | 15 | FY23 vs FY24 (shared tables and separate pages), Q3 FY23 vs Q3 FY24, segment A vs B, policy version 3.1 vs 3.2 |
| `multi_hop` | 10 | two pages of one document |
| `cross_doc` | 13 | the same metric in Valmora's and Zephyra's documents (the company is named in the question, not always in the chunk) |
| `table_cell` | 13 | one cell: balance sheet, plant capacity, hotel limits, per diem |
| `glossary` | 10 | OTIF, bps, ROCE, NCD, KMP, "long-haul flight" |
| `hindi_doc` | 16 | Hindi questions on the Hindi notice |
| `xlingual` | 22 | Hindi and Hinglish questions on the English documents |
| `asr_noise` | 12 | Whisper-style errors: मुनापा for मुनाफा, "ice in" for ISIN, "Wall Mora", "Zefira", "per dime" |
| `unanswerable` | 38 | FY25 and FY22 questions, a segment that doesn't exist, the other company's metric, details the documents lack, off-topic |

Languages: 138 English, 41 Hindi, 15 Hinglish.

Edit questions in `scripts/eval/question_bank.py` (fact ids, not page numbers), then:

```bash
uv run scripts/eval/build_questions.py                 # rewrites questions.jsonl, validates it like CI does
uv run scripts/eval/build_questions.py --verify-docs   # also checks every answer string is in the text of its expected pages
uv run scripts/eval/build_questions.py --check         # CI-style: fail if the committed file is stale
```

## 3. Run the eval

Needs: the `ml` dependency group, the models and Docling artifacts (`scripts/setup/download_models.sh`), Qdrant (`docker compose -f infra/docker-compose.yml up -d qdrant`), and for `--answers` Ollama with the configured chat model. It loads BGE-M3 and the reranker, so close other model-heavy work first (the machine has 16 GB).

```bash
cd backend
uv run --group ml python -m app.evals.retrieval_eval
```

It ingests the five documents into a throwaway Qdrant collection (dropped at the end) with the real `IngestionService`, then asks every question the way the chat pipeline does and writes `data/eval/results/<timestamp>/{results.json,summary.md,chunks.jsonl}`. Expect about 1-2 minutes of ingestion (Docling, OCR, embeddings) and about 5 minutes of queries; `--answers` adds the LLM time of ~190 answers.

Environment (the same variables as `tests/integration`):

| Variable | Default | |
|---|---|---|
| `MODELS_ROOT` | `<repo>/data/models` | folder with `huggingface/` and `docling/` |
| `QDRANT_URL` | `http://127.0.0.1:6333` | |
| `EVAL_DOCS` | `<repo>/data/eval/docs` | the generated corpus |
| `EVAL_RESULTS` | `<repo>/data/eval/results` | where run folders go |
| `APP_CONFIG_FILE`, `SECTION__KEY` | `config/local.config.json` | any config override works, e.g. `EMBEDDINGS__DEVICE=cpu`, `LLM__CHAT_MODEL=qwen3:8b` |

Parameters to compare (comma-separated lists run every combination on one ingestion; a chunk size re-ingests):

| Flag | Config key | |
|---|---|---|
| `--prefetch-k 8,12,20` | `retrieval.prefetch_k` | candidates per search leg before fusion |
| `--rerank-top-n 5,8` | `retrieval.rerank_top_n` | passages kept after the reranker |
| `--chunk-size 300,500,700 [--overlap 80]` | `ingestion.chunking.*` | |
| `--fusion rrf,dbsf` | `retrieval.fusion` | |
| `--threshold 0.05` | `retrieval.min_rerank_score` | the abstention gate used for the confusion matrix (default: the configured value); the sweep covers all thresholds anyway |

Other flags: `--answers` (also run the answer LLM, off by default), `--variants raw,routed,english,clean`, `--categories xlingual,asr_noise`, `--languages hi,hinglish`, `--ids q001,q002`, `--limit 20` (quick smoke run), `--out DIR`, `--keep-collection` / `--reuse-collection NAME` (skip ingestion when only retrieval parameters change), `--render results.json` (re-render `summary.md` without any model).

Examples:

```bash
uv run --group ml python -m app.evals.retrieval_eval --prefetch-k 8,12,20 --threshold 0.05      # phase 2: how deep to search
uv run --group ml python -m app.evals.retrieval_eval --chunk-size 300,500,700                    # phase 2: chunk size
uv run --group ml python -m app.evals.retrieval_eval --answers --limit 40                        # answer quality on a slice
```

### What it measures, exactly

For each question it runs up to four **query variants**: `raw` = `retrieve(question)` (what the phase-1 chat does); `routed` = `retrieve(question, query_en=...)` for Hindi and Hinglish questions (both queries are searched and fused, the reranker scores the English one: what the phase-3 router will do); `english` = `retrieve(query_en)` alone; `clean` = the correctly transcribed question for ASR-noise questions. The headline **pipeline** view uses `routed` where a question has an English query and `raw` otherwise. Retrieval is `RetrievalService.retrieve` unchanged; the candidates before the reranker come from the public `RetrievalService.search`. The answer step (`--answers`) builds sources, the prompt and the citations with the same functions as `ChatTurnService`. The harness changes no retrieval behaviour.

## 4. Reading `summary.md`

- **Recall@1/3/5/all and MRR**, before the reranker ("candidates") and after it. A passage *hits* when it is in the right document and its page range includes an expected page (page-level), or, stricter, when its text contains the fact's evidence (fact-level: the right page but the wrong chunk is a miss). Recall@k is the share of expected evidence in the top k (a two-hop question with one page in the top 3 scores 0.5); Success@k needs all of it; MRR uses the first hit. Recall@all of the candidates is the ceiling the reranker cannot exceed: if it is low, raise `prefetch_k`; if candidates are fine and the reranked numbers are not, the reranker is the problem.
- **Page-citation accuracy**: would citing the best passage cite a right page (lenient: an expected page is in the cited range; exact: the range stays within the expected pages). **Context recall**: the sources handed to the answer step hold all expected evidence.
- **By category and by language**, **raw vs router-rewritten vs English-only query** (does `query_en` help or hurt, per category: the English documents and the Hindi one), **ASR noise** (noisy vs clean).
- **Abstention at the current `min_rerank_score`**, with the pipeline's own gate (`Confidence.above_threshold`: the best score reaches the threshold, the best passage states every fiscal year the question names, and some passage found is about the company the question names; a question whose year or company is missing is *vetoed* at any threshold): the 2x2 matrix (answered / abstained x the documents hold the answer / don't), answer precision and recall, **hallucination risk** (unanswerable questions the gate would answer) and **false-abstain rate**, per language and, for the unanswerable questions, **per subtype** (near misses need other signals than off-topic questions). The score-only matrix is in `results.json` (`abstention.score_gate`) for comparison.
- **Threshold sweep**: the same matrix (same gate) over thresholds, overall and per language; the thresholds that maximise F1 or keep 95% of answerable questions; the **AUC** of the top score, the gap to the runner-up and the dense similarity as signals (1.0 separates perfectly, 0.5 not at all); score distributions and histograms per language, answerable vs unanswerable. This is where "reranker scores aren't calibrated across languages" (§9.2) becomes a number.
- **Latency per stage** (embed, search, rerank, total; warmed up), **ingestion** per document with *facts lost in ingestion*: planted facts whose text is in no chunk, so a failure there is Docling, OCR or chunking, not retrieval.
- **Misses** and **abstention errors**: the concrete questions to look at.
- With several runs, a **comparison table** puts the grid side by side.

`chunks.jsonl` holds every chunk (pages, content type, heading path, text) for debugging a miss; `results.json` holds the aggregates and one compact row per query.

## 5. Tests

`backend/tests/test_eval_*.py` run in CI without models: the metrics (recall, MRR, sweep, confusion), the manifest and page-verification logic, the question schema, the committed manifest and questions (every `expected` document and page exists and matches its facts; categories, languages and `query_en` rules), and the whole harness end to end with fake providers (ingest, both query variants, gate, answers, grid, outputs).

## Limits worth knowing

- The documents are synthetic: clean layouts and consistent numbers. Real filings are messier; treat absolute scores as a regression baseline, not a forecast.
- DOCX questions can't be page-checked (Docling gives no pages); they are scored on text.
- Evidence strings of the OCR document are matched ignoring spaces and punctuation.
- Passage text for fact-level matching is the heading path plus the chunk text; the overlap borrowed from the previous chunk is excluded.
- Answer checking is string containment, so it is blind to a correct answer in other words; add alternatives with `|`.
- Retrieval is not bit-for-bit repeatable. The last candidates of the vector search (around rank 7 to 8 of `prefetch_k`) flip between runs of the same code on the same collection (fp16 on the GPU): between two full runs 14 of 318 rows had another top 3 that nothing in the code between them could explain, and Recall@5 moved by 0.6 points. Compare two runs row by row in `results.json` (`rows[].top`) before reading a small difference as an effect.
