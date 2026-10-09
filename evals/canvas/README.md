# Chart planner hold-out set, v2

`backend/tests/integration/canvas_planner_holdout_v2.json`: 36 requests to see a figure from the eval corpus, in English (18), Hindi (9, Devanagari) and Hinglish (9, romanised), each labelled with the table, the series and the chart kind a careful person would pick. It measures the visual planner (docs/DESIGN.md §12.1) on questions it has never been tuned on.

## How it was written (blind)

The author wrote it from the corpus, not from the planner:

- **Seen:** the generated documents (`uv run scripts/eval/build_corpus.py`, then the text of every page, slide and DOCX section with pypdf, python-pptx and python-docx), `scripts/eval/corpus/finance.py` (the numbers), `evals/README.md`, the head of `evals/retrieval/manifest.json`, and docs/DESIGN.md §12.1 (the visual kinds and the contract).
- **Never opened:** `backend/app/services/canvas/`, `canvas_planner_cases.json` (the tuning set), `canvas_planner_holdout.json` (hold-out v1), any `test_canvas_*` file except the lines of `test_canvas_planner_eval.py` that read a case (its docstring and the loop that consumes `question`, `language`, `query_en`, `kinds`, `datasets` and `series`), so that the case format could be matched.
- DESIGN §12.1 quotes three questions from hold-out v1 and summarises its misses (a lookalike table of the other company, series on bridge and balance-sheet questions). None of those questions was reused, but v2 does ask about some of the same tables (shareholding, the businesses' shares) in other words, and it deliberately includes lookalike tables of the other company, as the task asked. Treat v2 as a *second* hold-out, not as proof the planner is fixed: once its misses are tuned on, v2 is spent too, and v3 should be written the same way.

Table inventory used (every table that can be charted; 26 are the expected table of at least one question):

| Document | Tables |
|---|---|
| `valmora_annual_report_fy24.pdf` | p3 Financial highlights · p6 Business overview (shares) · p10 Operations and manufacturing footprint · p12 People and culture · p14 Corporate governance report (committees) · p18 Segment results · p19 Quarterly performance FY24 · p20 Quarterly performance FY23 · p21-22 Statement of profit and loss (parts 1 and 2) · p23 Balance sheet, assets · p24 Balance sheet, equity and liabilities · p25 Cash flow statement · p26 Note 14 Borrowings · p27 Note 31 Contingent liabilities · p28 Dividend · p28 Shareholding pattern |
| `zephyra_investor_deck_q4fy24.pptx` | slide 3 At a glance · 5 Segment overview · 9 Q4 FY24 results · 10 FY24 profit and loss summary · 12 Network and fleet · 15 Stock information (shareholding rows) |
| `valmora_travel_expense_policy.docx` | §5.1 Hotel limits per night |
| `suryodaya_yojana_soochna.docx` (Hindi) | §5 important dates · §7 courses and seats · §8 districts, targets and budget |

Not used: the scanned health policy (OCR text, nothing worth charting), the approval, eligibility and per-diem tables of the travel policy (mixed currencies or text), the risk, corporate-information and glossary tables.

## What is in it

| | en | hi | hinglish | total |
|---|---:|---:|---:|---:|
| questions | 18 | 9 | 9 | 36 |
| expected a visual | 16 | 8 | 8 | 32 |
| expected no visual (`kind: "none"`) | 2 | 1 | 1 | 4 |

Expected kind: grouped_bar 8, bar 6, donut 4, kpi 4, stacked_bar 3, line 2, waterfall 2, comparison 1, table 1, timeline 1, none 4. Expected table's company: Valmora 19 (18 annual report, 1 travel policy), Zephyra 10, the Hindi scheme notice 3, none 4. The question names the company in 26 cases and omits it in 10 (where only one table is plausible, or where a lookalike exists and the planner must choose: `tags` has `no_company_named`, `company_named`, `lookalike_table_elsewhere`).

It covers: trends over periods, segment comparisons, composition and share, KPI-style single figures with a change, bridges (Zephyra EBITDA to PAT on one slide; Valmora revenue to profit, split over two pages), FY23 against FY24, two cases that need two tables, a timeline, an "exact figures" table request, tables with subtotal rows or mixed units, and four requests for a chart of something no table holds.

## Format

```json
{"version": 2, "documents": ["valmora_annual_report_fy24.pdf", "..."],
 "cases": [{
   "id": "v2-en-05", "question": "…", "language": "en|hi|hinglish", "query_en": "… (hi / hinglish only)",
   "kinds": ["waterfall", "bar"], "datasets": [[1, "FY24 profit and loss summary"]], "series": ["EBITDA", "…"],
   "expected": {
     "kind": "waterfall", "acceptable_kinds": ["bar"], "company": "Zephyra",
     "document": "zephyra_investor_deck_q4fy24.pptx",
     "table": {"title": "FY24 profit and loss summary", "slide": 10},
     "series": ["EBITDA", "Other income", "…"], "periods": ["FY24"], "measure": "…",
     "acceptable_tables": [{"document": "…", "table": {"title": "…", "page": 22}, "series": ["…"]}],
     "additional_tables": [{"document": "…", "table": {"title": "…", "page": 20}, "periods": ["…"]}]},
   "tags": ["bridge", "…"], "notes": "why this is the answer, and the traps"}]}
```

- `expected` is the full label. `table` is the best table: its heading as printed, plus `page` (PDF), `slide` (PPTX) or `section` (DOCX). `acceptable_tables` are other tables that answer the question equally well; `additional_tables` are tables that must be combined with the primary one for a complete answer (the primary alone is a partial answer). `series` are the row or column labels the visual must show (the identity of what is plotted; for a grouped bar of segments by year, the segments), `periods` the period columns or rows involved, `measure` a plain-words note on which column is meant.
- `kind: "none"`: no table holds what is asked; `document`, `table` are `null`, `series` and `periods` are empty. The right behaviour is no visual, not the nearest chart.
- `kinds`, `datasets`, `series` (top level) and `query_en` are the fields the existing eval loop (`test_canvas_planner_eval.py`) reads, so v2 can be run through it with `SETS["holdout_v2"] = "canvas_planner_holdout_v2.json"`: `kinds` is `expected.kind` plus `acceptable_kinds`; `datasets` is `[index into documents, case-insensitive substring of the table's title]` for the primary, acceptable and additional tables; `series` is the union of the series of all of them, matched with "any". The title substrings come from the documents' headings, not from a Docling parse, so a mismatch there is a labelling issue to check before a planner fault. Two things the loop does not do yet: `documents` is written as a list (the loop's `names[d]`; a different shape in v1 would need a one-line change), and a `none` case needs a branch (correct = no spec; today the loop forces planning and treats a missing spec as a miss).

Suggested scoring: kind right when the plan's kind is in `kinds`; table right when every dataset it plans is in `datasets` (a plan that uses only one of two tables listed in `additional_tables` counts as a partial answer); series right when the plotted labels contain any (lenient, as the loop does) or all (strict) of `expected.series`; for `none`, right when no visual is planned. Report all three together and by language, and the four `none` cases apart.

## How the labels were checked

After writing, every label was checked against the generated documents: for each question's expected table (and each acceptable and additional one), the heading, every `series` label and every `periods` value must appear in the text of the stated page (pypdf), slide (python-pptx) or section (python-docx). 190 labels over 32 questions, all found; the check was also run on a deliberately corrupted copy (a wrong page, a made-up row, a wrong section) to confirm it fails. The `none` questions were checked by reading the text of every page and slide for a table holding the asked-for data.

To rebuild the documents: `uv run scripts/eval/build_corpus.py` (deterministic, no models). If the corpus layout changes, the page and slide numbers here must be re-checked.

## Limits

- 36 questions: one question is almost 3 points; read per-language and per-kind numbers as indications.
- The documents are synthetic and cleanly laid out; the tables are small (two year columns, a handful of rows). The corpus holds only FY23 and FY24 (and quarters), so there is no five-year series to ask about.
- The expected kind is the most natural one, not the only one; `acceptable_kinds` lists the others a reviewer would accept. Where a single table cannot answer the question (the Valmora bridge, FY24 against FY23 quarters), the labels say so rather than pick one arbitrarily.
- Written by one author (a model) in one sitting; the Hindi and Hinglish phrasing is natural to the author but is not a native-speaker survey.
