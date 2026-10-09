"""The live visual canvas (docs/DESIGN.md §12.1): the LLM picks, code builds; every number is a cell or a calculation.

parsing.py       numbers (Indian/Western grouping, negatives, footnotes, ₹/$), units ("(₹ in crore)"), periods
datasets.py      a parsed table → TypedDataset: header rows, label column, column types, units, totals, cell positions
chartability.py  time series, KPI, composition, categorical, timeline or not chartable, with a confidence
spec.py          VisualSpec (what the model fills) and resolve(), the validator against the real datasets
calculator.py    growth, CAGR, diff, ratio, share, sum: units, formula text, CellRef inputs
builder.py       resolved spec → Visual (contract v1), Citations, summary in the chat language; check_grounding()
overview.py      the project overview's specs (KPI tiles, a trend, a composition)
planner.py       visual_intent() heuristics and VisualPlanner (qwen3 JSON with per-request enums, bounded, timed out)
store.py         table_datasets, canvas_visuals, project_overviews rows
service.py       CanvasService: ingestion hook, backfill job, overview, chat canvas, prepare_visual() for turns

Imports here stay lazy: ``services.documents`` imports ``store`` for its cascade.
"""
