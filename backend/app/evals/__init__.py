"""Retrieval evaluation (docs/DESIGN.md §10, phases 2 and 9): corpus manifest, question set, metrics and the harness.

Everything here that is not the harness itself (``retrieval_eval``) is pure Python, so it is unit-tested in CI
without models: ``textnorm`` (matching), ``manifest`` (the planted facts and their page verification),
``questions`` (the question-set schema), ``metrics`` (recall, MRR, abstention, threshold sweep) and ``report``
(Markdown summary). See ``evals/README.md``.
"""
