"""Providers for every external capability. ``load_all()`` imports (and so registers) every implementation."""


def load_all() -> None:
    from . import ingestion, llm, retrieval, runtime, speech, storage, web_search  # noqa: F401
