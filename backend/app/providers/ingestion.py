"""Document parsing providers (docs/DESIGN.md §3.1). Parsing arrives in phase 1."""

from __future__ import annotations

from .base import HealthStatus, Provider, ProviderHealth
from .registry import register

# Directories `docling-tools models download` creates under artifacts_path.
DOCLING_LAYOUT = "docling-project--docling-layout-heron"
DOCLING_TABLES = "docling-project--docling-models"
OCR_ARTIFACTS = {"rapidocr": "RapidOcr"}


class DocumentParser(Provider):
    capability = "ingestion"


@register
class DoclingParser(DocumentParser):
    name = "docling"

    async def health(self) -> ProviderHealth:
        cfg = self.config
        path = self.ctx.settings.path(cfg.artifacts_path)  # type: ignore[attr-defined]
        if not path.is_dir():
            return self._health(HealthStatus.DOWN, f"{path} missing (run scripts/setup/download_models.sh docling)")
        needed = [DOCLING_LAYOUT, DOCLING_TABLES]
        if cfg.ocr and cfg.ocr_engine in OCR_ARTIFACTS:  # type: ignore[attr-defined]
            needed.append(OCR_ARTIFACTS[cfg.ocr_engine])  # type: ignore[attr-defined]
        missing = [d for d in needed if not (path / d).is_dir()]
        if missing:
            return self._health(HealthStatus.DOWN, f"missing model folders: {', '.join(missing)}")
        ocr = f"OCR {cfg.ocr_engine}" if cfg.ocr else "OCR off"  # type: ignore[attr-defined]
        return self._health(HealthStatus.OK, f"layout + table models present, {ocr}")
