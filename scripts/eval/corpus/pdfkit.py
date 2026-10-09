"""PDF generation with exact page control (reportlab).

Every page is laid out as its own ``KeepInFrame(mode="error")``: content that doesn't fit its page raises instead of
spilling onto the next one, so the page a fact is planted on *is* the page it is rendered on (and the build still
re-reads the finished file to prove it). Multi-page tables are separate tables on consecutive pages with the header
row repeated, as in a printed annual report.

Fonts: DejaVu Sans, taken from matplotlib's bundled copy (it has the rupee sign U+20B9, which the PDF base-14
fonts and macOS's Arial lack), so the build doesn't depend on system fonts.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepInFrame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .registry import FactSpec, Registry, plain

MARGIN = 18 * mm
FRAME_W = A4[0] - 2 * MARGIN
FRAME_H = A4[1] - 2 * MARGIN - 8 * mm  # room for the running header and footer

_FONT = "DV"
_fonts_ready = False
INK = colors.HexColor("#1b2733")
ACCENT = colors.HexColor("#0b4f6c")
RULE = colors.HexColor("#9aa7b2")
BAND = colors.HexColor("#e6edf2")


def register_fonts() -> None:
    global _fonts_ready
    if _fonts_ready:
        return
    import matplotlib

    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for suffix, file in (
        ("", "DejaVuSans.ttf"),
        ("-B", "DejaVuSans-Bold.ttf"),
        ("-I", "DejaVuSans-Oblique.ttf"),
        ("-BI", "DejaVuSans-BoldOblique.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(_FONT + suffix, str(folder / file)))
    pdfmetrics.registerFontFamily(_FONT, normal=_FONT, bold=_FONT + "-B", italic=_FONT + "-I", boldItalic=_FONT + "-BI")
    _fonts_ready = True


def styles(scale: float = 1.0) -> dict[str, ParagraphStyle]:
    register_fonts()

    def s(name: str, **kw) -> ParagraphStyle:
        base = dict(fontName=_FONT, textColor=INK, fontSize=9.2 * scale, leading=13.2 * scale)
        base.update(kw)
        return ParagraphStyle(name, **base)

    return {
        "title": s(
            "title", fontName=_FONT + "-B", fontSize=20 * scale, leading=25 * scale, textColor=ACCENT, spaceAfter=8
        ),
        "h1": s(
            "h1",
            fontName=_FONT + "-B",
            fontSize=14 * scale,
            leading=18 * scale,
            textColor=ACCENT,
            spaceBefore=2,
            spaceAfter=6,
        ),
        "h2": s("h2", fontName=_FONT + "-B", fontSize=10.6 * scale, leading=14 * scale, spaceBefore=6, spaceAfter=3),
        "body": s("body", spaceAfter=5),
        "bullet": s("bullet", leftIndent=12, bulletIndent=2, spaceAfter=2),
        "note": s("note", fontSize=7.4 * scale, leading=9.6 * scale, textColor=colors.HexColor("#46525d")),
        "cell": s("cell", fontSize=8.2 * scale, leading=10.4 * scale),
        "cellr": s("cellr", fontSize=8.2 * scale, leading=10.4 * scale, alignment=2),
        "cellb": s("cellb", fontName=_FONT + "-B", fontSize=8.2 * scale, leading=10.4 * scale),
        "cellbr": s("cellbr", fontName=_FONT + "-B", fontSize=8.2 * scale, leading=10.4 * scale, alignment=2),
        "cellh": s("cellh", fontName=_FONT + "-B", fontSize=8.2 * scale, leading=10.4 * scale, textColor=colors.white),
        "cellhr": s(
            "cellhr",
            fontName=_FONT + "-B",
            fontSize=8.2 * scale,
            leading=10.4 * scale,
            textColor=colors.white,
            alignment=2,
        ),
        "cover": s("cover", fontName=_FONT + "-B", fontSize=30 * scale, leading=36 * scale, textColor=ACCENT),
        "coversub": s("coversub", fontSize=14 * scale, leading=20 * scale),
    }


_AMP = re.compile(r"&(?!amp;|lt;|gt;|#\d+;)")


def esc(text: str) -> str:
    """Escape a bare ``&`` for reportlab's paragraph markup (existing entities and tags are left alone)."""
    return _AMP.sub("&amp;", text)


class PdfPage:
    def __init__(self, doc: PdfDoc, number: int) -> None:
        self.doc = doc
        self.number = number
        self.flow: list = []
        self.st = doc.st

    # ---- text
    def _plant(self, text: str, facts: tuple[FactSpec, ...]) -> None:
        for spec in facts:
            self.doc.registry.plant(spec, document=self.doc.name, text=plain(text), page=self.number)

    def title(self, text: str) -> None:
        self.flow.append(Paragraph(esc(text), self.st["h1"]))

    def styled(self, text: str, style: str, *facts: FactSpec) -> None:
        """A paragraph in a named style (cover lines, centred text, ...)."""
        self.flow.append(Paragraph(esc(text), self.st[style]))
        self._plant(text, facts)

    def h2(self, text: str) -> None:
        self.flow.append(Paragraph(esc(text), self.st["h2"]))

    def para(self, text: str, *facts: FactSpec) -> None:
        self.flow.append(Paragraph(esc(text), self.st["body"]))
        self._plant(text, facts)

    def bullets(self, items: list[str | tuple[str, tuple[FactSpec, ...]]]) -> None:
        for item in items:
            text, facts = (item, ()) if isinstance(item, str) else item
            self.flow.append(Paragraph(esc(text), self.st["bullet"], bulletText="•"))
            self._plant(text, facts)

    def note(self, text: str, *facts: FactSpec) -> None:
        """A footnote: small print under a rule."""
        self.flow.append(Spacer(1, 4))
        self.flow.append(HRFlowable(width="35%", thickness=0.5, color=RULE, hAlign="LEFT", spaceAfter=3))
        self.flow.append(Paragraph(esc(text), self.st["note"]))
        self._plant(text, facts)

    def space(self, h: float = 6) -> None:
        self.flow.append(Spacer(1, h))

    def rule(self) -> None:
        self.flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=3, spaceAfter=5))

    def raw(self, flowable) -> None:
        self.flow.append(flowable)

    # ---- tables
    def table(
        self,
        rows: list[list[str]],
        widths: list[float],
        *,
        header_rows: int = 1,
        facts: dict[int, tuple[FactSpec, ...] | FactSpec] | None = None,
        bold_rows: tuple[int, ...] = (),
        numeric_from: int = 1,
        caption: str | None = None,
        shade_rows: tuple[int, ...] = (),
    ) -> None:
        """A grid table. ``widths`` are fractions of the frame width. ``facts`` maps a row index (0 = first row,
        header included) to facts whose evidence must be in that row's cells."""
        st = self.st
        if caption:
            self.flow.append(Paragraph(esc(caption), st["h2"]))
        data = []
        for r, row in enumerate(rows):
            line = []
            for c, cell in enumerate(row):
                right = c >= numeric_from
                if r < header_rows:
                    style = st["cellhr" if right else "cellh"]
                elif r in bold_rows:
                    style = st["cellbr" if right else "cellb"]
                else:
                    style = st["cellr" if right else "cell"]
                line.append(Paragraph(esc(cell), style))
            data.append(line)
        tbl = Table(data, colWidths=[w * FRAME_W for w in widths], repeatRows=header_rows)
        style = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#6b7782")),
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        for r in bold_rows + shade_rows:
            style.append(("BACKGROUND", (0, r), (-1, r), BAND))
        tbl.setStyle(TableStyle(style))
        self.flow.append(tbl)
        self.flow.append(Spacer(1, 6))
        for r, specs in (facts or {}).items():
            specs = (specs,) if isinstance(specs, FactSpec) else specs
            row_text = " ".join(plain(c) for c in rows[r])
            for spec in specs:
                self.doc.registry.plant(spec, document=self.doc.name, text=row_text, page=self.number)


class PdfDoc:
    def __init__(
        self,
        registry: Registry,
        name: str,
        *,
        title: str,
        header: str,
        scale: float = 1.0,
        footer_page_numbers: bool = True,
    ) -> None:
        self.registry = registry
        self.name = name
        self.title = title
        self.header = header
        self.st = styles(scale)
        self.pages: list[PdfPage] = []
        self.footer_page_numbers = footer_page_numbers

    def page(self) -> PdfPage:
        pg = PdfPage(self, len(self.pages) + 1)
        self.pages.append(pg)
        return pg

    def build(self, path: Path) -> None:
        register_fonts()
        doc = BaseDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN + 4 * mm,
            bottomMargin=MARGIN + 4 * mm,
            title=self.title,
            author="Eval corpus generator (fictional company)",
            invariant=1,
        )
        frame = Frame(
            MARGIN, MARGIN + 4 * mm, FRAME_W, FRAME_H, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
        )

        def decorate(canvas, _doc) -> None:
            canvas.saveState()
            canvas.setFont(_FONT, 7.4)
            canvas.setFillColor(colors.HexColor("#5b6772"))
            canvas.drawString(MARGIN, A4[1] - MARGIN + 2 * mm, self.header)
            canvas.setStrokeColor(RULE)
            canvas.line(MARGIN, A4[1] - MARGIN, A4[0] - MARGIN, A4[1] - MARGIN)
            if self.footer_page_numbers:
                canvas.drawRightString(A4[0] - MARGIN, MARGIN - 2 * mm, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=decorate)])
        story: list = []
        for i, pg in enumerate(self.pages):
            story.append(KeepInFrame(FRAME_W, FRAME_H - 2, pg.flow, mode="error", name=f"page {pg.number}"))
            if i < len(self.pages) - 1:
                story.append(PageBreak())
        try:
            doc.build(story)
        except Exception as e:  # LayoutError names the page that overflowed
            raise RuntimeError(f"{self.name}: a page does not fit its frame ({type(e).__name__}: {e})") from e
