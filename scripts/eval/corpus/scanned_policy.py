# ruff: noqa: E501  (document text and evidence strings are data)
"""Document 5: Valmora's group health insurance policy schedule as a scanned PDF (3 image-only pages).

A vector PDF is laid out and its facts planted and verified like any other PDF; each page is then rasterised and
degraded a little (slight skew, noise, softer contrast, fixed seed) and saved as an image-only PDF, so only OCR
(RapidOCR, docs/DESIGN.md §9.4) can read it. Facts are marked ``ocr``: the harness matches their evidence loosely
(ignoring spaces and punctuation) because OCR output can drop slashes and commas.
"""

from __future__ import annotations

from pathlib import Path

from app.evals.manifest import DocumentEntry, RenderedDocument

from .pdfkit import PdfDoc
from .registry import F, FactSpec, Registry
from .render import render_pdf

NAME = "valmora_group_health_policy_scan.pdf"
TITLE = "Group Health Insurance Policy Schedule (scanned)"
POLICY_NO = "GHI/2024/00418377"
SEED = 20240322
_SOURCE: RenderedDocument | None = None


def _ocr(spec: FactSpec) -> FactSpec:
    return F(spec.id, list(spec.evidence), spec.statement, spec.kind, ocr=True)


def build(reg: Registry, out_dir: Path) -> DocumentEntry:
    global _SOURCE
    entry = DocumentEntry(
        name=NAME, title=TITLE, format="pdf", language="en", pages=3, scanned=True, company="Valmora Industries Limited"
    )
    reg.add_document(entry)
    doc = PdfDoc(
        reg,
        NAME,
        title=TITLE,
        header="Aarav General Insurance Company Limited  |  Group Health Insurance Policy Schedule",
        scale=1.3,
    )

    pg = doc.page()
    pg.title("Group Health Insurance Policy: Schedule")
    pg.para("This schedule forms part of the policy and must be read together with the policy wording.")
    pg.table(
        [
            ["Item", "Details"],
            ["Policy number", POLICY_NO],
            ["Policyholder", "Valmora Industries Limited"],
            ["Insurer", "Aarav General Insurance Company Limited"],
            ["Policy period", "1 April 2024 to 31 March 2025"],
            ["Sum insured per family (floater)", "₹ 5,00,000"],
            ["Employees covered", "9,842"],
            ["Dependants covered", "28,415"],
            ["Total annual premium (including GST)", "₹ 4,86,12,500"],
            ["Date of issue", "22 March 2024"],
            ["Third-party administrator", "Medisure Health TPA Services"],
        ],
        [0.46, 0.54],
        numeric_from=9,
        facts={
            1: _ocr(
                F(
                    "ghi.policy_no",
                    [POLICY_NO],
                    f"The group health insurance policy number is {POLICY_NO}",
                    "identifier",
                )
            ),
            4: _ocr(
                F(
                    "ghi.period",
                    ["1 April 2024 to 31 March 2025"],
                    "The group health policy period is 1 April 2024 to 31 March 2025",
                    "date",
                )
            ),
            5: _ocr(
                F(
                    "ghi.sum_insured",
                    ["5,00,000", "floater"],
                    "The sum insured is ₹ 5,00,000 per family on a floater basis",
                    "number",
                )
            ),
            6: _ocr(
                F(
                    "ghi.employees",
                    ["Employees covered", "9,842"],
                    "9,842 employees are covered under the group health policy",
                    "number",
                )
            ),
            7: _ocr(
                F(
                    "ghi.dependants",
                    ["Dependants covered", "28,415"],
                    "28,415 dependants are covered under the group health policy",
                    "number",
                )
            ),
            8: _ocr(
                F("ghi.premium", ["4,86,12,500"], "The total annual premium including GST is ₹ 4,86,12,500", "number")
            ),
            9: _ocr(
                F(
                    "ghi.issue_date",
                    ["Date of issue", "22 March 2024"],
                    "The group health policy was issued on 22 March 2024",
                    "date",
                )
            ),
            10: _ocr(
                F(
                    "ghi.tpa",
                    ["Third-party administrator", "Medisure Health TPA Services"],
                    "The third-party administrator is Medisure Health TPA Services",
                    "text",
                )
            ),
        },
    )

    pg = doc.page()
    pg.title("Coverage and limits")
    pg.table(
        [
            ["Benefit", "Limit", "Waiting period"],
            ["Room rent (per day)", "₹ 5,000", "None"],
            ["ICU charges (per day)", "₹ 10,000", "None"],
            ["Maternity: normal delivery", "₹ 75,000", "Waived"],
            ["Maternity: caesarean section", "₹ 1,00,000", "Waived"],
            ["Ambulance (per hospitalisation)", "₹ 3,000", "None"],
            ["Pre-existing diseases", "Covered from day one", "None"],
            ["Pre- and post-hospitalisation", "30 days before and 60 days after", "None"],
            ["Co-payment", "10 per cent for dependent parents", "Not applicable"],
        ],
        [0.42, 0.38, 0.2],
        numeric_from=9,
        facts={
            1: _ocr(
                F("ghi.room_rent", ["Room rent", "5,000"], "Room rent is covered up to ₹ 5,000 per day", "table_cell")
            ),
            2: _ocr(
                F("ghi.icu", ["ICU charges", "10,000"], "ICU charges are covered up to ₹ 10,000 per day", "table_cell")
            ),
            3: _ocr(
                F(
                    "ghi.maternity_normal",
                    ["normal delivery", "75,000"],
                    "Normal delivery maternity cover is ₹ 75,000",
                    "table_cell",
                )
            ),
            4: _ocr(
                F(
                    "ghi.maternity_caesarean",
                    ["caesarean", "1,00,000"],
                    "Caesarean maternity cover is ₹ 1,00,000",
                    "table_cell",
                )
            ),
            6: _ocr(
                F(
                    "ghi.preexisting",
                    ["Pre-existing diseases", "Covered from day one"],
                    "Pre-existing diseases are covered from day one",
                    "table_cell",
                )
            ),
            7: _ocr(
                F(
                    "ghi.prepost",
                    ["Pre- and post-hospitalisation", "30 days before and 60 days after"],
                    "Pre- and post-hospitalisation expenses are covered for 30 days before and 60 days after",
                    "table_cell",
                )
            ),
            8: _ocr(
                F(
                    "ghi.copay",
                    ["Co-payment", "10 per cent for dependent parents"],
                    "A 10% co-payment applies to dependent parents",
                    "table_cell",
                )
            ),
        },
    )
    pg.para("Limits apply per policy year. Sub-limits are inclusive of the sum insured and are not additional to it.")

    pg = doc.page()
    pg.title("Claims procedure and exclusions")
    pg.h2("How to claim")
    pg.bullets(
        [
            (
                "For planned hospitalisation, inform the TPA at least 48 hours before admission to use the cashless facility.",
                (
                    _ocr(
                        F(
                            "ghi.cashless_planned",
                            ["at least 48 hours before admission"],
                            "For planned hospitalisation, inform the TPA at least 48 hours before admission for cashless treatment",
                            "number",
                        )
                    ),
                ),
            ),
            (
                "For emergency hospitalisation, inform the TPA within 24 hours of admission.",
                (
                    _ocr(
                        F(
                            "ghi.cashless_emergency",
                            ["within 24 hours of admission"],
                            "For emergency hospitalisation, inform the TPA within 24 hours of admission",
                            "number",
                        )
                    ),
                ),
            ),
            (
                "Reimbursement claims must be submitted within 30 days of discharge.",
                (
                    _ocr(
                        F(
                            "ghi.reimbursement",
                            ["within 30 days of discharge"],
                            "Reimbursement claims must be submitted within 30 days of discharge",
                            "number",
                        )
                    ),
                ),
            ),
            (
                "The 24x7 claims helpline is 1800 419 5530.",
                (
                    _ocr(
                        F("ghi.helpline", ["1800 419 5530"], "The 24x7 claims helpline is 1800 419 5530", "identifier")
                    ),
                ),
            ),
        ]
    )
    pg.h2("Exclusions")
    pg.bullets(
        [
            (
                "Cosmetic and aesthetic surgery, unless needed after an accident.",
                (
                    _ocr(
                        F(
                            "ghi.excl_cosmetic",
                            ["Cosmetic and aesthetic surgery"],
                            "Cosmetic and aesthetic surgery is excluded unless needed after an accident",
                            "text",
                        )
                    ),
                ),
            ),
            "Dental treatment, unless caused by an accident.",
            (
                "Infertility treatment and assisted reproduction.",
                (
                    _ocr(
                        F(
                            "ghi.excl_infertility",
                            ["Infertility treatment"],
                            "Infertility treatment and assisted reproduction are excluded",
                            "text",
                        )
                    ),
                ),
            ),
            "Injuries from adventure sports and hazardous activities.",
        ]
    )
    pg.space(16)
    pg.para(
        "Authorised signatory: Neha Kapoor, Aarav General Insurance Company Limited. Place: Mumbai.",
        _ocr(
            F(
                "ghi.signatory",
                ["Authorised signatory", "Neha Kapoor"],
                "The authorised signatory is Neha Kapoor of Aarav General Insurance Company Limited",
                "text",
            )
        ),
    )

    assert len(doc.pages) == 3
    source = out_dir / "_scan_source.pdf"
    doc.build(source)
    _SOURCE = render_pdf(source)  # facts are verified on this text layer
    _rasterise(source, out_dir / NAME)
    source.unlink()
    return entry


def _rasterise(source: Path, target: Path) -> None:
    import numpy as np
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(SEED)
    pdf = pdfium.PdfDocument(str(source))
    images = []
    for i, angle in enumerate((0.45, -0.3, 0.25)):
        page = pdf[i]
        img = page.render(scale=2.2).to_pil().convert("L")
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
        arr = np.asarray(img, dtype=np.float32)
        arr = 255 - (255 - arr) * 0.9  # softer ink
        arr += rng.normal(0, 7.0, arr.shape)  # sensor noise
        arr += np.linspace(-6, 6, arr.shape[0], dtype=np.float32)[:, None]  # uneven lighting
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
        images.append(img.convert("RGB"))
    images[0].save(str(target), "PDF", save_all=True, append_images=images[1:], resolution=158.4)


def render_source(path: Path) -> RenderedDocument:
    """What to verify the scanned facts against: the text layer of the vector source. Also checks that the final
    file really is image-only and has the planned pages."""
    from pypdf import PdfReader

    assert _SOURCE is not None, "scanned_policy.build must run first"
    reader = PdfReader(str(path))
    if len(reader.pages) != 3:
        raise AssertionError(f"{path.name}: expected 3 scanned pages, found {len(reader.pages)}")
    text = "".join((p.extract_text() or "") for p in reader.pages).strip()
    if text:
        raise AssertionError(f"{path.name} still has a text layer ({len(text)} characters): it would not need OCR")
    return _SOURCE
