# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "reportlab>=4.2",
#   "python-docx>=1.1",
#   "python-pptx>=1.0",
#   "pypdf>=5.0",
#   "pypdfium2>=4.30",
#   "pillow>=10.4",
#   "numpy>=1.26",
#   "pydantic>=2.10",
#   "matplotlib>=3.9",  # only for its bundled DejaVu Sans fonts (the rupee sign); see corpus/pdfkit.py
# ]
# ///
# ruff: noqa: E501  (document text and evidence strings are data)
"""Build the retrieval eval corpus: fictional documents with known facts, and the manifest of those facts.

    uv run scripts/eval/build_corpus.py                      # build into data/eval/docs/ and verify
    uv run scripts/eval/build_corpus.py --update-manifest    # also rewrite evals/retrieval/manifest.json
    uv run scripts/eval/build_corpus.py --out DIR            # another output folder

Everything is deterministic and uses no ML. Documents (never committed: data/ is git-ignored):

    valmora_annual_report_fy24.pdf   29-page annual report: multi-page tables, footnotes, glossary, identifiers
    zephyra_investor_deck_q4fy24.pptx 15 slides from a second company with overlapping metric names
    valmora_travel_expense_policy.docx 9-page policy with numbered clauses, limits, exceptions
    suryodaya_yojana_soochna.docx    Hindi government scheme notice with dates and tables
    valmora_group_health_policy_scan.pdf 3 image-only pages (OCR)

While laying out each page the generator "plants" facts; afterwards it reads the finished files back (pypdf,
python-pptx, python-docx) and checks every planted page against the rendered text. A planted page whose text lacks
the evidence, evidence found nowhere, or a wrong page count fails the build loudly. The verified manifest is
compared with the committed evals/retrieval/manifest.json (the question set refers to it): a difference fails the
build unless --update-manifest is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))  # app.evals: manifest and text matching (pure Python)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # corpus/: the document content

from corpus import annual_report, hindi_notice, investor_deck, scanned_policy, travel_policy  # noqa: E402
from corpus.registry import Registry  # noqa: E402
from corpus.render import render_docx, render_pdf, render_pptx  # noqa: E402

from app.evals.manifest import (  # noqa: E402
    Manifest,
    ManifestError,
    RenderedDocument,
    finalize_manifest,
    load_manifest,
)

DEFAULT_OUT = ROOT / "data/eval/docs"
COMMITTED_MANIFEST = ROOT / "evals/retrieval/manifest.json"


def build_all(out: Path) -> tuple[Manifest, dict[str, RenderedDocument]]:
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    entries = [
        annual_report.build(reg, out),
        investor_deck.build(reg, out),
        travel_policy.build(reg, out),
        hindi_notice.build(reg, out),
        scanned_policy.build(reg, out),
    ]
    rendered: dict[str, RenderedDocument] = {}
    for entry in entries:
        path = out / entry.name
        if entry.format == "pdf":
            # Scanned PDFs have no text layer: their facts are verified on the vector source they were rasterised from.
            rendered[entry.name] = scanned_policy.render_source(path) if entry.scanned else render_pdf(path)
        elif entry.format == "pptx":
            rendered[entry.name] = render_pptx(path)
        else:
            rendered[entry.name] = render_docx(path)
    return finalize_manifest(reg.manifest(), rendered), rendered


def summary(manifest: Manifest) -> str:
    lines = ["", f"{'document':44} {'pages':>5} {'facts':>6}"]
    for d in manifest.documents:
        n = len(manifest.facts_of(d.name))
        lines.append(f"{d.name:44} {d.pages or '-'!s:>5} {n:>6}")
    pages = sum(d.pages or 0 for d in manifest.documents)
    lines.append(f"{'total':44} {pages:>5} {len(manifest.facts):>6}")
    return "\n".join(lines)


def extra_pages_report(manifest: Manifest) -> list[str]:
    """Facts stated on pages beyond the ones the generator planted them on (legitimate repeats, listed for review)."""
    out = []
    for f in manifest.facts:
        extra = [p for p in f.pages if p not in f.planted_pages]
        if extra:
            out.append(f"  {f.id}: planted {f.planted_pages}, also stated on {extra}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT.relative_to(ROOT)})"
    )
    ap.add_argument("--manifest", type=Path, default=COMMITTED_MANIFEST, help="the committed manifest to compare with")
    ap.add_argument("--update-manifest", action="store_true", help="rewrite the committed manifest from this build")
    ap.add_argument("--show-repeats", action="store_true", help="list facts stated on more pages than planted")
    args = ap.parse_args()

    try:
        manifest, _ = build_all(args.out)
    except ManifestError as e:
        print(f"\nVERIFICATION FAILED: the planted facts do not match the rendered files.\n{e}", file=sys.stderr)
        return 1
    print(
        f"built {len(manifest.documents)} documents in {args.out}; every planted page verified against the rendered text"
    )
    print(summary(manifest))
    repeats = extra_pages_report(manifest)
    print(f"\n{len(repeats)} facts are also stated on pages beyond the planted ones (all such pages count as correct)")
    if args.show_repeats:
        print("\n".join(repeats))

    text = manifest.dump_json()
    (args.out / "manifest.json").write_text(text, encoding="utf-8")
    if args.update_manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text, encoding="utf-8")
        print(f"wrote {args.manifest.relative_to(ROOT)}")
        return 0
    if not args.manifest.exists():
        print(f"no committed manifest at {args.manifest}; run with --update-manifest", file=sys.stderr)
        return 1
    committed = load_manifest(args.manifest)
    if committed != manifest:
        old, new = {f.id: f for f in committed.facts}, {f.id: f for f in manifest.facts}
        diffs = [f"fact {i} added" for i in sorted(new.keys() - old.keys())]
        diffs += [f"fact {i} removed" for i in sorted(old.keys() - new.keys())]
        diffs += [
            f"fact {i} changed: {old[i].pages} -> {new[i].pages}"
            for i in sorted(old.keys() & new.keys())
            if old[i] != new[i]
        ]
        if committed.documents != manifest.documents:
            diffs.append("document list changed")
        print(
            "\nthe build differs from the committed manifest (questions may now point at wrong pages):\n  - "
            + "\n  - ".join(diffs[:40]),
            file=sys.stderr,
        )
        print("re-run with --update-manifest after reviewing, then regenerate the questions", file=sys.stderr)
        return 1
    print("matches the committed manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
