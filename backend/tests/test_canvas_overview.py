"""The project overview with several documents (services/canvas/overview.py, docs/DESIGN.md §12.1): panels grouped by
document, titles in the project's main language that say which document a panel is from, data labels as printed.
Found in the final real-model run: the overview mixed three companies and two languages (a Hindi donut titled "सीटें"
in an English project)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain.canvas import Visual
from app.services.canvas.builder import build_visual
from app.services.canvas.overview import (
    document_title,
    in_language,
    main_language,
    overview_specs,
    overview_title,
    script_of,
)
from app.services.canvas.planner import builds
from app.services.canvas.spec import resolve

from .canvas_helpers import DISTRICTS_HI, ZEPHYRA_GLANCE, ZEPHYRA_SEGMENTS, report_datasets, typed
from .test_canvas_api import REPORT, chat, drain, project, upload

SEATS_HI = [
    ["पाठ्यक्रम", "अवधि (सप्ताह)", "सीटें"],
    ["सोलर पैनल तकनीशियन", "8", "120"],
    ["घरेलू इलेक्ट्रिशियन", "10", "90"],
    ["सिलाई एवं परिधान निर्माण", "12", "60"],
    ["कुल", "30", "270"],
]
FILES = {
    "doc_1": "valmora_annual_report_fy24.pdf",
    "doc_z": "zephyra_investor_deck_q4fy24.pptx",
    "doc_hi": "suryodaya_yojana_soochna.docx",
}


ZEPHYRA = (
    "Zephyra deck\n\nQuarterly.\f| Quarter | Revenue (₹ crore) |\n| Q1 FY24 | 900 |\n| Q2 FY24 | 950 |\n"
    "| Q3 FY24 | 990 |\n| Q4 FY24 | 1,030 |"
).encode()


def project_datasets():
    """Valmora's report (KPI, quarters, segments, …), Zephyra's glance and segments and a Hindi scheme document
    (course seats, district budgets)."""
    valmora = list(report_datasets("doc_1").values())
    zephyra = [
        typed(ZEPHYRA_GLANCE, heading=("Zephyra at a glance: FY24",), page=3, document_id="doc_z", dataset_id="ds_z"),
        typed(ZEPHYRA_SEGMENTS, heading=("Segment overview",), page=5, document_id="doc_z", dataset_id="ds_zseg"),
    ]
    scheme = [
        typed(SEATS_HI, heading=("7. पाठ्यक्रम और सीटें",), page=None, document_id="doc_hi", dataset_id="ds_seats"),
        typed(
            DISTRICTS_HI, heading=("8. जिलेवार लक्ष्य एवं बजट",), page=None, document_id="doc_hi", dataset_id="ds_districts"
        ),
    ]
    return valmora + zephyra + scheme


def panels(datasets, max_panels: int, language: str = "en") -> list[tuple[str, Visual]]:
    pool = {d.id: d for d in datasets}
    specs = overview_specs(datasets, language=language, max_panels=max_panels, check=builds(pool, FILES))
    several = len({d.document_id for d in datasets if d.chartability.kind != "none"}) > 1
    out = []
    for spec in specs:
        visual = build_visual(
            resolve(spec, pool, filenames=FILES),
            visual_id="vis_o",
            project_id="p",
            chat_id=None,
            filenames=FILES,
            now=datetime(2026, 10, 9, tzinfo=UTC),
        )
        document = pool[spec.datasets[0]].document_id
        label = document_title(FILES[document]) if several else None
        out.append((document, visual.model_copy(update={"title": overview_title(visual, label, language)})))
    return out


# ------------------------------------------------------------------ by document


def test_a_small_overview_covers_each_document_in_turn_most_tables_first():
    got = panels(project_datasets(), 3)
    # Valmora's report has the most chartable tables; then Zephyra's deck and the scheme (two each: in the order met)
    assert [(doc, v.kind) for doc, v in got] == [("doc_1", "kpi"), ("doc_z", "kpi"), ("doc_hi", "donut")]


def test_a_larger_overview_keeps_a_documents_panels_together_in_the_order_kpi_trend_composition():
    got = panels(project_datasets(), 6)
    assert [doc for doc, _ in got] == ["doc_1", "doc_1", "doc_1", "doc_z", "doc_z", "doc_hi"]
    assert [v.kind for doc, v in got if doc == "doc_1"] == ["kpi", "line", "donut"]
    assert [v.kind for doc, v in got if doc == "doc_z"] == ["kpi", "donut"]


def test_a_panel_is_built_from_the_tables_of_one_document():
    pool = {d.id: d for d in project_datasets()}
    pool_specs = overview_specs(list(pool.values()), max_panels=6)
    for spec in pool_specs:
        assert len({pool[d].document_id for d in spec.datasets}) == 1


# ------------------------------------------------------------------ titles


def test_the_hindi_donut_in_an_english_project_is_titled_in_english_with_its_document():
    got = dict(panels(project_datasets(), 6))
    donut = got["doc_hi"]
    assert main_language(project_datasets(), "en") == "en"
    assert donut.language == "en"
    assert donut.title == "Suryodaya yojana soochna: seats by course"
    assert [s.label for s in donut.series] == ["सीटें"]  # the data's labels stay as printed
    assert next(r.x for r in donut.rows) == "सोलर पैनल तकनीशियन"


def test_titles_say_which_document_they_are_from_unless_the_title_says_whose_it_is():
    titles = [v.title for _, v in panels(project_datasets(), 6)]
    assert titles[0] == "Valmora annual report FY24: Highlights"
    assert titles[1] == "Valmora annual report FY24: Revenue · EBITDA, Q1 FY23–Q4 FY24"  # noqa: RUF001
    assert titles[2] == "Valmora annual report FY24: Revenue FY24 by segment"
    assert titles[3] == "Zephyra at a glance, FY24"  # (its own title names Zephyra)
    assert titles[4] == "Zephyra investor deck Q4FY24: Revenue FY24 by segment"
    assert titles[5] == "Suryodaya yojana soochna: seats by course"


def test_a_project_with_one_document_needs_no_document_name():
    got = panels(list(report_datasets().values()), 3)
    assert [v.title for _, v in got] == [
        "Highlights",
        "Revenue · EBITDA, Q1 FY23–Q4 FY24",  # noqa: RUF001
        "Revenue FY24 by segment",
    ]


def test_a_hindi_project_has_hindi_titles_and_labels_as_printed():
    scheme = [d for d in project_datasets() if d.document_id == "doc_hi"]
    assert main_language(scheme, "en") == "hi"
    got = panels(scheme, 3, language="hi")
    assert got[0][1].language == "hi" and got[0][1].title == "पाठ्यक्रम के अनुसार सीटें"
    assert [s.label for s in got[0][1].series] == ["सीटें"]


def test_the_projects_language_is_that_of_most_of_its_documents():
    everything = project_datasets()
    assert main_language(everything, "hi") == "en"  # two of three documents are in English
    scheme = [d for d in everything if d.document_id == "doc_hi"]
    valmora = [d for d in everything if d.document_id == "doc_1"]
    assert main_language([*scheme, *valmora], "hi") == "hi"  # split evenly: the app's default
    assert main_language([*scheme, *valmora], "en") == "en"
    assert main_language([], "hi") == "hi"


def test_words_are_translated_only_when_all_of_them_are_known():
    assert in_language("सीटें", "en") == "seats"
    assert in_language("आवंटित बजट", "en") == "allocated budget"
    assert in_language("लक्ष्य (युवा)", "en") == "target youth"
    assert in_language("न्यूनतम योग्यता", "en") == "minimum qualification"
    assert in_language("अज्ञात शब्द", "en") == "अज्ञात शब्द"  # unknown: as printed
    assert in_language("सीटें अज्ञात", "en") == "सीटें अज्ञात"  # one unknown word keeps the label whole
    assert in_language("Revenue FY24", "en") == "Revenue FY24"  # already English
    assert in_language("budget", "hi") == "बजट"
    assert script_of("Revenue FY24") == "en" and script_of("सीटें") == "hi" and script_of("2024") is None


@pytest.mark.parametrize(
    ("filename", "label"),
    [
        ("valmora_annual_report_fy24.pdf", "Valmora annual report FY24"),
        ("zephyra_investor_deck_q4fy24.pptx", "Zephyra investor deck Q4FY24"),
        ("suryodaya_yojana_soochna.docx", "Suryodaya yojana soochna"),
    ],
)
def test_a_document_is_named_by_its_file(filename, label):
    assert document_title(filename) == label


# ------------------------------------------------------------------ through the service


def test_the_overview_of_a_project_with_two_documents_names_each_panels_document(make_app):
    # the grouping and order with three panels (the shipped value is 4)
    with make_app(CANVAS__OVERVIEW_PANELS="3") as app:
        p = project(app)
        upload(app, p, "valmora_report.txt", REPORT)
        upload(app, p, "zephyra_deck.txt", ZEPHYRA)
        drain(app)
        chat(app, p)
        body = app.get(f"/api/projects/{p}/overview").json()
        assert body["status"] == "ready"
        # grouped by document, the one with the most tables first; each title says which document it is from
        assert [v["title"].split(": ", 1)[0] for v in body["panels"]] == [
            "Valmora report",
            "Valmora report",
            "Zephyra deck",
        ]
        assert [v["kind"] for v in body["panels"]] == ["kpi", "line", "line"]
        assert {v["language"] for v in body["panels"]} == {"en"}
        assert body["panels"][2]["title"] == "Zephyra deck: Revenue, Q1 FY24–Q4 FY24"  # noqa: RUF001
