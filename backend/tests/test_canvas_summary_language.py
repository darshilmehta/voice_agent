"""A visual's text summary is written in the visual's language only (services/canvas/builder.py, overview.py's
``label_name``). Found when the overview of an English project held a Hindi document: its donut read "सिलाई एवं परिधान
निर्माण is the largest part of सीटें…", an English sentence around Hindi labels.

The frame is the visual's language; a data label of the other language is named in it (the document's own English
name when the label is printed in both, else the word list) with the printed label after it, or, when it can't be
named, quoted as printed. Every number stays one the visual's cells and calculations give."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from app.domain.canvas import Visual
from app.services.canvas.builder import build_visual, check_grounding
from app.services.canvas.overview import label_name
from app.services.canvas.spec import SpecError, VisualSpec, resolve

from .canvas_helpers import DISTRICTS_HI, typed
from .test_canvas_builder import all_specs, build
from .test_canvas_overview import SEATS_HI, panels, project_datasets

NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)
DEVANAGARI = re.compile("[ऀ-ॿ]")
LATIN = re.compile("[A-Za-z]")
_CODE_WORD = re.compile(r"[A-Z]+\d+|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec")

# The same course table in a document that prints English names beside the Hindi ones.
SEATS_BILINGUAL = [
    ["पाठ्यक्रम (Course)", "अवधि (सप्ताह)", "सीटें"],
    ["सोलर पैनल तकनीशियन (Solar panel technician)", "8", "120"],
    ["घरेलू इलेक्ट्रिशियन (Domestic electrician)", "10", "90"],
    ["सिलाई एवं परिधान निर्माण (Sewing and garment making)", "12", "150"],
    ["कुल", "30", "360"],
]
# A measure the word list doesn't have.
APPLICANTS_HI = [
    ["पाठ्यक्रम", "उम्मीदवार"],
    ["सोलर पैनल तकनीशियन", "120"],
    ["घरेलू इलेक्ट्रिशियन", "90"],
    ["सिलाई एवं परिधान निर्माण", "60"],
    ["कुल", "270"],
]


def donut(rows, language: str, *, series: str) -> Visual:
    ds = typed(rows, dataset_id="ds_t", document_id="doc_t", heading=("7. पाठ्यक्रम और सीटें",), page=None)
    spec = VisualSpec(kind="donut", datasets=["ds_t"], series=[f"ds_t:{series}"], language=language)
    return build_visual(resolve(spec, {"ds_t": ds}), visual_id="v", project_id="p", chat_id=None, filenames={}, now=NOW)


def outside_names(summary: str) -> str:
    """The summary without the data labels it mentions: quoted ones and the printed label after a translated name."""
    text = re.sub("“[^”]*”", "", summary)
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    return re.sub(
        r"\s*\([^()]*\)", lambda m: "" if DEVANAGARI.search(m.group(0)) or LATIN.search(m.group(0)) else m[0], text
    )


# ------------------------------------------------------------------ English summaries of Hindi data


def test_an_english_summary_of_hindi_data_has_an_english_frame_and_translates_what_it_can():
    v = donut(SEATS_HI, "en", series="c2")
    # "सीटें" is "seats" (the word list); the course has no English name in the document: quoted as printed
    assert v.summary == "The largest share of seats is “सोलर पैनल तकनीशियन”: 120 (44.4%)."
    assert v.language == "en" and not DEVANAGARI.search(outside_names(v.summary))
    assert check_grounding(v) == []


def test_a_name_the_document_prints_in_english_is_used_with_the_printed_label_after_it():
    v = donut(SEATS_BILINGUAL, "en", series="c2")
    assert v.summary == "The largest share of seats is Sewing and garment making (सिलाई एवं परिधान निर्माण): 150 (41.7%)."
    assert check_grounding(v) == []


def test_a_measure_that_cannot_be_named_in_english_is_left_to_the_title_and_the_legend():
    v = donut(APPLICANTS_HI, "en", series="c1")
    assert v.summary == "The largest part is “सोलर पैनल तकनीशियन”: 120 (44.4%)."
    assert [s.label for s in v.series] == ["उम्मीदवार"]  # the data's labels stay as printed
    assert check_grounding(v) == []


def test_a_unit_that_is_a_hindi_word_is_said_in_english():
    ds = typed(SEATS_HI, dataset_id="ds_t", document_id="doc_t", heading=("7. पाठ्यक्रम और सीटें",), page=None)
    spec = VisualSpec(kind="bar", datasets=["ds_t"], series=["ds_t:c1"], language="en")
    v = build_visual(resolve(spec, {"ds_t": ds}), visual_id="v", project_id="p", chat_id=None, filenames={}, now=NOW)
    top, low = "“सिलाई एवं परिधान निर्माण”", "“सोलर पैनल तकनीशियन”"
    assert v.summary == f"Duration: highest {top} (12 weeks), lowest {low} (8 weeks)."
    assert check_grounding(v) == []


def test_the_districts_table_reads_in_english_with_every_kind_of_summary():
    ds = typed(DISTRICTS_HI, dataset_id="ds_d", document_id="doc_t", heading=("जिलेवार बजट",), page=None)
    built = []
    for kind in ("bar", "kpi", "comparison", "table", "donut"):
        spec = VisualSpec(kind=kind, datasets=["ds_d"], series=["ds_d:c1", "ds_d:c2"], language="en")
        try:
            r = resolve(spec, {"ds_d": ds})
            v = build_visual(r, visual_id="v", project_id="p", chat_id=None, filenames={}, now=NOW)
        except SpecError:  # a kind this table doesn't allow
            continue
        built.append(v.kind)
        assert not DEVANAGARI.search(outside_names(v.summary)), (kind, v.summary)
    assert {"bar", "kpi", "table"} <= set(built)


def test_the_overview_of_an_english_project_reads_its_hindi_donut_in_english():
    donut_panel = dict(panels(project_datasets(), 6))["doc_hi"]
    assert donut_panel.language == "en"
    assert donut_panel.summary == "The largest share of seats is “सोलर पैनल तकनीशियन”: 120 (44.4%)."


# ------------------------------------------------------------------ Hindi summaries


def test_a_hindi_summary_of_hindi_data_is_hindi_throughout():
    v = donut(SEATS_HI, "hi", series="c2")
    assert v.summary == "सीटें में सबसे बड़ा हिस्सा सोलर पैनल तकनीशियन का है: 120 (44.4%)।"
    assert not LATIN.search(v.summary)
    assert check_grounding(v) == []


def test_the_overview_of_a_hindi_project_is_hindi():
    scheme = [d for d in project_datasets() if d.document_id == "doc_hi"]
    (_, donut_panel), *_ = panels(scheme, 3, language="hi")
    assert donut_panel.summary == "सीटें में सबसे बड़ा हिस्सा सोलर पैनल तकनीशियन का है: 120 (44.4%)।"
    assert not LATIN.search(donut_panel.summary)


def test_a_hindi_summary_of_english_data_has_a_hindi_frame_and_quotes_the_labels_it_cannot_translate():
    v = build(kind="donut", datasets=["ds_segments"], series=["ds_segments:revenue_fy24"], language="hi")
    # "Revenue" is on the word list, FY24 is a period; the segment's name stays as printed, quoted
    assert v.summary == "राजस्व FY24 में सबसे बड़ा हिस्सा “Specialty Chemicals” का है: ₹3,568 करोड़ (48.4%)।"
    trend = build(
        kind="line", datasets=["ds_highlights"], series=["ds_highlights:revenue_from_operations"], language="hi"
    )
    assert trend.summary == "“Revenue from operations” FY23 में ₹6,482 करोड़ से FY24 में ₹7,365 करोड़ रहा (+13.6%)।"
    assert not LATIN.search(re.sub("“[^”]*”|FY\\d+", "", trend.summary))


def test_percentage_points_are_said_in_hindi():
    v = build(kind="comparison", datasets=["ds_highlights"], series=["ds_highlights:ebitda_margin"], language="hi")
    assert "+1.2 प्रतिशत अंक" in v.summary and " pp" not in v.summary


# ------------------------------------------------------------------ English data in an English summary is as it was


def test_english_labels_in_an_english_summary_are_untouched():
    v = build(kind="donut", datasets=["ds_segments"], series=["ds_segments:revenue_fy24"], language="en")
    assert v.summary == "The largest share of Revenue FY24 is Specialty Chemicals: ₹3,568 crore (48.4%)."
    for label in ("Revenue FY24", "Specialty Chemicals", "EBITDA", "Q3 FY24", "31 Mar 2024"):
        assert label_name(label, "en") == (label, None, "own")


# ------------------------------------------------------------------ naming a label


def test_label_name_in_english():
    assert label_name("सीटें", "en") == ("seats", "सीटें", "translated")
    assert label_name("आवंटित बजट", "en") == ("allocated budget", "आवंटित बजट", "translated")
    assert label_name("सीटें (Seats)", "en") == ("Seats", "सीटें", "translated")
    assert label_name("Sewing and garment making / सिलाई एवं परिधान निर्माण", "en") == (
        "Sewing and garment making",
        "सिलाई एवं परिधान निर्माण",
        "translated",
    )
    assert label_name("सिलाई एवं परिधान निर्माण", "en") == (
        "“सिलाई एवं परिधान निर्माण”",
        None,
        "quoted",
    )
    assert label_name("सीटें अज्ञात", "en").kind == "quoted"  # one word unknown: the label stays whole
    assert label_name("आवंटित बजट (₹ लाख)", "en").kind == "quoted"  # (a unit in parentheses is not an English name)
    assert label_name("लक्ष्य (kW)", "en") == ("target kW", "लक्ष्य (kW)", "translated")  # (kW is no English name)


def test_label_name_in_hindi():
    assert label_name("Revenue FY24", "hi") == ("राजस्व FY24", "Revenue FY24", "translated")
    assert label_name("राजस्व (Revenue)", "hi") == ("राजस्व", "Revenue", "translated")
    assert label_name("Specialty Chemicals", "hi") == ("“Specialty Chemicals”", None, "quoted")
    assert label_name("Full year FY24", "hi") == ("पूर्ण वर्ष FY24", "Full year FY24", "translated")
    for code in ("FY24", "Q3 FY24", "EBITDA", "31 Mar 2024", "2024"):
        assert label_name(code, "hi") == (code, None, "own")
    assert label_name("सीटें", "hi") == ("सीटें", None, "own")


# ------------------------------------------------------------------ every visual the builder can make


def test_no_summary_mixes_languages_and_every_one_is_grounded():
    """Every spec the test datasets allow, in both languages (Hindi tables in English visuals and the other way)."""
    checked = 0
    for spec in all_specs():
        try:
            v = build(**spec)
        except Exception:
            continue
        checked += 1
        assert check_grounding(v) == [], v.summary
        if v.language == "en":
            assert not DEVANAGARI.search(outside_names(v.summary)), v.summary
        else:  # Latin letters only as codes: FY24, EBITDA, CAGR, 31 Mar 2024, and "x" (a ratio)
            words = re.findall(r"[A-Za-z][A-Za-z0-9]*", outside_names(v.summary))
            assert all(w == "x" or w.isupper() or _CODE_WORD.fullmatch(w) for w in words), v.summary
    assert checked > 1000
