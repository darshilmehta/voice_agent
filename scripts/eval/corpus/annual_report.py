# ruff: noqa: E501  (document text and evidence strings are data)
"""Document 1: Valmora Industries Limited, Annual Report 2023-24 (PDF, 29 pages).

Narrative sections, multi-page financial tables, footnotes, a glossary, identifiers, and near-duplicate facts that
differ by year (FY23 / FY24, on separate pages as well as in shared tables) or by segment (three segment pages
written from one template), so a wrong-year or wrong-segment retrieval is visible.
"""

from __future__ import annotations

from pathlib import Path

from app.evals.manifest import DocumentEntry

from .finance import Valmora as V
from .finance import growth, n0, n2, pct
from .pdfkit import PdfDoc
from .registry import F, Registry

NAME = "valmora_annual_report_fy24.pdf"
TITLE = "Valmora Industries Limited - Annual Report 2023-24"
PAGES = 29


def cr(x: float) -> str:
    return f"₹ {n0(x)} crore"


def build(reg: Registry, out_dir: Path) -> DocumentEntry:
    entry = DocumentEntry(name=NAME, title=TITLE, format="pdf", language="en", pages=PAGES, company=V.name)
    reg.add_document(entry)
    doc = PdfDoc(reg, NAME, title=TITLE, header="Valmora Industries Limited  |  Annual Report 2023-24")
    for make in (
        p01_cover,
        p02_corporate,
        p03_highlights,
        p04_chair,
        p05_ceo,
        p06_overview,
        p07_chemicals,
        p08_plastics,
        p09_digital,
        p10_operations,
        p11_sustainability,
        p12_people,
        p13_risk,
        p14_governance,
        p15_board,
        p16_outlook,
        p17_mda,
        p18_segments,
        p19_q_fy24,
        p20_q_fy23,
        p21_pl1,
        p22_pl2,
        p23_assets,
        p24_equity,
        p25_cashflow,
        p26_notes1,
        p27_notes2,
        p28_shareholders,
        p29_glossary,
    ):
        make(doc.page())
    assert len(doc.pages) == PAGES, len(doc.pages)
    doc.build(out_dir / NAME)
    return entry


# ------------------------------------------------------------------ 1-2: cover and corporate information


def p01_cover(pg) -> None:
    pg.space(110)
    pg.styled("VALMORA INDUSTRIES LIMITED", "cover")
    pg.space(10)
    pg.styled("Annual Report 2023-24", "coversub")
    pg.space(8)
    pg.styled("Chemistry that builds, plastics that last, software that scales", "coversub")
    pg.space(60)
    pg.rule()
    pg.styled(
        "Thirtieth Annual General Meeting",
        "body",
        F("vmr.agm_number", ["Thirtieth Annual General Meeting"], "This is the 30th AGM", "text"),
    )
    pg.styled(f"CIN: {V.cin}", "body", F("vmr.cin", [V.cin], f"Valmora's CIN is {V.cin}", "identifier"))
    pg.styled(
        "Registered office: 14 Sunrise Industrial Park, Vapi, Gujarat 396195",
        "body",
        F(
            "vmr.reg_office",
            ["14 Sunrise Industrial Park", "Vapi"],
            "Registered office: 14 Sunrise Industrial Park, Vapi, Gujarat 396195",
            "text",
        ),
    )
    pg.space(20)
    pg.styled("Specialty Chemicals  |  Engineered Plastics  |  Digital Services", "body")


def p02_corporate(pg) -> None:
    pg.title("Corporate information")
    rows = [
        ["Item", "Details"],
        ["Chairperson", "Dr. Meenakshi Raghavan (Independent Director)"],
        ["Managing Director and Chief Executive Officer", "Mr. Karthik Nambiar"],
        ["Chief Financial Officer", "Ms. Sunita Deshpande"],
        ["Company Secretary and Compliance Officer", "Mr. Rohan Kulkarni"],
        [
            "Statutory auditors",
            "Sharma Venkatesh and Associates, Chartered Accountants (Firm Registration No. 109473W)",
        ],
        ["Registrar and transfer agent", "Bridgepoint Registry Services Private Limited"],
        ["Principal bankers", "Unity Bank; Western Trust Bank"],
        ["Credit rating", "AA+ (Stable) for long-term bank facilities and non-convertible debentures"],
        [
            "Stock exchanges",
            f"BSE Limited (scrip code {V.bse_code}); National Stock Exchange of India Limited (symbol {V.nse_symbol})",
        ],
        ["ISIN", V.isin],
        ["Corporate Identity Number (CIN)", V.cin],
        ["Registered office", "14 Sunrise Industrial Park, Vapi, Gujarat 396195"],
        ["Website and investor contact", "www.valmora.example; investors@valmora.example"],
    ]
    pg.table(
        rows,
        [0.36, 0.64],
        numeric_from=9,
        facts={
            1: F(
                "vmr.chair", ["Chairperson", "Meenakshi Raghavan"], "Dr. Meenakshi Raghavan is the Chairperson", "text"
            ),
            2: F(
                "vmr.md_ceo",
                ["Managing Director and Chief Executive Officer", "Karthik Nambiar"],
                "Karthik Nambiar is MD and CEO",
                "text",
            ),
            3: F("vmr.cfo", ["Chief Financial Officer", "Sunita Deshpande"], "Sunita Deshpande is the CFO", "text"),
            4: F("vmr.cs", ["Company Secretary", "Rohan Kulkarni"], "Rohan Kulkarni is the Company Secretary", "text"),
            5: F(
                "vmr.auditor",
                ["Sharma Venkatesh and Associates", "109473W"],
                "Statutory auditors: Sharma Venkatesh and Associates, firm registration no. 109473W",
                "identifier",
            ),
            6: F(
                "vmr.registrar",
                ["Bridgepoint Registry Services Private Limited"],
                "The registrar and transfer agent is Bridgepoint Registry Services Private Limited",
                "text",
            ),
            8: F("vmr.rating", ["AA+", "Stable"], "Credit rating AA+ (Stable)", "text"),
            9: (
                F("vmr.bse_code", [f"scrip code {V.bse_code}"], f"BSE scrip code {V.bse_code}", "identifier"),
                F("vmr.nse_symbol", [f"symbol {V.nse_symbol}"], f"NSE symbol {V.nse_symbol}", "identifier"),
            ),
            10: F("vmr.isin", [V.isin], f"Valmora's ISIN is {V.isin}", "identifier"),
            11: F("vmr.cin", [V.cin], f"Valmora's CIN is {V.cin}", "identifier"),
            12: F(
                "vmr.reg_office",
                ["14 Sunrise Industrial Park", "Vapi"],
                "Registered office: 14 Sunrise Industrial Park, Vapi, Gujarat 396195",
                "text",
            ),
        },
    )
    pg.para(
        "Shareholders may write to the Company Secretary for any query on dividends, transfers or the Annual General "
        "Meeting. Queries on shares held in demat form should go to the depository participant."
    )


# ------------------------------------------------------------------ 3: highlights


def p03_highlights(pg) -> None:
    pg.title("Financial highlights")
    pg.para(
        "The table summarises Valmora's consolidated performance for the year ended 31 March 2024 (FY24) against the "
        "year ended 31 March 2023 (FY23). Amounts are in ₹ crore unless stated otherwise."
    )
    e24, e23 = V.ebitda["FY24"], V.ebitda["FY23"]
    r24, r23 = V.revenue["FY24"], V.revenue["FY23"]
    nd = V.net_debt
    rows = [
        ["Metric", "FY24", "FY23", "Change"],
        ["Revenue from operations (₹ crore)", n0(r24), n0(r23), "+" + growth(r24, r23)],
        ["EBITDA (₹ crore)", n0(e24), n0(e23), "+" + growth(e24, e23)],
        ["EBITDA margin", V.margin("FY24"), V.margin("FY23"), f"+{V.margin_expansion_bps()} bps"],
        [
            "Profit after tax (₹ crore)",
            n0(V.pat["FY24"]),
            n0(V.pat["FY23"]),
            "+" + growth(V.pat["FY24"], V.pat["FY23"]),
        ],
        ["Basic earnings per share (₹)", V.eps("FY24"), V.eps("FY23"), "+" + growth(V.pat["FY24"], V.pat["FY23"])],
        [
            "Net debt to EBITDA",
            V.net_debt_ebitda("FY24"),
            V.net_debt_ebitda("FY23"),
            f"{nd('FY24') / e24 - nd('FY23') / e23:+.2f}x",
        ],
        [
            "Return on capital employed (ROCE)",
            V.roce("FY24"),
            V.roce("FY23"),
            f"+{float(V.roce('FY24')[:-1]) - float(V.roce('FY23')[:-1]):.1f} pts",
        ],
        ["Dividend per share (₹)", n2(V.dividend_ps["FY24"]), n2(V.dividend_ps["FY23"]), "+25.0%"],
        [
            "Number of employees",
            n0(V.employees["FY24"]),
            n0(V.employees["FY23"]),
            "+" + growth(V.employees["FY24"], V.employees["FY23"]),
        ],
    ]
    pg.table(
        rows,
        [0.46, 0.18, 0.18, 0.18],
        facts={
            1: (
                F(
                    "vmr.rev.fy24",
                    ["Revenue from operations", n0(r24)],
                    f"FY24 revenue from operations was ₹ {n0(r24)} crore",
                ),
                F(
                    "vmr.rev.fy23",
                    ["Revenue from operations", n0(r23)],
                    f"FY23 revenue from operations was ₹ {n0(r23)} crore",
                ),
            ),
            2: (
                F("vmr.ebitda.fy24", ["EBITDA", n0(e24)], f"FY24 EBITDA was ₹ {n0(e24)} crore"),
                F("vmr.ebitda.fy23", ["EBITDA", n0(e23)], f"FY23 EBITDA was ₹ {n0(e23)} crore"),
            ),
            3: (
                F("vmr.margin.fy24", ["EBITDA margin", V.margin("FY24")], f"FY24 EBITDA margin was {V.margin('FY24')}"),
                F("vmr.margin.fy23", ["EBITDA margin", V.margin("FY23")], f"FY23 EBITDA margin was {V.margin('FY23')}"),
                F(
                    "vmr.margin.bps",
                    ["EBITDA margin", f"+{V.margin_expansion_bps()} bps"],
                    f"EBITDA margin expanded by {V.margin_expansion_bps()} basis points in FY24",
                ),
            ),
            4: (
                F(
                    "vmr.pat.fy24",
                    ["Profit after tax", n0(V.pat["FY24"])],
                    f"FY24 profit after tax was ₹ {n0(V.pat['FY24'])} crore",
                ),
                F(
                    "vmr.pat.fy23",
                    ["Profit after tax", n0(V.pat["FY23"])],
                    f"FY23 profit after tax was ₹ {n0(V.pat['FY23'])} crore",
                ),
            ),
            5: (
                F("vmr.eps.fy24", ["earnings per share", V.eps("FY24")], f"FY24 basic EPS was ₹ {V.eps('FY24')}"),
                F("vmr.eps.fy23", ["earnings per share", V.eps("FY23")], f"FY23 basic EPS was ₹ {V.eps('FY23')}"),
            ),
            6: (
                F(
                    "vmr.nd_ebitda.fy24",
                    ["Net debt to EBITDA", V.net_debt_ebitda("FY24")],
                    f"FY24 net debt to EBITDA was {V.net_debt_ebitda('FY24')}",
                ),
                F(
                    "vmr.nd_ebitda.fy23",
                    ["Net debt to EBITDA", V.net_debt_ebitda("FY23")],
                    f"FY23 net debt to EBITDA was {V.net_debt_ebitda('FY23')}",
                ),
            ),
            7: (
                F("vmr.roce.fy24", ["ROCE", V.roce("FY24")], f"FY24 ROCE was {V.roce('FY24')}"),
                F("vmr.roce.fy23", ["ROCE", V.roce("FY23")], f"FY23 ROCE was {V.roce('FY23')}"),
            ),
            8: (
                F(
                    "vmr.dps.fy24",
                    ["Dividend per share", n2(V.dividend_ps["FY24"])],
                    "FY24 dividend per share was ₹ 15.00",
                ),
                F(
                    "vmr.dps.fy23",
                    ["Dividend per share", n2(V.dividend_ps["FY23"])],
                    "FY23 dividend per share was ₹ 12.00",
                ),
            ),
            9: (
                F(
                    "vmr.employees.fy24",
                    ["Number of employees", n0(V.employees["FY24"])],
                    f"Valmora had {n0(V.employees['FY24'])} employees in FY24",
                ),
                F(
                    "vmr.employees.fy23",
                    ["Number of employees", n0(V.employees["FY23"])],
                    f"Valmora had {n0(V.employees['FY23'])} employees in FY23",
                ),
            ),
        },
    )
    pg.note(
        "Net debt is borrowings less cash and bank balances. ROCE is EBIT (profit before tax plus finance costs) divided "
        "by closing capital employed (total equity plus borrowings). EBITDA margin is EBITDA as a percentage of revenue "
        "from operations.",
        F(
            "vmr.footnote.net_debt",
            ["Net debt is borrowings less cash and bank balances"],
            "Net debt is borrowings less cash and bank balances",
            "footnote",
        ),
    )


# ------------------------------------------------------------------ 4-5: messages


def p04_chair(pg) -> None:
    pg.title("Message from the Chairperson")
    pg.para("Dear Shareholders,")
    pg.para(
        "FY24 was a year in which Valmora grew faster than its markets while strengthening its balance sheet. Demand "
        "for specialty chemicals recovered in the second half, our plastics customers in the automotive sector "
        "increased their schedules, and Digital Services continued to win larger, multi-year engagements. Revenue grew "
        "by more than 13 per cent and profit after tax by more than 31 per cent, and we ended the year with the "
        "lowest leverage in our recent history."
    )
    pg.para(
        "The Board has recommended a dividend of ₹ 15.00 per equity share, which is 150 per cent of the face value of "
        "₹ 10, for FY24, subject to your approval at the Annual General Meeting. The recommendation reflects our "
        "belief that shareholders should share in the improvement in cash generation.",
        F(
            "vmr.dividend_rec",
            ["₹ 15.00 per equity share", "150 per cent"],
            "The Board recommended a dividend of ₹ 15.00 per share (150% of face value) for FY24",
            "number",
        ),
    )
    pg.para(
        "Looking ahead, the Board has approved a capital expenditure programme of ₹ 1,100 crore over FY25 and FY26. "
        "About half is earmarked for specialty chemicals capacity and debottlenecking, a quarter for plastics "
        "automation, and the balance for Digital Services, including a new delivery centre in Pune that is scheduled "
        "to open in the first half of FY25.",
        F(
            "vmr.capex_plan",
            ["₹ 1,100 crore", "FY25 and FY26"],
            "Capex plan: ₹ 1,100 crore over FY25 and FY26",
            "number",
        ),
        F(
            "vmr.pune_centre",
            ["delivery centre in Pune", "first half of FY25"],
            "A new Digital Services delivery centre in Pune opens in the first half of FY25",
            "text",
        ),
    )
    pg.para(
        "Safety, sustainability and governance remain the foundations of our licence to operate. Our lost-time injury "
        "frequency fell for the third year in a row, absolute emissions declined, and the Board met seven times to "
        "review strategy, risk and performance. On behalf of the Board I thank our employees, customers, suppliers "
        "and shareholders for their trust."
    )
    pg.para("Dr. Meenakshi Raghavan<br/>Chairperson")


def p05_ceo(pg) -> None:
    pg.title("Review by the Managing Director and CEO")
    pg.para(
        "Three priorities shaped FY24: grow the businesses where we have a right to win, simplify how we operate, and "
        "decarbonise our plants. We made progress on all three."
    )
    pg.h2("Grow")
    pg.para(
        "In October 2023 we commissioned the Dahej fluorochemicals expansion, adding 60,000 tonnes per annum of "
        "capacity ahead of schedule and under budget. Digital Services onboarded 14 new enterprise clients and "
        "renewed every one of its top ten contracts.",
        F(
            "vmr.dahej",
            ["Dahej", "60,000 tonnes per annum"],
            "The Dahej fluorochemicals expansion added 60,000 tonnes per annum, commissioned in October 2023",
            "number",
        ),
        F(
            "vmr.digital_clients",
            ["14 new enterprise clients"],
            "Digital Services onboarded 14 new enterprise clients in FY24",
            "number",
        ),
    )
    pg.h2("Simplify")
    pg.para(
        "Our cost and complexity programme, Project Sudarshan, delivered savings of ₹ 112 crore in FY24 through "
        "procurement consolidation, energy efficiency and the closure of two low-volume product lines. It will "
        "continue in FY25 with a target of ₹ 140 crore.",
        F(
            "vmr.sudarshan",
            ["Project Sudarshan", "₹ 112 crore"],
            "Project Sudarshan delivered savings of ₹ 112 crore in FY24",
            "number",
        ),
        F(
            "vmr.sudarshan_target",
            ["Project Sudarshan", "₹ 140 crore"],
            "Project Sudarshan's FY25 savings target is ₹ 140 crore",
            "number",
        ),
    )
    pg.h2("Decarbonise")
    pg.para(
        "We added captive solar capacity at three plants and moved the Hosur polymer line to renewable power. The "
        "sustainability section of this report gives the detail."
    )
    pg.h2("Where we fell short")
    pg.para(
        "Engineered Plastics margins slipped because polymer feedstock prices rose about 9 per cent during the year "
        "and we could not pass on the full increase to customers with fixed-price contracts. We have changed the "
        "pricing clauses in new contracts so that feedstock movements are shared.",
        F(
            "vmr.plastics_reason",
            ["Engineered Plastics margins slipped", "feedstock prices rose about 9 per cent"],
            "Plastics margins slipped because feedstock prices rose about 9% and contracts were fixed-price",
            "text",
        ),
    )
    pg.para("Mr. Karthik Nambiar<br/>Managing Director and Chief Executive Officer")


# ------------------------------------------------------------------ 6-9: business and segments


def p06_overview(pg) -> None:
    pg.title("Business overview")
    pg.para(
        "Valmora operates three businesses that share a common platform of engineering, procurement and customer "
        "relationships. Specialty Chemicals makes fluorochemicals, agrochemical intermediates and pigments. "
        "Engineered Plastics makes polymer compounds and moulded components, mainly for automotive and appliance "
        "customers. Digital Services builds and runs software platforms for enterprise clients in manufacturing, "
        "banking and retail."
    )
    tot = V.revenue["FY24"]
    mix = {s: pct(V.seg_rev(s, "FY24"), tot) for s in V.segments}
    pg.table(
        [
            ["Segment", "Share of FY24 revenue", "Share of FY24 EBITDA"],
            *[[s, mix[s], pct(V.seg_ebitda(s, "FY24"), V.ebitda["FY24"])] for s in V.segments],
        ],
        [0.5, 0.25, 0.25],
        facts={
            1: F(
                "vmr.mix.chem",
                ["Specialty Chemicals", mix["Specialty Chemicals"]],
                f"Specialty Chemicals was {mix['Specialty Chemicals']} of FY24 revenue",
                "table_cell",
            ),
            2: F(
                "vmr.mix.plastics",
                ["Engineered Plastics", mix["Engineered Plastics"]],
                f"Engineered Plastics was {mix['Engineered Plastics']} of FY24 revenue",
                "table_cell",
            ),
            3: F(
                "vmr.mix.digital",
                ["Digital Services", mix["Digital Services"]],
                f"Digital Services was {mix['Digital Services']} of FY24 revenue",
                "table_cell",
            ),
        },
    )
    pg.para(
        "Exports contributed 37 per cent of consolidated revenue in FY24, up from 34 per cent in FY23, with the "
        "Middle East, Europe and South-East Asia the largest destinations. The Company served more than 2,100 "
        "customers during the year, and no single customer accounted for more than 6 per cent of revenue.",
        F(
            "vmr.exports",
            ["Exports contributed 37 per cent", "34 per cent in FY23"],
            "Exports were 37% of FY24 revenue (FY23: 34%)",
            "number",
        ),
        F("vmr.customers", ["more than 2,100 customers"], "Valmora served more than 2,100 customers in FY24", "number"),
    )
    pg.para(
        "Each business has its own leadership team and performance targets, but capital allocation, treasury, "
        "procurement of common inputs and sustainability targets are managed centrally by the Managing Director and "
        "the Chief Financial Officer."
    )


def seg_facts(seg: str, slug: str) -> dict[str, F]:
    """Facts about a segment, stated both on its review page and in the segment results table."""
    r24, r23 = V.seg_rev(seg, "FY24"), V.seg_rev(seg, "FY23")
    e24 = V.seg_ebitda(seg, "FY24")
    return {
        "rev.fy24": F(f"vmr.seg.{slug}.rev.fy24", [seg, n0(r24)], f"{seg} revenue was ₹ {n0(r24)} crore in FY24"),
        "rev.fy23": F(f"vmr.seg.{slug}.rev.fy23", [seg, n0(r23)], f"{seg} revenue was ₹ {n0(r23)} crore in FY23"),
        "ebitda.fy24": F(f"vmr.seg.{slug}.ebitda.fy24", [seg, n0(e24)], f"{seg} EBITDA was ₹ {n0(e24)} crore in FY24"),
        "margin.fy24": F(
            f"vmr.seg.{slug}.margin.fy24",
            [seg, V.seg_margin(seg, "FY24")],
            f"{seg} EBITDA margin was {V.seg_margin(seg, 'FY24')} in FY24",
        ),
        "margin.fy23": F(
            f"vmr.seg.{slug}.margin.fy23",
            [seg, V.seg_margin(seg, "FY23")],
            f"{seg} EBITDA margin was {V.seg_margin(seg, 'FY23')} in FY23",
        ),
    }


def _segment_page(pg, seg: str, slug: str, util: tuple[str, str], intro: str) -> None:
    r24, r23 = V.seg_rev(seg, "FY24"), V.seg_rev(seg, "FY23")
    e24 = V.seg_ebitda(seg, "FY24")
    f = seg_facts(seg, slug)
    pg.title(f"Segment review: {seg}")
    pg.para(intro)
    pg.para(
        f"{seg} segment revenue grew {growth(r24, r23)} to {cr(r24)} in FY24 (FY23: {cr(r23)}). {seg} EBITDA was "
        f"{cr(e24)}, a margin of {V.seg_margin(seg, 'FY24')} (FY23: {V.seg_margin(seg, 'FY23')}).",
        *f.values(),
    )
    pg.para(
        f"Capacity utilisation was {util[0]} in FY24 (FY23: {util[1]}).",
        F(
            f"vmr.seg.{slug}.util",
            [f"Capacity utilisation was {util[0]} in FY24", f"FY23: {util[1]}"],
            f"{seg} capacity utilisation was {util[0]} in FY24 (FY23: {util[1]})",
            "number",
        ),
    )


def p07_chemicals(pg) -> None:
    _segment_page(
        pg,
        "Specialty Chemicals",
        "chem",
        ("86%", "81%"),
        "Specialty Chemicals is Valmora's largest business, supplying fluoro-specialties, agrochemical intermediates "
        "and high-performance pigments to customers in more than 40 countries.",
    )
    pg.h2("Highlights")
    pg.bullets(
        [
            (
                "Fluoro-specialties grew 17 per cent in volume, helped by the Dahej expansion.",
                (
                    F(
                        "vmr.chem.fluoro_growth",
                        ["Fluoro-specialties grew 17 per cent in volume"],
                        "Fluoro-specialties volumes grew 17% in FY24",
                        "number",
                    ),
                ),
            ),
            ("Agrochemical intermediates volumes were flat as customers destocked in the first half.", ()),
            (
                "The pigments line qualified with two new European automotive coatings customers.",
                (
                    F(
                        "vmr.chem.pigments",
                        ["two new European automotive coatings customers"],
                        "The pigments line qualified with two new European automotive coatings customers",
                        "text",
                    ),
                ),
            ),
            (
                "Zero liquid discharge was achieved at the Dahej and Vapi sites.",
                (
                    F(
                        "vmr.chem.zld",
                        ["Zero liquid discharge", "Dahej and Vapi"],
                        "Zero liquid discharge was achieved at Dahej and Vapi",
                        "text",
                    ),
                ),
            ),
        ]
    )
    pg.para(
        "Raw material costs eased in the second half, and a richer product mix lifted the margin even as selling "
        "prices of commodity intermediates fell."
    )


def p08_plastics(pg) -> None:
    _segment_page(
        pg,
        "Engineered Plastics",
        "plastics",
        ("78%", "79%"),
        "Engineered Plastics makes polymer compounds and precision-moulded components for automotive, appliance and "
        "electrical customers from four plants in Tamil Nadu, Gujarat and Maharashtra.",
    )
    pg.h2("Highlights")
    pg.bullets(
        [
            (
                "Automotive customers accounted for 44 per cent of segment revenue.",
                (
                    F(
                        "vmr.plastics.auto",
                        ["Automotive customers accounted for 44 per cent"],
                        "Automotive was 44% of Engineered Plastics revenue",
                        "number",
                    ),
                ),
            ),
            ("The Hosur compounding line moved to renewable power in the fourth quarter.", ()),
            (
                "Segment margin fell by 0.6 percentage points as feedstock costs rose faster than prices.",
                (
                    F(
                        "vmr.plastics.margin_fall",
                        ["Segment margin fell by 0.6 percentage points"],
                        "Engineered Plastics margin fell 0.6 percentage points",
                        "number",
                    ),
                ),
            ),
            (
                "A robotic moulding cell at Sanand cut cycle time by 12 per cent.",
                (
                    F(
                        "vmr.plastics.sanand",
                        ["robotic moulding cell at Sanand", "12 per cent"],
                        "A robotic moulding cell at Sanand cut cycle time by 12%",
                        "number",
                    ),
                ),
            ),
        ]
    )
    pg.para(
        "New contracts signed after January 2024 include a feedstock pass-through clause, which should reduce margin "
        "volatility from FY25 onwards."
    )


def p09_digital(pg) -> None:
    _segment_page(
        pg,
        "Digital Services",
        "digital",
        ("82%", "76%"),
        "Digital Services designs, builds and operates software platforms for enterprise clients, delivering from "
        "centres in Ahmedabad, Bengaluru and Chennai.",
    )
    pg.h2("Highlights")
    pg.bullets(
        [
            (
                "Segment headcount rose to 3,950 from 3,410 a year earlier.",
                (
                    F(
                        "vmr.digital.headcount",
                        ["3,950", "3,410"],
                        "Digital Services headcount was 3,950 (FY23: 3,410)",
                        "number",
                    ),
                ),
            ),
            (
                "The order book stood at ₹ 2,310 crore on 31 March 2024, compared with ₹ 1,640 crore a year earlier.",
                (
                    F(
                        "vmr.digital.orderbook",
                        ["order book", "2,310 crore"],
                        "Digital Services order book was ₹ 2,310 crore at 31 March 2024 (FY23: ₹ 1,640 crore)",
                        "number",
                    ),
                ),
            ),
            (
                "Revenue from clients outside India was 58 per cent of the segment total.",
                (
                    F(
                        "vmr.digital.overseas",
                        ["58 per cent of the segment total"],
                        "58% of Digital Services revenue came from clients outside India",
                        "number",
                    ),
                ),
            ),
            (
                "The proprietary platform PRISM now runs the production planning of 31 manufacturing clients.",
                (
                    F(
                        "vmr.digital.prism",
                        ["PRISM", "31 manufacturing clients"],
                        "PRISM runs production planning for 31 manufacturing clients",
                        "text",
                    ),
                ),
            ),
        ]
    )
    pg.para(
        "Utilisation improved because new engineers were deployed on client projects faster, and the share of "
        "fixed-price work with higher margins increased."
    )


# ------------------------------------------------------------------ 10-15: operations, sustainability, people, risk, governance


def p10_operations(pg) -> None:
    pg.title("Operations and manufacturing footprint")
    pg.para(
        "Valmora runs eight manufacturing plants and three digital delivery centres. Installed capacity is stated in "
        "tonnes per annum (tpa) for plants and in seats for delivery centres."
    )
    rows = [
        ["Facility", "Location", "Segment", "Installed capacity", "Commissioned"],
        ["Dahej Complex", "Dahej, Gujarat", "Specialty Chemicals", "2,10,000 tpa", "2009 (expanded 2023)"],
        ["Vapi Unit I", "Vapi, Gujarat", "Specialty Chemicals", "90,000 tpa", "1996"],
        ["Vapi Unit II", "Vapi, Gujarat", "Specialty Chemicals", "75,000 tpa", "2004"],
        ["Taloja Works", "Taloja, Maharashtra", "Specialty Chemicals", "1,05,000 tpa", "2007"],
        ["Hosur Polymers", "Hosur, Tamil Nadu", "Engineered Plastics", "1,20,000 tpa", "2011"],
        ["Sanand Compounding", "Sanand, Gujarat", "Engineered Plastics", "95,000 tpa", "2015"],
        ["Pune Moulding Park", "Pune, Maharashtra", "Engineered Plastics", "55,000 tpa", "2018"],
        ["Aurangabad Films", "Aurangabad, Maharashtra", "Engineered Plastics", "40,000 tpa", "2020"],
        ["Ahmedabad Centre", "Ahmedabad, Gujarat", "Digital Services", "1,450 seats", "2014"],
        ["Bengaluru Centre", "Bengaluru, Karnataka", "Digital Services", "1,300 seats", "2017"],
        ["Chennai Centre", "Chennai, Tamil Nadu", "Digital Services", "1,150 seats", "2021"],
    ]
    pg.table(
        rows,
        [0.2, 0.23, 0.22, 0.17, 0.18],
        numeric_from=9,
        facts={
            1: F(
                "vmr.plant.dahej",
                ["Dahej Complex", "2,10,000 tpa"],
                "The Dahej Complex has 2,10,000 tpa of capacity",
                "table_cell",
            ),
            5: F(
                "vmr.plant.hosur",
                ["Hosur Polymers", "1,20,000 tpa"],
                "Hosur Polymers has 1,20,000 tpa of capacity",
                "table_cell",
            ),
            7: F(
                "vmr.plant.pune",
                ["Pune Moulding Park", "55,000 tpa", "2018"],
                "Pune Moulding Park has 55,000 tpa, commissioned in 2018",
                "table_cell",
            ),
            10: F(
                "vmr.plant.bengaluru",
                ["Bengaluru Centre", "1,300 seats"],
                "The Bengaluru centre has 1,300 seats",
                "table_cell",
            ),
        },
    )
    pg.para(
        "Total installed capacity of the chemicals plants is 4.8 lakh tpa and of the plastics plants 3.1 lakh tpa. "
        "Specialty Chemicals plants ran at 86 per cent utilisation in FY24 and Engineered Plastics plants at 78 per "
        "cent.",
        F(
            "vmr.capacity_total",
            ["4.8 lakh tpa", "3.1 lakh tpa"],
            "Chemicals capacity is 4.8 lakh tpa and plastics capacity 3.1 lakh tpa",
            "number",
        ),
    )
    pg.para(
        "Every plant holds ISO 9001 and ISO 14001 certification, and the Dahej and Taloja works are also certified "
        "to ISO 45001 for occupational health and safety.",
        F("vmr.iso45001", ["Dahej and Taloja", "ISO 45001"], "Dahej and Taloja are ISO 45001 certified", "text"),
    )


def p11_sustainability(pg) -> None:
    pg.title("Sustainability")
    pg.para(
        "Our sustainability strategy rests on three commitments: lower emissions, responsible use of water and "
        "inclusive growth for the communities around our plants."
    )
    pg.h2("Emissions and energy")
    pg.para(
        "Absolute Scope 1 and 2 emissions fell 14 per cent to 3.54 lakh tonnes of CO2 equivalent in FY24 from 4.12 "
        "lakh tonnes in FY23. Captive solar capacity reached 42 MW from 28 MW, and renewable sources supplied 37 per "
        "cent of our electricity, up from 29 per cent.",
        F(
            "vmr.esg.emissions",
            ["14 per cent", "3.54 lakh tonnes"],
            "Scope 1 and 2 emissions fell 14% to 3.54 lakh tonnes of CO2e in FY24 (FY23: 4.12 lakh)",
            "number",
        ),
        F("vmr.esg.solar", ["42 MW", "28 MW"], "Captive solar capacity reached 42 MW in FY24 from 28 MW", "number"),
        F(
            "vmr.esg.renewable_share",
            ["37 per cent of our electricity", "29 per cent"],
            "Renewables supplied 37% of electricity in FY24 (FY23: 29%)",
            "number",
        ),
    )
    pg.para(
        "We aim to reach net zero Scope 1 and 2 emissions by 2040, with an interim target of a 30 per cent reduction "
        "against FY23 by FY27.",
        F(
            "vmr.esg.netzero",
            ["net zero Scope 1 and 2 emissions by 2040"],
            "Valmora targets net zero Scope 1 and 2 emissions by 2040",
            "number",
        ),
        F(
            "vmr.esg.interim",
            ["30 per cent reduction", "FY27"],
            "Interim target: 30% reduction against FY23 by FY27",
            "number",
        ),
    )
    pg.h2("Water")
    pg.para(
        "Recycled water met 38 per cent of total water consumption in FY24 against 31 per cent in FY23. Zero liquid "
        "discharge operates at Dahej and Vapi.",
        F(
            "vmr.esg.water",
            ["38 per cent of total water consumption", "31 per cent"],
            "Recycled water met 38% of consumption in FY24 (FY23: 31%)",
            "number",
        ),
    )
    pg.h2("Communities")
    pg.para(
        "Corporate social responsibility spending was ₹ 17.4 crore in FY24 (FY23: ₹ 14.9 crore), in line with the "
        "statutory requirement of 2 per cent of average net profit of the preceding three years. Programmes focused on "
        "school science laboratories, vocational training and clean drinking water in villages near Dahej and Hosur.",
        F(
            "vmr.esg.csr",
            ["₹ 17.4 crore", "₹ 14.9 crore"],
            "CSR spend was ₹ 17.4 crore in FY24 (FY23: ₹ 14.9 crore)",
            "number",
        ),
    )


def p12_people(pg) -> None:
    pg.title("People and culture")
    pg.para(
        "Valmora employed 9,842 people at 31 March 2024. Hiring in Digital Services and in the new Dahej unit more "
        "than offset attrition elsewhere. The table gives our main people indicators."
    )
    rows = [
        ["Indicator", "FY24", "FY23"],
        ["Total employees", n0(V.employees["FY24"]), n0(V.employees["FY23"])],
        ["Women as a share of the workforce", "17.6%", "16.1%"],
        ["Voluntary attrition rate", "11.2%", "13.5%"],
        ["Average training hours per employee", "31", "27"],
        ["Lost-time injury frequency rate (per million man-hours)", "0.21", "0.34"],
        ["Average age of employees (years)", "33", "34"],
    ]
    pg.table(
        rows,
        [0.6, 0.2, 0.2],
        facts={
            2: (
                F(
                    "vmr.people.women.fy24",
                    ["Women as a share of the workforce", "17.6%"],
                    "Women were 17.6% of the workforce in FY24",
                    "table_cell",
                ),
                F(
                    "vmr.people.women.fy23",
                    ["Women as a share of the workforce", "16.1%"],
                    "Women were 16.1% of the workforce in FY23",
                    "table_cell",
                ),
            ),
            3: (
                F(
                    "vmr.people.attrition.fy24",
                    ["Voluntary attrition rate", "11.2%"],
                    "Voluntary attrition was 11.2% in FY24",
                    "table_cell",
                ),
                F(
                    "vmr.people.attrition.fy23",
                    ["Voluntary attrition rate", "13.5%"],
                    "Voluntary attrition was 13.5% in FY23",
                    "table_cell",
                ),
            ),
            4: F(
                "vmr.people.training",
                ["Average training hours per employee", "31", "27"],
                "Average training hours per employee were 31 in FY24 (FY23: 27)",
                "table_cell",
            ),
            5: (
                F(
                    "vmr.people.ltifr.fy24",
                    ["Lost-time injury frequency rate", "0.21"],
                    "LTIFR was 0.21 in FY24",
                    "table_cell",
                ),
                F(
                    "vmr.people.ltifr.fy23",
                    ["Lost-time injury frequency rate", "0.34"],
                    "LTIFR was 0.34 in FY23",
                    "table_cell",
                ),
            ),
        },
    )
    pg.para(
        "The Valmora ESOP 2021 granted 3.2 lakh stock options to 212 employees in FY24 at an exercise price of ₹ 1,150 "
        "per option. Options vest in four equal annual instalments starting one year after the grant.",
        F(
            "vmr.people.esop",
            ["ESOP 2021", "3.2 lakh stock options", "₹ 1,150"],
            "ESOP 2021 granted 3.2 lakh options in FY24 at an exercise price of ₹ 1,150",
            "number",
        ),
    )
    pg.para(
        "The ethics hotline, run by an independent provider, received 41 complaints in FY24, of which 38 were closed "
        "within 45 days. None involved a violation of the Code of Conduct by a director or senior manager.",
        F(
            "vmr.people.hotline",
            ["41 complaints", "38 were closed within 45 days"],
            "The ethics hotline received 41 complaints in FY24 and 38 were closed within 45 days",
            "number",
        ),
    )


def p13_risk(pg) -> None:
    pg.title("Risk management")
    pg.para(
        "The Risk Management Committee reviews the enterprise risk register every quarter. The principal risks and our "
        "responses are summarised below."
    )
    rows = [
        ["Risk", "How it could affect Valmora", "Response"],
        [
            "Raw material price volatility",
            "Propylene and fluorspar together make up 41 per cent of materials cost; price spikes squeeze margins.",
            "Annual supply contracts, a second source for every key input and feedstock pass-through clauses in new plastics contracts.",
        ],
        [
            "Foreign exchange",
            "Exports are 37 per cent of revenue; a stronger rupee reduces realisations.",
            "We hedge 70 per cent of the net exposure for the next twelve months with forward contracts.",
        ],
        [
            "Environmental and regulatory",
            "Stricter effluent and emission norms could require additional capital spend.",
            "Zero liquid discharge at Dahej and Vapi; early adoption of upcoming norms in new capacity.",
        ],
        [
            "Customer concentration",
            "The top ten customers contributed 29 per cent of FY24 revenue (FY23: 32 per cent).",
            "Broadening the customer base; multi-year contracts with the largest accounts.",
        ],
        [
            "Cyber security",
            "An attack on plant control systems or client platforms could stop production or delivery.",
            "24x7 security operations centre, quarterly penetration tests, and ISO 27001 certification of Digital Services.",
        ],
        [
            "Talent",
            "Shortage of process engineers and software specialists could delay projects.",
            "Campus programmes, ESOPs and training hours above industry averages.",
        ],
    ]
    pg.table(
        rows,
        [0.2, 0.42, 0.38],
        numeric_from=9,
        facts={
            1: F(
                "vmr.risk.materials",
                ["Propylene and fluorspar", "41 per cent of materials cost"],
                "Propylene and fluorspar make up 41% of materials cost",
                "table_cell",
            ),
            2: F(
                "vmr.risk.hedge",
                ["We hedge 70 per cent of the net exposure", "twelve months"],
                "Valmora hedges 70% of its net FX exposure for the next twelve months",
                "table_cell",
            ),
            4: F(
                "vmr.risk.concentration",
                ["top ten customers contributed 29 per cent", "32 per cent"],
                "The top ten customers contributed 29% of FY24 revenue (FY23: 32%)",
                "table_cell",
            ),
            5: F(
                "vmr.risk.cyber",
                ["24x7 security operations centre", "ISO 27001"],
                "Valmora runs a 24x7 security operations centre; Digital Services is ISO 27001 certified",
                "table_cell",
            ),
        },
    )
    pg.para(
        "The Risk Management Committee, chaired by the Managing Director, met four times in FY24. The Audit Committee "
        "reviews the adequacy of internal financial controls twice a year."
    )


def p14_governance(pg) -> None:
    pg.title("Corporate governance report")
    pg.para(
        "The Board has nine directors: two executive and seven non-executive, of whom five are independent. Three "
        "directors are women. The Board met seven times in FY24, and average attendance at Board and committee "
        "meetings was 96 per cent.",
        F(
            "vmr.gov.board_size",
            ["nine directors", "five are independent"],
            "The Board has nine directors, five independent",
            "number",
        ),
        F(
            "vmr.gov.board_meetings",
            ["met seven times in FY24", "96 per cent"],
            "The Board met seven times in FY24; average attendance was 96%",
            "number",
        ),
    )
    rows = [
        ["Committee", "Chair", "Members", "Meetings in FY24"],
        ["Audit Committee", "Mr. Aditya Rao", "4", "5 meetings"],
        ["Nomination and Remuneration Committee", "Prof. Lakshmi Subramanian", "4", "3 meetings"],
        ["Stakeholders Relationship Committee", "Ms. Ishita Banerjee", "3", "2 meetings"],
        ["Corporate Social Responsibility Committee", "Dr. Meenakshi Raghavan", "4", "2 meetings"],
        ["Risk Management Committee", "Mr. Karthik Nambiar", "6", "4 meetings"],
    ]
    pg.table(
        rows,
        [0.38, 0.28, 0.14, 0.2],
        numeric_from=2,
        facts={
            1: F(
                "vmr.gov.audit",
                ["Audit Committee", "Aditya Rao", "5 meetings"],
                "The Audit Committee, chaired by Aditya Rao, met 5 times in FY24",
                "table_cell",
            ),
            2: F(
                "vmr.gov.nrc",
                ["Nomination and Remuneration Committee", "Lakshmi Subramanian"],
                "The Nomination and Remuneration Committee is chaired by Prof. Lakshmi Subramanian",
                "table_cell",
            ),
            4: F(
                "vmr.gov.csr",
                ["Corporate Social Responsibility Committee", "Meenakshi Raghavan", "2 meetings"],
                "The CSR Committee, chaired by Dr. Meenakshi Raghavan, met twice in FY24",
                "table_cell",
            ),
            5: F(
                "vmr.gov.risk",
                ["Risk Management Committee", "Karthik Nambiar", "4 meetings"],
                "The Risk Management Committee, chaired by Karthik Nambiar, met 4 times",
                "table_cell",
            ),
        },
    )
    pg.para(
        "Independent directors held one meeting without executives, as required, and confirmed that they meet the "
        "criteria of independence. The Board evaluated its own performance, that of its committees and of each "
        "director, with the help of an external facilitator.",
        F(
            "vmr.gov.evaluation",
            ["external facilitator"],
            "The Board evaluation was run with an external facilitator",
            "text",
        ),
    )
    pg.para(
        "Key managerial personnel remuneration is disclosed in the Annexure to the Directors' Report. The Managing "
        "Director's remuneration for FY24 was ₹ 6.8 crore, including performance pay.",
        F(
            "vmr.gov.md_pay",
            ["Managing Director's remuneration", "₹ 6.8 crore"],
            "The Managing Director's FY24 remuneration was ₹ 6.8 crore",
            "number",
        ),
    )


def p15_board(pg) -> None:
    pg.title("Board of directors")
    for name, text, facts in (
        (
            "Dr. Meenakshi Raghavan, Chairperson",
            "Independent director and Chairperson since 2019. A chemical engineer and former head of a national research laboratory, she chairs the CSR Committee.",
            (
                F(
                    "vmr.board.chair_since",
                    ["Chairperson since 2019"],
                    "Dr. Meenakshi Raghavan has been Chairperson since 2019",
                    "text",
                ),
            ),
        ),
        (
            "Mr. Karthik Nambiar, Managing Director and CEO",
            "Joined Valmora in 2003 and has led the Company as Managing Director since 2016. He chairs the Risk Management Committee.",
            (
                F(
                    "vmr.board.md_since",
                    ["Managing Director since 2016"],
                    "Karthik Nambiar has been Managing Director since 2016",
                    "text",
                ),
            ),
        ),
        (
            "Mr. Aditya Rao, Independent Director",
            "A retired chief financial officer of a listed manufacturer, he chairs the Audit Committee and has served on the Board since 2018.",
            (
                F(
                    "vmr.board.rao",
                    ["Aditya Rao", "chairs the Audit Committee", "since 2018"],
                    "Aditya Rao chairs the Audit Committee and joined the Board in 2018",
                    "text",
                ),
            ),
        ),
        (
            "Ms. Ishita Banerjee, Independent Director",
            "An expert in cyber security and data protection, she joined the Board in 2021 and chairs the Stakeholders Relationship Committee.",
            (
                F(
                    "vmr.board.banerjee",
                    ["Ishita Banerjee", "cyber security", "joined the Board in 2021"],
                    "Ishita Banerjee is a cyber security expert who joined the Board in 2021",
                    "text",
                ),
            ),
        ),
        (
            "Prof. Lakshmi Subramanian, Independent Director",
            "A professor of chemical engineering for 25 years, she chairs the Nomination and Remuneration Committee.",
            (
                F(
                    "vmr.board.subramanian",
                    ["Lakshmi Subramanian", "professor of chemical engineering for 25 years"],
                    "Prof. Lakshmi Subramanian was a professor of chemical engineering for 25 years",
                    "text",
                ),
            ),
        ),
        (
            "Mr. Vikram Oberoi, Non-Executive Director",
            "Nominated by the promoter group. He has 30 years of experience in industrial investments and is not an independent director.",
            (
                F(
                    "vmr.board.oberoi",
                    ["Vikram Oberoi", "Nominated by the promoter group"],
                    "Vikram Oberoi is a non-executive director nominated by the promoter group",
                    "text",
                ),
            ),
        ),
    ):
        pg.para(f"<b>{name}.</b> {text}", *facts)
    pg.para(
        "Brief profiles of the remaining three directors are available on the Company's website. No director is "
        "related to any other director."
    )


# ------------------------------------------------------------------ 16-17: management discussion and analysis


def p16_outlook(pg) -> None:
    pg.title("Management discussion and analysis: industry outlook and prior-year recap")
    pg.h2("Industry outlook")
    pg.para(
        "Indian specialty chemicals demand is expected to grow at 9 to 11 per cent a year through FY27, supported by "
        "global supply-chain diversification. Automotive production growth of about 6 per cent should support "
        "demand for engineered plastics, while enterprise technology spending is expected to grow 8 per cent.",
        F(
            "vmr.mda.industry",
            ["9 to 11 per cent a year through FY27"],
            "Specialty chemicals demand is expected to grow 9 to 11% a year through FY27",
            "number",
        ),
    )
    pg.para(
        "Risks to the outlook include volatile feedstock prices, freight disruptions in the Red Sea and slower "
        "discretionary spending in Europe."
    )
    pg.h2("Recap of FY23, the comparative year")
    r23, e23 = V.revenue["FY23"], V.ebitda["FY23"]
    pg.para(
        f"In FY23, Valmora's revenue from operations was {cr(r23)} and EBITDA was {cr(e23)}, an EBITDA margin of "
        f"{V.margin('FY23')}. Profit after tax was {cr(V.pat['FY23'])} and basic earnings per share were ₹ {V.eps('FY23')}. "
        f"Net debt stood at {cr(V.net_debt('FY23'))} at 31 March 2023, or {V.net_debt_ebitda('FY23')} EBITDA.",
        F(
            "vmr.recap.fy23.margin",
            ["In FY23", "EBITDA margin", V.margin("FY23")],
            f"In FY23 the EBITDA margin was {V.margin('FY23')}",
        ),
        F(
            "vmr.recap.fy23.pat",
            ["In FY23", "Profit after tax was", n0(V.pat["FY23"])],
            f"In FY23 profit after tax was ₹ {n0(V.pat['FY23'])} crore",
        ),
        F(
            "vmr.recap.fy23.net_debt",
            ["Net debt stood at", n0(V.net_debt("FY23")), "31 March 2023"],
            f"Net debt was ₹ {n0(V.net_debt('FY23'))} crore at 31 March 2023",
            "number",
        ),
    )
    pg.para(
        "FY23 was affected by elevated raw material prices in the first half and by the ramp-up costs of the Hosur "
        "polymer line. The effective tax rate was 25.1 per cent, and the final dividend for FY23 was ₹ 12.00 per share.",
        F(
            "vmr.recap.fy23.tax",
            ["effective tax rate was 25.1 per cent"],
            "FY23 effective tax rate was 25.1%",
            "number",
        ),
    )


def p17_mda(pg) -> None:
    r24, e24 = V.revenue["FY24"], V.ebitda["FY24"]
    pg.title("Management discussion and analysis: FY24 financial performance")
    pg.para(
        f"Revenue from operations increased {growth(r24, V.revenue['FY23'])} to {cr(r24)} in FY24. Digital Services "
        f"contributed the fastest growth at {growth(V.seg_rev('Digital Services', 'FY24'), V.seg_rev('Digital Services', 'FY23'))}, "
        "while Specialty Chemicals added the largest absolute amount.",
        F(
            "vmr.mda.rev_growth",
            ["Revenue from operations increased 13.6%", "7,365 crore"],
            "FY24 revenue from operations grew 13.6% to ₹ 7,365 crore",
            "number",
        ),
    )
    pg.para(
        f"EBITDA rose {growth(e24, V.ebitda['FY23'])} to {cr(e24)}. The EBITDA margin was {V.margin('FY24')} in FY24, up "
        f"{V.margin_expansion_bps()} basis points from {V.margin('FY23')} in FY23, on operating leverage, Project "
        "Sudarshan savings and a richer product mix.",
        F(
            "vmr.mda.margin",
            ["The EBITDA margin was 21.0% in FY24", "118 basis points", "19.8% in FY23"],
            "FY24 EBITDA margin was 21.0%, up 118 basis points from 19.8% in FY23",
            "number",
        ),
    )
    pg.para(
        f"Finance costs fell {growth(V.finance['FY24'], V.finance['FY23']).lstrip('-')} to {cr(V.finance['FY24'])} as the Company repaid "
        f"₹ 214 crore of term loans, and the effective tax rate was {pct(V.tax['FY24'], V.pbt('FY24'))}. Profit after tax grew "
        f"{growth(V.pat['FY24'], V.pat['FY23'])} to {cr(V.pat['FY24'])}.",
        F(
            "vmr.mda.finance_costs",
            ["Finance costs fell 18.6%", "96 crore"],
            "Finance costs fell 18.6% to ₹ 96 crore in FY24",
            "number",
        ),
        F("vmr.mda.tax_rate", ["effective tax rate was 25.0%"], "FY24 effective tax rate was 25.0%", "number"),
        F(
            "vmr.mda.term_loans",
            ["repaid ₹ 214 crore of term loans"],
            "Valmora repaid ₹ 214 crore of term loans in FY24",
            "number",
        ),
    )
    pg.para(
        "Other income included a one-time gain of ₹ 41 crore on the sale of surplus land at Taloja, which is not "
        "expected to recur. Net working capital days improved to 41 from 48, with inventory days at 62 (FY23: 70) and "
        "receivable days at 67 (FY23: 68).",
        F(
            "vmr.land_gain",
            ["one-time gain of ₹ 41 crore", "Taloja"],
            "Other income included a one-time gain of ₹ 41 crore on the sale of surplus land at Taloja",
            "number",
        ),
        F(
            "vmr.mda.wc_days",
            ["Net working capital days improved to 41 from 48"],
            "Net working capital days improved to 41 from 48",
            "number",
        ),
    )
    pg.para(
        f"Net debt declined to {cr(V.net_debt('FY24'))} from {cr(V.net_debt('FY23'))}, and ROCE rose to {V.roce('FY24')} from "
        f"{V.roce('FY23')}. Free cash flow, defined as operating cash flow less capital expenditure, was "
        f"{cr(V.free_cash_flow('FY24'))} against {cr(V.free_cash_flow('FY23'))} in FY23.",
        F(
            "vmr.mda.fcf",
            ["Free cash flow", "586 crore", "410 crore"],
            "FY24 free cash flow was ₹ 586 crore against ₹ 410 crore in FY23",
            "number",
        ),
        F(
            "vmr.mda.net_debt",
            ["Net debt declined to", "831 crore"],
            "Net debt declined to ₹ 831 crore at 31 March 2024",
            "number",
        ),
    )
    pg.para(
        "Internal financial controls were tested by the internal auditors during the year and no material weakness was "
        "reported. The Audit Committee reviewed the results of the testing."
    )


# ------------------------------------------------------------------ 18-20: segment and quarterly tables


def p18_segments(pg) -> None:
    pg.title("Segment results")
    pg.para("Segment revenue and EBITDA for FY24 and FY23 are shown below (₹ crore).")
    head = [
        "Segment",
        "Revenue FY24",
        "Revenue FY23",
        "EBITDA FY24",
        "EBITDA FY23",
        "EBITDA margin FY24",
        "EBITDA margin FY23",
    ]
    rows = [head]
    facts = {}
    keys = {"Specialty Chemicals": "chem", "Engineered Plastics": "plastics", "Digital Services": "digital"}
    for i, seg in enumerate(V.segments, start=1):
        rows.append(
            [
                seg + ("<super>1</super>" if seg == "Digital Services" else ""),
                n0(V.seg_rev(seg, "FY24")),
                n0(V.seg_rev(seg, "FY23")),
                n0(V.seg_ebitda(seg, "FY24")),
                n0(V.seg_ebitda(seg, "FY23")),
                V.seg_margin(seg, "FY24"),
                V.seg_margin(seg, "FY23"),
            ]
        )
        facts[i] = tuple(seg_facts(seg, keys[seg]).values())
    rows.append(
        [
            "Total",
            n0(V.revenue["FY24"]),
            n0(V.revenue["FY23"]),
            n0(V.ebitda["FY24"]),
            n0(V.ebitda["FY23"]),
            V.margin("FY24"),
            V.margin("FY23"),
        ]
    )
    pg.table(rows, [0.2, 0.12, 0.12, 0.12, 0.12, 0.16, 0.16], facts=facts, bold_rows=(4,))
    pg.note(
        "1. Digital Services revenue is stated net of ₹ 96 crore of inter-segment sales to Specialty Chemicals and "
        "Engineered Plastics, which are eliminated on consolidation.",
        F(
            "vmr.footnote.intersegment",
            ["₹ 96 crore of inter-segment sales"],
            "Digital Services revenue is net of ₹ 96 crore of inter-segment sales eliminated on consolidation",
            "footnote",
        ),
    )
    pg.note("Segment EBITDA is stated after allocating all corporate costs. There are no unallocated items.")
    pg.para(
        "Digital Services is the only segment whose margin expanded by more than 4 percentage points. Engineered "
        "Plastics is the only segment whose margin declined."
    )


def _quarter_page(pg, year: str, title_note: str, headline: str, facts_key: str) -> None:
    q = V.quarters[year]
    rows = [["Quarter", "Revenue", "EBITDA", "EBITDA margin", "Profit after tax"]]
    facts = {}
    for i, (name, (rev, e, p)) in enumerate(q.items(), start=1):
        rows.append([f"{name} {year}", n0(rev), n0(e), V.q_margin(year, name), n0(p)])
        facts[i] = F(
            f"vmr.q.{facts_key}.{name.lower()}",
            [f"{name} {year}", n0(rev), n0(e), V.q_margin(year, name)],
            f"{name} {year}: revenue ₹ {n0(rev)} crore, EBITDA margin {V.q_margin(year, name)}",
            "table_cell",
        )
    rows.append([f"Full year {year}", n0(V.revenue[year]), n0(V.ebitda[year]), V.margin(year), n0(V.pat[year])])
    pg.title(title_note)
    pg.para(headline)
    pg.table(rows, [0.28, 0.18, 0.16, 0.2, 0.18], facts=facts, bold_rows=(5,))


def p19_q_fy24(pg) -> None:
    q4 = V.quarters["FY24"]["Q4"]
    _quarter_page(
        pg,
        "FY24",
        "Quarterly performance: FY24",
        f"Revenue and profit rose in every quarter of FY24. The fourth quarter was the strongest, with revenue of "
        f"₹ {n0(q4[0])} crore and an EBITDA margin of {V.q_margin('FY24', 'Q4')} (₹ crore).",
        "fy24",
    )
    pg.para(
        "Quarterly margins improved sequentially through FY24 as feedstock costs eased and Digital Services "
        "utilisation rose. Q2 FY24 was affected by a planned two-week shutdown at Vapi Unit I.",
        F(
            "vmr.q.fy24.shutdown",
            ["Q2 FY24", "planned two-week shutdown at Vapi Unit I"],
            "Q2 FY24 was affected by a two-week planned shutdown at Vapi Unit I",
            "text",
        ),
    )


def p20_q_fy23(pg) -> None:
    q4 = V.quarters["FY23"]["Q4"]
    _quarter_page(
        pg,
        "FY23",
        "Quarterly performance: FY23 (comparative year)",
        f"FY23 started slowly as raw material costs peaked in the first quarter. The fourth quarter of FY23 was the "
        f"best, with revenue of ₹ {n0(q4[0])} crore and an EBITDA margin of {V.q_margin('FY23', 'Q4')} (₹ crore).",
        "fy23",
    )
    pg.para(
        "Q3 FY23 included the first full quarter of the Hosur polymer line, whose ramp-up costs were charged to the "
        "income statement.",
        F(
            "vmr.q.fy23.hosur",
            ["Q3 FY23", "first full quarter of the Hosur polymer line"],
            "Q3 FY23 included the first full quarter of the Hosur polymer line",
            "text",
        ),
    )


# ------------------------------------------------------------------ 21-25: financial statements


def p21_pl1(pg) -> None:
    pg.title("Consolidated statement of profit and loss for the year ended 31 March 2024 (part 1)")
    pg.para("All amounts are in ₹ crore.")
    r = [
        ["Particulars", "Note", "FY24", "FY23"],
        ["Revenue from operations", "22", n0(V.revenue["FY24"]), n0(V.revenue["FY23"])],
        ["Other income<super>2</super>", "23", n0(V.other_income["FY24"]), n0(V.other_income["FY23"])],
        ["Total income", "", n0(V.total_income("FY24")), n0(V.total_income("FY23"))],
        ["Expenses", "", "", ""],
        ["Cost of materials consumed", "24", n0(V.materials["FY24"]), n0(V.materials["FY23"])],
        ["Employee benefits expense", "25", n0(V.employee["FY24"]), n0(V.employee["FY23"])],
        ["Other expenses", "26", n0(V.other_exp["FY24"]), n0(V.other_exp["FY23"])],
        [
            "Expenses before depreciation and finance costs",
            "",
            n0(V.revenue["FY24"] - V.ebitda["FY24"]),
            n0(V.revenue["FY23"] - V.ebitda["FY23"]),
        ],
    ]
    pg.table(
        r,
        [0.58, 0.1, 0.16, 0.16],
        numeric_from=2,
        bold_rows=(3, 8),
        facts={
            1: F(
                "vmr.pl.revenue",
                ["Revenue from operations", n0(V.revenue["FY24"]), n0(V.revenue["FY23"])],
                "Revenue from operations: ₹ 7,365 crore (FY24), ₹ 6,482 crore (FY23)",
                "table_cell",
            ),
            5: (
                F(
                    "vmr.pl.materials.fy24",
                    ["Cost of materials consumed", n0(V.materials["FY24"])],
                    f"Cost of materials consumed was ₹ {n0(V.materials['FY24'])} crore in FY24",
                    "table_cell",
                ),
                F(
                    "vmr.pl.materials.fy23",
                    ["Cost of materials consumed", n0(V.materials["FY23"])],
                    f"Cost of materials consumed was ₹ {n0(V.materials['FY23'])} crore in FY23",
                    "table_cell",
                ),
            ),
            6: (
                F(
                    "vmr.pl.employee.fy24",
                    ["Employee benefits expense", n0(V.employee["FY24"])],
                    f"Employee benefits expense was ₹ {n0(V.employee['FY24'])} crore in FY24",
                    "table_cell",
                ),
                F(
                    "vmr.pl.employee.fy23",
                    ["Employee benefits expense", n0(V.employee["FY23"])],
                    f"Employee benefits expense was ₹ {n0(V.employee['FY23'])} crore in FY23",
                    "table_cell",
                ),
            ),
            7: F(
                "vmr.pl.other_expenses",
                ["Other expenses", n0(V.other_exp["FY24"]), n0(V.other_exp["FY23"])],
                f"Other expenses were ₹ {n0(V.other_exp['FY24'])} crore (FY24) and ₹ {n0(V.other_exp['FY23'])} crore (FY23)",
                "table_cell",
            ),
        },
    )
    pg.note(
        "2. Other income for FY24 includes a one-time gain of ₹ 41 crore on the sale of surplus land at Taloja, "
        "Maharashtra.",
        F(
            "vmr.land_gain",
            ["one-time gain of ₹ 41 crore", "Taloja"],
            "Other income included a one-time gain of ₹ 41 crore on the sale of surplus land at Taloja",
            "number",
        ),
    )
    pg.para("The statement continues on the next page.")


def p22_pl2(pg) -> None:
    pg.title("Consolidated statement of profit and loss (part 2, continued)")
    pg.para("All amounts are in ₹ crore. Figures for the year ended 31 March 2024 (FY24) and 31 March 2023 (FY23).")
    cur24, def24, cur23, def23 = 274, 16, 212, 11
    assert cur24 + def24 == V.tax["FY24"] and cur23 + def23 == V.tax["FY23"]
    tot24 = V.revenue["FY24"] - V.ebitda["FY24"] + V.depreciation["FY24"] + V.finance["FY24"]
    tot23 = V.revenue["FY23"] - V.ebitda["FY23"] + V.depreciation["FY23"] + V.finance["FY23"]
    assert V.total_income("FY24") - tot24 == V.pbt("FY24") and V.total_income("FY23") - tot23 == V.pbt("FY23")
    r = [
        ["Particulars", "Note", "FY24", "FY23"],
        ["Depreciation and amortisation expense", "27", n0(V.depreciation["FY24"]), n0(V.depreciation["FY23"])],
        ["Finance costs", "28", n0(V.finance["FY24"]), n0(V.finance["FY23"])],
        ["Total expenses", "", n0(tot24), n0(tot23)],
        ["Profit before tax", "", n0(V.pbt("FY24")), n0(V.pbt("FY23"))],
        ["Current tax", "29", n0(cur24), n0(cur23)],
        ["Deferred tax", "29", n0(def24), n0(def23)],
        ["Profit for the year", "", n0(V.pat["FY24"]), n0(V.pat["FY23"])],
        ["Other comprehensive income, net of tax", "", "8", "(5)"],
        ["Total comprehensive income for the year", "", "879", "659"],
        ["Basic and diluted earnings per equity share (₹)", "30", V.eps("FY24"), V.eps("FY23")],
    ]
    pg.table(
        r,
        [0.58, 0.1, 0.16, 0.16],
        numeric_from=2,
        bold_rows=(4, 7, 9),
        facts={
            1: F(
                "vmr.pl.depreciation",
                ["Depreciation and amortisation expense", n0(V.depreciation["FY24"]), n0(V.depreciation["FY23"])],
                "Depreciation and amortisation was ₹ 371 crore (FY24) and ₹ 342 crore (FY23)",
                "table_cell",
            ),
            2: F(
                "vmr.pl.finance",
                ["Finance costs", n0(V.finance["FY24"])],
                "Finance costs were ₹ 96 crore in FY24",
                "table_cell",
            ),
            4: F(
                "vmr.pl.pbt",
                ["Profit before tax", n0(V.pbt("FY24")), n0(V.pbt("FY23"))],
                "Profit before tax was ₹ 1,161 crore (FY24) and ₹ 887 crore (FY23)",
                "table_cell",
            ),
            5: F(
                "vmr.pl.current_tax",
                ["Current tax", "274", "212"],
                "Current tax was ₹ 274 crore (FY24) and ₹ 212 crore (FY23)",
                "table_cell",
            ),
            7: F(
                "vmr.pl.pat",
                ["Profit for the year", n0(V.pat["FY24"]), n0(V.pat["FY23"])],
                "Profit for the year was ₹ 871 crore (FY24) and ₹ 664 crore (FY23)",
                "table_cell",
            ),
            9: F(
                "vmr.pl.tci",
                ["Total comprehensive income", "879", "659"],
                "Total comprehensive income was ₹ 879 crore (FY24) and ₹ 659 crore (FY23)",
                "table_cell",
            ),
        },
    )
    pg.para(
        "The weighted average number of equity shares outstanding was 12.5 crore in both years. The face value of "
        "each share is ₹ 10.",
        F(
            "vmr.shares",
            ["12.5 crore", "face value of each share is ₹ 10"],
            "Valmora has 12.5 crore equity shares of face value ₹ 10",
            "number",
        ),
    )


def p23_assets(pg) -> None:
    b24, b23 = V.bs["FY24"], V.bs["FY23"]

    def row(label, key, note=""):
        return [label, note, n0(b24[key]), n0(b23[key])]

    nca24 = sum(b24[k] for k in ("ppe", "cwip", "intang", "invest", "other_nca"))
    nca23 = sum(b23[k] for k in ("ppe", "cwip", "intang", "invest", "other_nca"))
    ca24 = sum(b24[k] for k in ("inv", "rec", "cash", "other_ca"))
    ca23 = sum(b23[k] for k in ("inv", "rec", "cash", "other_ca"))
    pg.title("Consolidated balance sheet as at 31 March 2024: assets")
    pg.para("All amounts are in ₹ crore. The equity and liabilities side is on the next page.")
    rows = [
        ["Particulars", "Note", "31 Mar 2024", "31 Mar 2023"],
        ["Non-current assets", "", "", ""],
        row("Property, plant and equipment", "ppe", "3"),
        row("Capital work-in-progress", "cwip", "4"),
        row("Intangible assets", "intang", "5"),
        row("Non-current investments", "invest", "6"),
        row("Other non-current assets", "other_nca", "7"),
        ["Total non-current assets", "", n0(nca24), n0(nca23)],
        ["Current assets", "", "", ""],
        row("Inventories", "inv", "8"),
        row("Trade receivables", "rec", "9"),
        row("Cash and bank balances", "cash", "10"),
        row("Other current assets", "other_ca", "11"),
        ["Total current assets", "", n0(ca24), n0(ca23)],
        ["Total assets", "", n0(V.total_assets("FY24")), n0(V.total_assets("FY23"))],
    ]
    pg.table(
        rows,
        [0.5, 0.1, 0.2, 0.2],
        numeric_from=2,
        bold_rows=(7, 13, 14),
        facts={
            2: F(
                "vmr.bs.ppe",
                ["Property, plant and equipment", n0(b24["ppe"]), n0(b23["ppe"])],
                "PPE was ₹ 3,612 crore (31 Mar 2024) and ₹ 3,240 crore (31 Mar 2023)",
                "table_cell",
            ),
            3: F(
                "vmr.bs.cwip",
                ["Capital work-in-progress", n0(b24["cwip"]), n0(b23["cwip"])],
                "CWIP was ₹ 358 crore (2024) and ₹ 412 crore (2023)",
                "table_cell",
            ),
            9: F(
                "vmr.bs.inventories",
                ["Inventories", n0(b24["inv"]), n0(b23["inv"])],
                "Inventories were ₹ 1,248 crore (2024) and ₹ 1,126 crore (2023)",
                "table_cell",
            ),
            10: F(
                "vmr.bs.receivables",
                ["Trade receivables", n0(b24["rec"]), n0(b23["rec"])],
                "Trade receivables were ₹ 1,357 crore (2024) and ₹ 1,204 crore (2023)",
                "table_cell",
            ),
            11: F(
                "vmr.bs.cash",
                ["Cash and bank balances", n0(b24["cash"]), n0(b23["cash"])],
                "Cash and bank balances were ₹ 612 crore (2024) and ₹ 486 crore (2023)",
                "table_cell",
            ),
            14: F(
                "vmr.bs.total_assets",
                ["Total assets", n0(V.total_assets("FY24")), n0(V.total_assets("FY23"))],
                "Total assets were ₹ 7,941 crore (2024) and ₹ 7,132 crore (2023)",
                "table_cell",
            ),
        },
    )
    pg.para("Property, plant and equipment is stated at cost less accumulated depreciation.")


def p24_equity(pg) -> None:
    b24, b23 = V.bs["FY24"], V.bs["FY23"]

    def row(label, key, note=""):
        return [label, note, n0(b24[key]), n0(b23[key])]

    ncl24 = b24["nc_borrow"] + b24["other_ncl"]
    ncl23 = b23["nc_borrow"] + b23["other_ncl"]
    cl24 = b24["st_borrow"] + b24["payables"] + b24["other_cl"]
    cl23 = b23["st_borrow"] + b23["payables"] + b23["other_cl"]
    pg.title("Consolidated balance sheet as at 31 March 2024: equity and liabilities")
    pg.para("All amounts are in ₹ crore.")
    rows = [
        ["Particulars", "Note", "31 Mar 2024", "31 Mar 2023"],
        ["Equity", "", "", ""],
        row("Equity share capital", "share_cap", "12"),
        row("Other equity", "other_eq", "13"),
        ["Total equity", "", n0(V.equity("FY24")), n0(V.equity("FY23"))],
        ["Non-current liabilities", "", "", ""],
        row("Borrowings", "nc_borrow", "14"),
        row("Other non-current liabilities<super>3</super>", "other_ncl", "15"),
        ["Total non-current liabilities", "", n0(ncl24), n0(ncl23)],
        ["Current liabilities", "", "", ""],
        row("Short-term borrowings", "st_borrow", "14"),
        row("Trade payables", "payables", "16"),
        row("Other current liabilities<super>4</super>", "other_cl", "17"),
        ["Total current liabilities", "", n0(cl24), n0(cl23)],
        ["Total equity and liabilities", "", n0(V.total_assets("FY24")), n0(V.total_assets("FY23"))],
    ]
    pg.table(
        rows,
        [0.5, 0.1, 0.2, 0.2],
        numeric_from=2,
        bold_rows=(4, 8, 13, 14),
        facts={
            4: F(
                "vmr.bs.total_equity",
                ["Total equity", n0(V.equity("FY24")), n0(V.equity("FY23"))],
                "Total equity was ₹ 4,290 crore (2024) and ₹ 3,561 crore (2023)",
                "table_cell",
            ),
            6: F(
                "vmr.bs.nc_borrowings",
                ["Borrowings", n0(b24["nc_borrow"]), n0(b23["nc_borrow"])],
                "Non-current borrowings were ₹ 1,052 crore (2024) and ₹ 1,206 crore (2023)",
                "table_cell",
            ),
            11: F(
                "vmr.bs.payables",
                ["Trade payables", n0(b24["payables"]), n0(b23["payables"])],
                "Trade payables were ₹ 1,160 crore (2024) and ₹ 1,044 crore (2023)",
                "table_cell",
            ),
        },
    )
    pg.note(
        "3. Other non-current liabilities comprise provisions for employee benefits and deferred tax liabilities. "
        "4. Other current liabilities for FY24 include customer advances of ₹ 211 crore.",
        F(
            "vmr.footnote.advances",
            ["customer advances of ₹ 211 crore"],
            "Other current liabilities for FY24 include customer advances of ₹ 211 crore",
            "footnote",
        ),
    )


def p25_cashflow(pg) -> None:
    c24, c23 = V.cash_flow["FY24"], V.cash_flow["FY23"]

    def neg(x: int) -> str:
        return f"({n0(-x)})" if x < 0 else n0(x)

    pg.title("Consolidated cash flow statement for the year ended 31 March 2024")
    pg.para("All amounts are in ₹ crore.")
    rows = [
        ["Particulars", "FY24", "FY23"],
        ["Net cash generated from operating activities", n0(c24[0]), n0(c23[0])],
        ["Net cash used in investing activities", neg(c24[1]), neg(c23[1])],
        ["Net cash used in financing activities", neg(c24[2]), neg(c23[2])],
        ["Net increase in cash and cash equivalents", n0(sum(c24)), n0(sum(c23))],
        ["Cash and cash equivalents at the beginning of the year", n0(V.bs["FY23"]["cash"]), "382"],
        ["Cash and cash equivalents at the end of the year", n0(V.bs["FY24"]["cash"]), n0(V.bs["FY23"]["cash"])],
        ["Memo: purchase of property, plant, equipment and intangibles", n0(V.capex["FY24"]), n0(V.capex["FY23"])],
        ["Memo: dividends paid", "150", "120"],
    ]
    pg.table(
        rows,
        [0.62, 0.19, 0.19],
        bold_rows=(4, 6),
        facts={
            1: F(
                "vmr.cf.operating",
                ["Net cash generated from operating activities", "1,284", "1,052"],
                "Operating cash flow was ₹ 1,284 crore (FY24) and ₹ 1,052 crore (FY23)",
                "table_cell",
            ),
            2: F(
                "vmr.cf.investing",
                ["Net cash used in investing activities", "(712)", "(655)"],
                "Investing cash flow was ₹ -712 crore (FY24) and ₹ -655 crore (FY23)",
                "table_cell",
            ),
            3: F(
                "vmr.cf.financing",
                ["Net cash used in financing activities", "(446)", "(293)"],
                "Financing cash flow was ₹ -446 crore (FY24) and ₹ -293 crore (FY23)",
                "table_cell",
            ),
            7: F(
                "vmr.cf.capex",
                ["purchase of property, plant, equipment and intangibles", "698", "642"],
                "Capital expenditure was ₹ 698 crore (FY24) and ₹ 642 crore (FY23)",
                "table_cell",
            ),
            8: F(
                "vmr.cf.dividends_paid",
                ["dividends paid", "150", "120"],
                "Dividends paid were ₹ 150 crore in FY24 and ₹ 120 crore in FY23",
                "table_cell",
            ),
        },
    )
    pg.para(
        "Financing outflows in FY24 comprised repayment of term loans of ₹ 214 crore, a net reduction of ₹ 77 crore "
        "in working capital borrowings, interest paid and the final dividend for FY23 of ₹ 150 crore, equal to "
        "₹ 12.00 per share.",
        F(
            "vmr.cf.dividend_paid_fy23",
            ["final dividend for FY23 of ₹ 150 crore", "₹ 12.00 per share"],
            "The FY23 final dividend of ₹ 150 crore (₹ 12.00 per share) was paid in FY24",
            "number",
        ),
    )
    pg.para(
        "Operating cash flow rose because of higher profit and better working capital. Capital expenditure was spent "
        "mainly on the Dahej expansion and on solar capacity."
    )


# ------------------------------------------------------------------ 26-27: notes


def p26_notes1(pg) -> None:
    pg.title("Notes to the consolidated financial statements (1)")
    pg.h2("Note 1: Significant accounting policies")
    pg.para(
        "Revenue from the sale of goods is recognised when control passes to the customer, normally on dispatch or "
        "delivery as per the contract terms. Revenue from fixed-price software projects is recognised over time using "
        "the percentage-of-completion method; time-and-material revenue is recognised as hours are worked.",
        F(
            "vmr.note.revenue_policy",
            ["percentage-of-completion method", "fixed-price software projects"],
            "Fixed-price software revenue is recognised using the percentage-of-completion method",
            "text",
        ),
    )
    pg.para(
        "Inventories are valued at the lower of cost and net realisable value; cost is determined on a weighted "
        "average basis. Property, plant and equipment is depreciated on a straight-line basis over useful lives "
        "specified in Schedule II, except for plant and machinery, which has a useful life of 20 years based on "
        "technical assessment.",
        F(
            "vmr.note.useful_life",
            ["plant and machinery", "useful life of 20 years"],
            "Plant and machinery has a useful life of 20 years",
            "number",
        ),
    )
    pg.h2("Note 14: Borrowings")
    rows = [
        ["Instrument", "Terms", "31 Mar 2024", "31 Mar 2023"],
        ["Secured term loans from banks", "Floating rate; repayable by FY30", "752", "906"],
        ["Non-convertible debentures", "7.85% p.a., repayable on 15 March 2027", "300", "300"],
        ["Non-current borrowings", "", "1,052", "1,206"],
        ["Working capital demand loans", "Repayable on demand, secured by current assets", "391", "423"],
        ["Commercial paper", "Matured in FY24", "-", "45"],
        ["Short-term borrowings", "", "391", "468"],
        ["Total borrowings", "", "1,443", "1,674"],
    ]
    pg.table(
        rows,
        [0.3, 0.4, 0.15, 0.15],
        numeric_from=2,
        bold_rows=(3, 6, 7),
        facts={
            1: F(
                "vmr.note.term_loans",
                ["Secured term loans from banks", "752", "906"],
                "Secured term loans were ₹ 752 crore (2024) and ₹ 906 crore (2023)",
                "table_cell",
            ),
            2: F(
                "vmr.note.ncd",
                ["Non-convertible debentures", "7.85%", "15 March 2027"],
                "The ₹ 300 crore NCDs carry 7.85% and are repayable on 15 March 2027",
                "table_cell",
            ),
            7: F(
                "vmr.note.total_borrowings",
                ["Total borrowings", "1,443", "1,674"],
                "Total borrowings were ₹ 1,443 crore (2024) and ₹ 1,674 crore (2023)",
                "table_cell",
            ),
        },
    )
    pg.para(
        "Term loans are secured by a first charge on the fixed assets of the Dahej Complex and Hosur Polymers. "
        "There were no defaults in the repayment of principal or interest during the year.",
        F(
            "vmr.note.security",
            ["first charge on the fixed assets of the Dahej Complex and Hosur Polymers"],
            "Term loans are secured by a first charge on the fixed assets of Dahej and Hosur",
            "text",
        ),
    )


def p27_notes2(pg) -> None:
    pg.title("Notes to the consolidated financial statements (2)")
    pg.h2("Note 31: Contingent liabilities")
    rows = [
        ["Matter", "31 Mar 2024", "31 Mar 2023"],
        ["Disputed excise duty and GST demands, under appeal", "64", "58"],
        ["Disputed income tax demands, under appeal", "38", "31"],
        ["Claims by customers not acknowledged as debt", "17", "17"],
    ]
    pg.table(
        rows,
        [0.64, 0.18, 0.18],
        facts={
            1: F(
                "vmr.note.contingent_gst",
                ["Disputed excise duty and GST demands", "64"],
                "Disputed excise duty and GST demands were ₹ 64 crore at 31 March 2024",
                "table_cell",
            ),
            2: F(
                "vmr.note.contingent_tax",
                ["Disputed income tax demands", "38"],
                "Disputed income tax demands were ₹ 38 crore at 31 March 2024",
                "table_cell",
            ),
        },
    )
    pg.h2("Note 33: Related party transactions")
    pg.para(
        "Valmora Digital Services Private Limited (CIN U72900GJ2012PTC070918) is a wholly owned subsidiary. Purchases "
        "of software services from it were ₹ 212 crore in FY24 (FY23: ₹ 188 crore) and are eliminated on "
        "consolidation. The remuneration of key managerial personnel, including the Managing Director, was ₹ 14.6 "
        "crore.",
        F(
            "vmr.note.subsidiary_cin",
            ["Valmora Digital Services Private Limited", "U72900GJ2012PTC070918"],
            "The subsidiary Valmora Digital Services Private Limited has CIN U72900GJ2012PTC070918",
            "identifier",
        ),
        F(
            "vmr.note.kmp_pay",
            ["key managerial personnel", "₹ 14.6 crore"],
            "Key managerial personnel remuneration was ₹ 14.6 crore",
            "number",
        ),
    )
    pg.h2("Note 34: Auditor's remuneration")
    pg.para(
        "Statutory audit fees were ₹ 1.2 crore and fees for limited review and certification were ₹ 0.4 crore, "
        "excluding taxes and reimbursement of expenses.",
        F(
            "vmr.note.audit_fees",
            ["Statutory audit fees were ₹ 1.2 crore"],
            "Statutory audit fees were ₹ 1.2 crore",
            "number",
        ),
    )
    pg.h2("Note 36: Insurance")
    pg.para(
        "The Company has taken a Directors and Officers liability insurance policy, policy number "
        "DNO/VAL/2023-24/558120, with a sum insured of ₹ 50 crore and a policy period from 1 April 2023 to 31 March "
        "2024. Property and business interruption cover is held under separate policies.",
        F(
            "vmr.note.do_policy",
            ["Directors and Officers liability insurance policy", "DNO/VAL/2023-24/558120"],
            "The D&O liability policy number is DNO/VAL/2023-24/558120",
            "identifier",
        ),
        F(
            "vmr.note.do_sum",
            ["DNO/VAL/2023-24/558120", "sum insured of ₹ 50 crore"],
            "The D&O policy has a sum insured of ₹ 50 crore",
            "number",
        ),
    )
    pg.h2("Note 37: Events after the reporting period")
    pg.para(
        "On 12 April 2024 the Board approved the incorporation of a wholly owned subsidiary, Valmora Specialties "
        "(Singapore) Pte. Ltd., to hold export marketing rights for fluorochemicals. No other event after the "
        "reporting date requires disclosure.",
        F(
            "vmr.note.subsequent",
            ["12 April 2024", "Valmora Specialties (Singapore) Pte. Ltd."],
            "On 12 April 2024 the Board approved a Singapore subsidiary, Valmora Specialties (Singapore) Pte. Ltd.",
            "text",
        ),
    )


# ------------------------------------------------------------------ 28-29: shareholders and glossary


def p28_shareholders(pg) -> None:
    pg.title("Shareholder information")
    pg.h2("Dividend")
    rows = [
        ["Financial year", "Dividend per share (₹)", "Dividend rate", "Total outflow (₹ crore)", "Payout ratio"],
        ["FY24 (recommended)", "15.00", "150%", "187.50", pct(187.5, V.pat["FY24"])],
        ["FY23 (paid in FY24)", "12.00", "120%", "150.00", pct(150, V.pat["FY23"])],
    ]
    pg.table(
        rows,
        [0.26, 0.22, 0.16, 0.22, 0.14],
        numeric_from=1,
        facts={
            1: F(
                "vmr.div.fy24",
                ["FY24 (recommended)", "15.00", "187.50"],
                "FY24 recommended dividend: ₹ 15.00 per share, total outflow ₹ 187.50 crore",
                "table_cell",
            ),
            2: F(
                "vmr.div.fy23",
                ["FY23 (paid in FY24)", "12.00", "150.00"],
                "FY23 dividend: ₹ 12.00 per share, total outflow ₹ 150.00 crore",
                "table_cell",
            ),
        },
    )
    pg.para(
        "The record date for the FY24 dividend is 16 August 2024. If approved, the dividend will be paid on or after "
        "4 September 2024 to shareholders whose names appear in the register of members on the record date.",
        F(
            "vmr.div.record_date",
            ["record date for the FY24 dividend is 16 August 2024"],
            "The record date for the FY24 dividend is 16 August 2024",
            "number",
        ),
        F(
            "vmr.div.pay_date",
            ["paid on or after 4 September 2024"],
            "The FY24 dividend will be paid on or after 4 September 2024",
            "number",
        ),
    )
    pg.h2("Shareholding pattern at 31 March 2024")
    rows = [
        ["Category", "Share of equity capital"],
        ["Promoters and promoter group", "54.8%"],
        ["Foreign portfolio investors", "17.3%"],
        ["Mutual funds and financial institutions", "14.6%"],
        ["Public and others", "13.3%"],
    ]
    pg.table(
        rows,
        [0.64, 0.36],
        facts={
            1: F(
                "vmr.holding.promoters",
                ["Promoters and promoter group", "54.8%"],
                "Promoters held 54.8% at 31 March 2024",
                "table_cell",
            ),
            2: F(
                "vmr.holding.fpi",
                ["Foreign portfolio investors", "17.3%"],
                "Foreign portfolio investors held 17.3%",
                "table_cell",
            ),
        },
    )
    pg.h2("Market data")
    pg.para(
        "The market capitalisation on 31 March 2024 was ₹ 22,640 crore. The share price on the National Stock "
        "Exchange ranged between a low of ₹ 1,226 and a high of ₹ 1,912 during FY24; the high was recorded on 14 "
        "February 2024. The Company had 1,18,426 shareholders on 31 March 2024.",
        F(
            "vmr.market.cap",
            ["market capitalisation on 31 March 2024 was ₹ 22,640 crore"],
            "Market capitalisation at 31 March 2024 was ₹ 22,640 crore",
            "number",
        ),
        F(
            "vmr.market.range",
            ["low of ₹ 1,226", "high of ₹ 1,912"],
            "FY24 share price range: low ₹ 1,226, high ₹ 1,912",
            "number",
        ),
        F(
            "vmr.market.holders",
            ["1,18,426 shareholders"],
            "Valmora had 1,18,426 shareholders on 31 March 2024",
            "number",
        ),
    )


def p29_glossary(pg) -> None:
    pg.title("Glossary and abbreviations")
    rows = [
        ["Term", "Meaning in this report"],
        [
            "EBITDA",
            "Earnings before interest, tax, depreciation and amortisation: revenue from operations less operating expenses, excluding other income.",
        ],
        ["PAT", "Profit after tax: profit for the year attributable to shareholders."],
        [
            "ROCE",
            "Return on capital employed: EBIT divided by closing capital employed (total equity plus borrowings).",
        ],
        ["bps", "Basis points: one hundredth of a percentage point."],
        ["tpa", "Tonnes per annum, the unit of installed plant capacity."],
        ["LTIFR", "Lost-time injury frequency rate: lost-time injuries per million man-hours worked."],
        [
            "OTIF",
            "On-time, in-full: the share of customer orders delivered on the promised date and in the promised quantity.",
        ],
        [
            "KMP",
            "Key managerial personnel: the Managing Director, the Chief Financial Officer and the Company Secretary.",
        ],
        ["ESOP", "Employee stock option plan: a scheme granting employees options to buy shares at a fixed price."],
        ["NCD", "Non-convertible debenture: a debt instrument that cannot be converted into shares."],
        ["CSR", "Corporate social responsibility: spending on community programmes as required by law."],
        [
            "Scope 1 and 2",
            "Direct emissions from our plants and indirect emissions from purchased electricity and steam.",
        ],
        ["PRISM", "Valmora's proprietary production planning platform sold by Digital Services."],
        ["Project Sudarshan", "Valmora's cost and complexity reduction programme."],
    ]
    keys = {
        1: (
            "vmr.gl.ebitda",
            "EBITDA",
            ["EBITDA", "Earnings before interest, tax, depreciation and amortisation"],
            "EBITDA means earnings before interest, tax, depreciation and amortisation",
        ),
        3: (
            "vmr.gl.roce",
            "ROCE",
            ["ROCE", "Return on capital employed", "closing capital employed"],
            "ROCE is EBIT divided by closing capital employed (total equity plus borrowings)",
        ),
        4: (
            "vmr.gl.bps",
            "bps",
            ["bps", "one hundredth of a percentage point"],
            "A basis point is one hundredth of a percentage point",
        ),
        6: (
            "vmr.gl.ltifr",
            "LTIFR",
            ["LTIFR", "lost-time injuries per million man-hours"],
            "LTIFR is lost-time injuries per million man-hours worked",
        ),
        7: ("vmr.gl.otif", "OTIF", ["OTIF", "On-time, in-full"], "OTIF means on-time, in-full delivery"),
        8: (
            "vmr.gl.kmp",
            "KMP",
            ["KMP", "Key managerial personnel", "Chief Financial Officer and the Company Secretary"],
            "KMP are the Managing Director, the CFO and the Company Secretary",
        ),
        10: (
            "vmr.gl.ncd",
            "NCD",
            ["NCD", "cannot be converted into shares"],
            "An NCD is a debenture that cannot be converted into shares",
        ),
        12: (
            "vmr.gl.scope",
            "Scope 1 and 2",
            ["Scope 1 and 2", "indirect emissions from purchased electricity"],
            "Scope 1 and 2 are direct emissions and indirect emissions from purchased electricity and steam",
        ),
    }
    facts = {r: F(fid, ev, st, "definition") for r, (fid, _t, ev, st) in keys.items()}
    pg.table(rows, [0.2, 0.8], numeric_from=9, facts=facts)
    pg.h2("Notice of the 30th Annual General Meeting")
    pg.para(
        "The 30th Annual General Meeting of Valmora Industries Limited will be held on Tuesday, 27 August 2024 at "
        "11:00 a.m. IST through video conferencing. Remote e-voting opens on 23 August 2024 and closes on 26 August "
        "2024.",
        F(
            "vmr.agm.date",
            ["Tuesday, 27 August 2024", "11:00 a.m. IST"],
            "The 30th AGM is on Tuesday, 27 August 2024 at 11:00 a.m. IST by video conferencing",
            "date",
        ),
        F(
            "vmr.agm.evoting",
            ["Remote e-voting opens on 23 August 2024", "closes on 26 August 2024"],
            "Remote e-voting runs from 23 August 2024 to 26 August 2024",
            "date",
        ),
    )
