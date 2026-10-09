# ruff: noqa: E501  (document text and evidence strings are data)
"""Document 3: Valmora's Travel and Expense Policy (DOCX, 9 nominal pages).

Numbered clauses (4.2.1, 8.1.2), eligibility by grade, limits by city tier, exceptions, and a revision history with
older values (wrong-version traps). Docling reads no page numbers from DOCX, so facts are located by heading.
"""

from __future__ import annotations

from pathlib import Path

from app.evals.manifest import DocumentEntry

from .docxkit import DocxDoc
from .registry import F, Registry

NAME = "valmora_travel_expense_policy.docx"
TITLE = "Valmora Industries Limited: Travel and Expense Policy"
POLICY_NO = "HR-POL-014"
TRAVEL_INSURANCE_POLICY = "TRV/2024/0097215"


def build(reg: Registry, out_dir: Path) -> DocumentEntry:
    entry = DocumentEntry(
        name=NAME, title=TITLE, format="docx", language="en", pages=None, company="Valmora Industries Limited"
    )
    reg.add_document(entry)
    d = DocxDoc(reg, NAME, title=TITLE)

    d.title(f"Valmora Industries Limited: Travel and Expense Policy ({POLICY_NO}, version 3.2)")
    d.para(
        f"Policy number {POLICY_NO}. Version 3.2, effective 1 April 2024. Policy owner: the Chief Human Resources "
        "Officer. Next review: 1 April 2025.",
        F("trv.policy_no", [POLICY_NO], f"The Travel and Expense Policy is numbered {POLICY_NO}", "identifier"),
        F(
            "trv.effective",
            ["Version 3.2", "effective 1 April 2024"],
            "Version 3.2 of the Travel and Expense Policy is effective from 1 April 2024",
            "date",
        ),
        F(
            "trv.owner",
            ["Policy owner", "Chief Human Resources Officer"],
            "The policy owner is the Chief Human Resources Officer",
            "text",
        ),
    )

    # ---------------------------------------------------------------- 1, 2
    d.h1("1. Purpose and Scope")
    d.para(
        "1.1 This policy sets out how employees of Valmora Industries Limited plan, approve, book and claim the cost of business travel and related expenses."
    )
    d.para(
        "1.2 The policy applies to all permanent employees in grades L1 to L7 and to directors travelling on company business.",
        F(
            "trv.scope",
            ["1.2", "grades L1 to L7"],
            "Clause 1.2: the policy applies to permanent employees in grades L1 to L7 and to directors",
            "clause",
        ),
    )
    d.para(
        "1.3 Consultants and contractors are not covered by this policy; their travel is governed by the terms of their contracts.",
        F(
            "trv.contractors",
            ["1.3", "Consultants and contractors are not covered"],
            "Clause 1.3: consultants and contractors are not covered by the policy",
            "clause",
        ),
    )
    d.para(
        "1.4 Interns are treated as grade L1 for travel purposes, but only when they travel on an assignment away from their base office.",
        F(
            "trv.interns",
            ["1.4", "Interns are treated as grade L1"],
            "Clause 1.4: interns are treated as grade L1 when travelling on assignment away from their base office",
            "clause",
        ),
    )
    d.h1("2. Definitions")
    d.para(
        "2.1 Tier-1 cities are Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Kolkata and Pune.",
        F(
            "trv.tier1",
            ["2.1", "Tier-1 cities are Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Kolkata and Pune"],
            "Clause 2.1: Tier-1 cities are Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Kolkata and Pune",
            "clause",
        ),
    )
    d.para("2.2 Tier-2 cities are all other cities and towns in India.")
    d.para(
        "2.3 A long-haul flight is a flight whose scheduled duration is more than 6 hours.",
        F(
            "trv.longhaul",
            ["2.3", "long-haul flight", "more than 6 hours"],
            "Clause 2.3: a long-haul flight is one scheduled for more than 6 hours",
            "clause",
        ),
    )
    d.para(
        "2.4 The Line Manager is the employee's direct reporting manager; the Department Head is the head of the function the employee belongs to."
    )
    d.page_break()

    # ---------------------------------------------------------------- 3
    d.h1("3. Approvals and Booking")
    d.para(
        "3.1 All business travel must be approved in the Travel Desk system before any booking is made. Travel without prior approval is reimbursed only with the written approval of the Chief Financial Officer.",
        F(
            "trv.prior_approval",
            ["3.1", "approved in the Travel Desk system before any booking"],
            "Clause 3.1: business travel must be approved in the Travel Desk system before booking",
            "clause",
        ),
    )
    d.h2("3.2 Booking window")
    d.para(
        "3.2.1 Domestic flights must be booked at least 7 days before departure.",
        F(
            "trv.book_domestic",
            ["3.2.1", "at least 7 days before departure"],
            "Clause 3.2.1: domestic flights must be booked at least 7 days before departure",
            "clause",
        ),
    )
    d.para(
        "3.2.2 International flights must be booked at least 21 days before departure.",
        F(
            "trv.book_international",
            ["3.2.2", "at least 21 days before departure"],
            "Clause 3.2.2: international flights must be booked at least 21 days before departure",
            "clause",
        ),
    )
    d.para(
        "3.2.3 Bookings made less than 48 hours before departure need the approval of the Department Head.",
        F(
            "trv.last_minute",
            ["3.2.3", "less than 48 hours before departure", "Department Head"],
            "Clause 3.2.3: bookings less than 48 hours before departure need Department Head approval",
            "clause",
        ),
    )
    d.h2("3.3 Approval matrix")
    d.table(
        [
            ["Estimated trip cost", "Approver"],
            ["Up to ₹ 25,000", "Line Manager"],
            ["₹ 25,001 to ₹ 1,00,000", "Department Head"],
            ["Above ₹ 1,00,000", "Department Head and Business Unit Finance Controller"],
            ["Any international trip", "Department Head and Chief Human Resources Officer"],
        ],
        facts={
            1: F(
                "trv.matrix.low",
                ["Up to ₹ 25,000", "Line Manager"],
                "Trips costing up to ₹ 25,000 are approved by the Line Manager",
                "table_cell",
            ),
            3: F(
                "trv.matrix.high",
                ["Above ₹ 1,00,000", "Business Unit Finance Controller"],
                "Trips above ₹ 1,00,000 need the Department Head and the Business Unit Finance Controller",
                "table_cell",
            ),
            4: F(
                "trv.matrix.international",
                ["Any international trip", "Chief Human Resources Officer"],
                "Any international trip needs the Department Head and the Chief Human Resources Officer",
                "table_cell",
            ),
        },
    )
    d.page_break()

    # ---------------------------------------------------------------- 4
    d.h1("4. Air, Rail and Road Travel")
    d.h2("4.1 Eligibility by grade")
    d.table(
        [
            ["Grade", "Domestic air", "International air", "Rail", "Local conveyance"],
            ["L1 to L2", "Economy", "Economy", "AC 3-tier", "Auto-rickshaw or cab with receipt"],
            ["L3 to L4", "Economy", "Economy", "AC 2-tier", "Cab with receipt"],
            ["L5", "Economy", "Business class on long-haul flights", "AC first class", "Cab with receipt"],
            [
                "L6 to L7",
                "Premium economy or business class",
                "Business class on long-haul flights",
                "AC first class",
                "Company-arranged car",
            ],
        ],
        facts={
            1: F(
                "trv.elig.l1", ["L1 to L2", "AC 3-tier"], "Grades L1 to L2 travel by AC 3-tier on trains", "table_cell"
            ),
            3: F(
                "trv.elig.l5",
                ["L5", "Business class on long-haul flights", "AC first class"],
                "Grade L5 may fly business class on long-haul flights and travels AC first class by train",
                "table_cell",
            ),
        },
    )
    d.h2("4.2 Air travel")
    d.para(
        "4.2.1 Employees in grades L1 to L5 travel economy class on domestic flights.",
        F(
            "trv.air_domestic",
            ["4.2.1", "grades L1 to L5 travel economy class on domestic flights"],
            "Clause 4.2.1: grades L1 to L5 fly economy on domestic flights",
            "clause",
        ),
    )
    d.para(
        "4.2.2 On long-haul flights, employees in grades L5 to L7 may travel business class; all other grades travel economy class.",
        F(
            "trv.air_business",
            ["4.2.2", "grades L5 to L7 may travel business class"],
            "Clause 4.2.2: grades L5 to L7 may fly business class on long-haul flights",
            "clause",
        ),
    )
    d.para(
        "4.2.3 Fares must be the lowest logical fare within two hours of the preferred departure time. Refundable fares are not permitted unless the trip is likely to change."
    )
    d.para(
        "4.2.4 Exception: in a medical emergency affecting the traveller or an immediate family member, business class may be booked on any flight with the approval of the Chief Human Resources Officer.",
        F(
            "trv.air_medical",
            ["4.2.4", "medical emergency", "approval of the Chief Human Resources Officer"],
            "Clause 4.2.4: in a medical emergency business class is allowed with the CHRO's approval",
            "clause",
        ),
    )
    d.para("4.2.5 Airline loyalty points earned on business travel belong to the employee.")
    d.page_break()
    d.h2("4.3 Rail travel")
    d.para(
        "4.3.1 Rail travel is by the class shown in the eligibility table. Tatkal and premium-tatkal quotas may be used only when approved in advance."
    )
    d.h2("4.4 Own vehicle")
    d.para(
        "4.4.1 When an employee uses a personal vehicle for business travel, mileage is reimbursed at ₹ 11 per km for cars and ₹ 4.50 per km for two-wheelers.",
        F(
            "trv.mileage",
            ["4.4.1", "₹ 11 per km for cars", "₹ 4.50 per km for two-wheelers"],
            "Clause 4.4.1: mileage is ₹ 11 per km for cars and ₹ 4.50 per km for two-wheelers",
            "clause",
        ),
    )
    d.para("4.4.2 Tolls and parking are reimbursed on receipts. Fines are never reimbursed.")
    d.h2("4.5 Taxis and ride-hailing")
    d.para(
        "4.5.1 Cabs may be used for airport transfers and for travel after 9:00 p.m. when public transport is not available. Ride-hailing receipts must be attached to the claim.",
        F(
            "trv.cabs",
            ["4.5.1", "airport transfers", "after 9:00 p.m."],
            "Clause 4.5.1: cabs may be used for airport transfers and after 9:00 p.m. when public transport is unavailable",
            "clause",
        ),
    )
    d.page_break()

    # ---------------------------------------------------------------- 5
    d.h1("5. Accommodation")
    d.h2("5.1 Hotel limits per night")
    d.table(
        [
            ["Grade", "Tier-1 city (₹ per night)", "Tier-2 city (₹ per night)"],
            ["L1 to L2", "5,500", "3,800"],
            ["L3 to L4", "7,500", "5,200"],
            ["L5 to L7", "11,000", "7,500"],
        ],
        facts={
            1: F(
                "trv.hotel.l1",
                ["L1 to L2", "5,500", "3,800"],
                "Hotel limit for L1 to L2 is ₹ 5,500 per night in Tier-1 cities and ₹ 3,800 in Tier-2 cities",
                "table_cell",
            ),
            2: F(
                "trv.hotel.l3",
                ["L3 to L4", "7,500", "5,200"],
                "Hotel limit for L3 to L4 is ₹ 7,500 per night in Tier-1 cities and ₹ 5,200 in Tier-2 cities",
                "table_cell",
            ),
            3: F(
                "trv.hotel.l5",
                ["L5 to L7", "11,000", "7,500"],
                "Hotel limit for L5 to L7 is ₹ 11,000 per night in Tier-1 cities and ₹ 7,500 in Tier-2 cities",
                "table_cell",
            ),
        },
    )
    d.para("5.1.1 The limits include taxes. Breakfast included in the room rate is not claimed separately.")
    d.h2("5.2 Booking")
    d.para(
        "5.2.1 Hotels are booked through the Travel Desk from the approved hotel list. Employees may book directly only when no approved hotel is available, and must attach the Travel Desk's no-availability note."
    )
    d.h2("5.3 Exceptions to the hotel limits")
    d.para(
        "5.3.1 During conferences and trade fairs, an employee may stay in the event's official hotel at up to 20 per cent above the limit, with the approval of the Department Head.",
        F(
            "trv.hotel_conference",
            ["5.3.1", "up to 20 per cent above the limit", "Department Head"],
            "Clause 5.3.1: at conferences the hotel limit may be exceeded by up to 20% with Department Head approval",
            "clause",
        ),
    )
    d.para(
        "5.3.2 Where no hotel within the limit is available, the employee may exceed the limit by up to 10 per cent with the Line Manager's approval obtained before check-in.",
        F(
            "trv.hotel_noavail",
            ["5.3.2", "up to 10 per cent", "Line Manager's approval"],
            "Clause 5.3.2: if no hotel within the limit is available the limit may be exceeded by up to 10% with the Line Manager's approval before check-in",
            "clause",
        ),
    )
    d.h2("5.4 Stay with relatives or friends")
    d.para(
        "5.4.1 An employee who stays with relatives or friends may claim a lump sum of ₹ 1,000 per night without receipts.",
        F(
            "trv.relatives",
            ["5.4.1", "lump sum of ₹ 1,000 per night"],
            "Clause 5.4.1: staying with relatives or friends earns a lump sum of ₹ 1,000 per night without receipts",
            "clause",
        ),
    )
    d.h2("5.5 Serviced apartments")
    d.para(
        "5.5.1 For assignments longer than 14 consecutive nights, a serviced apartment may be booked at a monthly rent of up to ₹ 85,000 in Tier-1 cities.",
        F(
            "trv.apartment",
            ["5.5.1", "longer than 14 consecutive nights", "₹ 85,000"],
            "Clause 5.5.1: a serviced apartment is allowed for assignments longer than 14 consecutive nights, up to ₹ 85,000 a month in Tier-1 cities",
            "clause",
        ),
    )
    d.page_break()

    # ---------------------------------------------------------------- 6
    d.h1("6. Meals and Daily Allowance")
    d.h2("6.1 Domestic per diem")
    d.para(
        "6.1.1 For overnight trips within India, a per diem of ₹ 1,800 per day is paid to employees in grades L1 to L4 and ₹ 2,400 per day to employees in grades L5 to L7.",
        F(
            "trv.perdiem.l1",
            ["6.1.1", "₹ 1,800 per day", "grades L1 to L4"],
            "Clause 6.1.1: the domestic per diem is ₹ 1,800 per day for grades L1 to L4",
            "clause",
        ),
        F(
            "trv.perdiem.l5",
            ["6.1.1", "₹ 2,400 per day", "grades L5 to L7"],
            "Clause 6.1.1: the domestic per diem is ₹ 2,400 per day for grades L5 to L7",
            "clause",
        ),
    )
    d.para(
        "6.1.2 The per diem covers meals and incidental expenses. It is not paid for days on which the host provides all meals.",
        F(
            "trv.perdiem.covers",
            ["6.1.2", "covers meals and incidental expenses"],
            "Clause 6.1.2: the per diem covers meals and incidental expenses",
            "clause",
        ),
    )
    d.para(
        "6.1.3 For day trips of more than 8 hours, half of the per diem is paid.",
        F(
            "trv.perdiem.daytrip",
            ["6.1.3", "more than 8 hours", "half of the per diem"],
            "Clause 6.1.3: day trips of more than 8 hours earn half of the per diem",
            "clause",
        ),
    )
    d.h2("6.2 International per diem")
    d.table(
        [
            ["Region", "Per diem per day"],
            ["North America", "USD 85"],
            ["Europe", "EUR 70"],
            ["Asia-Pacific", "USD 60"],
            ["Middle East and Africa", "USD 65"],
        ],
        facts={
            1: F(
                "trv.intl.na",
                ["North America", "USD 85"],
                "The international per diem for North America is USD 85 per day",
                "table_cell",
            ),
            2: F(
                "trv.intl.europe",
                ["Europe", "EUR 70"],
                "The international per diem for Europe is EUR 70 per day",
                "table_cell",
            ),
            3: F(
                "trv.intl.apac",
                ["Asia-Pacific", "USD 60"],
                "The international per diem for Asia-Pacific is USD 60 per day",
                "table_cell",
            ),
        },
    )
    d.h2("6.3 Meals on actuals")
    d.para(
        "6.3.1 An employee who does not claim the per diem may claim meals on actuals with receipts.",
        F(
            "trv.actuals",
            ["6.3.1", "meals on actuals with receipts"],
            "Clause 6.3.1: meals may be claimed on actuals with receipts instead of the per diem",
            "clause",
        ),
    )
    d.para(
        "6.3.2 Alcohol is never reimbursable, including when it forms part of a client meal.",
        F(
            "trv.alcohol",
            ["6.3.2", "Alcohol is never reimbursable"],
            "Clause 6.3.2: alcohol is never reimbursable",
            "clause",
        ),
    )
    d.h2("6.4 Client entertainment")
    d.para(
        "6.4.1 Client entertainment is reimbursed up to ₹ 2,500 per head. Events for more than 8 guests need the prior approval of the Department Head.",
        F(
            "trv.entertainment",
            ["6.4.1", "₹ 2,500 per head", "more than 8 guests"],
            "Clause 6.4.1: client entertainment is reimbursed up to ₹ 2,500 per head; more than 8 guests need Department Head approval",
            "clause",
        ),
    )
    d.page_break()

    # ---------------------------------------------------------------- 7
    d.h1("7. International Travel")
    d.h2("7.1 Eligibility")
    d.para(
        "7.1.1 An employee must have completed at least 6 months of service to be eligible for international business travel.",
        F(
            "trv.intl_service",
            ["7.1.1", "at least 6 months of service"],
            "Clause 7.1.1: international travel needs at least 6 months of service",
            "clause",
        ),
    )
    d.para(
        "7.1.2 The employee's passport must be valid for at least 6 months beyond the planned return date.",
        F(
            "trv.passport",
            ["7.1.2", "passport must be valid for at least 6 months"],
            "Clause 7.1.2: the passport must be valid for at least 6 months beyond the return date",
            "clause",
        ),
    )
    d.h2("7.2 Visa and foreign exchange")
    d.para("7.2.1 Visa fees and related service charges are paid by the company.")
    d.para(
        "7.2.2 A foreign-exchange advance of up to 80 per cent of the estimated per diem may be drawn through the Finance department.",
        F(
            "trv.forex",
            ["7.2.2", "up to 80 per cent of the estimated per diem"],
            "Clause 7.2.2: a forex advance of up to 80% of the estimated per diem may be drawn",
            "clause",
        ),
    )
    d.h2("7.3 Travel insurance")
    d.para(
        f"7.3.1 Travel insurance is mandatory for all international trips and is provided under the company's group travel policy, policy number {TRAVEL_INSURANCE_POLICY}.",
        F(
            "trv.insurance_policy",
            ["7.3.1", TRAVEL_INSURANCE_POLICY],
            f"Clause 7.3.1: the group travel insurance policy number is {TRAVEL_INSURANCE_POLICY}",
            "identifier",
        ),
    )
    d.para(
        "7.3.2 In an emergency abroad, call the 24x7 assistance helpline on +91 22 5550 0198 before incurring medical expenses.",
        F(
            "trv.assistance",
            ["7.3.2", "+91 22 5550 0198"],
            "Clause 7.3.2: the 24x7 emergency assistance helpline abroad is +91 22 5550 0198",
            "identifier",
        ),
    )
    d.page_break()

    # ---------------------------------------------------------------- 8
    d.h1("8. Expense Claims and Reimbursement")
    d.h2("8.1 Submission")
    d.para("8.1.1 Claims are submitted in the Expense portal with the approved Travel Desk request number.")
    d.para(
        "8.1.2 Claims must be submitted within 15 days of the end of the trip.",
        F(
            "trv.claim_15",
            ["8.1.2", "within 15 days"],
            "Clause 8.1.2: claims must be submitted within 15 days of the end of the trip",
            "clause",
        ),
    )
    d.para(
        "8.1.3 Claims submitted more than 45 days after the end of the trip are not reimbursed unless the Chief Financial Officer approves an exception.",
        F(
            "trv.claim_45",
            ["8.1.3", "more than 45 days", "Chief Financial Officer approves an exception"],
            "Clause 8.1.3: claims later than 45 days are not reimbursed unless the CFO approves an exception",
            "clause",
        ),
    )
    d.h2("8.2 Receipts")
    d.para(
        "8.2.1 Receipts are mandatory for any single expense above ₹ 500.",
        F(
            "trv.receipts",
            ["8.2.1", "any single expense above ₹ 500"],
            "Clause 8.2.1: receipts are mandatory for any single expense above ₹ 500",
            "clause",
        ),
    )
    d.para(
        "8.2.2 If a receipt is lost, a self-declaration is accepted for amounts up to ₹ 1,500 per claim.",
        F(
            "trv.lost_receipt",
            ["8.2.2", "self-declaration", "₹ 1,500 per claim"],
            "Clause 8.2.2: a self-declaration is accepted for lost receipts up to ₹ 1,500 per claim",
            "clause",
        ),
    )
    d.h2("8.3 Reimbursement")
    d.para(
        "8.3.1 Approved claims are reimbursed within 10 working days of approval.",
        F(
            "trv.reimburse_days",
            ["8.3.1", "within 10 working days of approval"],
            "Clause 8.3.1: approved claims are reimbursed within 10 working days of approval",
            "clause",
        ),
    )
    d.h2("8.4 Advances")
    d.para(
        "8.4.1 An advance of up to 75 per cent of the estimated trip cost may be requested.",
        F(
            "trv.advance",
            ["8.4.1", "up to 75 per cent of the estimated trip cost"],
            "Clause 8.4.1: an advance of up to 75% of the estimated trip cost may be requested",
            "clause",
        ),
    )
    d.para(
        "8.4.2 The advance must be settled within 7 days of the employee's return; unsettled advances are recovered from the next salary.",
        F(
            "trv.advance_settle",
            ["8.4.2", "settled within 7 days"],
            "Clause 8.4.2: advances must be settled within 7 days of return",
            "clause",
        ),
    )
    d.page_break()

    # ---------------------------------------------------------------- 9, annex
    d.h1("9. Exceptions, Compliance and Contacts")
    d.para(
        "9.1 Any exception to this policy requires the prior written approval of the Chief Financial Officer, except where a clause names another approver.",
        F(
            "trv.exceptions",
            ["9.1", "prior written approval of the Chief Financial Officer"],
            "Clause 9.1: exceptions to the policy need the prior written approval of the CFO unless a clause names another approver",
            "clause",
        ),
    )
    d.para(
        "9.2 False or inflated claims are a violation of the Code of Conduct and may lead to disciplinary action and recovery of the amount.",
        F(
            "trv.misreporting",
            ["9.2", "False or inflated claims", "disciplinary action"],
            "Clause 9.2: false or inflated claims lead to disciplinary action and recovery",
            "clause",
        ),
    )
    d.para(
        "9.3 Questions about this policy can be sent to the Travel Desk on 1800 266 0145 or travel.desk@valmora.example, from 9:00 a.m. to 6:00 p.m., Monday to Saturday.",
        F(
            "trv.helpdesk",
            ["9.3", "1800 266 0145", "travel.desk@valmora.example"],
            "Clause 9.3: the Travel Desk helpline is 1800 266 0145, travel.desk@valmora.example, 9:00 a.m. to 6:00 p.m. Monday to Saturday",
            "identifier",
        ),
    )
    d.h1("Annex A: Revision history")
    d.table(
        [
            ["Version", "Effective date", "Change"],
            [
                "3.2",
                "1 April 2024",
                "Hotel limits raised by 10 per cent; receipts threshold lowered from ₹ 1,000 to ₹ 500.",
            ],
            ["3.1", "1 April 2023", "Domestic per diem raised from ₹ 1,500 to ₹ 1,800 per day for grades L1 to L4."],
            ["3.0", "1 April 2022", "Policy restructured; Tier-1 and Tier-2 city definitions introduced."],
        ],
        facts={
            1: F(
                "trv.rev.v32",
                ["3.2", "receipts threshold lowered from ₹ 1,000 to ₹ 500"],
                "Version 3.2 lowered the receipts threshold from ₹ 1,000 to ₹ 500",
                "table_cell",
            ),
            2: F(
                "trv.rev.v31",
                ["3.1", "Domestic per diem raised from ₹ 1,500 to ₹ 1,800"],
                "Version 3.1 raised the domestic per diem from ₹ 1,500 to ₹ 1,800 per day",
                "table_cell",
            ),
        },
    )
    assert d.nominal_pages == 9, d.nominal_pages
    d.save(out_dir / NAME)
    return entry
