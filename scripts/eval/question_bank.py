# ruff: noqa: E501
# fmt: off
"""The authored retrieval questions (compact form). ``build_questions.py`` turns them into
``evals/retrieval/questions.jsonl``, resolving fact ids to documents and pages through the corpus manifest.

Each entry: ``A(category, language, question, facts, answers, ...)`` for an answerable question, ``U(language,
question, subtype, ...)`` for one the documents do not answer.

  facts    list; each element is one piece of required evidence: a fact id, or a tuple of fact ids that are
           alternatives for the same evidence (every page of any of them counts). Several elements = all required.
  answers  strings the answer must contain ("a|b" = either spelling); numbers as written in the documents.
  en       the English query the router would produce (required for Hindi and Hinglish questions).
  clean    asr_noise only: the question as the user actually said it.
"""

from __future__ import annotations

Entry = dict

QUESTIONS: list[Entry] = []


def A(
    category: str,
    language: str,
    question: str,
    facts: list,
    answers: list[str],
    *,
    en: str | None = None,
    clean: str | None = None,
    note: str | None = None,
) -> None:
    QUESTIONS.append(
        dict(
            category=category,
            language=language,
            question=question,
            facts=facts,
            answers=answers,
            en=en,
            clean=clean,
            note=note,
            abstain=False,
        )
    )


def U(language: str, question: str, subtype: str, *, en: str | None = None, note: str | None = None) -> None:
    QUESTIONS.append(
        dict(
            category="unanswerable",
            language=language,
            question=question,
            facts=[],
            answers=[],
            en=en,
            clean=None,
            note=note,
            subtype=subtype,
            abstain=True,
        )
    )


# =========================================================================== exact facts and numbers (English)
A("exact_fact", "en", "What dividend per share did the Board recommend for FY24?", [("vmr.dividend_rec", "vmr.dps.fy24")], ["15.00"])
A("exact_fact", "en", "How much capital expenditure has the Board approved for FY25 and FY26?", ["vmr.capex_plan"], ["1,100"])
A("exact_fact", "en", "How many tonnes per annum of capacity did the Dahej fluorochemicals expansion add?", ["vmr.dahej"], ["60,000"])
A("exact_fact", "en", "How much did Project Sudarshan save in FY24?", ["vmr.sudarshan"], ["112"])
A("exact_fact", "en", "How many new enterprise clients did Digital Services onboard in FY24?", ["vmr.digital_clients"], ["14"])
A("exact_fact", "en", "What share of Valmora's FY24 revenue came from exports?", ["vmr.exports"], ["37"])
A("exact_fact", "en", "What was Valmora's net debt at 31 March 2024?", ["vmr.mda.net_debt"], ["831"])
A("exact_fact", "en", "What was the effective tax rate in FY24?", ["vmr.mda.tax_rate"], ["25.0"])
A("exact_fact", "en", "How much free cash flow did Valmora generate in FY24?", ["vmr.mda.fcf"], ["586"])
A("exact_fact", "en", "How many shareholders did Valmora have on 31 March 2024?", ["vmr.market.holders"], ["1,18,426|118,426"])
A("exact_fact", "en", "What was Valmora's market capitalisation on 31 March 2024?", ["vmr.market.cap"], ["22,640"])
A("exact_fact", "en", "When is the 30th Annual General Meeting of Valmora?", ["vmr.agm.date"], ["27 August 2024"])
A("exact_fact", "en", "What is the exercise price of the stock options granted under ESOP 2021?", ["vmr.people.esop"], ["1,150"])
A("exact_fact", "en", "How many complaints did the ethics hotline receive in FY24?", ["vmr.people.hotline"], ["41"])
A("exact_fact", "en", "What is the record date for the FY24 dividend?", ["vmr.div.record_date"], ["16 August 2024"])
A("exact_fact", "en", "What is Zephyra's FY25 revenue growth guidance?", ["zph.guide.revenue"], ["12 to 14|12-14|12% to 14%"])

# =========================================================================== paraphrases (no keyword overlap with the passage)
A("paraphrase", "en", "Out of every hundred rupees of sales in FY24, how much was left as operating profit before depreciation?", ["vmr.margin.fy24"], ["21.0"], note="passage: EBITDA margin 21.0%")
A("paraphrase", "en", "By how much did Valmora cut its interest burden in FY24?", ["vmr.mda.finance_costs"], ["18.6"], note="passage: finance costs fell 18.6%")
A("paraphrase", "en", "Where is Valmora planning to open a new site for its technology business?", ["vmr.pune_centre"], ["Pune"])
A("paraphrase", "en", "What squeezed the profitability of the polymer business last year?", ["vmr.plastics_reason"], ["feedstock"])
A("paraphrase", "en", "How much money will the Board spend on new plants and equipment over the coming two years?", ["vmr.capex_plan"], ["1,100"])
A("paraphrase", "en", "What proportion of staff left voluntarily in the latest year?", ["vmr.people.attrition.fy24"], ["11.2"])
A("paraphrase", "en", "How early must I reserve a plane ticket for a trip within India?", ["trv.book_domestic"], ["7 days|seven days"])
A("paraphrase", "en", "How much is paid each day for food and incidentals on an overnight domestic trip for a junior employee?", ["trv.perdiem.l1"], ["1,800"])
A("paraphrase", "en", "If the hotel at a trade fair costs more than the usual cap, how much extra may I spend?", ["trv.hotel_conference"], ["20 per cent|20%|20 percent"])
A("paraphrase", "en", "Can I get the company to pay for beer during a dinner with a client?", ["trv.alcohol"], ["never|not"])
A("paraphrase", "en", "What happens to an expense report I send in two months after the trip?", ["trv.claim_45"], ["45 days"])
A("paraphrase", "en", "How many shipments does the logistics firm's software platform keep an eye on each month?", ["zph.digital.zentrack"], ["1.9 million"])
A("paraphrase", "en", "How much did the logistics company owe lenders after subtracting its cash at the end of the last financial year?", [("zph.net_debt", "zph.bs.net_debt")], ["912"])
A("paraphrase", "en", "Which firm handles the cashless hospital claims for employees?", ["ghi.tpa"], ["Medisure"])
A("paraphrase", "en", "What does the company hope to achieve on its emissions by the year 2040?", ["vmr.esg.netzero"], ["net zero|zero"])
A("paraphrase", "en", "How many hours of training did the average Valmora worker receive last year?", ["vmr.people.training"], ["31"])

# =========================================================================== identifiers
A("identifier", "en", "What is the CIN of Valmora Industries?", ["vmr.cin"], ["L24119GJ1994PLC023871"])
A("identifier", "en", "What is Valmora's ISIN?", ["vmr.isin"], ["INE958R01014"])
A("identifier", "en", "What is the BSE scrip code of Valmora?", ["vmr.bse_code"], ["543871"])
A("identifier", "en", "Under which symbol is Valmora listed on the National Stock Exchange?", ["vmr.nse_symbol"], ["VALMORA"])
A("identifier", "en", "What is the CIN of Valmora Digital Services Private Limited?", ["vmr.note.subsidiary_cin"], ["U72900GJ2012PTC070918"])
A("identifier", "en", "What is the policy number of the Directors and Officers liability insurance?", ["vmr.note.do_policy"], ["DNO/VAL/2023-24/558120"])
A("identifier", "en", "What does clause 4.2.1 of the travel policy say?", ["trv.air_domestic"], ["economy"])
A("identifier", "en", "What does clause 8.1.3 say about late expense claims?", ["trv.claim_45"], ["45 days"])
A("identifier", "en", "What is the policy number of the Travel and Expense Policy?", ["trv.policy_no"], ["HR-POL-014"])
A("identifier", "en", "What is the group travel insurance policy number?", ["trv.insurance_policy"], ["TRV/2024/0097215"])
A("identifier", "en", "What is the group health insurance policy number?", ["ghi.policy_no"], ["GHI/2024/00418377"])
A("identifier", "en", "What is the firm registration number of Valmora's statutory auditors?", ["vmr.auditor"], ["109473W"])
A("identifier", "en", "What is the 24x7 emergency assistance helpline number for travellers abroad?", ["trv.assistance"], ["+91 22 5550 0198|22 5550 0198"])

# =========================================================================== wrong-year and wrong-segment traps
A("wrong_year_trap", "en", "What was Valmora's EBITDA margin in FY23?", ["vmr.margin.fy23"], ["19.8"])
A("wrong_year_trap", "en", "What was Valmora's revenue from operations in FY23?", ["vmr.rev.fy23"], ["6,482"])
A("wrong_year_trap", "en", "What was the EBITDA margin in the third quarter of FY23?", ["vmr.q.fy23.q3"], ["19.9"])
A("wrong_year_trap", "en", "What was the EBITDA margin in the third quarter of FY24?", ["vmr.q.fy24.q3"], ["21.1"])
A("wrong_year_trap", "en", "What was the capacity utilisation of Engineered Plastics in FY24?", ["vmr.seg.plastics.util"], ["78"])
A("wrong_year_trap", "en", "What was the EBITDA margin of Specialty Chemicals in FY23?", ["vmr.seg.chem.margin.fy23"], ["23.2"])
A("wrong_year_trap", "en", "What was the EBITDA margin of Digital Services in FY24?", ["vmr.seg.digital.margin.fy24"], ["22.6"])
A("wrong_year_trap", "en", "What was the EBITDA margin of Engineered Plastics in FY24?", ["vmr.seg.plastics.margin.fy24"], ["15.0"])
A("wrong_year_trap", "en", "What was Digital Services revenue in FY23?", ["vmr.seg.digital.rev.fy23"], ["1,142"])
A("wrong_year_trap", "en", "What dividend per share did Valmora pay for FY23?", ["vmr.dps.fy23"], ["12.00|12"])
A("wrong_year_trap", "en", "How many employees did Valmora have at the end of FY23?", ["vmr.employees.fy23"], ["9,310"])
A("wrong_year_trap", "en", "What was Valmora's voluntary attrition rate in FY23?", ["vmr.people.attrition.fy23"], ["13.5"])
A("wrong_year_trap", "en", "What was the EBITDA margin in the first quarter of FY24?", ["vmr.q.fy24.q1"], ["20.2"])
A("wrong_year_trap", "en", "What was the domestic per diem before version 3.1 of the travel policy?", ["trv.rev.v31"], ["1,500"])
A("wrong_year_trap", "en", "What was the receipts threshold before version 3.2 of the travel policy?", ["trv.rev.v32"], ["1,000"])

# =========================================================================== multi-hop within one document (two pages)
A("multi_hop", "en", "How many times did the Audit Committee meet in FY24, and since which year has its chair been on the Board?", ["vmr.gov.audit", "vmr.board.rao"], ["5|five", "2018"])
A("multi_hop", "en", "What dividend did the Board recommend for FY24, and what is the total dividend outflow?", [("vmr.dividend_rec",), "vmr.div.fy24"], ["15.00", "187.50"])
A("multi_hop", "en", "How much capex has the Board planned for FY25 and FY26, and how much did Valmora spend on capex in FY24?", ["vmr.capex_plan", "vmr.cf.capex"], ["1,100", "698"])
A("multi_hop", "en", "What was Valmora's net debt to EBITDA in FY24, and what was the net debt amount?", ["vmr.nd_ebitda.fy24", "vmr.mda.net_debt"], ["0.54", "831"])
A("multi_hop", "en", "How much capacity did the Dahej expansion add, and how big is the Dahej Complex in total?", ["vmr.dahej", "vmr.plant.dahej"], ["60,000", "2,10,000|210,000"])
A("multi_hop", "en", "What is the hotel limit for an L3 employee in a Tier-1 city, and which cities are Tier-1?", ["trv.hotel.l3", "trv.tier1"], ["7,500", "Mumbai"])
A("multi_hop", "en", "Can an L5 employee fly business class, and what counts as a long-haul flight?", ["trv.air_business", "trv.longhaul"], ["business", "6 hours"])
A("multi_hop", "en", "What is the domestic per diem for an L6 employee, and within how many days must the claim be submitted?", ["trv.perdiem.l5", "trv.claim_15"], ["2,400", "15 days"])
A("multi_hop", "en", "When does the group health insurance policy end, and what is the claims helpline number?", ["ghi.period", "ghi.helpline"], ["31 March 2025", "1800 419 5530"])
A("multi_hop", "en", "What was Zephyra's FY24 revenue, and what revenue growth does it guide for FY25?", ["zph.rev.fy24", "zph.guide.revenue"], ["4,986", "12 to 14|12-14|12% to 14%"])

# =========================================================================== cross-document disambiguation
A("cross_doc", "en", "What was Zephyra's EBITDA margin in FY24?", ["zph.margin.fy24"], ["12.7"])
A("cross_doc", "en", "What was Valmora's EBITDA margin in FY24?", ["vmr.margin.fy24"], ["21.0"])
A("cross_doc", "en", "What was Zephyra's profit after tax in FY24?", ["zph.pat.fy24"], ["268"])
A("cross_doc", "en", "What was Valmora's profit after tax in FY24?", ["vmr.pat.fy24"], ["871"])
A("cross_doc", "en", "How many employees does Zephyra have?", ["zph.employees"], ["14,120"])
A("cross_doc", "en", "How many employees does Valmora have?", ["vmr.employees.fy24"], ["9,842"])
A("cross_doc", "en", "What is Zephyra's ISIN?", ["zph.isin"], ["INE413T01011"])
A("cross_doc", "en", "What was Zephyra's Digital Services revenue in FY24?", ["zph.seg.digital.rev.fy24"], ["481"])
A("cross_doc", "en", "What was Valmora's Digital Services revenue in FY24?", ["vmr.seg.digital.rev.fy24"], ["1,494"])
A("cross_doc", "en", "What is Zephyra's net debt to EBITDA?", ["zph.nd_ebitda"], ["1.4"])
A("cross_doc", "en", "What is Valmora's net debt to EBITDA in FY24?", ["vmr.nd_ebitda.fy24"], ["0.54"])
A("cross_doc", "en", "What share of Zephyra's workforce are women?", ["zph.esg.women"], ["14.2"])
A("cross_doc", "en", "Which company reported an EBITDA margin of 12.7% in FY24?", ["zph.margin.fy24"], ["Zephyra"])

# =========================================================================== table cells
A("table_cell", "en", "What was property, plant and equipment at 31 March 2024?", ["vmr.bs.ppe"], ["3,612"])
A("table_cell", "en", "What was total equity at 31 March 2024?", ["vmr.bs.total_equity"], ["4,290"])
A("table_cell", "en", "What was depreciation and amortisation expense in FY24?", ["vmr.pl.depreciation"], ["371"])
A("table_cell", "en", "What was profit before tax in FY24?", ["vmr.pl.pbt"], ["1,161"])
A("table_cell", "en", "What was net cash used in investing activities in FY24?", ["vmr.cf.investing"], ["712"])
A("table_cell", "en", "How much capacity does the Hosur Polymers plant have?", ["vmr.plant.hosur"], ["1,20,000|120,000"])
A("table_cell", "en", "How many seats does the Bengaluru delivery centre have?", ["vmr.plant.bengaluru"], ["1,300"])
A("table_cell", "en", "What interest rate do the non-convertible debentures carry?", ["vmr.note.ncd"], ["7.85"])
A("table_cell", "en", "What was EBITDA in the fourth quarter of FY24?", ["vmr.q.fy24.q4"], ["422"])
A("table_cell", "en", "What share of Valmora's equity did the promoters hold at 31 March 2024?", ["vmr.holding.promoters"], ["54.8"])
A("table_cell", "en", "What is the hotel limit for an L5 to L7 employee in a Tier-2 city?", ["trv.hotel.l5"], ["7,500"])
A("table_cell", "en", "What is the international per diem for Europe?", ["trv.intl.europe"], ["70"])
A("table_cell", "en", "How many branches did Zephyra have in FY24?", ["zph.net.branches"], ["214"])

# =========================================================================== glossary and abbreviations
A("glossary", "en", "What does OTIF stand for in the annual report?", ["vmr.gl.otif"], ["on-time|on time"])
A("glossary", "en", "What does bps mean?", ["vmr.gl.bps"], ["hundredth"])
A("glossary", "en", "How does Valmora define ROCE?", [("vmr.gl.roce",)], ["EBIT", "capital employed"])
A("glossary", "en", "What is LTIFR?", ["vmr.gl.ltifr"], ["lost-time|lost time"])
A("glossary", "en", "What is an NCD?", ["vmr.gl.ncd"], ["debenture"])
A("glossary", "en", "Who counts as key managerial personnel (KMP)?", ["vmr.gl.kmp"], ["Managing Director"])
A("glossary", "en", "What are Scope 1 and 2 emissions?", ["vmr.gl.scope"], ["direct"])
A("glossary", "en", "How is EBITDA defined in the annual report?", ["vmr.gl.ebitda"], ["interest"])
A("glossary", "en", "How does the annual report define net debt?", ["vmr.footnote.net_debt"], ["borrowings", "cash"])
A("glossary", "en", "What is a long-haul flight under the travel policy?", ["trv.longhaul"], ["6 hours"])

# =========================================================================== Hindi questions on the Hindi document
A("hindi_doc", "hi", "योजना का कुल परिव्यय कितना है?", ["hin.outlay"], ["148"], en="What is the total outlay of the scheme?")
A("hindi_doc", "hi", "आवेदन की अंतिम तिथि क्या है?", ["hin.date.deadline"], ["30 नवंबर 2024"], en="What is the last date to apply?")
A("hindi_doc", "hi", "प्रशिक्षुओं को हर महीने कितना वजीफ़ा मिलेगा?", ["hin.stipend"], ["3,500"], en="How much stipend will trainees get every month?")
A("hindi_doc", "hi", "आवेदक की आयु कितनी होनी चाहिए?", ["hin.age"], ["18", "35"], en="What age must the applicant be?")
A("hindi_doc", "hi", "परिवार की सालाना आय की सीमा कितनी है?", ["hin.income"], ["2,50,000"], en="What is the annual family income limit?")
A("hindi_doc", "hi", "टूलकिट के लिए कितनी सहायता मिलती है?", ["hin.toolkit"], ["10,000"], en="How much toolkit assistance is provided?")
A("hindi_doc", "hi", "बैंक ऋण पर कितना अनुदान मिलेगा?", ["hin.subsidy"], ["35", "1,75,000"], en="How much subsidy is given on a bank loan?")
A("hindi_doc", "hi", "महिलाओं के लिए कितनी सीटें आरक्षित हैं?", ["hin.reservation"], ["33"], en="How many seats are reserved for women?")
A("hindi_doc", "hi", "सहायता के लिए हेल्पलाइन नंबर क्या है?", ["hin.helpline"], ["1800-419-0731|1800 419 0731"], en="What is the helpline number for help?")
A("hindi_doc", "hi", "इस सूचना का अधिसूचना क्रमांक क्या है?", ["hin.notice_no"], ["0317"], en="What is the notification number of this notice?")
A("hindi_doc", "hi", "कृषि ड्रोन ऑपरेटर पाठ्यक्रम कितने सप्ताह का है?", ["hin.course.drone"], ["24"], en="How many weeks is the agricultural drone operator course?")
A("hindi_doc", "hi", "सूर्यगढ़ जिले का लक्ष्य कितना है?", ["hin.district.suryagarh"], ["3,100"], en="What is the target for Suryagarh district?")
A("hindi_doc", "hi", "प्रशिक्षण कब शुरू होगा?", ["hin.date.training"], ["6 जनवरी 2025"], en="When will the training begin?")
A("hindi_doc", "hi", "स्वयं सहायता समूह के शुद्ध मुनाफे का कितना हिस्सा साझा कोष में जाएगा?", ["hin.shg_profit"], ["10 प्रतिशत|10%|10 percent|दस प्रतिशत"], en="What share of a self-help group's net profit goes to the common fund?")
A("hindi_doc", "hi", "कितने दिन लगातार अनुपस्थित रहने पर नामांकन रद्द हो जाएगा?", ["hin.absence"], ["7 दिन|सात दिन|7 days"], en="After how many days of continuous absence is enrolment cancelled?")
A("hindi_doc", "hi", "आवेदन के लिए कितना शुल्क देना होगा?", ["hin.no_fee"], ["कोई शुल्क नहीं|शुल्क नहीं|नहीं"], en="How much fee must be paid to apply?")

# =========================================================================== Hindi / Hinglish questions on the English documents
A("xlingual", "hi", "वाल्मोरा का FY24 में EBITDA मार्जिन कितना था?", ["vmr.margin.fy24"], ["21.0"], en="What was Valmora's EBITDA margin in FY24?")
A("xlingual", "hi", "वाल्मोरा का FY23 में शुद्ध लाभ कितना था?", ["vmr.pat.fy23"], ["664"], en="What was Valmora's net profit in FY23?")
A("xlingual", "hi", "वाल्मोरा का पंजीकृत कार्यालय कहाँ है?", ["vmr.reg_office"], ["वापी|Vapi"], en="Where is Valmora's registered office?")
A("xlingual", "hi", "वाल्मोरा ने FY24 के लिए प्रति शेयर कितना लाभांश प्रस्तावित किया?", [("vmr.dividend_rec", "vmr.dps.fy24")], ["15"], en="What dividend per share did Valmora propose for FY24?")
A("xlingual", "hi", "वाल्मोरा के पास कितने कर्मचारी हैं?", ["vmr.employees.fy24"], ["9,842"], en="How many employees does Valmora have?")
A("xlingual", "hi", "वाल्मोरा का शुद्ध ऋण 31 मार्च 2024 को कितना था?", ["vmr.mda.net_debt"], ["831"], en="What was Valmora's net debt on 31 March 2024?")
A("xlingual", "hi", "यात्रा नीति में घरेलू उड़ानों के लिए किस श्रेणी में यात्रा की अनुमति है?", ["trv.air_domestic"], ["इकोनॉमी|economy|Economy"], en="Which class of travel is allowed for domestic flights in the travel policy?")
A("xlingual", "hi", "टियर-1 शहर में L3 कर्मचारी के होटल की प्रति रात की सीमा कितनी है?", ["trv.hotel.l3"], ["7,500"], en="What is the per-night hotel limit for an L3 employee in a Tier-1 city?")
A("xlingual", "hi", "खर्च का दावा यात्रा खत्म होने के कितने दिनों के अंदर जमा करना होता है?", ["trv.claim_15"], ["15 दिन|15 days|15"], en="Within how many days of the end of the trip must the expense claim be submitted?")
A("xlingual", "hi", "ग्रुप हेल्थ इंश्योरेंस पॉलिसी का नंबर क्या है?", ["ghi.policy_no"], ["GHI/2024/00418377"], en="What is the group health insurance policy number?")
A("xlingual", "hi", "ज़ेफ़िरा का FY24 में EBITDA मार्जिन कितना है?", ["zph.margin.fy24"], ["12.7"], en="What is Zephyra's EBITDA margin in FY24?")
A("xlingual", "hi", "ज़ेफ़िरा ने FY25 के लिए राजस्व वृद्धि का क्या अनुमान दिया है?", ["zph.guide.revenue"], ["12", "14"], en="What revenue growth guidance has Zephyra given for FY25?")
A("xlingual", "hi", "वाल्मोरा की डहेज़ विस्तार परियोजना से कितनी क्षमता बढ़ी?", ["vmr.dahej"], ["60,000"], en="How much capacity did Valmora's Dahej expansion add?")
A("xlingual", "hinglish", "Valmora ka FY24 mein revenue kitna tha?", ["vmr.rev.fy24"], ["7,365"], en="What was Valmora's revenue from operations in FY24?")
A("xlingual", "hinglish", "FY25 aur FY26 ke liye capex plan kitne crore ka hai?", ["vmr.capex_plan"], ["1,100"], en="How many crore is the capex plan for FY25 and FY26?")
A("xlingual", "hinglish", "Valmora ki Annual General Meeting kab hai?", ["vmr.agm.date"], ["27 August 2024"], en="When is Valmora's Annual General Meeting?")
A("xlingual", "hinglish", "Dahej expansion se kitni capacity badhi?", ["vmr.dahej"], ["60,000"], en="How much capacity did the Dahej expansion add?")
A("xlingual", "hinglish", "Travel policy mein alcohol reimburse hota hai kya?", ["trv.alcohol"], ["never|not|nahi|नहीं"], en="Is alcohol reimbursed under the travel policy?")
A("xlingual", "hinglish", "L2 grade ko domestic trip pe per diem kitna milta hai?", ["trv.perdiem.l1"], ["1,800"], en="How much per diem does a grade L2 employee get on a domestic trip?")
A("xlingual", "hinglish", "Zephyra ke paas kitne warehouses hain?", ["zph.contract.warehouses"], ["112"], en="How many warehouses does Zephyra have?")
A("xlingual", "hinglish", "Group health policy mein family ke liye sum insured kitna hai?", ["ghi.sum_insured"], ["5,00,000|500,000"], en="What is the sum insured per family in the group health policy?")
A("xlingual", "hinglish", "Zephyra ka Q4 FY24 ka EBITDA margin kya raha?", ["zph.q4.margin"], ["13.1"], en="What was Zephyra's EBITDA margin in Q4 FY24?")

# =========================================================================== speech-recognition noise (Whisper-style errors)
A("asr_noise", "hi", "कंपनी का शुद्ध मुनापा FY24 में कितना था?", ["vmr.pat.fy24"], ["871"], en="What was the company's net profit in FY24?", clean="कंपनी का शुद्ध मुनाफा FY24 में कितना था?", note="मुनापा for मुनाफा (profit)")
A("asr_noise", "hi", "वाल्मोरा का एबिटडा मार्जिन FY24 में कितना था?", ["vmr.margin.fy24"], ["21.0"], en="What was Valmora's EBITDA margin in FY24?", clean="वाल्मोरा का EBITDA मार्जिन FY24 में कितना था?", note="EBITDA transliterated as एबिटडा")
A("asr_noise", "hi", "बैंक ऋण पर कितना अनुधान मिलेगा?", ["hin.subsidy"], ["35", "1,75,000"], en="How much subsidy is given on a bank loan?", clean="बैंक ऋण पर कितना अनुदान मिलेगा?", note="अनुधान for अनुदान (subsidy)")
A("asr_noise", "hi", "सोलर पैनल तकनीशियन कोर्स में कितनी सीटे हैं?", ["hin.course.solar"], ["2,400"], en="How many seats are in the solar panel technician course?", clean="सोलर पैनल तकनीशियन कोर्स में कितनी सीटें हैं?", note="सीटे for सीटें")
A("asr_noise", "hi", "वाल्मोरा का मुनापा FY23 में कितना रहा?", ["vmr.pat.fy23"], ["664"], en="What was Valmora's profit in FY23?", clean="वाल्मोरा का मुनाफा FY23 में कितना रहा?", note="मुनापा for मुनाफा")
A("asr_noise", "en", "What is Valmora's ice in?", ["vmr.isin"], ["INE958R01014"], clean="What is Valmora's ISIN?", note="ISIN heard as 'ice in'")
A("asr_noise", "en", "How much is the cap ex planned for FY25 and FY26?", ["vmr.capex_plan"], ["1,100"], clean="How much is the capex planned for FY25 and FY26?", note="capex split in two words")
A("asr_noise", "en", "How much capacity did the Dahez expansion add?", ["vmr.dahej"], ["60,000"], clean="How much capacity did the Dahej expansion add?", note="Dahej misspelt")
A("asr_noise", "en", "What was Wall Mora's net debt at the end of March 2024?", ["vmr.mda.net_debt"], ["831"], clean="What was Valmora's net debt at the end of March 2024?", note="Valmora heard as 'Wall Mora'")
A("asr_noise", "en", "What is Zefira's revenue growth guidance for FY25?", ["zph.guide.revenue"], ["12 to 14|12-14|12% to 14%"], clean="What is Zephyra's revenue growth guidance for FY25?", note="Zephyra misspelt")
A("asr_noise", "en", "How much is the per dime for L1 employees on domestic trips?", ["trv.perdiem.l1"], ["1,800"], clean="How much is the per diem for L1 employees on domestic trips?", note="per diem heard as 'per dime'")
A("asr_noise", "hinglish", "Valmora ka munafa FY24 mein kitna tha?", ["vmr.pat.fy24"], ["871"], en="What was Valmora's profit in FY24?", clean="Valmora ka munafa FY24 mein kitna tha?", note="Hinglish spelling variant 'munafa'; clean = same words")

# =========================================================================== unanswerable (plausible, not in the documents)
# near_miss_year
U("en", "What is Valmora's EBITDA margin for FY25?", "near_miss_year", note="only FY23 and FY24 exist")
U("en", "What was Valmora's revenue from operations in FY22?", "near_miss_year")
U("en", "What was Valmora's profit after tax in FY21?", "near_miss_year")
U("en", "What dividend per share is expected for FY25?", "near_miss_year")
U("en", "What was Valmora's revenue in the first quarter of FY25?", "near_miss_year")
U("en", "What was Zephyra's revenue in FY22?", "near_miss_year")
U("en", "What is Zephyra's FY25 EBITDA margin guidance?", "near_miss_year", note="the deck guides revenue growth and capex, not margin")
U("en", "What was Zephyra's profit after tax in the first quarter of FY25?", "near_miss_year")
# near_miss_segment
U("en", "What was the EBITDA margin of Valmora's Healthcare segment?", "near_miss_segment", note="no such segment")
U("en", "What was the revenue of Zephyra's Cold Chain segment?", "near_miss_segment")
U("en", "What was the capacity utilisation of Digital Services in FY22?", "near_miss_segment")
# near_miss_year
U("hi", "वाल्मोरा का FY25 का शुद्ध मुनाफा कितना रहेगा?", "near_miss_year", en="What will Valmora's net profit be in FY25?")
U("hinglish", "Valmora ka FY25 ka EBITDA margin kya hai?", "near_miss_year", en="What is Valmora's EBITDA margin for FY25?")
U("hinglish", "Zephyra ka FY22 ka revenue kitna tha?", "near_miss_year", en="What was Zephyra's revenue in FY22?")
# other_company
U("en", "What is Valmora's order book in Contract Logistics?", "other_company", note="a Zephyra metric")
U("en", "How many warehouses does Valmora operate?", "other_company")
U("en", "How many electric vehicles does Valmora run in last-mile delivery?", "other_company")
U("en", "What is Zephyra's installed chemical production capacity?", "other_company")
U("en", "How many manufacturing plants does Zephyra operate?", "other_company")
U("en", "What is Zephyra's dividend per share?", "other_company", note="the deck gives no dividend")
U("en", "What is the ESOP exercise price at Zephyra?", "other_company")
U("hi", "ज़ेफ़िरा का कितना रासायनिक उत्पादन है?", "other_company", en="How much chemical production does Zephyra have?")
U("hinglish", "Valmora ka fleet size kitna hai?", "other_company", en="How big is Valmora's fleet?")
# missing_detail
U("en", "Who is the head of Human Resources at Valmora?", "missing_detail")
U("en", "What is the notice period for an employee who resigns?", "missing_detail")
U("en", "What is the daily per diem for an employee in grade L8?", "missing_detail", note="the policy covers grades L1 to L7")
U("en", "What is the hotel limit in a Tier-3 city?", "missing_detail", note="the policy has Tier-1 and Tier-2 only")
U("en", "What is the policy number of Valmora's property insurance?", "missing_detail", note="the report names only the D&O policy number")
U("en", "What is the sum insured of the group travel insurance policy?", "missing_detail", note="only the policy number is given")
U("hi", "इस योजना के तहत प्रशिक्षुओं को यात्रा भत्ता कितना मिलेगा?", "missing_detail", en="How much travel allowance will trainees get under this scheme?")
U("hi", "इस योजना में वृद्धावस्था पेंशन कितनी मिलती है?", "missing_detail", en="How much old-age pension is given under this scheme?")
U("hi", "चयन परीक्षा कब होगी?", "missing_detail", en="When will the selection exam be held?", note="selection is by document verification, no exam")
# off_topic
U("en", "What is the capital of France?", "off_topic")
U("en", "Who won the cricket world cup in 2011?", "off_topic")
U("hi", "आज मुंबई में मौसम कैसा है?", "off_topic", en="What is the weather in Mumbai today?")
U("hinglish", "Aaj gold ka rate kya hai?", "off_topic", en="What is the gold rate today?")
# near_miss_year
U("hi", "वाल्मोरा का FY25 का लाभांश कितना रहेगा?", "near_miss_year", en="What will Valmora's dividend be for FY25?")
# other_company
U("hinglish", "Zephyra ka dividend per share kitna hai?", "other_company", en="What is Zephyra's dividend per share?")
