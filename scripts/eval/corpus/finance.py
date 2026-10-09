# ruff: noqa: RUF012  (a table of constants: the class attributes are never mutated)
"""The financial models of the two fictional companies. One place for every number, so tables, prose, slides and the
question set can't drift apart; the arithmetic is asserted (segments add up, the balance sheet balances).

All amounts are in INR crore unless a name says otherwise. FY24 = year ended 31 March 2024.
"""

from __future__ import annotations


def n0(x: float) -> str:
    """1234567 -> '1,234,567' (no decimals). Amounts here stay below 1 lakh crore, where Indian and Western
    grouping agree."""
    return f"{x:,.0f}"


def n1(x: float) -> str:
    return f"{x:,.1f}"


def n2(x: float) -> str:
    return f"{x:,.2f}"


def pct(num: float, den: float, digits: int = 1) -> str:
    return f"{num / den * 100:.{digits}f}%"


def growth(new: float, old: float) -> str:
    return f"{(new / old - 1) * 100:.1f}%"


def inr(n: int) -> str:
    """Indian digit grouping for rupee amounts: 250000 -> '2,50,000'."""
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join([*parts, tail])
    return ("-" if n < 0 else "") + s


def bps(new_num: float, new_den: float, old_num: float, old_den: float) -> int:
    return round((new_num / new_den - old_num / old_den) * 10000)


# =============================================================== Valmora Industries Limited


class Valmora:
    """Valmora Industries Limited: specialty chemicals, engineered plastics, digital services."""

    name = "Valmora Industries Limited"
    short = "Valmora"
    cin = "L24119GJ1994PLC023871"
    isin = "INE958R01014"
    bse_code = "543871"
    nse_symbol = "VALMORA"
    shares_crore = 12.5  # equity shares of face value INR 10 (INR 125 crore share capital)

    # segment -> {year: (revenue, EBITDA)}
    segments = {
        "Specialty Chemicals": {"FY23": (3214, 745), "FY24": (3568, 862)},
        "Engineered Plastics": {"FY23": (2126, 331), "FY24": (2303, 346)},
        "Digital Services": {"FY23": (1142, 207), "FY24": (1494, 337)},
    }
    revenue = {"FY23": 6482, "FY24": 7365}
    ebitda = {"FY23": 1283, "FY24": 1545}
    # profit and loss
    other_income = {"FY23": 64, "FY24": 83}
    materials = {"FY23": 3058, "FY24": 3402}
    employee = {"FY23": 1104, "FY24": 1257}
    other_exp = {"FY23": 1037, "FY24": 1161}
    depreciation = {"FY23": 342, "FY24": 371}
    finance = {"FY23": 118, "FY24": 96}
    tax = {"FY23": 223, "FY24": 290}
    pat = {"FY23": 664, "FY24": 871}
    # quarterly (revenue, EBITDA, PAT)
    quarters = {
        "FY24": {"Q1": (1742, 352, 196), "Q2": (1801, 372, 212), "Q3": (1889, 399, 227), "Q4": (1933, 422, 236)},
        "FY23": {"Q1": (1512, 285, 148), "Q2": (1598, 312, 163), "Q3": (1661, 331, 171), "Q4": (1711, 355, 182)},
    }
    # balance sheet (31 March)
    bs = {
        "FY23": dict(
            ppe=3240,
            cwip=412,
            intang=188,
            invest=96,
            other_nca=142,
            inv=1126,
            rec=1204,
            cash=486,
            other_ca=238,
            share_cap=125,
            other_eq=3436,
            nc_borrow=1206,
            other_ncl=332,
            st_borrow=468,
            payables=1044,
            other_cl=521,
        ),
        "FY24": dict(
            ppe=3612,
            cwip=358,
            intang=214,
            invest=118,
            other_nca=161,
            inv=1248,
            rec=1357,
            cash=612,
            other_ca=261,
            share_cap=125,
            other_eq=4165,
            nc_borrow=1052,
            other_ncl=358,
            st_borrow=391,
            payables=1160,
            other_cl=690,
        ),
    }
    dividend_ps = {"FY23": 12.0, "FY24": 15.0}
    cash_flow = {"FY23": (1052, -655, -293), "FY24": (1284, -712, -446)}  # operating, investing, financing
    capex = {"FY23": 642, "FY24": 698}
    employees = {"FY23": 9310, "FY24": 9842}

    @classmethod
    def check(cls) -> None:
        for y in ("FY23", "FY24"):
            assert sum(v[y][0] for v in cls.segments.values()) == cls.revenue[y], y
            assert sum(v[y][1] for v in cls.segments.values()) == cls.ebitda[y], y
            assert cls.materials[y] + cls.employee[y] + cls.other_exp[y] == cls.revenue[y] - cls.ebitda[y], y
            pbt = cls.ebitda[y] + cls.other_income[y] - cls.depreciation[y] - cls.finance[y]
            assert pbt - cls.tax[y] == cls.pat[y], (y, pbt)
            b = cls.bs[y]
            assets = sum(
                b[k] for k in ("ppe", "cwip", "intang", "invest", "other_nca", "inv", "rec", "cash", "other_ca")
            )
            equity = b["share_cap"] + b["other_eq"]
            liabilities = sum(b[k] for k in ("nc_borrow", "other_ncl", "st_borrow", "payables", "other_cl"))
            assert assets == equity + liabilities, (y, assets, equity + liabilities)
            q = cls.quarters[y]
            assert sum(v[0] for v in q.values()) == cls.revenue[y]
            assert sum(v[1] for v in q.values()) == cls.ebitda[y]
            assert sum(v[2] for v in q.values()) == cls.pat[y]
            assert sum(cls.cash_flow[y]) == (126 if y == "FY24" else 104)
        assert cls.bs["FY24"]["cash"] - cls.bs["FY23"]["cash"] == sum(cls.cash_flow["FY24"])
        # equity roll-forward FY24: other equity + PAT - FY23 dividend paid + other comprehensive income
        assert 3436 + 871 - 150 + 8 == 4165

    # ---- derived
    @classmethod
    def seg_rev(cls, seg: str, y: str) -> int:
        return cls.segments[seg][y][0]

    @classmethod
    def seg_ebitda(cls, seg: str, y: str) -> int:
        return cls.segments[seg][y][1]

    @classmethod
    def seg_margin(cls, seg: str, y: str) -> str:
        return pct(cls.seg_ebitda(seg, y), cls.seg_rev(seg, y))

    @classmethod
    def margin(cls, y: str) -> str:
        return pct(cls.ebitda[y], cls.revenue[y])

    @classmethod
    def pbt(cls, y: str) -> int:
        return cls.ebitda[y] + cls.other_income[y] - cls.depreciation[y] - cls.finance[y]

    @classmethod
    def total_income(cls, y: str) -> int:
        return cls.revenue[y] + cls.other_income[y]

    @classmethod
    def eps(cls, y: str) -> str:
        return n2(cls.pat[y] / cls.shares_crore)

    @classmethod
    def borrowings(cls, y: str) -> int:
        return cls.bs[y]["nc_borrow"] + cls.bs[y]["st_borrow"]

    @classmethod
    def net_debt(cls, y: str) -> int:
        return cls.borrowings(y) - cls.bs[y]["cash"]

    @classmethod
    def net_debt_ebitda(cls, y: str) -> str:
        return f"{cls.net_debt(y) / cls.ebitda[y]:.2f}x"

    @classmethod
    def equity(cls, y: str) -> int:
        return cls.bs[y]["share_cap"] + cls.bs[y]["other_eq"]

    @classmethod
    def roce(cls, y: str) -> str:
        """EBIT (profit before tax + finance costs) over closing capital employed (equity + borrowings)."""
        return pct(cls.pbt(y) + cls.finance[y], cls.equity(y) + cls.borrowings(y))

    @classmethod
    def total_assets(cls, y: str) -> int:
        b = cls.bs[y]
        return sum(b[k] for k in ("ppe", "cwip", "intang", "invest", "other_nca", "inv", "rec", "cash", "other_ca"))

    @classmethod
    def free_cash_flow(cls, y: str) -> int:
        return cls.cash_flow[y][0] - cls.capex[y]

    @classmethod
    def margin_expansion_bps(cls) -> int:
        return bps(cls.ebitda["FY24"], cls.revenue["FY24"], cls.ebitda["FY23"], cls.revenue["FY23"])

    @classmethod
    def q_margin(cls, y: str, q: str) -> str:
        rev, ebitda, _ = cls.quarters[y][q]
        return pct(ebitda, rev)


Valmora.check()


# =============================================================== Zephyra Logistics Limited


class Zephyra:
    """Zephyra Logistics Limited: freight, contract logistics, digital services. Overlaps Valmora on metric names
    (revenue, EBITDA margin, PAT, employees, a Digital Services segment) so documents must be told apart."""

    name = "Zephyra Logistics Limited"
    short = "Zephyra"
    cin = "L63090MH2005PLC156042"
    isin = "INE413T01011"
    bse_code = "544019"
    nse_symbol = "ZEPHYRALOG"

    segments = {
        "Freight Services": {"FY23": (2365, 233), "FY24": (2609, 267)},
        "Contract Logistics": {"FY23": (1742, 238), "FY24": (1896, 276)},
        "Digital Services": {"FY23": (330, 57), "FY24": (481, 90)},
    }
    revenue = {"FY23": 4437, "FY24": 4986}
    ebitda = {"FY23": 528, "FY24": 633}
    pat = {"FY23": 212, "FY24": 268}
    net_debt = {"FY23": 1075, "FY24": 912}
    roce = {"FY23": "12.9%", "FY24": "14.8%"}
    employees = {"FY23": 13460, "FY24": 14120}
    q4 = {"Q4 FY24": (1329, 174, 78), "Q4 FY23": (1186, 146, 61), "Q3 FY24": (1263, 158, 70)}  # revenue, EBITDA, PAT
    capex = {"FY23": 338, "FY24": 412}
    cfo = {"FY23": 497, "FY24": 604}

    @classmethod
    def check(cls) -> None:
        for y in ("FY23", "FY24"):
            assert sum(v[y][0] for v in cls.segments.values()) == cls.revenue[y], y
            assert sum(v[y][1] for v in cls.segments.values()) == cls.ebitda[y], y

    @classmethod
    def seg_margin(cls, seg: str, y: str) -> str:
        rev, ebitda = cls.segments[seg][y]
        return pct(ebitda, rev)

    @classmethod
    def margin(cls, y: str) -> str:
        return pct(cls.ebitda[y], cls.revenue[y])

    @classmethod
    def pat_margin(cls, y: str) -> str:
        return pct(cls.pat[y], cls.revenue[y])

    @classmethod
    def nd_ebitda(cls, y: str) -> str:
        return f"{cls.net_debt[y] / cls.ebitda[y]:.1f}x"

    @classmethod
    def q_margin(cls, q: str) -> str:
        rev, ebitda, _ = cls.q4[q]
        return pct(ebitda, rev)

    @classmethod
    def fcf(cls, y: str) -> int:
        return cls.cfo[y] - cls.capex[y]


Zephyra.check()
