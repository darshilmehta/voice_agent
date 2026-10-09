# ruff: noqa: RUF001, RUF002  (documents print en dashes, minus, times and divide signs: used on purpose)
"""Deterministic arithmetic for visuals (docs/DESIGN.md §12.1, workstream 4). The model only asks for a calculation;
this module computes it from cells and records the inputs (CellRefs), the unit and a readable formula.

    growth  (to ÷ from − 1) × 100          %        both inputs in the same unit, from > 0
    cagr    ((to ÷ from)^(1/years) − 1) × 100   %   years from the two periods' end dates, both > 0
    diff    to − from                      the inputs' unit; percentages give percentage points (pp)
    ratio   a ÷ b                          x
    share   part ÷ total × 100             %        total: a total cell, or the sum of the parts
    sum     a + b + …                      the inputs' unit

Results are rounded for display (1 decimal for %, 2 for ratios, the inputs' decimals otherwise) and the rounded value
is what the visual shows and what the grounding check recomputes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ...domain.canvas import CalcOp, Calculation, CellRef, Unit
from ...domain.datasets import PeriodInfo
from .parsing import PERCENT, RATIO

PP = Unit(kind="percent", label="pp")


class CalculationError(ValueError):
    """The inputs don't support the calculation (different units, zero or negative base, no time between them)."""


@dataclass(frozen=True, slots=True)
class Operand:
    """One input: a cell's value with its unit, where it came from, and its label in the visual."""

    value: float
    unit: Unit | None
    cell: CellRef
    decimals: int = 0
    label: str = ""  # the x it sits at ("FY24") or the series' name
    period: PeriodInfo | None = None


# ------------------------------------------------------------------ number formatting


def group_digits(integer: int, indian: bool) -> str:
    s = str(abs(integer))
    if indian and len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join([*parts, tail])
    elif len(s) > 3:
        s = f"{abs(integer):,}"
    return s


def format_number(value: float, decimals: int, *, indian: bool = True) -> str:
    """7365 → "7,365"; 421000 → "4,21,000" (Indian grouping); -712 → "−712"."""
    rounded = round(value, decimals)
    sign = "−" if rounded < 0 else ""
    integer = int(abs(rounded))
    text = group_digits(integer, indian)
    if decimals > 0:
        frac = f"{abs(rounded):.{decimals}f}".split(".")[1]
        text = f"{text}.{frac}"
    return sign + text


def is_indian(unit: Unit | None) -> bool:
    return unit is None or unit.currency in (None, "INR") or unit.scale in ("crore", "lakh")


def format_value(value: float, unit: Unit | None, decimals: int, language: str = "en") -> str:
    """A value as the summary and formulas print it: "₹7,365 crore", "21.0%", "0.54x", "1.2 pp", "2,10,000 tpa"."""
    from .parsing import CURRENCY_SYMBOLS, SCALE_HI

    number = format_number(value, decimals, indian=is_indian(unit))
    if unit is None:
        return number
    if unit.kind == "percent":
        return f"{number} pp" if unit.label == "pp" else f"{number}%"
    if unit.kind == "ratio":
        return f"{number}x"
    if unit.kind == "currency":
        symbol = CURRENCY_SYMBOLS.get(unit.currency or "", unit.currency or "")
        scale = (SCALE_HI.get(unit.scale, unit.scale) if language == "hi" else unit.scale) if unit.scale else ""
        if number.startswith("−"):
            return f"−{symbol}{number[1:]}" + (f" {scale}" if scale else "")
        return f"{symbol}{number}" + (f" {scale}" if scale else "")
    return f"{number} {unit.label}".strip()


# ------------------------------------------------------------------ operations


def _same_unit(a: Unit | None, b: Unit | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (a.kind, a.currency, a.scale, a.label) == (b.kind, b.currency, b.scale, b.label)


def _require_same(operands: Sequence[Operand], op: str) -> None:
    first = operands[0].unit
    for o in operands[1:]:
        if not _same_unit(first, o.unit):
            raise CalculationError(f"{op}: inputs are in different units ({_name(first)} and {_name(o.unit)})")


def _name(unit: Unit | None) -> str:
    return unit.label or unit.kind if unit else "no unit"


def _fmt(o: Operand) -> str:
    return format_number(o.value, o.decimals, indian=is_indian(o.unit))


_LABELS: dict[str, dict[CalcOp, str]] = {
    "en": {
        "growth": "{series} growth, {a} to {b}",
        "cagr": "{series} CAGR, {a} to {b}",
        "diff": "{series} change, {a} to {b}",
        "ratio": "{series} to {other}{at}",
        "share": "{part} share of {total}{at}",
        "sum": "Total {series}{at}",
    },
    "hi": {
        "growth": "{series} वृद्धि, {a} से {b}",
        "cagr": "{series} CAGR, {a} से {b}",
        "diff": "{series} बदलाव, {a} से {b}",
        "ratio": "{series} और {other} का अनुपात{at}",
        "share": "{total} में {part} का हिस्सा{at}",
        "sum": "कुल {series}{at}",
    },
}


def _label(op: CalcOp, language: str, **parts: str) -> str:
    return _LABELS["hi" if language == "hi" else "en"][op].format(**parts)


def growth(old: Operand, new: Operand, *, series: str, language: str = "en") -> Calculation:
    _require_same([old, new], "growth")
    if old.value <= 0:
        raise CalculationError(f"growth: the starting value {_fmt(old)} must be positive")
    value = round((new.value / old.value - 1) * 100, 1)
    formula = f"({_fmt(new)} − {_fmt(old)}) ÷ {_fmt(old)} × 100 = {format_number(value, 1)}%"
    return Calculation(
        label=_label("growth", language, series=series, a=old.label, b=new.label),
        op="growth",
        value=value,
        unit=PERCENT,
        inputs=[old.cell, new.cell],
        formula_text=formula,
    )


def years_between(old: PeriodInfo | None, new: PeriodInfo | None) -> float:
    if old is None or new is None or old.end is None or new.end is None:
        raise CalculationError("cagr: both inputs must be periods with known dates")
    months = (new.end.year - old.end.year) * 12 + (new.end.month - old.end.month)
    if months < 12:
        raise CalculationError("cagr: the periods must be at least a year apart")
    return round(months / 12, 2)


def cagr(old: Operand, new: Operand, *, series: str, language: str = "en") -> Calculation:
    _require_same([old, new], "cagr")
    if old.value <= 0 or new.value <= 0:
        raise CalculationError("cagr: both values must be positive")
    years = years_between(old.period, new.period)
    value = round(((new.value / old.value) ** (1 / years) - 1) * 100, 1)
    years_text = f"{years:g}"
    formula = f"({_fmt(new)} ÷ {_fmt(old)})^(1/{years_text}) − 1 = {format_number(value, 1)}%"
    return Calculation(
        label=_label("cagr", language, series=series, a=old.label, b=new.label),
        op="cagr",
        value=value,
        unit=PERCENT,
        inputs=[old.cell, new.cell],
        formula_text=formula,
    )


def diff(old: Operand, new: Operand, *, series: str, language: str = "en", label: str | None = None) -> Calculation:
    _require_same([old, new], "diff")
    decimals = max(old.decimals, new.decimals)
    value = round(new.value - old.value, decimals)
    percent = old.unit is not None and old.unit.kind == "percent"
    unit = PP if percent else old.unit
    shown = format_value(value, unit, decimals)
    sign = "+" if value > 0 else ""
    formula = f"{_fmt(new)} − {_fmt(old)} = {sign}{shown}"
    return Calculation(
        label=label or _label("diff", language, series=series, a=old.label, b=new.label),
        op="diff",
        value=value,
        unit=unit,
        inputs=[old.cell, new.cell],
        formula_text=formula,
    )


def ratio(a: Operand, b: Operand, *, series: str, other: str, at: str = "", language: str = "en") -> Calculation:
    if b.value == 0:
        raise CalculationError("ratio: division by zero")
    value = round(a.value / b.value, 2)
    formula = f"{_fmt(a)} ÷ {_fmt(b)} = {format_number(value, 2)}x"
    return Calculation(
        label=_label("ratio", language, series=series, other=other, at=f" ({at})" if at else ""),
        op="ratio",
        value=value,
        unit=RATIO,
        inputs=[a.cell, b.cell],
        formula_text=formula,
    )


def share(
    part: Operand, total: Operand | Sequence[Operand], *, total_label: str, at: str = "", language: str = "en"
) -> Calculation:
    """``part`` as a percentage of ``total``: a total cell, or the parts themselves (summed; ``part`` among them)."""
    parts = [total] if isinstance(total, Operand) else list(total)
    _require_same([part, *parts], "share")
    whole = sum(o.value for o in parts)
    if whole <= 0 or part.value < 0:
        raise CalculationError("share: the total must be positive and the part not negative")
    value = round(part.value / whole * 100, 1)
    # inputs: the part, then what the whole is made of (a total cell, or every part, the part itself included)
    inputs = [part.cell, *(o.cell for o in parts)]
    whole_text = _fmt(parts[0]) if len(parts) == 1 else "(" + " + ".join(_fmt(o) for o in parts) + ")"
    formula = f"{_fmt(part)} ÷ {whole_text} × 100 = {format_number(value, 1)}%"
    return Calculation(
        label=_label("share", language, part=part.label, total=total_label, at=f" ({at})" if at else ""),
        op="share",
        value=value,
        unit=PERCENT,
        inputs=inputs,
        formula_text=formula,
    )


def total(operands: Sequence[Operand], *, series: str, at: str = "", language: str = "en") -> Calculation:
    """op "sum": the operands added up (a negative operand subtracts: a bridge from revenue to profit)."""
    if len(operands) < 2:
        raise CalculationError("sum: needs at least two inputs")
    _require_same(operands, "sum")
    decimals = max(o.decimals for o in operands)
    value = round(sum(o.value for o in operands), decimals)
    terms = _fmt(operands[0]) + "".join(
        f" − {format_number(-o.value, o.decimals, indian=is_indian(o.unit))}" if o.value < 0 else f" + {_fmt(o)}"
        for o in operands[1:]
    )
    formula = f"{terms} = {format_number(value, decimals, indian=is_indian(operands[0].unit))}"
    return Calculation(
        label=_label("sum", language, series=series, at=f" ({at})" if at else ""),
        op="sum",
        value=value,
        unit=operands[0].unit,
        inputs=[o.cell for o in operands],
        formula_text=formula,
    )


def recompute(calc: Calculation, values: Sequence[float]) -> float:
    """The calculation's value from its input values, as computed above: the grounding check's reference."""
    if calc.op == "growth":
        return round((values[1] / values[0] - 1) * 100, 1)
    if calc.op == "diff":
        return values[1] - values[0]
    if calc.op == "ratio":
        return round(values[0] / values[1], 2)
    if calc.op == "share":
        return round(values[0] / sum(values[1:]) * 100, 1)
    if calc.op == "sum":
        return sum(values)
    if calc.op == "cagr":
        exponent = calc.formula_text.split("^(1/")[1].split(")")[0]
        return round(((values[1] / values[0]) ** (1 / float(exponent)) - 1) * 100, 1)
    raise ValueError(calc.op)


def close(a: float, b: float, decimals: int = 2) -> bool:
    return math.isclose(a, b, abs_tol=0.5 * 10**-decimals + 1e-9)
