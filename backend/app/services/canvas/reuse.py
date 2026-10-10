"""One panel per figure (docs/DESIGN.md §12.1, "A visual already on the canvas", polish round).

Walking the demo's fact questions ("revenue?", "and the EBITDA margin?", "how many employees?") left six identical
"Financial highlights" KPI panels on the canvas: each answer with a headline figure drew the same table's tiles
again. Before a turn's visual is added, it is compared with the chat's unpinned panels by what they show, their data
points (every cell a value came from, every calculation with its inputs), not by title or spec:

- **same**: the same kind and the same points (the same chart drawn again) → the new one takes the old one's place;
- **covered**: every point of the new one is on an older panel of the same kind, or the new one is KPI tiles whose
  cells an older panel of any kind already shows (a tile of FY24 revenue with the revenue chart on screen) → the
  older panel stays as it is (rebuilt with the turn's citations), under the new turn's id, in its place: nothing is
  added;
- **extends**: an older panel of the same kind shows only some of the new one's points (FY24 revenue, then revenue
  and EBITDA for FY24) → the new one takes its place;
- **merge**: KPI tiles from the same table as an older KPI panel, some of them new (the real run: the highlights'
  revenue, EBITDA and margin, then revenue, EBITDA and net profit, two "Financial highlights" panels) → one panel with
  the tiles of both, in its place (when they fit on one panel).

Pinned panels are never taken over (the user keeps them). Otherwise the visual is added as before. The newest panel
that matches wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ...domain.canvas import CellRef, Visual

Reuse = Literal["same", "covered", "extends", "merge"]
Point = tuple[object, ...]


def _cell(c: CellRef) -> Point:
    return ("cell", c.document_id, c.table_id, c.row, c.col)


def points(v: Visual) -> frozenset[Point]:
    """What a visual shows: the cells its values came from and its calculations (operation and input cells)."""
    out: set[Point] = set()
    for row in v.rows:
        for key, cell in row.cells.items():
            if cell is not None:
                out.add(_cell(cell))
            elif row.values.get(key) is not None:
                out.add(("value", row.x, key, row.values[key]))
    for tile in v.tiles:
        if tile.cell is not None:
            out.add(_cell(tile.cell))
        else:
            out.add(("tile", tile.label, tile.value))
    for event in v.events:
        out.add(_cell(event.cell) if event.cell is not None else ("event", event.date, event.label))
    for calc in v.calculations:
        out.add(("calc", calc.op, tuple(sorted(_cell(c) for c in calc.inputs))))
    return frozenset(out)


def _cells(pts: frozenset[Point]) -> frozenset[Point]:
    return frozenset(p for p in pts if p[0] == "cell")


def reuse_of(new: Visual, panels: Sequence[Visual]) -> tuple[Visual, Reuse] | None:
    """The unpinned panel ``new`` should take the place of, and how (module docstring); None: add it."""
    mine = points(new)
    if not mine:
        return None
    for old in sorted((p for p in panels if not p.pinned and p.id != new.id), key=lambda p: -p.position):
        theirs = points(old)
        if not theirs:
            continue
        if new.kind == old.kind and mine == theirs:
            return old, "same"
        if mine <= theirs and new.kind == old.kind:
            return old, "covered"
        if new.kind == "kpi" and _cells(mine) and _cells(mine) <= _cells(theirs):  # its figures, on another chart
            return old, "covered"
        if theirs < mine and new.kind == old.kind:
            return old, "extends"
        if new.kind == old.kind == "kpi" and _tables(mine) and _tables(mine) == _tables(theirs):
            return old, "merge"
    return None


def _tables(pts: frozenset[Point]) -> frozenset[object]:
    return frozenset(p[2] for p in pts if p[0] == "cell")
