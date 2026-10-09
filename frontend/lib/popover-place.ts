/**
 * Where the citation popover goes (components/Citations.tsx). Pure, so it is checked in Node.
 *
 * The popover opens under its chip, above it when it doesn't fit below, and, when it fits on neither side, on the side
 * with more room, shortened to fit (it scrolls inside). On the voice stage the chips sit right under the live captions,
 * and the popover must never cover the words being spoken: rectangles marked `data-cite-avoid` (the captions) are kept
 * clear. A chip under the captions then opens downwards over the controls; when that is too little room to read a
 * table passage in, the popover goes above the captions instead (over the visuals, which the person isn't reading
 * while a citation is open), never over them.
 */

export const GAP = 6;
export const EDGE = 8;
/** The least height worth showing a shortened popover at (its head and a few lines of the passage). */
export const MIN_HEIGHT = 120;
/** Room under the chip that is worth taking when the alternative is a popover above the captions, away from its chip. */
export const COMFORT_HEIGHT = 220;

/** Where the popover goes: its top-left corner, whether it opened above the chip, and a height cap when it was shortened. */
export interface PopoverPlace {
  top: number;
  left: number;
  above: boolean;
  maxHeight: number | null;
}

export interface Span {
  top: number;
  bottom: number;
}

type Vertical = Pick<PopoverPlace, "top" | "above" | "maxHeight">;

/**
 * The vertical place of a popover `height` tall for a chip spanning `chip` (viewport pixels) in a viewport `vh` tall.
 * `keepOut`: rectangles that must stay uncovered (the live captions). In order: under the chip when it fits, above the
 * chip when it fits there, above the keep-out rectangles that sit over the chip when it fits there, else shortened to
 * the room on the side that has most (below, unless above has more and at least `MIN_HEIGHT`).
 */
export function placePopover(chip: Span, height: number, vh: number, keepOut: readonly Span[] = []): Vertical {
  const below = chip.bottom + GAP;
  const floor = Math.min(vh - EDGE, ...keepOut.filter((b) => b.top >= chip.bottom - 1).map((b) => b.top - GAP));
  const over = keepOut.filter((b) => b.bottom <= chip.top + 1);
  const ceiling = Math.max(EDGE, ...over.map((b) => b.bottom + GAP));
  const roomBelow = floor - below;
  const roomAbove = chip.top - GAP - ceiling;
  if (height <= roomBelow) return { top: below, above: false, maxHeight: null };
  if (height <= roomAbove) return { top: chip.top - GAP - height, above: true, maxHeight: null };
  // Above the captions the chip sits under: the popover's bottom edge on their top edge.
  const roof = over.length > 0 ? Math.min(...over.map((b) => b.top)) - GAP : null;
  const roomRoof = roof === null ? 0 : roof - EDGE;
  if (roof !== null && roomBelow < COMFORT_HEIGHT && roomRoof >= Math.min(height, MIN_HEIGHT)) {
    const shown = Math.min(height, roomRoof);
    return { top: Math.round(roof - shown), above: true, maxHeight: shown < height ? Math.floor(shown) : null };
  }
  if (roomAbove > roomBelow && roomAbove >= MIN_HEIGHT) {
    return { top: ceiling, above: true, maxHeight: Math.floor(roomAbove) };
  }
  const room = Math.max(MIN_HEIGHT, Math.floor(roomBelow));
  return { top: Math.round(Math.max(EDGE, Math.min(below, vh - EDGE - room))), above: false, maxHeight: room };
}
