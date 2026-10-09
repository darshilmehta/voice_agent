/** Canvas order: by `position`, then oldest first, then as received. Kept apart so the reducer needs no contract reader. */

export function sortPanels<T extends { position: number; created_at: string }>(panels: T[]): T[] {
  return panels
    .map((p, i) => ({ p, i }))
    .sort((a, b) => a.p.position - b.p.position || a.p.created_at.localeCompare(b.p.created_at) || a.i - b.i)
    .map(({ p }) => p);
}
