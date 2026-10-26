/** Workspace layout: sidebar widths in CSS pixels, collapse state. */
export interface Layout {
  libraryOpen: boolean;
  libraryWidth: number;
  setOpen: boolean;
  setWidth: number;
}

export const DEFAULT_LAYOUT: Layout = { libraryOpen: true, libraryWidth: 380, setOpen: true, setWidth: 380 };

export const MIN_SIDEBAR = 240;
export const MIN_CENTRE = 420;

/** Keep a sidebar between its minimum and whatever leaves the centre its minimum,
 * given the other sidebar's width (0 when collapsed). */
export function clampSidebar(width: number, total: number, other: number): number {
  const max = Math.max(MIN_SIDEBAR, total - other - MIN_CENTRE);
  return Math.round(Math.min(max, Math.max(MIN_SIDEBAR, width)));
}

export const UI_SCALE_MIN = 85;
export const UI_SCALE_MAX = 140;

export function clampScale(scale: number): number {
  if (!Number.isFinite(scale)) return 100;
  return Math.round(Math.min(UI_SCALE_MAX, Math.max(UI_SCALE_MIN, scale)));
}
