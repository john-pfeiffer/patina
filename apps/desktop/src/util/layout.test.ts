import { describe, expect, it } from 'vitest';
import { clampScale, clampSidebar, MIN_CENTRE, MIN_SIDEBAR } from './layout';

describe('layout', () => {
  it('keeps sidebars within their minimum and the centre minimum', () => {
    expect(clampSidebar(100, 1440, 380)).toBe(MIN_SIDEBAR);
    expect(clampSidebar(380, 1440, 380)).toBe(380);
    expect(clampSidebar(900, 1440, 380)).toBe(1440 - 380 - MIN_CENTRE);
    // a narrow window: the other sidebar collapsed gives more room
    expect(clampSidebar(500, 960, 0)).toBe(500);
    expect(clampSidebar(600, 960, 0)).toBe(960 - MIN_CENTRE);
    expect(clampSidebar(500, 960, 380)).toBe(MIN_SIDEBAR);
  });

  it('clamps the UI scale', () => {
    expect(clampScale(100)).toBe(100);
    expect(clampScale(10)).toBe(85);
    expect(clampScale(400)).toBe(140);
    expect(clampScale(Number.NaN)).toBe(100);
  });
});
