// Readout formatting. Readouts are readouts: fixed precision, units, dashes
// for missing values.

export function num(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

export function signed(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s;
}

export function db(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? '—' : `${signed(v, digits)} dB`;
}

export function hz(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  return v >= 1000 ? `${(v / 1000).toFixed(1)} kHz` : `${Math.round(v)} Hz`;
}

export function pct(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)} %`;
}

export function secs(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${v.toFixed(2)} s`;
}

export function ratio(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined ? '—' : `×${v.toFixed(digits)}`;
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
