interface SparklineProps {
  values: readonly number[];
  /** Fixed range so bands are comparable between rows; defaults to the data range. */
  min?: number;
  max?: number;
  width?: number;
  height?: number;
  title?: string;
}

/** Tiny bar sparkline for 10-band readouts (tilt, noise spectrum, deltas). */
export function Sparkline({ values, min, max, width = 44, height = 12, title }: SparklineProps) {
  if (values.length === 0) return <span className="muted">—</span>;
  const lo = min ?? Math.min(...values);
  const hi = max ?? Math.max(...values);
  const span = hi - lo || 1;
  const bw = width / values.length;
  const zero = lo < 0 && hi > 0 ? height - ((0 - lo) / span) * height : height;
  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-label={title}>
      {title && <title>{title}</title>}
      {lo < 0 && hi > 0 && <line x1={0} x2={width} y1={zero} y2={zero} className="sparkline-zero" />}
      {values.map((v, i) => {
        const y = height - ((v - lo) / span) * height;
        const top = Math.min(y, zero);
        const h = Math.max(1, Math.abs(zero - y));
        return <rect key={i} x={i * bw + 0.5} y={top} width={Math.max(1, bw - 1)} height={h} />;
      })}
    </svg>
  );
}
