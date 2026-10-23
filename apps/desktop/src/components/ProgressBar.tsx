interface ProgressBarProps {
  /** 0..1 */
  value: number;
  label?: string;
  width?: number;
}

export function ProgressBar({ value, label, width = 80 }: ProgressBarProps) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <span className="progress" style={{ width }} title={label}>
      <span className="progress-fill" style={{ width: `${pct}%` }} />
      <span className="progress-text">{label ?? `${pct}%`}</span>
    </span>
  );
}
