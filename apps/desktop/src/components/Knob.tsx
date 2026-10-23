import { useEffect, useState } from 'react';

interface KnobProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  digits?: number;
  disabled?: boolean;
  /** Fires on every movement (local UI state). */
  onChange?: (value: number) => void;
  /** Fires once the drag ends or the keyboard releases (API-bound state). */
  onCommit?: (value: number) => void;
}

/** A knob looks like a knob: a range input with its numeric readout beside it. */
export function Knob({ label, value, min, max, step = 1, unit = '', digits = 0, disabled, onChange, onCommit }: KnobProps) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  const commit = () => {
    if (onCommit && local !== value) onCommit(local);
  };
  return (
    <label className="knob">
      <span className="knob-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={local}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          setLocal(v);
          onChange?.(v);
        }}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
      />
      <span className="readout">
        {local.toFixed(digits)}
        {unit}
      </span>
    </label>
  );
}
