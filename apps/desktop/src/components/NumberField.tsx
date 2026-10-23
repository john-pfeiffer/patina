import { useEffect, useState } from 'react';

interface NumberFieldProps {
  label?: string;
  value: number | null;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  unit?: string;
  width?: number;
  disabled?: boolean;
  /** Commits on blur or Enter; `null` when the field was cleared (if `allowEmpty`). */
  onCommit: (value: number | null) => void;
  allowEmpty?: boolean;
}

/** Numeric input that commits on blur/Enter so typing does not spam the API. */
export function NumberField({ label, value, min, max, step, placeholder, unit, width, disabled, onCommit, allowEmpty }: NumberFieldProps) {
  const text = value === null ? '' : String(value);
  const [local, setLocal] = useState(text);
  useEffect(() => setLocal(text), [text]);
  const commit = () => {
    if (local === text) return;
    if (local.trim() === '') {
      if (allowEmpty) onCommit(null);
      else setLocal(text);
      return;
    }
    const v = Number(local);
    if (Number.isNaN(v)) {
      setLocal(text);
      return;
    }
    onCommit(v);
  };
  return (
    <label className="field">
      {label && <span className="field-label">{label}</span>}
      <input
        type="number"
        value={local}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        disabled={disabled}
        style={width ? { width } : undefined}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          if (e.key === 'Escape') setLocal(text);
        }}
      />
      {unit && <span className="unit">{unit}</span>}
    </label>
  );
}
