import { useEffect, useState } from 'react';

interface TextFieldProps {
  label?: string;
  value: string;
  placeholder?: string;
  width?: number;
  onCommit: (value: string) => void;
}

/** Text input that commits on blur/Enter. */
export function TextField({ label, value, placeholder, width, onCommit }: TextFieldProps) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  const commit = () => {
    if (local !== value) onCommit(local);
  };
  return (
    <label className="field">
      {label && <span className="field-label">{label}</span>}
      <input
        type="text"
        value={local}
        placeholder={placeholder}
        style={width ? { width } : undefined}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          if (e.key === 'Escape') setLocal(value);
        }}
      />
    </label>
  );
}
