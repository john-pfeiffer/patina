import { useEffect, type ReactNode } from 'react';

interface DialogProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** Optional controls rendered at the right of the title bar. */
  tools?: ReactNode;
  width?: number;
}

/** Modal overlay. Escape or the backdrop closes it; the page behind stays
 * mounted so playback keeps running. */
export function Dialog({ title, onClose, children, tools, width = 720 }: DialogProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="dialog-backdrop" onPointerDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} style={{ width }}>
        <div className="dialog-head">
          <h2>{title}</h2>
          <span className="spacer" />
          {tools}
          <button type="button" onClick={onClose} title="close (Esc)">
            ×
          </button>
        </div>
        <div className="dialog-body">{children}</div>
      </div>
    </div>
  );
}
