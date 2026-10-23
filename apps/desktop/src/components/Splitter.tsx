import { useRef } from 'react';

interface SplitterProps {
  onStart?: () => void;
  /** Pixel delta since the drag started, called on every move. */
  onDrag: (deltaPx: number) => void;
  onEnd?: () => void;
}

/** Vertical drag handle between panes. */
export function Splitter({ onStart, onDrag, onEnd }: SplitterProps) {
  const startX = useRef(0);
  return (
    <div
      className="splitter"
      role="separator"
      aria-orientation="vertical"
      onPointerDown={(e) => {
        startX.current = e.clientX;
        e.currentTarget.setPointerCapture(e.pointerId);
        onStart?.();
      }}
      onPointerMove={(e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
        onDrag(e.clientX - startX.current);
      }}
      onPointerUp={(e) => {
        e.currentTarget.releasePointerCapture(e.pointerId);
        onEnd?.();
      }}
    />
  );
}
