import { useRef } from 'react';
import { Splitter } from './components/Splitter';
import { LibraryPane } from './panes/LibraryPane';
import { ProgressionPane } from './panes/ProgressionPane';
import { SetPane } from './panes/SetPane';
import { TransportPane } from './panes/TransportPane';
import { useStore } from './store';

const MIN_FRACTION = 0.3;

function ErrorBar() {
  const error = useStore((s) => s.lastError);
  const clearError = useStore((s) => s.clearError);
  if (!error) return null;
  return (
    <div className="error-bar" role="alert">
      <span>{error}</span>
      <span className="spacer" />
      <button type="button" onClick={clearError}>
        dismiss
      </button>
    </div>
  );
}

export function App() {
  const fractions = useStore((s) => s.ui.paneFractions);
  const setUi = useStore((s) => s.setUi);
  const panesRef = useRef<HTMLDivElement>(null);
  const dragStart = useRef<[number, number, number]>(fractions);

  /** Moves width between pane `i` and `i + 1`, keeping the total constant. */
  const drag = (i: 0 | 1) => (deltaPx: number) => {
    const width = panesRef.current?.clientWidth ?? 1;
    const total = dragStart.current.reduce((a, b) => a + b, 0);
    const delta = (deltaPx / width) * total;
    const next: [number, number, number] = [...dragStart.current];
    const a = next[i] + delta;
    const b = next[i + 1]! - delta;
    if (a < MIN_FRACTION || b < MIN_FRACTION) return;
    next[i] = a;
    next[i + 1] = b;
    setUi({ paneFractions: next });
  };
  const beginDrag = () => {
    dragStart.current = useStore.getState().ui.paneFractions;
  };

  return (
    <div className="app">
      <ErrorBar />
      <div
        className="panes"
        ref={panesRef}
        style={{ gridTemplateColumns: `${fractions[0]}fr 6px ${fractions[1]}fr 6px ${fractions[2]}fr` }}
      >
        <LibraryPane />
        <Splitter onStart={beginDrag} onDrag={drag(0)} />
        <SetPane />
        <Splitter onStart={beginDrag} onDrag={drag(1)} />
        <ProgressionPane />
      </div>
      <TransportPane />
    </div>
  );
}
