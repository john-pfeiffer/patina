import { useEffect, useRef } from 'react';
import { Splitter } from './components/Splitter';
import { LibraryPane } from './panes/LibraryPane';
import { PreferencesDialog } from './panes/PreferencesDialog';
import { ProgressionPane } from './panes/ProgressionPane';
import { RegionEditor } from './panes/RegionEditor';
import { SetPane } from './panes/SetPane';
import { TransportPane } from './panes/TransportPane';
import { useStore } from './store';
import { clampSidebar } from './util/layout';

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

/** Workspace: Library sidebar (left) · Progression (centre) · Set inspector
 * (right) · Transport (bottom). Sidebars collapse to a rail and resize by
 * dragging; the centre takes what is left. */
export function App() {
  const layout = useStore((s) => s.ui.layout);
  const uiScale = useStore((s) => s.ui.uiScale);
  const dialog = useStore((s) => s.ui.dialog);
  const setUi = useStore((s) => s.setUi);
  const workspace = useRef<HTMLDivElement>(null);
  const dragStart = useRef(layout);

  // Cmd/Ctrl+, opens Preferences, as on every desktop app.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === ',') {
        e.preventDefault();
        setUi({ dialog: 'preferences' });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setUi]);

  // UI scale: page zoom, which scales every px value (WebKit and Chromium).
  useEffect(() => {
    const root = document.getElementById('root');
    if (root) (root.style as CSSStyleDeclaration & { zoom?: string }).zoom = `${uiScale / 100}`;
  }, [uiScale]);

  const beginDrag = () => {
    dragStart.current = useStore.getState().ui.layout;
  };
  const total = () => workspace.current?.clientWidth ?? 1200;
  const dragLibrary = (deltaPx: number) =>
    setUi({ layout: { ...layout, libraryWidth: clampSidebar(dragStart.current.libraryWidth + deltaPx, total(), dragStart.current.setOpen ? dragStart.current.setWidth : 0) } });
  const dragSet = (deltaPx: number) =>
    setUi({ layout: { ...layout, setWidth: clampSidebar(dragStart.current.setWidth - deltaPx, total(), dragStart.current.libraryOpen ? dragStart.current.libraryWidth : 0) } });

  return (
    <div className="app">
      <ErrorBar />
      <div className="workspace" ref={workspace}>
        {layout.libraryOpen ? (
          <>
            <div className="sidebar" style={{ width: layout.libraryWidth }}>
              <LibraryPane />
            </div>
            <Splitter onStart={beginDrag} onDrag={dragLibrary} />
          </>
        ) : (
          <button type="button" className="rail" onClick={() => setUi({ layout: { ...layout, libraryOpen: true } })} title="show the library">
            <span>Library</span>
          </button>
        )}
        <div className="centre">
          <ProgressionPane />
        </div>
        {layout.setOpen ? (
          <>
            <Splitter onStart={beginDrag} onDrag={dragSet} />
            <div className="sidebar" style={{ width: layout.setWidth }}>
              <SetPane />
            </div>
          </>
        ) : (
          <button type="button" className="rail" onClick={() => setUi({ layout: { ...layout, setOpen: true } })} title="show the set">
            <span>Set</span>
          </button>
        )}
      </div>
      <TransportPane />
      {dialog === 'preferences' && <PreferencesDialog />}
      {dialog === 'region' && <RegionEditor />}
    </div>
  );
}
