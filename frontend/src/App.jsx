import { useEffect, useRef, useState } from 'react';
import Header from './components/Header';
import CaseInputs from './components/CaseInputs';
import Progress from './components/Progress';
import Report from './components/Report';
import { analyzeCase, checkBackend } from './lib/api';
import { EMPTY_SLOTS, TESTS } from './lib/tests';

export default function App() {
  const [files, setFiles] = useState(EMPTY_SLOTS);
  const [previews, setPreviews] = useState(EMPTY_SLOTS);
  const [options, setOptions] = useState({ maxRounds: 2, gradcam: false });
  const [backend, setBackend] = useState('checking');
  const [busy, setBusy] = useState(false);
  const [startedAt, setStartedAt] = useState(null);
  const [results, setResults] = useState(null);
  const [generatedAt, setGeneratedAt] = useState(null);
  const [error, setError] = useState(null);
  const reportRef = useRef(null);

  // Backend status, rechecked every 15 s
  useEffect(() => {
    let alive = true;
    const run = async () => {
      const ok = await checkBackend();
      if (alive) setBackend(ok ? 'online' : 'offline');
    };
    run();
    const id = setInterval(run, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // Scroll to the report when it arrives
  useEffect(() => {
    if (results) reportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [results]);

  const setSlot = (key, file) => {
    const url = file && file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    setFiles((f) => ({ ...f, [key]: file }));
    setPreviews((p) => ({ ...p, [key]: url }));
  };

  const onSelect = (key, file) => {
    if (previews[key]) URL.revokeObjectURL(previews[key]);
    setSlot(key, file);
    setResults(null);
    setError(null);
  };
  const onRemove = (key) => {
    if (previews[key]) URL.revokeObjectURL(previews[key]);
    setSlot(key, null);
    setResults(null);
  };

  const onReset = () => {
    Object.values(previews).forEach((u) => u && URL.revokeObjectURL(u));
    setFiles(EMPTY_SLOTS);
    setPreviews(EMPTY_SLOTS);
    setResults(null);
    setError(null);
  };

  const onLoadSample = async () => {
    onReset();
    try {
      const loaded = await Promise.all(TESTS.map(async (t) => {
        const res = await fetch(t.sample.path);
        if (!res.ok) throw new Error(t.sample.name);
        return [t.key, new File([await res.blob()], t.sample.name, { type: t.sample.type })];
      }));
      loaded.forEach(([key, file]) => setSlot(key, file));
    } catch {
      setError('The sample files could not be loaded. Choose files from your computer instead.');
    }
  };

  const onAnalyze = async () => {
    setBusy(true);
    setError(null);
    setResults(null);
    setStartedAt(Date.now());
    try {
      const data = await analyzeCase(files, options);
      setResults(data);
      setGeneratedAt(new Date());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const count = Object.values(files).filter(Boolean).length;
  const selectedTitles = TESTS.filter((t) => files[t.key]).map((t) => t.title).join(', ');

  return (
    <div className="min-h-screen flex flex-col">
      <Header backend={backend} onReset={onReset} busy={busy} hasCase={count > 0 || !!results} />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 py-6 space-y-6">
        {backend === 'offline' && (
          <p className="px-4 py-3 rounded border border-alert/30 bg-alert-wash text-sm text-alert no-print">
            The backend is not reachable. Start it with <code className="num">uvicorn backend.api:app --port 8000</code> from the project folder.
          </p>
        )}

        <CaseInputs
          files={files} previews={previews}
          onSelect={onSelect} onRemove={onRemove}
          onAnalyze={onAnalyze} onLoadSample={onLoadSample}
          busy={busy} options={options} setOptions={setOptions}
        />

        {error && (
          <p role="alert" className="px-4 py-3 rounded border border-alert/30 bg-alert-wash text-sm text-alert no-print">
            {error}
          </p>
        )}

        {busy && <Progress startedAt={startedAt} tests={selectedTitles} />}

        {results && (
          <div ref={reportRef} className="scroll-mt-4">
            <Report results={results} previews={previews} generatedAt={generatedAt} />
          </div>
        )}
      </main>

      <footer className="no-print border-t border-line bg-paper">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 text-xs text-muted flex flex-wrap justify-between gap-2">
          <span>Multimodal Diagnostic Support · Final Year Project 2026–27</span>
          <span>Research use only. Not for clinical decisions.</span>
        </div>
      </footer>
    </div>
  );
}
