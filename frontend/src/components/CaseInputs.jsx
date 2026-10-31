import { useRef, useState } from 'react';
import { FileText, Upload } from 'lucide-react';
import { TESTS, formatBytes } from '../lib/tests';

function InputRow({ test, file, preview, onSelect, onRemove, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const pick = () => inputRef.current?.click();
  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onSelect(test.key, dropped);
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-[13rem_1fr] gap-3 sm:gap-6 py-4 border-b border-line last:border-b-0">
      <div>
        <div className="font-medium">{test.title}</div>
        <div className="text-sm text-muted">{test.detail}</div>
        <div className="text-xs text-faint mt-0.5">Model: {test.model}</div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={test.accept}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.[0]) onSelect(test.key, e.target.files[0]);
          e.target.value = '';
        }}
      />

      {file ? (
        <div className="flex items-center gap-4 min-w-0">
          <div className="w-28 h-20 shrink-0 rounded border border-line bg-wash flex items-center justify-center overflow-hidden">
            {preview ? (
              <img src={preview} alt={`${test.title} preview`} className="max-w-full max-h-full object-contain" />
            ) : (
              <FileText className="w-6 h-6 text-faint" aria-hidden="true" />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium truncate">{file.name}</div>
            <div className="text-xs text-muted num">{formatBytes(file.size)}</div>
            <div className="mt-1.5 flex gap-4 text-sm">
              <button type="button" onClick={pick} disabled={disabled}
                className="text-accent hover:underline disabled:opacity-50">Replace</button>
              <button type="button" onClick={() => onRemove(test.key)} disabled={disabled}
                className="text-muted hover:text-alert hover:underline disabled:opacity-50">Remove</button>
            </div>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={pick}
          disabled={disabled}
          onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`h-20 w-full rounded border border-dashed text-sm flex items-center justify-center gap-2 transition-colors
            ${dragging ? 'border-accent bg-accent-wash text-accent' : 'border-line bg-wash text-muted hover:border-faint hover:text-ink'}
            disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          <Upload className="w-4 h-4" aria-hidden="true" />
          Drop a file here or <span className="text-accent underline underline-offset-2">browse</span>
        </button>
      )}
    </div>
  );
}

export default function CaseInputs({
  files, previews, onSelect, onRemove, onAnalyze, onLoadSample, busy, options, setOptions,
}) {
  const count = Object.values(files).filter(Boolean).length;

  return (
    <section className="bg-paper border border-line rounded no-print">
      <div className="px-5 sm:px-6 pt-5 pb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold">Case inputs</h2>
          <p className="text-sm text-muted">Add any combination of tests. Each is analysed by its own model, then reviewed together.</p>
        </div>
        <button type="button" onClick={onLoadSample} disabled={busy}
          className="text-sm text-accent hover:underline disabled:opacity-50">
          Load sample case
        </button>
      </div>

      <div className="px-5 sm:px-6">
        {TESTS.map((t) => (
          <InputRow key={t.key} test={t} file={files[t.key]} preview={previews[t.key]}
            onSelect={onSelect} onRemove={onRemove} disabled={busy} />
        ))}
      </div>

      <div className="px-5 sm:px-6 py-4 border-t border-line bg-wash rounded-b flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <details className="text-sm">
          <summary className="cursor-pointer text-muted hover:text-ink select-none">Advanced options</summary>
          <div className="mt-3 flex flex-col gap-3">
            <label className="flex items-center gap-3">
              <span className="text-muted w-40">Discussion rounds (max)</span>
              <select
                value={options.maxRounds}
                onChange={(e) => setOptions((o) => ({ ...o, maxRounds: Number(e.target.value) }))}
                disabled={busy}
                className="border border-line rounded bg-paper px-2 py-1"
              >
                {[1, 2, 3].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-3">
              <span className="text-muted w-40">ECG attention heatmap</span>
              <input type="checkbox" aria-label="ECG attention heatmap" checked={options.gradcam} disabled={busy}
                onChange={(e) => setOptions((o) => ({ ...o, gradcam: e.target.checked }))} />
              <span className="text-xs text-faint">Grad-CAM, adds a few seconds</span>
            </label>
          </div>
        </details>

        <button
          type="button"
          onClick={onAnalyze}
          disabled={count === 0 || busy}
          className="px-5 py-2 rounded bg-accent text-white text-sm font-medium hover:bg-accent-dark
            disabled:bg-faint disabled:cursor-not-allowed"
        >
          {busy ? 'Analysing…' : count === 0 ? 'Add a test to analyse' : `Analyse ${count} ${count === 1 ? 'test' : 'tests'}`}
        </button>
      </div>
    </section>
  );
}
