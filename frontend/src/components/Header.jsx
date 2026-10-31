export default function Header({ backend, onReset, busy, hasCase }) {
  const status = {
    checking: { dot: 'bg-faint', text: 'Checking backend…' },
    online: { dot: 'bg-ok', text: 'Backend connected' },
    offline: { dot: 'bg-alert', text: 'Backend not reachable' },
  }[backend];

  return (
    <header className="bg-paper border-b border-line no-print">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <img src="/favicon.svg" alt="" className="w-7 h-7" />
          <div className="min-w-0">
            <div className="font-semibold leading-tight truncate">Multimodal Diagnostic Support</div>
            <div className="text-xs text-muted leading-tight">Research prototype · FYP 2026–27</div>
          </div>
        </div>
        <div className="flex items-center gap-5 text-sm">
          <span className="hidden sm:flex items-center gap-2 text-muted" role="status">
            <span className={`w-2 h-2 rounded-full ${status.dot}`} aria-hidden="true" />
            {status.text}
          </span>
          {hasCase && (
            <button type="button" onClick={onReset} disabled={busy}
              className="px-3 py-1.5 rounded border border-line hover:bg-wash disabled:opacity-50">
              New case
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
