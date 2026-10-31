import { useEffect, useState } from 'react';

// The backend runs everything in one request, so this shows elapsed time and what the
// request does, without pretending to know which step is currently running.
export default function Progress({ startedAt, tests }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const s = Math.max(0, Math.floor((now - startedAt) / 1000));
  const elapsed = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

  return (
    <section className="bg-paper border border-line rounded px-5 sm:px-6 py-5 no-print" aria-live="polite">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="w-4 h-4 rounded-full border-2 border-line border-t-accent animate-spin" aria-hidden="true" />
          <span className="font-medium">Analysing {tests}</span>
        </div>
        <span className="num text-sm text-muted">{elapsed}</span>
      </div>
      <p className="mt-3 text-sm text-muted">
        The image models and report reader run first (a few seconds), then the panel discussion
        reviews their outputs. This usually takes about one minute.
      </p>
    </section>
  );
}
