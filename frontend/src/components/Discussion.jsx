import Markdown from './Markdown';

const ROLES = {
  advocate: { name: 'Advocate', role: 'proposes the most likely diagnosis' },
  skeptic: { name: 'Skeptic', role: 'argues for alternatives and checks for over-reach' },
  evidence: { name: 'Evidence check', role: 'compares claims with the retrieved guideline passages' },
  moderator: { name: 'Moderator', role: 'open question for the next round' },
};

/** Splits "Role: text" turns and numbers the rounds (a round starts with the Advocate). */
function parseTurns(transcript) {
  let round = 0;
  return transcript.map((raw, i) => {
    const m = raw.match(/^(Advocate|Skeptic|Evidence[- ]?Checker|Moderator[^:]*)\s*:\s*/i);
    const label = m ? m[1].toLowerCase() : 'moderator';
    const kind = label.startsWith('advocate') ? 'advocate'
      : label.startsWith('skeptic') ? 'skeptic'
      : label.startsWith('evidence') ? 'evidence' : 'moderator';
    if (kind === 'advocate') round += 1;
    return { i, kind, round: Math.max(round, 1), text: m ? raw.slice(m[0].length) : raw };
  });
}

/** "[SOURCE, Page N]:\ntext" blocks from the RAG context. */
function parsePassages(context) {
  if (!context) return [];
  const parts = context.split(/\[([^\]]+?), Page ([^\]]+?)\]:\n/);
  // split() with two capture groups gives [before, source, page, text, source, page, text, ...]
  const out = [];
  for (let k = 1; k < parts.length; k += 3) {
    out.push({ source: parts[k], page: String(parts[k + 1]).replace(/\.0$/, ''), text: (parts[k + 2] || '').trim() });
  }
  return out;
}

export default function Discussion({ transcript = [], retrievedContext = '' }) {
  if (!transcript.length) return null;
  const turns = parseTurns(transcript);
  const rounds = turns.reduce((n, t) => Math.max(n, t.round), 1);
  const passages = parsePassages(retrievedContext);

  return (
    <section className="px-5 sm:px-6 py-5 border-b border-line">
      <details>
        <summary className="cursor-pointer select-none">
          <span className="text-sm font-semibold text-muted uppercase tracking-wide">Panel discussion</span>
          <span className="ml-2 text-sm text-muted">
            {turns.length} contributions over {rounds} {rounds === 1 ? 'round' : 'rounds'}
          </span>
        </summary>

        <p className="mt-3 text-sm text-muted max-w-3xl">
          Four language-model agents review the model outputs. They see probabilities only, not the images,
          and cannot add findings the models did not report.
        </p>

        <ol className="mt-4 space-y-5">
          {turns.map((t, idx) => {
            const r = ROLES[t.kind];
            const newRound = idx === 0 || turns[idx - 1].round !== t.round;
            return (
              <li key={t.i}>
                {newRound && rounds > 1 && (
                  <div className="mb-3 text-xs font-semibold text-faint uppercase tracking-wide">Round {t.round}</div>
                )}
                <div className="pl-4 border-l-2 border-line">
                  <div className="text-sm">
                    <span className="font-semibold">{r.name}</span>
                    <span className="text-muted"> — {r.role}</span>
                  </div>
                  <div className="mt-1.5">
                    <Markdown>{t.text}</Markdown>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>

        {passages.length > 0 && (
          <details className="mt-6">
            <summary className="cursor-pointer select-none text-sm font-semibold">
              Guideline passages retrieved ({passages.length})
            </summary>
            <ul className="mt-3 space-y-3">
              {passages.map((p, i) => (
                <li key={i} className="text-sm">
                  <div className="text-muted">{p.source}, page <span className="num">{p.page}</span></div>
                  <p className="mt-1 text-ink/80 bg-wash border border-line rounded px-3 py-2 whitespace-pre-line">{p.text}</p>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-faint">
              Retrieved from WHO documents in the knowledge base. Coverage of ECG and dermoscopy classes is limited.
            </p>
          </details>
        )}
      </details>
    </section>
  );
}
