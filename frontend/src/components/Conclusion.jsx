import Markdown from './Markdown';
import { TESTS, pct } from '../lib/tests';

// Follow-up suggestions for the tests that were actually uploaded or mentioned in the findings.
function nextSteps(modalities, verdictText) {
  const mods = modalities.map((m) => String(m).toLowerCase());
  const text = verdictText.toLowerCase();
  const has = (key, words) => mods.includes(key) || words.some((w) => text.includes(w));
  const steps = [];
  if (has('ecg', ['myocardial', 'conduction', 'hypertrophy', 'st/t'])) {
    steps.push('Cardiac: review symptoms and vitals, obtain troponin and a repeat 12-lead ECG; cardiology review if abnormal.');
  }
  if (has('xray', ['cardiomegaly', 'x-ray', 'xray', 'pneumonia', 'effusion', 'opacity'])) {
    steps.push('Chest: radiologist review of the film; echocardiography if cardiomegaly is reported.');
  }
  if (has('dermoscopy', ['nevus', 'melanoma', 'lesion', 'keratosis', 'carcinoma', 'dermatofibroma'])) {
    steps.push('Skin: dermatology review; biopsy if the lesion changes or a malignant class is flagged.');
  }
  if (mods.includes('lab_values')) {
    steps.push('Blood count: interpret together with the imaging findings.');
  }
  return steps;
}

export default function Conclusion({ results }) {
  const verdict = results.final_verdict;
  const preds = results.individual_cnn_predictions || {};
  const modalities = results.modalities_processed || Object.keys(preds);
  const secondary = Array.isArray(verdict?.secondary_findings) ? verdict.secondary_findings : [];

  // Low-confidence flags raised by the image models themselves
  const flags = TESTS.filter((t) => preds[t.resultKey]?.prediction?.abstain)
    .map((t) => ({ title: t.title, reason: preds[t.resultKey].prediction.abstain_reason }));

  if (!verdict) {
    return (
      <section className="px-5 sm:px-6 py-5 border-b border-line">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">Conclusion</h3>
        <p className="mt-2 text-sm text-muted">The panel discussion did not return a conclusion. Model results are shown below.</p>
      </section>
    );
  }

  const isReferral = /refer|inconclusive/i.test(verdict.final_verdict || '');
  const steps = nextSteps(modalities, `${verdict.final_verdict} ${secondary.join(' ')}`);

  return (
    <section className="px-5 sm:px-6 py-5 border-b border-line">
      <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">Conclusion</h3>

      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <p className="text-2xl font-semibold leading-tight">{verdict.final_verdict}</p>
        <p className="text-sm text-muted">
          Confidence <span className="num text-ink font-medium">{pct(verdict.confidence_score, 0)}</span>
          <span className="text-faint"> · not above the model probability</span>
        </p>
      </div>

      {isReferral && (
        <p className="mt-3 px-3 py-2 rounded border border-caution/30 bg-caution-wash text-sm text-caution">
          Specialist review advised. The panel did not reach a confident single conclusion.
        </p>
      )}
      {flags.map((f) => (
        <p key={f.title} className="mt-3 px-3 py-2 rounded border border-caution/30 bg-caution-wash text-sm text-caution">
          {f.title} model flagged a low-confidence result{f.reason ? `: ${f.reason}` : '.'}
        </p>
      ))}

      <div className="mt-5 grid gap-6 md:grid-cols-2">
        <div>
          <h4 className="text-sm font-semibold">Other findings</h4>
          {secondary.length > 0 ? (
            <ul className="mt-1.5 space-y-1 text-sm list-disc pl-5">
              {secondary.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          ) : (
            <p className="mt-1.5 text-sm text-muted">None reported.</p>
          )}
        </div>
        <div>
          <h4 className="text-sm font-semibold">Suggested next steps</h4>
          <ul className="mt-1.5 space-y-1 text-sm list-disc pl-5">
            {steps.map((s, i) => <li key={i}>{s}</li>)}
            <li className="text-muted">All findings must be confirmed by a qualified clinician.</li>
          </ul>
        </div>
      </div>

      <details className="mt-5 group">
        <summary className="cursor-pointer text-sm font-semibold select-none">
          Panel rationale <span className="font-normal text-muted group-open:hidden">(show)</span>
        </summary>
        <div className="mt-2 pl-3 border-l-2 border-line">
          <Markdown>{verdict.audit_trail}</Markdown>
        </div>
      </details>
    </section>
  );
}
