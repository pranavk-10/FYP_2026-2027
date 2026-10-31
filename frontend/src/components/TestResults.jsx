import { useState } from 'react';
import { TESTS, TIER_LABEL, probOf, pct } from '../lib/tests';

const TIER_STYLE = {
  detected: { bar: 'bg-ink', text: 'text-ink font-medium' },
  watchlist: { bar: 'bg-caution', text: 'text-caution' },
  unlikely: { bar: 'bg-faint/60', text: 'text-muted' },
};

function ProbabilityCell({ p, tier }) {
  const style = TIER_STYLE[tier] || { bar: 'bg-accent', text: 'text-ink' };
  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="h-1.5 w-12 sm:w-32 bg-line rounded-full overflow-hidden" aria-hidden="true">
        <div className={`h-full ${style.bar}`} style={{ width: `${Math.min(100, p * 100)}%` }} />
      </div>
      <span className="num text-sm w-14 text-right">{pct(p)}</span>
    </div>
  );
}

function ImagePanel({ preview, heatmap, title }) {
  const [view, setView] = useState('image');
  const src = view === 'heatmap' && heatmap ? `data:image/png;base64,${heatmap}` : preview;
  if (!src) return null;
  return (
    <figure>
      <div className="h-44 rounded border border-line bg-wash flex items-center justify-center overflow-hidden">
        <img src={src} alt={`${title} ${view}`} className="max-w-full max-h-full object-contain" />
      </div>
      {heatmap && (
        <div className="mt-2 inline-flex text-xs border border-line rounded overflow-hidden no-print" role="group">
          {['image', 'heatmap'].map((v) => (
            <button key={v} type="button" onClick={() => setView(v)}
              className={`px-2.5 py-1 ${view === v ? 'bg-ink text-white' : 'bg-paper text-muted hover:text-ink'}`}>
              {v === 'image' ? 'Image' : 'Attention heatmap'}
            </button>
          ))}
        </div>
      )}
    </figure>
  );
}

function ModelResult({ test, data, preview }) {
  const [showAll, setShowAll] = useState(false);
  const pred = data.prediction || {};
  const rows = [...(pred.findings || [])].sort((a, b) => probOf(b) - probOf(a));
  // Long lists (X-ray: 19 findings) are shortened; short ones are always shown in full
  const limit = rows.length > 8 ? 6 : rows.length;
  const visible = showAll ? rows : rows.slice(0, limit);
  const hasTiers = rows.some((r) => r.risk_tier);
  const note = test.key === 'ecg' ? (data.evidence?.negative_findings || []).slice(-1)[0] : null;

  return (
    <div className="py-5 border-b border-line last:border-b-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="font-semibold">{test.title}</h4>
        <span className="text-xs text-faint">{test.model} · {test.trainedOn}</span>
      </div>

      <div className={`mt-3 grid gap-5 ${preview ? 'md:grid-cols-[15rem_1fr]' : ''}`}>
        {preview && <ImagePanel preview={preview} heatmap={data.explainability?.heatmap_png_base64} title={test.title} />}

        <div className="min-w-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted border-b border-line">
                <th className="font-medium py-1.5 pr-3">Finding</th>
                <th className="font-medium py-1.5 pr-3">Model probability</th>
                {hasTiers && <th className="font-medium py-1.5">Status</th>}
              </tr>
            </thead>
            <tbody>
              {visible.map((f, i) => {
                const style = TIER_STYLE[f.risk_tier];
                return (
                  <tr key={f.raw_class || f.label} className="border-b border-line/70 last:border-b-0">
                    <td className={`py-1.5 pr-3 ${i === 0 ? 'font-medium' : ''}`}>{f.label}</td>
                    <td className="py-1.5 pr-3"><ProbabilityCell p={probOf(f)} tier={f.risk_tier} /></td>
                    {hasTiers && (
                      <td className={`py-1.5 text-sm ${style?.text || ''}`}
                        title={f.threshold != null ? `Decision threshold ${pct(f.threshold, 0)}` : undefined}>
                        {TIER_LABEL[f.risk_tier] || '—'}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {rows.length > limit && (
            <button type="button" onClick={() => setShowAll((s) => !s)}
              className="mt-2 text-sm text-accent hover:underline no-print">
              {showAll ? 'Show fewer' : `Show all ${rows.length} findings`}
            </button>
          )}

          <div className="mt-3 space-y-1 text-sm text-muted">
            {pred.normality_score != null && (
              <p>Probability of a normal study: <span className="num text-ink">{pct(pred.normality_score)}</span></p>
            )}
            {pred.abstain && (
              <p className="text-caution">Low-confidence result: {pred.abstain_reason || 'specialist review advised.'}</p>
            )}
            {note && <p className="text-xs text-faint">{note}</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

const CBC_LABELS = {
  hemoglobin: 'Hemoglobin', rbc: 'RBC count', wbc: 'WBC count', platelets: 'Platelets',
  hematocrit: 'Hematocrit', mcv: 'MCV', mch: 'MCH', mchc: 'MCHC', neutrophils: 'Neutrophils',
  lymphocytes: 'Lymphocytes', monocytes: 'Monocytes', eosinophils: 'Eosinophils', basophils: 'Basophils',
};

function LabResult({ test, data }) {
  if (data.error) {
    return (
      <div className="py-5 border-b border-line last:border-b-0">
        <h4 className="font-semibold">{test.title}</h4>
        <p className="mt-2 text-sm text-alert">The report could not be read: {data.error}</p>
      </div>
    );
  }
  // Flags are only computed for the three headline values
  const flags = { hemoglobin: data.hemoglobin?.flag, wbc: data.wbc_count?.flag, platelets: data.platelets?.flag };
  const entries = Object.entries(data.cbc || {}).filter(([, v]) => v && v.value != null);
  const patient = data.patient || {};

  return (
    <div className="py-5 border-b border-line last:border-b-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="font-semibold">{test.title}</h4>
        <span className="text-xs text-faint">{test.model}</span>
      </div>
      {(patient.name || patient.age || patient.gender) && (
        <p className="mt-1 text-sm text-muted">
          Report for {patient.name || 'unnamed patient'}
          {patient.age ? `, ${patient.age} y` : ''}{patient.gender ? `, ${patient.gender}` : ''}
        </p>
      )}
      <table className="mt-3 w-full max-w-xl text-sm">
        <thead>
          <tr className="text-left text-xs text-muted border-b border-line">
            <th className="font-medium py-1.5 pr-3">Parameter</th>
            <th className="font-medium py-1.5 pr-3 text-right">Value</th>
            <th className="font-medium py-1.5 pr-3">Unit</th>
            <th className="font-medium py-1.5">Flag</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, v]) => {
            const flag = flags[key];
            const flagClass = flag === 'high' ? 'text-alert font-medium' : flag === 'low' ? 'text-caution font-medium' : 'text-muted';
            return (
              <tr key={key} className="border-b border-line/70 last:border-b-0">
                <td className="py-1.5 pr-3">{CBC_LABELS[key] || key}</td>
                <td className="py-1.5 pr-3 text-right num">{Number(v.value).toLocaleString()}</td>
                <td className="py-1.5 pr-3 text-muted">{v.unit}</td>
                <td className={`py-1.5 ${flagClass}`}>{flag ? flag[0].toUpperCase() + flag.slice(1) : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {data.other_abnormalities?.length > 0 && (
        <p className="mt-3 text-sm text-caution">Other abnormalities: {data.other_abnormalities.join('; ')}</p>
      )}
      <p className="mt-2 text-xs text-faint">Flags are checked for hemoglobin, WBC and platelets against standard reference ranges.</p>
    </div>
  );
}

export default function TestResults({ results, previews }) {
  const preds = results.individual_cnn_predictions || {};
  const present = TESTS.filter((t) => preds[t.resultKey]);
  if (present.length === 0) return null;

  return (
    <section className="px-5 sm:px-6 pt-5 pb-1 border-b border-line">
      <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">Results by test</h3>
      {present.map((t) =>
        t.key === 'blood'
          ? <LabResult key={t.key} test={t} data={preds[t.resultKey]} />
          : <ModelResult key={t.key} test={t} data={preds[t.resultKey]} preview={previews[t.key]} />
      )}
    </section>
  );
}
