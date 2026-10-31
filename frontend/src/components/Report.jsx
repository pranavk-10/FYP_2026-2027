import Conclusion from './Conclusion';
import TestResults from './TestResults';
import Discussion from './Discussion';
import { TESTS } from '../lib/tests';

export default function Report({ results, previews, generatedAt }) {
  const preds = results.individual_cnn_predictions || {};
  const included = TESTS.filter((t) => preds[t.resultKey]).map((t) => t.title);

  return (
    <article className="bg-paper border border-line rounded print-plain">
      <header className="px-5 sm:px-6 py-4 border-b border-line flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Report</h2>
          <p className="text-sm text-muted">
            {included.join(', ')}
            {generatedAt && <> · <time dateTime={generatedAt.toISOString()}>{generatedAt.toLocaleString()}</time></>}
          </p>
        </div>
        <button type="button" onClick={() => window.print()}
          className="text-sm px-3 py-1.5 rounded border border-line hover:bg-wash no-print">
          Print / save as PDF
        </button>
      </header>

      <Conclusion results={results} />
      <TestResults results={results} previews={previews} />
      <Discussion transcript={results.debate_transcript || []} retrievedContext={results.retrieved_context || ''} />

      <footer className="px-5 sm:px-6 py-4 text-xs text-muted">
        Research prototype (Final Year Project 2026–27). Not a medical device. Model outputs and the panel
        discussion support, but do not replace, assessment by a qualified clinician.
      </footer>
    </article>
  );
}
