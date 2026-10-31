import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Renders agent text (paragraphs, lists, tables) as plain report typography.
const components = {
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
};

export default function Markdown({ children }) {
  return (
    <div className="prose-report">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children || ''}
      </ReactMarkdown>
    </div>
  );
}
