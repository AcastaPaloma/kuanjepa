// The write-up's text lives in content/*.md so it can be edited without touching code. This file turns one of
// those files into reading steps, and formats prose: **bold**, *italic*, `code`, [links](url), $tex$ and $$tex$$.

export const escapeHtml = value => String(value).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

export const tex = (source, display = false) => window.katex
  ? window.katex.renderToString(source, { displayMode: display, throwOnError: false })
  : escapeHtml(source);

const prose = text => escapeHtml(text)
  .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) =>
    `<a href="${url}"${/^https?:/.test(url) ? ' target="_blank" rel="noopener"' : ''}>${label}</a>`)
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[^*\w])\*(?!\s)(.+?)\*(?!\w)/g, '$1<em>$2</em>');

// Punctuation right after inline math stays on its line.
export const rich = source => String(source).split(/(`[^`]+`|\$\$[^$]+\$\$|\$[^$]+\$[.,;:)]*)/).map(part => {
  if (part.length > 1 && part.startsWith('`')) return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
  if (part.startsWith('$$')) return tex(part.slice(2, -2), true);
  const math = /^\$([^$]+)\$(.*)$/.exec(part);
  if (math) return math[2] ? `<span class="nobr">${tex(math[1])}${escapeHtml(math[2])}</span>` : tex(math[1]);
  return prose(part);
}).join('');

// "<!-- id: x · scene: y -->" → { id: 'x', scene: 'y' }
const meta = line => Object.fromEntries(line.replace(/^<!--|-->$/g, '').split('·')
  .map(pair => pair.split(':').map(s => s.trim())).filter(([k, v]) => k && v));

// Paragraphs are separated by blank lines. A paragraph that is all "- " lines is a list, "> " a note, and one that
// starts and ends with $$ is display math.
const blockHtml = para => {
  const lines = para.split('\n');
  if (lines.every(l => /^- /.test(l))) return `<ul>${lines.map(l => `<li>${rich(l.slice(2))}</li>`).join('')}</ul>`;
  if (lines.every(l => /^>/.test(l))) return `<p class="note">${rich(lines.map(l => l.replace(/^> ?/, '')).join(' '))}</p>`;
  const text = lines.join(' ').trim();
  const widget = /^\{\{\s*([\w-]+)\s*\}\}$/.exec(text); // {{name}}: a chart or table the page fills in
  if (widget) return `<figure class="widget" data-widget="${widget[1]}"></figure>`;
  if (/^\$\$[\s\S]+\$\$$/.test(text)) return `<div class="math">${tex(text.slice(2, -2), true)}</div>`;
  return `<p>${rich(text)}</p>`;
};

export function parseSteps(source) {
  const text = source.replace(/^<!--[\s\S]*?-->\s*/, ''); // the editing notes at the top of the file
  const steps = [];
  let step = null, buffer = [];
  const flush = () => {
    if (!step) return;
    step.html = buffer.join('\n').split(/\n\s*\n/).map(p => p.trim()).filter(Boolean).map(blockHtml).join('');
    steps.push(step);
  };
  for (const line of text.split('\n')) {
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      flush();
      step = { level: Math.max(2, heading[1].length), title: heading[2].trim(), first: heading[1].length === 1 };
      buffer = [];
      continue;
    }
    if (step && !step.id && /^<!--.*-->\s*$/.test(line.trim())) { Object.assign(step, meta(line.trim())); continue; }
    buffer.push(line);
  }
  flush();
  steps.forEach((s, i) => { s.id ||= `step-${i + 1}`; });
  return steps;
}
