// The v2 tab: content/v2.md as one plain page (a comparison, not a scroll-driven walkthrough).
import { parseSteps, rich, escapeHtml } from './md.js';

const article = document.getElementById('article');
const katexReady = new Promise(resolve => {
  if (window.katex) return resolve();
  document.querySelector('script[src*="katex"]').addEventListener('load', resolve, { once: true });
});
const [source] = await Promise.all([fetch('./content/v2.md', { cache: 'no-cache' }).then(r => r.text()), katexReady]);
parseSteps(source).forEach(step => {
  const section = document.createElement('section');
  section.className = 'reading-block';
  section.id = step.id;
  section.innerHTML = (step.first
    ? `<header class="title-block"><h1><span>${escapeHtml(step.title.split(',')[0])}</span>${escapeHtml(step.title.slice(step.title.split(',')[0].length))}</h1>
       <p class="byline">Kuan Yi Wang · <a href="https://github.com/AcastaPaloma/kuanjepa" target="_blank" rel="noopener">source on GitHub</a></p></header>`
    : `${step.chapter ? `<p class="chapter-kicker">${escapeHtml(step.chapter)}</p>` : ''}<h2>${rich(step.title)}</h2>`) + step.html;
  article.append(section);
});
