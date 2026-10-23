let activeEditor;
function initializeEditor() {
activeEditor?.abort();
activeEditor = new AbortController();
const signal = activeEditor.signal;
// Full page exits ensure the browser warns about an unsaved owner draft.
document.querySelectorAll('a[href]').forEach(link => link.setAttribute('data-astro-reload', ''));
const $ = id => document.getElementById(id);
let csrf = '', path = '', sha = '', dirty = false, structured = null;
const status = message => { $('status').textContent = message; };
function markDirty(value = true) { dirty = value; $('dirty').textContent = value ? 'Unsaved changes' : ''; }
async function api(action, params = {}, body) {
  const query = new URLSearchParams({ action, ...params });
  const response = await fetch(`/api/editor?${query}`, { method: body ? 'POST' : 'GET', credentials: 'same-origin', headers: body ? { 'Content-Type': 'application/json', 'X-Editor-CSRF': csrf } : {}, ...(body ? { body: JSON.stringify(body) } : {}) });
  const data = await response.json();
  if (!response.ok) { if (response.status === 401 && action !== 'login') { $('workspace').hidden = true; $('login').hidden = false; } throw new Error(data.error || 'Request failed.'); }
  return data;
}
const guard = fn => async event => { event?.preventDefault(); try { await fn(event); } catch (error) { status(error.message); } };
function makeField(label, value, multiline = false) {
  const wrapper = document.createElement('label'); wrapper.textContent = label;
  const input = document.createElement(multiline ? 'textarea' : 'input'); input.value = value; if (multiline) input.rows = 5;
  input.addEventListener('input', () => markDirty()); wrapper.append(input); return { wrapper, input };
}
function renderStructured(data) {
  $('structured').replaceChildren(); structured = data;
  const fields = path.endsWith('about.json') ? [['lead', 'Opening sentence'], ['introduction', 'Introduction']] : [['headLine', 'Headline'], ['headLine2', 'Introduction']];
  for (const [key, label] of fields) {
    const field = makeField(label, data[key] || '', true); field.input.addEventListener('input', () => { structured[key] = field.input.value; }); $('structured').append(field.wrapper);
  }
  if (path.endsWith('home.json')) {
    const field = makeField('Intro lines (one per line)', data.shortIntros.join('\n'), true); field.input.addEventListener('input', () => { structured.shortIntros = field.input.value.split('\n').filter(Boolean); }); $('structured').append(field.wrapper); return;
  }
  data.sections.forEach((section, index) => {
    const box = document.createElement('div'); box.className = 'section-editor';
    const title = makeField('Section heading', section.title); title.input.addEventListener('input', () => { section.title = title.input.value; });
    const paragraphs = makeField('Paragraphs (separate with a blank line)', section.paragraphs.join('\n\n'), true); paragraphs.input.rows = Math.min(18, Math.max(6, section.paragraphs.length * 3)); paragraphs.input.addEventListener('input', () => { section.paragraphs = paragraphs.input.value.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean); });
    const buttons = document.createElement('div'); buttons.className = 'toolbar';
    for (const [name, operation] of [['Move up', () => { if (index) [data.sections[index - 1], data.sections[index]] = [data.sections[index], data.sections[index - 1]]; }], ['Move down', () => { if (index < data.sections.length - 1) [data.sections[index + 1], data.sections[index]] = [data.sections[index], data.sections[index + 1]]; }], ['Remove section', () => { if (confirm('Remove this section from the draft?')) data.sections.splice(index, 1); }]]) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary'; button.textContent = name; button.onclick = () => { operation(); markDirty(); renderStructured(data); }; buttons.append(button);
    }
    box.append(title.wrapper, paragraphs.wrapper, buttons); $('structured').append(box);
  });
  const add = document.createElement('button'); add.type = 'button'; add.textContent = 'Add section'; add.onclick = () => { data.sections.push({ title: 'New section', paragraphs: [''] }); markDirty(); renderStructured(data); }; $('structured').append(add);
}
function content() { return structured ? JSON.stringify(structured, null, 2) + '\n' : $('source').value; }
function showDocument(file, modified = false) {
  path = file.path; sha = file.sha || ''; $('document-title').textContent = path.split('/').pop(); $('editing').hidden = false; $('history').hidden = true; $('preview-panel').hidden = true;
  const isJSON = path.endsWith('.json'); $('source').hidden = isJSON; $('source-label').hidden = isJSON; $('structured').hidden = !isJSON;
  if (isJSON) renderStructured(JSON.parse(file.content)); else { structured = null; $('source').value = file.content; }
  markDirty(modified);
}
async function listFiles() {
  const { files } = await api('list'); $('files').replaceChildren();
  files.sort((a,b) => a.path.localeCompare(b.path)).forEach(file => { const option = document.createElement('option'); option.value = file.path; option.textContent = file.path.replace('src/content/', '').replace('src/data/', ''); $('files').append(option); });
  if ([...$('files').options].some(option => option.value === 'src/data/about.json')) $('files').value = 'src/data/about.json';
}
async function enter() { $('login').hidden = true; $('workspace').hidden = false; $('password').value = ''; await listFiles(); status('Signed in. Choose a page to start editing.'); }
$('login').addEventListener('submit', guard(async () => { const button = $('login').querySelector('button'); button.disabled = true; try { const data = await api('login', {}, { password: $('password').value }); csrf = data.csrf; await enter(); } finally { button.disabled = false; } }));
$('load').onclick = guard(async () => { if (dirty && !confirm('Discard unsaved changes?')) return; showDocument(await api('read', { path: $('files').value })); status('Page opened.'); });
$('source').addEventListener('input', () => markDirty());
$('logout').onclick = guard(async () => { if (dirty && !confirm('Sign out and discard unsaved changes?')) return; await api('logout', {}, {}); csrf = ''; markDirty(false); $('workspace').hidden = true; $('login').hidden = false; $('source').value = ''; $('structured').replaceChildren(); status('Signed out.'); });
$('new').onclick = () => { $('new-entry').hidden = !$('new-entry').hidden; };
$('new-entry').onsubmit = guard(async () => {
  if (dirty && !confirm('Discard unsaved changes?')) return;
  const kind = $('kind').value, slug = $('slug').value;
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(slug)) throw new Error('Use lowercase letters, numbers, and hyphens in the URL name.');
  const target = `src/content/${kind}/${slug}.md`;
  if ([...$('files').options].some(o => o.value.replace(/\.mdx$/, '.md') === target)) throw new Error('That URL name is already in use.');
  const yaml = value => JSON.stringify(value);
  const draft = `---\ntitle: ${yaml($('new-title').value)}\nstartDate: ${new Date().toISOString().slice(0,10)}\ndescription: ${yaml($('new-description').value)}\ntags: []\n${kind === 'projects' ? `subject: ${yaml($('subject').value)}\ninProgress: true\n` : 'draft: true\n'}---\n\nWrite your introduction here.\n`;
  showDocument({ path: target, content: draft }, true); $('new-entry').hidden = true; status('Draft created locally in the editor. Review it before publishing.');
});
$('preview-button').onclick = guard(async () => {
  const preview = $('preview'); preview.replaceChildren();
  const append = (tag, text) => { const element = document.createElement(tag); element.textContent = text; preview.append(element); };
  if (structured) {
    append('p', structured.lead || structured.headLine); append('p', structured.introduction || structured.headLine2);
    if (structured.sections) structured.sections.forEach(s => { append('h3', s.title); s.paragraphs.forEach(p => append('p', p)); });
    else structured.shortIntros.forEach(p => append('p', p));
  } else {
    const markdown = $('source').value.replace(/^---\n[\s\S]*?\n---\n/, '');
    markdown.split(/\n\s*\n/).filter(Boolean).forEach(block => { const heading = block.match(/^#{1,6}\s+(.+)$/); append(heading ? 'h3' : 'p', heading ? heading[1] : block); });
  }
  $('preview-panel').hidden = false; $('preview-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
});
$('save').onclick = guard(async () => {
  if (!dirty) { status('There are no unsaved changes.'); return; }
  if (!confirm('Publish these changes to your website? A new revision will be saved to GitHub.')) return;
  $('save').disabled = true;
  try { const result = await api('save', { path }, { content: content(), sha }); sha = result.sha; markDirty(false); status(result.message); await listFiles(); $('files').value = path; } finally { $('save').disabled = false; }
});
$('history-button').onclick = guard(async () => {
  const result = await api('history', { path }); $('revisions').replaceChildren();
  for (const revision of result.commits) {
    const button = document.createElement('button'); button.className = 'secondary'; button.textContent = `${new Date(revision.date).toLocaleString()} — ${revision.message}`;
    button.onclick = guard(async () => { if (dirty && !confirm('Replace unsaved changes with this revision?')) return; const file = await api('revision', { path, ref: revision.sha }); const currentSHA = sha; showDocument(file, true); sha = currentSHA; status('Earlier revision loaded for review. Publish to restore it.'); }); $('revisions').append(button);
  }
  $('history').hidden = false;
});
window.addEventListener('beforeunload', event => { if (dirty && document.getElementById('workspace')) { event.preventDefault(); event.returnValue = ''; } }, { signal });
(async () => { try { const session = await api('session'); csrf = session.csrf; await enter(); } catch (error) { $('login').hidden = false; status(error.message); } })();

}
document.addEventListener("astro:page-load", () => { if (document.getElementById("workspace")) initializeEditor(); });
