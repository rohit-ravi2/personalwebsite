import test from 'node:test';
import assert from 'node:assert/strict';
import handler from '../api/editor.mjs';
import { allowedPath, assertData, hashPassword, checkPassword, issueSession, readSession, COOKIE } from '../editor-lib/security.mjs';
const secret = 'test-session-secret-that-is-long-enough';
const markdown = '---\ntitle: "A page"\nstartDate: 2026-10-23\ndescription: "A description"\nsubject: "Neural decoding"\ntags: ["ML"]\n---\n\nA paragraph.\n';
test('password hashes and expiring signed sessions', () => {
  const hash = hashPassword('a strong test password');
  assert.equal(checkPassword('a strong test password', hash), true);
  assert.equal(checkPassword('wrong', hash), false);
  assert.equal(checkPassword('any', 'invalid'), false);
  const token = issueSession(secret, 100000);
  assert.ok(readSession(token, secret, 100001));
  assert.equal(readSession(token + 'x', secret, 100001), null);
  assert.equal(readSession(token.split('.')[0] + '.' + 'é'.repeat(43), secret, 100001), null);
  assert.equal(readSession(token, secret, 100000 + 5 * 60 * 60 * 1000), null);
});
test('allowlist blocks traversal, code files and other private content', () => {
  for (const path of ['src/data/about.json', 'src/data/home.json', 'src/content/projects/example.mdx', 'src/content/posts/series/entry.md', 'src/content/experiences/nyu.mdx', 'src/content/certifications/course.mdx']) assert.ok(allowedPath(path));
  for (const path of ['.env', 'src/pages/about.astro', 'src/content/posts/../config.ts', 'src/content/posts/%2e%2e/file.md', 'private/notes.md', 'src/content/posts/a/b/c.md']) assert.equal(allowedPath(path), false);
});
test('metadata validates types, real dates, duplicate YAML keys and aliases', () => {
  assert.doesNotThrow(() => assertData('src/content/projects/example.md', markdown));
  for (const content of [markdown.replace('2026-10-23', '2026-02-30'), markdown.replace('tags: ["ML"]', 'tags: ML'), markdown.replace('title: "A page"', 'title: "A page"\ntitle: "Duplicate"'), markdown.replace('title: "A page"', 'title: &a "A page"\nextra: *a')]) assert.throws(() => assertData('src/content/projects/example.md', content));
});
test('prose changes preserve executable MDX blocks including multiline bodies', () => {
  const old = markdown + '\nexport const Demo = () => {\n  doSomething();\n  return null;\n};\n\nMore prose.\n';
  assert.doesNotThrow(() => assertData('src/content/projects/example.mdx', old.replace('More prose.', 'New prose.'), old));
  assert.throws(() => assertData('src/content/projects/example.mdx', old.replace('doSomething();', 'runAnotherFunction();'), old));
  assert.throws(() => assertData('src/content/projects/example.mdx', markdown + '\n<script>alert(1)</script>\n', markdown));
  assert.throws(() => assertData('src/content/projects/example.mdx', old.replace('doSomething();', 'doSomething();\n  injectedFunction();'), old));
  assert.throws(() => assertData('src/content/projects/example.mdx', markdown + '\n[click](javascript:alert(1))\n', markdown));
});
test('JSON content is validated and treated as text', () => {
  const about = { lead: '<script>literal text</script>', introduction: 'Intro', sections: [{ title: 'Section', paragraphs: ['Text'] }] };
  assert.doesNotThrow(() => assertData('src/data/about.json', JSON.stringify(about)));
  assert.throws(() => assertData('src/data/about.json', JSON.stringify({ ...about, sections: [{ title: 'Section', paragraphs: [1] }] })));
});
function request(url, { method = 'GET', body, cookie, csrf, origin = 'https://www.rohitravi.com' } = {}) {
  return { url, method, body, headers: { origin, 'content-type': 'application/json', ...(cookie ? { cookie } : {}), ...(csrf ? { 'x-editor-csrf': csrf } : {}) }, socket: { remoteAddress: 'test' } };
}
async function invoke(req) {
  const res = { headers: {}, code: 0, setHeader(k, v) { this.headers[k] = v; }, status(code) { this.code = code; return this; }, json(data) { this.data = data; return this; } };
  await handler(req, res); return res;
}
test('API requires configuration, authentication, origin and CSRF before GitHub writes', async () => {
  const previous = { ...process.env }, originalFetch = globalThis.fetch;
  try {
    delete process.env.EDITOR_SESSION_SECRET;
    assert.equal((await invoke(request('/api/editor'))).code, 503);
    Object.assign(process.env, { EDITOR_SESSION_SECRET: secret, EDITOR_PASSWORD_HASH: hashPassword('test password'), EDITOR_GITHUB_TOKEN: 'test-not-a-real-token' });
    assert.equal((await invoke(request('/api/editor?action=list'))).code, 401);
    const badOrigin = await invoke(request('/api/editor?action=login', { method: 'POST', body: { password: 'test password' }, origin: 'https://other.example' }));
    assert.equal(badOrigin.code, 403);
    const login = await invoke(request('/api/editor?action=login', { method: 'POST', body: { password: 'test password' } }));
    assert.equal(login.code, 200);
    assert.match(login.headers['Set-Cookie'], /HttpOnly; Secure; SameSite=Strict/);
    const cookie = login.headers['Set-Cookie'].split(';')[0], csrf = login.data.csrf;
    assert.equal((await invoke(request('/api/editor?action=save&path=src/data/about.json', { method: 'POST', cookie, body: {} }))).code, 403);
    assert.equal((await invoke(request('/api/editor?action=read&path=.env', { cookie }))).code, 400);
    const calls = [];
    globalThis.fetch = async (url, init) => {
      calls.push([url, init]);
      if (init.method === 'PUT') return new Response(JSON.stringify({ content: { sha: 'newsha' }, commit: { sha: 'commitsha' } }));
      if (url.includes('/commits?')) return new Response(JSON.stringify([{ sha: 'a'.repeat(40), commit: { message: 'Edit page', author: { date: '2026-10-23' } } }]));
      return new Response(JSON.stringify({ type: 'file', sha: 'oldsha', content: Buffer.from(markdown).toString('base64') }));
    };
    const mismatch = await invoke(request('/api/editor?action=save&path=src/content/projects/example.mdx', { method: 'POST', cookie, csrf, body: { sha: 'stale', content: markdown } }));
    assert.equal(mismatch.code, 409); assert.equal(calls.some(c => c[1].method === 'PUT'), false);
    const save = await invoke(request('/api/editor?action=save&path=src/content/projects/example.mdx', { method: 'POST', cookie, csrf, body: { sha: 'oldsha', content: markdown.replace('A paragraph.', 'An edited paragraph.') } }));
    assert.equal(save.code, 200); assert.equal(save.data.sha, 'newsha');
    const write = calls.find(c => c[1].method === 'PUT'); assert.equal(JSON.parse(write[1].body).sha, 'oldsha');
    const history = await invoke(request('/api/editor?action=history&path=src/content/projects/example.mdx', { cookie })); assert.equal(history.data.commits[0].message, 'Edit page');
    const logout = await invoke(request('/api/editor?action=logout', { method: 'POST', cookie, csrf, body: {} })); assert.match(logout.headers['Set-Cookie'], /Max-Age=0/);
    assert.equal((await invoke(request('/api/editor?action=login', { method: 'POST', body: null }))).code, 400);
  } finally { globalThis.fetch = originalFetch; for (const key of Object.keys(process.env)) if (!(key in previous)) delete process.env[key]; Object.assign(process.env, previous); }
});

test('skill groups validate editable content and block executable links', () => {
  const data = { groups: [{title: 'Programming', items: ['Python', 'Java', 'C++'], project: {label: 'Projects', href: '/projects'}}] };
  assert.equal(allowedPath('src/data/skills.json'), true);
  assert.doesNotThrow(() => assertData('src/data/skills.json', JSON.stringify(data)));
  assert.throws(() => assertData('src/data/skills.json', JSON.stringify({groups: []})));
  data.groups[0].project.href = 'javascript:alert(1)';
  assert.throws(() => assertData('src/data/skills.json', JSON.stringify(data)));
});

test('guided-project metadata validates certificates and local PDF paths', () => {
  const metadata = `---\ntitle: Guided projects\nissuer: Project Network\nissueDate: 2025-05-19\nguidedProjects:\n  - group: Model architectures\n    title: Siamese networks\n    description: Learning similarities between inputs.\n    frameworks: [PyTorch]\n    verifyUrl: https://coursera.org/verify/example\n    pdfPath: /certifications/example.pdf\n---\n\nGuided project work.\n`;
  const path = 'src/content/certifications/example.md';
  assert.doesNotThrow(() => assertData(path, metadata));
  assert.throws(() => assertData(path, metadata.replace('https://coursera.org/verify/example', 'javascript:alert(1)')));
  assert.throws(() => assertData(path, metadata.replace('/certifications/example.pdf', '/../private/example.pdf')));
  assert.throws(() => assertData(path, metadata.replace('frameworks: [PyTorch]', 'frameworks: []')));
});
