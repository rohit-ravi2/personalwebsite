import { allowedPath, assertData, checkPassword, cookieValue, issueSession, readSession, sessionCookie } from '../editor-lib/security.mjs';
const failures = new Map();
function error(status, message) { const e = new Error(message); e.status = status; throw e; }
function config() {
  const secret = process.env.EDITOR_SESSION_SECRET;
  const passwordHash = process.env.EDITOR_PASSWORD_HASH;
  const token = process.env.EDITOR_GITHUB_TOKEN;
  const origin = process.env.EDITOR_ORIGIN || 'https://www.rohitravi.com';
  try { if (new URL(origin).origin !== origin || !origin.startsWith('https://')) error(503, 'Owner editing has not been configured yet.'); } catch { error(503, 'Owner editing has not been configured yet.'); }
  const repository = process.env.EDITOR_GITHUB_REPOSITORY || 'rohit-ravi2/personalwebsite';
  if (!secret || secret.length < 32 || !/^scrypt:[a-f0-9]{32}:[a-f0-9]{128}$/.test(passwordHash || '') || !token || !/^[\w.-]+\/[\w.-]+$/.test(repository)) error(503, 'Owner editing has not been configured yet.');
  return { secret, passwordHash, token, origin, repository, branch: process.env.EDITOR_GITHUB_BRANCH || 'main' };
}
async function github(cfg, endpoint, init = {}) {
  const response = await fetch(`https://api.github.com/repos/${cfg.repository}/${endpoint}`, {
    ...init, signal: AbortSignal.timeout(15000),
    headers: { Authorization: `Bearer ${cfg.token}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', 'Content-Type': 'application/json', ...init.headers },
  });
  if (response.status === 404) return null;
  if ([409, 422].includes(response.status)) error(409, 'This page changed since you opened it. Reload before saving.');
  if (!response.ok) error(502, 'The content service could not complete the request.');
  return response.json();
}
const encodePath = path => path.split('/').map(encodeURIComponent).join('/');
async function readFile(cfg, path, ref = cfg.branch) {
  const file = await github(cfg, `contents/${encodePath(path)}?ref=${encodeURIComponent(ref)}`);
  if (!file || file.type !== 'file') return null;
  return { path, sha: file.sha, content: Buffer.from(file.content, 'base64').toString('utf8') };
}
async function parseBody(req) {
  if (Number(req.headers['content-length']) > 500000) error(413, 'Content is too large.');
  if (!String(req.headers['content-type'] || '').startsWith('application/json')) error(415, 'Use JSON requests.');
  if (req.body === null || Array.isArray(req.body)) error(400, 'Invalid JSON.');
  if (req.body && typeof req.body === 'object' && !Array.isArray(req.body)) return req.body;
  let raw = typeof req.body === 'string' ? req.body : '';
  if (!raw) for await (const chunk of req) { raw += chunk; if (Buffer.byteLength(raw) > 500000) error(413, 'Content is too large.'); }
  try { const data = JSON.parse(raw); if (!data || typeof data !== 'object' || Array.isArray(data)) error(400, 'Invalid JSON.'); return data; } catch { error(400, 'Invalid JSON.'); }
}
export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  try {
    const cfg = config();
    const method = req.method;
    if (!['GET', 'POST'].includes(method)) { res.setHeader('Allow', 'GET, POST'); error(405, 'Method not allowed.'); }
    const url = new URL(req.url, cfg.origin);
    const action = url.searchParams.get('action') || 'session';
    if (method === 'POST' && req.headers.origin !== cfg.origin) error(403, 'Request origin is not allowed.');
    const session = readSession(cookieValue(req), cfg.secret);
    if (action === 'login' && method === 'POST') {
      const key = String(req.headers['x-forwarded-for'] || req.socket?.remoteAddress || 'unknown').split(',')[0];
      const now = Date.now();
      for (const [ip, entry] of failures) if (entry.until < now) failures.delete(ip);
      const attempt = failures.get(key);
      if (attempt?.count >= 5 && attempt.until > now) error(429, 'Too many attempts. Try again in fifteen minutes.');
      const body = await parseBody(req);
      if (!checkPassword(body.password, cfg.passwordHash)) {
        if (failures.size > 10000) error(429, 'Please try again later.');
        failures.set(key, { count: (attempt?.count || 0) + 1, until: now + 15 * 60 * 1000 });
        await new Promise(resolve => setTimeout(resolve, 800));
        error(401, 'Incorrect password.');
      }
      failures.delete(key);
      const token = issueSession(cfg.secret);
      res.setHeader('Set-Cookie', sessionCookie(token));
      return res.status(200).json({ csrf: readSession(token, cfg.secret).csrf });
    }
    if (!session) error(401, 'Sign in to edit.');
    if (method === 'POST' && req.headers['x-editor-csrf'] !== session.csrf) error(403, 'Session verification failed. Reload the editor.');
    if (action === 'session' && method === 'GET') return res.status(200).json({ csrf: session.csrf });
    if (action === 'logout' && method === 'POST') { res.setHeader('Set-Cookie', sessionCookie('', true)); return res.status(200).json({ ok: true }); }
    if (action === 'list' && method === 'GET') {
      const tree = await github(cfg, `git/trees/${encodeURIComponent(cfg.branch)}?recursive=1`);
      if (!tree || tree.truncated) error(502, 'Unable to list all content.');
      return res.status(200).json({ files: tree.tree.filter(f => f.type === 'blob' && allowedPath(f.path)).map(f => ({ path: f.path })) });
    }
    const path = url.searchParams.get('path');
    if (!allowedPath(path)) error(400, 'This content path is not editable.');
    if (action === 'read' && method === 'GET') {
      const file = await readFile(cfg, path);
      if (!file) error(404, 'Page not found.');
      return res.status(200).json(file);
    }
    if (action === 'history' && method === 'GET') {
      const commits = await github(cfg, `commits?sha=${encodeURIComponent(cfg.branch)}&path=${encodeURIComponent(path)}&per_page=15`);
      return res.status(200).json({ commits: (commits || []).map(c => ({ sha: c.sha, message: c.commit.message.split('\n')[0], date: c.commit.author.date })) });
    }
    if (action === 'revision' && method === 'GET') {
      const ref = url.searchParams.get('ref');
      if (!/^[a-f0-9]{40}$/.test(ref || '')) error(400, 'Invalid revision.');
      const file = await readFile(cfg, path, ref);
      if (!file) error(404, 'Revision not found.');
      return res.status(200).json(file);
    }
    if (action === 'save' && method === 'POST') {
      const body = await parseBody(req);
      const current = await readFile(cfg, path);
      if (current && body.sha !== current.sha || !current && body.sha) error(409, 'This page changed since you opened it. Reload before saving.');
      if (!current && path.endsWith('.md')) {
        if (await readFile(cfg, path + 'x')) error(409, 'An entry already uses that URL name.');
        const folder = await github(cfg, `contents/${encodePath(path.slice(0, -3))}?ref=${encodeURIComponent(cfg.branch)}`);
        if (Array.isArray(folder) || folder?.type === 'dir') error(409, 'A collection already uses that URL name.');
      }
      if (!current && !/^src\/content\/(posts|projects)\/.+\.md$/.test(path)) error(400, 'New entries must be Markdown files.');
      try { assertData(path, body.content, current?.content || ''); } catch (e) { error(400, e.message); }
      const saved = await github(cfg, `contents/${encodePath(path)}`, { method: 'PUT', body: JSON.stringify({ message: `Edit ${path.split('/').pop()} through owner editor`, content: Buffer.from(body.content).toString('base64'), branch: cfg.branch, ...(current ? { sha: current.sha } : {}) }) });
      return res.status(200).json({ sha: saved.content.sha, commit: saved.commit.sha, message: 'Saved to GitHub. The website deployment will start automatically.' });
    }
    error(400, 'Unknown editor action.');
  } catch (e) { return res.status(e.status || 500).json({ error: e.status ? e.message : 'The editor could not complete this request.' }); }
}
