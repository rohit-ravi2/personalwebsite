import { createProcessor } from '@mdx-js/mdx';
import { parseDocument } from 'yaml';
import { createHmac, randomBytes, scryptSync, timingSafeEqual } from 'node:crypto';
export const COOKIE = '__Host-rohit-editor';
const TTL = 60 * 60 * 4;
export function hashPassword(password, salt = randomBytes(16).toString('hex')) {
  return `scrypt:${salt}:${scryptSync(password, salt, 64).toString('hex')}`;
}
export function checkPassword(password, hash) {
  if (typeof password !== 'string' || password.length > 512) return false;
  const [kind, salt, value] = String(hash).split(':');
  if (kind !== 'scrypt' || !/^[a-f0-9]{32}$/.test(salt) || !/^[a-f0-9]{128}$/.test(value)) return false;
  return timingSafeEqual(scryptSync(password, salt, 64), Buffer.from(value, 'hex'));
}
const mac = (text, secret) => createHmac('sha256', secret).update(text).digest('base64url');
export function issueSession(secret, now = Date.now()) {
  const data = Buffer.from(JSON.stringify({ expires: Math.floor(now / 1000) + TTL, csrf: randomBytes(24).toString('hex') })).toString('base64url');
  return `${data}.${mac(data, secret)}`;
}
export function readSession(token, secret, now = Date.now()) {
  if (!token || token.length > 1024) return null;
  const [data, signature, extra] = token.split('.');
  if (extra || !data || !signature) return null;
  const expected = mac(data, secret);
  if (Buffer.byteLength(signature) !== Buffer.byteLength(expected) || !timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) return null;
  try {
    const session = JSON.parse(Buffer.from(data, 'base64url').toString());
    return session.expires > now / 1000 && /^[a-f0-9]{48}$/.test(session.csrf) ? session : null;
  } catch { return null; }
}
export function cookieValue(req) {
  return (req.headers.cookie || '').split(';').map(s => s.trim()).find(s => s.startsWith(COOKIE + '='))?.slice(COOKIE.length + 1);
}
export function sessionCookie(token, clear = false) {
  return `${COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=${clear ? 0 : TTL}`;
}
export function allowedPath(path) {
  if (typeof path !== 'string') return false;
  if (['src/data/about.json', 'src/data/home.json'].includes(path)) return true;
  return /^src\/content\/(projects|posts|experiences|certifications)\/[a-z0-9][a-z0-9-]*(?:\/[a-z0-9][a-z0-9-]*)?\.mdx?$/.test(path);
}
export function assertData(path, content, previous = '') {
  if (!allowedPath(path) || typeof content !== 'string' || Buffer.byteLength(content) > 350000) throw new Error('Invalid content or path.');
  if (path.endsWith('.json')) {
    let data;
    try { data = JSON.parse(content); } catch { throw new Error('Content must be valid JSON.'); }
    const str = (v, max = 20000) => typeof v === 'string' && v.length <= max;
    if (path.endsWith('about.json')) {
      if (!str(data.lead) || !str(data.introduction) || !Array.isArray(data.sections) || data.sections.length > 30 || !data.sections.every(s => str(s.title, 200) && Array.isArray(s.paragraphs) && s.paragraphs.length <= 50 && s.paragraphs.every(p => str(p)))) throw new Error('Invalid About content.');
    } else if (!str(data.headLine) || !str(data.headLine2) || !Array.isArray(data.shortIntros) || data.shortIntros.length > 8 || !data.shortIntros.every(p => str(p, 500))) throw new Error('Invalid homepage content.');
    if (path.endsWith('home.json') && data.bio !== undefined && !str(data.bio)) throw new Error('Invalid greeting panel bio.');
    return;
  }
  // Parse MDX, then preserve complete executable nodes (including multiline JS,
  // JSX attributes and expression bodies). Paragraphs inside a component are
  // conservatively protected with that component; ordinary prose stays editable.
  const code = source => {
    const body = source.replace(/^---\n[\s\S]*?\n---(?:\n|$)/, '');
    let tree;
    try { tree = createProcessor().parse(body); } catch { throw new Error('The Markdown contains invalid component syntax.'); }
    const nodes = [];
    const visit = node => {
      if (node.type.startsWith('mdx')) { nodes.push(body.slice(node.position.start.offset, node.position.end.offset)); return; }
      for (const child of node.children || []) visit(child);
    };
    visit(tree); return nodes;
  };
  if (JSON.stringify(code(previous)) !== JSON.stringify(code(content))) throw new Error('Component and code blocks must stay unchanged. Edit the prose around them.');
  const unsafe = s => s.match(/(?:javascript|vbscript|data)\s*:/gi) || [];
  if (JSON.stringify(unsafe(previous)) !== JSON.stringify(unsafe(content))) throw new Error('Executable links are not allowed.');
  const match = content.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  if (!match) throw new Error('Include page metadata between opening --- lines.');
  const document = parseDocument(match[1], { uniqueKeys: true });
  if (document.errors.length) throw new Error('Page metadata is invalid YAML.');
  let data;
  try { data = document.toJS({ maxAliasCount: 0 }); } catch { throw new Error('Metadata aliases are not supported.'); }
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Page metadata must contain named fields.');
  const requiredString = key => { if (typeof data[key] !== 'string' || !data[key].trim() || data[key].length > 5000) throw new Error(`Include a valid ${key} in the page metadata.`); };
  const date = key => {
    if (typeof data[key] !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(data[key]) || !Number.isFinite(Date.parse(data[key])) || new Date(data[key]).toISOString().slice(0,10) !== data[key]) throw new Error(`Use a valid YYYY-MM-DD date for ${key}.`);
  };
  if (data.slug !== undefined || data.layout !== undefined) throw new Error('URL names and page layouts are managed separately from page content.');
  requiredString('title');
  if (path.includes('/certifications/')) { requiredString('issuer'); date('issueDate'); }
  else date('startDate');
  if (data.endDate !== undefined) date('endDate');
  if (/\/(projects|posts)\//.test(path)) requiredString('description');
  if (path.includes('/projects/')) requiredString('subject');
  if (path.includes('/experiences/')) requiredString('company');
  if (data.tags !== undefined && (!Array.isArray(data.tags) || !data.tags.every(t => typeof t === 'string'))) throw new Error('Tags must be a list of text values.');
  for (const key of ['inProgress', 'draft']) if (data[key] !== undefined && typeof data[key] !== 'boolean') throw new Error(`${key} must be true or false.`);
  if (data.image !== undefined && (!data.image || typeof data.image.url !== 'string' || typeof data.image.alt !== 'string')) throw new Error('Images need a URL and alt text.');
  for (const key of ['canonical', 'verifyUrl', 'pdfPath']) if (data[key] !== undefined && typeof data[key] !== 'string') throw new Error(`${key} must be text.`);
  for (const key of ['courseCount', 'order']) if (data[key] !== undefined && (typeof data[key] !== 'number' || !Number.isFinite(data[key]))) throw new Error(`${key} must be a number.`);
}
