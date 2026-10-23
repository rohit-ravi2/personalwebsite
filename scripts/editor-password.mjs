import { hashPassword } from '../editor-lib/security.mjs';
import { Writable } from 'node:stream';
import { createInterface } from 'node:readline';
const silent = new Writable({ write(_chunk, _encoding, done) { done(); } });
if (!process.stdin.isTTY) { process.stderr.write('Run this helper in an interactive terminal.\n'); process.exit(1); }
const prompt = createInterface({ input: process.stdin, output: silent, terminal: true });
process.stdout.write('Choose an owner password (at least 16 characters; input hidden): ');
prompt.question('', password => {
  prompt.close(); process.stdout.write('\n');
  if (password.length < 16) { process.stderr.write('Use at least 16 characters.\n'); process.exitCode = 1; return; }
  process.stdout.write(hashPassword(password) + '\n');
});
