# Owner editing

The `/editor` page lets Rohit edit the homepage introduction, About narrative, existing projects, blog posts, experience entries and certification descriptions. It also creates Markdown project and blog entries. The homepage has a discreet owner sign-in link. The link is an entry point, not the security boundary.

## One-time Vercel setup

The editor stays locked until all required server-side environment variables are configured in the existing Vercel project:

| Variable | Value |
| --- | --- |
| `EDITOR_PASSWORD_HASH` | scrypt hash generated below; do not enter a plaintext password here |
| `EDITOR_SESSION_SECRET` | independently generated random secret, at least 32 characters |
| `EDITOR_GITHUB_TOKEN` | fine-grained GitHub token restricted to `rohit-ravi2/personalwebsite`, repository Contents read/write access |
| `EDITOR_ORIGIN` | `https://www.rohitravi.com` (the canonical origin, without trailing slash) |
| `EDITOR_GITHUB_REPOSITORY` | `rohit-ravi2/personalwebsite` (default) |
| `EDITOR_GITHUB_BRANCH` | `main` (default) |

Generate the password hash on a trusted machine using the hidden-input helper:

```sh
node scripts/editor-password.mjs
```

It prompts without echoing the password and prints its salted hash. Generate the independent session secret with `openssl rand -hex 32`. Store the hash, secret and token in Vercel environment settings. Never paste the token or password into source files. Redeploy after changing environment variables. The editor API requires HTTPS, including for local end-to-end testing; unit/API tests run without external services.

## Editing and publishing

1. Open the homepage owner link or `/editor` and sign in.
2. Choose an existing page, or create a new project/blog draft.
3. About sections can be edited, added, removed and reordered. Homepage text uses structured fields. Article pages use Markdown and opening YAML metadata.
4. Preview changes. Preview intentionally renders plain text and headings, not executable components or raw HTML.
5. Publish to create a GitHub commit. The existing GitHub–Vercel integration starts a deployment. The editor confirms the commit, not deployment success; the public site changes only after a successful build.
6. Revision history can load one of the last 15 file revisions. Publishing an earlier version restores it as a new commit, preserving history.

New blog entries begin with `draft: true`; change it to `false` before publishing a finished article. The current site still builds draft URLs with a visible draft label. Set a project’s `subject` to an existing subject and `inProgress` as appropriate. Markdown image URLs can point to existing hosted assets; file uploads are not part of this release.

Existing MDX imports, component blocks and expression blocks are protected from modification through the editor. Their surrounding prose and frontmatter remain editable. Code changes and new interactive demos continue to go through development. An outdated editor view cannot overwrite a newer GitHub revision: reload on a conflict. Unsaved drafts remain in the current browser tab and are not saved automatically.

## Security and operational behavior

- Password verification uses salted scrypt; four-hour sessions are signed and stored in Secure, HttpOnly, SameSite=Strict, `__Host-` cookies. Rotating the session secret invalidates existing sessions.
- Mutations require the configured Origin and a session-bound CSRF header. All API responses are private/no-store.
- Content writes are restricted to the two content JSON files and the public content collections. Source code, credentials, private notes and arbitrary paths cannot be accessed.
- Login throttling uses per-instance counters and a delay after failure. In Vercel this is not a global distributed rate limit. Configure a Vercel Firewall rate-limit rule for POST `/api/editor?action=login` for additional distributed protection; keep the owner password long and unique.
- Publishing writes to `main` using a repository-limited token. The hidden UI is never relied on to protect content. Removing `EDITOR_GITHUB_TOKEN` disables the editor immediately after environment redeployment.

Run `node --test tests/editor.test.mjs` for authentication, CSRF, path, metadata, MDX preservation and GitHub-write behavior checks. The API must be checked on Vercel after deployment because Astro’s static preview server does not execute Vercel Functions.

Implementation references: [Vercel Node.js functions](https://vercel.com/docs/functions/runtimes/node-js), [GitHub repository content API](https://docs.github.com/en/rest/repos/contents).
