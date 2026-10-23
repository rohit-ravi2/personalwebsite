# Website rebuild — 23 October 2026

Before: production commit `87f2d93`. The approved About narrative was first saved in local commit `ebbb3b2`; this rebuild includes it.

## Changes

- A connected About narrative about illness, athletic identity, Vedanta, research motivation, and current work, with the existing professional portrait.
- A clearer homepage header, personal About teaser, selected work, visible technical focus, and role-focused experience.
- Revised titles and descriptions in 18 projects and nine blog posts; targeted prose cleanup preserves numerical findings and caveats.
- Subject navigation, visible project dates, and labels for work explicitly still in progress.
- Selected certificates followed by an expandable list; contact and Vivekananda quote retained.
- A still cream/forest presentation, accessible focus styles, and reduced motion.
- Correct UTC rendering of date-only content metadata.
- Owner editor for About/home text, projects, posts, experience and certificate content, new project/post drafts, text previews, revision restoration, and conflict protection.

## Verification

- `pnpm build`: 109 static pages built successfully.
- `pnpm test`: six authentication, API, metadata, MDX and content validation tests passed.
- All 38 existing content files passed the editor validator unchanged.
- Firefox desktop (1440px) and mobile (390px) checks: homepage, About, projects, writing, experience and editor; no horizontal overflow, broken images or JavaScript page errors; mobile navigation passed.
- Editor browser workflow tested with simulated API responses: sign-in, structured editing, preview, publish payloads, history, new post, sign-out and returning through navigation. No live repository writes were made by the browser tests.
- Screenshots here record the live site before and local build after. JSON files record the browser results.

## Activation

Public deployment uses the existing GitHub-to-Vercel integration. Owner editing fails closed until the server-side secrets in [EDITOR_SETUP.md](../../EDITOR_SETUP.md) are configured. The browser workflow tests do not substitute for a live authenticated production check after activation.

The unrelated pre-existing untracked checkpoint, `docs/`, and `enveda.md` are excluded from this rebuild.
