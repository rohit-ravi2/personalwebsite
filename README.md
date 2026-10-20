# rohitravi.com

Source for my personal website and portfolio, live at https://www.rohitravi.com.

Built with Astro, React, TypeScript, and Tailwind CSS. Hosted on Vercel — pushes to `main` auto-deploy to production.

## What's here

- `src/content/projects/` — one MDX page per project; most embed an interactive demo
- `src/components/react/` — the demos (ONNX Runtime classifiers, Three.js substrate viewer, connectome explorer, simulators)
- `src/content/posts/` — blog posts
- `scripts/figures/` — scripts that regenerate figures from each project's saved result files
- `AnestheticSimulator/`, `scripts/brain/` — research code behind the anesthesia and substrate pages

## Local development

```bash
pnpm install
pnpm dev    # http://localhost:4312
```

## License

Code is MIT (see `LICENSE`). Content under `src/content/` and assets under `public/` are © Rohit Ravi — all rights reserved unless noted.
