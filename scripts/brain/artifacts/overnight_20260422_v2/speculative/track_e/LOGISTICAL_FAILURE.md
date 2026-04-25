# Track E (speculative) — LOGISTICAL_FAILURE

**EXPLORATORY — not yet rigorous.**

## Status

**LOGISTICAL_FAILURE: C. elegans lineage data not accessible via WebFetch.**

## What was attempted

1. Local filesystem search (`~/Desktop/...`, `~/Desktop/website/...`) for any file containing "lineage" or "sulston" — **no results**.
2. WebSearch identified multiple data sources:
   - Packer & Zhu 2019 (Science) — ~1068 annotated branches, behind Science paywall/supplementary
   - Tintori 2016 — Developmental Cell, supplementary
   - Sulston 1983 — WormAtlas HTML (not machine-readable CSV)
   - `livinlrg/C.elegans_C.briggsae_Embryo_Single_Cell` GitHub repo — contains `cell_data.txt` with lineage + cell_type columns but stored via Git LFS
3. Direct `curl` attempt on `raw.githubusercontent.com/livinlrg/C.elegans_C.briggsae_Embryo_Single_Cell/main/cell_data.txt` — returned HTTP 404 (file likely on LFS pointer not raw path).
4. WebFetch on `raw.githubusercontent.com` — blocked by domain verification.
5. WebFetch on Packer 2019 supplementary — not attempted (large file, unlikely to succeed).

## Why no toy/synthetic substitute

The spec requires pre-specified PASS/FAIL thresholds on real data:
- PASS: GNCA test accuracy > baseline by ≥ 10 percentage points
- FAIL: GNCA test accuracy ≤ baseline + 10pp

Building a hand-coded toy lineage from memorized Sulston structure would not satisfy this — the test would reduce to "can GNCA memorize a small hand-coded dataset," which doesn't support the scientific hypothesis the track was designed to probe ("do local update rules capture lineage-fate structure").

Fabricating a test would violate the explicit guardrail: "Do not speculate about novelty, impact, or publishability. Report measurements and pre-specified classifications only."

## What would unblock

- User downloads `cell_data.txt` from `livinlrg` repo via Git LFS locally, or
- User provides Packer 2019 supplementary tables (Tables S2, S5 contain cell-fate mappings), or
- User provides the original Sulston 1983 enumeration in a parseable format

Once data is available locally, the GNCA + baseline infrastructure can be written and run in ~2 hours.

## Infrastructure partially prepared

- Track directory: `artifacts/overnight_20260422_v2/speculative/track_e/`
- Architecture outline deferred to follow-up session when data is in place.

## Not attempted (time budget)

- Dryad mirror fetch
- bioRxiv supplementary fetch (Packer 2019 preprint)
- WormAtlas HTML-to-CSV scraping

These are candidate recovery options for a dedicated follow-up session.
