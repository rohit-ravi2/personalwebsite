#!/usr/bin/env bash
# Comprehensive data-pull manifest. Not meant to be run end-to-end —
# each block is self-contained so we can start / monitor individual
# groups. Stored under scripts/ only for reference; actual execution
# is via nohup curl processes kicked off from the orchestrator.

set -e

DATA=/home/rohit/Desktop/website/personalwebsite/data/external

# ─── Round A — quick small wins (all < 15 GB each, < 35 GB total) ───
#
# NeuroPAL atlas (DANDI 000715): entire dandiset is 0.84 GB across 10 assets.
# Stern full ablation feature file (Figshare 5086825): 10.5 GB.
# Remaining 7 rhythm-generator zips (Zenodo 5089834):
#   IRsurgery 1–8 (skipping 5 which we already have), Optogenetics 6 review.

# ─── Round B — Atanas 8 more worms (~200 GB) ───
# DANDI 000776 assets 3..10 (already have 1,2).

# ─── Round C — Randi atlas selective assets (~150 GB of 4 TB) ───
# DANDI 001075: grab 5 smallest assets.

# ─── Round D — unexplored DANDI C. elegans dandisets ───
# 000449, 000472, 000541, 000565, 000680, 000692, 000714, 000981

# ─── Round E — Zenodo additional worm corpora + Hallinen ───
echo "manifest only — run individual blocks from the orchestrator"
