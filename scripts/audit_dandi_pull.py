"""Audit local DANDI downloads against DANDI API expected sizes.

Strategy:
- Fetch all assets for each target dandiset via the DANDI REST API.
- For each on-disk file, identify its asset by (1) exact size match,
  then (2) sub-ID parsed from filename. If still ambiguous, ask for help.
- Classify each asset as complete / truncated / missing.
- Write a JSON manifest (_pull_audit.json) with a download queue.

No non-stdlib deps. Run with any python3 ≥ 3.10.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/home/rohit/Desktop/website/personalwebsite/data/external")

# Which dandisets we care about and how many assets to keep (None = all).
# Ordering: by asset path alphabetical. "limit" caps at the first N assets.
TARGETS = {
    "000776": {"subdir": "atanas2023",            "limit": 10, "new_name": "atanas",       "subs_filter": None},
    "000472": {"subdir": "dandi_000472",          "limit": None, "new_name": "dandi_000472", "subs_filter": None},
    "000541": {"subdir": "dandi_000541",          "limit": None, "new_name": "dandi_000541", "subs_filter": None},
    "000565": {"subdir": "dandi_000565",          "limit": None, "new_name": "dandi_000565", "subs_filter": None},
    "000692": {"subdir": "dandi_000692",          "limit": None, "new_name": "dandi_000692", "subs_filter": None},
    "000714": {"subdir": "dandi_000714",          "limit": None, "new_name": "dandi_000714", "subs_filter": None},
    "000715": {"subdir": "dandi_000715_neuropal", "limit": None, "new_name": "neuropal",    "subs_filter": None},
    "000981": {"subdir": "dandi_000981",          "limit": None, "new_name": "dandi_000981", "subs_filter": None},
    # Randi dandiset is 4TB total — previous session curated 5 specific
    # "imaging" assets (one per sub). Keep exactly those.
    "001075": {"subdir": "dandi_001075_randi",    "limit": None, "new_name": "randi",
               "subs_filter": {"sub-24", "sub-25", "sub-29", "sub-56", "sub-59"},
               "path_contains": "_desc-imaging_"},
}

SUB_RE = re.compile(r"(sub-[A-Za-z0-9\-]+)")

def fetch_all_assets(dandiset_id: str) -> list[dict]:
    url = (f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/"
           f"versions/draft/assets/?page_size=100")
    out = []
    while url:
        with urllib.request.urlopen(url, timeout=30) as r:
            page = json.load(r)
        out.extend(page["results"])
        url = page.get("next")
    out.sort(key=lambda a: a["path"])
    return out

def sub_of(path: str) -> str | None:
    """Parse the sub-ID from a path or filename. DANDI paths start with sub-X/."""
    m = SUB_RE.search(path)
    return m.group(1) if m else None

def canonical_name(scheme: str, asset_path: str) -> str:
    """Filename for NEW downloads."""
    sub = sub_of(asset_path) or "noid"
    basename = asset_path.rsplit("/", 1)[-1]
    if scheme == "atanas":
        return f"atanas_{sub}.nwb"
    if scheme == "randi":
        return f"randi_{sub}.nwb"
    if scheme == "neuropal":
        return basename
    if scheme.startswith("dandi_"):
        did = scheme.replace("dandi_", "")
        # mimic prior convention: "{did}_{flatpath}"
        return f"{did}_{asset_path.replace('/', '_')}"
    return basename

def audit():
    report = {
        "complete": [],
        "truncated": [],
        "missing": [],
        "unexpected": [],
        "download_queue": [],
    }

    for dandiset_id, conf in TARGETS.items():
        subdir = conf["subdir"]
        local_dir = ROOT / subdir
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            assets = fetch_all_assets(dandiset_id)
        except Exception as e:
            print(f"[WARN] fetch {dandiset_id}: {e}", file=sys.stderr)
            continue

        if conf.get("subs_filter"):
            assets = [a for a in assets if sub_of(a["path"]) in conf["subs_filter"]]
        if conf.get("path_contains"):
            assets = [a for a in assets if conf["path_contains"] in a["path"]]
        if conf.get("limit") and not conf.get("subs_filter"):
            assets = assets[: conf["limit"]]

        on_disk = {p.name: p.stat().st_size for p in local_dir.glob("*.nwb")
                   if not p.name.endswith(".partial")}

        # Build lookups
        size_index: dict[int, list[dict]] = {}
        sub_index: dict[str, list[dict]] = {}
        for a in assets:
            size_index.setdefault(a["size"], []).append(a)
            s = sub_of(a["path"])
            if s:
                sub_index.setdefault(s, []).append(a)

        # Claim each asset: which local file (if any) covers it?
        claimed_by = {}  # asset_id -> local filename

        # Pass 1: exact size match (any asset with matching size claims the file)
        size_disk: dict[int, list[str]] = {}
        for fn, sz in on_disk.items():
            size_disk.setdefault(sz, []).append(fn)

        for sz, fnames in size_disk.items():
            matching_assets = size_index.get(sz, [])
            # Pair up greedily — if exactly one file + one asset, claim
            for fn, a in zip(fnames, matching_assets):
                claimed_by[a["asset_id"]] = fn

        # Pass 2: sub-ID match for unclaimed partials
        claimed_fn = set(claimed_by.values())
        for fn, sz in on_disk.items():
            if fn in claimed_fn:
                continue
            sub = sub_of(fn)
            if not sub:
                continue
            candidates = [a for a in sub_index.get(sub, [])
                          if a["asset_id"] not in claimed_by]
            if len(candidates) == 1:
                claimed_by[candidates[0]["asset_id"]] = fn

        # Classify each asset
        for a in assets:
            fn = claimed_by.get(a["asset_id"])
            sz_local = on_disk.get(fn) if fn else None
            rec = {
                "dandiset": dandiset_id,
                "subdir": subdir,
                "asset_id": a["asset_id"],
                "asset_path": a["path"],
                "expected_size": a["size"],
                "local_filename": fn,
                "local_size": sz_local,
                "canonical_name": canonical_name(conf["new_name"], a["path"]),
            }
            if fn is None:
                report["missing"].append(rec)
                report["download_queue"].append(rec)
            elif sz_local == a["size"]:
                report["complete"].append(rec)
            else:
                report["truncated"].append(rec)
                report["download_queue"].append(rec)

        # Unexpected = on-disk files not claimed
        for fn in on_disk:
            if fn not in claimed_by.values():
                report["unexpected"].append({
                    "subdir": subdir, "filename": fn,
                    "local_size": on_disk[fn],
                })

    return report

def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def print_summary(r):
    print("=" * 72)
    print(f"DANDI pull audit")
    print("=" * 72)
    print(f"  complete:   {len(r['complete'])}")
    print(f"  truncated:  {len(r['truncated'])}")
    print(f"  missing:    {len(r['missing'])}")
    print(f"  unexpected: {len(r['unexpected'])}")

    if r["truncated"]:
        print("\n--- TRUNCATED (delete + redownload) ---")
        for rec in r["truncated"]:
            pct = 100 * rec["local_size"] / rec["expected_size"]
            print(f"  {rec['subdir']}/{rec['local_filename']}: "
                  f"{human(rec['local_size'])}/{human(rec['expected_size'])} "
                  f"({pct:.1f}%)")

    if r["unexpected"]:
        print("\n--- UNEXPECTED (not matched to any asset) ---")
        for rec in r["unexpected"]:
            print(f"  {rec['subdir']}/{rec['filename']}: {human(rec['local_size'])}")

    if r["missing"]:
        print(f"\n--- MISSING ({len(r['missing'])}) ---")
        for rec in r["missing"][:8]:
            print(f"  {rec['subdir']}/{rec['canonical_name']}: {human(rec['expected_size'])}")
        if len(r["missing"]) > 8:
            print(f"  ... and {len(r['missing']) - 8} more")

    total_dl = sum(rec["expected_size"] for rec in r["download_queue"])
    total_complete = sum(rec["expected_size"] for rec in r["complete"])
    print()
    print(f"Already complete:  {human(total_complete)} ({len(r['complete'])} files)")
    print(f"Needs download:    {human(total_dl)} ({len(r['download_queue'])} files)")

if __name__ == "__main__":
    r = audit()
    print_summary(r)
    out = ROOT / "_pull_audit.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\nAudit written to: {out}")
