#!/usr/bin/env python3
"""Track C4 — Citation audit for 7 project-cited references.

Verify each claim via PubMed search + abstract extraction. Label each:
  VERIFIED — paper exists, abstract supports claim
  MIS_ATTRIBUTED — paper exists but claim not supported
  UNVERIFIED — paper couldn't be located
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_task7_pubmed import pubmed_search, pubmed_fetch

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260422_v2" / "task_c_parallel_analysis" / "c4_citation_audit"
OUT_MD = OUT_DIR / "summary.md"

CITATIONS = [
    {
        "name": "Nath 2016 — FLP-13/ALA sleep",
        "claim_key_terms": ["FLP-13", "ALA", "sleep"],
        "search": "Nath[Author] 2016[Year] FLP-13 ALA sleep C elegans",
    },
    {
        "name": "de Bono 1998 — FLP-21/NPR-1 aggregation",
        "claim_key_terms": ["NPR-1", "aggregation"],
        "search": "de Bono[Author] 1998[Year] NPR-1 aggregation C elegans",
    },
    {
        "name": "Wang 2013 — NLP-40/defecation",
        "claim_key_terms": ["NLP-40", "defecation", "pacemaker"],
        "search": "Wang[Author] 2013[Year] NLP-40 defecation C elegans",
    },
    {
        "name": "Li 2003 — DAF-28/dauer",
        "claim_key_terms": ["DAF-28", "insulin", "dauer"],
        "search": "Li[Author] 2003[Year] daf-28 insulin C elegans",
    },
    {
        "name": "Mellem 2008 — AVA voltage-clamp",
        "claim_key_terms": ["AVA", "action potential", "voltage-clamp"],
        "search": "Mellem[Author] 2008[Year] action potential C elegans AVA",
    },
    {
        "name": "Nelson 2013 — NLP-22/RIA sleep",
        "claim_key_terms": ["NLP-22", "RIA", "sleep"],
        "search": "Nelson[Author] 2013[Year] NLP-22 C elegans sleep",
    },
    {
        "name": "Cohen 2009 — FLP-18/NPR signaling",
        "claim_key_terms": ["FLP-18", "NPR", "foraging"],
        "search": "Cohen[Author] 2009[Year] FLP-18 NPR foraging",
    },
]


def claim_supported(abstract: str, key_terms: list[str]) -> float:
    """Return fraction of key terms present in abstract (case-insensitive)."""
    if not abstract:
        return 0.0
    a = abstract.lower()
    hits = sum(1 for t in key_terms if t.lower() in a)
    return hits / max(1, len(key_terms))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for c in CITATIONS:
        print(f"Search: {c['name']}")
        pmids = pubmed_search(c["search"], max_results=5)
        if not pmids:
            results.append({
                "name": c["name"], "status": "UNVERIFIED",
                "pmid": None, "reason": "no PubMed results",
            })
            print(f"  → UNVERIFIED (no results)")
            continue
        # Fetch each result and score; prefer top-supported
        best = None
        for pmid in pmids[:3]:
            time.sleep(1.5)
            rec = pubmed_fetch(pmid)
            if "error" in rec:
                continue
            support = claim_supported(rec["abstract"], c["claim_key_terms"])
            score = (support, len(rec.get("abstract", "")))
            if best is None or score > best[0]:
                best = (score, rec)
        if best is None:
            results.append({
                "name": c["name"], "status": "UNVERIFIED",
                "pmid": pmids[0], "reason": "fetch errors on all candidates",
            })
            print(f"  → UNVERIFIED (fetch failed)")
            continue
        (support_score, _), rec = best
        status = ("VERIFIED" if support_score >= 0.66 else
                  "PARTIAL" if support_score >= 0.34 else
                  "MIS_ATTRIBUTED")
        results.append({
            "name": c["name"], "status": status,
            "pmid": rec["pmid"],
            "title": rec["title"],
            "authors": rec["authors"],
            "year": rec["year"], "journal": rec["journal"],
            "doi": rec["doi"],
            "support_score": round(support_score, 2),
            "key_terms_found": [t for t in c["claim_key_terms"]
                                 if t.lower() in rec["abstract"].lower()],
            "abstract_snippet": rec["abstract"][:250],
        })
        print(f"  → {status} (PMID {rec['pmid']}, support={support_score:.2f})")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Track C4 — Citation audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| claim | status | PMID | support score | title |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        title = r.get("title", "-")[:60]
        lines.append(
            f"| {r['name']} | **{r['status']}** | "
            f"{r.get('pmid', '-')} | "
            f"{r.get('support_score', '-')} | {title} |"
        )
    lines.append("")
    lines.append("## Per-claim detail")
    lines.append("")
    for r in results:
        lines.append(f"### {r['name']} — {r['status']}")
        if r.get("pmid"):
            lines.append(f"- PMID {r['pmid']} / DOI {r.get('doi', '-')}")
            lines.append(f"- {r.get('authors', '-')} "
                         f"— *{r.get('journal', '-')}* ({r.get('year', '-')})")
            lines.append(f"- Title: {r.get('title', '-')}")
            lines.append(f"- Key terms found: {r.get('key_terms_found', [])}")
            lines.append(f"- Abstract: {r.get('abstract_snippet', '-')}")
        else:
            lines.append(f"- {r.get('reason', '-')}")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))

    verified = sum(1 for r in results if r["status"] == "VERIFIED")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    mis = sum(1 for r in results if r["status"] == "MIS_ATTRIBUTED")
    unver = sum(1 for r in results if r["status"] == "UNVERIFIED")

    status_md = ART / "overnight_20260422_v2" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Track C4: citation audit\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: verified={verified}, partial={partial}, "
                f"misattributed={mis}, unverified={unver}\n")


if __name__ == "__main__":
    main()
