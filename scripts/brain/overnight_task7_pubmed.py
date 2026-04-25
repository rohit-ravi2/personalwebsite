#!/usr/bin/env python3
"""Overnight Task 7 — PubMed metadata for T4-5 candidate citations.

Fetches PubMed metadata (title, abstract, DOI) for the 5 T4-5 candidate
peptide references. Uses NCBI E-utilities API.

Candidates + claimed refs:
  FLP-13 — Nath et al. 2016 (ALA sleep)
  FLP-18 — Rogers et al. 2003 (NPR-1 aggregation)
  FLP-21 — de Bono & Bargmann 1998 (NPR-1 aggregation)
  NLP-40 — Wang et al. 2013 (defecation motor)
  DAF-28 — Li et al. 2003 (dauer/longevity)

Output:
  task7_pubmed/t4_5_citation_check.md
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import xml.etree.ElementTree as ET

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260421" / "task7_pubmed"
OUT_MD = OUT_DIR / "t4_5_citation_check.md"

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

CANDIDATES = [
    ("FLP-13", "Nath", "2016", "ALA sleep FLP-13 C elegans"),
    ("FLP-18", "Rogers", "2003", "FLP-18 NPR-1 C elegans"),
    ("FLP-21", "de Bono", "1998", "NPR-1 aggregation Bargmann"),
    ("NLP-40", "Wang", "2013", "NLP-40 defecation motor C elegans"),
    ("DAF-28", "Li", "2003", "daf-28 insulin C elegans"),
]


def pubmed_search(query: str, max_results: int = 3) -> list[str]:
    """Return PMIDs matching query."""
    url = (f"{NCBI_BASE}/esearch.fcgi?db=pubmed&term={quote_plus(query)}"
           f"&retmax={max_results}&retmode=json")
    try:
        req = Request(url, headers={"User-Agent": "phase0-audit/1.0"})
        with urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
            return data.get("esearchresult", {}).get("idlist", [])
    except (URLError, HTTPError) as e:
        print(f"  esearch failed: {e}")
        return []


def pubmed_fetch(pmid: str) -> dict:
    """Return title, abstract, DOI, authors for a PMID."""
    url = f"{NCBI_BASE}/efetch.fcgi?db=pubmed&id={pmid}&rettype=xml"
    try:
        req = Request(url, headers={"User-Agent": "phase0-audit/1.0"})
        with urlopen(req, timeout=15) as r:
            xml = r.read().decode()
    except (URLError, HTTPError) as e:
        return {"pmid": pmid, "error": str(e)}
    try:
        root = ET.fromstring(xml)
        article = root.find(".//Article")
        if article is None:
            return {"pmid": pmid, "error": "no Article element"}
        title = (article.findtext(".//ArticleTitle", "") or "").strip()
        abstract = " ".join(
            (t.text or "") for t in article.findall(".//AbstractText")
        ).strip()
        journal = article.findtext(".//Journal/Title", "") or ""
        year = article.findtext(".//PubDate/Year", "") or ""
        authors = []
        for a in article.findall(".//AuthorList/Author")[:5]:
            ln = a.findtext("LastName", "")
            fn = a.findtext("ForeName", "")
            if ln:
                authors.append(f"{ln} {fn[:1]}" if fn else ln)
        # DOI — can be in ArticleIdList
        doi = ""
        for aid in root.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""
                break
        return {
            "pmid": pmid, "title": title, "abstract": abstract[:500],
            "journal": journal, "year": year,
            "authors": ", ".join(authors), "doi": doi,
        }
    except Exception as e:
        return {"pmid": pmid, "error": f"parse: {e}"}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Task 7 — T4-5 candidate citation check",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PubMed metadata for the 5 T4-5 candidate peptide references. ",
        "Use this as the starting point for manual verification tomorrow.",
        "",
    ]

    for peptide, author, year, query in CANDIDATES:
        print(f"Searching: {peptide} — {author} {year}")
        full_query = f"{author}[Author] AND {year}[Year] AND ({query})"
        pmids = pubmed_search(full_query, max_results=3)
        if not pmids:
            # Try looser query
            pmids = pubmed_search(f"{query} {author} {year}", max_results=5)

        lines.append(f"## {peptide} — claimed: {author} {year}")
        lines.append("")
        if not pmids:
            lines.append("**No PubMed results matched.** Try manual search.")
            lines.append("")
            continue

        for pmid in pmids[:3]:
            rec = pubmed_fetch(pmid)
            time.sleep(0.4)  # politeness
            if "error" in rec:
                lines.append(f"- PMID {pmid}: fetch error ({rec['error']})")
                continue
            lines.append(f"- **PMID {rec['pmid']}**"
                         + (f" (DOI: {rec['doi']})" if rec['doi'] else ""))
            lines.append(f"  - **{rec['title']}**")
            lines.append(f"  - {rec['authors']} — *{rec['journal']}* "
                         f"({rec['year']})")
            if rec['abstract']:
                lines.append(f"  - _{rec['abstract'][:300]}..._")
            lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Total wall: {time.time()-t0:.1f}s")

    status_md = ART / "overnight_20260421" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Task 7: PubMed metadata\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: PubMed queries attempted for "
                f"{len(CANDIDATES)} T4-5 candidates\n")
        f.write(f"- Output: task7_pubmed/\n")


if __name__ == "__main__":
    main()
