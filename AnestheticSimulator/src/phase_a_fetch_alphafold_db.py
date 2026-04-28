"""Phase A — fetch AlphaFold DB structures for Tier-1 targets.

This script does TWO jobs at once:

1. Pull canonical UniProt entries for each Tier-1 C. elegans gene by querying
   UniProt REST API by gene name + organism (taxonomy 6239 = C. elegans).
   This sidesteps the unverified UniProt IDs in tier1_targets.csv.

2. Download the AlphaFold DB predicted structure (PDB format, latest model
   version) for each verified UniProt accession into artifacts/structures/.

Side effect: produces a verified-id mapping at
`artifacts/structures/uniprot_id_audit.csv` that becomes the corrected
identifier source for the CSV. Mismatches between the CSV's claim and the
canonical lookup are surfaced as warnings.

Usage:
    python src/phase_a_fetch_alphafold_db.py --dry-run
    python src/phase_a_fetch_alphafold_db.py --limit 5    # fetch first 5
    python src/phase_a_fetch_alphafold_db.py              # fetch all Tier-1

Network costs: one UniProt REST query + one AF DB download per target.
~25 queries + ~25 downloads for Tier-1. ~3 MB per PDB. Total ~75 MB, minutes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests


ROOT = Path(__file__).resolve().parents[1]
TIER1_CSV = ROOT / "targets" / "tier1_targets.csv"
TIER2_CSV = ROOT / "targets" / "tier2_targets.csv"
STRUCTURES_DIR = ROOT / "artifacts" / "structures"
AUDIT_CSV = STRUCTURES_DIR / "uniprot_id_audit.csv"
LOG = STRUCTURES_DIR / "fetch_log.txt"

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
ALPHAFOLD_FILE = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"
WORM_TAX_ID = "6239"

# Per-row acceptable mismatch level for stdout warnings
WARN_ON_MISMATCH = True


def query_uniprot_by_gene(gene_name: str, prefer_reviewed: bool = True) -> Optional[dict]:
    """Query UniProt for a C. elegans gene; prefer Swiss-Prot reviewed entries.

    Returns dict with keys: accession, entry_name, gene_names, protein_name,
    reviewed (bool), length, wormbase_xref. Returns None if no hit.
    """
    # Lowercase gene name for matching; UniProt is case-insensitive on gene
    g = gene_name.strip()

    # Strip any standalone uppercase markers (e.g. "UNC-49" -> "unc-49")
    g_lower = g.lower()

    queries_to_try = []
    if prefer_reviewed:
        queries_to_try.append(
            f"(gene:{g_lower}) AND (organism_id:{WORM_TAX_ID}) AND (reviewed:true)"
        )
    queries_to_try.append(
        f"(gene:{g_lower}) AND (organism_id:{WORM_TAX_ID})"
    )

    for query in queries_to_try:
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,id,gene_names,protein_name,reviewed,length,xref_wormbase",
            "size": "5",
        }
        try:
            r = requests.get(UNIPROT_REST, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                continue

            # Pick best match: exact gene name match preferred over fuzzy
            best = None
            for hit in results:
                hit_genes = []
                for gn in hit.get("genes", []):
                    if "geneName" in gn:
                        hit_genes.append(gn["geneName"]["value"].lower())
                    if "synonyms" in gn:
                        for syn in gn["synonyms"]:
                            hit_genes.append(syn["value"].lower())
                if g_lower in hit_genes:
                    best = hit
                    break
            if best is None:
                best = results[0]

            wb_id = ""
            for xref in best.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "WormBase":
                    wb_id = xref.get("id", "")
                    break

            primary_gene = ""
            for gn in best.get("genes", []):
                if "geneName" in gn:
                    primary_gene = gn["geneName"]["value"]
                    break

            protein_name = ""
            pn = best.get("proteinDescription", {})
            if "recommendedName" in pn:
                protein_name = pn["recommendedName"].get("fullName", {}).get("value", "")
            elif "submissionNames" in pn and pn["submissionNames"]:
                protein_name = pn["submissionNames"][0].get("fullName", {}).get("value", "")

            return {
                "accession": best.get("primaryAccession", ""),
                "entry_name": best.get("uniProtkbId", ""),
                "gene_name": primary_gene,
                "protein_name": protein_name,
                "reviewed": best.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)",
                "length": best.get("sequence", {}).get("length", 0),
                "wormbase_id": wb_id,
            }
        except requests.RequestException as e:
            print(f"  [warn] UniProt query failed for {g}: {e}", file=sys.stderr)
            return None

    return None


def fetch_alphafold(accession: str, out_path: Path) -> tuple[bool, str, dict]:
    """Query AF DB API for the latest version metadata, then download the PDB.

    Returns (ok, msg, metadata) where metadata has plddt distribution + global score.
    """
    meta: dict = {}
    if not accession:
        return False, "no accession", meta

    # Query API for latest version + confidence metrics
    api_url = ALPHAFOLD_API.format(acc=accession)
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 404:
            return False, f"AF DB has no entry for {accession} (404)", meta
        r.raise_for_status()
        data = r.json()
        if not data:
            return False, f"AF DB API returned empty list for {accession}", meta
        entry = data[0]
        ver = entry.get("latestVersion", 4)
        meta = {
            "global_plddt": entry.get("globalMetricValue"),
            "frac_plddt_very_low": entry.get("fractionPlddtVeryLow"),
            "frac_plddt_low": entry.get("fractionPlddtLow"),
            "frac_plddt_confident": entry.get("fractionPlddtConfident"),
            "frac_plddt_very_high": entry.get("fractionPlddtVeryHigh"),
            "version": ver,
            "model_created": entry.get("modelCreatedDate"),
        }
    except requests.RequestException as e:
        return False, f"AF DB API error: {e}", meta
    except (ValueError, KeyError) as e:
        return False, f"AF DB API response parse error: {e}", meta

    # Download the PDB file at the latest version
    file_url = ALPHAFOLD_FILE.format(acc=accession, ver=ver)
    try:
        r = requests.get(file_url, timeout=60)
        if r.status_code == 404:
            return False, f"AF DB v{ver} PDB 404 for {accession}", meta
        r.raise_for_status()
        out_path.write_bytes(r.content)
        size_msg = f"v{ver} {len(r.content)/1024:.1f} KB pLDDT={meta.get('global_plddt')}"
        return True, size_msg, meta
    except requests.RequestException as e:
        return False, f"download error: {e}", meta


def read_target_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip downloads; only run UniProt audit lookup")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N rows (0 = all)")
    ap.add_argument("--csv", type=Path, default=TIER1_CSV,
                    help="Target CSV (default: tier1)")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to sleep between API calls (default 0.5)")
    args = ap.parse_args()

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_target_csv(args.csv)
    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"Targets to process: {len(rows)}")
    print(f"Source CSV: {args.csv}")
    print(f"Output directory: {STRUCTURES_DIR}")
    print(f"Dry run: {args.dry_run}")
    print()

    audit_rows = []
    success = 0
    af_hits = 0

    for i, row in enumerate(rows, 1):
        gene = row.get("gene_name", "").strip()
        csv_uniprot = row.get("uniprot_id", "").strip()
        csv_wormbase = row.get("wormbase_id", "").strip()

        if not gene:
            continue

        if args.verbose or i <= 3:
            print(f"[{i}/{len(rows)}] {gene}")
        else:
            print(f"[{i}/{len(rows)}] {gene}", end=" ", flush=True)

        result = query_uniprot_by_gene(gene)

        if result is None:
            print(f"  -> NO UNIPROT HIT (CSV had {csv_uniprot})")
            audit_rows.append({
                "gene_name": gene,
                "csv_uniprot_id": csv_uniprot,
                "csv_wormbase_id": csv_wormbase,
                "verified_uniprot_id": "",
                "verified_wormbase_id": "",
                "verified_reviewed": "",
                "verified_protein_name": "",
                "verified_length": "",
                "uniprot_match": "NO_HIT",
                "wormbase_match": "",
                "alphafold_db_status": "no_uniprot",
                "alphafold_pdb_path": "",
            })
            time.sleep(args.sleep)
            continue

        verified_acc = result["accession"]
        verified_wb = result["wormbase_id"]
        uniprot_match = "OK" if csv_uniprot == verified_acc else "MISMATCH"
        wormbase_match = "OK" if csv_wormbase == verified_wb else (
            "MISMATCH" if verified_wb else "no_xref"
        )
        if args.verbose or uniprot_match == "MISMATCH":
            print(f"  CSV UniProt: {csv_uniprot} | verified: {verified_acc} ({uniprot_match})")
            print(f"  CSV WB:      {csv_wormbase} | verified: {verified_wb} ({wormbase_match})")
            print(f"  reviewed: {result['reviewed']}, length: {result['length']}")
            print(f"  protein: {result['protein_name']}")

        # Download AF DB structure
        af_status = "skipped_dry_run"
        af_path_str = ""
        af_meta: dict = {}
        if not args.dry_run:
            out_pdb = STRUCTURES_DIR / f"{gene}_{verified_acc}.pdb"
            ok, msg, af_meta = fetch_alphafold(verified_acc, out_pdb)
            if ok:
                af_status = f"OK {msg}"
                af_path_str = str(out_pdb.relative_to(ROOT))
                af_hits += 1
            else:
                af_status = f"FAIL {msg}"
            if args.verbose or not ok:
                print(f"  AF DB: {af_status}")

        audit_rows.append({
            "gene_name": gene,
            "csv_uniprot_id": csv_uniprot,
            "csv_wormbase_id": csv_wormbase,
            "verified_uniprot_id": verified_acc,
            "verified_wormbase_id": verified_wb,
            "verified_reviewed": str(result["reviewed"]),
            "verified_protein_name": result["protein_name"],
            "verified_length": str(result["length"]),
            "uniprot_match": uniprot_match,
            "wormbase_match": wormbase_match,
            "alphafold_db_status": af_status,
            "alphafold_pdb_path": af_path_str,
            "alphafold_global_plddt": af_meta.get("global_plddt", ""),
            "alphafold_version": af_meta.get("version", ""),
            "alphafold_frac_high_confidence": (
                "" if af_meta.get("frac_plddt_very_high") is None
                else f"{af_meta['frac_plddt_very_high']:.3f}"
            ),
        })
        success += 1
        time.sleep(args.sleep)

    # Write audit CSV
    if audit_rows:
        with open(AUDIT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)
        print(f"\nAudit CSV: {AUDIT_CSV}")

    # Summary
    n_total = len(audit_rows)
    n_uniprot_hit = sum(1 for r in audit_rows if r["verified_uniprot_id"])
    n_uniprot_match = sum(1 for r in audit_rows if r["uniprot_match"] == "OK")
    n_uniprot_mismatch = sum(1 for r in audit_rows if r["uniprot_match"] == "MISMATCH")
    n_uniprot_nohit = sum(1 for r in audit_rows if r["uniprot_match"] == "NO_HIT")
    n_wb_match = sum(1 for r in audit_rows if r["wormbase_match"] == "OK")
    n_wb_mismatch = sum(1 for r in audit_rows if r["wormbase_match"] == "MISMATCH")

    print("\n" + "=" * 60)
    print(f"PHASE A — UniProt + AlphaFold DB audit summary")
    print("=" * 60)
    print(f"Targets processed:         {n_total}")
    print(f"UniProt hits:              {n_uniprot_hit}/{n_total}")
    print(f"  CSV id matched:          {n_uniprot_match}/{n_uniprot_hit}")
    print(f"  CSV id MISMATCHED:       {n_uniprot_mismatch}/{n_uniprot_hit}  <-- correct CSV with verified IDs")
    print(f"  No hit:                  {n_uniprot_nohit}/{n_total}")
    print(f"WormBase matches:          {n_wb_match}/{n_uniprot_hit}")
    print(f"WormBase MISMATCHES:       {n_wb_mismatch}/{n_uniprot_hit}")
    if not args.dry_run:
        print(f"AlphaFold DB downloads:    {af_hits}/{n_uniprot_hit}")
    print()
    print(f"Audit log: {AUDIT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
