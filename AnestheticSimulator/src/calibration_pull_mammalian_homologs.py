"""Stage 2 — pull AlphaFold DB structures for mammalian-homolog calibration targets.

Targets (after pushback adjustments — Lu 2007 NALCN dropped):

| C. elegans | Mammalian homolog | UniProt | Anesthetic class |
|---|---|---|---|
| UNC-49 | GABA-A α1 (GABRA1) | P14867 | gaba_potentiation |
| AVR-14 / GLC-1/2 | GlyR α1 (GLRA1) | P23415 | glucl_potentiation (homolog) |
| ACR-16 / nAChR | nAChR α4 (CHRNA4) | P43681 | nachr_antagonism |
| TWK-18/29 | TREK-1 (KCNK2) | O95069 | k2p_potentiation |
| GAS-1 | NDUFS2 (Complex I) | O75306 | complex_i_block |

Re-uses the AF DB v6 fetch logic from `src/phase_a_fetch_alphafold_db.py`.
Output: artifacts/calibration/structures/{HUMAN_GENE}_{UNIPROT}.pdb

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/calibration_pull_mammalian_homologs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "calibration" / "structures"
OUT_AUDIT = ROOT / "artifacts" / "calibration" / "mammalian_homolog_structures.csv"

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
ALPHAFOLD_FILE = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"

TARGETS = [
    {"celegans": "UNC-49", "human_gene": "GABRA1", "uniprot": "P14867",
     "mech_class": "gaba_potentiation",
     "homolog_note": "Mammalian GABA-A α1 subunit; canonical orthosteric/allosteric anesthetic site"},
    {"celegans": "AVR-14", "human_gene": "GLRA1", "uniprot": "P23415",
     "mech_class": "glucl_potentiation",
     "homolog_note": "Mammalian glycine receptor α1; closest pentameric Cys-loop homolog of GluCl"},
    {"celegans": "ACR-16", "human_gene": "CHRNA4", "uniprot": "P43681",
     "mech_class": "nachr_antagonism",
     "homolog_note": "Mammalian nAChR α4 subunit; volatile anesthetic block"},
    {"celegans": "TWK-18", "human_gene": "KCNK2", "uniprot": "O95069",
     "mech_class": "k2p_potentiation",
     "homolog_note": "Mammalian TREK-1; K2P channel activated by volatile anesthetics"},
    {"celegans": "GAS-1", "human_gene": "NDUFS2", "uniprot": "O75306",
     "mech_class": "complex_i_block",
     "homolog_note": "Mammalian NDUFS2; Complex I 49-kDa subunit (gas-1 ortholog)"},
]


def fetch_af(acc: str, out_path: Path) -> tuple[bool, str, dict]:
    meta: dict = {}
    try:
        r = requests.get(ALPHAFOLD_API.format(acc=acc), timeout=30)
        if r.status_code == 404:
            return False, f"AF DB has no entry for {acc} (404)", meta
        r.raise_for_status()
        data = r.json()
        if not data:
            return False, f"AF DB API empty for {acc}", meta
        e = data[0]
        ver = e.get("latestVersion", 4)
        meta = {
            "global_plddt": e.get("globalMetricValue"),
            "frac_plddt_very_high": e.get("fractionPlddtVeryHigh"),
            "version": ver,
        }
    except requests.RequestException as ex:
        return False, f"API err: {ex}", meta

    try:
        r = requests.get(ALPHAFOLD_FILE.format(acc=acc, ver=ver), timeout=60)
        if r.status_code == 404:
            return False, f"AF DB v{ver} 404 for {acc}", meta
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True, f"v{ver} {len(r.content)/1024:.1f}KB pLDDT={meta.get('global_plddt')}", meta
    except requests.RequestException as ex:
        return False, f"download err: {ex}", meta


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = []
    n_ok = 0
    for t in TARGETS:
        out = OUT_DIR / f"{t['human_gene']}_{t['uniprot']}.pdb"
        print(f"[{t['celegans']:8s} -> {t['human_gene']} {t['uniprot']}] {t['mech_class']:25s}", end=" ", flush=True)
        ok, msg, meta = fetch_af(t["uniprot"], out)
        if ok:
            print(f"OK — {msg}")
            n_ok += 1
        else:
            print(f"FAIL — {msg}")
        audit.append({
            "celegans_target": t["celegans"],
            "human_gene": t["human_gene"],
            "uniprot": t["uniprot"],
            "mech_class": t["mech_class"],
            "homolog_note": t["homolog_note"],
            "structure_path": str(out.relative_to(ROOT)) if ok else "",
            "global_plddt": meta.get("global_plddt", ""),
            "frac_plddt_very_high": meta.get("frac_plddt_very_high", ""),
            "af_version": meta.get("version", ""),
            "status": "OK" if ok else "FAIL",
            "message": msg,
        })

    fieldnames = list(audit[0].keys()) if audit else []
    with open(OUT_AUDIT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(audit)

    print()
    print(f"Mammalian homolog structures: {n_ok}/{len(TARGETS)} downloaded")
    print(f"Audit: {OUT_AUDIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
