"""Phase A — LOCAL ESMFold for the Greene-deferred NALCN-complex targets.

Replaces the Greene SOL27 dependency with an 8GB-safe local run. The original
phase_a_esmfold_missed.py OOM'd at MODEL LOAD (the ~3B-param ESM-2 LM in fp16 ~5.9 GB
barely fits the 8 GB GPU before any residue is folded) — not a sequence-length issue.

Fix: run on CPU (no VRAM ceiling; the 62 GB host RAM holds the 3B model + even a
~3000-residue pair tensor), with trunk chunking to bound the O(L^2) activations.
Slower than a >=24 GB GPU, but it completes. NCA-1 is the priority (pore subunit of
the nca mechanism class); UNC-80 is a ~3000-aa IDR-rich scaffold whose ESMFold quality
is poor and whose docking value is marginal (opt-in).

HONEST SCOPE: this is a COMPLETENESS item, not load-bearing. P13-SOL28 already settled
the nca MAGNITUDE question locally (PASS); NALCN has no published Kd so nca_block stays
structurally uncalibratable regardless; and NCA-2 (G5EDM1) is already folded as a paralog
docking proxy. Local SOL27 completes the docking panel (30->32) + adds direct NCA-1
engagement data.

Usage:
  python src/phase_a_esmfold_local.py nca1            # NCA-1 only (priority), CPU
  python src/phase_a_esmfold_local.py nca1 unc80      # both
  python src/phase_a_esmfold_local.py nca1 --device cuda-trunk   # LM on CPU, trunk on GPU
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "artifacts" / "structures"
OUT = ROOT / "artifacts" / "p13_sol27_local"

# corrected + legacy accessions (tried in order); UniProt/EBI both attempted
TARGETS = {
    "nca1": {"gene": "NCA-1", "accs": ["Q9N4D6", "Q6Q762"]},
    "unc80": {"gene": "UNC-80", "accs": ["Q9N5E6", "Q9XV66"]},
}

UNIPROT = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
EBI = "https://www.ebi.ac.uk/proteins/api/proteins/{acc}"


def fetch_seq(accs: list[str], retries: int = 5, wait: float = 20.0) -> tuple[str, str] | None:
    """Try UniProt then EBI for each accession, with retries; cache to disk."""
    for acc in accs:
        cache = STRUCT / f"{acc}.fasta"
        if cache.exists():
            body = cache.read_text()
            seq = "".join(l.strip() for l in body.splitlines() if not l.startswith(">"))
            if seq:
                return acc, seq
        for attempt in range(retries):
            for url, parse in ((UNIPROT, "fasta"), (EBI, "ebi")):
                try:
                    r = requests.get(url.format(acc=acc), timeout=60,
                                     headers={"Accept": "text/x-fasta"} if parse == "ebi" else {})
                    if r.status_code == 200:
                        if parse == "fasta":
                            seq = "".join(l.strip() for l in r.text.splitlines() if not l.startswith(">"))
                        else:
                            seq = "".join(l.strip() for l in r.text.splitlines() if not l.startswith(">")) \
                                  if r.text.startswith(">") else json.loads(r.text).get("sequence", {}).get("sequence", "")
                        if seq:
                            STRUCT.mkdir(parents=True, exist_ok=True)
                            cache.write_text(r.text)
                            return acc, seq
                    elif r.status_code == 404:
                        break  # try next accession
                except Exception:
                    pass
            time.sleep(wait)
    return None


def fold(gene: str, acc: str, seq: str, device: str) -> dict:
    import torch
    from transformers import EsmForProteinFolding, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).eval()
    # chunk the trunk to bound O(L^2) activations (smaller for longer seqs)
    cs = 64 if len(seq) <= 1800 else 32
    model.trunk.set_chunk_size(cs)
    if device == "cuda-trunk" and torch.cuda.is_available():
        model.esm = model.esm.cpu().float()        # 3B LM stays on CPU (RAM)
        model.trunk = model.trunk.cuda()            # small trunk on GPU
        model.lm_head = model.lm_head.cpu() if hasattr(model, "lm_head") else model.lm_head
        run_device = "mixed"
    else:
        model = model.cpu().float()                 # all-CPU: no VRAM ceiling
        run_device = "cpu"

    print(f"  folding {gene} ({len(seq)} aa) on {run_device}, chunk_size={cs} ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"]
        out = model(ids)
    pdb = _to_pdb(out)[0]
    OUT.mkdir(parents=True, exist_ok=True)
    pdb_path = OUT / f"{gene}_{acc}_ESMFold.pdb"
    pdb_path.write_text(pdb)
    plddt = _mean_plddt(pdb_path)
    dt = (time.time() - t0) / 60.0
    res = {"gene": gene, "acc": acc, "length": len(seq), "device": run_device,
           "chunk_size": cs, "pdb": str(pdb_path), "mean_plddt": plddt, "wall_min": dt}
    print(f"  OK {gene}: mean pLDDT {plddt:.1f}, {dt:.1f} min -> {pdb_path}", flush=True)
    return res


def _to_pdb(outputs):
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    pos = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    pos = pos.cpu().numpy()
    mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        pred = OFProtein(aatype=outputs["aatype"][i], atom_positions=pos[i], atom_mask=mask[i],
                         residue_index=outputs["residue_index"][i] + 1, b_factors=outputs["plddt"][i],
                         chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None)
        pdbs.append(to_pdb(pred))
    return pdbs


def _mean_plddt(p: Path) -> float:
    vals = []
    for line in open(p):
        if line.startswith(("ATOM", "HETATM")):
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else 0.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    device = "cpu"
    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device") + 1]
    keys = [a for a in args if a in TARGETS] or ["nca1"]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for k in keys:
        t = TARGETS[k]
        print(f"\n[{t['gene']}] resolving sequence ...", flush=True)
        got = fetch_seq(t["accs"])
        if got is None:
            print(f"  SEQUENCE UNAVAILABLE (UniProt+EBI unreachable for {t['accs']}); "
                  f"skipping {t['gene']} — NOT fabricating a structure.", flush=True)
            results.append({"gene": t["gene"], "status": "SEQUENCE_UNAVAILABLE", "accs": t["accs"]})
            continue
        acc, seq = got
        try:
            results.append(fold(t["gene"], acc, seq, device))
        except Exception as e:
            print(f"  FOLD FAILED {t['gene']}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            results.append({"gene": t["gene"], "acc": acc, "length": len(seq),
                            "status": "FOLD_FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"})
    json.dump({"results": results,
               "note": "Local replacement for Greene SOL27; completeness not load-bearing "
                       "(SOL28 settled nca magnitude; NALCN has no Kd; NCA-2 paralog already folded)."},
              open(OUT / "sol27_local_verdict.json", "w"), indent=2)
    print(f"\nverdict -> {OUT / 'sol27_local_verdict.json'}")


if __name__ == "__main__":
    main()
