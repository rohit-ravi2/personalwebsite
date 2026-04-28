"""Phase A fallback — predict structures for targets missing from AlphaFold DB
using ESMFold (fair-esm 2.0).

Targets: NCA-1 (Q6Q762), UNC-80 (Q9XV66) — both auxiliary subunits of the
NALCN channel complex; large IDR-rich proteins. AlphaFold DB does not host
predictions for these UniProt accessions; we predict locally on the RTX 4060 Ti.

Notes:
- ESMFold v1 quality is comparable to AF2 for short / well-folded proteins;
  noticeably worse for very large / disordered / multi-domain proteins.
- Memory: model is ~3 GB on GPU + per-residue activations. UNC-80 may be
  too long; we will retry with chunk_size override or CPU offload as needed.
- Output: PDB file with ESMFold pLDDT in B-factor column (per ESM convention).

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/phase_a_esmfold_missed.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES_DIR = ROOT / "artifacts" / "structures"

MISSED = [
    {"gene": "NCA-1", "uniprot": "Q6Q762"},
    {"gene": "UNC-80", "uniprot": "Q9XV66"},
]

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def fetch_fasta(acc: str) -> str | None:
    r = requests.get(UNIPROT_FASTA.format(acc=acc), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    body = r.text
    seq = "".join(line.strip() for line in body.splitlines() if not line.startswith(">"))
    return seq


def main() -> int:
    import torch
    from transformers import EsmForProteinFolding, AutoTokenizer

    if not torch.cuda.is_available():
        print("CUDA not available; ESMFold will be very slow on CPU.")
        device = "cpu"
    else:
        device = "cuda"

    print(f"Loading ESMFold v1 (HF transformers) on {device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.eval()
    if device == "cuda":
        # bfloat16 for trunk to fit 8 GB; keep folding head fp32 for numerical stability
        model.esm = model.esm.half()
        model = model.to(device)
        # Set chunk size to reduce activation memory for long sequences.
        model.trunk.set_chunk_size(64)
    print(f"  loaded in {time.time()-t0:.1f}s")

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    for entry in MISSED:
        gene = entry["gene"]
        acc = entry["uniprot"]
        print(f"\n[{gene} {acc}]")
        seq = fetch_fasta(acc)
        if seq is None:
            print(f"  FAIL — UniProt FASTA 404 for {acc}")
            continue
        print(f"  length: {len(seq)} aa")
        if len(seq) > 1500 and device == "cpu":
            print("  skipping on CPU (too long)")
            continue
        if len(seq) > 2200 and device == "cuda":
            print("  warning: very long sequence; will likely OOM on 8 GB.")
            print("  trying anyway with chunk_size=32")
            model.trunk.set_chunk_size(32)

        try:
            t1 = time.time()
            with torch.no_grad():
                tokenized_input = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"]
                tokenized_input = tokenized_input.to(device)
                output = model(tokenized_input)
            pdb_str = convert_outputs_to_pdb(output)[0]
            dt = time.time() - t1

            out_pdb = STRUCTURES_DIR / f"{gene}_{acc}_ESMFold.pdb"
            out_pdb.write_text(pdb_str)
            sz = out_pdb.stat().st_size / 1024
            mean_plddt = compute_mean_plddt(out_pdb)
            print(f"  OK — {sz:.1f} KB, mean pLDDT (B-factor): {mean_plddt:.2f}, {dt:.1f}s")
            torch.cuda.empty_cache() if device == "cuda" else None
        except (RuntimeError, OSError) as e:
            print(f"  FAIL — {type(e).__name__}: {str(e)[:200]}")
            torch.cuda.empty_cache() if device == "cuda" else None
            continue

    return 0


def convert_outputs_to_pdb(outputs):
    """Convert HF transformers ESMFold outputs to PDB string list."""
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def compute_mean_plddt(pdb_path: Path) -> float:
    vals = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                b = float(line[60:66])
                vals.append(b)
            except ValueError:
                continue
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


if __name__ == "__main__":
    sys.exit(main())
