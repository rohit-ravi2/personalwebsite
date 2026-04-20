#!/usr/bin/env python3
"""Phase 3a step 1 — build a signed connectome adjacency matrix for the
LIF brain.

This is the C. elegans analog of the pre-processing Shiu et al. 2024
did for their *Drosophila* brain model (`philshiu/Drosophila_brain_model`):
take a wiring diagram with synapse counts and a neurotransmitter
identity table, produce a signed adjacency matrix where

    W_chem[pre, post] = sign(NT_pre) * raw_serial_section_count

and an unsigned symmetric gap-junction matrix.

Sources (cite these verbatim on the project page):
  - Cook et al. 2019, *Whole-animal connectomes of both Caenorhabditis
    elegans sexes*, Nature (adjacency from SI 5, corrected July 2020).
  - Loer & Rand 2022, WormAtlas — *The Evidence for Classical
    Neurotransmitters in C. elegans Neurons* (sheet: Hermaphrodite,
    sorted by neuron).

Sign convention for LIF brain v1 (consensus from the worm
compmodelling literature: Varshney 2011 / Izquierdo & Beer 2013 /
Kunert-Graf 2014 / Sarma 2018):
  - Acetylcholine (ACh) → +1  (fast excitatory via nAChR)
  - GABA                → -1  (fast inhibitory via GABA-A)
  - Glutamate (Glu)     → -1  (dominant postsynaptic receptor in CNS
                               is GluCl — inhibitory. Some iGluR targets
                               are +1 but for LIF v1 we use the
                               dominant sign. Refine later.)
  - Monoamines (5-HT, DA, Tyr, OA)  →  0 (slow/modulatory, not fast
                                          synaptic current — injected
                                          separately in Phase 5+)
  - Neuropeptides       →  0 (same rationale)

Output: scripts/brain/artifacts/connectome.npz with arrays:
  - names         (N,)   str    canonical neuron names, stable order
  - nt_primary    (N,)   str    primary NT per neuron (verbatim from L&R)
  - nt_secondary  (N,)   str    secondary NT (or "")
  - sign          (N,)   int8   signed NT convention (+1 / -1 / 0)
  - klass         (N,)   str    WormAtlas anatomical category
  - W_chem        (N,N)  float32  signed pre→post chemical weights
  - W_chem_raw    (N,N)  float32  unsigned raw serial-section counts
  - W_gap         (N,N)  float32  symmetric gap-junction weights

Run:
    /home/rohit/miniconda3/envs/ml/bin/python \\
        scripts/brain/build_connectome_matrix.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CELEGANS = Path("/home/rohit/Desktop/C-Elegans")
COOK_SI5 = (
    CELEGANS
    / "data" / "connectome" / "cook2019"
    / "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
)
NT_XLSX = (
    CELEGANS
    / "data" / "expression" / "neurotransmitter"
    / "Ce_NTtables_Loer&Rand2022.xlsx"
)
OUT = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"

# NT → sign mapping for LIF brain v1 (see module docstring).
# Keys are normalised NT category labels. The Loer & Rand sheet uses
# several string variants ("Serotonin / 5HT", "Tyramine (TA)", etc.) —
# _normalise_nt() maps them all to these canonical keys before lookup.
NT_SIGN = {
    "ACh": +1,
    "GABA": -1,
    "Glu": -1,
    "5HT": 0,       # modulatory in LIF v1
    "DA": 0,
    "OA": 0,
    "TA": 0,
    "unknown": 0,
}


def _normalise_nt(s: str) -> str:
    """Collapse Loer & Rand NT strings to canonical categories.

    Accepts anything like:
      'Acetylcholine (ACh)', 'ACh (unc-17, no cho-1)'  → 'ACh'
      'Glutamate (Glu)'                                  → 'Glu'
      'Serotonin (5-HT)', 'Serotonin / 5HT'              → '5HT'
      'GABA'                                             → 'GABA'
      'Dopamine (DA)'                                    → 'DA'
      'Octopamine (OA)'                                  → 'OA'
      'Tyramine (Tyr)', 'Tyramine (TA)'                  → 'TA'
      'Unknown', 'unknown', '', NaN                      → 'unknown'
    """
    if not s or str(s).strip().lower() in ("nan", "unknown", ""):
        return "unknown"
    s_lower = str(s).lower()
    if "ach" in s_lower or "acetylcholine" in s_lower:
        return "ACh"
    if "gaba" in s_lower:
        return "GABA"
    if "glu" in s_lower:
        return "Glu"
    if "5ht" in s_lower or "5-ht" in s_lower or "serotonin" in s_lower:
        return "5HT"
    if "dopamine" in s_lower or s_lower.strip() == "da":
        return "DA"
    if "octopamine" in s_lower or "(oa)" in s_lower:
        return "OA"
    if "tyramine" in s_lower or "(ta)" in s_lower or "(tyr)" in s_lower:
        return "TA"
    return "unknown"


def _strip_zero_pad(name: str) -> str:
    """Cook 2019 uses zero-padded motor-neuron names (AS01, DA09, etc.).
    Loer & Rand and most other sources use no padding (AS1, DA9).
    Normalise by stripping a leading zero when it follows a letter
    prefix."""
    import re
    # e.g. AS01 → AS1, DA09 → DA9, VB11 → VB11 (no change for 2-digit ≠ 0x)
    m = re.match(r"^([A-Za-z]+)0(\d)$", name)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return name


# ---------- Cook 2019 SI 5 parsing ----------

def _parse_cook_matrix(sheet: str) -> pd.DataFrame:
    """Return a (pre, post) DataFrame indexed by neuron name.

    The Excel layout has three header rows and three header cols:
      rows 0..2 = category / blank / column headers
      cols 0..2 = category / blank / row headers
      data     = rows 3..end, cols 3..end
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(COOK_SI5, sheet_name=sheet, header=None)

    col_names = np.array(
        [_strip_zero_pad(str(x).strip()) for x in raw.iloc[2, 3:].values]
    )
    row_names = np.array(
        [_strip_zero_pad(str(x).strip()) for x in raw.iloc[3:, 2].values]
    )
    data = raw.iloc[3:, 3:].values

    # Drop trailing rows/cols whose header is NaN / empty
    valid_cols = [i for i, n in enumerate(col_names) if n and n.lower() != "nan"]
    valid_rows = [i for i, n in enumerate(row_names) if n and n.lower() != "nan"]

    col_names = col_names[valid_cols]
    row_names = row_names[valid_rows]
    data = data[np.ix_(valid_rows, valid_cols)]

    # Force numeric, NaN → 0
    data = pd.DataFrame(data, index=row_names, columns=col_names)
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return data


# ---------- Loer & Rand NT parsing ----------

def _parse_nt() -> pd.DataFrame:
    """Return a per-neuron DataFrame with columns
    [name, nclass, nt1, nt2, soma]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(
            NT_XLSX, sheet_name="Hermaphrodite, sorted by neuron", header=None
        )
    # Header row is the first one containing the literal "Neuron" in col 3.
    hdr_row = None
    for i in range(raw.shape[0]):
        if str(raw.iloc[i, 3]).strip().lower() == "neuron":
            hdr_row = i
            break
    assert hdr_row is not None, "could not find header row in NT sheet"

    body = raw.iloc[hdr_row + 1:, 2:8].copy()
    body.columns = ["nclass", "name", "sex", "nt1", "nt2", "soma"]
    body["name"] = body["name"].astype(str).str.strip()
    body = body[body["name"].str.match(r"^[A-Z]")]  # drop blanks / footers
    body["nclass"] = body["nclass"].ffill().astype(str).str.strip()
    body["nt1"] = body["nt1"].fillna("unknown").astype(str).str.strip()
    body["nt2"] = body["nt2"].fillna("").astype(str).str.strip()
    body["soma"] = body["soma"].fillna("").astype(str).str.strip()
    return body.reset_index(drop=True)


def _nt_to_sign(nt1: str, nt2: str) -> int:
    """Return the signed-synapse sign for a presynaptic neuron.

    Normalises NT strings first, then: if primary has a defined sign,
    use it. Else fall back to secondary. Else 0.
    """
    s1 = NT_SIGN.get(_normalise_nt(nt1), 0)
    if s1 != 0:
        return s1
    return NT_SIGN.get(_normalise_nt(nt2), 0)


# ---------- main ----------

def main() -> None:
    print(f"Reading Cook 2019 SI 5 from:\n  {COOK_SI5}")
    chem = _parse_cook_matrix("hermaphrodite chemical")
    gap = _parse_cook_matrix("hermaphrodite gap jn symmetric")
    print(f"  chemical: {chem.shape}  total weight = {chem.values.sum():.0f}")
    print(f"  gap:      {gap.shape}  total weight = {gap.values.sum():.0f}")

    print(f"\nReading NT table from:\n  {NT_XLSX}")
    nt = _parse_nt()
    print(f"  {len(nt)} neurons in NT table")

    # Canonical neuron set = neurons present in BOTH Cook chemical rows/cols
    # AND the NT table. We intersect all three so everything aligns.
    cook_rows = set(chem.index)
    cook_cols = set(chem.columns)
    nt_names = set(nt["name"])
    canonical = sorted(cook_rows & cook_cols & nt_names)

    # Sanity: hermaphrodite has 302 somatic neurons. Cook SI 5 includes
    # 20 pharyngeal + 2 CAN + body-wall muscles. The NT table lists the
    # neuronal subset, so the intersection should be ≈302.
    print(f"\n|Cook rows|={len(cook_rows)}  |Cook cols|={len(cook_cols)}  "
          f"|NT names|={len(nt_names)}  |intersection|={len(canonical)}")

    dropped_cook = (cook_rows | cook_cols) - nt_names
    dropped_nt = nt_names - (cook_rows & cook_cols)
    if dropped_cook:
        print(f"  Cook-only (not in NT table): {sorted(dropped_cook)[:12]}"
              f"{' …' if len(dropped_cook) > 12 else ''}")
    if dropped_nt:
        print(f"  NT-only (not in Cook):       {sorted(dropped_nt)[:12]}"
              f"{' …' if len(dropped_nt) > 12 else ''}")

    # Build arrays in canonical order
    N = len(canonical)
    nt_by_name = nt.set_index("name")
    names = np.array(canonical, dtype=object)
    nt_primary = np.array([nt_by_name.loc[n, "nt1"] for n in canonical], dtype=object)
    nt_secondary = np.array([nt_by_name.loc[n, "nt2"] for n in canonical], dtype=object)
    klass = np.array([nt_by_name.loc[n, "nclass"] for n in canonical], dtype=object)
    sign = np.array(
        [_nt_to_sign(nt_by_name.loc[n, "nt1"], nt_by_name.loc[n, "nt2"])
         for n in canonical],
        dtype=np.int8,
    )

    W_chem_raw = chem.loc[canonical, canonical].values.astype(np.float32)
    W_gap = gap.loc[canonical, canonical].values.astype(np.float32)
    # Force gap junctions to be symmetric (Cook's "symmetric" sheet should
    # already be, but guarantee numerically).
    W_gap = (W_gap + W_gap.T) / 2.0

    # Signed chemical weights: sign multiplies each row (presynaptic).
    W_chem = (sign[:, None].astype(np.float32) * W_chem_raw).astype(np.float32)

    # Diagnostic stats
    n_chem_edges = int(np.sum(W_chem_raw > 0))
    n_gap_edges = int(np.sum(W_gap > 0)) // 2  # undirected
    n_excitatory = int(np.sum((W_chem_raw > 0) & (sign[:, None] > 0)))
    n_inhibitory = int(np.sum((W_chem_raw > 0) & (sign[:, None] < 0)))
    n_modulatory = int(np.sum((W_chem_raw > 0) & (sign[:, None] == 0)))
    print()
    print(f"Final matrix: {N} neurons")
    print(f"  chemical edges:   {n_chem_edges}  "
          f"(exc {n_excitatory}, inh {n_inhibitory}, "
          f"mod/zero-signed {n_modulatory})")
    print(f"  gap junctions:    {n_gap_edges}")

    # NT distribution (normalised)
    print("\nNT distribution (normalised primary):")
    norm_primary = np.array([_normalise_nt(v) for v in nt_primary])
    vals, counts = np.unique(norm_primary, return_counts=True)
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
        sgn = NT_SIGN.get(v, 0)
        print(f"  {v:<10} {c:>4}  sign={sgn:+d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        names=names,
        nt_primary=nt_primary,
        nt_secondary=nt_secondary,
        klass=klass,
        sign=sign,
        W_chem=W_chem,
        W_chem_raw=W_chem_raw,
        W_gap=W_gap,
    )
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
