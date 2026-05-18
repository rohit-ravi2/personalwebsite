"""AF-derived metadata for NCA complex assembly.
Used to weight accessory-protein contribution per-cell.
"""
from __future__ import annotations

AF_METADATA = {'nca-2': {'n_residues': 1785, 'mean_plddt': 74.10733333333333, 'median_plddt': 86.31, 'min_plddt': 22.08, 'max_plddt': 95.69, 'fraction_high_confidence': 0.7109243697478992, 'fraction_very_high': 0.29299719887955183, 'uniprot': 'G5EDM1', 'description': 'NCA-2: pore-forming subunit'}, 'unc-79': {'n_residues': 870, 'mean_plddt': 83.14996551724138, 'median_plddt': 92.09, 'min_plddt': 21.86, 'max_plddt': 98.69, 'fraction_high_confidence': 0.828735632183908, 'fraction_very_high': 0.5873563218390805, 'uniprot': 'P42173', 'description': 'UNC-79: obligate primary accessory (direct NCA binding)'}, 'nlf-1': {'n_residues': 438, 'mean_plddt': 67.89566210045662, 'median_plddt': 72.75, 'min_plddt': 25.95, 'max_plddt': 97.25, 'fraction_high_confidence': 0.5273972602739726, 'fraction_very_high': 0.19406392694063926, 'uniprot': 'M4Q8W4', 'description': 'NLF-1: peripheral modulator'}, 'weights': {'unc79': 0.6226399470023186, 'unc80': 0.3113199735011593, 'nlf1': 0.06604007949652203}}
