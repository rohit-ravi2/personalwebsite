# Wave P — Dependencies

**Status:** SCAFFOLDED. None of these are installed yet in Wave P-isolated venvs.

---

## 0. Zero external spend commitment

Wave P operates entirely on local hardware (RTX 4060 Ti, 8 GB VRAM) with **$0 external spend**. The following are explicitly excluded from the canonical path:

- **No cloud bursts** (Lambda Labs / AWS / GCP). FEP cloud spend is dropped.
- **No Colab Pro / paid quotas.** Colab free tier (T4, ~12 hr/day) is an *overflow-only* option, not a default path.
- **No commercial licenses.** All tools are open-source (MIT / Apache 2.0 / BSD / LGPL / GPL) or free academic.

Every load-bearing computation has a free open-source pathway. Where a tool is academic-use-only (AlphaFold-Multimer, RoseTTAFold-AllAtom), an MIT/Apache-licensed substitute is staged as a non-load-bearing alternative. See §1.1 (structure prediction) and §3 (license notes).

If at any future point the user reverses this commitment, the deferred acceleration paths are documented in `compute_budget.md` §4.

---

## 1. Tool inventory

### 1.1 Structural prediction (Phase A)

**Compute defaults:** Primary path is **RTX 4060 Ti local**. Overflow path is **Colab free tier (T4, ~12 hr/day)** when 8 GB VRAM is insufficient for a specific oligomer. Pentameric receptors are the main VRAM-stressing case; see §4.1 and `risk/risk_register.md` for fallback ladder.

| Tool | Version (target) | Purpose | License | Load-bearing? |
|---|---|---|---|---|
| ESMFold | Lin et al. 2023 *Science* (DOI 10.1126/science.ade2574) | Single-sequence transformer-based structure prediction; ~10× faster than AF2, slightly less accurate at low-MSA targets | MIT | YES — primary fallback for VRAM-tight pentamers |
| OpenFold | Ahdritz et al. 2024 *Nature Methods* | Open reproduction of AlphaFold2 with retrainable weights; runs on consumer GPUs with chunking | Apache 2.0 | YES — secondary fallback |
| Boltz-1 | Wohlwend et al. 2024 (preprint) | Open complex-prediction model comparable to AlphaFold3 for small protein-protein and protein-ligand complexes | MIT | YES — pentameric / multimer fallback |
| ColabFold | 1.5+ | AlphaFold-Multimer / AlphaFold2 driver, MMseqs2-based MSAs | Apache 2.0 (ColabFold); AlphaFold weights non-commercial (DeepMind) | Optional — overflow only via free tier |
| AlphaFold-Multimer | 2.3+ | Pentameric / oligomeric structure | CC BY-NC 4.0 (non-commercial academic use) | Non-load-bearing — open-source fallbacks (ESMFold, OpenFold, Boltz-1) cover the same workload |
| RoseTTAFold-AllAtom | 2024 release | All-atom backup predictor | Custom academic license | Non-load-bearing — open-source fallbacks available |
| AlphaFold DB pulls | n/a | Pre-computed monomers | CC BY 4.0 | YES — monomer baseline |
| FoldSeek | 8+ | Structural similarity search | GPL-3.0 | YES |
| TM-align | 2019+ | Quantitative alignment | free academic | YES |
| PyMOL | 2.5+ | Visualization, RMSD | open-source community version | YES |
| ChimeraX | 1.7+ | Alternative visualization | non-commercial free | Optional |

**When to use each predictor (Phase A):**

- **AlphaFold DB pull (free, pre-computed):** first attempt for any monomer. 25/25 Tier-1 monomers should resolve this way.
- **ESMFold local:** first attempt for any *C. elegans* sequence not in AF DB. Single-sequence inference fits comfortably in 8 GB VRAM for monomers up to ~700 aa. Lin et al. 2023 reports ~10× speedup vs AF2 at modest accuracy cost.
- **OpenFold local:** when ESMFold confidence (pLDDT) at the pocket is too low or the target needs MSA-based refinement. Runs on 8 GB with `--chunk-size` flag.
- **Boltz-1 local:** primary tool for **pentameric and multimer** assemblies. Wohlwend et al. 2024 reports comparable accuracy to AlphaFold3 on small complexes; MIT license; designed for consumer hardware.
- **ColabFold (free tier, overflow only):** if Boltz-1 fails on a specific pentamer, escalate to ColabFold free tier on T4 GPU. ~12 hr/day cap; ~30 hr cumulative budget across the phase = ~3 calendar days at the cap.
- **AlphaFold-Multimer / RoseTTAFold-AllAtom:** available for academic-use research, but **non-load-bearing**. Do not gate Phase A entry on these.

**Install commands (Phase A activation):**

```bash
# ESMFold (no MSA required, transformer-based)
pip install fair-esm[esmfold]
# or via the standalone package:
pip install esm

# OpenFold (Apache 2.0 reproduction of AlphaFold2)
git clone https://github.com/aqlaboratory/openfold
cd openfold && pip install -e .

# Boltz-1 (open AlphaFold3-style complex predictor, MIT)
pip install boltz
# or from source:
git clone https://github.com/jwohlwend/boltz
cd boltz && pip install -e .
```

These three substitutes are MIT/Apache-2.0 and impose **no commercial-use restriction**, which removes the AF-Multimer / RFAA license question from the critical path. The license-verification deliverable in Phase A is retained as bookkeeping but is no longer a blocking item.

### 1.2 Docking and binding (Phase B)

| Tool | Version | Purpose | License |
|---|---|---|---|
| AutoDock Vina | 1.2.5+ | Rigid docking | Apache 2.0 |
| AutoDockTools | 1.5.7+ | Receptor / ligand prep | LGPL |
| DiffDock | 2024 release | Generative pose ensemble | MIT |
| GNINA | 1.1+ | CNN rescoring | Apache 2.0 |
| fpocket | 4.0+ | Cavity detection | MIT |
| OpenBabel | 3.1+ | Ligand format conversion | GPL-2.0 |
| RDKit | 2024+ | Cheminformatics, charges | BSD-3-Clause |

### 1.3 MD and FEP (Phase D)

| Tool | Version | Purpose | License |
|---|---|---|---|
| OpenMM | 8.0+ | MD engine | LGPL / MIT |
| AMBER ff14SB | n/a | Protein force field | open-source |
| GAFF2 | n/a | Ligand force field | open-source |
| AM1-BCC | via Antechamber | Partial charges | open-source (AmberTools academic) |
| CHARMM-GUI | web | Membrane builder | free academic |
| LIPID17 | n/a | AMBER lipid force field | open-source |
| ParmEd | 4.0+ | Topology manipulation | LGPL |
| MDAnalysis | 2.4+ | Trajectory analysis | GPL-2.0 |
| YANK | 0.25+ | FEP framework | MIT |
| HOLE | 2.2+ | Pore radius analysis | free academic |

### 1.4 Network simulation (Phases E, F, G)

| Tool | Version | Purpose | License |
|---|---|---|---|
| Brian2 | 2.6+ | Spiking simulator | CeCILL-2.1 |
| Brian2GeNN | 1.7+ (optional) | GPU acceleration | GPL-2.0 |
| numpy | 1.24+ | Arrays | BSD |
| scipy | 1.10+ | Optimization, statistics | BSD |
| matplotlib | 3.7+ | Visualization | BSD |
| pandas | 2.0+ | Tables | BSD |

### 1.5 Differentiable simulator (Phase I, stretch)

| Tool | Version | Purpose | License |
|---|---|---|---|
| JAX | 0.4+ | Differentiable arrays | Apache 2.0 |
| optax | 0.1+ | Optimizers | Apache 2.0 |
| equinox | 0.11+ (optional) | Module abstractions | Apache 2.0 |

### 1.6 Network signatures (Phase J, stretch)

| Tool | Version | Purpose | License |
|---|---|---|---|
| PyPhi | 1.2+ | Integrated information | GPL-3.0 |
| networkx | 3.1+ | Graph metrics | BSD |
| umap-learn | 0.5+ | Manifold embedding | BSD |
| scikit-learn | 1.3+ | t-SNE, PCA | BSD |

---

## 2. Isolated environment plan

Wave P uses **three isolated environments** to prevent conflicts:

### 2.1 `~/venvs/wave-p-md/` — Phase D MD pipeline

Heavy: OpenMM, AMBER tools, MDAnalysis, ParmEd, HOLE, YANK.

```bash
python3 -m venv ~/venvs/wave-p-md
source ~/venvs/wave-p-md/bin/activate
pip install --upgrade pip
pip install openmm openmmtools mdanalysis parmed pdb-fixer
# AMBER tools via conda is more reliable:
# conda install -c conda-forge ambertools
```

### 2.2 `~/venvs/wave-p-dock/` — Phase B docking

Vina + RDKit + OpenBabel + DiffDock + GNINA. GNINA has heavy dependencies (PyTorch, OpenBabel native build); recommend conda for GNINA.

```bash
python3 -m venv ~/venvs/wave-p-dock
source ~/venvs/wave-p-dock/bin/activate
pip install --upgrade pip
pip install vina rdkit-pypi
# DiffDock from source: github.com/gcorso/DiffDock
# GNINA via conda recommended:
# conda env create -f environment.yml -n wave-p-gnina
```

### 2.3 `~/venvs/wave-p-jax/` — Phase I JAX simulator

```bash
python3 -m venv ~/venvs/wave-p-jax
source ~/venvs/wave-p-jax/bin/activate
pip install --upgrade pip
pip install jax[cuda12_pip] optax equinox
pip install brian2 numpy scipy matplotlib pandas
```

### 2.4 `~/miniconda3/envs/ml/` — Brain conda env (shared, do NOT modify)

This is the production-simulator environment used by Wave 2 and the notebook pipeline. **Wave P does not install into this env.** Phase G's network runs may need to read from this env's Brian2 channel modules; that is read-only access via filesystem path import, not package install.

---

## 3. License notes (load-bearing for publication)

### 3.1 AlphaFold-Multimer non-commercial restriction (non-load-bearing)

DeepMind's AlphaFold weights and AlphaFold-Multimer are licensed under **CC BY-NC 4.0** (non-commercial). This is acceptable for academic publication and for Wave P's research-program scope. AF-Multimer is **non-load-bearing** in the canonical plan: ESMFold (MIT), OpenFold (Apache 2.0), and Boltz-1 (MIT) cover the same workload without license restrictions. AF-Multimer remains available as a cross-check for academic publication but does not gate any phase.

For commercial redistribution from `scripts/brain/`: structures should be regenerated using the open-source predictors. The redistribution question is covered in `integration/production_simulator_handoff.md`.

### 3.2 RoseTTAFold-AllAtom license (non-load-bearing)

The Baker lab released RFAA under a custom license (verify against the latest release at `github.com/baker-laboratory/RoseTTAFold-All-Atom`). Wave P's use, if any, is academic and exploratory only. Boltz-1 (MIT) is the canonical open all-atom complex predictor in the revised plan.

### 3.2a Open-source structure-prediction substitutes (load-bearing)

- **ESMFold** — Lin et al. 2023 *Science* (DOI 10.1126/science.ade2574). MIT licensed via `fair-esm`. No restrictions; suitable for paper and production.
- **OpenFold** — Ahdritz et al. 2024 *Nature Methods*. Apache 2.0. Full reproduction of AlphaFold2 with retrainable weights.
- **Boltz-1** — Wohlwend et al. 2024 (preprint, github.com/jwohlwend/boltz). MIT. Open AlphaFold3-style model for protein-ligand and protein-protein complexes.

All three are unrestricted for academic publication and commercial redistribution. The MIT/Apache stack is the canonical structure-prediction layer for Wave P.

### 3.3 ColabFold

ColabFold itself is Apache 2.0. The bundled MMseqs2 database is open. The AlphaFold weights it loads are subject to the DeepMind license noted above. Practical effect: same as AlphaFold-Multimer.

### 3.4 GNINA

Apache 2.0. Trained CNN model weights (`crossdock_default2018`) bundled with the release — same license unless specified otherwise.

### 3.5 Brian2, scipy, numpy, JAX, optax — permissive

All BSD or Apache 2.0 except Brian2 (CeCILL-2.1, similar to LGPL). No commercial restrictions.

### 3.6 Action items before Phase A entry

- License-verification deliverable retained as bookkeeping but **demoted from blocking-item status**: ESMFold / OpenFold / Boltz-1 are the load-bearing tools and carry MIT / Apache 2.0 only.
- If AF-Multimer or RFAA is used in cross-validation runs, document the use and confirm academic-only context — not blocking for Phase A entry.

---

## 4. Hardware compatibility notes

### 4.1 RTX 4060 Ti 8 GB VRAM constraints

- **OpenMM MD on 50,000-atom systems**: fits comfortably. ~30-50 ns/day. Tested.
- **ESMFold (monomers)**: fits comfortably for sequences ≤ 700 aa. Primary local predictor.
- **Boltz-1 (small multimers / pentamers)**: fits with chunked attention; the engineered fallback for VRAM-tight pentameric cases.
- **OpenFold**: fits with `--chunk-size` for moderate sequences; tighter for full pentamers.
- **AlphaFold-Multimer (full pentamers)**: typically does NOT fit on 8 GB without aggressive chunking. Treated as non-load-bearing; pentameric workload routed to Boltz-1 / ESMFold instead.
- **RoseTTAFold-AllAtom**: tight on 8 GB. Some pentameric assemblies may OOM. Non-load-bearing.
- **DiffDock**: tight on 8 GB for large receptors. Use truncated receptor (30 Å around fpocket cavity); free-tier Colab as overflow.
- **GNINA scoring**: 8 GB sufficient for scoring; training would not be.
- **JAX simulator**: 300-cell network is small; 8 GB ample.
- **Complex I full-assembly (~45 subunits)**: explicitly out of scope on local hardware. Phase A focuses on **single-subunit-per-anesthetic-binding-site** modeling (GAS-1 primary per Morgan & Sedensky 1995; NUO-1 through NUO-6 individually). Full-assembly modeling is DEFERRED.

### 4.2 Colab free tier (overflow only)

Free Colab provides T4 GPU access with ~12 hr/day session caps. Not used by default. Reserved for cases where local 8 GB cannot fit a specific pentamer / multimer / DiffDock receptor and ESMFold + Boltz-1 + OpenFold have all failed. Cumulative budget across the program: ~30 hr (≈ 3 calendar days at the cap). No Colab Pro, no A100 quota required.

### 4.3 Cloud burst budget — DEFERRED

**No cloud bursts in the canonical Wave P plan.** External spend is $0. Cloud spend is recorded here as a *deferred enhancement* only: if the user later decides absolute affinity (FEP) is needed for top-10 hits, Lambda Labs A100 at ~$25-40/hour would be the path. Until that decision, all phases run on local hardware + free-tier overflow only.

---

## 5. Setup-time-estimate per environment

- `~/venvs/wave-p-md/`: 1-2 hours including AmberTools conda install.
- `~/venvs/wave-p-dock/`: 2-3 hours including DiffDock + GNINA setup.
- `~/venvs/wave-p-jax/`: 30 minutes (JAX CUDA 12 wheels are clean).
- ColabFold: ~30 minutes for first run (MMseqs2 database download).
- CHARMM-GUI: per-system, ~10 minutes web setup.

Total Wave P environment setup wall-clock: half a day.
