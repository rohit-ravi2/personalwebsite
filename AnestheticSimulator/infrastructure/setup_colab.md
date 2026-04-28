# Wave P — Colab pipeline setup

**Status:** SCAFFOLDED. Notebook templates not yet authored.

---

## 1. Why Colab (overflow only)

**Default path is RTX 4060 Ti local.** Colab is reserved for cases where local 8 GB VRAM is insufficient AND ESMFold / Boltz-1 / OpenFold have all failed locally. Free Colab provides T4 GPU access with ~12 hr/day session caps; cumulative budget across the program is ~30 hours.

Wave P uses Colab specifically for:

- Phase A pentameric edge cases that don't fit in 8 GB even with chunked attention (~10 hr cumulative).
- Phase B DiffDock for receptors where 30-Å truncation isn't sufficient (~8 hr cumulative).
- Phase I JAX overflow if local GPU is saturated during MD-heavy weeks (~12 hr cumulative).

**No Colab Pro required.** No A100 quota required. Free T4 tier is sufficient for the canonical plan.

---

## 2. ColabFold pipeline (Phase A)

### 2.1 Setup notebook (one-time per Phase A entry)

```python
# Cell 1 — install ColabFold
!pip install -U "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
!sudo apt-get install -y aria2

# Cell 2 — mount Drive (optional, for caching MMseqs2 results)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — clone Wave P targets CSV
!wget https://raw.githubusercontent.com/<USER>/<REPO>/main/AnestheticSimulator/targets/tier1_targets.csv \
    -O /content/tier1_targets.csv

# Cell 4 — run ColabFold for one target
target = "UNC-49"
sequence = "MNSEQENVENGTV..."  # 5x for pentamer
!mkdir -p /content/inputs
!echo ">{target}_pentamer\n{sequence}:{sequence}:{sequence}:{sequence}:{sequence}" > /content/inputs/{target}.fasta

!colabfold_batch --num-models 5 --num-recycle 3 --use-gpu-relax \
    --rank pae /content/inputs/{target}.fasta /content/results/{target}/

# Cell 5 — download to local
!tar czf /content/{target}_results.tar.gz /content/results/{target}/
from google.colab import files
files.download(f'/content/{target}_results.tar.gz')
```

### 2.2 Batch protocol

For Phase A's 12 multimer cases, the typical Colab session can run 1-2 multimers per day on the free tier (each multimer takes 1-3 hours; daily quotas reset at unpredictable times).

Recommended batching: **one multimer per session, multiple sessions per week.** Start the run, leave the tab open, download the result before the session times out.

### 2.3 MMseqs2 caching

ColabFold uses MMseqs2 for MSA generation. This is the slowest step (~30 minutes per sequence). Caching: store MSAs under `/content/drive/MyDrive/colabfold_cache/` and pass `--templates` to skip MSA regeneration.

---

## 3. DiffDock pipeline (Phase B)

### 3.1 Setup notebook

```python
# Cell 1 — clone DiffDock
!git clone https://github.com/gcorso/DiffDock /content/DiffDock
%cd /content/DiffDock
!pip install -r requirements.txt
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cell 2 — load receptor and ligand
RECEPTOR_PDB = "/content/drive/MyDrive/wavep/structures/UNC-49_multimer_rank_001.pdb"
LIGAND_SDF = "/content/drive/MyDrive/wavep/anesthetics/halothane.sdf"

# Cell 3 — run DiffDock
!python -m inference --protein_path {RECEPTOR_PDB} \
    --ligand {LIGAND_SDF} \
    --out_dir /content/results/UNC-49_halothane/ \
    --inference_steps 20 \
    --samples_per_complex 40 \
    --batch_size 10
```

### 3.2 Batch protocol

DiffDock per-pair on free-tier T4: ~15-25 minutes; on local 4060 Ti (truncated receptor): ~5-10 minutes. Canonical plan runs DiffDock locally on truncated receptors; Colab T4 absorbs only the cases that don't truncate cleanly.

---

## 4. JAX overflow (Phase I, stretch)

If local 4060 Ti is saturated during Phase I work blocks:

```python
# Cell 1 — JAX on Colab free-tier T4
!pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install optax equinox

# Cell 2 — clone Wave P JAX simulator
!git clone <repo> /content/wavep
%cd /content/wavep/AnestheticSimulator/src

# Cell 3 — run inverse optimization
!python phase_i_inverse_jax.py --inverse-fit \
    --empirical /content/drive/MyDrive/wavep/atanas_anesthetic.npz \
    --output /content/inverse_occupancy.npz
```

---

## 5. Data transfer protocol

Wave P uses **Google Drive** as the staging area for Colab transfer. Workflow:

1. Local: copy structure / ligand / data files into `~/Desktop/wavep_colab_staging/` (Drive-synced folder).
2. Colab: mount Drive; reference files at `/content/drive/MyDrive/wavep/...`.
3. Colab: write outputs back to Drive at `/content/drive/MyDrive/wavep/results/<phase>/<config>/`.
4. Local: copy results from Drive into `artifacts/<phase>/`.

This avoids large file uploads / downloads in Colab cells (slow and quota-eating).

---

## 6. Quota management

Colab free tier (T4):

- Daily quotas reset at unpredictable times (typically every 12-24 hours).
- Sessions time out after 12 hours of activity or ~90 minutes of idle.
- Quota varies based on past usage.

Wave P workflow:

- Run one multimer per session.
- Download immediately after completion.
- Don't leave a session idle for 90+ min.
- Distribute across multiple Colab accounts if necessary (one Wave P-dedicated Google account is sufficient).

If free-tier T4 quotas become a blocker in Phase A or B, **do not escalate to Colab Pro** — the user has committed to $0 external spend. Instead, fall back to subunit-by-subunit pocket modeling (Phase A) or further receptor truncation (Phase B). The deferred enhancement options are documented in `compute_budget.md` §4 but are not authorized by default.

---

## 7. Reproducibility note

All Colab notebooks for Wave P should:

- Save the random seed at the top of the notebook.
- Log ColabFold / DiffDock version with `pip show <package>` in a cell.
- Save the input fasta / sdf / pdb checksum.
- Save the output checksum.
- Push the notebook source (with no secrets) to the Wave P git repo.

Notebooks live in `infrastructure/colab_notebooks/` (to be created when Phase A activates).
