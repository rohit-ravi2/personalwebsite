# Upstream issue draft — AVAR_simulation_iclamp.py missing

**Status:** DRAFT — awaiting user authorization before filing.
**Target repo:** `github.com/ModelDBRepository/2017403` (Nicoletti et al. 2024, ModelDB 2017403)
**Drafted by:** Phase β-pre v3 engineering session, 2026-04-26
**Confidence:** high (verified by file listing + import-line check + reproduced workaround)

---

## Suggested title

`AVAR_simulation.py imports AVAR_simulation_iclamp but the module file is missing from the repo head tree`

## Description

`AVAR_simulation.py` (line 12) executes:

```python
from AVAR_simulation_iclamp import AVA_simulation_iclamp
```

However, the corresponding file `AVAR_simulation_iclamp.py` is **not present** in the repository head tree. Running `AVAR_simulation.py` end-to-end produces:

```
ModuleNotFoundError: No module named 'AVAR_simulation_iclamp'
```

For comparison, the AVAL counterpart files are both present:

- `AVAL_simulations.py` (wrapper) — imports `AVAL_simulation_iclamp`
- `AVAL_simulation_iclamp.py` (impl) — present, exports `AVA_simulation_iclamp`

The AVAR analog `AVAR_simulation_iclamp.py` (impl) is the one missing.

The `AVAR_simulation.py` wrapper itself looks complete — it sets up the AVAR-specific conductance vector (line 28), surface area (1121.79e-8 cm²), and channel set comment ("EGL19, LEAK, IRK, NCA, UNC103") that confirms AVAR uses UNC-103 in addition to AVAL's channel set. The voltage-clamp counterpart `AVAR_simulation_vclamp.py` IS present, which suggests `AVAR_simulation_iclamp.py` was developed but accidentally not committed (or removed by accident in a later commit).

## Impact

Without `AVAR_simulation_iclamp.py`:

- `AVAR_simulation.py` cannot complete its current-clamp simulation block (lines 54-87).
- The figures derived from that block (Fig 1B AVAR current-clamp panel in the published paper) cannot be regenerated end-to-end from the released code.
- Downstream users (such as our Wave 2 channel-translation effort) who validate Brian2 / GENESIS / NEST translations against Nicoletti's NEURON reference cannot run AVAR's iclamp protocol without manual reconstruction. We attempted to reuse `AVAL_simulation_iclamp.py` directly with AVAR's parameter vector as a fallback (since AVAR shares EGL19+LEAK+IRK+NCA with AVAL), but the AVAL iclamp script does **not** insert the `unc103` channel mod, leading to a non-physiological resting potential of ~+11 mV for AVAR (vs experimental anchor ~−25 mV) because the K+ rectifier UNC-103 is absent.

## Workaround we used

Reconstructed `AVAR_simulation_iclamp.py` (functionally) as a standalone patch in our wave2 directory at `scripts/brain/wave2/avar_unc103_patch.py`. The patch:

1. Mirrors `AVAL_simulation_iclamp.py`'s structure verbatim.
2. Substitutes AVAR's surface area (`1121.79e-8 cm²`) and parameter vector from `AVAR_simulation.py` line 28.
3. Adds `soma.insert("unc103")` and assigns `seg.unc103.gbar` from the rescaled AVAR parameter vector at index 4 (UNC-103 conductance: `0.0481669 × 1e-9 / 1121.79e-8` S/cm² after `gScm2` rescale).
4. Otherwise preserves all upstream constants — IClamp delay 1023 ms, dur 1000 ms, simdur 2500 ms, dt 0.025 ms, finitialize -60 mV, eca 60 mV, ek -80 mV.

With this patch, the AVAR resting potential lands at `-24.25 mV` (mean across 7 current steps), well within the physiologically expected range.

## Confidence in the workaround

- **Channel set verified** against the comment in `AVAR_simulation.py` line 27 ("EGL19, LEAK, IRK, NCA, UNC103").
- **Conductance values verified** by reading the parameter vector directly (line 28).
- **Surface area + ELEAK + CM verified** against the same source.
- **`unc103.mod` is shipped** in the repo and compiles cleanly into the local NEURON mechanism library; the patch only restores the missing Python glue.
- **Cannot fully verify against a reference** because the original `AVAR_simulation_iclamp.py` is what we'd compare against. Verifying the patch matches the upstream's *intended* behavior requires either restoring the original file or contacting the corresponding author. Plateau values from the patch run (`-127, -96, -62, -24, +16, +49, +80 mV` across the 7 steps) are consistent with the published Fig 1B AVAR Model trace (digitized blue Model plateau anchors `-125, -91, -58, -22, +20, +56, +85 mV` from the published panel).

## Suggested fix

Restore `AVAR_simulation_iclamp.py` to the repo. The most likely source is the corresponding author's local working tree at the time of submission. The voltage-clamp counterpart `AVAR_simulation_vclamp.py` is shipped; the iclamp counterpart was likely developed in parallel and accidentally omitted from the release commit.

If restoring from a working tree is not feasible, an alternative is to publish a small wrapper module — analogous to ours, but maintained upstream — that explicitly inserts `unc103` into AVAL's iclamp infrastructure with AVAR's parameters. Either approach would unblock end-to-end AVAR reproduction from the released code.

## Reproducibility

- **Repo commit checked:** ModelDB 2017403 head tree as locally cloned to `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` (commit hash 78a17ca per Phase β-pre v2 notes).
- **Verification command:** `ls AVAR_simulation_iclamp.py && python3 AVAR_simulation.py` — first command fails with "No such file or directory."
- **Patch + reproduction code:** available at `[our project URL pending]/scripts/brain/wave2/avar_unc103_patch.py`. Channel insertion logic: 7 lines of NEURON Python; conductance assignment: 1 line; otherwise identical to AVAL's iclamp module.

## Contact

Drafted by [user to fill] for the Wave 2 channel-translation effort. Happy to share the standalone patch as reference for the upstream restoration.
