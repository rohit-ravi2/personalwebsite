# X.2b — NeuroML2-native simulation (sketch)

**Status:** Speculative architectural investigation. Sketch only — no prototype.

**Frame:** This sketch characterizes the alternative of bypassing Brian2 entirely and consuming c302's NeuroML2 + LEMS data structures natively via jNeuroML or libNeuroML / pyNeuroML / NetPyNE. This corresponds to the architectural plan's **Path 3D** (originally rejected on backend-analysis grounds — see `phase_v_w2_backend_architecture_analysis.md`).

---

## 1. The core idea

c302 already has:

- **Per-cell NeuroML2 morphologies** (305 cell.nml files in `c302/NeuroML2/`).
- **Channel definitions** in NeuroML2 (LeakConductance, etc., though channel coverage is far less than Nicoletti's 22).
- **LEMS network templates** for assembling cells into network simulations.
- **Synapses** in NeuroML2 (chemical, electrical).

NeuroML2 is a declarative XML-based standard for neural models. **jNeuroML** (Java) and **pyNeuroML** (Python wrapper) execute NeuroML2 models, typically by translating to NEURON or LEMS and running.

**Path 3D = bypass Brian2; run c302 declarative models directly via NeuroML2 ecosystem.**

---

## 2. What changes vs Path 3A (Brian2 + parameter import)

### 2.1 Channel definitions

**Path 3A:** Translate Nicoletti's 22 NEURON .mod files → Brian2 eqs strings.
**Path 3D:** Translate Nicoletti's 22 NEURON .mod files → NeuroML2 ChannelML format.

The translation work changes target language but is similar in volume. Possibly **harder for ChannelML** because ChannelML's declarative format is more restrictive than Brian2's eqs strings — some NMODL idioms may not have clean ChannelML equivalents (closed-form `calcium(V)` per F13, hidden unit-conversion factors per F6, etc.).

### 2.2 Cell models

**Path 3A:** Per-cell Brian2 NeuronGroup with eqs combining all channels.
**Path 3D:** Per-cell NeuroML2 cell.nml file with `<biophysicalProperties>` referencing channels by ID. Already exists in c302 form (305 files); just needs Nicoletti channels added to the channel-ID dictionary.

### 2.3 Network specification

**Path 3A:** Brian2 Synapses + connectome import (existing infrastructure in production).
**Path 3D:** NeuroML2 + LEMS network specification. c302 already provides LEMS templates.

### 2.4 Runtime backend

**Path 3D's awkwardness:** jNeuroML doesn't actually have its own integrator at scale — it usually compiles to NEURON or to another simulator. So "NeuroML2-native" in practice means:
- `pyNeuroML` → NEURON (most common)
- `pyNeuroML` → Brian2 (less mature)
- `pyNeuroML` → LEMS interpreter (very slow, reference only)
- `pyNeuroML` → NetPyNE → NEURON

The "native" framing is misleading. **NeuroML2 is a model description language; it requires a backend simulator.**

### 2.5 Compute implications

If pyNeuroML→NEURON is the backend: we are paying NEURON's 10-100× CPU performance penalty vs Brian2 (per the architectural plan's NEURON benchmark finding — "BrainPy and Brian2 demonstrate ... one to two orders of magnitude over NEURON"). For 60-second simulations at 25 µs dt × 302 cells × 30 segments, this matters.

If pyNeuroML→Brian2 backend: less mature pipeline, may have its own translation defects. We'd still own the Brian2 channel definitions effectively (via NeuroML2 → Brian2 translation), so we'd duplicate Path 3A's translation work in NeuroML2 form.

---

## 3. Pros (acknowledged)

- **Native morphology support:** c302's morphology data flows directly into the simulation without custom parsing.
- **Declarative model spec:** model definitions are XML, readable, version-controlled, decoupled from simulator code.
- **Community standard:** NeuroML2 is the OpenWorm community's lingua franca; staying in NeuroML2 means easier collaboration with c302 / OpenWorm groups.
- **Multi-backend portability:** if a project later wants to switch backends (e.g., NetPyNE for declarative network spec), NeuroML2 substrate makes it easier.

---

## 4. Cons (load-bearing)

- **Performance:** NEURON backend is 10-100× slower than Brian2 (per architectural plan §"NEURON benchmark finding"). For RTX 4060 Ti with 60-second runs over 302 cells, this is a load-bearing constraint.
- **Channel translation harder, not easier:** ChannelML is less expressive than Brian2 eqs strings. Closed-form `calcium(V)` (F13), hidden unit-conversion machinery (F6), and other NMODL idioms may not translate cleanly. Per Phase β findings, Brian2 eqs+namespace dict was already painful; ChannelML would be moreso.
- **Project's existing infrastructure is Brian2:** scenario pipeline, classifier readouts, voltage-domain LIF cells — all live in Brian2. Path 3D requires either rewriting all of this (large) or building a Brian2-NeuroML2 bridge (also large).
- **jNeuroML Java dependency:** adds JVM to runtime stack. For long-term maintenance and reproducibility, this is friction.
- **Less mature than NEURON+Python or Brian2 for biophysical work:** the jNeuroML stack is mostly used for declarative model description + network-level NEURON simulation, not for the kind of channel-translation precision that Phase β has been doing.
- **Smaller community for biophysical detail:** OpenWorm community is active but smaller than NEURON or Brian2 communities. Bug fixes and edge-case handling may be slower.

---

## 5. Why Path 3D was rejected in `phase_v_w2_backend_architecture_analysis.md`

(Per the architectural plan's mention.) Summarizing the existing rationale:

- Path 3A's compute advantage (Brian2 ~10-100× faster than NEURON) is load-bearing for RTX 4060 Ti compute budget.
- Path 3D's translation work duplicates Path 3A's effort in a less-expressive target language.
- Existing project infrastructure (scenario pipeline, classifier readouts, modulator system) would need wholesale port.
- No clear research advantage that justifies the cost.

These rationales **remain valid** under condition 6 — switching to Path 3D doesn't address the 2b failure (which is about morphology, not backend). Path 3D would still need its own multi-compartment AVAL setup with per-segment Ca-pools, and would do so at higher cost than Brian2 SpatialNeuron.

---

## 6. Comparison to current Path 3A (Brian2)

| Aspect | Path 3A (Brian2) | Path 3D (NeuroML2-native) |
|---|---|---|
| Channel translation effort | NMODL → Brian2 eqs (Phase β has translated 7) | NMODL → ChannelML (untranslated; harder per channel) |
| Morphology integration | Brian2 SpatialNeuron + custom NeuroML2 import | Native NeuroML2 (better here) |
| Compute (CPU) | Fast (~10-100× over NEURON) | Slow (NEURON-backend speed) |
| Existing infrastructure | All in Brian2 | All would need rewriting/bridging |
| Community | Brian2 community + Nicoletti's NEURON code | OpenWorm + NeuroML2 community |
| Expressiveness | Brian2 eqs handle closed-form, residuals, custom scalars trivially | ChannelML requires registered patterns |
| Maturity for biophysics | High (Phase β has 13 NMODL→Brian2 patterns systematized) | Lower for the level of detail required |

---

## 7. Comparison to morphology fork (X.2a) and GNN Variant A

| Aspect | X.2a (Brian2 SpatialNeuron) | X.2b (NeuroML2-native) | GNN Variant A |
|---|---|---|---|
| Effort to implement under condition 6 | ~3-4 weeks | ~3 months (full backend port) | ~2-3 months for production-grade |
| Effort to validate Gate 2 | Standard 2a/2b in Brian2 | Need to redo all channel validation in ChannelML | Add ablation tests + multi-protocol |
| Mechanistic interpretability | Full | Full (declarative is even more transparent) | Full (Variant A) |
| Compute impact | Acceptable | Bad (NEURON-backend speed) | Acceptable |
| Wins existing infrastructure investment | Yes | No (would discard most Phase β work) | Yes |

**Path 3D's value proposition under condition 6 is weakest.** It doesn't address the morphology issue any better than Path 3A + multi-compartment, and it discards the Phase β channel-translation investment.

---

## 8. Where Path 3D would be the right answer

Despite being rejected for Wave 2, there are scenarios where Path 3D becomes attractive:

- **If the project pivots to OpenWorm-community collaboration** as a primary integration goal (paper trajectory shifts from research-tool → community-substrate).
- **If c302/NetPyNE establishes a robust NEURON-Brian2 bridge** that absorbs the speed penalty.
- **If anesthesia-mechanism research with NetPyNE collaboration becomes the focus** (mentioned in arch plan §"What would invalidate Path A?" condition 2).
- **If a future Wave focuses on declarative model versioning + community submission to OSB (Open Source Brain).**

None of these are the current focus, so Path 3D remains a **future-option**, not a Wave 2/3 work item.

---

## 9. Brief sub-investigation: NetPyNE as a middle path

NetPyNE (high-level declarative wrapper around NEURON) sits between Path 3A and Path 3D. It uses NEURON as backend (slow vs Brian2) but provides Python-native declarative model spec instead of NeuroML2 XML.

- **Pros over Path 3D:** Python-native, faster iteration, mature ecosystem, no JVM.
- **Pros over Path 3A:** declarative model spec, easier to refactor cell models.
- **Cons:** NEURON-backend speed penalty.
- **Use case:** if Wave 4+ wants declarative cell models without giving up Brian2 entirely, NetPyNE could provide model spec while Brian2 provides simulation. This is a non-trivial bridge.

NetPyNE is an honest mention but not a Wave 2 commitment.

---

## 10. Summary

| Question | Answer |
|---|---|
| Architectural feasibility | Yes, but loses Brian2 performance advantage. |
| Effort to switch | ~3 months to redo channel translation + infrastructure rebuild. |
| Does it address condition 6? | No — same morphology work needed regardless of backend. |
| Comparison to Path 3A under condition 6 | Worse: same morphology work + new backend infrastructure cost. |
| Comparison to X.2a (Brian2 multi-compartment) | Worse: same architectural goal, higher cost, slower runtime. |
| When would it become primary? | If project pivots to OpenWorm-community substrate or anesthesia-NetPyNE collaboration. Not the current focus. |
| Recommendation | **Reject as condition-6 response.** Architectural plan's prior rejection of Path 3D remains correct. Reconsider only if community-collaboration goals shift. |
