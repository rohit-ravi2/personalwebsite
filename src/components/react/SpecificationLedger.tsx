import * as React from "react";
import { useState } from "react";

/**
 * SpecificationLedger — "Where is the specified information in the C. elegans connectome?"
 *
 * A data-grounded decomposition of what makes the reproducible C. elegans nerve-ring
 * connectome reproducible. Every number is a real result from the analysis arc
 * (contact-geometry noise-ceiling, developmental-arrival mediation, presynaptic-NT bias)
 * and the systematic falsification of the "hidden molecular code" hypotheses.
 *
 * No number here is illustrative. Provenance is noted per block.
 * Honest ceiling: this is a rigorous DECOMPOSITION, not a discovered blueprint.
 */

// ---- Panel 1: contact geometry vs the animal-to-animal reproducibility ceiling ----
// Source: connectome-seed (Witvliet adult #7/#8; nerve-ring contacting pairs).
const CEILING = [
  { thr: "≥1 synapse", geom: 0.839, ceiling: 0.819, n: "1,257 core" },
  { thr: "≥2 synapses", geom: 0.879, ceiling: 0.843, n: "784 core" },
  { thr: "≥4 synapses", geom: 0.905, ceiling: 0.880, n: "458 core" },
];

// ---- Panel 2: the specification ledger (incremental AUC, directed reproducible chem synapse) ----
// Source: developmental-rule branch Phase 7 (9,878 directed contacting pairs, 5-fold CV).
const LEDGER = [
  { label: "Contact geometry", from: 0.5, to: 0.8245, note: "physical contact + area + degree — the dominant term", tone: "geo" },
  { label: "+ Developmental arrival", from: 0.8245, to: 0.8851, note: "birth-order timing (from L1–L3 stages, not the adult target)", tone: "time" },
  { label: "+ Presynaptic NT bias", from: 0.8851, to: 0.8925, note: "small, real: Glu sources over-wire, ACh/GABA under-wire", tone: "nt" },
  { label: "+ Lineage grammar", from: 0.8925, to: 0.8943, note: "negligible (+0.0018) — adds nothing beyond geometry + timing", tone: "lin" },
];
const RESIDUAL = { from: 0.8943, to: 1.0, label: "Residual", note: "animal-to-animal variability + unmodeled — not a hidden code" };

// ---- Panel 3: what it is NOT — the falsified "hidden code" hypotheses ----
const FALSIFIED = [
  { name: "Molecular address code", verdict: "DEAD", stat: "expression homophily adds +0.0004 over contact+degree; fails degree-preserving null (z=1.8)", },
  { name: "Low-rank relational code", verdict: "NULL", stat: "learned source×target latent vectors collapse to ‖U‖≈0.001 — no relational structure beyond geometry", },
  { name: "GRN-generated repertoire", verdict: "DEAD", stat: "random non-TF genes predict channel/receptor repertoire as well as TFs (0.693 vs 0.704); real CelEsT regulators ≈ degree-null", },
  { name: "Hidden lineage generator", verdict: "FAILS", stat: "lineage tree compresses 1.23× over random; large sublineages are unique — the shortest description is the trace itself", },
  { name: "Developmental-expression code", verdict: "DEAD", stat: "embryonic effector expression does not beat adult, is not effector-specific, and its edge-CV gain is leakage", },
  { name: "Structure → function dynamics", verdict: "WALL", stat: "signed multilayer structure ρ≈0.10 on the Randi propagation kernel; running dynamics on the connectome does worse than static features", },
];

const TONE: Record<string, string> = {
  geo: "#38bdf8", time: "#a78bfa", nt: "#f59e0b", lin: "#64748b",
};

function pct(v: number, lo = 0.5, hi = 1.0) {
  return ((v - lo) / (hi - lo)) * 100;
}

export default function SpecificationLedger() {
  const [open, setOpen] = useState<number | null>(null);

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-700/60 bg-slate-900/70 p-5 sm:p-7 text-slate-200">
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-[0.2em] text-sky-400/80">
          C. elegans connectome · specification analysis
        </div>
        <h3 className="mt-1 text-xl font-semibold text-slate-50">
          What makes the reproducible connectome reproducible?
        </h3>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          Physical contact geometry and developmental birth-timing account for the reproducible
          wiring <em>up to the animal-to-animal noise ceiling</em>, with a small real presynaptic-transmitter
          bias. There is no recoverable molecular address code beneath it.
        </p>
      </div>

      {/* Panel 1 — noise ceiling */}
      <section className="mb-8">
        <h4 className="mb-1 text-sm font-semibold text-slate-100">
          1 · Contact geometry reaches the reproducibility ceiling
        </h4>
        <p className="mb-4 text-xs text-slate-400">
          The most any model can achieve is set by how well one animal's wiring predicts another's
          (the <span className="text-slate-300">noise ceiling</span>). At every synapse threshold, contact
          geometry <span className="text-sky-300">meets or exceeds it</span> — so the leftover is individual
          variability, not a missing code.
        </p>
        <div className="space-y-4">
          {CEILING.map((r) => (
            <div key={r.thr}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">{r.thr}</span>
                <span className="text-slate-500">{r.n}</span>
              </div>
              <div className="relative h-7 w-full overflow-hidden rounded-md bg-slate-800/80">
                <div
                  className="absolute inset-y-0 left-0 rounded-md bg-sky-500/70"
                  style={{ width: `${pct(r.geom)}%` }}
                  title={`geometry AUC ${r.geom}`}
                />
                {/* ceiling marker */}
                <div
                  className="absolute inset-y-0 w-[2px] bg-rose-400"
                  style={{ left: `${pct(r.ceiling)}%` }}
                  title={`animal-to-animal ceiling ${r.ceiling}`}
                />
                <div className="absolute inset-0 flex items-center justify-between px-2 text-[11px]">
                  <span className="font-semibold text-white">geometry {r.geom.toFixed(3)}</span>
                  <span className="text-rose-200">ceiling {r.ceiling.toFixed(3)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-slate-500">
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-3 rounded-sm bg-sky-500/70" /> contact-geometry AUC</span>
          <span className="inline-flex items-center gap-1.5"><span className="h-3 w-[2px] bg-rose-400" /> animal-to-animal noise ceiling</span>
          <span>axis: 0.50 (chance) → 1.00</span>
        </div>
      </section>

      {/* Panel 2 — the ledger */}
      <section className="mb-8">
        <h4 className="mb-1 text-sm font-semibold text-slate-100">
          2 · The specification ledger
        </h4>
        <p className="mb-4 text-xs text-slate-400">
          Incremental predictive power for a reproducible chemical synapse (held-out, 9,878 directed
          contacting pairs). Contact geometry is the dominant term; developmental arrival mediates the rest;
          the transmitter bias is small; lineage adds essentially nothing.
        </p>
        <div className="relative h-12 w-full overflow-hidden rounded-lg bg-slate-800/60">
          {LEDGER.map((s, i) => (
            <div
              key={i}
              className="absolute inset-y-0 border-r border-slate-900/60"
              style={{ left: `${pct(s.from)}%`, width: `${pct(s.to) - pct(s.from)}%`, background: TONE[s.tone] + "cc" }}
              title={`${s.label}: ${s.from.toFixed(3)} → ${s.to.toFixed(3)}`}
            />
          ))}
          <div
            className="absolute inset-y-0 bg-[repeating-linear-gradient(45deg,#334155_0,#334155_6px,#1e293b_6px,#1e293b_12px)]"
            style={{ left: `${pct(RESIDUAL.from)}%`, width: `${pct(RESIDUAL.to) - pct(RESIDUAL.from)}%` }}
            title="residual = animal noise + unmodeled"
          />
          <div className="absolute inset-0 flex items-center px-3 text-[11px] text-white/90">
            <span className="font-semibold">AUC 0.50 → 0.89 across the model stack</span>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {LEDGER.map((s, i) => (
            <div key={i} className="flex items-start gap-2 text-xs">
              <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: TONE[s.tone] }} />
              <div>
                <span className="font-medium text-slate-200">{s.label}</span>
                <span className="ml-1 text-slate-500">
                  ({(s.to - s.from >= 0 ? "+" : "")}{(s.to - s.from).toFixed(4).replace(/0+$/, "").replace(/\.$/, "")})
                </span>
                <div className="text-slate-500">{s.note}</div>
              </div>
            </div>
          ))}
          <div className="flex items-start gap-2 text-xs">
            <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm bg-[repeating-linear-gradient(45deg,#334155_0,#334155_3px,#1e293b_3px,#1e293b_6px)]" />
            <div>
              <span className="font-medium text-slate-200">{RESIDUAL.label}</span>
              <div className="text-slate-500">{RESIDUAL.note}</div>
            </div>
          </div>
        </div>
      </section>

      {/* Panel 3 — what it's NOT */}
      <section>
        <h4 className="mb-1 text-sm font-semibold text-slate-100">
          3 · What it is <span className="italic">not</span> — the falsified hypotheses
        </h4>
        <p className="mb-4 text-xs text-slate-400">
          Six candidate "hidden codes" were tested with pre-registered gates and proper nulls
          (degree-preserving swaps, random-gene-set controls, cross-replicate and neuron-holdout).
          Each was ruled out. Tap for the kill statistic.
        </p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {FALSIFIED.map((f, i) => (
            <button
              key={i}
              onClick={() => setOpen(open === i ? null : i)}
              className="rounded-lg border border-slate-700/60 bg-slate-800/40 p-3 text-left transition hover:border-slate-600 hover:bg-slate-800/70"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-200">{f.name}</span>
                <span className="rounded-full bg-rose-500/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-rose-300">
                  {f.verdict}
                </span>
              </div>
              {open === i && (
                <p className="mt-2 text-xs leading-relaxed text-slate-400">{f.stat}</p>
              )}
            </button>
          ))}
        </div>
      </section>

      <p className="mt-6 border-t border-slate-800 pt-4 text-[11px] leading-relaxed text-slate-500">
        <span className="font-semibold text-slate-400">Reading this honestly.</span>{" "}
        This is a decomposition, not a discovered blueprint. Contact-dominance echoes Brittin (2021)
        and reproducibility echoes Witvliet (2021); the contribution here is the noise-ceiling framing,
        the developmental-arrival mediation, the controlled transmitter bias, and the systematic
        falsification of the molecular-code hypotheses. The residual is largely animal-to-animal
        variability — <em>not</em> a hidden code waiting to be found.
      </p>
    </div>
  );
}
