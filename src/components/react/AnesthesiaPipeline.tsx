import * as React from "react";
import { useEffect, useMemo, useState } from "react";

/**
 * AnesthesiaPipeline — Wave P digital pharmacology playground.
 *
 * Pipeline overview, interactive binding profile, calibration story, and
 * documented boundary findings. All numbers traceable to artifacts under
 * AnestheticSimulator/ in the repo.
 */

// ===== Types ============================================================

type VerdictCategory =
  | "VERIFIED"
  | "STRUCTURALLY_GROUNDED_BY_HOMOLOG"
  | "STRUCTURALLY_GROUNDED_AWAITING_WETLAB"
  | "STRUCTURALLY_UNCALIBRATED"
  | "BOUNDARY_FAIL";

type Confidence = "HIGH" | "MEDIUM" | "LOW" | "—";

type AnestheticMeta = {
  name: string;
  smiles: string;
  mw: number | null;
  logP: number | null;
  clinical_aqueous_EC50_uM: number | null;
  kp_partition: number | null;
  chem_class: string;
};

type TargetMeta = {
  gene: string;
  uniprot: string;
  mechanism_class: string;
  structure_status: string;
  rationale: string;
};

type Prediction = {
  occupancy_v1: number | null;
  occupancy_v2_corrected: number | null;
  mechanism_class: string;
  kinetic_param: string | null;
  kinetic_value: number | string | null;
  verdict_category: VerdictCategory;
  verdict_confidence: Confidence;
  verdict_comment: string;
};

type BindingProfile = {
  anesthetics: AnestheticMeta[];
  targets: TargetMeta[];
  predictions: Record<string, Record<string, Prediction>>;
  _meta: {
    f_allo_correction: number;
    correction_source: string;
    overlay_version: string;
  };
};

type NegLigand = {
  name: string;
  rationale: string;
  eger_status: string;
  engagement_count_at_1mM: number;
  n_targets_dock: number;
  median_predicted_Kd_uM: number | null;
  per_target: Record<string, { vina_dG: number; predicted_Kd_uM: number; occupancy_at_1mM: number }>;
};

type NegativeControls = {
  ligands: NegLigand[];
  _meta: { concentration_uM: number; engagement_threshold_occupancy: number };
};

type ChemClassMetric = {
  chem_class: string;
  n: number;
  pre_signed_mean: number;
  post_signed_mean: number;
  pre_mean_abs: number;
  post_mean_abs: number;
  post_pct_10x: number;
  post_pct_3x: number;
};

type CalibrationSummary = {
  f_allo_correction: number;
  strict_subset_n: number;
  strict_subset_pre: { within_10x_count: number; within_10x_pct: number; signed_mean_log_err: number; mean_abs_log_err: number };
  strict_subset_post: { within_10x_count: number; within_10x_pct: number; within_3x_count: number; within_3x_pct: number; signed_mean_log_err: number; mean_abs_log_err: number };
  per_chem_class: ChemClassMetric[];
  verdict_counts: Record<string, number>;
  verdict_descriptions: Record<string, string>;
  rigor_pass_summary: Array<{ cp: string; topic: string; verdict: string; key_number: string }>;
};

type DoseResponse = {
  anesthetic: string;
  substrate: string;
  doses: Array<{ dose_multiplier: number; firing_rate_Hz: number; n_spikes: number; max_class_occupancy: number; hyperpolarization_pA: number }>;
  honest_gap: { demo_50pct_suppression_dose: number; literature_behavioral_EC50_dose: number; fold_off_from_literature: number; interpretation: string };
};

type PipelineMeta = {
  phases: Array<{ id: string; title: string; status: string; summary: string; deferred?: string }>;
  computational_scope: { external_spend_USD: number; compute: string; envs: string[]; deferred_due_to_compute: string[] };
  key_anchors: Record<string, number>;
};

type CaseStudy = {
  filename: string;
  title: string;
  summary: string;
  word_count: number;
  github_path: string;
};

type CaseStudies = {
  case_studies: CaseStudy[];
  _meta: { total_word_count: number; umbrella_thesis: string };
};

// ===== Helpers ==========================================================

const DATA_BASE = "/data/anesthesia";

function fetchJSON<T>(path: string): Promise<T> {
  return fetch(`${DATA_BASE}/${path}`).then((r) => {
    if (!r.ok) throw new Error(`fetch ${path} failed: ${r.status}`);
    return r.json() as Promise<T>;
  });
}

function occupancyColor(occ: number | null): string {
  if (occ === null || occ === undefined) return "#e5e7eb";
  // Sequential colormap: 0 → light gray, 1 → deep purple
  const t = Math.max(0, Math.min(1, occ));
  // Interpolate between #f3f4f6 (gray-100) and #581c87 (purple-900)
  const r1 = 243, g1 = 244, b1 = 246;
  const r2 = 88, g2 = 28, b2 = 135;
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r},${g},${b})`;
}

function verdictColor(v: VerdictCategory): string {
  switch (v) {
    case "VERIFIED": return "bg-emerald-100 text-emerald-900 border-emerald-300";
    case "STRUCTURALLY_GROUNDED_BY_HOMOLOG": return "bg-sky-100 text-sky-900 border-sky-300";
    case "STRUCTURALLY_GROUNDED_AWAITING_WETLAB": return "bg-amber-100 text-amber-900 border-amber-300";
    case "STRUCTURALLY_UNCALIBRATED": return "bg-zinc-100 text-zinc-700 border-zinc-300";
    case "BOUNDARY_FAIL": return "bg-rose-100 text-rose-900 border-rose-300";
  }
}

function verdictShortLabel(v: VerdictCategory): string {
  switch (v) {
    case "VERIFIED": return "verified";
    case "STRUCTURALLY_GROUNDED_BY_HOMOLOG": return "homolog-grounded";
    case "STRUCTURALLY_GROUNDED_AWAITING_WETLAB": return "awaiting wet-lab";
    case "STRUCTURALLY_UNCALIBRATED": return "uncalibrated";
    case "BOUNDARY_FAIL": return "boundary FAIL";
  }
}

function confidenceBadge(c: Confidence): string {
  switch (c) {
    case "HIGH": return "bg-emerald-600 text-white";
    case "MEDIUM": return "bg-amber-500 text-white";
    case "LOW": return "bg-rose-500 text-white";
    default: return "bg-zinc-300 text-zinc-700";
  }
}

const MECHANISM_LABELS: Record<string, string> = {
  gaba_potentiation: "GABA-A potentiation",
  glucl_potentiation: "GluCl potentiation",
  k2p_potentiation: "K2P potentiation",
  nachr_antagonism: "nAChR antagonism",
  complex_i_block: "Complex I block",
  complex_ii_block: "Complex II block",
  snare_cooperativity: "SNARE release",
  nca_block: "NCA leak block",
};

const MECHANISM_ORDER = [
  "gaba_potentiation",
  "glucl_potentiation",
  "k2p_potentiation",
  "nachr_antagonism",
  "complex_i_block",
  "complex_ii_block",
  "snare_cooperativity",
  "nca_block",
];

// ===== Subcomponents ====================================================

function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-3">
      <h3 className="text-base font-semibold tracking-tight text-zinc-900">{title}</h3>
      {sub && <p className="text-xs text-zinc-600 mt-0.5">{sub}</p>}
    </div>
  );
}

function PipelineDiagram({ meta }: { meta: PipelineMeta }) {
  const statusFill = (status: string): string => {
    if (status.startsWith("SHIPPED")) return "#10b981"; // emerald-500
    if (status.startsWith("IN_PROGRESS")) return "#f59e0b"; // amber-500
    if (status.includes("PARAMETER_LOCKED") || status.includes("SCAFFOLDED")) return "#94a3b8"; // slate-400
    return "#cbd5e1"; // slate-300
  };
  const statusLabel = (status: string): string => {
    if (status.startsWith("SHIPPED")) return "shipped";
    if (status.startsWith("IN_PROGRESS")) return "in progress";
    if (status.includes("PARAMETER_LOCKED")) return "parameter-locked";
    if (status.includes("SCAFFOLDED")) return "scaffold";
    return "deferred";
  };
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-6">
      <SectionHeader
        title="Pipeline architecture"
        sub="Honest scope per phase. Hover for one-line summary."
      />
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
        {meta.phases.map((p) => (
          <div
            key={p.id}
            className="rounded-md border border-zinc-200 px-2.5 py-2 hover:border-zinc-400 transition-colors"
            title={p.summary}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono font-semibold text-zinc-900">Phase {p.id}</span>
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ backgroundColor: statusFill(p.status) }}
              />
            </div>
            <div className="font-medium text-zinc-800 leading-tight">{p.title}</div>
            <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wide">
              {statusLabel(p.status)}
            </div>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-zinc-500 mt-3 leading-relaxed">
        <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1 align-middle" /> shipped &nbsp;·&nbsp;
        <span className="inline-block w-2 h-2 rounded-full bg-amber-500 mr-1 align-middle" /> in progress &nbsp;·&nbsp;
        <span className="inline-block w-2 h-2 rounded-full bg-slate-400 mr-1 align-middle" /> scaffold / parameter-locked &nbsp;·&nbsp;
        <span className="inline-block w-2 h-2 rounded-full bg-slate-300 mr-1 align-middle" /> deferred
      </p>
    </div>
  );
}

function CompoundSelector({
  anesthetics,
  selected,
  onSelect,
  comparisonMode,
  onToggleComparison,
  selectedB,
  onSelectB,
  customSMILES,
  onSetCustomSMILES,
  negLigands,
}: {
  anesthetics: AnestheticMeta[];
  selected: string;
  onSelect: (n: string) => void;
  comparisonMode: boolean;
  onToggleComparison: () => void;
  selectedB: string;
  onSelectB: (n: string) => void;
  customSMILES: string;
  onSetCustomSMILES: (s: string) => void;
  negLigands: NegLigand[];
}) {
  const allCompounds = useMemo(() => {
    const an = anesthetics.map((a) => ({ name: a.name, kind: "anesthetic" as const }));
    const ng = negLigands.map((n) => ({ name: n.name, kind: "negative" as const }));
    return [...an, ...ng];
  }, [anesthetics, negLigands]);

  const validSMILES = useMemo(() => {
    if (!customSMILES) return null;
    // Basic syntax check: balanced parens/brackets and only allowed chars
    const cleaned = customSMILES.trim();
    if (cleaned.length === 0) return null;
    if (cleaned.length > 200) return false;
    if (!/^[A-Za-z0-9@+\-\[\]\(\)=#$/\\.%:]+$/.test(cleaned)) return false;
    let parens = 0, brackets = 0;
    for (const c of cleaned) {
      if (c === "(") parens++;
      else if (c === ")") parens--;
      else if (c === "[") brackets++;
      else if (c === "]") brackets--;
      if (parens < 0 || brackets < 0) return false;
    }
    return parens === 0 && brackets === 0;
  }, [customSMILES]);

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Compound"
        sub="14 cached compounds (6 anesthetics + 8 negative controls). Custom SMILES requires backend pipeline run — see note below."
      />
      <div className="flex flex-col sm:flex-row gap-3 items-start">
        <div className="flex-1 min-w-0">
          <label className="text-xs font-medium text-zinc-700 block mb-1">
            Primary compound
          </label>
          <select
            value={selected}
            onChange={(e) => onSelect(e.target.value)}
            className="w-full rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm font-mono"
          >
            <optgroup label="Clinical anesthetics (6)">
              {anesthetics.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name} · {a.chem_class.replace(/_/g, " ").toLowerCase()}
                </option>
              ))}
            </optgroup>
            <optgroup label="Negative controls (8)">
              {negLigands.map((n) => (
                <option key={n.name} value={n.name}>
                  {n.name.replace(/_/g, " ")} · {n.eger_status.toLowerCase().split(" (")[0]}
                </option>
              ))}
            </optgroup>
          </select>
        </div>
        <div className="flex items-center gap-2 sm:pt-5">
          <input
            type="checkbox"
            id="comparison-toggle"
            checked={comparisonMode}
            onChange={onToggleComparison}
            className="h-4 w-4 rounded border-zinc-300"
          />
          <label htmlFor="comparison-toggle" className="text-sm text-zinc-700 cursor-pointer">
            Comparison mode
          </label>
        </div>
        {comparisonMode && (
          <div className="flex-1 min-w-0">
            <label className="text-xs font-medium text-zinc-700 block mb-1">
              Comparison compound
            </label>
            <select
              value={selectedB}
              onChange={(e) => onSelectB(e.target.value)}
              className="w-full rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm font-mono"
            >
              <optgroup label="Clinical anesthetics">
                {anesthetics.map((a) => (
                  <option key={a.name} value={a.name}>{a.name}</option>
                ))}
              </optgroup>
              <optgroup label="Negative controls">
                {negLigands.map((n) => (
                  <option key={n.name} value={n.name}>{n.name.replace(/_/g, " ")}</option>
                ))}
              </optgroup>
            </select>
          </div>
        )}
      </div>

      <div className="mt-3 pt-3 border-t border-zinc-100">
        <label className="text-xs font-medium text-zinc-700 block mb-1">
          Custom SMILES (research-only — see caveat)
        </label>
        <input
          type="text"
          value={customSMILES}
          onChange={(e) => onSetCustomSMILES(e.target.value)}
          placeholder="e.g., CC(C)c1cccc(C(C)C)c1O"
          className="w-full rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-xs font-mono"
        />
        {customSMILES && (
          <div className={`text-[11px] mt-1.5 ${validSMILES === false ? "text-rose-600" : "text-zinc-600"}`}>
            {validSMILES === false ? (
              <span>Invalid SMILES syntax (atoms/bonds malformed or unbalanced parens/brackets).</span>
            ) : validSMILES === true ? (
              <span>
                SMILES syntax OK. <strong>Predictions for custom compounds require a backend pipeline run</strong> (Phase A AlphaFold + Phase B Vina; ~30 min on RTX 4060 Ti). This static deployment ships predictions only for the 14 cached compounds. Run the pipeline locally — see the GitHub repo at <code className="font-mono text-[10px]">AnestheticSimulator/</code>.
              </span>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function BindingHeatmap({
  preds,
  targets,
  selectedTarget,
  onSelectTarget,
  preds2,
}: {
  preds: Record<string, Prediction>;
  targets: TargetMeta[];
  selectedTarget: string | null;
  onSelectTarget: (g: string | null) => void;
  preds2?: Record<string, Prediction>;
}) {
  // Group targets by mechanism class for display
  const byClass = useMemo(() => {
    const groups: Record<string, TargetMeta[]> = {};
    for (const t of targets) {
      const k = t.mechanism_class || "other";
      if (!groups[k]) groups[k] = [];
      groups[k].push(t);
    }
    return groups;
  }, [targets]);

  const classOrder = useMemo(
    () => MECHANISM_ORDER.filter((c) => byClass[c]).concat(
      Object.keys(byClass).filter((c) => !MECHANISM_ORDER.includes(c))
    ),
    [byClass]
  );

  const cellW = 18;
  const cellH = 26;
  const labelH = 70;

  const totalTargets = targets.length;
  const svgW = totalTargets * cellW + 40;
  const svgH = labelH + cellH * (preds2 ? 2 : 1) + 8;

  let xPos = 20;

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Binding profile heatmap"
        sub="Predicted occupancy at 1× clinical EC50, post-allosteric correction (CP5 f_allo = 2.50×). Click any cell for target detail."
      />
      <div className="overflow-x-auto">
        <svg width={svgW} height={svgH} role="img" aria-label="Binding profile heatmap">
          {/* Class group headers + cells */}
          {classOrder.map((cls) => {
            const tgs = byClass[cls];
            const groupX = xPos;
            const groupW = tgs.length * cellW;
            xPos += groupW + 4; // small gap between classes
            return (
              <g key={cls}>
                {/* Class label rotated */}
                <text
                  x={groupX + groupW / 2}
                  y={labelH - 4}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#6b7280"
                  fontWeight={500}
                  className="select-none"
                >
                  {(MECHANISM_LABELS[cls] || cls).replace(" potentiation", "+").replace(" antagonism", "−").replace(" block", "↓").replace(" release", "↓r")}
                </text>
                {/* Group separator line */}
                <line
                  x1={groupX}
                  y1={labelH - 18}
                  x2={groupX + groupW}
                  y2={labelH - 18}
                  stroke="#9ca3af"
                  strokeWidth={1}
                />
                {tgs.map((t, i) => {
                  const cellX = groupX + i * cellW;
                  const p = preds[t.gene];
                  const occ = p?.occupancy_v2_corrected ?? null;
                  const isSelected = selectedTarget === t.gene;
                  return (
                    <g key={t.gene}>
                      {/* Target gene label rotated -45° above heatmap */}
                      <text
                        x={cellX + cellW / 2}
                        y={labelH - 22}
                        fontSize={8}
                        fill="#374151"
                        textAnchor="end"
                        transform={`rotate(-65, ${cellX + cellW / 2}, ${labelH - 22})`}
                        className="select-none font-mono"
                      >
                        {t.gene}
                      </text>
                      {/* Primary cell */}
                      <rect
                        x={cellX + 1}
                        y={labelH}
                        width={cellW - 2}
                        height={cellH - 2}
                        fill={occupancyColor(occ)}
                        stroke={isSelected ? "#7c3aed" : "#e5e7eb"}
                        strokeWidth={isSelected ? 2 : 1}
                        onClick={() => onSelectTarget(isSelected ? null : t.gene)}
                        style={{ cursor: "pointer" }}
                      >
                        <title>
                          {t.gene} · {(MECHANISM_LABELS[cls] || cls)} · occupancy {occ !== null ? occ.toFixed(3) : "—"}
                          {p?.verdict_category ? ` · ${verdictShortLabel(p.verdict_category)}` : ""}
                        </title>
                      </rect>
                      {/* Comparison row if comparison mode */}
                      {preds2 && (() => {
                        const occ2 = preds2[t.gene]?.occupancy_v2_corrected ?? null;
                        return (
                          <rect
                            x={cellX + 1}
                            y={labelH + cellH}
                            width={cellW - 2}
                            height={cellH - 2}
                            fill={occupancyColor(occ2)}
                            stroke={"#e5e7eb"}
                            strokeWidth={1}
                            onClick={() => onSelectTarget(isSelected ? null : t.gene)}
                            style={{ cursor: "pointer" }}
                          >
                            <title>
                              {t.gene} (B) · occupancy {occ2 !== null ? occ2.toFixed(3) : "—"}
                            </title>
                          </rect>
                        );
                      })()}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      {/* Legend */}
      <div className="flex items-center gap-3 text-[11px] text-zinc-600 mt-2 flex-wrap">
        <span className="font-medium">Occupancy:</span>
        <div className="flex items-center gap-1">
          {[0.0, 0.25, 0.5, 0.75, 1.0].map((v) => (
            <div key={v} className="flex items-center gap-1">
              <div
                className="w-4 h-3 border border-zinc-300"
                style={{ backgroundColor: occupancyColor(v) }}
              />
              <span>{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TargetDetailPanel({
  target,
  pred,
  predB,
  comparisonMode,
  primaryName,
  comparisonName,
}: {
  target: TargetMeta | null;
  pred: Prediction | null;
  predB: Prediction | null;
  comparisonMode: boolean;
  primaryName: string;
  comparisonName: string;
}) {
  if (!target) {
    return (
      <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 mb-4 text-sm text-zinc-500">
        Click a heatmap cell to see target detail (predicted Kd, kinetic shift, verdict category, anchor).
      </div>
    );
  }

  const renderPredCol = (p: Prediction | null, name: string) => {
    if (!p) return <div className="text-zinc-400 text-sm">no prediction</div>;
    const occ = p.occupancy_v2_corrected ?? p.occupancy_v1;
    return (
      <div>
        <div className="text-xs text-zinc-500 mb-1 uppercase tracking-wide">{name}</div>
        <div className="space-y-1.5 text-sm">
          <div>
            <span className="text-zinc-500">occupancy @ 1× EC50:&nbsp;</span>
            <span className="font-mono font-semibold">{occ !== null ? occ.toFixed(3) : "—"}</span>
            {p.occupancy_v1 !== null && p.occupancy_v2_corrected !== null && (
              <span className="text-[10px] text-zinc-400 ml-1.5">
                (v1 {p.occupancy_v1.toFixed(2)} → v2 {p.occupancy_v2_corrected.toFixed(2)})
              </span>
            )}
          </div>
          {p.kinetic_param && p.kinetic_value !== null && (
            <div>
              <span className="text-zinc-500">kinetic shift:&nbsp;</span>
              <span className="font-mono">{p.kinetic_param}</span>
              <span className="text-zinc-700">&nbsp;= </span>
              <span className="font-mono font-semibold">
                {typeof p.kinetic_value === "number" ? p.kinetic_value.toFixed(3) : String(p.kinetic_value)}
              </span>
            </div>
          )}
          <div className="pt-1">
            <span
              className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold border ${verdictColor(p.verdict_category)}`}
            >
              {verdictShortLabel(p.verdict_category)}
            </span>
            <span
              className={`inline-block ml-1 px-1.5 py-0.5 rounded text-[10px] font-semibold ${confidenceBadge(p.verdict_confidence)}`}
            >
              {p.verdict_confidence}
            </span>
          </div>
          <div className="text-[11px] text-zinc-600 leading-snug pt-0.5">{p.verdict_comment}</div>
        </div>
      </div>
    );
  };

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="font-mono text-base font-semibold text-zinc-900">{target.gene}</div>
          <div className="text-xs text-zinc-500 font-mono">{target.uniprot}</div>
          <div className="text-[11px] text-zinc-600 mt-1 max-w-xl leading-snug">{target.rationale}</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">mechanism</div>
          <div className="text-xs font-medium text-zinc-800">
            {MECHANISM_LABELS[target.mechanism_class] || target.mechanism_class}
          </div>
        </div>
      </div>
      <div className={`grid gap-4 ${comparisonMode ? "sm:grid-cols-2" : "grid-cols-1"}`}>
        {renderPredCol(pred, primaryName)}
        {comparisonMode && renderPredCol(predB, comparisonName)}
      </div>
    </div>
  );
}

function MechanismSummary({
  predictions,
  targets,
  primaryName,
  predictionsB,
  comparisonMode,
  comparisonName,
}: {
  predictions: Record<string, Prediction>;
  targets: TargetMeta[];
  primaryName: string;
  predictionsB?: Record<string, Prediction>;
  comparisonMode: boolean;
  comparisonName: string;
}) {
  const byClass = useMemo(() => {
    const out: Record<string, { sum: number; max: number; count: number; sumB: number; maxB: number; countB: number }> = {};
    for (const t of targets) {
      const cls = t.mechanism_class || "other";
      if (!out[cls]) out[cls] = { sum: 0, max: 0, count: 0, sumB: 0, maxB: 0, countB: 0 };
      const p = predictions[t.gene];
      const occ = p?.occupancy_v2_corrected ?? null;
      if (occ !== null) {
        out[cls].sum += occ;
        out[cls].max = Math.max(out[cls].max, occ);
        out[cls].count += 1;
      }
      if (predictionsB) {
        const pb = predictionsB[t.gene];
        const occB = pb?.occupancy_v2_corrected ?? null;
        if (occB !== null) {
          out[cls].sumB += occB;
          out[cls].maxB = Math.max(out[cls].maxB, occB);
          out[cls].countB += 1;
        }
      }
    }
    return out;
  }, [predictions, predictionsB, targets]);

  const orderedClasses = MECHANISM_ORDER.filter((c) => byClass[c]).concat(
    Object.keys(byClass).filter((c) => !MECHANISM_ORDER.includes(c))
  );

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Mechanism class engagement"
        sub="Max per-class occupancy at 1× clinical EC50. Multi-target binders engage many classes; selective ligands few."
      />
      <div className="space-y-2">
        {orderedClasses.map((cls) => {
          const d = byClass[cls];
          const maxOcc = d.max;
          const maxOccB = d.maxB;
          return (
            <div key={cls} className="text-xs">
              <div className="flex justify-between mb-0.5">
                <span className="text-zinc-700 font-medium">
                  {MECHANISM_LABELS[cls] || cls}
                </span>
                <span className="text-zinc-500 font-mono">
                  {maxOcc.toFixed(2)}{comparisonMode ? ` vs ${maxOccB.toFixed(2)}` : ""}
                  <span className="text-zinc-400 ml-1">({d.count} targets)</span>
                </span>
              </div>
              <div className="w-full h-3 bg-zinc-100 rounded-full overflow-hidden flex">
                <div
                  className="h-full bg-purple-600 transition-all"
                  style={{ width: `${maxOcc * 100}%` }}
                  title={`${primaryName}: ${maxOcc.toFixed(3)}`}
                />
              </div>
              {comparisonMode && (
                <div className="w-full h-3 bg-zinc-100 rounded-full overflow-hidden flex mt-0.5">
                  <div
                    className="h-full bg-rose-500 transition-all"
                    style={{ width: `${maxOccB * 100}%` }}
                    title={`${comparisonName}: ${maxOccB.toFixed(3)}`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
      {comparisonMode && (
        <div className="text-[11px] text-zinc-600 mt-3 flex gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 bg-purple-600 rounded-sm" />
            {primaryName}
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 bg-rose-500 rounded-sm" />
            {comparisonName}
          </span>
        </div>
      )}
    </div>
  );
}

function DoseResponseChart({ data, currentAnesthetic }: { data: DoseResponse; currentAnesthetic: string }) {
  const w = 600;
  const h = 220;
  const pad = { top: 10, right: 16, bottom: 36, left: 44 };
  const innerW = w - pad.left - pad.right;
  const innerH = h - pad.top - pad.bottom;

  const doses = data.doses;
  const maxRate = Math.max(...doses.map((d) => d.firing_rate_Hz), 1);
  const xMin = Math.log10(doses[0].dose_multiplier);
  const xMax = Math.log10(doses[doses.length - 1].dose_multiplier);

  const xScale = (dose: number) =>
    pad.left + ((Math.log10(dose) - xMin) / (xMax - xMin)) * innerW;
  const yScale = (rate: number) => pad.top + innerH - (rate / maxRate) * innerH;

  const pathD = doses
    .map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d.dose_multiplier).toFixed(1)} ${yScale(d.firing_rate_Hz).toFixed(1)}`)
    .join(" ");

  const isHalothane = currentAnesthetic === "halothane";

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Predicted network effect (Phase G dose-response)"
        sub={
          isHalothane
            ? "Halothane on a minimal Brian2 LIF demo network (40 E + 10 I). Consumes wave2_overlay_v2 + Phase G perturbation hooks."
            : "Phase G calibration in progress — halothane is the canonical demo. Other anesthetics will land with LIFBrain integration."
        }
      />
      {isHalothane ? (
        <>
          <svg width={w} height={h} role="img" aria-label="Halothane dose-response">
            {/* Y axis */}
            <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + innerH} stroke="#9ca3af" />
            {[0, 0.25, 0.5, 0.75, 1.0].map((f) => {
              const v = f * maxRate;
              const y = yScale(v);
              return (
                <g key={f}>
                  <line x1={pad.left - 4} y1={y} x2={pad.left} y2={y} stroke="#9ca3af" />
                  <text
                    x={pad.left - 6}
                    y={y + 3}
                    textAnchor="end"
                    fontSize={10}
                    fill="#6b7280"
                  >
                    {v.toFixed(0)}
                  </text>
                </g>
              );
            })}
            <text
              x={-pad.top - innerH / 2}
              y={14}
              transform={`rotate(-90)`}
              textAnchor="middle"
              fontSize={11}
              fill="#374151"
            >
              firing rate (Hz)
            </text>

            {/* X axis */}
            <line
              x1={pad.left}
              y1={pad.top + innerH}
              x2={pad.left + innerW}
              y2={pad.top + innerH}
              stroke="#9ca3af"
            />
            {doses.map((d) => (
              <g key={d.dose_multiplier}>
                <line
                  x1={xScale(d.dose_multiplier)}
                  y1={pad.top + innerH}
                  x2={xScale(d.dose_multiplier)}
                  y2={pad.top + innerH + 4}
                  stroke="#9ca3af"
                />
                <text
                  x={xScale(d.dose_multiplier)}
                  y={pad.top + innerH + 14}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#6b7280"
                >
                  {d.dose_multiplier < 0.01
                    ? d.dose_multiplier.toFixed(3)
                    : d.dose_multiplier.toFixed(d.dose_multiplier < 0.1 ? 2 : 1)}
                </text>
              </g>
            ))}
            <text
              x={pad.left + innerW / 2}
              y={h - 6}
              textAnchor="middle"
              fontSize={11}
              fill="#374151"
            >
              dose × clinical EC50 (log scale)
            </text>

            {/* 50% line */}
            <line
              x1={pad.left}
              y1={yScale(maxRate / 2)}
              x2={pad.left + innerW}
              y2={yScale(maxRate / 2)}
              stroke="#cbd5e1"
              strokeDasharray="4 3"
            />
            <text
              x={pad.left + innerW - 4}
              y={yScale(maxRate / 2) - 3}
              textAnchor="end"
              fontSize={9}
              fill="#94a3b8"
            >
              50% baseline
            </text>

            {/* Literature 1× EC50 marker */}
            <line
              x1={xScale(1.0)}
              y1={pad.top}
              x2={xScale(1.0)}
              y2={pad.top + innerH}
              stroke="#f59e0b"
              strokeDasharray="3 3"
            />
            <text
              x={xScale(1.0) + 4}
              y={pad.top + 12}
              fontSize={10}
              fill="#b45309"
              fontWeight={500}
            >
              1× clinical EC50 (Crowder 1996)
            </text>

            {/* Curve */}
            <path d={pathD} fill="none" stroke="#7c3aed" strokeWidth={2} />
            {doses.map((d) => (
              <circle
                key={d.dose_multiplier}
                cx={xScale(d.dose_multiplier)}
                cy={yScale(d.firing_rate_Hz)}
                r={3}
                fill="#7c3aed"
              >
                <title>
                  dose {d.dose_multiplier}× → {d.firing_rate_Hz.toFixed(1)} Hz, {d.hyperpolarization_pA.toFixed(0)} pA hyperpol
                </title>
              </circle>
            ))}
          </svg>
          <div className="mt-3 rounded-md bg-amber-50 border border-amber-200 px-3 py-2 text-[11px] text-amber-900 leading-relaxed">
            <strong>Honest gap:</strong> demo network 50%-suppression at {data.honest_gap.demo_50pct_suppression_dose}× clinical EC50 — {data.honest_gap.fold_off_from_literature}× tighter than Crowder 1996 PMID 8873562 behavioral anchor at 1×. Two factors: binding-side saturation (occupancy ≈ 1 across all 30 targets at 1× EC50, compressing dose-response) and demo-network coupling sensitivity (no muscle buffer, no graded-potential redundancy). Behavioral threshold calibration is the next gap; LIFBrain integration is the next bet.
          </div>
        </>
      ) : (
        <div className="rounded-md bg-zinc-50 border border-zinc-200 px-3 py-3 text-xs text-zinc-600 leading-relaxed">
          Phase G dose-response is currently calibrated only for halothane (the canonical anesthetic with the cleanest binding-side anchor: KCNK2 log_err 0.001). Other anesthetics will ship dose-response curves once Phase G integrates with LIFBrain on the full 300-neuron substrate. The selected compound's binding profile (above) is fully shipped.
        </div>
      )}
    </div>
  );
}

function CalibrationStoryPanel({ summary }: { summary: CalibrationSummary }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Calibration story (CP1-CP8 rigor pass)"
        sub={`Single-parameter allosteric correction f_allo = ${summary.f_allo_correction}× shifts strict T1 subset within-10× from ${summary.strict_subset_pre.within_10x_pct.toFixed(0)}% to ${summary.strict_subset_post.within_10x_pct.toFixed(0)}%.`}
      />

      <div className="grid sm:grid-cols-2 gap-3 mb-3">
        <div className="rounded-md bg-zinc-50 border border-zinc-200 p-2.5">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">Pre-correction (T1, n={summary.strict_subset_n})</div>
          <div className="text-sm space-y-0.5">
            <div><span className="font-mono">{summary.strict_subset_pre.within_10x_pct.toFixed(0)}%</span> within 10×</div>
            <div className="text-zinc-600">signed mean log_err: <span className="font-mono">{summary.strict_subset_pre.signed_mean_log_err > 0 ? "+" : ""}{summary.strict_subset_pre.signed_mean_log_err.toFixed(2)}</span></div>
            <div className="text-zinc-600">mean |log_err|: <span className="font-mono">{summary.strict_subset_pre.mean_abs_log_err.toFixed(2)}</span></div>
          </div>
        </div>
        <div className="rounded-md bg-emerald-50 border border-emerald-200 p-2.5">
          <div className="text-[10px] text-emerald-700 uppercase tracking-wide mb-1">Post-correction (T1, n={summary.strict_subset_n})</div>
          <div className="text-sm space-y-0.5">
            <div><span className="font-mono font-semibold">{summary.strict_subset_post.within_10x_pct.toFixed(0)}%</span> within 10×, <span className="font-mono">{summary.strict_subset_post.within_3x_pct.toFixed(0)}%</span> within 3×</div>
            <div className="text-zinc-600">signed mean log_err: <span className="font-mono">{summary.strict_subset_post.signed_mean_log_err > 0 ? "+" : ""}{summary.strict_subset_post.signed_mean_log_err.toFixed(2)}</span></div>
            <div className="text-zinc-600">mean |log_err|: <span className="font-mono">{summary.strict_subset_post.mean_abs_log_err.toFixed(2)}</span></div>
          </div>
        </div>
      </div>

      <div className="mb-3">
        <div className="text-xs font-medium text-zinc-700 mb-1.5">Per-chemical-class metrics (post-correction)</div>
        <div className="overflow-x-auto">
          <table className="text-[11px] w-full">
            <thead className="bg-zinc-50 text-zinc-600">
              <tr>
                <th className="text-left px-2 py-1">class</th>
                <th className="text-right px-2 py-1">n</th>
                <th className="text-right px-2 py-1">signed_mean</th>
                <th className="text-right px-2 py-1">mean |log_err|</th>
                <th className="text-right px-2 py-1">% within 10×</th>
              </tr>
            </thead>
            <tbody>
              {summary.per_chem_class.map((c) => (
                <tr key={c.chem_class} className="border-t border-zinc-100">
                  <td className="px-2 py-1 font-mono">{c.chem_class.replace(/_/g, " ").toLowerCase()}</td>
                  <td className="px-2 py-1 text-right">{c.n}</td>
                  <td className="px-2 py-1 text-right font-mono">{c.post_signed_mean > 0 ? "+" : ""}{c.post_signed_mean.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right font-mono">{c.post_mean_abs.toFixed(2)}</td>
                  <td className={`px-2 py-1 text-right font-mono ${c.post_pct_10x === 100 ? "text-emerald-700 font-semibold" : ""}`}>
                    {c.post_pct_10x.toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mb-3">
        <div className="text-xs font-medium text-zinc-700 mb-1.5">Verdict counts (replaces v1 5/5 PASS)</div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {Object.entries(summary.verdict_counts).map(([k, v]) => (
            <div key={k} className={`rounded-md px-2 py-2 border text-center ${verdictColor(k as VerdictCategory)}`}>
              <div className="text-2xl font-bold tabular-nums">{v}</div>
              <div className="text-[10px] uppercase tracking-tight leading-tight">{verdictShortLabel(k as VerdictCategory)}</div>
            </div>
          ))}
        </div>
      </div>

      <details className="text-xs">
        <summary className="cursor-pointer text-zinc-700 font-medium">Rigor checkpoint summary (CP1-CP8)</summary>
        <div className="mt-2 space-y-1.5 text-[11px]">
          {summary.rigor_pass_summary.map((cp) => (
            <div key={cp.cp} className="flex gap-2">
              <span className="font-mono font-semibold text-zinc-700 w-10 shrink-0">{cp.cp}</span>
              <span className="text-zinc-600">
                <span className="font-medium text-zinc-800">{cp.topic}</span> · {cp.verdict} ·{" "}
                <span className="text-zinc-500">{cp.key_number}</span>
              </span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

function BoundaryFindingsCard({ negative }: { negative: NegativeControls }) {
  // Find the Eger non-immobilizers + cis-DCE for the headline
  const find = (n: string) => negative.ligands.find((l) => l.name === n);
  const cisDCE = find("cis_12_dichloroethylene");
  const transDCE = find("trans_12_dichloroethylene");
  const hfe = find("hexafluoroethane");

  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50/50 p-4 mb-4">
      <SectionHeader
        title="Boundary findings — what the binding pipeline cannot do"
        sub="Two explicit boundary tests. Documented as the limit of the binding-pipeline output, not as failures of the broader project."
      />
      <div className="grid sm:grid-cols-2 gap-3 text-xs">
        <div className="rounded-md bg-white border border-rose-200 p-2.5">
          <div className="text-[10px] text-rose-700 uppercase tracking-wide mb-1">CP3 — conformational specificity</div>
          <div className="text-sm font-medium text-zinc-900 mb-1">
            cis-1,2-DCE vs trans-1,2-DCE (Eger 2001)
          </div>
          {cisDCE && transDCE && (
            <div className="text-[11px] text-zinc-700 leading-relaxed">
              cis-DCE (anesthetic per Eger): engages <strong>{cisDCE.engagement_count_at_1mM}/{cisDCE.n_targets_dock}</strong> targets at 1 mM.<br />
              trans-DCE (non-anesthetic per Eger): engages <strong>{transDCE.engagement_count_at_1mM}/{transDCE.n_targets_dock}</strong>.<br />
              <span className="text-rose-700 font-medium">Max gap = 0 across 0.1-30 mM.</span> Pipeline cannot distinguish stereoisomers.
            </div>
          )}
        </div>
        <div className="rounded-md bg-white border border-rose-200 p-2.5">
          <div className="text-[10px] text-rose-700 uppercase tracking-wide mb-1">CP7 — Eger non-immobilizer</div>
          <div className="text-sm font-medium text-zinc-900 mb-1">
            Hexafluoroethane vs cis-DCE
          </div>
          {hfe && cisDCE && (
            <div className="text-[11px] text-zinc-700 leading-relaxed">
              hexafluoroethane (Eger non-immobilizer): engages <strong>{hfe.engagement_count_at_1mM}/{hfe.n_targets_dock}</strong> targets at 1 mM.<br />
              cis-DCE (anesthetic positive control): engages <strong>{cisDCE.engagement_count_at_1mM}/{cisDCE.n_targets_dock}</strong>.<br />
              <span className="text-rose-700 font-medium">Pipeline gives the non-immobilizer a STRONGER binding profile than the anesthetic.</span> Eger discrimination is not solvable at the binding-pipeline level.
            </div>
          )}
        </div>
      </div>
      <div className="text-[11px] text-zinc-700 mt-3 leading-relaxed">
        <strong>Interpretation:</strong> the binding pipeline is a lipophilic-pocket-fit detector, not an anesthetic-specificity detector. The Eger non-immobilizer puzzle is a network-level problem: anesthetic specificity emerges from how multi-target engagement integrates at the network and behavioral threshold layers. Phase G's perturbation manager + LIFBrain integration is the next bet for capturing this — see the <em>Status</em> section.
      </div>
    </div>
  );
}

function NegativeControlsTable({ negative }: { negative: NegativeControls }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Negative-control engagement (8 ligands, 30 targets, 1 mM aqueous post-correction)"
        sub="Includes the 4 Eger 2001 non-immobilizers (cis/trans-DCE pair, hexafluoroethane) + 4 weak-narcotic / weak-anesthetic controls."
      />
      <div className="overflow-x-auto">
        <table className="text-xs w-full">
          <thead className="bg-zinc-50 text-zinc-600">
            <tr>
              <th className="text-left px-2 py-1">ligand</th>
              <th className="text-left px-2 py-1">Eger status</th>
              <th className="text-right px-2 py-1">engagement / 30</th>
              <th className="text-right px-2 py-1">median Kd (µM)</th>
            </tr>
          </thead>
          <tbody>
            {negative.ligands.map((l) => (
              <tr key={l.name} className="border-t border-zinc-100">
                <td className="px-2 py-1 font-mono">{l.name.replace(/_/g, " ")}</td>
                <td className="px-2 py-1 text-zinc-600">{l.eger_status}</td>
                <td className="px-2 py-1 text-right font-mono font-semibold">
                  {l.engagement_count_at_1mM}
                </td>
                <td className="px-2 py-1 text-right font-mono">
                  {l.median_predicted_Kd_uM !== null ? l.median_predicted_Kd_uM.toLocaleString() : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CaseStudiesPanel({ data }: { data: CaseStudies }) {
  const repoBase = "https://github.com/rohit-ravi2/personalwebsite/blob/main";
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader
        title="Methodology paper case studies (drafts)"
        sub={`5 case studies, ~${data._meta.total_word_count.toLocaleString()} words. Drafts only — not yet integrated into a paper manuscript.`}
      />
      <div className="space-y-2">
        {data.case_studies.map((c) => (
          <div key={c.filename} className="rounded-md border border-zinc-200 p-2.5 hover:border-zinc-400 transition-colors">
            <div className="flex items-center justify-between mb-1">
              <a
                href={`${repoBase}/${c.github_path}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-semibold text-zinc-900 hover:text-purple-700"
              >
                {c.title} →
              </a>
              <span className="text-[10px] text-zinc-500 font-mono">{c.word_count.toLocaleString()} words</span>
            </div>
            <p className="text-[11px] text-zinc-600 leading-snug">{c.summary}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-zinc-100">
        <div className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">Umbrella thesis</div>
        <p className="text-[11px] text-zinc-700 leading-relaxed italic">{data._meta.umbrella_thesis}</p>
      </div>
    </div>
  );
}

function StatusPanel({ meta }: { meta: PipelineMeta }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 mb-4">
      <SectionHeader title="Status & roadmap" sub="What's shipped vs in-progress vs deferred." />
      <div className="grid sm:grid-cols-2 gap-3 text-xs">
        <div>
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">Computational scope</div>
          <ul className="space-y-1 text-zinc-700">
            <li>External spend: <span className="font-mono">${meta.computational_scope.external_spend_USD}</span></li>
            <li>Compute: <span className="font-mono">{meta.computational_scope.compute}</span></li>
            <li>Envs: {meta.computational_scope.envs.join(", ")}</li>
          </ul>
        </div>
        <div>
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">Key anchors</div>
          <ul className="space-y-1 text-zinc-700 font-mono">
            <li>{meta.key_anchors.anesthetics_validated} anesthetics × {meta.key_anchors.tier1_targets} Tier-1 targets</li>
            <li>{meta.key_anchors.vina_dockings_run} Vina dockings</li>
            <li>{meta.key_anchors.calibration_dataset_size} calibration anchors</li>
            <li>{meta.key_anchors.rigor_checkpoints_passed} CP1-CP8 checkpoints</li>
            <li>{meta.key_anchors.case_studies_drafted} methodology case studies</li>
          </ul>
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-zinc-100">
        <div className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">Deferred (out of $0 / 8 GB VRAM scope)</div>
        <ul className="text-[11px] text-zinc-700 leading-snug list-disc pl-4">
          {meta.computational_scope.deferred_due_to_compute.map((d) => (
            <li key={d}>{d}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ===== Main component ===================================================

export function AnesthesiaPipeline() {
  const [binding, setBinding] = useState<BindingProfile | null>(null);
  const [negative, setNegative] = useState<NegativeControls | null>(null);
  const [calibration, setCalibration] = useState<CalibrationSummary | null>(null);
  const [doseResponse, setDoseResponse] = useState<DoseResponse | null>(null);
  const [pipelineMeta, setPipelineMeta] = useState<PipelineMeta | null>(null);
  const [caseStudies, setCaseStudies] = useState<CaseStudies | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [primary, setPrimary] = useState<string>("halothane");
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparison, setComparison] = useState<string>("hexafluoroethane");
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [customSMILES, setCustomSMILES] = useState("");

  useEffect(() => {
    Promise.all([
      fetchJSON<BindingProfile>("binding_profile.json"),
      fetchJSON<NegativeControls>("negative_controls.json"),
      fetchJSON<CalibrationSummary>("calibration_summary.json"),
      fetchJSON<DoseResponse>("dose_response.json"),
      fetchJSON<PipelineMeta>("pipeline_meta.json"),
      fetchJSON<CaseStudies>("case_studies.json"),
    ])
      .then(([b, n, c, d, m, s]) => {
        setBinding(b);
        setNegative(n);
        setCalibration(c);
        setDoseResponse(d);
        setPipelineMeta(m);
        setCaseStudies(s);
      })
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="my-6 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
        Failed to load pipeline data: {error}
      </div>
    );
  }
  if (!binding || !negative || !calibration || !doseResponse || !pipelineMeta || !caseStudies) {
    return (
      <div className="my-6 rounded-md border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-500 animate-pulse">
        Loading anesthesia-pipeline data…
      </div>
    );
  }

  // Resolve primary predictions
  const primaryIsAnesthetic = binding.predictions[primary] !== undefined;
  const primaryPreds: Record<string, Prediction> = primaryIsAnesthetic
    ? binding.predictions[primary]
    : (() => {
        // Negative control: synthesize Prediction-shaped records from negative_controls.json
        const lig = negative.ligands.find((l) => l.name === primary);
        if (!lig) return {};
        const out: Record<string, Prediction> = {};
        for (const [gene, info] of Object.entries(lig.per_target)) {
          out[gene] = {
            occupancy_v1: info.occupancy_at_1mM,
            occupancy_v2_corrected: info.occupancy_at_1mM,
            mechanism_class: binding.targets.find((t) => t.gene === gene)?.mechanism_class ?? "",
            kinetic_param: null,
            kinetic_value: null,
            verdict_category: "BOUNDARY_FAIL",
            verdict_confidence: "—",
            verdict_comment: `Negative-control compound. Engagement at 1 mM aqueous post-correction = ${info.occupancy_at_1mM.toFixed(3)}; predicted Kd ${info.predicted_Kd_uM.toFixed(0)} µM. Eger 2001 non-immobilizers should NOT engage productively at clinical concentrations — pipeline's high engagement here is the documented boundary finding (CP3, CP7).`,
          };
        }
        return out;
      })();

  const comparisonIsAnesthetic = binding.predictions[comparison] !== undefined;
  const comparisonPreds: Record<string, Prediction> = comparisonIsAnesthetic
    ? binding.predictions[comparison]
    : (() => {
        const lig = negative.ligands.find((l) => l.name === comparison);
        if (!lig) return {};
        const out: Record<string, Prediction> = {};
        for (const [gene, info] of Object.entries(lig.per_target)) {
          out[gene] = {
            occupancy_v1: info.occupancy_at_1mM,
            occupancy_v2_corrected: info.occupancy_at_1mM,
            mechanism_class: binding.targets.find((t) => t.gene === gene)?.mechanism_class ?? "",
            kinetic_param: null,
            kinetic_value: null,
            verdict_category: "BOUNDARY_FAIL",
            verdict_confidence: "—",
            verdict_comment: `Negative-control compound. Predicted Kd ${info.predicted_Kd_uM.toFixed(0)} µM at 1 mM aqueous.`,
          };
        }
        return out;
      })();

  const targetMeta = selectedTarget ? binding.targets.find((t) => t.gene === selectedTarget) ?? null : null;
  const targetPred = selectedTarget ? primaryPreds[selectedTarget] ?? null : null;
  const targetPredB = selectedTarget ? comparisonPreds[selectedTarget] ?? null : null;

  return (
    <div className="my-6">
      <PipelineDiagram meta={pipelineMeta} />

      <CompoundSelector
        anesthetics={binding.anesthetics}
        selected={primary}
        onSelect={setPrimary}
        comparisonMode={comparisonMode}
        onToggleComparison={() => setComparisonMode(!comparisonMode)}
        selectedB={comparison}
        onSelectB={setComparison}
        customSMILES={customSMILES}
        onSetCustomSMILES={setCustomSMILES}
        negLigands={negative.ligands}
      />

      <BindingHeatmap
        preds={primaryPreds}
        targets={binding.targets}
        selectedTarget={selectedTarget}
        onSelectTarget={setSelectedTarget}
        preds2={comparisonMode ? comparisonPreds : undefined}
      />

      <TargetDetailPanel
        target={targetMeta}
        pred={targetPred}
        predB={targetPredB}
        comparisonMode={comparisonMode}
        primaryName={primary}
        comparisonName={comparison}
      />

      <MechanismSummary
        predictions={primaryPreds}
        targets={binding.targets}
        primaryName={primary}
        predictionsB={comparisonMode ? comparisonPreds : undefined}
        comparisonMode={comparisonMode}
        comparisonName={comparison}
      />

      <DoseResponseChart data={doseResponse} currentAnesthetic={primary} />

      <CalibrationStoryPanel summary={calibration} />

      <BoundaryFindingsCard negative={negative} />

      <NegativeControlsTable negative={negative} />

      <CaseStudiesPanel data={caseStudies} />

      <StatusPanel meta={pipelineMeta} />
    </div>
  );
}

export default AnesthesiaPipeline;
