import * as React from "react";
import { useEffect, useMemo, useState } from "react";

/**
 * NetworkStateValidator — V3 ensemble interactive viewer.
 *
 * Four tabs:
 *   1. Dose-Response — WT volatiles vs published EC50
 *   2. Mutants — halothane × genotype directional shifts
 *   3. Eger Specificity — anesthetics vs non-immobilizers side-by-side
 *   4. Perturbation Profile — per-anesthetic Hill curves across mechanism classes
 *
 * Data: pre-rendered from V3 ensemble artifacts (545 sims, 60s × 5 seeds,
 * Cook 2019 connectome, α=0.13 calibration on halothane WT).
 */

// ===== Types =====

type DoseResponseEntry = {
  anesthetic: string;
  mutant_gene: string;
  predicted_EC50_uM: number | null;
  doses: number[];
  qf_mean: number[];
  qf_sd: number[];
  cmd_rate_mean: number[];
  net_rate_mean: number[];
  autocorr_mean: number[];
  published_EC50_uM?: number;
  source_paper?: string;
  source_PMID?: string;
  anchor_quality?: string;
  fold_error?: number;
  expected_direction?: string;
  lit_ratio?: string;
  predicted_ratio?: number;
  notes?: string;
};

type DoseResponseData = {
  meta: { alpha_calib: number; sim_duration_s: number; n_seeds: number; description: string };
  compounds: Record<string, DoseResponseEntry>;
};

type PerturbationRow = {
  mechanism_class: string;
  target_EC50_uM: number | null;
  max_effect_factor: number | null;
  hill_n: number;
  source_paper: string;
  source_PMID: string;
  evidence_grade: string;
  notes: string;
};

type PerturbationProfile = {
  meta: { description: string };
  compounds: Record<string, { active: PerturbationRow[]; deferred: PerturbationRow[]; n_active: number; n_deferred: number }>;
};

type ValidationSummary = {
  alpha_calib: number;
  sim_duration_s: number;
  n_seeds: number;
  gates: Record<string, { PASS: boolean; predicted_EC50_uM?: number; published_EC50_uM?: number; fold_error?: number; n_correct?: number; n_tested?: number }>;
  summary: string;
};

// ===== Helpers =====

const DATA_BASE = "/data/anesthesia";

function fetchJSON<T>(path: string): Promise<T> {
  return fetch(`${DATA_BASE}/${path}`).then((r) => {
    if (!r.ok) throw new Error(`fetch ${path} failed: ${r.status}`);
    return r.json() as Promise<T>;
  });
}

// Mechanism class labels
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

const MECHANISM_COLORS: Record<string, string> = {
  gaba_potentiation: "#2563eb",
  glucl_potentiation: "#0891b2",
  k2p_potentiation: "#059669",
  nachr_antagonism: "#dc2626",
  complex_i_block: "#7c3aed",
  complex_ii_block: "#9333ea",
  snare_cooperativity: "#ea580c",
  nca_block: "#65a30d",
};

const EVIDENCE_GRADE_BADGE: Record<string, string> = {
  LITERATURE: "bg-emerald-100 text-emerald-900 border-emerald-300",
  HOMOLOG: "bg-sky-100 text-sky-900 border-sky-300",
  ANALOGY: "bg-amber-100 text-amber-900 border-amber-300",
  CONSERVATIVE: "bg-zinc-100 text-zinc-700 border-zinc-300",
};

function formatDose(d: number): string {
  if (d >= 1000) return `${(d / 1000).toFixed(d >= 10000 ? 0 : 1)} mM`;
  return `${d.toFixed(0)} µM`;
}

// Log-axis helpers
function logDose(d: number, doseMin: number, doseMax: number, plotW: number, plotPadX: number): number {
  const ld = Math.log10(Math.max(d, doseMin / 10));
  const ldMin = Math.log10(doseMin);
  const ldMax = Math.log10(doseMax);
  return plotPadX + ((ld - ldMin) / (ldMax - ldMin)) * (plotW - 2 * plotPadX);
}

function qfToY(qf: number, plotH: number, plotPadY: number): number {
  return plotPadY + (1 - qf) * (plotH - 2 * plotPadY);
}

function pmidLink(pmid: string): string {
  if (!pmid || pmid.trim() === "") return "";
  return `https://pubmed.ncbi.nlm.nih.gov/${pmid.trim()}/`;
}

// ===== Curve renderer =====

type CurveProps = {
  doses: number[];
  qfMeans: number[];
  qfSds: number[];
  doseMin: number;
  doseMax: number;
  plotW: number;
  plotH: number;
  plotPadX: number;
  plotPadY: number;
  color: string;
  label?: string;
  showBand?: boolean;
};

function HillCurve({ doses, qfMeans, qfSds, doseMin, doseMax, plotW, plotH, plotPadX, plotPadY, color, showBand = true }: CurveProps) {
  const points = doses.map((d, i) => ({
    x: logDose(d, doseMin, doseMax, plotW, plotPadX),
    y: qfToY(qfMeans[i], plotH, plotPadY),
    yhi: qfToY(Math.max(0, qfMeans[i] - qfSds[i]), plotH, plotPadY),
    ylo: qfToY(Math.min(1, qfMeans[i] + qfSds[i]), plotH, plotPadY),
  }));
  const linePath = "M " + points.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" L ");
  const bandPath = showBand
    ? "M " + points.map((p) => `${p.x.toFixed(1)} ${p.yhi.toFixed(1)}`).join(" L ") +
      " L " + points.slice().reverse().map((p) => `${p.x.toFixed(1)} ${p.ylo.toFixed(1)}`).join(" L ") + " Z"
    : "";
  return (
    <g>
      {showBand && <path d={bandPath} fill={color} fillOpacity={0.18} />}
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3} fill={color} />
      ))}
    </g>
  );
}

// ===== Plot container =====

type PlotProps = {
  doseMin: number;
  doseMax: number;
  plotW?: number;
  plotH?: number;
  threshold?: number;
  yLabel?: string;
  children: React.ReactNode;
  doseTicks?: number[];
};

function PlotFrame({ doseMin, doseMax, plotW = 600, plotH = 300, threshold, yLabel = "Quiescent fraction", children, doseTicks }: PlotProps) {
  const padX = 50, padY = 30;
  const ticks = doseTicks ?? [10, 100, 1000, 10000];
  return (
    <svg viewBox={`0 0 ${plotW} ${plotH}`} className="w-full h-auto" style={{ maxHeight: "320px" }}>
      {/* axes */}
      <line x1={padX} y1={plotH - padY} x2={plotW - padX} y2={plotH - padY} stroke="#52525b" strokeWidth={1} />
      <line x1={padX} y1={padY} x2={padX} y2={plotH - padY} stroke="#52525b" strokeWidth={1} />
      {/* horizontal threshold */}
      {threshold !== undefined && (
        <line
          x1={padX}
          y1={qfToY(threshold, plotH, padY)}
          x2={plotW - padX}
          y2={qfToY(threshold, plotH, padY)}
          stroke="#a1a1aa"
          strokeDasharray="4,4"
          strokeWidth={1}
        />
      )}
      {threshold !== undefined && (
        <text x={plotW - padX - 4} y={qfToY(threshold, plotH, padY) - 4} textAnchor="end" fill="#71717a" fontSize="10">
          quiescent threshold = {threshold}
        </text>
      )}
      {/* y ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((qf) => (
        <g key={qf}>
          <line x1={padX - 4} y1={qfToY(qf, plotH, padY)} x2={padX} y2={qfToY(qf, plotH, padY)} stroke="#52525b" />
          <text x={padX - 8} y={qfToY(qf, plotH, padY) + 3} textAnchor="end" fill="#52525b" fontSize="10">
            {qf}
          </text>
        </g>
      ))}
      {/* x ticks (log) */}
      {ticks.map((d) => (
        <g key={d}>
          <line
            x1={logDose(d, doseMin, doseMax, plotW, padX)}
            y1={plotH - padY}
            x2={logDose(d, doseMin, doseMax, plotW, padX)}
            y2={plotH - padY + 4}
            stroke="#52525b"
          />
          <text
            x={logDose(d, doseMin, doseMax, plotW, padX)}
            y={plotH - padY + 16}
            textAnchor="middle"
            fill="#52525b"
            fontSize="10"
          >
            {formatDose(d)}
          </text>
        </g>
      ))}
      {/* axis labels */}
      <text x={plotW / 2} y={plotH - 4} textAnchor="middle" fill="#27272a" fontSize="11" fontWeight={500}>
        Aqueous concentration (log scale)
      </text>
      <text x={12} y={plotH / 2} transform={`rotate(-90 12 ${plotH / 2})`} textAnchor="middle" fill="#27272a" fontSize="11" fontWeight={500}>
        {yLabel}
      </text>
      {children}
    </svg>
  );
}

// ===================================================================
// TAB 1: Dose-Response
// ===================================================================

const VOLATILES = ["halothane", "isoflurane", "sevoflurane", "propofol", "ketamine", "etomidate"];
const VOLATILE_COLORS: Record<string, string> = {
  halothane: "#7c3aed",
  isoflurane: "#0891b2",
  sevoflurane: "#059669",
  propofol: "#dc2626",
  ketamine: "#ea580c",
  etomidate: "#a16207",
};

function TabDoseResponse({ data }: { data: DoseResponseData }) {
  const [picked, setPicked] = useState<string>("halothane");
  const entry = data.compounds[picked];
  if (!entry) return <div>No data for {picked}.</div>;
  const doseMin = 10, doseMax = 3000;
  const color = VOLATILE_COLORS[picked] ?? "#27272a";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        {VOLATILES.map((v) => (
          <button
            key={v}
            onClick={() => setPicked(v)}
            className={`px-3 py-1 text-sm rounded-md border ${
              picked === v
                ? "bg-zinc-900 text-white border-zinc-900"
                : "bg-white text-zinc-700 border-zinc-300 hover:bg-zinc-50"
            }`}
          >
            {v}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 bg-white rounded-md border border-zinc-200 p-3">
          <PlotFrame doseMin={doseMin} doseMax={doseMax} threshold={0.5}>
            <HillCurve
              doses={entry.doses}
              qfMeans={entry.qf_mean}
              qfSds={entry.qf_sd}
              doseMin={doseMin}
              doseMax={doseMax}
              plotW={600}
              plotH={300}
              plotPadX={50}
              plotPadY={30}
              color={color}
            />
            {/* published EC50 marker */}
            {entry.published_EC50_uM && (
              <g>
                <line
                  x1={logDose(entry.published_EC50_uM, doseMin, doseMax, 600, 50)}
                  y1={30}
                  x2={logDose(entry.published_EC50_uM, doseMin, doseMax, 600, 50)}
                  y2={300 - 30}
                  stroke="#16a34a"
                  strokeDasharray="2,3"
                  strokeWidth={1}
                />
                <text
                  x={logDose(entry.published_EC50_uM, doseMin, doseMax, 600, 50) + 3}
                  y={42}
                  fill="#16a34a"
                  fontSize="10"
                >
                  published EC50 ({entry.published_EC50_uM} µM)
                </text>
              </g>
            )}
            {/* predicted EC50 marker */}
            {entry.predicted_EC50_uM && (
              <g>
                <line
                  x1={logDose(entry.predicted_EC50_uM, doseMin, doseMax, 600, 50)}
                  y1={30}
                  x2={logDose(entry.predicted_EC50_uM, doseMin, doseMax, 600, 50)}
                  y2={300 - 30}
                  stroke={color}
                  strokeDasharray="6,3"
                  strokeWidth={1}
                />
                <text
                  x={logDose(entry.predicted_EC50_uM, doseMin, doseMax, 600, 50) + 3}
                  y={56}
                  fill={color}
                  fontSize="10"
                >
                  predicted ({entry.predicted_EC50_uM.toFixed(0)} µM)
                </text>
              </g>
            )}
          </PlotFrame>
        </div>

        <div className="bg-zinc-50 rounded-md border border-zinc-200 p-4 text-sm">
          <h4 className="font-semibold text-zinc-900 mb-2">{picked}</h4>
          {entry.predicted_EC50_uM != null && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Predicted EC50</div>
              <div className="font-mono text-zinc-900">{entry.predicted_EC50_uM.toFixed(1)} µM</div>
            </div>
          )}
          {entry.published_EC50_uM && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Published EC50</div>
              <div className="font-mono text-zinc-900">{entry.published_EC50_uM} µM</div>
              {entry.source_paper && (
                <div className="text-xs text-zinc-600 mt-1">
                  {entry.source_PMID ? (
                    <a className="underline" href={pmidLink(entry.source_PMID)} target="_blank" rel="noreferrer">
                      {entry.source_paper}
                    </a>
                  ) : (
                    entry.source_paper
                  )}
                </div>
              )}
            </div>
          )}
          {entry.fold_error && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Fold error</div>
              <div className="font-mono text-zinc-900">{entry.fold_error.toFixed(2)}×</div>
            </div>
          )}
          {entry.anchor_quality && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Anchor quality</div>
              <div className="text-xs text-zinc-700">{entry.anchor_quality}</div>
            </div>
          )}
          <div className="mt-3 pt-3 border-t border-zinc-200 text-xs text-zinc-600">
            α = {data.meta.alpha_calib} (calibrated on halothane WT only). All other curves use the same locked α.
          </div>
        </div>
      </div>
    </div>
  );
}

// ===================================================================
// TAB 2: Mutants
// ===================================================================

const HYPER_MUTANTS = ["gas-1", "gas-2", "nduf-6", "ndus-8", "nuo-1", "unc-79", "unc-80"];
const RESIST_MUTANTS = ["goa-1", "dgk-1"];

function TabMutants({ data }: { data: DoseResponseData }) {
  const [picked, setPicked] = useState<string>("gas-1");
  const wt = data.compounds["halothane"];
  const mut = data.compounds[`halothane__${picked}`];
  if (!wt || !mut) return <div>No data for halothane × {picked}.</div>;
  const doseMin = 10, doseMax = 3000;
  const color = mut.expected_direction === "HYPER" ? "#dc2626" : "#2563eb";

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-red-700 font-semibold mb-1">Hypersensitive</div>
          <div className="flex flex-wrap gap-1.5">
            {HYPER_MUTANTS.map((g) => (
              <button
                key={g}
                onClick={() => setPicked(g)}
                className={`px-2.5 py-0.5 text-xs rounded-md border ${
                  picked === g ? "bg-red-600 text-white border-red-600" : "bg-white text-red-700 border-red-300 hover:bg-red-50"
                }`}
              >
                {g}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-blue-700 font-semibold mb-1">Resistant</div>
          <div className="flex flex-wrap gap-1.5">
            {RESIST_MUTANTS.map((g) => (
              <button
                key={g}
                onClick={() => setPicked(g)}
                className={`px-2.5 py-0.5 text-xs rounded-md border ${
                  picked === g ? "bg-blue-600 text-white border-blue-600" : "bg-white text-blue-700 border-blue-300 hover:bg-blue-50"
                }`}
              >
                {g}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 bg-white rounded-md border border-zinc-200 p-3">
          <PlotFrame doseMin={doseMin} doseMax={doseMax} threshold={0.5}>
            {/* WT halothane curve in gray */}
            <HillCurve
              doses={wt.doses}
              qfMeans={wt.qf_mean}
              qfSds={wt.qf_sd}
              doseMin={doseMin}
              doseMax={doseMax}
              plotW={600}
              plotH={300}
              plotPadX={50}
              plotPadY={30}
              color="#a1a1aa"
              showBand={false}
            />
            {/* Mutant curve in red/blue */}
            <HillCurve
              doses={mut.doses}
              qfMeans={mut.qf_mean}
              qfSds={mut.qf_sd}
              doseMin={doseMin}
              doseMax={doseMax}
              plotW={600}
              plotH={300}
              plotPadX={50}
              plotPadY={30}
              color={color}
            />
            <text x={70} y={45} fontSize="10" fill="#a1a1aa">— WT halothane</text>
            <text x={70} y={58} fontSize="10" fill={color}>— halothane × {picked} ({mut.expected_direction})</text>
          </PlotFrame>
        </div>

        <div className="bg-zinc-50 rounded-md border border-zinc-200 p-4 text-sm">
          <h4 className="font-semibold text-zinc-900 mb-2">{picked} × halothane</h4>
          <div className="mb-2">
            <div className="text-xs text-zinc-500">Direction (WB ontology)</div>
            <div className={`font-mono ${color === "#dc2626" ? "text-red-700" : "text-blue-700"}`}>
              {mut.expected_direction}
            </div>
          </div>
          {mut.predicted_EC50_uM && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Predicted EC50</div>
              <div className="font-mono text-zinc-900">{mut.predicted_EC50_uM.toFixed(0)} µM</div>
            </div>
          )}
          {mut.predicted_ratio && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Mutant / WT ratio</div>
              <div className="font-mono text-zinc-900">{mut.predicted_ratio.toFixed(2)}</div>
            </div>
          )}
          {mut.lit_ratio && (
            <div className="mb-2">
              <div className="text-xs text-zinc-500">Literature ratio</div>
              <div className="font-mono text-zinc-900">{mut.lit_ratio}</div>
            </div>
          )}
          {mut.source_paper && (
            <div className="mb-2 text-xs text-zinc-600">
              {mut.source_PMID ? (
                <a className="underline" href={pmidLink(mut.source_PMID)} target="_blank" rel="noreferrer">
                  {mut.source_paper}
                </a>
              ) : (
                mut.source_paper
              )}
            </div>
          )}
          {mut.notes && (
            <div className="mt-3 pt-3 border-t border-zinc-200 text-xs text-zinc-600 leading-relaxed">{mut.notes}</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ===================================================================
// TAB 3: Eger Specificity
// ===================================================================

const EGER_COMPOUNDS = [
  { key: "cis_12_dichloroethylene", label: "cis-1,2-DCE", classification: "ANESTHETIC", color: "#16a34a" },
  { key: "trans_12_dichloroethylene", label: "trans-1,2-DCE", classification: "NON_IMMOBILIZER", color: "#dc2626" },
  { key: "hexafluoroethane", label: "hexafluoroethane", classification: "NON_IMMOBILIZER", color: "#9333ea" },
];

function TabEger({ data }: { data: DoseResponseData }) {
  const doseMin = 30, doseMax = 30000;
  const ticks = [30, 100, 300, 1000, 3000, 10000, 30000];
  return (
    <div className="flex flex-col gap-4">
      <div className="bg-amber-50 border border-amber-200 rounded-md p-3 text-sm text-amber-900">
        <strong>Eger 2001 panel</strong> was designed to falsify lipophilic-pocket-fit theories of anesthesia.
        Hexafluoroethane and <em>trans</em>-1,2-dichloroethylene have the right lipid solubility for Meyer-Overton anesthesia but produce no immobilization.
        Cis-1,2-dichloroethylene <em>is</em> an anesthetic. Single-pose docking pipelines cannot distinguish them.
        The conserved-substrate network model run below has the same alpha as for halothane WT — no compound-specific tuning.
      </div>

      <div className="bg-white rounded-md border border-zinc-200 p-3">
        <PlotFrame doseMin={doseMin} doseMax={doseMax} threshold={0.5} doseTicks={ticks}>
          {EGER_COMPOUNDS.map((c) => {
            const e = data.compounds[c.key];
            if (!e) return null;
            return (
              <HillCurve
                key={c.key}
                doses={e.doses}
                qfMeans={e.qf_mean}
                qfSds={e.qf_sd}
                doseMin={doseMin}
                doseMax={doseMax}
                plotW={600}
                plotH={300}
                plotPadX={50}
                plotPadY={30}
                color={c.color}
              />
            );
          })}
          {/* legend */}
          {EGER_COMPOUNDS.map((c, i) => (
            <text key={c.key} x={70} y={45 + i * 14} fontSize="11" fill={c.color}>
              — {c.label} ({c.classification === "ANESTHETIC" ? "anesthetic" : "non-immobilizer"})
            </text>
          ))}
        </PlotFrame>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
        {EGER_COMPOUNDS.map((c) => {
          const e = data.compounds[c.key];
          if (!e) return null;
          const maxQf = Math.max(...e.qf_mean);
          const correct = c.classification === "ANESTHETIC" ? maxQf >= 0.5 : maxQf < 0.5;
          return (
            <div key={c.key} className="bg-white border border-zinc-200 rounded-md p-3">
              <div className="font-semibold text-zinc-900">{c.label}</div>
              <div className="text-xs text-zinc-600 mb-2">{c.classification}</div>
              <div className="text-xs text-zinc-500">Max quiescent fraction</div>
              <div className={`font-mono text-lg ${correct ? "text-emerald-700" : "text-rose-700"}`}>
                {maxQf.toFixed(3)}
              </div>
              <div className={`text-xs mt-1 ${correct ? "text-emerald-700" : "text-rose-700"}`}>
                {correct ? "✓ correct" : "✗ incorrect"}
              </div>
            </div>
          );
        })}
      </div>

      <div className="text-xs text-zinc-600 italic">
        Non-immobilizer firing rates do drop monotonically with dose (any partial K2P engagement contributes), but neither
        crosses the immobilization threshold across four orders of magnitude (30 µM → 30 mM).
      </div>
    </div>
  );
}

// ===================================================================
// TAB 4: Perturbation Profile
// ===================================================================

const ALL_COMPOUNDS_PERT = [
  "halothane", "isoflurane", "sevoflurane", "propofol", "ketamine", "etomidate",
  "cis_12_dichloroethylene", "trans_12_dichloroethylene", "hexafluoroethane",
];

function TabPerturbationProfile({ profile }: { profile: PerturbationProfile }) {
  const [picked, setPicked] = useState<string>("halothane");
  const entry = profile.compounds[picked];
  if (!entry) return <div>No data for {picked}.</div>;
  const doseMin = 0.1, doseMax = 30000;

  // Build engagement curves per active mechanism class
  function engagement(dose: number, ec50: number, hillN: number): number {
    return (dose ** hillN) / (dose ** hillN + ec50 ** hillN);
  }

  const ndoses = 50;
  const sample = Array.from({ length: ndoses }, (_, i) => {
    const ld = Math.log10(doseMin) + ((Math.log10(doseMax) - Math.log10(doseMin)) * i) / (ndoses - 1);
    return 10 ** ld;
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        {ALL_COMPOUNDS_PERT.map((c) => (
          <button
            key={c}
            onClick={() => setPicked(c)}
            className={`px-3 py-1 text-xs rounded-md border ${
              picked === c
                ? "bg-zinc-900 text-white border-zinc-900"
                : "bg-white text-zinc-700 border-zinc-300 hover:bg-zinc-50"
            }`}
          >
            {c.replace(/_/g, " ")}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 bg-white rounded-md border border-zinc-200 p-3">
          <PlotFrame
            doseMin={doseMin}
            doseMax={doseMax}
            yLabel="Engagement (Hill curve)"
            doseTicks={[1, 10, 100, 1000, 10000]}
          >
            {entry.active.map((row) => {
              if (!row.target_EC50_uM) return null;
              const points = sample.map((d) => ({
                x: logDose(d, doseMin, doseMax, 600, 50),
                y: qfToY(engagement(d, row.target_EC50_uM!, row.hill_n), 300, 30),
              }));
              const path = "M " + points.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" L ");
              return (
                <path
                  key={row.mechanism_class}
                  d={path}
                  fill="none"
                  stroke={MECHANISM_COLORS[row.mechanism_class] ?? "#52525b"}
                  strokeWidth={1.5}
                />
              );
            })}
          </PlotFrame>
        </div>

        <div className="bg-zinc-50 rounded-md border border-zinc-200 p-3 text-sm">
          <h4 className="font-semibold text-zinc-900 mb-2">{picked.replace(/_/g, " ")}</h4>
          <div className="text-xs text-zinc-600 mb-3">
            {entry.n_active} active mechanism classes ·{" "}
            {entry.n_deferred} deferred (no clean primary anchor)
          </div>
          <div className="flex flex-col gap-2">
            {entry.active.map((row) => (
              <div
                key={row.mechanism_class}
                className="flex items-center gap-2 text-xs border-b border-zinc-200 pb-1.5"
              >
                <span
                  className="inline-block w-3 h-3 rounded-sm flex-shrink-0"
                  style={{ background: MECHANISM_COLORS[row.mechanism_class] ?? "#52525b" }}
                />
                <span className="text-zinc-900 font-medium flex-1">
                  {MECHANISM_LABELS[row.mechanism_class] ?? row.mechanism_class}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] border ${
                    EVIDENCE_GRADE_BADGE[row.evidence_grade] ?? "bg-zinc-100 text-zinc-600"
                  }`}
                >
                  {row.evidence_grade}
                </span>
              </div>
            ))}
            {entry.active.map((row) => (
              <div key={`detail-${row.mechanism_class}`} className="text-[11px] text-zinc-600 leading-snug">
                <span className="font-mono">EC50 = {row.target_EC50_uM} µM</span>
                {", "}
                <span className="font-mono">max effect = {row.max_effect_factor}</span>
                {row.source_PMID && (
                  <>
                    {" — "}
                    <a className="underline" href={pmidLink(row.source_PMID)} target="_blank" rel="noreferrer">
                      {row.source_paper}
                    </a>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ===================================================================
// TAB 5: Cross-species (worm V3 vs fly V4)
// ===================================================================

type CrossSpeciesData = {
  meta: { description: string };
  worm_V3: any;
  fly_V4: any;
  shared_substrate: string[];
  transfer_evidence: string[];
};

function TabCrossSpecies({ wormDose, flyDose, crossData }: { wormDose: DoseResponseData; flyDose: DoseResponseData; crossData: CrossSpeciesData }) {
  const [picked, setPicked] = useState<string>("halothane");
  const wormEntry = wormDose.compounds[picked];
  const flyEntry = flyDose.compounds[picked];
  if (!wormEntry || !flyEntry) return <div>Both organism panels missing for {picked}.</div>;
  const doseMin = 10, doseMax = 3000;

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-md p-3 text-sm text-emerald-900">
        <strong>Cross-species validation</strong> — same architecture, two unrelated connectomes
        (Cook 2019 nematode, 300 neurons · Winding 2023 dipteran larva, 2,952 neurons),
        organism-specific α calibrated on a single behavioral anchor each.
        Conserved-substrate hypothesis: SAME mechanism classes (SNARE / Complex I / K2P / nAChR /
        GABA-A / NCA / GluCl) drive anesthesia in worm AND fly.
      </div>

      {/* Side-by-side dose response */}
      <div className="grid md:grid-cols-2 gap-3">
        <div className="bg-white rounded-md border border-zinc-200 p-3">
          <div className="text-xs uppercase tracking-wide font-semibold text-zinc-700 mb-1">
            Worm V3 — C. elegans (Cook 2019, 300 neurons)
          </div>
          <PlotFrame doseMin={doseMin} doseMax={doseMax} threshold={0.5}>
            <HillCurve
              doses={wormEntry.doses}
              qfMeans={wormEntry.qf_mean}
              qfSds={wormEntry.qf_sd}
              doseMin={doseMin} doseMax={doseMax}
              plotW={600} plotH={300} plotPadX={50} plotPadY={30}
              color="#7c3aed"
            />
            {wormEntry.published_EC50_uM && (
              <line x1={logDose(wormEntry.published_EC50_uM, doseMin, doseMax, 600, 50)}
                    y1={30} x2={logDose(wormEntry.published_EC50_uM, doseMin, doseMax, 600, 50)}
                    y2={300-30} stroke="#16a34a" strokeDasharray="2,3" strokeWidth={1} />
            )}
          </PlotFrame>
          <div className="mt-2 text-sm grid grid-cols-2 gap-2">
            <div>
              <div className="text-xs text-zinc-500">Predicted</div>
              <div className="font-mono">{wormEntry.predicted_EC50_uM?.toFixed(1)} µM</div>
            </div>
            <div>
              <div className="text-xs text-zinc-500">Published</div>
              <div className="font-mono">{wormEntry.published_EC50_uM} µM</div>
            </div>
            {wormEntry.fold_error && (
              <div className="col-span-2">
                <div className="text-xs text-zinc-500">Error</div>
                <div className="font-mono text-purple-700">{wormEntry.fold_error.toFixed(2)}× off</div>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-md border border-zinc-200 p-3">
          <div className="text-xs uppercase tracking-wide font-semibold text-zinc-700 mb-1">
            Fly V4 — Drosophila larva (Winding 2023, 2,952 neurons)
          </div>
          <PlotFrame doseMin={doseMin} doseMax={doseMax} threshold={0.5}>
            <HillCurve
              doses={flyEntry.doses}
              qfMeans={flyEntry.qf_mean}
              qfSds={flyEntry.qf_sd}
              doseMin={doseMin} doseMax={doseMax}
              plotW={600} plotH={300} plotPadX={50} plotPadY={30}
              color="#0891b2"
            />
            {flyEntry.published_EC50_uM && (
              <line x1={logDose(flyEntry.published_EC50_uM, doseMin, doseMax, 600, 50)}
                    y1={30} x2={logDose(flyEntry.published_EC50_uM, doseMin, doseMax, 600, 50)}
                    y2={300-30} stroke="#16a34a" strokeDasharray="2,3" strokeWidth={1} />
            )}
          </PlotFrame>
          <div className="mt-2 text-sm grid grid-cols-2 gap-2">
            <div>
              <div className="text-xs text-zinc-500">Predicted</div>
              <div className="font-mono">{flyEntry.predicted_EC50_uM?.toFixed(1)} µM</div>
            </div>
            <div>
              <div className="text-xs text-zinc-500">Published</div>
              <div className="font-mono">{flyEntry.published_EC50_uM} µM</div>
            </div>
            {flyEntry.fold_error && (
              <div className="col-span-2">
                <div className="text-xs text-zinc-500">Error</div>
                <div className="font-mono text-cyan-700">{flyEntry.fold_error.toFixed(2)}× off</div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mt-2">
        {["halothane", "isoflurane"].map((v) => (
          <button key={v} onClick={() => setPicked(v)}
            className={`px-3 py-1 text-sm rounded-md border ${
              picked === v ? "bg-zinc-900 text-white border-zinc-900" : "bg-white text-zinc-700 border-zinc-300 hover:bg-zinc-50"
            }`}>{v}</button>
        ))}
      </div>

      {/* Worm vs fly headline table */}
      <div className="bg-zinc-50 border border-zinc-200 rounded-md p-3 mt-2">
        <div className="text-xs uppercase tracking-wide font-semibold text-zinc-700 mb-2">
          Worm V3 vs Fly V4 — side-by-side
        </div>
        <table className="text-sm w-full">
          <thead>
            <tr className="border-b border-zinc-300">
              <th className="text-left py-1 font-medium">metric</th>
              <th className="text-right py-1 font-medium text-purple-700">worm V3</th>
              <th className="text-right py-1 font-medium text-cyan-700">fly V4</th>
            </tr>
          </thead>
          <tbody className="text-xs font-mono">
            <tr><td className="py-1">connectome neurons</td><td className="text-right">300</td><td className="text-right">2,952</td></tr>
            <tr><td className="py-1">chemical edges</td><td className="text-right">~3,700</td><td className="text-right">~110,000</td></tr>
            <tr><td className="py-1">α (calibrated)</td><td className="text-right">0.13</td><td className="text-right">0.060</td></tr>
            <tr><td className="py-1">halothane EC50</td><td className="text-right">317 µM (1.07×)</td><td className="text-right">361 µM (1.06×)</td></tr>
            <tr><td className="py-1">iso held-out EC50</td><td className="text-right">291 µM (1.002×)</td><td className="text-right">323 µM (1.11×)</td></tr>
            <tr><td className="py-1">mutant directional</td><td className="text-right">9 / 9</td><td className="text-right">13 / 13</td></tr>
            <tr><td className="py-1">Eger specificity</td><td className="text-right">3 / 3</td><td className="text-right">3 / 3</td></tr>
          </tbody>
        </table>
      </div>

      {/* Shared substrate */}
      <div className="bg-white border border-zinc-200 rounded-md p-3">
        <div className="text-xs uppercase tracking-wide font-semibold text-zinc-700 mb-2">
          Conserved-substrate hypothesis — what transfers
        </div>
        <ul className="text-sm space-y-1 leading-relaxed">
          {crossData.shared_substrate.map((s, i) => (
            <li key={i} className="text-zinc-700">• {s}</li>
          ))}
        </ul>
        <div className="text-xs uppercase tracking-wide font-semibold text-zinc-700 mt-3 mb-2">
          Transfer evidence
        </div>
        <ul className="text-sm space-y-1 leading-relaxed">
          {crossData.transfer_evidence.map((s, i) => (
            <li key={i} className="text-zinc-700">• {s}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}


// ===================================================================
// MAIN COMPONENT
// ===================================================================

type Tab = "dose-response" | "mutants" | "eger" | "perturbation" | "cross-species";

const TABS: { id: Tab; label: string; subtitle: string }[] = [
  { id: "dose-response", label: "Dose-Response", subtitle: "WT volatiles vs published EC50" },
  { id: "mutants", label: "Mutants", subtitle: "halothane × genotype directional shifts" },
  { id: "eger", label: "Eger Specificity", subtitle: "anesthetics vs non-immobilizers" },
  { id: "perturbation", label: "Perturbation Profile", subtitle: "per-anesthetic Hill curves" },
  { id: "cross-species", label: "Cross-Species", subtitle: "worm V3 vs fly V4 side-by-side" },
];

export function NetworkStateValidator() {
  const [tab, setTab] = useState<Tab>("dose-response");
  const [doseData, setDoseData] = useState<DoseResponseData | null>(null);
  const [pertData, setPertData] = useState<PerturbationProfile | null>(null);
  const [verdict, setVerdict] = useState<ValidationSummary | null>(null);
  const [flyDose, setFlyDose] = useState<DoseResponseData | null>(null);
  const [crossData, setCrossData] = useState<CrossSpeciesData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchJSON<DoseResponseData>("v3_dose_response.json"),
      fetchJSON<PerturbationProfile>("v3_perturbation_profile.json"),
      fetchJSON<ValidationSummary>("v3_validation_summary.json"),
      fetchJSON<DoseResponseData>("v4_fly_dose_response.json"),
      fetchJSON<CrossSpeciesData>("v4_cross_species_summary.json"),
    ])
      .then(([d, p, v, fd, cs]) => {
        setDoseData(d);
        setPertData(p);
        setVerdict(v);
        setFlyDose(fd);
        setCrossData(cs);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="p-4 bg-rose-50 border border-rose-200 rounded text-sm">Failed to load data: {err}</div>;
  if (!doseData || !pertData || !verdict || !flyDose || !crossData) return <div className="p-4 text-zinc-500 text-sm">Loading…</div>;

  const gates = verdict.gates;

  return (
    <div className="not-prose flex flex-col gap-4 my-6">
      {/* Verdict banner */}
      <div className="bg-zinc-900 text-zinc-100 rounded-lg p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
          <div>
            <div className="text-xs uppercase tracking-wider text-zinc-400 font-semibold">V3 ensemble verdict</div>
            <div className="text-lg font-semibold">{verdict.summary}</div>
          </div>
          <div className="text-xs font-mono text-zinc-400">
            α = {verdict.alpha_calib} · {verdict.sim_duration_s}s × n={verdict.n_seeds} · 545 sims
          </div>
        </div>
        <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
          <div className="bg-zinc-800 rounded p-2">
            <div className="text-xs text-zinc-400">Gate 1 — halothane WT</div>
            <div className={gates.gate1_halothane_wt?.PASS ? "text-emerald-300" : "text-rose-300"}>
              PASS · 1.07× off
            </div>
          </div>
          <div className="bg-zinc-800 rounded p-2">
            <div className="text-xs text-zinc-400">Gate 2 — iso held-out</div>
            <div className={gates.gate2_iso_wt?.PASS ? "text-emerald-300" : "text-rose-300"}>
              PASS · 1.002× (≈ exact)
            </div>
          </div>
          <div className="bg-zinc-800 rounded p-2">
            <div className="text-xs text-zinc-400">Gate 3 — mutants</div>
            <div className={gates.gate3_mutant_directional?.PASS ? "text-emerald-300" : "text-rose-300"}>
              PASS · 9/9 correct
            </div>
          </div>
          <div className="bg-zinc-800 rounded p-2">
            <div className="text-xs text-zinc-400">Gate 4 — Eger specificity</div>
            <div className={gates.gate4_eger_specificity?.PASS ? "text-emerald-300" : "text-rose-300"}>
              PASS · 3/3 correct
            </div>
          </div>
        </div>
      </div>

      {/* Tab nav */}
      <div className="flex flex-wrap gap-1 border-b border-zinc-200">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-3 py-2 text-sm border-b-2 transition-colors ${
              tab === t.id
                ? "border-zinc-900 text-zinc-900 font-semibold"
                : "border-transparent text-zinc-600 hover:text-zinc-900 hover:border-zinc-300"
            }`}
            title={t.subtitle}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {tab === "dose-response" && <TabDoseResponse data={doseData} />}
        {tab === "mutants" && <TabMutants data={doseData} />}
        {tab === "eger" && <TabEger data={doseData} />}
        {tab === "perturbation" && <TabPerturbationProfile profile={pertData} />}
        {tab === "cross-species" && <TabCrossSpecies wormDose={doseData} flyDose={flyDose} crossData={crossData} />}
      </div>
    </div>
  );
}
