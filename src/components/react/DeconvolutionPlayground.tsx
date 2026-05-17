import * as React from "react";
import { useCallback, useEffect, useMemo, useState } from "react";

type Meta = {
  model_name: string;
  atlas_cell_types: string[];
  output_layout: { cancer_logit: number; tissue_logits: number[]; tumor_fraction_regression: number };
  tissue_labels: string[];
  note: string;
};

type Lookup = {
  cell_types: string[];
  cancer_types: string[];
  tumor_fractions: number[];
  healthy_baseline: number[];
  cancer_to_tissue: Record<string, string>;
  lookup: Record<string, Record<string, number[]>>;
};

const MODEL_URL = "/models/cfdna-cancer-detection/multitask_mlp.onnx";
const META_URL = "/models/cfdna-cancer-detection/meta.json";
const LOOKUP_URL = "/models/cfdna-cancer-detection/deconv_lookup.json";

type ORT = typeof import("onnxruntime-web");

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}
function softmax(x: number[]): number[] {
  const m = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}

async function loadOrt(): Promise<ORT> {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
  return ort as unknown as ORT;
}

// Interpolate the deconv vector for an arbitrary tumor fraction between known grid points
function interpolateDeconv(
  lookup: Lookup,
  cancer: string,
  tf: number
): number[] {
  const fractions = lookup.tumor_fractions;
  const cancerKey = cancer in lookup.lookup ? cancer : "healthy";
  // Find bounding fractions
  let lo = fractions[0];
  let hi = fractions[fractions.length - 1];
  for (let i = 0; i < fractions.length - 1; i++) {
    if (tf >= fractions[i] && tf <= fractions[i + 1]) {
      lo = fractions[i];
      hi = fractions[i + 1];
      break;
    }
  }
  if (tf <= fractions[0]) return [...lookup.lookup[cancerKey][String(fractions[0])]];
  if (tf >= fractions[fractions.length - 1])
    return [...lookup.lookup[cancerKey][String(fractions[fractions.length - 1])]];

  const vLo = lookup.lookup[cancerKey][String(lo)];
  const vHi = lookup.lookup[cancerKey][String(hi)];
  const alpha = (tf - lo) / (hi - lo);
  return vLo.map((v, i) => v + alpha * (vHi[i] - v));
}

// Color palette for major cell-type categories
const cellColor = (ct: string): string => {
  if (ct.startsWith("Blood-T") || ct.startsWith("Blood-B") || ct.startsWith("Blood-NK")) return "#5b8def";
  if (ct.startsWith("Blood-Granul") || ct.startsWith("Blood-Mono")) return "#3568c8";
  if (ct.startsWith("Eryth") || ct.startsWith("Megak")) return "#c0392b";
  if (ct === "Colon-Ep" || ct === "Colon-Fibro" || ct === "Small-Int-Ep") return "#2f8d46";
  if (ct === "Liver-Hep" || ct.startsWith("Liver")) return "#a05a00";
  if (ct.startsWith("Lung")) return "#d1962a";
  if (ct.startsWith("Breast")) return "#d6604d";
  if (ct.startsWith("Pancreas")) return "#7a4197";
  if (ct === "Endothel" || ct.includes("Fibro") || ct === "Adipocytes" || ct.includes("Musc") || ct === "Heart-Cardio") return "#888";
  return "#bdbdbd";
};

const CANCER_OPTIONS = [
  { key: "healthy", label: "Healthy (no cancer)", description: "Real healthy cfDNA baseline (mean of 23 deep WGBS samples, GSE186458)" },
  { key: "COAD", label: "Colorectal (COAD)", description: "Colon-Ep is the tissue-of-origin marker" },
  { key: "LIHC", label: "Liver (LIHC)", description: "Liver-Hep is the tissue-of-origin marker" },
  { key: "LUAD", label: "Lung adenocarcinoma (LUAD)", description: "Lung-Ep-Alveo is the tissue-of-origin marker — note: cross-cohort failure case" },
  { key: "BRCA", label: "Breast (BRCA)", description: "Breast-Luminal-Ep is the tissue-of-origin marker" },
];

export function DeconvolutionPlayground() {
  const [ort, setOrt] = useState<ORT | null>(null);
  const [session, setSession] = useState<any>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [lookup, setLookup] = useState<Lookup | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const [cancer, setCancer] = useState<string>("healthy");
  const [tumorFrac, setTumorFrac] = useState<number>(0);

  const [cancerProb, setCancerProb] = useState<number | null>(null);
  const [tissueProbs, setTissueProbs] = useState<number[] | null>(null);
  const [tfPred, setTfPred] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const ortMod = await loadOrt();
        if (!alive) return;
        setOrt(ortMod);
        const [metaR, lookupR] = await Promise.all([fetch(META_URL), fetch(LOOKUP_URL)]);
        const m = (await metaR.json()) as Meta;
        const l = (await lookupR.json()) as Lookup;
        if (!alive) return;
        setMeta(m);
        setLookup(l);
        const s = await ortMod.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        if (!alive) return;
        setSession(s);
        setStatus("ready");
      } catch (e: any) {
        console.error("ORT load failed", e);
        if (!alive) return;
        setLoadErr(String(e?.message || e));
        setStatus("error");
      }
    })();
    return () => { alive = false; };
  }, []);

  // Current deconv vector (interpolated from grid)
  const currentVec = useMemo(() => {
    if (!lookup) return null;
    return interpolateDeconv(lookup, cancer === "healthy" ? "healthy" : cancer, tumorFrac);
  }, [lookup, cancer, tumorFrac]);

  // Run inference whenever the vector changes
  useEffect(() => {
    if (!ort || !session || !currentVec) return;
    let alive = true;
    (async () => {
      setBusy(true);
      try {
        const f32 = new Float32Array(currentVec);
        const tensor = new ort.Tensor("float32", f32, [1, 40]);
        const result = await session.run({ deconv_fractions: tensor });
        if (!alive) return;
        const out = Array.from(result.multitask_output.data as Float32Array);
        const cLogit = out[0];
        const tLogits = out.slice(1, 6);
        const tf = out[6];
        setCancerProb(sigmoid(cLogit));
        setTissueProbs(softmax(tLogits));
        setTfPred(Math.max(0, Math.min(0.5, tf)));
      } catch (e) {
        console.error(e);
      } finally {
        if (alive) setBusy(false);
      }
    })();
    return () => { alive = false; };
  }, [ort, session, currentVec]);

  if (status === "loading") {
    return (
      <div className="my-6 rounded-lg border p-4 text-sm text-muted-foreground">
        Loading deconvolution model…
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="my-6 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm">
        Model failed to load: {loadErr}
      </div>
    );
  }
  if (!lookup || !meta || !currentVec) return null;

  // Top cell types by fraction
  const topCells = currentVec
    .map((v, i) => ({ cell: lookup.cell_types[i], v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 8);
  const targetTissue =
    cancer !== "healthy" ? lookup.cancer_to_tissue[cancer] : null;

  return (
    <div className="my-6 flex flex-col gap-5">
      {/* Controls */}
      <div className="flex flex-col gap-4 rounded-lg border p-4 bg-card">
        <div>
          <label className="block text-sm font-medium mb-2">
            Sample type
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-1.5">
            {CANCER_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                onClick={() => setCancer(opt.key)}
                className={`rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  cancer === opt.key
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-muted hover:border-primary/40"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {CANCER_OPTIONS.find((o) => o.key === cancer)?.description}
          </p>
        </div>

        <div>
          <label className="flex items-center justify-between text-sm font-medium mb-1">
            <span>Tumor fraction in cfDNA</span>
            <span className="font-mono text-primary">
              {(tumorFrac * 100).toFixed(1)}%
            </span>
          </label>
          <input
            type="range"
            min={0}
            max={20}
            step={0.5}
            value={tumorFrac * 100}
            onChange={(e) => setTumorFrac(Number(e.target.value) / 100)}
            className="w-full"
            disabled={cancer === "healthy"}
          />
          <div className="mt-1 flex justify-between text-xs text-muted-foreground">
            <span>0%</span>
            <span>0.5% (stage 1)</span>
            <span>5% (stage 3)</span>
            <span>20%</span>
          </div>
          {cancer === "healthy" && (
            <p className="mt-1 text-xs text-muted-foreground">
              Tumor fraction is fixed at 0% for healthy samples.
            </p>
          )}
        </div>
      </div>

      {/* Outputs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Cell-type composition */}
        <div className="rounded-lg border p-4 bg-card">
          <h4 className="text-sm font-semibold mb-3">
            Cell-type deconvolution (top 8 of 40)
          </h4>
          <div className="space-y-1.5">
            {topCells.map(({ cell, v }) => {
              const isTarget = cell === targetTissue;
              return (
                <div key={cell}>
                  <div className="flex justify-between text-xs">
                    <span
                      className={`font-mono ${
                        isTarget ? "font-bold text-primary" : ""
                      }`}
                    >
                      {cell}
                      {isTarget && " ←"}
                    </span>
                    <span className="font-mono">{(v * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2 rounded bg-muted overflow-hidden">
                    <div
                      className="h-full transition-[width] duration-200"
                      style={{
                        width: `${Math.min(100, v * 100 * 3)}%`,
                        backgroundColor: cellColor(cell),
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <details className="mt-3 text-xs text-muted-foreground">
            <summary className="cursor-pointer">How is this computed?</summary>
            <p className="mt-1">
              NNLS deconvolution over the Loyfer atlas (9520 atlas blocks × 40 cell types).
              For demo speed, this UI interpolates between precomputed grid points
              (tumor fraction × cancer type) rather than re-running NNLS in the browser.
            </p>
          </details>
        </div>

        {/* Model outputs */}
        <div className="rounded-lg border p-4 bg-card">
          <h4 className="text-sm font-semibold mb-3">Model predictions</h4>

          {/* Cancer probability gauge */}
          {cancerProb !== null && (
            <div className="mb-4">
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium">Cancer probability</span>
                <span className="font-mono font-bold">
                  {(cancerProb * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-3 rounded bg-muted overflow-hidden relative">
                <div
                  className={`h-full transition-[width] duration-300`}
                  style={{
                    width: `${cancerProb * 100}%`,
                    backgroundColor:
                      cancerProb > 0.7
                        ? "#c0392b"
                        : cancerProb > 0.4
                        ? "#d6604d"
                        : "#2f8d46",
                  }}
                />
                <div
                  className="absolute top-0 h-full w-px bg-foreground/30"
                  style={{ left: "50%" }}
                />
              </div>
            </div>
          )}

          {/* Tissue of origin */}
          {tissueProbs && (
            <div className="mb-4">
              <div className="text-xs font-medium mb-2">
                Tissue of origin (5-way softmax)
              </div>
              <div className="space-y-1.5">
                {meta.tissue_labels.map((label, i) => (
                  <div key={label}>
                    <div className="flex justify-between text-xs">
                      <span className="font-mono">{label}</span>
                      <span className="font-mono">
                        {(tissueProbs[i] * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded bg-muted overflow-hidden">
                      <div
                        className="h-full bg-primary transition-[width] duration-200"
                        style={{ width: `${tissueProbs[i] * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tumor fraction regression */}
          {tfPred !== null && (
            <div className="text-xs">
              <span className="font-medium">Tumor fraction (predicted):</span>{" "}
              <span className="font-mono">{(tfPred * 100).toFixed(2)}%</span>
              <span className="text-muted-foreground">
                {" "}
                vs input {(tumorFrac * 100).toFixed(2)}%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Honesty notice */}
      <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-xs leading-relaxed">
        <strong>What this demo is.</strong> This is the Day-1 multitask MLP (~ 200 KB)
        running real inference in your browser via ONNX Runtime. It learns from the in-silico
        mixture training distribution, so it tracks tumor-tissue elevation well but is the
        Day-1 model that scored cross-cohort AUROC 0.217 on GSE122126. The Day-3 model
        (cross-cohort AUROC 0.85) is 158 MB and too large for the browser. See the project
        page below for the real numbers and the four mechanisms that closed the cross-cohort
        gap.
      </div>

      {busy && (
        <div className="text-xs text-muted-foreground">running inference…</div>
      )}
    </div>
  );
}

export default DeconvolutionPlayground;
