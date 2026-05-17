import * as React from "react";
import { useEffect, useMemo, useState } from "react";

type Meta = {
  model_name: string;
  feature_names: string[];
  drug_type_labels: string[];
  sponsor_class_labels: string[];
  demo_eval_auroc: number;
  full_model_audited_auroc: number;
  headline_real_audited_auroc: number;
  note: string;
};

type ExampleTrial = {
  nct_id: string;
  features: Record<string, number>;
  actual_success: boolean;
  predicted_proba: number;
};

const MODEL_URL = "/models/clinical-trial-failure-prediction/demo_trial_lgbm.onnx";
const META_URL = "/models/clinical-trial-failure-prediction/meta.json";
const EXAMPLES_URL = "/models/clinical-trial-failure-prediction/example_trials.json";

type ORT = typeof import("onnxruntime-web");

async function loadOrt(): Promise<ORT> {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
  return ort as unknown as ORT;
}

export function TrialOutcomePredictor() {
  const [ort, setOrt] = useState<ORT | null>(null);
  const [session, setSession] = useState<any>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [examples, setExamples] = useState<ExampleTrial[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [proba, setProba] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  // 8 form inputs
  const [phaseP3only, setPhaseP3only] = useState(1);
  const [drugType, setDrugType] = useState(0);
  const [sponsor, setSponsor] = useState(0);
  const [enrollment, setEnrollment] = useState(200);
  const [duration, setDuration] = useState(730);
  const [oncology, setOncology] = useState(0);
  const [nPriorDrug, setNPriorDrug] = useState(2);
  const [nPriorTarget, setNPriorTarget] = useState(5);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const ortMod = await loadOrt();
        if (!alive) return;
        setOrt(ortMod);
        const [mr, er] = await Promise.all([fetch(META_URL), fetch(EXAMPLES_URL)]);
        const m = (await mr.json()) as Meta;
        const ex = (await er.json()) as ExampleTrial[];
        if (!alive) return;
        setMeta(m);
        setExamples(ex);
        const s = await ortMod.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        if (!alive) return;
        setSession(s);
        setStatus("ready");
      } catch (e: any) {
        console.error("Model load failed", e);
        if (!alive) return;
        setLoadErr(String(e?.message || e));
        setStatus("error");
      }
    })();
    return () => { alive = false; };
  }, []);

  const features = useMemo(
    () => [
      phaseP3only,
      drugType,
      sponsor,
      Math.log1p(enrollment),
      duration,
      oncology,
      nPriorDrug,
      nPriorTarget,
    ],
    [phaseP3only, drugType, sponsor, enrollment, duration, oncology, nPriorDrug, nPriorTarget]
  );

  // Predict whenever inputs change
  useEffect(() => {
    if (!ort || !session) return;
    let alive = true;
    (async () => {
      setBusy(true);
      try {
        const f32 = new Float32Array(features);
        const tensor = new ort.Tensor("float32", f32, [1, 8]);
        const result = await session.run({ trial_features: tensor });
        if (!alive) return;
        // LightGBM ONNX output: [labels, probabilities]
        // probabilities is (1,2) — col 1 is positive class
        const outNames = Object.keys(result);
        const probsTensor = result[outNames[1]] ?? result[outNames[0]];
        const data = Array.from(probsTensor.data as Float32Array);
        // 2-class softmax output, [prob_neg, prob_pos]
        const pos = data.length === 2 ? data[1] : data[0];
        setProba(pos);
      } catch (e) {
        console.error(e);
      } finally {
        if (alive) setBusy(false);
      }
    })();
    return () => { alive = false; };
  }, [ort, session, features]);

  if (status === "loading") {
    return (
      <div className="my-6 rounded-lg border p-4 text-sm text-muted-foreground">
        Loading trial outcome model…
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
  if (!meta) return null;

  return (
    <div className="my-6 flex flex-col gap-4">
      {/* Form + result side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Form */}
        <div className="rounded-lg border p-4 bg-card space-y-3">
          <h4 className="text-sm font-semibold mb-1">Trial design inputs</h4>

          {/* Phase */}
          <div>
            <label className="block text-xs font-medium mb-1">Phase</label>
            <div className="flex gap-1.5">
              <button
                onClick={() => setPhaseP3only(1)}
                className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs ${
                  phaseP3only === 1
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-muted hover:border-primary/40"
                }`}
              >
                Phase 3
              </button>
              <button
                onClick={() => setPhaseP3only(0)}
                className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs ${
                  phaseP3only === 0
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-muted hover:border-primary/40"
                }`}
              >
                Phase 2/3
              </button>
            </div>
          </div>

          {/* Drug type */}
          <div>
            <label className="block text-xs font-medium mb-1">Intervention type</label>
            <select
              value={drugType}
              onChange={(e) => setDrugType(Number(e.target.value))}
              className="w-full rounded-md border bg-background px-2 py-1.5 text-xs"
            >
              {meta.drug_type_labels.map((l, i) => (
                <option key={l} value={i}>{l}</option>
              ))}
            </select>
          </div>

          {/* Sponsor */}
          <div>
            <label className="block text-xs font-medium mb-1">Sponsor class</label>
            <select
              value={sponsor}
              onChange={(e) => setSponsor(Number(e.target.value))}
              className="w-full rounded-md border bg-background px-2 py-1.5 text-xs"
            >
              {meta.sponsor_class_labels.map((l, i) => (
                <option key={l} value={i}>{l}</option>
              ))}
            </select>
          </div>

          {/* Oncology */}
          <div>
            <label className="block text-xs font-medium mb-1">Indication</label>
            <div className="flex gap-1.5">
              <button
                onClick={() => setOncology(0)}
                className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs ${
                  oncology === 0
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-muted hover:border-primary/40"
                }`}
              >
                Non-oncology
              </button>
              <button
                onClick={() => setOncology(1)}
                className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs ${
                  oncology === 1
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-muted hover:border-primary/40"
                }`}
              >
                Oncology
              </button>
            </div>
          </div>

          {/* Enrollment */}
          <div>
            <label className="flex justify-between text-xs font-medium mb-1">
              <span>Planned enrollment</span>
              <span className="font-mono text-primary">{enrollment.toLocaleString()}</span>
            </label>
            <input
              type="range"
              min={10}
              max={5000}
              step={10}
              value={enrollment}
              onChange={(e) => setEnrollment(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Duration */}
          <div>
            <label className="flex justify-between text-xs font-medium mb-1">
              <span>Planned duration (days)</span>
              <span className="font-mono text-primary">{duration} ({(duration / 365.25).toFixed(1)}y)</span>
            </label>
            <input
              type="range"
              min={30}
              max={2920}
              step={30}
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Prior trials */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="flex justify-between text-xs font-medium mb-1">
                <span>Prior trials, same drug</span>
                <span className="font-mono text-primary">{nPriorDrug}</span>
              </label>
              <input
                type="range"
                min={0}
                max={30}
                step={1}
                value={nPriorDrug}
                onChange={(e) => setNPriorDrug(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="flex justify-between text-xs font-medium mb-1">
                <span>Prior, same target</span>
                <span className="font-mono text-primary">{nPriorTarget}</span>
              </label>
              <input
                type="range"
                min={0}
                max={80}
                step={1}
                value={nPriorTarget}
                onChange={(e) => setNPriorTarget(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Result */}
        <div className="rounded-lg border p-4 bg-card flex flex-col justify-center items-stretch">
          <h4 className="text-sm font-semibold mb-3">Predicted P3 success probability</h4>

          {proba !== null ? (
            <>
              <div className="text-center mb-4">
                <div
                  className="text-6xl font-bold"
                  style={{
                    color: proba > 0.5 ? "#2f8d46" : proba > 0.25 ? "#d1962a" : "#c0392b",
                  }}
                >
                  {(proba * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  base rate (P3 cohort): ~11.8%
                </div>
              </div>

              {/* Sigmoid bar */}
              <div className="h-4 rounded bg-muted overflow-hidden relative">
                <div
                  className="h-full transition-[width] duration-200"
                  style={{
                    width: `${proba * 100}%`,
                    backgroundColor: proba > 0.5 ? "#2f8d46" : proba > 0.25 ? "#d1962a" : "#c0392b",
                  }}
                />
                <div
                  className="absolute top-0 h-full w-px bg-foreground/30"
                  style={{ left: "11.8%" }}
                  title="cohort base rate"
                />
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                <span>0%</span>
                <span className="italic">↑ cohort base rate</span>
                <span>100%</span>
              </div>

              <div className="mt-4 text-xs text-muted-foreground leading-relaxed">
                <strong>Interpretation.</strong>{" "}
                {proba > 0.5
                  ? "Above this cohort's base rate. Some combination of indication, sponsor, intervention type, and trial-design choices puts this design profile in a historically more-likely-to-complete bucket."
                  : proba > 0.25
                  ? "Above base rate but not by much. Closer to ambient cohort risk than to a clear success signal."
                  : "At or below the cohort's historical base rate of completed P3 trials reaching primary-endpoint success."}
              </div>
            </>
          ) : (
            <div className="text-sm text-muted-foreground text-center">
              {busy ? "running model…" : "set inputs to predict"}
            </div>
          )}
        </div>
      </div>

      {/* Honesty banner */}
      <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-xs leading-relaxed">
        <strong>What this demo is.</strong> This is a <em>simplified</em> 8-feature LightGBM
        model exported to ONNX and running locally in your browser. The full leak-clean
        Day-2 LightGBM v2 has 134 features and achieves audited AUROC = {meta.headline_real_audited_auroc}{" "}
        on the same p3/time test split (this demo: AUROC = {meta.demo_eval_auroc.toFixed(4)}). The full
        Day-3 multimodal fusion (ChemBERTa + ESM2 + SciBERT + tabular) hits 0.8746 on the
        same split. The demo is for interaction; the headline number on the project page is
        from the audited full model.
      </div>

      {/* Example trials */}
      {examples.length > 0 && (
        <details className="rounded-lg border bg-card">
          <summary className="cursor-pointer px-4 py-2 text-sm font-medium">
            5 real test-set trials (predicted vs actual)
          </summary>
          <div className="px-4 pb-3 space-y-2">
            {examples.map((ex) => (
              <div key={ex.nct_id} className="text-xs flex flex-wrap items-center gap-2 border-t pt-2 first:border-t-0">
                <a
                  href={`https://clinicaltrials.gov/study/${ex.nct_id}`}
                  target="_blank"
                  rel="noopener"
                  className="font-mono underline text-primary"
                >
                  {ex.nct_id}
                </a>
                <span className="font-mono opacity-70">
                  → predicted {(ex.predicted_proba * 100).toFixed(1)}%
                </span>
                <span
                  className={
                    ex.actual_success
                      ? "text-emerald-600 font-medium"
                      : "text-red-600 font-medium"
                  }
                >
                  actual: {ex.actual_success ? "success" : "failure"}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

export default TrialOutcomePredictor;
