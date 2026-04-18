import * as React from "react";
import { useCallback, useEffect, useRef, useState } from "react";

type Meta = {
  classes: string[];
  input_shape: number[];
  preprocessing: string;
  note: string;
};

const MODEL_URL = "/models/art-classifier.onnx";
const META_URL = "/models/meta.json";
const IMG_SIZE = 128;

// Type for the onnxruntime-web module, loaded lazily so SSR doesn't try.
type ORT = typeof import("onnxruntime-web");

function softmax(x: number[]): number[] {
  const m = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}

async function loadOrt(): Promise<ORT> {
  const ort = await import("onnxruntime-web");
  // Prefer WebGL > WASM SIMD > WASM
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
  return ort as unknown as ORT;
}

function imageToTensorRGB(img: HTMLImageElement, size: number): Float32Array {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  // CHW, normalised to [0,1]
  const out = new Float32Array(3 * size * size);
  const planeSize = size * size;
  for (let i = 0; i < planeSize; i++) {
    out[i] = data[i * 4] / 255;                // R
    out[planeSize + i] = data[i * 4 + 1] / 255; // G
    out[2 * planeSize + i] = data[i * 4 + 2] / 255; // B
  }
  return out;
}

export function ArtClassifierDemo() {
  const [ort, setOrt] = useState<ORT | null>(null);
  const [session, setSession] = useState<any>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [saliency, setSaliency] = useState<ImageData | null>(null);
  const [saliencyOn, setSaliencyOn] = useState(false);

  const overlayCanvas = useRef<HTMLCanvasElement>(null);

  // Load ORT + model + meta on mount.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const ortMod = await loadOrt();
        if (!alive) return;
        setOrt(ortMod);
        const mr = await fetch(META_URL);
        const m = mr.ok ? ((await mr.json()) as Meta) : {
          classes: ["Human", "AI"],
          input_shape: [1, 3, IMG_SIZE, IMG_SIZE],
          preprocessing: "resize to 128x128, divide by 255",
          note: "",
        };
        if (!alive) return;
        setMeta(m);

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

  // Image upload handling
  const onFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setImgUrl(url);
    setProbs(null);
    setSaliency(null);
    setSaliencyOn(false);
    const img = new Image();
    img.onload = () => setImgEl(img);
    img.src = url;
  }, []);

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (f) onFile(f);
  };
  const onDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0]; if (f) onFile(f);
  };
  const onDragOver = (e: React.DragEvent<HTMLLabelElement>) => e.preventDefault();

  // Core classification
  const classify = useCallback(async () => {
    if (!ort || !session || !imgEl) return;
    setBusy(true);
    try {
      const pixels = imageToTensorRGB(imgEl, IMG_SIZE);
      const tensor = new ort.Tensor("float32", pixels, [1, 3, IMG_SIZE, IMG_SIZE]);
      const feeds: Record<string, any> = { input: tensor };
      const result = await session.run(feeds);
      const outName = Object.keys(result)[0];
      const out = Array.from(result[outName].data as Float32Array);
      const p = softmax(out);
      setProbs(p);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  }, [ort, session, imgEl]);

  // Occlusion-based saliency — slide a gray patch, measure drop in predicted class.
  const computeSaliency = useCallback(async () => {
    if (!ort || !session || !imgEl || !probs) return;
    setBusy(true);
    try {
      const predClass = probs.indexOf(Math.max(...probs));
      const baseline = probs[predClass];
      const patchSize = 24;
      const stride = 12;
      const steps = Math.floor((IMG_SIZE - patchSize) / stride) + 1;

      // Draw full pixels once into a reusable canvas
      const base = document.createElement("canvas");
      base.width = IMG_SIZE; base.height = IMG_SIZE;
      const bctx = base.getContext("2d")!;
      bctx.drawImage(imgEl, 0, 0, IMG_SIZE, IMG_SIZE);
      const baseData = bctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

      // Saliency grid in steps x steps
      const drops = new Float32Array(steps * steps);
      for (let gy = 0; gy < steps; gy++) {
        for (let gx = 0; gx < steps; gx++) {
          // Re-draw and occlude
          bctx.putImageData(baseData, 0, 0);
          bctx.fillStyle = "rgb(128,128,128)";
          bctx.fillRect(gx * stride, gy * stride, patchSize, patchSize);

          // Pixels -> tensor
          const d = bctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
          const out = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
          const plane = IMG_SIZE * IMG_SIZE;
          for (let i = 0; i < plane; i++) {
            out[i] = d[i * 4] / 255;
            out[plane + i] = d[i * 4 + 1] / 255;
            out[2 * plane + i] = d[i * 4 + 2] / 255;
          }
          const t = new ort.Tensor("float32", out, [1, 3, IMG_SIZE, IMG_SIZE]);
          const res = await session.run({ input: t });
          const outName = Object.keys(res)[0];
          const p = softmax(Array.from(res[outName].data as Float32Array));
          drops[gy * steps + gx] = Math.max(0, baseline - p[predClass]);
        }
      }

      // Normalise + upsample to full IMG_SIZE and colorise.
      const maxD = Math.max(1e-6, Math.max(...drops));
      const canvas = document.createElement("canvas");
      canvas.width = IMG_SIZE; canvas.height = IMG_SIZE;
      const ctx = canvas.getContext("2d")!;
      const imgd = ctx.createImageData(IMG_SIZE, IMG_SIZE);
      for (let y = 0; y < IMG_SIZE; y++) {
        const gy = Math.min(steps - 1, Math.floor(y / stride));
        for (let x = 0; x < IMG_SIZE; x++) {
          const gx = Math.min(steps - 1, Math.floor(x / stride));
          const v = drops[gy * steps + gx] / maxD;
          const i = (y * IMG_SIZE + x) * 4;
          // red→yellow heatmap
          imgd.data[i] = Math.round(255 * Math.min(1, v * 1.4));
          imgd.data[i + 1] = Math.round(180 * Math.max(0, v - 0.15));
          imgd.data[i + 2] = 0;
          imgd.data[i + 3] = Math.round(180 * v);
        }
      }
      setSaliency(imgd);
      setSaliencyOn(true);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  }, [ort, session, imgEl, probs]);

  // Draw overlay when toggled
  useEffect(() => {
    const c = overlayCanvas.current;
    if (!c || !imgEl) return;
    c.width = imgEl.naturalWidth || IMG_SIZE;
    c.height = imgEl.naturalHeight || IMG_SIZE;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (saliencyOn && saliency) {
      // Upscale the 128x128 saliency to full image size
      const tmp = document.createElement("canvas");
      tmp.width = IMG_SIZE; tmp.height = IMG_SIZE;
      tmp.getContext("2d")!.putImageData(saliency, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmp, 0, 0, c.width, c.height);
    }
  }, [saliencyOn, saliency, imgEl]);

  const isPlaceholder = meta?.note?.includes("PLACEHOLDER");

  return (
    <div className="my-6 flex flex-col gap-4">
      {isPlaceholder && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs">
          <strong>Preview.</strong> The in-browser pipeline is live, but the model weights
          currently loaded are a placeholder (random init) with the same architecture as the
          trained classifier. Predictions are ~uniform. A trained model is on the way.
        </div>
      )}

      {status === "loading" && (
        <div className="rounded-lg border p-4 text-sm text-muted-foreground">Loading model…</div>
      )}
      {status === "error" && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm">
          Model failed to load: {loadErr}
        </div>
      )}

      {status === "ready" && (
        <>
          <label
            htmlFor="art-upload"
            onDrop={onDrop}
            onDragOver={onDragOver}
            className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/40 p-6 text-center text-sm transition-colors hover:border-primary"
          >
            <span className="font-medium">Drop an image here</span>
            <span className="text-xs text-muted-foreground">or click to choose a file (JPG / PNG)</span>
            <input id="art-upload" type="file" accept="image/*" onChange={onPick} className="hidden" />
          </label>

          {imgUrl && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="relative rounded-lg border overflow-hidden bg-muted">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={imgUrl} alt="upload" className="w-full h-auto block" />
                <canvas
                  ref={overlayCanvas}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  style={{ mixBlendMode: saliencyOn ? "multiply" : "normal" }}
                />
              </div>
              <div className="flex flex-col gap-3 text-sm">
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={classify}
                    disabled={busy || !imgEl}
                    className="rounded-md bg-primary px-3 py-1.5 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                  >
                    {busy ? "Working…" : "Classify"}
                  </button>
                  {probs && (
                    <>
                      <button
                        onClick={computeSaliency}
                        disabled={busy}
                        className="rounded-md border px-3 py-1.5 hover:bg-accent disabled:opacity-50"
                      >
                        {busy ? "Computing…" : saliency ? "Recompute saliency" : "Show saliency"}
                      </button>
                      {saliency && (
                        <button
                          onClick={() => setSaliencyOn((v) => !v)}
                          className="rounded-md border px-3 py-1.5 hover:bg-accent"
                        >
                          {saliencyOn ? "Hide overlay" : "Show overlay"}
                        </button>
                      )}
                    </>
                  )}
                </div>

                {probs && meta && (
                  <div className="flex flex-col gap-2">
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">Predicted</div>
                    {meta.classes.map((cls, i) => (
                      <div key={cls}>
                        <div className="flex justify-between text-xs">
                          <span className="font-medium">{cls}</span>
                          <span>{(probs[i] * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 rounded bg-muted overflow-hidden">
                          <div
                            className="h-full bg-primary transition-[width]"
                            style={{ width: `${(probs[i] * 100).toFixed(1)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {meta?.note && (
                  <details className="mt-2 text-xs text-muted-foreground">
                    <summary className="cursor-pointer">Model info</summary>
                    <p className="mt-1">{meta.note}</p>
                    <p className="mt-1">Preprocessing: {meta.preprocessing}</p>
                  </details>
                )}
              </div>
            </div>
          )}

          {!imgUrl && (
            <p className="text-xs text-muted-foreground">
              The entire classification runs locally in your browser via ONNX Runtime Web. No image
              ever leaves your device.
            </p>
          )}
        </>
      )}
    </div>
  );
}

export default ArtClassifierDemo;
