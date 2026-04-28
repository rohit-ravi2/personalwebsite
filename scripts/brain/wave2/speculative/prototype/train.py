"""
X.1d prototype — train the 2-node mechanistic GNN against single-compartment ground truth.

Bounded prototype:
- Subsample trajectories aggressively (8000 → 200 steps) so inner loop completes fast.
- Loss: MSE on V_0 trajectory.
- Adam, ~30 epochs.
- Stdout flushed each line.

Output: results.json with summary statistics and trained_model.pt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from gnn_prototype import TwoNodeGNN, load_data, EGL19, LEAK


def log(msg):
    print(msg, flush=True)


def train_one_epoch(model, optimizer, V_target, I_inputs, batch_size=8):
    model.train()
    n = V_target.shape[0]
    perm = torch.randperm(n)
    losses = []
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        I_batch = I_inputs[idx]
        V_batch = V_target[idx]
        optimizer.zero_grad()
        V_pred = model(I_batch)
        loss = ((V_pred - V_batch) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, V_target, I_inputs):
    model.eval()
    V_pred = model(I_inputs)
    err = (V_pred - V_target).abs()
    mae_per_trace = err.mean(dim=1)
    return {
        "mae_per_trace": mae_per_trace.cpu().numpy(),
        "mae_mean": float(mae_per_trace.mean().item()),
        "mae_max": float(mae_per_trace.max().item()),
    }


def main():
    here = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    data = load_data(here / "train_data.npz")
    V = data["V_traces"].to(device)
    I = data["I_inputs"].to(device)
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    n_steps = V.shape[1]
    dt_ms = float(data["times"][1] - data["times"][0])
    log(f"Raw data: {V.shape[0]} traces, {n_steps} steps, dt={dt_ms} ms")
    log(f"V range: [{V.min().item():.2f}, {V.max().item():.2f}] mV")

    # Subsample: 8000 → 1600 (×5). Forward Euler on EGL-19 dynamics needs dt < ~0.2 ms
    # for stability; we'll use dt_sub = 0.125 ms which is comfortably stable.
    # Trace length 1600 × dt_sub × 200 ms backprop is the bound.
    SUBSAMPLE = 5
    V_sub = V[:, ::SUBSAMPLE]
    I_sub = I[:, ::SUBSAMPLE]
    dt_sub = dt_ms * SUBSAMPLE
    log(f"Subsampled: {V_sub.shape[1]} steps, dt={dt_sub} ms")

    V_train = V_sub[train_idx]
    I_train = I_sub[train_idx]
    V_test = V_sub[test_idx]
    I_test = I_sub[test_idx]

    model = TwoNodeGNN(n_steps=V_train.shape[1], dt_ms=dt_sub).to(device)

    init_eval = evaluate(model, V_test, I_test)
    init_axial_g = float(torch.exp(model.log_axial_g).item())
    log(f"Init test MAE: {init_eval['mae_mean']:.3f} mV (max {init_eval['mae_max']:.3f}); axial_g={init_axial_g:.4e}")

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {"train_loss": [], "test_mae": [], "axial_g": []}
    n_epochs = 40
    t0 = time.time()
    for epoch in range(n_epochs):
        loss = train_one_epoch(model, optimizer, V_train, I_train, batch_size=8)
        scheduler.step()
        ev = evaluate(model, V_test, I_test)
        ag = float(torch.exp(model.log_axial_g).item())
        history["train_loss"].append(loss)
        history["test_mae"].append(ev["mae_mean"])
        history["axial_g"].append(ag)
        if epoch % 2 == 0 or epoch == n_epochs - 1:
            log(f"Ep {epoch:3d}  loss={loss:8.4f}  test MAE={ev['mae_mean']:.3f} mV  axial_g={ag:.4e}")

    elapsed = time.time() - t0
    final = evaluate(model, V_test, I_test)
    log(f"\nFinal test MAE: {final['mae_mean']:.3f} mV (max {final['mae_max']:.3f})")
    log(f"Final axial_g: {float(torch.exp(model.log_axial_g).item()):.4e}")
    log(f"Final gbar_egl19: {float(torch.exp(model.log_gbar_egl19).item()):.4e}  (init {EGL19['gbar_Scm2']:.4e})")
    log(f"Final gleak0: {float(torch.exp(model.log_gleak0).item()):.4e}  (init {LEAK['g_leak_Scm2']:.4e})")
    log(f"Trained in {elapsed:.1f}s")

    results = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_steps_subsampled": int(V_train.shape[1]),
        "dt_ms_subsampled": dt_sub,
        "n_epochs": n_epochs,
        "elapsed_s": elapsed,
        "init_test_mae_mV": init_eval["mae_mean"],
        "init_test_mae_max_mV": init_eval["mae_max"],
        "final_test_mae_mV": final["mae_mean"],
        "final_test_mae_max_mV": final["mae_max"],
        "final_axial_g_S": float(torch.exp(model.log_axial_g).item()),
        "init_axial_g_S": init_axial_g,
        "final_gbar_egl19_Scm2": float(torch.exp(model.log_gbar_egl19).item()),
        "init_gbar_egl19_Scm2": float(EGL19["gbar_Scm2"]),
        "final_gleak0_Scm2": float(torch.exp(model.log_gleak0).item()),
        "init_gleak_Scm2": float(LEAK["g_leak_Scm2"]),
        "final_gleak1_Scm2": float(torch.exp(model.log_gleak1).item()),
        "history": history,
        "device": str(device),
    }
    out_path = here / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved results to {out_path}")

    torch.save(model.state_dict(), here / "trained_model.pt")
    log(f"Saved model to {here / 'trained_model.pt'}")

    # Pass criteria for prototype:
    # The single-compartment ground truth is what node 0 must reproduce.
    # If the GNN can't get test MAE < 5 mV, the prototype fails its sanity check.
    # If it does get there, the prototype passes.
    pass_threshold = 5.0
    passed = final["mae_mean"] < pass_threshold
    log(f"\nPROTOTYPE OUTCOME: {'PASS' if passed else 'FAIL'} (threshold {pass_threshold} mV)")
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
