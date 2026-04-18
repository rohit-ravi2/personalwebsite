#!/usr/bin/env python3
"""Train the art classifier on HuggingFace Hemg/AI-Generated-vs-Real-Images-Datasets parquet shards.

Already-downloaded shards live at /tmp/art-data/shard-{0,1,4}.parquet.
Dataset's native label: 0=AI, 1=Human. We remap to our {0=Human, 1=AI}
so the UI's `classes: ["Human", "AI"]` ordering is correct.

Exports to public/models/art-classifier.onnx (overwrites placeholder)
and updates public/models/meta.json.
"""
from __future__ import annotations

import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_art_classifier import TunedCNN, export_onnx, OUT_DIR, MODEL_PATH  # noqa: E402

IMG_SIZE = 128
SHARDS = [Path(f"/tmp/art-data/shard-{s}.parquet") for s in (0, 1, 4)]
TIME_BUDGET_S = int(__import__("os").environ.get("TIME_BUDGET_S", "4200"))  # 70 min
MIN_VAL_ACC = 0.70


def load_dataframe() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in SHARDS if p.exists()]
    df = pd.concat(parts, ignore_index=True)
    # Remap: dataset 0=AI -> our 1; dataset 1=Human -> our 0
    df["y"] = 1 - df["label"].astype(int)
    return df


def balance(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    h = df[df["y"] == 0].sample(n=per_class, random_state=0)
    a = df[df["y"] == 1].sample(n=per_class, random_state=0)
    out = pd.concat([h, a]).sample(frac=1, random_state=0).reset_index(drop=True)
    return out


class ParquetImageDs(Dataset):
    def __init__(self, frame: pd.DataFrame, tf):
        self.frame = frame.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        try:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        return self.tf(img), int(row["y"])


def main():
    df = load_dataframe()
    print(f"loaded {len(df):,} rows  class balance: {df['y'].value_counts().to_dict()}")

    # Balance + cap. Take min(counts) per class but cap at 15k per class for time.
    per_class = min(df[df["y"] == 0].shape[0], df[df["y"] == 1].shape[0], 15000)
    df = balance(df, per_class)
    n = len(df)
    n_val = int(0.1 * n)
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]
    print(f"train={len(train_df):,}  val={len(val_df):,}  per_class={per_class}")

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    model = TunedCNN(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Mild LR decay
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    BATCH = 128
    train_loader = DataLoader(
        ParquetImageDs(train_df, tf_train), batch_size=BATCH, shuffle=True,
        num_workers=4, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        ParquetImageDs(val_df, tf_eval), batch_size=256, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
    )

    t0 = time.time()
    best_acc = 0.0
    best_state = None
    history = []
    MAX_EPOCHS = 15

    for epoch in range(MAX_EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[time] budget exhausted at epoch {epoch}"); break
        model.train()
        running = 0.0; seen = 0
        for bi, (x, y) in enumerate(train_loader):
            if time.time() - t0 > TIME_BUDGET_S: break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0); seen += x.size(0)
        sched.step()
        train_loss = running / max(1, seen)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item(); total += y.numel()
        acc = correct / max(1, total)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_acc": round(acc, 4), "elapsed_min": round(elapsed/60, 2)})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_acc={acc:.4f} (best={best_acc:.4f}) elapsed={elapsed/60:.1f}m")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None or best_acc < MIN_VAL_ACC:
        print(f"[abort] best_acc={best_acc:.4f} < floor {MIN_VAL_ACC}. Not replacing placeholder.", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    export_onnx(
        model.cpu(),
        MODEL_PATH,
        ["Human", "AI"],
        note=(
            f"Binary Human-vs-AI classifier (3-conv CNN, 32/64/64 filters + dense 128). "
            f"Trained on a ~{2*per_class:,}-image subset of Hemg/AI-Generated-vs-Real-Images-Datasets "
            f"(HuggingFace); Human=label 0, AI=label 1. Held-out validation accuracy: {best_acc:.3f}. "
            f"A separate fuller model described on this page reaches 93.71% on the original 270k-image "
            f"47-category dataset — the in-browser demo intentionally uses a smaller, faster variant."
        ),
    )
    # Also write training history
    (OUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"DONE best_acc={best_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
