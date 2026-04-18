#!/usr/bin/env python3
"""Train (or emit a placeholder) art classifier and export to ONNX.

Two modes:
  - PLACEHOLDER=1 ./train_art_classifier.py
      Emit a randomly-initialized 3-conv CNN matching the notebook's
      `custom_tuned_cnn` architecture. Useful for shipping the UI
      before real training finishes.

  - ./train_art_classifier.py
      Pull the Kaggle dataset, train a compact binary classifier, emit
      ONNX. Time-budget enforced (default 80 min). Hard floor on
      validation accuracy before shipping.

Output: public/models/art-classifier.onnx + public/models/meta.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_SIZE = 128
OUT_DIR = Path(__file__).resolve().parents[1] / "public" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "art-classifier.onnx"
META_PATH = OUT_DIR / "meta.json"


class TunedCNN(nn.Module):
    """Architecture mirror of the notebook's `custom_tuned_cnn`.

    Notebook used TF/Keras `channels_last`; PyTorch default is
    `channels_first`, so the exported ONNX expects input of shape
    (N, 3, 128, 128) already normalised to [0,1].
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Conv block 1: 32 ch
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Conv block 2: 64 ch
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Conv block 3: 64 ch
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # After 3 maxpools of 128: 128 -> 64 -> 32 -> 16, so flat = 64*16*16
        self.drop = nn.Dropout(0.42)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)
        feat = F.relu(self.bn3(self.conv3(x))); x = F.max_pool2d(feat, 2)
        x = x.flatten(1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def export_onnx(model: nn.Module, path: Path, classes: list[str], note: str):
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    meta = {
        "classes": classes,
        "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
        "preprocessing": "resize to 128x128, divide by 255 (no further normalisation)",
        "note": note,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({size_kb:.1f} KB)")
    print(f"wrote {META_PATH}")


def emit_placeholder():
    torch.manual_seed(1)
    model = TunedCNN(num_classes=2)
    export_onnx(
        model,
        MODEL_PATH,
        ["Human", "AI"],
        note="PLACEHOLDER — randomly-initialised weights matching the architecture from the notebook. Predictions are ~uniform; replace with trained weights via scripts/train_art_classifier.py.",
    )


def main():
    if os.environ.get("PLACEHOLDER"):
        emit_placeholder()
        return

    # Real training path — requires Kaggle credentials + GPU + dataset.
    kcred = Path.home() / ".kaggle" / "kaggle.json"
    if not kcred.exists():
        print("[skip] no ~/.kaggle/kaggle.json — emitting placeholder instead", file=sys.stderr)
        emit_placeholder()
        return

    # Deferred heavy imports so the placeholder path stays fast
    import shutil
    import subprocess
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import random

    TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", "4800"))  # 80 min default
    MIN_VAL_ACC = float(os.environ.get("MIN_VAL_ACC", "0.70"))
    MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "20000"))

    data_dir = Path(os.environ.get("ART_DATA_DIR", "/tmp/art-data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Download a subset of the dataset via kaggle CLI. The full dataset is huge;
    # we rely on kaggle datasets download with -p then unzip. Safety: cap by
    # image count during the dataset scan.
    if not any(data_dir.iterdir()):
        print("[data] downloading adamelkholyy/human-ai-artwork-dataset ...")
        rc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "adamelkholyy/human-ai-artwork-dataset", "-p", str(data_dir), "--unzip"],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print("[skip] kaggle download failed:", rc.stderr[:500], file=sys.stderr)
            emit_placeholder()
            return

    # Collect files: anything under a folder whose name starts with Human_ or AI_.
    files, labels = [], []
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cls = p.parent.name
        if cls.startswith("Human"):
            files.append(p); labels.append(0)
        elif cls.startswith("AI"):
            files.append(p); labels.append(1)

    if len(files) < 2000:
        print(f"[skip] too few images ({len(files)}) — staying with placeholder", file=sys.stderr)
        emit_placeholder()
        return

    # Balance + subsample
    human = [f for f, l in zip(files, labels) if l == 0]
    ai = [f for f, l in zip(files, labels) if l == 1]
    random.seed(0)
    random.shuffle(human); random.shuffle(ai)
    per_class = min(len(human), len(ai), MAX_IMAGES // 2)
    human = human[:per_class]; ai = ai[:per_class]
    all_files = [(f, 0) for f in human] + [(f, 1) for f in ai]
    random.shuffle(all_files)
    split = int(len(all_files) * 0.85)
    train_items = all_files[:split]
    val_items = all_files[split:]
    print(f"[data] {len(train_items)} train / {len(val_items)} val (per-class = {per_class})")

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # [0,1]
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    class ArtDs(Dataset):
        def __init__(self, items, tf):
            self.items = items; self.tf = tf
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            fp, lbl = self.items[i]
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
            return self.tf(img), lbl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")
    model = TunedCNN(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_loader = DataLoader(ArtDs(train_items, tf_train), batch_size=64, shuffle=True, num_workers=4, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(ArtDs(val_items, tf_eval), batch_size=128, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))

    best_acc = 0.0
    best_state = None
    MAX_EPOCHS = 12
    for epoch in range(MAX_EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[train] time budget exhausted at epoch {epoch}")
            break
        model.train()
        for x, y in train_loader:
            if time.time() - t0 > TIME_BUDGET_S:
                break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item(); total += y.numel()
        acc = correct / max(1, total)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}] val_acc={acc:.4f} (best={best_acc:.4f})  elapsed={elapsed/60:.1f}m")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None or best_acc < MIN_VAL_ACC:
        print(f"[skip] best_acc={best_acc:.4f} < floor {MIN_VAL_ACC} — emitting placeholder", file=sys.stderr)
        emit_placeholder()
        return

    model.load_state_dict(best_state)
    export_onnx(
        model.cpu(),
        MODEL_PATH,
        ["Human", "AI"],
        note=f"Binary AI-vs-Human classifier, trained on {len(train_items)} images for in-browser inference. Val accuracy: {best_acc:.3f}. Architecture from finalnotebook.ipynb (tuned CNN, 3 conv blocks 32-64-64).",
    )


if __name__ == "__main__":
    main()
