#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Canonical class order (must match 02_build_features_lstm.py)
# -------------------------
CANON = ["normal", "dos", "spoof", "fdi", "stopped_conv"]

# -------------------------
# Model: Temporal CNN (Conv1D over time)
# Input windows: (B, T, F)
# Conv1D expects: (B, C, L) -> we use C=F, L=T
# -------------------------
class TemporalCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int,
                 ch1: int = 128, ch2: int = 128, ch3: int = 128,
                 k1: int = 7, k2: int = 5, k3: int = 3,
                 dropout: float = 0.25):
        super().__init__()

        def block(cin, cout, k):
            pad = k // 2
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=pad),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.net = nn.Sequential(
            block(n_features, ch1, k1),
            block(ch1, ch2, k2),
            block(ch2, ch3, k3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch3, n_classes)

    def forward(self, x):
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.head(h)

# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_npz(path: str):
    z = np.load(path, allow_pickle=True)

    X_train = z["X_train"].astype(np.float32)
    y_train = z["y_train"]
    X_val   = z["X_val"].astype(np.float32)
    y_val   = z["y_val"]
    X_test  = z["X_test"].astype(np.float32)
    y_test  = z["y_test"]

    feature_names = None
    if "feature_names" in z.files:
        feature_names = [str(s) for s in z["feature_names"].tolist()]

    # NEW: prefer label_map stored by 02_build_features_lstm.py
    label_map = None
    if "label_map" in z.files:
        label_map = [str(x) for x in z["label_map"].tolist()]
    else:
        label_map = CANON[:]  # fallback

    # also store these if present (for reproducibility / proxy defaults)
    window = int(z["window"][0]) if "window" in z.files else None
    stride = int(z["stride"][0]) if "stride" in z.files else None
    log1p_counts = int(z["log1p_counts"][0]) if "log1p_counts" in z.files else 0

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, label_map, window, stride, log1p_counts

def fit_scaler(X_train):
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_scaler(X, mu, sd):
    return (X - mu[None, None, :]) / sd[None, None, :]

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def macro_f1_from_cm(cm: np.ndarray):
    K = cm.shape[0]
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        f1s.append(f1)
    return float(np.mean(f1s))

@torch.no_grad()
def eval_epoch(model, loader, device, n_classes):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        total += int(yb.numel())
        correct += int((pred == yb).sum().item())
        for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy()):
            cm[int(t), int(p)] += 1
    acc = correct / max(1, total)
    mf1 = macro_f1_from_cm(cm)
    return acc, mf1, cm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lstm_windows.npz")
    ap.add_argument("--outdir", default="models_tcnn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weighting", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.25)
    return ap.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    Xtr, ytr, Xva, yva, Xte, yte, feature_names, label_map, win_npz, stride_npz, log1p_counts = load_npz(args.data)

    # --------- NEW: enforce canonical label mapping ----------
    # If y are ints already (0..K-1), keep them.
    # If y are strings, map them to ids using label_map.
    label2id = {label_map[i]: i for i in range(len(label_map))}
    id2label = {i: label_map[i] for i in range(len(label_map))}
    n_classes = len(label_map)

    def encode_y(y):
        arr = np.asarray(y)
        if arr.dtype.kind in ("U", "S", "O"):
            return np.array([label2id[str(v)] for v in arr.tolist()], dtype=np.int64)
        return arr.astype(np.int64)

    ytr_i = encode_y(ytr)
    yva_i = encode_y(yva)
    yte_i = encode_y(yte)

    # safety check: ensure labels are in range
    if ytr_i.size and (ytr_i.min() < 0 or ytr_i.max() >= n_classes):
        raise ValueError(f"y_train contains labels outside [0,{n_classes-1}]. Found min={ytr_i.min()} max={ytr_i.max()}")

    n_train, T, F = Xtr.shape

    # scaler
    mu, sd = fit_scaler(Xtr)
    Xtr_s = apply_scaler(Xtr, mu, sd)
    Xva_s = apply_scaler(Xva, mu, sd)
    Xte_s = apply_scaler(Xte, mu, sd)

    # class weights
    class_w = None
    if args.weighting == "balanced":
        counts = np.bincount(ytr_i, minlength=n_classes).astype(np.float32)
        w = counts.sum() / np.maximum(1.0, counts)
        w = w / w.mean()
        class_w = torch.from_numpy(w)

    # loaders
    train_ds = WindowDataset(Xtr_s, ytr_i)
    val_ds   = WindowDataset(Xva_s, yva_i)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCNN(n_features=F, n_classes=n_classes, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss(weight=(class_w.to(device) if class_w is not None else None))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_path = os.path.join(args.outdir, "temporal_cnn.pt")
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(float(loss.item()))

        val_acc, val_f1, _ = eval_epoch(model, val_loader, device, n_classes)
        print(f"[epoch {ep:02d}] train_loss={np.mean(losses):.6f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] patience={args.patience}")
                break

    # save artifacts
    np.savez(os.path.join(args.outdir, "scaler.npz"), mu=mu, sd=sd)

    cfg = {
        "model": "TemporalCNN",
        "n_features": int(F),
        "window": int(T),
        "n_classes": int(n_classes),

        # NEW: human-readable labels
        "label_map": label_map,
        "label2id": label2id,
        "id2label": id2label,

        "feature_names": feature_names,
        "seed": args.seed,
        "dropout": args.dropout,

        # NEW: keep preprocessing flags for proxy correctness
        "log1p_counts": int(log1p_counts),
        "stride": int(stride_npz) if stride_npz is not None else None,
    }

    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[save] model -> {best_path}")
    print(f"[save] scaler+config -> {args.outdir}/")
    print("[labels]", id2label)

if __name__ == "__main__":
    main()
