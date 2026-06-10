#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CANON = ["normal", "dos", "spoof", "fdi", "stopped_conv"]
NORMAL_ID = 0

class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, latent: int = 48, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=(dropout if num_layers > 1 else 0.0))
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(input_size=hidden, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=(dropout if num_layers > 1 else 0.0))
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        enc_out, _ = self.encoder(x)         # [B,T,H]
        h_last = enc_out[:, -1, :]           # [B,H]
        z = self.to_latent(h_last)           # [B,Z]
        h0 = torch.tanh(self.from_latent(z)).unsqueeze(0)  # [1,B,H]
        c0 = torch.zeros_like(h0)

        B, T, _ = x.shape
        dec_in = h0.transpose(0, 1).repeat(1, T, 1)        # [B,T,H]
        dec_out, _ = self.decoder(dec_in, (h0, c0))        # [B,T,H]
        y = self.out(dec_out)                               # [B,T,F]
        return y

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lstm_windows.npz")
    ap.add_argument("--outdir", default="models_lstm")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--latent", type=int, default=48)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)

    ap.add_argument("--thr-mode", choices=["quantile", "f1"], default="f1",
                    help="quantile = threshold from VAL normals quantile; f1 = threshold maximizing VAL F1 (recommended).")
    ap.add_argument("--thr-quantile", type=float, default=0.995)
    return ap.parse_args()

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def standardize_fit(X):
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)

def standardize_apply(X, mu, sd):
    return (X - mu[None, None, :]) / sd[None, None, :]

def recon_error(model, X, device):
    if X is None or X.shape[0] == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = torch.from_numpy(X[i:i+512]).to(device)
            yb = model(xb)
            e = (yb - xb).pow(2).mean(dim=(1, 2)).detach().cpu().numpy()
            out.append(e)
    return np.concatenate(out, axis=0) if out else np.array([], dtype=np.float32)

def _label_counts(y):
    return {name: int((y == i).sum()) for i, name in enumerate(CANON)}

def f1_best_threshold(scores: np.ndarray, y_bin: np.ndarray):
    # brute sweep over unique sorted scores (fast enough)
    if scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    ths = np.unique(scores)
    best = (0.0, 0.0, 0.0, float("inf"))  # f1, prec, rec, thr
    # sample thresholds for speed if too many
    if ths.size > 5000:
        idx = np.linspace(0, ths.size - 1, 5000).astype(int)
        ths = ths[idx]

    for t in ths:
        yp = (scores > t).astype(int)
        tp = int(((yp == 1) & (y_bin == 1)).sum())
        fp = int(((yp == 1) & (y_bin == 0)).sum())
        fn = int(((yp == 0) & (y_bin == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[0]:
            best = (f1, prec, rec, float(t))
    return best  # f1, prec, rec, thr

def main():
    a = parse_args()
    set_seed(a.seed)

    d = np.load(a.data, allow_pickle=True)
    X_train, y_train = d["X_train"].astype(np.float32), d["y_train"].astype(int)
    X_val, y_val     = d["X_val"].astype(np.float32),   d["y_val"].astype(int)

    feature_names = d["feature_names"].astype(object).tolist() if "feature_names" in d else None

    X_train_n = X_train[y_train == NORMAL_ID]
    X_val_n   = X_val[y_val == NORMAL_ID]

    print("[data]")
    print("  train:", X_train.shape, _label_counts(y_train))
    print("  val  :", X_val.shape,   _label_counts(y_val))
    print("  train_normal_windows:", int(X_train_n.shape[0]))
    print("  val_normal_windows  :", int(X_val_n.shape[0]))

    if X_train_n.shape[0] < 100:
        raise ValueError("Too few NORMAL windows. Increase normal runtime/runs or reduce window/stride.")

    mu, sd = standardize_fit(X_train_n)
    X_train_n_s = standardize_apply(X_train_n, mu, sd)
    X_val_n_s   = standardize_apply(X_val_n,   mu, sd)

    os.makedirs(a.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAE(
        n_features=X_train.shape[-1],
        hidden=a.hidden,
        latent=a.latent,
        num_layers=a.layers,
        dropout=a.dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_n_s)),
        batch_size=a.batch, shuffle=True, drop_last=False
    )

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, a.epochs + 1):
        model.train()
        losses = []

        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            yb = model(xb)
            loss = loss_fn(yb, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        val_scores = recon_error(model, X_val_n_s, device)
        val_err = float(val_scores.mean())
        tr_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.6f} val_recon_mse={val_err:.6f}")

        if val_err < best_val - 1e-6:
            best_val = val_err
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= a.patience:
                print("[early stop]")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- threshold selection ----------
    if a.thr_mode == "quantile":
        val_scores = recon_error(model, X_val_n_s, device)
        thr = float(np.quantile(val_scores, a.thr_quantile))
        thr_info = {"mode": "quantile", "thr_quantile": a.thr_quantile}

        print("[thr] mode=quantile quantile", a.thr_quantile, "->", thr)

    else:
        # compute on FULL val set (normals + attacks) for best F1
        X_val_s = standardize_apply(X_val, mu, sd)
        scores_full = recon_error(model, X_val_s, device)
        y_bin = (y_val != NORMAL_ID).astype(int)

        f1, prec, rec, thr = f1_best_threshold(scores_full, y_bin)
        thr_info = {"mode": "f1", "val_f1": f1, "val_precision": prec, "val_recall": rec}

        print(f"[thr] mode=f1 best_val: thr={thr:.6f} f1={f1:.3f} precision={prec:.3f} recall={rec:.3f}")

    # save
    torch.save(model.state_dict(), os.path.join(a.outdir, "lstm_ae.pt"))
    np.savez(os.path.join(a.outdir, "scaler.npz"), mu=mu, sd=sd)
    with open(os.path.join(a.outdir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"recon_mse_threshold": float(thr), **thr_info}, f, indent=2)

    cfg = {
        "hidden": a.hidden,
        "latent": a.latent,
        "layers": a.layers,
        "dropout": a.dropout,
        "window": int(d["window"][0]) if "window" in d else int(X_train.shape[1]),
        "n_features": int(X_train.shape[2]),
        "label_map": CANON,
        "feature_names": feature_names,
    }
    with open(os.path.join(a.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[save] model+scaler+thresholds+config -> {a.outdir}/")

if __name__ == "__main__":
    main()
