#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, confusion_matrix
)

CANON = ["normal", "dos", "spoof", "fdi", "stopped_conv"]
NORMAL_ID = 0

class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(input_size=hidden, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]
        z = self.to_latent(h_last)
        h0 = torch.tanh(self.from_latent(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        B, T, _ = x.shape
        dec_in = h0.transpose(0, 1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        return self.out(dec_out)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lstm_windows.npz")
    ap.add_argument("--modeldir", default="models_lstm")
    ap.add_argument("--plotdir", default="plots_lstm")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--threshold-file", default="thresholds.json")
    return ap.parse_args()

def standardize_apply(X, mu, sd):
    return (X - mu[None, None, :]) / sd[None, None, :]

def recon_scores_and_feature_mse(model, X, device):
    if X is None or X.shape[0] == 0:
        return np.array([], dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    model.eval()
    scores = []
    feat_mse = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 256):
            xb = torch.from_numpy(X[i:i+256]).to(device)
            yb = model(xb)
            err = (yb - xb).pow(2)
            s = err.mean(dim=(1,2)).detach().cpu().numpy()
            f = err.mean(dim=1).detach().cpu().numpy()
            scores.append(s)
            feat_mse.append(f)
    return np.concatenate(scores, axis=0), np.concatenate(feat_mse, axis=0)

def plot_confusion_matrix(cm, labels, outpath, title):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() * 0.5 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    a = parse_args()
    os.makedirs(a.plotdir, exist_ok=True)

    d = np.load(a.data, allow_pickle=True)
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(int)

    feature_names = d["feature_names"].astype(object).tolist() if "feature_names" in d else [f"f{i}" for i in range(X_test.shape[-1])]

    scaler = np.load(os.path.join(a.modeldir, "scaler.npz"))
    mu, sd = scaler["mu"].astype(np.float32), scaler["sd"].astype(np.float32)

    with open(os.path.join(a.modeldir, a.threshold_file), "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["recon_mse_threshold"])

    with open(os.path.join(a.modeldir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAE(
        n_features=cfg["n_features"],
        hidden=cfg["hidden"],
        latent=cfg["latent"],
        num_layers=cfg["layers"],
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(a.modeldir, "lstm_ae.pt"), map_location=device))

    Xs = standardize_apply(X_test, mu, sd)
    scores, feat_mse = recon_scores_and_feature_mse(model, Xs, device)

    y_bin = (y_test != NORMAL_ID).astype(int)
    y_pred = (scores > threshold).astype(int)

    pos_rate = float(y_bin.mean()) if y_bin.size else 0.0
    baseline_pr = pos_rate

    prec = precision_score(y_bin, y_pred, zero_division=0)
    rec  = recall_score(y_bin, y_pred, zero_division=0)
    f1   = f1_score(y_bin, y_pred, zero_division=0)
    cm   = confusion_matrix(y_bin, y_pred)

    print("[binary CM]\n", cm)
    print(f"[binary] precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} thr={threshold:.6f}")
    print(f"[prevalence] attack_rate={pos_rate:.3f} baseline_PR={baseline_pr:.3f}")

    pr_p, pr_r, _ = precision_recall_curve(y_bin, scores)
    roc_fpr, roc_tpr, _ = roc_curve(y_bin, scores)
    pr_auc = auc(pr_r, pr_p)
    roc_auc = auc(roc_fpr, roc_tpr)
    print(f"[curves] PR-AUC={pr_auc:.3f} ROC-AUC={roc_auc:.3f}")

    plot_confusion_matrix(cm, labels=["normal", "attack"],
                          outpath=os.path.join(a.plotdir, "confusion_matrix_binary.png"),
                          title="Confusion Matrix (binary)")

    # PR curve
    plt.figure()
    plt.plot(pr_r, pr_p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AUC={pr_auc:.3f})  baseline={baseline_pr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(a.plotdir, "pr_curve.png"), dpi=300)
    plt.close()

    # ROC curve
    plt.figure()
    plt.plot(roc_fpr, roc_tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC curve (AUC={roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(a.plotdir, "roc_curve.png"), dpi=300)
    plt.close()

    # Boxplot per scenario (fixed warning)
    data, labels = [], []
    for sid, name in enumerate(CANON):
        mask = (y_test == sid)
        if mask.sum() == 0:
            continue
        data.append(scores[mask])
        labels.append(name)

    plt.figure()
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(threshold, linestyle="--")
    plt.title("Score boxplot by scenario")
    plt.ylabel("Reconstruction MSE score")
    plt.tight_layout()
    plt.savefig(os.path.join(a.plotdir, "score_boxplot.png"), dpi=300)
    plt.close()

    # Threshold sweep
    qs = np.linspace(0.80, 0.999, 80)
    ths = np.quantile(scores, qs) if scores.size else np.array([])
    rows = []
    for t in ths:
        yp = (scores > t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_bin, yp, zero_division=0)),
            "recall": float(recall_score(y_bin, yp, zero_division=0)),
            "f1": float(f1_score(y_bin, yp, zero_division=0)),
        })
    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(os.path.join(a.plotdir, "threshold_sweep.csv"), index=False)

    # Per-scenario flag rate
    per_rows = []
    for sid, name in enumerate(CANON):
        mask = (y_test == sid)
        if mask.sum() == 0:
            continue
        per_rows.append({
            "scenario": name,
            "windows": int(mask.sum()),
            "flag_rate": float(y_pred[mask].mean()),
            "mean_score": float(scores[mask].mean()),
            "median_score": float(np.median(scores[mask])),
        })
    pd.DataFrame(per_rows).to_csv(os.path.join(a.plotdir, "per_scenario_flag_rate.csv"), index=False)

    # Summary
    summary = pd.DataFrame([{
        "threshold": threshold,
        "precision": prec, "recall": rec, "f1": f1,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "attack_rate": pos_rate,
        "baseline_pr": baseline_pr,
        "n_test": int(len(y_test)),
    }])
    summary.to_csv(os.path.join(a.plotdir, "metrics_summary.csv"), index=False)

    pd.DataFrame({"feature": feature_names}).to_csv(os.path.join(a.plotdir, "feature_schema.csv"), index=False)

    print(f"[save] plots+metrics -> {a.plotdir}/")

if __name__ == "__main__":
    main()
