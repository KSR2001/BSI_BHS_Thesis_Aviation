#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
)

# ---------------- Model ----------------
class TemporalCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.25):
        super().__init__()

        def block(cin, cout, k, pdrop):
            pad = k // 2
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=pad),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout(pdrop),
            )

        self.net = nn.Sequential(
            block(n_features, 128, 7, dropout),
            block(128, 128, 5, dropout),
            block(128, 128, 3, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B,T,F] -> [B,F,T]
        x = x.transpose(1, 2)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.head(h)

# ---------------- Utils ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lstm_windows.npz")
    ap.add_argument("--modeldir", default="models_tcnn")
    ap.add_argument("--plotdir", default="plots_tcnn")
    ap.add_argument("--batch", type=int, default=512)

    ap.add_argument("--importance", choices=["saliency", "permutation", "none"], default="permutation")
    ap.add_argument("--imp-subsample", type=int, default=2000)
    ap.add_argument("--perm-repeats", type=int, default=3)

    ap.add_argument("--focus-pairs", default="normal->fdi,stopped_conv->dos")
    ap.add_argument("--telemetry-dt", type=float, default=1.0,
                    help="Only used for reporting detection delay if not stored in config.")
    return ap.parse_args()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def apply_scaler(X, mu, sd):
    return (X - mu[None, None, :]) / (sd[None, None, :] + 1e-6)

def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def load_label_maps(cfg, fallback_labels: Optional[list] = None) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Robustly loads id2label/label2id from config.json.
    """
    id2label = {}
    raw = cfg.get("id2label", None)

    if isinstance(raw, dict):
        # keys may be strings
        id2label = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        id2label = {int(i): str(v) for i, v in enumerate(raw)}
    else:
        # fallback: try label_map in cfg, else fallback_labels, else numeric
        lm = cfg.get("label_map", None)
        if isinstance(lm, list):
            id2label = {i: str(lm[i]) for i in range(len(lm))}
        elif fallback_labels:
            id2label = {i: str(fallback_labels[i]) for i in range(len(fallback_labels))}
        else:
            n = int(cfg.get("n_classes", 0))
            id2label = {i: str(i) for i in range(n)}

    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def plot_confusion(cm: np.ndarray, labels: list, out_png: str, normalize: bool = False, title: str = ""):
    cm2 = cm.astype(np.float64)
    if normalize:
        row_sums = cm2.sum(axis=1, keepdims=True)
        cm2 = np.divide(cm2, np.maximum(row_sums, 1e-12))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm2, interpolation="nearest")
    plt.title(title or ("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"))
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm2.max() * 0.5 if cm2.size else 0.5
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            txt = f"{cm2[i, j]:.2f}" if normalize else str(int(cm2[i, j]))
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if cm2[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def one_vs_rest_pr_curves(y_true: np.ndarray, prob: np.ndarray, class_names: list, out_png: str, out_csv: str):
    rows = []
    plt.figure(figsize=(8, 6))
    for k, name in enumerate(class_names):
        yk = (y_true == k).astype(int)
        pk = prob[:, k]
        prec, rec, _ = precision_recall_curve(yk, pk)
        ap = average_precision_score(yk, pk) if yk.sum() > 0 else float("nan")
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        rows.append({"class": k, "class_name": name, "average_precision": float(ap)})

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision–Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(class_names):
        yt = (y_true == k)
        tp = int(((y_pred == k) & yt).sum())
        fp = int(((y_pred == k) & (~yt)).sum())
        fn = int(((y_pred != k) & yt).sum())
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        rows.append({"class": k, "class_name": name, "precision": prec, "recall": rec, "f1": f1, "support": int(yt.sum())})
    return pd.DataFrame(rows)

def plot_bar_metrics(df_per: pd.DataFrame, out_png: str):
    names = df_per["class_name"].tolist()
    x = np.arange(len(names))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, df_per["precision"].values, width=w, label="precision")
    plt.bar(x,      df_per["recall"].values,    width=w, label="recall")
    plt.bar(x + w,  df_per["f1"].values,        width=w, label="f1")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Per-class Precision / Recall / F1 (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_confidence_distributions(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, class_names: list, out_dir: str):
    maxp = prob.max(axis=1)
    correct = (y_true == y_pred)

    plt.figure(figsize=(8, 5))
    plt.hist(maxp[correct], bins=50, alpha=0.6, density=True, label="correct")
    plt.hist(maxp[~correct], bins=50, alpha=0.6, density=True, label="incorrect")
    plt.xlabel("Max softmax probability (confidence)")
    plt.ylabel("Density")
    plt.title("Confidence distribution (correct vs incorrect)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_confidence_correct_vs_incorrect.png"), dpi=300)
    plt.close()

    true_prob = prob[np.arange(len(y_true)), y_true]
    data, labels = [], []
    for k, name in enumerate(class_names):
        m = (y_true == k)
        if m.sum() == 0:
            continue
        data.append(true_prob[m])
        labels.append(name)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("P(true class)")
    plt.title("True-class probability distribution by class (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_true_class_probability.png"), dpi=300)
    plt.close()

def misclassification_breakdown(y_true: np.ndarray, y_pred: np.ndarray, class_names: list,
                                rid: Optional[np.ndarray], out_dir: str, focus_pairs: str):
    rows = []
    K = len(class_names)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            cnt = int(((y_true == i) & (y_pred == j)).sum())
            if cnt > 0:
                rows.append({"true": class_names[i], "pred": class_names[j], "count": cnt})
    df_pairs = pd.DataFrame(rows).sort_values("count", ascending=False)
    df_pairs.to_csv(os.path.join(out_dir, "misclassification_pairs.csv"), index=False)

    focus = []
    for part in focus_pairs.split(","):
        part = part.strip()
        if "->" in part:
            a, b = part.split("->", 1)
            focus.append((a.strip(), b.strip()))

    focus_rows = []
    for a, b in focus:
        if a in class_names and b in class_names:
            ia = class_names.index(a)
            ib = class_names.index(b)
            cnt = int(((y_true == ia) & (y_pred == ib)).sum())
            focus_rows.append({"true": a, "pred": b, "count": cnt})
    pd.DataFrame(focus_rows).to_csv(os.path.join(out_dir, "misclassification_focus.csv"), index=False)

    if rid is not None and len(rid) == len(y_true):
        mis = (y_true != y_pred)
        df_mis = pd.DataFrame({
            "rid": rid.astype(str),
            "true": [class_names[i] for i in y_true],
            "pred": [class_names[i] for i in y_pred],
        })
        df_mis = df_mis[mis].copy()
        df_mis.to_csv(os.path.join(out_dir, "misclassified_samples.csv"), index=False)

def measure_inference_time(model, device, X: np.ndarray, batch: int = 512) -> Dict[str, float]:
    model.eval()
    xb = torch.from_numpy(X[:min(len(X), batch)]).float().to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(xb)

    n = min(len(X), 5000)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch):
            xbi = torch.from_numpy(X[i:i+batch]).float().to(device)
            _ = model(xbi)
    t1 = time.perf_counter()

    total = max(1, n)
    sec = max(1e-9, (t1 - t0))
    ms_per_window = (sec / total) * 1000.0
    windows_per_sec = total / sec
    return {"timed_windows": int(total), "total_seconds": float(sec), "ms_per_window": float(ms_per_window), "windows_per_second": float(windows_per_sec)}

@torch.no_grad()
def forward_logits(model, device, X: np.ndarray, batch: int = 512) -> np.ndarray:
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i+batch]).float().to(device)
        logits = model(xb).detach().cpu().numpy()
        outs.append(logits)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)

def permutation_importance(model, device, X: np.ndarray, y: np.ndarray,
                           feature_names: list, out_dir: str,
                           subsample: int = 2000, repeats: int = 3, batch: int = 512):
    rng = np.random.default_rng(42)
    n = min(len(X), subsample)
    idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
    X0 = X[idx].copy()
    y0 = y[idx].copy()

    logits0 = forward_logits(model, device, X0, batch=batch)
    prob0 = softmax_np(logits0)
    pred0 = prob0.argmax(axis=1)
    base_macro_f1 = float(f1_score(y0, pred0, average="macro"))

    F = X0.shape[2]
    drops = np.zeros((F,), dtype=np.float64)

    for f in range(F):
        drop_vals = []
        for _ in range(repeats):
            Xp = X0.copy()
            perm = rng.permutation(n)
            Xp[:, :, f] = Xp[perm, :, f]
            logits = forward_logits(model, device, Xp, batch=batch)
            prob = softmax_np(logits)
            pred = prob.argmax(axis=1)
            mf1 = float(f1_score(y0, pred, average="macro"))
            drop_vals.append(base_macro_f1 - mf1)
        drops[f] = float(np.mean(drop_vals))

    df = pd.DataFrame({"feature": feature_names, "macro_f1_drop": drops.astype(float)})
    df = df.sort_values("macro_f1_drop", ascending=False)
    df.to_csv(os.path.join(out_dir, "feature_importance_permutation.csv"), index=False)

    top = df.head(25).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"].values, top["macro_f1_drop"].values)
    plt.title("Top feature importances (permutation, Temporal CNN)")
    plt.xlabel("Macro-F1 drop when permuted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_permutation_top25.png"), dpi=300)
    plt.close()

# ---------------- Main ----------------
def main():
    args = parse_args()
    ensure_dir(args.plotdir)

    with open(os.path.join(args.modeldir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    scaler = np.load(os.path.join(args.modeldir, "scaler.npz"))
    mu = scaler["mu"].astype(np.float32)
    sd = scaler["sd"].astype(np.float32)

    z = np.load(args.data, allow_pickle=True)
    Xte = z["X_test"].astype(np.float32)
    yte = z["y_test"]
    rid = z["rid_test"] if "rid_test" in z.files else None

    feature_names = z["feature_names"].astype(object).tolist() if "feature_names" in z.files else [f"f{i}" for i in range(Xte.shape[-1])]
    fallback_labels = [str(x) for x in z["label_map"].tolist()] if "label_map" in z.files else None

    id2label, label2id = load_label_maps(cfg, fallback_labels=fallback_labels)
    n_classes = int(cfg["n_classes"])
    class_names = [id2label.get(i, str(i)) for i in range(n_classes)]

    # ensure y numeric ids
    if np.asarray(yte).dtype.kind in ("U", "S", "O"):
        yte_i = np.array([label2id[str(v)] for v in np.asarray(yte).tolist()], dtype=np.int64)
    else:
        yte_i = np.asarray(yte, dtype=np.int64)

    Xte_s = apply_scaler(Xte, mu, sd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCNN(
        n_features=int(cfg["n_features"]),
        n_classes=n_classes,
        dropout=float(cfg.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.modeldir, "temporal_cnn.pt"), map_location=device))
    model.eval()

    logits = forward_logits(model, device, Xte_s, batch=args.batch)
    prob = softmax_np(logits)
    y_pred = prob.argmax(axis=1)

    cm = confusion_matrix(yte_i, y_pred, labels=list(range(n_classes)))
    acc = float((y_pred == yte_i).mean())
    macro_f1 = float(f1_score(yte_i, y_pred, average="macro"))
    micro_f1 = float(f1_score(yte_i, y_pred, average="micro"))
    weighted_f1 = float(f1_score(yte_i, y_pred, average="weighted"))

    df_per = metrics_per_class(yte_i, y_pred, class_names)
    df_per.to_csv(os.path.join(args.plotdir, "per_class_metrics.csv"), index=False)

    rep = classification_report(yte_i, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(args.plotdir, "classification_report.csv"))

    plot_confusion(cm, class_names, os.path.join(args.plotdir, "confusion_matrix_raw.png"),
                   normalize=False, title="Confusion Matrix (Test) - Raw Counts")
    plot_confusion(cm, class_names, os.path.join(args.plotdir, "confusion_matrix_normalized.png"),
                   normalize=True, title="Confusion Matrix (Test) - Row-normalized")

    one_vs_rest_pr_curves(
        yte_i, prob, class_names,
        out_png=os.path.join(args.plotdir, "pr_curves_ovr.png"),
        out_csv=os.path.join(args.plotdir, "pr_auc_ovr.csv"),
    )

    plot_bar_metrics(df_per, os.path.join(args.plotdir, "per_class_bars_precision_recall_f1.png"))
    plot_confidence_distributions(yte_i, y_pred, prob, class_names, args.plotdir)

    misclassification_breakdown(yte_i, y_pred, class_names, rid, args.plotdir, args.focus_pairs)

    speed = measure_inference_time(model, device, Xte_s, batch=args.batch)

    window = int(cfg.get("window", 60))
    telemetry_dt = float(cfg.get("telemetry_dt", args.telemetry_dt))
    min_detection_delay_sec = float(window * telemetry_dt)

    if args.importance == "permutation":
        permutation_importance(model, device, Xte_s, yte_i, feature_names, args.plotdir,
                               subsample=args.imp_subsample, repeats=args.perm_repeats, batch=args.batch)

    out = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "labels": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class": df_per.to_dict(orient="records"),
        "inference_speed": speed,
        "window": window,
        "telemetry_dt": telemetry_dt,
        "min_detection_delay_seconds": min_detection_delay_sec,
    }
    with open(os.path.join(args.plotdir, "metrics_tcnn_full.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[test] accuracy=", round(acc, 4),
          "macro_f1=", round(macro_f1, 4),
          "weighted_f1=", round(weighted_f1, 4))
    print("[speed] ms/window=", round(speed["ms_per_window"], 3),
          "windows/sec=", round(speed["windows_per_second"], 2))
    print("[delay] min_detection_delay_seconds=", min_detection_delay_sec)
    print(f"[save] thesis plots + csv + json -> {args.plotdir}")

if __name__ == "__main__":
    main()
