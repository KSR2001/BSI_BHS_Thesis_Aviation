#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

LOG_TEXT = r"""
[epoch 01] train_loss=0.517603 val_acc=0.7361 val_macro_f1=0.5695
[epoch 02] train_loss=0.186388 val_acc=0.8533 val_macro_f1=0.6785
[epoch 03] train_loss=0.118725 val_acc=0.8663 val_macro_f1=0.6977
[epoch 04] train_loss=0.082245 val_acc=0.8462 val_macro_f1=0.6572
[epoch 05] train_loss=0.061393 val_acc=0.8722 val_macro_f1=0.7019
[epoch 06] train_loss=0.043269 val_acc=0.8663 val_macro_f1=0.6812
[epoch 07] train_loss=0.037643 val_acc=0.8604 val_macro_f1=0.6729
[epoch 08] train_loss=0.031073 val_acc=0.8556 val_macro_f1=0.6613
[epoch 09] train_loss=0.026854 val_acc=0.8521 val_macro_f1=0.6656
[epoch 10] train_loss=0.023713 val_acc=0.8497 val_macro_f1=0.6587
[epoch 11] train_loss=0.019890 val_acc=0.8675 val_macro_f1=0.6786
[epoch 12] train_loss=0.020027 val_acc=0.8710 val_macro_f1=0.6852
[epoch 13] train_loss=0.019140 val_acc=0.8710 val_macro_f1=0.6861
"""

OUTDIR = "plots_tcnn_training_diagnostics_clean"
os.makedirs(OUTDIR, exist_ok=True)

pattern = re.compile(
    r"\[epoch\s+(\d+)\]\s+train_loss=([0-9.]+)\s+val_acc=([0-9.]+)\s+val_macro_f1=([0-9.]+)"
)

rows = []
for line in LOG_TEXT.strip().splitlines():
    m = pattern.search(line.strip())
    if m:
        rows.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "val_acc": float(m.group(3)),
            "val_macro_f1": float(m.group(4)),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "temporal_cnn_train_vs_val_curve.csv"), index=False)

best_idx = df["val_macro_f1"].idxmax()
best_epoch = int(df.loc[best_idx, "epoch"])
best_f1 = float(df.loc[best_idx, "val_macro_f1"])

# 1) Train loss only
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], color="tab:blue", marker="o", label="Train loss")
plt.xlabel("Epoch")
plt.ylabel("Train loss")
plt.title("Temporal CNN: Training loss across epochs")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "temporal_cnn_train_loss.png"), dpi=300)
plt.close()

# 2) Validation macro-F1 only
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["val_macro_f1"], color="tab:orange", marker="o", label="Validation macro-F1")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch = {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("Validation macro-F1")
plt.title("Temporal CNN: Validation macro-F1 across epochs")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "temporal_cnn_val_macro_f1.png"), dpi=300)
plt.close()

# 3) Combined clean subplot version
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

axes[0].plot(df["epoch"], df["train_loss"], color="tab:blue", marker="o")
axes[0].set_ylabel("Train loss")
axes[0].set_title("Temporal CNN: Training diagnostics")

axes[1].plot(df["epoch"], df["val_macro_f1"], color="tab:orange", marker="o")
axes[1].axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch = {best_epoch}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation macro-F1")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "temporal_cnn_training_diagnostics_combined.png"), dpi=300)
plt.close()

print(f"[save] clean Temporal CNN plots -> {OUTDIR}/")
print(f"[best] epoch={best_epoch}, val_macro_f1={best_f1:.4f}")