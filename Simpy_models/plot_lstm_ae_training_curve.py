#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

LOG_TEXT = r"""
[epoch 01] train_loss=0.879727 val_recon_mse=0.867906
[epoch 02] train_loss=0.832051 val_recon_mse=0.846401
[epoch 03] train_loss=0.815661 val_recon_mse=0.837596
[epoch 04] train_loss=0.809899 val_recon_mse=0.829190
[epoch 05] train_loss=0.807973 val_recon_mse=0.824678
[epoch 06] train_loss=0.795368 val_recon_mse=0.822213
[epoch 07] train_loss=0.787670 val_recon_mse=0.819436
[epoch 08] train_loss=0.785895 val_recon_mse=0.820221
[epoch 09] train_loss=0.784683 val_recon_mse=0.816244
[epoch 10] train_loss=0.783907 val_recon_mse=0.813527
[epoch 11] train_loss=0.776762 val_recon_mse=0.813490
[epoch 12] train_loss=0.778345 val_recon_mse=0.810465
[epoch 13] train_loss=0.773687 val_recon_mse=0.809981
[epoch 14] train_loss=0.774040 val_recon_mse=0.805347
[epoch 15] train_loss=0.767802 val_recon_mse=0.805094
[epoch 16] train_loss=0.764981 val_recon_mse=0.803149
[epoch 17] train_loss=0.765381 val_recon_mse=0.800658
[epoch 18] train_loss=0.759940 val_recon_mse=0.799559
[epoch 19] train_loss=0.760809 val_recon_mse=0.799075
[epoch 20] train_loss=0.766089 val_recon_mse=0.794823
[epoch 21] train_loss=0.758527 val_recon_mse=0.794885
[epoch 22] train_loss=0.755143 val_recon_mse=0.792422
[epoch 23] train_loss=0.751433 val_recon_mse=0.791495
[epoch 24] train_loss=0.749681 val_recon_mse=0.792455
[epoch 25] train_loss=0.752815 val_recon_mse=0.791168
[epoch 26] train_loss=0.748423 val_recon_mse=0.788868
[epoch 27] train_loss=0.747065 val_recon_mse=0.788433
[epoch 28] train_loss=0.747360 val_recon_mse=0.791710
[epoch 29] train_loss=0.746950 val_recon_mse=0.787047
[epoch 30] train_loss=0.749399 val_recon_mse=0.785211
[epoch 31] train_loss=0.748588 val_recon_mse=0.788508
[epoch 32] train_loss=0.747852 val_recon_mse=0.783305
[epoch 33] train_loss=0.739712 val_recon_mse=0.783222
[epoch 34] train_loss=0.742330 val_recon_mse=0.784643
[epoch 35] train_loss=0.739923 val_recon_mse=0.781573
[epoch 36] train_loss=0.735360 val_recon_mse=0.780527
[epoch 37] train_loss=0.732960 val_recon_mse=0.781599
[epoch 38] train_loss=0.732645 val_recon_mse=0.782377
[epoch 39] train_loss=0.748986 val_recon_mse=0.780958
[epoch 40] train_loss=0.732539 val_recon_mse=0.782106
[epoch 41] train_loss=0.731924 val_recon_mse=0.776417
[epoch 42] train_loss=0.729739 val_recon_mse=0.778651
[epoch 43] train_loss=0.724775 val_recon_mse=0.780737
[epoch 44] train_loss=0.729953 val_recon_mse=0.779342
[epoch 45] train_loss=0.726411 val_recon_mse=0.777296
[epoch 46] train_loss=0.725769 val_recon_mse=0.773756
[epoch 47] train_loss=0.723058 val_recon_mse=0.775693
[epoch 48] train_loss=0.722076 val_recon_mse=0.781185
[epoch 49] train_loss=0.720775 val_recon_mse=0.771037
[epoch 50] train_loss=0.719503 val_recon_mse=0.774330
[epoch 51] train_loss=0.718785 val_recon_mse=0.773182
[epoch 52] train_loss=0.718885 val_recon_mse=0.774323
[epoch 53] train_loss=0.717186 val_recon_mse=0.771192
[epoch 54] train_loss=0.714588 val_recon_mse=0.772905
[epoch 55] train_loss=0.718886 val_recon_mse=0.773269
[epoch 56] train_loss=0.711836 val_recon_mse=0.767939
[epoch 57] train_loss=0.712650 val_recon_mse=0.770123
[epoch 58] train_loss=0.711848 val_recon_mse=0.767465
[epoch 59] train_loss=0.710786 val_recon_mse=0.774071
[epoch 60] train_loss=0.712059 val_recon_mse=0.769024
"""

OUTDIR = "plots_lstm_training_diagnostics_clean"
os.makedirs(OUTDIR, exist_ok=True)

pattern = re.compile(
    r"\[epoch\s+(\d+)\]\s+train_loss=([0-9.]+)\s+val_recon_mse=([0-9.]+)"
)

rows = []
for line in LOG_TEXT.strip().splitlines():
    m = pattern.search(line.strip())
    if m:
        rows.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "val_recon_mse": float(m.group(3)),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "lstm_ae_train_vs_val_curve.csv"), index=False)

best_idx = df["val_recon_mse"].idxmin()
best_epoch = int(df.loc[best_idx, "epoch"])

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], color="tab:blue", marker="o", markersize=3, label="Train loss")
plt.plot(df["epoch"], df["val_recon_mse"], color="tab:orange", marker="o", markersize=3, label="Validation reconstruction MSE")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch = {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("Loss / Reconstruction error")
plt.title("LSTM-AE: Training loss vs Validation reconstruction MSE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "lstm_ae_training_diagnostics_clean.png"), dpi=300)
plt.close()

print(f"[save] clean LSTM-AE plot -> {OUTDIR}/")
print(f"[best] epoch={best_epoch}")