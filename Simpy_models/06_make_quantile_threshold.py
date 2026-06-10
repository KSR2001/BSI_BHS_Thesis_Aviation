#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn

CANON = ["normal", "dos", "spoof", "fdi", "stopped_conv"]
NORMAL_ID = 0

class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int, latent: int, num_layers: int = 1, dropout: float = 0.0):
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
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]
        z = self.to_latent(h_last)
        h0 = torch.tanh(self.from_latent(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        _, T, _ = x.shape
        dec_in = h0.transpose(0, 1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        return self.out(dec_out)

def standardize_apply(X, mu, sd):
    return (X - mu[None, None, :]) / (sd[None, None, :] + 1e-6)

def recon_error(model, X, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = torch.from_numpy(X[i:i+512]).to(device)
            yb = model(xb)
            e = (yb - xb).pow(2).mean(dim=(1,2)).detach().cpu().numpy()
            out.append(e)
    return np.concatenate(out, axis=0) if out else np.array([], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lstm_windows.npz")
    ap.add_argument("--modeldir", default="models_lstm")
    ap.add_argument("--q", type=float, default=0.999)
    ap.add_argument("--outname", default=None)
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    X_val_n = X_val[y_val == NORMAL_ID]
    if X_val_n.shape[0] < 10:
        raise ValueError("Too few normal windows in validation to compute quantile threshold.")

    scaler = np.load(os.path.join(a.modeldir, "scaler.npz"))
    mu, sd = scaler["mu"].astype(np.float32), scaler["sd"].astype(np.float32)

    with open(os.path.join(a.modeldir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAE(
        n_features=int(cfg["n_features"]),
        hidden=int(cfg["hidden"]),
        latent=int(cfg["latent"]),
        num_layers=int(cfg["layers"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(a.modeldir, "lstm_ae.pt"), map_location=device))
    model.eval()

    Xs = standardize_apply(X_val_n, mu, sd)
    scores = recon_error(model, Xs, device)
    thr = float(np.quantile(scores, a.q))

    out = {
        "recon_mse_threshold": thr,
        "mode": "quantile_baseline",
        "quantile": float(a.q),
        "n_val_normals": int(scores.shape[0]),
        "note": "Threshold computed from validation NORMAL reconstruction errors (deployment-like baseline)."
    }

    if a.outname is None:
        tag = str(a.q).replace(".", "")
        a.outname = f"thresholds_quantile_q{tag}.json"

    outpath = os.path.join(a.modeldir, a.outname)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[save] {outpath}")
    print(f"threshold={thr:.6f} from q={a.q} (n={scores.shape[0]})")

if __name__ == "__main__":
    main()