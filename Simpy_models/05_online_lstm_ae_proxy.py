#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import time
import numpy as np
import torch
import torch.nn as nn


# ---------------- Model ----------------

class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 32,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        enc_out, _ = self.encoder(x)          # [B,T,H]
        h_last = enc_out[:, -1, :]            # [B,H]
        z = self.to_latent(h_last)            # [B,Z]
        h0 = torch.tanh(self.from_latent(z)).unsqueeze(0)  # [1,B,H]
        c0 = torch.zeros_like(h0)

        _, T, _ = x.shape
        dec_in = h0.transpose(0, 1).repeat(1, T, 1)        # [B,T,H]
        dec_out, _ = self.decoder(dec_in, (h0, c0))        # [B,T,H]
        y = self.out(dec_out)                               # [B,T,F]
        return y


# ---------------- Utils ----------------

def standardize_apply(X, mu, sd):
    return (X - mu[None, None, :]) / (sd[None, None, :] + 1e-6)

def safe_send(sock, obj: dict):
    if not sock:
        return
    try:
        sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))
    except Exception:
        pass

def parse_component_from_feature(feat: str) -> str:
    if not feat:
        return "UNKNOWN"
    if "__" in feat:
        return feat.split("__", 1)[0]
    return feat


# ---------------- Proxy ----------------

class Proxy:
    def __init__(self, args):
        self.args = args

        with open(f"{args.modeldir}/config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(f"{args.modeldir}/thresholds.json", "r", encoding="utf-8") as f:
            thr = json.load(f)

        scaler = np.load(f"{args.modeldir}/scaler.npz")
        self.mu = scaler["mu"].astype(np.float32)
        self.sd = scaler["sd"].astype(np.float32)

        self.base_threshold = float(thr["recon_mse_threshold"])
        self.online_threshold = float(self.base_threshold * float(self.args.online_thr_mult))

        self.feature_names = cfg.get("feature_names", None)
        if not self.feature_names:
            raise ValueError("config.json missing feature_names.")

        self.f_index = {str(name): i for i, name in enumerate(self.feature_names)}
        self.F = int(cfg["n_features"])

        # Prefer config values if not provided
        if args.window <= 0:
            self.args.window = int(cfg.get("window", 60))
        if args.telemetry_dt <= 0 and "telemetry_dt" in cfg:
            self.args.telemetry_dt = float(cfg["telemetry_dt"])

        # log1p flag (match training)
        cfg_log1p = cfg.get("log1p_counts", None)
        if cfg_log1p is None:
            self.log1p_counts = bool(args.log1p_counts)
        else:
            self.log1p_counts = bool(int(cfg_log1p)) if isinstance(cfg_log1p, (int, str)) else bool(cfg_log1p)

        # indices
        self.q_idxs = [idx for name, idx in self.f_index.items() if name.endswith("__queue_length")]
        self.b_idxs = [idx for name, idx in self.f_index.items() if name.endswith("__busy")]
        self.telemetry_idxs = sorted(set(self.q_idxs + self.b_idxs))

        self.idx_total_queue = self.f_index.get("TOTAL_QUEUE", None)
        self.idx_total_busy  = self.f_index.get("TOTAL_BUSY", None)
        self.idx_throughput  = self.f_index.get("THROUGHPUT", None)

        # log1p indices (count-like)
        log1p_names = []
        for name in self.f_index.keys():
            if (
                name.endswith("__queue_length") or
                name.endswith("__exit_count") or
                name.endswith("__decision_count") or
                name.endswith("__mismatch_count") or
                name.endswith("__hit_count") or
                name == "TOTAL_QUEUE"
            ):
                log1p_names.append(name)
        self.log1p_idx = np.array([self.f_index[n] for n in log1p_names], dtype=np.int64)

        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMAE(
            n_features=self.F,
            hidden=int(cfg["hidden"]),
            latent=int(cfg["latent"]),
            num_layers=int(cfg["layers"]),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(torch.load(f"{args.modeldir}/lstm_ae.pt", map_location=self.device))
        self.model.eval()

        # sockets
        self.sim_sock = None
        self.unity_sock = None

        # run tracking
        self.last_sim_ts = None
        self.last_run_id = None

        # calibration
        self.calib_scores = []
        self.calib_threshold = None
        self.calib_done = False

        self.reset_state(keep_calibration=False)

    # -------- state reset --------

    def reset_state(self, keep_calibration: bool = True):
        self.buffer = {}
        self.latest_tbin = -1
        self.tele_cnt = {}

        # forward-fill memory for telemetry signals
        self.last_telemetry_vec = np.zeros((self.F,), dtype=np.float32)

        # scoring once per completed bin
        self.last_seen_tbin = -1
        self.last_scored_tbin = -1

        # alert gating
        self.last_alert_wall = 0.0
        self.above_count = 0

        if not keep_calibration:
            self.calib_scores = []
            self.calib_threshold = None
            self.calib_done = False

    # -------- thresholds --------

    def active_threshold(self) -> float:
        """
        Final threshold used for alerting:
          - During warmup/calibration: no alerts anyway (see should_alert)
          - After calibration: used calibrated threshold if available, else online threshold.
          - Optional floor: base_threshold * min_thr_mult.
        """
        thr = self.online_threshold
        if self.calib_threshold is not None:
            thr = float(self.calib_threshold)

        if self.args.min_thr_mult > 0:
            floor = float(self.base_threshold * self.args.min_thr_mult)
            thr = float(max(thr, floor))

        return float(thr)

    def in_calibration_phase(self, sim_ts: float) -> bool:
        # warmup handled separately; this checks the post-warmup calibration window
        return (self.args.calib_sec > 0) and (sim_ts < (self.args.warmup_sec + self.args.calib_sec))

    # -------- network --------

    def connect_upstream_retry(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.args.upstream_host, self.args.upstream_port))
                s.settimeout(1.0)
                self.sim_sock = s
                print(f"[proxy] connected upstream {self.args.upstream_host}:{self.args.upstream_port}")
                return
            except Exception as e:
                print(f"[proxy] upstream not ready ({e}). retrying in 1s...")
                time.sleep(1.0)

    def wait_for_client(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.args.listen_host, self.args.listen_port))
        srv.listen(1)
        print(f"[proxy] listening for Unity on {self.args.listen_host}:{self.args.listen_port}")
        conn, addr = srv.accept()
        conn.settimeout(1.0)
        self.unity_sock = conn
        print(f"[proxy] Unity connected from {addr}")

    # -------- feature buffering --------

    def _ensure_row(self, tbin: int):
        if tbin not in self.buffer:
            row = np.zeros((self.F,), dtype=np.float32)
            # initialize telemetry features with last seen values (forward-fill)
            if self.telemetry_idxs:
                row[self.telemetry_idxs] = self.last_telemetry_vec[self.telemetry_idxs]
            self.buffer[tbin] = row

    def _tbin(self, ts: float) -> int:
        return int(float(ts) / float(self.args.telemetry_dt))

    def _tele_update_mean(self, tbin: int, idx: int, value: float):
        # online mean aggregation within a time bin (matches offline mean)
        if tbin not in self.tele_cnt:
            self.tele_cnt[tbin] = {}
        c = self.tele_cnt[tbin].get(idx, 0)
        prev = float(self.buffer[tbin][idx])
        new = (prev * c + float(value)) / float(c + 1)
        self.buffer[tbin][idx] = float(new)
        self.tele_cnt[tbin][idx] = c + 1
        self.last_telemetry_vec[idx] = float(new)

    def update_from_msg(self, msg: dict):
        evt = msg.get("event", "")
        ts = float(msg.get("timestamp", 0.0))
        comp = str(msg.get("component", ""))

        tbin = self._tbin(ts)
        if tbin > self.latest_tbin:
            self.latest_tbin = tbin
        self._ensure_row(tbin)
        row = self.buffer[tbin]

        if evt == "telemetry":
            q = float(msg.get("queue_length", 0.0) or 0.0)
            b = float(msg.get("busy", 0.0) or 0.0)
            kq = f"{comp}__queue_length"
            kb = f"{comp}__busy"
            if kq in self.f_index:
                self._tele_update_mean(tbin, self.f_index[kq], q)
            if kb in self.f_index:
                self._tele_update_mean(tbin, self.f_index[kb], b)

        elif evt == "conveyor_exit":
            tt = float(msg.get("transit_time", 0.0) or 0.0)
            kc = f"{comp}__exit_count"
            km = f"{comp}__mean_transit_time"
            if kc in self.f_index:
                idxc = self.f_index[kc]
                idxm = self.f_index.get(km, None)
                prev_cnt = float(row[idxc])
                new_cnt = prev_cnt + 1.0
                row[idxc] = float(new_cnt)
                if idxm is not None:
                    prev_mean = float(row[idxm])
                    row[idxm] = float((prev_mean * prev_cnt + tt) / max(1e-6, new_cnt))

        elif evt == "diverter_decision":
            exp = str(msg.get("expected_branch", ""))
            ch  = str(msg.get("chosen_branch", ""))
            mismatch = 1.0 if (exp != ch and exp and ch) else 0.0
            kd = f"{comp}__decision_count"
            km = f"{comp}__mismatch_count"
            if kd in self.f_index:
                row[self.f_index[kd]] += 1.0
            if km in self.f_index:
                row[self.f_index[km]] += mismatch

        elif evt == "sensor_trigger":
            ks = f"{comp}__hit_count"
            if ks in self.f_index:
                row[self.f_index[ks]] += 1.0

        elif evt == "throughput":
            v = float(msg.get("throughput", 0.0) or 0.0)
            if self.idx_throughput is not None:
                row[self.idx_throughput] = float(v)

        # bound memory
        min_keep = self.latest_tbin - (self.args.window + 10)
        for k in list(self.buffer.keys()):
            if k < min_keep:
                del self.buffer[k]
                if k in self.tele_cnt:
                    del self.tele_cnt[k]

    # -------- scoring --------

    def score_window_ending_at(self, end_tbin: int):
        if end_tbin < (self.args.window - 1):
            return None

        start = end_tbin - self.args.window + 1
        X = np.zeros((self.args.window, self.F), dtype=np.float32)

        for i, tb in enumerate(range(start, end_tbin + 1)):
            if tb in self.buffer:
                X[i] = self.buffer[tb]

        # derived totals
        if self.idx_total_queue is not None and self.q_idxs:
            X[:, self.idx_total_queue] = X[:, self.q_idxs].sum(axis=1)
        if self.idx_total_busy is not None and self.b_idxs:
            X[:, self.idx_total_busy] = X[:, self.b_idxs].sum(axis=1)

        # throughput forward-fill inside window
        if self.idx_throughput is not None:
            last = 0.0
            for i in range(X.shape[0]):
                if X[i, self.idx_throughput] != 0.0:
                    last = float(X[i, self.idx_throughput])
                else:
                    X[i, self.idx_throughput] = float(last)

        # log1p counts if training used it
        if self.log1p_counts and self.log1p_idx.size > 0:
            X[:, self.log1p_idx] = np.log1p(np.maximum(X[:, self.log1p_idx], 0.0))

        # scale
        Xs = standardize_apply(X[None, :, :], self.mu, self.sd)
        xb = torch.from_numpy(Xs).to(self.device)

        with torch.no_grad():
            yb = self.model(xb)
            err = (yb - xb).pow(2)  # [1,T,F]
            score = float(err.mean().item())
            per_feat = err.mean(dim=(0, 1)).detach().cpu().numpy()
        return score, per_feat

    def localize(self, per_feat: np.ndarray):
        if per_feat is None or per_feat.size == 0:
            return "UNKNOWN", "UNKNOWN"
        idx = int(np.argmax(per_feat))
        feat = self.feature_names[idx] if idx < len(self.feature_names) else "UNKNOWN"
        comp = parse_component_from_feature(str(feat))
        return str(feat), comp

    # -------- calibration --------

    def maybe_calibrate(self, sim_ts: float, score: float):
        if self.args.calib_sec <= 0 or self.calib_done:
            return

        t0 = self.args.warmup_sec
        t1 = self.args.warmup_sec + self.args.calib_sec

        if sim_ts < t0:
            return

        if sim_ts <= t1:
            self.calib_scores.append(float(score))
            return

        # finalize once after calibration window ends
        if len(self.calib_scores) >= int(self.args.calib_min_n):
            q = float(self.args.calib_quantile)
            self.calib_threshold = float(np.quantile(np.array(self.calib_scores, dtype=np.float32), q))
            self.calib_done = True
            print(f"[proxy] calibrated threshold={self.calib_threshold:.6f} (q={q}, n={len(self.calib_scores)})")
        else:
            self.calib_done = True
            print(f"[proxy] calibration skipped (n={len(self.calib_scores)} < min_n={self.args.calib_min_n})")

    # -------- alert gating --------

    def should_alert(self, sim_ts: float, score: float) -> bool:
        # warmup: ignore everything
        if sim_ts < self.args.warmup_sec:
            self.above_count = 0
            return False

        
        if self.in_calibration_phase(sim_ts):
            self.above_count = 0
            return False

        thr = self.active_threshold()

        if score > thr:
            self.above_count += 1
        else:
            self.above_count = 0

        if self.above_count < self.args.require_consecutive:
            return False

        # cooldown in wall-clock seconds
        now_wall = time.time()
        if now_wall - self.last_alert_wall < self.args.cooldown_sec:
            return False

        self.last_alert_wall = now_wall
        return True

    # -------- main loop --------

    def run(self):
        self.connect_upstream_retry()
        self.wait_for_client()

        print(f"[proxy] window={self.args.window} dt={self.args.telemetry_dt}")
        print(f"[proxy] log1p_counts={self.log1p_counts}")
        print(f"[proxy] threshold base={self.base_threshold:.6f} online={self.online_threshold:.6f} "
              f"(mult={self.args.online_thr_mult})")
        print(f"[proxy] warmup_sec={self.args.warmup_sec} require_consecutive={self.args.require_consecutive} "
              f"cooldown_sec={self.args.cooldown_sec}")
        print(f"[proxy] calib_sec={self.args.calib_sec} calib_quantile={self.args.calib_quantile} "
              f"calib_min_n={self.args.calib_min_n} min_thr_mult={self.args.min_thr_mult} "
              f"reset_on_new_run={self.args.reset_on_new_run}")

        buf = b""
        while True:
            try:
                chunk = self.sim_sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                continue
            except Exception:
                break

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", errors="ignore").strip()
                if not s:
                    continue
                try:
                    msg = json.loads(s)
                except json.JSONDecodeError:
                    continue

                # forward raw msg to Unity
                msg.setdefault("msg_type", "event")
                safe_send(self.unity_sock, msg)

                sim_ts_msg = float(msg.get("timestamp", 0.0))
                run_id = msg.get("run_id", None)

                # reset on new run_id / backwards jump
                if self.args.reset_on_new_run:
                    if self.last_run_id is None and run_id is not None:
                        self.last_run_id = run_id

                    if run_id is not None and self.last_run_id is not None and run_id != self.last_run_id:
                        print(f"[proxy] run_id changed {self.last_run_id} -> {run_id}. resetting state.")
                        self.last_run_id = run_id
                        self.last_sim_ts = None
                        self.reset_state(keep_calibration=False)

                    if self.last_sim_ts is not None and sim_ts_msg + 5.0 < self.last_sim_ts:
                        print(f"[proxy] timestamp jumped backwards {self.last_sim_ts:.2f} -> {sim_ts_msg:.2f}. resetting state.")
                        self.last_sim_ts = None
                        self.reset_state(keep_calibration=False)

                self.last_sim_ts = sim_ts_msg

                # update features
                self.update_from_msg(msg)

                # score once per completed time-bin
                tbin = self._tbin(sim_ts_msg)
                if self.last_seen_tbin < 0:
                    self.last_seen_tbin = tbin
                if tbin <= self.last_seen_tbin:
                    continue

                completed_end = self.last_seen_tbin
                self.last_seen_tbin = tbin

                if completed_end <= self.last_scored_tbin:
                    continue

                scored = self.score_window_ending_at(completed_end)
                self.last_scored_tbin = completed_end
                if scored is None:
                    continue

                score, per_feat = scored
                sim_ts = float(completed_end * self.args.telemetry_dt)

                # calibration updates (finalizes after warmup+calib_sec)
                self.maybe_calibrate(sim_ts, score)

                if self.should_alert(sim_ts, score):
                    top_feat, affected_comp = self.localize(per_feat)
                    thr = self.active_threshold()
                    alert = {
                        "msg_type": "alert",
                        "timestamp": sim_ts,
                        "attack_type": "anomaly",
                        "confidence": float(min(0.999, score / max(1e-9, thr))),
                        "top_feature": top_feat,
                        "affected_component": affected_comp,
                        "description": f"LSTM-AE score={score:.6f} thr={thr:.6f} "
                                       f"(base={self.base_threshold:.6f}, online={self.online_threshold:.6f})",
                    }
                    print("[ALERT]", alert)
                    safe_send(self.unity_sock, alert)


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeldir", default="models_lstm")

    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8765)

    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=9001)

    ap.add_argument("--telemetry-dt", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=60)

    ap.add_argument("--warmup-sec", type=float, default=120.0)
    ap.add_argument("--require-consecutive", type=int, default=3)
    ap.add_argument("--cooldown-sec", type=float, default=3.0)
    ap.add_argument("--online-thr-mult", type=float, default=1.05)

    ap.add_argument("--log1p-counts", action="store_true")

    # calibration + reset
    ap.add_argument("--calib-sec", type=float, default=300.0,
                    help="Collect scores after warmup for this many seconds and set calibrated threshold.")
    ap.add_argument("--calib-quantile", type=float, default=0.999,
                    help="Quantile for calibrated threshold (e.g., 0.99–0.999).")
    ap.add_argument("--calib-min-n", type=int, default=60,
                    help="Minimum number of samples required to finalize calibration.")
    ap.add_argument("--reset-on-new-run", action="store_true",
                    help="Reset proxy buffers when run_id changes or timestamps jump backwards.")

    # optional floor to prevent too-low calibrated threshold
    ap.add_argument("--min-thr-mult", type=float, default=0.0,
                    help="Optional floor: base_threshold * min_thr_mult. Set 0 to disable.")

    return ap.parse_args()

def main():
    args = parse_args()

    # keep log1p ON by default (match your training)
    if not args.log1p_counts:
        args.log1p_counts = True

    # reset protection ON by default
    if not args.reset_on_new_run:
        args.reset_on_new_run = True

    Proxy(args).run()

if __name__ == "__main__":
    main()
