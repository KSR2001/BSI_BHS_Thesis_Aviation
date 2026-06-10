#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, socket, time
import numpy as np
import torch
import torch.nn as nn


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
        x = x.transpose(1, 2)   # (B,T,F)->(B,F,T)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.head(h)


# ---------------- Utils ----------------
def safe_send(sock, obj: dict):
    if not sock:
        return
    try:
        sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))
    except Exception:
        pass


def standardize_apply(X, mu, sd):
    return (X - mu[None, None, :]) / (sd[None, None, :] + 1e-6)


def load_id2label(cfg, n_classes: int):
    raw = cfg.get("id2label", None)
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {int(i): str(v) for i, v in enumerate(raw)}
    lm = cfg.get("label_map", None)
    if isinstance(lm, list) and len(lm) == n_classes:
        return {i: str(lm[i]) for i in range(n_classes)}
    return {i: str(i) for i in range(n_classes)}


# ---------------- Args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeldir", default="models_tcnn")

    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8765)

    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=9002)

    ap.add_argument("--telemetry-dt", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)

    ap.add_argument("--warmup-sec", type=float, default=120.0)
    ap.add_argument("--thr-conf", type=float, default=0.75)
    ap.add_argument("--require-consecutive", type=int, default=3)
    ap.add_argument("--cooldown-sec", type=float, default=2.0)

    ap.add_argument("--log1p-counts", action="store_true",
                    help="Apply log1p to count-like channels before scaling (MUST match training).")

    ap.add_argument("--reset-on-new-run", action="store_true")

    ap.add_argument("--send-predictions", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # Heuristic to fix dos vs stopped_conv online confusion
    ap.add_argument("--heuristic-stopped", action="store_true",
                    help="If model predicts dos but looks like conveyor stop, relabel to stopped_conv.")

    # Stop signature knobs
    ap.add_argument("--stop-throughput-eps", type=float, default=1e-6,
                    help="Throughput <= eps counts as ~0.")
    ap.add_argument("--stop-queue-min", type=float, default=1.0,
                    help="Minimum TOTAL_QUEUE to consider a stop meaningful.")
    ap.add_argument("--stop-zero-frac", type=float, default=0.8,
                    help="Fraction of last K seconds where throughput is ~0 to call stopped_conv.")
    ap.add_argument("--stop-lookback", type=int, default=15,
                    help="K seconds lookback for stop heuristic.")

    # NEW: require exit activity to be near zero (prevents DoS being relabeled as stop)
    ap.add_argument("--stop-exits-max", type=float, default=0.5,
                    help="Max TOTAL_EXITS per second (avg over last K) to consider 'stopped'. "
                         "If you still see some exits during stop, raise slightly; "
                         "if DoS is misclassified as stop, lower this.")

    return ap.parse_args()


# ---------------- Proxy ----------------
class Proxy:
    def __init__(self, args):
        self.args = args

        with open(f"{args.modeldir}/config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        scaler = np.load(f"{args.modeldir}/scaler.npz")
        self.mu = scaler["mu"].astype(np.float32)
        self.sd = scaler["sd"].astype(np.float32)

        self.feature_names = cfg.get("feature_names")
        if not self.feature_names:
            raise ValueError("config.json missing feature_names.")

        self.f_index = {str(name): i for i, name in enumerate(self.feature_names)}
        self.F = int(cfg["n_features"])
        self.n_classes = int(cfg["n_classes"])
        self.args.window = int(cfg.get("window", self.args.window))

        # prefer training settings if present
        if cfg.get("stride") is not None:
            self.args.stride = int(cfg["stride"])
        if not args.log1p_counts:
            self.args.log1p_counts = bool(int(cfg.get("log1p_counts", 0)))

        self.id2label = load_id2label(cfg, self.n_classes)

        if self.args.verbose:
            print("[proxy] id2label:", self.id2label)
            print("[proxy] log1p_counts:", self.args.log1p_counts,
                  "| stride:", self.args.stride, "| window:", self.args.window)

        # indices
        self.q_idxs = [idx for name, idx in self.f_index.items() if name.endswith("__queue_length")]
        self.b_idxs = [idx for name, idx in self.f_index.items() if name.endswith("__busy")]
        self.exit_idxs = [idx for name, idx in self.f_index.items() if name.endswith("__exit_count")]

        self.idx_total_queue = self.f_index.get("TOTAL_QUEUE", None)
        self.idx_total_busy  = self.f_index.get("TOTAL_BUSY", None)
        self.idx_throughput  = self.f_index.get("THROUGHPUT", None)

        # Optional TOTAL_EXITS if you stored it in training
        self.idx_total_exits = self.f_index.get("TOTAL_EXITS", None)

        # log1p targets
        self.log1p_idxs = [
            idx for name, idx in self.f_index.items()
            if name.endswith("__queue_length")
            or name.endswith("__exit_count")
            or name.endswith("__decision_count")
            or name.endswith("__mismatch_count")
            or name.endswith("__hit_count")
            or name == "TOTAL_QUEUE"
            or name == "TOTAL_EXITS"
        ]
        self.log1p_idxs = np.array(sorted(set(self.log1p_idxs)), dtype=np.int64)

        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TemporalCNN(self.F, self.n_classes, dropout=float(cfg.get("dropout", 0.25))).to(self.device)
        self.model.load_state_dict(torch.load(f"{args.modeldir}/temporal_cnn.pt", map_location=self.device))
        self.model.eval()

        # rolling buffer
        self.buffer = {}          # tbin -> row[F]
        self.tele_cnt = {}        # tbin -> {feat_idx: count}
        self.latest_tbin = -1
        self.last_pred_tbin = -999999
        self.last_telemetry_vec = np.zeros((self.F,), dtype=np.float32)

        # run tracking
        self.last_sim_ts = None
        self.last_run_id = None

        # gating
        self.last_alert_wall = 0.0
        self.above_count = 0
        self.pending_label = None

        # sockets
        self.sim_sock = None
        self.unity_sock = None
        self.server_sock = None

    # ----- networking -----
    def connect_upstream(self):
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

    def start_server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.args.listen_host, self.args.listen_port))
        srv.listen(1)
        self.server_sock = srv
        print(f"[proxy] listening for Unity on {self.args.listen_host}:{self.args.listen_port}")

    def accept_unity(self):
        conn, addr = self.server_sock.accept()
        conn.settimeout(1.0)
        self.unity_sock = conn
        print(f"[proxy] Unity connected from {addr}")

    def unity_send_line(self, raw_line: str):
        if not self.unity_sock:
            return
        try:
            self.unity_sock.sendall((raw_line.rstrip("\n") + "\n").encode("utf-8"))
        except Exception:
            try:
                self.unity_sock.close()
            except Exception:
                pass
            self.unity_sock = None
            print("[proxy] Unity disconnected. Waiting for reconnect...")

    # ----- time bins -----
    def _tbin(self, ts: float) -> int:
        return int(float(ts) / float(self.args.telemetry_dt))

    def reset_state(self):
        self.buffer = {}
        self.tele_cnt = {}
        self.latest_tbin = -1
        self.last_pred_tbin = -999999
        self.last_telemetry_vec[:] = 0.0
        self.last_alert_wall = 0.0
        self.above_count = 0
        self.pending_label = None

    def _ensure_row(self, tbin: int):
        if tbin in self.buffer:
            return
        row = np.zeros((self.F,), dtype=np.float32)
        # forward-fill telemetry into new bin
        if self.q_idxs:
            row[self.q_idxs] = self.last_telemetry_vec[self.q_idxs]
        if self.b_idxs:
            row[self.b_idxs] = self.last_telemetry_vec[self.b_idxs]
        self.buffer[tbin] = row

    def _ensure_gap_rows(self, new_tbin: int):
        if self.latest_tbin < 0:
            self._ensure_row(new_tbin)
            return
        for tb in range(self.latest_tbin + 1, new_tbin + 1):
            self._ensure_row(tb)

    def _tele_update_mean(self, tbin: int, idx: int, value: float):
        if tbin not in self.tele_cnt:
            self.tele_cnt[tbin] = {}
        c = self.tele_cnt[tbin].get(idx, 0)
        prev = float(self.buffer[tbin][idx])
        new = (prev * c + float(value)) / float(c + 1)
        self.buffer[tbin][idx] = float(new)
        self.tele_cnt[tbin][idx] = c + 1
        self.last_telemetry_vec[idx] = float(new)

    def update_from_msg(self, msg: dict):
        evt = msg.get("event", "") or msg.get("event_name", "")
        ts = float(msg.get("timestamp", 0.0))
        run_id = msg.get("run_id", None)

        if ts <= 0.0:
            return

        if self.args.reset_on_new_run:
            if self.last_run_id is None and run_id is not None:
                self.last_run_id = run_id
            if run_id is not None and self.last_run_id is not None and run_id != self.last_run_id:
                print(f"[proxy] run_id changed {self.last_run_id} -> {run_id}. resetting state.")
                self.last_run_id = run_id
                self.last_sim_ts = None
                self.reset_state()
            if self.last_sim_ts is not None and ts + 5.0 < self.last_sim_ts:
                print(f"[proxy] timestamp jumped backwards {self.last_sim_ts:.2f} -> {ts:.2f}. resetting state.")
                self.last_sim_ts = None
                self.reset_state()

        self.last_sim_ts = ts

        tbin = self._tbin(ts)
        self._ensure_gap_rows(tbin)
        self.latest_tbin = max(self.latest_tbin, tbin)
        row = self.buffer[tbin]

        comp = str(msg.get("component", ""))

        # telemetry
        if evt == "telemetry":
            q = float(msg.get("queue_length", 0.0) or 0.0)
            b = float(msg.get("busy", 0.0) or 0.0)
            kq = f"{comp}__queue_length"
            kb = f"{comp}__busy"
            if kq in self.f_index:
                self._tele_update_mean(tbin, self.f_index[kq], q)
            if kb in self.f_index:
                self._tele_update_mean(tbin, self.f_index[kb], b)
            return

        # per-bin counters
        if evt == "conveyor_exit":
            kc = f"{comp}__exit_count"
            km = f"{comp}__mean_transit_time"
            tt = float(msg.get("transit_time", 0.0) or 0.0)
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
            ch = str(msg.get("chosen_branch", ""))
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
        min_keep = self.latest_tbin - (self.args.window + 20)
        for k in list(self.buffer.keys()):
            if k < min_keep:
                del self.buffer[k]
                if k in self.tele_cnt:
                    del self.tele_cnt[k]

    def build_latest_window(self):
        if self.latest_tbin < (self.args.window - 1):
            return None
        if (self.latest_tbin - self.last_pred_tbin) < self.args.stride:
            return None
        self.last_pred_tbin = self.latest_tbin

        end = self.latest_tbin
        start = end - self.args.window + 1

        X = np.zeros((self.args.window, self.F), dtype=np.float32)
        for i, tb in enumerate(range(start, end + 1)):
            if tb in self.buffer:
                X[i] = self.buffer[tb]

        # derived totals
        if self.idx_total_queue is not None and self.q_idxs:
            X[:, self.idx_total_queue] = X[:, self.q_idxs].sum(axis=1)
        if self.idx_total_busy is not None and self.b_idxs:
            X[:, self.idx_total_busy] = X[:, self.b_idxs].sum(axis=1)

        # derived TOTAL_EXITS (per-second count) if present in features
        if self.idx_total_exits is not None and self.exit_idxs:
            X[:, self.idx_total_exits] = X[:, self.exit_idxs].sum(axis=1)

        # keep RAW throughput BEFORE forward-fill for heuristic
        thr_raw = None
        if self.idx_throughput is not None:
            thr_raw = X[:, self.idx_throughput].copy()

        # keep RAW exits (even if TOTAL_EXITS not in features)
        exits_raw = None
        if self.exit_idxs:
            exits_raw = X[:, self.exit_idxs].sum(axis=1).copy()
        elif self.idx_total_exits is not None:
            exits_raw = X[:, self.idx_total_exits].copy()

        # forward-fill throughput for the MODEL window
        if self.idx_throughput is not None:
            last = 0.0
            for i in range(X.shape[0]):
                if X[i, self.idx_throughput] != 0.0:
                    last = float(X[i, self.idx_throughput])
                else:
                    X[i, self.idx_throughput] = float(last)

        # log1p
        if self.args.log1p_counts and self.log1p_idxs.size > 0:
            X[:, self.log1p_idxs] = np.log1p(np.maximum(X[:, self.log1p_idxs], 0.0))

        if self.args.verbose:
            nz = np.count_nonzero(X)
            print(f"[dbg] tbin={self.latest_tbin} nz_ratio={nz / max(1, X.size):.3f}")

        Xs = standardize_apply(X[None, :, :], self.mu, self.sd)
        return Xs, X, thr_raw, exits_raw

    @torch.no_grad()
    def predict(self, Xs):
        xb = torch.from_numpy(Xs).float().to(self.device)
        logits = self.model(xb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return pred_id, float(probs[pred_id]), probs

    def apply_heuristics(self, pred_label: str, raw_window_model: np.ndarray,
                        thr_raw: np.ndarray, exits_raw: np.ndarray) -> str:
        """
        Online-only relabeling. Uses RAW throughput and RAW exit activity.
        Goal: relabel ONLY true conveyor stops, not DoS slowdowns.
        """
        if not self.args.heuristic_stopped:
            return pred_label

        # only fix known confusion: dos -> stopped_conv
        if pred_label != "dos":
            return pred_label

        if thr_raw is None or self.idx_total_queue is None:
            return pred_label

        K = max(1, int(self.args.stop_lookback))
        K = min(K, len(thr_raw))
        thr_tail = thr_raw[-K:]
        zero_frac = float((thr_tail <= self.args.stop-throughput-eps if False else 0.0))

        # (the line above is a trap in case you copy/paste wrong; use the correct one below)
        zero_frac = float((thr_tail <= self.args.stop_throughput_eps).mean())

        tq = float(raw_window_model[-1, self.idx_total_queue])

        # NEW: exits must be ~0 on average in last K seconds to call "stopped"
        exits_ok = True
        exits_avg = float("nan")
        if exits_raw is not None:
            K2 = min(K, len(exits_raw))
            exits_tail = exits_raw[-K2:]
            exits_avg = float(exits_tail.mean())  # exits per second (per bin)
            exits_ok = (exits_avg <= float(self.args.stop_exits_max))

        if self.args.verbose:
            print(f"[dbg-stop] thr_zero_frac(last {K}s)={zero_frac:.2f} "
                  f"total_queue={tq:.2f} exits_avg(last {min(K, len(exits_raw)) if exits_raw is not None else K}s)={exits_avg}")

        # Stop signature:
        # - throughput ~0 most of last K seconds
        # - queue exists
        # - exit activity ~0 (prevents DoS being relabeled)
        if (zero_frac >= float(self.args.stop_zero_frac)) and (tq >= self.args.stop_queue_min) and exits_ok:
            return "stopped_conv"

        return pred_label

    def should_alert(self, sim_ts: float, label: str, conf: float) -> bool:
        if sim_ts < self.args.warmup_sec:
            self.above_count = 0
            self.pending_label = None
            return False

        if label == "normal" or conf < self.args.thr_conf:
            self.above_count = 0
            self.pending_label = None
            return False

        if self.pending_label != label:
            self.pending_label = label
            self.above_count = 1
        else:
            self.above_count += 1

        if self.above_count < self.args.require_consecutive:
            return False

        now = time.time()
        if now - self.last_alert_wall < self.args.cooldown_sec:
            return False

        self.last_alert_wall = now
        return True

    def run(self):
        self.start_server()
        self.connect_upstream()
        self.accept_unity()

        print(f"[proxy] window={self.args.window} stride={self.args.stride} dt={self.args.telemetry_dt}")
        print(f"[proxy] warmup_sec={self.args.warmup_sec} thr_conf={self.args.thr_conf} "
              f"require_consecutive={self.args.require_consecutive} cooldown_sec={self.args.cooldown_sec}")
        print(f"[proxy] log1p_counts={self.args.log1p_counts} reset_on_new_run={self.args.reset_on_new_run}")
        print(f"[proxy] heuristic_stopped={self.args.heuristic_stopped} "
              f"(lookback={self.args.stop_lookback}s zero_frac={self.args.stop_zero_frac} "
              f"queue_min={self.args.stop_queue_min} exits_max={self.args.stop_exits_max})")

        buf = b""
        while True:
            if self.unity_sock is None:
                self.accept_unity()

            try:
                chunk = self.sim_sock.recv(4096)
                if not chunk:
                    print("[proxy] upstream closed. reconnecting...")
                    try:
                        self.sim_sock.close()
                    except Exception:
                        pass
                    self.sim_sock = None
                    self.connect_upstream()
                    continue
                buf += chunk
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[proxy] upstream error: {e}. reconnecting...")
                try:
                    self.sim_sock.close()
                except Exception:
                    pass
                self.sim_sock = None
                self.connect_upstream()
                continue

            while b"\n" in buf:
                line_b, buf = buf.split(b"\n", 1)
                raw_line = line_b.decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                # passthrough to Unity
                self.unity_send_line(raw_line)

                try:
                    msg = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                self.update_from_msg(msg)

                built = self.build_latest_window()
                if built is None:
                    continue

                Xs, Xmodel, thr_raw, exits_raw = built

                pred_id, conf, probs = self.predict(Xs)
                pred_label = self.id2label.get(pred_id, f"class_{pred_id}")

                # optional heuristic relabel
                pred_label = self.apply_heuristics(pred_label, Xmodel, thr_raw, exits_raw)

                sim_ts = float(msg.get("timestamp", 0.0))

                if self.args.verbose:
                    top2 = np.argsort(-probs)[:2]
                    print(
                        f"[pred] t={sim_ts:.1f} tbin={self.latest_tbin} -> {pred_label} p={conf:.3f} | top2: "
                        f"{self.id2label.get(int(top2[0]))}:{probs[top2[0]]:.3f}, "
                        f"{self.id2label.get(int(top2[1]))}:{probs[top2[1]]:.3f}"
                    )

                if self.args.send_predictions:
                    safe_send(self.unity_sock, {
                        "msg_type": "prediction",
                        "timestamp": sim_ts,
                        "predicted_class": pred_label,
                        "confidence": conf,
                    })

                if self.should_alert(sim_ts, pred_label, conf):
                    alert = {
                        "msg_type": "alert",
                        "timestamp": sim_ts,
                        "attack_type": pred_label,
                        "confidence": conf,
                        "description": f"TemporalCNN predicted {pred_label} (p={conf:.3f})",
                    }
                    print("[ALERT]", alert)
                    safe_send(self.unity_sock, alert)


def main():
    args = parse_args()
    if not args.reset_on_new_run:
        args.reset_on_new_run = True
    Proxy(args).run()


if __name__ == "__main__":
    main()
