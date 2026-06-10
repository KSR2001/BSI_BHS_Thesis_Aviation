#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

CANON = ["normal", "dos", "spoof", "fdi", "stopped_conv"]
LABEL_MAP = {k: i for i, k in enumerate(CANON)}

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/lstm_windows.npz")
    ap.add_argument("--window", type=int, default=60, help="timesteps per window")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.30)   # fraction of runs per scenario
    ap.add_argument("--val-size", type=float, default=0.20)    # fraction of remaining runs per scenario
    ap.add_argument("--random-state", type=int, default=42)

    # IMPORTANT: correct window labels using attack window
    ap.add_argument("--label-mode", choices=["run", "attack_window"], default="attack_window",
                    help="run = all windows in an attack run are attack; "
                         "attack_window = only windows overlapping attack interval are attack (recommended).")

    # Make heavy-tailed count features easier for AE to learn
    ap.add_argument("--log1p-counts", action="store_true",
                    help="apply log(1+x) to count-like features (recommended).")
    return ap.parse_args()

# ---------------- IO ----------------

def load_run_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # enforce dtypes safely
    df["timestamp"] = pd.to_numeric(df.get("timestamp", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["event"] = df.get("event", "").astype(str)
    df["component"] = df.get("component", "").astype(str)

    # ensure numeric columns exist
    for col in ["queue_length", "busy", "throughput", "transit_time"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["expected_branch", "chosen_branch", "bag_id"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    return df

# ---------------- Split ----------------

def stratified_run_split(manifest_df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    rng = np.random.default_rng(seed)
    train_runs, val_runs, test_runs = [], [], []

    for scen in CANON:
        runs = manifest_df.loc[manifest_df["scenario"] == scen, "run_id"].astype(str).tolist()
        runs = list(dict.fromkeys(runs))  # unique preserve order
        n = len(runs)
        if n == 0:
            continue

        perm = rng.permutation(runs).tolist()

        if n >= 3:
            n_test = max(1, int(np.floor(n * test_size)))
            n_test = min(n_test, n - 2)

            rem = n - n_test
            n_val = max(1, int(np.floor(rem * val_size)))
            n_val = min(n_val, rem - 1)
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0

        test_s = perm[:n_test]
        val_s  = perm[n_test:n_test + n_val]
        train_s = perm[n_test + n_val:]

        test_runs.extend(test_s)
        val_runs.extend(val_s)
        train_runs.extend(train_s)

    return np.array(train_runs), np.array(val_runs), np.array(test_runs)

# ---------------- Feature engineering ----------------

def _tbin(ts: pd.Series, dt: float) -> pd.Series:
    return (ts / dt).astype(int)

def build_run_timeseries(
    df: pd.DataFrame,
    runtime: float,
    dt: float,
    components: list,
    conveyors: list,
    diverters: list,
    sensors: list,
    log1p_counts: bool,
) -> tuple[np.ndarray, list]:
    T = int(np.floor(runtime / dt))
    if T <= 1:
        raise ValueError(f"runtime/dt too small -> T={T}. Increase runtime or decrease dt.")

    # ---------- TELEMETRY ----------
    tele = df[df["event"] == "telemetry"].copy()
    if tele.empty:
        raise ValueError("No telemetry rows found. Ensure bhs_sim_behavioral.py logs event='telemetry'")

    tele["tbin"] = _tbin(tele["timestamp"], dt)
    tele = tele[(tele["tbin"] >= 0) & (tele["tbin"] < T)]
    tele = tele[tele["component"].isin(components)].copy()

    q = tele.pivot_table(index="tbin", columns="component", values="queue_length", aggfunc="mean")
    q = q.reindex(range(T)).fillna(0.0)
    q = q.reindex(columns=components).fillna(0.0)

    b = tele.pivot_table(index="tbin", columns="component", values="busy", aggfunc="mean")
    b = b.reindex(range(T)).fillna(0.0)
    b = b.reindex(columns=components).fillna(0.0)

    q_arr = q.values.astype(np.float32)
    b_arr = b.values.astype(np.float32)

    total_queue = q_arr.sum(axis=1, keepdims=True).astype(np.float32)
    total_busy  = b_arr.sum(axis=1, keepdims=True).astype(np.float32)

    # ---------- CONVEYOR EXIT ----------
    conv_exit = df[df["event"] == "conveyor_exit"].copy()
    if not conv_exit.empty:
        conv_exit["tbin"] = _tbin(conv_exit["timestamp"], dt)
        conv_exit = conv_exit[(conv_exit["tbin"] >= 0) & (conv_exit["tbin"] < T)]
        conv_exit = conv_exit[conv_exit["component"].isin(conveyors)]
        conv_exit["transit_time"] = pd.to_numeric(conv_exit["transit_time"], errors="coerce").fillna(0.0)

        exit_cnt = conv_exit.groupby(["tbin", "component"]).size().unstack("component").reindex(range(T)).fillna(0.0)
        exit_cnt = exit_cnt.reindex(columns=conveyors).fillna(0.0)

        exit_mean = conv_exit.groupby(["tbin", "component"])["transit_time"].mean().unstack("component").reindex(range(T)).fillna(0.0)
        exit_mean = exit_mean.reindex(columns=conveyors).fillna(0.0)
    else:
        exit_cnt = pd.DataFrame(0.0, index=range(T), columns=conveyors)
        exit_mean = pd.DataFrame(0.0, index=range(T), columns=conveyors)

    exit_cnt_arr = exit_cnt.values.astype(np.float32)
    exit_mean_arr = exit_mean.values.astype(np.float32)

    # ---------- DIVERTERS ----------
    div = df[df["event"] == "diverter_decision"].copy()
    if not div.empty:
        div["tbin"] = _tbin(div["timestamp"], dt)
        div = div[(div["tbin"] >= 0) & (div["tbin"] < T)]
        div = div[div["component"].isin(diverters)]

        exp = div["expected_branch"].astype(str)
        ch  = div["chosen_branch"].astype(str)
        div["mismatch"] = (exp != ch).astype(int)

        div_cnt = div.groupby(["tbin", "component"]).size().unstack("component").reindex(range(T)).fillna(0.0)
        div_cnt = div_cnt.reindex(columns=diverters).fillna(0.0)

        mis_cnt = div.groupby(["tbin", "component"])["mismatch"].sum().unstack("component").reindex(range(T)).fillna(0.0)
        mis_cnt = mis_cnt.reindex(columns=diverters).fillna(0.0)
    else:
        div_cnt = pd.DataFrame(0.0, index=range(T), columns=diverters)
        mis_cnt = pd.DataFrame(0.0, index=range(T), columns=diverters)

    div_cnt_arr = div_cnt.values.astype(np.float32)
    mis_cnt_arr = mis_cnt.values.astype(np.float32)

    # ---------- SENSORS ----------
    s = df[df["event"] == "sensor_trigger"].copy()
    if not s.empty:
        s["tbin"] = _tbin(s["timestamp"], dt)
        s = s[(s["tbin"] >= 0) & (s["tbin"] < T)]
        s = s[s["component"].isin(sensors)]

        hit_cnt = s.groupby(["tbin", "component"]).size().unstack("component").reindex(range(T)).fillna(0.0)
        hit_cnt = hit_cnt.reindex(columns=sensors).fillna(0.0)
    else:
        hit_cnt = pd.DataFrame(0.0, index=range(T), columns=sensors)

    hit_cnt_arr = hit_cnt.values.astype(np.float32)

    # ---------- THROUGHPUT ----------
    thr = df[df["event"] == "throughput"].copy()
    throughput = np.zeros((T,), dtype=np.float32)
    if not thr.empty:
        thr["tbin"] = _tbin(thr["timestamp"], dt)
        thr = thr[(thr["tbin"] >= 0) & (thr["tbin"] < T)]
        thr["throughput"] = pd.to_numeric(thr["throughput"], errors="coerce").fillna(0.0)

        per = thr.groupby("tbin")["throughput"].mean()
        for tb, v in per.items():
            throughput[int(tb)] = float(v)

        last = 0.0
        for i in range(T):
            if throughput[i] != 0.0:
                last = float(throughput[i])
            else:
                throughput[i] = last

    throughput_arr = throughput.reshape(-1, 1).astype(np.float32)

    # ---------- Optional log1p on counts ----------
    if log1p_counts:
        q_arr = np.log1p(q_arr)
        total_queue = np.log1p(total_queue)
        exit_cnt_arr = np.log1p(exit_cnt_arr)
        div_cnt_arr = np.log1p(div_cnt_arr)
        mis_cnt_arr = np.log1p(mis_cnt_arr)
        hit_cnt_arr = np.log1p(hit_cnt_arr)

    # ---------- CONCAT ----------
    feature_names = []
    feature_names += [f"{c}__queue_length" for c in components]
    feature_names += [f"{c}__busy" for c in components]
    feature_names += [f"{c}__exit_count" for c in conveyors]
    feature_names += [f"{c}__mean_transit_time" for c in conveyors]
    feature_names += [f"{d}__decision_count" for d in diverters]
    feature_names += [f"{d}__mismatch_count" for d in diverters]
    feature_names += [f"{s}__hit_count" for s in sensors]
    feature_names += ["THROUGHPUT", "TOTAL_QUEUE", "TOTAL_BUSY"]

    X = np.concatenate(
        [q_arr, b_arr, exit_cnt_arr, exit_mean_arr, div_cnt_arr, mis_cnt_arr,
         hit_cnt_arr, throughput_arr, total_queue, total_busy],
        axis=1
    ).astype(np.float32)

    return X, feature_names

# ---------------- Main ----------------

def main():
    a = parse_args()
    m = pd.read_csv(a.manifest)

    required = {"run_id", "scenario", "path", "runtime", "telemetry_dt"}
    if not required.issubset(set(m.columns)):
        raise ValueError("manifest.csv must include run_id, scenario, path, runtime, telemetry_dt")

    m["scenario"] = m["scenario"].astype(str)
    m = m[m["scenario"].isin(CANON)].copy()

    # resolve paths
    manifest_dir = os.path.dirname(os.path.abspath(a.manifest))
    def _resolve(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        cand1 = os.path.abspath(p)
        if os.path.exists(cand1):
            return cand1
        return os.path.abspath(os.path.join(manifest_dir, p))

    m["path"] = m["path"].apply(_resolve)

    # discover components from telemetry
    all_components = set()
    diverters_set = set()
    sensors_set = set()
    for p in m["path"].tolist():
        df = load_run_csv(p)
        tele = df[df["event"] == "telemetry"]
        all_components.update(tele["component"].dropna().astype(str).unique().tolist())
        diverters_set.update(df.loc[df["event"] == "diverter_decision", "component"].astype(str).unique().tolist())
        sensors_set.update(df.loc[df["event"] == "sensor_trigger", "component"].astype(str).unique().tolist())

    components = sorted([c for c in all_components if c])
    conveyors  = sorted([c for c in components if c.startswith("CONV_")])
    diverters  = sorted([d for d in diverters_set if d])
    sensors    = sorted([s for s in sensors_set if s])

    # build per-run series + keep manifest info for labeling
    run_series = {}
    run_meta = {}  # attack_start/duration per run_id
    feature_names_final = None

    for _, row in m.iterrows():
        run_id = str(row["run_id"])
        scenario = str(row["scenario"])
        path = str(row["path"])
        runtime = float(row["runtime"])
        dt = float(row["telemetry_dt"])

        atk_start = row.get("attack_start", "")
        atk_dur   = row.get("attack_duration", "")
        atk_start = float(atk_start) if str(atk_start).strip() != "" else 0.0
        atk_dur   = float(atk_dur)   if str(atk_dur).strip() != "" else 0.0

        df = load_run_csv(path)
        X, feat_names = build_run_timeseries(
            df, runtime=runtime, dt=dt,
            components=components, conveyors=conveyors,
            diverters=diverters, sensors=sensors,
            log1p_counts=a.log1p_counts
        )

        run_series[run_id] = (X, scenario, dt, runtime)
        run_meta[run_id] = (atk_start, atk_dur)

        if feature_names_final is None:
            feature_names_final = feat_names
        elif feat_names != feature_names_final:
            raise ValueError("Feature schema mismatch across runs. Ensure consistent components/events across runs.")

    # split runs
    run_ids_train, run_ids_val, run_ids_test = stratified_run_split(
        m, test_size=a.test_size, val_size=a.val_size, seed=a.random_state
    )

    def run_counts(run_ids):
        out = {k: 0 for k in CANON}
        for rid in run_ids:
            scen = run_series[str(rid)][1]
            out[scen] += 1
        return out

    print("[split] runs per scenario:")
    print("  train:", run_counts(run_ids_train))
    print("  val  :", run_counts(run_ids_val))
    print("  test :", run_counts(run_ids_test))

    # window labeling
    def window_label(run_id: str, t0: float, t1: float, scenario: str) -> int:
        if scenario == "normal":
            return LABEL_MAP["normal"]

        if a.label_mode == "run":
            return LABEL_MAP[scenario]

        # attack_window mode
        atk_start, atk_dur = run_meta.get(run_id, (0.0, 0.0))
        atk_end = atk_start + atk_dur
        # overlap?
        overlaps = (t0 < atk_end) and (t1 > atk_start)
        return LABEL_MAP[scenario] if overlaps else LABEL_MAP["normal"]

    # create windows
    def make_windows(run_id: str):
        X, scenario, dt, _runtime = run_series[run_id]
        w, s = a.window, a.stride
        T = X.shape[0]

        xs, ys, rids, t0s = [], [], [], []
        for tb0 in range(0, T - w + 1, s):
            t0 = tb0 * dt
            t1 = (tb0 + w) * dt
            xs.append(X[tb0:tb0+w])
            ys.append(window_label(run_id, t0, t1, scenario))
            rids.append(run_id)
            t0s.append(t0)

        if not xs:
            return (np.zeros((0, w, X.shape[1]), dtype=np.float32),
                    np.zeros((0,), dtype=int),
                    np.zeros((0,), dtype=object),
                    np.zeros((0,), dtype=np.float32))

        return (np.stack(xs, axis=0).astype(np.float32),
                np.array(ys, dtype=int),
                np.array(rids, dtype=object),
                np.array(t0s, dtype=np.float32))

    def stack_runs(run_ids_list):
        Xs, Ys, RIDs, T0s = [], [], [], []
        for r in run_ids_list:
            r = str(r)
            x, y, rid, t0 = make_windows(r)
            if x.shape[0] == 0:
                continue
            Xs.append(x); Ys.append(y); RIDs.append(rid); T0s.append(t0)

        if not Xs:
            w = a.window
            F = next(iter(run_series.values()))[0].shape[1]
            return (np.zeros((0, w, F), dtype=np.float32),
                    np.zeros((0,), dtype=int),
                    np.zeros((0,), dtype=object),
                    np.zeros((0,), dtype=np.float32))

        return (np.concatenate(Xs, axis=0),
                np.concatenate(Ys, axis=0),
                np.concatenate(RIDs, axis=0),
                np.concatenate(T0s, axis=0))

    X_train, y_train, rid_train, t0_train = stack_runs(run_ids_train)
    X_val,   y_val,   rid_val,   t0_val   = stack_runs(run_ids_val)
    X_test,  y_test,  rid_test,  t0_test  = stack_runs(run_ids_test)

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        a.out,
        X_train=X_train, y_train=y_train, rid_train=rid_train, t0_train=t0_train,
        X_val=X_val, y_val=y_val, rid_val=rid_val, t0_val=t0_val,
        X_test=X_test, y_test=y_test, rid_test=rid_test, t0_test=t0_test,
        feature_names=np.array(feature_names_final, dtype=object),
        components=np.array(components, dtype=object),
        conveyors=np.array(conveyors, dtype=object),
        diverters=np.array(diverters, dtype=object),
        sensors=np.array(sensors, dtype=object),
        label_map=np.array(CANON, dtype=object),
        window=np.array([a.window], dtype=int),
        stride=np.array([a.stride], dtype=int),
        random_state=np.array([a.random_state], dtype=int),
        label_mode=np.array([a.label_mode], dtype=object),
        log1p_counts=np.array([int(a.log1p_counts)], dtype=int),
    )

    def counts(y):
        s = pd.Series(y).value_counts().sort_index()
        return {CANON[i]: int(s.get(i, 0)) for i in range(len(CANON))}

    print(f"[save] {a.out}")
    print("train windows:", X_train.shape, "label counts:", counts(y_train))
    print("val   windows:", X_val.shape,   "label counts:", counts(y_val))
    print("test  windows:", X_test.shape,  "label counts:", counts(y_test))
    print("n_features:", X_train.shape[-1] if X_train.size else X_val.shape[-1] if X_val.size else X_test.shape[-1])

if __name__ == "__main__":
    main()
