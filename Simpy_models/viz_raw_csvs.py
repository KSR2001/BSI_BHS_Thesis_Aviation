#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resolve_paths(manifest_path: str, paths: pd.Series) -> pd.Series:
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    out = []
    for p in paths.astype(str):
        if os.path.isabs(p) and os.path.exists(p):
            out.append(p); continue
        cand1 = os.path.abspath(p)
        if os.path.exists(cand1):
            out.append(cand1); continue
        out.append(os.path.abspath(os.path.join(manifest_dir, p)))
    return pd.Series(out)

def safe_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def plot_event_composition(df: pd.DataFrame, title: str, outpath: str, topk: int = 10):
    vc = df["event"].value_counts().head(topk)
    plt.figure()
    plt.bar(vc.index.astype(str), vc.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Row count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_total_queue_busy(df: pd.DataFrame, atk_start: float, atk_end: float, title: str, outpath: str):
    tele = df[df["event"] == "telemetry"].copy()
    if tele.empty:
        return
    tele["timestamp"] = safe_num(tele["timestamp"])
    tele["queue_length"] = safe_num(tele.get("queue_length", 0))
    tele["busy"] = safe_num(tele.get("busy", 0))

    agg = tele.groupby("timestamp").agg(
        TOTAL_QUEUE=("queue_length", "sum"),
        TOTAL_BUSY=("busy", "sum"),
    ).reset_index()

    plt.figure()
    plt.plot(agg["timestamp"], agg["TOTAL_QUEUE"], label="TOTAL_QUEUE")
    plt.plot(agg["timestamp"], agg["TOTAL_BUSY"], label="TOTAL_BUSY")
    if atk_end > atk_start:
        plt.axvspan(atk_start, atk_end, alpha=0.2)
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Sum over components")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_throughput(df: pd.DataFrame, atk_start: float, atk_end: float, title: str, outpath: str):
    thr = df[df["event"] == "throughput"].copy()
    if thr.empty:
        return
    thr["timestamp"] = safe_num(thr["timestamp"])
    thr["throughput"] = safe_num(thr.get("throughput", 0))
    agg = thr.groupby("timestamp")["throughput"].mean().reset_index()

    plt.figure()
    plt.plot(agg["timestamp"], agg["throughput"])
    if atk_end > atk_start:
        plt.axvspan(atk_start, atk_end, alpha=0.2)
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Throughput (bags/s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_diverter_mismatch(df: pd.DataFrame, atk_start: float, atk_end: float, title: str, outpath: str):
    div = df[df["event"] == "diverter_decision"].copy()
    if div.empty:
        return
    div["timestamp"] = safe_num(div["timestamp"])
    div["expected_branch"] = div.get("expected_branch", "").astype(str)
    div["chosen_branch"] = div.get("chosen_branch", "").astype(str)
    div["mismatch"] = (div["expected_branch"] != div["chosen_branch"]).astype(int)

    agg = div.groupby("timestamp").agg(
        decisions=("mismatch", "size"),
        mismatches=("mismatch", "sum"),
    ).reset_index()
    agg["mismatch_rate"] = agg["mismatches"] / np.maximum(1, agg["decisions"])

    plt.figure()
    plt.plot(agg["timestamp"], agg["mismatch_rate"])
    if atk_end > atk_start:
        plt.axvspan(atk_start, atk_end, alpha=0.2)
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Mismatch rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_sensor_hits(df: pd.DataFrame, atk_start: float, atk_end: float, title: str, outpath: str):
    s = df[df["event"] == "sensor_trigger"].copy()
    if s.empty:
        return
    
    s["timestamp"] = safe_num(s["timestamp"])
    s["tsec"] = np.floor(s["timestamp"]).astype(int)
    
    if "attack_flag" not in s.columns:
        s["attack_flag"] = ""
    if "bag_id" not in s.columns:
        s["bag_id"] = ""

    s["attack_flag"] = s["attack_flag"].fillna("").astype(str)
    s["bag_id"] = s["bag_id"].fillna("").astype(str)

    
    is_phantom = (s["attack_flag"] == "spoof_sensor") | (s["bag_id"].str.startswith("PHANTOM_"))
    is_real = ~is_phantom

    
    total = s.groupby("tsec").size().reset_index(name="hit_total")
    ph = s[is_phantom].groupby("tsec").size().reset_index(name="hit_phantom")
    rl = s[is_real].groupby("tsec").size().reset_index(name="hit_real")

    
    agg = total.merge(ph, on="tsec", how="left").merge(rl, on="tsec", how="left")
    agg["hit_phantom"] = agg["hit_phantom"].fillna(0.0)
    agg["hit_real"] = agg["hit_real"].fillna(0.0)

    
    plt.figure()
    plt.plot(agg["tsec"], agg["hit_real"], label="real sensor hits")
    plt.plot(agg["tsec"], agg["hit_phantom"], label="phantom (spoof) hits")
    plt.plot(agg["tsec"], agg["hit_total"], label="total hits", linestyle="--")

    if atk_end > atk_start:
        plt.axvspan(atk_start, atk_end, alpha=0.2)

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Sensor hit count per second")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to manifest.csv")
    ap.add_argument("--outdir", default="plots_raw", help="output folder for PNGs")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="optional run_ids to plot; if omitted, plots first run per scenario")
    ap.add_argument("--max-per-scenario", type=int, default=1, help="how many runs per scenario when --runs not used")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    m = pd.read_csv(args.manifest)
    m["path"] = resolve_paths(args.manifest, m["path"])
    m["attack_start"] = safe_num(m.get("attack_start", 0.0))
    m["attack_duration"] = safe_num(m.get("attack_duration", 0.0))
    m["scenario"] = m["scenario"].astype(str)
    m["run_id"] = m["run_id"].astype(str)

    if args.runs:
        sel = m[m["run_id"].isin(args.runs)].copy()
    else:
        # first N runs per scenario
        sel = m.groupby("scenario", as_index=False).head(args.max_per_scenario).copy()

    for _, row in sel.iterrows():
        run_id = row["run_id"]
        scen = row["scenario"]
        path = row["path"]
        atk_start = float(row["attack_start"])
        atk_end = float(row["attack_start"] + row["attack_duration"])
        df = pd.read_csv(path)
        df["event"] = df.get("event", "").astype(str)

        prefix = f"{scen}_{run_id}"
        plot_event_composition(
            df,
            title=f"Raw CSV event composition ({scen}, {run_id})",
            outpath=os.path.join(args.outdir, f"{prefix}_event_composition.png"),
        )
        plot_total_queue_busy(
            df, atk_start, atk_end,
            title=f"Raw telemetry: TOTAL_QUEUE & TOTAL_BUSY ({scen}, {run_id})",
            outpath=os.path.join(args.outdir, f"{prefix}_total_queue_busy.png"),
        )
        plot_throughput(
            df, atk_start, atk_end,
            title=f"Raw event: THROUGHPUT ({scen}, {run_id})",
            outpath=os.path.join(args.outdir, f"{prefix}_throughput.png"),
        )
        plot_diverter_mismatch(
            df, atk_start, atk_end,
            title=f"Diverter mismatch rate ({scen}, {run_id})",
            outpath=os.path.join(args.outdir, f"{prefix}_diverter_mismatch.png"),
        )
        plot_sensor_hits(
            df, atk_start, atk_end,
            title=f"Sensor hit count ({scen}, {run_id})",
            outpath=os.path.join(args.outdir, f"{prefix}_sensor_hits.png"),
        )

    print(f"[done] saved plots to: {args.outdir}")

if __name__ == "__main__":
    main()