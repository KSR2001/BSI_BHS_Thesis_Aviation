#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airport BHS SimPy model with:
- run_id written into every CSV row
- event-log CSV (checkin, conveyor_enter/exit, diverter_decision, storage/mcs enter/exit, build_arrival)
- TRUE spatio-temporal telemetry ticks at fixed dt (queue_len + busy per component)
- optional streaming to Unity/online proxy (newline JSON)
- --generate mode that produces exactly N runs per case AND writes manifest.csv

Attacks:
- dos: slows conveyors (speed factor)
- stopped_conv: stops conveyors
- spoof: RFID/tag tampering + optional phantom sensor hits
- fdi: diverter decision flipping

Important:
- CSV logging does NOT include conveyor_progress (huge files). Progress is optional streaming only.
"""

import argparse
import csv
import json
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import simpy


# ----------------- Data structures -----------------

@dataclass
class Bag:
    bag_id: str
    flight_id: str
    destination: str          # observed/tag destination (CAN be spoofed)
    true_destination: str     # ground truth destination (NEVER spoofed)
    weight: float
    t_start: float
    meta: dict = field(default_factory=dict)
    is_early: bool = False
    requires_mcs: bool = False


@dataclass
class AttackWindow:
    enabled: bool
    start: float
    duration: float
    params: dict = field(default_factory=dict)

    def active(self, t: float) -> bool:
        return self.enabled and (self.start <= t < self.start + self.duration)


# ----------------- Helpers -----------------

def expo(mean: float) -> float:
    return random.expovariate(1.0 / mean) if mean and mean > 0 else 0.0

def now(env) -> float:
    return float(env.now)

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def make_attack_window(runtime: float, rng: random.Random) -> Tuple[float, float]:
    # randomize start/duration to reduce "model memorizes exact timing"
    start = rng.uniform(0.2 * runtime, 0.7 * runtime)
    dur = rng.uniform(0.2 * runtime, 0.5 * runtime)
    if start + dur >= runtime:
        dur = max(30.0, runtime - start - 1.0)
    return float(start), float(dur)


# ----------------- Streaming -----------------

class Streamer:
    """TCP line-based server; Unity/proxy connects as client and reads newline JSON."""
    def __init__(self, host_port: Optional[str], net_jitter: bool,
                 drop_prob=0.03, delay_mean=0.05):
        self.sock = None
        self._listener = None
        self.lock = threading.Lock()
        self.net_jitter = net_jitter
        self.drop_prob = drop_prob
        self.delay_mean = delay_mean

        if not host_port:
            return

        host, port = host_port.split(":")
        addr = (host, int(port))

        try:
            self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._listener.bind(addr)
            self._listener.listen(1)
            print(f"[stream] Listening on {addr} for client...")

            self.sock, client_addr = self._listener.accept()
            self.sock.settimeout(0.0)
            print(f"[stream] Client connected from {client_addr}")
        except Exception as e:
            print(f"[stream] ERROR: could not listen on {addr}: {e}")
            self.sock = None

    def send(self, msg: dict):
        if self.sock is None:
            return
        try:
            msg = dict(msg)
            msg.setdefault("msg_type", "event")
            payload = (json.dumps(msg) + "\n").encode("utf-8")

            if self.net_jitter:
                r = random.random()
                if r < self.drop_prob:
                    return
                delay = random.expovariate(1.0 / max(1e-6, self.delay_mean))
                delay = min(delay, 1.5 * self.delay_mean)
                time.sleep(delay)

            with self.lock:
                self.sock.sendall(payload)
        except Exception:
            pass

    def close(self):
        for s in (self.sock, self._listener):
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self.sock = None
        self._listener = None


# ----------------- CSV Logger -----------------

class CsvLogger:
    """
    Compact schema for event logs + telemetry.
    NOTE: 'attack_flag' is for debugging/labeling only — do NOT use it as an input feature.
    """
    FIELDS = [
        "run_id",
        "timestamp",
        "bag_id",
        "event",
        "component",

        # bag attributes (mainly on checkin)
        "flight_id",
        "dest",
        "true_dest",
        "weight",
        "entry_line",
        "is_early",
        "requires_mcs",

        # diverter
        "expected_branch",
        "chosen_branch",

        # telemetry/system
        "queue_length",
        "busy",
        "throughput",

        # durations
        "transit_time",

        # misc flags
        "attack_flag",
        "scenario",
    ]

    def __init__(self, path: Optional[str], scenario: str, run_id: str):
        self.path = path
        self.scenario = scenario
        self.run_id = run_id
        self.fp = None
        self.writer = None
        if path:
            ensure_dir(path)
            self.fp = open(path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDS)
            self.writer.writeheader()

    def log_row(self, row: dict):
        if not self.writer:
            return
        out = {k: row.get(k) for k in self.FIELDS}
        out["scenario"] = self.scenario
        out["run_id"] = self.run_id
        self.writer.writerow(out)

    def close(self):
        if self.fp:
            try:
                self.fp.close()
            except Exception:
                pass
        self.fp = None
        self.writer = None


# ----------------- Components -----------------

class Conveyor:
    """
    Logs:
      - conveyor_enter
      - conveyor_exit (transit_time)
    Streams (optional):
      - conveyor_progress (Unity visuals)
    """
    def __init__(self, env, name: str, length_m=10.0, speed_mps=1.0,
                 dos_attack: AttackWindow = None,
                 stop_attack: AttackWindow = None,
                 streamer=None, logger: CsvLogger = None,
                 stream_progress: bool = False,
                 progress_step_sim: float = 0.2):
        self.env = env
        self.name = name
        self.length = float(length_m)
        self.speed = float(speed_mps)
        self.resource = simpy.Resource(env, capacity=1)
        self.queue_len = 0

        self.dos_attack = dos_attack
        self.stop_attack = stop_attack

        self.streamer = streamer
        self.logger = logger

        self.stream_progress = stream_progress
        self.progress_step_sim = float(progress_step_sim)

    def is_busy(self) -> int:
        return 1 if self.resource.count > 0 else 0

    def _attack_flag(self) -> Optional[str]:
        t = self.env.now
        if self.stop_attack and self.stop_attack.active(t):
            return "stopped_conv"
        if self.dos_attack and self.dos_attack.active(t):
            return "dos"
        return None

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def process(self, bag: Bag):
        # queueing
        self.queue_len += 1
        with self.resource.request() as req:
            yield req
            self.queue_len -= 1

            t_enter = self.env.now
            self._emit({
                "timestamp": now(self.env),
                "event": "conveyor_enter",
                "component": self.name,
                "bag_id": bag.bag_id,
                "attack_flag": self._attack_flag(),
            })

            total_len = self.length
            dist = 0.0
            step_sim = self.progress_step_sim

            while dist < total_len:
                v = self.speed
                if self.dos_attack and self.dos_attack.active(self.env.now):
                    factor = float(self.dos_attack.params.get("speed_factor", 0.2))
                    v *= factor
                if self.stop_attack and self.stop_attack.active(self.env.now):
                    v = 0.0

                yield self.env.timeout(step_sim)
                if v > 1e-6:
                    dist += v * step_sim
                    dist = min(dist, total_len)

                if self.streamer and self.stream_progress:
                    self.streamer.send({
                        "timestamp": now(self.env),
                        "event": "conveyor_progress",
                        "component": self.name,
                        "bag_id": bag.bag_id,
                        "progress": float(dist / total_len),
                    })

            t_exit = self.env.now
            self._emit({
                "timestamp": now(self.env),
                "event": "conveyor_exit",
                "component": self.name,
                "bag_id": bag.bag_id,
                "transit_time": float(t_exit - t_enter),
                "attack_flag": self._attack_flag(),
            })


class Sensor:
    """Edge sensor; can be spoofed to emit phantom hits."""
    def __init__(self, env, name: str,
                 spoof_attack: AttackWindow = None,
                 streamer=None, logger: CsvLogger = None,
                 phantom_period: float = 0.5):
        self.env = env
        self.name = name
        self.spoof_attack = spoof_attack
        self.streamer = streamer
        self.logger = logger
        self.phantom_period = float(phantom_period)

        if self.spoof_attack and self.spoof_attack.enabled:
            env.process(self._phantom_hits())

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def _phantom_hits(self):
        while True:
            yield self.env.timeout(self.phantom_period)
            if self.spoof_attack.active(self.env.now):
                self._emit({
                    "timestamp": now(self.env),
                    "event": "sensor_trigger",
                    "component": self.name,
                    "bag_id": f"PHANTOM_{int(self.env.now * 1000)}",
                    "attack_flag": "spoof_sensor",
                })

    def sense(self, bag: Bag):
        self._emit({
            "timestamp": now(self.env),
            "event": "sensor_trigger",
            "component": self.name,
            "bag_id": bag.bag_id,
            "attack_flag": None,
        })


class Diverter:
    """
    expected_branch uses TRUE destination (ground truth).
    chosen_branch uses controller view (observed destination), optionally flipped by FDI.
    """
    def __init__(self, env, name: str, target_dest: str,
                 fdi_attack: AttackWindow = None,
                 streamer=None, logger: CsvLogger = None):
        self.env = env
        self.name = name
        self.target_dest = target_dest
        self.fdi_attack = fdi_attack
        self.streamer = streamer
        self.logger = logger

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def decide(self, bag: Bag) -> str:
        expected = "branch" if bag.true_destination == self.target_dest else "straight"
        controller_expected = "branch" if bag.destination == self.target_dest else "straight"
        chosen = controller_expected

        fdi_flipped = False
        if self.fdi_attack and self.fdi_attack.active(self.env.now):
            p = float(self.fdi_attack.params.get("flip_prob", 0.7))
            if random.random() < p:
                chosen = "branch" if controller_expected == "straight" else "straight"
                fdi_flipped = True

        if fdi_flipped:
            bag.meta["fdi_flipped"] = True

        attack_flag = None
        if chosen != expected:
            if bag.meta.get("spoofed", False):
                attack_flag = "spoof_rfid"
            elif fdi_flipped:
                attack_flag = "fdi_diverter"
            else:
                attack_flag = "misroute"

        self._emit({
            "timestamp": now(self.env),
            "event": "diverter_decision",
            "component": self.name,
            "bag_id": bag.bag_id,
            "expected_branch": expected,
            "chosen_branch": chosen,
            "attack_flag": attack_flag,
        })
        return chosen


class BuildArea:
    """Sink; records completion; flags final misroute vs true_destination."""
    def __init__(self, env, name: str, streamer=None, logger: CsvLogger = None):
        self.env = env
        self.name = name
        self.count = 0
        self.streamer = streamer
        self.logger = logger

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def complete(self, bag: Bag):
        self.count += 1
        delivered = self.name.replace("BUILD_", "")
        attack_flag = None
        if delivered != bag.true_destination:
            if bag.meta.get("spoofed", False):
                attack_flag = "spoof_rfid"
            elif bag.meta.get("fdi_flipped", False):
                attack_flag = "fdi_diverter"
            else:
                attack_flag = "misroute"

        self._emit({
            "timestamp": now(self.env),
            "event": "build_arrival",
            "component": self.name,
            "bag_id": bag.bag_id,
            "transit_time": float(self.env.now - bag.t_start),
            "attack_flag": attack_flag,
        })


class EarlyBagStorage:
    def __init__(self, env, name: str, mean_hold: float,
                 streamer=None, logger: CsvLogger = None):
        self.env = env
        self.name = name
        self.mean_hold = float(mean_hold)
        self.streamer = streamer
        self.logger = logger

    def is_busy(self) -> int:
        # storage is modeled as delay only; treat as not-a-resource
        return 0

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def process(self, bag: Bag):
        self._emit({"timestamp": now(self.env), "event": "storage_enter", "component": self.name, "bag_id": bag.bag_id})
        yield self.env.timeout(expo(self.mean_hold))
        self._emit({"timestamp": now(self.env), "event": "storage_exit", "component": self.name, "bag_id": bag.bag_id})


class ManualEncodingStation:
    def __init__(self, env, name: str, capacity: int, mean_service: float,
                 streamer=None, logger: CsvLogger = None):
        self.env = env
        self.name = name
        self.res = simpy.Resource(env, capacity=int(capacity))
        self.mean_service = float(mean_service)
        self.streamer = streamer
        self.logger = logger

    def is_busy(self) -> int:
        return 1 if self.res.count > 0 else 0

    def queue_len(self) -> int:
        return len(self.res.queue)

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def process(self, bag: Bag):
        with self.res.request() as req:
            yield req
            self._emit({"timestamp": now(self.env), "event": "mcs_enter", "component": self.name, "bag_id": bag.bag_id})
            yield self.env.timeout(expo(self.mean_service))
            self._emit({"timestamp": now(self.env), "event": "mcs_exit", "component": self.name, "bag_id": bag.bag_id})


# ----------------- Source -----------------

class Source:
    def __init__(self, env, rate_per_s: float,
                 streamer=None, logger: CsvLogger = None,
                 spoof_attack: AttackWindow = None,
                 early_prob: float = 0.2,
                 unreadable_prob: float = 0.05):
        self.env = env
        self.rate = float(rate_per_s)
        self.streamer = streamer
        self.logger = logger
        self.spoof_attack = spoof_attack
        self.seq = 0
        self.early_prob = float(early_prob)
        self.unreadable_prob = float(unreadable_prob)

    def _emit(self, evt: dict):
        if self.streamer:
            self.streamer.send(evt)
        if self.logger:
            self.logger.log_row(evt)

    def new_bag(self) -> Bag:
        self.seq += 1
        bag_id = f"BAG{self.seq:06d}"
        true_dest = random.choice(["A", "B", "C", "D"])
        observed_dest = true_dest

        entry_line = 1 if random.random() < 0.5 else 2
        flight_id = f"FL{random.randint(100, 999)}"
        weight = round(random.uniform(8.0, 28.0), 1)

        bag = Bag(
            bag_id=bag_id,
            flight_id=flight_id,
            destination=observed_dest,
            true_destination=true_dest,
            weight=weight,
            t_start=self.env.now,
            meta={"entry": entry_line},
        )

        if random.random() < self.early_prob:
            bag.is_early = True
        if random.random() < self.unreadable_prob:
            bag.requires_mcs = True

        # spoof changes observed destination (not true destination)
        if self.spoof_attack and self.spoof_attack.active(self.env.now):
            mode = self.spoof_attack.params.get("mode", "flip_dest")
            if mode == "flip_dest":
                other = [d for d in ["A", "B", "C", "D"] if d != bag.destination]
                bag.destination = random.choice(other)
                bag.meta["spoofed"] = True
            elif mode == "swap_id":
                bag.bag_id = f"SWAP{bag.bag_id}"
                bag.meta["spoofed"] = True

        return bag

    def run(self, sink_process):
        while True:
            yield self.env.timeout(expo(1.0 / self.rate if self.rate > 0 else 0.0))
            bag = self.new_bag()
            entry = bag.meta.get("entry", 1)

            self._emit({
                "timestamp": now(self.env),
                "event": "checkin",
                "component": f"CHECKIN{entry}",
                "bag_id": bag.bag_id,
                "flight_id": bag.flight_id,
                "dest": bag.destination,
                "true_dest": bag.true_destination,
                "weight": bag.weight,
                "entry_line": entry,
                "is_early": int(bag.is_early),
                "requires_mcs": int(bag.requires_mcs),
                "attack_flag": "spoof_rfid" if bag.meta.get("spoofed", False) else None,
            })

            self.env.process(sink_process(bag))


# ----------------- Model -----------------

class BHSModel:
    def __init__(self, env, args, streamer=None, logger=None):
        random.seed(int(args.seed))
        self.env = env
        self.args = args
        self.run_id = args.run_id

        self.attacks = self._build_attacks(args)

        if args.mode == "normal":
            scenario_name = "normal"
        else:
            scenario_name = "+".join(sorted(args.attacks)) if args.attacks else "attack"

        self.streamer = streamer if streamer is not None else Streamer(args.stream, args.net_jitter)
        self.logger = logger if logger is not None else CsvLogger(args.out, scenario=scenario_name, run_id=self.run_id)

        # conveyors
        self.conv_ci1 = Conveyor(env, "CONV_CI1", 8.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                 self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_ci2 = Conveyor(env, "CONV_CI2", 8.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                 self.streamer, self.logger, stream_progress=args.stream_progress)

        self.conv_x1 = Conveyor(env, "CONV_X1", 10.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_x2 = Conveyor(env, "CONV_X2", 10.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                self.streamer, self.logger, stream_progress=args.stream_progress)

        self.storage = EarlyBagStorage(env, "STORAGE", args.early_hold_mean, self.streamer, self.logger)
        self.mcs = ManualEncodingStation(env, "MCS", args.mcs_capacity, args.mcs_mean_service, self.streamer, self.logger)

        self.conv_to_storage = Conveyor(env, "CONV_TO_STORAGE", 5.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                        self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_storage_to_main = Conveyor(env, "CONV_STORAGE_TO_MAIN", 5.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                             self.streamer, self.logger, stream_progress=args.stream_progress)

        self.conv_to_mcs = Conveyor(env, "CONV_TO_MCS", 5.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                    self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_mcs_to_main = Conveyor(env, "CONV_MCS_TO_MAIN", 5.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                         self.streamer, self.logger, stream_progress=args.stream_progress)

        self.conv_main = Conveyor(env, "CONV_MAIN", 12.0, 1.2, self.attacks["dos"], self.attacks["stopped_conv"],
                                  self.streamer, self.logger, stream_progress=args.stream_progress)

        self.sensor_main = Sensor(env, "S_MAIN", self.attacks["spoof"], self.streamer, self.logger)

        self.div1 = Diverter(env, "D1", "A", self.attacks["fdi"], self.streamer, self.logger)
        self.conv_after_d1 = Conveyor(env, "CONV_MAIN_D1", 6.0, 1.2, self.attacks["dos"], self.attacks["stopped_conv"],
                                      self.streamer, self.logger, stream_progress=args.stream_progress)

        self.div2 = Diverter(env, "D2", "B", self.attacks["fdi"], self.streamer, self.logger)
        self.conv_after_d2 = Conveyor(env, "CONV_MAIN_D2", 6.0, 1.2, self.attacks["dos"], self.attacks["stopped_conv"],
                                      self.streamer, self.logger, stream_progress=args.stream_progress)

        self.div3 = Diverter(env, "D3", "C", self.attacks["fdi"], self.streamer, self.logger)
        self.conv_after_d3 = Conveyor(env, "CONV_MAIN_D3", 6.0, 1.2, self.attacks["dos"], self.attacks["stopped_conv"],
                                      self.streamer, self.logger, stream_progress=args.stream_progress)

        self.div4 = Diverter(env, "D4", "D", self.attacks["fdi"], self.streamer, self.logger)

        self.conv_to_A = Conveyor(env, "CONV_TO_A", 8.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                  self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_to_B = Conveyor(env, "CONV_TO_B", 8.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                  self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_to_C = Conveyor(env, "CONV_TO_C", 8.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                  self.streamer, self.logger, stream_progress=args.stream_progress)
        self.conv_to_D = Conveyor(env, "CONV_TO_D", 8.0, 1.0, self.attacks["dos"], self.attacks["stopped_conv"],
                                  self.streamer, self.logger, stream_progress=args.stream_progress)

        self.buildA = BuildArea(env, "BUILD_A", self.streamer, self.logger)
        self.buildB = BuildArea(env, "BUILD_B", self.streamer, self.logger)
        self.buildC = BuildArea(env, "BUILD_C", self.streamer, self.logger)
        self.buildD = BuildArea(env, "BUILD_D", self.streamer, self.logger)

        self.source = Source(env, args.arrival_rate, self.streamer, self.logger,
                             spoof_attack=self.attacks["spoof"],
                             early_prob=args.early_prob,
                             unreadable_prob=args.unreadable_prob)

        # throughput
        self._sensor_count = 0
        self._window = float(args.throughput_window)
        env.process(self._throughput_meter())

        # spatio-temporal telemetry ticks
        self.telemetry_dt = float(args.telemetry_dt)
        env.process(self._telemetry_tick())

    def _build_attacks(self, args) -> Dict[str, AttackWindow]:
        def make_window(name):
            enabled = (args.mode == "attack") and (name in args.attacks)
            return AttackWindow(enabled=enabled, start=float(args.attack_start), duration=float(args.attack_duration), params={})

        cfg = {
            "dos": make_window("dos"),
            "spoof": make_window("spoof"),
            "fdi": make_window("fdi"),
            "stopped_conv": make_window("stopped_conv"),
        }
        if cfg["dos"].enabled:
            cfg["dos"].params["speed_factor"] = float(args.dos_speed_factor)
        if cfg["spoof"].enabled:
            cfg["spoof"].params["mode"] = str(args.rfid_mode)
        if cfg["fdi"].enabled:
            cfg["fdi"].params["flip_prob"] = float(args.fdi_flip_prob)
        return cfg

    def _throughput_meter(self):
        last_sample = self.env.now
        while True:
            yield self.env.timeout(self._window)
            now_t = self.env.now
            rate = self._sensor_count / max(1e-6, (now_t - last_sample))
            last_sample = now_t
            self._sensor_count = 0
            if self.logger:
                self.logger.log_row({
                    "timestamp": now(self.env),
                    "event": "throughput",
                    "component": "S_MAIN",
                    "bag_id": None,
                    "throughput": float(rate),
                })
            if self.streamer:
                self.streamer.send({
                    "timestamp": now(self.env),
                    "event": "throughput",
                    "component": "S_MAIN",
                    "bag_id": None,
                    "throughput": float(rate),
                })

    def _telemetry_tick(self):
        """
        Logs fixed-dt telemetry rows:
          event=telemetry, component=<name>, queue_length, busy
        This is what makes the dataset truly spatio-temporal.
        """
        comps = [
            self.conv_ci1, self.conv_ci2, self.conv_x1, self.conv_x2,
            self.conv_to_storage, self.conv_storage_to_main,
            self.conv_to_mcs, self.conv_mcs_to_main,
            self.conv_main, self.conv_after_d1, self.conv_after_d2, self.conv_after_d3,
            self.conv_to_A, self.conv_to_B, self.conv_to_C, self.conv_to_D,
        ]
        # also include MCS as a "component" telemetry source
        while True:
            yield self.env.timeout(self.telemetry_dt)
            t = now(self.env)

            # conveyors
            for c in comps:
                row = {
                    "timestamp": t,
                    "event": "telemetry",
                    "component": c.name,
                    "bag_id": None,
                    "queue_length": int(c.queue_len),
                    "busy": int(c.is_busy()),
                }
                if self.logger:
                    self.logger.log_row(row)
                if self.streamer:
                    self.streamer.send(row)

            # MCS telemetry
            row_mcs = {
                "timestamp": t,
                "event": "telemetry",
                "component": "MCS",
                "bag_id": None,
                "queue_length": int(self.mcs.queue_len()),
                "busy": int(self.mcs.is_busy()),
            }
            if self.logger:
                self.logger.log_row(row_mcs)
            if self.streamer:
                self.streamer.send(row_mcs)

    def run_bag(self, bag: Bag):
        entry = bag.meta.get("entry", 1)

        if entry == 1:
            yield self.env.process(self.conv_ci1.process(bag))
            yield self.env.process(self.conv_x1.process(bag))
        else:
            yield self.env.process(self.conv_ci2.process(bag))
            yield self.env.process(self.conv_x2.process(bag))

        if bag.requires_mcs:
            yield self.env.process(self.conv_to_mcs.process(bag))
            yield self.env.process(self.mcs.process(bag))
            yield self.env.process(self.conv_mcs_to_main.process(bag))

        if bag.is_early:
            yield self.env.process(self.conv_to_storage.process(bag))
            yield self.env.process(self.storage.process(bag))
            yield self.env.process(self.conv_storage_to_main.process(bag))

        yield self.env.process(self.conv_main.process(bag))

        self.sensor_main.sense(bag)
        self._sensor_count += 1

        decision = self.div1.decide(bag)
        if decision == "branch":
            yield self.env.process(self.conv_to_A.process(bag))
            self.buildA.complete(bag)
            return
        else:
            yield self.env.process(self.conv_after_d1.process(bag))

        decision = self.div2.decide(bag)
        if decision == "branch":
            yield self.env.process(self.conv_to_B.process(bag))
            self.buildB.complete(bag)
            return
        else:
            yield self.env.process(self.conv_after_d2.process(bag))

        decision = self.div3.decide(bag)
        if decision == "branch":
            yield self.env.process(self.conv_to_C.process(bag))
            self.buildC.complete(bag)
            return
        else:
            yield self.env.process(self.conv_after_d3.process(bag))

        _ = self.div4.decide(bag)
        yield self.env.process(self.conv_to_D.process(bag))
        self.buildD.complete(bag)

    def start(self):
        self.env.process(self.source.run(self.run_bag))

    def close(self):
        if self.logger:
            self.logger.close()
        if self.streamer and hasattr(self.streamer, "close"):
            self.streamer.close()


# ----------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser(description="Airport BHS SimPy (process-aware logs + spatio-temporal telemetry)")

    p.add_argument("--mode", choices=["normal", "attack"], default="normal")
    p.add_argument("--attacks", nargs="*", default=[], help="dos spoof fdi stopped_conv")

    p.add_argument("--attack-start", type=float, default=60.0)
    p.add_argument("--attack-duration", type=float, default=30.0)
    p.add_argument("--randomize-attack-window", action="store_true",
                   help="if set, ignores --attack-start/--attack-duration and randomizes them per run (useful in --generate)")

    p.add_argument("--dos-speed-factor", type=float, default=0.2)
    p.add_argument("--fdi-flip-prob", type=float, default=0.7)
    p.add_argument("--rfid-mode", choices=["flip_dest", "swap_id"], default="flip_dest")

    p.add_argument("--arrival-rate", type=float, default=0.2)
    p.add_argument("--runtime", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--early-prob", type=float, default=0.2)
    p.add_argument("--early-hold-mean", type=float, default=120.0)
    p.add_argument("--unreadable-prob", type=float, default=0.05)
    p.add_argument("--mcs-mean-service", type=float, default=20.0)
    p.add_argument("--mcs-capacity", type=int, default=1)

    p.add_argument("--telemetry-dt", type=float, default=1.0, help="fixed tick dt for telemetry rows")
    p.add_argument("--throughput-window", type=float, default=10.0)

    p.add_argument("--out", type=str, default=None, help="CSV output path")
    p.add_argument("--run-id", type=str, default="run0", help="run identifier written to CSV")

    p.add_argument("--stream", type=str, default=None, help="host:port for streaming events")
    p.add_argument("--net-jitter", action="store_true")
    p.add_argument("--stream-progress", action="store_true",
                   help="stream conveyor_progress (Unity visuals only; not written to CSV)")
    p.add_argument("--realtime", type=float, default=0.0)

    # generation
    p.add_argument("--generate", action="store_true",
                   help="generate multiple runs for each case into --out-dir and write manifest.csv")
    p.add_argument("--runs-per-case", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="data/raw")
    p.add_argument("--manifest", type=str, default="data/raw/manifest.csv")
    return p.parse_args()


def run_once(args) -> None:
    env = simpy.Environment()
    model = BHSModel(env, args)
    model.start()

    t0_wall = time.perf_counter()
    if args.realtime and args.realtime > 0:
        try:
            while env.now < args.runtime:
                env.step()
                target = t0_wall + env.now / args.realtime
                now_t = time.perf_counter()
                if target > now_t:
                    time.sleep(target - now_t)
        finally:
            model.close()
    else:
        env.run(until=args.runtime)
        model.close()


def main():
    args = parse_args()

    if args.generate:
        cases = [
            ("normal", "normal", []),
            ("fdi", "attack", ["fdi"]),
            ("spoof", "attack", ["spoof"]),
            ("dos", "attack", ["dos"]),
            ("stopped_conv", "attack", ["stopped_conv"]),
        ]
        os.makedirs(args.out_dir, exist_ok=True)
        ensure_dir(args.manifest)

        base_seed = int(args.seed)
        rows = []
        case_idx_map = {name: i for i, (name, _, _) in enumerate(cases)}
        rng = random.Random(base_seed)

        print(f"[gen] out_dir={args.out_dir} runs_per_case={args.runs_per_case} manifest={args.manifest}")
        for case_name, mode, attacks in cases:
            case_idx = case_idx_map[case_name]
            for i in range(1, args.runs_per_case + 1):
                # stable, non-mutating seed
                seed = base_seed + case_idx * 1000 + i
                run_id = f"{case_name}_run{i:02d}"
                out_csv = os.path.join(args.out_dir, f"{run_id}.csv")

                # copy args to avoid mutation
                class A: pass
                a = A()
                a.__dict__.update(args.__dict__)
                a.mode = mode
                a.attacks = list(attacks)
                a.seed = seed
                a.run_id = run_id
                a.out = out_csv

                # randomize attack window per run (recommended)
                if mode == "attack" and args.randomize_attack_window:
                    s, d = make_attack_window(args.runtime, rng)
                    a.attack_start = s
                    a.attack_duration = d

                print(f"[gen] {run_id} mode={mode} attacks={attacks} seed={seed} -> {out_csv}")
                run_once(a)

                rows.append({
                    "run_id": run_id,
                    "scenario": case_name,
                    "seed": seed,
                    "path": out_csv.replace("\\", "/"),
                    "arrival_rate": float(args.arrival_rate),
                    "runtime": float(args.runtime),
                    "attack_start": float(getattr(a, "attack_start", 0.0)) if mode == "attack" else 0.0,
                    "attack_duration": float(getattr(a, "attack_duration", 0.0)) if mode == "attack" else 0.0,
                    "telemetry_dt": float(args.telemetry_dt),
                })

        # write manifest
        with open(args.manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print("[gen] done.")
        return

    # single run
    env = simpy.Environment()
    model = BHSModel(env, args)
    model.start()
    print(f"[sim] start run_id={args.run_id} mode={args.mode} attacks={args.attacks} T={args.runtime}s")
    t0_wall = time.perf_counter()

    if args.realtime and args.realtime > 0:
        try:
            while env.now < args.runtime:
                env.step()
                target = t0_wall + env.now / args.realtime
                now_t = time.perf_counter()
                if target > now_t:
                    time.sleep(target - now_t)
        finally:
            model.close()
    else:
        env.run(until=args.runtime)
        model.close()

    dt = time.perf_counter() - t0_wall
    print(f"[sim] finished sim-time={args.runtime}s wall={dt:.2f}s")


if __name__ == "__main__":
    main()
