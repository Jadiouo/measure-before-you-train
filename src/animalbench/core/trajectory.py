"""Trajectory container + npz I/O.

agent_fields[name]  : (T, N) or (T, N, ...)   per decision interval
global_fields[name] : (T, ...)                per decision interval (e.g. 'D' = (T, N, N))
low[name]           : (F, N) or (F, ...)      low-level frames (stride), with low['t']
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    agent_fields: dict
    global_fields: dict
    low: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    commands: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def agent_array(self, names):
        """Stack named agent fields into (T, N, F)."""
        return np.stack([np.asarray(self.agent_fields[n], float) for n in names], axis=-1)

    def save(self, path):
        arrs = {}
        for k, v in self.agent_fields.items():
            arrs[f"agent/{k}"] = np.asarray(v)
        for k, v in self.global_fields.items():
            arrs[f"global/{k}"] = np.asarray(v)
        for k, v in self.low.items():
            arrs[f"low/{k}"] = np.asarray(v)
        for k, v in self.events.items():
            arrs[f"events/{k}"] = np.asarray(v)
        arrs["meta_json"] = np.frombuffer(json.dumps(self.meta, default=str).encode(), dtype=np.uint8)
        arrs["commands_json"] = np.frombuffer(json.dumps(self.commands, default=str).encode(), dtype=np.uint8)
        np.savez_compressed(path, **arrs)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        d = {"agent": {}, "global": {}, "low": {}, "events": {}}
        for k in z.files:
            if "/" in k:
                g, n = k.split("/", 1)
                d[g][n] = z[k]
        meta = json.loads(bytes(z["meta_json"]).decode())
        cmds = json.loads(bytes(z["commands_json"]).decode())
        return Trajectory(d["agent"], d["global"], d["low"], d["events"], cmds, meta)
