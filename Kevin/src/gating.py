from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json

import numpy as np
import torch
import torch.nn as nn

import const

SCHEMA_VERSION = 1

ARMS = {
    "off": {"architecture": "fnn", "adaptive": False, "stage": "A"},
    "on": {"architecture": "modified_mlp", "adaptive": True, "stage": "A"},
    "gate": {"architecture": "modified_mlp", "adaptive": False, "stage": "B"},
    "adw": {"architecture": "fnn", "adaptive": True, "stage": "B"},
}

N_LEVELS = (3000, 30)
SEEDS_BY_N = {3000: tuple(range(5)), 30: tuple(range(10))}

SORET_TRUE = {"mu": const.SS_MU_TRUE, "D": const.SS_D_TRUE, "DT": const.SS_DT_TRUE}
ELL = 0.1
PARAMS = ("mu", "D", "DT")

PHASE3_REF = {
    "hard_n30_med_geo_err": 0.1429,
    "soft_n3000_med_geo_err": 0.0266,
    "hard_n30_collapsed_seeds_0_4": 2,
}
BASELINE_REL_DEV_MAX = 0.20

COLLAPSE_MU = 0.1 * SORET_TRUE["mu"]
COLLAPSE_D = 0.1 * SORET_TRUE["D"]

CRIT_63_RATIO_MAX = 0.7
CRIT_63_DT_MAX = 0.048


def _job(arm, scheme, n_samples, seed):
    return {"id": f"{arm}-{scheme}-n{n_samples}-s{seed}", "kind": "soret",
            "arm": arm, "stage": ARMS[arm]["stage"],
            "architecture": ARMS[arm]["architecture"], "adaptive": ARMS[arm]["adaptive"],
            "scheme": scheme, "n_samples": n_samples, "seed": seed,
            "lift": "constant" if scheme == "hard" else None, "ell": ELL}


JOBS = {j["id"]: j for j in (
    [_job(arm, scheme, n, s)
     for n in N_LEVELS for scheme in ("hard", "soft") for s in SEEDS_BY_N[n]
     for arm in ("off", "on")]
    + [_job(arm, scheme, n, s)
       for n in N_LEVELS for scheme in ("hard", "soft") for s in SEEDS_BY_N[n]
       for arm in ("gate", "adw")]
)}

STAGE_A_IDS = [i for i, j in JOBS.items() if j["stage"] == "A"]


def geo_err(record):
    e = record["rel_err_refit"]
    return float(np.prod([e[k] for k in PARAMS]) ** (1.0 / len(PARAMS)))


def is_collapsed(record):
    return bool(record["mu_refit"] < COLLAPSE_MU or record["D_refit"] < COLLAPSE_D)


def load_stage_a_runs(runs_dir):
    runs = {}
    missing = []
    for job_id in STAGE_A_IDS:
        path = runs_dir / f"{job_id}.json"
        if not path.exists():
            missing.append(job_id)
            continue
        with open(path, encoding="utf-8") as f:
            runs[job_id] = json.load(f)
    if missing:
        raise SystemExit(f"Stage A incomplete: {len(missing)} run(s) missing "
                         f"(e.g. {missing[:5]}); finish stage A before evaluating triggers.")
    stale = [i for i, r in runs.items() if r.get("schema") != SCHEMA_VERSION]
    if stale:
        raise SystemExit(f"Stale stage-A JSONs (schema mismatch): {stale[:5]} -- rerun with --force.")
    return runs


def _med_geo(runs, arm, scheme, n, seeds):
    return float(np.median([geo_err(runs[f"{arm}-{scheme}-n{n}-s{s}"]) for s in seeds]))


def evaluate_triggers(runs):
    off_hard30 = _med_geo(runs, "off", "hard", 30, range(5))
    off_soft3000 = _med_geo(runs, "off", "soft", 3000, range(5))
    dev_hard = abs(off_hard30 - PHASE3_REF["hard_n30_med_geo_err"]) / PHASE3_REF["hard_n30_med_geo_err"]
    dev_soft = abs(off_soft3000 - PHASE3_REF["soft_n3000_med_geo_err"]) / PHASE3_REF["soft_n3000_med_geo_err"]
    collapsed_04 = sum(is_collapsed(runs[f"off-hard-n30-s{s}"]) for s in range(5))
    gate_62 = bool(dev_hard < BASELINE_REL_DEV_MAX and dev_soft < BASELINE_REL_DEV_MAX
                   and collapsed_04 == PHASE3_REF["hard_n30_collapsed_seeds_0_4"])

    on_soft3000 = _med_geo(runs, "on", "soft", 3000, SEEDS_BY_N[3000])
    off_soft3000_all = _med_geo(runs, "off", "soft", 3000, SEEDS_BY_N[3000])
    ratio_63 = on_soft3000 / off_soft3000_all
    dt_med_63 = float(np.median([runs[f"on-soft-n3000-s{s}"]["rel_err_refit"]["DT"]
                                 for s in SEEDS_BY_N[3000]]))
    trig_63 = bool(ratio_63 <= CRIT_63_RATIO_MAX and dt_med_63 <= CRIT_63_DT_MAX)

    on_collapsed = sum(is_collapsed(runs[f"on-hard-n30-s{s}"]) for s in SEEDS_BY_N[30])
    off_collapsed = sum(is_collapsed(runs[f"off-hard-n30-s{s}"]) for s in SEEDS_BY_N[30])
    trig_64 = bool(on_collapsed == 0)

    return {
        "gate_6_2": {"pass": gate_62,
                     "off_hard_n30_med_geo_err_seeds_0_4": off_hard30,
                     "off_soft_n3000_med_geo_err_seeds_0_4": off_soft3000,
                     "rel_dev_hard_n30": dev_hard, "rel_dev_soft_n3000": dev_soft,
                     "collapsed_seeds_0_4": collapsed_04,
                     "phase3_ref": dict(PHASE3_REF), "rel_dev_max": BASELINE_REL_DEV_MAX},
        "trigger_6_3": {"triggered": trig_63, "cell": ["soft", 3000],
                        "on_med_geo_err": on_soft3000, "off_med_geo_err": off_soft3000_all,
                        "ratio": ratio_63, "ratio_max": CRIT_63_RATIO_MAX,
                        "on_DT_med_err": dt_med_63, "dt_max": CRIT_63_DT_MAX},
        "trigger_6_4": {"triggered": trig_64, "cell": ["hard", 30],
                        "on_collapsed": on_collapsed, "off_collapsed": off_collapsed,
                        "n_seeds": len(SEEDS_BY_N[30]),
                        "collapse_def": f"mu_refit < {COLLAPSE_MU} or D_refit < {COLLAPSE_D}"},
    }


class ModifiedMLP(nn.Module):
    def __init__(self, layers=(4, 64, 64, 64, 64, 4)):
        super().__init__()
        in_dim = layers[0]
        hidden_dims = layers[1:-1]
        out_dim = layers[-1]
        if len(set(hidden_dims)) != 1:
            raise ValueError(
                f"ModifiedMLP requires all hidden layers to share the same width, got {hidden_dims}"
            )
        hidden_dim = hidden_dims[0]

        self.activation = nn.Tanh()
        self.branch_u = nn.Linear(in_dim, hidden_dim)
        self.branch_v = nn.Linear(in_dim, hidden_dim)
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.gate_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(hidden_dims) - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        for m in [self.branch_u, self.branch_v, self.input_layer, *self.gate_layers, self.output_layer]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        u = self.activation(self.branch_u(x))
        v = self.activation(self.branch_v(x))
        h = self.activation(self.input_layer(x))
        for gate_layer in self.gate_layers:
            z = self.activation(gate_layer(h))
            h = (1 - z) * u + z * v
        return self.output_layer(h)
