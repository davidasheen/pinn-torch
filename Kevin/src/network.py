from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layers=(4, 64, 64, 64, 64, 4)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    def __init__(self, layers, lo, hi, feature=None, n_feature=0, n_param=0):
        super().__init__()
        self.layers = tuple(int(v) for v in layers)
        head = self.layers[0]
        self.n_param = int(n_param)
        self.feature, self.n_feature = feature, int(n_feature)
        n_coord = head - self.n_feature
        self.net = FNN(self.layers)
        lo = torch.as_tensor(lo, dtype=torch.float64).reshape(-1)
        hi = torch.as_tensor(hi, dtype=torch.float64).reshape(-1)
        if lo.numel() != n_coord or hi.numel() != n_coord:
            raise ValueError(f"lo/hi must have {n_coord} entries (layers[0] {head} - {self.n_feature} feature "
                             f"columns), got {lo.numel()}/{hi.numel()}")
        if (self.feature is None) != (self.n_feature == 0):
            raise ValueError("feature and n_feature must be given together")
        span = hi - lo
        self.register_buffer("lo", lo)
        self.register_buffer("span", torch.where(span > 0, span, torch.ones_like(span)))

    def forward(self, coords):
        x = 2.0 * (coords - self.lo) / self.span - 1.0
        if self.feature is not None:
            x = torch.cat([x, self.feature(coords).to(x.dtype)], dim=1)
        return self.net(x)


class Theta(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.spec = {}
        s0 = []
        for name, spec in specs.items():
            spec = dict(spec) if isinstance(spec, dict) else {"init": spec}
            init = float(spec["init"])
            bounds = spec.get("bounds")
            log = bool(spec.get("log", False))
            if bounds is None:
                s0.append(init)
            else:
                lo, hi = float(bounds[0]), float(bounds[1])
                if not lo < hi:
                    raise ValueError(f"{name}: bounds must satisfy lo < hi, got {bounds!r}")
                if log and lo <= 0:
                    raise ValueError(f"{name}: log bounds need lo > 0, got {bounds!r}")
                f = ((math.log(init) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log
                     else (init - lo) / (hi - lo))
                if not 0 < f < 1:
                    raise ValueError(f"{name}: init {init!r} must lie strictly inside bounds {bounds!r}")
                s0.append(math.log(f / (1 - f)))
                bounds = (lo, hi)
            self.spec[name] = {"init": init, "bounds": bounds, "log": log}
        self.names = tuple(self.spec)
        self.s = nn.Parameter(torch.tensor(s0, dtype=torch.float64))

    def value(self, name):
        i = self.names.index(name)
        spec = self.spec[name]
        if spec["bounds"] is None:
            return self.s[i]
        lo, hi = spec["bounds"]
        f = torch.sigmoid(self.s[i])
        if spec["log"]:
            return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * f)
        return lo + (hi - lo) * f

    def fractions(self):
        return {name: None if self.spec[name]["bounds"] is None else float(torch.sigmoid(self.s[i]).detach())
                for i, name in enumerate(self.names)}

    def values(self):
        return {name: float(self.value(name).detach()) for name in self.names}

    def describe(self):
        return {name: {"init": spec["init"], "bounds": None if spec["bounds"] is None else list(spec["bounds"]),
                       "log": spec["log"]} for name, spec in self.spec.items()}
