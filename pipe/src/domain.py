from __future__ import annotations

import numpy as np
import torch

import const


class BoxDomain:
    def __init__(self, lo, hi, power=None):
        self.lo = np.asarray(lo, dtype=float).reshape(-1)
        self.hi = np.asarray(hi, dtype=float).reshape(-1)
        if self.lo.size == 0 or self.lo.size != self.hi.size:
            raise ValueError(f"BoxDomain needs lo/hi of the same nonzero length, got {self.lo.size} and {self.hi.size}")
        if np.any(self.hi <= self.lo):
            raise ValueError("BoxDomain needs hi > lo on every axis (the network scales inputs by hi - lo)")
        self.power = None if power is None else np.asarray(power, dtype=float).reshape(-1)
        if self.power is not None and (self.power.size != self.lo.size or np.any(self.power <= 0)):
            raise ValueError(f"power needs {self.lo.size} positive entries, got {power}")

    @property
    def bounds(self):
        return self.lo.copy(), self.hi.copy()

    def describe(self):
        out = {"type": "BoxDomain", "lo": self.lo.tolist(), "hi": self.hi.tolist()}
        if self.power is not None:
            out["power"] = self.power.tolist()
        return out

    def sample(self, n, generator, dtype):
        device = generator.device
        xi = torch.rand(n, self.lo.size, generator=generator, dtype=torch.float64, device=device)
        if self.power is not None:
            xi = xi ** torch.as_tensor(self.power, device=device)
        lo = torch.as_tensor(self.lo, device=device)
        hi = torch.as_tensor(self.hi, device=device)
        return (lo + (hi - lo) * xi).to(dtype)

    def sample_boundary(self, where, n, generator, dtype):
        if where["side"] not in const.BC_SIDES:
            raise ValueError(f"side must be one of {const.BC_SIDES}, got {where['side']!r}")
        axis = int(where["axis"])
        x = self.sample(n, generator, dtype)
        x[:, axis] = float((self.lo if where["side"] == "lo" else self.hi)[axis])
        return x
