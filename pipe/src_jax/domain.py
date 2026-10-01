from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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

    def sample(self, key, n, dtype):
        xi = jax.random.uniform(key, (n, self.lo.size), dtype=jnp.float64)
        if self.power is not None:
            xi = xi ** self.power
        return (self.lo + (self.hi - self.lo) * xi).astype(dtype)

    def sample_boundary(self, where, key, n, dtype):
        if where["side"] not in const.BC_SIDES:
            raise ValueError(f"side must be one of {const.BC_SIDES}, got {where['side']!r}")
        sides = (("_lo", "lo"), ("_hi", "hi")) if where["side"] == "pair" else (("", where["side"]),)
        axis = int(where["axis"])
        base = self.sample(key, n, dtype)
        return {suffix: base.at[:, axis].set(float((self.lo if side == "lo" else self.hi)[axis])) for suffix, side in sides}
