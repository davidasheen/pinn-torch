from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import hashlib

import numpy as np
import torch

import pde_residual
import pinn_training


class FrozenVelocity:

    def __init__(self, model, name="frozen"):
        self.base, self.name, self._cache = model, name, {}
        self.n_in = int(model.lo.numel())

    def digest(self):
        h = hashlib.sha256()
        for k, v in self.base.state_dict().items():
            h.update(k.encode())
            h.update(np.ascontiguousarray(v.detach().cpu().double().numpy()).tobytes())
        return h.hexdigest()[:16]

    def _model(self, dtype, device):
        key = (str(dtype), str(device))
        if key not in self._cache:
            m = copy.deepcopy(self.base).to(dtype=dtype, device=device)
            m.eval()
            self._cache[key] = m
        return self._cache[key]

    def __call__(self, X):
        P = X.detach()[:, :self.n_in]
        with torch.no_grad():
            return self._model(P.dtype, P.device)(P)[:, :3]


def make_species(provider):
    def species(X, c, D):
        c_x, c_y, c_z, c_t, c_lap = pde_residual.derivatives(c, X)
        uvw = provider(X)
        return {"species": c_t + uvw[:, 0:1] * c_x + uvw[:, 1:2] * c_y + uvw[:, 2:3] * c_z - D * c_lap}
    species.__name__ = species.__qualname__ = f"species_{provider.name}"
    return species


class CNetTraining(pinn_training.PINNTraining):

    def __init__(self, *args, provider, **kwargs):
        self.provider = provider
        super().__init__(*args, **kwargs)

    def data_sha(self):
        h = hashlib.sha256(super().data_sha().encode())
        h.update(self.provider.digest().encode())
        return h.hexdigest()[:16]
