from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules += [nn.Linear(layers[i], layers[i + 1]), nn.Tanh()]
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    def __init__(self, layers, lo, hi, transform=None, log_input=None):
        super().__init__()
        layers = [int(v) for v in layers]
        self.log_input = None if log_input is None else dict(log_input)
        if self.log_input is not None:
            a, eps = self.log_input["axis"], self.log_input["eps"]
            if not 0 <= a < layers[0] or eps <= 0:
                raise ValueError(f"log_input needs a valid axis and eps > 0, got {log_input}")
            self.log_span = math.log1p((float(hi[a]) - float(lo[a])) / eps)
            layers[0] += bool(self.log_input["keep_linear"])
        self.net = FNN(layers)
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float64).reshape(-1))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float64).reshape(-1))
        self.transform = transform

    def forward(self, x):
        z = 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0
        if self.log_input is not None:
            a, eps = self.log_input["axis"], self.log_input["eps"]
            za = (2.0 * torch.log1p((x[:, a] - self.lo[a]) / eps) / self.log_span - 1.0)[:, None]
            z = torch.cat([z, za], dim=1) if self.log_input["keep_linear"] else torch.cat([z[:, :a], za, z[:, a + 1:]], dim=1)
        y = self.net(z)
        return y if self.transform is None else self.transform(x, y)
