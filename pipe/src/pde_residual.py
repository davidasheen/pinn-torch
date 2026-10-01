from __future__ import annotations

import torch


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def on_axis(X):
    return torch.stack([torch.zeros_like(X[:, 0]), X[:, 1]], dim=1)


def pipe_startup(F, X, nu):
    du = grad(F(X)["u"], X)
    u_rr = grad(du[:, 0], X)[:, 0]
    G = F(on_axis(X))["G"]
    r = X[:, 0]
    return {"momentum_x": r * (du[:, 1] - G) - nu * (r * u_rr + du[:, 0])}


def channel_startup(F, X, nu):
    du = grad(F(X)["u"], X)
    u_yy = grad(du[:, 0], X)[:, 0]
    G = F(on_axis(X))["G"]
    return {"momentum_x": du[:, 1] - G - nu * u_yy}
