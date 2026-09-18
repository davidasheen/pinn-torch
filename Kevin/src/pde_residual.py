from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


def grad(f, coords):
    return torch.autograd.grad(f, coords, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def derivatives(f, coords):
    g = grad(f, coords)
    lap = grad(g[:, 0:1], coords)[:, 0:1] + grad(g[:, 1:2], coords)[:, 1:2] + grad(g[:, 2:3], coords)[:, 2:3]
    return g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4], lap


def navier_stokes(X, u, v, w, p, rho, mu):
    u_x, u_y, u_z, u_t, u_lap = derivatives(u, X)
    v_x, v_y, v_z, v_t, v_lap = derivatives(v, X)
    w_x, w_y, w_z, w_t, w_lap = derivatives(w, X)
    grad_p = grad(p, X)
    return {"continuity": u_x + v_y + w_z,
            "momentum_u": rho * (u_t + u * u_x + v * u_y + w * u_z) + grad_p[:, 0:1] - mu * u_lap,
            "momentum_v": rho * (v_t + u * v_x + v * v_y + w * v_z) + grad_p[:, 1:2] - mu * v_lap,
            "momentum_w": rho * (w_t + u * w_x + v * w_y + w * w_z) + grad_p[:, 2:3] - mu * w_lap}


def heat(X, T, alpha):
    _, _, _, T_t, T_lap = derivatives(T, X)
    return {"energy": T_t - alpha * T_lap}


def species_soret(X, Y, T, u, v, w, rho, D, DT):
    Y_x, Y_y, Y_z, Y_t, Y_lap = derivatives(Y, X)
    _, _, _, _, T_lap = derivatives(T, X)
    return {"species": rho * (Y_t + u * Y_x + v * Y_y + w * Y_z) - D * Y_lap - DT * T_lap}


def diffusion(X, c, D):
    _, _, _, c_t, c_lap = derivatives(c, X)
    return {"species": c_t - D * c_lap}


def advection_diffusion(X, c, u, v, w, D):
    c_x, c_y, c_z, c_t, c_lap = derivatives(c, X)
    return {"species": c_t + u * c_x + v * c_y + w * c_z - D * c_lap}
