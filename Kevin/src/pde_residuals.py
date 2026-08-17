from __future__ import annotations

import torch


def grad(f, coords):
    return torch.autograd.grad(f, coords, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def navier_stokes_residual(coords, u, v, w, p, rho, mu):
    grad_u, grad_v, grad_w, grad_p = grad(u, coords), grad(v, coords), grad(w, coords), grad(p, coords)
    u_x, u_y, u_z, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3], grad_u[:, 3:4]
    v_x, v_y, v_z, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3], grad_v[:, 3:4]
    w_x, w_y, w_z, w_t = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3], grad_w[:, 3:4]
    p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

    u_xx, u_yy, u_zz = grad(u_x, coords)[:, 0:1], grad(u_y, coords)[:, 1:2], grad(u_z, coords)[:, 2:3]
    v_xx, v_yy, v_zz = grad(v_x, coords)[:, 0:1], grad(v_y, coords)[:, 1:2], grad(v_z, coords)[:, 2:3]
    w_xx, w_yy, w_zz = grad(w_x, coords)[:, 0:1], grad(w_y, coords)[:, 1:2], grad(w_z, coords)[:, 2:3]

    continuity = u_x + v_y + w_z
    momentum_u = rho * (u_t + u * u_x + v * u_y + w * u_z) + p_x - mu * (u_xx + u_yy + u_zz)
    momentum_v = rho * (v_t + u * v_x + v * v_y + w * v_z) + p_y - mu * (v_xx + v_yy + v_zz)
    momentum_w = rho * (w_t + u * w_x + v * w_y + w * w_z) + p_z - mu * (w_xx + w_yy + w_zz)

    return continuity, momentum_u, momentum_v, momentum_w
