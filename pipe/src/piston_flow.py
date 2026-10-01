from __future__ import annotations

import numpy as np
from scipy import optimize, special

CHUNK = 64
QUAD_PANELS = 400
QUAD_ORDER = 16


def j2_zeros(n):
    return special.jn_zeros(2, n)


def quadrature(n_panels=QUAD_PANELS, order=QUAD_ORDER):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, 1.0, n_panels + 1)
    half = 0.5 * np.diff(edges)[:, None]
    mid = 0.5 * (edges[:-1] + edges[1:])[:, None]
    return (mid + half * x).ravel(), (half * w).ravel()


def mean_velocity(u, r, w):
    return float(np.sum(w * 2.0 * r * u))


def series(alpha, weight, shape, r, t):
    r, t = np.broadcast_arrays(np.asarray(r, float), np.asarray(t, float))
    rr, tt = r.reshape(1, -1), t.reshape(1, -1)
    out = np.zeros(rr.shape[1])
    for i in range(0, alpha.size, CHUNK):
        a, wt = alpha[i:i + CHUNK, None], weight[i:i + CHUNK, None]
        out += np.sum(wt * shape(a, rr) * np.exp(-a ** 2 * tt), axis=0)
    return out.reshape(r.shape)


def poiseuille(r):
    return 2.0 * (1.0 - np.asarray(r, float) ** 2)


def eigenfunction(a, r):
    return special.j0(a * r) - special.j0(a)


def _ones(a, r):
    return np.ones(np.broadcast_shapes(np.shape(a), np.shape(r)))


def at_coefficients(n_terms):
    alpha = j2_zeros(n_terms)
    return alpha, 4.0 / (alpha ** 2 * special.j0(alpha))


def project(u0, n_terms):
    r, w = quadrature()
    f = u0(r)
    flux = mean_velocity(f, r, w)
    if abs(flux - 1.0) > 1e-10:
        raise ValueError(f"initial profile must have unit mean velocity, got {flux:.12f}")
    alpha = j2_zeros(n_terms)
    rhs = w * r * (f - poiseuille(r))
    num = np.concatenate([eigenfunction(alpha[i:i + CHUNK, None], r[None]) @ rhs
                          for i in range(0, n_terms, CHUNK)])
    return alpha, num / (alpha ** 2 * special.j0(alpha) ** 2 / 8.0)


def expansion_velocity(r, t, alpha, coef):
    return poiseuille(r) + series(alpha, coef, eigenfunction, r, t)


def expansion_pressure_gradient(t, alpha, coef):
    return 8.0 + 2.0 * series(alpha, coef * alpha * special.j1(alpha), _ones, 0.0, t)


def expansion_wall_shear(t, alpha, coef):
    return 0.5 * expansion_pressure_gradient(t, alpha, coef)


def at_velocity(r, t, n_terms):
    return expansion_velocity(r, t, *at_coefficients(n_terms))


def at_pressure_gradient(t, n_terms):
    alpha = j2_zeros(n_terms)
    return 4.0 * (2.0 + series(alpha, np.ones_like(alpha), _ones, 0.0, t))


def at_wall_shear(t, n_terms):
    return 0.5 * at_pressure_gradient(t, n_terms)


def rgs_velocity(r, t, n_terms):
    lam = special.jn_zeros(0, n_terms)
    return 1.0 - np.asarray(r, float) ** 2 - series(lam, 8.0 / (lam ** 3 * special.j1(lam)),
                                                    lambda a, rr: special.j0(a * rr), r, t)


def da_velocity(r, t, t0, n_terms):
    alpha = j2_zeros(n_terms)
    wt = 2.0 / (alpha ** 3 * special.j1(alpha))
    h = lambda a, rr: special.j0(a) - special.j0(a * rr)
    r, t = np.broadcast_arrays(np.asarray(r, float), np.asarray(t, float))
    s = series(alpha, wt, h, r, t)
    accel = (2.0 * t * (1 - r ** 2) + (1 - r ** 4) / 8.0 - (1 - r ** 2) / 6.0 + s) / t0
    steady = 2.0 * (1 - r ** 2) + (s - series(alpha, wt, h, r, np.maximum(t - t0, 0.0))) / t0
    return np.where(t <= t0, accel, steady)


def plug_scale(delta):
    r, w = quadrature()
    return 1.0 / mean_velocity(np.tanh((1.0 - r) / delta), r, w)


def smooth_plug(r, delta):
    return plug_scale(delta) * np.tanh((1.0 - np.asarray(r, float)) / delta)


def tan_zeros(n):
    f = lambda lam: np.sin(lam) - lam * np.cos(lam)
    return np.array([optimize.brentq(f, k * np.pi + 1e-9, (k + 0.5) * np.pi, xtol=1e-14) for k in range(1, n + 1)])


def channel_mean_velocity(u, y, w):
    return float(np.sum(w * u))


def channel_poiseuille(y):
    return 1.5 * (1.0 - np.asarray(y, float) ** 2)


def channel_eigenfunction(a, y):
    return np.cos(a * y) - np.cos(a)


def channel_at_coefficients(n_terms):
    lam = tan_zeros(n_terms)
    return lam, 2.0 / (lam ** 2 * np.cos(lam))


def channel_project(u0, n_terms):
    y, w = quadrature()
    f = u0(y)
    flux = channel_mean_velocity(f, y, w)
    if abs(flux - 1.0) > 1e-10:
        raise ValueError(f"initial profile must have unit mean velocity, got {flux:.12f}")
    lam = tan_zeros(n_terms)
    rhs = w * (f - channel_poiseuille(y))
    num = np.concatenate([channel_eigenfunction(lam[i:i + CHUNK, None], y[None]) @ rhs
                          for i in range(0, n_terms, CHUNK)])
    return lam, num / (np.sin(lam) ** 2 / 2.0)


def channel_expansion_velocity(y, t, lam, coef):
    return channel_poiseuille(y) + series(lam, coef, channel_eigenfunction, y, t)


def channel_expansion_pressure_gradient(t, lam, coef):
    return 3.0 + series(lam, coef * lam * np.sin(lam), _ones, 0.0, t)


def sparrow_channel_velocity(y, x_star, n_terms):
    lam = tan_zeros(n_terms)
    return channel_poiseuille(y) + series(lam, 2.0 / lam ** 2, lambda a, yy: np.cos(a * yy) / np.cos(a) - 1.0,
                                          y, x_star)


def channel_plug_scale(delta):
    y, w = quadrature()
    return 1.0 / channel_mean_velocity(np.tanh((1.0 - y) / delta), y, w)


def channel_smooth_plug(y, delta):
    return channel_plug_scale(delta) * np.tanh((1.0 - np.asarray(y, float)) / delta)
