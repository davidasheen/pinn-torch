from __future__ import annotations

import numpy as np


def taylor_green_fields(x, y, z, t, mu, rho=1.0):
    nu = mu / rho
    decay1 = np.exp(-2.0 * nu * t)
    decay2 = np.exp(-4.0 * nu * t)
    u = np.cos(x) * np.sin(y) * decay1
    v = -np.sin(x) * np.cos(y) * decay1
    w = np.zeros_like(x)
    p = -0.25 * rho * (np.cos(2 * x) + np.cos(2 * y)) * decay2
    return u, v, w, p


def sample_synthetic_dataset(
    n_samples,
    mu,
    rho=1.0,
    x_range=(0.0, 2 * np.pi),
    y_range=(0.0, 2 * np.pi),
    z_range=(0.0, 1.0),
    t_range=(0.0, 1.0),
    noise_std=0.0,
    seed=None,
):
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, size=(n_samples, 1))
    y = rng.uniform(*y_range, size=(n_samples, 1))
    z = rng.uniform(*z_range, size=(n_samples, 1))
    t = rng.uniform(*t_range, size=(n_samples, 1))

    u, v, w, p = taylor_green_fields(x, y, z, t, mu, rho)

    if noise_std > 0:
        u = u + rng.normal(0.0, noise_std, u.shape)
        v = v + rng.normal(0.0, noise_std, v.shape)
        w = w + rng.normal(0.0, noise_std, w.shape)
        p = p + rng.normal(0.0, noise_std, p.shape)

    X = np.hstack([x, y, z, t])
    return {"X": X, "u": u, "v": v, "w": w, "p": p, "mu_true": mu, "rho": rho}


def save_dataset(dataset, path):
    np.savez(
        path,
        X=dataset["X"],
        u=dataset["u"],
        v=dataset["v"],
        w=dataset["w"],
        p=dataset["p"],
        mu_true=dataset["mu_true"],
        rho=dataset["rho"],
    )


def load_dataset(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}
