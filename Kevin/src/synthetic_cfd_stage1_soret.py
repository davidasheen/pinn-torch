from __future__ import annotations

import numpy as np
from scipy.special import erfc


def velocity_field(z, t, U0, mu, rho):
    nu = mu / rho
    u = U0 * erfc(z / (2.0 * np.sqrt(nu * t)))
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    p = np.zeros_like(u)
    return u, v, w, p


def temperature_field(z, t, alpha, Tw=1.0):
    return Tw * erfc(z / (2.0 * np.sqrt(alpha * t)))


def species_field(z, t, D, D_T, alpha, rho, Yw=1.0, Tw=1.0):
    if np.isclose(D, rho * alpha):
        raise ValueError(
            f"D={D} is too close to rho*alpha={rho * alpha} (Lewis number ~= 1); the closed-form "
            "solution used by this module degenerates at D == rho*alpha (division by zero in the "
            "formula) -- pick a different D/alpha pair with a clear separation."
        )
    C = D_T * Tw / (D - rho * alpha)
    K = Yw + C
    return K * erfc(z / (2.0 * np.sqrt((D / rho) * t))) - C * erfc(z / (2.0 * np.sqrt(alpha * t)))


def sample_synthetic_stage1_soret_dataset(
    n_samples,
    mu,
    D,
    D_T,
    alpha,
    U0=1.0,
    rho=1.0,
    Tw=1.0,
    Yw=1.0,
    x_range=(0.0, 1.0),
    y_range=(0.0, 1.0),
    z_range=(0.0, 2.5),
    t_range=(0.1, 1.0),
    noise_std=0.0,
    seed=None,
):
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, size=(n_samples, 1))
    y = rng.uniform(*y_range, size=(n_samples, 1))
    z = rng.uniform(*z_range, size=(n_samples, 1))
    t = rng.uniform(*t_range, size=(n_samples, 1))

    u, v, w, p = velocity_field(z, t, U0, mu, rho)
    T = temperature_field(z, t, alpha, Tw)
    Y = species_field(z, t, D, D_T, alpha, rho, Yw, Tw)

    if noise_std > 0:
        u = u + rng.normal(0.0, noise_std, u.shape)
        v = v + rng.normal(0.0, noise_std, v.shape)
        w = w + rng.normal(0.0, noise_std, w.shape)
        p = p + rng.normal(0.0, noise_std, p.shape)
        T = T + rng.normal(0.0, noise_std, T.shape)
        Y = Y + rng.normal(0.0, noise_std, Y.shape)

    X = np.hstack([x, y, z, t])
    return {
        "X": X, "u": u, "v": v, "w": w, "p": p, "T": T, "Y": Y,
        "mu_true": mu, "D_true": D, "DT_true": D_T, "alpha": alpha,
        "U0_true": U0, "rho": rho, "Tw": Tw, "Yw": Yw,
    }


def save_dataset(dataset, path):
    np.savez(path, **dataset)


def load_dataset(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}
