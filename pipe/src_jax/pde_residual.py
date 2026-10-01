from __future__ import annotations

import jax
import jax.numpy as jnp


def derivatives(F, X):
    y = F(X)
    dy = jax.jacfwd(F)(X)

    def second(e):
        return jax.jvp(lambda z: jax.jvp(F, (z,), (e,))[1], (X,), (e,))[1]

    d2 = jax.vmap(second)(jnp.eye(X.shape[0], dtype=X.dtype)[:3])
    return y, dy, {f: v.sum() for f, v in d2.items()}


def navier_stokes(F, X, rho, mu):
    y, dy, lap = derivatives(F, X)
    u, v, w = y["u"], y["v"], y["w"]

    def momentum(f, i):
        g = dy[f]
        return rho * (g[3] + u * g[0] + v * g[1] + w * g[2]) + dy["p"][i] - mu * lap[f]

    return {"continuity": dy["u"][0] + dy["v"][1] + dy["w"][2],
            "momentum_u": momentum("u", 0), "momentum_v": momentum("v", 1), "momentum_w": momentum("w", 2)}


def pipe_startup(F, X, nu):
    u = lambda x: F(x)["u"]
    e_r = jnp.eye(2, dtype=X.dtype)[0]
    du = jax.jacfwd(u)(X)
    u_rr = jax.jvp(lambda z: jax.jvp(u, (z,), (e_r,))[1], (X,), (e_r,))[1]
    G = F(X.at[0].set(0.0))["G"]
    r = X[0]
    return {"momentum_x": r * (du[1] - G) - nu * (r * u_rr + du[0])}


def channel_startup(F, X, nu):
    u = lambda x: F(x)["u"]
    e_y = jnp.eye(2, dtype=X.dtype)[0]
    du = jax.jacfwd(u)(X)
    u_yy = jax.jvp(lambda z: jax.jvp(u, (z,), (e_y,))[1], (X,), (e_y,))[1]
    G = F(X.at[0].set(0.0))["G"]
    return {"momentum_x": du[1] - G - nu * u_yy}
