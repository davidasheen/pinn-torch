from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp


def _mlp(n_in, n_out, hidden, key):
    if len(set(hidden)) != 1:
        raise ValueError(f"eqx.nn.MLP needs equal hidden widths, got {hidden}")
    k_mlp, k_init = jax.random.split(key)
    mlp = eqx.nn.MLP(n_in, n_out, width_size=hidden[0], depth=len(hidden), activation=jnp.tanh, key=k_mlp)
    weights = []
    for layer, k in zip(mlp.layers, jax.random.split(k_init, len(mlp.layers))):
        lim = math.sqrt(6.0 / (layer.in_features + layer.out_features))
        weights.append(jax.random.uniform(k, layer.weight.shape, minval=-lim, maxval=lim))
    mlp = eqx.tree_at(lambda m: [layer.weight for layer in m.layers], mlp, weights)
    return eqx.tree_at(lambda m: [layer.bias for layer in m.layers], mlp,
                       [jnp.zeros_like(layer.bias) for layer in mlp.layers])


class PINN(eqx.Module):
    mlp: object
    theta: dict
    lo: tuple = eqx.field(static=True)
    hi: tuple = eqx.field(static=True)
    transform: object = eqx.field(static=True)
    subnets: object = eqx.field(static=True)
    log_input: object = eqx.field(static=True)

    def __init__(self, n_in, n_out, hidden, lo, hi, theta, key, transform=None, subnets=None, log_input=None):
        if log_input is not None:
            log_input = (int(log_input[0]), float(log_input[1]), bool(log_input[2]))
            if subnets is not None or not 0 <= log_input[0] < n_in or log_input[1] <= 0:
                raise ValueError(f"log_input needs a valid axis, eps > 0 and no subnets, got {log_input}")
        self.log_input = log_input
        if subnets is None:
            self.mlp = _mlp(n_in + (log_input is not None and log_input[2]), n_out, hidden, key)
        else:
            subnets = tuple((tuple(int(a) for a in axes), tuple(int(o) for o in outs)) for axes, outs in subnets)
            if sorted(o for _, outs in subnets for o in outs) != list(range(n_out)):
                raise ValueError(f"subnets must cover outputs 0..{n_out - 1} exactly once, got {subnets}")
            if any(not axes or min(axes) < 0 or max(axes) >= n_in for axes, _ in subnets):
                raise ValueError(f"subnet input axes must be within 0..{n_in - 1}, got {subnets}")
            self.mlp = tuple(_mlp(len(axes), len(outs), hidden, jax.random.fold_in(key, i))
                             for i, (axes, outs) in enumerate(subnets))
        self.subnets = subnets
        self.theta = {str(k): jnp.asarray(float(v)) for k, v in theta.items()}
        self.lo = tuple(float(v) for v in lo)
        self.hi = tuple(float(v) for v in hi)
        self.transform = transform

    def __call__(self, x):
        net0 = self.mlp if self.subnets is None else self.mlp[0]
        dtype = net0.layers[0].weight.dtype
        lo, hi = jnp.asarray(self.lo, dtype), jnp.asarray(self.hi, dtype)
        x = x.astype(dtype)
        z = 2.0 * (x - lo) / (hi - lo) - 1.0
        if self.log_input is not None:
            a, eps, keep = self.log_input
            span = math.log1p((self.hi[a] - self.lo[a]) / eps)
            za = 2.0 * jnp.log1p((x[a] - lo[a]) / eps) / span - 1.0
            z = jnp.append(z, za) if keep else z.at[a].set(za)
        if self.subnets is None:
            y = self.mlp(z)
        else:
            parts = jnp.concatenate([net(z[jnp.asarray(axes)]) for net, (axes, _) in zip(self.mlp, self.subnets)])
            order = [o for _, outs in self.subnets for o in outs]
            y = parts[jnp.asarray(sorted(range(len(order)), key=order.__getitem__))]
        return y if self.transform is None else self.transform(x, y)


def param_groups(params):
    labels = jax.tree.map(lambda _: "net", params)
    return eqx.tree_at(lambda m: m.theta, labels, jax.tree.map(lambda _: "theta", params.theta))
