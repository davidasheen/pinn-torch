from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import pickle
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import const
import network


def _short(value, width=90):
    text = repr(value)
    return text if len(text) <= width else text[:width - 3] + "..."


def _host(tree):
    return jax.tree.map(np.asarray, tree)


def _device(tree):
    return jax.tree.map(jnp.asarray, tree)


def _astype(tree, dtype):
    return jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


def _leaves(tree):
    return None if tree is None else [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _restore(template, leaves):
    arrays, static = eqx.partition(template, eqx.is_array)
    return eqx.combine(jax.tree.unflatten(jax.tree.structure(arrays), [jnp.asarray(x) for x in leaves]), static)


ADAM = optax.multi_transform(
    {"net": optax.chain(optax.clip_by_global_norm(const.GRAD_CLIP), optax.adam(const.ADAM_LR)),
     "theta": optax.adam(const.INVERSE_LR)},
    network.param_groups)
LBFGS = optax.lbfgs(memory_size=const.LBFGS_HISTORY_SIZE)


class Problem(eqx.Module):
    fields: tuple = eqx.field(static=True)
    pde: tuple = eqx.field(static=True)
    boundary: tuple = eqx.field(static=True)
    inverse: tuple = eqx.field(static=True)
    known: tuple = eqx.field(static=True)

    def field_fn(self, model):
        return lambda x: dict(zip(self.fields, model(x)))

    def coef(self, model, name):
        return model.theta[name] if name in self.inverse else dict(self.known)[name]

    def point_residual(self, spec, model, pts, prefix):
        name, fn, inputs = spec
        F = self.field_fn(model)
        kwargs = {pname: F if kind == "F" else pts[key] if kind == "X" else self.coef(model, key)
                  for pname, kind, key in inputs}
        r = fn(**kwargs)
        if isinstance(r, dict):
            return {f"{name}_{k}" if prefix else str(k): jnp.atleast_1d(v) for k, v in r.items()}
        return {name: jnp.concatenate([jnp.atleast_1d(v) for v in r])}

    def residual_pde(self, model, X):
        res = {}
        for spec in self.pde:
            res.update(jax.vmap(lambda x, s=spec: self.point_residual(s, model, {"": x}, False))(X))
        return res

    def residual_bc(self, i, model, pts):
        return jax.vmap(lambda p: self.point_residual(self.boundary[i], model, p, True))(pts)

    def loss(self, model, weights, data, colloc, bc_pts):
        terms, X = {}, colloc
        if data is not None:
            terms["data"] = jnp.mean((jax.vmap(model)(data["X"]) - data["Y"]) ** 2)
            X = jnp.concatenate([data["X"], colloc])
        for k, r in self.residual_pde(model, X).items():
            terms[k] = jnp.mean(r ** 2)
        for i, pts in enumerate(bc_pts):
            for k, r in self.residual_bc(i, model, pts).items():
                terms[k] = jnp.mean(r ** 2)
        if weights is None:
            return sum(terms.values()), terms
        return sum(weights[k] * v for k, v in terms.items()), terms


@eqx.filter_jit
def eval_loss(problem, model, weights, *batch):
    return problem.loss(model, weights, *batch)


@eqx.filter_jit
def adam_step(problem, model, opt_state, weights, *batch):
    grads = eqx.filter_grad(lambda m: problem.loss(m, weights, *batch)[0])(model)
    updates, opt_state = ADAM.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), opt_state


@eqx.filter_jit
def term_grad_norms(problem, model, *batch):
    params, static = eqx.partition(model, eqx.is_array)
    jac = jax.jacrev(lambda p: problem.loss(eqx.combine(p, static), None, *batch)[1])(params)
    return {k: jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(tree.mlp))) for k, tree in jac.items()}


@eqx.filter_jit
def lbfgs_block(problem, model, state, n_steps, weights, *batch):
    params, static = eqx.partition(model, eqx.is_array)
    fn = lambda p: problem.loss(eqx.combine(p, static), weights, *batch)[0]
    value_and_grad = optax.value_and_grad_from_state(fn)

    def body(_, carry):
        p, s = carry
        value, grad = value_and_grad(p, state=s)
        updates, s = LBFGS.update(grad, s, p, value=value, grad=grad, value_fn=fn)
        return optax.apply_updates(p, updates), s

    params, state = jax.lax.fori_loop(0, n_steps, body, (params, state))
    return eqx.combine(params, static), state


@eqx.filter_jit
def residuals(problem, model, pts, which):
    return problem.residual_pde(model, pts) if which is None else problem.residual_bc(which, model, pts)


@eqx.filter_jit
def rar_score(problem, model, pts, which):
    return sum((r ** 2).sum(axis=1) for r in residuals(problem, model, pts, which).values())


@eqx.filter_jit
def predict(model, X):
    return jax.vmap(model)(X)


class PINNTraining:
    def __init__(
        self,
        dataset,
        fields,
        pde=(),
        inverse=None,
        known=None,
        domain=None,
        boundary=(),
        hidden=const.HIDDEN_LAYERS,
        adam_epochs=const.ADAM_EPOCHS,
        lbfgs_iterations=const.LBFGS_ITERATIONS,
        seed=const.SEED,
        log_every=const.LOG_EVERY,
        record_every=const.RECORD_EVERY,
        checkpoint_path=None,
        checkpoint_every=const.CHECKPOINT_EVERY,
        resume="auto",
        num_node=None,
        transform=None,
        loss_balance=None,
        subnets=None,
        log_input=None,
    ):
        self.dataset = dataset
        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("fields must not be empty")
        if dataset is not None:
            missing = [f for f in self.fields if f not in dataset]
            if missing:
                raise ValueError(f"fields {missing} not in dataset; dataset keys: {sorted(dataset)}")
        if (dataset is None) != (num_node is not None):
            raise ValueError("num_node is required with dataset=None and not allowed otherwise")
        self.num_node = num_node
        self.inverse = {str(k): float(v) for k, v in (inverse or {}).items()}
        self.known = {str(k): float(v) for k, v in (known or {}).items()}
        both = sorted(set(self.inverse) & set(self.known))
        if both:
            raise ValueError(f"{both} given as both inverse and known")
        if domain is None:
            raise ValueError("PINNTraining needs a domain (domain.BoxDomain)")
        self.domain = domain
        self.lo, self.hi = domain.bounds
        self.n_in = int(self.lo.size)
        self.transform = transform
        self.loss_balance = self.normalize_balance(loss_balance)
        self.subnets = None if subnets is None else {str(f): [int(a) for a in axes] for f, axes in subnets.items()}
        if self.subnets is not None and sorted(self.subnets) != sorted(self.fields):
            raise ValueError(f"subnets must name every field once: fields {self.fields}, got {sorted(self.subnets)}")
        self.log_input = None if log_input is None else {"axis": int(log_input["axis"]), "eps": float(log_input["eps"]),
                                                           "keep_linear": bool(log_input.get("keep_linear", False))}
        self.hidden = tuple(int(h) for h in hidden)
        self.layers = (self.n_in, *self.hidden, len(self.fields))
        self.adam_epochs = int(adam_epochs)
        self.lbfgs_iterations = int(lbfgs_iterations)
        self.seed = int(seed)
        self.log_every = int(log_every)
        self.record_every = int(record_every)
        self.checkpoint_path = None if checkpoint_path is None else pathlib.Path(checkpoint_path)
        self.checkpoint_every = int(checkpoint_every)
        if resume not in const.RESUME_MODES:
            raise ValueError(f"resume must be one of {const.RESUME_MODES}, got {resume!r}")
        self.resume = resume
        self.dtype = jnp.dtype(const.TRAIN_DTYPE)
        self.lbfgs_dtype = jnp.dtype(const.LBFGS_DTYPE)
        self.t_construct = time.perf_counter()
        self.pde = [self.normalize_spec(s, "pde") for s in pde]
        self.boundary = [self.normalize_spec(s, "bc") for s in boundary]
        self.split_data()

        base = jax.random.key(self.seed)
        self.key_colloc = jax.random.fold_in(base, const.SEED_OFFSET_COLLOCATION)
        self.key_valid = jax.random.fold_in(base, const.SEED_OFFSET_VALIDATION)
        for i, spec in enumerate(self.boundary):
            spec["key"] = jax.random.fold_in(base, const.SEED_OFFSET_BOUNDARY + i)
            spec["colloc"] = {}
        net_groups = None if self.subnets is None else [(axes, [self.fields.index(f)])
                                                        for f, axes in self.subnets.items()]
        self.model = _astype(network.PINN(self.n_in, len(self.fields), self.hidden, self.lo, self.hi, self.inverse,
                                          key=jax.random.fold_in(base, const.SEED_OFFSET_NETWORK),
                                          transform=transform, subnets=net_groups,
                                          log_input=None if self.log_input is None else tuple(self.log_input.values())),
                             self.dtype)
        self.opt_state = ADAM.init(eqx.filter(self.model, eqx.is_array))
        as_tuple = lambda s: (s["name"], s["residual"], tuple((p, kind, key) for p, (kind, key) in s["inputs"].items()))
        self.problem = Problem(fields=self.fields, pde=tuple(as_tuple(s) for s in self.pde),
                               boundary=tuple(as_tuple(s) for s in self.boundary),
                               inverse=tuple(self.inverse), known=tuple(sorted(self.known.items())))
        self.register_terms()
        self.weights = None if self.loss_balance is None else {k: 1.0 for k in self.term_names}
        self.weight_arrays = None

        raw = dict(layers=list(self.layers), fields=list(self.fields), inverse=self.inverse, known=self.known,
                   bounds=[self.lo.tolist(), self.hi.tolist()],
                   pde=[self.describe_spec(s) for s in self.pde],
                   boundary=[self.describe_spec(s) for s in self.boundary],
                   domain=self.domain.describe(), split_pattern=list(const.SPLIT_PATTERN),
                   num_domain=const.NUM_DOMAIN, rar_pool=const.RAR_POOL, resample_every=const.RESAMPLE_EVERY,
                   adam_epochs=self.adam_epochs, adam_lr=const.ADAM_LR, inverse_lr=const.INVERSE_LR,
                   grad_clip=const.GRAD_CLIP, lbfgs_iterations=self.lbfgs_iterations, lbfgs_chunk=const.LBFGS_CHUNK,
                   lbfgs_history=const.LBFGS_HISTORY_SIZE, seed=self.seed,
                   dtype=const.TRAIN_DTYPE, lbfgs_dtype=const.LBFGS_DTYPE,
                   terms={"pde": self.pde_terms, "bc": self.bc_terms},
                   **{f"n_{s}": 0 if d is None else int(d["X"].shape[0])
                      for s, d in zip(const.SPLITS, (self.train_data, self.valid_data, self.test_data))},
                   n_colloc=dict(self.target_n), n_node=dict(self.n_node),
                   n_boundary={spec["name"]: dict(spec["n"]) for spec in self.boundary},
                   record_every=self.record_every, checkpoint_every=self.checkpoint_every,
                   restore_best=True, data_sha=self.data_sha(),
                   transform=None if transform is None else getattr(transform, "__qualname__", repr(transform)),
                   loss_balance=self.loss_balance, subnets=self.subnets, log_input=self.log_input)
        self.config = json.loads(json.dumps(raw, default=float))
        self.config["fingerprint"] = self.fingerprint(self.config)
        self.history = {k: [] for k in const.HISTORY_KEYS}
        self.history.update({k: [] for k in self.inverse})
        self.history.update({f"{s}_term_{k}": [] for s in ("train", "val") for k in self.term_names})
        if self.weights is not None:
            self.history.update({f"weight_{k}": [] for k in self.term_names})
        self.best = {"val": float("inf"), "epoch": None, "phase": None, "model": None}
        self.adam_epoch, self.adam_done, self.lbfgs_done = 0, False, False
        self.epoch = 0
        self.colloc = None
        self.result = None
        if self.checkpoint_path is not None and self.resume is not False and self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as fh:
                ck = pickle.load(fh)
            stale = self.checkpoint_diff(ck)
            if not stale:
                self.resume_from_checkpoint(ck)
            elif self.resume == "auto":
                print(f"[resume] STALE {self.checkpoint_path}: starting fresh\n  " + "\n  ".join(stale), flush=True)
            else:
                raise ValueError(f"checkpoint {self.checkpoint_path} does not match this run "
                                 f"(pass resume='auto' to retrain or resume=False):\n  " + "\n  ".join(stale))


    def normalize_spec(self, spec, kind):
        if callable(spec):
            spec = {"residual": spec}
        spec = dict(spec)
        fn = spec["residual"]
        out = {"name": str(spec.get("name") or getattr(fn, "__name__", "residual")), "residual": fn}
        if kind == "bc":
            out["num"] = int(spec.get("num", const.NUM_BOUNDARY))
            side = str(spec.get("side"))
            if side not in const.BC_SIDES:
                raise ValueError(f"boundary {out['name']!r}: side must be one of {const.BC_SIDES}, got {side!r}")
            out["where"] = {"axis": int(spec["axis"]), "side": side}
        out["inputs"] = self.bind(out, kind)
        return out

    def bind(self, spec, kind):
        name = spec["name"]
        pair = kind == "bc" and spec["where"]["side"] == "pair"
        points = tuple("X" + s for s in const.PAIR_SUFFIXES) if pair else ("X",)
        coefs = (*self.inverse, *self.known)
        inputs = {}
        for pname, param in inspect.signature(spec["residual"]).parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise ValueError(f"{name}: *args/**kwargs are not supported in residual functions")
            if pname == "F":
                inputs[pname] = ("F", "")
            elif pname in points:
                inputs[pname] = ("X", pname[1:])
            elif pname in coefs:
                inputs[pname] = ("coef", pname)
            else:
                raise ValueError(f"{name}: argument {pname!r} is not F, a point {points}, or a coefficient {coefs}")
        absent = [p for p in points if p not in inputs]
        if absent:
            raise ValueError(f"{name}: residual must take the point argument(s) {absent}")
        return inputs

    def describe_spec(self, spec):
        out = {"name": spec["name"], "function": getattr(spec["residual"], "__qualname__", repr(spec["residual"])),
               "inputs": {p: f"{kind}:{key}" for p, (kind, key) in spec["inputs"].items()},
               "outputs": list(spec["outputs"])}
        if "where" in spec:
            out["num"] = spec["num"]
            out["where"] = spec["where"]
        return out

    def register_terms(self):
        x = jnp.zeros(self.n_in, self.dtype)
        self.pde_terms, self.bc_terms = [], []
        for specs, compiled, prefix, register in ((self.pde, self.problem.pde, False, self.pde_terms),
                                                  (self.boundary, self.problem.boundary, True, self.bc_terms)):
            for spec, spec_t in zip(specs, compiled):
                pts = {key: x for kind, key in spec["inputs"].values() if kind == "X"}
                out = eqx.filter_eval_shape(lambda m, q, s=spec_t, pre=prefix: self.problem.point_residual(s, m, q, pre),
                                            self.model, pts)
                spec["outputs"] = list(out)
                register.extend(out)
        names = ["data", *self.pde_terms, *self.bc_terms]
        dup = sorted({k for k in names if names.count(k) > 1})
        if dup:
            raise ValueError(f"residual names must be unique across data/pde/boundary, duplicated: {dup}")
        self.term_names = [*(["data"] if self.dataset is not None else []), *self.pde_terms, *self.bc_terms]

    @staticmethod
    def normalize_balance(spec):
        if spec is None:
            return None
        spec = dict(spec)
        if spec.get("method") not in const.LOSS_BALANCE_METHODS:
            raise ValueError(f"loss_balance method must be one of {const.LOSS_BALANCE_METHODS}, got {spec.get('method')!r}")
        out = {"method": spec["method"], "every": int(spec.get("every", const.GRAD_NORM_EVERY)),
               "alpha": float(spec.get("alpha", const.GRAD_NORM_ALPHA)), "select": "lbfgs"}
        if out["every"] <= 0 or not 0.0 <= out["alpha"] < 1.0:
            raise ValueError(f"loss_balance needs every > 0 and 0 <= alpha < 1, got {out}")
        if spec.get("max_weight") is not None:
            out["max_weight"] = float(spec["max_weight"])
            if out["max_weight"] <= 0:
                raise ValueError(f"loss_balance max_weight must be > 0, got {out['max_weight']}")
        return out

    def current_weights(self):
        if self.weights is None:
            return None
        if self.weight_arrays is None:
            self.weight_arrays = {k: jnp.asarray(v, self.dtype) for k, v in self.weights.items()}
        return self.weight_arrays

    def update_weights(self):
        norms = {k: float(v) for k, v in term_grad_norms(self.problem, self.model, *self.batch("train")).items()}
        total, a = sum(norms.values()), self.loss_balance["alpha"]
        cap = self.loss_balance.get("max_weight", float("inf"))
        self.weights = {k: min(cap, a * w + (1.0 - a) * total / norms[k]) if norms[k] > 0 else w
                        for k, w in self.weights.items()}
        self.weight_arrays = None


    @staticmethod
    def split_index(n, pattern):
        label = np.digitize(np.arange(n) % int(sum(pattern)), np.cumsum(pattern)[:-1])
        return tuple(np.flatnonzero(label == i) for i in range(3))

    def split_counts(self, n):
        return {s: int(rows.size) for s, rows in zip(const.SPLITS, self.split_index(n, const.SPLIT_PATTERN))}

    def split_data(self):
        self.target_n = self.split_counts(const.NUM_DOMAIN)
        for spec in self.boundary:
            spec["n"] = self.split_counts(spec["num"])
        if self.dataset is None:
            self.train_data = self.valid_data = self.test_data = self.data = None
            self.n_node = self.split_counts(self.num_node)
            return
        n = int(np.shape(self.dataset["X"])[0])
        splits = []
        for name, rows in zip(const.SPLITS, self.split_index(n, const.SPLIT_PATTERN)):
            if rows.size == 0:
                raise ValueError(f"split {name!r} is empty under pattern {const.SPLIT_PATTERN}")
            splits.append({k: (np.asarray(v)[rows] if np.ndim(v) >= 1 and np.shape(v)[0] == n else v)
                           for k, v in self.dataset.items()})
        self.train_data, self.valid_data, self.test_data = splits
        self.data = self.device_data()
        self.n_node = {s: int(np.shape(d["X"])[0]) for s, d in zip(const.SPLITS, splits)}

    def device_data(self):
        if self.dataset is None:
            return None
        return {s: {"X": jnp.asarray(d["X"], dtype=self.dtype),
                    "Y": jnp.asarray(np.column_stack([np.asarray(d[f], float).reshape(-1) for f in self.fields]),
                                     dtype=self.dtype)}
                for s, d in (("train", self.train_data), ("valid", self.valid_data))}

    def cast(self, dtype):
        self.dtype = jnp.dtype(dtype)
        self.model = _astype(self.model, self.dtype)
        self.weight_arrays = None
        self.data = self.device_data()
        if self.colloc is not None:
            self.colloc = _astype(self.colloc, self.dtype)
        for spec in self.boundary:
            spec["colloc"] = _astype(spec["colloc"], self.dtype)

    def data_sha(self):
        if self.dataset is None:
            return None
        h = hashlib.sha256()
        for k in ("X", *self.fields):
            h.update(np.ascontiguousarray(np.asarray(self.dataset[k], dtype=float)).tobytes())
        return h.hexdigest()[:16]

    @staticmethod
    def fingerprint(config):
        subset = {k: config.get(k) for k in const.FINGERPRINT_KEYS}
        subset.update({k: config[k] for k in const.FINGERPRINT_OPTIONAL_KEYS if config.get(k) is not None})
        return hashlib.sha256(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def checkpoint_diff(self, ck):
        cfg = ck.get("config") or {}
        if cfg.get("fingerprint") == self.config["fingerprint"]:
            return []
        diffs = [f"{key}: {_short(cfg.get(key))} -> {_short(self.config.get(key))}"
                 for key in (*const.FINGERPRINT_KEYS, *const.FINGERPRINT_OPTIONAL_KEYS)
                 if cfg.get(key) != self.config.get(key)]
        return diffs or ["fingerprint differs"]

    def param_values(self):
        return {k: float(v) for k, v in self.model.theta.items()}


    def bc_colloc(self, split):
        return [spec["colloc"][split] for spec in self.boundary]

    def batch(self, split):
        return None if self.data is None else self.data[split], self.colloc[split], self.bc_colloc(split)

    def train_loss(self):
        return eval_loss(self.problem, self.model, self.current_weights(), *self.batch("train"))

    @staticmethod
    def rar_pick(key, score, n):
        return jax.random.choice(key, score.shape[0], (n,), replace=False, p=score + score.mean())

    def sample_fixed(self):
        keys = jax.random.split(self.key_valid, 2 * (1 + len(self.boundary)))
        self.colloc = {}
        for j, s in enumerate(("valid", "test")):
            self.colloc[s] = self.domain.sample(keys[j], self.n_node[s] + self.target_n[s], self.dtype)
            for i, spec in enumerate(self.boundary):
                spec["colloc"][s] = self.domain.sample_boundary(spec["where"], keys[2 + 2 * i + j], spec["n"][s],
                                                                self.dtype)

    def resample(self):
        if self.colloc is None:
            self.sample_fixed()
        k_node, k_cand, k_pick = jax.random.split(jax.random.fold_in(self.key_colloc, self.epoch), 3)
        n = self.target_n["train"]
        cand = self.domain.sample(k_cand, int(round(n * const.RAR_POOL)), self.dtype)
        keep = self.rar_pick(k_pick, rar_score(self.problem, self.model, cand, None), n)
        self.colloc["train"] = jnp.concatenate([self.domain.sample(k_node, self.n_node["train"], self.dtype), cand[keep]])
        for i, spec in enumerate(self.boundary):
            k_cand, k_pick = jax.random.split(jax.random.fold_in(spec["key"], self.epoch))
            n = spec["n"]["train"]
            cand = self.domain.sample_boundary(spec["where"], k_cand, int(round(n * const.RAR_POOL)), self.dtype)
            keep = self.rar_pick(k_pick, rar_score(self.problem, self.model, cand, i), n)
            spec["colloc"]["train"] = {s: X[keep] for s, X in cand.items()}


    def term_parts(self, terms):
        return (float(terms["data"]) if "data" in terms else float("nan"),
                sum(float(terms[k]) for k in self.pde_terms), sum(float(terms[k]) for k in self.bc_terms))

    def record(self, phase, total, terms):
        v_total, v_terms = eval_loss(self.problem, self.model, self.current_weights(), *self.batch("valid"))
        vt = float(v_total)
        eligible = self.loss_balance is None or phase == self.loss_balance["select"]
        if eligible and vt < self.best["val"]:
            self.best.update(val=vt, epoch=int(self.epoch), phase=phase, model=self.model)
        values = (int(self.epoch), phase, float(total), *self.term_parts(terms), vt, *self.term_parts(v_terms))
        for k, v in zip(const.HISTORY_KEYS, values):
            self.history[k].append(v)
        for k, v in self.param_values().items():
            self.history[k].append(v)
        for s, tm in (("train", terms), ("val", v_terms)):
            for k, v in tm.items():
                self.history.setdefault(f"{s}_term_{k}", []).append(float(v))
        for k, v in (self.weights or {}).items():
            self.history.setdefault(f"weight_{k}", []).append(v)

    def show(self, phase, total, terms):
        td, tp, tb = self.term_parts(terms)
        msg = f"[{phase}] epoch {self.epoch:6d}  loss={float(total):.4e}"
        if self.dataset is not None:
            msg += f"  data={td:.2e}"
        msg += f"  pde={tp:.2e}"
        if self.boundary:
            msg += f"  bc={tb:.2e}"
        for k, v in self.param_values().items():
            msg += f"  {k}={v:.5g}"
        if self.weights is not None:
            msg += "  weights=[" + " ".join(f"{v:.3g}" for v in self.weights.values()) + "]"
        msg += f"  elapsed={time.perf_counter() - self.t_construct:.0f}s"
        print(msg, flush=True)

    def maybe_log(self, phase):
        do_record = self.record_every and self.epoch % self.record_every == 0
        do_show = self.log_every and self.epoch % self.log_every == 0
        if not (do_record or do_show):
            return
        total, terms = self.train_loss()
        if do_record:
            self.record(phase, total, terms)
        if do_show:
            self.show(phase, total, terms)


    def save_checkpoint(self, adam_epoch, adam_done, lbfgs_done):
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"config": self.config, "dtype": str(self.dtype), "model": _leaves(self.model), "history": self.history,
                 "train_state": {"adam_epoch": int(adam_epoch), "adam_done": bool(adam_done),
                                 "lbfgs_done": bool(lbfgs_done), "epoch": int(self.epoch),
                                 "opt_state": None if adam_done else _leaves(self.opt_state),
                                 "best": {**self.best, "model": _leaves(self.best["model"])},
                                 "weights": self.weights,
                                 "colloc": _host(self.colloc),
                                 "bc_colloc": [_host(spec["colloc"]) for spec in self.boundary]}}
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh)
        tmp.replace(self.checkpoint_path)

    def resume_from_checkpoint(self, ck):
        ts = ck["train_state"]
        self.adam_epoch, self.adam_done, self.lbfgs_done = ts["adam_epoch"], ts["adam_done"], ts["lbfgs_done"]
        self.epoch = int(ts["epoch"])
        if ck["dtype"] != str(self.dtype):
            self.cast(ck["dtype"])
        self.model = _restore(self.model, ck["model"])
        self.history = ck["history"]
        best = ts["best"]["model"]
        self.best = {**ts["best"], "model": None if best is None else _restore(self.model, best)}
        if ts["opt_state"] is not None:
            self.opt_state = _restore(self.opt_state, ts["opt_state"])
        if ts.get("weights") is not None:
            self.weights, self.weight_arrays = dict(ts["weights"]), None
        if ts["colloc"] is not None:
            self.colloc = _device(ts["colloc"])
            for spec, saved in zip(self.boundary, ts["bc_colloc"]):
                spec["colloc"] = _device(saved)
        if self.log_every:
            print(f"[resume] {self.checkpoint_path}: adam_epoch={self.adam_epoch}/{self.adam_epochs} "
                  f"adam_done={self.adam_done} lbfgs_done={self.lbfgs_done}", flush=True)


    def train_adam(self):
        if self.colloc is None:
            self.resample()
        e = self.adam_epoch
        try:
            for e in range(self.adam_epoch, self.adam_epochs):
                if e > 0 and e % const.RESAMPLE_EVERY == 0:
                    self.resample()
                if self.weights is not None and e % self.loss_balance["every"] == 0:
                    self.update_weights()
                self.model, self.opt_state = adam_step(self.problem, self.model, self.opt_state,
                                                       self.current_weights(), *self.batch("train"))
                self.maybe_log("adam")
                self.epoch += 1
                if self.checkpoint_every and (e + 1) % self.checkpoint_every == 0:
                    self.save_checkpoint(e + 1, False, False)
        except KeyboardInterrupt:
            self.save_checkpoint(e, False, False)
            print(f"[adam] interrupted at epoch {e}, checkpoint saved; call train again to resume", flush=True)
            raise
        total, terms = self.train_loss()
        self.record("adam", total, terms)
        if self.log_every:
            self.show("adam", total, terms)
        self.adam_epoch, self.adam_done = self.adam_epochs, True
        self.save_checkpoint(self.adam_epochs, True, False)

    def train_lbfgs(self):
        if self.dtype != self.lbfgs_dtype:
            self.cast(self.lbfgs_dtype)
            if self.log_every:
                print(f"[lbfgs] cast to {self.dtype}", flush=True)
        self.resample()
        chunk = min(const.LBFGS_CHUNK, self.lbfgs_iterations)
        step = min(self.record_every, chunk) if self.record_every else chunk
        n_iter = min(max(self.epoch - self.adam_epochs, 0), self.lbfgs_iterations)
        since = 0
        shown = self.epoch
        state = LBFGS.init(eqx.filter(self.model, eqx.is_array))
        try:
            while n_iter < self.lbfgs_iterations:
                if self.record_every:
                    total, terms = self.train_loss()
                    self.record("lbfgs", total, terms)
                    if self.log_every and self.epoch - shown >= self.log_every:
                        shown = self.epoch
                        self.show("lbfgs", total, terms)
                n = min(step, self.lbfgs_iterations - n_iter, chunk - since)
                self.model, state = lbfgs_block(self.problem, self.model, state, jnp.asarray(n),
                                                self.current_weights(), *self.batch("train"))
                n_iter += n
                self.epoch += n
                since += n
                if since < chunk and n_iter < self.lbfgs_iterations:
                    continue
                since = 0
                if self.checkpoint_every:
                    self.save_checkpoint(self.adam_epochs, True, False)
                if n_iter < self.lbfgs_iterations:
                    self.resample()
                    state = LBFGS.init(eqx.filter(self.model, eqx.is_array))
        except KeyboardInterrupt:
            self.save_checkpoint(self.adam_epochs, True, False)
            print("[lbfgs] interrupted, checkpoint saved; call train again to resume "
                  "(L-BFGS restarts from the saved weights)", flush=True)
            raise
        total, terms = self.train_loss()
        self.record("lbfgs", total, terms)
        if self.log_every:
            self.show("lbfgs", total, terms)
            print(f"[lbfgs] stopped after {n_iter} iterations", flush=True)
        self.lbfgs_done = True

    def restore_best(self):
        if self.best["model"] is None:
            return False
        self.model = _astype(self.best["model"], self.dtype)
        if self.log_every:
            print(f"[best] restored epoch {self.best['epoch']} ({self.best['phase']}) "
                  f"val_total={self.best['val']:.4e}", flush=True)
        return True

    def train(self):
        t_start = time.perf_counter()
        if not self.adam_done:
            self.train_adam()
        if self.lbfgs_iterations and not self.lbfgs_done:
            self.train_lbfgs()
        self.lbfgs_done = True
        restored = self.restore_best()
        self.save_checkpoint(self.adam_epochs, True, True)
        if self.colloc is None:
            self.resample()
        total, terms = self.train_loss()
        self.result = {"params": self.param_values(),
                       "weights": self.weights,
                       "history": self.history,
                       "best": {"epoch": self.best["epoch"], "phase": self.best["phase"],
                                "val": None if self.best["model"] is None else self.best["val"],
                                "restored": restored},
                       "loss": float(total),
                       "terms": {k: float(v) for k, v in terms.items()},
                       "wall_seconds": time.perf_counter() - t_start,
                       "config": self.config}
        return self.result


    def evaluate(self, coords):
        y = np.asarray(predict(self.model, jnp.asarray(coords, dtype=self.dtype)))
        return {name: y[:, i] for i, name in enumerate(self.fields)}

    def evaluate_residual(self, split):
        if self.colloc is None:
            self.resample()
        out = {k: np.asarray(v).ravel() for k, v in residuals(self.problem, self.model, self.colloc[split], None).items()}
        for i, spec in enumerate(self.boundary):
            res = residuals(self.problem, self.model, spec["colloc"][split], i)
            out.update({k: np.asarray(v).ravel() for k, v in res.items()})
        return out

    @staticmethod
    def error_record(y, yhat):
        y = np.asarray(y, float).ravel()
        yhat = np.asarray(yhat, float).ravel()
        if y.size == 0:
            return {"n": 0}
        span = float(y.max() - y.min())
        resid = yhat - y
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {"n": int(y.size), "span": span,
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "span_rms": float(np.sqrt(np.mean(resid ** 2)) / span) if span > 0 else None,
                "within_band": float(np.mean(np.abs(resid) <= const.DIAG_BAND * span)),
                "r2": (1.0 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot > 0 else None,
                "computed": y.tolist(), "predicted": yhat.tolist()}

    def diag_record(self, split="test", data=None):
        if data is None:
            data = {"train": self.train_data, "valid": self.valid_data, "test": self.test_data}[split]
        pred = self.evaluate(data["X"])
        return {f: self.error_record(data[f], pred[f]) for f in self.fields if f in data}


    @staticmethod
    def finish(fig, path=None):
        if path is not None:
            fig.savefig(path)
        if plt.get_backend().lower() != "agg":
            plt.show()

    @staticmethod
    def history_axis(history):
        return np.asarray(history["epoch"]), "epoch"

    @staticmethod
    def best_axis(best):
        return None if not best else best.get("epoch")

    @staticmethod
    def plot_loss(history, path=None, title="Loss history", best_epoch=None, ax=None):
        own = ax is None
        if own:
            fig, ax = plt.subplots()
        it, xlabel = PINNTraining.history_axis(history)
        vt = np.asarray([np.nan if v is None else v for v in history["val_total"]], float)
        ax.semilogy(it, history["train_total"], color="tab:blue", label="train total")
        ax.semilogy(it, vt, color="tab:orange", label="validation total")
        adam = [i for i, ph in zip(it, history["phase"]) if ph == "adam"]
        if adam and max(adam) < it.max():
            ax.axvline(max(adam), color="0.5", lw=1, ls=":", label="Adam -> L-BFGS")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="tab:red", lw=1, ls="--", label="best epoch")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("total loss")
        if own:
            ax.set_title(title)
            ax.legend()
            PINNTraining.finish(fig, path)
        return ax

    @staticmethod
    def plot_diag(records, fields, grid=None, path=None, title="Test diagonal", band=const.DIAG_BAND,
                  units=None, axes=None):
        own = axes is None
        if own:
            nrows, ncols = grid or (1, len(fields))
            fig, axarr = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows + 1), squeeze=False)
            axes = axarr.ravel()
        axes = np.ravel(axes)
        units = const.FIELD_UNITS if units is None else units
        for ax, f in zip(axes, fields):
            label, scale = units.get(f, (f, 1.0))
            rec = records[f]
            if not rec.get("n") or "computed" not in rec:
                ax.text(0.5, 0.5, f"no {f} points", transform=ax.transAxes, ha="center", va="center")
                ax.set_xlabel(f"computed {label}")
                ax.set_ylabel(f"predicted {label}")
                continue
            y = np.asarray(rec["computed"], float) * scale
            yh = np.asarray(rec["predicted"], float) * scale
            half = band * rec["span"] * scale
            lo = float(min(y.min(), yh.min()))
            hi = float(max(y.max(), yh.max()))
            pad = 0.05 * (hi - lo) if hi > lo else 0.01
            lo, hi = lo - pad, hi + pad
            ax.fill_between([lo, hi], [lo - half, hi - half], [lo + half, hi + half],
                            color="0.85", label=f"{100 * band:.0f}% of span")
            ax.plot([lo, hi], [lo, hi], color="0.3", lw=1)
            ax.plot(y, yh, ".", ms=4, alpha=0.6, color="tab:blue")
            note = f"span-RMS {100 * rec['span_rms']:.1f}%" if rec.get("span_rms") else ""
            if rec.get("within_band") is not None:
                note += f"\nwithin band {100 * rec['within_band']:.0f}%"
            ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=10)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"computed {label}")
            ax.set_ylabel(f"predicted {label}")
        if own:
            for ax, f in zip(axes, fields):
                ax.set_title(f"test {f}")
            for ax in axes[len(fields):]:
                ax.axis("off")
            axes[0].legend(loc="lower right")
            fig.suptitle(title)
            fig.tight_layout()
            PINNTraining.finish(fig, path)
        return axes

    @staticmethod
    def residual_stats(resid, whiskers=(5, 95)):
        out = {}
        for k, r in resid.items():
            a = np.abs(np.asarray(r, float)).ravel()
            q = np.percentile(a, [whiskers[0], 25, 50, 75, whiskers[1]])
            out[k] = {"n": int(a.size), "rms": float(np.sqrt(np.mean(a ** 2))),
                      "whislo": float(q[0]), "q1": float(q[1]), "med": float(q[2]),
                      "q3": float(q[3]), "whishi": float(q[4])}
        return out

    @staticmethod
    def plot_residual(residual, path=None, title="Collocation residual", ax=None):
        own = ax is None
        if own:
            fig, ax = plt.subplots()
        stats = {s: (v if v and isinstance(next(iter(v.values())), dict)
                     else PINNTraining.residual_stats(v))
                 for s, v in residual.items()}
        names = list(next(iter(stats.values())))
        splits = list(stats)
        for i, s in enumerate(splits):
            boxes = [{k: stats[s][n][k] for k in ("med", "q1", "q3", "whislo", "whishi")}
                     for n in names]
            pos = np.arange(len(names)) + (i - (len(splits) - 1) / 2) * 0.8 / len(splits)
            props = {"color": const.SPLIT_COLORS.get(s, "0.3")}
            art = ax.bxp(boxes, positions=pos, widths=0.7 / len(splits), vert=False, showfliers=False,
                         manage_ticks=False, boxprops=props, whiskerprops=props, capprops=props,
                         medianprops=props)
            art["boxes"][0].set_label(s)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([const.RESIDUAL_LABELS.get(n, n) for n in names])
        ax.set_ylim(len(names) - 0.5, -1.3)
        ax.set_xscale("log")
        ax.set_xlabel("|residual|")
        if own:
            ax.set_title(title)
            ax.legend(loc="upper right", ncol=len(splits))
            PINNTraining.finish(fig, path)
        return ax
