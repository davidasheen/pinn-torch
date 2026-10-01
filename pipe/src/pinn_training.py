from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import inspect
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import const
import network


def _short(value, width=90):
    text = repr(value)
    return text if len(text) <= width else text[:width - 3] + "..."


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


class PINNTraining:
    def __init__(
        self,
        fields,
        pde=(),
        known=None,
        domain=None,
        boundary=(),
        num_node=0,
        hidden=const.HIDDEN_LAYERS,
        adam_epochs=const.ADAM_EPOCHS,
        lbfgs_iterations=const.LBFGS_ITERATIONS,
        seed=const.SEED,
        log_every=const.LOG_EVERY,
        record_every=const.RECORD_EVERY,
        checkpoint_path=None,
        checkpoint_every=const.CHECKPOINT_EVERY,
        resume="auto",
        transform=None,
        loss_balance=None,
        log_input=None,
        device=None,
    ):
        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("fields must not be empty")
        self.known = {str(k): float(v) for k, v in (known or {}).items()}
        if domain is None:
            raise ValueError("PINNTraining needs a domain (domain.BoxDomain)")
        self.domain = domain
        self.lo, self.hi = domain.bounds
        self.n_in = int(self.lo.size)
        self.num_node = int(num_node)
        self.transform = transform
        self.loss_balance = self.normalize_balance(loss_balance)
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
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, const.TRAIN_DTYPE)
        self.lbfgs_dtype = getattr(torch, const.LBFGS_DTYPE)
        self.t_construct = time.perf_counter()
        self.pde = [self.normalize_spec(s, "pde") for s in pde]
        self.boundary = [self.normalize_spec(s, "bc") for s in boundary]
        self.n_colloc = self.split_counts(const.NUM_DOMAIN)
        self.n_node = self.split_counts(self.num_node)
        for spec in self.boundary:
            spec["n"] = self.split_counts(spec["num"])
            spec["colloc"] = {}

        torch.manual_seed(self.seed)
        self.model = network.PINN(self.layers, self.lo, self.hi, transform=transform, log_input=self.log_input)
        self.model.to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=const.ADAM_LR)
        self.register_terms()
        self.weights = None if self.loss_balance is None else {k: 1.0 for k in self.term_names}

        raw = dict(layers=list(self.layers), fields=list(self.fields), known=self.known,
                   bounds=[self.lo.tolist(), self.hi.tolist()],
                   pde=[self.describe_spec(s) for s in self.pde],
                   boundary=[self.describe_spec(s) for s in self.boundary],
                   domain=self.domain.describe(), split_pattern=list(const.SPLIT_PATTERN),
                   num_domain=const.NUM_DOMAIN, rar_pool=const.RAR_POOL, resample_every=const.RESAMPLE_EVERY,
                   adam_epochs=self.adam_epochs, adam_lr=const.ADAM_LR, grad_clip=const.GRAD_CLIP,
                   lbfgs_iterations=self.lbfgs_iterations, lbfgs_chunk=const.LBFGS_CHUNK,
                   lbfgs_history=const.LBFGS_HISTORY_SIZE, seed=self.seed,
                   dtype=const.TRAIN_DTYPE, lbfgs_dtype=const.LBFGS_DTYPE,
                   terms={"pde": self.pde_terms, "bc": self.bc_terms},
                   n_colloc=dict(self.n_colloc), n_node=dict(self.n_node),
                   n_boundary={spec["name"]: dict(spec["n"]) for spec in self.boundary},
                   record_every=self.record_every, checkpoint_every=self.checkpoint_every,
                   transform=None if transform is None else getattr(transform, "__qualname__", repr(transform)),
                   loss_balance=self.loss_balance, log_input=self.log_input,
                   backend=f"torch {torch.__version__}")
        self.config = json.loads(json.dumps(raw, default=float))
        self.config["fingerprint"] = self.fingerprint(self.config)
        self.history = {k: [] for k in const.HISTORY_KEYS}
        self.history.update({f"{s}_term_{k}": [] for s in ("train", "val") for k in self.term_names})
        if self.weights is not None:
            self.history.update({f"weight_{k}": [] for k in self.term_names})
        self.best = {"val": float("inf"), "epoch": None, "phase": None, "state": None}
        self.adam_epoch, self.adam_done, self.lbfgs_done = 0, False, False
        self.epoch = 0
        self.colloc = None
        self.result = None
        if self.checkpoint_path is not None and self.resume is not False and self.checkpoint_path.exists():
            ck = torch.load(self.checkpoint_path, map_location=self.device)
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
        out["inputs"] = self.bind(out)
        return out

    def bind(self, spec):
        name = spec["name"]
        inputs = {}
        for pname, param in inspect.signature(spec["residual"]).parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise ValueError(f"{name}: *args/**kwargs are not supported in residual functions")
            if pname in ("F", "X"):
                inputs[pname] = pname
            elif pname in self.known:
                inputs[pname] = "coef"
            else:
                raise ValueError(f"{name}: argument {pname!r} is not F, X, or a known coefficient {tuple(self.known)}")
        if "X" not in inputs:
            raise ValueError(f"{name}: residual must take the point argument X")
        return inputs

    def describe_spec(self, spec):
        out = {"name": spec["name"], "function": getattr(spec["residual"], "__qualname__", repr(spec["residual"])),
               "inputs": dict(spec["inputs"]), "outputs": list(spec["outputs"])}
        if "where" in spec:
            out["num"] = spec["num"]
            out["where"] = spec["where"]
        return out

    def register_terms(self):
        X = torch.as_tensor(np.stack([self.lo + f * (self.hi - self.lo) for f in (0.25, 0.75)]),
                            dtype=self.dtype, device=self.device)
        self.pde_terms, self.bc_terms = [], []
        for specs, prefix, names in ((self.pde, False, self.pde_terms), (self.boundary, True, self.bc_terms)):
            for spec in specs:
                spec["outputs"] = list(self.call_residual(spec, X, prefix))
                names.extend(spec["outputs"])
        names = [*self.pde_terms, *self.bc_terms]
        dup = sorted({k for k in names if names.count(k) > 1})
        if dup:
            raise ValueError(f"residual names must be unique across pde/boundary, duplicated: {dup}")
        self.term_names = names

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
        return out

    @staticmethod
    def split_index(n, pattern):
        label = np.digitize(np.arange(n) % int(sum(pattern)), np.cumsum(pattern)[:-1])
        return tuple(np.flatnonzero(label == i) for i in range(3))

    def split_counts(self, n):
        return {s: int(rows.size) for s, rows in zip(const.SPLITS, self.split_index(n, const.SPLIT_PATTERN))}

    @staticmethod
    def fingerprint(config):
        subset = {k: config.get(k) for k in const.FINGERPRINT_KEYS}
        return hashlib.sha256(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def checkpoint_diff(self, ck):
        cfg = ck.get("config") or {}
        if cfg.get("fingerprint") == self.config["fingerprint"]:
            return []
        diffs = [f"{key}: {_short(cfg.get(key))} -> {_short(self.config.get(key))}"
                 for key in const.FINGERPRINT_KEYS if cfg.get(key) != self.config.get(key)]
        return diffs or ["fingerprint differs"]

    def cast(self, dtype):
        self.dtype = dtype
        self.model.to(dtype)
        if self.colloc is not None:
            self.colloc = {s: x.to(dtype) for s, x in self.colloc.items()}
        for spec in self.boundary:
            spec["colloc"] = {s: x.to(dtype) for s, x in spec["colloc"].items()}

    def field_fn(self, x):
        return dict(zip(self.fields, self.model(x).unbind(dim=1)))

    def call_residual(self, spec, X, prefix):
        X = X.detach().requires_grad_(True)
        kwargs = {p: self.field_fn if kind == "F" else X if kind == "X" else self.known[p]
                  for p, kind in spec["inputs"].items()}
        r = spec["residual"](**kwargs)
        if isinstance(r, dict):
            return {f"{spec['name']}_{k}" if prefix else str(k): v.reshape(X.shape[0], -1) for k, v in r.items()}
        return {spec["name"]: torch.stack([v.reshape(-1) for v in r], dim=1)}

    def residual_pde(self, X):
        res = {}
        for spec in self.pde:
            res.update(self.call_residual(spec, X, False))
        return res

    def residual_bc(self, i, X):
        return self.call_residual(self.boundary[i], X, True)

    def loss(self, split, weights=None):
        terms = {k: (r ** 2).mean() for k, r in self.residual_pde(self.colloc[split]).items()}
        for i, spec in enumerate(self.boundary):
            terms.update({k: (r ** 2).mean() for k, r in self.residual_bc(i, spec["colloc"][split]).items()})
        if weights is None:
            return sum(terms.values()), terms
        return sum(weights[k] * v for k, v in terms.items()), terms

    def eval_loss(self, split):
        total, terms = self.loss(split, self.weights)
        return float(total.detach()), {k: float(v.detach()) for k, v in terms.items()}

    def update_weights(self):
        _, terms = self.loss("train")
        params = list(self.model.parameters())
        norms = {}
        for k, v in terms.items():
            grads = torch.autograd.grad(v, params, retain_graph=True)
            norms[k] = float(torch.cat([g.reshape(-1) for g in grads]).norm())
        total, a = sum(norms.values()), self.loss_balance["alpha"]
        self.weights = {k: a * w + (1.0 - a) * total / norms[k] if norms[k] > 0 else w
                        for k, w in self.weights.items()}

    def generator(self, stream, epoch=0):
        return torch.Generator(device=self.device).manual_seed(self.seed * 10 ** 8 + stream * 10 ** 6 + epoch)

    @staticmethod
    def rar_score(res):
        return sum((r.detach() ** 2).sum(dim=1) for r in res.values())

    @staticmethod
    def rar_pick(score, n, generator):
        return torch.multinomial(score + score.mean(), n, replacement=False, generator=generator)

    def sample_fixed(self):
        g = self.generator(const.SEED_OFFSET_VALIDATION)
        self.colloc = {"valid": self.domain.sample(self.n_node["valid"] + self.n_colloc["valid"], g, self.dtype)}
        for spec in self.boundary:
            spec["colloc"]["valid"] = self.domain.sample_boundary(spec["where"], spec["n"]["valid"], g, self.dtype)

    def resample(self):
        if self.colloc is None:
            self.sample_fixed()
        g = self.generator(const.SEED_OFFSET_COLLOCATION, self.epoch)
        n = self.n_colloc["train"]
        node = self.domain.sample(self.n_node["train"], g, self.dtype)
        cand = self.domain.sample(int(round(n * const.RAR_POOL)), g, self.dtype)
        keep = self.rar_pick(self.rar_score(self.residual_pde(cand)), n, g)
        self.colloc["train"] = torch.cat([node, cand[keep]])
        for i, spec in enumerate(self.boundary):
            g = self.generator(const.SEED_OFFSET_BOUNDARY + i, self.epoch)
            n = spec["n"]["train"]
            cand = self.domain.sample_boundary(spec["where"], int(round(n * const.RAR_POOL)), g, self.dtype)
            spec["colloc"]["train"] = cand[self.rar_pick(self.rar_score(self.residual_bc(i, cand)), n, g)]

    def term_parts(self, terms):
        return sum(terms[k] for k in self.pde_terms), sum(terms[k] for k in self.bc_terms)

    def record(self, phase, total, terms):
        v_total, v_terms = self.eval_loss("valid")
        eligible = self.loss_balance is None or phase == self.loss_balance["select"]
        if eligible and v_total < self.best["val"]:
            state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.best.update(val=v_total, epoch=int(self.epoch), phase=phase, state=state)
        values = (int(self.epoch), phase, total, *self.term_parts(terms), v_total, *self.term_parts(v_terms))
        for k, v in zip(const.HISTORY_KEYS, values):
            self.history[k].append(v)
        for s, tm in (("train", terms), ("val", v_terms)):
            for k, v in tm.items():
                self.history[f"{s}_term_{k}"].append(v)
        for k, v in (self.weights or {}).items():
            self.history[f"weight_{k}"].append(v)

    def show(self, phase, total, terms):
        pde, bc = self.term_parts(terms)
        msg = f"[{phase}] epoch {self.epoch:6d}  loss={total:.4e}  pde={pde:.2e}"
        if self.boundary:
            msg += f"  bc={bc:.2e}"
        if self.weights is not None:
            msg += "  weights=[" + " ".join(f"{v:.3g}" for v in self.weights.values()) + "]"
        msg += f"  elapsed={time.perf_counter() - self.t_construct:.0f}s"
        print(msg, flush=True)

    def maybe_log(self, phase):
        do_record = self.record_every and self.epoch % self.record_every == 0
        do_show = self.log_every and self.epoch % self.log_every == 0
        if not (do_record or do_show):
            return
        total, terms = self.eval_loss("train")
        if do_record:
            self.record(phase, total, terms)
        if do_show:
            self.show(phase, total, terms)

    def save_checkpoint(self, adam_epoch, adam_done, lbfgs_done):
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"config": self.config, "dtype": _dtype_name(self.dtype), "model": self.model.state_dict(),
                 "history": self.history,
                 "train_state": {"adam_epoch": int(adam_epoch), "adam_done": bool(adam_done),
                                 "lbfgs_done": bool(lbfgs_done), "epoch": int(self.epoch),
                                 "optimizer": None if adam_done else self.optimizer.state_dict(),
                                 "best": self.best, "weights": self.weights, "colloc": self.colloc,
                                 "bc_colloc": [spec["colloc"] for spec in self.boundary]}}
        tmp = self.checkpoint_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(self.checkpoint_path)

    def resume_from_checkpoint(self, ck):
        ts = ck["train_state"]
        self.adam_epoch, self.adam_done, self.lbfgs_done = ts["adam_epoch"], ts["adam_done"], ts["lbfgs_done"]
        self.epoch = int(ts["epoch"])
        dtype = getattr(torch, ck["dtype"])
        if dtype != self.dtype:
            self.cast(dtype)
        self.model.load_state_dict(ck["model"])
        self.history = ck["history"]
        self.best = ts["best"]
        if ts["optimizer"] is not None:
            self.optimizer.load_state_dict(ts["optimizer"])
        if ts["weights"] is not None:
            self.weights = dict(ts["weights"])
        if ts["colloc"] is not None:
            self.colloc = ts["colloc"]
            for spec, saved in zip(self.boundary, ts["bc_colloc"]):
                spec["colloc"] = saved
        if self.log_every:
            print(f"[resume] {self.checkpoint_path}: adam_epoch={self.adam_epoch}/{self.adam_epochs} "
                  f"adam_done={self.adam_done} lbfgs_done={self.lbfgs_done}", flush=True)

    def adam_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        total, _ = self.loss("train", self.weights)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=const.GRAD_CLIP)
        self.optimizer.step()

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
                self.adam_step()
                self.maybe_log("adam")
                self.epoch += 1
                if self.checkpoint_every and (e + 1) % self.checkpoint_every == 0:
                    self.save_checkpoint(e + 1, False, False)
        except KeyboardInterrupt:
            self.save_checkpoint(e, False, False)
            print(f"[adam] interrupted at epoch {e}, checkpoint saved; call train again to resume", flush=True)
            raise
        total, terms = self.eval_loss("train")
        self.record("adam", total, terms)
        if self.log_every:
            self.show("adam", total, terms)
        self.adam_epoch, self.adam_done = self.adam_epochs, True
        self.save_checkpoint(self.adam_epochs, True, False)

    def make_lbfgs(self):
        return torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=1,
                                 tolerance_grad=const.LBFGS_TOLERANCE_GRAD, tolerance_change=0.0,
                                 history_size=const.LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe")

    def train_lbfgs(self):
        if self.dtype != self.lbfgs_dtype:
            self.cast(self.lbfgs_dtype)
            if self.log_every:
                print(f"[lbfgs] cast to {_dtype_name(self.dtype)}", flush=True)
        self.resample()
        chunk = min(const.LBFGS_CHUNK, self.lbfgs_iterations)
        step = min(self.record_every, chunk) if self.record_every else chunk
        n_iter = min(max(self.epoch - self.adam_epochs, 0), self.lbfgs_iterations)
        since = 0
        shown = self.epoch
        first = next(self.model.parameters())
        lbfgs = self.make_lbfgs()

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, _ = self.loss("train", self.weights)
            total.backward()
            return total

        try:
            while n_iter < self.lbfgs_iterations:
                if self.record_every:
                    total, terms = self.eval_loss("train")
                    self.record("lbfgs", total, terms)
                    if self.log_every and self.epoch - shown >= self.log_every:
                        shown = self.epoch
                        self.show("lbfgs", total, terms)
                group = lbfgs.param_groups[0]
                group["max_iter"] = min(step, self.lbfgs_iterations - n_iter, chunk - since)
                group["max_eval"] = int(group["max_iter"] * const.LBFGS_MAX_EVAL_FACTOR)
                before = lbfgs.state[first].get("n_iter", 0)
                lbfgs.step(closure)
                done = lbfgs.state[first].get("n_iter", 0) - before
                n_iter += done
                self.epoch += done
                since += done
                if since < chunk and done and n_iter < self.lbfgs_iterations:
                    continue
                since = 0
                if self.checkpoint_every:
                    self.save_checkpoint(self.adam_epochs, True, False)
                if not done:
                    print("[lbfgs] gradient below tolerance, stopping", flush=True)
                    break
                if n_iter < self.lbfgs_iterations:
                    self.resample()
                    lbfgs = self.make_lbfgs()
        except KeyboardInterrupt:
            self.save_checkpoint(self.adam_epochs, True, False)
            print("[lbfgs] interrupted, checkpoint saved; call train again to resume "
                  "(L-BFGS restarts from the saved weights)", flush=True)
            raise
        total, terms = self.eval_loss("train")
        self.record("lbfgs", total, terms)
        if self.log_every:
            self.show("lbfgs", total, terms)
            print(f"[lbfgs] stopped after {n_iter} iterations", flush=True)
        self.lbfgs_done = True

    def restore_best(self):
        if self.best["state"] is None:
            return False
        self.model.load_state_dict(self.best["state"])
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
        total, terms = self.eval_loss("train")
        self.result = {"weights": self.weights,
                       "history": self.history,
                       "best": {"epoch": self.best["epoch"], "phase": self.best["phase"],
                                "val": None if self.best["state"] is None else self.best["val"],
                                "restored": restored},
                       "loss": total,
                       "terms": terms,
                       "wall_seconds": time.perf_counter() - t_start,
                       "config": self.config}
        return self.result

    def evaluate(self, coords):
        X = torch.as_tensor(np.asarray(coords, dtype=float), dtype=self.dtype, device=self.device)
        with torch.no_grad():
            y = self.model(X).cpu().numpy()
        return {name: y[:, i] for i, name in enumerate(self.fields)}

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
