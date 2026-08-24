from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import const
import network
import pde_residual


class PINNTraining:
    def __init__(
        self,
        dataset,
        inverse=None,
        known=None,
        bounds=None,
        boundary=(),
        group=None,
        hidden=const.HIDDEN_LAYERS,
        split_pattern=const.SPLIT_PATTERN,
        num_domain=const.NUM_DOMAIN,
        rar_pool=const.RAR_POOL,
        rar_chunk=const.RAR_CHUNK,
        resample_every=const.RESAMPLE_EVERY,
        adam_iterations=const.ADAM_ITERATIONS,
        adam_lr=const.ADAM_LR,
        grad_clip=const.GRAD_CLIP,
        lbfgs_iterations=const.LBFGS_ITERATIONS,
        dtype=getattr(torch, const.TRAIN_DTYPE),
        lbfgs_dtype=getattr(torch, const.LBFGS_DTYPE),
        device=None,
        seed=const.SEED,
        log_every=const.LOG_EVERY,
        record_every=const.RECORD_EVERY,
        checkpoint_path=None,
        checkpoint_every=const.CHECKPOINT_EVERY,
        resume=True,
        restore_best=True,
    ):
        self.dataset = dataset
        self.inverse = {k: float(v) for k, v in (inverse or {}).items()}
        self.known = {k: float(v) for k, v in (known or {}).items()}
        self.bounds = bounds
        self.group = None if group is None else np.asarray(group).reshape(-1)
        self.boundary = []
        for spec in boundary:
            if spec["side"] not in const.BC_SIDES:
                raise ValueError(f"boundary {spec['name']!r}: side must be one of {const.BC_SIDES}, "
                                 f"got {spec['side']!r}")
            self.boundary.append({"name": str(spec["name"]), "axis": int(spec["axis"]), "side": str(spec["side"]),
                                  "residual": spec["residual"], "num": int(spec.get("num", const.NUM_BOUNDARY))})
        self.hidden = tuple(int(h) for h in hidden)
        self.split_pattern = const.SPLIT_PATTERN if split_pattern is None else split_pattern
        self.num_domain = int(num_domain)
        self.rar_pool = float(rar_pool)
        self.rar_chunk = rar_chunk
        self.resample_every = int(resample_every)
        self.adam_iterations = int(adam_iterations)
        self.adam_lr = float(adam_lr)
        self.grad_clip = grad_clip
        self.lbfgs_iterations = int(lbfgs_iterations)
        self.train_dtype = dtype
        self.dtype = dtype
        self.lbfgs_dtype = lbfgs_dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = int(seed)
        self.log_every = int(log_every)
        self.record_every = int(record_every)
        self.checkpoint_path = None if checkpoint_path is None else pathlib.Path(checkpoint_path)
        self.checkpoint_every = int(checkpoint_every)
        self.resume = bool(resume)
        self.restore_best = bool(restore_best)
        self.fields = tuple(f for f in const.FIELD_NAMES if f in dataset)
        self.layers = (4, *self.hidden, len(self.fields))

        self.load_data()
        self.initialize_model()

    @staticmethod
    def split_index(n, pattern, group=None):
        pattern = tuple(int(v) for v in pattern)
        if len(pattern) != 3 or min(pattern) < 1:
            raise ValueError(f"pattern must be three positive ints, got {pattern!r}")
        key = np.arange(n) if group is None else np.asarray(group).reshape(-1)
        if key.size != n:
            raise ValueError(f"group has {key.size} entries but expects {n}")
        if key.dtype.kind == "f":
            key = np.round(key, const.SPLIT_ROUND_DECIMALS)
        uniq, inv = np.unique(key, return_inverse=True)
        label = np.digitize(np.arange(uniq.size) % int(sum(pattern)),
                            np.cumsum(pattern)[:-1])[inv.reshape(-1)]
        return tuple(np.flatnonzero(label == i) for i in range(3))

    def split_data(self):
        self.split_pattern = tuple(int(v) for v in self.split_pattern)
        n = int(np.shape(self.dataset["X"])[0])
        splits = []
        for name, rows in zip(const.SPLITS, self.split_index(n, self.split_pattern, self.group)):
            if rows.size == 0:
                raise ValueError(f"split {name!r} is empty under pattern {self.split_pattern}")
            splits.append({k: (np.asarray(v)[rows] if np.ndim(v) >= 1 and np.shape(v)[0] == n else v)
                           for k, v in self.dataset.items()})
        self.train_data, self.valid_data, self.test_data = splits

    def to_torch(self, d):
        out = {"X": torch.as_tensor(np.asarray(d["X"], dtype=float), dtype=self.dtype, device=self.device)}
        for f in self.fields:
            out[f] = torch.as_tensor(np.asarray(d[f], dtype=float).reshape(-1, 1), dtype=self.dtype, device=self.device)
        return out

    def sample_box(self, n, generator=None):
        g = self.generator if generator is None else generator
        return self.lo + (self.hi - self.lo) * torch.rand((n, 4), generator=g, dtype=self.dtype, device=self.device)

    def split_rows(self, n):
        idx = self.split_index(n, self.split_pattern)
        return {s: torch.as_tensor(rows, dtype=torch.long, device=self.device) for s, rows in zip(const.SPLITS, idx)}

    def rar_score(self, coords):
        res = self.residual_pde(coords, self.model(coords))
        return sum(r.detach() ** 2 for r in res.values()).flatten()

    def resample(self):
        pool = self.sample_box(self.n_pool)
        parts = {s: pool[r] for s, r in self.pool_rows.items()}
        if self.rar_pool > 1:
            cand = parts["train"]
            with torch.enable_grad():
                if self.rar_chunk:
                    step = int(self.rar_chunk)
                    r = torch.cat([self.rar_score(cand[lo:lo + step].detach().requires_grad_(True))
                                   for lo in range(0, cand.shape[0], step)])
                else:
                    r = self.rar_score(cand.detach().requires_grad_(True))
            keep = torch.multinomial(r + r.mean(), self.target_n["train"], replacement=False,
                                     generator=self.generator)
            parts["train"] = cand[keep]
            parts["valid"] = parts["valid"][:self.target_n["valid"]]
            parts["test"] = parts["test"][:self.target_n["test"]]
        self.colloc = {s: p.detach() for s, p in parts.items()}
        for spec in self.boundary:
            base = self.sample_box(spec["num"], spec["generator"])
            faces = {}
            for side in (("lo", "hi") if spec["side"] == "pair" else (spec["side"],)):
                X = base.clone()
                X[:, spec["axis"]] = (self.lo if side == "lo" else self.hi)[spec["axis"]]
                faces[side] = X.detach()
            if spec["side"] == "pair":
                spec["colloc"] = {s: (faces["lo"][r], faces["hi"][r]) for s, r in spec["rows"].items()}
            else:
                spec["colloc"] = {s: faces[spec["side"]][r] for s, r in spec["rows"].items()}

    def load_data(self):
        self.split_data()
        self.train_torch = self.to_torch(self.train_data)
        self.valid_torch = self.to_torch(self.valid_data)
        if self.bounds is None:
            X = np.asarray(self.train_data["X"], dtype=float)
            self.bounds = (X.min(axis=0), X.max(axis=0))
        self.lo = torch.as_tensor(np.asarray(self.bounds[0], dtype=float), dtype=self.dtype, device=self.device)
        self.hi = torch.as_tensor(np.asarray(self.bounds[1], dtype=float), dtype=self.dtype, device=self.device)
        self.n_pool = int(round(self.num_domain * self.rar_pool)) if self.rar_pool > 1 else self.num_domain
        self.pool_rows = self.split_rows(self.n_pool)
        self.target_n = {s: int(r.numel()) for s, r in self.split_rows(self.num_domain).items()}
        for spec in self.boundary:
            spec["rows"] = self.split_rows(spec["num"])
        self.colloc = None

    def initialize_model(self):
        torch.manual_seed(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed + const.SEED_OFFSET_COLLOCATION)
        for i, spec in enumerate(self.boundary):
            spec["generator"] = torch.Generator(device=self.device).manual_seed(
                self.seed + const.SEED_OFFSET_BOUNDARY + i)
        self.model = network.FNN(self.layers).to(device=self.device, dtype=self.dtype)
        self.params = {k: nn.Parameter(torch.tensor(v, dtype=self.dtype, device=self.device))
                       for k, v in self.inverse.items()}
        self.optimizer = torch.optim.Adam(self.all_params, lr=self.adam_lr)
        self.config = dict(layers=list(self.layers), fields=list(self.fields), inverse=self.inverse, known=self.known,
                           bounds=[self.lo.cpu().numpy().tolist(), self.hi.cpu().numpy().tolist()],
                           boundary=[{k: spec[k] for k in ("name", "axis", "side", "num")} for spec in self.boundary],
                           split_pattern=list(self.split_pattern), num_domain=self.num_domain,
                           rar_pool=self.rar_pool, rar_chunk=self.rar_chunk, resample_every=self.resample_every,
                           n_train=int(self.train_torch["X"].shape[0]), n_valid=int(self.valid_torch["X"].shape[0]),
                           n_test=int(np.shape(self.test_data["X"])[0]),
                           n_colloc=dict(self.target_n),
                           n_boundary={spec["name"]: {s: int(r.numel()) for s, r in spec["rows"].items()}
                                       for spec in self.boundary},
                           adam_iterations=self.adam_iterations, adam_lr=self.adam_lr, grad_clip=self.grad_clip,
                           lbfgs_iterations=self.lbfgs_iterations,
                           dtype=str(self.train_dtype).replace("torch.", ""),
                           lbfgs_dtype=str(self.lbfgs_dtype).replace("torch.", ""),
                           seed=self.seed, record_every=self.record_every, checkpoint_every=self.checkpoint_every,
                           restore_best=self.restore_best)
        self.history = {"iteration": [], "phase": [], "train_total": [], "train_data": [], "train_pde": [],
                        "train_bc": [], "val_total": [], "val_data": [], "val_pde": [], "val_bc": [],
                        **{k: [] for k in self.params}}
        self.best = {"val": float("inf"), "iteration": None, "phase": None, "state": None, "params": None}
        self.adam_it, self.adam_done, self.lbfgs_done = 0, False, False
        self.it = 0
        self.result = None
        if self.checkpoint_path is not None and self.resume and self.checkpoint_path.exists():
            self.resume_from_checkpoint()

    @property
    def all_params(self):
        return list(self.model.parameters()) + list(self.params.values())

    def cast(self, new_dtype):
        self.dtype = new_dtype
        self.model.to(dtype=new_dtype)
        self.params = {k: nn.Parameter(p.detach().to(new_dtype)) for k, p in self.params.items()}
        self.train_torch = self.to_torch(self.train_data)
        self.valid_torch = self.to_torch(self.valid_data)
        self.lo = self.lo.to(new_dtype)
        self.hi = self.hi.to(new_dtype)
        if self.colloc is not None:
            self.colloc = {s: c.to(new_dtype) for s, c in self.colloc.items()}
        for spec in self.boundary:
            if "colloc" in spec:
                spec["colloc"] = {s: (tuple(x.to(new_dtype) for x in c) if isinstance(c, tuple) else c.to(new_dtype))
                                  for s, c in spec["colloc"].items()}

    def coef(self, name):
        return self.params[name] if name in self.params else self.known[name]

    def fields_of(self, y):
        return {f: y[:, i:i + 1] for i, f in enumerate(self.fields)}

    def residual_pde(self, coords, y):
        f = self.fields_of(y)
        continuity, mom_u, mom_v, mom_w = pde_residual.navier_stokes_residual(
            coords, f["u"], f["v"], f["w"], f["p"], self.coef("rho"), self.coef("mu"))
        res = {"continuity": continuity, "momentum_u": mom_u, "momentum_v": mom_v, "momentum_w": mom_w}
        if len(self.fields) == 6:
            energy, T_lap = pde_residual.energy_residual(coords, f["T"], self.coef("alpha"))
            res["energy"] = energy
            res["species"] = pde_residual.species_residual(coords, f["Y"], T_lap, f["u"], f["v"], f["w"],
                                                           self.coef("rho"), self.coef("D"), self.coef("DT"))
        return res

    def residual_bc(self, spec, split):
        colloc = spec["colloc"][split]
        if spec["side"] == "pair":
            X_lo = colloc[0].clone().requires_grad_(True)
            X_hi = colloc[1].clone().requires_grad_(True)
            r = spec["residual"](X_lo, self.fields_of(self.model(X_lo)), X_hi, self.fields_of(self.model(X_hi)))
        else:
            X = colloc.clone().requires_grad_(True)
            r = spec["residual"](X, self.fields_of(self.model(X)))
        if isinstance(r, dict):
            return r
        if isinstance(r, (list, tuple)):
            r = torch.cat(list(r), dim=1)
        return {spec["name"]: r}

    def loss(self, data, colloc, split):
        n = data["X"].shape[0]
        coords = torch.cat([data["X"], colloc], dim=0).requires_grad_(True)
        y = self.model(coords)
        terms = {"data": sum(torch.mean((y[:n, i:i + 1] - data[f]) ** 2) for i, f in enumerate(self.fields))}
        res = self.residual_pde(coords, y)
        terms["continuity"] = torch.mean(res["continuity"] ** 2)
        terms["momentum"] = sum(torch.mean(res[k] ** 2) for k in ("momentum_u", "momentum_v", "momentum_w"))
        if "energy" in res:
            terms["energy"] = torch.mean(res["energy"] ** 2)
            terms["species"] = torch.mean(res["species"] ** 2)
        if self.boundary:
            terms["bc"] = sum(torch.mean(r ** 2) for spec in self.boundary
                              for r in self.residual_bc(spec, split).values())
        total = sum(terms.values())
        return total, terms

    def record(self, it, phase, total, terms):
        with torch.enable_grad():
            v_total, v_terms = self.loss(self.valid_torch, self.colloc["valid"], "valid")
        vt, vd = float(v_total.detach()), float(v_terms["data"].detach())
        vb = float(v_terms["bc"].detach()) if "bc" in v_terms else 0.0
        if vt < self.best["val"]:
            self.best.update(val=vt, iteration=int(it), phase=phase,
                             state={k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()},
                             params={k: float(p.detach()) for k, p in self.params.items()})
        t_total, t_data = float(total.detach()), float(terms["data"].detach())
        t_bc = float(terms["bc"].detach()) if "bc" in terms else 0.0
        self.history["iteration"].append(int(it))
        self.history["phase"].append(phase)
        self.history["train_total"].append(t_total)
        self.history["train_data"].append(t_data)
        self.history["train_pde"].append(t_total - t_data - t_bc)
        self.history["train_bc"].append(t_bc)
        self.history["val_total"].append(vt)
        self.history["val_data"].append(vd)
        self.history["val_pde"].append(vt - vd - vb)
        self.history["val_bc"].append(vb)
        for k, p in self.params.items():
            self.history[k].append(float(p.detach()))

    def show(self, it, phase, total, terms):
        msg = (f"[{phase}] iter {it:6d}  loss={float(total.detach()):.4e}  "
               f"data={float(terms['data'].detach()):.2e}")
        if "bc" in terms:
            msg += f"  bc={float(terms['bc'].detach()):.2e}"
        for k, p in self.params.items():
            msg += f"  {k}={float(p.detach()):.5f}"
        print(msg, flush=True)

    def maybe_log(self, k, it, phase, total, terms):
        if self.record_every and k % self.record_every == 0:
            self.record(it, phase, total, terms)
        if self.log_every and k % self.log_every == 0:
            self.show(it, phase, total, terms)

    def save_checkpoint(self, adam_it, adam_done, lbfgs_done, optimizer_state=None):
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(),
                    "layers": list(self.layers),
                    "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
                    "params": {k: float(p.detach()) for k, p in self.params.items()},
                    "known": dict(self.known),
                    "history": self.history,
                    "config": self.config,
                    "train_state": {"adam_it": int(adam_it), "adam_done": bool(adam_done),
                                    "lbfgs_done": bool(lbfgs_done), "optimizer_state_dict": optimizer_state,
                                    "best": self.best,
                                    "generator_state": self.generator.get_state(),
                                    "bc_generator_state": [spec["generator"].get_state()
                                                           for spec in self.boundary],
                                    "colloc": None if self.colloc is None
                                    else {s: c.cpu() for s, c in self.colloc.items()},
                                    "bc_colloc": [{s: (tuple(x.cpu() for x in c) if isinstance(c, tuple)
                                                       else c.cpu()) for s, c in spec["colloc"].items()}
                                                  if "colloc" in spec else None
                                                  for spec in self.boundary]}}, self.checkpoint_path)

    def resume_from_checkpoint(self):
        ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        for key in ("layers", "fields", "inverse", "known", "num_domain", "boundary", "adam_iterations",
                    "rar_pool", "rar_chunk", "resample_every"):
            if ck["config"].get(key) != self.config[key]:
                raise ValueError(f"checkpoint {key}={ck['config'].get(key)!r} does not match "
                                 f"{self.config[key]!r}; pass resume=False to start fresh")
        ts = ck["train_state"]
        self.adam_it, self.adam_done, self.lbfgs_done = ts["adam_it"], ts["adam_done"], ts["lbfgs_done"]
        if ck["dtype"] != str(self.dtype).replace("torch.", ""):
            self.cast(getattr(torch, ck["dtype"]))
        self.model.load_state_dict(ck["model_state_dict"])
        with torch.no_grad():
            for k, p in self.params.items():
                p.copy_(torch.tensor(ck["params"][k], dtype=p.dtype, device=self.device))
        self.history = ck["history"]
        if ts.get("best") is not None:
            self.best.update(ts["best"])
        if not self.adam_done and ts.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ts["optimizer_state_dict"])
        if ts.get("generator_state") is not None:
            self.generator.set_state(ts["generator_state"])
        for spec, state in zip(self.boundary, ts.get("bc_generator_state") or []):
            spec["generator"].set_state(state)
        if ts.get("colloc") is not None:
            self.colloc = {s: c.to(device=self.device) for s, c in ts["colloc"].items()}
        for spec, saved in zip(self.boundary, ts.get("bc_colloc") or []):
            if saved is not None:
                spec["colloc"] = {s: (tuple(x.to(device=self.device) for x in c) if isinstance(c, tuple)
                                      else c.to(device=self.device)) for s, c in saved.items()}
        if self.log_every:
            print(f"[resume] {self.checkpoint_path}: adam_it={self.adam_it}/{self.adam_iterations} "
                  f"adam_done={self.adam_done} lbfgs_done={self.lbfgs_done}", flush=True)

    @staticmethod
    def load_checkpoint(path, device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(path, map_location=device, weights_only=False)
        model = network.FNN(state["layers"]).to(device=device, dtype=getattr(torch, state["dtype"]))
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model, state

    def train_adam(self):
        k = self.adam_it
        if self.colloc is None:
            self.resample()
        try:
            for k in range(self.adam_it, self.adam_iterations):
                if self.resample_every and k > 0 and k % self.resample_every == 0:
                    self.resample()
                self.optimizer.zero_grad(set_to_none=True)
                total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
                total.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                self.maybe_log(k, self.it, "adam", total, terms)
                self.it += 1
                if self.checkpoint_every and (k + 1) % self.checkpoint_every == 0:
                    self.save_checkpoint(k + 1, False, False, self.optimizer.state_dict())
        except KeyboardInterrupt:
            self.save_checkpoint(k, False, False, self.optimizer.state_dict())
            print(f"[adam] interrupted at iter {k}, checkpoint saved; call train again to resume", flush=True)
            raise
        total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        self.record(self.it, "adam", total, terms)
        if self.log_every:
            self.show(self.it, "adam", total, terms)
        self.adam_it, self.adam_done = self.adam_iterations, True
        self.save_checkpoint(self.adam_iterations, True, False)

    def train_lbfgs(self):
        if self.lbfgs_dtype != self.dtype:
            self.cast(self.lbfgs_dtype)
            if self.log_every:
                print(f"[lbfgs] cast to {str(self.dtype).replace('torch.', '')}", flush=True)
        self.resample()
        all_params = self.all_params
        lbfgs = torch.optim.LBFGS(
            all_params, lr=1.0, max_iter=self.lbfgs_iterations,
            max_eval=int(self.lbfgs_iterations * const.LBFGS_MAX_EVAL_FACTOR),
            tolerance_grad=const.LBFGS_TOLERANCE_GRAD, tolerance_change=0.0,
            history_size=const.LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe")
        calls = {"n": 0, "last_ckpt": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
            total.backward()
            calls["n"] += 1
            self.maybe_log(calls["n"], self.it + calls["n"], "lbfgs", total, terms)
            n_iter = lbfgs.state[all_params[0]].get("n_iter", 0)
            if self.checkpoint_every and n_iter - calls["last_ckpt"] >= self.checkpoint_every:
                calls["last_ckpt"] = n_iter
                self.save_checkpoint(self.adam_iterations, True, False)
            return total

        try:
            lbfgs.step(closure)
        except KeyboardInterrupt:
            self.save_checkpoint(self.adam_iterations, True, False)
            print("[lbfgs] interrupted, checkpoint saved; call train again to resume "
                  "(L-BFGS restarts from the saved weights)", flush=True)
            raise
        self.it += calls["n"]
        total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        self.record(self.it, "lbfgs", total, terms)
        if self.log_every:
            self.show(self.it, "lbfgs", total, terms)
        self.lbfgs_done = True

    def restore_best_snapshot(self):
        if not self.restore_best or self.best["state"] is None:
            return False
        self.model.load_state_dict({k: v.to(dtype=self.dtype, device=self.device)
                                    for k, v in self.best["state"].items()})
        with torch.no_grad():
            for k, p in self.params.items():
                p.copy_(torch.tensor(self.best["params"][k], dtype=p.dtype, device=self.device))
        if self.log_every:
            print(f"[best] restored iteration {self.best['iteration']} ({self.best['phase']}) "
                  f"val_total={self.best['val']:.4e}", flush=True)
        return True

    def train(self):
        t_start = time.perf_counter()
        self.it = self.adam_it if not self.adam_done else max(self.adam_iterations, *self.history["iteration"], 0)
        if not self.adam_done:
            self.train_adam()
        if self.lbfgs_iterations and not self.lbfgs_done:
            self.train_lbfgs()
        self.lbfgs_done = self.lbfgs_done or not self.lbfgs_iterations
        restored = self.restore_best_snapshot()
        self.save_checkpoint(self.adam_iterations, True, self.lbfgs_done)
        total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        self.result = {"model": self.model,
                       "params": {k: float(p.detach()) for k, p in self.params.items()},
                       "known": self.known,
                       "history": self.history,
                       "best": {"iteration": self.best["iteration"], "phase": self.best["phase"],
                                "val": None if self.best["state"] is None else self.best["val"],
                                "restored": restored},
                       "loss": float(total.detach()),
                       "terms": {k: float(v.detach()) for k, v in terms.items()},
                       "wall_seconds": time.perf_counter() - t_start,
                       "config": self.config,
                       "train_state": {"adam_it": self.adam_iterations, "adam_done": True,
                                       "lbfgs_done": bool(self.lbfgs_done), "optimizer_state_dict": None,
                                       "best": self.best}}
        return self.result

    @staticmethod
    def evaluation(model, coords, fields):
        param = next(model.parameters())
        coords = np.asarray(coords, dtype=float)
        with torch.no_grad():
            y = model(torch.as_tensor(coords, dtype=param.dtype, device=param.device))
        y = y.detach().cpu().numpy()
        return {name: y[:, i] for i, name in enumerate(fields)}

    def evaluate(self, coords):
        return self.evaluation(self.model, coords, self.fields)

    def evaluate_residual(self, split):
        if self.colloc is None:
            self.resample()
        with torch.enable_grad():
            coords = self.colloc[split].clone().requires_grad_(True)
            res = self.residual_pde(coords, self.model(coords))
            out = {k: v.detach().cpu().numpy().ravel() for k, v in res.items()}
            for spec in self.boundary:
                for k, r in self.residual_bc(spec, split).items():
                    out[k] = r.detach().cpu().numpy().ravel()
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

    def diag_record(self, split="test"):
        data = {"train": self.train_data, "valid": self.valid_data, "test": self.test_data}[split]
        pred = self.evaluate(data["X"])
        return {f: self.error_record(data[f], pred[f]) for f in self.fields if f in data}

    def plot_all(self, out_dir, prefix="", fields=None, grid=None):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        head = f"{prefix}_" if prefix else ""
        self.plot_loss(self.history, out_dir / f"{head}loss.png", title=f"{prefix} loss history".strip(),
                       best_iteration=self.best["iteration"])
        records = self.diag_record("test")
        fields = list(records) if fields is None else list(fields)
        self.plot_diag(records, fields, grid, out_dir / f"{head}diag.png",
                       title=f"{prefix} test diagonal".strip())
        self.plot_residual({s: self.evaluate_residual(s) for s in const.SPLITS},
                           out_dir / f"{head}residual.png",
                           title=f"{prefix} collocation residual".strip())
        return records

    @staticmethod
    def finish(fig, path=None):
        if path is not None:
            fig.savefig(path)
        if plt.get_backend().lower() != "agg":
            plt.show()

    @staticmethod
    def plot_loss(history, path=None, title="Loss history", best_iteration=None, ax=None):
        own = ax is None
        if own:
            fig, ax = plt.subplots()
        it = np.asarray(history["iteration"])
        vt = np.asarray([np.nan if v is None else v for v in history["val_total"]], float)
        ax.semilogy(it, history["train_total"], color="tab:blue", label="train total")
        ax.semilogy(it, vt, color="tab:orange", label="validation total")
        adam = [i for i, ph in zip(it, history["phase"]) if ph == "adam"]
        if adam and max(adam) < it.max():
            ax.axvline(max(adam), color="0.5", lw=1, ls=":", label="Adam -> L-BFGS")
        if best_iteration is not None:
            ax.axvline(best_iteration, color="tab:red", lw=1, ls="--", label="best epoch")
        ax.set_xlabel("iteration")
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
            ax.bxp(boxes, positions=pos, widths=0.7 / len(splits), vert=False, showfliers=False,
                   manage_ticks=False, boxprops=props, whiskerprops=props, capprops=props,
                   medianprops=props)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([const.RESIDUAL_LABELS.get(n, n) for n in names])
        ax.set_ylim(len(names) - 0.5, -1.3)
        ax.set_xscale("log")
        ax.set_xlabel("|residual|")
        if own:
            ax.set_title(title)
            ax.legend(handles=[plt.Line2D([], [], color=const.SPLIT_COLORS.get(s, "0.3"), label=s)
                               for s in splits], loc="upper right", ncol=len(splits))
            PINNTraining.finish(fig, path)
        return ax
