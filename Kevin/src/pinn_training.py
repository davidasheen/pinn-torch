from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import inspect
import json
import math
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

from . import const
from . import domain
from . import network

def map_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """
    Recursively applies a function to all torch tensors within a nested structure.

    Args:
        obj: The object to process (tensor, dict, list, or tuple).
        fn: The function to apply to each tensor.

    Returns:
        The object with tensors transformed by fn.
    """
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(map_tensors(v, fn) for v in obj)
    if isinstance(obj, list):
        return [map_tensors(v, fn) for v in obj]
    return obj

def _short(value: Any, width: int = 90) -> str:
    """
    Shortens a string representation of a value for logging.

    Args:
        value: The value to shorten.
        width: Maximum length of the resulting string.

    Returns:
        A shortened string representation.
    """
    text = repr(value)
    return text if len(text) <= width else text[:width - 3] + "..."

class PINNTraining:
    """
    Orchestrates the training of a Physics-Informed Neural Network (PINN).

    Handles data splitting, collocation point sampling, optimization (Adam and L-BFGS),
    and checkpointing for both forward and inverse problems.
    """

    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        fields: Optional[List[str]] = None,
        pde: Tuple[Union[Callable, Dict], ...] = (),
        inverse: Optional[Dict[str, Union[float, Dict]]] = None,
        known: Optional[Dict[str, float]] = None,
        domain: domain.MeshDomain = None,
        boundary: Tuple[Union[Callable, Dict], ...] = (),
        group: Optional[np.ndarray] = None,
        sigma: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        hidden: Tuple[int, ...] = const.HIDDEN_LAYERS,
        feature: Optional[Callable] = None,
        num_node: Optional[int] = None,
        num_domain: Optional[int] = None,
        rar_pool: Optional[float] = None,
        num_centers: int = 11,
        batch_size: Optional[int] = const.BATCH_SIZE,
        adam_epochs: int = const.ADAM_EPOCHS,
        lbfgs_iterations: int = const.LBFGS_ITERATIONS,
        lbfgs_chunk: int = const.LBFGS_CHUNK,
        dtype: torch.dtype = getattr(torch, const.TRAIN_DTYPE),
        device: Optional[str] = None,
        seed: int = const.SEED,
        log_every: int = const.LOG_EVERY,
        record_every: int = const.RECORD_EVERY,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = const.CHECKPOINT_EVERY,
        resume: str = "auto",
    ):
        """
        Initializes the PINNTraining instance.

        Args:
            dataset: Dictionary containing 'X' and target fields.
            fields: List of target field names. Defaults to names in dataset.
            pde: Tuple of PDE residual specifications.
            inverse: Dictionary of parameters to be recovered (inverse problem).
            known: Dictionary of fixed physical parameters.
            domain: The domain object for sampling collocation points.
            boundary: Tuple of boundary condition specifications.
            group: Array for guided data splitting.
            sigma: Weighting factors for each field in the data loss.
            weights: Weights for specific residual terms.
            hidden: Architecture of the hidden layers.
            feature: Optional feature encoding function.
            num_node: Number of nodes to use in collocation.
            num_domain: Number of collocation points to sample.
            rar_pool: RAR sampling pool size.
            num_centers: Number of centers for RBF encoding.
            batch_size: Training batch size.
            adam_epochs: Number of Adam epochs.
            lbfgs_iterations: Maximum L-BFGS iterations.
            lbfgs_chunk: L-BFGS checkpointing chunk size.
            dtype: Torch dtype for training.
            device: Computation device.
            seed: Random seed.
            log_every: Epoch interval for logging.
            record_every: Epoch interval for recording history.
            checkpoint_path: Path to save/load checkpoints.
            checkpoint_every: Epoch interval for saving checkpoints.
            resume: Resume mode ("auto", True, False).
        """
        self.dataset = dataset
        self.fields = tuple(fields) if fields is not None else tuple(f for f in const.FIELD_NAMES if f in dataset)
        missing = [f for f in self.fields if f not in dataset]
        if not self.fields or missing:
            raise ValueError(f"fields {missing or '()'} not in dataset; dataset keys: {sorted(dataset)}")
        self.inverse = {}
        for name, spec in (inverse or {}).items():
            spec = dict(spec) if isinstance(spec, dict) else {"init": spec}
            self.inverse[str(name)] = {"init": float(spec["init"]),
                                       "bounds": None if spec.get("bounds") is None
                                       else (float(spec["bounds"][0]), float(spec["bounds"][1])),
                                       "log": bool(spec.get("log", False))}
        self.known = {str(k): float(v) for k, v in (known or {}).items()}
        both = sorted(set(self.inverse) & set(self.known))
        if both:
            raise ValueError(f"{both} given as both inverse and known")
        bad = sorted(set(sigma or {}) - set(self.fields))
        if bad:
            raise ValueError(f"sigma for unknown fields {bad}; fields are {self.fields}")
        self.sigma = {f: float((sigma or {}).get(f, 1.0)) for f in self.fields}
        self.weights = {str(k): float(v) for k, v in (weights or {}).items()}
        self.pde_weight = float(const.PDE_WEIGHT)
        self.pde = [self.normalize_spec(s, "pde") for s in pde]
        self.boundary = [self.normalize_spec(s, "bc") for s in boundary]
        if domain is None:
            raise ValueError("PINNTraining needs a domain (domain.MeshDomain)")
        self.domain = domain
        self.group = None if group is None else np.asarray(group).reshape(-1)
        self.hidden = tuple(int(h) for h in hidden)
        self.split_pattern = const.SPLIT_PATTERN
        self.num_domain = int(num_domain) if num_domain is not None else int(const.NUM_DOMAIN)
        self.num_node = None if num_node is None else int(num_node)
        self.rar_pool = float(rar_pool) if rar_pool is not None else float(const.RAR_POOL)
        self.rar_chunk = const.RAR_CHUNK
        self.rar_boundary = bool(const.RAR_BOUNDARY)
        self.resample_every = int(const.RESAMPLE_EVERY)
        self.num_centers = int(num_centers)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.adam_epochs = int(adam_epochs)
        self.adam_lr = float(const.ADAM_LR)
        self.inverse_lr = float(const.INVERSE_LR)
        self.grad_clip = const.GRAD_CLIP
        self.lbfgs_iterations = int(lbfgs_iterations)
        self.lbfgs_chunk = int(lbfgs_chunk or 0)
        self.lbfgs_bound_tol = float(const.LBFGS_BOUND_TOL)
        self.lbfgs_resample = bool(const.LBFGS_RESAMPLE)
        self.train_dtype = dtype
        self.dtype = dtype
        self.lbfgs_dtype = getattr(torch, const.LBFGS_DTYPE)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = int(seed)
        self.deterministic = bool(const.DETERMINISTIC)
        self.log_every = int(log_every)
        self.record_every = int(record_every)
        self.checkpoint_path = None if checkpoint_path is None else pathlib.Path(checkpoint_path)
        self.checkpoint_every = int(checkpoint_every)
        if resume not in const.RESUME_MODES:
            raise ValueError(f"resume must be one of {const.RESUME_MODES}, got {resume!r}")
        self.resume = resume
        self.t_construct = time.perf_counter()
        self.dim = int(np.shape(dataset["X"])[1])
        self.feature = feature
        self.n_feature = 0 if feature is None else int(feature(torch.zeros(1, self.dim, dtype=torch.float64)).shape[1])
        self.layers = (self.dim + self.n_feature, *self.hidden, len(self.fields))

        self.load_data()
        self.initialize_model()

    def normalize_spec(self, spec: Union[Callable, Dict], kind: str) -> Dict[str, Any]:
        """
        Normalizes a PDE or boundary specification into a standard dictionary format.

        Args:
            spec: The specification (callable or dict).
            kind: The type of specification ("pde" or "bc").

        Returns:
            A normalized specification dictionary.
        """
        if callable(spec):
            spec = {"residual": spec}
        spec = dict(spec)
        fn = spec["residual"]
        name = str(spec.get("name") or getattr(fn, "__name__", "residual"))
        out = {"name": name, "residual": fn, "map": {str(k): str(v) for k, v in (spec.get("map") or {}).items()},
               "scale": spec.get("scale", 1.0), "weight": spec.get("weight", 1.0),
               "table": {str(k): np.asarray(v) for k, v in (spec.get("table") or {}).items()}}
        if kind == "bc":
            out["num"] = int(spec.get("num", const.NUM_BOUNDARY))
            if "faces" in spec:
                out["where"] = {"faces": tuple(sorted({int(f) for f in np.atleast_1d(spec["faces"])}))}
            elif "axis" in spec:
                side = str(spec.get("side"))
                if side not in const.BC_SIDES:
                    raise ValueError(f"boundary {name!r}: side must be one of {const.BC_SIDES}, got {side!r}")
                out["where"] = {"axis": int(spec["axis"]), "side": side}
            else:
                raise ValueError(f"boundary {name!r}: give either faces=[...] or axis=/side=")
        out["inputs"] = self.bind(out, kind)
        return out

    def bind(self, spec: Dict[str, Any], kind: str) -> Dict[str, Tuple[str, str, str]]:
        """
        Binds the arguments of a residual function to their data sources.

        Args:
            spec: The normalized specification.
            kind: The type of specification ("pde" or "bc").

        Returns:
            A dictionary mapping argument names to source tuples (type, key, suffix).
        """
        fn, name = spec["residual"], spec["name"]
        pair = kind == "bc" and spec["where"].get("side") == "pair"
        coefs = tuple(self.inverse) + tuple(self.known)
        inputs = {}
        for pname, param in inspect.signature(fn).parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise ValueError(f"{name}: *args/**kwargs are not supported in residual functions")
            if param.default is inspect.Parameter.empty and pname not in spec["map"]:
                # If the parameter has no default and isn't mapped, it's a problem.
                # But we check if it's a reserved arg first in the original code.
                # Let's refine: if it's mapped, we map it. If it's not mapped but has a default, we skip it.
                pass

            key, suffix = spec["map"].get(pname, pname), ""
            if pair and key.endswith(const.PAIR_SUFFIXES):
                key, suffix = key[:-3], key[-3:]
            if key == "fields":
                source = ("all_fields", key, "")
            elif key in const.RESERVED_ARGS:
                if kind == "pde" and key != "X":
                    raise ValueError(f"{name}: {key!r} is only available to boundary residuals")
                source = ("extra", key, suffix)
            elif key in self.fields:
                source = ("field", key, suffix)
            elif key in coefs:
                source = ("coef", key, "")
            elif key in spec["table"]:
                source = ("table", key, "")
            elif param.default is not inspect.Parameter.empty:
                # Skip arguments with default values that aren't mapped
                continue
            else:
                raise ValueError(f"{name}: argument {pname!r} (as {key!r}) is not a field {self.fields}, "
                                 f"a coefficient {coefs}, a table {tuple(spec['table'])}, or one of {const.RESERVED_ARGS}"
                                 + (f" with a {const.PAIR_SUFFIXES} suffix" if pair else ""))
            if pair and source[0] in ("extra", "field") and not suffix:
                raise ValueError(f"{name}: pair boundary argument {pname!r} needs a _lo/_hi suffix")
            inputs[pname] = source
        return inputs

    def describe_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provides a human-readable description of a specification.

        Args:
            spec: The specification to describe.

        Returns:
            A description dictionary.
        """
        out = {"name": spec["name"], "function": getattr(spec["residual"], "__qualname__", repr(spec["residual"])),
               "map": spec["map"], "inputs": {p: f"{kind}:{key}{suffix}" for p, (kind, key, suffix) in spec["inputs"].items()},
               "outputs": list(spec.get("outputs", [])), "scale": spec["scale"], "weight": spec["weight"],
               "table": {k: f"{tuple(v.shape)}:{hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]}"
                         for k, v in spec["table"].items()}}
        if "where" in spec:
            out["num"] = spec["num"]
            out["where"] = spec["where"]
        return out

    @staticmethod
    def split_index(n: int, pattern: Tuple[int, ...], group: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Calculates indices for splitting data based on a pattern.

        Args:
            n: Total number of samples.
            pattern: Split ratio (e.g., 8:1:1).
            group: Optional array for guided splitting.

        Returns:
            A tuple of index arrays for each split.
        """
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

    def split_data(self) -> None:
        """Splits the dataset into training, validation, and test sets."""
        self.split_pattern = tuple(int(v) for v in self.split_pattern)
        n = int(np.shape(self.dataset["X"])[0])
        splits = []
        for name, rows in zip(const.SPLITS, self.split_index(n, self.split_pattern, self.group)):
            if rows.size == 0:
                raise ValueError(f"split {name!r} is empty under pattern {self.split_pattern}")
            splits.append({k: (np.asarray(v)[rows] if np.ndim(v) >= 1 and np.shape(v)[0] == n else v)
                           for k, v in self.dataset.items()})
        self.train_data, self.valid_data, self.test_data = splits

    def to_torch(self, d: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Converts a numpy-based dataset dictionary to torch tensors.

        Args:
            d: Dataset dictionary.

        Returns:
            Dictionary of torch tensors.
        """
        out = {"X": torch.as_tensor(np.asarray(d["X"], dtype=float), dtype=self.dtype, device=self.device)}
        for f in self.fields:
            out[f] = torch.as_tensor(np.asarray(d[f], dtype=float).reshape(-1, 1), dtype=self.dtype, device=self.device)
        return out

    def split_counts(self, n: int) -> Dict[str, int]:
        """
        Returns the number of samples in each split.

        Args:
            n: Total samples.

        Returns:
            Dictionary of split counts.
        """
        return {s: int(rows.size) for s, rows in zip(const.SPLITS, self.split_index(n, self.split_pattern))}

    def rar_score(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Computes the residual score for RAR sampling.

        Args:
            coords: Coordinate tensor.

        Returns:
            Score tensor.
        """
        coords_list = [coords[:, i:i+1] for i in range(coords.shape[1])]
        res = self.residual_pde(coords_list, self.fields_of(self.model(coords_list)))
        score = sum((r.detach() / self.term_scale[k]) ** 2 for k, r in res.items()).flatten()
        if torch.isnan(score).any() or torch.isinf(score).any():
            print(f"RAR score contains nan/inf. Terms: {list(res.keys())}")
            for k, r in res.items():
                print(f"  {k}: min={r.min().item():.2e}, max={r.max().item():.2e}, mean={r.mean().item():.2e}")
        return score

    def bc_score(self, spec: Dict[str, Any], pieces: Any) -> torch.Tensor:
        """
        Computes the residual score for boundary condition RAR sampling.

        Args:
            spec: Boundary specification.
            pieces: Sampled boundary coordinates.

        Returns:
            Score tensor.
        """
        res = self.residual_pieces(spec, pieces)
        return sum(((r.detach() / self.term_scale[k]) ** 2).sum(dim=1) for k, r in res.items()).flatten()

    def sample_fixed(self) -> None:
        """Samples fixed collocation points for validation and test sets."""
        g = self.valid_generator
        self.colloc = {s: torch.cat([self.domain.sample_vertices(self.n_node[s], g, self.dtype, self.device),
                                     self.domain.sample(self.target_n[s], g, self.dtype, self.device)]).detach()
                       for s in ("valid", "test")}
        self.colloc_batch = {}
        for spec in self.boundary:
            spec["colloc"] = {}
            spec["colloc_batch"] = {}
            for s in ("valid", "test"):
                pieces = self.domain.sample_boundary(spec["where"], spec["n"][s], g, self.dtype, self.device)
                spec["colloc"][s] = map_tensors(pieces, lambda x: x.detach())

    def resample(self) -> None:
        """Performs RAR-based collocation sampling for the training set."""
        if self.colloc is None:
            self.sample_fixed()
        n = self.target_n["train"]
        if self.rar_pool > 1:
            cand = self.domain.sample(int(round(n * self.rar_pool)), self.generator, self.dtype, self.device)
            with torch.enable_grad():
                if self.rar_chunk:
                    step = int(self.rar_chunk)
                    r = torch.cat([self.rar_score(cand[lo:lo + step].detach().requires_grad_(True))
                                   for lo in range(0, cand.shape[0], step)])
                else:
                    r = self.rar_score(cand.detach().requires_grad_(True))
            keep = torch.multinomial(r + r.mean(), n, replacement=False, generator=self.generator)
            train = cand[keep]
        else:
            train = self.domain.sample(n, self.generator, self.dtype, self.device)
        nodes = self.domain.sample_vertices(self.n_node["train"], self.generator, self.dtype, self.device)
        parts = [nodes, train]
        self.colloc["train"] = torch.cat(parts).detach()
        self.colloc_batch["train"] = [torch.cat(chunks).detach()
                                      for chunks in zip(*(p.tensor_split(self.n_batch) for p in parts))]
        for spec in self.boundary:
            n_bc = spec["n"]["train"]
            if self.rar_boundary and self.rar_pool > 1:
                cand = self.domain.sample_boundary(spec["where"], int(round(n_bc * self.rar_pool)), spec["generator"],
                                                   self.dtype, self.device)
                with torch.enable_grad():
                    score = self.bc_score(spec, cand)
                keep = torch.multinomial(score + score.mean(), n_bc, replacement=False, generator=spec["generator"])
                pieces = map_tensors(cand, lambda x, k=keep: x[k])
            else:
                pieces = self.domain.sample_boundary(spec["where"], n_bc, spec["generator"], self.dtype, self.device)
            spec["colloc"]["train"] = map_tensors(pieces, lambda x: x.detach())
            spec["colloc_batch"]["train"] = [map_tensors(spec["colloc"]["train"], lambda x, i=b: x.tensor_split(self.n_batch)[i])
                                             for b in range(self.n_batch)]

    def load_data(self) -> None:
        """Splits and loads the dataset into torch tensors."""
        self.split_data()
        self.train_torch = self.to_torch(self.train_data)
        self.valid_torch = self.to_torch(self.valid_data)
        n_train = int(self.train_torch["X"].shape[0])
        self.n_batch = 1 if self.batch_size is None else max(1, math.ceil(n_train / self.batch_size))
        self.bounds = self.domain.bounds
        self.bounds = (np.asarray(self.bounds[0], dtype=float).reshape(-1), np.asarray(self.bounds[1], dtype=float).reshape(-1))
        if self.bounds[0].size != self.dim or self.bounds[1].size != self.dim:
            raise ValueError(f"dataset X has {self.dim} columns but the domain bounds have "
                             f"{self.bounds[0].size}/{self.bounds[1].size} entries (an s axis on one side only?)")
        self.lo = torch.as_tensor(self.bounds[0], dtype=self.dtype, device=self.device)
        self.hi = torch.as_tensor(self.bounds[1], dtype=self.dtype, device=self.device)
        self.target_n = self.split_counts(self.num_domain)
        self.n_node =({s: int(np.shape(d["X"])[0]) for s, d in zip(const.SPLITS, (self.train_data, self.valid_data, self.test_data))}
                       if self.num_node is None else self.split_counts(self.num_node))
        for spec in self.boundary:
            spec["n"] = self.split_counts(spec["num"])
            spec["colloc_batch"] = {}
        self.colloc, self.colloc_batch = None, {}

    @staticmethod
    def set_determinism(enabled: bool) -> None:
        """Sets torch and cudnn to deterministic mode."""
        torch.use_deterministic_algorithms(bool(enabled))
        torch.backends.cudnn.deterministic = bool(enabled)
        torch.backends.cudnn.benchmark = not enabled

    def initialize_model(self) -> None:
        """Initializes the network, optimizer, and training configuration."""
        self.set_determinism(self.deterministic)
        torch.manual_seed(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed + const.SEED_OFFSET_COLLOCATION)
        for i, spec in enumerate(self.boundary):
            spec["generator"] = torch.Generator(device=self.device).manual_seed(
                self.seed + const.SEED_OFFSET_BOUNDARY + i)
        self.seed_valid = self.seed + const.SEED_OFFSET_VALIDATION
        self.valid_generator = torch.Generator(device=self.device).manual_seed(self.seed_valid)
        self.batch_generator = torch.Generator(device=self.device).manual_seed(self.seed + const.SEED_OFFSET_BATCH)
        for spec in (*self.pde, *self.boundary):
            spec["table_t"] = {k: torch.as_tensor(v, device=self.device).to(self.dtype) if np.issubdtype(v.dtype, np.floating)
                               else torch.as_tensor(v, device=self.device) for k, v in spec["table"].items()}
        self.n_param = int(getattr(self.domain, "n_param", 0))
        self.model = network.PINN(self.layers, self.bounds[0], self.bounds[1], feature=self.feature,
                                  n_feature=self.n_feature, n_param=self.n_param, num_centers=self.num_centers).to(device=self.device, dtype=self.dtype)
        self.theta = network.Theta(self.inverse).to(device=self.device, dtype=self.dtype) if self.inverse else None
        groups = [{"params": list(self.model.parameters()), "lr": self.adam_lr}]
        if self.theta is not None:
            groups.append({"params": list(self.theta.parameters()), "lr": self.inverse_lr})
        self.optimizer = torch.optim.Adam(groups)
        self.dry_run()
        raw = dict(layers=list(self.layers), fields=list(self.fields),
                   inverse=self.theta.describe() if self.theta is not None else {}, known=self.known,
                   bounds=[self.bounds[0].tolist(), self.bounds[1].tolist()],
                   pde=[self.describe_spec(s) for s in self.pde],
                   boundary=[self.describe_spec(s) for s in self.boundary],
                   domain=self.domain.describe(), feature=self.describe_feature(),
                   split_pattern=list(self.split_pattern), num_domain=self.num_domain,
                   rar_pool=self.rar_pool, rar_chunk=self.rar_chunk, rar_boundary=self.rar_boundary,
                   resample_every=self.resample_every, batch_size=self.batch_size, n_batch=self.n_batch,
                   adam_epochs=self.adam_epochs, adam_lr=self.adam_lr, inverse_lr=self.inverse_lr,
                   grad_clip=self.grad_clip, lbfgs_iterations=self.lbfgs_iterations, lbfgs_chunk=self.lbfgs_chunk,
                   lbfgs_bound_tol=self.lbfgs_bound_tol, lbfgs_resample=self.lbfgs_resample,
                   seed=self.seed, deterministic=self.deterministic,
                   seed_valid=self.seed_valid, sigma=self.sigma, weights=self.term_weight, scale=self.term_scale, pde_weight=self.pde_weight,
                   terms={"pde": self.pde_terms, "bc": self.bc_terms},
                   dtype=str(self.train_dtype).replace("torch.", ""),
                   lbfgs_dtype=str(self.lbfgs_dtype).replace("torch.", ""),
                   n_train=int(self.train_torch["X"].shape[0]), n_valid=int(self.valid_torch["X"].shape[0]),
                   n_test=int(np.shape(self.test_data["X"])[0]),
                   n_colloc=dict(self.target_n), n_node=dict(self.n_node),
                   n_param=self.n_param,
                   n_boundary={spec["name"]: dict(spec["n"]) for spec in self.boundary},
                   record_every=self.record_every, checkpoint_every=self.checkpoint_every,
                   restore_best=True, data_sha=self.data_sha())
        self.config = json.loads(json.dumps(raw, default=float))
        self.config["fingerprint"] = self.fingerprint(self.config)
        self.history = {k: [] for k in const.HISTORY_KEYS}
        self.history.update({k: [] for k in self.inverse})
        self.best = {"val": float("inf"), "epoch": None, "iteration": None, "phase": None,
                     "state": None, "theta_s": None}
        self.adam_epoch, self.adam_done, self.lbfgs_done = 0, False, False
        self.lbfgs_stop = None
        self.epoch, self.iteration = 0, 0
        self.result = None
        if self.checkpoint_path is not None and self.resume is not False and self.checkpoint_path.exists():
            ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            stale = self.checkpoint_diff(ck)
            if not stale:
                self.resume_from_checkpoint(ck)
            elif self.resume == "auto":
                print(f"[resume] STALE {self.checkpoint_path}: starting fresh\n  " + "\n  ".join(stale), flush=True)
            else:
                raise ValueError(f"checkpoint {self.checkpoint_path} does not match this run "
                                 f"(pass resume='auto' to retrain or resume=False):\n  " + "\n  ".join(stale))

    def data_sha(self) -> str:
        """Computes a SHA-256 fingerprint of the training data."""
        h = hashlib.sha256()
        for k in ("X", *self.fields):
            h.update(np.ascontiguousarray(np.asarray(self.dataset[k], dtype=float)).tobytes())
        if self.group is not None:
            h.update(np.ascontiguousarray(self.group).tobytes())
        return h.hexdigest()[:16]

    @staticmethod
    def fingerprint(config: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint of the training configuration."""
        subset = {k: config.get(k) for k in const.FINGERPRINT_KEYS}
        return hashlib.sha256(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def checkpoint_diff(self, ck: Dict[str, Any]) -> List[str]:
        """Compares the current configuration with a checkpoint configuration."""
        cfg = ck.get("config") or {}
        if "fingerprint" not in cfg:
            return ["checkpoint predates fingerprints (old format)"]
        if cfg["fingerprint"] == self.config["fingerprint"]:
            return []
        diffs = [f"{key}: {_short(cfg.get(key))} -> {_short(self.config.get(key))}"
                 for key in const.FINGERPRINT_KEYS if cfg.get(key) != self.config.get(key)]
        return diffs or ["fingerprint differs"]

    def describe_feature(self) -> Optional[Dict[str, Any]]:
        """Provides a description and fingerprint of the feature encoding function."""
        if self.feature is None:
            return None
        digest = getattr(self.feature, "digest", None)
        if callable(digest):
            sha = str(digest())
        else:
            g = torch.Generator(device=self.device).manual_seed(const.SEED_DRY_RUN)
            X = self.domain.sample(const.FEATURE_SHA_POINTS, g, torch.float64, self.device)
            vals = np.ascontiguousarray(self.feature(X).detach().cpu().numpy().astype(np.float64))
            sha = hashlib.sha256(vals.tobytes()).hexdigest()[:16]
        return {"name": getattr(self.feature, "__name__", type(self.feature).__name__), "n": self.n_feature,
                "sha": sha}

    def dry_run(self) -> None:
        """
        Performs a single forward pass to verify residual shapes and register terms.
        """
        g = torch.Generator(device=self.device).manual_seed(const.SEED_DRY_RUN)
        n = const.DRY_RUN_POINTS
        self.pde_terms, self.bc_terms, self.term_scale, self.term_weight = [], [], {}, {}
        with torch.enable_grad():
            X_tensor = self.domain.sample(n, g, self.dtype, self.device).requires_grad_(True)
            X_list = [X_tensor[:, i:i+1] for i in range(X_tensor.shape[1])]
            y = self.fields_of(self.model(X_list))
            for spec in self.pde:
                self.finish_spec(spec, self.call_residual(spec, {"": {"X": X_list, "y": y, "normal": None, "node": None,
                                                                      "segment": None}}),
                                 self.pde_terms)
            for spec in self.boundary:
                pieces = self.domain.sample_boundary(spec["where"], n, g, self.dtype, self.device)
                self.finish_spec(spec, self.residual_pieces(spec, pieces), self.bc_terms)
        names = ["data", *self.pde_terms, *self.bc_terms]
        dup = sorted({k for k in names if names.count(k) > 1})
        if dup:
            raise ValueError(f"residual names must be unique across data/pde/boundary, duplicated: {dup}")
        unknown = sorted(set(self.weights) - set(names))
        if unknown:
            raise ValueError(f"weights for unknown terms {unknown}; terms are {names}")
        self.term_weight["data"] = self.weights.get("data", 1.0)

    def finish_spec(self, spec: Dict[str, Any], res: Dict[str, torch.Tensor], register: List[str]) -> None:
        """
        Registers a residual's output terms and their scaling/weighting.

        Args:
            spec: The residual specification.
            res: The computed residuals.
            register: The list to which term names are added.
        """
        spec["outputs"] = list(res)
        for k, r in res.items():
            if not torch.is_tensor(r) or r.ndim != 2:
                raise ValueError(f"{spec['name']}: residual {k!r} must be a (n, k) tensor, got {type(r)}")
            for attr, store in (("scale", self.term_scale), ("weight", self.term_weight)):
                v = spec[attr]
                store[k] = float(v[k] if isinstance(v, dict) else v)
            if k in self.weights:
                self.term_weight[k] = self.weights[k]
            register.append(k)

    @property
    def all_params(self) -> List[torch.Tensor]:
        """Returns a list of all optimizable parameters (network weights and inverse params)."""
        return list(self.model.parameters()) + ([] if self.theta is None else list(self.theta.parameters()))

    def cast(self, new_dtype: torch.dtype) -> None:
        """
        Casts the model and all stored tensors to a new dtype.

        Args:
            new_dtype: The target torch dtype.
        """
        self.dtype = new_dtype
        self.model.to(dtype=new_dtype)
        if self.theta is not None:
            self.theta.to(dtype=new_dtype)
        self.train_torch = self.to_torch(self.train_data)
        self.valid_torch = self.to_torch(self.valid_data)
        self.lo = self.lo.to(new_dtype)
        self.hi = self.hi.to(new_dtype)
        if self.colloc is not None:
            self.colloc = {s: c.to(new_dtype) for s, c in self.colloc.items()}
        for spec in self.boundary:
            if "colloc" in spec:
                spec["colloc"] = map_tensors(spec["colloc"], lambda x: x.to(new_dtype) if x.is_floating_point() else x)
        for spec in (*self.pde, *self.boundary):
            spec["table_t"] = map_tensors(spec["table_t"], lambda x: x.to(new_dtype) if x.is_floating_point() else x)

    def coef(self, name: str) -> torch.Tensor:
        """
        Retrieves the value of a physical parameter.

        Args:
            name: The name of the parameter.

        Returns:
            The current value as a torch tensor.
        """
        if self.theta is not None and name in self.theta.names:
            return self.theta.value(name)
        return self.known[name]

    def param_values(self) -> Dict[str, float]:
        """Returns the current values of all inverse parameters."""
        return {} if self.theta is None else self.theta.values()

    def pinned_params(self) -> List[str]:
        """Returns a list of inverse parameters that have hit their search bounds."""
        if self.theta is None or not self.lbfgs_bound_tol:
            return []
        frac = self.theta.fractions()
        return [name for name, f in frac.items()
                if f is not None and min(f, 1.0 - f) < self.lbfgs_bound_tol]

    @staticmethod
    def tail_estimate(history: Dict[str, List[float]], names: Optional[List[str]] = None,
                      fraction: float = const.LBFGS_TAIL_FRACTION, chunk: int = const.LBFGS_CHUNK) -> Optional[Dict[str, Any]]:
        """
        Computes a statistical estimate of parameters based on the tail of the L-BFGS history.

        Args:
            history: The training history dictionary.
            names: Optional list of parameters to estimate.
            fraction: The fraction of the L-BFGS phase to consider as the tail.
            chunk: Chunk size for SEM calculation.

        Returns:
            A dictionary of estimates (median, SD, SEM, etc.) or None.
        """
        names = list(names) if names is not None else [k for k in history if k not in const.HISTORY_KEYS]
        phase = np.asarray(history.get("phase", []))
        mask = phase == "lbfgs"
        if not names or not mask.any():
            return None
        it = np.asarray(history["iteration"], dtype=float)[mask]
        lo = it[-1] - (it[-1] - it[0]) * float(fraction)
        keep = it >= lo
        n_chunk = max(1, int(round((it[-1] - lo) / chunk))) if chunk else int(keep.sum())
        out = {"fraction": float(fraction), "window": [float(lo), float(it[-1])],
               "n": int(keep.sum()), "n_chunk": n_chunk,
               "params": {}, "sd": {}, "sem": {}, "p16": {}, "p84": {}, "drift": {}}
        for name in names:
            v = np.asarray(history[name], dtype=float)[mask][keep]
            out["params"][name] = float(np.median(v))
            out["sd"][name] = float(v.std())
            out["sem"][name] = float(v.std() / np.sqrt(n_chunk))
            out["p16"][name] = float(np.percentile(v, 16))
            out["p84"][name] = float(np.percentile(v, 84))
            out["drift"][name] = (0.0 if v.size < 2 else
                                  float(np.median(v[v.size // 2:]) - np.median(v[:v.size // 2])))
        return out

    @staticmethod
    def scale_estimate(est: Optional[Dict[str, Any]], scale: Dict[str, float]) -> Dict[str, Any]:
        """
        Scales the parameter estimates by the provided scale factors.

        Args:
            est: The estimate dictionary.
            scale: Scaling factors for each parameter.

        Returns:
            A dictionary of scaled estimates.
        """
        if est is None:
            return {}
        return {k: {"value": est["params"][k] * scale[k], "sd": est["sd"][k] * scale[k],
                    "sem": est["sem"][k] * scale[k], "p16": est["p16"][k] * scale[k],
                    "p84": est["p84"][k] * scale[k], "drift": est["drift"][k] * scale[k]}
                for k in est["params"]}

    def fields_of(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Splits the network output tensor into a dictionary of named fields.

        Args:
            y: The output tensor of the model.

        Returns:
            Dictionary mapping field names to tensors.
        """
        return {f: y[:, i:i + 1] for i, f in enumerate(self.fields)}

    def call_residual(self, spec: Dict[str, Any], pieces: Dict[str, Any], prefix: bool = False) -> Dict[str, torch.Tensor]:
        """
        Calls a residual function with arguments bound from the specification.

        Args:
            spec: The normalized residual specification.
            pieces: Coordinate and field data for the current point/batch.
            prefix: Whether to prefix residual names with the spec name.

        Returns:
            A dictionary of computed residuals.
        """
        kwargs = {}
        for pname, (kind, key, suffix) in spec["inputs"].items():
            if kind == "all_fields":
                # Provide all output fields as a dictionary
                kwargs[pname] = {f: piece["y"][f] for f in self.fields}
                continue
            if kind == "coef":
                kwargs[pname] = self.coef(key)
                continue
            if kind == "table":
                kwargs[pname] = spec["table_t"][key]
                continue
            piece = pieces[suffix]
            if kind == "field":
                kwargs[pname] = piece["y"][key]
            else:
                value = piece.get(key)
                if value is None:
                    raise ValueError(f"{spec['name']}: {key!r} is not available on this boundary "
                                     "(normal needs a mesh face or a spatial box face, node needs a time face of MeshDomain, "
                                     "segment needs a time face of a MeshDomain built with s_values)")
                if key == "X" and torch.is_tensor(value) and value.ndim == 2:
                    value = [value[:, i:i+1] for i in range(value.shape[1])]
                kwargs[pname] = value
        r = spec["residual"](**kwargs)
        if isinstance(r, dict):
            return {f"{spec['name']}_{k}" if prefix else str(k): v for k, v in r.items()}
        if isinstance(r, (list, tuple)):
            r = torch.cat(list(r), dim=1)
        return {spec["name"]: r}

    def residual_pde(self, coords: Union[torch.Tensor, List[torch.Tensor]], y: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes all PDE residuals for a given set of coordinates.

        Args:
            coords: Coordinate tensor or list of coordinate tensors.
            y: Dictionary of field tensors.

        Returns:
            Dictionary of all PDE residuals.
        """
        res = {}
        pieces = {"": {"X": coords, "y": y, "normal": None, "node": None, "segment": None}}
        for spec in self.pde:
            res.update(self.call_residual(spec, pieces))
        return res

    def residual_pieces(self, spec: Dict[str, Any], pieces: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes residuals for a specific boundary piece.

        Args:
            spec: The boundary specification.
            pieces: A dictionary of boundary pieces.

        Returns:
            Dictionary of computed residuals.
        """
        live = {}
        for suffix, piece in pieces.items():
            X = piece["X"].clone().requires_grad_(True)
            live[suffix] = {"X": X, "y": self.fields_of(self.model(X)), "normal": piece["normal"], "node": piece["node"],
                            "segment": piece.get("segment")}
        return self.call_residual(spec, live, prefix=True)

    def residual_bc(self, spec: Dict[str, Any], split: str, bidx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Computes boundary residuals for a specific split.

        Args:
            spec: The boundary specification.
            split: The split to evaluate ("train", "valid", "test").
            bidx: Optional batch index.

        Returns:
            Dictionary of computed residuals.
        """
        pieces = spec["colloc"][split] if bidx is None else spec["colloc_batch"][split][bidx]
        return self.residual_pieces(spec, pieces)

    def loss(self, data: Dict[str, Any], colloc: torch.Tensor, split: str, bidx: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the total PINN loss (Data + PDE + BC).

        Args:
            data: Data dictionary.
            colloc: Collocation points.
            split: The current split.
            bidx: Optional batch index.

        Returns:
            A tuple containing the total loss and a dictionary of individual term losses.
        """
        n = data["X"].shape[0]
        coords_tensor = torch.cat([data["X"], colloc], dim=0).detach().requires_grad_(True)
        coords_list = [coords_tensor[:, i:i+1] for i in range(coords_tensor.shape[1])]
        y = self.fields_of(self.model(coords_list))
        terms = {"data": sum(torch.mean(((y[f][:n] - data[f]) / self.sigma[f]) ** 2) for f in self.fields) / len(self.fields)}
        for k, r in self.residual_pde(coords_list, y).items():
            terms[k] = torch.mean((r / self.term_scale[k]) ** 2)
        for spec in self.boundary:
            for k, r in self.residual_bc(spec, split, bidx).items():
                terms[k] = torch.mean((r / self.term_scale[k]) ** 2)
        model_part = sum(self.term_weight[k] * terms[k] for k in (*self.pde_terms, *self.bc_terms))
        total = self.term_weight["data"] * terms["data"] + self.pde_weight * model_part
        return total, terms

    def term_parts(self, terms: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        """
        Splits the total loss into data, PDE, and BC components.

        Args:
            terms: Dictionary of individual term losses.

        Returns:
            Tuple of (data_loss, pde_loss, bc_loss).
        """
        data = self.term_weight["data"] * float(terms["data"].detach())
        pde = self.pde_weight * sum(self.term_weight[k] * float(terms[k].detach()) for k in self.pde_terms)
        bc = self.pde_weight * sum(self.term_weight[k] * float(terms[k].detach()) for k in self.bc_terms)
        return data, pde, bc

    def record(self, phase: str, total: torch.Tensor, terms: Dict[str, torch.Tensor]) -> None:
        """
        Records training metrics in the history.

        Args:
            phase: The current training phase ("adam" or "lbfgs").
            total: The total loss.
            terms: Dictionary of individual term losses.
        """
        with torch.enable_grad():
            v_total, v_terms = self.loss(self.valid_torch, self.colloc["valid"], "valid")
        vt = float(v_total.detach())
        vd, vp, vb = self.term_parts(v_terms)
        if vt < self.best["val"]:
            self.best.update(val=vt, epoch=int(self.epoch), iteration=int(self.iteration), phase=phase,
                             state={k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()},
                             theta_s=None if self.theta is None else self.theta.s.detach().clone().cpu())
        td, tp, tb = self.term_parts(terms)
        self.history["epoch"].append(int(self.epoch))
        self.history["iteration"].append(int(self.iteration))
        self.history["phase"].append(phase)
        self.history["train_total"].append(float(total.detach()))
        self.history["train_data"].append(td)
        self.history["train_pde"].append(tp)
        self.history["train_bc"].append(tb)
        self.history["val_total"].append(vt)
        self.history["val_data"].append(vd)
        self.history["val_pde"].append(vp)
        self.history["val_bc"].append(vb)
        for k, v in self.param_values().items():
            self.history[k].append(v)

    def show(self, phase: str, total: torch.Tensor, terms: Dict[str, torch.Tensor]) -> None:
        """
        Prints the current training status to the console.

        Args:
            phase: The current training phase.
            total: The total loss.
            terms: Dictionary of individual term losses.
        """
        td, tp, tb = self.term_parts(terms)
        msg = f"[{phase}] epoch {self.epoch:6d}"
        if self.n_batch > 1:
            msg += f" (iter {self.iteration})"
        msg += f"  loss={float(total.detach()):.4e}  data={td:.2e}  pde={tp:.2e}"
        if self.boundary:
            msg += f"  bc={tb:.2e}"
        for k, v in self.param_values().items():
            msg += f"  {k}={v:.5g}"
        msg += f"  elapsed={time.perf_counter() - self.t_construct:.0f}s"
        print(msg, flush=True)

    def maybe_log(self, phase: str, total: Optional[torch.Tensor] = None, terms: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Conditionally logs and records training status.

        Args:
            phase: The current training phase.
            total: The total loss.
            terms: Dictionary of individual term losses.
        """
        do_record = self.record_every and self.epoch % self.record_every == 0
        do_show = self.log_every and self.epoch % self.log_every == 0
        if not (do_record or do_show):
            return
        if total is None:
            with torch.enable_grad():
                total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        if do_record:
            self.record(phase, total, terms)
        if do_show:
            self.show(phase, total, terms)

    def save_checkpoint(self, adam_epoch: int, adam_done: bool, lbfgs_done: bool, optimizer_state: Optional[Dict] = None) -> None:
        """
        Saves the training state to a checkpoint file.

        Args:
            adam_epoch: Current Adam epoch.
            adam_done: Whether Adam phase is complete.
            lbfgs_done: Whether L-BFGS phase is complete.
            optimizer_state: Current state of the optimizer.
        """
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        to_cpu = lambda x: x.cpu()
        torch.save({"model_state_dict": self.model.state_dict(),
                    "layers": self.config["layers"], "n_param": self.n_param,
                    "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
                    "params": self.param_values(),
                    "theta_s": None if self.theta is None else self.theta.s.detach().cpu(),
                    "known": dict(self.known),
                    "history": self.history,
                    "config": self.config,
                    "train_state": {"adam_epoch": int(adam_epoch), "adam_done": bool(adam_done),
                                    "lbfgs_done": bool(lbfgs_done), "optimizer_state_dict": optimizer_state,
                                    "epoch": int(self.epoch), "iteration": int(self.iteration),
                                    "best": self.best, "lbfgs_stop": self.lbfgs_stop,
                                    "generator_state": self.generator.get_state(),
                                    "batch_generator_state": self.batch_generator.get_state(),
                                    "bc_generator_state": [spec["generator"].get_state() for spec in self.boundary],
                                    "colloc": None if self.colloc is None else map_tensors(self.colloc, to_cpu),
                                    "bc_colloc": [map_tensors(spec["colloc"], to_cpu) if "colloc" in spec else None
                                                  for spec in self.boundary]}}, self.checkpoint_path)

    def resume_from_checkpoint(self, ck: Dict[str, Any]) -> None:
        """
        Restores the training state from a checkpoint.

        Args:
            ck: The checkpoint dictionary.
        """
        ts = ck["train_state"]
        self.adam_epoch, self.adam_done, self.lbfgs_done = ts["adam_epoch"], ts["adam_done"], ts["lbfgs_done"]
        self.epoch, self.iteration = int(ts["epoch"]), int(ts["iteration"])
        if ck["dtype"] != str(self.dtype).replace("torch.", ""):
            self.cast(getattr(torch, ck["dtype"]))
        self.model.load_state_dict(ck["model_state_dict"])
        if self.theta is not None and ck.get("theta_s") is not None:
            with torch.no_grad():
                self.theta.s.copy_(ck["theta_s"].to(dtype=self.theta.s.dtype, device=self.device))
        self.history = ck["history"]
        if ts.get("best") is not None:
            self.best.update(ts["best"])
        self.lbfgs_stop = ts.get("lbfgs_stop")
        if not self.adam_done and ts.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ts["optimizer_state_dict"])
        if ts.get("generator_state") is not None:
            self.generator.set_state(ts["generator_state"].cpu())
        if ts.get("batch_generator_state") is not None:
            self.batch_generator.set_state(ts["batch_generator_state"].cpu())
        for spec, state in zip(self.boundary, ts.get("bc_generator_state") or []):
            spec["generator"].set_state(state.cpu())
        to_dev = lambda x: x.to(device=self.device)
        if ts.get("colloc") is not None:
            self.colloc = map_tensors(ts["colloc"], to_dev)
        for spec, saved in zip(self.boundary, ts.get("bc_colloc") or []):
            if saved is not None:
                spec["colloc"] = map_tensors(saved, to_dev)
        if self.log_every:
            print(f"[resume] {self.checkpoint_path}: adam_epoch={self.adam_epoch}/{self.adam_epochs} "
                  f"adam_done={self.adam_done} lbfgs_done={self.lbfgs_done}", flush=True)

    @staticmethod
    def load_checkpoint(path: str, device: Optional[str] = None, feature: Optional[Callable] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Loads a trained model and its state from a checkpoint file.

        Args:
            path: Path to the checkpoint.
            device: Computation device.
            feature: Feature encoding function used during training.

        Returns:
            A tuple of (trained_model, checkpoint_dictionary).
        """
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(path, map_location=torch_device, weights_only=False)
        lo, hi = state["config"]["bounds"]
        described = state["config"].get("feature")
        if (described is None) != (feature is None):
            raise ValueError(f"checkpoint {path} was trained with feature={described}; pass the same feature callable")
        model = network.PINN(state["layers"], lo, hi, feature=feature, n_feature=0 if described is None else described["n"],
                             n_param=state.get("n_param", 0)).to(device=torch_device, dtype=getattr(torch, state["dtype"]))
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model, state

    def train_adam(self) -> None:
        """
        Executes the Adam optimization phase.
        """
        if self.colloc is None or "train" not in self.colloc_batch:
            self.resample()
        n_train = self.train_torch["X"].shape[0]
        e = self.adam_epoch
        try:
            for e in range(self.adam_epoch, self.adam_epochs):
                if self.resample_every and e > 0 and e % self.resample_every == 0:
                    self.resample()
                order = (None if self.n_batch == 1 else
                         torch.randperm(n_train, generator=self.batch_generator, device=self.device))
                for b, idx in enumerate([None] if order is None else order.tensor_split(self.n_batch)):
                    data = self.train_torch if idx is None else {k: v[idx] for k, v in self.train_torch.items()}
                    colloc = self.colloc["train"] if idx is None else self.colloc_batch["train"][b]
                    self.optimizer.zero_grad(set_to_none=True)
                    total, terms = self.loss(data, colloc, "train", None if idx is None else b)
                    total.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.optimizer.step()
                self.maybe_log("adam")
                self.epoch += 1
                self.iteration += self.n_batch
                if self.checkpoint_every and (e + 1) % self.checkpoint_every == 0:
                    self.save_checkpoint(e + 1, False, False, self.optimizer.state_dict())
        except KeyboardInterrupt:
            self.save_checkpoint(e, False, False, self.optimizer.state_dict())
            print(f"[adam] interrupted at epoch {e}, checkpoint saved; call train again to resume", flush=True)
            raise
        with torch.enable_grad():
            total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        self.record("adam", total, terms)
        if self.log_every:
            self.show("adam", total, terms)
        self.adam_epoch, self.adam_done = self.adam_epochs, True
        self.save_checkpoint(self.adam_epochs, True, False)

    def train_lbfgs(self) -> None:
        """
        Executes the L-BFGS optimization phase.
        """
        if self.lbfgs_dtype != self.dtype:
            self.cast(self.lbfgs_dtype)
            if self.log_every:
                print(f"[lbfgs] cast to {str(self.dtype).replace('torch.', '')}", flush=True)
        self.resample()
        all_params = self.all_params
        chunk = min(self.lbfgs_chunk, self.lbfgs_iterations) if self.lbfgs_chunk else self.lbfgs_iterations
        step_iter = min(self.record_every, chunk) if self.record_every else chunk

        def make_lbfgs():
            return torch.optim.LBFGS(
                all_params, lr=1.0, max_iter=step_iter, max_eval=int(step_iter * const.LBFGS_MAX_EVAL_FACTOR),
                tolerance_grad=const.LBFGS_TOLERANCE_GRAD, tolerance_change=0.0,
                history_size=const.LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe")

        lbfgs = make_lbfgs()
        calls = {"n": 0, "record": False, "shown": self.epoch}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
            total.backward()
            calls["n"] += 1
            if calls["record"]:
                calls["record"] = False
                if self.record_every:
                    self.record("lbfgs", total, terms)
                if self.log_every and self.epoch - calls["shown"] >= self.log_every:
                    calls["shown"] = self.epoch
                    self.show("lbfgs", total, terms)
            return total

        n_iter = min(max(int(self.epoch) - int(self.adam_epochs), 0), self.lbfgs_iterations)
        since_chunk = 0
        self.lbfgs_stop = {"reason": "max_iter", "n_iter": n_iter, "pinned": []}
        try:
            while n_iter < self.lbfgs_iterations:
                before = int(lbfgs.state[all_params[0]].get("n_iter", 0))
                calls["record"] = True
                group = lbfgs.param_groups[0]
                group["max_iter"] = min(step_iter, self.lbfgs_iterations - n_iter, chunk - since_chunk)
                group["max_eval"] = int(group["max_iter"] * const.LBFGS_MAX_EVAL_FACTOR)
                lbfgs.step(closure)
                step_iters = int(lbfgs.state[all_params[0]].get("n_iter", 0)) - before
                n_iter += step_iters
                self.epoch += step_iters
                self.iteration += step_iters
                since_chunk += step_iters
                self.lbfgs_stop["n_iter"] = n_iter
                if since_chunk < chunk and step_iters and n_iter < self.lbfgs_iterations:
                    continue
                since_chunk = 0
                if self.checkpoint_every:
                    self.save_checkpoint(self.adam_epochs, True, False)
                self.lbfgs_stop["pinned"] = self.pinned_params()
                if step_iters == 0:
                    self.lbfgs_stop["reason"] = "gradient tolerance"
                    break
                if self.lbfgs_resample and n_iter < self.lbfgs_iterations:
                    self.resample()
                    lbfgs = make_lbfgs()
        except KeyboardInterrupt:
            self.save_checkpoint(self.adam_epochs, True, False)
            print("[lbfgs] interrupted, checkpoint saved; call train again to resume "
                  "(L-BFGS restarts from the saved weights)", flush=True)
            raise
        with torch.enable_grad():
            total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        self.record("lbfgs", total, terms)
        if self.log_every:
            self.show("lbfgs", total, terms)
            print(f"[lbfgs] stopped after {n_iter} iterations ({calls['n']} evals): {self.lbfgs_stop['reason']}"
                  + (f"   WARNING: {'/'.join(self.lbfgs_stop['pinned'])} sitting on a search bound, "
                     f"this inversion ran off the rails" if self.lbfgs_stop["pinned"] else ""), flush=True)
            est = self.tail_estimate(self.history)
            if est is not None:
                print(f"[lbfgs] tail estimate over iterations {est['window'][0]:.0f}-{est['window'][1]:.0f} "
                      f"({est['n_chunk']} chunks, {est['n']} records): "
                      + ", ".join(f"{k} = {est['params'][k]:.5g} +- {est['sem'][k]:.2g} (sd {est['sd'][k]:.2g})"
                                  for k in est["params"]), flush=True)
        self.lbfgs_done = True

    def restore_best_snapshot(self) -> bool:
        """
        Restores the model and parameters to the best state recorded during validation.

        Returns:
            True if a snapshot was restored, False otherwise.
        """
        if self.best["state"] is None:
            return False
        self.model.load_state_dict({k: v.to(dtype=self.dtype, device=self.device)
                                    for k, v in self.best["state"].items()})
        if self.theta is not None and self.best.get("theta_s") is not None:
            with torch.no_grad():
                self.theta.s.copy_(self.best["theta_s"].to(dtype=self.theta.s.dtype, device=self.device))
        if self.log_every:
            print(f"[best] restored epoch {self.best['epoch']} ({self.best['phase']}) "
                  f"val_total={self.best['val']:.4e}", flush=True)
        return True

    def train(self) -> Dict[str, Any]:
        """
        Orchestrates the complete training pipeline (Adam -> L-BFGS -> Best Restore).

        Returns:
            A result dictionary containing the trained model, parameters, and history.
        """
        t_start = time.perf_counter()
        if not self.adam_done:
            self.train_adam()
        if self.lbfgs_iterations and not self.lbfgs_done:
            self.train_lbfgs()
        self.lbfgs_done = self.lbfgs_done or not self.lbfgs_iterations
        restored = self.restore_best_snapshot()
        self.save_checkpoint(self.adam_epochs, True, self.lbfgs_done)
        if self.colloc is None:
            self.resample()
        with torch.enable_grad():
            total, terms = self.loss(self.train_torch, self.colloc["train"], "train")
        self.result = {"model": self.model,
                       "params": self.param_values(),
                       "estimate": self.tail_estimate(self.history),
                       "theta_s": None if self.theta is None else self.theta.s.detach().cpu().tolist(),
                       "known": self.known,
                       "history": self.history,
                       "best": {"epoch": self.best["epoch"], "iteration": self.best["iteration"],
                                "phase": self.best["phase"],
                                "val": None if self.best["state"] is None else self.best["val"],
                                "restored": restored},
                       "loss": float(total.detach()),
                       "terms": {k: float(v.detach()) for k, v in terms.items()},
                       "lbfgs": self.lbfgs_stop,
                       "wall_seconds": time.perf_counter() - t_start,
                       "config": self.config,
                       "train_state": {"adam_epoch": self.adam_epochs, "adam_done": True,
                                       "lbfgs_done": bool(self.lbfgs_done), "optimizer_state_dict": None,
                                       "epoch": int(self.epoch), "iteration": int(self.iteration),
                                       "best": self.best}}
        return self.result

    @staticmethod
    def evaluation(model: nn.Module, coords: np.ndarray, fields: List[str]) -> Dict[str, np.ndarray]:
        """
        Performs inference using a trained model.

        Args:
            model: The trained PINN model.
            coords: Coordinate array.
            fields: List of field names to evaluate.

        Returns:
            Dictionary mapping field names to predicted values.
        """
        param = next(model.parameters())
        coords = np.asarray(coords, dtype=float)
        with torch.no_grad():
            y = model(torch.as_tensor(coords, dtype=param.dtype, device=param.device))
        y = y.detach().cpu().numpy()
        return {name: y[:, i] for i, name in enumerate(fields)}

    def evaluate(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluates the model at the given coordinates.

        Args:
            coords: Coordinate array.

        Returns:
            Dictionary of predicted field values.
        """
        return self.evaluation(self.model, coords, self.fields)

    def evaluate_residual(self, split: str) -> Dict[str, np.ndarray]:
        """
        Computes the PDE residuals for a given data split.

        Args:
            split: The split to evaluate ("train", "valid", "test").

        Returns:
            Dictionary of residuals.
        """
        if self.colloc is None:
            self.resample()
        with torch.enable_grad():
            coords = self.colloc[split].clone().requires_grad_(True)
            res = self.residual_pde(coords, self.fields_of(self.model(coords)))
            out = {k: (v / self.term_scale[k]).detach().cpu().numpy().ravel() for k, v in res.items()}
            for spec in self.boundary:
                for k, r in self.residual_bc(spec, split).items():
                    out[k] = (r / self.term_scale[k]).detach().cpu().numpy().ravel()
        return out

    @staticmethod
    def error_record(y: np.ndarray, yhat: np.ndarray) -> Dict[str, Any]:
        """
        Computes error metrics between true and predicted values.

        Args:
            y: True values.
            yhat: Predicted values.

        Returns:
            Dictionary of error metrics (RMSE, R2, etc.).
        """
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

    def diag_record(self, split: str = "test") -> Dict[str, Dict[str, Any]]:
        """
        Generates diagnostic error records for all fields.

        Args:
            split: The split to evaluate.

        Returns:
            Dictionary mapping field names to error records.
        """
        data = {"train": self.train_data, "valid": self.valid_data, "test": self.test_data}[split]
        pred = self.evaluate(data["X"])
        return {f: self.error_record(data[f], pred[f]) for f in self.fields if f in data}

    @staticmethod
    def finish(fig: plt.Figure, path: Optional[str] = None) -> None:
        """
        Saves and/or displays a matplotlib figure.

        Args:
            fig: The figure to finish.
            path: Optional path to save the figure.
        """
        if path is not None:
            fig.savefig(path)
        if plt.get_backend().lower() != "agg":
            plt.show()

    @staticmethod
    def history_axis(history: Dict[str, List[float]]) -> Tuple[np.ndarray, str]:
        """
        Determines the x-axis (epoch or iteration) for history plots.

        Args:
            history: The training history.

        Returns:
            Tuple of (axis_values, axis_label).
        """
        key = "epoch" if "epoch" in history else "iteration"
        return np.asarray(history[key]), key

    @staticmethod
    def best_axis(best: Dict[str, Any]) -> Optional[Union[int, float]]:
        """
        Retrieves the axis value at which the best validation loss occurred.

        Args:
            best: The best state dictionary.

        Returns:
            The epoch or iteration of the best state.
        """
        return None if not best else best.get("epoch", best.get("iteration"))

    @staticmethod
    def plot_loss(history: Dict[str, List[float]], path: Optional[str] = None,
                  title: str = "Loss history", best_epoch: Optional[int] = None,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plots the training and validation loss history.

        Args:
            history: Training history.
            path: Optional path to save the plot.
            title: Plot title.
            best_epoch: Epoch of the best validation loss.
            ax: Optional existing axes to plot on.

        Returns:
            The axes used for plotting.
        """
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
    def plot_diag(records: Dict[str, Dict[str, Any]], fields: List[str],
                  grid: Optional[Tuple[int, int]] = None, path: Optional[str] = None,
                  title: str = "Test diagonal", band: float = const.DIAG_BAND,
                  units: Optional[Dict[str, Tuple[str, float]]] = None,
                  axes: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plots diagnostic scatter plots (True vs Predicted) for all fields.

        Args:
            records: Error records dictionary.
            fields: List of field names.
            grid: Grid layout (rows, cols).
            path: Optional path to save the plot.
            title: Plot title.
            band: Relative error band width.
            units: Dictionary of units and scales for fields.
            axes: Optional existing axes.

        Returns:
            The axes used for plotting.
        """
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
    def residual_stats(resid: Dict[str, torch.Tensor], whiskers: Tuple[int, int] = (5, 95)) -> Dict[str, Dict[str, float]]:
        """
        Computes statistics (median, quartiles, whiskers) for residuals.

        Args:
            resid: Dictionary of residuals.
            whiskers: Percentiles for the whiskers.

        Returns:
            Dictionary of statistics for each residual.
        """
        out = {}
        for k, r in resid.items():
            a = np.abs(np.asarray(r, float)).ravel()
            q = np.percentile(a, [whiskers[0], 25, 50, 75, whiskers[1]])
            out[k] = {"n": int(a.size), "rms": float(np.sqrt(np.mean(a ** 2))),
                      "whislo": float(q[0]), "q1": float(q[1]), "med": float(q[2]),
                      "q3": float(q[3]), "whishi": float(q[4])}
        return out

    @staticmethod
    def plot_residual(residual: Dict[str, torch.Tensor], path: Optional[str] = None, title: str = "Collocation residual", ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plots the distribution of residuals using boxplots.

        Args:
            residual: Dictionary of residuals.
            path: Optional path to save the plot.
            title: Plot title.
            ax: Optional existing axes.

        Returns:
            The axes used for plotting.
        """
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
        ax.set_xlabel("|scaled residual|")
        if own:
            ax.set_title(title)
            ax.legend(loc="upper right", ncol=len(splits))
            PINNTraining.finish(fig, path)
        return ax

    @staticmethod
    def plot_params(history: Dict[str, List[float]], params: Optional[Dict[str, float]] = None,
                   true: Optional[Dict[str, float]] = None, path: Optional[str] = None,
                   title: str = "Inverse parameters", best_epoch: Optional[int] = None,
                   estimate: Optional[Dict[str, Any]] = None, axes: Optional[plt.Axes] = None,
                   label: Optional[str] = None) -> plt.Axes:
        """
        Plots the recovery of inverse parameters over the training history.

        Args:
            history: Training history.
            params: Final recovered parameters.
            true: True physical parameter values.
            path: Optional path to save the plot.
            title: Plot title.
            best_epoch: Epoch of the best validation loss.
            estimate: Tail estimate dictionary.
            axes: Optional existing axes.
            label: Label for the recovered parameter line.

        Returns:
            The axes used for plotting.
        """
        names = list(params) if params is not None else [k for k in history if k not in const.HISTORY_KEYS]
        reported = ({k: v["value"] for k, v in estimate.items()} if estimate
                    else params if isinstance(params, dict) else {})
        own = axes is None
        if own:
            fig, axarr = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
            axes = axarr.ravel()
        axes = np.ravel(axes)
        it, xlabel = PINNTraining.history_axis(history)
        for ax, k in zip(axes, names):
            ax.plot(it, history[k], label=label or f"recovered {k}")
            if estimate is not None and k in estimate:
                e = estimate[k]
                ax.axhspan(e["p16"], e["p84"], color="tab:orange", alpha=0.18)
                ax.axhline(e["value"], color="tab:orange", lw=1.5,
                           label=f"reported {k} (tail median)")
            if true is not None and k in true:
                ax.axhline(true[k], color="k", ls="--", label=f"true {k} = {true[k]:.4g}")
            if best_epoch is not None:
                ax.axvline(best_epoch, color="tab:red", lw=1, ls="--", label="best epoch")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(k)
        if own:
            for ax, k in zip(axes, names):
                ax.set_title(f"{k}: recovered = {reported.get(k, history[k][-1]):.5g}")
                ax.legend()
            if title:
                fig.suptitle(title)
            fig.tight_layout()
            PINNTraining.finish(fig, path)
        return axes
