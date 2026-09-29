from __future__ import annotations

import hashlib
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from . import const
from . import comsol_case


def _uniform(shape, generator, dtype, device, lo=0.0, hi=1.0):
    return lo + (hi - lo) * torch.rand(shape, generator=generator, dtype=dtype, device=device)


class NodeFeature:
    def __init__(self, vertices, values, length_scale=1.0, name="node_feature"):
        vertices = np.asarray(vertices, float) / float(length_scale)
        values = np.asarray(values, float).reshape(-1)
        if values.shape[0] != vertices.shape[0]:
            raise ValueError(f"{values.shape[0]} values for {vertices.shape[0]} vertices")
        ok = np.isfinite(values)
        self.vertices, self.values = vertices[ok], values[ok]
        self.n_dropped = int((~ok).sum())
        self.__name__ = str(name)
        self._cache = {}

    def digest(self):
        h = hashlib.sha256()
        for arr in (self.vertices, self.values):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]

    def _tensors(self, device, dtype):
        key = (str(device), str(dtype))
        if key not in self._cache:
            self._cache[key] = (torch.as_tensor(self.vertices, dtype=dtype, device=device),
                                torch.as_tensor(self.values, dtype=dtype, device=device))
        return self._cache[key]

    def __call__(self, X):
        vert, val = self._tensors(X.device, X.dtype)
        P = X[:, :3].detach()
        out = torch.empty(P.shape[0], dtype=X.dtype, device=X.device)
        step = int(const.FEATURE_CHUNK)
        for lo in range(0, P.shape[0], step):
            out[lo:lo + step] = val[torch.cdist(P[lo:lo + step], vert).argmin(dim=1)]
        return out.reshape(-1, 1)


def _sides(where):
    if where["side"] not in const.BC_SIDES:
        raise ValueError(f"side must be one of {const.BC_SIDES}, got {where['side']!r}")
    return (("_lo", "lo"), ("_hi", "hi")) if where["side"] == "pair" else (("", where["side"]),)


class BoxDomain:
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float).reshape(-1)
        self.hi = np.asarray(hi, dtype=float).reshape(-1)
        if self.lo.size != self.hi.size:
            raise ValueError(f"lo and hi must have same size, got {self.lo.size} and {self.hi.size}")

    @property
    def bounds(self):
        return self.lo.copy(), self.hi.copy()

    def describe(self):
        return {"type": "BoxDomain", "lo": self.lo.tolist(), "hi": self.hi.tolist()}

    def sample(self, n, generator, dtype, device):
        lo = torch.as_tensor(self.lo, dtype=dtype, device=device)
        hi = torch.as_tensor(self.hi, dtype=dtype, device=device)
        return lo + (hi - lo) * torch.rand((n, self.lo.size), generator=generator, dtype=dtype, device=device)

    def sample_vertices(self, n, generator, dtype, device):
        return self.sample(n, generator, dtype, device)

    def sample_boundary(self, where, n, generator, dtype, device):
        if "faces" in where:
            raise ValueError("BoxDomain has no mesh faces; boundary specs must use axis/side (or use MeshDomain)")
        axis = int(where["axis"])
        base = self.sample(n, generator, dtype, device)
        out = {}
        for suffix, side in _sides(where):
            X = base.clone()
            X[:, axis] = float((self.lo if side == "lo" else self.hi)[axis])
            normal = None
            if axis < self.lo.size - 1:
                normal = torch.zeros((n, self.lo.size - 1), dtype=dtype, device=device)
                normal[:, axis] = -1.0 if side == "lo" else 1.0
            out[suffix] = {"X": X, "normal": normal, "node": None}
        return out


class MeshDomain:
    def __init__(self, mesh, length_scale=1.0, t_range=(0.0, 1.0), s_values=None, s_discrete=False):
        self.length_scale = float(length_scale)
        self.t_range = (float(t_range[0]), float(t_range[1]))
        self.s_values = None if s_values is None else np.asarray(s_values, dtype=float).reshape(-1)
        if self.s_values is not None and (self.s_values.size < 2 or not np.all(np.isfinite(self.s_values))):
            raise ValueError(f"s_values needs at least two finite entries, got {s_values!r}")
        self.dim = 4 if self.s_values is None else 5
        self.n_param = 0 if self.s_values is None else 1
        self.s_discrete = bool(s_discrete)
        if self.s_discrete and self.s_values is None:
            raise ValueError("s_discrete needs s_values")
        self.vertices = np.asarray(mesh["vertices"], dtype=float) / self.length_scale
        self.tet = np.asarray(mesh["tet"], dtype=np.int64)
        self.tet_volume = np.asarray(mesh["tet_volume"], dtype=float)
        tris, entity, normal = [], [], []
        if "tri" in mesh:
            tris.append(np.asarray(mesh["tri"], dtype=np.int64))
            entity.append(np.asarray(mesh["tri_entity"], dtype=np.int64))
            normal.append(np.asarray(mesh["tri_normal"], dtype=float))
        if "quad" in mesh:
            quad = np.asarray(mesh["quad"], dtype=np.int64)
            for idx in const.COMSOL_QUAD_TRIS:
                tris.append(quad[:, list(idx)])
                entity.append(np.asarray(mesh["quad_entity"], dtype=np.int64))
                normal.append(np.asarray(mesh["quad_normal"], dtype=float))
        self.tri = np.concatenate(tris)
        self.tri_entity = np.concatenate(entity)
        area, unit, _ = comsol_case.tri_geometry(self.vertices, self.tri)
        ref = np.concatenate(normal)
        sign = np.where(np.einsum("ij,ij->i", unit, ref) < 0, -1.0, 1.0)
        self.tri_area = area
        self.tri_normal = unit * sign[:, None]
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(mesh["vertices"]).tobytes())
        h.update(np.ascontiguousarray(self.tet).tobytes())
        h.update(np.ascontiguousarray(self.tri).tobytes())
        self.mesh_sha = h.hexdigest()[:16]
        self._cache = {}

    @property
    def bounds(self):
        lo = np.append(self.vertices.min(axis=0), self.t_range[0])
        hi = np.append(self.vertices.max(axis=0), self.t_range[1])
        if self.s_values is not None:
            lo, hi = np.append(lo, self.s_values.min()), np.append(hi, self.s_values.max())
        return lo, hi

    @property
    def faces(self):
        return tuple(int(f) for f in np.unique(self.tri_entity))

    def describe(self):
        out = {"type": "MeshDomain", "n_vertices": int(self.vertices.shape[0]), "n_tet": int(self.tet.shape[0]),
               "n_tri": int(self.tri.shape[0]), "mesh_sha": self.mesh_sha, "weight": "volume",
               "length_scale": self.length_scale, "t_range": list(self.t_range)}
        if self.s_values is not None:
            out["s_values"] = self.s_values.tolist()
        if self.s_discrete:
            out["s_discrete"] = True
        return out

    def _tensors(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = {
                "vert": torch.as_tensor(self.vertices, dtype=torch.float64, device=device),
                "tet": torch.as_tensor(self.tet, dtype=torch.long, device=device),
                "tet_w": torch.as_tensor(self.tet_volume, dtype=torch.float64, device=device),
                "tri": torch.as_tensor(self.tri, dtype=torch.long, device=device),
                "tri_entity": torch.as_tensor(self.tri_entity, dtype=torch.long, device=device),
                "tri_w": torch.as_tensor(self.tri_area, dtype=torch.float64, device=device),
                "tri_normal": torch.as_tensor(self.tri_normal, dtype=torch.float64, device=device)}
        return self._cache[key]

    def _time(self, n, generator, dtype, device):
        return _uniform((n, 1), generator, dtype, device, *self.t_range)

    def _with_s(self, X, generator, dtype, device):
        if self.s_values is None:
            return X
        if self.s_discrete:
            segment = torch.randint(0, int(self.s_values.size), (X.shape[0],), generator=generator, device=device)
            s = torch.as_tensor(self.s_values, dtype=dtype, device=device)[segment].reshape(-1, 1)
        else:
            s = _uniform((X.shape[0], 1), generator, dtype, device, float(self.s_values.min()), float(self.s_values.max()))
        return torch.cat([X, s], dim=1)

    def sample(self, n, generator, dtype, device):
        m = self._tensors(device)
        k = torch.multinomial(m["tet_w"], n, replacement=True, generator=generator)
        r = torch.sort(torch.rand((n, 3), generator=generator, dtype=torch.float64, device=device), dim=1).values
        bary = torch.stack([r[:, 0], r[:, 1] - r[:, 0], r[:, 2] - r[:, 1], 1.0 - r[:, 2]], dim=1)
        xyz = (bary.unsqueeze(2) * m["vert"][m["tet"][k]]).sum(dim=1)
        X = torch.cat([xyz.to(dtype), self._time(n, generator, dtype, device)], dim=1)
        return self._with_s(X, generator, dtype, device)

    def sample_vertices(self, n, generator, dtype, device):
        return self.sample_nodes(n, generator, dtype, device)["X"]

    def sample_faces(self, faces, n, generator, dtype, device):
        m = self._tensors(device)
        faces = torch.as_tensor(sorted({int(f) for f in np.atleast_1d(faces)}), dtype=torch.long, device=device)
        idx = torch.nonzero(torch.isin(m["tri_entity"], faces)).flatten()
        if idx.numel() == 0:
            raise ValueError(f"no boundary triangles on faces {faces.tolist()}; mesh has faces {self.faces}")
        k = idx[torch.multinomial(m["tri_w"][idx], n, replacement=True, generator=generator)]
        r = torch.rand((n, 2), generator=generator, dtype=torch.float64, device=device)
        fold = r.sum(dim=1) > 1.0
        r[fold] = 1.0 - r[fold]
        p0, p1, p2 = (m["vert"][m["tri"][k, i]] for i in range(3))
        xyz = p0 + r[:, 0:1] * (p1 - p0) + r[:, 1:2] * (p2 - p0)
        X = torch.cat([xyz.to(dtype), self._time(n, generator, dtype, device)], dim=1)
        return {"X": self._with_s(X, generator, dtype, device), "normal": m["tri_normal"][k].to(dtype),
                "node": None, "segment": None}

    def sample_nodes(self, n, generator, dtype, device, discrete_s=False):
        m = self._tensors(device)
        node = torch.randint(0, m["vert"].shape[0], (n,), generator=generator, device=device)
        X = torch.cat([m["vert"][node].to(dtype), self._time(n, generator, dtype, device)], dim=1)
        segment = None
        if self.s_values is not None and discrete_s:
            segment = torch.randint(0, int(self.s_values.size), (n,), generator=generator, device=device)
            s = torch.as_tensor(self.s_values, dtype=dtype, device=device)[segment].reshape(-1, 1)
            X = torch.cat([X, s], dim=1)
        else:
            X = self._with_s(X, generator, dtype, device)
        return {"X": X, "normal": None, "node": node, "segment": segment}

    def sample_boundary(self, where, n, generator, dtype, device):
        if "faces" in where:
            return {"": self.sample_faces(where["faces"], n, generator, dtype, device)}
        axis = int(where["axis"])
        lo, hi = self.bounds
        out = {}
        if axis == 3:
            base = self.sample_nodes(n, generator, dtype, device, discrete_s=True)
            for suffix, side in _sides(where):
                X = base["X"].clone()
                X[:, 3] = float((lo if side == "lo" else hi)[3])
                out[suffix] = {"X": X, "normal": None, "node": base["node"], "segment": base["segment"]}
            return out
        base = self.sample(n, generator, dtype, device)
        for suffix, side in _sides(where):
            X = base.clone()
            X[:, axis] = float((lo if side == "lo" else hi)[axis])
            normal = torch.zeros((n, 3), dtype=dtype, device=device)
            normal[:, axis] = -1.0 if side == "lo" else 1.0
            out[suffix] = {"X": X, "normal": normal, "node": None, "segment": None}
        return out
