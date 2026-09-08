from __future__ import annotations

import pathlib
import re

import numpy as np

import const
import json_util


def smooth_step(s, ncontder):
    s = np.clip(s, 0.0, 1.0)
    if ncontder <= 0:
        return s
    if ncontder == 1:
        return s * s * (3.0 - 2.0 * s)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def pulse_waveform(t, schedule):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for pulse in schedule:
        width = pulse["smooth"]
        if width > 0:
            rise = smooth_step((t - (pulse["lower"] - 0.5 * width)) / width, pulse["ncontder"])
            fall = smooth_step((t - (pulse["upper"] - 0.5 * width)) / width, pulse["ncontder"])
        else:
            rise = (t >= pulse["lower"]).astype(float)
            fall = (t >= pulse["upper"]).astype(float)
        out = out + pulse["baseline"] + (pulse["amplitude"] - pulse["baseline"]) * (rise - fall)
    return out


def tri_geometry(vertices, tri):
    p0, p1, p2 = vertices[tri[:, 0]], vertices[tri[:, 1]], vertices[tri[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(cross, axis=1)
    return 0.5 * norm, cross / np.maximum(norm, 1e-300)[:, None], (p0 + p1 + p2) / 3.0


def tet_volume(vertices, tet):
    a = vertices[tet[:, 1]] - vertices[tet[:, 0]]
    b = vertices[tet[:, 2]] - vertices[tet[:, 0]]
    c = vertices[tet[:, 3]] - vertices[tet[:, 0]]
    return np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0


def face_quadrature(mesh, faces):
    vertices = np.asarray(mesh["vertices"], float)
    faces = [int(f) for f in np.atleast_1d(faces)]
    tris, normals = [], []
    if "tri" in mesh:
        keep = np.isin(np.asarray(mesh["tri_entity"]), faces)
        tris.append(np.asarray(mesh["tri"], dtype=np.int64)[keep])
        normals.append(np.asarray(mesh["tri_normal"], float)[keep])
    if "quad" in mesh:
        keep = np.isin(np.asarray(mesh["quad_entity"]), faces)
        quad = np.asarray(mesh["quad"], dtype=np.int64)[keep]
        for idx in const.COMSOL_QUAD_TRIS:
            tris.append(quad[:, list(idx)])
            normals.append(np.asarray(mesh["quad_normal"], float)[keep])
    tri = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    if tri.shape[0] == 0:
        raise ValueError(f"no boundary element on faces {faces}")
    normal = np.concatenate(normals)
    area, _, _ = tri_geometry(vertices, tri)
    node_ids = np.unique(tri.ravel())
    W = np.zeros((node_ids.size, 3))
    slot = np.searchsorted(node_ids, tri)
    contrib = (area / 3.0)[:, None] * normal
    for j in range(3):
        np.add.at(W, slot[:, j], contrib)
    return node_ids, W


def slug(label):
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", label)).strip("_").lower()


def segment_labels(meta):
    out = {}
    for dataset in meta["datasets"]:
        match = re.match(r"\s*(pulse|purge)\s*(\d+)\s*//", str(dataset["label"]), re.IGNORECASE)
        if match is None:
            continue
        key = (match.group(1).lower(), int(match.group(2)))
        if key in out:
            raise ValueError(f"two datasets look like {key}: {out[key]!r} and {dataset['label']!r}")
        out[key] = dataset["label"]
    return out


def load_case(outdir):
    outdir = pathlib.Path(outdir)
    meta = json_util.load_json(outdir / "meta.json")
    mesh = dict(np.load(outdir / "mesh.npz"))
    fields = {d["label"]: outdir / f"field_{slug(d['label'])}.npz" for d in meta["datasets"]}
    return meta, mesh, fields


def match_nodes(vertices, spatial, tol=1e-9):
    vertices, spatial = np.asarray(vertices, float), np.asarray(spatial, float)
    if vertices.shape != spatial.shape:
        raise ValueError(f"vertices {vertices.shape} and spatial {spatial.shape} differ")
    key_v = np.round(vertices / tol).astype(np.int64)
    key_s = np.round(spatial / tol).astype(np.int64)
    order_v = np.lexsort(key_v.T[::-1])
    order_s = np.lexsort(key_s.T[::-1])
    if not np.array_equal(key_v[order_v], key_s[order_s]):
        raise ValueError("mesh vertices and field spatial are not the same point set (up to tol)")
    perm = np.empty(len(vertices), dtype=np.int64)
    perm[order_v] = order_s
    return perm


def geodesic_from_faces(mesh, faces):
    import scipy.sparse
    import scipy.sparse.csgraph

    vertices = np.asarray(mesh["vertices"], float)
    tet = np.asarray(mesh["tet"], dtype=np.int64)
    edges = [tet[:, [a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    for key in ("tri", "quad"):
        if key in mesh:
            poly = np.asarray(mesh[key], dtype=np.int64)
            edges += [poly[:, [i, (i + 1) % poly.shape[1]]] for i in range(poly.shape[1])]
    pairs = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
    weight = np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]], axis=1)
    n = vertices.shape[0]
    graph = scipy.sparse.coo_matrix((weight, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    faces = [int(f) for f in np.atleast_1d(faces)]
    source = np.unique(np.asarray(mesh["tri"])[np.isin(mesh["tri_entity"], faces)].ravel())
    if "quad" in mesh:
        source = np.union1d(source, np.unique(np.asarray(mesh["quad"])[np.isin(mesh["quad_entity"], faces)].ravel()))
    if source.size == 0:
        raise ValueError(f"no boundary element on faces {faces}")
    return scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=source, min_only=True)


def faces_of(meta, physics, role):
    roles = meta["boundary_conditions"][physics]["face_roles"]
    out = tuple(sorted(int(f) for f, r in roles.items() if role in r))
    if not out:
        have = sorted({x for r in roles.values() for x in r})
        raise KeyError(f"no face with role {role!r} in {physics!r}; roles present: {have}")
    return out
