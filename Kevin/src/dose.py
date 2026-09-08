from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import comsol_case
import const


def outlet_quadrature(mesh, meta):
    return comsol_case.face_quadrature(mesh, comsol_case.faces_of(meta, "tds", "outlet"))


def cfd_flux(field, perm, node_ids, W):
    rows = np.asarray(perm)[np.asarray(node_ids)]
    c = np.asarray(field["conc_data"])[rows, :, 0]
    v = np.asarray(field["vel_data"])[rows, :, 1:4]
    return np.einsum("it,itk,ik->t", c, v, np.asarray(W, float))


def known_flow_rate(meta):
    props = next(f for f in meta["boundary_conditions"]["spf"]["features"] if f.get("role") == "inlet")["properties"]
    number = lambda expr: float(str(expr).split("[")[0])
    q_std = meta["parameters"]["Fin"]["value"] / const.CFD_HALF_MODEL_FACTOR * const.SCCM
    return q_std / number(props["mnst"]["value"]) * number(props["Mn"]["value"]) / meta["transport"]["values"]["rho"]


def _outlet_sums(c_model, v_model, xyz, W, t_star, scales, c_fields, v_fields, s, chunk):
    param = next((c_model or v_model).parameters())
    dtype, device = param.dtype, param.device
    xyz = np.asarray(xyz, float) / float(scales["L"])
    t_star = np.asarray(t_star, float).reshape(-1)
    Wt = torch.as_tensor(np.asarray(W, float), dtype=dtype, device=device)
    c_col = {f: i for i, f in enumerate(c_fields)} if c_model is not None else {}
    v_col = {f: i for i, f in enumerate(v_fields)}
    sum_cf, sum_f = [], []
    for lo in range(0, t_star.size, int(chunk)):
        tt = t_star[lo:lo + int(chunk)]
        n_t, n_node = tt.size, xyz.shape[0]
        cols = [np.tile(xyz, (n_t, 1)), np.repeat(tt, n_node)[:, None]]
        if s is not None:
            cols.append(np.full((n_t * n_node, 1), float(s)))
        X = torch.as_tensor(np.column_stack(cols), dtype=dtype, device=device)
        with torch.no_grad():
            y = v_model(X.to(next(v_model.parameters()).dtype)).to(dtype)
            f = (y[:, v_col["u"]] * Wt[:, 0].repeat(n_t) + y[:, v_col["v"]] * Wt[:, 1].repeat(n_t)
                 + y[:, v_col["w"]] * Wt[:, 2].repeat(n_t))
            sum_f.append(f.reshape(n_t, n_node).sum(dim=1).cpu().numpy())
            if c_model is not None:
                c = c_model(X)[:, c_col["c"]]
                sum_cf.append((c * f).reshape(n_t, n_node).sum(dim=1).cpu().numpy())
    return (np.concatenate(sum_cf) if c_model is not None else None), np.concatenate(sum_f)


def pinn_flux(c_model, v_model, xyz, W, t_star, scales, c_fields, v_fields, q_known, s=None, chunk=64):
    sum_cf, sum_f = _outlet_sums(c_model, v_model, xyz, W, t_star, scales, c_fields, v_fields, s, chunk)
    return sum_cf / sum_f * np.asarray(q_known, float) * float(scales["C"])


def pinn_qvol(v_model, xyz, W, t_star, scales, v_fields, s=None, chunk=64):
    _, sum_f = _outlet_sums(None, v_model, xyz, W, t_star, scales, (), v_fields, s, chunk)
    return sum_f * float(scales["U"])


def window_dose(t, q, t_lo, t_hi):
    t, q = np.asarray(t, float), np.asarray(q, float)
    keep = (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    if keep.sum() < 2:
        raise ValueError(f"fewer than two samples inside [{t_lo}, {t_hi}]")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(q[keep], t[keep]))


def mass_rate(q_mol_s):
    return np.asarray(q_mol_s, float) * const.CFD_HALF_MODEL_FACTOR * const.CFD_PRECURSOR_MOLAR_MASS * 1e3
