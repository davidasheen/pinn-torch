from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from typing import Dict, List, Union
import pinn_torch.differentiation as diff

def navier_stokes(
    X: List[torch.Tensor],
    fields: Dict[str, torch.Tensor],
    p: torch.Tensor,
    rho: Union[float, torch.Tensor],
    mu: Union[float, torch.Tensor],
    spatial_dims: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Computes the residuals for the incompressible Navier-Stokes equations.

    This implementation is agnostic to dimensionality, supporting both 2D and 3D
    spatial domains with a temporal dimension.

    Args:
        X: List of coordinate tensors [x, y, (z), t].
        fields: Dictionary containing the velocity fields (e.g., {'u': ..., 'v': ..., 'w': ...}).
        p: Pressure field tensor.
        rho: Fluid density (scalar or tensor).
        mu: Dynamic viscosity (scalar or tensor).
        spatial_dims: Number of spatial dimensions. Used to compute the Laplacian.

    Returns:
        A dictionary containing the residuals for continuity and momentum:
        - 'continuity': The divergence of the velocity field.
        - 'momentum_u', 'momentum_v', ...: The momentum residuals for each velocity component.
    """
    # Velocity vector: use all fields starting with 'u', 'v', 'w' etc in order
    vel_names = sorted([k for k in fields if k in ("u", "v", "w")])
    U = [fields[name] for name in vel_names]

    # Divergence of velocity: Continuity Equation
    continuity = diff.div_vector(U, X)

    momentum = []
    for i, u_i in enumerate(U):
        # Temporal derivative (assumed to be the last coordinate in X)
        u_it = diff.del_scalar(u_i, X)[-1]

        # Convective term: (U . grad) u_i
        u_i_grads = diff.del_scalar(u_i, X)
        convection = sum(U[j] * u_i_grads[j] for j in range(len(U)))

        # Pressure gradient
        p_grads = diff.del_scalar(p, X)
        grad_p_i = p_grads[i]

        # Diffusion term: mu * Laplacian(u_i)
        lap_u_i = diff.laplacian(u_i, X, spatial_dims=spatial_dims)

        # rho * (u_it + convection) + grad_p - mu * lap
        res = rho * (u_it + convection) + grad_p_i - mu * lap_u_i
        momentum.append(res)

    results = {"continuity": continuity}
    for i, name in enumerate(vel_names):
        results[f"momentum_{name}"] = momentum[i]

    return results

def heat(
    X: List[torch.Tensor],
    fields: Dict[str, torch.Tensor],
    alpha: Union[float, torch.Tensor],
    spatial_dims: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Computes the residual for the heat equation (diffusion of temperature).

    Args:
        X: List of coordinate tensors [x, y, ..., t].
        fields: Dictionary containing the temperature field {'T': ...}.
        alpha: Thermal diffusivity (scalar or tensor).
        spatial_dims: Number of spatial dimensions.

    Returns:
        Dictionary containing the residual: {'energy': T_t - alpha * Laplacian(T)}.
    """
    T = fields["T"]
    T_t = diff.del_scalar(T, X)[-1]
    T_lap = diff.laplacian(T, X, spatial_dims=spatial_dims)
    return {"energy": T_t - alpha * T_lap}

def species_soret(
    X: List[torch.Tensor],
    fields: Dict[str, torch.Tensor],
    rho: Union[float, torch.Tensor],
    D: Union[float, torch.Tensor],
    DT: Union[float, torch.Tensor],
    spatial_dims: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Computes the residual for species transport including the Soret effect (thermophoresis).

    Args:
        X: List of coordinate tensors [x, y, ..., t].
        fields: Dictionary containing velocity ('u', 'v', 'w'), species concentration ('Y'),
                and temperature ('T').
        rho: Density (scalar or tensor).
        D: Diffusion coefficient (scalar or tensor).
        DT: Soret coefficient (scalar or tensor).
        spatial_dims: Number of spatial dimensions.

    Returns:
        Dictionary containing the residual: {'species': rho*(Y_t + U.grad(Y)) - D*Lap(Y) - DT*Lap(T)}.
    """
    Y = fields["Y"]
    T = fields["T"]
    # Velocity vector for convection
    vel_names = sorted([k for k in fields if k in ("u", "v", "w")])
    U = [fields[name] for name in vel_names]

    Y_t = diff.del_scalar(Y, X)[-1]
    Y_grads = diff.del_scalar(Y, X)
    convection = sum(U[j] * Y_grads[j] for j in range(len(U)))

    Y_lap = diff.laplacian(Y, X, spatial_dims=spatial_dims)
    T_lap = diff.laplacian(T, X, spatial_dims=spatial_dims)

    return {"species": rho * (Y_t + convection) - D * Y_lap - DT * T_lap}

def diffusion(
    X: List[torch.Tensor],
    fields: Dict[str, torch.Tensor],
    D: Union[float, torch.Tensor],
    spatial_dims: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Computes the residual for the pure diffusion equation.

    Args:
        X: List of coordinate tensors [x, y, ..., t].
        fields: Dictionary containing the concentration field {'c': ...}.
        D: Diffusion coefficient (scalar or tensor).
        spatial_dims: Number of spatial dimensions.

    Returns:
        Dictionary containing the residual: {'species': c_t - D * Laplacian(c)}.
    """
    c = fields["c"]
    c_t = diff.del_scalar(c, X)[-1]
    c_lap = diff.laplacian(c, X, spatial_dims=spatial_dims)
    return {"species": c_t - D * c_lap}

def advection_diffusion(
    X: List[torch.Tensor],
    fields: Dict[str, torch.Tensor],
    D: Union[float, torch.Tensor],
    spatial_dims: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Computes the residual for the advection-diffusion equation.

    Args:
        X: List of coordinate tensors [x, y, ..., t].
        fields: Dictionary containing the concentration field ('c') and velocity fields ('u', 'v', 'w').
        D: Diffusion coefficient (scalar or tensor).
        spatial_dims: Number of spatial dimensions.

    Returns:
        Dictionary containing the residual: {'species': c_t + U.grad(c) - D * Laplacian(c)}.
    """
    c = fields["c"]
    vel_names = sorted([k for k in fields if k in ("u", "v", "w")])
    U = [fields[name] for name in vel_names]

    c_t = diff.del_scalar(c, X)[-1]
    c_grads = diff.del_scalar(c, X)
    convection = sum(U[j] * c_grads[j] for j in range(len(U)))

    c_lap = diff.laplacian(c, X, spatial_dims=spatial_dims)
    return {"species": c_t + convection - D * c_lap}
