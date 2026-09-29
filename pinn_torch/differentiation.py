import numpy as np
import torch

def del_scalar(f, X):
    """
    Computes the gradient of scalar function f with respect to Cartesian coordinates X.

    Args:
        f: Scalar tensor.
        X: List or tuple of coordinate tensors [x1, x2, ..., xn].

    Returns:
        A tuple of tensors (df/dx1, df/dx2, ..., df/dxn).
    """
    return torch.autograd.grad(f.sum(), X, create_graph=True)


def grad_scalar(f, X, coord_system='cartesian'):
    """Compute the gradient of a scalar function with respect to the given
    coordinate system.

    For ``'cartesian'`` this is identical to :func:`del_scalar`.
    For ``'cylindrical'`` the returned tuple is ``(∂f/∂r, (1/r)∂f/∂θ, ∂f/∂z)``.

    Args:
        f: Scalar tensor.
        X: List or tuple of coordinate tensors. For cylindrical coordinates
           ``X`` should be ``(r, theta, z)``.
        coord_system: Either ``'cartesian'`` (default) or ``'cylindrical'``.

    Returns:
        A tuple of gradient components appropriate for the chosen coordinate
        system.
    """
    grads = del_scalar(f, X)
    if coord_system == 'cartesian':
        return grads
    elif coord_system == 'cylindrical':
        if len(X) < 2:
            raise ValueError('Cylindrical coordinates require at least r and theta')
        r = X[0]
        # Avoid division by zero – rely on autograd to propagate NaNs if r==0
        theta_grad = grads[1] / r
        # Preserve any additional dimensions (e.g., z) unchanged
        if len(grads) > 2:
            return (grads[0], theta_grad, grads[2])
        else:
            return (grads[0], theta_grad)
    else:
        raise ValueError(f"Unsupported coordinate system: {coord_system}")

def div_vector(F, X, coord_system='cartesian'):
    """
    Computes the divergence of vector function F.

    Args:
        F: List or tuple of scalar fields [f1, f2, ..., fn].
        X: List or tuple of coordinate tensors [x1, x2, ..., xn].
        coord_system: Coordinate system ('cartesian' or 'cylindrical').

    Returns:
        The divergence.
    """
    if coord_system == 'cartesian':
        return sum(del_scalar(F[i], X)[i] for i in range(len(F)))
    elif coord_system == 'cylindrical':
        # F = [Fr, Ftheta, Fz], X = [r, theta, z]
        # div F = dFr/dr + Fr/r + (1/r) dFtheta/dtheta + dFz/dz
        r = X[0]
        df_dr = del_scalar(F[0], X)[0]
        df_dtheta = del_scalar(F[1], X)[1]
        df_dz = del_scalar(F[2], X)[2]
        return df_dr + F[0]/r + (1/r)*df_dtheta + df_dz
    else:
        raise ValueError(f"Unsupported coordinate system: {coord_system}")

def laplacian(f, X, spatial_dims=None, coord_system='cartesian'):
    """
    Computes the spatial Laplacian of scalar function f.

    Args:
        f: Scalar tensor.
        X: List or tuple of coordinate tensors [x1, x2, ..., xn].
        spatial_dims: Number of spatial dimensions to sum over.
                     If None, uses all dimensions in X.
        coord_system: Coordinate system ('cartesian' or 'cylindrical').

    Returns:
        The Laplacian.
    """
    if coord_system == 'cartesian':
        if spatial_dims is None:
            spatial_dims = len(X)

        grad_f = del_scalar(f, X)
        return sum(del_scalar(grad_f[i], X)[i] for i in range(spatial_dims))
    elif coord_system == 'cylindrical':
        # X = [r, theta, z]
        # lap f = d^2f/dr^2 + (1/r) df/dr + (1/r^2) d^2f/dtheta^2 + d^2f/dz^2
        r = X[0]
        df_dr = del_scalar(f, X)[0]
        df_dtheta = del_scalar(f, X)[1]
        df_dz = del_scalar(f, X)[2]
        return del_scalar(df_dr, X)[0] + (1/r)*df_dr + (1/(r**2))*del_scalar(df_dtheta, X)[1] + del_scalar(df_dz, X)[2]
    else:
        raise ValueError(f"Unsupported coordinate system: {coord_system}")

def curl_vector(F, X, coord_system='cartesian'):
    """
    Computes the curl of vector function F.
    Currently supports 2D and 3D.
    """
    if coord_system == 'cartesian':
        # Compute Jacobian
        # jac[i][j] = dF_i / dX_j
        jac = [del_scalar(f, X) for f in F]

        dims = len(X)
        if dims == 3:
            # curl F = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
            return (
                jac[2][1] - jac[1][2],
                jac[0][2] - jac[2][0],
                jac[1][0] - jac[0][1]
            )
        elif dims == 2:
            # 2D curl is a scalar: dF_y/dx - dF_x/dy
            return jac[1][0] - jac[0][1]
        else:
            raise ValueError(f"Cartesian curl_vector only supported for 2D and 3D, got {dims}D")
    elif coord_system == 'cylindrical':
        # F = [Fr, Ftheta, Fz], X = [r, theta, z]
        r = X[0]
        # curl F = ( (1/r) dFz/dtheta - dFtheta/dz ) er + ( dFr/dz - dFz/dr ) etheta + ( 1/r ) ( d(r*Ftheta)/dr - dFr/dtheta ) ez
        dfz_dtheta = del_scalar(F[2], X)[1]
        dftheta_dz = del_scalar(F[1], X)[2]
        dfr_dz = del_scalar(F[0], X)[2]
        dfz_dr = del_scalar(F[2], X)[0]
        dftheta_dr = del_scalar(F[1], X)[0]
        dfr_dtheta = del_scalar(F[0], X)[1]

        curl_r = (1/r) * dfz_dtheta - dftheta_dz
        curl_theta = dfr_dz - dfz_dr
        curl_z = (1/r) * (F[1] + r * dftheta_dr - dfr_dtheta)
        return (curl_r, curl_theta, curl_z)
    else:
        raise ValueError(f"Unsupported coordinate system: {coord_system}")
