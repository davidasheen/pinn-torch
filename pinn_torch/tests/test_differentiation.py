# tests/test_differentiation.py
"""
Unit tests for pinn_torch.differentiation

These tests cover:
* del_scalar (Cartesian gradients)
* div_vector   – Cartesian & cylindrical
* laplacian    – Cartesian & cylindrical
* curl_vector  – Cartesian & cylindrical
"""

import math

import torch
import pytest

# Import the module under test
from pinn_torch.differentiation import (
    del_scalar,
    grad_scalar,
    div_vector,
    laplacian,
    curl_vector,
)

def tensor(val):
    """Helper to create a float tensor with gradient tracking."""
    return torch.tensor([val], dtype=torch.float64, requires_grad=True)

# ----------------------------------------------------------------------
# 1. del_scalar – basic Cartesian gradient
# ----------------------------------------------------------------------
def test_del_scalar_cartesian():
    x = tensor(2.0)
    y = tensor(3.0)
    f = x ** 2 + y ** 2  # f = x^2 + y^2

    grads = del_scalar(f, (x, y))

    # Expected: df/dx = 2*x, df/dy = 2*y
    assert torch.allclose(grads[0], torch.tensor([4.0]))
    assert torch.allclose(grads[1], torch.tensor([6.0]))

def test_grad_scalar_cartesian_matches_del_scalar():
    x = tensor(2.0)
    y = tensor(3.0)
    f = x ** 2 + y ** 2
    # grad_scalar should be identical to del_scalar for Cartesian
    grads_del = del_scalar(f, (x, y))
    grads_grad = grad_scalar(f, (x, y), coord_system='cartesian')
    for g1, g2 in zip(grads_del, grads_grad):
        assert torch.allclose(g1, g2)

def test_grad_scalar_cylindrical_basic():
    # f = r * theta   -> df/dr = theta, df/dtheta = r
    r = tensor(2.0)
    theta = tensor(0.5)
    z = tensor(0.0)  # not used
    f = r * theta
    grads = grad_scalar(f, (r, theta, z), coord_system='cylindrical')
    # Expected: (∂f/∂r = theta, (1/r)∂f/∂theta = (1/r)*r = 1, ∂f/∂z = 0)
    assert torch.allclose(grads[0], theta)           # df/dr
    assert torch.allclose(grads[1], torch.tensor([1.0]))  # (1/r) df/dtheta
    assert torch.allclose(grads[2], torch.tensor([0.0]))  # df/dz

def test_grad_scalar_cylindrical_zero_r():
    # Verify that division by zero propagates NaN (autograd behavior)
    r = tensor(0.0)
    theta = tensor(1.0)
    z = tensor(0.0)
    f = r * theta
    grads = grad_scalar(f, (r, theta, z), coord_system='cylindrical')
    # df/dr = theta = 1, (1/r) df/dtheta = NaN, df/dz = 0
    assert torch.allclose(grads[0], torch.tensor([1.0]))
    assert torch.isnan(grads[1])
    assert torch.allclose(grads[2], torch.tensor([0.0]))

# ----------------------------------------------------------------------
# 2. Divergence
# ----------------------------------------------------------------------
def test_div_vector_cartesian():
    # Simple linear field: F = (x, y, z)
    x = tensor(1.0)
    y = tensor(2.0)
    z = tensor(3.0)

    F = (x, y, z)
    X = (x, y, z)

    div = div_vector(F, X, coord_system="cartesian")
    # ∂x/∂x = 1, ∂y/∂y = 1, ∂z/∂z = 1  →  divergence = 3
    assert torch.allclose(div, torch.tensor([3.0]))

def test_div_vector_cylindrical():
    # Field: F = (r, theta, z)  in cylindrical coords
    r = tensor(1.0)          # avoid division by zero
    theta = tensor(math.pi)  # any value, derivative of theta w.r.t. theta = 1
    z = tensor(0.5)

    F = (r, theta, z)
    X = (r, theta, z)

    div = div_vector(F, X, coord_system="cylindrical")
    # Analytical result:  dFr/dr + Fr/r + (1/r) dFθ/dθ + dFz/dz
    #   = 1 + 1/1 + (1/1)*1 + 1 = 4
    assert torch.allclose(div, torch.tensor([4.0]))

# ----------------------------------------------------------------------
# 3. Laplacian
# ----------------------------------------------------------------------
def test_laplacian_cartesian():
    # f = x^2 + y^2 + z^2  →  ∇²f = 2 + 2 + 2 = 6
    x = tensor(1.0)
    y = tensor(-2.0)
    z = tensor(0.5)

    f = x ** 2 + y ** 2 + z ** 2
    X = (x, y, z)

    lap = laplacian(f, X, spatial_dims=3, coord_system="cartesian")
    assert torch.allclose(lap, torch.tensor([6.0]))

def test_laplacian_cylindrical():
    # f = r^2  →  ∇²f = d²f/dr² + (1/r) df/dr = 2 + 2 = 4
    r = tensor(1.0)
    theta = tensor(0.0)  # dummy, not used
    z = tensor(0.0)      # dummy, not used

    f = r ** 2
    X = (r, theta, z)

    lap = laplacian(f, X, coord_system="cylindrical")
    assert torch.allclose(lap, torch.tensor([4.0]))

# ----------------------------------------------------------------------
# 4. Curl
# ----------------------------------------------------------------------
def test_curl_vector_cartesian_3d():
    # Choose a field whose curl we know analytically:
    #   F = (0, 0, f) with f = x*y
    #   curl F = (∂f/∂y, -∂f/∂x, 0) = (x, -y, 0)
    x = tensor(1.5)
    y = tensor(-0.7)
    z = tensor(0.0)  # not used

    f = x * y
    F = (torch.zeros_like(x), torch.zeros_like(y), f)
    X = (x, y, z)

    curl = curl_vector(F, X, coord_system="cartesian")
    expected = (x, -y, torch.tensor([0.0]))
    for c, e in zip(curl, expected):
        assert torch.allclose(c, e)

def test_curl_vector_cylindrical_zero():
    # Purely radial field F = (r, 0, 0) in cylindrical coordinates has zero curl
    r = tensor(2.0)
    theta = tensor(0.0)
    z = tensor(1.0)

    F = (r, torch.zeros_like(theta), torch.zeros_like(z))
    X = (r, theta, z)

    curl = curl_vector(F, X, coord_system="cylindrical")
    zero = torch.tensor([0.0])

    for component in curl:
        assert torch.allclose(component, zero)

# ----------------------------------------------------------------------
# 5. Guard against mismatched component length (cylindrical only)
# ----------------------------------------------------------------------
def test_div_vector_cylindrical_invalid_length():
    r = tensor(1.0)
    theta = tensor(0.0)
    # Missing the third component (z) should raise a clear error
    F = (r, theta)          # only two components
    X = (r, theta, tensor(0.0))

    with pytest.raises(IndexError):
        # The function will attempt to access F[2] -> IndexError
        div_vector(F, X, coord_system="cylindrical")
