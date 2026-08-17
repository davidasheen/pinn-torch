from __future__ import annotations

import torch


def update_loss_weights_by_grad_norm(terms, net_params, weights, alpha):
    names = list(terms.keys())
    grad_norms = {}
    for i, name in enumerate(names):
        grads = torch.autograd.grad(
            terms[name], net_params, retain_graph=(i < len(names) - 1), allow_unused=True
        )
        flat = torch.cat([g.flatten() for g in grads if g is not None])
        grad_norms[name] = flat.abs().mean().item() + 1e-12

    max_norm = max(grad_norms.values())
    for name in names:
        target = max_norm / grad_norms[name]
        weights[name] = (1 - alpha) * weights[name] + alpha * target
