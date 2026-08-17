from __future__ import annotations

from pathlib import Path

import numpy as np
import skopt
import torch
import torch.nn as nn

from networks import (
    FNN,
    ModifiedMLP,
    build_network,
    load_network_from_checkpoint,
)
from loss_weighting import update_loss_weights_by_grad_norm
from pde_residuals import navier_stokes_residual


def run_inverse_viscosity(
    dataset,
    rho=1.0,
    mu_init=1.0,
    x_range=(0.0, 2 * np.pi),
    y_range=(0.0, 2 * np.pi),
    z_range=(0.0, 1.0),
    t_range=(0.0, 1.0),
    layers=(4, 64, 64, 64, 64, 4),
    architecture="fnn",
    fourier_features=0,
    fourier_scale=1.0,
    num_domain=4000,
    adam_iterations=15000,
    adam_lr=1e-3,
    log_every=500,
    run_lbfgs=True,
    lbfgs_iterations=15000,
    seed=0,
    device=None,
    checkpoint_path=None,
    checkpoint_every=None,
    resume=True,
    adaptive_loss_weights=False,
    loss_weight_alpha=0.1,
    loss_weight_update_every=None,
):
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training on device: {device}")

    checkpoint_every = checkpoint_every or log_every
    loss_weight_update_every = loss_weight_update_every or log_every

    with torch.device(device):
        net = build_network(architecture, layers, fourier_features, fourier_scale)
        mu = nn.Parameter(torch.tensor(float(mu_init), dtype=torch.float32))

        X_data = torch.tensor(dataset["X"], dtype=torch.float32)
        u_true = torch.tensor(dataset["u"], dtype=torch.float32)
        v_true = torch.tensor(dataset["v"], dtype=torch.float32)
        w_true = torch.tensor(dataset["w"], dtype=torch.float32)
        p_true = torch.tensor(dataset["p"], dtype=torch.float32)

    lo = np.array([x_range[0], y_range[0], z_range[0], t_range[0]])
    hi = np.array([x_range[1], y_range[1], z_range[1], t_range[1]])

    def sample_collocation(n, dtype):
        unit_cube = np.asarray(skopt.sampler.Hammersly().generate([(0.0, 1.0)] * 4, n + 1)[1:])
        points = lo + (hi - lo) * unit_cube
        with torch.device(device):
            return torch.tensor(points, dtype=dtype)

    domain_coords = sample_collocation(num_domain, dtype=X_data.dtype)

    n_data = X_data.shape[0]

    def compute_loss_terms(coords):
        y_pde = net(coords)
        y_data = y_pde[:n_data]
        data_loss = (
            torch.mean((y_data[:, 0:1] - u_true) ** 2)
            + torch.mean((y_data[:, 1:2] - v_true) ** 2)
            + torch.mean((y_data[:, 2:3] - w_true) ** 2)
            + torch.mean((y_data[:, 3:4] - p_true) ** 2)
        )

        u, v, w, p = y_pde[:, 0:1], y_pde[:, 1:2], y_pde[:, 2:3], y_pde[:, 3:4]
        continuity, mom_u, mom_v, mom_w = navier_stokes_residual(coords, u, v, w, p, rho, mu)
        continuity_loss = torch.mean(continuity**2)
        momentum_loss = torch.mean(mom_u**2) + torch.mean(mom_v**2) + torch.mean(mom_w**2)

        return {"data": data_loss, "continuity": continuity_loss, "momentum": momentum_loss}

    def compute_loss():
        coords = torch.cat([X_data, domain_coords], dim=0)
        coords.requires_grad_(True)
        terms = compute_loss_terms(coords)
        pde_loss = terms["continuity"] + terms["momentum"]

        if loss_weights is None:
            total_loss = terms["data"] + pde_loss
        else:
            total_loss = (
                loss_weights["data"] * terms["data"]
                + loss_weights["continuity"] * terms["continuity"]
                + loss_weights["momentum"] * terms["momentum"]
            )

        return total_loss, terms["data"], pde_loss

    def update_loss_weights():
        coords = torch.cat([X_data, domain_coords], dim=0)
        coords.requires_grad_(True)
        terms = compute_loss_terms(coords)
        update_loss_weights_by_grad_norm(terms, list(net.parameters()), loss_weights, loss_weight_alpha)

    loss_weights = {"data": 1.0, "continuity": 1.0, "momentum": 1.0} if adaptive_loss_weights else None

    params = list(net.parameters()) + [mu]
    optimizer = torch.optim.Adam(params, lr=adam_lr)
    history = {"iteration": [], "loss": [], "data_loss": [], "pde_loss": [], "mu": [], "loss_weights": []}

    start_it = 0
    adam_done = False
    lbfgs_done = False

    if checkpoint_path is not None and resume and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if tuple(state["layers"]) != tuple(layers):
            raise ValueError(
                f"Network structure in checkpoint layers={state['layers']} does not match "
                f"the currently passed layers={layers}, cannot resume directly -- either "
                "change layers back to match, or pass resume=False (or delete the checkpoint "
                "file) to start fresh."
            )
        ckpt_architecture = state.get("architecture", "fnn")
        if ckpt_architecture != architecture:
            raise ValueError(
                f"Network architecture in checkpoint architecture={ckpt_architecture!r} does not match "
                f"the currently passed architecture={architecture!r}, cannot resume directly -- either "
                "change architecture back to match, or pass resume=False (or delete the checkpoint "
                "file) to start fresh."
            )
        ckpt_ff = int(state.get("fourier_features", 0))
        ckpt_fs = float(state.get("fourier_scale", 1.0))
        if ckpt_ff != int(fourier_features) or (ckpt_ff and ckpt_fs != float(fourier_scale)):
            raise ValueError(
                f"Fourier feature encoding in checkpoint (features={ckpt_ff}, scale={ckpt_fs}) does not "
                f"match the currently passed (features={fourier_features}, scale={fourier_scale}), "
                "cannot resume directly -- either change them back to match, or pass resume=False "
                "(or delete the checkpoint file) to start fresh."
            )
        net.load_state_dict(state["model_state_dict"])
        with torch.no_grad():
            mu.copy_(torch.tensor(state["mu_value"], dtype=mu.dtype, device=mu.device))
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        history.setdefault("loss_weights", [])
        start_it = state["adam_it"]
        adam_done = state["adam_done"]
        lbfgs_done = state["lbfgs_done"]
        if loss_weights is not None:
            loss_weights = state.get("loss_weights") or loss_weights
        print(
            f"Resuming from checkpoint: {checkpoint_path} (completed adam_it={start_it}/{adam_iterations}, "
            f"adam_done={adam_done}, lbfgs_done={lbfgs_done})"
        )

    def save_checkpoint(adam_it):
        if checkpoint_path is None:
            return
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "mu_value": mu.item(),
                "optimizer_state_dict": optimizer.state_dict(),
                "adam_it": adam_it,
                "adam_done": adam_done,
                "lbfgs_done": lbfgs_done,
                "history": history,
                "layers": layers,
                "architecture": architecture,
                "fourier_features": fourier_features,
                "fourier_scale": fourier_scale,
                "loss_weights": loss_weights,
            },
            checkpoint_path,
        )

    if not adam_done:
        it = start_it
        try:
            for it in range(start_it, adam_iterations):
                if loss_weights is not None and it % loss_weight_update_every == 0:
                    update_loss_weights()
                    history["loss_weights"].append({"iteration": it, **loss_weights})

                optimizer.zero_grad()
                loss, data_loss, pde_loss = compute_loss()
                loss.backward()
                optimizer.step()

                if it % log_every == 0 or it == adam_iterations - 1:
                    history["iteration"].append(it)
                    history["loss"].append(loss.item())
                    history["data_loss"].append(data_loss.item())
                    history["pde_loss"].append(pde_loss.item())
                    history["mu"].append(mu.item())
                    weight_str = (
                        f"  weights={{{', '.join(f'{k}={v:.3g}' for k, v in loss_weights.items())}}}"
                        if loss_weights is not None
                        else ""
                    )
                    print(
                        f"[adam] iter {it:6d}  loss={loss.item():.4e}  "
                        f"data={data_loss.item():.4e}  pde={pde_loss.item():.4e}  mu={mu.item():.5f}{weight_str}"
                    )

                if (it + 1) % checkpoint_every == 0:
                    save_checkpoint(it + 1)
        except KeyboardInterrupt:
            save_checkpoint(it)
            print(f"[adam] Training interrupted, checkpoint saved at iter {it}; rerun this function to resume")
            raise

        adam_done = True
        save_checkpoint(adam_iterations)
    else:
        print("[adam] Already completed (from checkpoint), skipping")

    if run_lbfgs and lbfgs_done:
        print("[lbfgs] Already completed (from checkpoint), skipping")
    elif run_lbfgs:

        lbfgs = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=lbfgs_iterations,
            max_eval=int(lbfgs_iterations * 1.25),
            tolerance_grad=1e-8,
            tolerance_change=0,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0
        last_checkpointed_n_iter = 0

        def closure():
            nonlocal closure_calls, last_checkpointed_n_iter
            lbfgs.zero_grad()
            loss, data_loss, pde_loss = compute_loss()
            loss.backward()
            closure_calls += 1
            if closure_calls % log_every == 0:
                print(
                    f"[lbfgs] closure call {closure_calls:6d}  loss={loss.item():.4e}  "
                    f"data={data_loss.item():.4e}  pde={pde_loss.item():.4e}  mu={mu.item():.5f}"
                )

            n_iter = lbfgs.state[params[0]].get("n_iter", 0)
            if n_iter - last_checkpointed_n_iter >= checkpoint_every:
                last_checkpointed_n_iter = n_iter
                history["iteration"].append(adam_iterations + n_iter)
                history["loss"].append(loss.item())
                history["data_loss"].append(data_loss.item())
                history["pde_loss"].append(pde_loss.item())
                history["mu"].append(mu.item())
                save_checkpoint(adam_iterations)

            return loss

        try:
            lbfgs.step(closure)
        except KeyboardInterrupt:
            save_checkpoint(adam_iterations)
            print("[lbfgs] Training interrupted, checkpoint saved for current parameter state")
            raise
        n_iter = lbfgs.state[params[0]].get("n_iter", 0)

        final_loss, final_data_loss, final_pde_loss = compute_loss()
        history["iteration"].append(adam_iterations + n_iter)
        history["loss"].append(final_loss.item())
        history["data_loss"].append(final_data_loss.item())
        history["pde_loss"].append(final_pde_loss.item())
        history["mu"].append(mu.item())
        hit_cap = " (hit max_iter cap, may not have converged)" if n_iter >= lbfgs_iterations else ""
        print(
            f"[lbfgs] final: {n_iter}/{lbfgs_iterations} outer iterations, "
            f"{closure_calls} closure calls{hit_cap}"
        )
        print(f"[lbfgs] final loss={final_loss.item():.4e}  mu={mu.item():.5f}")

        lbfgs_done = True
        save_checkpoint(adam_iterations)

    return {
        "model": net,
        "mu": mu,
        "mu_recovered": mu.item(),
        "history": history,
        "architecture": architecture,
        "loss_weights": loss_weights,
    }


def load_model(checkpoint_path, device=None):
    net, state = load_network_from_checkpoint(checkpoint_path, device=device)
    return net, state["mu_value"]
