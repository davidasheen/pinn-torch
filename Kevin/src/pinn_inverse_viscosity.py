from __future__ import annotations

from pathlib import Path

import numpy as np
import skopt
import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layers=(4, 64, 64, 64, 64, 4)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def predict(self, X):
        param = next(self.parameters())
        self.eval()
        with torch.no_grad():
            y = self(torch.as_tensor(X, dtype=param.dtype, device=param.device))
        return y.cpu().numpy()


def _grad(f, coords):
    return torch.autograd.grad(f, coords, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def navier_stokes_residual(coords, y, rho, mu):
    u, v, w, p = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    grad_u, grad_v, grad_w, grad_p = _grad(u, coords), _grad(v, coords), _grad(w, coords), _grad(p, coords)
    u_x, u_y, u_z, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3], grad_u[:, 3:4]
    v_x, v_y, v_z, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3], grad_v[:, 3:4]
    w_x, w_y, w_z, w_t = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3], grad_w[:, 3:4]
    p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

    u_xx, u_yy, u_zz = _grad(u_x, coords)[:, 0:1], _grad(u_y, coords)[:, 1:2], _grad(u_z, coords)[:, 2:3]
    v_xx, v_yy, v_zz = _grad(v_x, coords)[:, 0:1], _grad(v_y, coords)[:, 1:2], _grad(v_z, coords)[:, 2:3]
    w_xx, w_yy, w_zz = _grad(w_x, coords)[:, 0:1], _grad(w_y, coords)[:, 1:2], _grad(w_z, coords)[:, 2:3]

    continuity = u_x + v_y + w_z
    momentum_u = rho * (u_t + u * u_x + v * u_y + w * u_z) + p_x - mu * (u_xx + u_yy + u_zz)
    momentum_v = rho * (v_t + u * v_x + v * v_y + w * v_z) + p_y - mu * (v_xx + v_yy + v_zz)
    momentum_w = rho * (w_t + u * w_x + v * w_y + w * w_z) + p_z - mu * (w_xx + w_yy + w_zz)

    return continuity, momentum_u, momentum_v, momentum_w


def run_inverse_viscosity(
    dataset,
    rho=1.0,
    mu_init=1.0,
    x_range=(0.0, 2 * np.pi),
    y_range=(0.0, 2 * np.pi),
    z_range=(0.0, 1.0),
    t_range=(0.0, 1.0),
    layers=(4, 64, 64, 64, 64, 4),
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
):
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training on device: {device}")

    checkpoint_every = checkpoint_every or log_every

    with torch.device(device):
        net = FNN(layers)
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

    def compute_loss():
        coords = torch.cat([X_data, domain_coords], dim=0)
        coords.requires_grad_(True)
        y_pde = net(coords)
        y_data = y_pde[:n_data]
        data_loss = (
            torch.mean((y_data[:, 0:1] - u_true) ** 2)
            + torch.mean((y_data[:, 1:2] - v_true) ** 2)
            + torch.mean((y_data[:, 2:3] - w_true) ** 2)
            + torch.mean((y_data[:, 3:4] - p_true) ** 2)
        )

        continuity, mom_u, mom_v, mom_w = navier_stokes_residual(coords, y_pde, rho, mu)
        pde_loss = (
            torch.mean(continuity**2) + torch.mean(mom_u**2) + torch.mean(mom_v**2) + torch.mean(mom_w**2)
        )

        return data_loss + pde_loss, data_loss, pde_loss

    params = list(net.parameters()) + [mu]
    optimizer = torch.optim.Adam(params, lr=adam_lr)
    history = {"iteration": [], "loss": [], "data_loss": [], "pde_loss": [], "mu": []}

    start_it = 0
    adam_done = False
    lbfgs_done = False

    if checkpoint_path is not None and resume and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if tuple(state["layers"]) != tuple(layers):
            raise ValueError(
                f"checkpoint layers={state['layers']} does not match the given "
                f"layers={layers}, cannot resume training directly -- either revert to the "
                "matching layers, or pass resume=False (or delete this checkpoint file) to start over."
            )
        net.load_state_dict(state["model_state_dict"])
        with torch.no_grad():
            mu.copy_(torch.tensor(state["mu_value"], dtype=mu.dtype, device=mu.device))
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        start_it = state["adam_it"]
        adam_done = state["adam_done"]
        lbfgs_done = state["lbfgs_done"]
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
            },
            checkpoint_path,
        )

    if not adam_done:
        it = start_it
        try:
            for it in range(start_it, adam_iterations):
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
                    print(
                        f"[adam] iter {it:6d}  loss={loss.item():.4e}  "
                        f"data={data_loss.item():.4e}  pde={pde_loss.item():.4e}  mu={mu.item():.5f}"
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
    }


def load_model(checkpoint_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    net = FNN(state["layers"]).to(device)
    net.load_state_dict(state["model_state_dict"])
    net.eval()

    return net, state["mu_value"]
