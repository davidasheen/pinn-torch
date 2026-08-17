from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PredictMixin:

    def predict(self, X):
        param = next(self.parameters())
        self.eval()
        with torch.no_grad():
            y = self(torch.as_tensor(X, dtype=param.dtype, device=param.device))
        return y.cpu().numpy()


class FourierFeatures(nn.Module):

    def __init__(self, in_dim, num_features, scale=1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, num_features) * scale)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, x):
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


def _make_encoder(in_dim, fourier_features, fourier_scale):
    if not fourier_features:
        return None, in_dim
    enc = FourierFeatures(in_dim, fourier_features, fourier_scale)
    return enc, enc.out_dim


class FNN(_PredictMixin, nn.Module):

    def __init__(self, layers=(4, 64, 64, 64, 64, 4), fourier_features=0, fourier_scale=1.0):
        super().__init__()
        self.encoder, in_dim = _make_encoder(layers[0], fourier_features, fourier_scale)
        dims = (in_dim, *layers[1:])
        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.net(x)


class ModifiedMLP(_PredictMixin, nn.Module):

    def __init__(self, layers=(4, 64, 64, 64, 64, 4), fourier_features=0, fourier_scale=1.0):
        super().__init__()
        self.encoder, in_dim = _make_encoder(layers[0], fourier_features, fourier_scale)
        hidden_dims = layers[1:-1]
        out_dim = layers[-1]
        if len(set(hidden_dims)) != 1:
            raise ValueError(
                f"ModifiedMLP requires all hidden layers to share the same width, got {hidden_dims}"
            )
        hidden_dim = hidden_dims[0]

        self.activation = nn.Tanh()
        self.branch_u = nn.Linear(in_dim, hidden_dim)
        self.branch_v = nn.Linear(in_dim, hidden_dim)
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.gate_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(hidden_dims) - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        for m in [self.branch_u, self.branch_v, self.input_layer, *self.gate_layers, self.output_layer]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        u = self.activation(self.branch_u(x))
        v = self.activation(self.branch_v(x))
        h = self.activation(self.input_layer(x))
        for gate_layer in self.gate_layers:
            z = self.activation(gate_layer(h))
            h = (1 - z) * u + z * v
        return self.output_layer(h)


def build_network(architecture, layers, fourier_features=0, fourier_scale=1.0):
    kwargs = dict(fourier_features=fourier_features, fourier_scale=fourier_scale)
    if architecture == "fnn":
        return FNN(layers, **kwargs)
    if architecture == "modified_mlp":
        return ModifiedMLP(layers, **kwargs)
    raise ValueError(f"Unknown architecture={architecture!r}, expected 'fnn' or 'modified_mlp'")


def load_network_from_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    net = build_network(
        state.get("architecture", "fnn"),
        state["layers"],
        state.get("fourier_features", 0),
        state.get("fourier_scale", 1.0),
    )
    net = net.to(device=device, dtype=getattr(torch, state.get("dtype", "float32")))
    net.load_state_dict(state["model_state_dict"])
    net.eval()

    return net, state
