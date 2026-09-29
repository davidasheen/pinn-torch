from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import pinn_torch as pinn

class PINN(nn.Module):
    def __init__(self, layers, lo, hi, feature=None, n_feature=0, n_param=0, num_centers=11):
        super().__init__()
        # layers is expected to be (input_dim, h1, h2, ..., h_n, output_dim)
        # We use these to configure the modular components

        # Domain normalization for RBF
        self.lo = torch.as_tensor(lo, dtype=torch.float64)
        self.hi = torch.as_tensor(hi, dtype=torch.float64)
        domain_sizes = (self.hi - self.lo).tolist()

        # 1. Input Encoding: Radial Basis Function
        # num_centers is now configurable to manage memory/capacity
        self.encoder = pinn.pinn.RadialBasisFunction(
            num_centers=num_centers,
            domain_sizes=domain_sizes
        )

        # 2. Backbone: Residual Fully Connected MLP
        # Input dim is the number of RBF centers
        num_inputs = self.encoder.rbf_centers.shape[1]
        nodes_per_layer = layers[1] if len(layers) > 1 else 32
        num_layers = len(layers) - 2 if len(layers) > 2 else 1

        self.backbone = pinn.pinn.ResidualFullyConnected(
            num_inputs=num_inputs,
            num_layers=num_layers,
            nodes_per_layer=nodes_per_layer
        )

        # 3. Output Head: PhysicsLayer
        # We assume all outputs are linear unless specified otherwise.
        # For the Kevin experiments, the output dimension is layers[-1].
        num_outputs = layers[-1]
        self.head_layer = pinn.pinn.PhysicsLayer(
            num_inputs=nodes_per_layer,
            num_linear_outputs=num_outputs
        )

        # 4. Composition: PhysicsHead or PhysicsHeadSteady
        # Use PhysicsHead if it's time-dependent (dim > 3 for x,y,z,t), else PhysicsHeadSteady
        if len(lo) > 3:
            self.model = pinn.pinn.PhysicsHead(
                layers=(self.encoder, self.backbone, self.head_layer),
                num_spatial=len(lo) - 1
            )
        else:
            self.model = pinn.pinn.PhysicsHeadSteady(
                layers=(self.encoder, self.backbone, self.head_layer),
                num_spatial=len(lo)
            )

        self.feature, self.n_feature = feature, int(n_feature)
        self.n_param = int(n_param)

    def forward(self, coords):
        # The Kevin training driver passes a single tensor (N, dim).
        if isinstance(coords, torch.Tensor):
            coords_list = [coords[:, i:i+1] for i in range(coords.shape[1])]
        else:
            coords_list = coords

        if isinstance(self.model, pinn.pinn.PhysicsHead):
            # Split into spatial and temporal parts
            spatial = coords_list[:-1]
            t = coords_list[-1]
            return self.model(spatial, t)
        else:
            # PhysicsHeadSteady expects only spatial
            return self.model(coords_list)

class Theta(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.spec = {}
        s0 = []
        for name, spec in specs.items():
            spec = dict(spec) if isinstance(spec, dict) else {"init": spec}
            init = float(spec["init"])
            bounds = spec.get("bounds")
            log = bool(spec.get("log", False))
            if bounds is None:
                s0.append(init)
            else:
                lo, hi = float(bounds[0]), float(bounds[1])
                if not lo < hi:
                    raise ValueError(f"{name}: bounds must satisfy lo < hi, got {bounds!r}")
                if log and lo <= 0:
                    raise ValueError(f"{name}: log bounds need lo > 0, got {bounds!r}")
                f = ((math.log(init) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log
                     else (init - lo) / (hi - lo))
                if not 0 < f < 1:
                    raise ValueError(f"{name}: init {init!r} must lie strictly inside bounds {bounds!r}")
                s0.append(math.log(f / (1 - f)))
                bounds = (lo, hi)
            self.spec[name] = {"init": init, "bounds": bounds, "log": log}
        self.names = tuple(self.spec)
        self.s = nn.Parameter(torch.tensor(s0, dtype=torch.float64))

    def value(self, name):
        i = self.names.index(name)
        spec = self.spec[name]
        if spec["bounds"] is None:
            return self.s[i]
        lo, hi = spec["bounds"]
        f = torch.sigmoid(self.s[i])
        if spec["log"]:
            return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * f)
        return lo + (hi - lo) * f

    def fractions(self):
        return {name: None if self.spec[name]["bounds"] is None else float(torch.sigmoid(self.s[i]).detach())
                for i, name in enumerate(self.names)}

    def values(self):
        return {name: float(self.value(name).detach()) for name in self.names}

    def describe(self):
        return {name: {"init": spec["init"], "bounds": None if spec["bounds"] is None else list(spec["bounds"]),
                       "log": spec["log"]} for name, spec in self.spec.items()}
