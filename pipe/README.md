```
├── notebook/
│   ├── pinn_piston_flow.ipynb        ← torch
│   └── pinn_piston_flow_jax.ipynb    ← JAX
├── src/                              ← torch library
├── src_jax/                          ← JAX library
└── result/
    ├── pinn_piston_flow/             ← torch results
    │   ├── pipe/seed_{0,1,2}.json
    │   ├── channel/seed_{0,1,2}.json
    │   └── figure/*.png
    └── pinn_piston_flow_jax/         ← JAX results
        ├── pipe/seed_{0,1,2}.json
        ├── channel/seed_{0,1,2}.json
        └── figure/*.png
```

## JAX on GPU

```bash
conda create -n jax python=3.12 -y
conda activate jax
pip install "jax[cuda12]==0.11.2" equinox==0.13.8 optax==0.2.8 scipy matplotlib ipykernel
python -m ipykernel install --user --name jax --display-name jax --env XLA_PYTHON_CLIENT_PREALLOCATE false
python -c "import jax; print(jax.devices())"
```
