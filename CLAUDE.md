# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- **Installation**: `pip install -e .`
- **Run Training Script**: `PYTHONPATH=. python Kevin/src/pinn_training.py`
- **Run Notebooks**: `jupyter notebook Kevin/notebook/`

## Architecture and Structure

### Core Library (`pinn_torch/`)
- `pinn.py`: Core PINN implementation and base classes.
- `differentiation.py`: Utilities for calculating PDE derivatives.

### Experiments (`Kevin/`)
- `src/`: Implementation of specific PDE problems.
    - `pde_residual.py`: Logic for defining and calculating PDE residuals.
    - `pinn_training.py`: Training orchestration via the `PINNTraining` class.
- `notebook/`: Research notebooks for data analysis and visualization.

### Workflow
1. **Data**: Define training datasets.
2. **Residuals**: Implement PDE residuals using `pinn_torch.differentiation`.
3. **Training**: Use `PINNTraining` to execute a two-stage optimization:
    - **Stage 1**: Adam optimizer for initial convergence.
    - **Stage 2**: L-BFGS optimizer for fine-tuning.
4. **Evaluation**: Visualize results via notebooks.
