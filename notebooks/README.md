# Notebooks

Each notebook mirrors one example from `examples/`, with extra prose narrating
the model and discussing the results. The committed `.ipynb` files include
all outputs (including embedded matplotlib figures), so they can be browsed
on GitHub without running anything.

| # | Notebook | Topic |
|---|---|---|
| 01 | [`01_drone.ipynb`](01_drone.ipynb) | The Fig. 48 drone, monolithic form. |
| 02 | [`02_integer_optimization.ipynb`](02_integer_optimization.ipynb) | `x + y ≥ ⌈√x⌉ + ⌈√y⌉ + c` over ℕ×ℕ (Sec. VI-D), with full Kleene trace. |
| 03 | [`03_auv_seabed.ipynb`](03_auv_seabed.ipynb) | AUV seabed surveying with cyclic constraints (Sec. VIII). |
| 04 | [`04_uncertain_and_ode.ipynb`](04_uncertain_and_ode.ipynb) | `UncertainDP` brackets and `ODE_DP` steady states. |
| 05 | [`05_visualize_kleene.ipynb`](05_visualize_kleene.ipynb) | Plots the Kleene ascent (structure of Fig. 36). |
| 06 | [`06_drone_mcdpl_syntax.ipynb`](06_drone_mcdpl_syntax.ipynb) | Same drone, MCDPL-style declarative builder. |
| 07 | [`07_drone_modular.ipynb`](07_drone_modular.ipynb) | Same drone, modular composition: `Module` classes + operator-overloaded constraints. |
| 08 | [`08_vehicle_modular.ipynb`](08_vehicle_modular.ipynb) | Motor catalog + chassis + battery, multi-point Pareto front, with `>=` constraint syntax. |
| 09 | [`09_robotic_arm.ipynb`](09_robotic_arm.ipynb) | Five-module robotic arm exercising non-trivial cyclic constraints. |
| 10 | [`10_solver_trace.ipynb`](10_solver_trace.ipynb) | Solver observability: `trace`, `verbose`, `on_iteration`, and the `status` field, with a convergence plot. |
| 11 | [`11_uncertain_drone.ipynb`](11_uncertain_drone.ipynb) | Set-based deterministic uncertainty: `Box` vs `Ellipsoid` worst case. |
| 12 | [`12_stochastic_drone.ipynb`](12_stochastic_drone.ipynb) | Stochastic uncertainty with a Gaussian copula; mean, p95, CVaR95, and a histogram of the MC distribution. |

## Running locally

```bash
pip install -e ".[viz]"
pip install jupyter
jupyter lab notebooks/
```

## Regenerating

The notebooks are built from the script `build_notebooks.py` in the project
root. To rebuild them all (after a code change, say):

```bash
pip install nbformat nbconvert ipykernel matplotlib
python build_notebooks.py
```

This produces fresh `.ipynb` files with their outputs re-executed.
