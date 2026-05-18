# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-18

Initial release.

### Added
- Posets: `Reals`, `Naturals`, `NamedProduct`, `Discrete`.
- `Antichain` with normalisation, `union_min`, `filter_above`.
- Six DP primitives: `AlgebraicDP`, `FunctionDP`, `CatalogDP`, `ConstraintDP`,
  `ODE_DP`, `UncertainDP`.
- Three composition operators: `Series`, `Parallel`, `Loop`
  (with `series`, `par`, `loop` aliases).
- Reusable building blocks: `adder`, `multiplier`, `scale`, `constant`,
  `identity`.
- Solver: `kleene_loop`, `solve`, `minimize_cost`, `SolveResult`.
- `MCDP` builder for MCDPL-style declarative composition (operator API).
- `System` builder for modular composition with named subsystems and
  algebraic connection constraints.
- Eight worked examples covering: the Fig. 48 drone (monolithic, MCDPL
  syntax, and modular forms), the Sec. VI-D integer optimisation with
  Kleene-trace visualisation, the Sec. VIII AUV seabed surveying, the
  `UncertainDP` brackets and `ODE_DP` steady states, and a motor +
  chassis + battery vehicle producing a multi-point Pareto front.
- Smoke tests covering posets, antichains, all primitives, all three
  composition operators, and the System builder.
- Eight executed Jupyter notebooks under `notebooks/`, one per example,
  with narrative prose and (for notebook 05) embedded Kleene-trace plots.
  Notebooks are regenerable via `python build_notebooks.py`.
