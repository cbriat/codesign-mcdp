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
  algebraic connection constraints. Supports two equivalent surface
  syntaxes:
    - Operator-overloaded: `provides`, `requires`, and `add` return port
      handles; arithmetic operators on handles build expression trees;
      `>=` between a port handle and an expression registers a constraint.
      Reads like the textbook inequalities. Type errors (constraining an
      outer F, using a module F port as a value, comparing ports from
      different systems) are caught at expression-build time with
      explanatory messages.
    - Legacy lambda form (`sys.constrain("module.f_port", lambda x: ...)`)
      still supported; both styles compile to the same internal
      constraint list and produce identical results.
- `Module` declarative base class: subclass and define `F`, `R` as
  class-level dicts plus an `h(self, f)` method to get a fully-formed
  `DesignProblem`. Parameterised modules via overridden `__init__`.
- Helper functions `sqrt`, `exp`, `log` usable inside constraint
  expressions.
- Nine worked examples covering: the Fig. 48 drone (monolithic, MCDPL
  syntax, and modular forms), the Sec. VI-D integer optimisation with
  Kleene-trace visualisation, the Sec. VIII AUV seabed surveying, the
  `UncertainDP` brackets and `ODE_DP` steady states, a motor + chassis +
  battery vehicle producing a multi-point Pareto front, and a five-module
  robotic arm exercising non-trivial cyclic constraints.
- Smoke tests covering posets, antichains, all primitives, all three
  composition operators, the System builder, the operator-overloaded
  DSL, and the type-error guards for misused ports.
- Nine executed Jupyter notebooks under `notebooks/`, one per example,
  with narrative prose and (for notebook 05) embedded Kleene-trace plots.
  Notebooks are regenerable via `python build_notebooks.py`.
- Full LaTeX reference manual under `docs/manual/`, covering the
  theoretical background, all data types, primitives, composition
  operators, both builders (with both surface syntaxes), the solver,
  every worked example, and modelling guidelines. Pre-built PDF
  (`codesign-mcdp-manual.pdf`) committed alongside the LaTeX source.
  Rebuilt with `make` in that directory.
