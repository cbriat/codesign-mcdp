# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Renamed `NamedProduct` to `Ports`** to match the library's everyday
  vocabulary (port handles, outer F port, module R port, the operator
  DSL is built on port handles). The old name is retained as a
  backward-compatible alias (`NamedProduct = Ports`), so existing code
  importing `NamedProduct` continues to work. All internal modules,
  examples, notebooks, and documentation have been migrated to the new
  name; the LaTeX manual now uses `Ports` throughout and explains the
  alias.
- Module-level docstrings in `codesign/posets.py` expanded with worked
  rationale for each class and a clearer summary at the top.

### Added

#### Online learning (compositional, elimination-based)
- New `codesign.online` module implementing the optimistic-evaluator
  solver from Alharbi, Dahleh & Zardini (arXiv:2604.22624).
- `OptimisticEvaluator` abstract base maintains an observation history
  and exposes a `bound(candidate) -> (lower, upper)` interface; the
  default fallback is `(0, +inf)` for every R component.
- Three concrete evaluators:
    - `MonotonicityEvaluator(features, r_components)` — assumes
      component-wise monotonicity in the features. Aggressive when
      applicable, only correct if monotonicity genuinely holds.
    - `LipschitzEvaluator(features, r_components, L)` — assumes
      Lipschitz output with a user-supplied constant. Safe default
      across most problems; `L` can be a scalar or a dict per R component.
    - `LinearParametricEvaluator(features, r_components,
      confidence=2.0, min_obs=3)` — fits a running OLS model and bounds
      by a confidence band on the regressor. Fastest in practice but
      least safe when the linear assumption breaks.
- `solve_online(candidate_fn, functionality, *, candidates, evaluator,
  budget=None)` runs the elimination loop: bound, pick the most
  promising survivor by UCB on lower bound, run the inner solve via
  `codesign.solver.solve`, merge into the incumbent antichain, prune
  newly dominated candidates, repeat until the candidates are
  exhausted or the budget is hit.
- `OnlineResult` dataclass with `antichain`, `n_evaluated`,
  `n_eliminated`, `n_candidates`, `history` (per-iteration log),
  `evaluated_ids`, `eliminated_ids`, and `incumbent_ids` (which
  evaluations contributed to the final antichain).
- New example `14_online_fleet.py`: 200-candidate heterogeneous robot
  fleet sizing, side-by-side comparison of the three evaluators with
  a feature-space elimination plot.
- New notebook `notebooks/14_online_fleet.ipynb` covering the same
  case study with an explanatory walk-through.
- New smoke test `test_online_solver`.

#### Visualisation helpers
- New `codesign.viz` module, importable as `from codesign import viz`.
- `viz.plot_antichain(result, axes)` renders the Pareto front as a 2D
  or 3D scatter (accepts a `SolveResult`, `UncertaintyResult`, or
  bare `Antichain`); optionally shades dominated regions.
- `viz.plot_convergence(result)` plots the Kleene delta-vs-iteration
  on a log axis; works on a `SolveResult` with a trace or a trace
  list directly.
- `viz.plot_uncertainty(unc_result, port, nominal=None)` draws a
  histogram of the MC samples for the named R port and marks the
  nominal, mean, p95, CVaR95 summaries.
- `viz.to_dot(dp, name=...)` produces a GraphViz dot string showing
  the system's modules, outer ports, and connection constraints.
- All helpers accept an optional `ax=` to compose into a larger figure.

#### Solver warm-start
- `solve(dp, f, ..., start_from=prev)` seeds the Kleene iteration
  from a previously computed `SolveResult` (or `Antichain`). The
  inner antichain is reused as the initial `A_0` for the new solve.
- `SolveResult._inner_antichain` carries the converged inner-loop
  antichain so it can be passed straight back as a warm start.
- Sweep tests of the microgrid example show roughly 10% fewer total
  Kleene iterations under warm-start versus a cold start at each
  parameter point.

#### Flagship microgrid case study
- New example `13_microgrid.py`: solar PV + lithium battery +
  diesel generator + structural frame with a cyclic mass coupling.
- Exercises: catalog choice over four battery chemistries, warm-
  started parameter sweep, stochastic sun-hours via `Stochastic`,
  and every visualisation helper.
- New notebook `notebooks/13_microgrid.ipynb`.

#### Solver observability
- `TraceEntry` dataclass capturing per-iteration state: `iteration`,
  `antichain`, `n_points`, `delta` (max absolute change for numeric
  posets, set-change indicator otherwise), and `elapsed_ms`.
- `solve(..., trace=True)` collects a list of `TraceEntry` on
  `result.trace`. Default is `False`, so existing call sites pay nothing.
- `solve(..., verbose=0|1|2)` controls live printing: silent, one summary
  line at the end, or a per-iteration progress feed.
- `solve(..., on_iteration=callable)` callback receives each `TraceEntry`
  as it is produced, suitable for live plots or custom logging.
- New `SolveResult.status` field with three values: `"converged"`,
  `"max_iter"`, `"diverged"`. Orthogonal to `feasible`. The previous
  `converged` field is preserved as a backward-compat alias for
  `status == "converged"`.
- Divergence guard: when any numeric value in the antichain crosses
  `DIVERGENCE_CAP = 1e30` before the iteration settles, the solver
  stops with `status="diverged"`. Distinguishes numerical blow-up from
  clean ⊤-infeasibility.
- Legacy `record_trace=True` and `trace_out=[...]` keyword arguments are
  preserved as backward-compatible aliases.

#### Uncertainty layer
- `UncertaintySet` abstract base for deterministic, set-based parameter
  uncertainty, with concrete implementations:
    - `Box(name=(lo, hi[, direction]), ...)`: axis-aligned interval
      product. Each parameter can carry a "direction of badness" token
      (`"more_is_better"`, `"more_is_worse"`, `"less_is_better"`,
      `"less_is_worse"`); declared directions give an analytic worst case,
      undeclared directions trigger a 2^n endpoint search.
    - `Ellipsoid(center, cov, params, directions=None,
      boundary_samples=8)`: n-D ellipsoid in parameter space. Analytic
      worst case when all directions are declared; boundary sampling
      otherwise.
    - 2D conveniences `Disk(center, radius, ...)` and
      `Circle(center, radius, ...)` reduce to isotropic ellipsoids.
- `Stochastic(marginals, copula=Independence())`: joint distribution
  built from scipy-stats frozen marginals plus a copula.
- Copulas: `Independence()` (default) and
  `GaussianCopula(correlation=...)`, sampled by Cholesky factorisation
  followed by the standard-normal CDF.
- `solve(dp, f, uncertainty=[...], n_samples=1000, rng_seed=None)`:
  unified entry point. Allowed summary labels: `"worst_case"`,
  `"mean"`, `"p95"`, `"cvar95"`, `"samples"`. Multiple summaries can be
  requested in a single call.
- `UncertaintyResult` dataclass with optional fields per requested
  summary, plus `feasibility_rate` and `n_samples_used`.
- A `Module` instance carries optional `uncertain_set` and
  `uncertain_dist` attributes; the uncertainty solver walks the
  `_codesign_modules` attribute attached to the built DP by
  `System.build`. Module parameters are saved before each sample and
  restored afterwards so nominal values are never clobbered.

#### Examples, notebooks, and tests
- `10_solver_trace.py` / notebook 10: every observability feature in
  turn, including a deliberately under-iterated solve to show the
  `"max_iter"` status and a deliberately infeasible solve to show the
  `"diverged"` status. The notebook plots the delta-vs-iteration curve.
- `11_uncertain_drone.py` / notebook 11: drone from example 7 with two
  uncertain internal parameters on the battery; worst case under a
  `Box` versus an `Ellipsoid`.
- `12_stochastic_drone.py` / notebook 12: same drone with stochastic
  uncertainty under a Gaussian copula; all summaries from a single
  solve call, plus a histogram of the MC distribution (matplotlib in
  the notebook, ASCII in the script).
- Three new smoke tests: `test_solver_trace_and_status`,
  `test_uncertainty_box`, `test_uncertainty_stochastic`.

#### System builder
- `System.build()` now attaches a `_codesign_modules` dict to the
  returned DP, exposing the module instances for the uncertainty
  solver and other inspection tools.

### Changed
- `SolveResult.trace` is now `None` when tracing is disabled (instead of
  an empty list), so missing traces are distinguishable from empty ones.
  The default behaviour is unchanged: with no flags, `trace` is `None`.
- `solve` and `kleene_loop` now use keyword-only arguments for the new
  observability and uncertainty options, to avoid call-site ambiguity.

## [0.1.0] - 2026-05-18

Initial release.

### Added
- Posets: `Reals`, `Naturals`, `Ports`, `Discrete`.
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
