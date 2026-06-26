# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Example 17: full-vehicle co-design across ICE, hybrid, and EV
  architectures** (`examples/17_car_codesign.py`,
  `notebooks/17_car_codesign.ipynb`,
  `docs/diagrams/car_ice.{svg,png}`,
  `docs/diagrams/car_hev.{svg,png}`,
  `docs/diagrams/car_ev.{svg,png}`). A 3300-line example that
  decomposes a passenger car into 18 to 24 MCDP modules per
  architecture and solves three powertrain variants in parallel:
  conventional ICE (engine block, forced induction, fuel injection,
  exhaust aftertreatment, cooling, lubrication, multi-speed
  transmission, mechanical differential, fuel tank, 12V electrical),
  parallel power-split hybrid (Atkinson engine, motor-generator,
  small HV battery, planetary power-split, power electronics, plus
  the ICE accessories at reduced sizing), and battery-electric
  (traction motor, large HV battery, power electronics, on-board
  charger, single-speed reducer, battery thermal management). All
  three share a common chassis core (body frame, front and rear
  suspension, front and rear brakes, steering, tires, wheels) and
  auxiliary core (HVAC, interior, safety, lighting and infotainment).

  Two coupled cycles close inside the constraint graph and are
  resolved by the Kleene iteration: the mass spiral, in which every
  load-bearing subsystem reads the design mass as an F input, and
  the energy-storage loop, in which fuel-tank or battery size
  depends on consumption × range while the storage's own mass
  contributes to curb weight. For EVs the battery cycle dominates
  the mass spiral (the pack is 20 to 35% of curb weight) so
  convergence takes 30 to 50 iterations vs 15 to 25 for ICE.

  The example provides four representative missions (Urban Compact,
  Family Daily, Suburban Utility, Performance) and sweeps every
  architecture across all four, emitting a Pareto-by-mission table
  plus a global 10-year TCO summary. Sample headline finding for
  Family Daily (5 passengers, 700 km range, 9 s 0-100): ICE cheapest
  at $35,523 but 196 g/km CO2; HEV best TCO at $56,427 with 91 g/km;
  EV lowest CO2 at 58 g/km but $64,666 upfront and 2434 kg with a
  100 kWh 800V pack. A 7-passenger 800 km EV remains infeasible
  against the modelled 2024 pack-level energy density, matching the
  real-world absence of such a product.

  Calibration values cite Genta (chassis), Bosch handbook (engines),
  Heywood and Pulkrabek (ICE thermodynamics), Hofmann (hybrid
  topologies), Naunheimer (transmissions), Larminie and Lowry
  (electric powertrain), IEA Global EV Outlook (battery pricing),
  EPA (fleet averages). New `EV_XL` tire entry (800 kg load index,
  240 km/h speed rating) extends the catalog for heavier EVs.
  Smoke test in `tests/test_smoke.py::test_car_codesign_smoke`
  builds and solves one car of each architecture and checks
  headline metrics fall in literature ranges.

- **Block-diagram rendering** (`codesign/diagram.py`,
  `System.draw_diagram()`). A new visualisation that turns any
  System into a Simulink-style block diagram via GraphViz: one box
  per subsystem with F ports listed on the left and R ports on the
  right, outer functionalities as teal ellipses on the diagram's
  left margin, outer resources as navy ellipses on the right.
  Constraint wiring is rendered port-to-port whenever the constraint
  was written in the operator-overloaded form
  (`m1.r_port >= m2.r_port * ...`). Lambda-based constraints get a
  dashed edge from a small "λ" marker so they remain visible.
  Strongly-connected components of size > 1 are detected via
  Tarjan's algorithm and their internal edges are coloured amber
  (`#B45309`), so the Kleene-iteration cycle is visible at a glance.
  Returns a `graphviz.Digraph` that is Jupyter-displayable inline or
  exportable via `.render(filename, format="svg")` to SVG, PDF, or
  PNG.

  Requires the optional `graphviz` Python package (added under a
  new `diagram` extras group in `pyproject.toml`) plus the `dot`
  binary on PATH. The rest of the package remains importable
  without these dependencies; the diagram module is loaded lazily.

  Sample output: example 7 (drone modular) shows the
  battery ↔ actuator feedback cycle in amber, example 9 (robotic
  arm) shows a five-module star with no cycle, example 15
  (bioprocess) shows three lambda-aggregated outer R nodes via the
  λ marker.

- **Tier 1 online-solver enhancements** (`codesign/online.py`,
  `tests/test_online_tier1.py`):
  - **Warm-start mechanism**: `solve_online` now accepts a
    `warm_start` argument that pre-populates the evaluator with seed
    observations before the picker takes over. Accepts either a list
    of candidate indices (manually picked corner runs) or an integer
    `n` (greedy farthest-point heuristic over the feature space).
    Particularly useful for the `MonotonicityEvaluator`, which is
    uninformative without observations at the low-feature corner; in
    example 16, four corner warm-start runs lift its Pareto recovery
    from 0 / 4 to at least 1 / 4.
  - **Pluggable picker strategies**: `solve_online` now accepts a
    `picker` argument. Built-in options are `"lcb"` (the default,
    minimises the sum of lower-bound components, exploitation-only),
    `"ucb"` (lower bound minus `kappa * (upper - lower)` exploration
    bonus, tunable via `picker=("ucb", {"kappa": 1.0})`), and
    `"random"` (uniform baseline for comparing the value of
    structural priors). Custom callables are also accepted.
  - **`GaussianProcessEvaluator`**: a new evaluator class with a
    zero-mean GP and RBF kernel, implemented in pure numpy (no
    scikit-learn dependency). Tunable `length_scale`, `sigma_f`,
    `noise`, `confidence`, and `min_obs`. For nearly-additive
    response surfaces such as the example 16 bioprocess effect
    model, `LinearParametricEvaluator` is empirically competitive;
    GP is more useful when the response has strong local nonlinearity
    that a global linear fit misses.

- **Example 16: monoclonal antibody fed-batch online DOE**
  (`examples/16_online_doe.py` and `notebooks/16_online_doe.ipynb`).
  Takes the example 15 model, fixes CHO-K1 and the 100 kg/yr mission,
  and sweeps a 5x5x5x3 = 375-point grid of operating conditions
  (temperature, pH, glucose target, feed start day). Compares
  factorial DOE (75 runs at the pH=7.1 slice), random sampling
  (40 runs), and the three online evaluators (Lipschitz,
  Monotonicity, LinearParametric) at a 40-run budget. Both
  LinearParametric and tuned Lipschitz recover 3 of 4 Pareto classes,
  matching the 75-run factorial DOE at 53% of the experimental cost.
  Monotonicity alone is uninformative without warm-start, which the
  example flags explicitly.

- **Example 15: monoclonal antibody fed-batch co-design** (`examples/15_bioprocess.py`
  and `notebooks/15_bioprocess.ipynb`). A worked biotech upstream
  application with realistic parameters from the 2024-2026
  bioprocessing literature. Four subsystems (CellLine, Media,
  Bioreactor, FeedStrategy) coupled cyclically through peak cell
  density; the Kleene iteration resolves the cycle automatically.
  Sources: Reinhart 2021 (CHO specific productivity), BioProcess
  International 2024 (kLa and OUR characterisation), Sustainability
  Atlas 2026 (capex), Khattak 2010 (metabolic constraints), CHO media
  market report 2025 (media pricing). The example produces a genuine
  2-point Pareto front per mission scale, showing the CHO-K1 (cheap
  COGS, larger footprint) vs CHO-MK (smaller footprint, higher
  licence fee) tradeoff.

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
