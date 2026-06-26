# codesign-mcdp

A Python library for **Monotone Co-Design Problems (MCDPs)**, following the
mathematical framework of Andrea Censi, *A Mathematical Theory of Co-Design*
([arXiv:1512.08055](https://arxiv.org/abs/1512.08055)). It is a from-scratch
alternative to [MCDPL](https://co-design.science/software/), built around
composable Python objects rather than a separate DSL.

A *design problem* is a relation between a **functionality** poset `F` and a
**resource** poset `R`. Given a target functionality `f`, the problem asks for
the minimal resources needed to deliver it (an antichain in `R`, the Pareto
front). Design problems compose under three operators (series, parallel,
feedback) and the resulting class is closed under composition; monotonicity is
preserved. Solutions are found by Kleene fixed-point iteration in the lattice
of antichains.

![Kleene iteration trace, c=4](docs/images/kleene_trace_c4.png)

*Kleene fixed-point iteration for `x + y >= ceil(sqrt(x)) + ceil(sqrt(y)) + 4`
over `N x N`. The antichain grows monotonically from the seed `S_0 = {(0,0)}`
and converges in six iterations to the five-point Pareto front
`{(0,7), (3,6), (4,4), (6,3), (7,0)}`. See `examples/05_visualize_kleene.py`.*

## Installation

The library has no required runtime dependencies beyond the Python standard
library; only some examples and the optional layers use third-party packages
(matplotlib for plots, graphviz for diagrams, numpy/scipy for the online and
stochastic layers).

Install the latest version directly from Git:

```bash
pip install git+https://github.com/cbriat/codesign-mcdp.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/cbriat/codesign-mcdp.git
cd codesign-mcdp
pip install -e ".[dev]"      # everything: tests, plots, diagrams, notebooks
```

The optional-dependency groups are `viz` (matplotlib), `diagram` (graphviz),
`online` (numpy + scipy), `nb` (notebook tooling), and `dev` (all of the
above plus pytest). A bare `pip install -e .` pulls nothing extra and still
runs the full solver. Python 3.9 or newer is required.

Run the test suite with:

```bash
pytest
```

## A 30-second example

A battery: given a required capacity, the minimal mass is capacity divided by
specific energy (1.8 MJ/kg for Li-ion).

```python
from codesign import Reals, Ports, AlgebraicDP, solve

F = Ports({"capacity": Reals(unit="J")})
R = Ports({"mass": Reals(unit="kg")})

battery = AlgebraicDP(
    F=F, R=R,
    equations={"mass": lambda f: f["capacity"] / 1.8e6},
)

result = solve(battery, {"capacity": 3.6e6})  # 1 kWh
print(result)
# SolveResult(iters=0, converged=True, feasible=True)
#   Antichain[(mass=2 kg)]
```

## Composition

`series`, `par` (parallel), and `loop` (feedback) close arbitrary co-design
problems over the primitive DPs. The three operators are direct
implementations of Defs. 14, 15, 16 in the paper.

```python
from codesign import series, par, loop

chained = series(battery, shipping)     # cost-of-shipping by mass
parallel = par(battery, actuator)       # independent resources combined
feedback = loop(drone_inner, axis="battery_mass")  # close a recursive constraint
```

`loop` triggers the Kleene fixed-point iteration; everything else evaluates in
closed form.

## Modular composition with `System`

`series`, `par`, and `loop` are low-level operators on whole DPs. For larger
designs it is more natural to define each subsystem (battery, actuator,
chassis, sensor, ...) independently with its own F and R, then wire them
together with named algebraic constraints. The `System` builder does exactly
that, and the operator-overloaded syntax lets you write the wiring as
inequalities that read like the textbook math:

```python
from codesign import Module, Reals, System, solve

# 1. Define subsystems as Module subclasses. F, R, and h are declared
#    inline; the constructor wires them into a DesignProblem.
class Battery(Module):
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}
    def h(self, f):
        return {"mass": f["capacity"] / 1.8e6}

class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}
    def h(self, f):
        return {"power": 10.0 * f["lift_force"] ** 2}

# 2. Assemble. Each provides/requires/add returns a port handle; arithmetic
#    operators on handles build expression trees, and >= registers a
#    constraint with the system.
sys = System("drone")
endurance     = sys.provides("endurance",     unit="s")
extra_payload = sys.provides("extra_payload", unit="kg")
extra_power   = sys.provides("extra_power",   unit="W")
total_mass    = sys.requires("total_mass",    unit="kg")

battery  = sys.add("battery",  Battery())
actuator = sys.add("actuator", Actuator())

# 3. Wire it: each line is a constraint that reads like a textbook inequality.
battery.capacity    >= (actuator.power + extra_power) * endurance
actuator.lift_force >= 9.81 * (battery.mass + extra_payload)
total_mass          >= battery.mass + extra_payload

drone = sys.build()
result = solve(drone, {"endurance": 300, "extra_payload": 0.5, "extra_power": 5.0})
```

Each `>=` constraint compiles to the same internal callable as the legacy
lambda form. Under the hood, `build()` produces a single `Loop` whose axis
bundles every subsystem's R; the Kleene iteration converges over all of them
simultaneously, and the resulting DP is no different from one written in
operator form. Systems can themselves be added as subsystems of other
Systems (recursive composition).

### Operator-overloaded constraint syntax: the rules

`sys.provides`, `sys.requires`, and `sys.add` each return a port handle:

- `provides(name)` returns an **outer F** port (right-hand side only).
- `requires(name)` returns an **outer R** port (can be a constraint target).
- `add(module_name, dp)` returns a `ModuleHandle`. Attribute access on the
  handle (`battery.capacity`) yields a port: F ports can be constraint
  targets, R ports can appear in expressions.

The DSL refuses category mistakes at the line where they happen: putting an
F port on the right of an expression, or constraining an R port externally,
raises immediately with an explanatory message.

For complex demands that don't fit algebraic expressions, the legacy lambda
form remains available and can be mixed freely with the operator form:

```python
# Equivalent to: battery.capacity >= (actuator.power + extra_power) * endurance
sys.constrain("battery.capacity",
              lambda x: (x["actuator.power"] + x["extra_power"]) * x["endurance"])
```

Both styles produce identical results.

### Multi-valued antichains

When a subsystem returns a multi-valued antichain, for example a `CatalogDP`
with Pareto-incomparable motors, the System takes the Cartesian product
across subsystems and lets the outer Min prune dominated combinations. The
result is a genuine system-level Pareto front:

```text
Medium load: payload=10 kg, mission_energy=1 MJ
   Pareto front (2 points):
      total_mass=22.49 kg,  total_cost=$475.00   (heavier, cheaper motor)
      total_mass=20.41 kg,  total_cost=$969.00   (lighter, more expensive motor)
```

See `examples/09_robotic_arm.py` for a five-module composition that uses
both the operator syntax and parameterised Module subclasses to express
mechanical and electrical couplings that don't fit a clean series/parallel
pattern.


## Primitive DP types

| Type | Use it for |
|---|---|
| `AlgebraicDP` | each resource is a closed-form monotone formula in `f` |
| `FunctionDP` | the user supplies `f -> Antichain` directly (multi-valued antichains, branchy logic) |
| `CatalogDP` | choose from a finite catalog of implementations (motors, batteries, sensors) |
| `ConstraintDP` | feasibility predicate plus a scalar cost, lifted to `Min` |
| `ODE_DP` | derive the relation from a differential equation (steady state or final value) |
| `UncertainDP` | wrap a lower and an upper bracket around an unknown `h` (Sec. VII) |

## The drone example (paper's Fig. 48)

The MCDPL example from Fig. 48 of the paper closes a feedback loop between a
battery and an actuator: the battery must store enough energy to carry the
payload, but it also adds to the payload itself.

```python
# see examples/01_drone.py

drone = make_drone()   # Loop DP wrapping battery + actuator
result = solve(drone, {
    "endurance": 300.0,        # seconds
    "extra_payload": 0.5,      # kg
    "extra_power": 5.0,        # W
})
# iters=22, feasible=True, Antichain[(report_mass=0.04921 kg)]
```

When the mission becomes infeasible (e.g. 30 minutes of endurance with a 1 kg
payload), the iteration drives the loop variable to ⊤ and the result reports
`feasible=False` rather than diverging.

## Worked examples in `examples/`

* `01_drone.py` – battery + actuator with payload feedback (Fig. 48), monolithic form.
* `02_integer_optimization.py` – `x + y >= ceil(sqrt(x)) + ceil(sqrt(y)) + c` over `N x N` (Sec. VI-D).
* `03_auv_seabed.py` – AUV seabed surveying, cyclic constraints on time, energy, cost (Sec. VIII).
* `04_uncertain_and_ode.py` – `UncertainDP` brackets and `ODE_DP` steady states.
* `05_visualize_kleene.py` – plots the Kleene ascent `S_0, S_1, ...` (reproduces the structure of Fig. 36).
* `06_drone_mcdpl_syntax.py` – drone rebuilt with the MCDPL-style declarative builder.
* `07_drone_modular.py` – the same drone with `System`, written with `Module` classes and the operator-overloaded `>=` constraint syntax.
* `08_vehicle_modular.py` – motor catalog + chassis + battery wired with the operator syntax, producing a multi-point Pareto front.
* `09_robotic_arm.py` – five subsystems (two joints, sensor, controller, battery) with non-trivial cyclic couplings, demonstrating where the operator syntax really pays off.
* `10_solver_trace.py` – the solver's observability features: `trace`, `verbose`, `on_iteration` callback, and the `status` field.
* `11_uncertain_drone.py` – set-based deterministic uncertainty on internal parameters: worst case under a `Box` and an `Ellipsoid`.
* `12_stochastic_drone.py` – Monte Carlo with a Gaussian copula, returning worst-case + mean + p95 + CVaR95 from a single solve call.
* `13_microgrid.py` – flagship case study: solar + battery + diesel + frame with cyclic coupling, warm-started parameter sweep, stochastic sun hours, and the full visualisation suite.
* `14_online_fleet.py` – online elimination-based co-design over a 200-candidate robot catalog using three flavours of optimistic evaluator.
* `15_bioprocess.py` – monoclonal antibody fed-batch co-design with realistic CHO cell-line, media, bioreactor, and feed-strategy catalogues sourced from the 2024-2026 bioprocessing literature. Produces a 3-way (COGS, footprint, CO2 per gram) Pareto front across three mission scales (clinical 10 kg/yr, commercial 100 kg/yr, large 500 kg/yr).
* `16_online_doe.py` – online Design of Experiments for the example 15 mAb process. Fixes CHO-K1 and the 100 kg/yr mission, then sweeps a 5x5x5x3 = 375-point grid of operating conditions (temperature, pH, glucose target, feed start day). Compares factorial DOE, random sampling, and the three online evaluators at a 40-run budget. LinearParametric and Lipschitz match a 75-run factorial DOE quality at 53% of the experimental cost.

Run any of them with `python -m examples.NN_name`. The visualization example
also needs matplotlib (`pip install matplotlib`).

## Documentation

A full reference manual is provided as both LaTeX source and a pre-built PDF
under [`docs/manual/`](docs/manual/). It covers the mathematical background
from Censi (2015), every data type and primitive, both builders, the solver,
worked examples, and modelling guidelines. Rebuild from source with
`make` in that directory if you have LaTeX installed.

The notebook companion to each example is under
[`notebooks/`](notebooks/README.md), with outputs and figures pre-rendered so
they read directly on GitHub.

## Notebooks

Each example also has a Jupyter notebook companion under `notebooks/`, with
extra prose explaining the model and the results. The committed `.ipynb`
files include all outputs (including embedded figures from notebook 05), so
they render on GitHub without running anything. See
[`notebooks/README.md`](notebooks/README.md) for the index.

To run them locally:

```bash
pip install -e ".[viz]"
pip install jupyter
jupyter lab notebooks/
```

To regenerate them after a code change:

```bash
pip install nbformat nbconvert ipykernel matplotlib
python build_notebooks.py
```

## Solving and ranking

`solve(dp, functionality)` returns a `SolveResult`:

```python
result = solve(dp, {"capacity": 3.6e6})
result.antichain    # the Pareto front (an Antichain[R])
result.iterations   # number of Kleene steps (0 if no loop)
result.status       # "converged" | "max_iter" | "diverged"
result.feasible     # True iff at least one finite minimal resource bundle exists
result.trace        # list of TraceEntry when trace=True, else None
result.converged    # backward-compat alias for status == "converged"
```

`status` and `feasible` are orthogonal. `status="converged"` with `feasible=False` is a clean infeasibility (the antichain settled at ⊤). `status="max_iter"` with `feasible=True` means the iteration cap was reached while the run still looked feasible; usually a sign to increase `max_iter`. `status="diverged"` means a numeric value crossed the divergence cap before the iteration could settle.

### Watching the solver work

```python
# Live printing
result = solve(dp, f, verbose=1)   # one summary line at the end
result = solve(dp, f, verbose=2)   # per-iteration progress feed

# Structured trace
result = solve(dp, f, trace=True)
for entry in result.trace:
    print(entry.iteration, entry.n_points, entry.delta, entry.elapsed_ms)

# Callback
def my_logger(entry):
    if entry.iteration % 10 == 0:
        print(f"iter {entry.iteration}: delta={entry.delta}")
result = solve(dp, f, on_iteration=my_logger)
```

`trace=False` by default, so the existing call sites pay nothing.

### Cost minimisation

When the antichain has multiple incomparable points (genuine tradeoffs),
`minimize_cost` collapses it to one design under a scalar objective:

```python
from codesign import minimize_cost

best = minimize_cost(result, cost_fn=lambda r: r["weight"] + 0.1 * r["cost"])
```

## Uncertainty

Modules can carry deterministic, set-based uncertainty on their internal parameters (`uncertain_set`), stochastic uncertainty (`uncertain_dist`), or both. A single `solve(..., uncertainty=[...])` call returns the worst-case answer alongside statistical summaries:

```python
from scipy import stats
from codesign import Module, Reals, Box, Stochastic, GaussianCopula, System, solve

class Battery(Module):
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}
    def __init__(self, specific_energy=1.8e6, efficiency=0.85):
        self.specific_energy = specific_energy
        self.efficiency = efficiency
        super().__init__()
    def h(self, f):
        return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}

b = Battery()
# Deterministic set: worst case is the corner where both params are at their
# lowest declared values.
b.uncertain_set = Box(
    specific_energy=(1.6e6, 2.0e6, "more_is_better"),
    efficiency=(0.80, 0.90, "more_is_better"),
)
# Stochastic with correlation: two uniform marginals tied by a Gaussian copula.
b.uncertain_dist = Stochastic(
    marginals={
        "specific_energy": stats.uniform(loc=1.6e6, scale=0.4e6),
        "efficiency":      stats.uniform(loc=0.80, scale=0.10),
    },
    copula=GaussianCopula(correlation=[[1.0, 0.4], [0.4, 1.0]]),
)

# ... wire b into a System ...

result = solve(drone, f,
               uncertainty=["worst_case", "mean", "p95", "cvar95", "samples"],
               n_samples=1000, rng_seed=42)

result.worst_case        # SolveResult-equivalent at the worst point of the set
result.mean              # dict[r_port -> mean across MC samples]
result.p95               # dict[r_port -> 95th percentile]
result.cvar95            # dict[r_port -> CVaR at the 95% level]
result.samples           # list of antichains, one per MC sample
result.feasibility_rate  # fraction of MC samples that came back feasible
```

The uncertainty sets supported in v1 are `Box` (n-D, axis-aligned), `Ellipsoid` (n-D, possibly tilted), plus the 2D conveniences `Disk` and `Circle`. Stochastic dependence is described by a `Copula` (`Independence` by default, `GaussianCopula(correlation=...)` for correlated marginals). Each `Box`/`Ellipsoid` parameter can be declared with a "direction of badness" (`"more_is_better"`, `"more_is_worse"`, etc.); declared directions enable an analytic worst-case computation, undeclared directions trigger a boundary search.

## Online learning

When a co-design problem has many discrete candidates (catalog entries, robot types, component families) and each candidate's inner solve is non-trivial, evaluating every one is wasteful. The `codesign.online` module implements the elimination-based solver from Alharbi, Dahleh & Zardini (arXiv:2604.22624): maintain *optimistic bounds* on each candidate's inner-solve output, evaluate the most promising one, then prune any candidate whose lower bound is already dominated by the incumbent.

```python
from codesign import (
    solve_online, LipschitzEvaluator,
    MonotonicityEvaluator, LinearParametricEvaluator,
)

# `candidate_fn(robot) -> DP` builds the inner DP for one robot type;
# `candidates` is a list of feature dicts (one per robot type).
ev = LipschitzEvaluator(
    features=["speed", "payload", "unit_cost"],
    r_components=["total_cost"],
    L={"total_cost": 300.0},   # or a scalar L
)
result = solve_online(
    candidate_fn, mission,
    candidates=candidates,
    evaluator=ev,
    budget=50,                  # max inner solves; None = unbounded
)

result.antichain         # Min over the evaluated, surviving candidates
result.n_evaluated       # actual inner solves performed
result.n_eliminated      # candidates pruned without evaluation
result.evaluated_ids     # indices into `candidates` that were evaluated
result.eliminated_ids    # indices pruned by the bound
result.incumbent_ids     # indices whose evaluation contributed to the final antichain
result.history           # per-iteration log: pick, antichain, remaining, evaluated, eliminated
```

Three evaluators are provided out of the box:

* `LipschitzEvaluator(features, r_components, L)` — bounds tighten by `L * ||features|` around each observation. Safe default: with a sensible `L` it never prunes a Pareto-optimal candidate.
* `MonotonicityEvaluator(features, r_components)` — assumes the output is component-wise monotone in the features. Aggressive when applicable (often dozens of evaluations instead of thousands), but only correct if monotonicity genuinely holds.
* `LinearParametricEvaluator(features, r_components, confidence, min_obs)` — fits a running OLS model with a confidence band. Fastest in practice but can wrongly prune when the linear assumption breaks.

Subclass `OptimisticEvaluator` for custom assumptions; the only method to override is `bound(candidate) -> (lower, upper)` mapping each R component to its current lower and upper bound.

## Visualisation

The `codesign.viz` module provides four matplotlib- and GraphViz-based helpers, all importable from the top-level `codesign` namespace:

```python
from codesign import viz

ax = viz.plot_antichain(result, axes=["mass", "cost"])    # 2D or 3D Pareto scatter
ax = viz.plot_convergence(result)                          # delta-vs-iteration on log axis
ax = viz.plot_uncertainty(unc_result, port="total_mass",   # histogram with summaries
                          nominal=nominal_mass)
dot = viz.to_dot(dp, name="my_dp")                         # System structure as GraphViz dot
```

Each helper accepts an existing matplotlib axes (`ax=...`) for composition into larger figures. `to_dot` returns a string suitable for piping into `dot -Tpng` or pasting into [graphviz online](https://dreampuf.github.io/GraphvizOnline/).

### Block diagrams of Systems

For richer Simulink-style block diagrams of `System`-built designs, `codesign.diagram` produces port-level wiring with cycle detection:

```python
from codesign import draw_system          # also: system.draw_diagram()

dot = system.draw_diagram(rankdir="LR")    # returns a graphviz.Digraph
dot.render("bioprocess", format="svg", cleanup=True)
```

Each subsystem becomes a box with its F ports on the left and R ports on the right; outer functionalities and outer resources appear as ellipses on the diagram's margins; constraint wiring resolves to specific ports rather than to whole modules. Strongly-connected components are detected automatically and their internal edges are coloured amber, so the Kleene-iteration cycle (where one exists) is visible at a glance. Lambda-based constraints get a dashed edge from a small `λ` marker.

Optional dependency: `pip install codesign-mcdp[diagram]` plus the `dot` binary on PATH (`apt-get install graphviz` or `brew install graphviz`).

## How the solver works

`solve` dispatches on the top-level operator. For non-loop DPs the answer is
`dp.h(f)` in closed form. For a `Loop`, it runs the Kleene ascent of Prop. 4:

1. Seed `A_0 = {⊥_R}`.
2. For each point `r ∈ A_k`, evaluate the inner DP at `f ⊕ {axis: r[axis]}`,
   intersect with `↑ r`, take the union over `r`, and apply `Min`. That is
   `A_{k+1} = Φ(A_k)`.
3. Stop when `A_{k+1} = A_k` (fixed point), when `A_{k+1}` is empty
   (no feasible extension), or when every point's loop axis reaches `⊤`
   (provably infeasible).
4. Project out the loop axis to land in the outer resource poset.

The implementation includes a divergence cap to convert numerical blow-up into
infeasibility, which is essential for floating-point loops where a few
iterations of unbounded growth would otherwise overflow before the algorithm
recognises divergence.

## Modeling guidelines

A few patterns that come up repeatedly:

* **Expose loop variables you care about.** The `Loop` operator projects its
  axis out of the outer `R`. To inspect the converged loop value, include it
  in the inner `R` *under a different name* (e.g. both `battery_mass` for the
  loop axis and `report_mass` mirrored for the outer R). The `System` builder
  handles this automatically; you only need it in the operator-level API.
* **Cap physical maxima.** When a design variable has a physical ceiling
  (`v_max`, `r_max`), make `h` return a `⊤`-valued antichain once the
  iteration's loop input exceeds it. The Kleene ascent will then converge to
  infeasible rather than oscillating.
* **Generate antichain breadth from `FunctionDP` or `CatalogDP`.**
  `AlgebraicDP` always returns a single point. When you want a true Pareto
  front, use `FunctionDP` and enumerate the tradeoffs explicitly, or use a
  `CatalogDP` with several incomparable entries.
* **Scalar-objective optimization is `minimize_cost` over the antichain.**
  The MCDP solver returns the Pareto front; the engineer's choice of which
  point to ship is a downstream scalarization.

## Architecture

```
codesign/
  posets.py        Reals, Naturals, Ports, Discrete
  antichains.py    Antichain: normalised, Min-closed, with union_min and filter_above
  dp.py            DesignProblem, AlgebraicDP, FunctionDP, CatalogDP, ConstraintDP, ODE_DP, UncertainDP
  composition.py   Series, Parallel, Loop  (and series, par, loop aliases)
  primitives.py    adder, multiplier, scale, constant, identity
  solver.py        kleene_loop, solve, minimize_cost, SolveResult
  mcdpl.py         MCDP builder, MCDPL-style provides/requires/constraint
  system.py        System builder, modular composition of named subsystems
```

The dependency graph is acyclic: `posets <- antichains <- dp <- composition`,
with `solver` reading all four. `mcdpl` and `system` are thin builders on top.

## Running the tests

```bash
python -m tests.test_smoke
```

A CI workflow at `.github/workflows/test.yml` runs the same smoke test on
every push.

## What this is and isn't

This is a from-scratch implementation of the *algorithmic* core of the paper:
the antichain calculus, the three composition operators, the Kleene
fixed-point iteration that closes loops, and a modular builder for assembling
multi-subsystem designs. It does **not** ship:

* the original MCDPL parser and its concrete `mcdp { ... }` text syntax (the
  `MCDP` builder in `mcdpl.py` provides the same shape in Python),
* approximation strategies for non-finitely-representable antichains beyond
  the bracket pattern of `UncertainDP`,
* visualization of the design graph itself.

These are tractable extensions on top of the current core.

## License

MIT. See [LICENSE](LICENSE).

## References

* Andrea Censi, *A Mathematical Theory of Co-Design*, arXiv:1512.08055 (2015).
* Davey and Priestley, *Introduction to Lattices and Order*, CUP (2002).
