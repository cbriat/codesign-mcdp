"""Generate Jupyter notebooks for each example, then execute them so the
outputs (text, tables, embedded matplotlib figures) are captured in the
.ipynb files committed to the repository.

Run from the project root:

    python build_notebooks.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def make_notebook(cells):
    """Build a notebook from a sequence of (kind, source) tuples."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    for kind, source in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        elif kind == "code":
            nb.cells.append(nbf.v4.new_code_cell(source))
        else:
            raise ValueError(f"unknown cell kind {kind!r}")
    return nb


def execute_notebook(nb, cwd: Path) -> None:
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(cwd)}})


def write(name: str, cells) -> Path:
    nb = make_notebook(cells)
    path = NOTEBOOKS_DIR / name
    print(f"[building] {path.name}")
    execute_notebook(nb, NOTEBOOKS_DIR.parent)
    nbf.write(nb, path)
    print(f"[wrote   ] {path}")
    return path


# ---------------------------------------------------------------------------
# 01 Drone (monolithic)
# ---------------------------------------------------------------------------

NB_01 = [
    ("md", """# 01. The drone example (Fig. 48)

This is the canonical MCDP from Censi (2015), Fig. 48: a battery powers an actuator that has to lift the battery plus an extra payload. Because the battery's own mass appears in the lift it has to produce, the design problem is intrinsically recursive and only closes once the **Kleene fixed-point iteration** converges.

The model is built here as a single `FunctionDP` wrapped in a `Loop` on `battery_mass`. Compare with notebook **07** for the same model written modularly with the `System` builder.
"""),
    ("md", "## Imports"),
    ("code", """import math
from codesign import (
    Antichain, FunctionDP, Loop, Ports, Reals, solve,
)"""),
    ("md", """## Physical constants

These follow Fig. 48 of the paper: a Li-ion battery (1.8 MJ/kg), gravity, and a quadratic drag coefficient for the actuator (10 W per N² of lift)."""),
    ("code", """ALPHA = 1.8e6      # Li-ion specific energy, J/kg
G = 9.81           # gravity, m/s^2
C_LIFT = 10.0      # actuator coefficient, W per N^2 of lift"""),
    ("md", """## The inner design problem

The functionality is `(endurance, extra_payload, extra_power, battery_mass)` and the resource is `(battery_mass, report_mass)`. Note that `battery_mass` appears on *both* sides: the inner DP receives the current iterate as a functionality input and emits a tightened estimate as a resource output. The `report_mass` is a mirrored copy of the same value so the outer R retains visibility of it (the `Loop` operator projects out the loop axis).
"""),
    ("code", """# Outer F has the mission spec plus the loop axis as an input.
F = Ports({
    "endurance":     Reals(unit="s"),
    "extra_payload": Reals(unit="kg"),
    "extra_power":   Reals(unit="W"),
    "battery_mass":  Reals(unit="kg"),    # current iterate, fed back in
})
# Inner R emits the tightened battery_mass plus a "report" mirror.
R = Ports({
    "battery_mass": Reals(unit="kg"),     # what Loop closes on
    "report_mass":  Reals(unit="kg"),     # what the outer world sees
})

def h(f):
    # Short-circuit: if any input has already diverged to +inf, propagate it.
    # This keeps the iteration well-defined when an intermediate iterate
    # exceeds the divergence cap and starts producing infinities.
    if (f["battery_mass"] == math.inf or
        f["endurance"] == math.inf or
        f["extra_payload"] == math.inf or
        f["extra_power"] == math.inf):
        return Antichain.singleton(R, {
            "battery_mass": math.inf, "report_mass": math.inf,
        })
    # Physics:
    #   lift force      = (battery + payload) * g
    #   actuator power  = C_LIFT * lift^2
    #   total power     = actuator + avionics extra
    #   energy required = total_power * endurance
    #   battery mass    = energy / specific_energy
    lift = (f["battery_mass"] + f["extra_payload"]) * G
    actuator_power = C_LIFT * lift * lift
    total_power = actuator_power + f["extra_power"]
    energy = total_power * f["endurance"]
    mass = energy / ALPHA
    # Both R components carry the same number; report_mass is the outer view.
    return Antichain.singleton(R, {
        "battery_mass": mass, "report_mass": mass,
    })

inner = FunctionDP(F=F, R=R, h_fn=h, name="drone")
# Close the loop on battery_mass: solve() will run the Kleene iteration
# until the iterate settles (or diverges to +inf).
drone = Loop(inner, axis="battery_mass")
drone"""),
    ("md", """## Solving for several mission profiles

We sweep over short, medium, longer, marginal, and clearly-infeasible missions. The marginal and infeasible cases are correctly flagged: the loop axis is driven to `⊤` (infinity) when the recursion does not close on a finite battery mass.
"""),
    ("code", """# Five mission profiles, ordered roughly by difficulty.
cases = [
    ("Short, light",   dict(endurance=60.0,   extra_payload=0.10, extra_power=1.0)),
    ("Medium, modest", dict(endurance=300.0,  extra_payload=0.50, extra_power=5.0)),
    ("Longer mission", dict(endurance=600.0,  extra_payload=0.50, extra_power=5.0)),
    ("Marginal",       dict(endurance=600.0,  extra_payload=1.00, extra_power=10.0)),
    ("Infeasible",     dict(endurance=1800.0, extra_payload=1.00, extra_power=10.0)),
]
for label, f in cases:
    # Each solve runs an independent Kleene iteration. max_iter caps the
    # number of fixed-point steps; the feasible cases converge in ~10-25.
    result = solve(drone, f, max_iter=200)
    print(f"{label:<16} iters={result.iterations:>3}  "
          f"feasible={result.feasible}  {result.antichain}")"""),
    ("md", """## Reading the result

For the feasible cases, the antichain converges to a single battery mass that grows roughly with mission energy. The marginal and infeasible cases hit the divergence cap: the actuator can't lift a battery large enough to satisfy its own energy demand, so the Kleene ascent walks the loop axis to `⊤` and the solver reports `feasible=False`.

In notebook **07** we'll rebuild the same problem with the `System` builder, where battery and actuator are *independent* subsystems wired together with constraint equations.
"""),
]


# ---------------------------------------------------------------------------
# 02 Integer optimization
# ---------------------------------------------------------------------------

NB_02 = [
    ("md", """# 02. Integer optimization (Sec. VI-D)

The Sec. VI-D example of Censi (2015):

$$
x + y \\geq \\lceil\\sqrt{x}\\rceil + \\lceil\\sqrt{y}\\rceil + c
$$

over $\\mathbb{N} \\times \\mathbb{N}$. For each $c$ we want the Pareto-minimal set of $(x, y)$ pairs satisfying the inequality. The Kleene ascent starts from the seed $\\{(0,0)\\}$ and converges in a handful of steps. This is the example whose trace appears in Fig. 36 of the paper.
"""),
    ("md", "## Imports"),
    ("code", """import math
from codesign import (
    Antichain, FunctionDP, Loop, Ports, Naturals, solve,
)"""),
    ("md", """## The model

The inner DP enumerates every splitting $(c_1, c_2)$ of the deficit into the two coordinates, giving the antichain of points $(x, y)$ with $x + y$ exactly meeting the constraint. The `Loop` on the axis `xy` closes $x_{out} \\geq x_{in}$ and $y_{out} \\geq y_{in}$ simultaneously.
"""),
    ("code", """def make_looped(c_value: int):
    # Inner posets: x and y are naturals (with +inf as top); xy bundles them.
    N = Naturals()
    XY = Ports({"x": N, "y": N})
    # F has the parameter c plus the current iterate xy fed back from the loop.
    F = Ports({"c": N, "xy": XY})
    # R emits the next iterate xy plus a "report" copy the outer world sees.
    R = Ports({"xy": XY, "xy_report": XY})

    def h(f):
        c = int(f["c"])
        x_in, y_in = f["xy"]["x"], f["xy"]["y"]
        # Propagate top through: if either coordinate is +inf, give up.
        if x_in == math.inf or y_in == math.inf:
            top = {"x": math.inf, "y": math.inf}
            return Antichain.singleton(R, {"xy": top, "xy_report": top})

        # ceil(sqrt(x_in)) without floating point: integer-square-root.
        # If isqrt(n)**2 < n then n is not a perfect square; round up.
        sx = math.isqrt(int(x_in)) + (1 if math.isqrt(int(x_in)) ** 2 < int(x_in) else 0)
        sy = math.isqrt(int(y_in)) + (1 if math.isqrt(int(y_in)) ** 2 < int(y_in) else 0)
        # The constraint x + y >= ceil(sqrt(x)) + ceil(sqrt(y)) + c, with the
        # incoming x_in, y_in bounding the ceiling terms.
        target = sx + sy + c

        # Enumerate every (x_out, y_out) with x_out + y_out == target and
        # both coordinates >= the ceiling lower bound. This gives the full
        # antichain of splits, which Min will prune as the iteration grows.
        pts = []
        for x_out in range(sx, target - sy + 1):
            y_out = target - x_out
            if y_out < sy:
                break
            pts.append({
                "xy": {"x": x_out, "y": y_out},
                "xy_report": {"x": x_out, "y": y_out},
            })
        if not pts:
            return Antichain.empty(R)
        return Antichain.from_set(R, pts)

    inner = FunctionDP(F=F, R=R, h_fn=h, name=f"sqrt_sum(c={c_value})")
    # Close on xy so the iterate feeds itself back through the inequality.
    return Loop(inner, axis="xy")"""),
    ("md", "## Run with trace"),
    ("code", """def pretty(p):
    return f"({p['xy_report']['x']}, {p['xy_report']['y']})"

def run(c_value, show_trace=True):
    looped = make_looped(c_value)
    # record_trace=True keeps the full sequence A_0, A_1, ... on result.trace.
    result = solve(looped, {"c": c_value}, max_iter=50, record_trace=show_trace)
    print(f"c = {c_value}: iters = {result.iterations}, feasible = {result.feasible}")
    if show_trace and result.trace:
        # Print every iterate to show how the antichain evolves.
        for k, entry in enumerate(result.trace):
            A = entry.antichain
            pts = ", ".join(f"({p['xy']['x']}, {p['xy']['y']})" for p in A.points)
            print(f"   S_{k}: {{ {pts} }}")
    pts = ", ".join(pretty(p) for p in result.antichain.points)
    print(f"   M(c={c_value}) = {{ {pts} }}\\n")
    return result

# Walk through the values from the paper; c=20 is silent to keep output short.
_ = run(0)
_ = run(1)
_ = run(4)
_ = run(20, show_trace=False)"""),
    ("md", """## Comments

The `c = 0` case is trivial: the empty mission has the solution $(0, 0)$.

For `c = 4` we get a five-point Pareto front $\\{(0,7), (3,6), (4,4), (6,3), (7,0)\\}$ in six iterations. Notice that the antichain is not necessarily monotone in cardinality: it grows, then shrinks, as some interior points get dominated by newly-discovered ones.

Note: the paper claims $M(1) = \\{(1,0), (0,1)\\}$, but those points fail the constraint ($1 \\geq 1 + 0 + 1 = 2$ is false). The correct answer is $\\{(0,3), (3,0)\\}$, which is what our solver finds. The paper has a typo there.

The next notebook (**05**) renders these traces graphically.
"""),
]


# ---------------------------------------------------------------------------
# 03 AUV seabed surveying
# ---------------------------------------------------------------------------

NB_03 = [
    ("md", """# 03. AUV seabed surveying (Sec. VIII)

An autonomous underwater vehicle must sweep an area $A$ at fixed depth, moving at velocity $v$ with sensor field of view $r$. The constraints couple time, energy, and cost cyclically:

- Coverage: $v \\cdot T \\cdot r \\geq k \\cdot A$
- Actuation: $P_{act} \\geq \\psi(v)$ (drag scales as $v^3$)
- Sensing: $P_{sens} \\geq \\chi(r)$ (sensor power scales with $r$)
- Energy: $E \\geq (P_{act} + P_{sens}) \\cdot T$

Larger $v$ shortens $T$ but costs more drag; wider $r$ shortens $T$ but costs more sensor power. The MCDP solver returns the (time, energy, cost) Pareto front for each area to cover.
"""),
    ("md", "## Imports"),
    ("code", """import math
from codesign import (
    Antichain, FunctionDP, Loop, Ports, Reals,
    solve, minimize_cost,
)"""),
    ("md", "## Build the AUV model"),
    ("code", """def make_auv():
    # Physical constants for this toy model.
    K_GEOM = 1.0        # geometric coverage constant (mission area / sweep)
    V_MAX = 3.0         # vehicle speed cap, m/s
    R_MAX = 5.0         # sensor footprint cap, m
    PSI_A = 30.0        # drag power coefficient (P_drag = PSI * v^3)
    CHI_A = 50.0        # sensor power coefficient (P_sens = CHI * r)
    SENSOR_COST_A = 200.0  # capex per metre of sensor footprint

    # The design is parameterised by speed v and footprint r; the loop
    # closes on (v, r) so the iteration converges to a self-consistent set.
    Design = Ports({"v": Reals(unit="m/s"), "r": Reals(unit="m")})
    F = Ports({"A": Reals(unit="m^2"), "design": Design})
    R = Ports({
        "design": Design,
        "T": Reals(unit="s"),
        "E": Reals(unit="J"),
        "cost": Reals(unit="$"),
    })

    def h(f):
        A = f["A"]
        v_in, r_in = f["design"]["v"], f["design"]["r"]
        # Propagate top through if a previous iterate already diverged.
        if v_in == math.inf or r_in == math.inf:
            return Antichain.singleton(R, {
                "design": {"v": math.inf, "r": math.inf},
                "T": math.inf, "E": math.inf, "cost": math.inf,
            })
        # Floor the inputs so we never compute 1/0; cap them at the physical
        # limits so the Kleene iteration converges to infeasibility cleanly
        # if the demand can't be met within V_MAX, R_MAX.
        v = max(float(v_in), 0.1)
        r = max(float(r_in), 0.5)
        if v > V_MAX or r > R_MAX:
            return Antichain.singleton(R, {
                "design": {"v": math.inf, "r": math.inf},
                "T": math.inf, "E": math.inf, "cost": math.inf,
            })

        # Enumerate a small grid of candidate (v_try, r_try) values.
        # The factors 1.0, 1.3, 1.7 produce three speeds and three
        # footprints, so up to nine combinations per iteration. Each gives
        # a (T, E, cost) triple, and Min over the antichain prunes the
        # dominated ones automatically.
        pts = []
        for v_try in (v, min(v*1.3, V_MAX), min(v*1.7, V_MAX)):
            for r_try in (r, min(r*1.3, R_MAX), min(r*1.7, R_MAX)):
                # Only consider candidates that satisfy the monotonicity
                # contract on the loop axis (output >= input).
                if v_try < v_in or r_try < r_in:
                    continue
                T_try = K_GEOM * A / (v_try * r_try)          # time to cover A
                E_try = (PSI_A * v_try**3 + CHI_A * r_try) * T_try   # energy
                cost_try = SENSOR_COST_A * r_try               # sensor capex
                pts.append({
                    "design": {"v": v_try, "r": r_try},
                    "T": T_try, "E": E_try, "cost": cost_try,
                })
        return Antichain.from_set(R, pts)

    inner = FunctionDP(F=F, R=R, h_fn=h, name="auv_inner")
    return Loop(inner, axis="design")

auv = make_auv()
auv"""),
    ("md", """## Run for three mission scales

Each area gives a small (T, E, cost) Pareto front. We then collapse it to a single design with `minimize_cost` using a composite scalar objective.
"""),
    ("code", """def show(result, label):
    print(label)
    print(f"   iters={result.iterations}, feasible={result.feasible}")
    if not result.feasible:
        return
    # Walk the antichain and print every Pareto point.
    for p in result.antichain.points:
        print(f"   T={p['T']:.0f}s, E={p['E']/1000:.1f}kJ, $={p['cost']:.0f}")
    # Scalarise the Pareto front: weighted sum of time, energy, cost.
    # The weights here are illustrative; in practice the engineer picks them.
    best = minimize_cost(
        result,
        cost_fn=lambda r: r["T"] + 0.05 * (r["E"] / 1000.0) + r["cost"],
    )
    if best is not None:
        print(f"   best composite: T={best['T']:.0f}s, "
              f"E={best['E']/1000:.1f}kJ, $={best['cost']:.0f}")
    print()

# Three mission scales: 100 m^2, 1000 m^2, 10000 m^2.
for A in (100.0, 1000.0, 10_000.0):
    result = solve(auv, {"A": A}, max_iter=50)
    show(result, f"Area = {A:g} m^2")"""),
    ("md", """## Interpretation

Each scenario produces three incomparable points along the (cost, speed) tradeoff: slow-and-cheap or fast-and-expensive sensors. As $A$ grows by a factor of 10, both $T$ and $E$ scale up by the same factor (linear in area), while cost stays the same (it's a one-off sensor purchase, not a per-mission cost).
"""),
]


# ---------------------------------------------------------------------------
# 04 UncertainDP and ODE_DP
# ---------------------------------------------------------------------------

NB_04 = [
    ("md", """# 04. `UncertainDP` and `ODE_DP`

Two of the more specialised primitives.

**`UncertainDP`** wraps a pair of design problems $(h^L, h^U)$ that bracket a true $h$ which is unknown or non-finitely-representable (Sec. VII). Solving with $h^L$ gives an optimistic Pareto front; solving with $h^U$ gives a pessimistic one. The true minimal resources sit between.

**`ODE_DP`** derives a monotone resource relation from a differential equation: integrate to a final value, or solve for the steady state. Useful when the resource depends on the trajectory of an underlying physical system rather than a closed-form expression.
"""),
    ("md", "## Imports"),
    ("code", """from codesign import (
    AlgebraicDP, Ports, ODE_DP, Reals, UncertainDP, solve,
)"""),
    ("md", """## UncertainDP demo: battery with uncertain specific energy

Old Li-ion cells average 1.6 MJ/kg; newer ones 2.0 MJ/kg. We bracket the unknown true value with the two limits and solve in both modes.
"""),
    ("code", """F = Ports({"capacity": Reals(unit="J")})
R = Ports({"mass": Reals(unit="kg")})

# Two algebraic brackets around the true h. Each captures one end of the
# specific-energy range; their solutions bracket the true required mass.
pessimistic = AlgebraicDP(
    F=F, R=R,
    equations={"mass": lambda f: f["capacity"] / 1.6e6},   # heavy battery
    name="battery_pessimistic",
)
optimistic = AlgebraicDP(
    F=F, R=R,
    equations={"mass": lambda f: f["capacity"] / 2.0e6},   # light battery
    name="battery_optimistic",
)
# UncertainDP composes the two brackets into a single DP that can be solved
# in either mode. mode="upper" means default to the pessimistic bracket.
uncertain = UncertainDP(F=F, R=R, lower=optimistic, upper=pessimistic, mode="upper")

print("Battery sizing under specific-energy uncertainty (1 kWh capacity):")
# Solve in both modes to see the interval of feasible designs.
for mode in ("lower", "upper"):
    result = solve(uncertain.with_mode(mode), {"capacity": 3.6e6})
    mass = list(result.antichain.points)[0]["mass"]
    label = "optimistic" if mode == "lower" else "pessimistic"
    print(f"   {label:<12} ({mode}): mass = {mass:.3f} kg")
print("\\nDesigns that survive the pessimistic case are robust to the uncertainty.")"""),
    ("md", """## ODE_DP demo: steady-state heater

A heated payload loses heat to the environment proportional to its temperature rise (Newton's cooling). At steady state the input power equals the heat-loss coefficient times the temperature delta. The ODE solver finds the steady state by Newton iteration on $\\dot x = 0$.
"""),
    ("code", """H_LOSS = 0.8  # W/K, heat-loss coefficient

# rhs: dP/dt = H_LOSS * delta_T - P (so steady state is P = H_LOSS * delta_T).
# steady_state mode runs Newton iteration on rhs = 0; extract pulls the
# scalar power out of the converged state.
heater = ODE_DP(
    F=Ports({"delta_T": Reals(unit="K")}),
    R=Ports({"power": Reals(unit="W")}),
    rhs=lambda x, t, f: H_LOSS * f["delta_T"] - x,
    extract=lambda x: {"power": float(x)},
    mode="steady_state",
    x0_fn=lambda f: 0.0,    # seed Newton at P = 0
    name="heater_ode",
)

print("Power required to hold a steady temperature rise (h_loss = 0.8 W/K):")
# Sweep over a few temperature deltas; each should produce P = H_LOSS * delta_T.
for dT in (5.0, 20.0, 50.0):
    result = solve(heater, {"delta_T": dT})
    p = list(result.antichain.points)[0]["power"]
    print(f"   delta_T = {dT:>4.0f} K  ->  P_in = {p:>5.1f} W  "
          f"(=  h_loss * delta_T = {H_LOSS * dT:.1f} W)")"""),
    ("md", """The ODE solver finds the steady state exactly because $\\dot x = 0$ has a closed form here. For nonlinear right-hand sides (radiative loss $\\propto T^4$, for example) it would still converge as long as the equation is monotone in the resource.
"""),
]


# ---------------------------------------------------------------------------
# 05 Visualizing the Kleene trace
# ---------------------------------------------------------------------------

NB_05 = [
    ("md", """# 05. Visualizing the Kleene ascent

This notebook reproduces the structure of Fig. 36 from Censi (2015): plotting each antichain $S_k$ of the Kleene iteration as it converges, for the integer-optimization problem from notebook **02**.

We render the trace for three values of $c$ ($c = 1, 4, 8$). The seed is always $\\{(0,0)\\}$; the iteration grows the antichain outward along the diagonal $x + y \\approx 2\\sqrt{x+y} + c$ until it stabilises.
"""),
    ("md", "## Imports"),
    ("code", """import math
import matplotlib.pyplot as plt

from codesign import (
    Antichain, FunctionDP, Loop, Ports, Naturals, solve,
)

%matplotlib inline"""),
    ("md", "## The same model as notebook 02"),
    ("code", """def make_looped(c_value):
    N = Naturals()
    XY = Ports({"x": N, "y": N})
    F = Ports({"c": N, "xy": XY})
    R = Ports({"xy": XY, "xy_report": XY})

    def h(f):
        c = int(f["c"])
        x_in, y_in = f["xy"]["x"], f["xy"]["y"]
        if x_in == math.inf or y_in == math.inf:
            top = {"x": math.inf, "y": math.inf}
            return Antichain.singleton(R, {"xy": top, "xy_report": top})
        sx = math.isqrt(int(x_in)) + (1 if math.isqrt(int(x_in))**2 < int(x_in) else 0)
        sy = math.isqrt(int(y_in)) + (1 if math.isqrt(int(y_in))**2 < int(y_in) else 0)
        target = sx + sy + c
        pts = []
        for x_out in range(sx, target - sy + 1):
            y_out = target - x_out
            if y_out < sy:
                break
            pts.append({
                "xy": {"x": x_out, "y": y_out},
                "xy_report": {"x": x_out, "y": y_out},
            })
        if not pts:
            return Antichain.empty(R)
        return Antichain.from_set(R, pts)
    return Loop(FunctionDP(F=F, R=R, h_fn=h), axis="xy")"""),
    ("md", "## Plot the trace for a given c"),
    ("code", """def plot_trace(c_value):
    looped = make_looped(c_value)
    # record_trace=True keeps every intermediate antichain on result.trace.
    result = solve(looped, {"c": c_value}, max_iter=50, record_trace=True)
    trace = result.trace

    # Auto-scale the plot to the largest finite coordinate seen across the trace.
    max_xy = 1
    for entry in trace:
        for p in entry.antichain.points:
            x, y = p["xy"]["x"], p["xy"]["y"]
            if x != math.inf: max_xy = max(max_xy, int(x))
            if y != math.inf: max_xy = max(max_xy, int(y))
    bound = max_xy + 3

    # Lay out a grid of subplots, one per Kleene iteration.
    n = len(trace)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = [axes] if rows * cols == 1 else (axes.flat if rows > 1 else axes)
    axes = list(axes)

    # Render each antichain as a scatter. +inf points are skipped (they
    # encode infeasibility, not real coordinates to plot).
    for k, entry in enumerate(trace):
        A = entry.antichain
        ax = axes[k]
        xs, ys = [], []
        for p in A.points:
            x, y = p["xy"]["x"], p["xy"]["y"]
            if x == math.inf or y == math.inf: continue
            xs.append(x); ys.append(y)
        ax.scatter(xs, ys, s=60, c="C3", zorder=3)
        ax.set_xlim(-0.5, bound); ax.set_ylim(-0.5, bound)
        ax.set_xticks(range(0, bound + 1, max(1, bound // 6)))
        ax.set_yticks(range(0, bound + 1, max(1, bound // 6)))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"$S_{{{k}}}$ ({len(A.points)} pts)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    # Hide any leftover axes from the grid.
    for k in range(len(trace), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(
        f"Kleene ascent for c = {c_value}: "
        f"converged in {result.iterations} iters, "
        f"|M| = {len(result.antichain.points)}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    return result"""),
    ("md", "## c = 1"),
    ("code", "_ = plot_trace(1)"),
    ("md", "## c = 4 (the example from Fig. 36)"),
    ("code", "_ = plot_trace(4)"),
    ("md", "## c = 8"),
    ("code", "_ = plot_trace(8)"),
    ("md", """## Reading the plots

Every step of the iteration is monotone-non-decreasing in the antichain order: each point grows or splits into incomparable successors; nothing ever moves "down" in the poset. The number of points can fluctuate (some interior points get dominated by points discovered nearby), but the final antichain dominates the seed.

For this problem the iteration converges in O($\\sqrt{c}$) steps because the diagonal grows by roughly $\\Delta(x+y) \\approx 2$ per iteration.
"""),
]


# ---------------------------------------------------------------------------
# 06 MCDPL-style declarative syntax
# ---------------------------------------------------------------------------

NB_06 = [
    ("md", """# 06. The same drone with MCDPL-style syntax

The `MCDP` builder in `codesign.mcdpl` mirrors the paper's concrete syntax (`mcdp { provides ...; requires ...; ... >= ... }`) directly in Python. It is a thin wrapper over `AlgebraicDP`, `FunctionDP`, and `Loop`, but reads almost identically to the MCDPL source in Fig. 48.

This notebook rebuilds the same drone as notebook **01** with this builder. Output is identical.
"""),
    ("md", "## Imports"),
    ("code", "from codesign import MCDP, solve"),
    ("md", "## Build the drone"),
    ("code", """# Same physical constants as notebook 01.
ALPHA = 1.8e6      # Li-ion specific energy, J/kg
G = 9.81           # gravity, m/s^2
C_LIFT = 10.0      # actuator coefficient, W per N^2 of lift

# The `with MCDP(...)` context manager builds an internal port and
# constraint table; the final m.build() compiles it to a Loop(FunctionDP).
with MCDP("drone") as m:
    # provides = outer functionality ports (what the user supplies).
    m.provides("endurance", unit="s")
    m.provides("extra_payload", unit="kg")
    m.provides("extra_power", unit="W")
    m.provides("battery_mass", unit="kg")   # loop axis on the F side

    # requires = outer resource ports (what the system needs).
    m.requires("battery_mass", unit="kg")   # loop axis on the R side
    m.requires("report_mass",  unit="kg")   # outer-visible mirror

    # The constraint reads exactly like the paper:
    # battery_mass >= ((battery+payload)*g)^2 * C_LIFT + extra_power) * endurance / alpha
    def battery_mass_eq(f):
        lift = (f["battery_mass"] + f["extra_payload"]) * G
        actuator_power = C_LIFT * lift * lift
        total_power = actuator_power + f["extra_power"]
        energy = total_power * f["endurance"]
        return energy / ALPHA

    # Same expression bound to both R ports.
    m.constraint("battery_mass", battery_mass_eq)
    m.constraint("report_mass",  battery_mass_eq)
    # Close the loop. The Kleene iteration runs on battery_mass and the
    # outer R only exposes report_mass to downstream consumers.
    m.loop_on("battery_mass")

drone = m.build()
drone"""),
    ("md", "## Run the same cases as notebook 01"),
    ("code", """# Identical mission profiles to notebook 01, for a direct comparison.
cases = [
    ("Short, light",   dict(endurance=60.0,   extra_payload=0.10, extra_power=1.0)),
    ("Medium, modest", dict(endurance=300.0,  extra_payload=0.50, extra_power=5.0)),
    ("Longer mission", dict(endurance=600.0,  extra_payload=0.50, extra_power=5.0)),
    ("Infeasible",     dict(endurance=1800.0, extra_payload=1.00, extra_power=10.0)),
]
for label, f in cases:
    result = solve(drone, f, max_iter=80)
    print(f"{label:<16} iters={result.iterations:>3}  "
          f"feasible={result.feasible}  {result.antichain}")"""),
    ("md", """## Compared with notebook 01

The numbers and iteration counts match exactly. The `MCDP` builder is purely syntactic sugar; it produces the same `Loop(FunctionDP(...))` underneath. The advantage is that it reads more like the MCDPL block in the paper, which is the natural notation for a single self-contained problem with provides/requires declarations.

For *modular* composition with named subsystems, the `System` builder in notebook **07** is more idiomatic.
"""),
]


# ---------------------------------------------------------------------------
# 07 Modular drone with System
# ---------------------------------------------------------------------------

NB_07 = [
    ("md", """# 07. The drone, modular: MCDPL-style operator syntax

The same MCDP as notebooks **01** and **06**, but with battery and actuator defined as **independent** subsystems and assembled with the `System` builder. This notebook uses the new operator-overloaded constraint syntax: each connection is written as a Python `>=` between port handles, mirroring how the same model would appear in the MCDPL source from the paper.

This is the recommended style for modular design. Each subsystem is a `Module` subclass with its own `F`, `R`, and `h`. The wiring at the bottom reads as a column of inequalities, the way a textbook would.
"""),
    ("md", "## Imports"),
    ("code", "from codesign import Module, Reals, System, solve"),
    ("md", """## Subsystems as class-based Modules

Each is a self-contained design problem. The constructor accepts parameters so a single class can be reused with different physical constants.
"""),
    ("code", """class Battery(Module):
    # F, R declared as class-level dicts; the Module base class wires them
    # into a DesignProblem during __init__.
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}

    def __init__(self, specific_energy=1.8e6):
        # Set the instance attribute BEFORE calling super().__init__() so
        # that any h() called during construction sees it.
        self.specific_energy = specific_energy
        super().__init__()

    def h(self, f):
        # mass = capacity / specific_energy. Returning a dict yields a
        # singleton antichain.
        return {"mass": f["capacity"] / self.specific_energy}


class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}

    def __init__(self, c_lift=10.0):
        self.c_lift = c_lift
        super().__init__()

    def h(self, f):
        # Quadratic drag-like power: P = c_lift * F^2.
        return {"power": self.c_lift * f["lift_force"] ** 2}


Battery(), Actuator()"""),
    ("md", """## Assembly via operator-overloaded constraints

`sys.provides`, `sys.requires`, and `sys.add` each return a port handle. Arithmetic operators on handles build expression trees lazily; the `>=` operator registers a constraint with the parent system.
"""),
    ("code", """G = 9.81

sys = System("drone")

# Outer interface: each call returns a Port handle.
endurance     = sys.provides("endurance",     unit="s")
extra_payload = sys.provides("extra_payload", unit="kg")
extra_power   = sys.provides("extra_power",   unit="W")
total_mass    = sys.requires("total_mass",    unit="kg")

# Subsystems: each call returns a ModuleHandle whose attributes are ports.
battery  = sys.add("battery",  Battery())
actuator = sys.add("actuator", Actuator())

# Connection constraints. Read like a textbook page of inequalities.
battery.capacity    >= (actuator.power + extra_power) * endurance
actuator.lift_force >= G * (battery.mass + extra_payload)
total_mass          >= battery.mass + extra_payload

print(sys)"""),
    ("md", "## Build and solve"),
    ("code", """# build() compiles the constraint table into a Loop(System_inner) DP.
drone = sys.build()
print(drone)
print()

# Five mission profiles, ordered roughly by difficulty.
cases = [
    ("Short, light",   dict(endurance=60.0,   extra_payload=0.10, extra_power=1.0)),
    ("Medium, modest", dict(endurance=300.0,  extra_payload=0.50, extra_power=5.0)),
    ("Longer mission", dict(endurance=600.0,  extra_payload=0.50, extra_power=5.0)),
    ("Marginal",       dict(endurance=600.0,  extra_payload=1.00, extra_power=10.0)),
    ("Infeasible",     dict(endurance=1800.0, extra_payload=1.00, extra_power=10.0)),
]
for label, f in cases:
    # The System builder bundles every subsystem's R into a single Kleene
    # axis; iteration counts here are typically a bit higher than the
    # monolithic Loop version of notebook 01, but converge to the same point.
    result = solve(drone, f, max_iter=200)
    print(f"{label:<16} iters={result.iterations:>3}  "
          f"feasible={result.feasible}  {result.antichain}")"""),
    ("md", """## What changed compared to notebook 01

The values are identical (for Medium, modest: `total_mass = 0.5492 kg = 0.04921 (battery) + 0.5 (payload)`). The Kleene loop now updates each subsystem's R ports in alternation, so the iteration count is somewhat larger than the monolithic version. The fixed point is the same.

The payoff is **clarity and modularity**: `Battery` and `Actuator` are reusable building blocks, the constraints between them read as math rather than dict lookups, and adding a third subsystem (notebook **08**) or building a heterogeneous network (notebook **09**) requires no extra machinery.

### The same example in the lambda-based legacy syntax

For comparison, here is what the constraint block looked like before:

```python
sys.constrain("battery.capacity",
              lambda x: (x["actuator.power"] + x["extra_power"]) * x["endurance"])
sys.constrain("actuator.lift_force",
              lambda x: G * (x["battery.mass"] + x["extra_payload"]))
sys.constrain("total_mass",
              lambda x: x["battery.mass"] + x["extra_payload"])
```

Both forms still work and compile to the same internal constraint list. The operator form is what we recommend for new code.
"""),
    ("md", """## System structure with `viz.to_dot`

The constraint graph is implicit in the operator overloading; `viz.to_dot` extracts it as a GraphViz dot string. Render it externally with `dot -Tpng` or paste into [graphviz online](https://dreampuf.github.io/GraphvizOnline/) to see modules, outer ports, and the edges connecting them.
"""),
    ("code", """from codesign import viz
dot = viz.to_dot(drone, name="drone")
print(dot)"""),
]


# ---------------------------------------------------------------------------
# 08 Modular vehicle with multi-point Pareto front
# ---------------------------------------------------------------------------

NB_08 = [
    ("md", """# 08. Motor + chassis + battery: a multi-point Pareto front

A small electric vehicle is co-designed from three subsystems:

- a **motor** picked from a discrete catalog (each entry has its own `(torque, mass, cost)` tuple),
- a **chassis** whose mass and cost scale with the load it supports,
- a **battery** sized to the mission energy.

The subsystems are coupled cyclically: the chassis must support the motor and battery, the motor's torque is sized by the total moving mass (which includes the chassis), and so on. The `System` builder closes the loop.

Because the motor catalog has Pareto-incomparable entries, the system-level Pareto front has multiple points: real engineering tradeoffs surface automatically.

This notebook also uses the operator-overloaded constraint syntax introduced in notebook 07.
"""),
    ("md", "## Imports"),
    ("code", """from codesign import (
    CatalogDP, CatalogEntry, Module, Ports, Reals,
    System, minimize_cost, solve,
)"""),
    ("md", """## Subsystem 1: a motor catalog

The catalog has Pareto-incomparable entries (lighter and more expensive vs. heavier and cheaper). `CatalogDP` is kept as a plain function constructor: multi-valued antichains don't fit the `Module` declarative pattern as cleanly.
"""),
    ("code", """# Seven entries spanning four torque classes. Notice that within each
# torque rating there can be multiple Pareto-incomparable entries (cheaper
# but heavier vs lighter but pricier); this is what produces the multi-point
# antichain at the system level.
motor = CatalogDP(
    F=Ports({"torque": Reals(unit="N*m")}),
    R=Ports({"mass": Reals(unit="kg"), "cost": Reals(unit="USD")}),
    catalog=[
        CatalogEntry(name="Tiny",          provides={"torque": 2.0},  costs={"mass": 0.20, "cost": 30.0}),
        CatalogEntry(name="Light-Premium", provides={"torque": 8.0},  costs={"mass": 0.50, "cost": 200.0}),
        CatalogEntry(name="Mid-Standard",  provides={"torque": 8.0},  costs={"mass": 0.80, "cost": 120.0}),
        CatalogEntry(name="Heavy-Budget",  provides={"torque": 20.0}, costs={"mass": 1.50, "cost": 90.0}),
        CatalogEntry(name="Light-Pro",     provides={"torque": 20.0}, costs={"mass": 0.90, "cost": 350.0}),
        CatalogEntry(name="XL-Budget",     provides={"torque": 80.0}, costs={"mass": 3.50, "cost": 180.0}),
        CatalogEntry(name="XL-Pro",        provides={"torque": 80.0}, costs={"mass": 2.20, "cost": 700.0}),
    ],
    name="motor",
)
motor"""),
    ("md", "## Subsystems 2 and 3: chassis and battery as Module classes"),
    ("code", """class Chassis(Module):
    # The chassis mass and cost scale linearly with the load it supports.
    # Because the chassis itself contributes to that load, this creates the
    # cyclic dependency that the Kleene iteration resolves.
    F = {"load": Reals(unit="kg")}
    R = {"mass": Reals(unit="kg"), "cost": Reals(unit="USD")}

    def h(self, f):
        return {
            "mass": 0.6  * f["load"],     # 60% mass per unit supported
            "cost": 20.0 * f["load"],     # $20 per kg supported
        }


class Battery(Module):
    # Sized purely by the mission energy demand; specific energy 1.8 MJ/kg,
    # cell cost $0.05 per Wh.
    F = {"energy": Reals(unit="J")}
    R = {"mass":   Reals(unit="kg"), "cost": Reals(unit="USD")}

    def h(self, f):
        return {
            "mass": f["energy"] / 1.8e6,
            "cost": 0.05 * f["energy"] / 3.6e3,  # $0.05 per Wh
        }"""),
    ("md", """## Wire it up with `>=`

The chassis must support payload plus motor plus battery; the motor's torque demand depends on the total moving mass; the battery's energy demand is set externally. Total mass and total cost aggregate from every subsystem.
"""),
    ("code", """G = 9.81
TORQUE_PER_KG = 0.25     # required torque to accelerate 1 kg

sys = System("vehicle")

# Outer ports: mission spec in, system-level aggregates out.
payload        = sys.provides("payload",        unit="kg")
mission_energy = sys.provides("mission_energy", unit="J")
total_mass     = sys.requires("total_mass",     unit="kg")
total_cost     = sys.requires("total_cost",     unit="USD")

# Three subsystems. Each returns a ModuleHandle whose attributes are ports.
m = sys.add("motor",   motor)
c = sys.add("chassis", Chassis())
b = sys.add("battery", Battery())

# Connection constraints. Each line reads like a textbook inequality:
#   chassis must support payload + motor + battery
#   motor torque must accelerate payload + chassis + battery
#   battery energy must meet the mission demand
c.load   >= payload + m.mass + b.mass
m.torque >= TORQUE_PER_KG * G * (payload + c.mass + b.mass)
b.energy >= mission_energy

# Outer-R aggregation: total mass and cost roll up from every subsystem.
total_mass >= payload + m.mass + c.mass + b.mass
total_cost >= m.cost + c.cost + b.cost

vehicle = sys.build()
print(sys)"""),
    ("md", "## Solve for three mission profiles"),
    ("code", """cases = [
    ("Small parcel", dict(payload=2.0,  mission_energy=2.0e5)),
    ("Medium load",  dict(payload=10.0, mission_energy=1.0e6)),
    ("Heavy + long", dict(payload=20.0, mission_energy=5.0e6)),
]
for label, f in cases:
    result = solve(vehicle, f, max_iter=200)
    f_str = ", ".join(f"{k}={v}" for k, v in f.items())
    print(f"{label}: {f_str}")
    print(f"   iters={result.iterations}, feasible={result.feasible}")
    if not result.feasible:
        # Heaviest case: even the largest motor can't supply enough torque.
        # The loop axis is driven to top; no special handling needed here.
        print()
        continue
    # Walk the Pareto front. Cases 1 and 2 produce a 2-point front from the
    # incomparable motor choices that survive after the chassis grows.
    print(f"   Pareto front ({len(result.antichain.points)} points):")
    for p in result.antichain.points:
        print(f"      total_mass={p['total_mass']:6.2f} kg,  "
              f"total_cost=${p['total_cost']:7.2f}")
    # Scalarise the front by picking the cheapest design.
    cheapest = minimize_cost(result, cost_fn=lambda r: r["total_cost"])
    if cheapest is not None:
        print(f"   cheapest: total_mass={cheapest['total_mass']:.2f} kg, "
              f"total_cost=${cheapest['total_cost']:.2f}")
    print()"""),
    ("md", """## Reading the Pareto fronts

For Small parcel and Medium load, two motor choices remain feasible after the chassis grows to support everything, and they trade total mass against total cost: one is lighter but more expensive, the other heavier but cheaper. Neither dominates, so both appear in the system Pareto front.

For Heavy + long, even the biggest motor in the catalog can't supply enough torque once the chassis (which scales with battery mass) grows large enough. The iteration drives the loop axis to `⊤` and the result is `feasible=False`, automatically and without any special handling in the example code.

This is the payoff of modularity: each subsystem has a clean local definition, and the global Pareto structure is discovered by the solver from the constraint graph alone.
"""),
    ("md", """## Visualising the Pareto front

`codesign.viz.plot_antichain` renders the front as a 2D scatter with dominance regions shaded. Each star is a Pareto-optimal design; the shaded quadrant is the set of (mass, cost) pairs it dominates.
"""),
    ("code", """import matplotlib.pyplot as plt
from codesign import viz

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (label, f) in zip(axes, cases[:2]):
    result = solve(vehicle, f, max_iter=200)
    viz.plot_antichain(result, axes=["total_mass", "total_cost"], ax=ax)
    ax.set_title(f"{label}: {len(result.antichain)} Pareto designs")
fig.tight_layout()
plt.show()"""),
]


# ---------------------------------------------------------------------------
# 09 A robotic arm: non-trivial connection topology
# ---------------------------------------------------------------------------

NB_09 = [
    ("md", """# 09. A robotic arm: non-trivial connection topology

This notebook exercises the operator-overloaded syntax on a problem where the connections don't follow a clean series / parallel / loop pattern. The arm has:

- two **joint** actuators (shoulder and elbow), each with their own torque and power characteristics,
- a **controller** driving both joints and ingesting sensor data,
- a **sensor** measuring arm position,
- a **battery** sized by total mission energy.

The connection graph is genuinely non-trivial: the controller's power demand depends on both incoming sensor data and outgoing commands, both joints share the same battery via a power bus, and the shoulder torque has to support the elbow's own mass at the end of the shoulder arm. The Kleene iteration resolves the resulting cycles automatically.

The point of this notebook is that the resulting code reads as a paragraph of physical relationships, not a sequence of dict-juggling lambdas.
"""),
    ("md", "## Imports"),
    ("code", "from codesign import Module, Reals, System, solve"),
    ("md", """## Joint, sensor, controller, battery

Each subsystem is a small `Module`. The `Joint` class is reused for both shoulder and elbow with different `motor_density` parameters.
"""),
    ("code", """G = 9.81


class Joint(Module):
    # A torque-and-speed joint. Mass scales with torque (motor_density is
    # kg per N*m); electric power is torque*speed divided by drivetrain
    # efficiency. Reused for shoulder (heavier) and elbow (lighter).
    F = {"torque": Reals(unit="N*m"), "speed": Reals(unit="rad/s")}
    R = {"mass":   Reals(unit="kg"),  "electric_power": Reals(unit="W")}

    def __init__(self, motor_density=0.20, efficiency=0.85):
        self.motor_density = motor_density
        self.efficiency = efficiency
        super().__init__()

    def h(self, f):
        return {
            "mass":           self.motor_density * f["torque"],
            "electric_power": f["torque"] * f["speed"] / self.efficiency,
        }


class Sensor(Module):
    # Affine power model: per-sample cost plus a fixed idle draw. Mass
    # is constant.
    F = {"sample_rate": Reals(unit="Hz")}
    R = {"power": Reals(unit="W"), "mass": Reals(unit="kg")}

    def h(self, f):
        return {"power": 0.02 * f["sample_rate"] + 0.5, "mass": 0.05}


class Controller(Module):
    # Power scales with both input (sensor) and output (command) rates.
    F = {"input_rate": Reals(unit="Hz"), "command_rate": Reals(unit="Hz")}
    R = {"power":      Reals(unit="W"),  "mass":         Reals(unit="kg")}

    def h(self, f):
        return {
            "power": 0.05 * (f["input_rate"] + f["command_rate"]) + 2.0,
            "mass":  0.15,
        }


class Battery(Module):
    # Same battery as notebook 07: mass = energy / specific_energy.
    F = {"energy": Reals(unit="J")}
    R = {"mass":   Reals(unit="kg")}

    def __init__(self, specific_energy=1.8e6):
        self.specific_energy = specific_energy
        super().__init__()

    def h(self, f):
        return {"mass": f["energy"] / self.specific_energy}"""),
    ("md", """## Wire everything together

Seven mission parameters drive the system; one resource (total mass) comes out. The eight constraint lines below capture the entire interconnection.
"""),
    ("code", """sys = System("robotic_arm")

# Mission parameters.
payload_mass    = sys.provides("payload_mass",    unit="kg")
operating_time  = sys.provides("operating_time",  unit="s")
elbow_speed     = sys.provides("elbow_speed",     unit="rad/s")
shoulder_speed  = sys.provides("shoulder_speed",  unit="rad/s")
control_rate    = sys.provides("control_rate",   unit="Hz")
elbow_arm       = sys.provides("elbow_arm",      unit="m")
shoulder_arm    = sys.provides("shoulder_arm",   unit="m")

total_mass = sys.requires("total_mass", unit="kg")

elbow      = sys.add("elbow",      Joint())
shoulder   = sys.add("shoulder",   Joint(motor_density=0.25))
sensor     = sys.add("sensor",     Sensor())
controller = sys.add("controller", Controller())
battery    = sys.add("battery",    Battery())

# Mechanical chain: the elbow lifts only the payload at its arm length,
# the shoulder lifts the payload plus the elbow joint itself at its
# longer arm.
elbow.torque    >= G * payload_mass * elbow_arm
elbow.speed     >= elbow_speed
shoulder.torque >= G * (payload_mass + elbow.mass) * shoulder_arm
shoulder.speed  >= shoulder_speed

# Electronics: sensor must Nyquist the control loop; controller must
# keep up with both incoming samples and outgoing commands.
sensor.sample_rate      >= 2.0 * control_rate
controller.input_rate   >= 2.0 * control_rate
controller.command_rate >= control_rate

# Energy budget: battery sized by integrated power over the mission.
battery.energy >= operating_time * (
    elbow.electric_power + shoulder.electric_power
    + controller.power + sensor.power
)

# Aggregate mass.
total_mass >= (
    payload_mass
    + elbow.mass + shoulder.mass
    + sensor.mass + controller.mass
    + battery.mass
)

print(sys)"""),
    ("md", "## Build and solve"),
    ("code", """arm = sys.build()

cases = [
    ("Pick-and-place light", dict(
        payload_mass=0.5, operating_time=300.0,
        elbow_speed=2.0, shoulder_speed=1.5,
        control_rate=100.0,
        elbow_arm=0.3, shoulder_arm=0.5,
    )),
    ("Heavier payload", dict(
        payload_mass=2.0, operating_time=600.0,
        elbow_speed=1.5, shoulder_speed=1.0,
        control_rate=200.0,
        elbow_arm=0.3, shoulder_arm=0.5,
    )),
    ("Long-reach precise", dict(
        payload_mass=1.0, operating_time=900.0,
        elbow_speed=1.0, shoulder_speed=0.8,
        control_rate=500.0,
        elbow_arm=0.6, shoulder_arm=0.9,
    )),
]
for label, f in cases:
    result = solve(arm, f, max_iter=300)
    print(f"{label}:")
    print(f"   iters={result.iterations}, feasible={result.feasible}")
    if result.feasible:
        for p in result.antichain.points:
            print(f"   total_mass = {p['total_mass']:.2f} kg")
    print()"""),
    ("md", """## What the DSL provides

The eight constraint lines above each express a single physical relationship. In the lambda-based form, the same model would have grown to roughly twenty-five lines of `lambda x: x["something.something"] + ...` boilerplate, with every port name as a string. The operator-overloaded version:

- catches typos: `elbow.toruqe` becomes an `AttributeError` on the line where the typo happens, not deep inside `build()`,
- catches semantic mistakes: trying to put a module F port (e.g. `sensor.sample_rate`) on the RHS of an expression raises immediately, with a message explaining the rule,
- prints constraints as readable inequalities in `print(sys)`,
- composes through expression trees so refactoring is just arithmetic.

The cyclic dependencies (shoulder torque depending on elbow mass; battery energy depending on both joints' powers, which depend on their torques, which depend on the elbow mass which depends on the payload) are resolved transparently by the Kleene iteration: four iterations for all three test cases.
"""),
]


# ---------------------------------------------------------------------------
# 10 Solver observability: trace, verbose, callback, status
# ---------------------------------------------------------------------------

NB_10 = [
    ("md", """# 10. Watching the solver work

The solver supports three observability features that turn it from a black box into a debugging tool. This notebook exercises each in turn.

- `verbose=0|1|2` controls live printing: silent, end-of-solve summary, or per-iteration progress feed.
- `trace=True` collects a structured `TraceEntry` per iteration on `result.trace`. Useful for plotting convergence behaviour or writing regression tests on iteration counts.
- `on_iteration=callable` is invoked with each `TraceEntry` as it is produced, useful for live plotting or custom logging.

And the `result.status` field (`"converged"`, `"max_iter"`, `"diverged"`) distinguishes the solver's termination reason from `result.feasible` (which is about the answer, not the iteration).
"""),
    ("md", "## A small drone for testing"),
    ("code", """from codesign import Module, Reals, System, solve

# Minimal two-subsystem drone for exercising the observability features.
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

sys = System("drone")
endurance     = sys.provides("endurance",     unit="s")
extra_payload = sys.provides("extra_payload", unit="kg")
extra_power   = sys.provides("extra_power",   unit="W")
total_mass    = sys.requires("total_mass",    unit="kg")
b = sys.add("battery",  Battery())
a = sys.add("actuator", Actuator())
# Same three-line wiring as notebook 07.
b.capacity    >= (a.power + extra_power) * endurance
a.lift_force  >= 9.81 * (b.mass + extra_payload)
total_mass    >= b.mass + extra_payload
drone = sys.build()
# Fixed mission used everywhere in this notebook.
f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}"""),
    ("md", "## verbose=1: a one-line summary"),
    ("code", "_ = solve(drone, f, verbose=1)"),
    ("md", "## verbose=2: a per-iteration feed"),
    ("code", "_ = solve(drone, f, verbose=2, max_iter=10)"),
    ("md", """## trace=True: collect the iterations as data

The trace is a list of `TraceEntry` objects, one per iteration (plus the seed at iteration 0). Each carries the antichain at that step, the number of points, the convergence delta (max absolute change in any port value), and the wall time spent on that step alone.
"""),
    ("code", """r = solve(drone, f, trace=True, max_iter=200)
print(f"status={r.status}, iters={r.iterations}, feasible={r.feasible}")
print(f"trace has {len(r.trace)} entries (iteration 0 = seed, then 1..N)")
print(f"first 5 deltas: {[e.delta for e in r.trace[:5]]}")
print(f"final delta: {r.trace[-1].delta}")"""),
    ("md", """## Plotting the convergence

The `codesign.viz` module provides `plot_convergence` as a one-liner; for this drone the deltas oscillate between two coupled axes (battery mass and actuator power) and decay to roughly machine precision.
"""),
    ("code", """import matplotlib.pyplot as plt
from codesign import viz

ax = viz.plot_convergence(r)
ax.set_title("Convergence of the drone fixed point")
plt.tight_layout()
plt.show()"""),
    ("md", """## on_iteration: a custom callback

The callback receives each `TraceEntry` as it is produced. The drone here is small, so we print every 5th iteration to keep things tidy. In a real GUI or notebook plotter you'd update a live figure instead.
"""),
    ("code", """def my_logger(entry):
    if entry.iteration % 5 == 0:
        d = "    -    " if entry.delta is None else f"{entry.delta:.3e}"
        print(f"   iter {entry.iteration:>3}: |A|={entry.n_points}, delta={d}")

_ = solve(drone, f, on_iteration=my_logger, max_iter=100)"""),
    ("md", """## status vs feasible

The `status` field describes the solver's termination reason. The `feasible` field describes the answer. They are orthogonal: a solve can terminate cleanly on an infeasible problem, and a max-iter cut might leave a still-converging-but-feasible run looking suspect.
"""),
    ("code", """# A run cut short:
r_short = solve(drone, f, max_iter=3)
print(f"max_iter=3:  status={r_short.status!r}, feasible={r_short.feasible}, iters={r_short.iterations}")

# A clean converged run:
r_ok = solve(drone, f, max_iter=200)
print(f"max_iter=200: status={r_ok.status!r}, feasible={r_ok.feasible}, iters={r_ok.iterations}")

# A genuinely infeasible run (too long an endurance for the battery):
r_inf = solve(drone, {"endurance": 1800.0, "extra_payload": 1.0, "extra_power": 10.0}, max_iter=200)
print(f"infeasible:  status={r_inf.status!r}, feasible={r_inf.feasible}, iters={r_inf.iterations}")"""),
    ("md", """The third case is interesting: the solver detects numerical divergence (some port value crossing the divergence cap of 1e30) and stops with `status='diverged'`, which is more informative than just `feasible=False`. With only the feasible flag you couldn't tell whether bumping `max_iter` would help.
"""),
]


# ---------------------------------------------------------------------------
# 11 Set-based deterministic uncertainty
# ---------------------------------------------------------------------------

NB_11 = [
    ("md", """# 11. Set-based deterministic uncertainty

The drone from notebook 07 is extended so the battery has two internal parameters (specific energy, efficiency) that are known only up to an uncertainty set. The question is: under the worst-case point of that set, how heavy does the drone become?

Two sets are exercised:

- A `Box`: rectangular ranges on the two parameters, with each range declared in the "more is better" direction so the worst case is the corner where both parameters take their lowest values.
- An `Ellipsoid`: a tilted, correlated set smaller than the box; the worst case lies on its boundary in the direction of badness, not at a corner.

Set-based uncertainty fits MCDP naturally: monotonicity means the worst case is always on the set's boundary, in the direction of badness, and the answer is a single antichain rather than a distribution.
"""),
    ("md", "## Imports and modules"),
    ("code", """from codesign import Box, Ellipsoid, Module, Reals, System, solve

class Battery(Module):
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}
    def __init__(self, specific_energy=1.8e6, efficiency=0.85):
        self.specific_energy = specific_energy
        self.efficiency = efficiency
        super().__init__()
    def h(self, f):
        return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}

class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}
    def h(self, f):
        return {"power": 10.0 * f["lift_force"] ** 2}

def make_drone(battery):
    sys = System("drone")
    endurance     = sys.provides("endurance",     unit="s")
    extra_payload = sys.provides("extra_payload", unit="kg")
    extra_power   = sys.provides("extra_power",   unit="W")
    total_mass    = sys.requires("total_mass",    unit="kg")
    b = sys.add("battery",  battery)
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_power) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_payload)
    total_mass    >= b.mass + extra_payload
    return sys.build()

f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}"""),
    ("md", "## Nominal: no uncertainty"),
    ("code", """drone = make_drone(Battery())
nominal = solve(drone, f)
nominal_mass = list(nominal.antichain.points)[0]["total_mass"]
print(f"nominal total_mass = {nominal_mass:.4f} kg")"""),
    ("md", """## Box uncertainty

The Box puts independent ranges on each parameter. Because both are declared "more is better," the worst case is the single corner where specific_energy = 1.6e6 and efficiency = 0.80.
"""),
    ("code", """bat = Battery()
# Box: independent ranges on each parameter. The direction-of-badness token
# tells the solver which endpoint is the "worst" - for both parameters here,
# lower is worse since they're declared "more_is_better".
bat.uncertain_set = Box(
    specific_energy=(1.6e6, 2.0e6, "more_is_better"),
    efficiency=(0.80, 0.90, "more_is_better"),
)
drone = make_drone(bat)

# uncertainty=["worst_case"] runs one solve at the worst corner of the box.
r_box = solve(drone, f, uncertainty=["worst_case"])
wc = list(r_box.worst_case.antichain.points)[0]["total_mass"]
print(f"Box worst case: {wc:.4f} kg  (penalty {wc - nominal_mass:+.4f} kg)")"""),
    ("md", """## Ellipsoid uncertainty (smaller, correlated set)

The Ellipsoid carves out the implausible corner where both parameters are simultaneously at their extremes. The worst case lies on the curved boundary in the direction of badness, which is closer to the centre than the box's worst-case corner.
"""),
    ("code", """bat = Battery()
# Ellipsoid: a tilted set centred at the nominal values. The negative
# off-diagonal cov entry encodes a negative correlation: pessimistic on
# specific_energy makes optimistic on efficiency more likely, and vice
# versa. The worst case is therefore not the joint-pessimistic corner.
bat.uncertain_set = Ellipsoid(
    center={"specific_energy": 1.8e6, "efficiency": 0.85},
    cov=[
        [1.0e10, -2.0e3],     # variance of specific_energy, covariance
        [-2.0e3,  2.5e-3],    # covariance, variance of efficiency
    ],
    params=["specific_energy", "efficiency"],
    directions={
        "specific_energy": "more_is_better",
        "efficiency":      "more_is_better",
    },
)
drone = make_drone(bat)

r_ell = solve(drone, f, uncertainty=["worst_case"])
wc_ell = list(r_ell.worst_case.antichain.points)[0]["total_mass"]
print(f"Ellipsoid worst case: {wc_ell:.4f} kg  (penalty {wc_ell - nominal_mass:+.4f} kg)")"""),
    ("md", """## Summary

| Set       | Worst-case mass | Penalty vs nominal |
|-----------|-----------------|--------------------|
| (nominal) | 0.5602 kg       | -                  |
| Box       | 0.5760 kg       | +0.0158 kg         |
| Ellipsoid | 0.5652 kg       | +0.0050 kg         |

The ellipsoid is the more honest model when you believe the two parameters are correlated, since it rejects the "both at the worst simultaneously" combination as implausible. The 2D conveniences `Disk(center, radius)` and `Circle(center, radius)` are special cases of `Ellipsoid`.
"""),
]


# ---------------------------------------------------------------------------
# 12 Stochastic uncertainty with Gaussian copula
# ---------------------------------------------------------------------------

NB_12 = [
    ("md", """# 12. Stochastic uncertainty with a Gaussian copula

When the parameters aren't just bounded but have probability distributions, the analysis shifts from "worst case" to "statistical summaries." This notebook exercises Monte Carlo sampling with a Gaussian copula glueing two marginals together, and shows how a single solve call returns several summaries at once.

The drone has both `uncertain_set` (a Box) and `uncertain_dist` (a Stochastic with a Gaussian copula) attached to its battery. We ask for both kinds of answers and see how they compare.
"""),
    ("md", "## Imports and model"),
    ("code", """from scipy import stats
from codesign import (
    Box, GaussianCopula, Module, Reals, Stochastic, System, solve,
)

class Battery(Module):
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}
    def __init__(self, specific_energy=1.8e6, efficiency=0.85):
        self.specific_energy = specific_energy
        self.efficiency = efficiency
        super().__init__()
    def h(self, f):
        return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}

class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}
    def h(self, f):
        return {"power": 10.0 * f["lift_force"] ** 2}

bat = Battery()
# Set-based bracket: the Box gives a deterministic worst-case answer.
bat.uncertain_set = Box(
    specific_energy=(1.6e6, 2.0e6, "more_is_better"),
    efficiency=(0.80, 0.90, "more_is_better"),
)
# Stochastic model: two uniform marginals tied by a Gaussian copula.
# The 0.4 off-diagonal entry means specific_energy and efficiency are
# positively correlated in their joint distribution (good cells tend to
# be efficient too), softening the joint-pessimistic case.
bat.uncertain_dist = Stochastic(
    marginals={
        "specific_energy": stats.uniform(loc=1.6e6, scale=0.4e6),
        "efficiency":      stats.uniform(loc=0.80, scale=0.10),
    },
    copula=GaussianCopula(correlation=[[1.0, 0.4],
                                       [0.4, 1.0]]),
)

# Wiring identical to notebook 07.
sys = System("drone")
endurance     = sys.provides("endurance",     unit="s")
extra_payload = sys.provides("extra_payload", unit="kg")
extra_power   = sys.provides("extra_power",   unit="W")
total_mass    = sys.requires("total_mass",    unit="kg")
b = sys.add("battery",  bat)
a = sys.add("actuator", Actuator())
b.capacity    >= (a.power + extra_power) * endurance
a.lift_force  >= 9.81 * (b.mass + extra_payload)
total_mass    >= b.mass + extra_payload
drone = sys.build()
f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}"""),
    ("md", "## A nominal solve, for reference"),
    ("code", """nominal = solve(drone, f)
nominal_mass = list(nominal.antichain.points)[0]["total_mass"]
print(f"Nominal mass: {nominal_mass:.4f} kg")"""),
    ("md", """## All summaries in one call

The solver gathers the deterministic worst case and runs the Monte Carlo for all the statistical summaries in a single pass.
"""),
    ("code", """# Each summary in `uncertainty` triggers one piece of work:
#   "worst_case" -> one deterministic solve at the Box corner
#   "mean" / "p95" / "cvar95" -> aggregated over the n_samples MC runs
#   "samples" -> keep the raw antichain per MC sample on res.samples
res = solve(
    drone, f,
    uncertainty=["worst_case", "mean", "p95", "cvar95", "samples"],
    n_samples=1000,
    rng_seed=42,           # for reproducible MC draws
    verbose=1,
)

wc = list(res.worst_case.antichain.points)[0]["total_mass"]
print()
# Canonical ordering: nominal < mean < p95 < CVaR95 < worst_case.
print(f"Worst case (Box):     {wc:.4f} kg")
print(f"Mean:                 {res.mean['total_mass']:.4f} kg")
print(f"95th percentile:      {res.p95['total_mass']:.4f} kg")
print(f"CVaR (worst 5% mean): {res.cvar95['total_mass']:.4f} kg")
print(f"Feasibility rate:     {res.feasibility_rate:.3f}")"""),
    ("md", """## Visualising the distribution

The raw antichain per MC sample is on `res.samples`. The `codesign.viz` module provides `plot_uncertainty` as a one-liner that draws the histogram and marks each summary statistic.
"""),
    ("code", """import matplotlib.pyplot as plt
from codesign import viz

ax = viz.plot_uncertainty(res, port="total_mass",
                          nominal=nominal_mass, bins=30)
wc = list(res.worst_case.antichain.points)[0]["total_mass"]
ax.axvline(wc, color="black", linestyle="--",
           label=f"worst case (Box) {wc:.4f}")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Monte Carlo distribution of total_mass, with summaries")
plt.tight_layout()
plt.show()"""),
    ("md", """## Reading the chart

- **nominal** is what you'd get if you ignored the uncertainty entirely (parameters at their declared centres).
- **mean** is the expected total_mass over the joint distribution. Slightly above nominal because the distribution is skewed (the parameters' product is harmonic-mean-like; small values dominate).
- **p95** is "95% of designs are no heavier than this." Pessimistic but useful for typical specification claims.
- **CVaR95** is "the average mass over the worst 5% of scenarios." Standard for engineering risk.
- **worst case (Box)** is the deterministic upper bound from the set-based analysis. Slightly above CVaR95 because the Box admits the implausible corner.

A natural ordering emerges: `nominal < mean < p95 < CVaR95 < worst_case`. For a normal-design specification you'd usually report the p95 or CVaR95; the worst case is the right answer when failures truly mean disaster.
"""),
]


# ---------------------------------------------------------------------------
# 13 The microgrid flagship: cycles + warm-start + uncertainty + viz
# ---------------------------------------------------------------------------

NB_13 = [
    ("md", """# 13. Microgrid: a flagship case study

An off-grid cabin must supply its daily energy and peak load without grid power. Four subsystems contribute:

- a **solar PV array** (cheap when the sun shines, but the sun is stochastic);
- a **lithium battery** with a discrete chemistry choice (LFP, NMC, LCO, NaIon);
- a **diesel generator** (reliable but carbon-heavy);
- a **structural frame** whose mass and cost scale with the total mass it must support, *including its own*, producing a genuine fixed-point coupling.

This notebook brings together every feature of the package: cyclic dependencies, parameter sweeps with solver warm-start, stochastic uncertainty on the sun, and the new visualisation helpers in `codesign.viz`.
"""),
    ("md", "## Modules"),
    ("code", """from codesign import Module, Reals, Stochastic, System, solve, viz
from scipy import stats
import numpy as np


class SolarArray(Module):
    # Required peak power is the larger of the user's stated peak demand
    # and the average daily-energy demand divided by available sun hours.
    # cost and mass scale linearly with this required peak (kW).
    F = {"peak_power_kw": Reals(unit="kW"),
         "daily_energy_kwh": Reals(unit="kWh")}
    R = {"cost_usd": Reals(unit="USD"), "mass_kg": Reals(unit="kg")}

    def __init__(self, cost_per_kw=1100.0, mass_per_kw=28.0,
                 sun_hours_per_day=3.0):
        self.cost_per_kw = cost_per_kw
        self.mass_per_kw = mass_per_kw
        self.sun_hours_per_day = sun_hours_per_day
        super().__init__()

    def h(self, f):
        # Guard against sun=0 (no sun, infinite array needed).
        sun = max(self.sun_hours_per_day, 1e-6)
        # Whichever constraint binds: instantaneous peak, or daily average.
        required_peak = max(f["peak_power_kw"],
                            f["daily_energy_kwh"] / sun)
        return {"cost_usd": required_peak * self.cost_per_kw,
                "mass_kg":  required_peak * self.mass_per_kw}


class Battery(Module):
    # Battery sized to storage demand. Four chemistries differ in specific
    # energy (Wh/kg), cost density (USD/kWh), and cycle life (equivalent
    # full-cycles before replacement). The catalogue is a static dict
    # rather than a CatalogDP since chemistry is a one-shot choice.
    F = {"storage_kwh": Reals(unit="kWh")}
    R = {"cost_usd": Reals(unit="USD"), "mass_kg": Reals(unit="kg"),
         "replacements": Reals()}
    CHEMISTRIES = {
        # name: (Wh/kg, USD/kWh, cycle life)
        "LFP":   (160.0, 130.0, 4000.0),
        "NMC":   (240.0, 175.0, 2000.0),
        "LCO":   (220.0, 180.0,  800.0),
        "NaIon": (110.0,  90.0, 3000.0),
    }
    def __init__(self, chemistry="LFP", daily_cycles=1.0, life_years=10.0):
        wh_kg, usd_kwh, life = self.CHEMISTRIES[chemistry]
        self.chemistry = chemistry
        self.specific_energy = wh_kg
        self.cost_density = usd_kwh
        self.cycle_life = life
        self.daily_cycles = daily_cycles
        self.life_years = life_years
        super().__init__()

    def h(self, f):
        kwh = f["storage_kwh"]
        # Number of replacements expected over the mission life.
        reps = (self.daily_cycles * 365.0 * self.life_years) / max(self.cycle_life, 1.0)
        return {"cost_usd": kwh * self.cost_density * (1.0 + reps),
                "mass_kg":  kwh * 1000.0 / max(self.specific_energy, 1e-6),
                "replacements": reps}


class DieselGenerator(Module):
    # Capital cost scales with kW capacity; fuel cost scales with kWh used
    # per month. CO2 emission likewise tracks kWh-per-month consumption.
    F = {"backup_power_kw": Reals(unit="kW"),
         "backup_hours":    Reals(unit="h")}
    R = {"cost_usd": Reals(unit="USD"), "mass_kg": Reals(unit="kg"),
         "co2_kg":   Reals(unit="kg")}
    def __init__(self, cost_per_kw=500.0, mass_per_kw=40.0,
                 fuel_cost_per_kwh=0.35, co2_per_kwh=0.95):
        self.cost_per_kw = cost_per_kw
        self.mass_per_kw = mass_per_kw
        self.fuel_cost_per_kwh = fuel_cost_per_kwh
        self.co2_per_kwh = co2_per_kwh
        super().__init__()
    def h(self, f):
        kw = f["backup_power_kw"]
        hrs = f["backup_hours"]
        kwh = kw * hrs
        capital = kw * self.cost_per_kw
        fuel = kwh * self.fuel_cost_per_kwh * 30.0  # 30 days/month
        return {"cost_usd": capital + fuel,
                "mass_kg":  kw * self.mass_per_kw,
                "co2_kg":   kwh * self.co2_per_kwh * 30.0}


class Frame(Module):
    # The frame mass is 18% of whatever it must support. Because the frame
    # supports itself too, this creates a self-referential constraint that
    # the Kleene iteration resolves.
    F = {"supported_mass_kg": Reals(unit="kg")}
    R = {"cost_usd": Reals(unit="USD"), "mass_kg": Reals(unit="kg")}
    def h(self, f):
        m = f["supported_mass_kg"] * 0.18
        return {"cost_usd": m * 6.0, "mass_kg": m}"""),

    ("md", """## Building the system

The frame's mass depends on the total supported mass *including the frame itself*. The Kleene iteration handles this cycle automatically.
"""),
    ("code", """def make_microgrid(*, chemistry="LFP", sun_hours_per_day=3.0,
                   solar_fraction=0.85, uncertainty=False):
    solar = SolarArray(sun_hours_per_day=sun_hours_per_day)
    if uncertainty:
        solar.uncertain_dist = Stochastic(
            marginals={"sun_hours_per_day":
                stats.truncnorm(a=-1.5, b=1.5,
                                loc=sun_hours_per_day, scale=1.0)},
        )
    sys = System(f"microgrid_{chemistry}")
    daily_load = sys.provides("daily_load_kwh", unit="kWh")
    peak_load  = sys.provides("peak_load_kw",   unit="kW")
    backup_h   = sys.provides("backup_hours",   unit="h")
    total_cost = sys.requires("total_cost_usd", unit="USD")
    total_mass = sys.requires("total_mass_kg",  unit="kg")
    annual_co2 = sys.requires("annual_co2_kg",  unit="kg/yr")
    s = sys.add("solar",   solar)
    b = sys.add("battery", Battery(chemistry=chemistry))
    d = sys.add("diesel",  DieselGenerator())
    fr = sys.add("frame",  Frame())
    s.daily_energy_kwh >= daily_load * solar_fraction
    s.peak_power_kw    >= peak_load
    b.storage_kwh      >= daily_load * solar_fraction
    d.backup_power_kw  >= peak_load
    d.backup_hours     >= backup_h
    fr.supported_mass_kg >= s.mass_kg + b.mass_kg + d.mass_kg + fr.mass_kg
    total_cost >= s.cost_usd + b.cost_usd + d.cost_usd + fr.cost_usd
    total_mass >= s.mass_kg + b.mass_kg + d.mass_kg + fr.mass_kg
    annual_co2 >= d.co2_kg * (365.0 / 30.0)
    return sys.build()


mission = {"daily_load_kwh": 15.0, "peak_load_kw": 3.0, "backup_hours": 12.0}"""),

    ("md", "## Compare the four chemistries"),
    ("code", """for chem in ["LFP", "NMC", "LCO", "NaIon"]:
    r = solve(make_microgrid(chemistry=chem), mission, max_iter=400)
    p = list(r.antichain.points)[0]
    print(f"{chem:<6} cost=${p['total_cost_usd']:>8.0f}  "
          f"mass={p['total_mass_kg']:>6.1f} kg  "
          f"CO2={p['annual_co2_kg']:>6.1f} kg/yr  "
          f"(iters={r.iterations})")"""),
    ("md", """A clear tradeoff: NaIon is cheapest but heaviest, LFP is the all-rounder, LCO is the priciest (short life means it pays for replacements). The frame's cyclic dependence forces the iteration to do real work; iteration counts in the 20-25 range are typical."""),

    ("md", """## Convergence trace with `viz.plot_convergence`

One call to render the delta-vs-iteration semilog.
"""),
    ("code", """import matplotlib.pyplot as plt

dp = make_microgrid(chemistry="LFP")
r = solve(dp, mission, max_iter=400, trace=True)
ax = viz.plot_convergence(r)
ax.set_title(f"Microgrid (LFP): converged in {r.iterations} Kleene steps")
plt.tight_layout()
plt.show()"""),

    ("md", """## Warm-started parameter sweep

Sweeping `daily_load_kwh` from 5 to 30 kWh in 50 steps. Warm-starting each solve from the previous fixed point saves Kleene iterations: the answer at 15.5 kWh is a good seed for 16.0 kWh.
"""),
    ("code", """dp = make_microgrid(chemistry="LFP")
loads = np.linspace(5.0, 30.0, 50)

# Cold sweep: each solve starts fresh from the bottom antichain.
cold_iters = 0
for L in loads:
    f = {"daily_load_kwh": float(L), "peak_load_kw": 3.0, "backup_hours": 12.0}
    r = solve(dp, f, max_iter=400)
    cold_iters += r.iterations

# Warm sweep: each solve reuses the previous fixed point as its seed.
# Since adjacent parameter values usually have nearby fixed points, the
# Kleene iteration finishes in fewer steps.
warm_iters = 0
prev = None
for L in loads:
    f = {"daily_load_kwh": float(L), "peak_load_kw": 3.0, "backup_hours": 12.0}
    r = solve(dp, f, max_iter=400, start_from=prev)
    warm_iters += r.iterations
    prev = r            # feed this result into the next iteration

print(f"cold total iters: {cold_iters}")
print(f"warm total iters: {warm_iters}")
print(f"speedup: {cold_iters / max(warm_iters, 1):.2f}x")"""),

    ("md", """## Stochastic sun hours

The sun is treated as a truncated-normal random variable around 3 hours/day. A Monte-Carlo solve over 200 samples gives the cost distribution, plus mean, p95, CVaR95.
"""),
    ("code", """dp_u = make_microgrid(chemistry="LFP", uncertainty=True)
res_u = solve(dp_u, mission,
              uncertainty=["mean", "p95", "cvar95", "samples"],
              n_samples=200, rng_seed=42, max_iter=400)
nominal = solve(make_microgrid(chemistry="LFP"), mission, max_iter=400)
nominal_cost = list(nominal.antichain.points)[0]["total_cost_usd"]
print(f"Nominal cost: ${nominal_cost:.0f}")
print(f"Mean cost (MC): ${res_u.mean['total_cost_usd']:.0f}")
print(f"p95 cost: ${res_u.p95['total_cost_usd']:.0f}")
print(f"CVaR95 cost: ${res_u.cvar95['total_cost_usd']:.0f}")
print(f"Feasibility rate: {res_u.feasibility_rate:.3f}")"""),

    ("md", """## Distribution plot with `viz.plot_uncertainty`

Histogram of cost across MC samples, with summary lines.
"""),
    ("code", """ax = viz.plot_uncertainty(res_u, port="total_cost_usd",
                          nominal=nominal_cost, bins=25)
ax.set_xlabel("total cost (USD)")
ax.set_title("Cost distribution under stochastic sun hours")
plt.tight_layout()
plt.show()"""),

    ("md", """## System structure as a graph

`viz.to_dot` returns a GraphViz dot string showing modules and the constraints linking them. Useful for documentation, less so for solving.
"""),
    ("code", """dot = viz.to_dot(dp, name="microgrid")
print(dot[:500] + "..." if len(dot) > 500 else dot)"""),

    ("md", """## What this notebook used

| Feature | Where |
|---|---|
| `Module` classes | four subsystems |
| Cyclic constraints | frame depends on its own mass |
| Warm-start (`start_from=`) | parameter sweep, ~10% faster |
| `Stochastic` + MC summaries | sun-hours uncertainty |
| `viz.plot_convergence` | delta-vs-iteration |
| `viz.plot_uncertainty` | MC cost distribution |
| `viz.to_dot` | system structure as GraphViz dot |

The microgrid is a realistic engineering scenario where MCDP's monotone structure pays off: real cycles, multiple objectives, and uncertainty are handled by the same `solve` call.
"""),
]


# ---------------------------------------------------------------------------
# 14 Online elimination: heterogeneous robot fleet
# ---------------------------------------------------------------------------

NB_14 = [
    ("md", """# 14. Online elimination-based co-design

When a co-design problem has many discrete candidates (catalog entries, robot types, component families) and each candidate's inner solve is non-trivial, evaluating every one is wasteful. The `codesign.online` module implements the elimination-based solver from Alharbi, Dahleh & Zardini (2026): maintain *optimistic bounds* on each candidate's inner-solve output, evaluate the most promising one, then prune any candidate whose lower bound is already dominated by what we know.

This notebook reproduces the spirit of the multi-robot fleet case study. A logistics service must hit a target throughput over a target range, and can buy robots from a catalog of N=200 candidate types. We solve it three ways:

- **Lipschitz**: the most general assumption (bounded variation), the most reliable, modest pruning.
- **Monotonicity**: cheapest when applicable; needs a feature under which the output is genuinely monotone.
- **LinearParametric**: now the *certified* confidence-polytope bound of the paper (Sec. V-C3). It is guaranteed never to prune a Pareto-optimal candidate; it is exact and aggressive only when the resource map is genuinely affine in the features.
"""),

    ("md", "## The problem"),
    ("code", """import math, random
from codesign import (
    AlgebraicDP, Reals, Ports, solve,
    solve_online, LipschitzEvaluator,
    MonotonicityEvaluator, LinearParametricEvaluator,
)

# Mission spec: deliver `target_throughput` packages per hour over
# `target_range` km of daily travel.
F = Ports({"target_throughput": Reals(unit="pkg/h"),
                  "target_range":      Reals(unit="km")})
# What the fleet costs: dollars to acquire + kWh to operate.
R = Ports({"total_cost":   Reals(unit="USD"),
                  "total_energy": Reals(unit="kWh/day")})

def make_dp(robot):
    # Closed-form inner solve per robot type. capacity = speed * payload
    # is packages/hour per single robot; total_cost is the fleet size
    # times unit_cost; total_energy is range times per-km energy times
    # 24 hours of duty cycle.
    s = robot["speed"]; p = robot["payload"]
    c = robot["unit_cost"]; e = robot["energy_per_km"]
    capacity = s * p  # pkg/h per robot
    # Default-argument trick captures the loop-variable values at lambda
    # creation time, so each AlgebraicDP closes over its own robot's specs.
    return AlgebraicDP(F, R, {
        "total_cost":   lambda f, cap=capacity, uc=c: (f["target_throughput"]/cap) * uc,
        "total_energy": lambda f, ek=e: f["target_range"] * ek * 24.0,
    })"""),

    ("md", "## The catalog (200 robot types)"),
    ("code", """def make_catalog(n=200, seed=42):
    # Random robot specs in physically plausible ranges. Seeded for
    # reproducibility across notebook runs.
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = rng.uniform(5, 30); p = rng.uniform(1, 20)
        c = rng.uniform(500, 5000); e = rng.uniform(0.05, 0.5)
        out.append({"name": f"r{i:03d}",
                    "speed": s, "payload": p,
                    "unit_cost": c, "energy_per_km": e,
                    # Derived feature: total_cost is exactly proportional
                    # to this scalar (T * cost_per_capacity), so it's a
                    # genuinely monotone feature the MonotonicityEvaluator
                    # can exploit aggressively.
                    "cost_per_capacity": c / (s * p)})
    return out

candidates = make_catalog()
mission = {"target_throughput": 100.0, "target_range": 50.0}
print(f"{len(candidates)} candidate robot types")"""),

    ("md", """## Exhaustive baseline

Every catalog entry gets solved; we record the true Pareto front for reference.
"""),
    ("code", """# Run solve() on every candidate. With 200 entries this is the cost
# we want the online learner to reduce.
points = []
for c in candidates:
    a = solve(make_dp(c), mission).antichain
    pt = dict(list(a.points)[0]); pt["name"] = c["name"]
    points.append(pt)

# Compute the true Pareto front by brute force: a point is Pareto-optimal
# if no other point dominates it (i.e., is <= in every R component and
# strictly < in at least one).
pareto = []
for p in points:
    dominated = any(
        q["total_cost"] <= p["total_cost"] and q["total_energy"] <= p["total_energy"]
        and (q["total_cost"] < p["total_cost"] or q["total_energy"] < p["total_energy"])
        for q in points
    )
    if not dominated:
        pareto.append(p)
pareto.sort(key=lambda x: x["total_cost"])
print(f"True Pareto front: {len(pareto)} non-dominated robot types")
for p in pareto:
    print(f"  {p['name']}: cost={p['total_cost']:7.2f}, energy={p['total_energy']:6.2f}")"""),

    ("md", """## Online with three evaluators

Each evaluator encodes a different prior belief about how candidate features relate to the inner-solve output.
"""),
    ("code", """# Three evaluators with different structural assumptions.
evs = [
    # Lipschitz: bounded variation. Safest default; L tunes the trade-off
    # between pruning aggressiveness and risk of missing a Pareto point.
    ("Lipschitz", LipschitzEvaluator(
        features=["speed", "payload", "unit_cost", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
        L={"total_cost": 300.0, "total_energy": 30.0},
    )),
    # Monotonicity on the derived feature: total_cost is exactly
    # proportional to cost_per_capacity, so one observation prunes
    # everything strictly worse in features.
    ("Monotonicity", MonotonicityEvaluator(
        features=["cost_per_capacity", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
    )),
    # LinearParametric: the certified confidence-polytope bound of the
    # paper (Sec. V-C3). It maintains the polytope of linear parameter
    # vectors consistent with every observation and lower-bounds each
    # query by one LP over that polytope, so the bound is guaranteed and
    # it never wrongly eliminates a Pareto point. min_obs=5 means it only
    # starts bounding after five evaluations.
    ("LinearParametric (certified)", LinearParametricEvaluator(
        features=["speed", "payload", "unit_cost", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
        min_obs=5,
    )),
]
results = []
for name, ev in evs:
    # solve_online prunes candidates by bound, then evaluates the most
    # promising survivor via the standard solver. Loop until exhausted
    # or budget hit (unset here, so unbounded).
    res = solve_online(make_dp, mission, candidates=candidates, evaluator=ev)
    results.append((name, res))
    print(f"{name:<18}: {res.n_evaluated:>3} evals, {res.n_eliminated:>3} eliminated, "
          f"antichain size {len(res.antichain)}")"""),

    ("md", """## Visualising the elimination cascade

Plotting the catalog in feature space, coloured by status. Stars mark the true Pareto front.
"""),
    ("code", """import matplotlib.pyplot as plt

pareto_names = {p["name"] for p in pareto}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (label, res) in zip(axes, results):
    evaluated  = {candidates[i]["name"] for i in res.evaluated_ids}
    eliminated = {candidates[i]["name"] for i in res.eliminated_ids}
    xe, ye, xx, yx, xp, yp = [], [], [], [], [], []
    for c in candidates:
        x, y = c["cost_per_capacity"], c["energy_per_km"]
        if c["name"] in pareto_names: xp.append(x); yp.append(y)
        elif c["name"] in evaluated: xe.append(x); ye.append(y)
        elif c["name"] in eliminated: xx.append(x); yx.append(y)
    ax.scatter(xx, yx, c="lightgrey", s=18, label=f"eliminated ({len(eliminated)})", edgecolor="none")
    ax.scatter(xe, ye, c="steelblue", s=22, label=f"evaluated ({len(evaluated) - len(pareto_names & evaluated)})", edgecolor="none")
    ax.scatter(xp, yp, c="crimson", s=55, marker="*", label=f"Pareto-optimal ({len(pareto_names)})", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("cost_per_capacity")
    ax.set_ylabel("energy_per_km")
    ax.set_title(label)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Online elimination across 200 candidates")
fig.tight_layout()
plt.show()"""),

    ("md", """## Reading the picture

- **Lipschitz** is conservative: with the L values we picked, almost every candidate gets evaluated. The pruning rate is modest because the Lipschitz bound only tightens by `L * distance` around each observation, so distant candidates remain plausible.
- **Monotonicity** is aggressive: because `cost_per_capacity` is genuinely monotone-related to `total_cost` (it's literally proportional), one observation eliminates everything with strictly worse features. Most of the catalog gets ruled out within a dozen evaluations.
- **LinearParametric** is now the *certified* bound: it evaluates 122/200 and provably recovers all five Pareto points. Because it maintains the whole polytope of linear parameter vectors consistent with the observations (rather than a single OLS fit with a confidence band), its lower bound can never drop a Pareto-optimal candidate. It prunes less than the local bounds here because `total_cost` is not exactly affine in the raw features; it becomes exact and aggressive only when the map truly is linear. (The former OLS +/- 3-sigma heuristic pruned harder -- about 35/200 -- but wrongly dropped one Pareto point on this seed.)

The choice between evaluators is a classical bias-variance tradeoff: stronger structural assumptions mean fewer evaluations needed. With the certified evaluators none of them can wrongly prune -- Lipschitz (with a valid L), Monotonicity (on a genuinely monotone feature), and the certified LinearParametric (on a genuinely affine map) all carry a guarantee. The remaining question is purely how much pruning you get: Monotonicity is the workhorse when you have a monotone feature, Lipschitz is the safe general default, and certified LinearParametric is exact and aggressive precisely when the resource map is linear in the features.

## Where this matters

This example uses a smooth analytic inner solve, so each evaluation is cheap and the speedup is mostly pedagogical. The real payoff appears when each inner solve is itself expensive: a multi-stage MILP, a system identification, or a co-design problem with cycles. With a 1000-entry catalog and a 100ms inner solve, going from 1000 to 50 evaluations is the difference between 100 seconds and 5.
"""),
]


# ---------------------------------------------------------------------------
# 15 mAb fed-batch bioprocess co-design
# ---------------------------------------------------------------------------

NB_15 = [
    ("md", """# 15. Monoclonal antibody fed-batch co-design

A biopharmaceutical company has to deliver a monoclonal antibody at a target titer (g/L) and annual demand (kg/year). The design choices coupled cyclically are:

- **cell line**: which CHO clone to use (productivity, batch length, oxygen demand)
- **media**: which commercial formulation (cost per litre, max supported cell density)
- **bioreactor**: which format and size (single-use vs stainless steel, kLa cap)
- **feed strategy**: how aggressively to feed glucose (titer vs metabolic waste, batch failure risk)

The cycles are real biology: higher titer needs higher cell density, which needs higher kLa (more capable bioreactor), which the media must support. Richer feed produces more lactate and ammonia, which inflates the cell density required to deliver the same titer (closing the loop). The Kleene iteration converges this automatically.

All parameters are taken from the bioprocessing literature: cell-line specific productivity (Reinhart 2019), oxygen uptake rates (BioProcess International 2024), media costs (CHO media market 2025), bioreactor capex (Sustainability Atlas 2026, BioPlan 2025), metabolic constraints (Khattak 2010, Lao & Toth 1997).
"""),
    ("md", "## Imports"),
    ("code", """import math
from codesign import (
    CatalogDP, CatalogEntry, Module, Ports, Reals,
    System, solve, viz,
)
import matplotlib.pyplot as plt"""),
    ("md", """## Catalogue parameters

Each cell line is characterised by an *effective integrated* specific productivity (calibrated so titer = qP * avg_VCD * batch_days / 1000 gives realistic numbers at literature-reported peak VCDs of 10-30 million cells/mL).

The media list spans HyClone-CD (workhorse standard), EX-CELL-CD-CHO and Cellvento-CHO-220 (Sigma/Merck modern CD), and BalanCD-HIP (high-intensity perfusion grade). The bioreactor list spans single-use 200L through stainless steel 25,000L, with kLa caps mapped to maximum supportable peak VCD via the rule "kLa=1 supports about 4 million cells/mL with pure O2 sparging."
"""),
    ("code", """# (name, qP_eff [pg/cell/day], qO2 [1e-10 mmol/cell/h], batch_days, license [USD/batch])
CELL_LINES = [
    ("CHO-S",     15.0, 5.0, 14.0,    500.0),   # biomass-favouring legacy
    ("CHO-DG44",  18.0, 6.0, 14.0,   1500.0),   # DHFR-amplified, mid-tier
    ("CHO-K1",    50.0, 7.0, 12.0,   5000.0),   # high-producer workhorse
    ("CHO-MK",   120.0, 7.5,  8.0,  25000.0),   # next-gen, short batch
]

# (name, cost [USD/L], max_vcd_supported [1e6 cells/mL])
MEDIA_OPTIONS = [
    ("HyClone-CD",         80.0, 15.0),
    ("EX-CELL-CD-CHO",    110.0, 25.0),
    ("Cellvento-CHO-220", 140.0, 35.0),
    ("BalanCD-HIP",       250.0, 80.0),    # premium HIP grade
]

# (name, working_vol [L], max_peak_vcd [1e6 cells/mL], capex [USD/batch],
#  footprint [m^2], co2_per_batch [kg])
BIOREACTORS = [
    ("SU-200",     200.0,  40.0,   3_000.0,  2.0,  150.0),
    ("SU-2000",   2000.0,  60.0,  20_000.0,  8.0, 1200.0),    # industry sweet spot
    ("SS-5000",   5000.0,  72.0,  35_000.0, 12.0, 2200.0),
    ("SS-12500", 12500.0,  80.0,  60_000.0, 22.0, 4800.0),
    ("SS-25000", 25000.0,  90.0, 100_000.0, 35.0, 8500.0),
]"""),
    ("md", """## The four subsystems

Each is a small Module or CatalogDP. The cell line maps a titer demand to a required cell density and oxygen demand; the bioreactor and media catalogues pick the smallest entry satisfying the (metabolically-inflated) cell-density demand; the feed strategy emits a U-shaped COGS multiplier capturing batch failure risk.
"""),
    ("code", """class CellLine(Module):
    # See the example's docstring for the unit derivation. With qP in
    # pg/cell/day and avg VCD in 1e6 cells/mL, titer = qP * avg_VCD *
    # batch_days * 1e-3 g/L, so avg_VCD = titer * 1e3 / (qP * batch_days).
    F = {"target_titer": Reals(unit="g/L")}
    R = {
        "avg_vcd":           Reals(unit="1e6 cells/mL"),
        "peak_vcd":          Reals(unit="1e6 cells/mL"),
        "oxygen_demand":     Reals(unit="mmol/L/h"),
        "batch_days":        Reals(unit="day"),
        "license_per_batch": Reals(unit="USD"),
    }

    def __init__(self, name, qp, qo2, batch_days, license_fee, peak_to_avg=2.0):
        self.cell_name = name
        self.qp = qp                 # effective integrated qP, pg/cell/day
        self.qo2 = qo2               # 1e-10 mmol/cell/h
        self.batch_days = batch_days
        self.license_fee = license_fee
        self.peak_to_avg = peak_to_avg
        super().__init__()

    def h(self, f):
        titer = f["target_titer"]
        avg_vcd  = titer * 1e3 / (self.qp * self.batch_days)
        peak_vcd = self.peak_to_avg * avg_vcd
        # OUR_peak = peak_VCD [1e6 cells/mL] * 1e9 cells/L * qO2 * 1e-10 mmol/cell/h
        #          = peak_VCD_million * qO2 * 0.1
        oxygen = peak_vcd * self.qo2 * 0.1
        return {
            "avg_vcd": avg_vcd, "peak_vcd": peak_vcd,
            "oxygen_demand": oxygen,
            "batch_days": float(self.batch_days),
            "license_per_batch": float(self.license_fee),
        }"""),
    ("code", """def make_bioreactor_dp():
    # CatalogDP that picks the smallest bioreactor supporting the
    # demanded peak VCD. Costs are per-batch; the working volume is
    # an output used later to compute COGS per gram.
    F = Ports({"peak_vcd": Reals(unit="1e6 cells/mL")})
    R = Ports({
        "working_volume":  Reals(unit="L"),
        "capex_per_batch": Reals(unit="USD"),
        "footprint_m2":    Reals(unit="m^2"),
        "co2_per_batch":   Reals(unit="kg"),
    })
    entries = [
        CatalogEntry(
            name=name,
            provides={"peak_vcd": max_vcd},
            costs={"working_volume": vol, "capex_per_batch": capex,
                   "footprint_m2": fp, "co2_per_batch": co2},
        )
        for (name, vol, max_vcd, capex, fp, co2) in BIOREACTORS
    ]
    return CatalogDP(F=F, R=R, catalog=entries, name="bioreactors")


def make_media_dp():
    # CatalogDP that picks the cheapest media supporting the peak VCD.
    F = Ports({"peak_vcd": Reals(unit="1e6 cells/mL")})
    R = Ports({"media_cost_per_l": Reals(unit="USD/L")})
    entries = [
        CatalogEntry(
            name=name,
            provides={"peak_vcd": max_vcd},
            costs={"media_cost_per_l": cost},
        )
        for (name, cost, max_vcd) in MEDIA_OPTIONS
    ]
    return CatalogDP(F=F, R=R, catalog=entries, name="media")


class FeedStrategy(Module):
    # The single design knob is the glucose set-point in mM.
    # Lower (around 5 mM) = HIPDOG-style, minimises lactate/ammonia
    # but harder to control near starvation threshold.
    # Higher (around 12-15 mM) = legacy, easier to control but produces
    # more waste, reducing growth (Khattak 2010).
    # The COGS multiplier captures U-shaped batch-failure risk.
    F = {"peak_vcd": Reals(unit="1e6 cells/mL"),
         "batch_days": Reals(unit="day")}
    R = {"feed_cost_per_l":  Reals(unit="USD/L"),
         "metabolic_factor": Reals(unit="rel"),
         "cogs_multiplier":  Reals(unit="rel")}

    def __init__(self, glucose_setpoint_mm=8.0):
        self.glucose = max(4.0, min(15.0, float(glucose_setpoint_mm)))
        super().__init__()

    def h(self, f):
        glucose_premium = 1.0 + 0.05 * (self.glucose - 5.0)
        feed_cost_per_l = 6.0 * glucose_premium * f["batch_days"] * 0.05
        # Khattak's correlation: ~+30% waste from 3x nutrient.
        waste_factor = 1.0 + 0.03 * (self.glucose - 5.0)
        # U-shape: low glucose risks starvation, high glucose risks waste.
        low_penalty  = 0.06  * max(0.0, 8.0 - self.glucose) ** 1.5
        high_penalty = 0.015 * max(0.0, self.glucose - 8.0) ** 1.5
        return {
            "feed_cost_per_l":  feed_cost_per_l,
            "metabolic_factor": waste_factor,
            "cogs_multiplier":  1.0 + low_penalty + high_penalty,
        }"""),
    ("md", """## Assemble the system

The outer F is the target titer; the outer R is the cost vector (COGS, footprint, CO2). The COGS expression integrates capex, license fee, media, feed cost, and the U-shaped batch-failure multiplier. Footprint depends on the annual demand (more demand needs more parallel reactor lines).
"""),
    ("code", """def make_bioprocess(*, cell_line, glucose_setpoint_mm,
                    annual_demand_kg, turnaround_days=5.0,
                    downstream_yield=0.7):
    sys = System(f"mAb_{cell_line[0]}")
    # Outer F: titer demand at harvest.
    target_titer = sys.provides("target_titer", unit="g/L")
    # Outer R: three-way cost vector for the engineer.
    sys.requires("cogs_per_g",   unit="USD/g")
    sys.requires("footprint_m2", unit="m^2")
    sys.requires("co2_per_g",    unit="kg/g")

    # Subsystems.
    cell  = sys.add("cell",  CellLine(*cell_line))
    feed  = sys.add("feed",  FeedStrategy(glucose_setpoint_mm))
    bior  = sys.add("bior",  make_bioreactor_dp())
    media = sys.add("media", make_media_dp())

    # Wiring: titer demand drives required cell density; feed strategy
    # inflates it by the metabolic factor; bioreactor and media must
    # both support the inflated peak VCD.
    cell.target_titer >= target_titer
    feed.peak_vcd     >= cell.peak_vcd
    feed.batch_days   >= cell.batch_days
    bior.peak_vcd     >= cell.peak_vcd * feed.metabolic_factor
    media.peak_vcd    >= cell.peak_vcd * feed.metabolic_factor

    # Outer R aggregation. We use the dict-based constrain form because
    # several ports must be combined with closed-over parameters
    # (annual_demand_kg, downstream_yield, turnaround_days).
    def cogs_eq(x):
        vol = x["bior.working_volume"]
        mass_per_batch_g = x["target_titer"] * vol * downstream_yield
        cost_per_batch = (x["bior.capex_per_batch"]
                          + x["cell.license_per_batch"]
                          + x["media.media_cost_per_l"] * vol
                          + x["feed.feed_cost_per_l"] * vol)
        # Multiply by feed-strategy COGS multiplier for batch-failure risk.
        return (cost_per_batch / max(mass_per_batch_g, 1e-6)
                * x["feed.cogs_multiplier"])

    def footprint_eq(x):
        # Annual demand / per-batch mass = batches/year needed.
        # Each line delivers 365/(batch+turnaround) batches/year.
        # 3x multiplier for downstream and utilities space.
        vol = x["bior.working_volume"]
        cycle_d = x["cell.batch_days"] + turnaround_days
        max_batches_per_line = 365.0 / cycle_d
        mass_per_batch_g = x["target_titer"] * vol * downstream_yield
        batches_needed = annual_demand_kg * 1000.0 / max(mass_per_batch_g, 1e-6)
        parallel_lines = max(1.0, batches_needed / max_batches_per_line)
        return parallel_lines * x["bior.footprint_m2"] * 3.0

    def co2_eq(x):
        vol = x["bior.working_volume"]
        mass_per_batch_g = x["target_titer"] * vol * downstream_yield
        return x["bior.co2_per_batch"] / max(mass_per_batch_g, 1e-6)

    sys.constrain("cogs_per_g",   cogs_eq)
    sys.constrain("footprint_m2", footprint_eq)
    sys.constrain("co2_per_g",    co2_eq)
    return sys.build()"""),
    ("md", """## Solve for a single scenario

A 100 kg/year mid-stage commercial program at 5 g/L titer with CHO-K1 and 8 mM glucose.
"""),
    ("code", """dp = make_bioprocess(
    cell_line=CELL_LINES[2],            # CHO-K1
    glucose_setpoint_mm=8.0,
    annual_demand_kg=100.0,
)
result = solve(dp, {"target_titer": 5.0}, max_iter=200)
print(f"status: {result.status}, feasible: {result.feasible}, "
      f"iters: {result.iterations}")
for p in result.antichain.points:
    print(f"   COGS      = ${p['cogs_per_g']:.2f}/g")
    print(f"   Footprint = {p['footprint_m2']:.1f} m^2")
    print(f"   CO2       = {p['co2_per_g']*1000:.1f} g CO2/g mAb")"""),
    ("md", """## Sweep across cell lines and feed strategies

For each cell-line / glucose-setpoint combination, solve and collect the (COGS, footprint, CO2) point. Compute the global Pareto front across all 12 design combinations.
"""),
    ("code", """def sweep(target_titer, annual_demand):
    # Every (cell_line, glucose) combination yields one design point.
    results = []
    for cl in CELL_LINES:
        for glu in (5.0, 8.0, 12.0):
            dp = make_bioprocess(cell_line=cl, glucose_setpoint_mm=glu,
                                 annual_demand_kg=annual_demand)
            r = solve(dp, {"target_titer": target_titer}, max_iter=200)
            if not r.feasible:
                continue
            for pt in r.antichain.points:
                if math.isinf(pt["cogs_per_g"]):
                    continue
                results.append({
                    "cell": cl[0], "glu": glu,
                    "label": f"{cl[0]}/glu={glu:.0f}mM",
                    "cogs": pt["cogs_per_g"],
                    "fp": pt["footprint_m2"],
                    "co2": pt["co2_per_g"],
                })
    # Global 3D Pareto front: a point is non-dominated if no other
    # point is <= in all three R components and strictly < in any.
    pareto = []
    for p in results:
        dominated = any(
            q["cogs"] <= p["cogs"] and q["fp"] <= p["fp"] and q["co2"] <= p["co2"]
            and (q["cogs"] < p["cogs"] or q["fp"] < p["fp"] or q["co2"] < p["co2"])
            for q in results
        )
        if not dominated:
            pareto.append(p)
    return results, sorted(pareto, key=lambda x: x["cogs"])

results, pareto = sweep(5.0, 100.0)
print(f"\\n100 kg/yr at 5 g/L: {len(results)} feasible designs, "
      f"{len(pareto)} on the Pareto front:")
for p in pareto:
    print(f"   {p['label']:<25} COGS=${p['cogs']:5.2f}/g  "
          f"footprint={p['fp']:5.1f} m^2  CO2={p['co2']*1000:.1f} g/g")"""),
    ("md", """## Visualise the Pareto front

A 2D scatter in (COGS, footprint) space. All evaluated designs in grey; the Pareto-optimal ones in red. The shape of the front reveals the real engineering tradeoff: shorter-batch high-producer cell lines (CHO-MK) win on footprint but pay more per gram in licence fees; longer-batch standard cell lines (CHO-K1) win on COGS but need more parallel lines for the same annual output.
"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: full design space + Pareto front.
ax = axes[0]
pareto_keys = {(p["cell"], p["glu"]) for p in pareto}
for r in results:
    color = "C3" if (r["cell"], r["glu"]) in pareto_keys else "0.7"
    size = 90 if (r["cell"], r["glu"]) in pareto_keys else 30
    marker = "*" if (r["cell"], r["glu"]) in pareto_keys else "o"
    ax.scatter(r["cogs"], r["fp"], c=color, s=size, marker=marker, zorder=3)
    if (r["cell"], r["glu"]) in pareto_keys:
        ax.annotate(r["label"], (r["cogs"], r["fp"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.set_xlabel("COGS (USD per gram)")
ax.set_ylabel("Facility footprint (m^2)")
ax.set_title("100 kg/yr commercial program, 5 g/L titer")
ax.grid(True, linestyle=":", alpha=0.5)

# Right panel: Pareto fronts for three mission scales.
ax = axes[1]
for label, titer, demand, marker in [
    ("Clinical 10 kg/yr",   3.0,  10.0, "o"),
    ("Mid 100 kg/yr",       5.0, 100.0, "s"),
    ("Large 500 kg/yr",     8.0, 500.0, "^"),
]:
    _, pf = sweep(titer, demand)
    xs = [p["cogs"] for p in pf]
    ys = [p["fp"] for p in pf]
    ax.plot(xs, ys, marker=marker, label=label, markersize=10, linewidth=1.5)
ax.set_xlabel("COGS (USD per gram)")
ax.set_ylabel("Facility footprint (m^2)")
ax.set_yscale("log")
ax.set_title("Pareto fronts across three mission scales")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)

fig.tight_layout()
plt.show()"""),
    ("md", """## What the framework just did

The Kleene iteration resolved the cyclic constraints automatically. The cycle that matters:

1. The CellLine module sees the titer demand and emits a peak VCD.
2. FeedStrategy reads that VCD and emits a metabolic factor.
3. The Bioreactor catalogue must support `peak_vcd * metabolic_factor`, which is larger than the bare cell-line demand.
4. Inside the catalogue lookup, the smallest sufficient bioreactor is chosen, with its own kLa cap acting as a feasibility wall.

The Pareto front is genuinely two-point in every scenario. CHO-K1 with 12-day batches is cheap per gram but needs more parallel lines for high annual demand, occupying more floor space. CHO-MK with 8-day batches has expensive licence fees per batch but turns over twice as often, so the same annual output fits in fewer parallel lines and less footprint. CHO-S and CHO-DG44 are dominated everywhere because their lower productivity demands more cells and more bioreactor volume, raising both COGS and footprint together.

The framework would extend in several useful directions: adding a perfusion mode subsystem (essentially a Loop with much higher peak VCD but continuous media exchange), modelling product quality attributes (glycosylation profile, aggregation rate) as additional R components, adding regulatory uncertainty as a UncertainDP wrapper, or running the same problem with `solve_online` to design an experimental campaign that finds the Pareto front in 20 to 30 bench runs rather than 200 to 500.
"""),
]


# ---------------------------------------------------------------------------
# 16 online DOE for the mAb fed-batch process
# ---------------------------------------------------------------------------

NB_16 = [
    ("md", """# 16. Online Design of Experiments for the mAb fed-batch process

Example 15 chose between four cell lines and three glucose set-points. Real bioprocess development never sees that small a space. A typical Phase II / III scale-up campaign for a monoclonal antibody asks roughly the opposite question: the cell line and media are fixed by previous decisions, and what remains is to find the best operating point in a 4D (or higher) space of process parameters.

Here we fix CHO-K1 and the 100 kg/year mission from example 15 and sweep over a $5 \\times 5 \\times 5 \\times 3 = 375$-point grid of operating conditions: temperature, pH, glucose target, feed start day. In a real campaign, each candidate is a 10 to 14-day bioreactor run costing $20,000 to $100,000 in materials, labour, and analytics, so running all 375 is unthinkable. Process scientists use factorial DOE designs (typically 30 to 100 runs) instead.

This notebook shows that the same elimination-based online solver from example 14 transfers directly to this setting: with a properly chosen evaluator, the Pareto front is recovered from 40 simulated runs (11% of the grid) at the same quality as a 75-run factorial DOE.
"""),
    ("md", "## Imports"),
    ("code", """import math
import random
from codesign import (
    AlgebraicDP, LinearParametricEvaluator, LipschitzEvaluator,
    MonotonicityEvaluator, Ports, Reals, solve, solve_online,
)
import matplotlib.pyplot as plt
import numpy as np"""),
    ("md", """## Mission and bioprocess parameters

The mission and bioreactor / media catalogues are inherited from example 15 (see that example for the literature calibration). The key difference: instead of CHO-K1 having a fixed `qP_eff = 50` pg/cell/day, the effective qP now depends on the operating conditions through a closed-form effect model.
"""),
    ("code", """TARGET_TITER_G_L  = 5.0
ANNUAL_DEMAND_KG  = 100.0
TURNAROUND_DAYS   = 5.0
DOWNSTREAM_YIELD  = 0.7

QP_BASE           = 35.0     # pg/cell/day (slightly lower than ex 15)
BATCH_DAYS_BASE   = 12.0
LICENSE_PER_BATCH = 5_000.0

BIOREACTORS = [
    ("SU-200",     200.0,  40.0,   3_000.0,  2.0,  150.0),
    ("SU-2000",   2000.0,  60.0,  20_000.0,  8.0, 1200.0),
    ("SS-5000",   5000.0,  72.0,  35_000.0, 12.0, 2200.0),
    ("SS-12500", 12500.0,  80.0,  60_000.0, 22.0, 4800.0),
    ("SS-25000", 25000.0,  90.0, 100_000.0, 35.0, 8500.0),
]
MEDIA = [
    ("HyClone-CD",         80.0, 15.0),
    ("EX-CELL-CD-CHO",    110.0, 25.0),
    ("Cellvento-CHO-220", 140.0, 35.0),
    ("BalanCD-HIP",       250.0, 80.0),
]"""),
    ("md", """## The effect model

Each candidate operating condition $(T, \\text{pH}, [\\text{glucose}], d_\\text{feed})$ runs through a closed-form effect model that yields one `(cogs, footprint, co2)` outcome. The four effects are calibrated to the bioprocess literature:

- **Temperature shift**: production-phase cold shift from 37 C to 32 to 34 C is a well-established productivity booster. qP rises by about 6% per degree of cold shift; growth rate drops by 8% per degree; batch length stretches by $1/\\mu$. (Yoon et al. 2003, Sou et al. 2015.)

- **pH set-point**: a U-shape around 7.05. Off-target pH shifts metabolism toward lactate accumulation. (Trummer et al. 2006.)

- **Glucose target**: low values (around 5 mM) follow HIPDOG-style efficient metabolism with high batch failure risk; high values (around 13 mM) accumulate waste. Failure rate is U-shaped around 8 mM. (Khattak et al. 2010, Gagnon et al. 2011.)

- **Feed start day**: earlier feed start delivers more integrated nutrients and supports higher peak VCD; later defers and reduces achievable peak.
"""),
    ("code", """def simulate_run(T_C, pH, glucose_mm, feed_start_day):
    # Temperature shift
    cold_shift = max(0.0, 37.0 - T_C)
    qp_factor = 1.0 + 0.06 * cold_shift
    mu_factor = max(0.5, 1.0 - 0.08 * cold_shift)
    batch_length_eff = BATCH_DAYS_BASE / mu_factor

    # pH penalty (stronger U-shape than in example 15)
    pH_distance = abs(pH - 7.05)
    pH_penalty = 1.0 + 2.5 * pH_distance

    # Glucose burden + failure rate
    waste_factor = 1.0 + 0.05 * max(0.0, glucose_mm - 5.0)
    low_glu_failure  = 0.20 * max(0.0, 8.0 - glucose_mm) ** 1.5
    high_glu_failure = 0.04 * max(0.0, glucose_mm - 8.0) ** 1.5
    failure_factor   = 1.0 + low_glu_failure + high_glu_failure

    # Feed start day
    feed_factor = 1.0 + 0.04 * (3.0 - feed_start_day)

    qp_eff = QP_BASE * qp_factor * feed_factor / pH_penalty
    avg_vcd = TARGET_TITER_G_L * 1e3 / (qp_eff * batch_length_eff)
    peak_vcd_eff = 2.0 * avg_vcd * waste_factor

    # Smallest bioreactor and media that support the demanded peak VCD.
    bior = next(((n, v, capex, fp, co2)
                 for (n, v, mv, capex, fp, co2) in BIOREACTORS
                 if mv >= peak_vcd_eff), None)
    media = next(((n, c) for (n, c, mv) in MEDIA if mv >= peak_vcd_eff), None)
    if bior is None or media is None:
        return dict(cogs_per_g=math.inf, footprint_m2=math.inf,
                    co2_per_g=math.inf)

    _, vol, capex, fp_per_batch, co2_per_batch = bior
    _, media_cost_l = media
    feed_cost_per_l = (6.0 * (1.0 + 0.05 * max(0.0, glucose_mm - 5.0))
                       * batch_length_eff * 0.05)
    mass_g = TARGET_TITER_G_L * vol * DOWNSTREAM_YIELD
    per_batch = capex + LICENSE_PER_BATCH + media_cost_l * vol + feed_cost_per_l * vol
    cogs = (per_batch / mass_g) * failure_factor
    cycle_d = batch_length_eff + TURNAROUND_DAYS
    batches_per_line = 365.0 / cycle_d
    needed = ANNUAL_DEMAND_KG * 1000.0 / mass_g
    parallel = max(1.0, needed / batches_per_line)
    footprint = parallel * fp_per_batch * 3.0
    return dict(cogs_per_g=cogs, footprint_m2=footprint,
                co2_per_g=co2_per_batch / mass_g)"""),
    ("md", """## Build the candidate grid

Each candidate carries both raw features (the 4D condition vector) and derived features. The normalised features (each scaled to roughly [0, 1] over the grid) are needed by the Lipschitz evaluator because its Euclidean metric otherwise gets dominated by whichever raw feature has the largest span (glucose, 5 to 13 mM). The monotone-bad features (`pH_distance`, `glucose_extremity`, `feed_delay`) are needed for the Monotonicity evaluator.
"""),
    ("code", """def make_grid():
    cands = []
    for T_C in (33, 34, 35, 36, 37):
        for pH in (6.9, 7.0, 7.1, 7.2, 7.3):
            for glu in (5, 7, 9, 11, 13):
                for feed_d in (2, 3, 4):
                    cands.append({
                        "T_C": float(T_C), "pH": float(pH),
                        "glucose_mm": float(glu),
                        "feed_start_day": float(feed_d),
                        # Normalised features for Lipschitz Euclidean metric.
                        "T_norm":   (37.0 - T_C) / 4.0,
                        "pH_norm":  abs(pH - 7.05) / 0.25,
                        "glu_norm": abs(glu - 8.0) / 5.0,
                        "feed_norm": (feed_d - 2.0) / 2.0,
                        # Monotone-bad features for Monotonicity.
                        "pH_distance":       abs(pH - 7.05),
                        "glucose_extremity": abs(glu - 8.0),
                        "feed_delay":        max(0.0, feed_d - 2.0),
                    })
    return cands

candidates = make_grid()
print(f"DOE grid: {len(candidates)} candidate operating conditions")"""),
    ("md", """## Wrap each candidate as an AlgebraicDP

Each candidate is run through `simulate_run` eagerly, producing a `(cogs, footprint, co2)` triple. The `AlgebraicDP` wraps these three constants as a standard DP so that the codesign solver can produce an antichain from the inner solve.
"""),
    ("code", """F_OUTER = Ports({"target_titer": Reals(unit="g/L")})
R_OUTER = Ports({
    "cogs_per_g":   Reals(unit="USD/g"),
    "footprint_m2": Reals(unit="m^2"),
    "co2_per_g":    Reals(unit="kg/g"),
})

def make_dp(candidate):
    out = simulate_run(candidate["T_C"], candidate["pH"],
                       candidate["glucose_mm"], candidate["feed_start_day"])
    return AlgebraicDP(F_OUTER, R_OUTER, {
        "cogs_per_g":   lambda f, v=out["cogs_per_g"]: v,
        "footprint_m2": lambda f, v=out["footprint_m2"]: v,
        "co2_per_g":    lambda f, v=out["co2_per_g"]: v,
    })"""),
    ("md", """## Exhaustive baseline (the "all 375 bioreactor runs" reference)

Run every candidate's inner solve, then compute the global Pareto front. This is what process development would have to do without the online solver: 375 wet bioreactor runs at $20,000 to $100,000 each, almost a year of work in a typical 4-bioreactor scale-down facility.
"""),
    ("code", """def is_dominated(p, points):
    return any(
        q["cogs_per_g"]   <= p["cogs_per_g"]
        and q["footprint_m2"] <= p["footprint_m2"]
        and q["co2_per_g"] <= p["co2_per_g"]
        and (q["cogs_per_g"]   < p["cogs_per_g"]
             or q["footprint_m2"] < p["footprint_m2"]
             or q["co2_per_g"]    < p["co2_per_g"])
        for q in points
    )

all_results = []
for cand in candidates:
    r = solve(make_dp(cand), {"target_titer": TARGET_TITER_G_L})
    if not r.feasible: continue
    for pt in r.antichain.points:
        if math.isinf(pt["cogs_per_g"]): continue
        all_results.append({**cand,
            "cogs_per_g":   pt["cogs_per_g"],
            "footprint_m2": pt["footprint_m2"],
            "co2_per_g":    pt["co2_per_g"]})
true_pareto = [p for p in all_results if not is_dominated(p, all_results)]
true_classes = {(round(p["cogs_per_g"], 2), round(p["footprint_m2"], 1))
                for p in true_pareto}
print(f"{len(all_results)} feasible candidates")
print(f"{len(true_pareto)} non-dominated points")
print(f"{len(true_classes)} distinct (cogs, footprint) Pareto classes:")
for c, f in sorted(true_classes):
    print(f"   cogs=${c:.2f}/g  footprint={f:.1f} m^2")"""),
    ("md", """## Compare four strategies

We compare against two non-online baselines and run the three online evaluators with a fixed budget of 40 inner solves:

1. **Factorial DOE at the pH=7.1 slice**: a 75-run subset that fixes one factor. This is roughly the design a process engineer with intuition about pH might run.

2. **Random sample of 40**: uniform sampling, no prior structure.

3. **Lipschitz online evaluator** on normalised features: assumes the outputs are $L$-Lipschitz in the 4D condition space.

4. **Monotonicity online evaluator** on the three monotone-bad features (`pH_distance`, `glucose_extremity`, `feed_delay`), restricted to bounding the cogs axis.

5. **LinearParametric online evaluator**: the certified confidence-polytope bound (Sec. V-C3). It carries a guarantee never to prune a Pareto-optimal candidate, but only bounds usefully when the resource map is genuinely affine in the features.

The metric is "Pareto classes recovered" by `(cogs, footprint)` value (since many candidates produce the same outcome, counting by candidate identity would understate recovery).
"""),
    ("code", """def recover_classes(ids):
    out = set()
    for i in ids:
        c = candidates[i]
        r = simulate_run(c["T_C"], c["pH"], c["glucose_mm"], c["feed_start_day"])
        out.add((round(r["cogs_per_g"], 2), round(r["footprint_m2"], 1)))
    return out

# Baseline 1: factorial DOE at pH=7.1
fac_results = []
for c in candidates:
    if c["pH"] == 7.1:
        r = solve(make_dp(c), {"target_titer": TARGET_TITER_G_L})
        if r.feasible:
            for pt in r.antichain.points:
                fac_results.append({**c,
                    "cogs_per_g": pt["cogs_per_g"],
                    "footprint_m2": pt["footprint_m2"],
                    "co2_per_g": pt["co2_per_g"]})
fac_pareto = [p for p in fac_results if not is_dominated(p, fac_results)]
fac_classes = {(round(p["cogs_per_g"], 2), round(p["footprint_m2"], 1))
               for p in fac_pareto}

# Baseline 2: random 40 picks
rng = random.Random(42)
rand_idx = rng.sample(range(len(candidates)), 40)
rand_results = []
for i in rand_idx:
    r = solve(make_dp(candidates[i]), {"target_titer": TARGET_TITER_G_L})
    if r.feasible:
        for pt in r.antichain.points:
            rand_results.append({**candidates[i],
                "cogs_per_g": pt["cogs_per_g"],
                "footprint_m2": pt["footprint_m2"],
                "co2_per_g": pt["co2_per_g"]})
rand_pareto = [p for p in rand_results if not is_dominated(p, rand_results)]
rand_classes = {(round(p["cogs_per_g"], 2), round(p["footprint_m2"], 1))
                for p in rand_pareto}

# Online evaluators
norm_feat = ["T_norm", "pH_norm", "glu_norm", "feed_norm"]
r_comp = ["cogs_per_g", "footprint_m2"]
evals = [
    ("Lipschitz",        LipschitzEvaluator(norm_feat, r_comp,
                            L={"cogs_per_g": 35.0, "footprint_m2": 10.0})),
    ("Monotonicity",     MonotonicityEvaluator(
                            ["pH_distance", "glucose_extremity", "feed_delay"],
                            ["cogs_per_g"])),
    ("LinearParametric", LinearParametricEvaluator(norm_feat, r_comp,
                            min_obs=10)),
]
online_results = {}
for name, ev in evals:
    res = solve_online(make_dp, {"target_titer": TARGET_TITER_G_L},
                       candidates=candidates, evaluator=ev, budget=40)
    online_results[name] = res

print(f"{'strategy':<25} {'runs':>5} {'classes':>10} {'recovery':>10}")
print("-" * 55)
print(f"{'Factorial DOE (pH=7.1)':<25} {75:>5} "
      f"{len(true_classes & fac_classes):>5}/{len(true_classes):>3} "
      f"{100*len(true_classes & fac_classes)/len(true_classes):>9.0f}%")
print(f"{'Random sample (seed 42)':<25} {40:>5} "
      f"{len(true_classes & rand_classes):>5}/{len(true_classes):>3} "
      f"{100*len(true_classes & rand_classes)/len(true_classes):>9.0f}%")
for name, res in online_results.items():
    rc = recover_classes(res.incumbent_ids)
    print(f"{name:<25} {res.n_evaluated:>5} "
          f"{len(true_classes & rc):>5}/{len(true_classes):>3} "
          f"{100*len(true_classes & rc)/len(true_classes):>9.0f}%")"""),
    ("md", """## Visualise the elimination cascade

The left panel shows every candidate in $(\\text{cogs}, \\text{footprint})$ space. Grey points are candidates the LinearParametric online solver did not evaluate (with the certified bound it never *eliminates* a candidate here -- see below). Coloured points are the 40 it did evaluate. Red stars mark the true Pareto front classes.

The right panel shows the trajectory of which candidate the LinearParametric picker chose at each iteration, plotted in the (`T_C`, `glucose_mm`) projection. Because this effect model is markedly nonlinear, the certified confidence polytope never tightens into a useful bound, so the picker keeps exploring broadly across the grid rather than concentrating -- and, correctly, recovers none of the Pareto classes rather than risk a wrong elimination.
"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: (cogs, footprint) space showing Pareto front and LP picks.
ax = axes[0]
all_cogs = [p["cogs_per_g"] for p in all_results]
all_fp   = [p["footprint_m2"] for p in all_results]
ax.scatter(all_cogs, all_fp, c="0.85", s=12, label="not evaluated", zorder=1)

lp_res = online_results["LinearParametric"]
lp_evaluated_cogs, lp_evaluated_fp = [], []
for i in lp_res.evaluated_ids:
    c = candidates[i]
    o = simulate_run(c["T_C"], c["pH"], c["glucose_mm"], c["feed_start_day"])
    lp_evaluated_cogs.append(o["cogs_per_g"])
    lp_evaluated_fp.append(o["footprint_m2"])
ax.scatter(lp_evaluated_cogs, lp_evaluated_fp, c="C0", s=35,
           edgecolors="white", linewidths=0.5,
           label=f"LP evaluated ({len(lp_res.evaluated_ids)})", zorder=2)

pareto_cogs = [c for c, _ in true_classes]
pareto_fp   = [f for _, f in true_classes]
ax.scatter(pareto_cogs, pareto_fp, marker="*", c="C3", s=180,
           edgecolors="black", linewidths=0.6,
           label=f"true Pareto ({len(true_classes)} classes)", zorder=3)
ax.set_xlabel("COGS (USD per gram)")
ax.set_ylabel("Facility footprint (m^2)")
ax.set_title("Where did the LinearParametric solver look?")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.4)

# Right: pick trajectory in (T_C, glucose) projection.
ax = axes[1]
xs = [candidates[i]["T_C"] for i in lp_res.evaluated_ids]
ys = [candidates[i]["glucose_mm"] for i in lp_res.evaluated_ids]
sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis", s=80,
                edgecolors="black", linewidths=0.4)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("iteration number")
# Highlight Pareto-class candidates in this projection.
for pt in true_pareto:
    ax.scatter(pt["T_C"], pt["glucose_mm"], marker="*", c="C3",
               s=200, edgecolors="black", linewidths=0.7, alpha=0.7)
ax.set_xlabel("Temperature (C)")
ax.set_ylabel("Glucose set-point (mM)")
ax.set_title("LinearParametric pick trajectory")
ax.set_xticks([33, 34, 35, 36, 37])
ax.set_yticks([5, 7, 9, 11, 13])
ax.grid(True, linestyle=":", alpha=0.4)

fig.tight_layout()
plt.show()"""),
    ("md", """## What this example demonstrates

At budget 40 (11% of the grid, an 89% reduction in wet-lab work), the tuned Lipschitz online solver recovers 3 of 4 Pareto classes, matching a 75-run factorial DOE at 53% of the experimental cost. Lipschitz is sensitive to the choice of $L$: a sweep over $L \\in \\{10, 15, 20, 25, 35\\}$ gives recovery $2, 1, 1, 1, 3$ respectively, so in a real campaign you would calibrate $L$ from preliminary scale-down or historical-batch data.

The certified LinearParametric evaluator, by contrast, recovers **zero** of the 4 classes here -- and correctly so. This effect model is markedly nonlinear (temperature U-shapes, a power-law failure rate), so no single affine parameter set fits the observations, the confidence polytope cannot support a non-trivial bound, and the evaluator falls back to the no-information value rather than risk eliminating a candidate it cannot certify as suboptimal. This is the safe-degradation property in action: on example 14's genuinely linear fleet catalogue the same evaluator recovers *every* Pareto point, whereas here it declines to guess. When you need reach on a smooth nonlinear surface at the price of an uncertified bound, a Gaussian process is the better tool.

The Monotonicity evaluator alone is uninformative on this problem because its lower bounds only tighten for candidates that lie above every observed candidate in the partial order on monotone-bad features. With no observation yet at the low-feature corner, every Pareto-optimal candidate has lower bound zero and the picker wanders. In a real campaign you would seed it with three to five hand-picked corner runs (one at each extreme of each feature) before letting it pick.

The general lesson for online co-design over expensive evaluations: **the structural prior is doing the work, the budget is the cost.** A predictive prior like LinearParametric, or a hybrid Gaussian-process-with-Lipschitz-tail, propagates information across the whole grid; pure local bounds (Lipschitz with conservative $L$, Monotonicity) need either denser observations or a warm-start.

For the bioprocess engineer, this translates to: identify the structural assumptions your process actually supports (smooth response surface? monotone in some directions?), pick the matching evaluator, and treat the first five to ten runs as a calibration phase rather than a search.
"""),
]


# ---------------------------------------------------------------------------
# NB 17: full-vehicle co-design (ICE / hybrid / EV)
# ---------------------------------------------------------------------------

NB_17 = [
    ("md", """# 17. Full-vehicle co-design across ICE, hybrid, and electric architectures

This is the largest example in the package and the first to model a system with three architecturally distinct variants in a single design study. A passenger vehicle is decomposed into 18 to 24 subsystems (depending on architecture), each a separate MCDP module with its own F (functionality demands) and R (resource outputs). For each of four representative missions we sweep over hundreds of catalog combinations and ask: which architecture, and which configuration within that architecture, minimises the relevant cost vector?

The example surfaces two coupled cycles that automotive engineers fight at every program: the *mass spiral* (subsystem weights sum to curb weight, which feeds back as the design mass that the suspension, brakes, tires, and powertrain must support), and the *energy-storage loop* (fuel-tank size for ICE or battery capacity for EV depends on consumption × range; the storage's own mass contributes to curb weight, which raises consumption). The Kleene iteration resolves both cycles simultaneously.

This notebook imports the example module from `examples/17_car_codesign.py` and demonstrates the workflow: build one car of each architecture for a chosen mission, solve, compare; then sweep all three across all missions and read off the Pareto winners.
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys

# The notebook executes with CWD = project root.
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'car_codesign', os.path.join(PROJECT_ROOT, 'examples', '17_car_codesign.py'))
ex17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex17)

from codesign import solve
print(f"Module loaded; {sum(1 for m in dir(ex17) if not m.startswith('_'))} public symbols")
print(f"  ICE engines      : {len(ex17.ICE_ENGINES)}")
print(f"  Hybrid engines   : {len(ex17.HYBRID_ENGINES)}")
print(f"  Transmissions    : {len(ex17.ICE_TRANSMISSIONS)}")
print(f"  Body styles      : {len(ex17.BODY_STYLES)}")
print(f"  Suspension types : {len(ex17.SUSPENSION_VARIANTS)}")
print(f"  Tire compounds   : {len(ex17.TIRES)}")
print(f"  EV motors        : {len(ex17.EV_MOTORS)}")
print(f"  EV batteries     : {len(ex17.EV_BATTERIES)}")"""),
    ("md", """## Picking a representative mission

We use the *Family Daily* mission as the showcase: 5 passengers, 500 L of cargo, 180 km/h top speed, 700 km of range, 0.9 g of braking authority, 0-100 km/h in 9 seconds. Realistic European D-segment territory.
"""),
    ("code", """mission = ex17.MISSIONS["Family Daily"]
for k, v in mission.items():
    print(f"  {k:<24} = {v}")"""),
    ("md", """## Build one car of each architecture

Each build function returns a `System.build()` design problem ready for `solve()`. The wiring inside each builder expresses the mass spiral and the energy-storage loop as MCDP constraints; the Kleene iteration finds the fixed point.
"""),
    ("code", """# ICE: 2.0L turbo gas + 8AT + comfort suspension + premium AS tires
ice_dp = ex17.build_ice_car(
    mission=mission,
    body=next(b for b in ex17.BODY_STYLES if b.style_name == "mid_sedan"),
    engine=next(e for e in ex17.ICE_ENGINES if e.name == "2.0L turbo gas"),
    forced_induction=ex17.ForcedInduction(kind="single_turbo"),
    transmission=next(t for t in ex17.ICE_TRANSMISSIONS if t.name == "8AT torque-converter"),
    suspension_type="comfort",
    tire_choice=next(t for t in ex17.TIRES if t.name == "premium_AS"),
    wheel_choice=next(w for w in ex17.WHEELS if w.name == "alloy_cast_17"),
    steering_choice=next(s for s in ex17.STEERING_OPTIONS if s.name == "EPS"),
    drivetrain_layout="fwd")

# HEV: 2.5L Atkinson + 120 kW motor + Li-NMC HV pack + SiC inverter
hev_dp = ex17.build_hybrid_car(
    mission=mission,
    body=next(b for b in ex17.BODY_STYLES if b.style_name == "mid_sedan"),
    engine=ex17.HYBRID_ENGINES[1],
    motor_peak_power_kW=120,
    hv_battery_chemistry="lithium_NMC",
    power_electronics_sic=True,
    suspension_type="comfort",
    tire_choice=next(t for t in ex17.TIRES if t.name == "premium_AS"),
    wheel_choice=next(w for w in ex17.WHEELS if w.name == "alloy_cast_17"),
    steering_choice=next(s for s in ex17.STEERING_OPTIONS if s.name == "EPS"),
    drivetrain_layout="fwd")

# EV: 180 kW PMSM + 100 kWh NMC 800V + SiC + heat pump
# (The 85 kWh pack is insufficient for 700 km at this mass; the 800V
# 100 kWh option dominates here.)
ev_dp = ex17.build_ev_car(
    mission=mission,
    body=next(b for b in ex17.BODY_STYLES if b.style_name == "mid_sedan"),
    motor=next(m for m in ex17.EV_MOTORS if "180kW" in m.name),
    battery=next(b for b in ex17.EV_BATTERIES if "100kWh" in b.name),
    power_electronics_sic=True,
    suspension_type="comfort",
    tire_choice=next(t for t in ex17.TIRES if t.name == "EV_XL"),
    wheel_choice=next(w for w in ex17.WHEELS if w.name == "alloy_cast_17"),
    steering_choice=next(s for s in ex17.STEERING_OPTIONS if s.name == "EPS"),
    drivetrain_layout="rwd")

print("All three System DPs built")"""),
    ("md", """## Solve and compare

Each `solve()` runs the Kleene iteration until the mass spiral converges. ICE typically takes 15 to 25 iterations, HEV 15 to 20, EV 30 to 50 (the battery dominates the mass cycle so convergence is slower).
"""),
    ("code", """for name, dp in (("ICE", ice_dp), ("HEV", hev_dp), ("EV", ev_dp)):
    res = solve(dp, dict(mission), max_iter=250, verbose=0)
    pt = list(res.antichain.points)[0] if res.feasible and res.antichain.points else None
    if not pt:
        print(f"{name:<3}: INFEASIBLE")
        continue
    cost = pt["production_cost"]
    wt = pt["curb_weight"]
    co2 = pt["co2_per_km"]
    durab = pt["durability"]
    fuel = pt.get("fuel_consumption", 0.0)
    energy = pt.get("energy_consumption", 0.0)
    if name == "EV":
        consumption = f"{energy:.1f} kWh/100km"
    else:
        consumption = f"{fuel:.1f} L/100km"
    print(f"{name:<3}: cost=${cost:>7,.0f}  weight={wt:>5.0f}kg  "
          f"{consumption:>14}  CO2={co2:>3.0f}g/km  durab={durab/1000:.0f}k km")"""),
    ("md", """## Block diagram

The framework's `draw_system` tool renders the module graph. For the EV the diagram shows 18 modules connected by ~80 constraint edges. The amber edges are the mass-spiral cycle, in which every load-bearing module feeds the total weight back to its own design-mass input.
"""),
    ("code", """import shutil
if shutil.which("dot"):
    from codesign import draw_system
    g = draw_system(ev_dp, name="car_ev_notebook", rankdir="LR")
    g.format = "svg"
    svg_bytes = g.pipe()
    from IPython.display import SVG, display
    display(SVG(svg_bytes))
else:
    print("graphviz not available; skipping diagram")"""),
    ("md", """## Sweeping the catalog

The `sweep_ice`, `sweep_hev`, `sweep_ev` helpers iterate over reasonable combinations of body, engine / motor, transmission / power electronics, suspension, and tires. Each combination is solved at the mission; infeasible designs are silently dropped. Below we evaluate Urban Compact across all three architectures and report the cheapest feasible design per architecture.
"""),
    ("code", """mission_urban = ex17.MISSIONS["Urban Compact"]
for arch_name, sweep_fn in (("ICE", ex17.sweep_ice),
                              ("HEV", ex17.sweep_hev),
                              ("EV",  ex17.sweep_ev)):
    results = sweep_fn(mission_urban)
    if not results:
        print(f"{arch_name:<3}: 0 feasible designs")
        continue
    label, pt = min(results, key=lambda r: r[1]["production_cost"])
    cost = pt["production_cost"]; wt = pt["curb_weight"]; co2 = pt["co2_per_km"]
    if arch_name == "EV":
        consumption = f"{pt.get('energy_consumption', 0):.1f} kWh/100km"
    else:
        consumption = f"{pt.get('fuel_consumption', 0):.1f} L/100km"
    print(f"{arch_name:<3}: {len(results):>4} feasible  cheapest: "
          f"${cost:>7,.0f} {wt:.0f}kg {consumption:>13} CO2={co2:.0f}g/km")"""),
    ("md", """## What the analysis surfaces

Across the four missions, three patterns emerge from the sweep:

1. **ICE always wins on upfront cost.** It does so by 15 to 30%, driven mainly by the absence of an HV battery and power electronics. The cheapest feasible ICE configuration sits around $30,000 for a compact and $40,000 for a 7-seat SUV.

2. **HEV always wins on 10-year TCO.** The 30% fuel-consumption credit from regenerative braking and engine-off idling compounds across 150,000 km of driving and overcomes the $4,000 to $6,000 hardware premium.

3. **EV wins on tank-to-wheel CO2 but is range-limited by the mass spiral.** A 7-passenger 800-km EV remains infeasible against today's pack-level energy density (~160 Wh/kg). This is a genuine engineering finding, not a calibration artefact: the analogous real-world product (a 7-seat EV with 800 km of range) does not exist in the 2024 market.

The framework's contribution is making each of these conclusions inspectable. Every R port of every module is exposed in the antichain; every constraint is a single line of operator-overloaded Python; and the diagram makes the mass spiral visible.
"""),
]


NB_18 = [
    ("md", """# 18. Metabolic architecture switching across carbon sources (temporal Case 1)

This is the first *temporal* example: a system whose best architecture changes over time because the environment changes. An organism alternates between two metabolic architectures as its carbon source changes. When glucose is abundant it runs a glycolytic, fast-growth network; when only acetate is available it must switch to a gluconeogenic architecture (the glyoxylate shunt) that is slower and biochemically costlier to operate. Switching is not free: re-acclimation means expressing a different enzyme complement, the lag phase of the classic diauxic shift.

The question is not "which architecture is best" but "what is the schedule of architectures across a changing environment, given that switching costs something". That is a small dynamic program over the discrete choice of architecture per epoch, solved here by `solve_schedule` (an exact Viterbi pass over the epoch/architecture lattice).
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'metabolic', os.path.join(PROJECT_ROOT, 'examples', '18_metabolic_switching.py'))
ex18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex18)

from codesign import solve_schedule
print("Module loaded.")
print("Architectures:", ex18.GLYCOLYTIC.name, "and", ex18.GLUCONEOGENIC.name)
"""),
    ("md", """## The two architectures

Each metabolic architecture is a small co-design problem: given a demanded biomass growth rate `mu` (the outer functionality of the epoch), it returns a scalar `burden`, a proxy for the proteomic and ATP cost of sustaining that growth on that substrate. The glycolytic architecture is cheap per unit growth but capped at a moderate growth ceiling; the gluconeogenic one is costlier per unit and carries a fixed shunt overhead, but reaches a higher ceiling.

We first solve each architecture directly across a sweep of demanded growth rates to see the burden curves that drive the scheduling decision.
"""),
    ("code", """from codesign import solve, minimize_cost
import numpy as np

mus = np.linspace(0.1, 1.15, 40)
burden_glyc, burden_gluco = [], []
for mu in mus:
    for arch, store in ((ex18.GLYCOLYTIC, burden_glyc),
                        (ex18.GLUCONEOGENIC, burden_gluco)):
        res = solve(arch.dp, {"mu": float(mu)})
        if res.feasible:
            pt = minimize_cost(res, ex18.burden_cost)
            store.append(ex18.burden_cost(pt) if pt else np.nan)
        else:
            store.append(np.nan)
print("glycolytic feasible up to mu =",
      f"{max(m for m,b in zip(mus,burden_glyc) if not np.isnan(b)):.2f}")
print("gluconeogenic feasible up to mu =",
      f"{max(m for m,b in zip(mus,burden_gluco) if not np.isnan(b)):.2f}")
"""),
    ("code", """import matplotlib.pyplot as plt

# MATLAB gem colours.
BLUE   = "#0072BD"
ORANGE = "#D95319"

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(mus, burden_glyc, color=BLUE, lw=3.0, label="glycolytic (glucose)")
ax.plot(mus, burden_gluco, color=ORANGE, lw=3.0, label="gluconeogenic (acetate)")
ax.set_xlabel("demanded growth rate  mu  (1/h)", fontsize=12)
ax.set_ylabel("metabolic burden", fontsize=12)
ax.set_title("Burden curves of the two metabolic architectures", fontsize=13)
ax.legend(fontsize=11, frameon=True, loc="upper left")
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)
fig.tight_layout()
plt.show()
"""),
    ("md", """The glycolytic curve (blue) is lower and cheaper but stops at its growth ceiling; the gluconeogenic curve (orange) continues to higher growth rates but sits above glycolytic everywhere they overlap, and it is offset upward by the fixed shunt overhead. Where both are feasible, glycolytic is cheaper to run. The scheduling tension is therefore entirely about *switching*: is it worth switching into the cheaper pathway for a short window if that costs two re-acclimations?
"""),
    ("md", """## The environment and the schedule

The environment is a sequence of epochs that alternate substrate and demand. Only the architecture matching the available substrate is admissible per epoch, except a deliberately contested `mixed` epoch (both substrates present) that is *flanked by acetate on both sides*. Choosing the locally cheaper glycolytic pathway for that single mixed epoch therefore forces two extra switches (acetate to glucose-type and back), whereas riding it out on the incumbent gluconeogenic pathway costs none.

We solve the schedule twice: once with a low re-acclimation cost, once with a high one.
"""),
    ("code", """epochs = ex18.build_environment()
print("Environment:")
for ep in epochs:
    subs = "/".join(sorted({c.tags["substrate"] for c in ep.candidates}))
    print(f"  {ep.name:<10s} mu={ep.functionality['mu']:.2f}  [{subs}]")

sched_lo = solve_schedule(epochs, cost_fn=ex18.burden_cost, switch_cost=0.05)
sched_hi = solve_schedule(epochs, cost_fn=ex18.burden_cost, switch_cost=0.8)

print("\\nLow switch cost (0.05):  ", " -> ".join(sched_lo.schedule),
      f"  ({sched_lo.n_switches} switches)")
print("High switch cost (0.8):  ", " -> ".join(sched_hi.schedule),
      f"  ({sched_hi.n_switches} switches)")
"""),
    ("code", """# Visualise the two schedules as coloured tracks across epochs.
fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.4), sharex=True)
names = [ep.name for ep in epochs]
xs = range(len(epochs))

def plot_track(ax, sched, title):
    for i, er in enumerate(sched.epochs):
        c = BLUE if er.architecture == "glycolytic" else ORANGE
        ax.barh(0, 1, left=i, height=0.6, color=c, edgecolor="white", linewidth=1.5)
        ax.text(i + 0.5, 0, er.architecture[:5], ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        if er.switch_cost:
            ax.plot(i, 0.42, marker="v", color="black", markersize=9)
    ax.set_yticks([])
    ax.set_xlim(0, len(epochs))
    ax.set_title(title, fontsize=12, loc="left")

plot_track(axes[0], sched_lo,
           f"Low re-acclimation cost: {sched_lo.n_switches} switches "
           f"(total burden {sched_lo.total_cost:.2f})")
plot_track(axes[1], sched_hi,
           f"High re-acclimation cost: {sched_hi.n_switches} switches "
           f"(total burden {sched_hi.total_cost:.2f})")
axes[1].set_xticks([i + 0.5 for i in xs])
axes[1].set_xticklabels(names, fontsize=10)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color=BLUE, label="glycolytic"),
                        Patch(color=ORANGE, label="gluconeogenic"),
                        plt.Line2D([], [], marker="v", color="black",
                                   linestyle="none", label="switch")],
               loc="upper right", fontsize=9, ncol=3, frameon=True)
fig.tight_layout()
plt.show()
"""),
    ("md", """## What the schedule reveals

The `mixed` epoch is the whole story. Under a low re-acclimation cost the organism switches into the cheaper glycolytic pathway for that epoch and pays for two extra switches (the black markers), because the per-epoch burden saving outweighs the switching cost. Under a high re-acclimation cost the same epoch is served by the incumbent gluconeogenic pathway: two switches now cost more than the burden they would save, so the organism rides it out.

This is the diauxic-shift intuition made quantitative. The optimal metabolic schedule is not a property of the environment alone; it depends on the *economics of switching*. The framework surfaces the flip point directly: identical environment, different switching cost, qualitatively different metabolic program. The same `solve_schedule` primitive that handles this organism would handle a vehicle reconfiguring between drivetrain modes or a sensor network swapping topology between survey and tracking phases.
"""),
]


NB_19 = [
    ("md", """# 19. Planetary rover module activation over a battery budget (temporal Case 2)

This example addresses the question "robots with modules that switch on and off depending on context", solved as a genuine dynamic program with a resource carried between stages. A small planetary rover carries modules that can be independently activated: a drive train, a science payload, a high-gain comms link, and a survival heater. Each mission phase demands a capability, and the rover runs on a battery that depletes when modules draw power and partially recharges from solar input between phases.

The outer object is a finite-horizon DP over mission phases; the per-phase decision is which module configuration (which "mode") to activate; the per-phase cost is obtained by *solving a co-design problem* that sizes the power bus and accumulates the mode's energy and objective cost; and the battery state of charge is the carried state linking phases. The framing follows standard spacecraft power-mode practice, where the power budget is organised into operational modes entered during different mission phases, and activities are scheduled around available battery reserve.
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'rover', os.path.join(PROJECT_ROOT, 'examples', '19_rover_modules.py'))
ex19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex19)

from codesign import solve_dynamic, rollout
print("Modes:", ", ".join(a.name for a in ex19.ALL_MODES))
print(f"Battery capacity {ex19.BATTERY_CAPACITY_WH:.0f} Wh, "
      f"solar recharge {ex19.SOLAR_RECHARGE_WH:.0f} Wh/phase")
"""),
    ("md", """## The four modes, as an energy/value trade

Each mode is a co-design problem that, given the phase's capability demand, sizes a power bus and returns the energy drawn (subtracted from the battery) and a cost. The cost is *energy spent plus an opportunity penalty* for objective value not delivered, kept non-negative because the co-design resource poset forbids negative resources. High-value modes carry a low penalty and are preferred when charge allows.
"""),
    ("code", """from codesign import solve, minimize_cost
rows = []
for arch in ex19.ALL_MODES:
    res = solve(arch.dp, {"cap": 5.0})
    pt = minimize_cost(res, ex19.mission_cost) if res.feasible else None
    if pt:
        rows.append((arch.name, pt["energy_Wh"], pt["cost"]))
print(f"{'mode':<9} {'energy_Wh':>10} {'cost':>8}")
for name, e, c in rows:
    print(f"{name:<9} {e:>10.1f} {c:>8.1f}")
"""),
    ("md", """## Solve the policy and roll out from full and depleted batteries

`solve_dynamic` runs the backward Bellman pass over the (phase, state-of-charge) lattice and returns a full state-indexed policy, valid for any starting charge. We roll it out from a full battery and from a nearly-empty one.
"""),
    ("code", """stages = ex19.build_mission(n_phases=6)
grid = ex19.StateGrid.linspace(0.0, ex19.BATTERY_CAPACITY_WH, 61)
policy = solve_dynamic(stages, grid, cost_fn=ex19.mission_cost,
                       terminal_cost=ex19.terminal_reward)

full = rollout(policy, stages, ex19.BATTERY_CAPACITY_WH)
low  = rollout(policy, stages, 90.0)

for label, res in (("full 300 Wh", full), ("low 90 Wh", low)):
    print(f"\\n{label}: {' -> '.join(res.schedule)}  (cost {res.total_cost:.1f})")
    for sr in res.stages:
        print(f"   {sr.stage:<8s} {sr.architecture:<9s} "
              f"soc {sr.state_in:6.1f} -> {sr.state_out:6.1f} Wh")
"""),
    ("code", """import matplotlib.pyplot as plt

# MATLAB gem colours, one per mode.
MODE_COLOR = {"drive": "#0072BD", "science": "#D95319",
              "comms": "#EDB120", "survival": "#7E2F8E"}

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

def plot_run(ax, res, title):
    trace_x, trace_y = [], []
    for i, s in enumerate(res.stages):
        trace_x += [i, i + 1]
        trace_y += [s.state_in, s.state_out]
    ax.plot(trace_x, trace_y, color="0.25", lw=3.0, zorder=3,
            label="state of charge")
    for i, s in enumerate(res.stages):
        ax.axvspan(i, i + 1, color=MODE_COLOR.get(s.architecture, "0.6"),
                   alpha=0.30, zorder=1)
        ax.text(i + 0.5, ex19.BATTERY_CAPACITY_WH * 0.92, s.architecture,
                ha="center", va="top", fontsize=9,
                color=MODE_COLOR.get(s.architecture, "0.3"), fontweight="bold")
    ax.set_ylim(0, ex19.BATTERY_CAPACITY_WH * 1.02)
    ax.set_ylabel("battery (Wh)", fontsize=11)
    ax.set_title(title, fontsize=12, loc="left")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)

plot_run(axes[0], full, "Start at full charge: science-heavy, then steps down")
plot_run(axes[1], low, "Start depleted: forced onto lighter modes throughout")
axes[1].set_xticks([i + 0.5 for i in range(len(full.stages))])
axes[1].set_xticklabels([s.stage for s in full.stages], fontsize=10)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color=c, alpha=0.4, label=m)
                        for m, c in MODE_COLOR.items()],
               loc="lower left", fontsize=9, ncol=4, frameon=True)
fig.tight_layout()
plt.show()
"""),
    ("md", """## What the policy reveals

The same policy produces two different module-activation schedules depending only on the starting charge. From a full battery the rover runs the high-value science mode until reserve falls, then steps down to a lighter mode as the battery approaches empty. Starting depleted, it can never afford science's large draw (which would take the battery negative), so it holds a lighter mode throughout and lets solar rebuild what it can.

This is the load-shedding behaviour real power-constrained missions use, and it falls out of the framework without any special-casing: the co-design solve sizes each mode, the carried state of charge couples the phases, and the backward DP finds the schedule. The out-of-bounds guard in the solver is what makes it correct, a mode whose draw would take the battery below zero is rejected before the grid snaps the state back into range, so the rover never plans a schedule it cannot energetically execute.
"""),
]


NB_20 = [
    ("md", """# 20. Antichain-valued sequential co-design (multi-objective DP)

The rover example (19) carried a battery budget and minimised a single scalar cost per phase, so its value function was one number per state. This notebook exercises the *antichain-valued* generalisation, the full sequential co-design object: the value at each stage and state is a whole Pareto front of cumulative resource totals, not a scalar, and the Bellman `min` becomes an antichain union-and-minimise.

The scenario is a multi-leg survey. At each leg the operator picks a mode that trades two incommensurable objectives, monetary cost and CO2, against each other, while drawing down a shared energy budget carried between legs. Because cost and CO2 are incomparable, the answer to "how should the whole mission be run" is not one plan but a Pareto front of whole-mission (cost, CO2) totals, each realised by a different schedule of per-leg modes. The antichain-valued DP computes that front exactly.
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'seqcd', os.path.join(PROJECT_ROOT, 'examples', '20_sequential_codesign.py'))
ex20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex20)

from codesign import solve_sequential, sum_combine, check_monotonicity
print("Modes:", ", ".join(m.name for m in ex20.MODES))
"""),
    ("md", """## The modes

Three incomparable modes per leg span the cost/CO2 trade: `eco` (low CO2, high cost), `balanced` (middle), and `rapid` (low cost, high CO2). Each draws a different amount from the shared energy budget. The Pareto structure comes from the union over these incomparable modes and from accumulating incomparable totals across legs.
"""),
    ("code", """print(f"{'mode':<9} {'cost':>5} {'co2':>5} {'energy':>7}")
for m, spec in (("eco",(10,1,4)),("balanced",(6,4,5)),("rapid",(2,9,7))):
    print(f"{m:<9} {spec[0]:>5} {spec[1]:>5} {spec[2]:>7}")
"""),
    ("md", """## Solve the antichain-valued DP

`solve_sequential` carries the scalar energy state (read by the transition) while accumulating the named cost axes `cost` and `co2` on the antichain. We use `sum_combine` because both objectives accumulate additively across legs.
"""),
    ("code", """n_legs = 4
stages = ex20.build_mission(n_legs)
grid = ex20.StateGrid.linspace(0.0, ex20.ENERGY_CAPACITY, 61)

res = solve_sequential(stages, grid, cost_axes=["cost", "co2"],
                       initial_state=ex20.ENERGY_CAPACITY, combine=sum_combine)

front = sorted(((p["cost"], p["co2"]) for p in res.value), key=lambda t: t[0])
print(f"Whole-mission Pareto front: {res.width} incomparable (cost, CO2) totals")
for c, e in front:
    print(f"   cost={c:6.1f}   co2={e:6.1f}")
"""),
    ("code", """import matplotlib.pyplot as plt

BLUE = "#0072BD"
xs = [c for c, e in front]
ys = [e for c, e in front]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.step(xs, ys, where="post", color=BLUE, lw=2.5, alpha=0.6, zorder=1)
ax.plot(xs, ys, "o", color=BLUE, markersize=10, zorder=3,
        markeredgecolor="white", markeredgewidth=1.2)
ax.annotate("all rapid\\n(cheap, dirty)", xy=(xs[0], ys[0]),
            xytext=(xs[0] + 4, ys[0] - 3), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="0.4"))
ax.annotate("all eco\\n(clean, costly)", xy=(xs[-1], ys[-1]),
            xytext=(xs[-1] - 13, ys[-1] + 2), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set_xlabel("total mission cost", fontsize=12)
ax.set_ylabel("total mission CO2", fontsize=12)
ax.set_title(f"Whole-mission Pareto front ({res.width} plans), {n_legs} legs",
             fontsize=13)
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)
fig.tight_layout()
plt.show()
"""),
    ("md", """## The front grows polynomially, not exponentially

A worry with antichain-valued DP is that the value front blows up with the horizon. The theory says the front size equals the width of the *reachable frontier*, and for a summed objective on a fixed number of axes it grows polynomially in the horizon. We verify this by sweeping the number of legs and plotting the front width against the raw plan count.
"""),
    ("code", """widths = []
legs_range = list(range(1, 9))
for nl in legs_range:
    st = ex20.build_mission(nl)
    g = ex20.StateGrid.linspace(0.0, ex20.ENERGY_CAPACITY, 61)
    r = solve_sequential(st, g, cost_axes=["cost", "co2"],
                         initial_state=ex20.ENERGY_CAPACITY, combine=sum_combine)
    widths.append(r.width)

ORANGE = "#D95319"
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(legs_range, widths, "-o", color=ORANGE, lw=3.0, markersize=9,
        markeredgecolor="white", markeredgewidth=1.2, label="Pareto front width")
ax.plot(legs_range, [2 ** n for n in legs_range], "--", color="0.5", lw=2.0,
        label="raw plan count (2^legs)")
ax.set_xlabel("number of legs", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.set_title("Front width grows polynomially, not exponentially", fontsize=13)
ax.set_yscale("log")
ax.legend(fontsize=11, frameon=True, loc="upper left")
ax.grid(True, alpha=0.3, linewidth=0.8, which="both")
ax.tick_params(labelsize=11)
fig.tight_layout()
plt.show()
"""),
    ("md", """## The monotonicity guard

The theory gives two conditions, (H1) and (H2), under which the value front is monotone in the carried state: more energy never shrinks the achievable front. `check_monotonicity` verifies them numerically on the state grid, distinguishing genuinely non-monotone (perishable) stages from the benign consumable-but-monotone case.
"""),
    ("code", """rep = check_monotonicity(stages, grid, cost_axes=["cost", "co2"])
print(rep)
print("Value front monotone in carried budget:", rep.monotone_value_guaranteed)
"""),
    ("md", """## What the antichain-valued DP delivers

The output is a single object, the whole-mission Pareto front, that a single-objective DP cannot produce: incomparable plans from all-rapid (cheapest, dirtiest) to all-eco (cleanest, costliest), with interior points mixing modes across legs. Every point is on the exact reachable frontier: achievable by some feasible mode sequence, and no achievable non-dominated total is missed.

Two structural facts make this practical rather than a combinatorial explosion. The front width grows polynomially in the horizon (linearly here) rather than as the raw plan count, because cost and CO2 accumulate on two fixed axes. And the value is monotone in the carried budget when (H1) and (H2) hold, which the guard confirms. This is the general sequential co-design object; the scalar rover DP of notebook 19 is its width-one special case.
"""),
]


NB_21 = [
    ("md", """# 21. Vector-state co-design for a self-reconfiguring robot

Examples 19 and 20 carried a single scalar between stages (a battery charge). Real reconfigurable systems carry a *state vector*. The Formula 1 seasonal co-design of Neumann, Zardini and colleagues carries two battery wear levels plus a regulatory flag; a self-reconfiguring modular robot on a multi-leg mission carries the accumulated wear of each drive module it can activate, plus a shared energy budget.

This notebook is the robot case, and it exercises the general **vector-state dynamic program**. A field robot reconfigures between three morphologies at each mission leg, drawing on two physical drive modules whose wear accumulates independently:

- **tracked**: uses the track module heavily. Low energy, high track wear.
- **wheeled**: uses the wheel module. Higher energy, low ops cost, high wheel wear.
- **hybrid**: splits load across both modules, wearing each a little.

The carried state is a vector of three axes (`track_wear`, `wheel_wear`, `energy`), and the optimal plan spreads wear across the two modules so neither hits its limit and forces an expensive fallback.
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'recon', os.path.join(PROJECT_ROOT, 'examples', '21_reconfigurable_robot.py'))
ex21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex21)

from codesign import solve_vector_sequential, check_vector_monotonicity, sum_combine
print("Morphologies:", ", ".join(m.name for m in ex21.MORPHOLOGIES))
print(f"energy capacity {ex21.ENERGY_CAPACITY:.0f} (+{ex21.SOLAR_RECHARGE:.0f}/leg), "
      f"wear limit {ex21.WEAR_LIMIT:.0f} per module")
"""),
    ("md", """## The morphologies, as an energy / ops / wear trade

Each morphology is a co-design problem returning the energy it draws, an operations-cost proxy, and the wear it applies to each module. Tracked and wheeled are deliberately *cost-incomparable* (tracked is cheaper on energy, wheeled on operations), so the mission has a genuine Pareto front rather than a single dominant plan.
"""),
    ("code", """print(f"{'morphology':<10} {'energy':>6} {'ops':>4} {'d_track':>8} {'d_wheel':>8}")
for m, spec in (("tracked",(2,4,3,0)),("wheeled",(4,2,0,3)),("hybrid",(3,3,1.5,1.5))):
    print(f"{m:<10} {spec[0]:>6} {spec[1]:>4} {spec[2]:>8} {spec[3]:>8}")
"""),
    ("md", """## Solve the vector-state DP

`solve_vector_sequential` carries the full three-axis state vector on a `VectorStateGrid` (a product of three continuous axes), accumulating (energy, ops) on the antichain while the transition advances the wear and energy axes.
"""),
    ("code", """n_legs = 5
stages = ex21.build_mission(n_legs)
grid = ex21.VectorStateGrid([
    ex21.ContinuousAxis('track_wear', 0.0, ex21.WEAR_LIMIT, 11),
    ex21.ContinuousAxis('wheel_wear', 0.0, ex21.WEAR_LIMIT, 11),
    ex21.ContinuousAxis('energy', 0.0, ex21.ENERGY_CAPACITY, 13),
])
print(f"grid size: {len(grid)} state nodes")

res = solve_vector_sequential(
    stages, grid, cost_axes=['energy', 'ops'],
    initial_state={'track_wear': 0.0, 'wheel_wear': 0.0, 'energy': ex21.ENERGY_CAPACITY},
    combine=sum_combine)

front = sorted(((p['energy'], p['ops']) for p in res.value), key=lambda t: t[0])
print(f"\\nMission Pareto front ({res.width} incomparable (energy, ops) totals):")
for e, o in front:
    print(f"   energy={e:6.1f}   ops={o:6.1f}")
"""),
    ("code", """import matplotlib.pyplot as plt

BLUE = "#0072BD"
xs = [e for e, o in front]
ys = [o for e, o in front]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.step(xs, ys, where="post", color=BLUE, lw=2.5, alpha=0.6, zorder=1)
ax.plot(xs, ys, "o", color=BLUE, markersize=11, zorder=3,
        markeredgecolor="white", markeredgewidth=1.4)
ax.set_xlabel("total mission energy", fontsize=12)
ax.set_ylabel("total mission ops cost", fontsize=12)
ax.set_title(f"Reconfigurable-robot Pareto front ({res.width} plans), {n_legs} legs",
             fontsize=13)
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)
fig.tight_layout()
plt.show()
"""),
    ("md", """## The three-axis carried state

The key point is that a scalar DP could not represent this problem: the value is parametrised by a *vector* of two wear levels plus energy. Below we confirm the monotonicity guard passes over the product order, so more spare wear budget and more energy never shrink the achievable front.
"""),
    ("code", """rep = check_vector_monotonicity(stages, grid, cost_axes=['energy', 'ops'])
print(rep)
print("value front monotone in the carried state vector:", rep.monotone_value_guaranteed)
"""),
    ("md", """## What the vector-state DP delivers

Each point on the front is a different morphology schedule across the five legs. The low-energy end leans on tracked (cheap energy), the low-ops end on wheeled (cheap operations), and the interior mixes so that neither module's wear reaches its limit. Reaching the fifth leg on a single morphology is infeasible (five wheeled legs would wear the wheel module past its limit), so the DP is forced to spread load, exactly the maintenance-aware behaviour an operator wants.

This is the structured multi-component state that the Formula 1 seasonal co-design carries (there: two batteries plus a regulatory flag; here: two drive modules plus energy). The single-axis case reduces to the scalar sequential DP of notebook 20, so the vector layer strictly generalises it.
"""),
]


NB_22 = [
    ("md", """# 22. Online feedback co-design of an adaptive sensor node

Every temporal example so far plans offline: the whole horizon is solved in advance and a policy is read out. This notebook closes the loop. A solar-powered environmental sensor node runs in the field, and at each control step it senses its current battery charge, reads the current data requirement and solar conditions, re-solves its co-design at those live conditions, applies the cheapest feasible configuration, and repeats.

The plan is never trusted to match reality: the next configuration is chosen from the *measured* state, so when conditions diverge from any forecast the loop simply re-solves against what actually happened. That is feedback, not open-loop replay. This is the co-design instance of **control co-design (CCD)** in its nested, receding-horizon form, here the myopic variant (re-solve a single static co-design at the current conditions each step).
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'onlinecd', os.path.join(PROJECT_ROOT, 'examples', '22_online_feedback_codesign.py'))
ex22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex22)

from codesign import run_online_codesign
print("Configs:", ", ".join(c.name for c in ex22.CONFIGS))
print(f"battery capacity {ex22.BATTERY_CAPACITY:.0f}, horizon {ex22.N_STEPS} steps")
"""),
    ("md", """## The scenario

The node runs in three sensing configurations (low_power, nominal, high_rate), each a co-design problem that sizes the radio and compute against the demanded data rate and is *gated by the measured charge* (a hungry configuration is infeasible when the battery is low, the feedback path). A storm raises the demanded data rate mid-run, and the solar recharge follows a day/storm/recovery profile.
"""),
    ("code", """print("step:   ", " ".join(f"{i:>3}" for i in range(ex22.N_STEPS)))
print("demand: ", " ".join(f"{d:>3}" for d in ex22.DEMAND_RATE))
print("solar:  ", " ".join(f"{s:>3}" for s in ex22.SOLAR))
"""),
    ("md", """## Run the closed loop

At each step the loop senses the true battery charge (held by the plant), reads the live requirement (demand plus available charge) and environment (solar), re-solves the co-design, applies the cheapest feasible configuration, steps the true plant, and logs the outcome.
"""),
    ("code", """sensor, requirement, environment, plant = ex22.make_scenario()
result = run_online_codesign(
    ex22.CONFIGS, n_steps=ex22.N_STEPS, sensor=sensor, requirement=requirement,
    environment=environment, plant=plant, cost_fn=ex22.cost_fn,
    initial_state=ex22.BATTERY_CAPACITY)

print(f"{'t':>2}  {'soc_in':>6}  {'rate':>4}  {'solar':>5}  {'config':<10} {'energy':>6}  {'ops':>4}")
for s in result.steps:
    e = s.point['energy'] if s.feasible else float('nan')
    o = s.point['ops'] if s.feasible else float('nan')
    cfg = s.architecture if s.feasible else 'INFEASIBLE'
    print(f"{s.step:>2}  {s.measured_state:>6.1f}  {s.requirement['rate']:>4.0f}  "
          f"{s.environment['solar']:>5.0f}  {cfg:<10} {e:>6.1f}  {o:>4.1f}")
print(f"\\nschedule: {' -> '.join(result.schedule)}")
print(f"total ops cost = {result.total_cost:.1f}, feasible = {result.feasible}")
"""),
    ("code", """import matplotlib.pyplot as plt

# MATLAB gem colours, one per config.
CFG_COLOR = {"low_power": "#0072BD", "nominal": "#D95319", "high_rate": "#7E2F8E"}

steps = result.steps
xs = list(range(len(steps)))
soc = [s.measured_state for s in steps]
demand = [s.requirement["rate"] for s in steps]

fig, ax = plt.subplots(figsize=(9.5, 5.0))
# Battery trace (thick).
ax.plot(xs, soc, color="0.25", lw=3.0, marker="o", markersize=6,
        zorder=4, label="battery charge")
# Config bands.
for s in steps:
    c = CFG_COLOR.get(s.architecture, "0.6")
    ax.axvspan(s.step - 0.5, s.step + 0.5, color=c, alpha=0.22, zorder=1)
# Demand overlay on a twin axis.
ax2 = ax.twinx()
ax2.plot(xs, demand, color="#EDB120", lw=2.5, ls="--", marker="s",
         markersize=5, zorder=3, label="data-rate demand")
ax2.set_ylabel("data-rate demand", fontsize=12, color="#B8860B")
ax2.tick_params(axis="y", labelcolor="#B8860B", labelsize=11)

ax.set_xlabel("control step", fontsize=12)
ax.set_ylabel("battery charge", fontsize=12)
ax.set_title("Online feedback co-design: charge, demand, and chosen config",
             fontsize=13)
ax.set_ylim(0, ex22.BATTERY_CAPACITY * 1.05)
ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)

from matplotlib.patches import Patch
handles = [Patch(color=c, alpha=0.35, label=n) for n, c in CFG_COLOR.items()]
handles.append(plt.Line2D([], [], color="0.25", lw=3.0, label="battery"))
handles.append(plt.Line2D([], [], color="#EDB120", lw=2.5, ls="--", label="demand"))
ax.legend(handles=handles, loc="lower center", fontsize=9, ncol=5, frameon=True)
fig.tight_layout()
plt.show()
"""),
    ("md", """## What the closed loop reveals

The node escalates to `high_rate` during the storm (the demand spike) when the rate justifies it, then the closed loop reads the depleting charge and the re-solve is gated: `high_rate` becomes infeasible, so the node falls back through `nominal` to `low_power`, and climbs back as solar recovers. No offline plan is followed. Each step is solved against the measured battery state, so the schedule adapts to how the deployment actually unfolds.

This is the myopic (option-a) form of online feedback co-design: re-solve a single static co-design at the live conditions each step. A receding-horizon lookahead that plans several steps ahead with the vector-state DP and commits only the first, and online learning of the co-design model from measurements, are the natural next increments; the model here is known, and measurements update the carried state and conditions rather than the model itself.
"""),
]


NB_23 = [
    ("md", """# 23. Hierarchical co-design for a Formula 1 season (precompute-then-DP)

This notebook reproduces, in the framework's vocabulary, the seasonal co-design of Neumann, Habermacher, Fieni, Cerofolini, Zardini, and Onder, *"Hierarchical Co-Design for Multi-Race Strategy Optimization in Formula 1"* (ITSC 2026). It is the canonical **precompute-then-DP** structure that motivated the `precompute_catalog` / `dp_over_catalog` helpers.

The idea is a clean separation into two layers:

1. **Race-level co-design** (uses the MCDP framework). For each track, battery size, and *incoming battery age*, a `CatalogDP` over energy-deployment strategies emits a Pareto front of `(race_time, wear_increment)`: deploying more electrical energy lowers race time but ages the battery faster, and an aged pack deploys less effectively. This front is solved **once** and frozen.

2. **Season-level dynamic program** (a scalar maximisation MDP). The state is a vector `(w1, w2, ex)`: the fractional wear of the two regulation-permitted battery units plus a flag recording whether a replacement penalty has been incurred. Each race the controls are which unit to run, which frozen implementation (deployment strategy) to pick, and whether to install a fresh unit (a grid penalty: 10 places for the first replacement, 5 for each later one). Race time maps to expected championship points through a time-gap and grid-penalty model integrated against the FIA points table. The DP maximises the season's total expected points.

No co-design solve happens inside the season sweep, which is exactly what distinguishes precompute-then-DP from the re-solving `solve_sequential`.
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'f1', os.path.join(PROJECT_ROOT, 'examples', '23_formula1_season.py'))
f1 = importlib.util.module_from_spec(_spec)
sys.modules['f1'] = f1   # register before exec so @dataclass(frozen=True) resolves
_spec.loader.exec_module(f1)

print("Season:", ", ".join(t.name for t in f1.SEASON))
print("Batteries:", ", ".join(f"{k} ({v} MJ)" for k, v in f1.BATTERIES.items()))
print(f"Wear limit {f1.WEAR_MAX:.0%}, deploy strategies {f1.DEPLOY_STRATEGIES}")
print("FIA points P1..P10:", [f1.FIA_POINTS[p] for p in range(1, 11)])
"""),
    ("md", """## Layer 1: the race Pareto front (the paper's Fig. 1)

Each deployment strategy is a `CatalogDP` entry emitting `(race_time, wear)`. `precompute_catalog` returns the Pareto front through a genuine MCDP solve. Faster race times cost more battery wear, so the whole set is non-dominated, the trade-off curve the season DP later selects along.
"""),
    ("code", """track = f1.SEASON[0]   # Monza
front_new = f1.precompute_catalog(
    [f1.Architecture('4MJ', f1.build_race_dp(track, 4.0, 0.0))],
    {'participate': 0.0}, ['race_time', 'wear'])
front_aged = f1.precompute_catalog(
    [f1.Architecture('4MJ', f1.build_race_dp(track, 4.0, 0.20))],
    {'participate': 0.0}, ['race_time', 'wear'])

print(f"{track.name} / 4MJ Pareto front (new pack):")
for _, p in sorted(front_new, key=lambda x: x[1]['race_time']):
    print(f"   {f1._deploy_label(p['wear']):<12} time={p['race_time']:8.1f}s  wear={p['wear']:.3f}")
"""),
    ("code", """import matplotlib.pyplot as plt

BLUE, ORANGE = "#0072BD", "#D95319"

def front_xy(front):
    pts = sorted((p for _, p in front), key=lambda p: p['wear'])
    return [p['wear'] for p in pts], [p['race_time'] for p in pts]

xn, yn = front_xy(front_new)
xa, ya = front_xy(front_aged)

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(xn, yn, "-o", color=BLUE, lw=2.8, markersize=10, label="new pack (0% wear)",
        markeredgecolor="white", markeredgewidth=1.3)
ax.plot(xa, ya, "-s", color=ORANGE, lw=2.8, markersize=10, label="aged pack (20% wear)",
        markeredgecolor="white", markeredgewidth=1.3)
ax.set_xlabel("battery wear increment  $\\\\Delta w_b$", fontsize=12)
ax.set_ylabel("race time (s)", fontsize=12)
ax.set_title(f"Race co-design Pareto front, {track.name} / 4MJ", fontsize=13)
ax.legend(fontsize=10, frameon=True)
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)
fig.tight_layout()
plt.show()
"""),
    ("md", """The aged pack sits above and to the left: with less usable energy it cannot reach the low race times a fresh pack can, no matter how hard it deploys. This is why incoming battery age is part of the DP state, and why the catalog is precomputed per age bucket.

## Reproducing the paper's figures (same format, illustrative numbers)

**Honest scope.** This section regenerates the paper's three key figures *in the same format*, so the framework's output can be compared side by side with the published ones. It does **not** reproduce the paper's numbers. The paper's fronts come from an optimal-control lap simulation and a battery-health degradation model that are not available here; the position model is fitted in the paper from a decade of FIA race data and the top-four constructors' results. What matches is the **structure**: the shape of the race-time-vs-wear Pareto fronts per (battery, age), the grid-position penalty curve with its saturation and track-difficulty ordering, and the finishing-position density. The numbers below are this example's stylised parameters. Treat this as proof that the framework produces the same *kind* of artefacts the paper reports, not as a numerical replication.

### Fig. 1: race Pareto fronts per (battery, initial age)

Race time on the x-axis (a tight band), wear increment in percent on the y-axis, one line per battery size and initial age (solid/dashed/dotted for 10/20/30% age), with the fastest point (node A, P1) and a representative trade-off (node B) highlighted, matching the paper's Fig. 1 layout.
"""),
    ("code", """import matplotlib.pyplot as plt

fra = f1.Track("Paul Ricard (FRA)", base_time=5060.0, overtake_difficulty=0.4)
fig, ax = plt.subplots(figsize=(8.4, 5.6))
f1.figure1_race_fronts(fra, ax=ax)
fig.tight_layout()
plt.show()
"""),
    ("md", """The layout matches the paper's Fig. 1: two battery sizes, three initial-age curves each, race times clustered in a narrow band, wear on the vertical axis, and the highlighted fastest (A) and trade-off (B) nodes. The absolute time band and wear range differ from the paper (illustrative parameters), but the co-design produces exactly this family of nondominated fronts.

### Fig. 2: grid-start position penalty $\\mu_{pos} \\pm \\sigma_{pos}$

The mean grid-start penalty as a function of the starting slot, with a $\\pm\\sigma$ band, for two tracks. The penalty is zero at the reference slot P3, a bonus for front starts, and saturates beyond ~P12; a hard-to-overtake track (Monaco) sits above an easy one, reproducing the paper's CAN-vs-MON ordering.
"""),
    ("code", """can = f1.Track("Villeneuve (CAN)", base_time=5100.0, overtake_difficulty=0.35)
mon = next(t for t in f1.SEASON if t.name == "Monaco")
fig, ax = plt.subplots(figsize=(8.4, 5.2))
f1.figure2_position_penalty([can, mon], ax=ax)
fig.tight_layout()
plt.show()
"""),
    ("md", """This reproduces the paper's Fig. 2 structure: the zero-crossing at the reference offset, the negative (bonus) region for strong starts, the saturation for poor starts, and Monaco penalising a bad grid slot more than Villeneuve, which is the whole point of the track-difficulty scaling.

### Fig. 3: finishing-position density $\\varphi(\\mathrm{Pos}_{\\mathrm{end}})$

The density over finishing positions for the fastest implementation, for two starting slots (P3 and P6). A worse start shifts the distribution back, as in the paper's Fig. 3.
"""),
    ("code", """fig, ax = plt.subplots(figsize=(8.4, 5.2))
f1.figure3_finishing_distribution(fra, grid_starts=(3, 6), ax=ax)
fig.tight_layout()
plt.show()
"""),
    ("md", """The P3 curve peaks at P1 (consistent with the baseline offset) and the P6 curve shifts back, matching the paper's qualitative Fig. 3 behaviour. The exact shift is milder here because the stylised $\\mu_{pos}$ is smaller than the paper's fitted value.

**Summary of the comparison.** All three figures reproduce the paper's *format and qualitative behaviour* from the framework's own co-design solves and position model. They are not a numerical match, and the notebook labels them as such. This is the honest sense in which the framework "does the same thing": it produces the same structured artefacts (Pareto fronts, penalty curves, finishing densities) that feed the same season-level dynamic program.

## Precompute every race front, once
"""),
    ("code", """catalogs = f1.precompute_race_catalogs(f1.SEASON)
print(f"{len(catalogs)} race fronts precomputed "
      f"({len(f1.SEASON)} tracks x {len(f1.BATTERIES)} batteries x "
      f"{len(f1.wear_buckets())} age buckets).")
print("These are frozen; the season DP below performs no further co-design solve.")
"""),
    ("md", """## Layer 2: solve the season dynamic program

Backward induction over the season, carrying the `(w1, w2, ex)` state and maximising total expected championship points.
"""),
    ("code", """result = f1.solve_season(f1.SEASON, catalogs)
print(f"Optimal season expected points: {result.total_points:.1f}\\n")
print(f"{'race':<12} {'unit':>4} {'batt':>5} {'deploy':>12} {'time':>8} "
      f"{'repl':>5} {'grid':>4} {'E[pts]':>6}")
for d in result.decisions:
    print(f"{d.track:<12} {d.battery_unit:>4} {d.battery_name:>5} "
          f"{d.deploy_name:>12} {d.race_time:>8.1f} "
          f"{'yes' if d.replaced else 'no':>5} {d.grid_start:>4} {d.exp_points:>6.1f}")
"""),
    ("code", """import numpy as np

races = [d.track for d in result.decisions]
pts = [d.exp_points for d in result.decisions]
repl = [d.replaced for d in result.decisions]
colors = [ORANGE if r else BLUE for r in repl]

fig, ax = plt.subplots(figsize=(9.5, 5.0))
bars = ax.bar(range(len(races)), pts, color=colors, edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(races)))
ax.set_xticklabels(races, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("expected championship points", fontsize=12)
ax.set_title(f"Optimal season policy, {result.total_points:.1f} points total", fontsize=13)
ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="race (no replacement)"),
                   Patch(color=ORANGE, label="battery replacement (grid penalty)")],
          fontsize=10, frameon=True)
fig.tight_layout()
plt.show()
"""),
    ("md", """## Finding 1: a local penalty for a global gain

The optimal policy takes a battery replacement at one race, accepting a grid penalty (and a low points haul that day, the orange bar) to keep a fresh, low-wear pack available for aggressive deployment across the remaining races. Sacrificing points locally raises the season total, the multi-race coupling the paper highlights.
"""),
    ("code", """n_repl = sum(1 for d in result.decisions if d.replaced)
repl_races = [d.track for d in result.decisions if d.replaced]
print(f"replacements: {n_repl} at {repl_races}")
print(f"season total: {result.total_points:.1f} expected points")
"""),
    ("md", """## Finding 2: race order shifts the optimal policy

The paper reports that race order does not change the attainable total reward but does change the optimal control policy. In this stylised model the grid-penalty cost is track-dependent (a replacement hurts far more at a hard-to-overtake track like Monaco than at Monza), so reordering the calendar lets the optimiser place the replacement on a cheaper track. The total is therefore *near*-invariant, and the policy adapts to the order, the temporal coupling of the multi-stage decision.
"""),
    ("code", """reversed_season = list(reversed(f1.SEASON))
rev_catalogs = f1.precompute_race_catalogs(reversed_season)
rev_result = f1.solve_season(reversed_season, rev_catalogs)

print(f"forward season total  = {result.total_points:.1f}")
print(f"reversed season total = {rev_result.total_points:.1f}  (near-invariant)")
print(f"forward replaces at  {[d.track for d in result.decisions if d.replaced] or 'never'}")
print(f"reversed replaces at {[d.track for d in rev_result.decisions if d.replaced] or 'never'}")
print("\\nThe optimal replacement moves to a cheaper-penalty track under reordering,")
print("illustrating the temporal coupling the paper emphasises.")
"""),
    ("md", """## Correctness

The season DP is validated against exhaustive brute-force enumeration on small instances in `tests/test_formula1.py` (the DP's optimal value matches the brute-force optimum exactly, for both single- and mixed-battery unit configurations). The race co-design fronts are genuine MCDP solves, aged packs deploy less, and the catalogs are correctly indexed by incoming battery age. This is the precompute-then-DP structure of the F1 paper, with the co-design layer supplied by the framework and the season MDP written out as a scalar-maximisation backward induction.
"""),
]


NB_24 = [
    ("md", """# 24. Catalog-driven car co-design from a single architecture table

Notebook **17** modelled the full-vehicle co-design with three hand-wired builders (`build_ice_car`, `build_hybrid_car`, `build_ev_car`), one per powertrain family. Each builder wires its own subsystem graph and knows, in Python, which modules an ICE, a hybrid, or an EV needs. That is expressive but it hard-codes the architecture space into three functions.

This notebook takes the *catalog-driven* stance instead. A single **12-row `ARCHITECTURE_CATALOG`** enumerates the modern powertrain spectrum -- pure ICE, diesel, mild hybrid (MHEV), full hybrid (FHEV), plug-in hybrid (PHEV), range-extender EV (REEV), and battery-electric (BEV) -- and **one** `build_architecture()` function assembles a solvable `System` for *any* row. The discrete powertrain choices (engine, transmission, e-motor) become one-entry `CatalogDP` slices of master catalogs; the parametric modules (cooling, fuel, battery, brakes, suspension, tyres, ...) size themselves from mission demand and from the converged curb mass. The body, suspension, and tyres stay full `CatalogDP`s, so the solver still picks the cheapest chassis that satisfies the mission.

The result is that adding an architecture is adding a row to a table, not writing a new builder. The same two coupled cycles as notebook 17 are closed by the Kleene iteration: the **mass spiral** (subsystem weights sum to curb weight, which feeds back as the design mass every load-bearing module must support) and the **BEV energy spiral** (battery sized to range x consumption, whose own mass raises consumption).

One design decision matters for the numbers: the energy metric is *unified primary energy* per 100 km (fuel lower-heating-value **plus** battery/grid electricity). This is why an ICE lands near 45-55 kWh/100 km while a BEV lands near 15 kWh/100 km -- the combustion path throws away most of the fuel's chemical energy as heat. All the calibration numbers are internet-sourced; see the `SOURCES` block in the example's module docstring (BNEF 2024 pack prices, pack-level Wh/kg, fuel LHV/CO2, Corolla curb-mass bracket).
"""),
    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'car_catalog', os.path.join(PROJECT_ROOT, 'examples', '24_car_catalog_codesign.py'))
car = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(car)

from codesign import solve
print("Module loaded.")
print(f"  Architecture rows : {len(car.ARCHITECTURE_CATALOG)}")
print(f"  Engine catalog    : {len(car.ENGINE_CATALOG)}")
print(f"  Transmission cat. : {len(car.TRANSMISSION_CATALOG)}")
print(f"  E-motor catalog   : {len(car.EMOTOR_CATALOG)}")
print(f"  Body catalog      : {len(car.BODY_CATALOG)}")
print(f"  Suspension catalog: {len(car.SUSPENSION_CATALOG)}")
print(f"  Tire catalog      : {len(car.TIRE_CATALOG)}")"""),
    ("md", """## The mission

One representative compact C-segment mission drives the whole study: 5 seats, 370 L of cargo, 170 km/h top speed, 500 km of range, 0-100 km/h in 11.5 s. The targets are bracketed by the 2024 Toyota Corolla figures cited in the example's SOURCES block.
"""),
    ("code", """print("Mission:", car.mission_str(car.MISSION))
for k, v in car.MISSION.items():
    print(f"  {k:<20} = {v}")"""),
    ("md", """## The architecture catalog

Each row pre-selects the discrete powertrain (an index into the engine, transmission, and e-motor catalogs) plus its energy strategy (target battery kWh, on-board charger, ICE cruise fraction, AWD/diesel flags). `ARCH_CLASS` groups the twelve rows into the six powertrain classes used by the comparison table at the end.
"""),
    ("code", """print(f"{'architecture':<18}{'class':<7}{'engine':<16}{'e-motor':<13}"
      f"{'batt kWh':>9}{'ice_f':>7}")
print("-" * 70)
for entry in car.ARCHITECTURE_CATALOG:
    arch = car.architecture_by_name(entry[0])
    eng = car.ENGINE_CATALOG[arch['engine_idx']][0]
    em  = car.EMOTOR_CATALOG[arch['emotor_idx']][0]
    print(f"{arch['name']:<18}{car.ARCH_CLASS[arch['name']]:<7}{eng:<16}{em:<13}"
          f"{arch['target_battery_kWh']:>9.0f}{arch['ice_fraction']:>7.2f}")"""),
    ("md", """## Solve the whole table

`solve_architecture` takes one architecture row and sweeps the eligible chassis catalog (body x suspension x tyre), building a single-valued `System` for each combination, closing the mass and energy spirals with `solve()`, and keeping the cheapest feasible converged design. We run it for all twelve rows and print the feasibility table -- the cheapest feasible design per architecture, with its curb mass, unified energy/100 km, fuel/100 km, and tailpipe CO2.
"""),
    ("code", """import time
t0 = time.time()

solved = []                 # (arch_name, best_point) for feasible rows
table  = []                 # (name, class, feasible, best) for every row
for entry in car.ARCHITECTURE_CATALOG:
    arch = car.architecture_by_name(entry[0])
    feasible, best = car.solve_architecture(arch, car.MISSION)
    if feasible:
        solved.append((arch['name'], best))
    table.append((arch['name'], car.ARCH_CLASS[arch['name']], feasible, best))
elapsed = time.time() - t0

header = (f"  {'Architecture':<18}{'cls':<6}{'feas':<6}"
          f"{'cost$':>9}{'mass kg':>9}{'kWh/100':>9}{'L/100':>7}{'CO2':>7}")
print(header)
print("  " + "-" * (len(header) - 2))
for name, cls, feasible, best in table:
    if not feasible:
        print(f"  {name:<18}{cls:<6}{'no':<6}{'-':>9}{'-':>9}{'-':>9}{'-':>7}{'-':>7}")
        continue
    print(f"  {name:<18}{cls:<6}{'yes':<6}"
          f"{best['production_cost_USD']:>9,.0f}{best['curb_weight_kg']:>9.0f}"
          f"{best['energy_per_100km_kWh']:>9.1f}{best['fuel_per_100km_L']:>7.1f}"
          f"{best['co2_per_km']:>7.0f}")
print(f"\\n{len(solved)}/{len(car.ARCHITECTURE_CATALOG)} architectures feasible "
      f"(whole table swept in {elapsed:.1f}s)")"""),
    ("md", """## The Pareto front over (cost, energy, mass)

The cheapest feasible design of each architecture is one point in a three-objective space: production cost, unified energy per 100 km, and curb mass. `_pareto` keeps the non-dominated ones -- the designs for which no other architecture is at least as good on all three axes and strictly better on one.
"""),
    ("code", """axes = ("production_cost_USD", "energy_per_100km_kWh", "curb_weight_kg")
front = car._pareto(solved, axes)
front_names = {name for name, _ in front}
print("Pareto front over (cost, energy/100km, mass):")
for name, p in sorted(front, key=lambda t: t[1]['production_cost_USD']):
    print(f"  {name:<18} ${p['production_cost_USD']:>8,.0f}"
          f"  {p['energy_per_100km_kWh']:>5.1f} kWh/100"
          f"  {p['curb_weight_kg']:>5.0f} kg")"""),
    ("code", """import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# MATLAB gem colours, one per powertrain class.
CLASS_COLOR = {"ICE": "#0072BD", "MHEV": "#4DBEEE", "FHEV": "#77AC30",
               "PHEV": "#EDB120", "REEV": "#D95319", "BEV": "#7E2F8E"}

fig, ax = plt.subplots(figsize=(9.0, 5.6))
for name, p in solved:
    cls = car.ARCH_CLASS[name]
    on_front = name in front_names
    ax.scatter(p['energy_per_100km_kWh'], p['production_cost_USD'],
               s=260 if on_front else 90,
               marker="*" if on_front else "o",
               color=CLASS_COLOR[cls],
               edgecolor="black" if on_front else "white",
               linewidth=1.4 if on_front else 0.8,
               zorder=4 if on_front else 2)
    # Label only the Pareto-optimal (star) points -- they are well separated,
    # so their names never collide. The clustered non-front designs (several
    # near-identical ICE/MHEV variants) are left to the class legend rather
    # than overprinting each other; the feasibility table above names them.
    if on_front:
        ax.annotate(name, (p['energy_per_100km_kWh'], p['production_cost_USD']),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=9, color="0.2", fontweight="bold")
ax.set_xlabel("unified primary energy  (kWh / 100 km)", fontsize=12)
ax.set_ylabel("production cost  (USD)", fontsize=12)
ax.set_title("Cheapest feasible design per architecture "
             "(stars = Pareto-optimal)", fontsize=13)
ax.margins(0.12)
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.tick_params(labelsize=11)

handles = [Line2D([], [], marker="o", linestyle="none", color=c, label=cls,
                  markeredgecolor="white") for cls, c in CLASS_COLOR.items()]
handles.append(Line2D([], [], marker="*", linestyle="none", color="0.4",
                      markersize=14, markeredgecolor="black", label="Pareto front"))
ax.legend(handles=handles, fontsize=9, frameon=True, ncol=2, loc="upper right")
fig.tight_layout()
plt.show()"""),
    ("md", """## Comparison by powertrain class

Collapsing to the cheapest feasible design in each of the six classes reads off the headline trade: cost climbs and tailpipe CO2 falls as the architecture electrifies, with the REEV and BEV eliminating (or nearly eliminating) tailpipe emissions at the top of the cost range.
"""),
    ("code", """print(f"  {'class':<6}{'best arch':<18}{'cost$':>9}{'mass kg':>9}"
      f"{'kWh/100':>9}{'CO2 g/km':>10}")
by_class = {}
for name, p in solved:
    cls = car.ARCH_CLASS[name]
    if cls not in by_class or \\
       p['production_cost_USD'] < by_class[cls][1]['production_cost_USD']:
        by_class[cls] = (name, p)
for cls in ("ICE", "MHEV", "FHEV", "PHEV", "REEV", "BEV"):
    if cls not in by_class:
        continue
    name, p = by_class[cls]
    print(f"  {cls:<6}{name:<18}{p['production_cost_USD']:>9,.0f}"
          f"{p['curb_weight_kg']:>9.0f}{p['energy_per_100km_kWh']:>9.1f}"
          f"{p['co2_per_km']:>10.0f}")"""),
    ("md", """## What the catalog-driven approach delivers

Compared with notebook 17's three hand-wired builders, this study spans the *same* architectural breadth -- and more (it adds MHEV, FHEV, PHEV, and REEV as distinct rows) -- from a single table and a single `build_architecture()` function. Every powertrain choice is a `CatalogDP` entry rather than a bespoke `Module`, and the mass and BEV-energy spirals are closed by the same Kleene solver.

Three patterns fall out of the Pareto front and the class comparison:

1. **ICE wins on upfront cost**, by a wide margin, because it carries no traction battery or power electronics. The cheapest feasible ICE sits well below any electrified option.

2. **Unified primary energy falls monotonically with electrification.** The ICE burns roughly three times the primary energy per 100 km of the BEV, because combustion rejects most of the fuel's chemical energy as heat -- the unified metric makes that visible where an "L/100 km vs kWh/100 km" comparison would hide it.

3. **Tailpipe CO2 tracks the ICE cruise fraction**, reaching zero for the pure BEVs. The PHEV/REEV rows interpolate: a small battery plus an engine buys most of the CO2 reduction at a fraction of the BEV's cost and mass.

The framework's contribution is the same as in notebook 17 -- every subsystem R port is exposed in the antichain and every constraint is one line -- but here the *architecture space itself* is data. Widening the study is editing a table, and the same monotone-resource machinery resolves each row's coupled spirals automatically. The calibration numbers are internet-sourced (see the example's `SOURCES` block); they are illustrative, not OEM-specific, and it is the co-design framework that is being validated.
"""),
]


# ---------------------------------------------------------------------------
# NB 25: reproducing the online co-design paper's synthetic benchmarks
# ---------------------------------------------------------------------------

NB_25 = [
    ("md", """# 25. Reproducing the online co-design paper's synthetic benchmarks

This notebook is a *replication study*. It re-runs the two synthetic benchmark families from Alharbi, Dahleh & Zardini, **"Compositional Online Learning for Multi-Objective System Co-Design"** (arXiv:2604.22624, 2026), and reports honestly on which of the paper's claims reproduce and which do not.

The paper studies online multi-objective decision-making in *monotone co-design*: functionalities and resources are partially ordered, and the agent must recover the target-feasible antichain of non-dominated resources using as few expensive evaluations as possible. Its engine is **Algorithm 1** (rejection sampler with optimistic evaluators): draw a candidate from a low-discrepancy base measure, compute history-dependent *optimistic* bounds on its resource and functionality, and skip (reject) it without evaluation when either its optimistic resource is already dominated by the incumbent (eq. 13) or its optimistic functionality can never meet the target (eq. 14). Only survivors are actually queried.

**What this example replicates** are the two synthetic families of Section VII-B:

- **E1 / Monotone** (Table I): a monotone step map `g: [0,1]^3 -> [0,1]^2`. The optimistic bound is the monotone join of eqs. (23)-(24), i.e. `codesign.online.MonotonicityEvaluator`.
- **E2 / Lipschitz** (Table II): an `L=2`-Lipschitz triangle-wave map `g: [0,1]^4 -> [0,1]^2`. The optimistic bound is the Lipschitz cone of eq. (25), i.e. `codesign.online.LipschitzEvaluator` with `L=2`.

The library's evaluators supply the optimistic lower bounds (their `bound()` reproduces eqs. 23-25 exactly); Algorithm 1 itself -- the scrambled-Halton base measure, the two elimination conditions, and the forced-acceptance knob `delta` -- is implemented in the example, because the library's `solve_online` *scans* a candidate list with an upper-confidence picker whereas the paper *samples* a base measure with rejection.

**What is not replicated**: E3 (intermodal mobility, a large multi-commodity-flow LP) and E4 (heterogeneous multi-robot, a planner-executor simulation) -- neither expensive block is in this repository. There is no public code/data release and the exact seeds, grid, and atom count `K` are unpublished, so this replication is *statistical, not bit-exact*: the precise table entries cannot be reproduced, only the qualitative claims.
"""),

    ("md", "## Imports and module load"),
    ("code", """import importlib.util, os, sys, time
PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    'ex25', os.path.join(PROJECT_ROOT, 'examples', '25_online_paper_benchmarks.py'))
ex25 = importlib.util.module_from_spec(_spec)
sys.modules['ex25'] = ex25
_spec.loader.exec_module(ex25)

print("Module loaded.")
print(f"  instances (M1..M8 / L1..L8) : {ex25.N_INSTANCES}")
print(f"  runs per instance           : {ex25.RUNS}")
print(f"  budget  (monotone/lipschitz): {ex25.ITERS_MONO} / {ex25.ITERS_LIP}")
print(f"  K atoms (unpublished)       : {ex25.K_ATOMS}")
print(f"  Lipschitz L                 : {ex25.LIPSCHITZ_L}")
print(f"  forced-acceptance delta     : {ex25.DELTA}")
try:
    import pymoo
    print(f"  pymoo {pymoo.__version__} present -> EA panel (NSGA-III / MOEA/D / RVEA) will run")
except Exception:
    print("  pymoo absent -> EA panel skipped")"""),

    ("md", """## The honest deviation on the monotone family

Before running, one caveat the example is careful to document. The paper says the target functionality is "fixed and satisfied by construction, so the learning problem reduces to recovering the non-dominated antichain of a resource map `g`". Taken literally this is *degenerate* for the monotone family: a monotone `g` is minimized at the bottom of the box, so its unconstrained minimal antichain collapses to the single point `g(0)=(0,0)` and there is nothing to learn.

The example therefore keeps the Lipschitz family as the literal unconstrained minimization, and for the monotone family makes the functionality target **binding**: it recovers `FixFunMinRes(f)`, the minimal antichain of `g` restricted to the feasible upper set `{x : g(x) >= f}`. This is the standard co-design query, it exercises *both* elimination conditions (13) and (14), and it makes the monotone benchmark non-degenerate. The Lipschitz family is unaffected. The metric in both panels is cumulative hypervolume difference (lower is better), summed over iterations exactly as in the paper's tables.
"""),

    ("md", """## Run the benchmark

We call the example's `run_family` driver directly, once per family, with the EA panel enabled. At the trimmed default scale this takes on the order of ten seconds; the module-level constants (`RUNS`, `ITERS_MONO`, `ITERS_LIP`) crank up to the paper's 100-run, 4000/2000-iteration scale.
"""),
    ("code", """t0 = time.time()

mono_means, mono_curves = ex25.run_family(
    "Monotone", ex25.make_monotone_map, d=3,
    feature_names=["x0", "x1", "x2"],
    make_evaluator=lambda ff: ex25.MonotonicityEvaluator(ff, list(ex25.RC)),
    constrained=True, budget=ex25.ITERS_MONO, with_ea=True)

lip_means, lip_curves = ex25.run_family(
    "Lipschitz", ex25.make_lipschitz_map, d=4,
    feature_names=["x0", "x1", "x2", "x3"],
    make_evaluator=lambda ff: ex25.LipschitzEvaluator(ff, list(ex25.RC),
                                                      L=ex25.LIPSCHITZ_L),
    constrained=False, budget=ex25.ITERS_LIP, with_ea=True)

print(f"\\nbenchmark runtime: {time.time() - t0:.1f} s")"""),

    ("md", """## Table I -- Monotone problems (E1)

Per-instance mean cumulative HVD for each method across the eight instances. `Ours` is Algorithm 1 with the monotone optimistic bound; `Halton` uses the same low-discrepancy proposals but *no* elimination; `Random` draws uniformly from the shared pool; the EA rows (if pymoo is present) run pymoo's NSGA-III / MOEA/D / RVEA over the continuous box.
"""),
    ("code", """ex25.print_panel("TABLE I  -- Monotone problems (E1), FixFunMinRes(f)",
                 "paper Table I OURS row: 8.81e1 .. 3.46e1", "M", mono_means)"""),

    ("md", """## Table II -- Lipschitz problems (E2)

The same panel for the `L=2` Lipschitz family (pure unconstrained minimization).
"""),
    ("code", """ex25.print_panel("TABLE II -- Lipschitz problems (E2), pure minimization",
                 "paper Table II OURS row: 5.89e1 .. 4.20e1", "L", lip_means)"""),

    ("md", """## Validation: which claims reproduce?

The example checks the paper's *reproducible qualitative* claims and prints per-claim verdicts. The absolute table entries are **not** bit-reproducible (unpublished seeds / grid / `K`); only the relative claims are testable.
"""),
    ("code", "ex25.validate(mono_means, lip_means)"),

    ("md", """### The example's own honest validation note

The critical outcome is claim (c). Quoting the example module's `Validation status` docstring verbatim -- this is an independent re-run at higher statistical strength (WP-P3), and the outcome is *not* softened:
"""),
    ("code", """doc = ex25.__doc__
start = doc.index("Validation status")
end = doc.index("Run\\n---")
print(doc[start:end].rstrip())"""),

    ("md", """## The HVD convergence figures

Mean hypervolume difference versus iteration (expensive evaluations), averaged over all instances and runs, on a log-y axis. `Ours` (red) should sit below `Halton` (blue dashed) throughout -- the claim (a)/(b) signal. The EA curves optimize the *continuous* box while Ours/Halton/Random are confined to the shared discrete pool, an asymmetry that favours the EAs on the smooth Lipschitz map and underlies the (c) non-reproduction.
"""),
    ("code", """import matplotlib.pyplot as plt
import numpy as np

STYLES = {"Ours": ("C3", "-", 2.4), "Halton": ("C0", "--", 1.6),
          "Random": ("C7", ":", 1.6), "NSGA-III": ("C2", "-.", 1.3),
          "MOEA/D": ("C1", "-.", 1.3), "RVEA": ("C4", "-.", 1.3)}

def plot_family(family, mean_curves, ax):
    for m, curve in mean_curves.items():
        if curve is None:
            continue
        color, ls, lw = STYLES.get(m, ("C5", "-", 1.2))
        x = np.arange(1, len(curve) + 1)
        y = np.clip(curve, 1e-6, None)   # floor for the log axis
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=m)
    ax.set_yscale("log")
    ax.set_xlabel("iteration (expensive evaluations)", fontsize=11)
    ax.set_ylabel("HV(A_ref) - HV(A_hat)", fontsize=11)
    ax.set_title(f"{family}: mean HVD over "
                 f"{ex25.N_INSTANCES} instances x {ex25.RUNS} runs", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
plot_family("Monotone (E1)", mono_curves, axes[0])
plot_family("Lipschitz (E2)", lip_curves, axes[1])
fig.tight_layout()
plt.show()"""),

    ("md", """## Takeaways

- **Claims (a) and (b) reproduce robustly.** On both families `Ours` (Algorithm 1 with the certified optimistic bound) beats plain `Halton` on the shared discrete pool -- exactly the elimination-vs-no-elimination comparison the paper makes, and the apples-to-apples one. In the higher-strength WP-P3 re-run, `Ours` was best-of-all on *every* monotone instance, even stronger than the paper's 7/8.

- **Claim (c) did NOT reproduce.** On the smooth Lipschitz family the continuous EAs (MOEA/D in particular) match or beat `Ours`, and MOEA/D shows *no* blow-ups here (max/median only ~1.5-4x, not the paper's ~20x on L8). The leading explanation is structural: the EA panel optimizes the continuous `[0,1]^d` box while Ours/Halton/Random are confined to a fixed discrete pool, an asymmetry that favours the EAs on smooth maps (their HVD is clipped at 0 against a discrete reference they can actually beat); the budget is also far below paper scale and the EAs are untuned. This is reported as measured and labelled **NOT REPRODUCED** rather than dressed up as a fragile PASS.

- **Why the honest split matters.** Claims (a)/(b) isolate the effect the paper is really about -- the value of structural optimism on a shared candidate pool -- and they hold cleanly. Claim (c) is a cross-paradigm comparison (rejection sampler on a discrete pool vs. continuous evolutionary search) that is sensitive to the harness, and the harness here does not reproduce it. A replication study earns its keep precisely by drawing that line.
"""),
]


def main():
    plan = [
        ("01_drone.ipynb", NB_01),
        ("02_integer_optimization.ipynb", NB_02),
        ("03_auv_seabed.ipynb", NB_03),
        ("04_uncertain_and_ode.ipynb", NB_04),
        ("05_visualize_kleene.ipynb", NB_05),
        ("06_drone_mcdpl_syntax.ipynb", NB_06),
        ("07_drone_modular.ipynb", NB_07),
        ("08_vehicle_modular.ipynb", NB_08),
        ("09_robotic_arm.ipynb", NB_09),
        ("10_solver_trace.ipynb", NB_10),
        ("11_uncertain_drone.ipynb", NB_11),
        ("12_stochastic_drone.ipynb", NB_12),
        ("13_microgrid.ipynb", NB_13),
        ("14_online_fleet.ipynb", NB_14),
        ("15_bioprocess.ipynb", NB_15),
        ("16_online_doe.ipynb", NB_16),
        ("17_car_codesign.ipynb", NB_17),
        ("18_metabolic_switching.ipynb", NB_18),
        ("19_rover_modules.ipynb", NB_19),
        ("20_sequential_codesign.ipynb", NB_20),
        ("21_reconfigurable_robot.ipynb", NB_21),
        ("22_online_feedback_codesign.ipynb", NB_22),
        ("23_formula1_season.ipynb", NB_23),
        ("24_car_catalog_codesign.ipynb", NB_24),
        ("25_online_paper_benchmarks.ipynb", NB_25),
    ]
    for name, cells in plan:
        write(name, cells)
    print(f"\nAll {len(plan)} notebooks built and executed in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
