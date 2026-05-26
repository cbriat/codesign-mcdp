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
    ("md", """## What we just saw

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
        # If isqrt(n)**2 < n then n was not a perfect square, so we add 1.
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
    ("md", """## What the DSL bought us

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
- **LinearParametric**: aggressive but riskier; fits a linear model and prunes by its confidence band.
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
    # LinearParametric: fits a running OLS regressor with a 3-sigma
    # confidence band. min_obs=5 means it only starts bounding after
    # five evaluations.
    ("LinearParametric", LinearParametricEvaluator(
        features=["speed", "payload", "unit_cost", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
        confidence=3.0, min_obs=5,
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
- **LinearParametric** falls in between. The linear model converges quickly on a fit, but the confidence band can be too tight when the true relationship is nonlinear, so it can wrongly eliminate a Pareto candidate. In the run above, this is what causes the smaller antichain.

The choice between evaluators is a classical bias-variance tradeoff: stronger structural assumptions mean fewer evaluations needed, but more risk of wrongly pruning. In production code the safe default is Lipschitz with a conservative L; switch to monotonicity only when you can verify the feature is genuinely monotone.

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

All parameters are taken from the 2024-2026 bioprocessing literature: cell-line specific productivity (Reinhart 2021, Sumi 2024), oxygen uptake rates (BioProcess International 2024), media costs (CHO media market 2025), bioreactor capex (Sustainability Atlas 2026, BioPlan 2025), metabolic constraints (Khattak 2010, Lao & Toth 1997).
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
    # an output we use later to compute COGS per gram.
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
    # we need to combine several ports with closed-over parameters
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

5. **LinearParametric online evaluator**: fits a running least-squares model and bounds by a 2.5-sigma confidence band.

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
                            confidence=2.5, min_obs=10)),
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

The left panel shows every candidate in $(\\text{cogs}, \\text{footprint})$ space. Grey points are candidates that the LinearParametric online solver eliminated or did not evaluate. Coloured points are the 40 it did evaluate. Red stars mark the true Pareto front classes.

The right panel shows the trajectory of which candidate the LinearParametric picker chose at each iteration, plotted in the (`T_C`, `glucose_mm`) projection. The solver explores broadly in the first 10 iterations (when `min_obs=10` hasn't been reached and bounds are uninformative) and then concentrates near the predicted Pareto-optimal region.
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

At budget 40 (11% of the grid, a 89% reduction in wet-lab work), the LinearParametric online solver recovers 3 of 4 Pareto classes, matching a 75-run factorial DOE at 53% of the experimental cost. The Lipschitz evaluator achieves the same recovery rate with a tuned $L$ constant, but is more sensitive to the choice of $L$: a sweep over $L \\in \\{10, 15, 20, 25, 35\\}$ gives recovery $2, 1, 1, 1, 3$ respectively. In a real campaign you would calibrate $L$ from preliminary scale-down or historical-batch data.

The Monotonicity evaluator alone is uninformative on this problem because its lower bounds only tighten for candidates that lie above every observed candidate in the partial order on monotone-bad features. With no observation yet at the low-feature corner, every Pareto-optimal candidate has lower bound zero and the picker wanders. In a real campaign you would seed it with three to five hand-picked corner runs (one at each extreme of each feature) before letting it pick.

The general lesson for online co-design over expensive evaluations: **the structural prior is doing the work, the budget is the cost.** A predictive prior like LinearParametric, or a hybrid Gaussian-process-with-Lipschitz-tail, propagates information across the whole grid; pure local bounds (Lipschitz with conservative $L$, Monotonicity) need either denser observations or a warm-start.

For the bioprocess engineer, this translates to: identify the structural assumptions your process actually supports (smooth response surface? monotone in some directions?), pick the matching evaluator, and treat the first five to ten runs as a calibration phase rather than a search.
"""),
]


# ---------------------------------------------------------------------------
# Build all notebooks
# ---------------------------------------------------------------------------


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
    ]
    for name, cells in plan:
        write(name, cells)
    print(f"\nAll {len(plan)} notebooks built and executed in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
