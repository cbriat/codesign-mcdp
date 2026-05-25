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
    Antichain, FunctionDP, Loop, NamedProduct, Reals, solve,
)"""),
    ("md", """## Physical constants

These follow Fig. 48 of the paper: a Li-ion battery (1.8 MJ/kg), gravity, and a quadratic drag coefficient for the actuator (10 W per N² of lift)."""),
    ("code", """ALPHA = 1.8e6      # Li-ion specific energy, J/kg
G = 9.81           # gravity, m/s^2
C_LIFT = 10.0      # actuator coefficient, W per N^2 of lift"""),
    ("md", """## The inner design problem

The functionality is `(endurance, extra_payload, extra_power, battery_mass)` and the resource is `(battery_mass, report_mass)`. Note that `battery_mass` appears on *both* sides: the inner DP receives the current iterate as a functionality input and emits a tightened estimate as a resource output. The `report_mass` is a mirrored copy of the same value so the outer R retains visibility of it (the `Loop` operator projects out the loop axis).
"""),
    ("code", """F = NamedProduct({
    "endurance": Reals(unit="s"),
    "extra_payload": Reals(unit="kg"),
    "extra_power": Reals(unit="W"),
    "battery_mass": Reals(unit="kg"),
})
R = NamedProduct({
    "battery_mass": Reals(unit="kg"),
    "report_mass": Reals(unit="kg"),
})

def h(f):
    if (f["battery_mass"] == math.inf or
        f["endurance"] == math.inf or
        f["extra_payload"] == math.inf or
        f["extra_power"] == math.inf):
        return Antichain.singleton(R, {
            "battery_mass": math.inf, "report_mass": math.inf,
        })
    lift = (f["battery_mass"] + f["extra_payload"]) * G
    actuator_power = C_LIFT * lift * lift
    total_power = actuator_power + f["extra_power"]
    energy = total_power * f["endurance"]
    mass = energy / ALPHA
    return Antichain.singleton(R, {
        "battery_mass": mass, "report_mass": mass,
    })

inner = FunctionDP(F=F, R=R, h_fn=h, name="drone")
drone = Loop(inner, axis="battery_mass")
drone"""),
    ("md", """## Solving for several mission profiles

We sweep over short, medium, longer, marginal, and clearly-infeasible missions. The marginal and infeasible cases are correctly flagged: the loop axis is driven to `⊤` (infinity) when the recursion does not close on a finite battery mass.
"""),
    ("code", """cases = [
    ("Short, light",   dict(endurance=60.0,   extra_payload=0.10, extra_power=1.0)),
    ("Medium, modest", dict(endurance=300.0,  extra_payload=0.50, extra_power=5.0)),
    ("Longer mission", dict(endurance=600.0,  extra_payload=0.50, extra_power=5.0)),
    ("Marginal",       dict(endurance=600.0,  extra_payload=1.00, extra_power=10.0)),
    ("Infeasible",     dict(endurance=1800.0, extra_payload=1.00, extra_power=10.0)),
]
for label, f in cases:
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
    Antichain, FunctionDP, Loop, NamedProduct, Naturals, solve,
)"""),
    ("md", """## The model

The inner DP enumerates every splitting $(c_1, c_2)$ of the deficit into the two coordinates, giving the antichain of points $(x, y)$ with $x + y$ exactly meeting the constraint. The `Loop` on the axis `xy` closes $x_{out} \\geq x_{in}$ and $y_{out} \\geq y_{in}$ simultaneously.
"""),
    ("code", """def make_looped(c_value: int):
    N = Naturals()
    XY = NamedProduct({"x": N, "y": N})
    F = NamedProduct({"c": N, "xy": XY})
    R = NamedProduct({"xy": XY, "xy_report": XY})

    def h(f):
        c = int(f["c"])
        x_in, y_in = f["xy"]["x"], f["xy"]["y"]
        if x_in == math.inf or y_in == math.inf:
            top = {"x": math.inf, "y": math.inf}
            return Antichain.singleton(R, {"xy": top, "xy_report": top})

        sx = math.isqrt(int(x_in)) + (1 if math.isqrt(int(x_in)) ** 2 < int(x_in) else 0)
        sy = math.isqrt(int(y_in)) + (1 if math.isqrt(int(y_in)) ** 2 < int(y_in) else 0)
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

    inner = FunctionDP(F=F, R=R, h_fn=h, name=f"sqrt_sum(c={c_value})")
    return Loop(inner, axis="xy")"""),
    ("md", "## Run with trace"),
    ("code", """def pretty(p):
    return f"({p['xy_report']['x']}, {p['xy_report']['y']})"

def run(c_value, show_trace=True):
    looped = make_looped(c_value)
    result = solve(looped, {"c": c_value}, max_iter=50, record_trace=show_trace)
    print(f"c = {c_value}: iters = {result.iterations}, feasible = {result.feasible}")
    if show_trace and result.trace:
        for k, entry in enumerate(result.trace):
            A = entry.antichain
            pts = ", ".join(f"({p['xy']['x']}, {p['xy']['y']})" for p in A.points)
            print(f"   S_{k}: {{ {pts} }}")
    pts = ", ".join(pretty(p) for p in result.antichain.points)
    print(f"   M(c={c_value}) = {{ {pts} }}\\n")
    return result

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
    Antichain, FunctionDP, Loop, NamedProduct, Reals,
    solve, minimize_cost,
)"""),
    ("md", "## Build the AUV model"),
    ("code", """def make_auv():
    K_GEOM = 1.0; V_MAX = 3.0; R_MAX = 5.0
    PSI_A = 30.0; CHI_A = 50.0; SENSOR_COST_A = 200.0

    Design = NamedProduct({"v": Reals(unit="m/s"), "r": Reals(unit="m")})
    F = NamedProduct({"A": Reals(unit="m^2"), "design": Design})
    R = NamedProduct({
        "design": Design,
        "T": Reals(unit="s"), "E": Reals(unit="J"), "cost": Reals(unit="$"),
    })

    def h(f):
        A = f["A"]
        v_in, r_in = f["design"]["v"], f["design"]["r"]
        if v_in == math.inf or r_in == math.inf:
            return Antichain.singleton(R, {
                "design": {"v": math.inf, "r": math.inf},
                "T": math.inf, "E": math.inf, "cost": math.inf,
            })
        v = max(float(v_in), 0.1); r = max(float(r_in), 0.5)
        if v > V_MAX or r > R_MAX:
            return Antichain.singleton(R, {
                "design": {"v": math.inf, "r": math.inf},
                "T": math.inf, "E": math.inf, "cost": math.inf,
            })

        pts = []
        for v_try in (v, min(v*1.3, V_MAX), min(v*1.7, V_MAX)):
            for r_try in (r, min(r*1.3, R_MAX), min(r*1.7, R_MAX)):
                if v_try < v_in or r_try < r_in:
                    continue
                T_try = K_GEOM * A / (v_try * r_try)
                E_try = (PSI_A * v_try**3 + CHI_A * r_try) * T_try
                cost_try = SENSOR_COST_A * r_try
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
    for p in result.antichain.points:
        print(f"   T={p['T']:.0f}s, E={p['E']/1000:.1f}kJ, $={p['cost']:.0f}")
    best = minimize_cost(
        result,
        cost_fn=lambda r: r["T"] + 0.05 * (r["E"] / 1000.0) + r["cost"],
    )
    if best is not None:
        print(f"   best composite: T={best['T']:.0f}s, "
              f"E={best['E']/1000:.1f}kJ, $={best['cost']:.0f}")
    print()

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
    AlgebraicDP, NamedProduct, ODE_DP, Reals, UncertainDP, solve,
)"""),
    ("md", """## UncertainDP demo: battery with uncertain specific energy

Old Li-ion cells average 1.6 MJ/kg; newer ones 2.0 MJ/kg. We bracket the unknown true value with the two limits and solve in both modes.
"""),
    ("code", """F = NamedProduct({"capacity": Reals(unit="J")})
R = NamedProduct({"mass": Reals(unit="kg")})

pessimistic = AlgebraicDP(
    F=F, R=R,
    equations={"mass": lambda f: f["capacity"] / 1.6e6},
    name="battery_pessimistic",
)
optimistic = AlgebraicDP(
    F=F, R=R,
    equations={"mass": lambda f: f["capacity"] / 2.0e6},
    name="battery_optimistic",
)
uncertain = UncertainDP(F=F, R=R, lower=optimistic, upper=pessimistic, mode="upper")

print("Battery sizing under specific-energy uncertainty (1 kWh capacity):")
for mode in ("lower", "upper"):
    result = solve(uncertain.with_mode(mode), {"capacity": 3.6e6})
    mass = list(result.antichain.points)[0]["mass"]
    label = "optimistic" if mode == "lower" else "pessimistic"
    print(f"   {label:<12} ({mode}): mass = {mass:.3f} kg")
print("\\nDesigns that survive the pessimistic case are robust to the uncertainty.")"""),
    ("md", """## ODE_DP demo: steady-state heater

A heated payload loses heat to the environment proportional to its temperature rise (Newton's cooling). At steady state the input power equals the heat-loss coefficient times the temperature delta. The ODE solver finds the steady state by Newton iteration on $\\dot x = 0$.
"""),
    ("code", """H_LOSS = 0.8  # W/K

heater = ODE_DP(
    F=NamedProduct({"delta_T": Reals(unit="K")}),
    R=NamedProduct({"power": Reals(unit="W")}),
    rhs=lambda x, t, f: H_LOSS * f["delta_T"] - x,
    extract=lambda x: {"power": float(x)},
    mode="steady_state",
    x0_fn=lambda f: 0.0,
    name="heater_ode",
)

print("Power required to hold a steady temperature rise (h_loss = 0.8 W/K):")
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
    Antichain, FunctionDP, Loop, NamedProduct, Naturals, solve,
)

%matplotlib inline"""),
    ("md", "## The same model as notebook 02"),
    ("code", """def make_looped(c_value):
    N = Naturals()
    XY = NamedProduct({"x": N, "y": N})
    F = NamedProduct({"c": N, "xy": XY})
    R = NamedProduct({"xy": XY, "xy_report": XY})

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
    result = solve(looped, {"c": c_value}, max_iter=50, record_trace=True)
    trace = result.trace

    max_xy = 1
    for entry in trace:
        for p in entry.antichain.points:
            x, y = p["xy"]["x"], p["xy"]["y"]
            if x != math.inf: max_xy = max(max_xy, int(x))
            if y != math.inf: max_xy = max(max_xy, int(y))
    bound = max_xy + 3

    n = len(trace)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = [axes] if rows * cols == 1 else (axes.flat if rows > 1 else axes)
    axes = list(axes)

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
    ("code", """ALPHA = 1.8e6
G = 9.81
C_LIFT = 10.0

with MCDP("drone") as m:
    m.provides("endurance", unit="s")
    m.provides("extra_payload", unit="kg")
    m.provides("extra_power", unit="W")
    m.provides("battery_mass", unit="kg")

    m.requires("battery_mass", unit="kg")    # loop axis
    m.requires("report_mass", unit="kg")     # mirror

    def battery_mass_eq(f):
        lift = (f["battery_mass"] + f["extra_payload"]) * G
        actuator_power = C_LIFT * lift * lift
        total_power = actuator_power + f["extra_power"]
        energy = total_power * f["endurance"]
        return energy / ALPHA

    m.constraint("battery_mass", battery_mass_eq)
    m.constraint("report_mass", battery_mass_eq)
    m.loop_on("battery_mass")

drone = m.build()
drone"""),
    ("md", "## Run the same cases as notebook 01"),
    ("code", """cases = [
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
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}

    def __init__(self, specific_energy=1.8e6):
        self.specific_energy = specific_energy
        super().__init__()

    def h(self, f):
        return {"mass": f["capacity"] / self.specific_energy}


class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}

    def __init__(self, c_lift=10.0):
        self.c_lift = c_lift
        super().__init__()

    def h(self, f):
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
    ("code", """drone = sys.build()
print(drone)
print()

cases = [
    ("Short, light",   dict(endurance=60.0,   extra_payload=0.10, extra_power=1.0)),
    ("Medium, modest", dict(endurance=300.0,  extra_payload=0.50, extra_power=5.0)),
    ("Longer mission", dict(endurance=600.0,  extra_payload=0.50, extra_power=5.0)),
    ("Marginal",       dict(endurance=600.0,  extra_payload=1.00, extra_power=10.0)),
    ("Infeasible",     dict(endurance=1800.0, extra_payload=1.00, extra_power=10.0)),
]
for label, f in cases:
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
    CatalogDP, CatalogEntry, Module, NamedProduct, Reals,
    System, minimize_cost, solve,
)"""),
    ("md", """## Subsystem 1: a motor catalog

The catalog has Pareto-incomparable entries (lighter and more expensive vs. heavier and cheaper). `CatalogDP` is kept as a plain function constructor: multi-valued antichains don't fit the `Module` declarative pattern as cleanly.
"""),
    ("code", """motor = CatalogDP(
    F=NamedProduct({"torque": Reals(unit="N*m")}),
    R=NamedProduct({"mass": Reals(unit="kg"), "cost": Reals(unit="USD")}),
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
    F = {"load": Reals(unit="kg")}
    R = {"mass": Reals(unit="kg"), "cost": Reals(unit="USD")}

    def h(self, f):
        return {
            "mass": 0.6  * f["load"],
            "cost": 20.0 * f["load"],
        }


class Battery(Module):
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
TORQUE_PER_KG = 0.25

sys = System("vehicle")

payload        = sys.provides("payload",        unit="kg")
mission_energy = sys.provides("mission_energy", unit="J")
total_mass     = sys.requires("total_mass",     unit="kg")
total_cost     = sys.requires("total_cost",     unit="USD")

m = sys.add("motor",   motor)
c = sys.add("chassis", Chassis())
b = sys.add("battery", Battery())

c.load   >= payload + m.mass + b.mass
m.torque >= TORQUE_PER_KG * G * (payload + c.mass + b.mass)
b.energy >= mission_energy

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
        print()
        continue
    print(f"   Pareto front ({len(result.antichain.points)} points):")
    for p in result.antichain.points:
        print(f"      total_mass={p['total_mass']:6.2f} kg,  "
              f"total_cost=${p['total_cost']:7.2f}")
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
    F = {"sample_rate": Reals(unit="Hz")}
    R = {"power": Reals(unit="W"), "mass": Reals(unit="kg")}

    def h(self, f):
        return {"power": 0.02 * f["sample_rate"] + 0.5, "mass": 0.05}


class Controller(Module):
    F = {"input_rate": Reals(unit="Hz"), "command_rate": Reals(unit="Hz")}
    R = {"power":      Reals(unit="W"),  "mass":         Reals(unit="kg")}

    def h(self, f):
        return {
            "power": 0.05 * (f["input_rate"] + f["command_rate"]) + 2.0,
            "mass":  0.15,
        }


class Battery(Module):
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
b.capacity    >= (a.power + extra_power) * endurance
a.lift_force  >= 9.81 * (b.mass + extra_payload)
total_mass    >= b.mass + extra_payload
drone = sys.build()
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
bat.uncertain_set = Box(
    specific_energy=(1.6e6, 2.0e6, "more_is_better"),
    efficiency=(0.80, 0.90, "more_is_better"),
)
drone = make_drone(bat)

r_box = solve(drone, f, uncertainty=["worst_case"])
wc = list(r_box.worst_case.antichain.points)[0]["total_mass"]
print(f"Box worst case: {wc:.4f} kg  (penalty {wc - nominal_mass:+.4f} kg)")"""),
    ("md", """## Ellipsoid uncertainty (smaller, correlated set)

The Ellipsoid carves out the implausible corner where both parameters are simultaneously at their extremes. The worst case lies on the curved boundary in the direction of badness, which is closer to the centre than the box's worst-case corner.
"""),
    ("code", """bat = Battery()
bat.uncertain_set = Ellipsoid(
    center={"specific_energy": 1.8e6, "efficiency": 0.85},
    cov=[
        [1.0e10, -2.0e3],
        [-2.0e3,  2.5e-3],
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
bat.uncertain_set = Box(
    specific_energy=(1.6e6, 2.0e6, "more_is_better"),
    efficiency=(0.80, 0.90, "more_is_better"),
)
bat.uncertain_dist = Stochastic(
    marginals={
        "specific_energy": stats.uniform(loc=1.6e6, scale=0.4e6),
        "efficiency":      stats.uniform(loc=0.80, scale=0.10),
    },
    copula=GaussianCopula(correlation=[[1.0, 0.4],
                                       [0.4, 1.0]]),
)

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
    ("code", """res = solve(
    drone, f,
    uncertainty=["worst_case", "mean", "p95", "cvar95", "samples"],
    n_samples=1000,
    rng_seed=42,
    verbose=1,
)

wc = list(res.worst_case.antichain.points)[0]["total_mass"]
print()
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
        sun = max(self.sun_hours_per_day, 1e-6)
        required_peak = max(f["peak_power_kw"],
                            f["daily_energy_kwh"] / sun)
        return {"cost_usd": required_peak * self.cost_per_kw,
                "mass_kg":  required_peak * self.mass_per_kw}


class Battery(Module):
    F = {"storage_kwh": Reals(unit="kWh")}
    R = {"cost_usd": Reals(unit="USD"), "mass_kg": Reals(unit="kg"),
         "replacements": Reals()}
    CHEMISTRIES = {
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
        reps = (self.daily_cycles * 365.0 * self.life_years) / max(self.cycle_life, 1.0)
        return {"cost_usd": kwh * self.cost_density * (1.0 + reps),
                "mass_kg":  kwh * 1000.0 / max(self.specific_energy, 1e-6),
                "replacements": reps}


class DieselGenerator(Module):
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

cold_iters = 0
for L in loads:
    f = {"daily_load_kwh": float(L), "peak_load_kw": 3.0, "backup_hours": 12.0}
    r = solve(dp, f, max_iter=400)
    cold_iters += r.iterations

warm_iters = 0
prev = None
for L in loads:
    f = {"daily_load_kwh": float(L), "peak_load_kw": 3.0, "backup_hours": 12.0}
    r = solve(dp, f, max_iter=400, start_from=prev)
    warm_iters += r.iterations
    prev = r

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
    AlgebraicDP, Reals, NamedProduct, solve,
    solve_online, LipschitzEvaluator,
    MonotonicityEvaluator, LinearParametricEvaluator,
)

F = NamedProduct({"target_throughput": Reals(unit="pkg/h"),
                  "target_range":      Reals(unit="km")})
R = NamedProduct({"total_cost":   Reals(unit="USD"),
                  "total_energy": Reals(unit="kWh/day")})

def make_dp(robot):
    s = robot["speed"]; p = robot["payload"]
    c = robot["unit_cost"]; e = robot["energy_per_km"]
    capacity = s * p  # pkg/h per robot
    return AlgebraicDP(F, R, {
        "total_cost":   lambda f, cap=capacity, uc=c: (f["target_throughput"]/cap) * uc,
        "total_energy": lambda f, ek=e: f["target_range"] * ek * 24.0,
    })"""),

    ("md", "## The catalog (200 robot types)"),
    ("code", """def make_catalog(n=200, seed=42):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = rng.uniform(5, 30); p = rng.uniform(1, 20)
        c = rng.uniform(500, 5000); e = rng.uniform(0.05, 0.5)
        out.append({"name": f"r{i:03d}",
                    "speed": s, "payload": p,
                    "unit_cost": c, "energy_per_km": e,
                    "cost_per_capacity": c / (s * p)})  # monotone-friendly feature
    return out

candidates = make_catalog()
mission = {"target_throughput": 100.0, "target_range": 50.0}
print(f"{len(candidates)} candidate robot types")"""),

    ("md", """## Exhaustive baseline

Every catalog entry gets solved; we record the true Pareto front for reference.
"""),
    ("code", """points = []
for c in candidates:
    a = solve(make_dp(c), mission).antichain
    pt = dict(list(a.points)[0]); pt["name"] = c["name"]
    points.append(pt)

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
    ("code", """evs = [
    ("Lipschitz", LipschitzEvaluator(
        features=["speed", "payload", "unit_cost", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
        L={"total_cost": 300.0, "total_energy": 30.0},
    )),
    ("Monotonicity", MonotonicityEvaluator(
        features=["cost_per_capacity", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
    )),
    ("LinearParametric", LinearParametricEvaluator(
        features=["speed", "payload", "unit_cost", "energy_per_km"],
        r_components=["total_cost", "total_energy"],
        confidence=3.0, min_obs=5,
    )),
]
results = []
for name, ev in evs:
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
    ]
    for name, cells in plan:
        write(name, cells)
    print(f"\nAll {len(plan)} notebooks built and executed in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
