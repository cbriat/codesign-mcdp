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
        for k, A in enumerate(result.trace):
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
    for A in trace:
        for p in A.points:
            x, y = p["xy"]["x"], p["xy"]["y"]
            if x != math.inf: max_xy = max(max_xy, int(x))
            if y != math.inf: max_xy = max(max_xy, int(y))
    bound = max_xy + 3

    n = len(trace)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = [axes] if rows * cols == 1 else (axes.flat if rows > 1 else axes)
    axes = list(axes)

    for k, A in enumerate(trace):
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
    ("md", """# 07. The drone, modular: independent battery and actuator

The same MCDP as notebooks **01** and **06**, but with battery and actuator defined **independently** as their own design problems, then assembled with the `System` builder.

This is the natural way to build larger designs: each subsystem is its own DP with its own F and R, and the `System` wires them together with named algebraic constraints. The Kleene iteration converges over the joint state of all subsystem R ports.
"""),
    ("md", "## Imports"),
    ("code", """from codesign import (
    AlgebraicDP, NamedProduct, Reals, System, solve,
)"""),
    ("md", """## Subsystems

Each is defined in isolation. The battery has no idea the actuator exists, and vice versa.
"""),
    ("code", """battery = AlgebraicDP(
    F=NamedProduct({"capacity": Reals(unit="J")}),
    R=NamedProduct({"mass": Reals(unit="kg")}),
    equations={"mass": lambda f: f["capacity"] / 1.8e6},
    name="battery",
)
actuator = AlgebraicDP(
    F=NamedProduct({"lift_force": Reals(unit="N")}),
    R=NamedProduct({"power": Reals(unit="W")}),
    equations={"power": lambda f: 10.0 * f["lift_force"] ** 2},
    name="actuator",
)
battery, actuator"""),
    ("md", """## Assembly

Declare the outer interface, add the subsystems, then attach connection constraints. Each constraint reads `target_port >= demand(ctx)` where `ctx` is a dict carrying the outer functionalities (`endurance`, `extra_payload`, `extra_power`) under their bare names and every subsystem R port under its dotted name (`battery.mass`, `actuator.power`).
"""),
    ("code", """G = 9.81

sys = System("drone")
sys.provides("endurance", unit="s")
sys.provides("extra_payload", unit="kg")
sys.provides("extra_power", unit="W")
sys.requires("total_mass", unit="kg")

sys.add("battery", battery)
sys.add("actuator", actuator)

sys.constrain(
    "battery.capacity",
    lambda x: (x["actuator.power"] + x["extra_power"]) * x["endurance"],
)
sys.constrain(
    "actuator.lift_force",
    lambda x: G * (x["battery.mass"] + x["extra_payload"]),
)
sys.constrain(
    "total_mass",
    lambda x: x["battery.mass"] + x["extra_payload"],
)
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

The values are identical (e.g. for Medium, modest: `total_mass = 0.5492 kg = 0.04921 (battery) + 0.5 (payload)`). The iteration count is roughly 2x larger, because the loop now updates `battery.mass` and `actuator.power` in alternation rather than in one coupled step. Same fixed point, finer-grained updates.

The payoff is *modularity*: the battery and actuator are reusable building blocks. The next notebook (**08**) adds a third subsystem (a motor catalog with discrete choices) and produces a multi-point Pareto front from the same machinery.
"""),
]


# ---------------------------------------------------------------------------
# 08 Modular vehicle with multi-point Pareto front
# ---------------------------------------------------------------------------

NB_08 = [
    ("md", """# 08. Motor + chassis + battery: a multi-point Pareto front

A small electric vehicle is co-designed from three independent subsystems:

- a **motor** picked from a discrete catalog (each entry has its own (torque, mass, cost) tuple),
- a **chassis** whose mass and cost scale with the load it must carry,
- a **battery** sized to the mission energy.

The three subsystems are coupled cyclically: the chassis must support the motor and battery, the motor's torque is sized by the total moving mass (which includes the chassis), and so on. The `System` builder closes the loop.

Because the motor catalog has Pareto-incomparable entries (lighter and more expensive, or heavier and cheaper), the system-level Pareto front has multiple points: real engineering tradeoffs surfaced automatically.
"""),
    ("md", "## Imports"),
    ("code", """from codesign import (
    AlgebraicDP, CatalogDP, CatalogEntry, NamedProduct, Reals,
    System, minimize_cost, solve,
)"""),
    ("md", """## Subsystem 1: a motor catalog

Each entry says "this motor can deliver up to X torque, and costs Y mass and Z dollars." Several entries are mutually incomparable on (mass, cost).
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
    ("md", "## Subsystem 2: the chassis"),
    ("code", """chassis = AlgebraicDP(
    F=NamedProduct({"load": Reals(unit="kg")}),
    R=NamedProduct({"mass": Reals(unit="kg"), "cost": Reals(unit="USD")}),
    equations={
        "mass": lambda f: 0.6 * f["load"],
        "cost": lambda f: 20.0 * f["load"],
    },
    name="chassis",
)"""),
    ("md", "## Subsystem 3: the battery"),
    ("code", """battery = AlgebraicDP(
    F=NamedProduct({"energy": Reals(unit="J")}),
    R=NamedProduct({"mass": Reals(unit="kg"), "cost": Reals(unit="USD")}),
    equations={
        "mass": lambda f: f["energy"] / 1.8e6,
        "cost": lambda f: 0.05 * f["energy"] / 3.6e3,  # $0.05 / Wh
    },
    name="battery",
)"""),
    ("md", """## Wire it up

The chassis must support payload + motor + battery, the motor's torque demand depends on the total moving mass, and the battery's energy demand is the externally-supplied mission energy. The total mass and total cost aggregate up from all subsystems.
"""),
    ("code", """G = 9.81
TORQUE_PER_KG = 0.25

sys = System("vehicle")
sys.provides("payload", unit="kg")
sys.provides("mission_energy", unit="J")
sys.requires("total_mass", unit="kg")
sys.requires("total_cost", unit="USD")

sys.add("motor", motor)
sys.add("chassis", chassis)
sys.add("battery", battery)

sys.constrain("chassis.load",
              lambda x: x["payload"] + x["motor.mass"] + x["battery.mass"])
sys.constrain("motor.torque",
              lambda x: TORQUE_PER_KG * G * (
                  x["payload"] + x["chassis.mass"] + x["battery.mass"]))
sys.constrain("battery.energy",
              lambda x: x["mission_energy"])
sys.constrain("total_mass",
              lambda x: x["payload"] + x["motor.mass"]
                        + x["chassis.mass"] + x["battery.mass"])
sys.constrain("total_cost",
              lambda x: x["motor.cost"] + x["chassis.cost"] + x["battery.cost"])

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
    ]
    for name, cells in plan:
        write(name, cells)
    print(f"\nAll {len(plan)} notebooks built and executed in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
