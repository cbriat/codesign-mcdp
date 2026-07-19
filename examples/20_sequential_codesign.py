"""
Example 20: antichain-valued sequential co-design (multi-objective DP).

The temporal Case 2 example (19) carried a battery budget and minimised a
single scalar cost per stage, so its value function was a single number
per state. This example exercises the *antichain-valued* generalisation:
the value at each stage and state is a whole Pareto front of cumulative
resource totals, not one number. It is the sequential co-design object of
the framework's theory, solved by an antichain-valued Bellman recursion.

The scenario is a multi-leg survey mission. At each leg the operator picks
an operating mode that trades two incommensurable objectives against each
other, monetary cost and CO2 emitted, while drawing down a shared energy
budget carried between legs. Because cost and CO2 are incomparable, the
right answer to "how should the whole mission be run" is not a single plan
but a Pareto front of whole-mission (cost, CO2) totals, each realised by a
different schedule of per-leg modes. The antichain-valued DP computes that
front exactly.

What the framework guarantees here
----------------------------------
* The value front equals the reachable frontier: every point returned is
  achievable by some feasible mode sequence, and no achievable
  non-dominated total is missed (front = reachable frontier).
* The front size is the width of the reachable set. With a summed
  (consumable) objective on a fixed number of axes it grows polynomially
  in the horizon, not exponentially, which is why the front here stays
  small and legible.
* The carried energy budget is a separate state axis from the accumulated
  (cost, CO2) resource: the transition reads energy off the chosen point
  while only cost and CO2 are accumulated on the antichain.

Modes (each a co-design problem)
--------------------------------
Three incomparable modes per leg, spanning the cost/CO2 trade:

    eco    : low CO2, higher cost, moderate energy draw
    balanced: middle cost and CO2
    rapid  : low cost, high CO2, higher energy draw

Run:  python -m examples.20_sequential_codesign
Expected output: the per-leg mode table, the whole-mission Pareto front of
(cost, CO2) totals (9 incomparable points), and a monotonicity report
confirming the value front is well-behaved in the carried budget.
"""

from __future__ import annotations

from codesign import (
    AlgebraicDP,
    Architecture,
    Ports,
    Reals,
    SeqStage,
    StateGrid,
    System,
    check_monotonicity,
    solve_sequential,
    sum_combine,
)

ENERGY_CAPACITY = 30.0   # carried budget units


# ---------------------------------------------------------------------------
# Mode builder: a co-design problem emitting (cost, co2, energy) for a leg.
# ---------------------------------------------------------------------------
def build_mode(name, *, cost, co2, energy):
    """A leg mode emitting fixed cost, CO2, and energy draw.

    A single point per mode keeps the example legible; the Pareto structure
    comes from the *union* over the three incomparable modes, and from the
    accumulation of incomparable totals across legs.
    """
    s = System(name)
    demand = s.provides("demand", unit="")
    s.requires("cost", unit="")
    s.requires("co2", unit="")
    s.requires("energy", unit="")
    s.add(
        "m",
        AlgebraicDP(
            F=Ports({"demand": Reals(unit="")}),
            R=Ports({
                "cost": Reals(),
                "co2": Reals(),
                "energy": Reals(),
            }),
            equations={
                "cost": lambda f, c=cost: c,
                "co2": lambda f, e=co2: e,
                "energy": lambda f, en=energy: en,
            },
        ),
    ).demand >= demand
    s.constrain("cost", lambda x: x["m.cost"])
    s.constrain("co2", lambda x: x["m.co2"])
    s.constrain("energy", lambda x: x["m.energy"])
    return s.build()


ECO = Architecture(
    "eco", build_mode("eco", cost=10.0, co2=1.0, energy=4.0),
    tags={"mode": "eco"},
)
BALANCED = Architecture(
    "balanced", build_mode("balanced", cost=6.0, co2=4.0, energy=5.0),
    tags={"mode": "balanced"},
)
RAPID = Architecture(
    "rapid", build_mode("rapid", cost=2.0, co2=9.0, energy=7.0),
    tags={"mode": "rapid"},
)
MODES = [ECO, BALANCED, RAPID]


# ---------------------------------------------------------------------------
# Mission: legs sharing a depleting energy budget.
# ---------------------------------------------------------------------------
def build_mission(n_legs):
    func = lambda state: {"demand": 1.0}
    transition = lambda state, point: state - point["energy"]
    admissible = lambda state: state >= -1e-9
    return [
        SeqStage(f"leg_{i+1}", functionality=func, transition=transition,
                 admissible=admissible, candidates=MODES)
        for i in range(n_legs)
    ]


def main():
    n_legs = 4
    stages = build_mission(n_legs)
    grid = StateGrid.linspace(0.0, ENERGY_CAPACITY, 61)

    print("Antichain-valued sequential co-design: a multi-leg survey")
    print("=" * 60)
    print(f"{n_legs} legs, energy budget {ENERGY_CAPACITY:.0f}, modes:",
          ", ".join(m.name for m in MODES))
    print("per-leg (cost, CO2, energy):")
    for m, spec in [("eco", (10, 1, 4)), ("balanced", (6, 4, 5)),
                    ("rapid", (2, 9, 7))]:
        print(f"  {m:<9s} cost={spec[0]:>2}  co2={spec[1]:>2}  energy={spec[2]}")
    print()

    res = solve_sequential(
        stages, grid,
        cost_axes=["cost", "co2"],
        initial_state=ENERGY_CAPACITY,
        combine=sum_combine,
    )

    print(f"Whole-mission Pareto front of (cost, CO2) totals "
          f"({res.width} incomparable points):")
    front = sorted(
        ((p["cost"], p["co2"]) for p in res.value), key=lambda t: t[0]
    )
    for c, e in front:
        print(f"  cost={c:6.1f}   co2={e:6.1f}")
    print()
    print("Each point is a different whole-mission plan: the low-cost end")
    print("runs rapid legs (cheap, dirty), the low-CO2 end runs eco legs")
    print("(clean, expensive), and the interior points mix modes across legs.")
    print("The front is the exact reachable frontier, and its size grows")
    print("polynomially (here linearly) with the number of legs, not")
    print("exponentially, because cost and CO2 accumulate on two fixed axes.")
    print()

    # Confirm the value is well-behaved: more energy is never worse, and the
    # transition is monotone in the state.
    rep = check_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    print("Monotonicity guard:", rep)
    if rep.monotone_value_guaranteed:
        print("  (H1) and (H2) hold, so the value front is monotone in the")
        print("  carried budget: more energy never shrinks the achievable front.")


if __name__ == "__main__":
    main()
