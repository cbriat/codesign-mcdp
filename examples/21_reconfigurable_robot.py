"""
Example 21: vector-state co-design for a self-reconfiguring robot.

Examples 19 and 20 carried a single scalar between stages (battery
charge). Real reconfigurable systems carry a *state vector*. The Formula 1
seasonal co-design of Neumann, Zardini and colleagues carries two battery
wear levels plus a regulatory-penalty flag; a self-reconfiguring modular
robot on a multi-leg mission carries the accumulated wear of each drive
module it can activate, plus a shared energy budget. This example is the
robot case, and it exercises the general vector-state dynamic program.

A field robot can reconfigure between three morphologies at each mission
leg, drawing on two physical drive modules whose wear accumulates
independently:

    tracked : uses the track module heavily. Robust on rough ground, low
              energy, but wears the track module fast.
    wheeled : uses the wheel module. Fast and efficient on flat ground,
              wears the wheel module fast, spares the track.
    hybrid  : splits load across both modules. Middling on every axis,
              wears both a little.

The carried state is a vector of three axes: ``track_wear``,
``wheel_wear``, and ``energy`` (a shared battery that depletes and
recharges). Each leg the robot picks a morphology by solving that
morphology's co-design problem (sizing its power draw against the leg's
terrain demand), the chosen morphology adds wear to the module(s) it uses
and draws energy, and the vector state advances. A module that wears past
its limit is forbidden (the morphology relying on it becomes infeasible),
so the optimal plan spreads wear across the two modules over the mission,
exactly the behaviour a maintenance-aware operator wants.

Because cost (energy plus an operations penalty) is the accumulated
resource and the two wear levels plus energy are the carried state, the
value is an antichain over the cost axes parametrised by the state
vector. The vector monotonicity guard confirms the value is well-behaved
in the carried state.

This is grounded in the self-reconfiguring-robot literature, where a
robot changes morphology to suit terrain and must manage module wear and
energy across a mission, and it mirrors the F1 paper's structured
multi-component state (there: two batteries; here: two drive modules).

Run directly to solve the vector-state DP and print the mission's Pareto
front of (energy, ops) totals plus the monotonicity report.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    AlgebraicDP,
    Architecture,
    ContinuousAxis,
    Ports,
    Reals,
    System,
    VecStage,
    VectorStateGrid,
    check_vector_monotonicity,
    solve_vector_sequential,
    state_as_dict,
    state_get,
    sum_combine,
)

ENERGY_CAPACITY = 24.0
SOLAR_RECHARGE = 4.0
WEAR_LIMIT = 10.0


# ---------------------------------------------------------------------------
# Morphology builder: a co-design problem returning (energy, ops) plus the
# per-module wear increments and energy draw it will apply this leg.
# ---------------------------------------------------------------------------
def build_morphology(name, *, energy, ops, track_wear, wheel_wear):
    """Build one morphology as a co-design problem.

    Emits four resource ports: ``energy`` (drawn this leg, subtracted from
    the battery), ``ops`` (an operations-cost proxy, minimised), and the
    two wear increments ``d_track`` / ``d_wheel`` applied to the carried
    wear axes.
    """
    s = System(name)
    demand = s.provides("demand", unit="")
    s.requires("energy", unit="")
    s.requires("ops", unit="")
    s.requires("d_track", unit="")
    s.requires("d_wheel", unit="")
    s.add("m", AlgebraicDP(
        F=Ports({"demand": Reals(unit="")}),
        R=Ports({
            "energy": Reals(), "ops": Reals(),
            "d_track": Reals(), "d_wheel": Reals(),
        }),
        equations={
            "energy": lambda f, e=energy: e,
            "ops": lambda f, o=ops: o,
            "d_track": lambda f, w=track_wear: w,
            "d_wheel": lambda f, w=wheel_wear: w,
        },
    )).demand >= demand
    s.constrain("energy", lambda x: x["m.energy"])
    s.constrain("ops", lambda x: x["m.ops"])
    s.constrain("d_track", lambda x: x["m.d_track"])
    s.constrain("d_wheel", lambda x: x["m.d_wheel"])
    return s.build()


TRACKED = Architecture(
    "tracked",
    build_morphology("tracked", energy=2.0, ops=4.0, track_wear=3.0, wheel_wear=0.0),
    tags={"morphology": "tracked"},
)
WHEELED = Architecture(
    "wheeled",
    build_morphology("wheeled", energy=4.0, ops=2.0, track_wear=0.0, wheel_wear=3.0),
    tags={"morphology": "wheeled"},
)
HYBRID = Architecture(
    "hybrid",
    build_morphology("hybrid", energy=3.0, ops=3.0, track_wear=1.5, wheel_wear=1.5),
    tags={"morphology": "hybrid"},
)
MORPHOLOGIES = [TRACKED, WHEELED, HYBRID]


# ---------------------------------------------------------------------------
# The mission: legs carrying a vector state (track_wear, wheel_wear, energy)
# ---------------------------------------------------------------------------
def leg_demand(_state):
    return {"demand": 1.0}


def make_transition():
    def transition(state, point):
        d = state_as_dict(state)
        new_energy = min(
            ENERGY_CAPACITY,
            d["energy"] - point["energy"] + SOLAR_RECHARGE,
        )
        return {
            "track_wear": d["track_wear"] + point["d_track"],
            "wheel_wear": d["wheel_wear"] + point["d_wheel"],
            "energy": new_energy,
        }
    return transition


def admissible(state):
    d = state_as_dict(state)
    return (
        d["track_wear"] <= WEAR_LIMIT + 1e-9
        and d["wheel_wear"] <= WEAR_LIMIT + 1e-9
        and d["energy"] >= -1e-9
    )


def build_mission(n_legs):
    transition = make_transition()
    return [
        VecStage(f"leg_{i+1}", functionality=leg_demand, transition=transition,
                 admissible=admissible, candidates=MORPHOLOGIES)
        for i in range(n_legs)
    ]


def main():
    n_legs = 5
    stages = build_mission(n_legs)

    # Product grid: two wear axes and an energy axis. Kept modest so the
    # example runs quickly; each axis is discretised explicitly.
    grid = VectorStateGrid([
        ContinuousAxis("track_wear", 0.0, WEAR_LIMIT, 11),
        ContinuousAxis("wheel_wear", 0.0, WEAR_LIMIT, 11),
        ContinuousAxis("energy", 0.0, ENERGY_CAPACITY, 13),
    ])

    print("Self-reconfiguring robot: vector-state co-design")
    print("=" * 60)
    print(f"{n_legs} legs, energy capacity {ENERGY_CAPACITY:.0f} "
          f"(+{SOLAR_RECHARGE:.0f}/leg), wear limit {WEAR_LIMIT:.0f} per module")
    print("morphologies (energy, ops, track_wear, wheel_wear):")
    for m, spec in (("tracked", (2, 4, 3, 0)), ("wheeled", (4, 2, 0, 3)),
                    ("hybrid", (3, 3, 1.5, 1.5))):
        print(f"  {m:<8s} {spec}")
    print(f"grid size: {len(grid)} state nodes")
    print()

    res = solve_vector_sequential(
        stages, grid,
        cost_axes=["energy", "ops"],
        initial_state={"track_wear": 0.0, "wheel_wear": 0.0,
                       "energy": ENERGY_CAPACITY},
        combine=sum_combine,
    )

    print(f"Mission Pareto front of (energy, ops) totals "
          f"({res.width} incomparable points):")
    front = sorted(((p["energy"], p["ops"]) for p in res.value),
                   key=lambda t: t[0])
    for e, o in front:
        print(f"  energy={e:6.1f}   ops={o:6.1f}")
    print()
    print("Each point is a different morphology schedule across the five")
    print("legs. The low-energy end leans on the wheeled morphology; the")
    print("low-ops end mixes to spread wear so no single module hits its")
    print("limit and forces an expensive fallback. The DP carries the full")
    print("three-axis state (two module wear levels plus energy), which a")
    print("scalar DP could not represent.")
    print()

    rep = check_vector_monotonicity(stages, grid, cost_axes=["energy", "ops"])
    print("Vector monotonicity guard:", rep)
    if rep.monotone_value_guaranteed:
        print("  (H1) and (H2) hold over the product order: more spare wear")
        print("  budget and more energy never shrink the achievable front.")


if __name__ == "__main__":
    main()
