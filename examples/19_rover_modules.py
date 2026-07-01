"""
Example 19: planetary rover module activation over a battery budget (Case 2).

A small planetary rover carries several modules that can be independently
activated or deactivated: a drive train, a drill, a high-gain
communications link, a science payload (spectrometer + camera), and a
survival heater. Each mission phase demands a different capability, and
the rover runs on a battery whose charge depletes when modules draw
power and partially recharges from solar input between phases. The
question is not a single static design; it is *which modules to activate
in each phase*, and how to size the shared power bus that serves them,
so that the whole multi-phase mission completes without the battery
going flat.

This is the second temporal case, with a genuinely carried resource. The
outer object is a finite-horizon dynamic program over mission phases; the
per-phase decision is which module configuration (which "mode") to
activate; the per-phase cost is obtained by *solving a co-design problem*
that sizes the power bus and accumulates the energy and data cost of the
active modules against the phase's requirement; and the battery state of
charge is carried from one phase to the next. A configuration that draws
heavily buys capability now at the cost of charge later, so the optimal
policy is a schedule of module activations threaded through the evolving
state of charge.

The framing follows standard spacecraft / rover power-mode practice,
where the power budget is organised into operational modes entered during
different mission phases (drive mode, science mode, comms mode, survival
mode), and activities are scheduled around available battery reserve.

Modes (each a co-design problem)
--------------------------------
Each mode is a small ``System`` that, given the phase's capability demand,
sizes a shared power bus and returns two outer resources:

    energy_Wh : the energy the configuration consumes this phase (this is
                the quantity subtracted from the battery state of charge)
    cost      : a mission-cost proxy combining bus mass and a penalty for
                leaving a phase's primary objective unmet

The modes differ in which modules they activate:

    drive       : drive train only, moderate draw, advances the traverse
    science      : payload + drive at low speed, high data value, high draw
    comms       : high-gain link + housekeeping, moderate draw
    survival    : heater + housekeeping only, low draw, no mission progress
                  (the "ride it out" option when charge is low)

Per-phase the dynamic program picks the mode that minimises mission cost
plus cost-to-go, subject to the battery never going negative. Between
phases the battery recharges by a fixed solar increment (capped at
capacity). When charge is plentiful the policy spends it on
high-value science and comms; when charge runs low it is forced into
survival mode, deferring objectives, exactly the load-shedding behaviour
real missions use.

Parameter values are illustrative, in the qualitative range of a small
solar rover (tens to low hundreds of Wh per phase, a battery of a few
hundred Wh). They are not drawn from a specific vehicle.

Run directly to solve the policy and roll it out from a full and from a
depleted initial battery, showing how the module-activation schedule
changes with available charge.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    AlgebraicDP,
    Architecture,
    Ports,
    Reals,
    Stage,
    StateGrid,
    System,
    rollout,
    solve_dynamic,
)

# Battery and solar parameters (Wh).
BATTERY_CAPACITY_WH = 300.0
SOLAR_RECHARGE_WH = 60.0          # gained between consecutive phases

# Highest objective value any single mode can deliver. Used to turn the
# per-mode objective value into a non-negative opportunity penalty
# (BEST - value), since the co-design resource poset forbids negative
# resources. Must be >= the largest objective_value assigned below.
BEST_OBJECTIVE_VALUE = 200.0


# ---------------------------------------------------------------------------
# Mode builders: each returns a co-design problem sizing a power bus and
# reporting (energy_Wh, cost). The capability demand `cap` is the outer
# functionality and means different things per mode (distance, data, link
# margin) but is kept a single scalar for a compact example.
# ---------------------------------------------------------------------------
def build_mode(
    name: str,
    *,
    base_draw_wh: float,
    draw_per_cap: float,
    bus_overhead_wh: float,
    objective_value: float,
    cap_ceiling: float,
) -> object:
    """Build one rover operating mode as a co-design problem.

    Parameters
    ----------
    name : str
        Mode name.
    base_draw_wh : float
        Fixed energy the mode draws each phase regardless of demand
        (housekeeping plus the always-on modules of this mode).
    draw_per_cap : float
        Additional energy per unit of capability demanded.
    bus_overhead_wh : float
        A fixed energy overhead for powering the shared bus in this mode,
        standing in for conversion and distribution losses that scale with
        the number of active modules.
    objective_value : float
        Mission value delivered by completing this mode's objective. Enters
        the cost as a negative term (credit) so that, all else equal,
        higher-value modes are preferred when charge allows.
    cap_ceiling : float
        Maximum capability the mode can deliver in one phase. Demands above
        this are infeasible for the mode.
    """
    sys_ = System(name)
    cap = sys_.provides("cap", unit="")
    sys_.requires("energy_Wh", unit="Wh")
    sys_.requires("cost", unit="")

    # Power-draw module: energy as a function of demanded capability, with
    # a hard ceiling above which the mode cannot deliver.
    draw = sys_.add(
        "draw",
        AlgebraicDP(
            F=Ports({"cap": Reals(unit="")}),
            R=Ports({"energy_Wh": Reals(unit="Wh")}),
            equations={
                "energy_Wh": lambda f, b=base_draw_wh, k=draw_per_cap, cc=cap_ceiling: (
                    b + k * f["cap"] if f["cap"] <= cc else float("inf")
                )
            },
        ),
    )
    draw.cap >= cap

    # Bus module: fixed distribution overhead for this mode.
    bus = sys_.add(
        "bus",
        AlgebraicDP(
            F=Ports({"cap": Reals(unit="")}),
            R=Ports({"energy_Wh": Reals(unit="Wh")}),
            equations={"energy_Wh": lambda f, o=bus_overhead_wh: o},
        ),
    )
    bus.cap >= cap

    # Outer energy is draw + bus overhead.
    sys_.constrain(
        "energy_Wh", lambda x: x["draw.energy_Wh"] + x["bus.energy_Wh"]
    )
    # Outer cost: total energy spent plus an opportunity penalty for the
    # objective value NOT delivered by this mode. The penalty is
    # ``BEST_OBJECTIVE_VALUE - objective_value``, so a high-value mode
    # carries a low penalty and is preferred when charge allows, while the
    # cost stays non-negative as the co-design resource poset requires
    # (resources cannot be negative, so a raw "energy minus value" credit
    # would be rejected as below the poset bottom).
    sys_.constrain(
        "cost",
        lambda x, pen=(BEST_OBJECTIVE_VALUE - objective_value): (
            x["draw.energy_Wh"] + x["bus.energy_Wh"] + pen
        ),
    )
    return sys_.build()


# Four modes spanning the capability/draw trade-off. Objective values are
# scaled so that, when charge allows, an active objective mode is
# preferred over idling in survival; the binding trade is then the battery
# budget, not the per-phase arithmetic. Science delivers the most value but
# draws the most energy, so it is affordable only with healthy reserve.
DRIVE = Architecture(
    "drive",
    build_mode(
        "drive",
        base_draw_wh=30.0, draw_per_cap=8.0, bus_overhead_wh=5.0,
        objective_value=110.0, cap_ceiling=10.0,
    ),
    tags={"modules": "drive"},
)
SCIENCE = Architecture(
    "science",
    build_mode(
        "science",
        base_draw_wh=45.0, draw_per_cap=12.0, bus_overhead_wh=10.0,
        objective_value=200.0, cap_ceiling=8.0,
    ),
    tags={"modules": "drive+payload"},
)
COMMS = Architecture(
    "comms",
    build_mode(
        "comms",
        base_draw_wh=35.0, draw_per_cap=6.0, bus_overhead_wh=8.0,
        objective_value=120.0, cap_ceiling=10.0,
    ),
    tags={"modules": "high-gain-comms"},
)
SURVIVAL = Architecture(
    "survival",
    build_mode(
        "survival",
        base_draw_wh=12.0, draw_per_cap=1.0, bus_overhead_wh=2.0,
        objective_value=0.0, cap_ceiling=10.0,
    ),
    tags={"modules": "heater"},
)

ALL_MODES = [DRIVE, SCIENCE, COMMS, SURVIVAL]


# ---------------------------------------------------------------------------
# Mission: a sequence of phases, each demanding some capability. The
# transition depletes the battery by the chosen mode's energy and then adds
# the solar recharge (capped at capacity).
# ---------------------------------------------------------------------------
def phase_demand(_state: float) -> dict:
    """Capability demanded in a phase (state-independent here)."""
    return {"cap": 5.0}


def make_transition():
    """Battery update: subtract energy spent, add solar, cap at capacity."""

    def transition(state: float, point) -> float:
        spent = point["energy_Wh"]
        recharged = min(BATTERY_CAPACITY_WH, state - spent + SOLAR_RECHARGE_WH)
        return recharged

    return transition


def battery_nonneg(state: float) -> bool:
    """The battery may not go below zero at any point."""
    return state >= 0.0


def build_mission(n_phases: int) -> list:
    """A mission of ``n_phases`` identical-demand phases.

    Identical demand keeps the example readable; the interesting variation
    comes from the carried state of charge, not from the per-phase demand.
    All four modes are admissible in every phase, so the policy is free to
    shed load to survival whenever charge is tight.
    """
    transition = make_transition()
    return [
        Stage(
            f"phase_{i+1}",
            functionality=phase_demand,
            transition=transition,
            admissible=battery_nonneg,
            candidates=ALL_MODES,
        )
        for i in range(n_phases)
    ]


def mission_cost(point) -> float:
    """Scalar cost minimised per phase: energy spent minus objective value."""
    return point["cost"]


def terminal_reward(state: float) -> float:
    """Credit leftover charge slightly, so ending with reserve is preferred.

    Encoded as a negative cost proportional to remaining charge. Small
    enough that it never dominates in-mission objective value, but it
    breaks ties toward leaving a margin.
    """
    return -0.05 * state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    stages = build_mission(n_phases=6)
    grid = StateGrid.linspace(0.0, BATTERY_CAPACITY_WH, 61)  # 5 Wh resolution

    print("Planetary rover module activation over a battery budget")
    print("=" * 60)
    print(
        f"battery capacity = {BATTERY_CAPACITY_WH:.0f} Wh, "
        f"solar recharge = {SOLAR_RECHARGE_WH:.0f} Wh/phase, "
        f"{len(stages)} phases"
    )
    print("modes:", ", ".join(a.name for a in ALL_MODES))
    print()

    # Solve the policy once; it is valid for any initial charge on the grid.
    policy = solve_dynamic(
        stages,
        grid,
        cost_fn=mission_cost,
        terminal_cost=terminal_reward,
    )

    # Roll out from a full battery: the rover can afford the high-value
    # science and comms modes.
    full = rollout(policy, stages, BATTERY_CAPACITY_WH)
    print(f"Start at full charge ({BATTERY_CAPACITY_WH:.0f} Wh):")
    _print_rollout(full)

    # Roll out from a depleted battery: the rover must shed load to
    # survival in the lean phases and can only spend on objectives once
    # solar has rebuilt some reserve.
    low = rollout(policy, stages, 90.0)
    print("\nStart at low charge (90 Wh):")
    _print_rollout(low)

    print(
        "\nSame policy, different starting charge: the module-activation "
        "schedule\nadapts. With a full battery the rover runs high-value "
        "science until\nreserve falls, then steps down to a lighter mode; "
        "starting depleted it\ncan never afford science's draw and holds a "
        "lighter mode throughout,\nshedding load exactly as a real "
        "power-constrained mission would."
    )


def _print_rollout(res) -> None:
    if not res.feasible:
        print("  INFEASIBLE from this state")
    for sr in res.stages:
        print(
            f"  {sr.stage:<8s} -> {sr.architecture:<9s} "
            f"soc_in={sr.state_in:6.1f} Wh  "
            f"spent/credit cost={sr.stage_cost:7.2f}  "
            f"soc_out={sr.state_out:6.1f} Wh"
        )
    print(
        f"  total cost = {res.total_cost:.2f} "
        f"(lower is better; energy spent plus opportunity penalty), "
        f"feasible = {res.feasible}"
    )


if __name__ == "__main__":
    main()
