"""
Example: demos for the advanced primitives UncertainDP and ODE_DP.

UncertainDP wraps a pair of design problems (h_lower, h_upper) that bracket
an underlying h that is either unknown or non-finitely-representable.
Solving with h_lower gives an *optimistic* Pareto front (lower bound on
resources); solving with h_upper gives a *pessimistic* Pareto front. The
true minimal resources sit between (Sec. VII of the paper).

ODE_DP derives a monotone resource relation from a differential equation
by integrating to a steady state or to a final value. Here we model the
energy needed to heat a payload from T_ambient to a target temperature,
where heat loss follows Newton's law of cooling:

    dT/dt = (P_in - h * (T - T_ambient)) / C

At steady state, P_in = h * (T_ss - T_ambient), so the power needed to
hold a temperature is monotone in the temperature setpoint.

Run:  python -m examples.04_uncertain_and_ode
Expected output: the optimistic vs. pessimistic battery mass for a 1 kWh
capacity, then the steady-state heater power for several temperature rises
(each matching the closed-form h_loss * delta_T).
"""
from __future__ import annotations

from codesign import (
    AlgebraicDP,
    Ports,
    ODE_DP,
    Reals,
    UncertainDP,
    solve,
)


# ---------------------------------------------------------------------------
# UncertainDP demo: a battery whose specific energy is only known to lie
# within a range [1.6e6, 2.0e6] J/kg. The pessimistic bound assumes the
# lower number (more mass needed); the optimistic bound assumes the
# higher number (less mass needed).
# ---------------------------------------------------------------------------


def battery_uncertainty_demo():
    F = Ports({"capacity": Reals(unit="J")})
    R = Ports({"mass": Reals(unit="kg")})

    # Pessimistic: assume only 1.6 MJ/kg (older cell chemistry).
    pessimistic = AlgebraicDP(
        F=F, R=R,
        equations={"mass": lambda f: f["capacity"] / 1.6e6},
        name="battery_pessimistic",
    )
    # Optimistic: assume 2.0 MJ/kg (newer cells).
    optimistic = AlgebraicDP(
        F=F, R=R,
        equations={"mass": lambda f: f["capacity"] / 2.0e6},
        name="battery_optimistic",
    )
    uncertain = UncertainDP(
        F=F, R=R, lower=optimistic, upper=pessimistic, mode="upper",
        name="battery_uncertain",
    )

    print("UncertainDP demo: battery sizing under specific-energy uncertainty\n")
    cap = 3.6e6  # 1 kWh
    for mode in ("lower", "upper"):
        result = solve(uncertain.with_mode(mode), {"capacity": cap})
        ac = result.antichain
        mass = list(ac.points)[0]["mass"]
        label = "optimistic" if mode == "lower" else "pessimistic"
        print(f"   {label:<12} ({mode}): mass = {mass:.3f} kg")
    print(
        "   ...the true mass for 1 kWh sits between the two bounds; design\n"
        "   choices that survive the pessimistic case are robust."
    )


# ---------------------------------------------------------------------------
# ODE_DP demo: steady-state heater. Functionality is the target temperature
# rise (delta T above ambient). Resource is the required input power.
# Newton's law of cooling: dT/dt = (P_in - h*dT) / C  --> at steady state,
# P_in = h * dT.
# ---------------------------------------------------------------------------


def heater_demo():
    F = Ports({"delta_T": Reals(unit="K")})
    R = Ports({"power": Reals(unit="W")})

    # Heat-loss coefficient h = 0.8 W/K.
    H_LOSS = 0.8

    # The "state" we track is the steady-state input power.
    # dx/dt = h * delta_T - x, where x is what's being delivered.
    # The root of this is x = h * delta_T.
    heater = ODE_DP(
        F=F, R=R,
        rhs=lambda x, t, f: H_LOSS * f["delta_T"] - x,
        extract=lambda x: {"power": float(x)},
        mode="steady_state",
        x0_fn=lambda f: 0.0,
        name="heater_ode",
    )

    print("\nODE_DP demo: power required to hold a steady temperature rise\n")
    for dT in (5.0, 20.0, 50.0):
        result = solve(heater, {"delta_T": dT})
        p = list(result.antichain.points)[0]["power"]
        print(f"   delta_T = {dT:>4.0f} K  -->  P_in = {p:>5.1f} W   "
              f"(=  h_loss * delta_T = {H_LOSS * dT:.1f} W)")


if __name__ == "__main__":
    battery_uncertainty_demo()
    heater_demo()
