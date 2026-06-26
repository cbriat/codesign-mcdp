"""
Drone co-design example, modeled after Fig. 48 in Censi's paper.

Two design choices vs. a naive version:

1. We expose ``battery_mass`` as an OUTER resource (in addition to being the
   feedback variable), so the final antichain reports the converged value
   rather than collapsing to a unit poset. The inner DP outputs two
   components with the same value: one for the feedback loop, one for the
   outer interface.

2. We sweep several missions and print the converged battery mass.

Physics (simple lumped model):
    actuation power     = c * weight^2
    weight              = g * (battery_mass + extra_payload)
    total power         = actuation_power + extra_power
    energy needed       = total_power * endurance
    required mass       = energy / specific_energy

Closing the loop: required_mass <= trial_mass.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    Reals, Ports, FunctionDP, loop, solve,
)


ALPHA = 1.8e6           # specific energy J/kg
G = 9.81                # gravity m/s^2
C_LIFT = 10.0           # W/N^2


def drone_inner_h(f):
    """Inner relation. battery_mass appears in F (input) AND R (output)."""
    battery_mass = f["battery_mass"]
    endurance = f["endurance"]
    extra_payload = f["extra_payload"]
    extra_power = f["extra_power"]

    weight = G * (battery_mass + extra_payload)
    actuation_power = C_LIFT * weight ** 2
    total_power = actuation_power + extra_power
    energy = total_power * endurance
    required_mass = energy / ALPHA

    return {
        "battery_mass": required_mass,    # the loop axis
        "report_mass": required_mass,     # mirrored as outer resource
    }


def build_drone():
    F = Ports({
        "endurance": Reals(unit="s"),
        "extra_payload": Reals(unit="kg"),
        "extra_power": Reals(unit="W"),
        "battery_mass": Reals(unit="kg"),
    })
    R = Ports({
        "battery_mass": Reals(unit="kg"),
        "report_mass": Reals(unit="kg"),
    })
    inner = FunctionDP(F, R, drone_inner_h, name="drone-inner")
    return loop(inner, axis="battery_mass", name="drone")


def design(drone, endurance, extra_payload, extra_power):
    return solve(drone, {
        "endurance": endurance,
        "extra_payload": extra_payload,
        "extra_power": extra_power,
    })


if __name__ == "__main__":
    drone = build_drone()
    print(drone)
    print()

    cases = [
        ("Short, light",   60.0, 0.1, 1.0),
        ("Medium, modest", 300.0, 0.5, 5.0),
        ("Longer mission", 600.0, 0.5, 5.0),
        ("Marginal",       600.0, 1.0, 10.0),
        ("Infeasible",    1800.0, 1.0, 10.0),
    ]
    for label, T, m_pay, P_ex in cases:
        res = design(drone, T, m_pay, P_ex)
        print(f"{label}: endurance={T:.0f}s, payload={m_pay:.2f}kg, extra_P={P_ex:.1f}W")
        print(f"   iters={res.iterations}, feasible={res.feasible}, {res.antichain}")
        print()
