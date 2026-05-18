"""
Modular drone: the same MCDP as ``examples/01_drone.py``, but assembled from
two independent subsystems (Battery and Actuator) plus connection
constraints, rather than one monolithic FunctionDP.

The battery and actuator are defined completely independently. They have
their own functionality (capacity, lift_force) and resources (mass, power).
The System ties them together with three constraints:

    battery.capacity   >= (actuator.power + extra_power) * endurance
    actuator.lift_force >= 9.81 * (battery.mass + extra_payload)
    total_mass         >= battery.mass + extra_payload

The first two have a feedback loop: actuator.power depends on battery.mass
and vice versa. The System builder closes the loop automatically over the
bundle of all subsystem R ports.
"""
from __future__ import annotations

from codesign import (
    AlgebraicDP,
    NamedProduct,
    Reals,
    System,
    solve,
)


# ---------------------------------------------------------------------------
# Subsystems: each defined independently, in isolation.
# ---------------------------------------------------------------------------


def make_battery(specific_energy_jpkg: float = 1.8e6) -> AlgebraicDP:
    """A battery: capacity in -> mass out, via specific energy."""
    return AlgebraicDP(
        F=NamedProduct({"capacity": Reals(unit="J")}),
        R=NamedProduct({"mass": Reals(unit="kg")}),
        equations={"mass": lambda f: f["capacity"] / specific_energy_jpkg},
        name="battery",
    )


def make_actuator(c_lift: float = 10.0) -> AlgebraicDP:
    """A simple aerodynamic actuator: power scales as the square of lift."""
    return AlgebraicDP(
        F=NamedProduct({"lift_force": Reals(unit="N")}),
        R=NamedProduct({"power": Reals(unit="W")}),
        equations={"power": lambda f: c_lift * f["lift_force"] ** 2},
        name="actuator",
    )


# ---------------------------------------------------------------------------
# System: wire the subsystems together.
# ---------------------------------------------------------------------------


def make_drone() -> System:
    G = 9.81

    sys = System("drone")

    # Outer interface.
    sys.provides("endurance", unit="s")
    sys.provides("extra_payload", unit="kg")
    sys.provides("extra_power", unit="W")
    sys.requires("total_mass", unit="kg")

    # Subsystems.
    sys.add("battery", make_battery())
    sys.add("actuator", make_actuator())

    # Connection constraints.
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

    return sys


if __name__ == "__main__":
    sys = make_drone()
    print(sys)
    print()

    drone = sys.build()
    print(drone)

    cases = [
        ("Short, light",    dict(endurance=60.0,  extra_payload=0.10, extra_power=1.0)),
        ("Medium, modest",  dict(endurance=300.0, extra_payload=0.50, extra_power=5.0)),
        ("Longer mission",  dict(endurance=600.0, extra_payload=0.50, extra_power=5.0)),
        ("Marginal",        dict(endurance=600.0, extra_payload=1.00, extra_power=10.0)),
        ("Infeasible",      dict(endurance=1800.0,extra_payload=1.00, extra_power=10.0)),
    ]
    for label, f in cases:
        result = solve(drone, f, max_iter=200)
        f_str = ", ".join(f"{k}={v}" for k, v in f.items())
        print(
            f"\n{label}: {f_str}"
            f"\n   iters={result.iterations}, feasible={result.feasible}, "
            f"{result.antichain}"
        )
