"""
Rebuild the Fig. 48 drone with the MCDPL-style declarative builder.

This is the same MCDP as ``examples/01_drone.py`` but expressed in
``codesign.mcdpl.MCDP(...)`` notation, which mirrors the paper's syntax:

    mcdp {
        provides endurance [s]
        provides extra_payload [kg]
        provides extra_power [W]
        requires battery_mass [kg]

        # ... constraint equations ...

        requires mass for battery     # i.e. close the loop on battery_mass
    }

Compare with examples/01_drone.py to see how the same model looks when
written purely through the operator API (FunctionDP + Loop).

Run:  python -m examples.06_drone_mcdpl_syntax
Expected output: the built drone DP signature, then one block per mission
with the Kleene iteration count, feasibility, and converged battery mass --
matching the numbers from examples/01_drone.py.
"""
from __future__ import annotations

from codesign import MCDP, solve


def make_drone():
    ALPHA = 1.8e6   # Li-ion specific energy (J/kg)
    G = 9.81
    C_LIFT = 10.0   # actuator coefficient (W per N^2 of lift)

    with MCDP("drone") as m:
        # ---- functionalities (what the design must deliver) -------------
        m.provides("endurance", unit="s")
        m.provides("extra_payload", unit="kg")
        m.provides("extra_power", unit="W")
        # battery_mass also appears as a functionality because it's the
        # loop variable; loop_on closes it below.
        m.provides("battery_mass", unit="kg")

        # ---- resources (what the design needs) --------------------------
        m.requires("battery_mass", unit="kg")    # loop axis
        m.requires("report_mass", unit="kg")     # mirror for visibility

        # ---- co-design constraints --------------------------------------
        def battery_mass_eq(f):
            # battery_mass >= energy / ALPHA, where energy = power * endurance,
            # power = c_lift * lift^2 + extra_power, lift = (battery + payload) * g
            lift = (f["battery_mass"] + f["extra_payload"]) * G
            actuator_power = C_LIFT * lift * lift
            total_power = actuator_power + f["extra_power"]
            energy = total_power * f["endurance"]
            return energy / ALPHA

        m.constraint("battery_mass", battery_mass_eq)
        m.constraint("report_mass", battery_mass_eq)  # mirror

        # Close the recursive constraint.
        m.loop_on("battery_mass")

    return m.build()


if __name__ == "__main__":
    drone = make_drone()
    print(drone)

    cases = [
        ("Short, light",    dict(endurance=60.0,  extra_payload=0.10, extra_power=1.0)),
        ("Medium, modest",  dict(endurance=300.0, extra_payload=0.50, extra_power=5.0)),
        ("Longer mission",  dict(endurance=600.0, extra_payload=0.50, extra_power=5.0)),
        ("Infeasible",      dict(endurance=1800.0,extra_payload=1.00, extra_power=10.0)),
    ]
    for label, f in cases:
        result = solve(drone, f, max_iter=80)
        print(
            f"\n{label}: " + ", ".join(f"{k}={v}" for k, v in f.items()) +
            f"\n   iters={result.iterations}, feasible={result.feasible}, "
            f"{result.antichain}"
        )
