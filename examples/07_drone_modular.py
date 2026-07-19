"""
Modular drone, MCDPL-style.

The same MCDP as ``examples/01_drone.py``, expressed with the new
operator-overloaded constraint syntax. Battery and actuator are defined
as ``Module`` subclasses with class-level F/R declarations, and the
``System`` is wired together with ``>=`` constraints that read like the
mathematical inequalities they represent.

Equivalent to the lambda-based ``sys.constrain(target, lambda x: ...)``
form. Both styles compile to the same internal constraint list and
produce identical results.

Run:  python -m examples.07_drone_modular
Expected output: the System wiring summary, the compiled DP signature, and
one block per mission with the Kleene iteration count, feasibility, and
converged total mass.
"""
from __future__ import annotations

from codesign import Module, Reals, System, solve


# ---------------------------------------------------------------------------
# Subsystems: declarative class definitions.
# ---------------------------------------------------------------------------


class Battery(Module):
    """Sizes a battery's mass from its required capacity (via specific energy)."""
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}

    def __init__(self, specific_energy: float = 1.8e6):
        self.specific_energy = specific_energy
        super().__init__()

    def h(self, f):
        return {"mass": f["capacity"] / self.specific_energy}


class Actuator(Module):
    """A simple aerodynamic actuator: power scales as lift squared."""
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}

    def __init__(self, c_lift: float = 10.0):
        self.c_lift = c_lift
        super().__init__()

    def h(self, f):
        return {"power": self.c_lift * f["lift_force"] ** 2}


# ---------------------------------------------------------------------------
# System assembly.
# ---------------------------------------------------------------------------


def make_drone() -> System:
    G = 9.81

    sys = System("drone")

    # Outer interface: capture each declaration as a Port handle for use in
    # expressions.
    endurance     = sys.provides("endurance",     unit="s")
    extra_payload = sys.provides("extra_payload", unit="kg")
    extra_power   = sys.provides("extra_power",   unit="W")
    total_mass    = sys.requires("total_mass",    unit="kg")

    # Subsystems: capture each as a ModuleHandle so its ports are accessible
    # via attribute lookup.
    battery  = sys.add("battery",  Battery())
    actuator = sys.add("actuator", Actuator())

    # Connection constraints: each line reads like an inequality from a
    # textbook. The LHS is the F port (or outer R) being constrained; the
    # RHS is an algebraic expression involving outer F values and module R
    # ports.
    battery.capacity    >= (actuator.power + extra_power) * endurance
    actuator.lift_force >= G * (battery.mass + extra_payload)
    total_mass          >= battery.mass + extra_payload

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
