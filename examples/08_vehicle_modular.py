"""
Modular vehicle: motor catalog + chassis + battery, with MCDPL-style constraints.

Three independent subsystems are wired with the operator-overloaded
``>=`` syntax. The motor catalog has Pareto-incomparable entries, so the
system-level result is a multi-point Pareto front over total mass and
total cost.
"""
from __future__ import annotations

from codesign import (
    CatalogDP,
    CatalogEntry,
    Module,
    Ports,
    Reals,
    System,
    minimize_cost,
    solve,
)


# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------


def make_motor_catalog() -> CatalogDP:
    """Discrete catalog of motors with Pareto-incomparable mass/cost tradeoffs.

    CatalogDP doesn't fit the standard Module pattern (multi-valued antichains
    are best declared at construction time), so we keep this one as a
    function returning a plain DesignProblem.
    """
    return CatalogDP(
        F=Ports({"torque": Reals(unit="N*m")}),
        R=Ports({
            "mass": Reals(unit="kg"),
            "cost": Reals(unit="USD"),
        }),
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


class Chassis(Module):
    """Chassis mass and cost both scale with the supported load."""
    F = {"load": Reals(unit="kg")}
    R = {
        "mass": Reals(unit="kg"),
        "cost": Reals(unit="USD"),
    }

    def __init__(self, mass_per_kg: float = 0.6, cost_per_kg: float = 20.0):
        self.mass_per_kg = mass_per_kg
        self.cost_per_kg = cost_per_kg
        super().__init__()

    def h(self, f):
        return {
            "mass": self.mass_per_kg * f["load"],
            "cost": self.cost_per_kg * f["load"],
        }


class Battery(Module):
    """Battery sized by energy storage required."""
    F = {"energy": Reals(unit="J")}
    R = {
        "mass": Reals(unit="kg"),
        "cost": Reals(unit="USD"),
    }

    def __init__(self, specific_energy: float = 1.8e6, cost_per_wh: float = 0.05):
        self.specific_energy = specific_energy
        self.cost_per_wh = cost_per_wh
        super().__init__()

    def h(self, f):
        return {
            "mass": f["energy"] / self.specific_energy,
            "cost": self.cost_per_wh * f["energy"] / 3.6e3,  # Wh -> J conversion
        }


# ---------------------------------------------------------------------------
# System assembly with operator-overloaded constraints
# ---------------------------------------------------------------------------


def make_vehicle() -> System:
    G = 9.81
    TORQUE_PER_KG = 0.25

    sys = System("vehicle")

    # Outer interface as Port handles.
    payload        = sys.provides("payload",        unit="kg")
    mission_energy = sys.provides("mission_energy", unit="J")
    total_mass     = sys.requires("total_mass",     unit="kg")
    total_cost     = sys.requires("total_cost",     unit="USD")

    # Subsystems as ModuleHandles.
    motor   = sys.add("motor",   make_motor_catalog())
    chassis = sys.add("chassis", Chassis())
    battery = sys.add("battery", Battery())

    # Connection constraints.
    chassis.load   >= payload + motor.mass + battery.mass
    motor.torque   >= TORQUE_PER_KG * G * (payload + chassis.mass + battery.mass)
    battery.energy >= mission_energy

    # Aggregations into outer R ports.
    total_mass >= payload + motor.mass + chassis.mass + battery.mass
    total_cost >= motor.cost + chassis.cost + battery.cost

    return sys


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys = make_vehicle()
    print(sys)
    print()

    vehicle = sys.build()
    print(vehicle)

    cases = [
        ("Small parcel", dict(payload=2.0,  mission_energy=2.0e5)),
        ("Medium load",  dict(payload=10.0, mission_energy=1.0e6)),
        ("Heavy + long", dict(payload=20.0, mission_energy=5.0e6)),
    ]
    for label, f in cases:
        result = solve(vehicle, f, max_iter=200)
        f_str = ", ".join(f"{k}={v}" for k, v in f.items())
        print(f"\n{label}: {f_str}")
        print(f"   iters={result.iterations}, feasible={result.feasible}")
        if not result.feasible:
            continue
        print(f"   Pareto front ({len(result.antichain.points)} points):")
        for p in result.antichain.points:
            print(f"      total_mass={p['total_mass']:6.2f} kg,  "
                  f"total_cost=${p['total_cost']:7.2f}")
        cheapest = minimize_cost(result, cost_fn=lambda r: r["total_cost"])
        if cheapest is not None:
            print(
                f"   cheapest: total_mass={cheapest['total_mass']:.2f} kg, "
                f"total_cost=${cheapest['total_cost']:.2f}"
            )
