"""
Modular motor + chassis + battery co-design.

Three independently-defined subsystems are wired together with a System.
The motor catalog contains four Pareto-incomparable entries (lighter but
more expensive, or cheaper but heavier), so the system returns a genuine
multi-point Pareto front over (total_mass, total_cost).

This is closer to the running co-design example of Censi (2015) Sec. IV:
the choice of motor influences chassis mass (heavier motor needs sturdier
chassis), which in turn influences the torque the motor must supply, which
loops back to the motor choice.
"""
from __future__ import annotations

from codesign import (
    AlgebraicDP,
    CatalogDP,
    CatalogEntry,
    NamedProduct,
    Reals,
    System,
    solve,
    minimize_cost,
)


# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------


def make_motor_catalog() -> CatalogDP:
    """A small catalog of motors with Pareto-incomparable mass/cost tradeoffs.

    Each motor can deliver up to a given peak torque (N*m); heavier motors
    are typically cheaper for the same torque rating.
    """
    F = NamedProduct({"torque": Reals(unit="N*m")})
    R = NamedProduct({
        "mass": Reals(unit="kg"),
        "cost": Reals(unit="USD"),
    })
    catalog = [
        CatalogEntry(
            name="Tiny",
            provides={"torque": 2.0},
            costs={"mass": 0.20, "cost": 30.0},
        ),
        CatalogEntry(
            name="Light-Premium",
            provides={"torque": 8.0},
            costs={"mass": 0.50, "cost": 200.0},
        ),
        CatalogEntry(
            name="Mid-Standard",
            provides={"torque": 8.0},
            costs={"mass": 0.80, "cost": 120.0},
        ),
        CatalogEntry(
            name="Heavy-Budget",
            provides={"torque": 20.0},
            costs={"mass": 1.50, "cost": 90.0},
        ),
        CatalogEntry(
            name="Light-Pro",
            provides={"torque": 20.0},
            costs={"mass": 0.90, "cost": 350.0},
        ),
        CatalogEntry(
            name="XL-Budget",
            provides={"torque": 80.0},
            costs={"mass": 3.50, "cost": 180.0},
        ),
        CatalogEntry(
            name="XL-Pro",
            provides={"torque": 80.0},
            costs={"mass": 2.20, "cost": 700.0},
        ),
    ]
    return CatalogDP(F=F, R=R, catalog=catalog, name="motor")


def make_chassis(coefficient: float = 0.6) -> AlgebraicDP:
    """Chassis mass and cost scale with the load it must carry.

    F: total payload it must support (kg).
    R: chassis mass and chassis cost.
    """
    return AlgebraicDP(
        F=NamedProduct({"load": Reals(unit="kg")}),
        R=NamedProduct({
            "mass": Reals(unit="kg"),
            "cost": Reals(unit="USD"),
        }),
        equations={
            "mass": lambda f: coefficient * f["load"],
            "cost": lambda f: 20.0 * f["load"],
        },
        name="chassis",
    )


def make_battery(specific_energy: float = 1.8e6) -> AlgebraicDP:
    return AlgebraicDP(
        F=NamedProduct({"energy": Reals(unit="J")}),
        R=NamedProduct({
            "mass": Reals(unit="kg"),
            "cost": Reals(unit="USD"),
        }),
        equations={
            "mass": lambda f: f["energy"] / specific_energy,
            "cost": lambda f: 0.05 * f["energy"] / 3.6e3,  # $0.05 / Wh
        },
        name="battery",
    )


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------


def make_vehicle() -> System:
    """A small electric vehicle: motor + chassis + battery, plus payload.

    Outer functionalities: payload (kg the user wants to carry),
    mission_energy (J the battery must store).

    Outer resources: total_mass and total_cost.

    Internal coupling:
      - chassis must support payload + motor + battery
      - motor must deliver torque proportional to total moving mass
      - battery's energy is supplied externally as a functionality

    The chassis-motor coupling creates the feedback loop.
    """
    G = 9.81
    TORQUE_PER_KG = 0.25   # N*m of motor torque required per kg of moving mass

    sys = System("vehicle")

    sys.provides("payload", unit="kg")
    sys.provides("mission_energy", unit="J")
    sys.requires("total_mass", unit="kg")
    sys.requires("total_cost", unit="USD")

    sys.add("motor", make_motor_catalog())
    sys.add("chassis", make_chassis())
    sys.add("battery", make_battery())

    # The chassis must support payload plus motor mass plus battery mass.
    sys.constrain(
        "chassis.load",
        lambda x: x["payload"] + x["motor.mass"] + x["battery.mass"],
    )
    # The motor torque is sized by the total moving mass.
    sys.constrain(
        "motor.torque",
        lambda x: TORQUE_PER_KG * G * (
            x["payload"] + x["chassis.mass"] + x["battery.mass"]
        ),
    )
    # The battery's energy demand is just the externally-supplied mission_energy.
    sys.constrain(
        "battery.energy",
        lambda x: x["mission_energy"],
    )

    # Aggregations.
    sys.constrain(
        "total_mass",
        lambda x: x["payload"] + x["motor.mass"] + x["chassis.mass"] + x["battery.mass"],
    )
    sys.constrain(
        "total_cost",
        lambda x: x["motor.cost"] + x["chassis.cost"] + x["battery.cost"],
    )

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
        ("Small parcel",   dict(payload=2.0,  mission_energy=2.0e5)),
        ("Medium load",    dict(payload=10.0, mission_energy=1.0e6)),
        ("Heavy + long",   dict(payload=20.0, mission_energy=5.0e6)),
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
        # Cost-only scalarisation: cheapest design that satisfies the spec.
        cheapest = minimize_cost(result, cost_fn=lambda r: r["total_cost"])
        if cheapest is not None:
            print(
                f"   cheapest: total_mass={cheapest['total_mass']:.2f} kg, "
                f"total_cost=${cheapest['total_cost']:.2f}"
            )
