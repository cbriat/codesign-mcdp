"""
Set-based deterministic uncertainty: Box and Ellipsoid.

The drone from example 7 is extended so its battery has two internal
parameters (specific energy and efficiency) that are known only up to an
uncertainty set. The user-facing question is: under the worst-case point
of that set, how heavy is the drone?

The nominal parameters (specific_energy = 2.0 MJ/kg, efficiency = 0.90)
are chosen so their product, 1.8 MJ/kg of *delivered* energy density,
matches the canonical drone of examples 1/6/7. The nominal solve
therefore converges to the same 0.5492 kg; the uncertainty sets perturb
around that shared reference.

Two sets are exercised:

- a :class:`Box`: rectangular ranges on the two parameters, with each
  range declared in the "more is better" direction so the worst case is
  the single corner where both parameters take their lowest values,
- an :class:`Ellipsoid`: a tilted, correlated set that's smaller than
  the box; the worst case lies on its boundary in the direction of
  badness.

For both, the worst-case mass is compared to the nominal-parameter
mass to highlight how much the uncertainty "costs."
"""
from __future__ import annotations

from codesign import (
    Box,
    Ellipsoid,
    Module,
    Reals,
    System,
    solve,
)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class Battery(Module):
    """Battery whose mass depends on two internal parameters."""
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}

    def __init__(self, specific_energy: float = 2.0e6, efficiency: float = 0.9):
        self.specific_energy = specific_energy
        self.efficiency = efficiency
        super().__init__()

    def h(self, f):
        return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}


class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}
    def h(self, f):
        return {"power": 10.0 * f["lift_force"] ** 2}


# ---------------------------------------------------------------------------
# System assembly. Battery and uncertainty are passed in as constructor
# args so we can build the drone with a different uncertainty per query.
# ---------------------------------------------------------------------------


def make_drone(battery: Battery):
    sys = System("drone")
    endurance     = sys.provides("endurance",     unit="s")
    extra_payload = sys.provides("extra_payload", unit="kg")
    extra_power   = sys.provides("extra_power",   unit="W")
    total_mass    = sys.requires("total_mass",    unit="kg")
    b = sys.add("battery",  battery)
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_power) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_payload)
    total_mass    >= b.mass + extra_payload
    return sys.build()


if __name__ == "__main__":
    f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}

    # ----- Nominal -----
    print("Nominal parameters (specific_energy=2.0 MJ/kg, efficiency=0.90):")
    print("   -> delivered 1.8 MJ/kg, same as the canonical drone (ex. 1/6/7)")
    bat = Battery()
    drone = make_drone(bat)
    r0 = solve(drone, f)
    print(f"   total_mass = {list(r0.antichain.points)[0]['total_mass']:.4f} kg")
    print()

    # ----- Box uncertainty -----
    print("Box uncertainty:")
    print("   specific_energy in [1.7, 2.3] MJ/kg  (more is better)")
    print("   efficiency      in [0.83, 0.97]      (more is better)")
    bat = Battery()
    bat.uncertain_set = Box(
        specific_energy=(1.7e6, 2.3e6, "more_is_better"),
        efficiency=(0.83, 0.97, "more_is_better"),
    )
    drone = make_drone(bat)
    r_box = solve(drone, f, uncertainty=["worst_case"])
    wc = list(r_box.worst_case.antichain.points)[0]["total_mass"]
    print(f"   worst-case total_mass = {wc:.4f} kg")
    print(f"   uncertainty penalty   = "
          f"{wc - list(r0.antichain.points)[0]['total_mass']:+.4f} kg")
    print()

    # ----- Ellipsoid uncertainty -----
    print("Ellipsoid uncertainty (smaller, correlated set):")
    print("   center (2.0 MJ/kg, 0.90), covariance:")
    print("       [[ 1.0e10  -2.0e3 ],")
    print("        [-2.0e3   2.5e-3]]")
    bat = Battery()
    bat.uncertain_set = Ellipsoid(
        center={"specific_energy": 2.0e6, "efficiency": 0.9},
        cov=[
            [1.0e10, -2.0e3],
            [-2.0e3, 2.5e-3],
        ],
        params=["specific_energy", "efficiency"],
        directions={
            "specific_energy": "more_is_better",
            "efficiency":      "more_is_better",
        },
    )
    drone = make_drone(bat)
    r_ell = solve(drone, f, uncertainty=["worst_case"])
    wc_ell = list(r_ell.worst_case.antichain.points)[0]["total_mass"]
    print(f"   worst-case total_mass = {wc_ell:.4f} kg")
    print(f"   uncertainty penalty   = "
          f"{wc_ell - list(r0.antichain.points)[0]['total_mass']:+.4f} kg")
    print()

    print("The ellipsoid worst case is less pessimistic than the box because")
    print("the ellipsoid carves out the implausible corner where both")
    print("parameters are simultaneously at their extremes.")
