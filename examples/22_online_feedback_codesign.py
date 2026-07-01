"""
Example 22: online feedback co-design of an adaptive sensor node.

Every temporal example so far plans offline: the whole horizon is solved
in advance and a policy is read out. This example closes the loop. A
solar-powered environmental sensor node runs in the field, and at each
control step it senses its current battery charge, reads the current data
requirement and environmental conditions (light, temperature), re-solves
its co-design at those live conditions, applies the cheapest feasible
configuration, and repeats. The plan is never trusted to match reality:
the next configuration is chosen from the measured state, so when
conditions diverge from any nominal forecast the loop simply re-solves
against what actually happened. That is feedback, not open-loop replay.

This is the co-design instance of control co-design (CCD) in its nested,
receding-horizon form, here the myopic variant: re-solve a single static
co-design at the current conditions each step. The model is known;
measurements update the carried state and the conditions, not the
co-design model itself (learning the model online is a later extension).

The node can run in three sensing configurations, each a co-design
problem that sizes the radio and compute against the demanded data rate
and returns an energy draw and an operations cost:

    low_power : minimal sampling and store-and-forward. Cheap, low energy,
                low data quality. Always feasible.
    nominal   : regular sampling and periodic uplink. Moderate on all axes.
    high_rate : dense sampling and live uplink. High data quality, high
                energy, feasible only when the demanded rate is high enough
                to justify it and charge allows.

The environment drives two things that change over the run: the demanded
data rate (a storm front raises the sampling requirement mid-run) and the
solar input (day/night cycle changes recharge). The loop adapts the
configuration online to both, and the audit log records every decision,
condition, and outcome, which is the artefact a field operator wants.

Run directly to simulate a multi-step deployment and print the closed-loop
schedule and audit trail, showing the node escalating to high_rate during
the storm and falling back when charge runs low at night.
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
    System,
    run_online_codesign,
)

BATTERY_CAPACITY = 100.0


# ---------------------------------------------------------------------------
# Configuration builder: a co-design problem sizing the node for a demanded
# data rate, returning (energy, ops). Infeasible above its rate ceiling.
# ---------------------------------------------------------------------------
def build_config(name, *, energy_base, energy_per_rate, ops, rate_ceiling,
                 rate_floor=0.0):
    """Build one sensing configuration as a co-design problem.

    The configuration demands both a data ``rate`` and the available
    ``charge`` (passed as functionality from the measured battery state).
    It is infeasible if its energy draw exceeds the available charge, which
    is how the measured state gates the co-design in the closed loop: a
    hungry configuration simply cannot be selected when the battery is low.

    Parameters
    ----------
    name : str
    energy_base, energy_per_rate : float
        Energy draw is ``energy_base + energy_per_rate * rate``.
    ops : float
        Operations-cost proxy (minimised).
    rate_ceiling, rate_floor : float
        Rate band the configuration supports; outside it, infeasible.
    """
    s = System(name)
    rate = s.provides("rate", unit="")
    charge = s.provides("charge", unit="")
    s.requires("energy", unit="")
    s.requires("ops", unit="")

    def energy_eq(f, b=energy_base, k=energy_per_rate, cc=rate_ceiling,
                  fl=rate_floor):
        if not (fl <= f["rate"] <= cc):
            return float("inf")
        draw = b + k * f["rate"]
        # Infeasible if the draw exceeds the charge available this step.
        return draw if draw <= f["charge"] else float("inf")

    def ops_eq(f, o=ops, cc=rate_ceiling, fl=rate_floor, b=energy_base,
               k=energy_per_rate):
        if not (fl <= f["rate"] <= cc):
            return float("inf")
        draw = b + k * f["rate"]
        return o if draw <= f["charge"] else float("inf")

    node = s.add("node", AlgebraicDP(
        F=Ports({"rate": Reals(unit=""), "charge": Reals(unit="")}),
        R=Ports({"energy": Reals(), "ops": Reals()}),
        equations={"energy": energy_eq, "ops": ops_eq},
    ))
    node.rate >= rate
    node.charge >= charge
    s.constrain("energy", lambda x: x["node.energy"])
    s.constrain("ops", lambda x: x["node.ops"])
    return s.build()


LOW_POWER = Architecture(
    "low_power",
    build_config("low_power", energy_base=2.0, energy_per_rate=1.0, ops=5.0,
                 rate_ceiling=100.0),
    tags={"config": "low_power"},
)
NOMINAL = Architecture(
    "nominal",
    build_config("nominal", energy_base=5.0, energy_per_rate=2.0, ops=3.0,
                 rate_ceiling=100.0, rate_floor=2.0),
    tags={"config": "nominal"},
)
HIGH_RATE = Architecture(
    "high_rate",
    build_config("high_rate", energy_base=10.0, energy_per_rate=3.0, ops=1.0,
                 rate_ceiling=100.0, rate_floor=6.0),
    tags={"config": "high_rate"},
)
CONFIGS = [LOW_POWER, NOMINAL, HIGH_RATE]


# ---------------------------------------------------------------------------
# The deployment scenario: a day/night solar cycle with a storm front that
# raises the data-rate requirement for a few steps.
# ---------------------------------------------------------------------------
N_STEPS = 12

# Demanded data rate per step. A storm at steps 4 to 7 raises the demand.
DEMAND_RATE = [3, 3, 3, 3, 8, 9, 9, 7, 3, 3, 3, 3]

# Solar recharge per step: high by day, dipping during the storm, then
# recovering as the weather clears so the node can climb back.
SOLAR = [14, 14, 14, 14, 12, 10, 8, 8, 10, 12, 14, 14]


def cost_fn(point):
    """Minimise operations cost; energy is tracked via the plant."""
    return point["ops"]


def make_scenario():
    """Return the sensor, requirement, environment, and plant callables.

    The plant holds the true battery charge and advances it by subtracting
    the chosen configuration's energy draw and adding solar recharge. The
    sensor reads that true charge back, closing the loop.
    """
    # Mutable cell holding the true state of charge, updated by the plant
    # and read by the sensor. This stands in for the physical battery.
    true_soc = {"value": BATTERY_CAPACITY}

    def sensor(step, prev_state):
        # Read back the true charge (in deployment, an ADC reading).
        return true_soc["value"]

    def requirement(step, measured_soc):
        # The demanded data rate at this step plus the available charge, so
        # the co-design is gated by the measured battery: an energy-hungry
        # configuration is infeasible when charge is low. This is the
        # feedback path, the measured state enters the solve directly.
        return {"rate": float(DEMAND_RATE[step]), "charge": measured_soc}

    def environment(step, measured_soc):
        return {"solar": SOLAR[step], "soc": measured_soc}

    def plant(step, measured_soc, arch_name, point):
        # Advance the true battery: subtract draw, add solar, clip to
        # [0, capacity]. If a configuration would take charge below zero it
        # should not have been chosen; guard by flooring at 0 and letting
        # the next step's solve see the depleted state.
        draw = point["energy"]
        new = measured_soc - draw + SOLAR[step]
        new = max(0.0, min(BATTERY_CAPACITY, new))
        true_soc["value"] = new
        return new

    return sensor, requirement, environment, plant


def main():
    sensor, requirement, environment, plant = make_scenario()

    print("Online feedback co-design of an adaptive sensor node")
    print("=" * 60)
    print(f"{N_STEPS} steps, battery capacity {BATTERY_CAPACITY:.0f}")
    print("configs:", ", ".join(c.name for c in CONFIGS))
    print("storm raises demand at steps 4-7; solar falls at night (steps 6+)")
    print()

    result = run_online_codesign(
        CONFIGS,
        n_steps=N_STEPS,
        sensor=sensor,
        requirement=requirement,
        environment=environment,
        plant=plant,
        cost_fn=cost_fn,
        initial_state=BATTERY_CAPACITY,
    )

    print("Closed-loop audit trail:")
    print(f"  {'t':>2}  {'soc_in':>6}  {'rate':>4}  {'solar':>5}  "
          f"{'config':<10} {'energy':>6}  {'ops':>4}")
    for step in result.steps:
        soc = step.measured_state
        rate = step.requirement["rate"]
        solar = step.environment["solar"]
        e = step.point["energy"] if step.feasible else float("nan")
        o = step.point["ops"] if step.feasible else float("nan")
        cfg = step.architecture if step.feasible else "INFEASIBLE"
        print(f"  {step.step:>2}  {soc:>6.1f}  {rate:>4.0f}  {solar:>5.0f}  "
              f"{cfg:<10} {e:>6.1f}  {o:>4.1f}")
    print()
    print(f"schedule: {' -> '.join(result.schedule)}")
    print(f"total ops cost = {result.total_cost:.1f}, "
          f"feasible = {result.feasible}")
    print()
    print("The node escalates to high_rate during the storm (steps 4-7) when")
    print("the demanded rate justifies it, then the closed loop reads the")
    print("depleted night-time charge and the re-solve falls back to cheaper")
    print("configurations. No offline plan is followed: each step is solved")
    print("against the measured battery state, so the schedule adapts to how")
    print("the deployment actually unfolds.")


if __name__ == "__main__":
    main()
