"""
Non-trivial composition: a robotic arm with two joints sharing a power bus.

This example shows the operator-overloaded syntax doing real work. The
arm has:

- two **joint** actuators (shoulder and elbow), each with their own torque
  and power characteristics,
- a **controller** that drives both joints and needs sensor input,
- a **sensor** measuring arm position,
- a **power bus** aggregating power demands from joints, controller, sensor,
- a **battery** sized to the mission energy.

The connection graph is not a clean series/parallel/loop: the controller's
power demand depends on the sensor's rate, both joints share the same
battery via the power bus, and the shoulder must support the elbow's mass.

In lambda form, the five-line constraint block at the end of ``make_arm``
would be twenty lines of nested dict lookups. With operator overloading it
reads like an equation sheet.
"""
from __future__ import annotations

from codesign import Module, Reals, System, minimize_cost, solve


G = 9.81


# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------


class Joint(Module):
    """A single revolute joint. F: torque + speed.  R: mass + electrical power."""
    F = {
        "torque": Reals(unit="N*m"),
        "speed":  Reals(unit="rad/s"),
    }
    R = {
        "mass":           Reals(unit="kg"),
        "electric_power": Reals(unit="W"),
    }

    def __init__(self, motor_density: float = 0.20, efficiency: float = 0.85):
        # mass = motor_density * peak_torque ; power = torque*speed / efficiency
        self.motor_density = motor_density
        self.efficiency = efficiency
        super().__init__()

    def h(self, f):
        return {
            "mass":           self.motor_density * f["torque"],
            "electric_power": f["torque"] * f["speed"] / self.efficiency,
        }


class Sensor(Module):
    """A position sensor. F: sample rate. R: power consumption and mass."""
    F = {"sample_rate": Reals(unit="Hz")}
    R = {
        "power": Reals(unit="W"),
        "mass":  Reals(unit="kg"),
    }

    def h(self, f):
        return {
            "power": 0.02 * f["sample_rate"] + 0.5,  # 0.5 W idle plus per-sample cost
            "mass":  0.05,                             # 50 g, fixed
        }


class Controller(Module):
    """The control computer. F: input sample rate, command bandwidth.
    R: own power, own mass.
    """
    F = {
        "input_rate":   Reals(unit="Hz"),
        "command_rate": Reals(unit="Hz"),
    }
    R = {
        "power": Reals(unit="W"),
        "mass":  Reals(unit="kg"),
    }

    def h(self, f):
        # Power grows with both the rate of incoming sensor samples and the
        # rate of outgoing commands; mass is a fixed mainboard.
        return {
            "power": 0.05 * (f["input_rate"] + f["command_rate"]) + 2.0,
            "mass":  0.15,
        }


class Battery(Module):
    F = {"energy": Reals(unit="J")}
    R = {"mass":   Reals(unit="kg")}

    def __init__(self, specific_energy: float = 1.8e6):
        self.specific_energy = specific_energy
        super().__init__()

    def h(self, f):
        return {"mass": f["energy"] / self.specific_energy}


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------


def make_arm() -> System:
    sys = System("robotic_arm")

    # Mission parameters.
    payload_mass      = sys.provides("payload_mass",   unit="kg")
    operating_time    = sys.provides("operating_time", unit="s")
    elbow_speed       = sys.provides("elbow_speed",    unit="rad/s")
    shoulder_speed    = sys.provides("shoulder_speed", unit="rad/s")
    control_rate      = sys.provides("control_rate",   unit="Hz")
    elbow_arm_length  = sys.provides("elbow_arm",      unit="m")
    shoulder_arm_length = sys.provides("shoulder_arm", unit="m")

    # System-level resources.
    total_mass = sys.requires("total_mass", unit="kg")

    # Subsystems.
    elbow      = sys.add("elbow",      Joint())
    shoulder   = sys.add("shoulder",   Joint(motor_density=0.25))   # heavier joint
    sensor     = sys.add("sensor",     Sensor())
    controller = sys.add("controller", Controller())
    battery    = sys.add("battery",    Battery())

    # Mechanical loading. The elbow lifts only the payload at its arm length.
    elbow.torque >= G * payload_mass * elbow_arm_length
    elbow.speed  >= elbow_speed

    # The shoulder lifts the payload and the elbow joint at the shoulder's
    # longer arm length. The elbow.mass dependence creates a cycle through
    # the elbow joint that the Kleene iteration resolves.
    shoulder.torque >= G * (payload_mass + elbow.mass) * shoulder_arm_length
    shoulder.speed  >= shoulder_speed

    # Sensor sample rate must keep up with twice the control loop (Nyquist).
    sensor.sample_rate >= 2.0 * control_rate

    # Controller demand: input rate matches the sensor rate, command rate
    # matches the control loop. Routing both demands through outer F values
    # rather than referencing F ports as values keeps the DSL happy.
    controller.input_rate   >= 2.0 * control_rate
    controller.command_rate >= control_rate

    # Battery sized by total electrical energy over the mission.
    # Power bus aggregates everything that consumes electricity.
    battery.energy >= operating_time * (
        elbow.electric_power
        + shoulder.electric_power
        + controller.power
        + sensor.power
    )

    # Total mass: payload + every subsystem mass.
    total_mass >= (
        payload_mass
        + elbow.mass + shoulder.mass
        + sensor.mass + controller.mass
        + battery.mass
    )

    return sys


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys = make_arm()
    print(sys)
    print()

    arm = sys.build()

    cases = [
        ("Pick-and-place light", dict(
            payload_mass=0.5, operating_time=300.0,
            elbow_speed=2.0, shoulder_speed=1.5,
            control_rate=100.0,
            elbow_arm=0.3, shoulder_arm=0.5,
        )),
        ("Heavier payload", dict(
            payload_mass=2.0, operating_time=600.0,
            elbow_speed=1.5, shoulder_speed=1.0,
            control_rate=200.0,
            elbow_arm=0.3, shoulder_arm=0.5,
        )),
        ("Long-reach precise", dict(
            payload_mass=1.0, operating_time=900.0,
            elbow_speed=1.0, shoulder_speed=0.8,
            control_rate=500.0,
            elbow_arm=0.6, shoulder_arm=0.9,
        )),
    ]
    for label, f in cases:
        result = solve(arm, f, max_iter=300)
        print(f"\n{label}:")
        print(f"   iters={result.iterations}, feasible={result.feasible}")
        if result.feasible:
            for p in result.antichain.points:
                print(f"   total_mass = {p['total_mass']:.2f} kg")
