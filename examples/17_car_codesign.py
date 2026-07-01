"""
Example 17: full-vehicle co-design (ICE, hybrid, electric).

This example pushes the codesign-mcdp framework to one of the
canonical engineering co-design problems: choosing the right
combination of powertrain, chassis, energy storage, and auxiliary
components for a passenger vehicle, given a target mission.

The example is deliberately *granular*: rather than modelling the
powertrain as a single block, every meaningful subsystem appears as
its own MCDP module with its own F / R port pair. This surfaces the
many interacting design constraints that automotive engineers reason
about implicitly and shows the framework's reach beyond textbook
two- or three-module problems.

Three powertrain architectures are modelled in parallel:

1. **Internal combustion (ICE)** -- a conventional gasoline or
   diesel layout with engine block, forced induction, fuel injection,
   exhaust aftertreatment, cooling, lubrication, multi-speed
   transmission, mechanical differential, fuel tank, 12V electrical
   system.
2. **Hybrid (HEV)** -- a power-split hybrid with a small Atkinson-
   cycle engine, a small high-voltage battery, an electric motor /
   generator, planetary power-split gearbox, plus the rest of the
   ICE accessories at reduced sizing.
3. **Battery electric (EV)** -- a fully electric drivetrain with
   large traction motor (single or dual), large high-voltage battery
   pack, power electronics, onboard charger, single-speed reducer,
   and battery thermal management. No engine, no fuel tank.

All three architectures share the chassis-and-running-gear modules
(body frame, suspension, brakes, steering, tires, wheels) and the
auxiliary modules (HVAC, interior, safety systems, lighting and
infotainment). The macro objectives compared across architectures
are: production cost (USD), curb weight (kg), energy consumption
(L/100 km equivalent for ICE/HEV, kWh/100 km for EV), tank-to-wheel
CO2 (g/km), maintenance cost (USD/year), and durability (km until
major overhaul).

The "weight death spiral" cycle that automotive engineers fight is
made explicit here. Every weight-sensitive subsystem (suspension,
brakes, tire load rating, engine power demand) reads the vehicle's
design curb weight as an F port input. The macro aggregation then
sums all module weights into the curb_weight outer R, and the
Kleene iteration converges this to its self-consistent fixed point.

Parameter sources
-----------------
- Genta, G. and Morello, L. (2009). *The Automotive Chassis*.
  Vol. 1: Components Design. Springer. Suspension stiffness,
  damper characteristics, brake sizing, tire load ratings.
- Bosch (2018). *Bosch Automotive Handbook*, 10th ed. Engine
  specific power, exhaust aftertreatment, electrical loads.
- Pulkrabek, W. (2003). *Engineering Fundamentals of the Internal
  Combustion Engine*, 2nd ed. Thermal efficiency, heat rejection,
  fuel flow rates.
- Heywood, J. (2018). *Internal Combustion Engine Fundamentals*,
  2nd ed. Combustion thermodynamics, BSFC ranges, durability.
- Hofmann, P. (2014). *Hybridfahrzeuge*, 2nd ed. Springer.
  Power-split topologies, sizing rules, efficiency models.
- Naunheimer, H. et al. (2011). *Automotive Transmissions*, 2nd ed.
  Springer. Gearbox losses, weights, costs.
- IEA (2023). *Global EV Outlook*. Battery pack cost ($/kWh) and
  pack-level energy density (Wh/kg) trends 2020-2030.
- Nemry, F. et al. (2008). *Environmental Improvement of Passenger
  Cars*. JRC Scientific and Technical Reports. CO2 emission factors,
  fuel densities, well-to-wheel.
- EPA (2024). *Automotive Trends Report*. Fleet-average fuel
  economy, weight class data, drag coefficient data.
- Larminie, J. and Lowry, J. (2012). *Electric Vehicle Technology
  Explained*, 2nd ed. Motor efficiency maps, inverter losses,
  charging system architecture.
- Mock, P. et al. (2023). *European Vehicle Market Statistics
  Pocketbook 2023/24*. International Council on Clean Transportation.
  Mass-vs-mission scatter data for calibration.

All numerical values are illustrative, drawn from the published
ranges in the references above. They are *not* claimed to reflect
any specific OEM product; they are calibrated to roughly match
fleet averages for each technology class.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    AlgebraicDP,
    Module,
    Ports,
    Reals,
    System,
    solve,
)


# ===========================================================================
# Engineering constants
# ===========================================================================

GRAVITY = 9.81                          # m/s^2

# Atmosphere (sea level, ISA)
AIR_DENSITY = 1.225                     # kg/m^3

# Fuel properties (gasoline as the default ICE fuel; diesel handled in engine
# catalogue entry by overriding these via the per-engine attributes)
GASOLINE_DENSITY      = 0.745           # kg/L
GASOLINE_LHV_MJ_PER_L = 32.0            # MJ/L lower heating value
DIESEL_DENSITY        = 0.832           # kg/L
DIESEL_LHV_MJ_PER_L   = 35.8            # MJ/L lower heating value

# Tank-to-wheel CO2 emission factors (kg CO2 per L of fuel burned)
GASOLINE_CO2_PER_L = 2.31               # kg CO2/L
DIESEL_CO2_PER_L   = 2.68               # kg CO2/L

# Grid carbon intensity for EV charging (tank-to-wheel from EV perspective).
# Default: EU-27 average grid intensity 2024 per EEA. Override per scenario.
GRID_CO2_PER_KWH = 0.295                # kg CO2/kWh

# Driver mass placeholder (added to curb weight to give "as-driven" mass).
# Per ISO 1176, driver is included in curb weight via a 75 kg allowance.
DRIVER_MASS_KG = 75.0

# Standard cargo / passenger mass for payload checks (per ISO 1176).
PASSENGER_MASS_KG = 75.0

# Conversion factors
KMH_TO_MS = 1.0 / 3.6
MS_TO_KMH = 3.6


# ===========================================================================
# Physics helpers
# ===========================================================================


def drag_power_kw(speed_kmh: float, cd: float, frontal_area_m2: float) -> float:
    """Aerodynamic drag power at the given cruise speed.

    P_drag = 0.5 * rho * Cd * A * v^3, returned in kW.
    """
    v = speed_kmh * KMH_TO_MS
    return 0.5 * AIR_DENSITY * cd * frontal_area_m2 * (v ** 3) / 1000.0


def rolling_power_kw(speed_kmh: float, mass_kg: float,
                     crr: float) -> float:
    """Rolling-resistance power at the given speed.

    P_roll = Crr * m * g * v, returned in kW.
    """
    v = speed_kmh * KMH_TO_MS
    return crr * mass_kg * GRAVITY * v / 1000.0


def cruise_road_load_kw(speed_kmh: float, mass_kg: float, cd: float,
                        frontal_area_m2: float, crr: float) -> float:
    """Steady-state road load at the wheels (drag + rolling), in kW."""
    return (drag_power_kw(speed_kmh, cd, frontal_area_m2)
            + rolling_power_kw(speed_kmh, mass_kg, crr))


def acceleration_power_kw(target_0_100_s: float, mass_kg: float) -> float:
    """Minimum average power required to reach 100 km/h in the target time.

    Uses a simple energy balance ignoring drag and roll during accel:
    E = 0.5 * m * v^2, P = E / t. Realistic peak power is roughly 2x the
    average, so for a 10 s 0-to-100 target on a 1500 kg car the average
    is about 58 kW and the peak is around 120 kW, matching the upper
    rated power of mainstream sedans.
    """
    if target_0_100_s <= 0.0:
        return float("inf")
    v = 100.0 * KMH_TO_MS
    avg = 0.5 * mass_kg * (v ** 2) / target_0_100_s / 1000.0
    return 2.0 * avg


def top_speed_power_kw(max_speed_kmh: float, mass_kg: float, cd: float,
                       frontal_area_m2: float, crr: float) -> float:
    """Peak power required to sustain the rated top speed."""
    # 10% headroom for grade and accessory load.
    return 1.10 * cruise_road_load_kw(max_speed_kmh, mass_kg, cd,
                                       frontal_area_m2, crr)


def harmonic_mean(values: List[float]) -> float:
    """Harmonic mean of a list of positive numbers."""
    inv_sum = 0.0
    for v in values:
        if v <= 0:
            return 0.0
        inv_sum += 1.0 / v
    return len(values) / inv_sum if inv_sum > 0 else 0.0


# ===========================================================================
# Common chassis and body modules (shared by ICE, hybrid, and EV)
# ===========================================================================


class BodyFrame(Module):
    """Vehicle body-in-white plus exterior trim.

    Represents the structural frame, body panels, doors, glass, and
    bumpers as a single subsystem characterised by its body style.
    The body style determines the aerodynamic profile (Cd, frontal
    area), the cargo and passenger envelope, and the structural mass.

    Inputs (F):
        target_passengers : minimum passenger capacity in seated
            adults the body must accommodate.
        target_cargo_L    : minimum cargo volume the body must
            accommodate (litres).

    Outputs (R):
        weight    : structural mass of body, frame, panels, doors,
            glass, bumpers (kg, including driver allowance).
        cost      : material + tooling + assembly cost (USD).
        cd        : aerodynamic drag coefficient (dimensionless).
        frontal_area : projected frontal area (m^2).
        passengers   : seated passenger capacity (people).
        cargo_L      : actual cargo volume (litres).
        durability   : structural fatigue life (km).
    """
    F = {
        "target_passengers": Reals(unit="people"),
        "target_cargo_L":    Reals(unit="L"),
    }
    R = {
        "weight":       Reals(unit="kg"),
        "cost":         Reals(unit="USD"),
        "cd":           Reals(unit=""),
        "frontal_area": Reals(unit="m^2"),
        "passengers":   Reals(unit="people"),
        "cargo_L":      Reals(unit="L"),
        "durability":   Reals(unit="km"),
    }

    def __init__(self, style_name: str, *,
                 base_weight_kg: float, base_cost_usd: float,
                 cd: float, frontal_area_m2: float,
                 max_passengers: int, max_cargo_L: float,
                 durability_km: float):
        self.style_name = style_name
        self.base_weight_kg = base_weight_kg
        self.base_cost_usd = base_cost_usd
        self.cd_val = cd
        self.frontal_area_m2 = frontal_area_m2
        self.max_passengers = max_passengers
        self.max_cargo_L = max_cargo_L
        self.durability_km = durability_km
        super().__init__()

    def h(self, f):
        if f["target_passengers"] > self.max_passengers:
            return {"weight": float("inf"), "cost": float("inf"),
                    "cd": self.cd_val, "frontal_area": self.frontal_area_m2,
                    "passengers": self.max_passengers,
                    "cargo_L": self.max_cargo_L,
                    "durability": 0.0}
        if f["target_cargo_L"] > self.max_cargo_L:
            return {"weight": float("inf"), "cost": float("inf"),
                    "cd": self.cd_val, "frontal_area": self.frontal_area_m2,
                    "passengers": self.max_passengers,
                    "cargo_L": self.max_cargo_L,
                    "durability": 0.0}
        return {
            "weight":       self.base_weight_kg,
            "cost":         self.base_cost_usd,
            "cd":           self.cd_val,
            "frontal_area": self.frontal_area_m2,
            "passengers":   float(self.max_passengers),
            "cargo_L":      float(self.max_cargo_L),
            "durability":   self.durability_km,
        }


class FrontSuspension(Module):
    """Front-axle suspension: springs, dampers, anti-roll bar, control arms.

    Sized to the vehicle's design curb weight (specifically, the front
    axle's share, taken as 55% for FWD/AWD and 45% for RWD; we use 50%
    as a neutral default). Heavier vehicles need stiffer springs and
    higher-rated dampers, which increase both the mass and cost of the
    components and reduce ride comfort.

    Inputs (F):
        design_mass        : vehicle design curb weight (kg).
        target_max_speed   : top speed the suspension must remain
            stable at (km/h). Sets damper rebound rate.

    Outputs (R):
        weight     : kg
        cost       : USD
        comfort    : 0-1 dimensionless comfort score
        durability : km until first major service
    """
    F = {
        "design_mass":      Reals(unit="kg"),
        "target_max_speed": Reals(unit="km/h"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "comfort":    Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self, type_name: str, *,
                 specific_mass_frac: float, base_cost_usd: float,
                 cost_per_kg_load: float, comfort_score: float,
                 durability_km: float, max_supportable_kg: float):
        self.type_name = type_name
        self.specific_mass_frac = specific_mass_frac
        self.base_cost_usd = base_cost_usd
        self.cost_per_kg_load = cost_per_kg_load
        self.comfort_score = comfort_score
        self.durability_km = durability_km
        self.max_supportable_kg = max_supportable_kg
        super().__init__()

    def h(self, f):
        m = f["design_mass"]
        if m > self.max_supportable_kg:
            return {"weight": float("inf"), "cost": float("inf"),
                    "comfort": 0.0, "durability": 0.0}
        # Front axle share of design mass.
        front_load = 0.55 * m
        # Suspension mass scales linearly with the axle load it must
        # carry; specific_mass_frac is the mass fraction of the
        # suspension relative to the axle load it supports.
        weight = self.specific_mass_frac * front_load
        cost   = self.base_cost_usd + self.cost_per_kg_load * front_load
        return {"weight": weight, "cost": cost,
                "comfort": self.comfort_score,
                "durability": self.durability_km}


class RearSuspension(Module):
    """Rear-axle suspension. Same structure as FrontSuspension but
    sized for the rear axle's 45% share of design mass."""
    F = {
        "design_mass":      Reals(unit="kg"),
        "target_max_speed": Reals(unit="km/h"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "comfort":    Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self, type_name: str, *,
                 specific_mass_frac: float, base_cost_usd: float,
                 cost_per_kg_load: float, comfort_score: float,
                 durability_km: float, max_supportable_kg: float):
        self.type_name = type_name
        self.specific_mass_frac = specific_mass_frac
        self.base_cost_usd = base_cost_usd
        self.cost_per_kg_load = cost_per_kg_load
        self.comfort_score = comfort_score
        self.durability_km = durability_km
        self.max_supportable_kg = max_supportable_kg
        super().__init__()

    def h(self, f):
        m = f["design_mass"]
        if m > self.max_supportable_kg:
            return {"weight": float("inf"), "cost": float("inf"),
                    "comfort": 0.0, "durability": 0.0}
        rear_load = 0.45 * m
        weight = self.specific_mass_frac * rear_load
        cost   = self.base_cost_usd + self.cost_per_kg_load * rear_load
        return {"weight": weight, "cost": cost,
                "comfort": self.comfort_score,
                "durability": self.durability_km}


class BrakesFront(Module):
    """Front brake assembly: rotors, calipers, pads (per axle, both wheels).

    Sized to the kinetic energy that must be dissipated in a
    full-stop from max speed. The front brakes do roughly 70% of the
    braking work (longitudinal load transfer under deceleration).
    Larger and heavier vehicles need larger rotors and harder pads,
    increasing both mass and cost; the durability falls with how
    aggressively the brakes are loaded.

    Inputs (F):
        design_mass      : kg (front axle does 70% of braking work)
        target_max_speed : km/h
        target_decel_g   : target longitudinal deceleration capacity (g)

    Outputs (R):
        weight     : kg
        cost       : USD
        durability : km
    """
    F = {
        "design_mass":      Reals(unit="kg"),
        "target_max_speed": Reals(unit="km/h"),
        "target_decel_g":   Reals(unit="g"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, base_weight_kg: float = 25.0,
                 weight_per_kw_brake: float = 0.012,
                 base_cost_usd: float = 280.0,
                 cost_per_kw_brake: float = 0.18,
                 base_durability_km: float = 60000.0):
        self.base_weight_kg = base_weight_kg
        self.weight_per_kw_brake = weight_per_kw_brake
        self.base_cost_usd = base_cost_usd
        self.cost_per_kw_brake = cost_per_kw_brake
        self.base_durability_km = base_durability_km
        super().__init__()

    def h(self, f):
        m = f["design_mass"]
        v_max = f["target_max_speed"] * KMH_TO_MS
        decel = f["target_decel_g"] * GRAVITY
        # Peak instantaneous brake power: F * v at v_max, where
        # F = m * decel (Newton's 2nd law). Front does 70%.
        peak_power_kw = 0.70 * m * decel * v_max / 1000.0
        # Sustained brake power for a 100-to-0 stop (front share)
        # determines rotor mass; we size to peak instantaneous power.
        weight = self.base_weight_kg + self.weight_per_kw_brake * peak_power_kw
        cost   = self.base_cost_usd + self.cost_per_kw_brake   * peak_power_kw
        # Durability falls with deceleration capacity squared (pads
        # work much harder at aggressive decel levels).
        durability = self.base_durability_km * (0.8 / max(f["target_decel_g"], 0.5)) ** 1.5
        return {"weight": weight, "cost": cost, "durability": durability}


class BrakesRear(Module):
    """Rear brake assembly. Smaller than front; does 30% of brake work."""
    F = {
        "design_mass":      Reals(unit="kg"),
        "target_max_speed": Reals(unit="km/h"),
        "target_decel_g":   Reals(unit="g"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self):
        self.base_weight_kg = 18.0
        self.weight_per_kw_brake = 0.010
        self.base_cost_usd = 220.0
        self.cost_per_kw_brake = 0.14
        self.base_durability_km = 80000.0
        super().__init__()

    def h(self, f):
        m = f["design_mass"]
        v_max = f["target_max_speed"] * KMH_TO_MS
        decel = f["target_decel_g"] * GRAVITY
        peak_power_kw = 0.30 * m * decel * v_max / 1000.0
        weight = self.base_weight_kg + self.weight_per_kw_brake * peak_power_kw
        cost   = self.base_cost_usd + self.cost_per_kw_brake   * peak_power_kw
        durability = self.base_durability_km * (0.8 / max(f["target_decel_g"], 0.5)) ** 1.5
        return {"weight": weight, "cost": cost, "durability": durability}


class SteeringSystem(Module):
    """Steering rack plus power assistance.

    Modern vehicles use electric power steering (EPS) almost
    exclusively; hydraulic systems remain only in heavy trucks. We
    model both as catalog entries with different weight, cost, and
    accessory-load profiles.

    Inputs (F):
        design_mass : kg (heavier cars need stronger steering torque).

    Outputs (R):
        weight        : kg
        cost          : USD
        electric_load : W (continuous accessory draw)
        durability    : km
    """
    F = {"design_mass": Reals(unit="kg")}
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "electric_load": Reals(unit="W"),
        "durability":    Reals(unit="km"),
    }

    def __init__(self, name: str, *,
                 base_weight_kg: float, weight_scale: float,
                 base_cost_usd: float, cost_scale: float,
                 electric_load_W: float, durability_km: float):
        self.base_weight_kg = base_weight_kg
        self.weight_scale = weight_scale
        self.base_cost_usd = base_cost_usd
        self.cost_scale = cost_scale
        self.electric_load_W = electric_load_W
        self.durability_km = durability_km
        super().__init__()
        self.name = name

    def h(self, f):
        m = f["design_mass"]
        weight = self.base_weight_kg + self.weight_scale * (m / 1500.0)
        cost   = self.base_cost_usd  + self.cost_scale   * (m / 1500.0)
        return {"weight": weight, "cost": cost,
                "electric_load": self.electric_load_W,
                "durability": self.durability_km}


class Tires(Module):
    """Four tires (a set), characterised by their performance category.

    Inputs (F):
        design_mass      : kg (sets load index requirement).
        target_max_speed : km/h (sets speed rating).

    Outputs (R):
        weight     : kg (set of 4)
        cost       : USD (set of 4)
        crr        : rolling-resistance coefficient (dimensionless)
        grip       : peak longitudinal/lateral friction coefficient
        durability : km (rated tread life)
        noise_dB   : pass-by noise (informational)
    """
    F = {
        "design_mass":      Reals(unit="kg"),
        "target_max_speed": Reals(unit="km/h"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "crr":        Reals(unit=""),
        "grip":       Reals(unit=""),
        "durability": Reals(unit="km"),
        "noise_dB":   Reals(unit="dB"),
    }

    def __init__(self, name: str, *,
                 mass_per_tire_kg: float, cost_per_tire_usd: float,
                 crr: float, grip: float, durability_km: float,
                 noise_dB: float,
                 max_load_kg: float, max_speed_kmh: float):
        self.mass_per_tire_kg = mass_per_tire_kg
        self.cost_per_tire_usd = cost_per_tire_usd
        self.crr_val = crr
        self.grip_val = grip
        self.durability_km = durability_km
        self.noise_dB_val = noise_dB
        self.max_load_kg = max_load_kg
        self.max_speed_kmh = max_speed_kmh
        super().__init__()
        self.name = name

    def h(self, f):
        per_tire_load = f["design_mass"] / 4.0
        if per_tire_load > self.max_load_kg:
            return {"weight": float("inf"), "cost": float("inf"),
                    "crr": self.crr_val, "grip": self.grip_val,
                    "durability": 0.0, "noise_dB": self.noise_dB_val}
        if f["target_max_speed"] > self.max_speed_kmh:
            return {"weight": float("inf"), "cost": float("inf"),
                    "crr": self.crr_val, "grip": self.grip_val,
                    "durability": 0.0, "noise_dB": self.noise_dB_val}
        return {
            "weight":     4.0 * self.mass_per_tire_kg,
            "cost":       4.0 * self.cost_per_tire_usd,
            "crr":        self.crr_val,
            "grip":       self.grip_val,
            "durability": self.durability_km,
            "noise_dB":   self.noise_dB_val,
        }


class Wheels(Module):
    """Four wheels (rims), characterised by material and diameter class.

    Inputs (F):
        design_mass : kg

    Outputs (R):
        weight      : kg (set of 4)
        cost        : USD (set of 4)
        unsprung_kg : kg of unsprung mass added per wheel (ride/handling)
    """
    F = {"design_mass": Reals(unit="kg")}
    R = {
        "weight":      Reals(unit="kg"),
        "cost":        Reals(unit="USD"),
        "unsprung_kg": Reals(unit="kg"),
    }

    def __init__(self, name: str, *,
                 mass_per_wheel_kg: float, cost_per_wheel_usd: float):
        self.mass_per_wheel_kg = mass_per_wheel_kg
        self.cost_per_wheel_usd = cost_per_wheel_usd
        super().__init__()
        self.name = name

    def h(self, f):
        return {
            "weight":      4.0 * self.mass_per_wheel_kg,
            "cost":        4.0 * self.cost_per_wheel_usd,
            "unsprung_kg": self.mass_per_wheel_kg,
        }


# ===========================================================================
# Auxiliary modules (shared by all architectures)
# ===========================================================================


class HVAC(Module):
    """Heating, ventilation, air conditioning.

    Modern HVAC systems use an electric or belt-driven compressor,
    cabin blower, evaporator core, heater core, and climate control
    ECU. Sizing is driven by cabin volume (passenger count proxy)
    and the climate envelope the car must operate in.

    Inputs (F):
        target_passengers : people (sets cabin volume).

    Outputs (R):
        weight        : kg
        cost          : USD
        electric_load : W (peak; AC compressor dominant)
        durability    : km
    """
    F = {"target_passengers": Reals(unit="people")}
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "electric_load": Reals(unit="W"),
        "durability":    Reals(unit="km"),
    }

    def __init__(self, *, electric_compressor: bool = False):
        self.electric_compressor = electric_compressor
        super().__init__()

    def h(self, f):
        n = max(2.0, f["target_passengers"])
        # Weight scales with cabin volume; AC compressor adds ~6 kg.
        base = 22.0 + 1.8 * (n - 2.0)
        cost = 580.0 + 90.0 * (n - 2.0)
        if self.electric_compressor:
            base += 4.0   # heavier compressor and motor
            cost += 220.0
            elec = 2800.0  # peak electric AC compressor draw
        else:
            elec = 350.0   # blower + electronics; belt-driven AC adds
                          # mechanical accessory load (handled in engine
                          # accessory budget, not here).
        return {"weight": base, "cost": cost,
                "electric_load": elec, "durability": 160000.0}


class InteriorTrim(Module):
    """Seats, dashboard, carpet, headliner, sound insulation, IP.

    Mass scales primarily with passenger count and trim grade. We use
    a single "trim_level" parameter on a 0-1 scale where 0 is economy
    cloth and 1 is full leather + power adjustments + heating + memory.

    Inputs (F):
        target_passengers : people

    Outputs (R):
        weight     : kg
        cost       : USD
        durability : km
    """
    F = {"target_passengers": Reals(unit="people")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, trim_level: float = 0.3):
        self.trim_level = max(0.0, min(1.0, trim_level))
        super().__init__()

    def h(self, f):
        n = max(2.0, f["target_passengers"])
        # Empty-vehicle interior; passenger mass is in payload, not curb.
        # Per-seat structure (seat frame + cushion + adjusters) is 10 to
        # 20 kg depending on trim grade; full leather + power seats adds
        # another 8 kg per seat. Dashboard, carpet, headliner, IP, sound
        # insulation account for a further fixed allowance.
        base_per_seat = 12.0 + 9.0 * self.trim_level
        weight = n * base_per_seat + 35.0  # 35 kg dashboard + carpet + trim
        cost   = n * (180.0 + 850.0 * self.trim_level) + 600.0
        durability = 250000.0 - 60000.0 * self.trim_level  # power seats fail sooner
        return {"weight": weight, "cost": cost, "durability": durability}


class SafetySystems(Module):
    """Airbags, seat-belt pretensioners, ABS module, ESC, sensors.

    Mandatory under EU GSR2 and similar regimes; baseline configuration
    is non-optional. Mass scales linearly with passenger count (front
    + curtain + side bags for each seating row).

    Inputs (F):
        target_passengers : people
        target_max_speed  : km/h (higher max speed -> more sensors)

    Outputs (R):
        weight        : kg
        cost          : USD
        electric_load : W
        durability    : km
    """
    F = {
        "target_passengers": Reals(unit="people"),
        "target_max_speed":  Reals(unit="km/h"),
    }
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "electric_load": Reals(unit="W"),
        "durability":    Reals(unit="km"),
    }

    def __init__(self, *, adas_level: int = 2):
        # ADAS level: 0 = AEB + ABS only, 1 = + lane keep + ACC,
        # 2 = + traffic-jam assist + auto park, 3 = highway pilot.
        self.adas_level = int(adas_level)
        super().__init__()

    def h(self, f):
        n = max(2.0, f["target_passengers"])
        base = 18.0 + 3.5 * (n - 2.0)
        cost = 450.0 + 120.0 * (n - 2.0)
        # Higher ADAS levels add cameras, radars, lidars, compute.
        adas_mass = [0.0, 4.0, 9.0, 16.0][min(self.adas_level, 3)]
        adas_cost = [0.0, 600.0, 1800.0, 4500.0][min(self.adas_level, 3)]
        base += adas_mass
        cost += adas_cost
        elec = 80.0 + 30.0 * self.adas_level
        return {"weight": base, "cost": cost,
                "electric_load": elec, "durability": 400000.0}


class LightingAndInfotainment(Module):
    """Headlights, tail-lights, interior lighting, infotainment head unit.

    Modest contribution to weight and cost but a meaningful continuous
    electrical load.

    Inputs (F):
        design_mass : kg (a weak scaling input; larger vehicles get
                     marginally larger infotainment systems).

    Outputs (R):
        weight        : kg
        cost          : USD
        electric_load : W
        durability    : km
    """
    F = {"design_mass": Reals(unit="kg")}
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "electric_load": Reals(unit="W"),
        "durability":    Reals(unit="km"),
    }

    def __init__(self, *, led_lighting: bool = True,
                 infotainment_grade: float = 0.5):
        self.led_lighting = led_lighting
        self.infotainment_grade = max(0.0, min(1.0, infotainment_grade))
        super().__init__()

    def h(self, f):
        # Base weight/cost plus a tiny mass-proportional scaling.
        size_factor = 1.0 + 0.0001 * max(f["design_mass"] - 1200.0, 0.0)
        weight = (18.0 + 12.0 * self.infotainment_grade) * size_factor
        cost   = (380.0 + 1200.0 * self.infotainment_grade) * size_factor
        elec   = (140.0 if self.led_lighting else 250.0) \
                 + 120.0 * self.infotainment_grade
        if not self.led_lighting:
            cost  -= 150.0   # halogens are cheaper
            weight += 2.0    # but slightly heavier per lumen
        return {"weight": weight, "cost": cost,
                "electric_load": elec, "durability": 200000.0}


# ===========================================================================
# ICE-specific powertrain modules
# ===========================================================================


class EngineBlock(Module):
    """Internal-combustion engine long block.

    Represents the cylinder block, head, crankshaft, pistons, valves,
    valvetrain, and ECU as a single subsystem. Each "engine" is a
    discrete catalogue entry parameterised by its peak power, peak
    torque, peak efficiency, cycle type, displacement, weight, and
    cost. The peak power and torque set the upper bound on what the
    engine can deliver; the peak efficiency determines fuel
    consumption at the best operating point; the peak heat rejection
    rate sizes the cooling system.

    Inputs (F):
        target_peak_power_kW  : minimum rated peak power needed.
        target_peak_torque_Nm : minimum rated peak torque needed.

    Outputs (R):
        weight                 : kg
        cost                   : USD
        displacement_L         : L
        peak_efficiency        : fraction (0-1)
        heat_rejection_at_peak : kW thermal load at rated power
        oil_capacity_L         : L of engine oil
        durability             : km
        accessory_power_load   : kW continuous mechanical draw
                                 for water pump, oil pump, alternator
        fuel_type              : 0.0 = gasoline, 1.0 = diesel (a
                                 numeric tag so downstream modules
                                 can route correctly)
    """
    F = {
        "target_peak_power_kW":  Reals(unit="kW"),
        "target_peak_torque_Nm": Reals(unit="Nm"),
    }
    R = {
        "weight":                 Reals(unit="kg"),
        "cost":                   Reals(unit="USD"),
        "displacement_L":         Reals(unit="L"),
        "peak_efficiency":        Reals(unit=""),
        "heat_rejection_at_peak": Reals(unit="kW"),
        "oil_capacity_L":         Reals(unit="L"),
        "durability":             Reals(unit="km"),
        "accessory_power_load":   Reals(unit="kW"),
        "fuel_type":              Reals(unit=""),
        # Rated capacity exposed as R so downstream modules (fuel
        # injection, exhaust, cooling, transmission) can size to it.
        "rated_peak_power_kW":    Reals(unit="kW"),
        "rated_peak_torque_Nm":   Reals(unit="Nm"),
    }

    def __init__(self, name: str, *,
                 peak_power_kW: float, peak_torque_Nm: float,
                 displacement_L: float, peak_efficiency: float,
                 weight_kg: float, cost_usd: float,
                 oil_capacity_L: float, durability_km: float,
                 cycle: str = "otto", fuel: str = "gasoline",
                 accessory_load_kW: float = 1.5):
        self.peak_power_kW = peak_power_kW
        self.peak_torque_Nm = peak_torque_Nm
        self.displacement_L = displacement_L
        self.peak_efficiency = peak_efficiency
        self.weight_kg = weight_kg
        self.cost_usd = cost_usd
        self.oil_capacity_L = oil_capacity_L
        self.durability_km = durability_km
        self.cycle = cycle
        self.fuel = fuel
        self.accessory_load_kW = accessory_load_kW
        super().__init__()
        self.name = name

    def h(self, f):
        if f["target_peak_power_kW"]  > self.peak_power_kW or \
           f["target_peak_torque_Nm"] > self.peak_torque_Nm:
            return {"weight": float("inf"), "cost": float("inf"),
                    "displacement_L": self.displacement_L,
                    "peak_efficiency": self.peak_efficiency,
                    "heat_rejection_at_peak": float("inf"),
                    "oil_capacity_L": self.oil_capacity_L,
                    "durability": 0.0,
                    "accessory_power_load": self.accessory_load_kW,
                    "fuel_type": 1.0 if self.fuel == "diesel" else 0.0,
                    "rated_peak_power_kW":  self.peak_power_kW,
                    "rated_peak_torque_Nm": self.peak_torque_Nm}
        # At peak power, the engine rejects (1/eta - 1) * P_peak as
        # heat to the coolant + exhaust + radiation. ~55% goes to
        # coolant, the rest to exhaust and radiation. Cooling system
        # is sized to the coolant share.
        heat_rejection_total = self.peak_power_kW * (1.0 / self.peak_efficiency - 1.0)
        cooling_share = 0.55
        return {
            "weight":                 self.weight_kg,
            "cost":                   self.cost_usd,
            "displacement_L":         self.displacement_L,
            "peak_efficiency":        self.peak_efficiency,
            "heat_rejection_at_peak": cooling_share * heat_rejection_total,
            "oil_capacity_L":         self.oil_capacity_L,
            "durability":             self.durability_km,
            "accessory_power_load":   self.accessory_load_kW,
            "fuel_type":              1.0 if self.fuel == "diesel" else 0.0,
            "rated_peak_power_kW":    self.peak_power_kW,
            "rated_peak_torque_Nm":   self.peak_torque_Nm,
        }


class ForcedInduction(Module):
    """Turbocharger or twin-turbo, plus charge-air intercooler.

    A turbo recovers exhaust enthalpy to boost intake pressure,
    raising volumetric efficiency and peak power without increasing
    displacement. Modelled as an optional add-on to the engine
    block: the "none" variant adds zero mass and cost and applies
    no power multiplier.

    Inputs (F):
        engine_peak_power_kW : the boost-target rated power of the
                               combined engine + turbo.

    Outputs (R):
        weight        : kg
        cost          : USD
        durability    : km
        intake_load_W : minor electric load (electric wastegate)
    """
    F = {"engine_peak_power_kW": Reals(unit="kW")}
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "durability":    Reals(unit="km"),
        "intake_load_W": Reals(unit="W"),
    }

    def __init__(self, kind: str = "none"):
        # 'none', 'single_turbo', 'twin_turbo'
        self.kind = kind
        super().__init__()

    def h(self, f):
        p = f["engine_peak_power_kW"]
        if self.kind == "none":
            return {"weight": 0.0, "cost": 0.0, "durability": 1.0e9,
                    "intake_load_W": 0.0}
        if self.kind == "single_turbo":
            return {"weight": 12.0 + 0.04 * p,
                    "cost":   450.0 + 4.5 * p,
                    "durability": 180000.0,
                    "intake_load_W": 25.0}
        if self.kind == "twin_turbo":
            return {"weight": 22.0 + 0.06 * p,
                    "cost":   1100.0 + 6.5 * p,
                    "durability": 150000.0,
                    "intake_load_W": 50.0}
        raise ValueError(f"unknown forced induction kind: {self.kind}")


class FuelInjection(Module):
    """Fuel rail, injectors, high-pressure pump.

    Sized to the peak fuel flow rate the engine will demand at rated
    power. Direct-injection systems are heavier and more expensive
    than port injection but enable higher compression ratios and
    higher efficiency.

    Inputs (F):
        engine_peak_power_kW : peak fuel demand setter
        engine_peak_efficiency : sets fuel mass flow at peak power

    Outputs (R):
        weight : kg
        cost   : USD
        durability : km
    """
    F = {
        "engine_peak_power_kW":   Reals(unit="kW"),
        "engine_peak_efficiency": Reals(unit=""),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, system: str = "direct"):
        # 'port' or 'direct'
        self.system = system
        super().__init__()

    def h(self, f):
        p = f["engine_peak_power_kW"]
        if self.system == "port":
            return {"weight": 5.0 + 0.03 * p, "cost": 180.0 + 2.2 * p,
                    "durability": 250000.0}
        return {"weight": 8.0 + 0.04 * p, "cost": 320.0 + 4.5 * p,
                "durability": 200000.0}


class ExhaustAftertreatment(Module):
    """Exhaust manifold, catalytic converter, particulate filter, muffler.

    Diesel systems carry the heaviest after-treatment burden: diesel
    particulate filter (DPF), selective catalytic reduction (SCR)
    with AdBlue injection, and NOx storage catalyst. Gasoline
    Otto-cycle engines need only a 3-way catalyst; gasoline DI engines
    also add a gasoline particulate filter (GPF) under Euro 7.

    Inputs (F):
        engine_peak_power_kW : sizes the catalyst light-off mass
        fuel_type            : 0 (gasoline) or 1 (diesel)

    Outputs (R):
        weight : kg
        cost   : USD
        durability : km
    """
    F = {
        "engine_peak_power_kW": Reals(unit="kW"),
        "fuel_type":            Reals(unit=""),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, includes_gpf: bool = True):
        # Under Euro 7 / EPA tier-4, GPF is essentially mandatory
        # for direct-injection gasoline. Older calibrations omit it.
        self.includes_gpf = includes_gpf
        super().__init__()

    def h(self, f):
        p = f["engine_peak_power_kW"]
        is_diesel = f["fuel_type"] > 0.5
        if is_diesel:
            # DPF + SCR + AdBlue tank + control unit
            weight = 28.0 + 0.18 * p
            cost   = 1800.0 + 5.0 * p
            durability = 240000.0
        else:
            # 3-way cat + muffler; optional GPF
            weight = 14.0 + 0.10 * p
            cost   = 520.0 + 3.0 * p
            if self.includes_gpf:
                weight += 6.0
                cost   += 380.0
            durability = 220000.0
        return {"weight": weight, "cost": cost, "durability": durability}


class CoolingSystem(Module):
    """Engine cooling system: radiator, water pump, thermostat, fan,
    expansion tank, hoses.

    Sized to the engine's peak heat-rejection rate (the coolant share
    of total waste heat). A radiator's heat-dissipation capacity per
    unit mass is roughly 4-6 kW/kg; we use 5 kW/kg as a calibrated
    average for cross-flow aluminium radiators.

    Inputs (F):
        heat_to_dissipate_kW : design heat-rejection rate (kW)

    Outputs (R):
        weight         : kg
        cost           : USD
        fan_power_kW   : peak electric fan draw
        coolant_mass_kg: kg of coolant (counts toward curb weight)
        durability     : km
    """
    F = {"heat_to_dissipate_kW": Reals(unit="kW")}
    R = {
        "weight":          Reals(unit="kg"),
        "cost":            Reals(unit="USD"),
        "fan_power_kW":    Reals(unit="kW"),
        "coolant_mass_kg": Reals(unit="kg"),
        "durability":      Reals(unit="km"),
    }

    def __init__(self):
        self.radiator_capacity_kW_per_kg = 5.0
        super().__init__()

    def h(self, f):
        q = max(f["heat_to_dissipate_kW"], 1.0)
        radiator_mass = q / self.radiator_capacity_kW_per_kg
        # Water pump + thermostat + hoses + expansion tank add ~30%
        # of radiator mass; coolant mass is roughly 4-5 L of glycol
        # mix for small engines, 8-10 L for large.
        accessories_mass = 0.3 * radiator_mass
        coolant_L = 3.0 + 0.05 * q
        coolant_mass = coolant_L * 1.06   # glycol/water mix density
        weight = radiator_mass + accessories_mass + coolant_mass
        cost   = 60.0 * radiator_mass + 180.0
        fan_power = 0.40 + 0.012 * q     # kW, electric fan peak draw
        return {"weight": weight, "cost": cost,
                "fan_power_kW": fan_power,
                "coolant_mass_kg": coolant_mass,
                "durability": 200000.0}


class LubricationSystem(Module):
    """Oil pump, oil filter, oil cooler, sump.

    Scales with engine displacement. The oil itself contributes a
    fixed mass that the engine block module already reports as
    oil_capacity_L; here we model only the pump, filter, cooler,
    and the sump structure.

    Inputs (F):
        engine_displacement_L : sets pump and sump size
        engine_oil_capacity_L : oil mass (1 L ~ 0.87 kg motor oil)

    Outputs (R):
        weight              : kg (pump + filter + cooler + oil mass)
        cost                : USD
        oil_change_interval : km (depends on filter and synthetic oil)
        durability          : km
    """
    F = {
        "engine_displacement_L": Reals(unit="L"),
        "engine_oil_capacity_L": Reals(unit="L"),
    }
    R = {
        "weight":              Reals(unit="kg"),
        "cost":                Reals(unit="USD"),
        "oil_change_interval": Reals(unit="km"),
        "durability":          Reals(unit="km"),
    }

    def __init__(self, *, synthetic_oil: bool = True,
                 oil_cooler: bool = False):
        self.synthetic_oil = synthetic_oil
        self.oil_cooler = oil_cooler
        super().__init__()

    def h(self, f):
        disp = f["engine_displacement_L"]
        oil_L = f["engine_oil_capacity_L"]
        # Pump + filter + sump weight scales with displacement.
        pump_sump = 4.0 + 2.5 * disp
        oil_mass = 0.87 * oil_L  # SAE 0W-30 ~ 0.87 kg/L
        weight = pump_sump + oil_mass
        cost = 80.0 + 35.0 * disp
        if self.oil_cooler:
            weight += 1.5
            cost   += 140.0
        interval = 15000.0 if self.synthetic_oil else 8000.0
        return {"weight": weight, "cost": cost,
                "oil_change_interval": interval,
                "durability": 300000.0}


class Transmission(Module):
    """Multi-speed gearbox plus clutch / torque converter.

    Catalogue entries cover 6MT, 6AT, 8AT, CVT, 6DCT. For a
    single-speed reducer (used in EVs), see SingleSpeedReducer.

    Inputs (F):
        input_peak_power_kW  : the engine's peak power it must transmit
        input_peak_torque_Nm : the engine's peak torque it must transmit

    Outputs (R):
        weight     : kg
        cost       : USD
        efficiency : fraction of input power that reaches the
                     differential (0.85 - 0.97 depending on type)
        durability : km
    """
    F = {
        "input_peak_power_kW":  Reals(unit="kW"),
        "input_peak_torque_Nm": Reals(unit="Nm"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "efficiency": Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self, name: str, *,
                 base_weight_kg: float, weight_per_kw: float,
                 base_cost_usd: float, cost_per_kw: float,
                 efficiency: float, durability_km: float,
                 max_torque_Nm: float):
        self.base_weight_kg = base_weight_kg
        self.weight_per_kw = weight_per_kw
        self.base_cost_usd = base_cost_usd
        self.cost_per_kw = cost_per_kw
        self.efficiency_val = efficiency
        self.durability_km = durability_km
        self.max_torque_Nm = max_torque_Nm
        super().__init__()
        self.name = name

    def h(self, f):
        if f["input_peak_torque_Nm"] > self.max_torque_Nm:
            return {"weight": float("inf"), "cost": float("inf"),
                    "efficiency": self.efficiency_val,
                    "durability": 0.0}
        p = f["input_peak_power_kW"]
        weight = self.base_weight_kg + self.weight_per_kw * p
        cost   = self.base_cost_usd  + self.cost_per_kw   * p
        return {"weight": weight, "cost": cost,
                "efficiency": self.efficiency_val,
                "durability": self.durability_km}


class DriveshaftDifferential(Module):
    """Driveshaft, propshaft, differential housing and gears.

    For FWD this is the transaxle final drive (small); for RWD or
    AWD it includes a longitudinal driveshaft and a rear differential
    (larger). Sized to peak transmitted torque.

    Inputs (F):
        peak_torque_Nm : Nm at the differential input

    Outputs (R):
        weight     : kg
        cost       : USD
        durability : km
    """
    F = {"peak_torque_Nm": Reals(unit="Nm")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, layout: str = "fwd"):
        self.layout = layout
        super().__init__()

    def h(self, f):
        t = f["peak_torque_Nm"]
        if self.layout == "fwd":
            return {"weight": 18.0 + 0.04 * t,
                    "cost": 320.0 + 0.6 * t,
                    "durability": 300000.0}
        if self.layout == "rwd":
            return {"weight": 38.0 + 0.06 * t,
                    "cost": 580.0 + 0.9 * t,
                    "durability": 320000.0}
        if self.layout == "awd":
            return {"weight": 65.0 + 0.08 * t,
                    "cost": 1400.0 + 1.5 * t,
                    "durability": 280000.0}
        raise ValueError(f"unknown layout: {self.layout}")


class FuelTank(Module):
    """Fuel tank and fuel-system plumbing (lines, pump, evap canister).

    Sized to deliver the target range at the expected fuel consumption
    rate. The tank capacity in litres equals
    range_km / 100 * fuel_consumption_L_per_100km. The fuel mass
    (when full) contributes to curb weight at half capacity per
    ECE R83 (the regulated "operational" mass).

    Inputs (F):
        target_range_km                : km
        fuel_consumption_L_per_100km   : L/100km (closes the cycle)

    Outputs (R):
        weight    : kg (empty tank + half-full fuel mass + pump)
        cost      : USD
        capacity_L : L
        durability: km
    """
    F = {
        "target_range_km":              Reals(unit="km"),
        "fuel_consumption_L_per_100km": Reals(unit="L/100km"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "capacity_L": Reals(unit="L"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, fuel_density_kg_per_L: float = GASOLINE_DENSITY):
        self.fuel_density_kg_per_L = fuel_density_kg_per_L
        super().__init__()

    def h(self, f):
        consumption = max(f["fuel_consumption_L_per_100km"], 1.0)
        capacity_L = f["target_range_km"] * consumption / 100.0 + 5.0  # reserve
        # Empty tank + plumbing: ~0.6 kg per L of capacity for
        # high-density polyethylene with metal lines, baffles, evap.
        tank_empty = 0.55 * capacity_L + 4.0
        half_fuel_mass = 0.5 * capacity_L * self.fuel_density_kg_per_L
        weight = tank_empty + half_fuel_mass
        cost = 110.0 + 4.5 * capacity_L
        return {"weight": weight, "cost": cost,
                "capacity_L": capacity_L,
                "durability": 250000.0}


class Alternator(Module):
    """Belt-driven 12V alternator (ICE only).

    Sized to meet the maximum simultaneous 12V accessory load with
    margin. Modern smart alternators include start-stop reverse
    capability but remain belt-driven.

    Inputs (F):
        peak_electric_load_W : peak 12V accessory draw

    Outputs (R):
        weight     : kg
        cost       : USD
        parasitic_load_kW : continuous mechanical draw on engine
        durability : km
    """
    F = {"peak_electric_load_W": Reals(unit="W")}
    R = {
        "weight":            Reals(unit="kg"),
        "cost":              Reals(unit="USD"),
        "parasitic_load_kW": Reals(unit="kW"),
        "durability":        Reals(unit="km"),
    }

    def __init__(self):
        super().__init__()

    def h(self, f):
        p_W = f["peak_electric_load_W"]
        weight = 4.5 + 0.0035 * p_W
        cost   = 140.0 + 0.05 * p_W
        # Belt + alternator drag: alternator is ~60% efficient
        # under load, so parasitic mech load = p_W / 0.6 / 1000.
        # Average accessory load is roughly half of peak.
        avg_load_kW = 0.5 * p_W / 1000.0
        parasitic = avg_load_kW / 0.60
        return {"weight": weight, "cost": cost,
                "parasitic_load_kW": parasitic,
                "durability": 200000.0}


class Battery12V(Module):
    """12V lead-acid (or AGM) battery for starting and accessories.

    Modern start-stop systems require AGM technology. Battery is
    sized for peak crank current plus accessory reserve.

    Inputs (F):
        peak_electric_load_W : sets the AGM capacity requirement

    Outputs (R):
        weight     : kg
        cost       : USD
        durability : km
    """
    F = {"peak_electric_load_W": Reals(unit="W")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, agm: bool = True):
        self.agm = agm
        super().__init__()

    def h(self, f):
        # Capacity scales with peak load; baseline is 60 Ah AGM at
        # ~18 kg. For 1 kW peak load we want 80 Ah at 22 kg.
        cap_Ah = 45.0 + 0.03 * f["peak_electric_load_W"]
        weight = 0.30 * cap_Ah  # AGM specific weight
        cost = 130.0 + 1.8 * cap_Ah if self.agm else 80.0 + 1.1 * cap_Ah
        durability = 100000.0 if self.agm else 60000.0
        return {"weight": weight, "cost": cost, "durability": durability}


class StarterMotor(Module):
    """Engine starter motor (ICE and hybrid only).

    A starter is sized to the engine's displacement and compression
    ratio (cranking torque requirement). Hybrid integrated
    starter-generators (ISG) replace the conventional starter; see
    HybridMotorGenerator.

    Inputs (F):
        engine_displacement_L : sets cranking torque demand

    Outputs (R):
        weight     : kg
        cost       : USD
        durability : km
    """
    F = {"engine_displacement_L": Reals(unit="L")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "durability": Reals(unit="km"),
    }

    def __init__(self):
        super().__init__()

    def h(self, f):
        d = f["engine_displacement_L"]
        weight = 3.5 + 0.7 * d
        cost = 90.0 + 25.0 * d
        return {"weight": weight, "cost": cost, "durability": 180000.0}


# ===========================================================================
# Hybrid-specific modules (power-split parallel-hybrid layout)
# ===========================================================================


class HybridPowerSplit(Module):
    """Planetary power-split gearset replacing the conventional gearbox.

    A Toyota Hybrid Synergy Drive-style topology: a planetary gearset
    couples the engine, a motor-generator (MG1, for starting and
    charging), and a traction motor (MG2, for drive assist and
    regenerative braking). There is no clutch and no torque
    converter. Mechanically simpler than a gearbox but
    electronically and software-complex.

    Inputs (F):
        engine_peak_power_kW : sizes the planetary gears
        motor_peak_power_kW  : sizes MG1 and MG2 carrier

    Outputs (R):
        weight     : kg (gearset + housing)
        cost       : USD
        efficiency : effective drive-train efficiency
        durability : km
    """
    F = {
        "engine_peak_power_kW": Reals(unit="kW"),
        "motor_peak_power_kW":  Reals(unit="kW"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "efficiency": Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self):
        super().__init__()

    def h(self, f):
        # Planetary set sized to combined power throughput.
        p = f["engine_peak_power_kW"] + f["motor_peak_power_kW"]
        weight = 28.0 + 0.18 * p
        cost   = 1200.0 + 12.0 * p
        return {"weight": weight, "cost": cost,
                "efficiency": 0.94,    # very efficient at cruise
                "durability": 350000.0}


class HybridMotorGenerator(Module):
    """Combined traction motor / generator for power-split hybrid.

    In a power-split hybrid, MG2 (traction motor) does drive assist
    and regenerative braking; MG1 (generator) handles starting and
    charging. We model them together because their sizing is
    coupled by the planetary-gearset ratio. The motor's peak power
    is roughly 30-50% of the engine's peak in a "strong" hybrid.

    Inputs (F):
        motor_peak_power_kW : sets the rotor and stator sizing
        battery_voltage_V   : sets the winding configuration

    Outputs (R):
        weight     : kg
        cost       : USD
        peak_efficiency : motor efficiency at peak (0.92-0.96)
        peak_torque_Nm  : Nm available at the motor shaft
        durability : km
    """
    F = {
        "motor_peak_power_kW": Reals(unit="kW"),
        "battery_voltage_V":   Reals(unit="V"),
    }
    R = {
        "weight":          Reals(unit="kg"),
        "cost":            Reals(unit="USD"),
        "peak_efficiency": Reals(unit=""),
        "peak_torque_Nm":  Reals(unit="Nm"),
        "durability":      Reals(unit="km"),
    }

    def __init__(self):
        # Permanent-magnet synchronous motor, ~1.5 kW/kg specific power
        self.specific_power_kW_per_kg = 1.6
        super().__init__()

    def h(self, f):
        p = f["motor_peak_power_kW"]
        weight = p / self.specific_power_kW_per_kg + 3.0
        cost = 220.0 + 26.0 * p   # PM motor with rare-earth magnets
        # Peak torque ~ 1.8 Nm per kW for low-speed traction motors.
        torque = 1.8 * p
        return {"weight": weight, "cost": cost,
                "peak_efficiency": 0.95,
                "peak_torque_Nm": torque,
                "durability": 280000.0}


class HVBatteryHybrid(Module):
    """High-voltage traction battery for hybrid vehicles.

    Power-split hybrids use a small (1-2 kWh) NiMH or Li-ion pack
    sized for power output rather than energy capacity. The battery
    is cycled shallowly (10-80% SOC window) for very long calendar
    life. Modern Prius packs are NiMH; PHEVs and "strong" hybrids
    use Li-ion.

    Inputs (F):
        motor_peak_power_kW : sets the power-cell discharge rating

    Outputs (R):
        weight     : kg
        cost       : USD
        voltage_V  : nominal pack voltage
        energy_kWh : usable energy capacity
        durability : km (warrantied to first major degradation)
    """
    F = {"motor_peak_power_kW": Reals(unit="kW")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "voltage_V":  Reals(unit="V"),
        "energy_kWh": Reals(unit="kWh"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, chemistry: str = "lithium_NMC"):
        # 'NiMH' or 'lithium_NMC'
        self.chemistry = chemistry
        super().__init__()

    def h(self, f):
        p = f["motor_peak_power_kW"]
        # Hybrids size battery to ~50 Wh per kW of motor power.
        energy_kWh = 0.05 * p
        if self.chemistry == "NiMH":
            specific_energy_Wh_per_kg = 60.0   # NiMH pack level
            cost_per_kWh = 480.0
            voltage = 200.0
        else:
            specific_energy_Wh_per_kg = 130.0  # Li-NMC, hybrid use
            cost_per_kWh = 320.0
            voltage = 240.0
        weight = energy_kWh * 1000.0 / specific_energy_Wh_per_kg + 12.0
        cost   = energy_kWh * cost_per_kWh + 480.0
        return {"weight": weight, "cost": cost,
                "voltage_V": voltage,
                "energy_kWh": energy_kWh,
                "durability": 250000.0}


class HybridPowerElectronics(Module):
    """DC-AC inverter, DC-DC converter (HV-to-12V), control.

    Sized to the traction motor's peak power. Modern SiC-based
    inverters are smaller and lighter than older IGBT designs but
    cost more per kW.

    Inputs (F):
        motor_peak_power_kW : sets the inverter rating

    Outputs (R):
        weight     : kg
        cost       : USD
        efficiency : combined inverter + DC-DC efficiency
        durability : km
    """
    F = {"motor_peak_power_kW": Reals(unit="kW")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "efficiency": Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, sic: bool = False):
        self.sic = sic
        super().__init__()

    def h(self, f):
        p = f["motor_peak_power_kW"]
        # IGBT inverters: ~0.20 kg/kW, $11/kW.
        # SiC inverters: ~0.13 kg/kW, $20/kW.
        if self.sic:
            weight = 0.13 * p + 4.0
            cost = 20.0 * p + 180.0
            eta = 0.97
        else:
            weight = 0.20 * p + 4.0
            cost = 11.0 * p + 140.0
            eta = 0.95
        return {"weight": weight, "cost": cost,
                "efficiency": eta,
                "durability": 220000.0}


# ===========================================================================
# Electric-vehicle-specific modules
# ===========================================================================


class ElectricMotor(Module):
    """Traction motor for full-electric vehicles.

    Permanent-magnet synchronous motors (PMSM) are the dominant
    technology for EVs because of their high specific power and
    high peak efficiency (>96% at best operating point). Induction
    motors are lighter on rare-earth cost and have wider efficient
    operating regions but lower peak. Some manufacturers (Tesla,
    BMW) use induction at the front and PMSM at the rear in dual-
    motor configurations.

    Inputs (F):
        target_peak_power_kW : sets the rotor and stator sizing
        target_peak_torque_Nm: sets winding configuration
        battery_voltage_V    : sets motor winding voltage

    Outputs (R):
        weight              : kg
        cost                : USD
        peak_efficiency     : motor efficiency at the rated point
        peak_torque_Nm      : Nm available at the motor shaft
        rated_peak_power_kW : kW the motor can deliver at peak (echoes
                              the catalog rating so downstream modules
                              such as the reducer and power electronics
                              can size to a resource port rather than to
                              an F port)
        durability          : km
    """
    F = {
        "target_peak_power_kW":  Reals(unit="kW"),
        "target_peak_torque_Nm": Reals(unit="Nm"),
        "battery_voltage_V":     Reals(unit="V"),
    }
    R = {
        "weight":              Reals(unit="kg"),
        "cost":                Reals(unit="USD"),
        "peak_efficiency":     Reals(unit=""),
        "peak_torque_Nm":      Reals(unit="Nm"),
        "rated_peak_power_kW": Reals(unit="kW"),
        "durability":          Reals(unit="km"),
    }

    def __init__(self, name: str, *,
                 specific_power_kW_per_kg: float, cost_per_kW: float,
                 base_cost_usd: float, peak_efficiency: float,
                 torque_per_kW: float, durability_km: float,
                 motor_type: str = "PMSM"):
        self.specific_power_kW_per_kg = specific_power_kW_per_kg
        self.cost_per_kW = cost_per_kW
        self.base_cost_usd = base_cost_usd
        self.peak_efficiency = peak_efficiency
        self.torque_per_kW = torque_per_kW
        self.durability_km = durability_km
        self.motor_type = motor_type
        super().__init__()
        self.name = name

    def h(self, f):
        # The motor is a catalog entry: its weight, cost, and torque
        # follow from the requested peak power and the fixed
        # specific-power / torque-per-kW ratings. A request the motor
        # cannot satisfy on torque returns an infeasible (infinite-cost)
        # bundle so the solver eliminates it.
        p = f["target_peak_power_kW"]
        weight = p / self.specific_power_kW_per_kg + 5.0
        cost = self.base_cost_usd + self.cost_per_kW * p
        torque = self.torque_per_kW * p
        if torque < f["target_peak_torque_Nm"]:
            return {"weight": float("inf"), "cost": float("inf"),
                    "peak_efficiency": self.peak_efficiency,
                    "peak_torque_Nm": torque,
                    "rated_peak_power_kW": p,
                    "durability": 0.0}
        return {"weight": weight, "cost": cost,
                "peak_efficiency": self.peak_efficiency,
                "peak_torque_Nm": torque,
                "rated_peak_power_kW": p,
                "durability": self.durability_km}


class HVBatteryEV(Module):
    """High-voltage traction battery pack for full-electric vehicles.

    The dominant cost and weight item in an EV. Sized to deliver the
    target range at the expected energy consumption, with an SOC
    usable window of 80% (10-90% nominal) for long life. Cell-to-
    pack architectures (CATL, BYD blade, Tesla 4680) push pack-level
    energy density up to 150-200 Wh/kg in 2024-2025; we use 160 Wh/kg
    as a calibrated default.

    Inputs (F):
        target_range_km        : km
        energy_kWh_per_100km   : kWh/100km (cycle-closing input)

    Outputs (R):
        weight        : kg
        cost          : USD
        voltage_V     : nominal pack voltage (400V or 800V class)
        energy_kWh    : usable energy capacity (after SOC window)
        peak_power_kW : peak discharge power
        durability    : km warranty (typical 8 yr / 160,000 km)
    """
    F = {
        "target_range_km":      Reals(unit="km"),
        "energy_kWh_per_100km": Reals(unit="kWh/100km"),
    }
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "voltage_V":     Reals(unit="V"),
        "energy_kWh":    Reals(unit="kWh"),
        "peak_power_kW": Reals(unit="kW"),
        "durability":    Reals(unit="km"),
    }

    def __init__(self, name: str, *,
                 chemistry: str = "NMC811",
                 voltage_V: float = 400.0,
                 pack_specific_energy_Wh_per_kg: float = 160.0,
                 cost_per_kWh_usd: float = 130.0,
                 c_rate_peak: float = 4.0,
                 usable_soc_fraction: float = 0.80,
                 warranty_km: float = 160000.0):
        # NMC811 is the mainstream 2024 cathode; LFP is cheaper,
        # heavier, longer-lived; NCA is older Tesla-style high-energy.
        self.chemistry = chemistry
        self.voltage_V = voltage_V
        self.pack_specific_energy = pack_specific_energy_Wh_per_kg
        self.cost_per_kWh = cost_per_kWh_usd
        self.c_rate_peak = c_rate_peak
        self.usable_soc = usable_soc_fraction
        self.warranty_km = warranty_km
        super().__init__()
        self.name = name

    def h(self, f):
        # Required usable energy for the rated range.
        usable_kWh = f["target_range_km"] * f["energy_kWh_per_100km"] / 100.0
        # Total pack capacity inflates by 1 / usable_soc to leave
        # SOC headroom.
        total_kWh = usable_kWh / self.usable_soc + 2.0  # buffer for new pack
        weight = total_kWh * 1000.0 / self.pack_specific_energy + 35.0
        # Pack-level cost: cells + module hardware + BMS + cooling
        # plate + structural. We add a fixed BMS+structural overhead.
        cost = total_kWh * self.cost_per_kWh + 1100.0
        peak_power = total_kWh * self.c_rate_peak
        return {"weight": weight, "cost": cost,
                "voltage_V": self.voltage_V,
                "energy_kWh": usable_kWh,
                "peak_power_kW": peak_power,
                "durability": self.warranty_km}


class EVPowerElectronics(Module):
    """Inverter, DC-DC converter, vehicle control unit for EV.

    Sized to the traction motor's peak power. For dual-motor cars,
    there are two inverters; we model the combined assembly.

    Inputs (F):
        peak_motor_power_kW : combined peak power across all motors

    Outputs (R):
        weight     : kg
        cost       : USD
        efficiency : combined inverter + DC-DC
        durability : km
    """
    F = {"peak_motor_power_kW": Reals(unit="kW")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "efficiency": Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, sic: bool = True):
        self.sic = sic
        super().__init__()

    def h(self, f):
        p = f["peak_motor_power_kW"]
        if self.sic:
            weight = 0.12 * p + 6.0
            cost   = 22.0 * p + 280.0
            eta = 0.975
        else:
            weight = 0.18 * p + 6.0
            cost   = 12.0 * p + 200.0
            eta = 0.96
        return {"weight": weight, "cost": cost,
                "efficiency": eta,
                "durability": 220000.0}


class OnboardCharger(Module):
    """Onboard AC charger (3.7-22 kW) plus DC fast-charge interface.

    The onboard charger converts AC mains to DC battery charging.
    Higher-rated chargers add weight and cost but reduce charging
    time at home. DC fast-charging up to 250 kW (350+ kW for 800V
    architectures) bypasses the onboard charger entirely.

    Outputs (R):
        weight     : kg
        cost       : USD
        ac_kw      : maximum AC charging power (kW)
        dc_kw      : maximum DC fast-charge power (kW)
        durability : km
    """
    F = {"battery_voltage_V": Reals(unit="V")}
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "ac_kw":      Reals(unit="kW"),
        "dc_kw":      Reals(unit="kW"),
        "durability": Reals(unit="km"),
    }

    def __init__(self, *, ac_rating_kW: float = 11.0,
                 dc_rating_kW: float = 100.0):
        self.ac_rating_kW = ac_rating_kW
        self.dc_rating_kW = dc_rating_kW
        super().__init__()

    def h(self, f):
        # 800V cars get bigger DC-fast charge interface even at
        # the same AC rating; their onboard charger is similar.
        v = f["battery_voltage_V"]
        # OBC weight scales with AC rating: ~0.35 kg/kW.
        weight = 0.35 * self.ac_rating_kW + 4.0
        cost = 28.0 * self.ac_rating_kW + 240.0
        # DC fast-charge interface is mostly contactors + cabling;
        # 800V hardware costs 30% more than 400V.
        dc_weight = 6.0 + 0.020 * self.dc_rating_kW
        dc_cost   = 380.0 + 0.9 * self.dc_rating_kW
        if v > 600.0:
            dc_cost *= 1.30
        return {"weight": weight + dc_weight,
                "cost":   cost + dc_cost,
                "ac_kw":  self.ac_rating_kW,
                "dc_kw":  self.dc_rating_kW,
                "durability": 200000.0}


class SingleSpeedReducer(Module):
    """Fixed-ratio reduction gearbox for EV / hybrid drive.

    Most EVs use a single-speed reducer (typical ratio 8-10:1)
    between the motor and the wheels. Much simpler and lighter than
    a multi-speed gearbox; the motor's wide torque band makes
    multi-speed unnecessary for most use cases.

    Inputs (F):
        input_peak_power_kW  : motor peak power
        input_peak_torque_Nm : motor peak torque

    Outputs (R):
        weight     : kg
        cost       : USD
        efficiency : ~0.97 (single fixed gear pair)
        durability : km
    """
    F = {
        "input_peak_power_kW":  Reals(unit="kW"),
        "input_peak_torque_Nm": Reals(unit="Nm"),
    }
    R = {
        "weight":     Reals(unit="kg"),
        "cost":       Reals(unit="USD"),
        "efficiency": Reals(unit=""),
        "durability": Reals(unit="km"),
    }

    def __init__(self):
        super().__init__()

    def h(self, f):
        # Compact helical or spur reducer.
        p = f["input_peak_power_kW"]
        t = f["input_peak_torque_Nm"]
        weight = 14.0 + 0.06 * p + 0.005 * t
        cost   = 280.0 + 1.8 * p + 0.4 * t
        return {"weight": weight, "cost": cost,
                "efficiency": 0.97,
                "durability": 400000.0}


class BatteryThermalManagement(Module):
    """Liquid cooling / heating system for EV battery pack.

    Cell life and fast-charge capability depend on pack temperature
    being held in a narrow window (15-35 deg C ideal). The thermal
    management system uses a separate coolant loop with heat
    exchanger, electric pump, heating element (PTC or heat pump),
    and a chiller plate built into the pack. Heat-pump-based
    systems (Tesla Model Y, ID.4) reuse waste heat from motor and
    inverter for cabin heating.

    Inputs (F):
        battery_energy_kWh : sets the thermal mass to control
        peak_motor_power_kW: sets peak charging-heating load

    Outputs (R):
        weight     : kg
        cost       : USD
        electric_load : W (average draw)
        durability : km
    """
    F = {
        "battery_energy_kWh":  Reals(unit="kWh"),
        "peak_motor_power_kW": Reals(unit="kW"),
    }
    R = {
        "weight":        Reals(unit="kg"),
        "cost":          Reals(unit="USD"),
        "electric_load": Reals(unit="W"),
        "durability":    Reals(unit="km"),
    }

    def __init__(self, *, heat_pump: bool = True):
        self.heat_pump = heat_pump
        super().__init__()

    def h(self, f):
        e = f["battery_energy_kWh"]
        p = f["peak_motor_power_kW"]
        # Cooling plate, pump, valves, hoses, expansion tank.
        weight = 8.0 + 0.15 * e + 0.020 * p
        cost = 240.0 + 8.0 * e + 1.5 * p
        # Heat pump replaces resistive heater for ~5x cabin-heating COP.
        if self.heat_pump:
            weight += 6.0
            cost   += 700.0
            elec = 900.0
        else:
            elec = 1500.0   # PTC heater adds significant draw
        return {"weight": weight, "cost": cost,
                "electric_load": elec,
                "durability": 220000.0}


# ===========================================================================
# Catalog data: engine, body, transmission, suspension, tire, wheel,
# steering, electric-motor, and EV-battery variants.
#
# All numbers are illustrative, drawn from the published ranges in the
# bibliography at the top of this file. They are not OEM-specific.
# ===========================================================================


# ---- ICE engine catalogue ----
# Each tuple: (name, peak_kW, peak_Nm, displacement_L, peak_eta, weight_kg,
#              cost_usd, oil_L, durability_km, cycle, fuel, accessory_kW)
ICE_ENGINES = [
    EngineBlock("1.5L NA gas",        peak_power_kW=85,  peak_torque_Nm=145,
                displacement_L=1.5,   peak_efficiency=0.34,
                weight_kg=125,        cost_usd=2400,
                oil_capacity_L=4.0,   durability_km=320000,
                cycle="otto", fuel="gasoline", accessory_load_kW=1.4),
    EngineBlock("2.0L NA gas",        peak_power_kW=110, peak_torque_Nm=200,
                displacement_L=2.0,   peak_efficiency=0.34,
                weight_kg=145,        cost_usd=3200,
                oil_capacity_L=4.5,   durability_km=340000,
                cycle="otto", fuel="gasoline", accessory_load_kW=1.6),
    EngineBlock("1.5L turbo gas",     peak_power_kW=125, peak_torque_Nm=250,
                displacement_L=1.5,   peak_efficiency=0.36,
                weight_kg=140,        cost_usd=3600,
                oil_capacity_L=4.5,   durability_km=280000,
                cycle="otto", fuel="gasoline", accessory_load_kW=1.7),
    EngineBlock("2.0L turbo gas",     peak_power_kW=185, peak_torque_Nm=380,
                displacement_L=2.0,   peak_efficiency=0.36,
                weight_kg=165,        cost_usd=4400,
                oil_capacity_L=5.5,   durability_km=260000,
                cycle="otto", fuel="gasoline", accessory_load_kW=1.9),
    EngineBlock("3.0L V6 turbo gas",  peak_power_kW=275, peak_torque_Nm=500,
                displacement_L=3.0,   peak_efficiency=0.35,
                weight_kg=200,        cost_usd=6800,
                oil_capacity_L=7.0,   durability_km=240000,
                cycle="otto", fuel="gasoline", accessory_load_kW=2.2),
    EngineBlock("2.0L turbo diesel",  peak_power_kW=140, peak_torque_Nm=400,
                displacement_L=2.0,   peak_efficiency=0.41,
                weight_kg=180,        cost_usd=4900,
                oil_capacity_L=6.0,   durability_km=420000,
                cycle="diesel", fuel="diesel", accessory_load_kW=1.8),
]


# ---- Hybrid engine catalogue (Atkinson cycle, lower power, higher eta) ----
HYBRID_ENGINES = [
    EngineBlock("1.8L Atkinson hybrid",
                peak_power_kW=72,  peak_torque_Nm=142,
                displacement_L=1.8, peak_efficiency=0.41,
                weight_kg=115,      cost_usd=2900,
                oil_capacity_L=4.0, durability_km=400000,
                cycle="atkinson", fuel="gasoline", accessory_load_kW=0.6),
    EngineBlock("2.5L Atkinson hybrid",
                peak_power_kW=110, peak_torque_Nm=210,
                displacement_L=2.5, peak_efficiency=0.41,
                weight_kg=140,      cost_usd=3600,
                oil_capacity_L=4.8, durability_km=380000,
                cycle="atkinson", fuel="gasoline", accessory_load_kW=0.7),
]


# ---- Multi-speed transmissions for ICE ----
ICE_TRANSMISSIONS = [
    Transmission("6MT manual",
                 base_weight_kg=35, weight_per_kw=0.12,
                 base_cost_usd=820, cost_per_kw=4.8,
                 efficiency=0.96, durability_km=320000,
                 max_torque_Nm=400),
    Transmission("6AT torque-converter",
                 base_weight_kg=55, weight_per_kw=0.20,
                 base_cost_usd=1450, cost_per_kw=8.0,
                 efficiency=0.91, durability_km=280000,
                 max_torque_Nm=450),
    Transmission("8AT torque-converter",
                 base_weight_kg=68, weight_per_kw=0.22,
                 base_cost_usd=1950, cost_per_kw=10.0,
                 efficiency=0.93, durability_km=300000,
                 max_torque_Nm=550),
    Transmission("CVT",
                 base_weight_kg=42, weight_per_kw=0.15,
                 base_cost_usd=1200, cost_per_kw=7.0,
                 efficiency=0.88, durability_km=240000,
                 max_torque_Nm=300),
    Transmission("6DCT dual-clutch",
                 base_weight_kg=60, weight_per_kw=0.18,
                 base_cost_usd=1850, cost_per_kw=9.5,
                 efficiency=0.94, durability_km=260000,
                 max_torque_Nm=500),
]


# ---- Body styles ----
# Body weight here is the body-in-white plus panels, doors, glass, bumpers
# (i.e. the structural and exterior shell). The remaining curb mass comes
# from suspension, brakes, tires, wheels, powertrain, interior, fluids,
# and driver. Calibrated so total curb weight after Kleene convergence
# matches typical OEM ranges for each segment.
BODY_STYLES = [
    BodyFrame("compact_hatchback", base_weight_kg=620, base_cost_usd=7800,
              cd=0.30, frontal_area_m2=2.10, max_passengers=5,
              max_cargo_L=380, durability_km=350000),
    BodyFrame("mid_sedan",         base_weight_kg=720, base_cost_usd=9200,
              cd=0.28, frontal_area_m2=2.25, max_passengers=5,
              max_cargo_L=510, durability_km=380000),
    BodyFrame("mid_suv",           base_weight_kg=880, base_cost_usd=11200,
              cd=0.34, frontal_area_m2=2.70, max_passengers=7,
              max_cargo_L=950, durability_km=380000),
    BodyFrame("pickup_truck",      base_weight_kg=1050, base_cost_usd=11800,
              cd=0.42, frontal_area_m2=2.90, max_passengers=5,
              max_cargo_L=1500, durability_km=420000),
    BodyFrame("sport_coupe",       base_weight_kg=680, base_cost_usd=14500,
              cd=0.26, frontal_area_m2=2.05, max_passengers=4,
              max_cargo_L=250, durability_km=320000),
]


# ---- Suspension variants (front + rear sized together) ----
# Each variant is a (front_kwargs, rear_kwargs) pair.
SUSPENSION_VARIANTS = {
    "economy": dict(
        front=dict(type_name="econ-strut",
                   specific_mass_frac=0.045, base_cost_usd=320,
                   cost_per_kg_load=0.45, comfort_score=0.55,
                   durability_km=180000, max_supportable_kg=2200),
        rear=dict(type_name="twist-beam",
                  specific_mass_frac=0.030, base_cost_usd=180,
                  cost_per_kg_load=0.30, comfort_score=0.50,
                  durability_km=200000, max_supportable_kg=2200),
    ),
    "comfort": dict(
        front=dict(type_name="comfort-multilink",
                   specific_mass_frac=0.055, base_cost_usd=520,
                   cost_per_kg_load=0.70, comfort_score=0.80,
                   durability_km=160000, max_supportable_kg=2500),
        rear=dict(type_name="multilink-hydraulic",
                  specific_mass_frac=0.045, base_cost_usd=420,
                  cost_per_kg_load=0.55, comfort_score=0.80,
                  durability_km=170000, max_supportable_kg=2500),
    ),
    "sport": dict(
        front=dict(type_name="sport-double-wishbone",
                   specific_mass_frac=0.060, base_cost_usd=780,
                   cost_per_kg_load=1.10, comfort_score=0.45,
                   durability_km=140000, max_supportable_kg=2400),
        rear=dict(type_name="sport-multilink",
                  specific_mass_frac=0.050, base_cost_usd=640,
                  cost_per_kg_load=0.90, comfort_score=0.45,
                  durability_km=150000, max_supportable_kg=2400),
    ),
    "off_road": dict(
        front=dict(type_name="off-road-coil",
                   specific_mass_frac=0.075, base_cost_usd=580,
                   cost_per_kg_load=0.65, comfort_score=0.50,
                   durability_km=220000, max_supportable_kg=3500),
        rear=dict(type_name="off-road-leaf",
                  specific_mass_frac=0.085, base_cost_usd=480,
                  cost_per_kg_load=0.50, comfort_score=0.45,
                  durability_km=250000, max_supportable_kg=3500),
    ),
}


# ---- Tire catalogue (4 tires per set) ----
TIRES = [
    Tires("economy_AS",      mass_per_tire_kg=9.5,  cost_per_tire_usd=80,
          crr=0.011,  grip=0.95, durability_km=60000, noise_dB=72,
          max_load_kg=550, max_speed_kmh=190),
    Tires("premium_AS",      mass_per_tire_kg=10.5, cost_per_tire_usd=130,
          crr=0.0085, grip=1.00, durability_km=70000, noise_dB=69,
          max_load_kg=600, max_speed_kmh=210),
    Tires("performance_S",   mass_per_tire_kg=11.5, cost_per_tire_usd=210,
          crr=0.012,  grip=1.20, durability_km=35000, noise_dB=74,
          max_load_kg=620, max_speed_kmh=270),
    Tires("all_terrain",     mass_per_tire_kg=13.5, cost_per_tire_usd=180,
          crr=0.015,  grip=1.05, durability_km=55000, noise_dB=77,
          max_load_kg=750, max_speed_kmh=180),
    Tires("eco_LRR",         mass_per_tire_kg=9.0,  cost_per_tire_usd=110,
          crr=0.0070, grip=0.92, durability_km=75000, noise_dB=68,
          max_load_kg=550, max_speed_kmh=200),
    # Heavy-duty XL load-index tire for EVs and SUVs above 2 ton; rated
    # for full payload plus battery mass. Lower noise and similar grip
    # to premium_AS but higher rolling resistance.
    Tires("EV_XL",           mass_per_tire_kg=12.5, cost_per_tire_usd=180,
          crr=0.0085, grip=1.00, durability_km=65000, noise_dB=70,
          max_load_kg=800, max_speed_kmh=240),
]


# ---- Wheel options ----
WHEELS = [
    Wheels("steel_16",       mass_per_wheel_kg=10.5, cost_per_wheel_usd=85),
    Wheels("alloy_cast_17",  mass_per_wheel_kg=8.5,  cost_per_wheel_usd=170),
    Wheels("alloy_cast_19",  mass_per_wheel_kg=10.2, cost_per_wheel_usd=260),
    Wheels("alloy_forged_19",mass_per_wheel_kg=7.5,  cost_per_wheel_usd=520),
]


# ---- Steering systems ----
STEERING_OPTIONS = [
    SteeringSystem("hydraulic",
                   base_weight_kg=14, weight_scale=3.0,
                   base_cost_usd=320, cost_scale=120,
                   electric_load_W=80, durability_km=180000),
    SteeringSystem("EPS",
                   base_weight_kg=11, weight_scale=2.0,
                   base_cost_usd=480, cost_scale=180,
                   electric_load_W=320, durability_km=200000),
]


# ---- EV traction motors ----
EV_MOTORS = [
    ElectricMotor("EV_60kW_PMSM",
                  specific_power_kW_per_kg=1.7, cost_per_kW=22,
                  base_cost_usd=380, peak_efficiency=0.96,
                  torque_per_kW=2.8, durability_km=400000,
                  motor_type="PMSM"),
    ElectricMotor("EV_130kW_PMSM",
                  specific_power_kW_per_kg=2.0, cost_per_kW=20,
                  base_cost_usd=520, peak_efficiency=0.96,
                  torque_per_kW=2.4, durability_km=400000),
    ElectricMotor("EV_180kW_PMSM",
                  specific_power_kW_per_kg=2.3, cost_per_kW=18,
                  base_cost_usd=680, peak_efficiency=0.97,
                  torque_per_kW=2.1, durability_km=420000),
    ElectricMotor("EV_300kW_PMSM",
                  specific_power_kW_per_kg=2.6, cost_per_kW=18,
                  base_cost_usd=950, peak_efficiency=0.97,
                  torque_per_kW=1.9, durability_km=380000),
    ElectricMotor("EV_220kW_induction",
                  specific_power_kW_per_kg=1.8, cost_per_kW=14,
                  base_cost_usd=520, peak_efficiency=0.94,
                  torque_per_kW=1.6, durability_km=450000,
                  motor_type="IM"),
]


# ---- EV battery packs ----
EV_BATTERIES = [
    HVBatteryEV("LFP_40kWh_400V",        chemistry="LFP",   voltage_V=350,
                pack_specific_energy_Wh_per_kg=140, cost_per_kWh_usd=105,
                c_rate_peak=3.0,  usable_soc_fraction=0.90,
                warranty_km=200000),
    HVBatteryEV("NMC_60kWh_400V",        chemistry="NMC811",voltage_V=400,
                pack_specific_energy_Wh_per_kg=160, cost_per_kWh_usd=130,
                c_rate_peak=4.0,  usable_soc_fraction=0.80,
                warranty_km=160000),
    HVBatteryEV("NMC_85kWh_400V",        chemistry="NMC811",voltage_V=400,
                pack_specific_energy_Wh_per_kg=170, cost_per_kWh_usd=128,
                c_rate_peak=4.0,  usable_soc_fraction=0.80,
                warranty_km=160000),
    HVBatteryEV("NMC_100kWh_800V",       chemistry="NMC811",voltage_V=800,
                pack_specific_energy_Wh_per_kg=180, cost_per_kWh_usd=145,
                c_rate_peak=5.0,  usable_soc_fraction=0.85,
                warranty_km=170000),
    HVBatteryEV("LFP_75kWh_400V",        chemistry="LFP",   voltage_V=350,
                pack_specific_energy_Wh_per_kg=145, cost_per_kWh_usd=98,
                c_rate_peak=3.0,  usable_soc_fraction=0.95,
                warranty_km=250000),
]


# ===========================================================================
# Helper utilities for the lambda-based macro aggregations
# ===========================================================================


def sum_total_mass(x: Dict[str, float], modules: List[str]) -> float:
    """Sum of weight ports for the given module names plus DRIVER_MASS_KG.

    Each entry of ``modules`` is a module name; the function pulls
    x[mod + ".weight"] for each one and returns the total. Driver
    mass per ISO 1176 is added once.
    """
    return DRIVER_MASS_KG + sum(x[f"{m}.weight"] for m in modules)


def total_cost(x: Dict[str, float], modules: List[str]) -> float:
    """Sum of cost ports across the named modules."""
    return sum(x[f"{m}.cost"] for m in modules)


def assembly_overhead_usd(curb_weight_kg: float) -> float:
    """Empirical assembly overhead: ~$1.50 per kg of vehicle. Captures
    labour, paint, BIW welding, final assembly, dealer prep."""
    return 1.5 * curb_weight_kg


def co2_per_km_ice(fuel_consumption_L_100km: float, is_diesel: float) -> float:
    """CO2 per km in g/km from fuel consumption and fuel type."""
    kg_per_L = DIESEL_CO2_PER_L if is_diesel > 0.5 else GASOLINE_CO2_PER_L
    return fuel_consumption_L_100km * kg_per_L * 10.0   # × 1000/100


# ===========================================================================
# Build function: ICE car
# ===========================================================================


def build_ice_car(*, mission: Dict[str, float],
                  body: BodyFrame,
                  engine: EngineBlock,
                  forced_induction: ForcedInduction,
                  transmission: Transmission,
                  suspension_type: str,
                  tire_choice: Tires,
                  wheel_choice: Wheels,
                  steering_choice: SteeringSystem,
                  drivetrain_layout: str = "fwd",
                  trim_level: float = 0.3,
                  adas_level: int = 2):
    """Assemble a System for an ICE vehicle.

    The wiring expresses three coupled cycles:

    1. **Mass spiral**: every load-bearing subsystem (suspension,
       brakes, tires, wheels, steering) has a ``design_mass`` F port
       constrained to be >= the sum of every module's weight R port
       plus the driver. Heavier subsystems imply higher loads, which
       imply heavier subsystems. The Kleene iteration converges this
       to its self-consistent fixed point.

    2. **Power coupling**: the engine's ``target_peak_power`` F port
       is constrained to cover the larger of the top-speed road load
       (which depends on mass) and the acceleration demand (which
       also depends on mass). The engine's weight feeds back into
       the mass spiral.

    3. **Fuel-energy coupling**: the fuel tank's capacity is sized
       to ``range × fuel_consumption``, where the consumption depends
       on the mass (and therefore on the tank size), closing a third
       loop through the tank's weight contribution.
    """
    name = f"ICE_{engine.name.replace(' ', '_')}_{body.style_name}_{transmission.name.replace(' ', '_')}"
    sys = System(name)

    # ---- Outer F: mission ---------------------------------------------
    f_pass  = sys.provides("target_passengers",  unit="people")
    f_cargo = sys.provides("target_cargo_L",     unit="L")
    f_speed = sys.provides("target_max_speed",   unit="km/h")
    f_range = sys.provides("target_range_km",    unit="km")
    f_decel = sys.provides("target_decel_g",     unit="g")
    f_0_100 = sys.provides("target_0_100_s",     unit="s")

    # ---- Outer R: macro objectives ------------------------------------
    sys.requires("production_cost",       unit="USD")
    sys.requires("curb_weight",           unit="kg")
    sys.requires("fuel_consumption",      unit="L/100km")
    sys.requires("co2_per_km",            unit="g/km")
    sys.requires("maintenance_per_year",  unit="USD/yr")
    sys.requires("durability",            unit="km")

    # ---- Modules ------------------------------------------------------
    body_m   = sys.add("body",  body)

    susp_F = SUSPENSION_VARIANTS[suspension_type]["front"]
    susp_R = SUSPENSION_VARIANTS[suspension_type]["rear"]
    fr_m   = sys.add("susp_front", FrontSuspension(**susp_F))
    rr_m   = sys.add("susp_rear",  RearSuspension(**susp_R))
    bf_m   = sys.add("brakes_front", BrakesFront())
    br_m   = sys.add("brakes_rear",  BrakesRear())
    st_m   = sys.add("steering",     steering_choice)
    ti_m   = sys.add("tires",        tire_choice)
    wh_m   = sys.add("wheels",       wheel_choice)

    hv_m   = sys.add("hvac",     HVAC(electric_compressor=False))
    in_m   = sys.add("interior", InteriorTrim(trim_level=trim_level))
    sf_m   = sys.add("safety",   SafetySystems(adas_level=adas_level))
    li_m   = sys.add("lights",   LightingAndInfotainment())

    en_m   = sys.add("engine",         engine)
    fi_m   = sys.add("forced_induction", forced_induction)
    inj_m  = sys.add("fuel_injection", FuelInjection(system="direct"))
    ex_m   = sys.add("exhaust",        ExhaustAftertreatment())
    co_m   = sys.add("cooling",        CoolingSystem())
    lu_m   = sys.add("lube",           LubricationSystem())
    tr_m   = sys.add("trans",          transmission)
    df_m   = sys.add("diff",           DriveshaftDifferential(layout=drivetrain_layout))
    tk_m   = sys.add("tank",           FuelTank(
        fuel_density_kg_per_L=(DIESEL_DENSITY if engine.fuel == "diesel"
                                else GASOLINE_DENSITY)))
    al_m   = sys.add("alt",            Alternator())
    bt_m   = sys.add("bat12",          Battery12V())
    sr_m   = sys.add("starter",        StarterMotor())

    # Convenience list of every module's name for the lambdas below.
    ICE_MODULES = ["body", "susp_front", "susp_rear",
                   "brakes_front", "brakes_rear",
                   "steering", "tires", "wheels",
                   "hvac", "interior", "safety", "lights",
                   "engine", "forced_induction", "fuel_injection",
                   "exhaust", "cooling", "lube", "trans", "diff",
                   "tank", "alt", "bat12", "starter"]

    # ---- Mass spiral: every load-bearing module reads total mass ------
    total_weight_expr = (
        body_m.weight + fr_m.weight + rr_m.weight
        + bf_m.weight + br_m.weight + st_m.weight
        + ti_m.weight + wh_m.weight
        + hv_m.weight + in_m.weight + sf_m.weight + li_m.weight
        + en_m.weight + fi_m.weight + inj_m.weight + ex_m.weight
        + co_m.weight + lu_m.weight + tr_m.weight + df_m.weight
        + tk_m.weight + al_m.weight + bt_m.weight + sr_m.weight
        + DRIVER_MASS_KG
    )

    fr_m.design_mass >= total_weight_expr
    rr_m.design_mass >= total_weight_expr
    bf_m.design_mass >= total_weight_expr
    br_m.design_mass >= total_weight_expr
    st_m.design_mass >= total_weight_expr
    ti_m.design_mass >= total_weight_expr
    wh_m.design_mass >= total_weight_expr
    li_m.design_mass >= total_weight_expr

    # ---- Mission propagation ------------------------------------------
    body_m.target_passengers >= f_pass
    body_m.target_cargo_L    >= f_cargo
    fr_m.target_max_speed    >= f_speed
    rr_m.target_max_speed    >= f_speed
    bf_m.target_max_speed    >= f_speed
    br_m.target_max_speed    >= f_speed
    bf_m.target_decel_g      >= f_decel
    br_m.target_decel_g      >= f_decel
    ti_m.target_max_speed    >= f_speed

    hv_m.target_passengers   >= f_pass
    in_m.target_passengers   >= f_pass
    sf_m.target_passengers   >= f_pass
    sf_m.target_max_speed    >= f_speed

    # ---- Powertrain internal wiring -----------------------------------
    # Forced induction, fuel injection, exhaust, transmission, and
    # differential are sized to the engine's rated capacity (R ports),
    # not its target demand (F port). The rated capacity is the engine's
    # actual ability to deliver; the demand is what other constraints
    # require. Using the R port keeps the operator-overloaded constraint
    # DSL legal (F ports cannot appear on RHS of >=).
    fi_m.engine_peak_power_kW       >= en_m.rated_peak_power_kW
    inj_m.engine_peak_power_kW      >= en_m.rated_peak_power_kW
    inj_m.engine_peak_efficiency    >= en_m.peak_efficiency
    ex_m.engine_peak_power_kW       >= en_m.rated_peak_power_kW
    ex_m.fuel_type                  >= en_m.fuel_type
    co_m.heat_to_dissipate_kW       >= en_m.heat_rejection_at_peak
    lu_m.engine_displacement_L      >= en_m.displacement_L
    lu_m.engine_oil_capacity_L      >= en_m.oil_capacity_L
    sr_m.engine_displacement_L      >= en_m.displacement_L

    tr_m.input_peak_power_kW        >= en_m.rated_peak_power_kW
    tr_m.input_peak_torque_Nm       >= en_m.rated_peak_torque_Nm
    df_m.peak_torque_Nm             >= en_m.rated_peak_torque_Nm

    # ---- Lambdas for engine sizing, fuel consumption, electrical -----
    # These are closed-form callables of the x context (module R ports).

    def total_mass_from_x(x):
        return DRIVER_MASS_KG + sum(x[f"{m}.weight"] for m in ICE_MODULES)

    def peak_power_demand_kW(x):
        """Engine peak power: max of top-speed and acceleration demand,
        plus drivetrain losses."""
        m_kg = total_mass_from_x(x)
        cd = x["body.cd"]
        A  = x["body.frontal_area"]
        crr = x["tires.crr"]
        p_top = top_speed_power_kw(mission["target_max_speed"], m_kg, cd, A, crr)
        p_accel = acceleration_power_kw(mission["target_0_100_s"], m_kg)
        # Engine must overcome drivetrain efficiency to deliver this at
        # the wheels. We divide by the transmission efficiency.
        eta_dt = max(x["trans.efficiency"], 0.5)
        return max(p_top, p_accel) / eta_dt

    def peak_torque_demand_Nm(x):
        """Engine peak torque requirement.

        In a multi-speed transmission, low gears multiply engine torque
        by 3 to 5x; the engine's intrinsic peak torque is rarely the
        binding acceleration constraint. We just impose a modest
        minimum tied to engine power (0.55 Nm per kW of demand), so the
        catalog's intrinsic (power, torque) pairing handles the rest.
        """
        return 0.55 * peak_power_demand_kW(x)

    def peak_electric_load_W(x):
        """Peak 12V electrical load: HVAC + steering + safety + lights
        + cooling fan."""
        return (x["hvac.electric_load"] + x["steering.electric_load"]
                + x["safety.electric_load"] + x["lights.electric_load"]
                + 1000.0 * x["cooling.fan_power_kW"])

    def fuel_consumption_L_100km(x):
        """Cruise fuel consumption at 100 km/h (highway-cycle proxy).

        Accounts for road load (drag + roll), accessory mechanical
        load (alternator + cooling fan + accessories), drivetrain
        efficiency, and engine part-load efficiency.
        """
        m_kg = total_mass_from_x(x)
        cd  = x["body.cd"]
        A   = x["body.frontal_area"]
        crr = x["tires.crr"]
        road_load_kW = cruise_road_load_kw(100.0, m_kg, cd, A, crr)
        accessory_kW = (x["alt.parasitic_load_kW"]
                        + x["engine.accessory_power_load"]
                        + x["cooling.fan_power_kW"])
        wheel_demand_kW = road_load_kW + accessory_kW
        eta_dt = max(x["trans.efficiency"], 0.5)
        engine_out_kW = wheel_demand_kW / eta_dt
        # At cruise, engine runs at ~80% of peak efficiency
        cruise_eta = max(0.80 * x["engine.peak_efficiency"], 0.05)
        fuel_power_kW = engine_out_kW / cruise_eta
        is_diesel = x["engine.fuel_type"] > 0.5
        lhv = (DIESEL_LHV_MJ_PER_L if is_diesel
               else GASOLINE_LHV_MJ_PER_L)
        # L/100 km = fuel_kW * 3.6 MJ/kWh / LHV[MJ/L]
        return fuel_power_kW * 3.6 / lhv

    # Engine sizing constraints (lambda-based since they depend on
    # the total mass which is itself a sum of module weights).
    sys.constrain("engine.target_peak_power_kW",  peak_power_demand_kW)
    sys.constrain("engine.target_peak_torque_Nm", peak_torque_demand_Nm)

    # Fuel tank sizing: range fixed by mission, consumption from lambda.
    sys.constrain("tank.target_range_km",
                  lambda x, r=mission["target_range_km"]: r)
    sys.constrain("tank.fuel_consumption_L_per_100km",
                  fuel_consumption_L_100km)

    # Electrical loads.
    sys.constrain("alt.peak_electric_load_W",   peak_electric_load_W)
    sys.constrain("bat12.peak_electric_load_W", peak_electric_load_W)

    # ---- Outer R aggregations -----------------------------------------
    sys.constrain("production_cost",
        lambda x: total_cost(x, ICE_MODULES)
                  + assembly_overhead_usd(total_mass_from_x(x)))
    sys.constrain("curb_weight", total_mass_from_x)
    sys.constrain("fuel_consumption", fuel_consumption_L_100km)
    sys.constrain("co2_per_km",
        lambda x: co2_per_km_ice(fuel_consumption_L_100km(x),
                                  x["engine.fuel_type"]))
    sys.constrain("maintenance_per_year", lambda x: (
        # Tire replacement (full set every durability_km)
        x["tires.cost"] * 15000.0 / max(x["tires.durability"], 1.0)
        # Brakes: pads, plus periodic rotor service. ~30% of cost
        # spread over component lifetime.
        + 0.30 * (x["brakes_front.cost"] + x["brakes_rear.cost"])
            * 15000.0 / max(min(x["brakes_front.durability"],
                                  x["brakes_rear.durability"]), 1.0)
        # Oil and filter changes
        + 80.0 * 15000.0 / max(x["lube.oil_change_interval"], 1.0)
        # Baseline scheduled service
        + 380.0
        # Battery (12V) replacement
        + x["bat12.cost"] * 15000.0 / max(x["bat12.durability"], 1.0)
    ))
    sys.constrain("durability", lambda x: harmonic_mean([
        x["engine.durability"], x["trans.durability"],
        x["body.durability"],   x["susp_front.durability"],
        x["susp_rear.durability"],
        # Battery and exhaust are also durability-critical
        x["bat12.durability"], x["exhaust.durability"],
    ]))

    return sys.build()


# ===========================================================================
# Build function: hybrid (HEV) car
# ===========================================================================


def build_hybrid_car(*, mission: Dict[str, float],
                     body: BodyFrame,
                     engine: EngineBlock,
                     motor_peak_power_kW: float,
                     hv_battery_chemistry: str,
                     power_electronics_sic: bool,
                     suspension_type: str,
                     tire_choice: Tires,
                     wheel_choice: Wheels,
                     steering_choice: SteeringSystem,
                     drivetrain_layout: str = "fwd",
                     trim_level: float = 0.3,
                     adas_level: int = 2):
    """Assemble a System for a parallel power-split hybrid vehicle.

    The hybrid topology replaces the multi-speed transmission of an ICE
    car with a planetary power-split unit driving an electric motor and
    the front wheels in parallel with an Atkinson-cycle engine. The
    motor draws from a small high-voltage battery; the engine and the
    motor each contribute to total wheel power, allowing the engine to
    run in a narrow high-efficiency band. The starter and alternator
    of the ICE car are replaced by the motor-generator and a DC-DC
    converter from the HV pack.

    The mass spiral closes the same way as the ICE car, with extra
    contributions from the motor, HV battery, and power electronics,
    offset by removing the starter and alternator.
    """
    name = (f"HEV_{engine.name.replace(' ', '_')}_"
            f"{int(motor_peak_power_kW)}kW_{body.style_name}")
    sys = System(name)

    # ---- Outer F: same as ICE car -------------------------------------
    f_pass  = sys.provides("target_passengers",  unit="people")
    f_cargo = sys.provides("target_cargo_L",     unit="L")
    f_speed = sys.provides("target_max_speed",   unit="km/h")
    f_range = sys.provides("target_range_km",    unit="km")
    f_decel = sys.provides("target_decel_g",     unit="g")
    f_0_100 = sys.provides("target_0_100_s",     unit="s")

    # ---- Outer R: same headline metrics as the ICE car. Electric
    # consumption is zero for a (non-plug-in) hybrid but is still
    # exposed so HEV and EV results line up on the same axes.
    sys.requires("production_cost",       unit="USD")
    sys.requires("curb_weight",           unit="kg")
    sys.requires("fuel_consumption",      unit="L/100km")
    sys.requires("co2_per_km",            unit="g/km")
    sys.requires("maintenance_per_year",  unit="USD/yr")
    sys.requires("durability",            unit="km")

    # ---- Common chassis and auxiliary modules -------------------------
    body_m   = sys.add("body",  body)

    susp_F = SUSPENSION_VARIANTS[suspension_type]["front"]
    susp_R = SUSPENSION_VARIANTS[suspension_type]["rear"]
    fr_m   = sys.add("susp_front", FrontSuspension(**susp_F))
    rr_m   = sys.add("susp_rear",  RearSuspension(**susp_R))
    bf_m   = sys.add("brakes_front", BrakesFront())
    br_m   = sys.add("brakes_rear",  BrakesRear())
    st_m   = sys.add("steering",     steering_choice)
    ti_m   = sys.add("tires",        tire_choice)
    wh_m   = sys.add("wheels",       wheel_choice)

    hv_m   = sys.add("hvac",     HVAC(electric_compressor=True))
    in_m   = sys.add("interior", InteriorTrim(trim_level=trim_level))
    sf_m   = sys.add("safety",   SafetySystems(adas_level=adas_level))
    li_m   = sys.add("lights",   LightingAndInfotainment())

    # ---- Powertrain: Atkinson engine + reduced ICE periphery ----------
    en_m   = sys.add("engine",         engine)
    inj_m  = sys.add("fuel_injection", FuelInjection(system="direct"))
    ex_m   = sys.add("exhaust",        ExhaustAftertreatment())
    co_m   = sys.add("cooling",        CoolingSystem())
    lu_m   = sys.add("lube",           LubricationSystem())

    # Hybrid replaces transmission with power-split + integrated motor.
    psg_m  = sys.add("power_split",       HybridPowerSplit())
    mg_m   = sys.add("motor_generator",   HybridMotorGenerator())
    pe_m   = sys.add("power_electronics", HybridPowerElectronics(sic=power_electronics_sic))
    hvb_m  = sys.add("hv_battery",        HVBatteryHybrid(chemistry=hv_battery_chemistry))

    df_m   = sys.add("diff",  DriveshaftDifferential(layout=drivetrain_layout))
    tk_m   = sys.add("tank",  FuelTank(
        fuel_density_kg_per_L=(DIESEL_DENSITY if engine.fuel == "diesel"
                                else GASOLINE_DENSITY)))
    # No alternator (DC-DC from HV pack), no starter (integrated
    # starter-generator built into the motor-generator).
    bt_m   = sys.add("bat12", Battery12V())

    HEV_MODULES = ["body", "susp_front", "susp_rear",
                   "brakes_front", "brakes_rear",
                   "steering", "tires", "wheels",
                   "hvac", "interior", "safety", "lights",
                   "engine", "fuel_injection", "exhaust",
                   "cooling", "lube",
                   "power_split", "motor_generator",
                   "power_electronics", "hv_battery",
                   "diff", "tank", "bat12"]

    # ---- Mass spiral --------------------------------------------------
    total_weight_expr = (
        body_m.weight + fr_m.weight + rr_m.weight
        + bf_m.weight + br_m.weight + st_m.weight
        + ti_m.weight + wh_m.weight
        + hv_m.weight + in_m.weight + sf_m.weight + li_m.weight
        + en_m.weight + inj_m.weight + ex_m.weight
        + co_m.weight + lu_m.weight
        + psg_m.weight + mg_m.weight
        + pe_m.weight + hvb_m.weight
        + df_m.weight + tk_m.weight + bt_m.weight
        + DRIVER_MASS_KG
    )

    fr_m.design_mass >= total_weight_expr
    rr_m.design_mass >= total_weight_expr
    bf_m.design_mass >= total_weight_expr
    br_m.design_mass >= total_weight_expr
    st_m.design_mass >= total_weight_expr
    ti_m.design_mass >= total_weight_expr
    wh_m.design_mass >= total_weight_expr
    li_m.design_mass >= total_weight_expr

    # ---- Mission propagation ------------------------------------------
    body_m.target_passengers >= f_pass
    body_m.target_cargo_L    >= f_cargo
    fr_m.target_max_speed    >= f_speed
    rr_m.target_max_speed    >= f_speed
    bf_m.target_max_speed    >= f_speed
    br_m.target_max_speed    >= f_speed
    bf_m.target_decel_g      >= f_decel
    br_m.target_decel_g      >= f_decel
    ti_m.target_max_speed    >= f_speed

    hv_m.target_passengers   >= f_pass
    in_m.target_passengers   >= f_pass
    sf_m.target_passengers   >= f_pass
    sf_m.target_max_speed    >= f_speed

    # ---- Powertrain internal wiring -----------------------------------
    inj_m.engine_peak_power_kW   >= en_m.rated_peak_power_kW
    inj_m.engine_peak_efficiency >= en_m.peak_efficiency
    ex_m.engine_peak_power_kW    >= en_m.rated_peak_power_kW
    ex_m.fuel_type               >= en_m.fuel_type
    co_m.heat_to_dissipate_kW    >= en_m.heat_rejection_at_peak
    lu_m.engine_displacement_L   >= en_m.displacement_L
    lu_m.engine_oil_capacity_L   >= en_m.oil_capacity_L

    # Hybrid-specific wiring: motor sized by chosen peak power,
    # battery sized by motor power, power electronics sized by motor.
    sys.constrain("motor_generator.motor_peak_power_kW",
                  lambda x, p=motor_peak_power_kW: p)
    sys.constrain("motor_generator.battery_voltage_V",
                  lambda x: x["hv_battery.voltage_V"])
    sys.constrain("hv_battery.motor_peak_power_kW",
                  lambda x, p=motor_peak_power_kW: p)
    sys.constrain("power_electronics.motor_peak_power_kW",
                  lambda x, p=motor_peak_power_kW: p)

    # Power-split unit sized by combined engine + motor power.
    psg_m.engine_peak_power_kW >= en_m.rated_peak_power_kW
    sys.constrain("power_split.motor_peak_power_kW",
                  lambda x, p=motor_peak_power_kW: p)
    df_m.peak_torque_Nm >= en_m.rated_peak_torque_Nm

    # ---- Power and consumption lambdas --------------------------------

    def total_mass_from_x(x):
        return DRIVER_MASS_KG + sum(x[f"{mod}.weight"] for mod in HEV_MODULES)

    def peak_power_demand_kW(x):
        """Combined engine + motor must cover top-speed and accel
        demand, accounting for the power-split unit's efficiency."""
        m_kg = total_mass_from_x(x)
        cd = x["body.cd"]
        A  = x["body.frontal_area"]
        crr = x["tires.crr"]
        p_top = top_speed_power_kw(mission["target_max_speed"], m_kg, cd, A, crr)
        p_accel = acceleration_power_kw(mission["target_0_100_s"], m_kg)
        eta_dt = max(x["power_split.efficiency"], 0.7)
        return max(p_top, p_accel) / eta_dt

    # Engine sized to peak power demand minus motor assist. The motor
    # provides peak power for short bursts; the engine handles sustained
    # cruise. We size the engine to (demand - 0.7 * motor_power), giving
    # a 70% motor-assist credit during peak demand.
    def engine_power_demand_kW(x):
        return max(peak_power_demand_kW(x) - 0.7 * motor_peak_power_kW, 0.0)

    sys.constrain("engine.target_peak_power_kW",  engine_power_demand_kW)
    sys.constrain("engine.target_peak_torque_Nm",
                  lambda x: 0.55 * engine_power_demand_kW(x))

    # Fuel tank sizing.
    def hev_fuel_consumption_L_100km(x):
        """Hybrid cruise consumption: 25 to 35% lower than ICE for the
        same vehicle, due to engine off at idle, optimal-RPM operation,
        and regenerative braking. We model this as a 30% reduction
        applied to the ICE-equivalent road-load expression."""
        m_kg = total_mass_from_x(x)
        cd = x["body.cd"]; A = x["body.frontal_area"]; crr = x["tires.crr"]
        road_load_kW = cruise_road_load_kw(100.0, m_kg, cd, A, crr)
        accessory_kW = (x["engine.accessory_power_load"]
                        + x["cooling.fan_power_kW"]
                        + 0.001 * x["hvac.electric_load"])
        wheel_demand_kW = road_load_kW + accessory_kW
        engine_out_kW = wheel_demand_kW / max(x["power_split.efficiency"], 0.7)
        # Hybrid engines run near their best-efficiency point much more
        # of the time than a conventional ICE; we credit 95% of peak eta.
        cruise_eta = max(0.95 * x["engine.peak_efficiency"], 0.05)
        fuel_power_kW = engine_out_kW / cruise_eta
        # 30% hybrid efficiency credit (regen braking, idle stop, etc.)
        fuel_power_kW *= 0.70
        is_diesel = x["engine.fuel_type"] > 0.5
        lhv = DIESEL_LHV_MJ_PER_L if is_diesel else GASOLINE_LHV_MJ_PER_L
        return fuel_power_kW * 3.6 / lhv

    sys.constrain("tank.target_range_km",
                  lambda x, r=mission["target_range_km"]: r)
    sys.constrain("tank.fuel_consumption_L_per_100km",
                  hev_fuel_consumption_L_100km)

    # 12V loads (still needed for legacy 12V accessories).
    def peak_electric_load_W(x):
        return (x["hvac.electric_load"] + x["steering.electric_load"]
                + x["safety.electric_load"] + x["lights.electric_load"]
                + 1000.0 * x["cooling.fan_power_kW"])
    sys.constrain("bat12.peak_electric_load_W", peak_electric_load_W)

    # ---- Outer R aggregations -----------------------------------------
    sys.constrain("production_cost",
        lambda x: total_cost(x, HEV_MODULES)
                  + assembly_overhead_usd(total_mass_from_x(x)))
    sys.constrain("curb_weight", total_mass_from_x)
    sys.constrain("fuel_consumption", hev_fuel_consumption_L_100km)
    sys.constrain("co2_per_km",
        lambda x: co2_per_km_ice(hev_fuel_consumption_L_100km(x),
                                  x["engine.fuel_type"]))
    sys.constrain("maintenance_per_year", lambda x: (
        x["tires.cost"] * 15000.0 / max(x["tires.durability"], 1.0)
        + 0.20 * (x["brakes_front.cost"] + x["brakes_rear.cost"])
            * 15000.0 / max(min(x["brakes_front.durability"],
                                  x["brakes_rear.durability"]), 1.0)
        + 80.0 * 15000.0 / max(x["lube.oil_change_interval"], 1.0)
        + 280.0  # slightly lower than ICE: regen reduces brake wear
        + x["bat12.cost"] * 15000.0 / max(x["bat12.durability"], 1.0)
    ))
    sys.constrain("durability", lambda x: harmonic_mean([
        x["engine.durability"], x["power_split.durability"],
        x["body.durability"],   x["susp_front.durability"],
        x["susp_rear.durability"],
        x["hv_battery.durability"], x["motor_generator.durability"],
        x["bat12.durability"], x["exhaust.durability"],
    ]))

    return sys.build()


# ===========================================================================
# Build function: full battery-electric vehicle (EV)
# ===========================================================================


def build_ev_car(*, mission: Dict[str, float],
                 body: BodyFrame,
                 motor: ElectricMotor,
                 battery: HVBatteryEV,
                 power_electronics_sic: bool = True,
                 obc_ac_kW: float = 11.0,
                 obc_dc_kW: float = 150.0,
                 battery_heat_pump: bool = True,
                 suspension_type: str = "comfort",
                 tire_choice: Tires = None,
                 wheel_choice: Wheels = None,
                 steering_choice: SteeringSystem = None,
                 drivetrain_layout: str = "fwd",
                 trim_level: float = 0.3,
                 adas_level: int = 2):
    """Assemble a System for a full battery-electric vehicle.

    The EV topology removes everything related to combustion (engine
    block, forced induction, fuel injection, exhaust aftertreatment,
    engine cooling, lubrication, transmission, fuel tank, alternator,
    12V starter) and replaces them with a traction motor, high-voltage
    battery, power electronics, on-board charger, single-speed reducer,
    and a separate battery thermal-management loop.

    The mass cycle here is dominated by the battery, which is sized to
    deliver the target range at the expected energy consumption. The
    battery's own mass contributes 20 to 35% of curb weight, so the
    Kleene iteration takes more steps to converge than for ICE.
    """
    name = (f"EV_{motor.name}_{battery.name}_{body.style_name}")
    sys = System(name)

    # ---- Outer F: same mission as ICE / HEV ---------------------------
    f_pass  = sys.provides("target_passengers",  unit="people")
    f_cargo = sys.provides("target_cargo_L",     unit="L")
    f_speed = sys.provides("target_max_speed",   unit="km/h")
    f_range = sys.provides("target_range_km",    unit="km")
    f_decel = sys.provides("target_decel_g",     unit="g")
    f_0_100 = sys.provides("target_0_100_s",     unit="s")

    # ---- Outer R: same shape as the ICE / HEV cars. Fuel consumption
    # is zero for an EV but is still reported so all three architectures
    # are directly comparable on the same cost vector.
    sys.requires("production_cost",       unit="USD")
    sys.requires("curb_weight",           unit="kg")
    sys.requires("fuel_consumption",      unit="L/100km")
    sys.requires("energy_consumption",    unit="kWh/100km")
    sys.requires("co2_per_km",            unit="g/km")
    sys.requires("maintenance_per_year",  unit="USD/yr")
    sys.requires("durability",            unit="km")

    # ---- Common chassis and auxiliary ---------------------------------
    body_m   = sys.add("body",  body)

    susp_F = SUSPENSION_VARIANTS[suspension_type]["front"]
    susp_R = SUSPENSION_VARIANTS[suspension_type]["rear"]
    fr_m   = sys.add("susp_front", FrontSuspension(**susp_F))
    rr_m   = sys.add("susp_rear",  RearSuspension(**susp_R))
    bf_m   = sys.add("brakes_front", BrakesFront())
    br_m   = sys.add("brakes_rear",  BrakesRear())
    st_m   = sys.add("steering",     steering_choice)
    ti_m   = sys.add("tires",        tire_choice)
    wh_m   = sys.add("wheels",       wheel_choice)

    hv_m   = sys.add("hvac",     HVAC(electric_compressor=True))
    in_m   = sys.add("interior", InteriorTrim(trim_level=trim_level))
    sf_m   = sys.add("safety",   SafetySystems(adas_level=adas_level))
    li_m   = sys.add("lights",   LightingAndInfotainment())

    # ---- EV powertrain ------------------------------------------------
    mot_m  = sys.add("motor",     motor)
    bat_m  = sys.add("hv_battery", battery)
    pe_m   = sys.add("power_electronics",
                     EVPowerElectronics(sic=power_electronics_sic))
    obc_m  = sys.add("charger", OnboardCharger(
        ac_rating_kW=obc_ac_kW, dc_rating_kW=obc_dc_kW))
    red_m  = sys.add("reducer", SingleSpeedReducer())
    btm_m  = sys.add("thermal", BatteryThermalManagement(
        heat_pump=battery_heat_pump))

    EV_MODULES = ["body", "susp_front", "susp_rear",
                  "brakes_front", "brakes_rear",
                  "steering", "tires", "wheels",
                  "hvac", "interior", "safety", "lights",
                  "motor", "hv_battery", "power_electronics",
                  "charger", "reducer", "thermal"]

    # ---- Mass spiral --------------------------------------------------
    total_weight_expr = (
        body_m.weight + fr_m.weight + rr_m.weight
        + bf_m.weight + br_m.weight + st_m.weight
        + ti_m.weight + wh_m.weight
        + hv_m.weight + in_m.weight + sf_m.weight + li_m.weight
        + mot_m.weight + bat_m.weight + pe_m.weight
        + obc_m.weight + red_m.weight + btm_m.weight
        + DRIVER_MASS_KG
    )

    fr_m.design_mass >= total_weight_expr
    rr_m.design_mass >= total_weight_expr
    bf_m.design_mass >= total_weight_expr
    br_m.design_mass >= total_weight_expr
    st_m.design_mass >= total_weight_expr
    ti_m.design_mass >= total_weight_expr
    wh_m.design_mass >= total_weight_expr
    li_m.design_mass >= total_weight_expr

    # ---- Mission propagation ------------------------------------------
    body_m.target_passengers >= f_pass
    body_m.target_cargo_L    >= f_cargo
    fr_m.target_max_speed    >= f_speed
    rr_m.target_max_speed    >= f_speed
    bf_m.target_max_speed    >= f_speed
    br_m.target_max_speed    >= f_speed
    bf_m.target_decel_g      >= f_decel
    br_m.target_decel_g      >= f_decel
    ti_m.target_max_speed    >= f_speed

    hv_m.target_passengers   >= f_pass
    in_m.target_passengers   >= f_pass
    sf_m.target_passengers   >= f_pass
    sf_m.target_max_speed    >= f_speed

    # ---- Powertrain wiring --------------------------------------------
    # ---- Powertrain wiring --------------------------------------------
    # The motor's peak power and torque are set by the mission demand
    # (lambda constraints below). The battery voltage feeds the motor
    # and the on-board charger; the battery capacity is sized from the
    # target range and energy consumption, which closes the energy
    # cycle. The reducer and power electronics size to the motor's
    # rated peak power (a resource port, so it is legal on the RHS of a
    # demand constraint).
    mot_m.battery_voltage_V    >= bat_m.voltage_V
    obc_m.battery_voltage_V    >= bat_m.voltage_V
    red_m.input_peak_power_kW  >= mot_m.rated_peak_power_kW
    red_m.input_peak_torque_Nm >= mot_m.peak_torque_Nm
    pe_m.peak_motor_power_kW   >= mot_m.rated_peak_power_kW

    # Battery thermal management is sized by the battery energy capacity
    # (thermal mass to hold in range) and the motor peak power (peak
    # heat load during hard acceleration and fast charging).
    btm_m.battery_energy_kWh   >= bat_m.energy_kWh
    sys.constrain("thermal.peak_motor_power_kW",
                  lambda x: peak_power_demand_kW(x))

    # Motor target_peak_power and torque set by mission demand.

    # ---- Power and consumption lambdas --------------------------------

    def total_mass_from_x(x):
        return DRIVER_MASS_KG + sum(x[f"{mod}.weight"] for mod in EV_MODULES)

    def peak_power_demand_kW(x):
        """Wheel power demand: max of top-speed road load and
        acceleration. Account for reducer + power electronics
        efficiencies on the motor side."""
        m_kg = total_mass_from_x(x)
        cd = x["body.cd"]; A = x["body.frontal_area"]; crr = x["tires.crr"]
        p_top = top_speed_power_kw(mission["target_max_speed"], m_kg, cd, A, crr)
        p_accel = acceleration_power_kw(mission["target_0_100_s"], m_kg)
        eta_dt = (max(x["reducer.efficiency"], 0.85)
                  * max(x["power_electronics.efficiency"], 0.85))
        return max(p_top, p_accel) / eta_dt

    def peak_torque_demand_Nm(x):
        """Motor torque, roughly 2 Nm/kW for typical EV traction motors."""
        return 2.0 * peak_power_demand_kW(x)

    sys.constrain("motor.target_peak_power_kW",  peak_power_demand_kW)
    sys.constrain("motor.target_peak_torque_Nm", peak_torque_demand_Nm)
    sys.constrain("power_electronics.peak_motor_power_kW",
                  peak_power_demand_kW)

    def ev_energy_kWh_per_100km(x):
        """EV cruise energy consumption at 100 km/h, in kWh / 100 km.

        Combines road load, accessories, drivetrain losses, and
        thermal-management overhead. kWh per 100 km is numerically
        equal to kW at a 100 km/h cruise speed (since 1 hour at
        100 km/h covers 100 km).
        """
        m_kg = total_mass_from_x(x)
        cd = x["body.cd"]; A = x["body.frontal_area"]; crr = x["tires.crr"]
        road_load_kW = cruise_road_load_kw(100.0, m_kg, cd, A, crr)
        # Accessories: HVAC + steering + safety + lights + thermal mgmt.
        accessory_kW = 0.001 * (x["hvac.electric_load"]
                                + x["steering.electric_load"]
                                + x["safety.electric_load"]
                                + x["lights.electric_load"]
                                + x["thermal.electric_load"])
        wheel_demand_kW = road_load_kW
        # Drivetrain: motor + reducer + power electronics
        motor_in_kW = (wheel_demand_kW
                       / max(x["reducer.efficiency"], 0.85)
                       / max(x["motor.peak_efficiency"], 0.85))
        battery_out_kW = motor_in_kW / max(x["power_electronics.efficiency"], 0.85)
        total_battery_kW = battery_out_kW + accessory_kW
        return total_battery_kW   # kWh per 100 km equals kW at 100 km/h

    # Battery sized to mission range × energy consumption.
    sys.constrain("hv_battery.target_range_km",
                  lambda x, r=mission["target_range_km"]: r)
    sys.constrain("hv_battery.energy_kWh_per_100km",
                  ev_energy_kWh_per_100km)

    # ---- Outer R aggregations -----------------------------------------
    sys.constrain("production_cost",
        lambda x: total_cost(x, EV_MODULES)
                  + assembly_overhead_usd(total_mass_from_x(x)))
    sys.constrain("curb_weight", total_mass_from_x)
    sys.constrain("fuel_consumption", lambda x: 0.0)   # no liquid fuel
    sys.constrain("energy_consumption", ev_energy_kWh_per_100km)
    sys.constrain("co2_per_km",
        lambda x: ev_energy_kWh_per_100km(x) * GRID_CO2_PER_KWH * 10.0)
    sys.constrain("maintenance_per_year", lambda x: (
        # Tires
        x["tires.cost"] * 15000.0 / max(x["tires.durability"], 1.0)
        # Brakes (regen reduces wear)
        + 0.15 * (x["brakes_front.cost"] + x["brakes_rear.cost"])
            * 15000.0 / max(min(x["brakes_front.durability"],
                                  x["brakes_rear.durability"]), 1.0)
        # No oil changes for an EV.
        # Baseline scheduled service much lower than ICE.
        + 150.0
        # Coolant / brake fluid / cabin filter etc.
        + 60.0
    ))
    sys.constrain("durability", lambda x: harmonic_mean([
        x["motor.durability"], x["reducer.durability"],
        x["body.durability"],  x["susp_front.durability"],
        x["susp_rear.durability"],
        x["hv_battery.durability"], x["power_electronics.durability"],
        x["thermal.durability"],
    ]))

    return sys.build()


# ===========================================================================
# Mission definitions
# ===========================================================================

# Four representative missions spanning the passenger-car market.
# Each is a dict of the outer F values + a human label.
MISSIONS = {
    "Urban Compact": {
        "target_passengers":  4,
        "target_cargo_L":     300,
        "target_max_speed":   130,
        "target_range_km":    600,
        "target_decel_g":     0.8,
        "target_0_100_s":     11.0,
    },
    "Family Daily": {
        "target_passengers":  5,
        "target_cargo_L":     500,
        "target_max_speed":   180,
        "target_range_km":    700,
        "target_decel_g":     0.9,
        "target_0_100_s":     9.0,
    },
    "Suburban Utility": {
        "target_passengers":  7,
        "target_cargo_L":     900,
        "target_max_speed":   170,
        "target_range_km":    800,
        "target_decel_g":     0.7,
        "target_0_100_s":     11.0,
    },
    "Performance": {
        "target_passengers":  4,
        "target_cargo_L":     200,
        "target_max_speed":   250,
        "target_range_km":    500,
        "target_decel_g":     1.0,
        "target_0_100_s":     5.5,
    },
}


# ===========================================================================
# Helper: pick body candidate set for a given mission
# ===========================================================================


def _eligible_bodies(mission: Mapping[str, float]) -> List[BodyFrame]:
    """Bodies that can physically accommodate the mission's passenger
    and cargo demands."""
    return [b for b in BODY_STYLES
            if b.max_passengers >= mission["target_passengers"]
            and b.max_cargo_L   >= mission["target_cargo_L"]]


def _eligible_tires(mission: Mapping[str, float]) -> List[Tires]:
    return [t for t in TIRES
            if t.max_speed_kmh >= mission["target_max_speed"]]


# ===========================================================================
# Per-architecture sweep functions
# ===========================================================================


def sweep_ice(mission: Mapping[str, float], *, verbose: bool = False
              ) -> List[Tuple[str, Dict[str, float]]]:
    """Solve every reasonable ICE catalogue combination for a mission.

    Returns a list of (label, antichain-point) tuples for feasible
    designs, ready for Pareto extraction.
    """
    results: List[Tuple[str, Dict[str, float]]] = []
    bodies = _eligible_bodies(mission)
    tires_ok = _eligible_tires(mission)
    if not tires_ok:
        tires_ok = TIRES   # fall back; per-design feasibility check
    perf = mission["target_0_100_s"] < 8.0   # performance flag
    for body in bodies:
        for engine in ICE_ENGINES:
            # Skip clearly mismatched: tiny engine on heavy SUV/truck
            if body.style_name in ("mid_suv", "pickup_truck") \
               and engine.peak_power_kW < 130:
                continue
            for fi_kind in ("none", "single_turbo"):
                # Twin-turbo only on the largest engine
                if engine.peak_power_kW < 200 and fi_kind == "twin_turbo":
                    continue
                for trans in ICE_TRANSMISSIONS:
                    # CVT not for performance trims
                    if perf and trans.name == "CVT":
                        continue
                    for tires in tires_ok:
                        for susp in ("economy", "comfort", "sport"):
                            try:
                                dp = build_ice_car(
                                    mission=mission, body=body,
                                    engine=engine,
                                    forced_induction=ForcedInduction(kind=fi_kind),
                                    transmission=trans,
                                    suspension_type=susp,
                                    tire_choice=tires,
                                    wheel_choice=WHEELS[1],
                                    steering_choice=STEERING_OPTIONS[1],
                                    drivetrain_layout=("rwd" if perf else "fwd"),
                                )
                                res = solve(dp, dict(mission),
                                             max_iter=80, verbose=0)
                            except Exception as e:
                                if verbose:
                                    print(f"  skip ICE {body.style_name}/"
                                          f"{engine.name}: {e}")
                                continue
                            if res.feasible and res.antichain.points:
                                pt = list(res.antichain.points)[0]
                                label = (f"ICE/{body.style_name}/"
                                         f"{engine.name}/{trans.name}/"
                                         f"{susp}/{tires.name}")
                                results.append((label, dict(pt)))
    return results


def sweep_hev(mission: Mapping[str, float], *, verbose: bool = False
              ) -> List[Tuple[str, Dict[str, float]]]:
    results: List[Tuple[str, Dict[str, float]]] = []
    bodies = _eligible_bodies(mission)
    tires_ok = _eligible_tires(mission) or TIRES
    perf = mission["target_0_100_s"] < 8.0
    motor_options = [60, 80, 110, 140] if not perf else [110, 140, 180]
    for body in bodies:
        for engine in HYBRID_ENGINES:
            for motor_kW in motor_options:
                for chem in ("lithium_NMC",):  # NiMH omitted: Li-NMC dominates modern strong hybrids
                    for tires in tires_ok:
                        for susp in ("economy", "comfort"):
                            try:
                                dp = build_hybrid_car(
                                    mission=mission, body=body,
                                    engine=engine,
                                    motor_peak_power_kW=motor_kW,
                                    hv_battery_chemistry=chem,
                                    power_electronics_sic=True,
                                    suspension_type=susp,
                                    tire_choice=tires,
                                    wheel_choice=WHEELS[1],
                                    steering_choice=STEERING_OPTIONS[1],
                                    drivetrain_layout="fwd",
                                )
                                res = solve(dp, dict(mission),
                                             max_iter=120, verbose=0)
                            except Exception as e:
                                if verbose:
                                    print(f"  skip HEV {body.style_name}/"
                                          f"{engine.name}/{motor_kW}: {e}")
                                continue
                            if res.feasible and res.antichain.points:
                                pt = list(res.antichain.points)[0]
                                label = (f"HEV/{body.style_name}/"
                                         f"{engine.name}/{motor_kW}kW/"
                                         f"{chem}/{susp}/{tires.name}")
                                results.append((label, dict(pt)))
    return results


def sweep_ev(mission: Mapping[str, float], *, verbose: bool = False
             ) -> List[Tuple[str, Dict[str, float]]]:
    results: List[Tuple[str, Dict[str, float]]] = []
    bodies = _eligible_bodies(mission)
    tires_ok = _eligible_tires(mission) or TIRES
    perf = mission["target_0_100_s"] < 8.0
    for body in bodies:
        for motor in EV_MOTORS:
            if perf and motor.specific_power_kW_per_kg < 2.0:
                continue
            for battery in EV_BATTERIES:
                # Skip tiny battery + long range combos that obviously fail
                if mission["target_range_km"] > 600 and "40kWh" in battery.name:
                    continue
                for tires in tires_ok:
                    for susp in ("comfort",):
                        try:
                            dp = build_ev_car(
                                mission=mission, body=body,
                                motor=motor, battery=battery,
                                power_electronics_sic=True,
                                suspension_type=susp,
                                tire_choice=tires,
                                wheel_choice=WHEELS[1],
                                steering_choice=STEERING_OPTIONS[1],
                                drivetrain_layout=("rwd" if perf else "fwd"),
                            )
                            res = solve(dp, dict(mission),
                                         max_iter=200, verbose=0)
                        except Exception as e:
                            if verbose:
                                print(f"  skip EV {body.style_name}/"
                                      f"{motor.name}/{battery.name}: {e}")
                            continue
                        if res.feasible and res.antichain.points:
                            pt = list(res.antichain.points)[0]
                            label = (f"EV/{body.style_name}/"
                                     f"{motor.name}/{battery.name}/"
                                     f"{susp}/{tires.name}")
                            results.append((label, dict(pt)))
    return results


# ===========================================================================
# Pareto extraction across architectures
# ===========================================================================


def _is_dominated(p: Dict[str, float], others: Sequence[Dict[str, float]],
                  axes: Sequence[str]) -> bool:
    for q in others:
        if q is p:
            continue
        if all(q[k] <= p[k] for k in axes) \
           and any(q[k] < p[k] for k in axes):
            return True
    return False


def extract_pareto(results: List[Tuple[str, Dict[str, float]]],
                   axes: Sequence[str] = ("production_cost",
                                          "curb_weight",
                                          "co2_per_km")
                   ) -> List[Tuple[str, Dict[str, float]]]:
    points = [p for _, p in results]
    return [(label, p) for label, p in results
            if not _is_dominated(p, points, axes)]


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 78)
    print("Example 17: Full-vehicle co-design (ICE / Hybrid / EV)")
    print("=" * 78)

    summary: Dict[str, Dict[str, Tuple[int, int, Optional[Dict[str, float]]]]] = {}

    for mission_name, mission in MISSIONS.items():
        print(f"\n--- Mission: {mission_name} ---")
        print(f"  passengers={mission['target_passengers']}, "
              f"cargo={mission['target_cargo_L']} L, "
              f"max speed={mission['target_max_speed']} km/h, "
              f"range={mission['target_range_km']} km, "
              f"0-100={mission['target_0_100_s']} s")

        mission_results = {}
        for arch_name, sweep_fn in (("ICE", sweep_ice),
                                     ("HEV", sweep_hev),
                                     ("EV",  sweep_ev)):
            res = sweep_fn(mission)
            feas = len(res)
            best = None
            if res:
                # Best by production_cost
                best = min(res, key=lambda r: r[1]["production_cost"])
            mission_results[arch_name] = (feas, best)

        summary[mission_name] = mission_results

        # Print a compact table of the cheapest feasible per architecture.
        print(f"  {'Arch':<5} {'Feasible':>10} {'Best cost':>12} "
              f"{'Weight':>9} {'Fuel/Energy':>13} {'CO2':>8}")
        for arch_name in ("ICE", "HEV", "EV"):
            feas, best = mission_results[arch_name]
            if best is None:
                print(f"  {arch_name:<5} {feas:>10} {'-':>12} {'-':>9} {'-':>13} {'-':>8}")
                continue
            label, pt = best
            cost = pt["production_cost"]
            wt = pt["curb_weight"]
            co2 = pt["co2_per_km"]
            if arch_name == "EV":
                fuel_or_energy = f"{pt.get('energy_consumption', 0):.1f} kWh"
            else:
                fuel_or_energy = f"{pt.get('fuel_consumption', 0):.1f} L"
            print(f"  {arch_name:<5} {feas:>10} ${cost:>10,.0f} "
                  f"{wt:>7.0f} kg {fuel_or_energy:>13} {co2:>5.0f} g/km")

    # Print global summary
    print("\n" + "=" * 78)
    print("Summary: cheapest feasible per (mission, architecture)")
    print("=" * 78)
    print(f"  {'Mission':<22} {'Arch':<5} {'Cost':>10} {'Weight':>9} "
          f"{'CO2':>8} {'10y TCO':>9}")
    for mission_name, arch_res in summary.items():
        for arch_name, (feas, best) in arch_res.items():
            if best is None:
                continue
            _, pt = best
            # Crude 10-year TCO: production + 10*maintenance + 10*energy_cost
            # Energy: 15000 km/yr * (fuel * 1.8 USD/L OR energy * 0.25 USD/kWh)
            if arch_name == "EV":
                annual_energy_cost = 15000 / 100 * pt.get("energy_consumption", 0) * 0.25
            else:
                annual_energy_cost = 15000 / 100 * pt.get("fuel_consumption", 0) * 1.8
            tco10 = (pt["production_cost"]
                     + 10 * pt["maintenance_per_year"]
                     + 10 * annual_energy_cost)
            print(f"  {mission_name:<22} {arch_name:<5} "
                  f"${pt['production_cost']:>8,.0f} {pt['curb_weight']:>7.0f}kg "
                  f"{pt['co2_per_km']:>5.0f}g/km ${tco10:>7,.0f}")

    return summary


if __name__ == "__main__":
    main()
