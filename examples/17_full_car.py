"""
Example 17: Full-car co-design across ICE, hybrid, and battery-electric powertrains.

A granular decomposition of a passenger vehicle into 22 subsystems
spread across three groups (powertrain, chassis, interior+aux), with
a 12-entry architecture catalog covering pure-ICE, mild hybrid, full
hybrid, plug-in hybrid, range-extender EV, and battery-electric
configurations. The MCDP framework resolves the classic automotive
"weight death spiral" cycle (vehicle mass drives required propulsion
power, which drives engine / motor / battery / fuel tank mass, which
contributes back to vehicle mass) by Kleene iteration on the
monotonic constraint network.

Subsystems
----------

POWERTRAIN (12):
    engine, air_intake, exhaust_system, engine_cooling,
    engine_lubrication, fuel_system, transmission, final_drive,
    electric_motor, battery_pack, power_electronics, battery_cooling

CHASSIS (7):
    body_shell, front_suspension, rear_suspension, front_brakes,
    rear_brakes, steering, wheels_tires

INTERIOR / AUX (3):
    seats, hvac, infotainment_safety

Outer mission (functionality)
-----------------------------

    passenger_capacity  (people)
    cargo_volume        (L)
    max_speed           (km/h)
    range               (km, single tank or charge)
    accel_0_100         (s, target 0-100 km/h time)
    max_payload         (kg, passengers + cargo + occasional)

Outer resources
---------------

    production_cost            (USD)
    curb_weight                (kg)
    energy_per_100km           (kWh, unified ICE + EV metric)
    co2_per_km                 (g/km, tailpipe only)
    maintenance_per_year       (USD)
    durability_km              (km until major overhaul)

Parameter calibration
---------------------

All component-level numbers are drawn from current automotive
engineering practice and published references (Genta, "Motor
Vehicle Dynamics"; SAE technical papers; EPA fuel-economy data;
Bosch Automotive Handbook; ICCT vehicle cost study; 2024-2025 OEM
disclosures). Values are within published ranges for the relevant
technology level but are not production-precision for any specific
vehicle. The framework is what's being validated; the numbers serve
the framework.

Architectures
-------------

The architecture catalog has 12 entries covering the modern
passenger-vehicle technology spectrum. Each entry pre-selects
discrete catalog choices (engine, transmission, motor topology,
battery strategy) and lets the parametric modules size everything
else from mission demand via the Kleene cycle.

References
----------

[1] G. Genta and L. Morello, "The Automotive Chassis", Springer.
[2] H. Heisler, "Advanced Vehicle Technology", Butterworth-Heinemann.
[3] Bosch, "Automotive Handbook", 11th ed., 2022.
[4] ICCT, "Electric vehicle cost evolution to 2030", 2024.
[5] BNEF, "Lithium-ion battery price survey", 2024 ($115/kWh pack).
[6] EPA, "Fuel Economy Guide", 2024 model year data.
[7] SAE J1349 for engine power rating conventions.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from codesign import (
    System,
    Reals,
    Ports,
    AlgebraicDP,
    CatalogDP,
    CatalogEntry,
    Module,
    solve,
    sqrt,
)


# ===========================================================================
# Physical constants and global calibration
# ===========================================================================

G                       = 9.81       # m/s^2
RHO_AIR                 = 1.225      # kg/m^3, sea level, 15 C
LHV_GASOLINE_KWH_PER_L  = 8.95       # 32.2 MJ/L
LHV_DIESEL_KWH_PER_L    = 9.94       # 35.8 MJ/L
DENSITY_GASOLINE        = 0.745      # kg/L
DENSITY_DIESEL          = 0.832      # kg/L
CO2_GASOLINE_KG_PER_L   = 2.31       # WTW tailpipe-only
CO2_DIESEL_KG_PER_L     = 2.68

DRIVER_MASS             = 75.0       # kg, included in curb mass
PASSENGER_MASS          = 75.0       # kg, occupant mass beyond driver
LUGGAGE_PER_PERSON_KG   = 20.0       # kg, baseline luggage estimate
ANNUAL_KM               = 15_000.0   # for maintenance amortisation

# Cruise/cycle reference speeds used in energy-consumption equations.
# Real WLTP cycles are weighted across low/medium/high/extra-high
# segments; we use a single representative cruise speed and scale.
CRUISE_SPEED_KMH        = 90.0       # representative WLTP-ish cruise
HIGHWAY_SPEED_KMH       = 120.0      # secondary reference for max speed

# Auxiliary loads (kW): always-on, regardless of powertrain.
ACCESSORY_LOAD_KW       = 0.6        # cabin electronics, lights, ECUs


# ===========================================================================
# SECTION 1: POWERTRAIN SUBSYSTEMS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.1 Engine catalog
# ---------------------------------------------------------------------------
# Eight combustion-engine archetypes spanning today's passenger-vehicle
# product space. Peak power and efficiency are nameplate values at the
# rated speed; durability is the typical mileage before major overhaul
# (rebuild, head gasket, valve job). The "none" entry is a placeholder
# for pure-EV architectures; it carries a small structural mass and
# cost so the System sees a real Antichain even when the engine is
# absent.

# Fields: (name, displacement_L, peak_power_kW, peak_torque_Nm,
#          fuel_type, peak_eff, weight_kg, cost_USD, durability_km)
ENGINE_CATALOG = [
    ("none_ev",        0.0,    0.0,   0.0,  "none",   1.00,  10.0,   100,  1_000_000),
    ("1.0L_3cyl_T",    1.0,   85.0, 200.0,  "gas",    0.32, 105.0,  2200,    250_000),
    ("1.5L_4cyl_NA",   1.5,   88.0, 150.0,  "gas",    0.30, 110.0,  1800,    220_000),
    ("1.5L_4cyl_T",    1.5,  130.0, 250.0,  "gas",    0.34, 135.0,  2800,    220_000),
    ("2.0L_4cyl_T",    2.0,  180.0, 350.0,  "gas",    0.35, 165.0,  3500,    200_000),
    ("2.0L_4cyl_TD",   2.0,  140.0, 400.0,  "diesel", 0.40, 175.0,  4200,    320_000),
    ("3.0L_V6_T",      3.0,  280.0, 450.0,  "gas",    0.34, 210.0,  5800,    200_000),
    ("1.0L_REX",       1.0,   50.0,  90.0,  "gas",    0.36,  75.0,  1800,    300_000),
]

ENGINE_F = Ports({
    "ice_power_demand_kW": Reals(unit="kW"),
})
ENGINE_R = Ports({
    "engine_weight_kg":    Reals(unit="kg"),
    "engine_cost_USD":     Reals(unit="USD"),
    "engine_displacement_L": Reals(unit="L"),
    "engine_efficiency":   Reals(unit=""),
    "engine_heat_kW":      Reals(unit="kW"),
    "engine_durability_km": Reals(unit="km"),
    "fuel_type_co2":       Reals(unit="kgCO2/L"),
    "fuel_type_density":   Reals(unit="kg/L"),
    "fuel_type_LHV":       Reals(unit="kWh/L"),
})


def make_engine_dp():
    entries = []
    for (name, disp, pkw, ptq, ftype, eff, wt, cost, dur) in ENGINE_CATALOG:
        # Heat rejection at peak power: P_heat = P_mech * (1/eff - 1).
        # The "none_ev" engine produces zero heat.
        if eff < 1.0:
            heat_kw = pkw * (1.0 / eff - 1.0)
        else:
            heat_kw = 0.0
        if ftype == "gas":
            co2, dens, lhv = CO2_GASOLINE_KG_PER_L, DENSITY_GASOLINE, LHV_GASOLINE_KWH_PER_L
        elif ftype == "diesel":
            co2, dens, lhv = CO2_DIESEL_KG_PER_L, DENSITY_DIESEL, LHV_DIESEL_KWH_PER_L
        else:
            co2, dens, lhv = 0.0, 0.0, 1.0  # avoid div-by-zero downstream
        entries.append(CatalogEntry(
            provides={"ice_power_demand_kW": pkw},   # max F it can satisfy
            costs={
                "engine_weight_kg":     wt,
                "engine_cost_USD":      cost,
                "engine_displacement_L": disp,
                "engine_efficiency":    eff,
                "engine_heat_kW":       heat_kw,
                "engine_durability_km": dur,
                "fuel_type_co2":        co2,
                "fuel_type_density":    dens,
                "fuel_type_LHV":        lhv,
            },
            name=name,
        ))
    return CatalogDP(F=ENGINE_F, R=ENGINE_R, catalog=entries,
                     name="engine_catalog")


# ---------------------------------------------------------------------------
# 1.2 Air intake (parametric: filter + turbo (if any) + intercooler + manifold)
# ---------------------------------------------------------------------------
# Sized to engine peak power and a boost-pressure parameter (passed in
# at construction). NA engines have boost=1.0 (no turbo); turbocharged
# engines run boost = 1.5-2.5. Weight and cost scale roughly linearly
# with airflow capacity, which scales with peak power.


class AirIntake(Module):
    """Air filter + turbo (if any) + intercooler + intake manifold.

    Weight and cost scale with peak engine power (proxy for required
    airflow). The ``boost_pressure`` parameter at construction time
    selects the turbo class: 1.0 (NA, light), 1.5 (mild boost), 2.0
    (heavy boost). The "none" engine yields zero weight and cost.
    """
    F = Ports({"engine_peak_power_kW": Reals(unit="kW")})
    R = Ports({
        "intake_weight_kg":     Reals(unit="kg"),
        "intake_cost_USD":      Reals(unit="USD"),
        "intake_durability_km": Reals(unit="km"),
    })

    def __init__(self, boost_pressure: float = 1.0):
        self.boost = float(boost_pressure)

    def h(self, f):
        p_kw = max(0.0, float(f["engine_peak_power_kW"]))
        boost_extra = max(0.0, self.boost - 1.0)
        if p_kw < 1.0:
            wt, cost, dur = 0.5, 20.0, 1_000_000.0
        else:
            wt   = 4.0 + p_kw * (0.03 + 0.04 * boost_extra)
            cost = 80.0 + p_kw * (12.0 + 18.0 * boost_extra)
            dur  = 250_000.0 - 40_000.0 * boost_extra
        return {
            "intake_weight_kg":     wt,
            "intake_cost_USD":      cost,
            "intake_durability_km": max(150_000.0, dur),
        }


class ExhaustSystem(Module):
    """Exhaust system sized to displacement. Diesel adds DPF+SCR overhead."""
    F = Ports({"displacement_L": Reals(unit="L")})
    R = Ports({
        "exhaust_weight_kg":     Reals(unit="kg"),
        "exhaust_cost_USD":      Reals(unit="USD"),
        "exhaust_durability_km": Reals(unit="km"),
    })

    def __init__(self, is_diesel: bool = False):
        self.diesel_factor = 1.4 if is_diesel else 1.0

    def h(self, f):
        d = max(0.0, float(f["displacement_L"]))
        if d < 0.05:
            return {"exhaust_weight_kg": 0.0,
                    "exhaust_cost_USD":  0.0,
                    "exhaust_durability_km": 1_000_000.0}
        wt   = (8.0 + 6.0 * d) * self.diesel_factor
        cost = (180.0 + 120.0 * d) * self.diesel_factor
        dur  = 180_000.0 if self.diesel_factor == 1.0 else 240_000.0
        return {"exhaust_weight_kg": wt,
                "exhaust_cost_USD":  cost,
                "exhaust_durability_km": dur}


class EngineCooling(Module):
    """Engine cooling system sized to heat rejection."""
    F = Ports({"heat_rejection_kW": Reals(unit="kW")})
    R = Ports({
        "engine_cool_weight_kg":     Reals(unit="kg"),
        "engine_cool_cost_USD":      Reals(unit="USD"),
        "engine_cool_aux_kW":        Reals(unit="kW"),
        "engine_cool_durability_km": Reals(unit="km"),
    })

    def __init__(self, safety_factor: float = 1.3):
        self.sf = float(safety_factor)

    def h(self, f):
        q = max(0.0, float(f["heat_rejection_kW"]))
        capacity = q * self.sf
        if capacity < 0.5:
            return {
                "engine_cool_weight_kg": 1.0,
                "engine_cool_cost_USD":  20.0,
                "engine_cool_aux_kW":    0.05,
                "engine_cool_durability_km": 500_000.0,
            }
        # Calibration: a typical 150 kW gasoline car (~280 kW heat capacity
        # after safety factor) ships with a ~12 kg cooling pack costing
        # ~$350. That puts coefficients at ~0.03 kg/kW + 4 kg base, and
        # $1.0/kW + $80 base.
        wt   = 4.0 + 0.03 * capacity
        cost = 80.0 + 1.0 * capacity
        aux  = 0.25 + 0.005 * capacity
        return {
            "engine_cool_weight_kg": wt,
            "engine_cool_cost_USD":  cost,
            "engine_cool_aux_kW":    aux,
            "engine_cool_durability_km": 200_000.0,
        }


class EngineLubrication(Module):
    """Engine lubrication: oil pan, pump, filter, optional cooler."""
    F = Ports({"displacement_L": Reals(unit="L")})
    R = Ports({
        "lube_weight_kg":     Reals(unit="kg"),
        "lube_cost_USD":      Reals(unit="USD"),
        "lube_durability_km": Reals(unit="km"),
    })

    def h(self, f):
        d = max(0.0, float(f["displacement_L"]))
        if d < 0.05:
            return {"lube_weight_kg": 0.0, "lube_cost_USD": 0.0,
                    "lube_durability_km": 1_000_000.0}
        oil_l = 2.0 + 1.5 * d
        wt    = oil_l * 0.9 + 5.0 + 2.0 * d
        cost  = 50.0 + 35.0 * d
        return {"lube_weight_kg": wt,
                "lube_cost_USD":  cost,
                "lube_durability_km": 220_000.0}


class FuelSystem(Module):
    """Fuel storage and delivery sized to range and consumption.

    Includes tank, fuel pump, injectors, lines. Carries average fuel
    mass (90% full) into curb weight per SAE convention.
    """
    F = Ports({
        "target_range_km":  Reals(unit="km"),
        "fuel_per_100km_L": Reals(unit="L/100km"),
        "fuel_density":     Reals(unit="kg/L"),
    })
    R = Ports({
        "fuel_weight_kg":     Reals(unit="kg"),
        "fuel_cost_USD":      Reals(unit="USD"),
        "max_fuel_L":         Reals(unit="L"),
        "fuel_durability_km": Reals(unit="km"),
    })

    def h(self, f):
        rng    = max(0.0, float(f["target_range_km"]))
        cons   = max(0.0, float(f["fuel_per_100km_L"]))
        dens   = max(0.0, float(f["fuel_density"]))
        max_l  = rng * cons / 100.0
        if max_l < 0.5:
            return {
                "fuel_weight_kg":     0.0,
                "fuel_cost_USD":      0.0,
                "max_fuel_L":         0.0,
                "fuel_durability_km": 1_000_000.0,
            }
        avg_fuel_kg = 0.9 * max_l * dens
        hw_kg   = 4.0 + 0.5 * max_l
        cost    = 250.0 + 8.0 * max_l + 200.0
        return {
            "fuel_weight_kg":     hw_kg + avg_fuel_kg,
            "fuel_cost_USD":      cost,
            "max_fuel_L":         max_l,
            "fuel_durability_km": 300_000.0,
        }


# ---------------------------------------------------------------------------
# 1.7 Transmission catalog
# ---------------------------------------------------------------------------
# Five archetypes: 6-speed manual, 6-speed auto, 8-speed auto, CVT,
# and single-speed reducer (used by EVs and Toyota-style HSD).

# Fields: (name, ratio_range, peak_torque_Nm, efficiency,
#          weight_kg, cost_USD, durability_km, requires_clutch)
TRANSMISSION_CATALOG = [
    ("6spd_manual",     5.5,  400.0, 0.97,  45.0, 1200,  280_000, True),
    ("6spd_auto",       5.0,  450.0, 0.92,  75.0, 2200,  220_000, False),
    ("8spd_auto",       7.0,  500.0, 0.94,  85.0, 3200,  240_000, False),
    ("CVT",             6.0,  300.0, 0.90,  60.0, 1900,  180_000, False),
    ("single_speed",    1.0,  600.0, 0.97,  25.0,  650,  500_000, False),
]

TRANSMISSION_F = Ports({
    "trans_peak_torque_Nm": Reals(unit="Nm"),
})
TRANSMISSION_R = Ports({
    "trans_weight_kg":       Reals(unit="kg"),
    "trans_cost_USD":        Reals(unit="USD"),
    "trans_efficiency":      Reals(unit=""),
    "trans_durability_km":   Reals(unit="km"),
})


def make_transmission_dp():
    entries = []
    for (name, ratio, ptq, eff, wt, cost, dur, _clutch) in TRANSMISSION_CATALOG:
        entries.append(CatalogEntry(
            provides={"trans_peak_torque_Nm": ptq},
            costs={
                "trans_weight_kg":     wt,
                "trans_cost_USD":      cost,
                "trans_efficiency":    eff,
                "trans_durability_km": dur,
            },
            name=name,
        ))
    return CatalogDP(F=TRANSMISSION_F, R=TRANSMISSION_R, catalog=entries,
                     name="transmission_catalog")


# ---------------------------------------------------------------------------
# 1.8 Final drive (parametric: differential + driveshafts + CV joints)
# ---------------------------------------------------------------------------

class FinalDrive(Module):
    """Differential + driveshafts + CV joints, sized to peak wheel torque.

    Wheel torque = transmission_output_torque * final_drive_ratio. We
    take peak_axle_torque as the F and produce hardware sized to
    survive it. AWD configurations get a 1.6x multiplier on weight
    and cost to account for the second axle's hardware.
    """
    F = Ports({"peak_axle_torque_Nm": Reals(unit="Nm")})
    R = Ports({
        "final_drive_weight_kg":     Reals(unit="kg"),
        "final_drive_cost_USD":      Reals(unit="USD"),
        "final_drive_durability_km": Reals(unit="km"),
    })

    def __init__(self, awd: bool = False):
        self.awd_factor = 1.6 if awd else 1.0

    def h(self, f):
        tq = max(0.0, float(f["peak_axle_torque_Nm"]))
        # Calibration: 1500 Nm peak axle torque (decent FWD) -> ~35 kg, ~$550
        wt   = (10.0 + 0.017 * tq) * self.awd_factor
        cost = (200.0 + 0.23  * tq) * self.awd_factor
        # Differential and CV joints are designed for life of the vehicle
        # in most cases; we discount durability with stress.
        dur  = max(180_000.0, 350_000.0 - 0.05 * tq)
        return {
            "final_drive_weight_kg":     wt,
            "final_drive_cost_USD":      cost,
            "final_drive_durability_km": dur,
        }


# ---------------------------------------------------------------------------
# 1.9 Electric motor catalog
# ---------------------------------------------------------------------------
# Six topology classes covering everything from "no motor" through
# dual-motor performance EV.

# Fields: (name, total_peak_power_kW, total_peak_torque_Nm,
#          weight_kg, cost_USD, efficiency, durability_km)
EMOTOR_CATALOG = [
    ("none_ice",      0.0,    0.0,    1.0,    50, 1.00, 1_000_000),
    ("isg_48V",      12.0,   60.0,   12.0,   650, 0.90,   400_000),
    ("front_small", 100.0,  280.0,   38.0,  1600, 0.94,   500_000),
    ("front_med",   150.0,  340.0,   48.0,  2200, 0.94,   500_000),
    ("front_large", 200.0,  410.0,   62.0,  2900, 0.93,   450_000),
    ("dual_perf",   400.0,  720.0,  118.0,  5800, 0.93,   400_000),
]

EMOTOR_F = Ports({
    "motor_power_demand_kW":  Reals(unit="kW"),
    "motor_torque_demand_Nm": Reals(unit="Nm"),
})
EMOTOR_R = Ports({
    "motor_weight_kg":     Reals(unit="kg"),
    "motor_cost_USD":      Reals(unit="USD"),
    "motor_efficiency":    Reals(unit=""),
    "motor_durability_km": Reals(unit="km"),
    "motor_peak_power_kW": Reals(unit="kW"),
    "motor_heat_kW":       Reals(unit="kW"),
})


def make_emotor_dp():
    entries = []
    for (name, pkw, ptq, wt, cost, eff, dur) in EMOTOR_CATALOG:
        # Heat at peak: motor loss is (1 - eff) * P_peak
        heat = pkw * (1.0 - eff) if eff < 1.0 else 0.0
        entries.append(CatalogEntry(
            provides={
                "motor_power_demand_kW":  pkw,
                "motor_torque_demand_Nm": ptq,
            },
            costs={
                "motor_weight_kg":     wt,
                "motor_cost_USD":      cost,
                "motor_efficiency":    eff,
                "motor_durability_km": dur,
                "motor_peak_power_kW": pkw,
                "motor_heat_kW":       heat,
            },
            name=name,
        ))
    return CatalogDP(F=EMOTOR_F, R=EMOTOR_R, catalog=entries,
                     name="emotor_catalog")


# ---------------------------------------------------------------------------
# 1.10 Battery pack (parametric: cells + BMS + housing)
# ---------------------------------------------------------------------------

class BatteryPack(Module):
    """High-voltage battery pack sized to range and energy demand.

    F: target_battery_kWh   (energy capacity needed)
       motor_peak_power_kW  (for C-rate sizing; pack must be able to
                             discharge at peak motor power)

    R: battery_weight_kg
       battery_cost_USD
       battery_volume_L
       battery_durability_km   (typically ~1500 cycles * useful range)

    Calibration follows BNEF 2024 cost survey ($115/kWh pack-level)
    and current production cell-pack specific energy (150 Wh/kg
    pack-level for NMC, ~110 Wh/kg for LFP). We use a blended
    130 Wh/kg / $130/kWh as a representative 2025 baseline.
    """
    F = Ports({
        "target_battery_kWh":  Reals(unit="kWh"),
        "motor_peak_power_kW": Reals(unit="kW"),
    })
    R = Ports({
        "battery_weight_kg":     Reals(unit="kg"),
        "battery_cost_USD":      Reals(unit="USD"),
        "battery_volume_L":      Reals(unit="L"),
        "battery_durability_km": Reals(unit="km"),
        "battery_capacity_kWh":  Reals(unit="kWh"),
    })

    def __init__(self, specific_energy_Wh_per_kg: float = 130.0,
                 cost_USD_per_kWh: float = 130.0,
                 max_c_rate: float = 3.0):
        self.spec_energy = float(specific_energy_Wh_per_kg)
        self.cost_per_kwh = float(cost_USD_per_kWh)
        self.max_c = float(max_c_rate)

    def h(self, f):
        kwh    = max(0.0, float(f["target_battery_kWh"]))
        p_peak = max(0.0, float(f["motor_peak_power_kW"]))
        # C-rate constraint: pack must be able to sustain peak discharge
        # for at least 30s. Size up if necessary.
        kwh_for_c_rate = p_peak / self.max_c
        kwh_final = max(kwh, kwh_for_c_rate)
        if kwh_final < 0.05:
            return {
                "battery_weight_kg":     0.0,
                "battery_cost_USD":      0.0,
                "battery_volume_L":      0.0,
                "battery_durability_km": 1_000_000.0,
                "battery_capacity_kWh":  0.0,
            }
        wt   = kwh_final * 1000.0 / self.spec_energy
        cost = kwh_final * self.cost_per_kwh + 800.0   # +$800 BMS/housing
        vol  = kwh_final * 4.5                          # ~4.5 L/kWh pack-level
        # Durability: 1500 EOL cycles * 80% usable * pack kWh / consumption.
        # We express in km by assuming a typical 18 kWh/100km consumption.
        cycles = 1500.0
        usable = 0.8
        dur_km = cycles * usable * kwh_final / 0.18  # 0.18 kWh/km baseline
        return {
            "battery_weight_kg":     wt + 25.0,        # +25 kg housing
            "battery_cost_USD":      cost,
            "battery_volume_L":      vol,
            "battery_durability_km": max(150_000.0, dur_km),
            "battery_capacity_kWh":  kwh_final,
        }


# ---------------------------------------------------------------------------
# 1.11 Power electronics (parametric: inverter + DC-DC + onboard charger)
# ---------------------------------------------------------------------------

class PowerElectronics(Module):
    """Inverter + DC-DC converter + onboard charger.

    Sized to peak motor power. Inverter dominates mass/cost; OBC adds
    a fixed overhead (3-22 kW AC charging hardware).
    """
    F = Ports({"motor_peak_power_kW": Reals(unit="kW")})
    R = Ports({
        "pe_weight_kg":     Reals(unit="kg"),
        "pe_cost_USD":      Reals(unit="USD"),
        "pe_durability_km": Reals(unit="km"),
        "pe_efficiency":    Reals(unit=""),
    })

    def __init__(self, charger_kW: float = 0.0):
        self.charger = float(charger_kW)

    def h(self, f):
        p = max(0.0, float(f["motor_peak_power_kW"]))
        if p < 1.0 and self.charger < 0.1:
            return {
                "pe_weight_kg":     0.5, "pe_cost_USD": 50.0,
                "pe_durability_km": 1_000_000.0, "pe_efficiency": 1.0,
            }
        # Inverter: ~0.05 kg/kW + 6 kg base, $15/kW + $200 base
        inv_wt   = 6.0 + 0.05 * p
        inv_cost = 200.0 + 15.0 * p
        # Onboard charger: ~1.5 kg/kW + 3 kg, $80/kW + $100
        chg_wt   = 3.0 + 1.5 * self.charger if self.charger > 0 else 0.0
        chg_cost = 100.0 + 80.0 * self.charger if self.charger > 0 else 0.0
        # DC-DC for 12V auxiliary: ~3 kg, $150 (always present if motor present)
        dcdc_wt   = 3.0
        dcdc_cost = 150.0
        return {
            "pe_weight_kg":     inv_wt + chg_wt + dcdc_wt,
            "pe_cost_USD":      inv_cost + chg_cost + dcdc_cost,
            "pe_durability_km": 300_000.0,
            "pe_efficiency":    0.95,    # combined inverter + DC-DC + OBC
        }


# ---------------------------------------------------------------------------
# 1.12 Battery cooling (parametric: liquid loop sized to battery kWh)
# ---------------------------------------------------------------------------

class BatteryCooling(Module):
    """Battery thermal management: liquid coolant loop sized to pack kWh.

    Small batteries (< 5 kWh, MHEV/FHEV) get a small passive loop;
    larger batteries (PHEV/BEV) get a dedicated liquid system with
    chiller and PTC heater.
    """
    F = Ports({"battery_capacity_kWh": Reals(unit="kWh")})
    R = Ports({
        "bcool_weight_kg":     Reals(unit="kg"),
        "bcool_cost_USD":      Reals(unit="USD"),
        "bcool_aux_kW":        Reals(unit="kW"),
        "bcool_durability_km": Reals(unit="km"),
    })

    def h(self, f):
        kwh = max(0.0, float(f["battery_capacity_kWh"]))
        if kwh < 0.5:
            return {"bcool_weight_kg": 0.0, "bcool_cost_USD": 0.0,
                    "bcool_aux_kW": 0.0, "bcool_durability_km": 1_000_000.0}
        if kwh < 5.0:
            # Small pack: passive air or minimal loop
            return {"bcool_weight_kg": 2.0 + 0.5 * kwh,
                    "bcool_cost_USD":  80.0 + 20.0 * kwh,
                    "bcool_aux_kW":    0.1,
                    "bcool_durability_km": 250_000.0}
        # Larger pack: dedicated thermal management
        wt   = 8.0 + 0.4 * kwh           # ~0.4 kg/kWh + 8 kg base
        cost = 250.0 + 18.0 * kwh        # ~$18/kWh + $250 base
        aux  = 0.3 + 0.02 * kwh          # chiller load
        return {"bcool_weight_kg": wt,
                "bcool_cost_USD":  cost,
                "bcool_aux_kW":    aux,
                "bcool_durability_km": 250_000.0}


# ===========================================================================
# SECTION 2: CHASSIS SUBSYSTEMS
# ===========================================================================

# ---------------------------------------------------------------------------
# 2.1 Body shell catalog
# ---------------------------------------------------------------------------
# Five archetypes spanning the modern passenger product space. The
# body provides passenger and cargo capacity, sets aerodynamic
# performance (Cd, frontal area), and contributes the dominant
# structural mass (BIW + panels + closures + glass).

# Fields: (name, n_seats, cargo_L, weight_kg, cost_USD, Cd,
#          frontal_area_m2, durability_km)
BODY_CATALOG = [
    ("compact_hatch", 5,  370,  280.0,  4200, 0.31, 2.20, 350_000),
    ("sedan",         5,  500,  340.0,  5400, 0.28, 2.30, 350_000),
    ("SUV",           7,  900,  470.0,  7200, 0.34, 2.75, 350_000),
    ("pickup",        5, 1500,  560.0,  7800, 0.42, 3.10, 400_000),
    ("sport_coupe",   4,  280,  300.0,  9500, 0.26, 2.10, 250_000),
]

BODY_F = Ports({
    "passenger_capacity": Reals(unit=""),
    "cargo_volume_L":     Reals(unit="L"),
})
BODY_R = Ports({
    "body_weight_kg":     Reals(unit="kg"),
    "body_cost_USD":      Reals(unit="USD"),
    "drag_coefficient":   Reals(unit=""),
    "frontal_area_m2":    Reals(unit="m^2"),
    "body_durability_km": Reals(unit="km"),
})


def make_body_dp():
    entries = []
    for (name, n_seats, cargo, wt, cost, cd, fa, dur) in BODY_CATALOG:
        entries.append(CatalogEntry(
            provides={
                "passenger_capacity": n_seats,
                "cargo_volume_L":     cargo,
            },
            costs={
                "body_weight_kg":     wt,
                "body_cost_USD":      cost,
                "drag_coefficient":   cd,
                "frontal_area_m2":    fa,
                "body_durability_km": dur,
            },
            name=name,
        ))
    return CatalogDP(F=BODY_F, R=BODY_R, catalog=entries, name="body_catalog")


# ---------------------------------------------------------------------------
# 2.2 Suspension (front and rear, same catalog, separate modules)
# ---------------------------------------------------------------------------
# Springs + dampers + anti-roll bar + control arms + bushings, per axle.
# Four setups spanning economy, comfort, sport, off-road.

# Fields per axle: (name, max_axle_load_kg, weight_kg, cost_USD,
#                   comfort_index, durability_km)
SUSPENSION_CATALOG = [
    ("economy",     900,   42.0,   420,  0.5, 150_000),
    ("comfort",    1100,   55.0,   850,  0.9, 180_000),
    ("sport",      1200,   58.0,  1400,  0.4, 130_000),
    ("offroad",    1400,   78.0,  1200,  0.6, 220_000),
]

SUSP_F = Ports({"axle_load_kg": Reals(unit="kg")})
SUSP_R = Ports({
    "susp_weight_kg":     Reals(unit="kg"),
    "susp_cost_USD":      Reals(unit="USD"),
    "susp_comfort":       Reals(unit=""),
    "susp_durability_km": Reals(unit="km"),
})


def make_suspension_dp(name_suffix: str = ""):
    entries = []
    for (name, load, wt, cost, comf, dur) in SUSPENSION_CATALOG:
        entries.append(CatalogEntry(
            provides={"axle_load_kg": load},
            costs={
                "susp_weight_kg":     wt,
                "susp_cost_USD":      cost,
                "susp_comfort":       comf,
                "susp_durability_km": dur,
            },
            name=name,
        ))
    return CatalogDP(F=SUSP_F, R=SUSP_R, catalog=entries,
                     name=f"suspension{name_suffix}")


# ---------------------------------------------------------------------------
# 2.3 Brakes (parametric per axle, sized to mass and max speed)
# ---------------------------------------------------------------------------

class Brakes(Module):
    """Brake hardware sized to vehicle mass and max speed.

    Brake energy at full stop from max_speed scales as
    0.5 * mass * v_max^2. Rotor + caliper + pad mass scales with that
    energy, with the front axle carrying ~60% of braking load.
    """
    F = Ports({
        "vehicle_mass_kg":  Reals(unit="kg"),
        "max_speed_kmh":    Reals(unit="km/h"),
    })
    R = Ports({
        "brake_weight_kg":     Reals(unit="kg"),
        "brake_cost_USD":      Reals(unit="USD"),
        "brake_durability_km": Reals(unit="km"),
    })

    def __init__(self, axle: str = "front"):
        # Front carries ~60% of braking energy, rear ~40%.
        self.frac = 0.60 if axle == "front" else 0.40
        self.axle = axle

    def h(self, f):
        m   = max(1.0, float(f["vehicle_mass_kg"]))
        v   = max(50.0, float(f["max_speed_kmh"]))
        v_ms = v / 3.6
        # Energy per stop, kJ. 60/40 front/rear split.
        ke = 0.5 * m * v_ms ** 2 / 1000.0 * self.frac
        # Calibration: ke=1400 kJ per front axle (typical 1500kg/200kmh)
        # -> ~18 kg, ~$560. Coefficients tuned to that target.
        wt   = 4.0 + 0.010 * ke
        cost = 80.0 + 0.35 * ke
        dur  = max(60_000.0, 150_000.0 - 0.02 * ke)
        return {
            "brake_weight_kg":     wt,
            "brake_cost_USD":      cost,
            "brake_durability_km": dur,
        }


# ---------------------------------------------------------------------------
# 2.4 Steering (parametric: rack + EPS pump + tie rods + column)
# ---------------------------------------------------------------------------

class Steering(Module):
    """Steering rack + EPS + tie rods, sized to front axle load."""
    F = Ports({"front_axle_load_kg": Reals(unit="kg")})
    R = Ports({
        "steering_weight_kg":     Reals(unit="kg"),
        "steering_cost_USD":      Reals(unit="USD"),
        "steering_aux_kW":        Reals(unit="kW"),
        "steering_durability_km": Reals(unit="km"),
    })

    def h(self, f):
        load = max(100.0, float(f["front_axle_load_kg"]))
        # Calibration: 800 kg front axle (sedan) -> ~14 kg, ~$420, 0.15 kW
        wt   = 6.0 + 0.010 * load
        cost = 250.0 + 0.21 * load
        aux  = 0.05 + 0.00012 * load     # EPS draws ~50W idle + 0.12 W/kg
        return {
            "steering_weight_kg":     wt,
            "steering_cost_USD":      cost,
            "steering_aux_kW":        aux,
            "steering_durability_km": 250_000.0,
        }


# ---------------------------------------------------------------------------
# 2.5 Wheels and tires catalog
# ---------------------------------------------------------------------------
# Five tire-set archetypes. Rolling resistance and durability are the
# two parameters that drive the energy/maintenance tradeoff.

# Fields: (name, max_load_kg, max_speed_kmh, weight_kg, cost_USD,
#          rolling_resistance, grip_coef, durability_km)
TIRE_CATALOG = [
    ("eco_lrr",         750, 200,  68.0,   480, 0.0080, 0.85, 50_000),
    ("all_season",      900, 210,  78.0,   560, 0.0105, 0.95, 45_000),
    ("performance",     900, 280,  82.0,   980, 0.0130, 1.10, 30_000),
    ("all_terrain",    1100, 200,  98.0,   780, 0.0155, 1.00, 60_000),
    ("offroad",        1300, 180, 115.0,   920, 0.0180, 1.05, 70_000),
]

TIRES_F = Ports({
    "tire_load_kg":  Reals(unit="kg"),
    "tire_max_kmh":  Reals(unit="km/h"),
})
TIRES_R = Ports({
    "tires_weight_kg":     Reals(unit="kg"),
    "tires_cost_USD":      Reals(unit="USD"),
    "rolling_resistance":  Reals(unit=""),
    "tires_durability_km": Reals(unit="km"),
})


def make_tires_dp():
    entries = []
    for (name, load, vmax, wt, cost, crr, _grip, dur) in TIRE_CATALOG:
        entries.append(CatalogEntry(
            provides={"tire_load_kg": load, "tire_max_kmh": vmax},
            costs={
                "tires_weight_kg":     wt,
                "tires_cost_USD":      cost,
                "rolling_resistance":  crr,
                "tires_durability_km": dur,
            },
            name=name,
        ))
    return CatalogDP(F=TIRES_F, R=TIRES_R, catalog=entries,
                     name="tires_catalog")


# ===========================================================================
# SECTION 3: INTERIOR / AUX SUBSYSTEMS
# ===========================================================================

# ---------------------------------------------------------------------------
# 3.1 Seats (parametric: front + rear, sized to passenger count)
# ---------------------------------------------------------------------------

class Seats(Module):
    """Seat assemblies (frames + cushions + upholstery + tracks).

    Mass and cost scale with passenger capacity. The ``trim_level``
    parameter at construction time multiplies cost (1.0 = base cloth,
    1.8 = mid leather, 3.2 = premium with heating/ventilation/memory).
    """
    F = Ports({"passenger_capacity": Reals(unit="")})
    R = Ports({
        "seats_weight_kg":     Reals(unit="kg"),
        "seats_cost_USD":      Reals(unit="USD"),
        "seats_durability_km": Reals(unit="km"),
    })

    def __init__(self, trim_level: float = 1.0):
        self.trim = float(trim_level)

    def h(self, f):
        n = max(1.0, float(f["passenger_capacity"]))
        # Two front bucket seats (~22 kg each) + rear bench scaled by n-2
        wt   = 44.0 + max(0.0, n - 2.0) * 14.0
        cost = (380.0 + max(0.0, n - 2.0) * 180.0) * self.trim
        return {
            "seats_weight_kg":     wt,
            "seats_cost_USD":      cost,
            "seats_durability_km": 350_000.0,
        }


# ---------------------------------------------------------------------------
# 3.2 HVAC (parametric: heater + AC + ducts, scales with cabin volume)
# ---------------------------------------------------------------------------

class HVAC(Module):
    """Heating, ventilation, air conditioning system.

    Sized to passenger capacity (proxy for cabin volume).
    """
    F = Ports({"passenger_capacity": Reals(unit="")})
    R = Ports({
        "hvac_weight_kg":     Reals(unit="kg"),
        "hvac_cost_USD":      Reals(unit="USD"),
        "hvac_aux_kW":        Reals(unit="kW"),
        "hvac_durability_km": Reals(unit="km"),
    })

    def h(self, f):
        n = max(1.0, float(f["passenger_capacity"]))
        # ~5 kg base, +1.5 kg per seat. Cost ~$280 + $40/seat.
        wt   = 5.0 + 1.5 * n
        cost = 280.0 + 40.0 * n
        # Average HVAC draw: heating uses waste heat for ICE, but for EV
        # everything is electric. We model the average parasitic.
        aux  = 0.5 + 0.15 * n
        return {
            "hvac_weight_kg":     wt,
            "hvac_cost_USD":      cost,
            "hvac_aux_kW":        aux,
            "hvac_durability_km": 200_000.0,
        }


# ---------------------------------------------------------------------------
# 3.3 Infotainment + safety (catalog: 3 trim levels)
# ---------------------------------------------------------------------------
# Bundles head unit, display, speakers, airbags, ABS/ESC, ADAS sensors.
# Modern cars bundle infotainment and safety because the same ECU
# array runs both.

# Fields: (name, weight_kg, cost_USD, durability_km, safety_score)
INFOTAINMENT_CATALOG = [
    ("basic",     35.0,  1200,  220_000, 1.0),
    ("mid",       55.0,  2600,  220_000, 1.2),
    ("premium",   85.0,  5800,  220_000, 1.4),
]

INFO_F = Ports({"trim_demand": Reals(unit="")})
INFO_R = Ports({
    "info_weight_kg":     Reals(unit="kg"),
    "info_cost_USD":      Reals(unit="USD"),
    "info_durability_km": Reals(unit="km"),
    "info_safety_score":  Reals(unit=""),
})


def make_infotainment_dp():
    entries = []
    # trim_demand ranges 1.0/1.2/1.4 (basic/mid/premium)
    for (name, wt, cost, dur, safety) in INFOTAINMENT_CATALOG:
        entries.append(CatalogEntry(
            provides={"trim_demand": safety},
            costs={
                "info_weight_kg":     wt,
                "info_cost_USD":      cost,
                "info_durability_km": dur,
                "info_safety_score":  safety,
            },
            name=name,
        ))
    return CatalogDP(F=INFO_F, R=INFO_R, catalog=entries,
                     name="infotainment_catalog")


# ===========================================================================
# SECTION 4: ENERGY-CONSUMPTION AND COST AGGREGATION
# ===========================================================================
#
# These are the lambda-based outer R aggregations. They synthesize the
# subsystem outputs into the macroscopic car metrics. They are
# attached to the System via sys.constrain(outer_r, lambda x: ...).
#
# Energy model: at steady cruise, the propulsive power requirement is
#
#   P_cruise [kW] = (0.5 * rho * Cd * A * v^3
#                    + Crr * m * g * v
#                    + P_accessory) / 1000
#
# where v is in m/s. Energy consumption per 100 km is then
# P_cruise * (100 / v_kmh) [kWh].
#
# For ICE: fuel_per_100km_L = energy_per_100km_kWh
#                              / (eta_engine * eta_trans * LHV_L_per_kWh)
# For EV : grid_per_100km_kWh = energy_per_100km_kWh
#                               / (eta_motor * eta_pe * eta_trans)
# For HEV: blended efficiency depending on operation mode.
# ---------------------------------------------------------------------------


def cruise_propulsive_kW(cd: float, area_m2: float, crr: float,
                         mass_kg: float, accessory_kW: float,
                         v_kmh: float = CRUISE_SPEED_KMH) -> float:
    """Tractive + accessory power demanded at cruise."""
    v = v_kmh / 3.6
    p_aero = 0.5 * RHO_AIR * cd * area_m2 * v ** 3 / 1000.0
    p_roll = crr * mass_kg * G * v / 1000.0
    return p_aero + p_roll + accessory_kW


def energy_per_100km_kWh(p_cruise_kW: float,
                         v_kmh: float = CRUISE_SPEED_KMH) -> float:
    """kWh per 100 km at the cruise speed."""
    if v_kmh < 1e-3:
        return float("inf")
    return p_cruise_kW * 100.0 / v_kmh


# ===========================================================================
# SECTION 5: ARCHITECTURE CATALOG
# ===========================================================================
#
# Each architecture pre-selects discrete catalog choices and sizes the
# energy storage strategy. The parametric modules (cooling, fuel,
# battery, brakes, suspension, etc.) then size themselves from the
# resulting demands via the Kleene cycle.
#
# Fields:
#   name
#   engine_idx          -> index into ENGINE_CATALOG (0 = none_ev)
#   transmission_idx    -> index into TRANSMISSION_CATALOG
#   emotor_idx          -> index into EMOTOR_CATALOG
#   boost_pressure      -> air-intake parameter (1.0 NA, 1.5/2.0 turbo)
#   ice_fraction        -> fraction of cruise power delivered by ICE
#                          (1.0 ICE-only, 0.5 FHEV, 0.0 BEV)
#   target_battery_kWh  -> intended battery capacity (architecture sets
#                          a floor; the Kleene cycle may resize upward
#                          for C-rate or range demands)
#   onboard_charger_kW
#   regen_braking       -> bool
#   awd                 -> bool
#   is_diesel           -> bool
# ---------------------------------------------------------------------------

ARCHITECTURE_CATALOG = [
    # name                       eng  trn  em  boost  ice_f  batt  chg  regen  awd  diesel
    ("ICE_economy",                1,  0,  0,  1.4,   1.00,   0.0,  0.0, False, False, False),
    ("ICE_family",                 3,  2,  0,  1.7,   1.00,   0.0,  0.0, False, False, False),
    ("ICE_premium",                4,  2,  0,  1.8,   1.00,   0.0,  0.0, False, False, False),
    ("ICE_performance",            6,  2,  0,  2.0,   1.00,   0.0,  0.0, False, True,  False),
    ("Diesel_long_range",          5,  2,  0,  1.8,   1.00,   0.0,  0.0, False, False, True),
    ("MHEV_48V",                   3,  2,  1,  1.7,   0.95,   1.0,  0.0, True,  False, False),
    ("HEV_full",                   2,  3,  2,  1.0,   0.60,   2.0,  0.0, True,  False, False),
    ("PHEV_small",                 3,  2,  3,  1.7,   0.50,  16.0,  7.0, True,  False, False),
    ("PHEV_large",                 4,  2,  4,  1.8,   0.40,  25.0, 11.0, True,  False, False),
    ("REEV",                       7,  4,  4,  1.2,   0.20,  55.0, 50.0, True,  False, False),
    ("BEV_long_range",             0,  4,  4,  1.0,   0.00,  80.0,150.0, True,  False, False),
    ("BEV_AWD_perf",               0,  4,  5,  1.0,   0.00, 100.0,250.0, True,  True,  False),
]


def architecture_by_name(name: str) -> dict:
    for entry in ARCHITECTURE_CATALOG:
        if entry[0] == name:
            (n, eng, trn, em, boost, icef, batt, chg, regen, awd, diesel) = entry
            return {
                "name": n, "engine_idx": eng, "transmission_idx": trn,
                "emotor_idx": em, "boost_pressure": boost,
                "ice_fraction": icef, "target_battery_kWh": batt,
                "onboard_charger_kW": chg, "regen_braking": regen,
                "awd": awd, "is_diesel": diesel,
            }
    raise KeyError(name)
