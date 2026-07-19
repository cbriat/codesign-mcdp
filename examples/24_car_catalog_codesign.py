"""
Example 24: Catalog-driven car co-design from a single architecture table.

A passenger vehicle is decomposed into 22 subsystems (powertrain,
chassis, interior + auxiliary). A single 12-row ARCHITECTURE_CATALOG
drives the whole study: each row pre-selects the discrete powertrain
choices (engine, transmission, electric-motor topology, boost, battery
strategy, charger, drivetrain) for one point on the modern technology
spectrum -- pure ICE, mild hybrid (MHEV), full hybrid (FHEV), plug-in
hybrid (PHEV), range-extender EV (REEV) and battery-electric (BEV) --
while the parametric modules (cooling, fuel, battery, brakes,
suspension, tyres, ...) size themselves from mission demand. The body,
tyres and suspension stay full CatalogDPs so the solver still chooses
the cheapest chassis that satisfies the mission.

This is the co-design counterpart of the hand-wired 17_car_codesign.py:
here one table + one build_architecture() function replaces the three
separate build_ice/build_hybrid/build_ev builders, and every powertrain
choice is a CatalogDP entry rather than a bespoke Module.

The classic automotive "weight death spiral" (vehicle mass -> required
power -> engine / motor / battery / fuel mass -> vehicle mass) and the
BEV energy spiral (mass -> consumption -> battery size -> mass) are both
closed by Kleene iteration on the monotone constraint network built by
System.

Outer mission (functionality)   Outer resources
    passenger_capacity (people)     production_cost      (USD)
    cargo_volume       (L)          curb_weight          (kg)
    max_speed          (km/h)       energy_per_100km     (kWh, ICE+EV unified)
    target_range       (km)         fuel_per_100km       (L)
    accel_0_100        (s)          co2_per_km           (g/km, tailpipe)
                                    durability           (km to major overhaul)

The unified energy metric is *primary energy input* per 100 km (fuel
lower-heating-value plus battery/grid electricity). This is why an ICE
lands near 50-60 kWh/100 km while a BEV lands near 14-18 kWh/100 km: the
combustion path throws away most of the fuel's chemical energy as heat.

SOURCES (values added or spot-checked against public data, 2024-2025)
--------------------------------------------------------------------
* Battery pack price 115 USD/kWh (volume-weighted global average) and
  BEV-pack 97 USD/kWh -- BloombergNEF 2024 Lithium-Ion Battery Price
  Survey, 10 Dec 2024.
  https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-see-largest-drop-since-2017-falling-to-115-per-kilowatt-hour-bloombergnef/
* Pack-level gravimetric energy density: LFP 125-145, NMC 140-180 Wh/kg
  (130 Wh/kg blended baseline used here) -- Wassiliadis et al., "From
  Cell to Pack", World Electric Vehicle Journal 16(9):484, 2025.
  https://www.mdpi.com/2032-6653/16/9/484
* Gasoline lower heating value ~34 MJ/L (8.95 kWh/L used), density
  0.745 kg/L, CO2 2.31 kg/L burned -- UBC Physics "Fossil Fuels for
  Transport" (46.4 MJ/kg, 0.737 kg/L -> 34.2 MJ/L; 2.36 kg CO2/L).
  https://c21.phas.ubc.ca/article/fossil-fuels-for-transport/
* 4-cylinder engine dry mass: 1.5 L ~110-116 kg, 2.0 L ~136-150 kg
  (turbo/ancillaries add mass) -- industry engine-weight references.
  https://autopartswd.com/car-engine-weight-guide/
* Compact-car curb mass ~1290-1560 kg (2024 Toyota Corolla) -- KBB.
  https://www.kbb.com/toyota/corolla/2024/specs/
* Compact BEV real-world consumption ~14-16 kWh/100 km (efficient
  compacts); half of the EPA fleet sits 12-16 kWh/100 km -- InsideEVs
  2024 consumption ranking / EPA fuel-economy data.
  https://insideevs.com/news/709706/electric-cars-energy-consumption-ranking/
* Diesel LHV 9.94 kWh/L (35.8 MJ/L), density 0.832 kg/L, CO2 2.68 kg/L
  -- U. Waterloo carbon-pricing heating-value note.
  https://uwaterloo.ca/centre-advanced-science-education/news/gasoline-and-diesel-fuel-carbon-pricing-and-heating-values
* Mission spec (5 seats, 370 L, 170 km/h, 500 km, 11.5 s 0-100) is a
  compact C-segment target, bracketed by the Corolla figures above.

Component masses/costs/efficiencies not individually footnoted are kept
from the original scaffold and lie within the published ranges of the
sources above (Bosch Automotive Handbook 2022; ICCT EV cost work; EPA
Automotive Trends). They are illustrative, not OEM-specific: the MCDP
framework is what is being validated; the numbers serve the framework.

Run:  python -m examples.24_car_catalog_codesign
Expected output: per-architecture feasibility over the 12-row table
(feasible / infeasible), the cheapest feasible design per architecture
with its curb mass, unified energy/100 km and tailpipe CO2, a Pareto
front over (cost, energy/100 km, mass), and a comparison table grouped
by ICE / MHEV / FHEV / PHEV / REEV / BEV. Runs in a few seconds.
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


def make_engine_dp(rows=ENGINE_CATALOG):
    entries = []
    for (name, disp, pkw, ptq, ftype, eff, wt, cost, dur) in rows:
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
            # Same durability as a sized tank: a value that FELL once the
            # (mass-coupled) fuel demand crossed this threshold would be
            # non-monotone and break the Kleene feedback loop.
            return {
                "fuel_weight_kg":     0.0,
                "fuel_cost_USD":      0.0,
                "max_fuel_L":         0.0,
                "fuel_durability_km": 300_000.0,
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


def make_transmission_dp(rows=TRANSMISSION_CATALOG):
    entries = []
    for (name, ratio, ptq, eff, wt, cost, dur, _clutch) in rows:
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


def make_emotor_dp(rows=EMOTOR_CATALOG):
    entries = []
    for (name, pkw, ptq, wt, cost, eff, dur) in rows:
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

    Calibration follows the BNEF 2024 Lithium-Ion Battery Price Survey
    (volume-weighted pack average $115/kWh; BEV packs $97/kWh) and
    pack-level specific energy from Wassiliadis et al. 2025 (LFP
    125-145, NMC 140-180 Wh/kg). We use a blended 130 Wh/kg / $115/kWh
    as a representative 2024-2025 baseline. See module SOURCES block.
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
                 cost_USD_per_kWh: float = 115.0,  # BNEF 2024 pack avg
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
        # Durability is held constant across pack sizes: a value that
        # dropped as the (fed-back) pack grew would be non-monotone and
        # break the Kleene feedback loop. Weight/cost/aux still scale up.
        if kwh < 0.5:
            return {"bcool_weight_kg": 0.0, "bcool_cost_USD": 0.0,
                    "bcool_aux_kW": 0.0, "bcool_durability_km": 250_000.0}
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


def make_body_dp(rows=BODY_CATALOG):
    entries = []
    for (name, n_seats, cargo, wt, cost, cd, fa, dur) in rows:
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


def make_suspension_dp(name_suffix: str = "", rows=SUSPENSION_CATALOG):
    entries = []
    for (name, load, wt, cost, comf, dur) in rows:
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
        # Durability depends on the axle (pad/rotor duty), NOT on the
        # fed-back curb mass: a resource that FELL as the design mass rose
        # would be non-monotone and break the Kleene feedback (the loop
        # requires every fed-back R port to be non-decreasing in F).
        dur  = 120_000.0 if self.axle == "front" else 150_000.0
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


def make_tires_dp(rows=TIRE_CATALOG):
    entries = []
    for (name, load, vmax, wt, cost, crr, _grip, dur) in rows:
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


def make_infotainment_dp(rows=INFOTAINMENT_CATALOG):
    entries = []
    # trim_demand ranges 1.0/1.2/1.4 (basic/mid/premium)
    for (name, wt, cost, dur, safety) in rows:
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


# Powertrain-class label for each architecture, used only for grouping the
# final comparison table (ICE / MHEV / FHEV / PHEV / REEV / BEV).
ARCH_CLASS = {
    "ICE_economy": "ICE", "ICE_family": "ICE", "ICE_premium": "ICE",
    "ICE_performance": "ICE", "Diesel_long_range": "ICE",
    "MHEV_48V": "MHEV", "HEV_full": "FHEV",
    "PHEV_small": "PHEV", "PHEV_large": "PHEV",
    "REEV": "REEV", "BEV_long_range": "BEV", "BEV_AWD_perf": "BEV",
}


# ===========================================================================
# SECTION 6: MISSION AND MODEL CONSTANTS
# ===========================================================================
#
# One representative compact C-segment mission drives the whole study. The
# targets are bracketed by the 2024 Toyota Corolla figures cited in the
# module SOURCES block (curb ~1290-1560 kg, 0-100 ~10-11 s).

MISSION = {
    "passenger_capacity": 5,       # people
    "cargo_volume_L":     370.0,   # L (compact-hatch class)
    "max_speed_kmh":      170.0,   # km/h
    "target_range_km":    500.0,   # km per tank or charge
    "accel_0_100_s":      11.5,    # s, 0-100 km/h target
}

# Drivetrain-efficiency and duty constants (dimensionless unless noted).
# These are the only "added" tuning constants; each is grounded below.
CRUISE_EFF_FACTOR = 0.80   # engine cruise eff as fraction of peak BSFC eff
                           # (part-load penalty at steady 90 km/h; Bosch
                           #  Automotive Handbook BSFC maps, engine runs
                           #  below its best point at light cruise load)
PE_EFF            = 0.95    # inverter + DC-DC combined (Larminie & Lowry,
                           #  "Electric Vehicle Technology Explained", 2ed)
ALT_EFF           = 0.60    # alternator electrical-to-mechanical (Bosch
                           #  Automotive Handbook; belt-driven claw-pole)
AUX_DUTY          = 0.30    # cruise duty factor on peak auxiliary loads
                           #  (HVAC/fans/pumps rarely run at peak at steady
                           #  cruise; 0.3 keeps derived L/100km realistic)
ACCEL_PEAK_FACTOR = 1.6    # peak/average power ratio over a 0-100 pull
                           #  (energy-balance average x 1.6; a full-throttle
                           #  pull is not a constant-power event)
FINAL_DRIVE_REDUCTION = 6.0  # representative engine/motor -> axle torque
                             #  multiplier (gearbox x final drive), tuned so
                             #  a 250 Nm engine gives ~1500 Nm axle torque,
                             #  matching the FinalDrive calibration comment

ASSEMBLY_USD_PER_KG = 1.5   # labour + paint + BIW + final assembly overhead
                            #  (per-kg proxy, same convention as ex. 17)


def _fuel_props(ftype: str):
    """(LHV kWh/L, density kg/L, CO2 kg/L) for a fuel-type tag."""
    if ftype == "gas":
        return LHV_GASOLINE_KWH_PER_L, DENSITY_GASOLINE, CO2_GASOLINE_KG_PER_L
    if ftype == "diesel":
        return LHV_DIESEL_KWH_PER_L, DENSITY_DIESEL, CO2_DIESEL_KG_PER_L
    return 1.0, 0.0, 0.0   # "none" (EV): no liquid fuel


# Module bookkeeping: (system name, weight port, cost port, durability port).
# Used by the mass / cost / durability aggregation lambdas.
MODULE_PORTS = [
    ("engine",       "engine_weight_kg",       "engine_cost_USD",       "engine_durability_km"),
    ("intake",       "intake_weight_kg",       "intake_cost_USD",       "intake_durability_km"),
    ("exhaust",      "exhaust_weight_kg",       "exhaust_cost_USD",      "exhaust_durability_km"),
    ("engine_cool",  "engine_cool_weight_kg",   "engine_cool_cost_USD",  "engine_cool_durability_km"),
    ("lube",         "lube_weight_kg",          "lube_cost_USD",         "lube_durability_km"),
    ("fuel",         "fuel_weight_kg",          "fuel_cost_USD",         "fuel_durability_km"),
    ("trans",        "trans_weight_kg",         "trans_cost_USD",        "trans_durability_km"),
    ("final_drive",  "final_drive_weight_kg",   "final_drive_cost_USD",  "final_drive_durability_km"),
    ("motor",        "motor_weight_kg",         "motor_cost_USD",        "motor_durability_km"),
    ("battery",      "battery_weight_kg",       "battery_cost_USD",      "battery_durability_km"),
    ("pe",           "pe_weight_kg",            "pe_cost_USD",           "pe_durability_km"),
    ("bcool",        "bcool_weight_kg",         "bcool_cost_USD",        "bcool_durability_km"),
    ("body",         "body_weight_kg",          "body_cost_USD",         "body_durability_km"),
    ("susp_front",   "susp_weight_kg",          "susp_cost_USD",         "susp_durability_km"),
    ("susp_rear",    "susp_weight_kg",          "susp_cost_USD",         "susp_durability_km"),
    ("brakes_front", "brake_weight_kg",         "brake_cost_USD",        "brake_durability_km"),
    ("brakes_rear",  "brake_weight_kg",         "brake_cost_USD",        "brake_durability_km"),
    ("steering",     "steering_weight_kg",      "steering_cost_USD",     "steering_durability_km"),
    ("tires",        "tires_weight_kg",         "tires_cost_USD",        "tires_durability_km"),
    ("seats",        "seats_weight_kg",         "seats_cost_USD",        "seats_durability_km"),
    ("hvac",         "hvac_weight_kg",          "hvac_cost_USD",         "hvac_durability_km"),
    ("info",         "info_weight_kg",          "info_cost_USD",         "info_durability_km"),
]


# ===========================================================================
# SECTION 7: SYSTEM ASSEMBLY FROM ONE ARCHITECTURE ROW
# ===========================================================================


def build_architecture(arch: dict, mission: Mapping[str, float], *,
                       body_row, susp_row, tire_row,
                       info_row=INFOTAINMENT_CATALOG[0]) -> Any:
    """Assemble a solvable System for one ARCHITECTURE_CATALOG row.

    The architecture fixes the discrete powertrain (engine / transmission /
    e-motor are one-entry CatalogDPs sliced from the master catalogs). The
    chassis catalog rows (body / suspension / tyre) are passed in one at a
    time by the sweep in :func:`solve_architecture`, so each System is
    single-valued and its Kleene loop carries only monotone quantities.
    Every parametric module (cooling, fuel, battery, brakes, steering, ...)
    sizes itself from the mission and from the converged curb mass through
    the loop.

    Two coupled cycles are closed by the loop:
      * mass spiral -- every load-bearing module reads total curb mass;
      * energy/battery spiral -- consumption depends on mass, and for BEVs
        the battery is sized to range x consumption, feeding back to mass.
    """
    # --- Known constants pulled from the pre-selected catalog rows -------
    eng = ENGINE_CATALOG[arch["engine_idx"]]
    trn = TRANSMISSION_CATALOG[arch["transmission_idx"]]
    em  = EMOTOR_CATALOG[arch["emotor_idx"]]
    disp    = eng[1]
    pkw_eng = eng[2]
    ptq_eng = eng[3]
    ftype   = eng[4]
    eff_eng = eng[5]
    eff_trn = trn[3]
    pkw_em  = em[1]
    ptq_em  = em[2]
    eff_em  = em[5]
    ice_f   = arch["ice_fraction"]
    lhv_kWh_per_L, fuel_dens, co2_kg_per_L = _fuel_props(ftype)
    engine_heat_kW = pkw_eng * (1.0 / eff_eng - 1.0) if eff_eng < 1.0 else 0.0

    sys = System(arch["name"])

    # --- Outer functionality (mission) ----------------------------------
    sys.provides("passenger_capacity", unit="")
    sys.provides("cargo_volume_L",     unit="L")
    sys.provides("max_speed_kmh",      unit="km/h")
    sys.provides("target_range_km",    unit="km")

    # --- Outer resources (macro objectives) -----------------------------
    sys.requires("production_cost_USD",   unit="USD")
    sys.requires("curb_weight_kg",        unit="kg")
    sys.requires("energy_per_100km_kWh",  unit="kWh")
    sys.requires("fuel_per_100km_L",      unit="L")
    sys.requires("co2_per_km",            unit="g/km")
    sys.requires("durability_km",         unit="km")

    # --- Modules --------------------------------------------------------
    # Powertrain: engine / transmission / motor are single-entry catalogs
    # sliced from the master catalogs by the architecture's indices.
    sys.add("engine", make_engine_dp([eng]))
    sys.add("intake", AirIntake(boost_pressure=arch["boost_pressure"]))
    sys.add("exhaust", ExhaustSystem(is_diesel=arch["is_diesel"]))
    sys.add("engine_cool", EngineCooling())
    sys.add("lube", EngineLubrication())
    sys.add("fuel", FuelSystem())
    sys.add("trans", make_transmission_dp([trn]))
    sys.add("final_drive", FinalDrive(awd=arch["awd"]))
    sys.add("motor", make_emotor_dp([em]))
    sys.add("battery", BatteryPack())
    sys.add("pe", PowerElectronics(charger_kW=arch["onboard_charger_kW"]))
    sys.add("bcool", BatteryCooling())

    # Chassis: one catalog row each, chosen by the sweep in
    # solve_architecture (keeps the System single-valued / monotone).
    sys.add("body", make_body_dp([body_row]))
    sys.add("susp_front", make_suspension_dp("_front", [susp_row]))
    sys.add("susp_rear", make_suspension_dp("_rear", [susp_row]))
    sys.add("brakes_front", Brakes(axle="front"))
    sys.add("brakes_rear", Brakes(axle="rear"))
    sys.add("steering", Steering())
    sys.add("tires", make_tires_dp([tire_row]))

    # Interior / auxiliary.
    sys.add("seats", Seats(trim_level=1.0))
    sys.add("hvac", HVAC())
    sys.add("info", make_infotainment_dp([info_row]))

    # --- Loop helpers (functions of the module-R context x) -------------

    def total_mass(x):
        m = DRIVER_MASS
        for name, wport, _c, _d in MODULE_PORTS:
            m += x[f"{name}.{wport}"]
        return m

    def peak_power_kW(x):
        """Combined powertrain peak demand: max of top-speed road load and
        acceleration, referred to the powertrain output through the box."""
        m = total_mass(x)
        cd  = x["body.drag_coefficient"]
        A   = x["body.frontal_area_m2"]
        crr = x["tires.rolling_resistance"]
        p_top = 1.10 * cruise_propulsive_kW(cd, A, crr, m, 0.0,
                                            mission["max_speed_kmh"])
        v = 100.0 / 3.6
        p_accel = ACCEL_PEAK_FACTOR * 0.5 * m * v ** 2 \
            / mission["accel_0_100_s"] / 1000.0
        return max(p_top, p_accel) / max(eff_trn, 0.5)

    def accessory_cruise_kW(x):
        """Continuous electrical accessory draw at cruise (kW)."""
        peak = (x["hvac.hvac_aux_kW"] + x["steering.steering_aux_kW"]
                + x["engine_cool.engine_cool_aux_kW"] + x["bcool.bcool_aux_kW"])
        return ACCESSORY_LOAD_KW + AUX_DUTY * peak

    def fuel_power_kW(x):
        """Primary fuel power at cruise (kW of fuel LHV). Zero for a BEV."""
        if ice_f <= 0.0 or lhv_kWh_per_L <= 0.0:
            return 0.0
        cd  = x["body.drag_coefficient"]
        A   = x["body.frontal_area_m2"]
        crr = x["tires.rolling_resistance"]
        p_tractive = cruise_propulsive_kW(cd, A, crr, total_mass(x), 0.0)
        eng_cruise_eff = max(CRUISE_EFF_FACTOR * eff_eng, 0.05)
        ice_tractive = ice_f * p_tractive
        fuel_kW = ice_tractive / (eng_cruise_eff * max(eff_trn, 0.5))
        # Accessory share carried by the engine via the alternator.
        acc_fuel = ice_f * accessory_cruise_kW(x) / (ALT_EFF * eng_cruise_eff)
        return fuel_kW + acc_fuel

    def electric_power_kW(x):
        """Primary electrical power at cruise (kW from battery/grid)."""
        cd  = x["body.drag_coefficient"]
        A   = x["body.frontal_area_m2"]
        crr = x["tires.rolling_resistance"]
        p_tractive = cruise_propulsive_kW(cd, A, crr, total_mass(x), 0.0)
        elec_tractive = (1.0 - ice_f) * p_tractive
        drive_eff = max(eff_em, 0.5) * PE_EFF * max(eff_trn, 0.5)
        elec_kW = elec_tractive / drive_eff if elec_tractive > 0.0 else 0.0
        # Accessory share carried electrically (battery/12V from HV pack).
        elec_kW += (1.0 - ice_f) * accessory_cruise_kW(x)
        return elec_kW

    def fuel_per_100km_L(x):
        if lhv_kWh_per_L <= 0.0:
            return 0.0
        e_kWh_100 = fuel_power_kW(x) * 100.0 / CRUISE_SPEED_KMH
        return e_kWh_100 / lhv_kWh_per_L

    def energy_per_100km(x):
        """Unified primary energy: fuel LHV + electricity, per 100 km."""
        return (fuel_power_kW(x) + electric_power_kW(x)) \
            * 100.0 / CRUISE_SPEED_KMH

    def battery_target_kWh(x):
        floor = arch["target_battery_kWh"]
        if ice_f <= 0.0:  # pure BEV: size to full electric range
            need = mission["target_range_km"] * energy_per_100km(x) / 100.0
            return max(floor, need)
        return floor

    # --- Mission propagation (outer F -> module F) ----------------------
    sys.constrain("body.passenger_capacity",
                  lambda x, v=mission["passenger_capacity"]: v)
    sys.constrain("body.cargo_volume_L",
                  lambda x, v=mission["cargo_volume_L"]: v)
    sys.constrain("seats.passenger_capacity",
                  lambda x, v=mission["passenger_capacity"]: v)
    sys.constrain("hvac.passenger_capacity",
                  lambda x, v=mission["passenger_capacity"]: v)
    sys.constrain("tires.tire_max_kmh",
                  lambda x, v=mission["max_speed_kmh"]: v)
    sys.constrain("brakes_front.max_speed_kmh",
                  lambda x, v=mission["max_speed_kmh"]: v)
    sys.constrain("brakes_rear.max_speed_kmh",
                  lambda x, v=mission["max_speed_kmh"]: v)
    sys.constrain("info.trim_demand", lambda x: 1.0)  # basic trim floor

    # --- Powertrain demands (mostly known constants; power is mass-coupled)
    sys.constrain("engine.ice_power_demand_kW",
                  lambda x: ice_f * peak_power_kW(x))
    sys.constrain("motor.motor_power_demand_kW",
                  lambda x: (1.0 - ice_f) * peak_power_kW(x))
    sys.constrain("motor.motor_torque_demand_Nm",
                  lambda x: 2.0 * (1.0 - ice_f) * peak_power_kW(x))
    sys.constrain("intake.engine_peak_power_kW", lambda x, p=pkw_eng: p)
    sys.constrain("exhaust.displacement_L", lambda x, d=disp: d)
    sys.constrain("engine_cool.heat_rejection_kW",
                  lambda x, q=engine_heat_kW: q)
    sys.constrain("lube.displacement_L", lambda x, d=disp: d)
    # Transmission/final-drive torque: engine torque, or motor torque per
    # driven axle when there is no engine (pure EV).
    n_axles = 2.0 if arch["awd"] else 1.0
    trans_torque = max(ptq_eng, ptq_em / n_axles)
    axle_torque = max(ptq_eng, ptq_em / n_axles) * FINAL_DRIVE_REDUCTION
    sys.constrain("trans.trans_peak_torque_Nm", lambda x, t=trans_torque: t)
    sys.constrain("final_drive.peak_axle_torque_Nm",
                  lambda x, t=axle_torque: t)
    sys.constrain("pe.motor_peak_power_kW", lambda x, p=pkw_em: p)
    sys.constrain("battery.motor_peak_power_kW", lambda x, p=pkw_em: p)
    sys.constrain("battery.target_battery_kWh", battery_target_kWh)
    sys.constrain("bcool.battery_capacity_kWh",
                  lambda x: x["battery.battery_capacity_kWh"])

    # --- Fuel system (range fixed, consumption mass-coupled) ------------
    sys.constrain("fuel.target_range_km",
                  lambda x, r=mission["target_range_km"]: r)
    sys.constrain("fuel.fuel_per_100km_L", fuel_per_100km_L)
    sys.constrain("fuel.fuel_density", lambda x, d=fuel_dens: d)

    # --- Mass spiral: load-bearing modules read total curb mass --------
    sys.constrain("susp_front.axle_load_kg", lambda x: 0.55 * total_mass(x))
    sys.constrain("susp_rear.axle_load_kg",  lambda x: 0.45 * total_mass(x))
    sys.constrain("steering.front_axle_load_kg",
                  lambda x: 0.55 * total_mass(x))
    # Per-tyre static load = curb mass / 4 (four contact patches). The
    # tyre catalog's max_load_kg is a per-tyre load rating.
    sys.constrain("tires.tire_load_kg", lambda x: 0.25 * total_mass(x))
    sys.constrain("brakes_front.vehicle_mass_kg", total_mass)
    sys.constrain("brakes_rear.vehicle_mass_kg", total_mass)

    # --- Outer resource aggregations -----------------------------------
    def production_cost(x):
        c = 0.0
        for name, _w, cport, _d in MODULE_PORTS:
            c += x[f"{name}.{cport}"]
        return c + ASSEMBLY_USD_PER_KG * total_mass(x)

    def durability(x):
        # Harmonic mean of the life-limiting subsystems (whichever wears
        # out first dominates the vehicle's useful life).
        keys = [
            "engine.engine_durability_km", "trans.trans_durability_km",
            "motor.motor_durability_km", "battery.battery_durability_km",
            "body.body_durability_km", "susp_front.susp_durability_km",
            "tires.tires_durability_km", "exhaust.exhaust_durability_km",
        ]
        inv = 0.0
        for k in keys:
            v = max(x[k], 1.0)
            inv += 1.0 / v
        return len(keys) / inv if inv > 0 else 0.0

    sys.constrain("production_cost_USD", production_cost)
    sys.constrain("curb_weight_kg", total_mass)
    sys.constrain("energy_per_100km_kWh", energy_per_100km)
    sys.constrain("fuel_per_100km_L", fuel_per_100km_L)
    sys.constrain("co2_per_km",
                  lambda x, c=co2_kg_per_L: fuel_per_100km_L(x) * c * 10.0)
    sys.constrain("durability_km", durability)

    return sys.build()


# ===========================================================================
# SECTION 8: SOLVE LOOP OVER THE ARCHITECTURE TABLE
# ===========================================================================


def _eligible_bodies(mission: Mapping[str, float]):
    """Body catalog rows that can seat/carry the mission."""
    return [r for r in BODY_CATALOG
            if r[1] >= mission["passenger_capacity"]
            and r[2] >= mission["cargo_volume_L"]]


def _eligible_tires(mission: Mapping[str, float]):
    """Tire catalog rows rated for the mission top speed."""
    return [r for r in TIRE_CATALOG if r[2] >= mission["max_speed_kmh"]]


def _solve_one(arch, mission, body_row, susp_row, tire_row, max_iter):
    dp = build_architecture(arch, mission, body_row=body_row,
                            susp_row=susp_row, tire_row=tire_row)
    res = solve(dp, dict(
        passenger_capacity=mission["passenger_capacity"],
        cargo_volume_L=mission["cargo_volume_L"],
        max_speed_kmh=mission["max_speed_kmh"],
        target_range_km=mission["target_range_km"],
    ), max_iter=max_iter, verbose=0)
    if not res.feasible or not res.antichain:
        return None
    finite = [dict(p) for p in res.antichain.points
              if all(math.isfinite(dict(p).get(k, float("inf")))
                     for k in ("production_cost_USD", "curb_weight_kg",
                               "energy_per_100km_kWh"))]
    if not finite:
        return None
    return min(finite, key=lambda p: p["production_cost_USD"])


def solve_architecture(arch: dict, mission: Mapping[str, float],
                       *, max_iter: int = 200):
    """Sweep the eligible chassis catalog rows and return the best design.

    For the pre-selected powertrain of ``arch``, this tries every eligible
    (body, suspension, tyre) combination -- each a single-valued System --
    and keeps the cheapest feasible converged design. Returns
    (feasible: bool, best_point_or_None).
    """
    best = None
    for body_row in _eligible_bodies(mission):
        for susp_row in SUSPENSION_CATALOG:
            for tire_row in _eligible_tires(mission):
                pt = _solve_one(arch, mission, body_row, susp_row,
                                tire_row, max_iter)
                if pt is None:
                    continue
                if best is None or \
                   pt["production_cost_USD"] < best["production_cost_USD"]:
                    best = pt
    return (best is not None), best


def _pareto(points: Sequence[Tuple[str, dict]],
            axes: Sequence[str]) -> List[Tuple[str, dict]]:
    """Non-dominated (label, point) pairs over the given minimise-axes."""
    out = []
    pts = [p for _, p in points]
    for label, p in points:
        dominated = False
        for q in pts:
            if q is p:
                continue
            if all(q[k] <= p[k] for k in axes) \
               and any(q[k] < p[k] for k in axes):
                dominated = True
                break
        if not dominated:
            out.append((label, p))
    return out


def main():
    print("=" * 74)
    print("Example 24: catalog-driven car co-design from one architecture table")
    print("=" * 74)
    print(f"Mission: {mission_str(MISSION)}")

    solved: List[Tuple[str, dict]] = []   # (arch_name, best_point)

    print("\n--- Per-architecture feasibility (cheapest feasible design) ---")
    header = (f"  {'Architecture':<18}{'cls':<6}{'feas':<6}"
              f"{'cost$':>9}{'mass kg':>9}{'kWh/100':>9}"
              f"{'L/100':>7}{'CO2':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in ARCHITECTURE_CATALOG:
        arch = architecture_by_name(entry[0])
        feasible, best = solve_architecture(arch, MISSION)
        cls = ARCH_CLASS[arch["name"]]
        if not feasible:
            print(f"  {arch['name']:<18}{cls:<6}{'no':<6}"
                  f"{'-':>9}{'-':>9}{'-':>9}{'-':>7}{'-':>7}")
            continue
        solved.append((arch["name"], best))
        print(f"  {arch['name']:<18}{cls:<6}{'yes':<6}"
              f"{best['production_cost_USD']:>9,.0f}"
              f"{best['curb_weight_kg']:>9.0f}"
              f"{best['energy_per_100km_kWh']:>9.1f}"
              f"{best['fuel_per_100km_L']:>7.1f}"
              f"{best['co2_per_km']:>7.0f}")

    if not solved:
        print("\nNo architecture was feasible for this mission.")
        return solved

    # --- Pareto front over (cost, energy, mass) ------------------------
    axes = ("production_cost_USD", "energy_per_100km_kWh", "curb_weight_kg")
    front = _pareto(solved, axes)
    print("\n--- Pareto front over (cost, energy/100km, mass) ---")
    for name, p in sorted(front, key=lambda t: t[1]["production_cost_USD"]):
        print(f"  {name:<18} ${p['production_cost_USD']:>8,.0f}"
              f"  {p['energy_per_100km_kWh']:>5.1f} kWh/100"
              f"  {p['curb_weight_kg']:>5.0f} kg")

    # --- Comparison grouped by powertrain class ------------------------
    print("\n--- Comparison by powertrain class (cheapest feasible each) ---")
    print(f"  {'class':<6}{'best arch':<18}{'cost$':>9}{'mass kg':>9}"
          f"{'kWh/100':>9}{'CO2 g/km':>10}")
    by_class: Dict[str, Tuple[str, dict]] = {}
    for name, p in solved:
        cls = ARCH_CLASS[name]
        if cls not in by_class or \
           p["production_cost_USD"] < by_class[cls][1]["production_cost_USD"]:
            by_class[cls] = (name, p)
    for cls in ("ICE", "MHEV", "FHEV", "PHEV", "REEV", "BEV"):
        if cls not in by_class:
            continue
        name, p = by_class[cls]
        print(f"  {cls:<6}{name:<18}{p['production_cost_USD']:>9,.0f}"
              f"{p['curb_weight_kg']:>9.0f}{p['energy_per_100km_kWh']:>9.1f}"
              f"{p['co2_per_km']:>10.0f}")

    return solved


def mission_str(m: Mapping[str, float]) -> str:
    return (f"{int(m['passenger_capacity'])} seats, "
            f"{int(m['cargo_volume_L'])} L, "
            f"{int(m['max_speed_kmh'])} km/h top, "
            f"{int(m['target_range_km'])} km range, "
            f"0-100 in {m['accel_0_100_s']:.1f} s")


if __name__ == "__main__":
    main()
