"""
Off-grid microgrid: a flagship case study.

A remote cabin must supply a daily energy demand and a peak load
without grid power. Four subsystems contribute:

- a solar PV array (cheap when the sun shines, but the sun is
  stochastic),
- a lithium battery (smooths solar over the diurnal cycle, with mass
  and aging penalties that depend on chemistry),
- a diesel generator (perfectly reliable, but expensive and carbon-
  heavy),
- a mounting frame whose cost and mass scale with the total mass it
  must support, frame included, producing a genuine fixed-point
  coupling.

Features the package now offers come together in this example:

1. **Catalog of chemistries** for the battery (LFP, NMC, LCO, NaIon).
2. **Cyclic coupling** via the frame: the Kleene iteration genuinely
   iterates, instead of resolving in one step.
3. **Warm-started parameter sweep** that reuses the prior fixed point
   at each step.
4. **Stochastic uncertainty** on the daily sun hours, propagated
   through the loop to a distribution of feasible designs.

The visualisation module is used at the end to render the convergence
trace, the MC distribution of total cost, and the system structure.

Running this script prints the chemistry comparison, the warm-start
speedup, and the stochastic robustness summary, then writes three
files to ``outputs/``: ``microgrid_convergence.png``,
``microgrid_uncertainty.png``, and ``microgrid.dot`` (Graphviz source
for the system diagram).
"""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

from codesign import Module, Reals, Stochastic, System, solve, viz


# ---------------------------------------------------------------------------
# Subsystem modules
# ---------------------------------------------------------------------------


class SolarArray(Module):
    """Solar PV array. Daily energy = peak_power * sun_hours_per_day."""
    F = {
        "peak_power_kw":    Reals(unit="kW"),
        "daily_energy_kwh": Reals(unit="kWh"),
    }
    R = {
        "cost_usd": Reals(unit="USD"),
        "mass_kg":  Reals(unit="kg"),
    }

    def __init__(self, cost_per_kw: float = 1100.0, mass_per_kw: float = 28.0,
                 sun_hours_per_day: float = 3.0):
        self.cost_per_kw = cost_per_kw
        self.mass_per_kw = mass_per_kw
        self.sun_hours_per_day = sun_hours_per_day
        super().__init__()

    def h(self, f):
        sun = max(self.sun_hours_per_day, 1e-6)
        required_peak = max(
            f["peak_power_kw"],
            f["daily_energy_kwh"] / sun,
        )
        return {
            "cost_usd": required_peak * self.cost_per_kw,
            "mass_kg":  required_peak * self.mass_per_kw,
        }


class Battery(Module):
    """Battery; chemistry is a discrete parameter (catalog choice)."""
    F = {"storage_kwh": Reals(unit="kWh")}
    R = {
        "cost_usd":     Reals(unit="USD"),
        "mass_kg":      Reals(unit="kg"),
        "replacements": Reals(),
    }

    # (Wh/kg, USD/kWh, equivalent full cycles to end of life)
    CHEMISTRIES = {
        "LFP":   (160.0, 130.0, 4000.0),
        "NMC":   (240.0, 175.0, 2000.0),
        "LCO":   (220.0, 180.0,  800.0),
        "NaIon": (110.0,  90.0, 3000.0),
    }

    def __init__(self, chemistry: str = "LFP",
                 daily_cycles: float = 1.0, life_years: float = 10.0):
        if chemistry not in self.CHEMISTRIES:
            raise ValueError(
                f"unknown chemistry {chemistry!r}; "
                f"options are {list(self.CHEMISTRIES)}"
            )
        self.chemistry = chemistry
        wh_per_kg, usd_per_kwh, cycle_life = self.CHEMISTRIES[chemistry]
        self.specific_energy = wh_per_kg
        self.cost_density = usd_per_kwh
        self.cycle_life = cycle_life
        self.daily_cycles = daily_cycles
        self.life_years = life_years
        super().__init__()

    def h(self, f):
        kwh = f["storage_kwh"]
        used_cycles = self.daily_cycles * 365.0 * self.life_years
        replacements = used_cycles / max(self.cycle_life, 1.0)
        return {
            "cost_usd":     kwh * self.cost_density * (1.0 + replacements),
            "mass_kg":      kwh * 1000.0 / max(self.specific_energy, 1e-6),
            "replacements": replacements,
        }


class DieselGenerator(Module):
    """Backup generator. Diesel cost = capital + fuel."""
    F = {
        "backup_power_kw": Reals(unit="kW"),
        "backup_hours":    Reals(unit="h"),
    }
    R = {
        "cost_usd": Reals(unit="USD"),
        "mass_kg":  Reals(unit="kg"),
        "co2_kg":   Reals(unit="kg"),
    }

    def __init__(self, cost_per_kw: float = 500.0, mass_per_kw: float = 40.0,
                 fuel_cost_per_kwh: float = 0.35,
                 co2_per_kwh: float = 0.95):
        self.cost_per_kw = cost_per_kw
        self.mass_per_kw = mass_per_kw
        self.fuel_cost_per_kwh = fuel_cost_per_kwh
        self.co2_per_kwh = co2_per_kwh
        super().__init__()

    def h(self, f):
        p = f["backup_power_kw"]
        h = f["backup_hours"]
        energy_kwh = p * h
        capital = p * self.cost_per_kw
        fuel = energy_kwh * self.fuel_cost_per_kwh
        return {
            "cost_usd": capital + fuel,
            "mass_kg":  p * self.mass_per_kw,
            "co2_kg":   energy_kwh * self.co2_per_kwh,
        }


class Frame(Module):
    """Mounting frame; cost and mass scale with supported mass.

    The cyclic constraint
    ``frame.supported_mass_kg >= sum(other.mass_kg) + frame.mass_kg``
    creates the genuine fixed point this model needs.
    """
    F = {"supported_mass_kg": Reals(unit="kg")}
    R = {
        "cost_usd": Reals(unit="USD"),
        "mass_kg":  Reals(unit="kg"),
    }

    def __init__(self, cost_per_kg: float = 6.0, mass_fraction: float = 0.18):
        self.cost_per_kg = cost_per_kg
        self.mass_fraction = mass_fraction
        super().__init__()

    def h(self, f):
        m = f["supported_mass_kg"]
        return {
            "cost_usd": m * self.cost_per_kg,
            "mass_kg":  m * self.mass_fraction,
        }


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------


def make_microgrid(*, chemistry: str = "LFP",
                   sun_hours_per_day: float = 3.0,
                   solar_fraction: float = 0.85,
                   uncertainty: bool = False):
    """Build the microgrid as a System with the given parameters."""
    solar = SolarArray(sun_hours_per_day=sun_hours_per_day)
    if uncertainty:
        solar.uncertain_dist = Stochastic(
            marginals={
                "sun_hours_per_day":
                    stats.truncnorm(a=-1.5, b=1.5,
                                    loc=sun_hours_per_day, scale=1.0)
            },
        )

    battery = Battery(chemistry=chemistry)
    diesel = DieselGenerator()
    frame = Frame()

    sys = System(f"microgrid_{chemistry}")
    daily_load = sys.provides("daily_load_kwh", unit="kWh")
    peak_load  = sys.provides("peak_load_kw",   unit="kW")
    backup_h   = sys.provides("backup_hours",   unit="h")
    total_cost = sys.requires("total_cost_usd", unit="USD")
    total_mass = sys.requires("total_mass_kg",  unit="kg")
    annual_co2 = sys.requires("annual_co2_kg",  unit="kg/yr")

    s = sys.add("solar",   solar)
    b = sys.add("battery", battery)
    d = sys.add("diesel",  diesel)
    fr = sys.add("frame",  frame)

    s.daily_energy_kwh >= daily_load * solar_fraction
    s.peak_power_kw    >= peak_load
    b.storage_kwh      >= daily_load * solar_fraction
    d.backup_power_kw  >= peak_load
    d.backup_hours     >= backup_h

    # Cyclic coupling: the frame must support everything, frame included.
    fr.supported_mass_kg >= s.mass_kg + b.mass_kg + d.mass_kg + fr.mass_kg

    total_cost >= s.cost_usd + b.cost_usd + d.cost_usd + fr.cost_usd
    total_mass >= s.mass_kg + b.mass_kg + d.mass_kg + fr.mass_kg
    annual_co2 >= d.co2_kg * (365.0 / 30.0)

    return sys.build()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("# 1. Compare four battery chemistries at a fixed load")
    print("-" * 60)
    mission = {
        "daily_load_kwh": 15.0,
        "peak_load_kw":   3.0,
        "backup_hours":   12.0,
    }
    print(f"   mission: {mission}\n")
    by_chem = {}
    for chem in ["LFP", "NMC", "LCO", "NaIon"]:
        dp = make_microgrid(chemistry=chem)
        r = solve(dp, mission, max_iter=400)
        if r.feasible:
            p = list(r.antichain.points)[0]
            by_chem[chem] = (p, r.iterations)
            print(f"   {chem:<6}  cost=${p['total_cost_usd']:>8.0f}  "
                  f"mass={p['total_mass_kg']:>6.1f}kg  "
                  f"CO2={p['annual_co2_kg']:>6.1f}kg/yr  "
                  f"(iters={r.iterations})")
        else:
            print(f"   {chem:<6}  INFEASIBLE")
    print()

    print("# 2. Sweep daily-load demand from 5 to 30 kWh (LFP)")
    print("-" * 60)
    dp = make_microgrid(chemistry="LFP")
    loads = np.linspace(5.0, 30.0, 50)

    cold_iters = 0
    for L in loads:
        f = {"daily_load_kwh": float(L), "peak_load_kw": 3.0, "backup_hours": 12.0}
        r = solve(dp, f, max_iter=400)
        cold_iters += r.iterations

    warm_iters = 0
    prev = None
    sweep_results = []
    for L in loads:
        f = {"daily_load_kwh": float(L), "peak_load_kw": 3.0, "backup_hours": 12.0}
        r = solve(dp, f, max_iter=400, start_from=prev)
        warm_iters += r.iterations
        if r.feasible:
            p = list(r.antichain.points)[0]
            sweep_results.append((float(L), p["total_cost_usd"],
                                  p["total_mass_kg"], p["annual_co2_kg"]))
        prev = r

    print(f"   cold-start total iters: {cold_iters}")
    print(f"   warm-start total iters: {warm_iters}")
    print(f"   speedup: {cold_iters / max(warm_iters, 1):.2f}x")
    print()
    print(f"   load (kWh) | cost ($) | mass (kg) | CO2 (kg/yr)")
    for (L, c, m, co2) in sweep_results[::6]:
        print(f"     {L:>5.1f}    {c:>8.0f}   {m:>7.1f}   {co2:>9.1f}")
    print()

    print("# 3. Stochastic sun hours: how robust is the LFP design?")
    print("-" * 60)
    dp_u = make_microgrid(chemistry="LFP", uncertainty=True)
    res = solve(
        dp_u, mission,
        uncertainty=["mean", "p95", "cvar95", "samples"],
        n_samples=500, rng_seed=0, max_iter=400,
    )
    nominal_cost = by_chem["LFP"][0]["total_cost_usd"]
    print(f"   nominal cost:      ${nominal_cost:.0f}")
    print(f"   MC mean cost:      ${res.mean['total_cost_usd']:.0f}")
    print(f"   p95 cost:          ${res.p95['total_cost_usd']:.0f}")
    print(f"   CVaR95 cost:       ${res.cvar95['total_cost_usd']:.0f}")
    print(f"   feasibility rate:  {res.feasibility_rate:.3f}")
    print()

    # ----- 4. Visualisations -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plot generation)")
        return

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    r_trace = solve(dp, mission, max_iter=400, trace=True)
    ax = viz.plot_convergence(r_trace, title="Microgrid Kleene convergence (LFP)")
    plt.tight_layout()
    conv_path = os.path.join(out_dir, "microgrid_convergence.png")
    plt.savefig(conv_path, dpi=110)
    plt.close()
    print(f"   saved {conv_path}")

    ax = viz.plot_uncertainty(res, "total_cost_usd",
                              nominal=nominal_cost,
                              title="MC distribution of total cost (LFP)")
    plt.tight_layout()
    unc_path = os.path.join(out_dir, "microgrid_uncertainty.png")
    plt.savefig(unc_path, dpi=110)
    plt.close()
    print(f"   saved {unc_path}")

    dot = viz.to_dot(dp, name="microgrid")
    dot_path = os.path.join(out_dir, "microgrid.dot")
    with open(dot_path, "w") as fh:
        fh.write(dot)
    print(f"   saved {dot_path}")


if __name__ == "__main__":
    main()
