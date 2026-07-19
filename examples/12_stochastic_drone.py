"""
Stochastic uncertainty: Monte Carlo with a Gaussian copula.

The drone's battery has two internal parameters (specific energy,
efficiency) with marginal distributions and a positive correlation
between them (more energy-dense cells also tend to be more efficient).
A Gaussian copula glues the two marginals together.

We ask for several statistical summaries in one call:

- ``mean``: the expected value of each R port across the MC samples,
- ``p95``: the 95th percentile per R port (95% of samples are at or below),
- ``cvar95``: the expected value of the worst 5%,
- ``samples``: the raw antichain per MC sample, for plotting.

These come out of a single Monte Carlo run; the worst-case (set-based)
summary can be requested in the same call if a ``uncertain_set`` is
also attached. Here we include it to show that all four summaries
behave consistently: nominal < mean < p95 < cvar95 < worst_case.

Running this script prints the nominal mass, the four statistical
summaries, a bar chart of their ordering, and an ASCII histogram of
the sampled total_mass distribution. No figures are produced.
"""
from __future__ import annotations

from scipy import stats

from codesign import (
    Box,
    GaussianCopula,
    Module,
    Reals,
    Stochastic,
    System,
    solve,
)


# ---------------------------------------------------------------------------
# Modules (same as example 11)
# ---------------------------------------------------------------------------


class Battery(Module):
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}

    def __init__(self, specific_energy: float = 1.8e6, efficiency: float = 0.85):
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
# Build the drone with both set-based and stochastic uncertainty on the battery
# ---------------------------------------------------------------------------


def make_drone():
    bat = Battery()

    # Set-based (deterministic) uncertainty
    bat.uncertain_set = Box(
        specific_energy=(1.6e6, 2.0e6, "more_is_better"),
        efficiency=(0.80, 0.90, "more_is_better"),
    )

    # Stochastic uncertainty with a positive correlation between the
    # two parameters.
    bat.uncertain_dist = Stochastic(
        marginals={
            "specific_energy": stats.uniform(loc=1.6e6, scale=0.4e6),  # U[1.6, 2.0]e6
            "efficiency":      stats.uniform(loc=0.80, scale=0.10),    # U[0.80, 0.90]
        },
        copula=GaussianCopula(correlation=[[1.0, 0.4],
                                           [0.4, 1.0]]),
    )

    sys = System("drone")
    endurance     = sys.provides("endurance",     unit="s")
    extra_payload = sys.provides("extra_payload", unit="kg")
    extra_power   = sys.provides("extra_power",   unit="W")
    total_mass    = sys.requires("total_mass",    unit="kg")
    b = sys.add("battery",  bat)
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_power) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_payload)
    total_mass    >= b.mass + extra_payload
    return sys.build(), bat


if __name__ == "__main__":
    drone, bat = make_drone()
    f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}

    # Nominal
    nominal = solve(drone, f)
    nominal_mass = list(nominal.antichain.points)[0]["total_mass"]
    print(f"Nominal mass: {nominal_mass:.4f} kg")
    print()

    # All summaries in one call
    print("All uncertainty summaries from a single solve():")
    res = solve(
        drone, f,
        uncertainty=["worst_case", "mean", "p95", "cvar95", "samples"],
        n_samples=1000,
        rng_seed=42,
        verbose=1,
    )

    print()
    print("Statistical summaries (across 1000 MC samples):")
    print(f"   mean total_mass   = {res.mean['total_mass']:.4f} kg")
    print(f"   p95 total_mass    = {res.p95['total_mass']:.4f} kg")
    print(f"   cvar95 total_mass = {res.cvar95['total_mass']:.4f} kg")
    print(f"   feasibility rate  = {res.feasibility_rate:.3f}")
    wc = list(res.worst_case.antichain.points)[0]["total_mass"]
    print(f"   worst-case mass   = {wc:.4f} kg   (Box, deterministic)")
    print()

    # Sanity-check the ordering
    print("Ordering of summaries:")
    summary = [
        ("nominal",   nominal_mass),
        ("mean",      res.mean["total_mass"]),
        ("p95",       res.p95["total_mass"]),
        ("cvar95",    res.cvar95["total_mass"]),
        ("worst_case", wc),
    ]
    for label, val in summary:
        bar = "#" * int((val - nominal_mass) / 0.005)
        print(f"   {label:<10} {val:7.4f} kg  {bar}")

    # The raw samples are also available
    feasible_samples = [s for s in res.samples
                        if not s.has_any_top() and not s.is_empty()]
    print()
    print(f"Raw samples: {len(res.samples)} antichains "
          f"({len(feasible_samples)} feasible)")
    print()
    print("Histogram (10 bins) of total_mass across samples:")
    masses = sorted(list(s.points)[0]["total_mass"] for s in feasible_samples)
    lo, hi = masses[0], masses[-1]
    nbins = 10
    width = (hi - lo) / nbins if hi > lo else 1.0
    counts = [0] * nbins
    for m in masses:
        idx = min(int((m - lo) / width), nbins - 1)
        counts[idx] += 1
    max_c = max(counts)
    for i, c in enumerate(counts):
        edge_lo = lo + i * width
        edge_hi = edge_lo + width
        bar = "#" * int(40 * c / max_c) if max_c else ""
        print(f"   [{edge_lo:.4f} .. {edge_hi:.4f}]  {c:>4}  {bar}")
