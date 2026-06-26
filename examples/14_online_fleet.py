"""
Online elimination-based co-design: heterogeneous robot fleet sizing.

This example reproduces the spirit of the multi-robot fleet case study
from Alharbi, Dahleh & Zardini (arXiv:2604.22624). A logistics service
must deliver a target throughput over a target range, and can purchase
robots from a large catalogue of N=200 candidate types, each described
by four "features": cruise speed, payload, unit cost, and energy per
kilometre.

For each candidate, the inner solve computes the required fleet cost
and the daily energy budget. Naively this is 200 inner solves. The
online solver from ``codesign.online`` uses an optimistic evaluator
(monotonicity, Lipschitz, or linear-parametric) to bound each
candidate's inner-solve output, eliminate candidates that are provably
suboptimal, and only run inner solves for the survivors.

The example shows three flavours of evaluator side by side:

- **Lipschitz** is the most general (no monotonicity required) and the
  most reliable: with a sensible L it never prunes a Pareto-optimal
  candidate.
- **Monotonicity** is the cheapest when applicable; here we manufacture
  a derived feature ``cost_per_capacity`` under which ``total_cost``
  is genuinely monotone. With the right feature, this is the workhorse.
- **Linear-parametric** is the most aggressive but the least safe: when
  the underlying map is nearly linear, the confidence band tightens
  fast and many candidates fall away; when it isn't, the bound can be
  too tight and Pareto-optimal candidates can be wrongly eliminated.

A plot at the end shades the feature plane by status (evaluated,
eliminated, optimal) so the reader can see how the elimination cascade
runs through the catalogue.
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # render to PNG without a display
import matplotlib.pyplot as plt

from codesign import (
    AlgebraicDP,
    LinearParametricEvaluator,
    LipschitzEvaluator,
    MonotonicityEvaluator,
    Ports,
    Reals,
    solve,
    solve_online,
)


# ---------------------------------------------------------------------------
# The inner DP per robot type
# ---------------------------------------------------------------------------


F = Ports({
    "target_throughput": Reals(unit="pkg/h"),
    "target_range":      Reals(unit="km"),
})
R = Ports({
    "total_cost":   Reals(unit="USD"),
    "total_energy": Reals(unit="kWh/day"),
})


def make_dp(robot: Dict[str, float]):
    """Return a fresh DP for a single robot type.

    Smooth (fractional-fleet) model so that small feature perturbations
    produce small output changes, which is what the Lipschitz and
    linear-parametric evaluators need to give meaningful bounds. The
    rounding-up to whole robots is not interesting for the demo.

    Resources:
      total_cost   = (target_throughput / (speed * payload)) * unit_cost
      total_energy = target_range * energy_per_km * 24 hours
    """
    s = robot["speed"]
    p = robot["payload"]
    c = robot["unit_cost"]
    eperkm = robot["energy_per_km"]
    capacity = s * p

    return AlgebraicDP(F, R, {
        "total_cost":   lambda f, cap=capacity, uc=c: (
            f["target_throughput"] / cap
        ) * uc,
        "total_energy": lambda f, ek=eperkm: f["target_range"] * ek * 24.0,
    })


# ---------------------------------------------------------------------------
# The catalogue: 200 candidate robot types
# ---------------------------------------------------------------------------


def make_catalog(n: int = 200, seed: int = 42) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    out: List[Dict[str, float]] = []
    for i in range(n):
        s = rng.uniform(5, 30)
        p = rng.uniform(1, 20)
        c = rng.uniform(500, 5000)
        e = rng.uniform(0.05, 0.5)
        out.append({
            "name":              f"r{i:03d}",
            "speed":             s,
            "payload":           p,
            "unit_cost":         c,
            "energy_per_km":     e,
            # Derived monotone-friendly feature: cost-per-capacity.
            # total_cost is exactly proportional to this (= throughput * c/(s*p)).
            "cost_per_capacity": c / (s * p),
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run():
    candidates = make_catalog()
    mission = {"target_throughput": 100.0, "target_range": 50.0}

    # ------------ Exhaustive baseline ------------
    print(f"=== Exhaustive baseline ({len(candidates)} candidates) ===")
    points: List[Dict[str, float]] = []
    for c in candidates:
        a = solve(make_dp(c), mission).antichain
        pt = list(a.points)[0]
        pt = {**pt, "name": c["name"]}
        points.append(pt)
    # True Pareto front
    pareto = []
    for p in points:
        dominated = any(
            q["total_cost"] <= p["total_cost"]
            and q["total_energy"] <= p["total_energy"]
            and (q["total_cost"] < p["total_cost"]
                 or q["total_energy"] < p["total_energy"])
            for q in points
        )
        if not dominated:
            pareto.append(p)
    pareto.sort(key=lambda x: x["total_cost"])
    print(f"True Pareto front: {len(pareto)} non-dominated candidates")
    for p in pareto:
        print(f"  {p['name']}: cost={p['total_cost']:7.2f}, "
              f"energy={p['total_energy']:6.2f}")
    print()

    # ------------ Online with three evaluators ------------
    evaluators = [
        ("Lipschitz (L=300/30)", LipschitzEvaluator(
            features=["speed", "payload", "unit_cost", "energy_per_km"],
            r_components=["total_cost", "total_energy"],
            L={"total_cost": 300.0, "total_energy": 30.0},
        )),
        ("Monotonicity (cost_per_capacity, energy_per_km)", MonotonicityEvaluator(
            features=["cost_per_capacity", "energy_per_km"],
            r_components=["total_cost", "total_energy"],
        )),
        ("LinearParametric (confidence=3.0)", LinearParametricEvaluator(
            features=["speed", "payload", "unit_cost", "energy_per_km"],
            r_components=["total_cost", "total_energy"],
            confidence=3.0,
            min_obs=5,
        )),
    ]

    all_results = []
    for name, ev in evaluators:
        res = solve_online(make_dp, mission,
                           candidates=candidates,
                           evaluator=ev,
                           verbose=0)
        all_results.append((name, res))

        recovered_pareto = set(c["name"] for p in pareto for c in candidates
                               if c["name"] == p["name"])
        found_pareto_names = set(candidates[i]["name"] for i in res.incumbent_ids)
        missing = recovered_pareto - found_pareto_names
        correct = "OK" if not missing else f"MISSED {sorted(missing)}"

        print(f"=== Online with {name} ===")
        print(f"  evaluated     = {res.n_evaluated:>3} / {res.n_candidates}")
        print(f"  eliminated    = {res.n_eliminated:>3}")
        print(f"  antichain size= {len(res.antichain)}")
        print(f"  Pareto recovery: {correct}")
        print()

    # ------------ Visualisation ------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pareto_names = {p["name"] for p in pareto}
    for ax, (label, res) in zip(axes, all_results):
        evaluated = {candidates[i]["name"] for i in res.evaluated_ids}
        eliminated = {candidates[i]["name"] for i in res.eliminated_ids}

        xs_e, ys_e = [], []   # evaluated, non-Pareto
        xs_x, ys_x = [], []   # eliminated
        xs_u, ys_u = [], []   # unexplored (budget hit; none here)
        xs_p, ys_p = [], []   # Pareto-optimal

        for c in candidates:
            x = c["cost_per_capacity"]
            y = c["energy_per_km"]
            if c["name"] in pareto_names:
                xs_p.append(x); ys_p.append(y)
            elif c["name"] in evaluated:
                xs_e.append(x); ys_e.append(y)
            elif c["name"] in eliminated:
                xs_x.append(x); ys_x.append(y)
            else:
                xs_u.append(x); ys_u.append(y)

        ax.scatter(xs_x, ys_x, c="lightgrey", s=18,
                   label=f"eliminated ({len(eliminated)})", edgecolor="none")
        ax.scatter(xs_u, ys_u, c="white", s=18, edgecolor="grey",
                   label=f"unexplored ({len(xs_u)})")
        ax.scatter(xs_e, ys_e, c="steelblue", s=22,
                   label=f"evaluated ({len(evaluated) - len(pareto_names & evaluated)})",
                   edgecolor="none")
        ax.scatter(xs_p, ys_p, c="crimson", s=55, marker="*",
                   label=f"Pareto-optimal ({len(pareto_names)})",
                   edgecolor="black", linewidth=0.5)

        ax.set_xlabel("cost_per_capacity (USD per pkg/h)")
        ax.set_ylabel("energy_per_km (kWh/km)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("Online elimination across 200 robot candidates", fontsize=12)
    fig.tight_layout()
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fleet_online.png")
    fig.savefig(out_path, dpi=110)
    print(f"Saved figure to {out_path}")

    # Convergence: the incumbent antichain grows as more candidates are
    # evaluated.
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for label, res in all_results:
        sizes = [h["evaluated"] for h in res.history]
        antichain_sizes = [len(h["antichain"]) for h in res.history]
        ax2.plot(sizes, antichain_sizes, marker=".", label=label)
    ax2.set_xlabel("inner solves performed")
    ax2.set_ylabel("incumbent antichain size")
    ax2.set_title("Incumbent antichain grows with evaluations")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    conv_path = os.path.join(out_dir, "fleet_online_convergence.png")
    fig2.savefig(conv_path, dpi=110)
    print(f"Saved convergence figure to {conv_path}")


if __name__ == "__main__":
    run()
