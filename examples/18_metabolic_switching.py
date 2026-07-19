"""
Example 18: metabolic architecture switching across carbon sources (Case 1).

An organism (or an engineered production strain) lives in an environment
whose available carbon source changes over time. When glucose is
abundant it runs a glycolytic, fast-growth architecture; when only
acetate is available it must switch to a gluconeogenic / glyoxylate-shunt
architecture that is slower and biochemically more expensive to operate.
Switching between the two is not free: re-acclimation requires expressing
a different enzyme complement, which costs time and resource (a lag phase
on the classic diauxic-shift picture).

This is the first temporal co-design case: *the best architecture changes
over time because the environment changes*, and the question is the
schedule of which metabolic architecture to run in each environmental
epoch, accounting for the cost of switching.

Two architectures, each a small co-design problem
-------------------------------------------------
Both architectures take the demanded biomass production rate `mu`
(1/h, the outer functionality of the epoch) and return a scalar
`burden` (a dimensionless proxy for the proteomic / ATP cost of
sustaining that growth rate on that substrate). Inside each, an enzyme
module must supply enough catalytic flux capacity to support the demand,
and a maintenance module adds the substrate-specific overhead:

    glycolytic   : cheap per unit flux, but capped at a maximum growth
                   rate; infeasible above it. Models fast growth on
                   glucose.
    gluconeogenic: usable up to a higher growth ceiling but with a larger
                   per-flux burden and a fixed shunt overhead. Models
                   slower, costlier growth on acetate.

The environment is a sequence of epochs that alternate substrate and
demand. Because only one substrate is available per epoch, each epoch
admits only the architecture matching its substrate (expressed via the
per-epoch candidate list). The interesting coupling is therefore not
which architecture is *feasible* (that is forced by the substrate) but
what the re-acclimation cost does to the total burden, and how a brief
unfavourable epoch may be cheaper to "ride out" on the incumbent
architecture if switching back and forth would cost more than it saves.
To show that effect we also include a mixed-substrate epoch in which both
architectures are admissible, so the scheduler has a real choice.

Parameter values are illustrative, chosen to sit in the qualitative
ranges reported for E. coli diauxic growth on glucose versus acetate
(growth rate on glucose is roughly twice that on acetate; the acetate
regime carries a measurable proteomic overhead from gluconeogenesis and
the glyoxylate shunt). They are not fitted to a specific strain.

Run:  python -m examples.18_metabolic_switching
Expected output: the substrate / growth-rate environment, then the optimal
architecture schedule under low (0.05) and high (0.8) per-switch cost; the
high-cost case rides out the mixed epoch on the incumbent pathway instead
of taking two switches.
"""

from __future__ import annotations

from codesign import (
    AlgebraicDP,
    Architecture,
    Epoch,
    Ports,
    Reals,
    System,
    solve_schedule,
)


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------
def build_metabolism(
    name: str,
    *,
    burden_per_mu: float,
    mu_max: float,
    shunt_overhead: float,
) -> object:
    """Build one metabolic architecture as a small co-design problem.

    Parameters
    ----------
    name : str
        System name.
    burden_per_mu : float
        Proteomic / ATP burden incurred per unit growth rate. The
        glycolytic architecture has a low value here; the gluconeogenic
        one a higher value.
    mu_max : float
        Maximum growth rate the architecture can sustain. Demands above
        this are infeasible (the architecture cannot supply the flux).
    shunt_overhead : float
        Fixed burden paid whenever the architecture is active, independent
        of growth rate. Captures the cost of expressing an alternative
        pathway (the glyoxylate shunt for acetate).
    """
    sys_ = System(name)
    mu = sys_.provides("mu", unit="1/h")          # demanded growth rate
    sys_.requires("burden", unit="")              # proteomic/ATP proxy

    # Enzyme module: supplies catalytic flux capacity for the demand, with
    # a hard ceiling. Burden grows linearly in the demanded rate; above
    # mu_max the architecture simply cannot deliver, so burden -> inf.
    enzyme = sys_.add(
        "enzyme",
        AlgebraicDP(
            F=Ports({"mu": Reals(unit="1/h")}),
            R=Ports({"burden": Reals(unit="")}),
            equations={
                "burden": lambda f, b=burden_per_mu, mm=mu_max: (
                    f["mu"] * b if f["mu"] <= mm else float("inf")
                )
            },
        ),
    )
    enzyme.mu >= mu

    # Maintenance module: the fixed shunt overhead, independent of mu.
    maint = sys_.add(
        "maint",
        AlgebraicDP(
            F=Ports({"mu": Reals(unit="1/h")}),
            R=Ports({"burden": Reals(unit="")}),
            equations={"burden": lambda f, o=shunt_overhead: o},
        ),
    )
    maint.mu >= mu

    # Outer burden is the sum of the two contributions.
    sys_.constrain("burden", lambda x: x["enzyme.burden"] + x["maint.burden"])
    return sys_.build()


# Glycolytic (glucose) regime: cheap per unit growth, no shunt overhead,
# fast growth ceiling.
GLYCOLYTIC = Architecture(
    "glycolytic",
    build_metabolism(
        "glycolytic", burden_per_mu=1.0, mu_max=0.95, shunt_overhead=0.0
    ),
    tags={"substrate": "glucose", "pathway": "glycolysis"},
)

# Gluconeogenic (acetate) regime: costlier per unit growth, a fixed shunt
# overhead, but a higher ceiling so it can be pushed when needed.
GLUCONEOGENIC = Architecture(
    "gluconeogenic",
    build_metabolism(
        "gluconeogenic", burden_per_mu=2.2, mu_max=1.20, shunt_overhead=0.6
    ),
    tags={"substrate": "acetate", "pathway": "glyoxylate-shunt"},
)


# ---------------------------------------------------------------------------
# Environment: a sequence of substrate epochs
# ---------------------------------------------------------------------------
def build_environment() -> list:
    """A diauxic-style environment with a contestable mixed epoch.

    The organism starts in glucose, moves onto acetate, then hits a brief
    mixed-substrate epoch (either pathway admissible) that is *flanked by
    acetate on both sides*, before finally returning to glucose. The
    flanking is what makes re-acclimation cost decisive: running the
    locally cheaper glycolytic pathway for the single mixed epoch forces
    two extra switches (acetate -> glucose-type -> acetate), whereas
    riding the mixed epoch out on the incumbent gluconeogenic pathway
    costs none. Whether the lower running burden is worth those two
    switches is exactly the trade the scheduler resolves.
    """
    return [
        Epoch("glucose_1", {"mu": 0.8}, candidates=[GLYCOLYTIC]),
        Epoch("acetate_1", {"mu": 0.5}, candidates=[GLUCONEOGENIC]),
        # Mixed: both substrates present at a modest demand both can meet,
        # surrounded by acetate so a switch here is "there and back".
        Epoch("mixed", {"mu": 0.6}, candidates=[GLYCOLYTIC, GLUCONEOGENIC]),
        Epoch("acetate_2", {"mu": 0.5}, candidates=[GLUCONEOGENIC]),
        Epoch("glucose_2", {"mu": 0.8}, candidates=[GLYCOLYTIC]),
    ]


def burden_cost(point) -> float:
    """Scalar cost: the proteomic/ATP burden of the chosen design point."""
    return point["burden"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    epochs = build_environment()

    print("Metabolic architecture switching across carbon sources")
    print("=" * 60)
    print("Environment (substrate / demanded growth rate mu):")
    for ep in epochs:
        subs = "/".join(
            sorted({c.tags["substrate"] for c in ep.candidates})
        )
        print(f"  {ep.name:<10s} mu={ep.functionality['mu']:.2f}  [{subs}]")
    print()

    # Low re-acclimation cost: the scheduler is free to switch whenever an
    # epoch-local choice is cheaper. In the mixed epoch it will pick the
    # cheaper glycolytic pathway even though that means two switches
    # (acetate -> glucose-type -> acetate-then-glucose).
    sched_lo = solve_schedule(
        epochs,
        cost_fn=burden_cost,
        switch_cost=0.05,
    )
    print("Low re-acclimation cost (0.05 per switch):")
    _print_schedule(sched_lo)

    # High re-acclimation cost: switching into and back out of the
    # glycolytic pathway just for the single mixed epoch no longer pays.
    # The organism rides out the mixed epoch on the gluconeogenic pathway
    # it is already running, even though that pathway is locally costlier.
    sched_hi = solve_schedule(
        epochs,
        cost_fn=burden_cost,
        switch_cost=0.8,
    )
    print("\nHigh re-acclimation cost (0.8 per switch):")
    _print_schedule(sched_hi)

    # The takeaway: identical environment, different switching economics,
    # different optimal metabolic schedule. The mixed epoch's pathway flips
    # from glycolytic (cheap to run, worth switching for) to gluconeogenic
    # (costlier to run, but not worth two switches) as re-acclimation cost
    # rises.
    mixed_lo = sched_lo.epochs[2].architecture
    mixed_hi = sched_hi.epochs[2].architecture
    print(
        f"\nMixed-epoch pathway: {mixed_lo} (cheap switching) "
        f"-> {mixed_hi} (costly switching)"
    )


def _print_schedule(sched) -> None:
    for er in sched.epochs:
        sw = f"  (+switch {er.switch_cost:.2f})" if er.switch_cost else ""
        print(
            f"  {er.epoch:<10s} -> {er.architecture:<14s} "
            f"burden={er.running_cost:6.3f}{sw}"
        )
    print(
        f"  total burden = {sched.total_cost:.3f}, "
        f"switches = {sched.n_switches}, feasible = {sched.feasible}"
    )


if __name__ == "__main__":
    main()
