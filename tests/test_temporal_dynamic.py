"""Tests for the temporal (Case 1) and dynamic (Case 2) co-design layers.

These cover the switching scheduler in :mod:`codesign.temporal` and the
finite-horizon architecture DP in :mod:`codesign.dynamic`, including the
regression for the grid-snap masking hazard (an out-of-bounds carried
resource must not be rescued by snapping back into range).

The tests build tiny architectures by hand so they exercise the temporal
machinery, not the larger example models, and they run without numpy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    AlgebraicDP,
    Architecture,
    Epoch,
    Ports,
    Reals,
    Stage,
    StateGrid,
    System,
    rollout,
    solve_and_rollout,
    solve_dynamic,
    solve_schedule,
)


# ---------------------------------------------------------------------------
# Helpers: tiny architectures
# ---------------------------------------------------------------------------
def _capped_producer(unit_cost, max_yield, name):
    """A producer whose cost is unit_cost*demand, infeasible above max_yield."""
    sys_ = System(name)
    demand = sys_.provides("demand", unit="u")
    sys_.requires("cost", unit="USD")
    sys_.add(
        "prod",
        AlgebraicDP(
            F=Ports({"demand": Reals(unit="u")}),
            R=Ports({"cost": Reals(unit="USD")}),
            equations={
                "cost": lambda f, uc=unit_cost, my=max_yield: (
                    f["demand"] * uc if f["demand"] <= my else float("inf")
                )
            },
        ),
    ).demand >= demand
    sys_.constrain("cost", lambda x: x["prod.cost"])
    return sys_.build()


def _fuel_producer(unit_cost, burn_div, name):
    """A producer reporting cost and a fuel draw = demand / burn_div."""
    sys_ = System(name)
    demand = sys_.provides("demand", unit="u")
    sys_.requires("cost", unit="USD")
    sys_.requires("fuel", unit="L")
    sys_.add(
        "p",
        AlgebraicDP(
            F=Ports({"demand": Reals(unit="u")}),
            R=Ports({"cost": Reals(unit="USD"), "fuel": Reals(unit="L")}),
            equations={
                "cost": lambda f, uc=unit_cost: f["demand"] * uc,
                "fuel": lambda f, bd=burn_div: f["demand"] / bd,
            },
        ),
    ).demand >= demand
    sys_.constrain("cost", lambda x: x["p.cost"])
    sys_.constrain("fuel", lambda x: x["p.fuel"])
    return sys_.build()


def _cost(point):
    return point["cost"]


# ---------------------------------------------------------------------------
# Case 1: switching scheduler
# ---------------------------------------------------------------------------
def test_schedule_greedy_without_switch_cost():
    """With zero switch cost the schedule is the epoch-local greedy choice."""
    cheap = Architecture("cheap", _capped_producer(1.0, 10.0, "cheap"))
    strong = Architecture("strong", _capped_producer(2.0, 100.0, "strong"))
    epochs = [
        Epoch("e1", {"demand": 5.0}),
        Epoch("e2", {"demand": 5.0}),
        Epoch("e3", {"demand": 50.0}),  # exceeds cheap's ceiling
        Epoch("e4", {"demand": 5.0}),
    ]
    sched = solve_schedule(epochs, [cheap, strong], cost_fn=_cost, switch_cost=0.0)
    assert sched.feasible
    assert sched.schedule == ["cheap", "cheap", "strong", "cheap"]
    assert sched.n_switches == 2
    print("schedule greedy:", sched.schedule)


def test_schedule_switch_cost_suppresses_switching():
    """A large switch cost should reduce the number of architecture changes."""
    cheap = Architecture("cheap", _capped_producer(1.0, 10.0, "cheap"))
    strong = Architecture("strong", _capped_producer(2.0, 100.0, "strong"))
    epochs = [
        Epoch("e1", {"demand": 5.0}),
        Epoch("e2", {"demand": 5.0}),
        Epoch("e3", {"demand": 50.0}),
        Epoch("e4", {"demand": 5.0}),
    ]
    greedy = solve_schedule(epochs, [cheap, strong], cost_fn=_cost, switch_cost=0.0)
    sticky = solve_schedule(epochs, [cheap, strong], cost_fn=_cost, switch_cost=1000.0)
    assert sticky.n_switches < greedy.n_switches
    assert sticky.feasible
    print("greedy switches:", greedy.n_switches, "sticky:", sticky.n_switches)


def test_schedule_per_epoch_candidates():
    """Per-epoch candidate lists restrict which architectures are admissible."""
    a = Architecture("a", _capped_producer(1.0, 100.0, "a"))
    b = Architecture("b", _capped_producer(1.5, 100.0, "b"))
    epochs = [
        Epoch("e1", {"demand": 5.0}, candidates=[a]),     # only a
        Epoch("e2", {"demand": 5.0}, candidates=[b]),     # only b
        Epoch("e3", {"demand": 5.0}, candidates=[a, b]),  # either
    ]
    sched = solve_schedule(epochs, cost_fn=_cost, switch_cost=0.0)
    assert sched.schedule[0] == "a"
    assert sched.schedule[1] == "b"
    assert sched.feasible
    print("per-epoch candidates:", sched.schedule)


# ---------------------------------------------------------------------------
# Case 2: dynamic architecture DP with carried resource
# ---------------------------------------------------------------------------
def _fuel_stages(n):
    eco = Architecture("eco", _fuel_producer(2.0, 10.0, "eco"))
    fast = Architecture("fast", _fuel_producer(1.0, 5.0, "fast"))
    func = lambda s: {"demand": 20.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    stages = [
        Stage(f"s{i}", functionality=func, transition=trans, admissible=adm)
        for i in range(n)
    ]
    return stages, [eco, fast]


def test_dynamic_rich_resource_prefers_cheaper_thirsty_arch():
    """With ample fuel the cheaper-but-thirstier architecture wins."""
    stages, arches = _fuel_stages(3)
    grid = StateGrid.linspace(0.0, 12.0, 25)
    res = solve_and_rollout(stages, grid, 12.0, cost_fn=_cost, architectures=arches)
    assert res.feasible
    assert res.schedule == ["fast", "fast", "fast"]
    print("rich:", res.schedule, "cost", res.total_cost)


def test_dynamic_scarce_resource_forces_thrifty_arch():
    """With scarce fuel the policy is forced onto the thriftier architecture.

    fast burns 4 L/stage (12 L over 3 stages); with only 7 L it would go
    negative, so the policy must use eco (2 L/stage, 6 L total). This is
    the regression for the grid-snap masking hazard: a transition to
    negative fuel must be rejected before snapping, not rounded up to 0.
    """
    stages, arches = _fuel_stages(3)
    grid = StateGrid.linspace(0.0, 12.0, 25)
    res = solve_and_rollout(stages, grid, 7.0, cost_fn=_cost, architectures=arches)
    assert res.feasible
    assert res.schedule == ["eco", "eco", "eco"]
    print("scarce:", res.schedule, "cost", res.total_cost)


def test_dynamic_insufficient_resource_is_infeasible():
    """Below the minimum total burn even the thriftiest path is infeasible."""
    stages, arches = _fuel_stages(3)
    grid = StateGrid.linspace(0.0, 12.0, 25)
    res = solve_and_rollout(stages, grid, 4.0, cost_fn=_cost, architectures=arches)
    # eco needs 6 L over 3 stages; 4 L cannot complete.
    assert not res.feasible
    print("insufficient: feasible =", res.feasible)


def test_dynamic_policy_queryable_off_path():
    """The returned policy is queryable at off-nominal states (closed loop)."""
    stages, arches = _fuel_stages(3)
    grid = StateGrid.linspace(0.0, 12.0, 25)
    policy = solve_dynamic(stages, grid, cost_fn=_cost, architectures=arches)
    # Cost-to-go must be finite where feasible and respect monotonicity:
    # more fuel never costs more to go.
    c_hi = policy.cost_to_go(0, 12.0)
    c_lo = policy.cost_to_go(0, 6.0)
    assert c_hi != float("inf")
    assert c_lo != float("inf")
    assert c_hi <= c_lo + 1e-9
    # An action exists at a healthy state.
    act = policy.action_at(0, 12.0)
    assert act is not None
    print("policy cost-to-go 12L:", c_hi, "6L:", c_lo)


def test_dynamic_state_dependent_functionality():
    """Stage functionality may depend on the carried state."""
    # Demand scales down as fuel runs low, so a low-fuel start faces an
    # easier mission. Verify the DP threads state through functionality.
    arch = Architecture("a", _fuel_producer(1.0, 5.0, "a"))
    func = lambda s: {"demand": 10.0 if s >= 8.0 else 5.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    stages = [Stage(f"s{i}", functionality=func, transition=trans, admissible=adm)
              for i in range(2)]
    grid = StateGrid.linspace(0.0, 10.0, 21)
    res = solve_and_rollout(stages, grid, 10.0, cost_fn=_cost, architectures=[arch])
    # At 10 L demand is 10 -> burns 2 L -> 8 L; next demand still 10 -> 2 L.
    assert res.feasible
    assert res.stages[0].point["fuel"] == 2.0
    print("state-dependent:", [s.point["fuel"] for s in res.stages])


# ---------------------------------------------------------------------------
# Script mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("ALL TEMPORAL/DYNAMIC TESTS PASSED")
