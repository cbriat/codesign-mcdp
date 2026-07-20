"""Tests for the antichain-valued sequential co-design DP.

Covers the antichain-valued Bellman recursion in :mod:`codesign.sequential`:
the multi-point value front, the front-equals-reachable-frontier identity
(Q2), the scalar reduction to classical DP, the monotonicity guard
(H1/H2, Q1) including a genuine perishable non-monotone case, and reset
detection with horizon factorisation (Q3). Runs without numpy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    AlgebraicDP,
    Architecture,
    Ports,
    Reals,
    SeqStage,
    StateGrid,
    System,
    check_monotonicity,
    detect_resets,
    factorise_at_resets,
    join_combine,
    solve_sequential,
    sum_combine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _point_arch(cost, co2, fuel, name):
    """An architecture emitting one fixed (cost, co2, fuel) point."""
    s = System(name)
    demand = s.provides("demand", unit="u")
    s.requires("cost", unit="")
    s.requires("co2", unit="")
    s.requires("fuel", unit="L")
    s.add(
        "p",
        AlgebraicDP(
            F=Ports({"demand": Reals(unit="u")}),
            R=Ports({"cost": Reals(), "co2": Reals(), "fuel": Reals(unit="L")}),
            equations={
                "cost": lambda f, c=cost: c,
                "co2": lambda f, e=co2: e,
                "fuel": lambda f, fu=fuel: fu,
            },
        ),
    ).demand >= demand
    s.constrain("cost", lambda x: x["p.cost"])
    s.constrain("co2", lambda x: x["p.co2"])
    s.constrain("fuel", lambda x: x["p.fuel"])
    return s.build()


def _two_mode_stages(n):
    clean = Architecture("clean", _point_arch(10.0, 1.0, 2.0, "clean"))
    cheap = Architecture("cheap", _point_arch(2.0, 8.0, 2.0, "cheap"))
    func = lambda s: {"demand": 1.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    stages = [
        SeqStage(f"s{i}", functionality=func, transition=trans,
                 admissible=adm, candidates=[clean, cheap])
        for i in range(n)
    ]
    return stages


# ---------------------------------------------------------------------------
# Antichain-valued value and the front=frontier identity (Q2)
# ---------------------------------------------------------------------------
def test_value_is_full_pareto_front():
    """Over 3 stages of two incomparable modes the value has 4 Pareto points."""
    stages = _two_mode_stages(3)
    grid = StateGrid.linspace(0.0, 6.0, 13)
    res = solve_sequential(stages, grid, cost_axes=["cost", "co2"],
                           initial_state=6.0, combine=sum_combine)
    assert res.feasible
    totals = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in res.value)
    # 0,1,2,3 clean choices give 4 incomparable totals.
    assert res.width == 4
    assert (30.0, 3.0) in totals   # all clean
    assert (6.0, 24.0) in totals   # all cheap
    assert (22.0, 10.0) in totals  # one cheap
    assert (14.0, 17.0) in totals  # two cheap
    print("front:", totals)


def test_front_equals_reachable_frontier():
    """Q2: the value front equals the brute-force reachable minimal totals."""
    stages = _two_mode_stages(3)
    grid = StateGrid.linspace(0.0, 6.0, 13)
    res = solve_sequential(stages, grid, cost_axes=["cost", "co2"],
                           initial_state=6.0, combine=sum_combine)
    # Brute force: enumerate all 2^3 choice sequences, sum, take Min.
    import itertools
    modes = {"clean": (10.0, 1.0), "cheap": (2.0, 8.0)}
    raw = []
    for seq in itertools.product(modes, repeat=3):
        c = sum(modes[m][0] for m in seq)
        e = sum(modes[m][1] for m in seq)
        raw.append((c, e))
    # Pareto-minimal of raw:
    front = []
    for (c, e) in raw:
        if not any((c2 <= c and e2 <= e and (c2, e2) != (c, e)) for (c2, e2) in raw):
            front.append((c, e))
    front = sorted(set(front))
    got = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in res.value)
    assert got == front, (got, front)
    print("reachable frontier matches:", got)


def test_scalar_reduction_to_classical_dp():
    """Width-1 resource poset reduces to single-valued DP."""
    mono = Architecture("mono", _point_arch(5.0, 0.0, 2.0, "mono"))
    func = lambda s: {"demand": 1.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    stages = [SeqStage(f"s{i}", functionality=func, transition=trans,
                       admissible=adm, candidates=[mono]) for i in range(3)]
    grid = StateGrid.linspace(0.0, 6.0, 13)
    res = solve_sequential(stages, grid, cost_axes=["cost"], initial_state=6.0)
    assert res.width == 1
    assert round(list(res.value)[0]["cost"], 1) == 15.0
    print("scalar:", [round(p["cost"], 1) for p in res.value])


def _case_study_modes():
    """The three survey-mission modes of ``examples/20_sequential_codesign.py``.

    Same (cost, CO2, energy) numbers as the example's ``eco`` / ``balanced`` /
    ``rapid`` architectures; the carried-energy axis is the ``fuel`` port of
    :func:`_point_arch`.
    """
    return [
        Architecture("eco", _point_arch(10.0, 1.0, 4.0, "eco")),
        Architecture("balanced", _point_arch(6.0, 4.0, 5.0, "balanced")),
        Architecture("rapid", _point_arch(2.0, 9.0, 7.0, "rapid")),
    ]


def test_scalar_reduction_matches_solve_dynamic_on_case_study():
    """On the paper's case-study instance the scalar reduction equals classical DP.

    Runs the four-leg survey mission (energy budget 30, 61-node grid) through
    (a) ``solve_sequential`` restricted to the single cost axis, whose value
    antichain must collapse to a singleton, and (b) the scalar-valued
    :func:`codesign.dynamic.solve_dynamic` on the same stages with
    ``cost_fn = point["cost"]``. The two must return the same optimum, which is
    the concrete instance behind the scalar-reduction claim in the paper's
    case study (the general statement is the scalar-reduction proposition).
    """
    from codesign import dynamic

    modes = _case_study_modes()
    cap = 30.0
    func = lambda s: {"demand": 1.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    grid = StateGrid.linspace(0.0, cap, 61)

    seq_stages = [SeqStage(f"leg_{i+1}", functionality=func, transition=trans,
                           admissible=adm, candidates=modes) for i in range(4)]
    seq = solve_sequential(seq_stages, grid, cost_axes=["cost"],
                           initial_state=cap, combine=sum_combine)
    assert seq.feasible
    assert seq.width == 1, seq.width
    seq_cost = list(seq.value)[0]["cost"]

    dyn_stages = [dynamic.Stage(name=f"leg_{i+1}", functionality=func,
                                transition=trans, admissible=adm,
                                candidates=modes) for i in range(4)]
    policy = dynamic.solve_dynamic(dyn_stages, grid,
                                   cost_fn=lambda p: p["cost"])
    dyn = dynamic.rollout(policy, dyn_stages, cap)
    assert dyn.feasible

    assert round(seq_cost, 6) == round(dyn.total_cost, 6), (seq_cost, dyn.total_cost)
    assert round(seq_cost, 1) == 8.0        # four rapid legs at cost 2 each
    assert dyn.schedule == ["rapid"] * 4
    print("scalar reduction vs solve_dynamic:", seq_cost, "==", dyn.total_cost,
          dyn.schedule)


def test_infeasible_when_resource_exhausted():
    """Not enough carried resource to complete the horizon is infeasible."""
    stages = _two_mode_stages(3)  # needs 6 L total
    grid = StateGrid.linspace(0.0, 6.0, 13)
    res = solve_sequential(stages, grid, cost_axes=["cost", "co2"],
                           initial_state=4.0, combine=sum_combine)
    assert not res.feasible
    print("exhausted feasible:", res.feasible)


# ---------------------------------------------------------------------------
# Q1: monotonicity guard
# ---------------------------------------------------------------------------
def test_monotonicity_holds_for_consistent_stage():
    """A consistently oriented stage passes (H1) and (H2)."""
    stages = _two_mode_stages(3)
    grid = StateGrid.linspace(0.0, 6.0, 13)
    rep = check_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    assert rep.monotone_value_guaranteed
    print("consistent:", rep)


def test_monotonicity_catches_perishable():
    """A U-shaped (perishable) demand in state is flagged as non-monotone."""
    arch = Architecture("a", _point_arch(0.0, 0.0, 1.0, "a"))

    # Rebuild with state-coupled demand: cost grows with demand, demand is
    # U-shaped in state (interior optimum) -> genuinely non-monotone.
    def make():
        s = System("u")
        demand = s.provides("demand", unit="u")
        s.requires("cost", unit="")
        s.requires("fuel", unit="L")
        s.add("p", AlgebraicDP(
            F=Ports({"demand": Reals(unit="u")}),
            R=Ports({"cost": Reals(), "fuel": Reals(unit="L")}),
            equations={"cost": lambda f: f["demand"], "fuel": lambda f: 1.0},
        )).demand >= demand
        s.constrain("cost", lambda x: x["p.cost"])
        s.constrain("fuel", lambda x: x["p.fuel"])
        return s.build()

    a = Architecture("a", make())
    func_U = lambda state: {"demand": 1.0 + abs(state - 2.0)}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    stages = [SeqStage(f"s{i}", functionality=func_U, transition=trans,
                       admissible=adm, candidates=[a]) for i in range(2)]
    grid = StateGrid.linspace(0.0, 4.0, 9)
    rep = check_monotonicity(stages, grid, cost_axes=["cost"])
    assert not rep.h1_ok
    assert not rep.monotone_value_guaranteed
    print("perishable:", rep, rep.h1_violations[:2])


# ---------------------------------------------------------------------------
# Q3: reset detection and factorisation
# ---------------------------------------------------------------------------
def test_no_resets_without_quiescence():
    stages = _two_mode_stages(3)
    grid = StateGrid.linspace(0.0, 6.0, 13)
    resets = detect_resets(stages, grid, cost_axes=["cost", "co2"])
    assert resets == []
    print("no resets:", resets)


def test_reset_detected_and_factorised():
    """A stage with a quiescent transition is detected and splits the horizon."""
    clean = Architecture("clean", _point_arch(10.0, 1.0, 2.0, "clean"))
    cheap = Architecture("cheap", _point_arch(2.0, 8.0, 2.0, "cheap"))
    func = lambda s: {"demand": 1.0}
    trans = lambda s, p: s - p["fuel"]
    reset_trans = lambda s, p: 0.0  # quiescent
    adm = lambda s: s >= -1e-9
    stages = [
        SeqStage("a", functionality=func, transition=trans, admissible=adm,
                 candidates=[clean, cheap]),
        SeqStage("reset", functionality=func, transition=reset_trans,
                 admissible=adm, candidates=[clean, cheap]),
        SeqStage("b", functionality=func, transition=trans, admissible=adm,
                 candidates=[clean, cheap]),
    ]
    grid = StateGrid.linspace(0.0, 6.0, 13)
    resets = detect_resets(stages, grid, cost_axes=["cost", "co2"])
    runs = factorise_at_resets(stages, resets)
    assert 1 in resets
    assert runs == [(0, 1), (2, 2)]
    print("resets:", resets, "runs:", runs)


def test_join_combine_renewable():
    """Renewable combination (join) keeps the front bounded across stages."""
    # Two modes; with join, the total is the per-axis max over stages, so
    # the front cannot exceed the single-stage front width.
    stages = _two_mode_stages(4)
    grid = StateGrid.linspace(0.0, 8.0, 17)
    res = solve_sequential(stages, grid, cost_axes=["cost", "co2"],
                           initial_state=8.0, combine=join_combine)
    # Under join the reachable totals are {max picks}; the front stays the
    # two single-stage points plus their combinations, bounded (<= a small
    # constant), not growing with the 4-stage horizon.
    assert res.feasible
    assert res.width <= 4
    print("join width:", res.width,
          "front:", sorted((round(p["cost"],1), round(p["co2"],1)) for p in res.value))


# ---------------------------------------------------------------------------
# Fix: full-resource-poset dedup (no cost-projection collision)
# ---------------------------------------------------------------------------
def _equal_cost_arch(fuel, name):
    """A stage-0 mode emitting a fixed (cost, co2)=(5, 5) but a chosen fuel."""
    s = System(name)
    d = s.provides("demand", unit="u")
    s.requires("cost")
    s.requires("co2")
    s.requires("fuel", unit="L")
    s.add("p", AlgebraicDP(
        F=Ports({"demand": Reals(unit="u")}),
        R=Ports({"cost": Reals(), "co2": Reals(), "fuel": Reals(unit="L")}),
        equations={"cost": lambda f: 5.0, "co2": lambda f: 5.0,
                   "fuel": lambda f, fu=fuel: fu},
    )).demand >= d
    s.constrain("cost", lambda x: x["p.cost"])
    s.constrain("co2", lambda x: x["p.co2"])
    s.constrain("fuel", lambda x: x["p.fuel"])
    return s.build()


def _state_priced_arch(name):
    """A stage-1 mode whose (cost, co2) depend on the incoming state."""
    s = System(name)
    d = s.provides("demand", unit="u")
    s.requires("cost")
    s.requires("co2")
    s.requires("fuel", unit="L")
    s.add("p", AlgebraicDP(
        F=Ports({"demand": Reals(unit="u")}),
        R=Ports({"cost": Reals(), "co2": Reals(), "fuel": Reals(unit="L")}),
        equations={"cost": lambda f: f["demand"],
                   "co2": lambda f: 50.0 - f["demand"],
                   "fuel": lambda f: 0.0},
    )).demand >= d
    s.constrain("cost", lambda x: x["p.cost"])
    s.constrain("co2", lambda x: x["p.co2"])
    s.constrain("fuel", lambda x: x["p.fuel"])
    return s.build()


def test_equal_cost_incomparable_resource_points_both_survive():
    """Two modes with equal cost but different fuel must NOT be deduplicated.

    Regression for the cost-projection dedup bug: the backward pass keyed
    candidate points on their rounded cost projection, so two points with
    equal (cost, co2) but a different carried axis (fuel) collided and one
    was silently dropped. Here the different fuel leads to different
    successor states, hence -- because stage 1 is priced off the state -- to
    two incomparable whole-horizon totals. Both must appear in the front.
    On the pre-fix code the front had width 1 (one path dropped).
    """
    A = Architecture("modeA", _equal_cost_arch(2.0, "modeA"))
    B = Architecture("modeB", _equal_cost_arch(3.0, "modeB"))
    M = Architecture("priced", _state_priced_arch("priced"))
    stage0 = SeqStage("s0", functionality=lambda s: {"demand": 1.0},
                      transition=lambda s, p: s - p["fuel"],
                      admissible=lambda s: s >= -1e-9, candidates=[A, B])
    stage1 = SeqStage("s1", functionality=lambda s: {"demand": s},
                      transition=lambda s, p: s - p["fuel"],
                      admissible=lambda s: s >= -1e-9, candidates=[M])
    grid = StateGrid.linspace(0.0, 12.0, 13)
    res = solve_sequential([stage0, stage1], grid, cost_axes=["cost", "co2"],
                           initial_state=10.0, combine=sum_combine)
    totals = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in res.value)
    # fuel=2 -> state 8 -> stage1 (8, 42) -> total (13, 47)
    # fuel=3 -> state 7 -> stage1 (7, 43) -> total (12, 48)   (incomparable)
    assert res.width == 2, totals
    assert (13.0, 47.0) in totals  # the modeA (fuel=2) path, dropped pre-fix
    assert (12.0, 48.0) in totals  # the modeB (fuel=3) path
    print("dedup survivors:", totals)


# ---------------------------------------------------------------------------
# Q1: extended certificate -- joint (H2) resource slice and (H3) down-set
# ---------------------------------------------------------------------------
def test_consistent_stage_passes_all_hypotheses():
    """The canonical consumable transitions satisfy (H1), joint (H2), (H3)."""
    stages = _two_mode_stages(3)
    grid = StateGrid.linspace(0.0, 6.0, 13)
    rep = check_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    assert rep.h1_ok and rep.h2_ok and rep.h2_joint_ok and rep.h3_ok
    assert rep.monotone_value_guaranteed
    print("all-hypotheses:", rep)


def test_renewable_stage_passes_all_hypotheses():
    """The renewable (join) canonical case also satisfies every hypothesis."""
    # check_monotonicity certifies the maps (h_k, phi_k, admissibility); the
    # monoid choice does not enter the certificate, so the same stages that
    # are solved with join_combine also pass every hypothesis.
    stages = _two_mode_stages(4)
    grid = StateGrid.linspace(0.0, 8.0, 17)
    res = solve_sequential(stages, grid, cost_axes=["cost", "co2"],
                           initial_state=8.0, combine=join_combine)
    assert res.feasible
    rep = check_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    assert rep.monotone_value_guaranteed
    assert rep.h2_joint_ok and rep.h3_ok


def test_monotonicity_flags_resource_slice_of_h2():
    """A transition decreasing in the resource violates joint (H2).

    The stage map is strictly harder as the state grows (committed
    orientation only), and the transition subtracts a resource axis that
    grows with the state, so a larger resource drives the successor *lower*
    -- the opposite of what joint monotonicity demands in that orientation.
    The state slice of (H2) still holds; only the resource slice fails.
    """
    def make():
        s = System("g")
        d = s.provides("demand", unit="u")
        s.requires("cost")
        s.requires("fuel", unit="L")
        s.add("p", AlgebraicDP(
            F=Ports({"demand": Reals(unit="u")}),
            R=Ports({"cost": Reals(), "fuel": Reals(unit="L")}),
            equations={"cost": lambda f: f["demand"], "fuel": lambda f: f["demand"]},
        )).demand >= d
        s.constrain("cost", lambda x: x["p.cost"])
        s.constrain("fuel", lambda x: x["p.fuel"])
        return s.build()

    g = Architecture("g", make())
    func = lambda state: {"demand": 1.0 + state}   # harder as state grows
    trans = lambda s, p: s - p["fuel"]             # decreasing in the resource
    adm = lambda s: s >= -1e-9
    stages = [SeqStage(f"s{i}", functionality=func, transition=trans,
                       admissible=adm, candidates=[g]) for i in range(2)]
    grid = StateGrid.linspace(0.0, 4.0, 9)
    rep = check_monotonicity(stages, grid, cost_axes=["cost"])
    assert rep.h1_ok           # consistently oriented (committed)
    assert rep.h2_ok           # state slice fine
    assert not rep.h2_joint_ok  # resource slice fails
    assert not rep.monotone_value_guaranteed
    assert rep.h2_joint_violations
    print("resource-slice:", rep, rep.h2_joint_violations[:2])


def test_monotonicity_flags_non_downset_admissibility():
    """An admissible region that is not a down-set violates (H3).

    The stage map is constant in the state (so (H1) and the state slice of
    (H2) hold in both orientations), but the admissible predicate excludes a
    middle band, so the admissible set is neither down- nor up-closed: no
    orientation makes it a down-set. (H3) must be flagged.
    """
    clean = Architecture("clean", _point_arch(10.0, 1.0, 2.0, "clean"))
    cheap = Architecture("cheap", _point_arch(2.0, 8.0, 2.0, "cheap"))
    func = lambda s: {"demand": 1.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: not (1.5 < s < 3.0)   # hole: neither up- nor down-closed
    stages = [SeqStage(f"s{i}", functionality=func, transition=trans,
                       admissible=adm, candidates=[clean, cheap])
              for i in range(2)]
    grid = StateGrid.linspace(0.0, 6.0, 13)
    rep = check_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    assert rep.h1_ok and rep.h2_ok and rep.h2_joint_ok
    assert not rep.h3_ok
    assert not rep.monotone_value_guaranteed
    assert rep.h3_violations
    print("non-down-set admissibility:", rep, rep.h3_violations[:2])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("ALL SEQUENTIAL TESTS PASSED")
