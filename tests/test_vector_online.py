"""Tests for vector-state DP, precompute-then-DP, and online co-design.

Covers the vector-state generalisation in :mod:`codesign.vector_dp` and
:mod:`codesign.state`, the precompute-then-DP helpers in
:mod:`codesign.sequential` (the Formula 1 structure), and the closed-loop
controller in :mod:`codesign.online_codesign`. Runs without numpy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codesign import (
    AlgebraicDP,
    Architecture,
    ContinuousAxis,
    DiscreteAxis,
    Ports,
    Reals,
    StateGrid,
    System,
    VecStage,
    VectorStateGrid,
    check_vector_monotonicity,
    dp_over_catalog,
    precompute_catalog,
    resolve_at,
    run_online_codesign,
    solve_vector_sequential,
    state_as_dict,
    state_get,
    sum_combine,
)
from codesign.state import make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _point_arch(cost, co2, fuel, name):
    s = System(name)
    d = s.provides("demand", unit="u")
    s.requires("cost", unit="")
    s.requires("co2", unit="")
    s.requires("fuel", unit="L")
    s.add("p", AlgebraicDP(
        F=Ports({"demand": Reals(unit="u")}),
        R=Ports({"cost": Reals(), "co2": Reals(), "fuel": Reals(unit="L")}),
        equations={"cost": lambda f, c=cost: c, "co2": lambda f, e=co2: e,
                   "fuel": lambda f, fu=fuel: fu})).demand >= d
    s.constrain("cost", lambda x: x["p.cost"])
    s.constrain("co2", lambda x: x["p.co2"])
    s.constrain("fuel", lambda x: x["p.fuel"])
    return s.build()


CLEAN = Architecture("clean", _point_arch(10.0, 1.0, 2.0, "clean"))
CHEAP = Architecture("cheap", _point_arch(2.0, 8.0, 2.0, "cheap"))
_COST = lambda r: r["cost"]


# ---------------------------------------------------------------------------
# state.py primitives
# ---------------------------------------------------------------------------
def test_state_vec_canonical_and_hashable():
    a = make_state(fuel=12.0, flag=0)
    b = make_state(flag=0, fuel=12.0)
    assert a == b                      # order-independent
    assert hash(a) == hash(b)          # usable as dict key
    assert state_get(a, "fuel") == 12.0
    assert state_as_dict(a) == {"fuel": 12.0, "flag": 0}


def test_vector_grid_product_and_order():
    grid = VectorStateGrid([
        ContinuousAxis("fuel", 0.0, 4.0, 3),      # 0,2,4
        DiscreteAxis("flag", [0, 1], order=[0, 1]),
    ])
    assert len(grid) == 6
    lo = make_state(fuel=0.0, flag=0)
    hi = make_state(fuel=4.0, flag=1)
    assert grid.leq(lo, hi)
    assert not grid.leq(hi, lo)
    # snapping onto the grid
    snapped = grid.snap({"fuel": 1.2, "flag": 1})
    assert state_get(snapped, "fuel") == 2.0
    # out-of-bounds rejected
    assert not grid.in_bounds({"fuel": 5.0, "flag": 0})


def test_discrete_axis_incomparable_without_order():
    ax = DiscreteAxis("regime", ["a", "b"])   # no order
    assert ax.leq("a", "a")
    assert not ax.leq("a", "b")
    assert not ax.leq("b", "a")


# ---------------------------------------------------------------------------
# vector_dp.py
# ---------------------------------------------------------------------------
def _fuel_flag_stages(n):
    func = lambda sv: {"demand": 1.0}

    def trans(sv, pt):
        d = state_as_dict(sv)
        nf = d["fuel"] - pt["fuel"]
        return {"fuel": nf, "flag": 1 if nf < 3.0 else d["flag"]}

    adm = lambda sv: state_get(sv, "fuel") >= -1e-9
    return [VecStage(f"s{i}", functionality=func, transition=trans,
                     admissible=adm, candidates=[CLEAN, CHEAP])
            for i in range(n)]


def test_vector_dp_reproduces_full_front():
    grid = VectorStateGrid([
        ContinuousAxis("fuel", 0.0, 6.0, 13),
        DiscreteAxis("flag", [0, 1], order=[0, 1]),
    ])
    res = solve_vector_sequential(_fuel_flag_stages(3), grid,
                                  cost_axes=["cost", "co2"],
                                  initial_state={"fuel": 6.0, "flag": 0},
                                  combine=sum_combine)
    totals = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in res.value)
    assert res.feasible and res.width == 4
    assert (30.0, 3.0) in totals and (6.0, 24.0) in totals
    print("vector front:", totals)


def test_vector_dp_single_axis_matches_scalar():
    # A one-axis vector grid must reproduce the scalar sequential result.
    from codesign import solve_sequential, SeqStage
    # scalar
    func = lambda s: {"demand": 1.0}
    trans_s = lambda s, p: s - p["fuel"]
    adm_s = lambda s: s >= -1e-9
    sstages = [SeqStage(f"s{i}", functionality=func, transition=trans_s,
                        admissible=adm_s, candidates=[CLEAN, CHEAP])
               for i in range(3)]
    sgrid = StateGrid.linspace(0.0, 6.0, 13)
    sres = solve_sequential(sstages, sgrid, cost_axes=["cost", "co2"],
                            initial_state=6.0, combine=sum_combine)
    # vector, one axis
    vgrid = VectorStateGrid([ContinuousAxis("fuel", 0.0, 6.0, 13)])
    vfunc = lambda sv: {"demand": 1.0}
    vtrans = lambda sv, p: {"fuel": state_get(sv, "fuel") - p["fuel"]}
    vadm = lambda sv: state_get(sv, "fuel") >= -1e-9
    vstages = [VecStage(f"s{i}", functionality=vfunc, transition=vtrans,
                        admissible=vadm, candidates=[CLEAN, CHEAP])
               for i in range(3)]
    vres = solve_vector_sequential(vstages, vgrid, cost_axes=["cost", "co2"],
                                   initial_state={"fuel": 6.0}, combine=sum_combine)
    st = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in sres.value)
    vt = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in vres.value)
    assert st == vt, (st, vt)
    print("scalar==vector single-axis:", vt)


def test_vector_monotonicity_clean_and_flagged():
    # Clean monotone two-continuous-axis transition passes.
    grid = VectorStateGrid([ContinuousAxis("fuel", 0.0, 6.0, 7),
                            ContinuousAxis("wear", 0.0, 10.0, 6)])
    func = lambda sv: {"demand": 1.0}

    def trans(sv, pt):
        d = state_as_dict(sv)
        return {"fuel": d["fuel"] - pt["fuel"], "wear": d["wear"] + pt["fuel"]}

    adm = lambda sv: state_get(sv, "fuel") >= -1e-9 and state_get(sv, "wear") <= 10.0 + 1e-9
    stages = [VecStage(f"s{i}", functionality=func, transition=trans,
                       admissible=adm, candidates=[CLEAN, CHEAP]) for i in range(2)]
    rep = check_vector_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    assert rep.monotone_value_guaranteed
    assert rep.h2_joint_ok and rep.h3_ok  # joint (H2) and (H3) also hold
    print("clean vector monotone:", rep)


def _res_priced_arch(name):
    """One-point mode whose cost/fuel grow with the incoming demand."""
    s = System(name)
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


def test_vector_monotonicity_flags_resource_slice_of_h2():
    """The vector guard flags a transition that is not jointly monotone."""
    g = Architecture("g", _res_priced_arch("g"))
    grid = VectorStateGrid([ContinuousAxis("fuel", 0.0, 4.0, 9)])
    func = lambda sv: {"demand": 1.0 + state_get(sv, "fuel")}  # harder as fuel grows
    trans = lambda sv, pt: {"fuel": state_get(sv, "fuel") - pt["fuel"]}
    adm = lambda sv: state_get(sv, "fuel") >= -1e-9
    stages = [VecStage(f"s{i}", functionality=func, transition=trans,
                       admissible=adm, candidates=[g]) for i in range(2)]
    rep = check_vector_monotonicity(stages, grid, cost_axes=["cost"])
    assert rep.h1_ok and rep.h2_ok
    assert not rep.h2_joint_ok
    assert not rep.monotone_value_guaranteed
    print("vector resource-slice flagged:", rep)


def test_vector_monotonicity_flags_non_downset_admissibility():
    """The vector guard flags an admissible region that is not a down-set."""
    grid = VectorStateGrid([ContinuousAxis("fuel", 0.0, 6.0, 13)])
    func = lambda sv: {"demand": 1.0}
    trans = lambda sv, pt: {"fuel": state_get(sv, "fuel") - pt["fuel"]}
    adm = lambda sv: not (1.5 < state_get(sv, "fuel") < 3.0)  # excluded band
    stages = [VecStage(f"s{i}", functionality=func, transition=trans,
                       admissible=adm, candidates=[CLEAN, CHEAP]) for i in range(2)]
    rep = check_vector_monotonicity(stages, grid, cost_axes=["cost", "co2"])
    assert rep.h1_ok and rep.h2_ok and rep.h2_joint_ok
    assert not rep.h3_ok
    assert not rep.monotone_value_guaranteed
    print("vector non-down-set flagged:", rep)


# ---------------------------------------------------------------------------
# precompute-then-DP (F1 structure)
# ---------------------------------------------------------------------------
def test_precompute_catalog_and_dp():
    cat = precompute_catalog([CLEAN, CHEAP], {"demand": 1.0}, ["cost", "co2"])
    assert len(cat) == 2
    names = {n for n, _ in cat}
    assert names == {"clean", "cheap"}
    grid = StateGrid.linspace(0.0, 6.0, 13)
    res = dp_over_catalog([cat, cat, cat], grid, cost_axes=["cost", "co2"],
                          initial_state=6.0,
                          transition=lambda s, p: s - p["fuel"],
                          combine=sum_combine,
                          admissible=lambda s: s >= -1e-9)
    totals = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in res.value)
    assert res.feasible and res.width == 4
    print("dp-over-catalog:", totals)


def test_precompute_matches_live_resolve():
    # The precompute-then-DP must equal solve_sequential when the co-design
    # is state-independent (the F1 regime).
    from codesign import solve_sequential, SeqStage
    func = lambda s: {"demand": 1.0}
    trans = lambda s, p: s - p["fuel"]
    adm = lambda s: s >= -1e-9
    sstages = [SeqStage(f"s{i}", functionality=func, transition=trans,
                        admissible=adm, candidates=[CLEAN, CHEAP]) for i in range(3)]
    grid = StateGrid.linspace(0.0, 6.0, 13)
    live = solve_sequential(sstages, grid, cost_axes=["cost", "co2"],
                            initial_state=6.0, combine=sum_combine)
    cat = precompute_catalog([CLEAN, CHEAP], {"demand": 1.0}, ["cost", "co2"])
    pre = dp_over_catalog([cat, cat, cat], grid, cost_axes=["cost", "co2"],
                          initial_state=6.0, transition=trans, combine=sum_combine,
                          admissible=adm)
    lt = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in live.value)
    pt = sorted((round(p["cost"], 1), round(p["co2"], 1)) for p in pre.value)
    assert lt == pt, (lt, pt)
    print("precompute==live:", pt)


# ---------------------------------------------------------------------------
# online_codesign.py
# ---------------------------------------------------------------------------
def test_resolve_at_picks_cheapest():
    name, pt, c = resolve_at([CLEAN, CHEAP], {"demand": 1.0}, _COST)
    assert name == "cheap" and c == 2.0
    assert pt is not None


def test_online_loop_tracks_changing_requirement():
    # A demand-dependent cost: when demand is high, only "clean" (higher
    # ceiling) stays feasible; wire an infeasibility ceiling into cheap.
    def capped_arch(cost, ceiling, name):
        s = System(name)
        d = s.provides("demand", unit="u")
        s.requires("cost", unit=""); s.requires("fuel", unit="L")
        s.add("p", AlgebraicDP(
            F=Ports({"demand": Reals(unit="u")}),
            R=Ports({"cost": Reals(), "fuel": Reals(unit="L")}),
            equations={"cost": lambda f, c=cost, cc=ceiling:
                       f["demand"] * c if f["demand"] <= cc else float("inf"),
                       "fuel": lambda f: 1.0})).demand >= d
        s.constrain("cost", lambda x: x["p.cost"])
        s.constrain("fuel", lambda x: x["p.fuel"])
        return s.build()

    weak = Architecture("weak", capped_arch(1.0, 5.0, "weak"))     # cheap, low ceiling
    strong = Architecture("strong", capped_arch(2.0, 100.0, "strong"))  # costly, high ceiling

    # Requirement spikes above the weak ceiling at step 2.
    demands = [3.0, 4.0, 20.0, 3.0]

    def sensor(t, prev):
        return prev if prev is not None else 0.0

    def requirement(t, state):
        return {"demand": demands[t]}

    def plant(t, state, arch, point):
        return state  # trivial process

    res = run_online_codesign([weak, strong], n_steps=4, sensor=sensor,
                              requirement=requirement, plant=plant, cost_fn=_COST)
    assert res.feasible
    # weak is cheaper when feasible; strong is forced only at the spike.
    assert res.schedule == ["weak", "weak", "strong", "weak"], res.schedule
    print("online schedule:", res.schedule)


def test_online_loop_feedback_state_advances():
    # The plant advances a state that the sensor reads back; verify the
    # loop threads measured state through requirement and plant.
    def sensor(t, prev):
        return 0.0 if prev is None else prev

    seen = []

    def requirement(t, state):
        seen.append(state)
        return {"demand": 1.0}

    def plant(t, state, arch, point):
        return state + 1.0  # deterministic advance

    res = run_online_codesign([CLEAN, CHEAP], n_steps=3, sensor=sensor,
                              requirement=requirement, plant=plant, cost_fn=_COST,
                              initial_state=0.0)
    # states seen by the requirement should be 0,1,2 as the plant advances.
    assert seen == [0.0, 1.0, 2.0], seen
    print("feedback states:", seen)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("ALL VECTOR/ONLINE TESTS PASSED")
