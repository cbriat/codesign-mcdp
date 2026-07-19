"""Validation-guard tests for the three release-polish error-handling fixes.

Each fix is covered by (a) at least one test asserting the new error is raised
with an informative message (matching on the interpolated names), and (b) one
test asserting that valid usage still works unchanged.

    1. ODE_DP     : a dict-valued (named) integrator state is rejected early.
    2. MCDP.loop_on : looping on an axis missing from provides()/requires()
                      raises a message explaining the both-sides requirement.
    3. Series     : mismatched dp1.R / dp2.F ports are caught upfront.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from codesign import (
    AlgebraicDP,
    CatalogDP,
    ContinuousAxis,
    Discrete,
    DiscreteAxis,
    FunctionDP,
    ODE_DP,
    Ports,
    Reals,
    Stage,
    StateGrid,
    System,
    UncertainDP,
    VectorStateGrid,
    loop,
    make_state,
    par,
    series,
    solve,
    solve_dynamic,
    state_get,
)
from codesign.mcdpl import MCDP


# ---------------------------------------------------------------------------
# Fix 1: ODE_DP dict-state guard
# ---------------------------------------------------------------------------


def test_ode_dp_dict_state_rejected():
    """A dict-valued state must fail early with a clear, named message."""
    F = Ports({"delta_T": Reals(unit="K")})
    R = Ports({"power": Reals(unit="W")})
    heater = ODE_DP(
        F=F,
        R=R,
        rhs=lambda x, t, f: 0.8 * f["delta_T"] - x,
        extract=lambda x: {"power": float(x)},
        mode="steady_state",
        x0_fn=lambda f: {"temp": 0.0, "pressure": 0.0},
        name="heater_ode",
    )
    with pytest.raises(TypeError) as excinfo:
        solve(heater, {"delta_T": 20.0})
    msg = str(excinfo.value)
    assert "sequence of floats" in msg, msg
    assert "dict" in msg, msg
    # the offending keys must be interpolated in
    assert "pressure" in msg and "temp" in msg, msg
    # a concrete fix must be suggested
    assert "positionally" in msg or "x[0]" in msg, msg
    assert "heater_ode" in msg, msg


def test_ode_dp_dict_state_rejected_final_value():
    """The same guard applies in the final_value integrator path."""
    F = Ports({"delta_T": Reals(unit="K")})
    R = Ports({"power": Reals(unit="W")})
    heater = ODE_DP(
        F=F,
        R=R,
        rhs=lambda x, t, f: 0.8 * f["delta_T"] - x,
        extract=lambda x: {"power": float(x)},
        mode="final_value",
        x0_fn=lambda f: {"temp": 0.0},
        name="ode",
    )
    with pytest.raises(TypeError, match=r"sequence of floats"):
        solve(heater, {"delta_T": 20.0})


def test_ode_dp_scalar_state_still_works():
    """A scalar-state ODE (the common case) is unaffected."""
    F = Ports({"delta_T": Reals(unit="K")})
    R = Ports({"power": Reals(unit="W")})
    heater = ODE_DP(
        F=F,
        R=R,
        rhs=lambda x, t, f: 0.8 * f["delta_T"] - x,
        extract=lambda x: {"power": float(x)},
        mode="steady_state",
        x0_fn=lambda f: 0.0,
        name="heater_ode",
    )
    res = solve(heater, {"delta_T": 20.0})
    pt = list(res.antichain)[0]
    assert abs(pt["power"] - 0.8 * 20.0) < 1e-6


def test_ode_dp_vector_state_still_works():
    """A positional (list) vector state is accepted and integrated."""
    F = Ports({"target": Reals()})
    R = Ports({"a": Reals(), "b": Reals()})
    dp = ODE_DP(
        F=F,
        R=R,
        rhs=lambda x, t, f: [f["target"] - x[0], 0.5 * f["target"] - x[1]],
        extract=lambda x: {"a": float(x[0]), "b": float(x[1])},
        mode="final_value",
        t_end=20.0,
        n_steps=400,
        x0_fn=lambda f: [0.0, 0.0],
        name="vec_ode",
    )
    res = solve(dp, {"target": 10.0})
    pt = list(res.antichain)[0]
    assert abs(pt["a"] - 10.0) < 0.5
    assert abs(pt["b"] - 5.0) < 0.5


# ---------------------------------------------------------------------------
# Fix 2: MCDP.loop_on both-sides guard
# ---------------------------------------------------------------------------


def test_loop_on_missing_provides():
    """loop_on on an axis absent from provides() explains the requirement."""
    m = MCDP("m")
    m.requires("battery_mass", unit="kg")
    with pytest.raises(ValueError) as excinfo:
        m.loop_on("battery_mass")
    msg = str(excinfo.value)
    assert "battery_mass" in msg, msg
    assert "provides()" in msg, msg
    assert "BOTH" in msg, msg
    # current declarations interpolated in
    assert "requires() = ['battery_mass']" in msg, msg


def test_loop_on_missing_requires():
    """loop_on on an axis absent from requires() mentions the mirror pattern."""
    m = MCDP("m")
    m.provides("battery_mass", unit="kg")
    with pytest.raises(ValueError) as excinfo:
        m.loop_on("battery_mass")
    msg = str(excinfo.value)
    assert "battery_mass" in msg, msg
    assert "requires()" in msg, msg
    assert "BOTH" in msg, msg
    # the report_mass mirror hint from examples/06
    assert "report_mass" in msg, msg
    assert "examples/06" in msg, msg


def test_loop_on_valid_still_works():
    """A correctly declared loop builds and solves unchanged."""
    with MCDP("m") as m:
        m.provides("endurance", unit="s")
        m.provides("battery_mass", unit="kg")  # loop variable
        m.requires("battery_mass", unit="kg")  # loop axis
        m.constraint(
            "battery_mass",
            lambda f: 0.01 * f["endurance"] + 0.1 * f["battery_mass"],
        )
        m.loop_on("battery_mass")
    dp = m.build()
    res = solve(dp, {"endurance": 100.0})
    assert not res.antichain.is_empty()
    assert not res.antichain.has_any_top()


# ---------------------------------------------------------------------------
# Fix 3: Series upfront interface check
# ---------------------------------------------------------------------------


def _battery():
    F1 = Ports({"capacity": Reals()})
    R1 = Ports({"mass": Reals()})
    return AlgebraicDP(F1, R1, {"mass": lambda f: f["capacity"] / 1.8e6}, name="battery")


def test_series_port_mismatch_rejected():
    """A dp2 requiring a port dp1 does not produce fails upfront, clearly."""
    battery = _battery()
    F2 = Ports({"power": Reals()})  # dp2 wants 'power', battery makes 'mass'
    R2 = Ports({"cost": Reals()})
    pricing = AlgebraicDP(F2, R2, {"cost": lambda f: f["power"] * 10.0}, name="pricing")
    with pytest.raises(ValueError) as excinfo:
        series(battery, pricing)
    msg = str(excinfo.value)
    assert "interface mismatch" in msg, msg
    assert "battery" in msg and "pricing" in msg, msg
    # the mismatched port names on each side must be reported
    assert "power" in msg, msg
    assert "mass" in msg, msg
    assert "missing" in msg.lower(), msg


def test_series_exact_match_still_works():
    """The canonical battery -> pricing chain still composes and solves."""
    battery = _battery()
    F2 = Ports({"mass": Reals()})
    R2 = Ports({"cost": Reals()})
    pricing = AlgebraicDP(F2, R2, {"cost": lambda f: f["mass"] * 10.0}, name="pricing")
    chain = series(battery, pricing)
    res = solve(chain, {"capacity": 3.6e6})
    pt = list(res.antichain)[0]
    assert abs(pt["cost"] - (3.6e6 / 1.8e6) * 10.0) < 1e-6


def test_series_extra_dp1_resource_allowed():
    """Extra dp1.R ports not consumed by dp2 are permitted (subset, not equality)."""
    F1 = Ports({"capacity": Reals()})
    R1 = Ports({"mass": Reals(), "heat": Reals()})  # 'heat' unused downstream
    battery = AlgebraicDP(
        F1, R1,
        {"mass": lambda f: f["capacity"] / 1.8e6, "heat": lambda f: f["capacity"] * 1e-7},
        name="battery",
    )
    F2 = Ports({"mass": Reals()})
    R2 = Ports({"cost": Reals()})
    pricing = AlgebraicDP(F2, R2, {"cost": lambda f: f["mass"] * 10.0}, name="pricing")
    chain = series(battery, pricing)  # must NOT raise
    res = solve(chain, {"capacity": 3.6e6})
    pt = list(res.antichain)[0]
    assert abs(pt["cost"] - (3.6e6 / 1.8e6) * 10.0) < 1e-6


# ===========================================================================
# Full error-message audit (release-polish WP-K): message-content tests for
# the upgraded raises across codesign/. Each asserts the new message states
# what went wrong and interpolates the offending name/value where available.
# ===========================================================================


# ---- ODE_DP mode validation (new validation, approved task 1) ----


def test_ode_dp_invalid_mode_rejected_at_construction():
    """An invalid mode fails at construction, not silently on the first solve."""
    F = Ports({"delta_T": Reals()})
    R = Ports({"power": Reals()})
    with pytest.raises(ValueError) as excinfo:
        ODE_DP(
            F=F, R=R,
            rhs=lambda x, t, f: 0.0,
            extract=lambda x: {"power": float(x)},
            mode="bogus",
            name="heater_ode",
        )
    msg = str(excinfo.value)
    assert "final_value" in msg and "steady_state" in msg, msg
    assert "bogus" in msg, msg          # offending value interpolated
    assert "heater_ode" in msg, msg     # dp name interpolated


def test_ode_dp_valid_modes_still_construct():
    F = Ports({"delta_T": Reals()})
    R = Ports({"power": Reals()})
    for mode in ("final_value", "steady_state"):
        ODE_DP(
            F=F, R=R,
            rhs=lambda x, t, f: f["delta_T"] - x,
            extract=lambda x: {"power": float(x)},
            mode=mode,
        )


# ---- UncertainDP mode ----


def test_uncertain_dp_invalid_mode_rejected():
    F = Ports({"a": Reals()})
    R = Ports({"b": Reals()})
    lo = AlgebraicDP(F, R, {"b": lambda f: f["a"]}, name="lo")
    hi = AlgebraicDP(F, R, {"b": lambda f: 2 * f["a"]}, name="hi")
    with pytest.raises(ValueError) as excinfo:
        UncertainDP(F, R, lo, hi, mode="middle")
    msg = str(excinfo.value)
    assert "lower" in msg and "upper" in msg, msg
    assert "middle" in msg, msg


# ---- AlgebraicDP / CatalogDP ----


def test_algebraic_dp_r_not_ports():
    with pytest.raises(TypeError, match=r"requires R to be a Ports"):
        AlgebraicDP(Ports({"a": Reals()}), Reals(), {})


def test_algebraic_dp_missing_equations():
    with pytest.raises(ValueError) as excinfo:
        AlgebraicDP(
            Ports({"a": Reals()}),
            Ports({"b": Reals(), "c": Reals()}),
            {"b": lambda f: 0.0},
        )
    msg = str(excinfo.value)
    assert "missing equations" in msg, msg
    assert "c" in msg, msg


def test_catalog_dp_requires_ports():
    with pytest.raises(TypeError, match=r"requires Ports F and R"):
        CatalogDP(Reals(), Ports({"b": Reals()}), [])


# ---- Parallel composition ----


def _dp(fname, rname, name):
    F = Ports({fname: Reals()})
    R = Ports({rname: Reals()})
    return AlgebraicDP(F, R, {rname: lambda f: 0.0}, name=name)


def test_parallel_needs_ports_functionality():
    bad = FunctionDP(Reals(), Ports({"r": Reals()}), lambda f: {"r": 0.0}, name="bad")
    ok = _dp("a", "b", "ok")
    with pytest.raises(TypeError) as excinfo:
        par(bad, ok)
    msg = str(excinfo.value)
    assert "Ports functionality" in msg, msg
    assert "bad" in msg and "ok" in msg, msg


def test_parallel_functionality_name_clash():
    a = _dp("x", "m", "A")
    b = _dp("x", "n", "B")
    with pytest.raises(ValueError) as excinfo:
        par(a, b)
    msg = str(excinfo.value)
    assert "functionality port names" in msg, msg
    assert "x" in msg and "A" in msg and "B" in msg, msg


def test_parallel_resource_name_clash():
    a = _dp("x", "m", "A")
    b = _dp("y", "m", "B")
    with pytest.raises(ValueError, match=r"resource port names"):
        par(a, b)


# ---- Loop composition ----


def test_loop_needs_ports_functionality():
    bad = FunctionDP(Reals(), Ports({"r": Reals()}), lambda f: {"r": 0.0}, name="badloop")
    with pytest.raises(TypeError, match=r"Ports functionality space"):
        loop(bad, "r")


def test_loop_axis_must_be_on_both_sides():
    inner = AlgebraicDP(
        Ports({"a": Reals()}), Ports({"b": Reals()}),
        {"b": lambda f: f["a"]}, name="inner",
    )
    with pytest.raises(ValueError) as excinfo:
        loop(inner, "a")
    msg = str(excinfo.value)
    assert "must appear in both F and R" in msg, msg
    assert "inner" in msg and "'a'" in msg, msg


# ---- MCDP build guards ----


def test_mcdp_build_needs_provides():
    m = MCDP("x")
    m.requires("r")
    with pytest.raises(ValueError, match=r"at least one provides"):
        m.build()


def test_mcdp_build_needs_requires():
    m = MCDP("x")
    m.provides("p")
    with pytest.raises(ValueError, match=r"at least one requires"):
        m.build()


def test_mcdp_build_missing_constraint():
    m = MCDP("x")
    m.provides("p")
    m.requires("r")
    with pytest.raises(ValueError) as excinfo:
        m.build()
    msg = str(excinfo.value)
    assert "have no constraint" in msg, msg
    assert "r" in msg, msg


# ---- Ports / posets ----


def test_ports_needs_component():
    with pytest.raises(ValueError, match=r"at least one component"):
        Ports({})


def test_ports_make_missing_and_unknown():
    P = Ports({"a": Reals(), "b": Reals()})
    with pytest.raises(ValueError) as e1:
        P.make(a=1.0)
    assert "missing component" in str(e1.value) and "b" in str(e1.value)
    with pytest.raises(ValueError) as e2:
        P.make(a=1.0, b=2.0, c=3.0)
    assert "unknown component" in str(e2.value) and "c" in str(e2.value)


def test_discrete_no_canonical_bottom_top():
    d = Discrete(["x", "y"])
    with pytest.raises(ValueError, match=r"no canonical bottom"):
        d.bottom()
    with pytest.raises(ValueError, match=r"no canonical top"):
        d.top()


# ---- StateGrid / axes ----


def test_state_grid_empty_and_linspace():
    with pytest.raises(ValueError, match=r"at least one node"):
        StateGrid([])
    with pytest.raises(ValueError, match=r"at least one node"):
        StateGrid.linspace(0.0, 1.0, 0)


def test_continuous_axis_needs_node():
    with pytest.raises(ValueError, match=r"at least one node"):
        ContinuousAxis("c", 0.0, 1.0, 0)


def test_discrete_axis_needs_value():
    with pytest.raises(ValueError, match=r"at least one value"):
        DiscreteAxis("d", [])


def test_vector_state_grid_empty_and_duplicate():
    with pytest.raises(ValueError, match=r"at least one axis"):
        VectorStateGrid([])
    with pytest.raises(ValueError) as excinfo:
        VectorStateGrid([ContinuousAxis("c", 0, 1, 2), ContinuousAxis("c", 0, 1, 2)])
    msg = str(excinfo.value)
    assert "must be unique" in msg and "c" in msg, msg


def test_state_get_missing_axis():
    s = make_state(fuel=1.0, flag=0)
    with pytest.raises(KeyError) as excinfo:
        state_get(s, "charge")
    assert "no axis" in str(excinfo.value), excinfo.value


# ---- online picker resolution ----


def test_online_unknown_picker():
    from codesign.online import _resolve_picker
    with pytest.raises(ValueError, match=r"unknown picker"):
        _resolve_picker("nonsense")
    with pytest.raises(ValueError) as excinfo:
        _resolve_picker(("nonsense", {}))
    assert "unknown picker" in str(excinfo.value), excinfo.value


# ---- System guards ----


def test_system_needs_requires():
    dp = AlgebraicDP(
        Ports({"a": Reals()}), Ports({"b": Reals()}),
        {"b": lambda f: f["a"]}, name="mod",
    )
    s = System("s")
    s.add("m", dp)
    with pytest.raises(ValueError, match=r"at least one requires"):
        s.build()


def test_system_module_name_no_dot():
    dp = AlgebraicDP(
        Ports({"a": Reals()}), Ports({"b": Reals()}),
        {"b": lambda f: f["a"]}, name="mod",
    )
    s = System("s")
    with pytest.raises(ValueError, match=r"reserved for port references"):
        s.add("bad.name", dp)


# ---- Dynamic (representative of the shared no-candidates message family;
#      sequential / vector_dp / temporal siblings share the identical text) ----


def test_dynamic_stage_no_candidates():
    stage = Stage(name="leg", functionality=lambda st: {}, transition=lambda st, p: st)
    with pytest.raises(ValueError) as excinfo:
        solve_dynamic([stage], StateGrid([0.0]), cost_fn=lambda p: 0.0)
    msg = str(excinfo.value)
    assert "no candidates" in msg and "leg" in msg, msg
    assert "architectures" in msg, msg


# ---- Uncertainty (needs numpy) ----


def test_box_lo_gt_hi():
    pytest.importorskip("numpy")
    from codesign import Box
    with pytest.raises(ValueError) as excinfo:
        Box(x=(5.0, 1.0))
    msg = str(excinfo.value)
    assert "lo > hi" in msg, msg
    assert "5.0" in msg and "1.0" in msg, msg


def test_disk_needs_two_parameters():
    pytest.importorskip("numpy")
    from codesign import Disk
    with pytest.raises(ValueError) as excinfo:
        Disk(center={"a": 0.0, "b": 0.0, "c": 0.0}, radius=1.0)
    assert "exactly 2 parameters" in str(excinfo.value), excinfo.value


def test_gaussian_copula_not_square():
    pytest.importorskip("numpy")
    from codesign import GaussianCopula
    with pytest.raises(ValueError, match=r"square matrix"):
        GaussianCopula([[1.0, 0.4]])


def test_stochastic_needs_marginal():
    pytest.importorskip("numpy")
    from codesign import Stochastic
    with pytest.raises(ValueError, match=r"at least one marginal"):
        Stochastic()


# ---- viz (needs matplotlib) ----


def test_plot_antichain_wrong_result_type():
    pytest.importorskip("matplotlib")
    from codesign import viz
    with pytest.raises(TypeError) as excinfo:
        viz.plot_antichain(42, ["a", "b"])
    assert "got int" in str(excinfo.value), excinfo.value


def test_plot_antichain_bad_axes_count():
    pytest.importorskip("matplotlib")
    from codesign import viz
    battery = _battery()
    res = solve(battery, {"capacity": 3.6e6})
    with pytest.raises(ValueError, match=r"2 or 3 R-port names"):
        viz.plot_antichain(res, ["mass"])


# ---------------------------------------------------------------------------
# Fix: CatalogDP construction guards (empty catalogue, duplicate entry names)
# ---------------------------------------------------------------------------


def _catalog_ports():
    return Ports({"speed": Reals()}), Ports({"cost": Reals()})


def test_catalogdp_empty_catalogue_rejected():
    """An empty catalogue must fail at construction, naming the DP."""
    from codesign import CatalogEntry  # noqa: F401
    F, R = _catalog_ports()
    with pytest.raises(ValueError) as excinfo:
        CatalogDP(F=F, R=R, catalog=[], name="engine_catalog")
    msg = str(excinfo.value)
    assert "engine_catalog" in msg, msg
    assert "at least one CatalogEntry" in msg, msg


def test_catalogdp_duplicate_names_rejected():
    """Two entries with the same name must be rejected, naming the clash."""
    from codesign import CatalogEntry
    F, R = _catalog_ports()
    entries = [
        CatalogEntry(provides={"speed": 1.0}, costs={"cost": 1.0}, name="A"),
        CatalogEntry(provides={"speed": 2.0}, costs={"cost": 2.0}, name="A"),
    ]
    with pytest.raises(ValueError) as excinfo:
        CatalogDP(F=F, R=R, catalog=entries, name="engine_catalog")
    msg = str(excinfo.value)
    assert "engine_catalog" in msg, msg
    assert "'A'" in msg and "duplicate" in msg, msg


def test_catalogdp_unnamed_entries_allowed():
    """Multiple unnamed (default name="") entries stay legitimate."""
    from codesign import CatalogEntry
    F, R = _catalog_ports()
    entries = [
        CatalogEntry(provides={"speed": 1.0}, costs={"cost": 1.0}),
        CatalogEntry(provides={"speed": 2.0}, costs={"cost": 2.0}),
    ]
    dp = CatalogDP(F=F, R=R, catalog=entries, name="engine_catalog")
    assert len(dp.catalog) == 2


# ---------------------------------------------------------------------------
# Fix: detect_resets aligns with the sibling "no candidates" ValueError
# ---------------------------------------------------------------------------


def test_detect_resets_no_candidates_rejected():
    """A stage with no candidates and no default architectures must raise,
    like solve_sequential / check_monotonicity, not vacuously reset."""
    from codesign import SeqStage, detect_resets
    stage = SeqStage(
        "s0",
        functionality=lambda x: {"demand": 1.0},
        transition=lambda s, p: 0.0,
    )
    grid = StateGrid.linspace(0.0, 4.0, 5)
    with pytest.raises(ValueError) as excinfo:
        detect_resets([stage], grid, cost_axes=["cost"])
    msg = str(excinfo.value)
    assert "'s0'" in msg, msg
    assert "no candidates" in msg, msg


# ---------------------------------------------------------------------------
# Fix: Naturals honours unit= in format() (previously silently ignored)
# ---------------------------------------------------------------------------


def test_naturals_unit_formatting():
    """Naturals(unit=...) prints the unit and folds it into the name,
    mirroring Reals; the default no-unit form is unchanged."""
    from codesign import Naturals
    import math
    n = Naturals(unit="parts")
    assert n.name == "N+[parts]"
    assert n.format(3) == "3 parts"
    assert n.format(3.0) == "3 parts"     # integer nature preserved
    assert n.format(math.inf) == "⊤"
    plain = Naturals()
    assert plain.name == "N+"
    assert plain.format(0) == "0"
    assert plain.format(25) == "25"


# ---------------------------------------------------------------------------
# Fix: to_dot walks .inner so Func/Neg-wrapped module refs keep their edge
# ---------------------------------------------------------------------------


def _inner_edge_system(rhs_builder):
    """Build a System whose battery.capacity constraint wraps actuator.power
    in a Func/Neg node, returning the dot source of the built DP."""
    from codesign import Module, System, viz
    from codesign.posets import Reals as _R

    class Act(Module):
        F = {"lift": _R()}
        R = {"power": _R()}
        def h(self, f):
            return {"power": f["lift"]}

    class Bat(Module):
        F = {"capacity": _R()}
        R = {"mass": _R()}
        def h(self, f):
            return {"mass": f["capacity"]}

    s = System("t")
    demand = s.provides("demand", unit="N")
    total_m = s.requires("total_mass", unit="kg")
    b = s.add("battery", Bat())
    a = s.add("actuator", Act())
    a.lift >= demand
    b.capacity >= rhs_builder(a)
    total_m >= b.mass
    dp = s.build()
    return viz.to_dot(dp)


def test_to_dot_inner_edge_sqrt():
    """A sqrt()-wrapped (Func) operand must still emit the module edge."""
    from codesign.sugar import sqrt
    dot = _inner_edge_system(lambda a: sqrt(a.power))
    # A constraint edge (color="#555") carrying the sqrt label must appear.
    assert 'color="#555"' in dot, dot
    assert 'label="sqrt(actuator.power)"' in dot, dot


def test_to_dot_inner_edge_neg():
    """A Neg-wrapped operand must still emit the module edge."""
    dot = _inner_edge_system(lambda a: -a.power)
    assert 'color="#555"' in dot, dot
    assert 'label="-actuator.power"' in dot, dot


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("All validation tests passed.")
