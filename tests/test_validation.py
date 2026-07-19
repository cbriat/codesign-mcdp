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
    ODE_DP,
    Ports,
    Reals,
    series,
    solve,
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("All validation tests passed.")
