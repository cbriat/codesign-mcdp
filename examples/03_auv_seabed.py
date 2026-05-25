"""
Example: AUV seabed surveying (Sec. VIII, Example 10 of Censi 2015).

An autonomous underwater vehicle (AUV) must sweep an area A [m^2] at a
fixed depth, moving at velocity v [m/s] with sensor field of view r [m].
The co-design constraints are:

    1. Coverage:      v * T * r  >=  k * A      (k geometry constant)
    2. Actuation:     P_act  >=  psi(v)         (hydrodynamic drag)
    3. Sensing:       P_sens >=  chi(r)         (sensor power)
    4. Energetics:    E      >=  (P_act + P_sens) * T
    5. Cost:          $      >=  sensor_cost(r) + ...

The structure is intrinsically cyclic: choosing a larger v shortens T but
costs more actuation power; a wider r shortens T but costs more sensor
power. The Kleene fixed-point iteration finds the Pareto front of
(time, energy, cost) for each value of the functionality A.

For clarity, we close the loop on the design variables (v, r) and expose
T, E, $ on the outer R.
"""
from __future__ import annotations

import math

from codesign import (
    Antichain,
    FunctionDP,
    Loop,
    Ports,
    Reals,
    solve,
    minimize_cost,
)


def make_auv():
    """Construct the AUV seabed surveying MCDP.

    Outer functionality is the area A to cover. Outer resources are the
    mission time T, energy E, and cost $. Internal design variables v
    (velocity) and r (sensor radius) are closed over by the loop.
    """
    # --- physical constants --------------------------------------------
    K_GEOM = 1.0          # coverage geometry constant
    V_MAX = 3.0           # max velocity (m/s)
    R_MAX = 5.0           # max sensor field of view (m)
    PSI_A = 30.0          # P_act = PSI_A * v^3  (drag scales as v^3)
    CHI_A = 50.0          # P_sens = CHI_A * r   (sensor power scales with r)
    SENSOR_COST_A = 200.0 # cost scales linearly in r

    # --- poset and DP definitions --------------------------------------
    # Loop axis 'design' bundles v and r; T, E, $ are the visible outputs.
    Design = Ports({
        "v": Reals(unit="m/s"),
        "r": Reals(unit="m"),
    })
    F = Ports({
        "A": Reals(unit="m^2"),
        "design": Design,
    })
    R = Ports({
        "design": Design,
        "T": Reals(unit="s"),
        "E": Reals(unit="J"),
        "cost": Reals(unit="$"),
    })

    def h(f):
        A = f["A"]
        v_in = f["design"]["v"]
        r_in = f["design"]["r"]

        # Saturate the design variables against physical caps. Once at the
        # cap the corresponding "minimal design that covers A" may be
        # infeasible, and the Kleene iteration will lift the loop axis to
        # ⊤ to signal that.
        if v_in == math.inf or r_in == math.inf:
            return Antichain.singleton(R, {
                "design": {"v": math.inf, "r": math.inf},
                "T": math.inf, "E": math.inf, "cost": math.inf,
            })

        # The minimal v, r that still cover A given the loop input v_in, r_in.
        # We must have v * T * r >= k * A with T = A / (v*r/k) so we just
        # need v, r that respect the loop constraints v >= v_in, r >= r_in
        # and produce a finite T. Use v = max(v_in, eps), r = max(r_in, eps).
        v = max(float(v_in), 0.1)
        r = max(float(r_in), 0.5)
        if v > V_MAX or r > R_MAX:
            return Antichain.singleton(R, {
                "design": {"v": math.inf, "r": math.inf},
                "T": math.inf, "E": math.inf, "cost": math.inf,
            })

        T = K_GEOM * A / (v * r)
        P_act = PSI_A * (v ** 3)
        P_sens = CHI_A * r
        E = (P_act + P_sens) * T
        cost = SENSOR_COST_A * r

        # The point reported as 'design' is the loop-closure value; the
        # Kleene iteration uses this to drive v_in, r_in upward. The
        # antichain at the fixed point is the Pareto front over (T, E, $).
        # Since the inner h is single-valued in this formulation, we
        # generate a small spread by enumerating a few candidate (v, r)
        # pairs above (v_in, r_in): each trades velocity for sensor width.
        pts = []
        for v_try in (v, min(v * 1.3, V_MAX), min(v * 1.7, V_MAX)):
            for r_try in (r, min(r * 1.3, R_MAX), min(r * 1.7, R_MAX)):
                if v_try < v_in or r_try < r_in:
                    continue
                T_try = K_GEOM * A / (v_try * r_try)
                E_try = (PSI_A * v_try ** 3 + CHI_A * r_try) * T_try
                cost_try = SENSOR_COST_A * r_try
                pts.append({
                    "design": {"v": v_try, "r": r_try},
                    "T": T_try, "E": E_try, "cost": cost_try,
                })
        return Antichain.from_set(R, pts)

    inner = FunctionDP(F=F, R=R, h_fn=h, name="auv_inner")
    return Loop(inner, axis="design")


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------


def show(result, label):
    print(f"\n{label}")
    print(f"   iters={result.iterations}, feasible={result.feasible}")
    if not result.feasible:
        return
    for p in result.antichain.points:
        T = p["T"]
        E = p["E"]
        cost = p["cost"]
        print(f"   T={T:.0f}s, E={E/1000:.1f}kJ, $={cost:.0f}")


if __name__ == "__main__":
    print("AUV seabed surveying (Sec. VIII, Example 10):")
    auv = make_auv()

    # Three areas at increasing scale.
    for A in (100.0, 1000.0, 10_000.0):
        result = solve(auv, {"A": A}, max_iter=50)
        show(result, f"Area = {A:g} m^2")

        # Optimize a scalar cost: $1 per second + $0.05 per kJ + $1 per $.
        # This collapses the Pareto front to one engineering choice.
        best = minimize_cost(
            result,
            cost_fn=lambda r: r["T"] + 0.05 * (r["E"] / 1000.0) + r["cost"],
        )
        if best is not None:
            print(
                f"   best by composite cost: T={best['T']:.0f}s, "
                f"E={best['E']/1000:.1f}kJ, $={best['cost']:.0f}"
            )
