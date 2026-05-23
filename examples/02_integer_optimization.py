"""
Example: integer optimization from Sec. VI-D of Censi (2015).

Find the minimal pair (x, y) in N x N such that

    x + y >= ceil(sqrt(x)) + ceil(sqrt(y)) + c

for a given constant c. This is the canonical MCDP example whose Kleene
ascent (Fig. 36 of the paper) converges in a handful of steps from the
seed antichain {(0, 0)}.

We model it with one FunctionDP wrapped in a Loop:

    inner :  (c, x_in, y_in)  ->  (x_out, y_out)
    where x_out = ceil(sqrt(x_in)) + c1, y_out = ceil(sqrt(y_in)) + c2,
    and c1 + c2 = c is enumerated to populate the antichain.

The outer loop closes x_out >= x_in, y_out >= y_in. Because the splitting
of c into (c1, c2) is non-deterministic, the antichain at the fixed point
has multiple incomparable points -- the Pareto front for the resource
pair (x, y).
"""
from __future__ import annotations

import math

from codesign import (
    Antichain,
    FunctionDP,
    Loop,
    NamedProduct,
    Naturals,
    SolveResult,
    solve,
)


def make_problem():
    """Build the MCDP for the sqrt+ceil+sum example."""
    N = Naturals()

    # Inner DP: functionality is (c, x_in, y_in), resource is (x_out, y_out).
    # The loop will identify x_in with x_out and y_in with y_out.
    F = NamedProduct({"c": N, "x_in": N, "y_in": N})
    R = NamedProduct({"x_out": N, "y_out": N})

    def h(f):
        c = f["c"]
        x_in = f["x_in"]
        y_in = f["y_in"]
        if c == math.inf or x_in == math.inf or y_in == math.inf:
            return Antichain.singleton(R, {"x_out": math.inf, "y_out": math.inf})

        # ceil(sqrt) of the current loop values.
        sx = math.isqrt(int(x_in))
        if sx * sx < int(x_in):
            sx += 1
        sy = math.isqrt(int(y_in))
        if sy * sy < int(y_in):
            sy += 1

        # The constraint is x_out + y_out >= sx + sy + c, with x_out >= sx
        # and y_out >= sy individually impossible to guarantee without
        # splitting the slack. Enumerate every split (c1, c2) of the deficit:
        # x_out = sx + c1, y_out = sy + c2, c1 + c2 = (sx + sy + c) - x_in - y_in
        # but clipped at zero. To stay sound we enumerate from the strongest
        # form: the antichain of (x, y) achieving x + y = sx + sy + c with
        # x >= sx, y >= sy.
        target = sx + sy + int(c)
        pts = []
        for x_out in range(sx, target - sy + 1):
            y_out = target - x_out
            if y_out < sy:
                break
            pts.append({"x_out": x_out, "y_out": y_out})
        if not pts:
            # Should never happen for c >= 0, but guard anyway.
            return Antichain.empty(R)
        return Antichain.from_set(R, pts)

    inner = FunctionDP(F=F, R=R, h_fn=h, name="sqrt_sum_inner")

    # Close the loop on both x and y. The Loop primitive in this library
    # supports a single axis; chain two loops to identify both pairs.
    # We rewire the inner so that its R also carries a copy of (x_in, y_in)
    # under the same names used by the outer F. To keep things simple we
    # express it as a single Loop on a composite axis: build an inner where
    # F has 'c' plus the loop variables under the SAME names they will close
    # over, and R exposes both the loop variables and a report.
    return inner


def make_looped(c_value: int):
    """Build the Loop DP that closes x_out -> x_in and y_out -> y_in."""
    N = Naturals()
    # We rewrite the inner so that F = (c, xy) and R = (xy) with xy = NamedProduct
    # to use a single loop axis.
    XY = NamedProduct({"x": N, "y": N})
    F = NamedProduct({"c": N, "xy": XY})
    R = NamedProduct({"xy": XY, "xy_report": XY})

    def h(f):
        c = int(f["c"])
        x_in = f["xy"]["x"]
        y_in = f["xy"]["y"]
        if x_in == math.inf or y_in == math.inf:
            top = {"x": math.inf, "y": math.inf}
            return Antichain.singleton(R, {"xy": top, "xy_report": top})

        sx = math.isqrt(int(x_in))
        if sx * sx < int(x_in):
            sx += 1
        sy = math.isqrt(int(y_in))
        if sy * sy < int(y_in):
            sy += 1
        target = sx + sy + c

        pts = []
        for x_out in range(sx, target - sy + 1):
            y_out = target - x_out
            if y_out < sy:
                break
            pts.append({
                "xy": {"x": x_out, "y": y_out},
                "xy_report": {"x": x_out, "y": y_out},
            })
        if not pts:
            return Antichain.empty(R)
        return Antichain.from_set(R, pts)

    inner = FunctionDP(F=F, R=R, h_fn=h, name=f"sqrt_sum(c={c_value})")
    return Loop(inner, axis="xy")


# ---------------------------------------------------------------------------
# Run the example and print the Kleene trace for a few values of c.
# ---------------------------------------------------------------------------


def pretty_xy(p):
    return f"({p['xy_report']['x']}, {p['xy_report']['y']})"


def run(c_value: int, show_trace: bool = True):
    looped = make_looped(c_value)
    result = solve(looped, {"c": c_value}, max_iter=50, record_trace=show_trace)
    print(f"\nc = {c_value}: iters = {result.iterations}, feasible = {result.feasible}")
    if show_trace and result.trace:
        for k, entry in enumerate(result.trace):
            A = entry.antichain
            pts = ", ".join(
                f"({p['xy']['x']}, {p['xy']['y']})" for p in A.points
            )
            print(f"   S_{k}: {{ {pts} }}")
    pts = ", ".join(pretty_xy(p) for p in result.antichain.points)
    print(f"   M(c={c_value}) = {{ {pts} }}")
    return result


if __name__ == "__main__":
    print("Sec. VI-D: minimal (x, y) in N x N such that")
    print("   x + y >= ceil(sqrt(x)) + ceil(sqrt(y)) + c")

    # c = 0 has the trivial solution (0, 0).
    run(0)
    # c = 1 has two minimal solutions, (1, 0) and (0, 1).
    run(1)
    # c = 4 reproduces the picture in Fig. 41 of the paper: an antichain
    # containing several incomparable (x, y) pairs.
    run(4)
    # c = 20 matches Fig. 36 of the paper.
    run(20, show_trace=False)
