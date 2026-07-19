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

Run:  python -m examples.02_integer_optimization
Expected output: for each c, the number of Kleene iterations, the trace of
antichains S_0, S_1, ... ascending to the fixed point, and the final set of
minimal (x, y) pairs M(c).
"""
from __future__ import annotations

import math

from codesign import (
    Antichain,
    FunctionDP,
    Loop,
    Ports,
    Naturals,
    solve,
)


def make_looped(c_value: int):
    """Build the Loop DP that closes x_out -> x_in and y_out -> y_in.

    Both loop variables are bundled into a single composite axis ``xy`` so
    that one Loop primitive closes the pair. The inner FunctionDP maps
    (c, xy) -> (xy, xy_report): ``xy`` feeds the loop, ``xy_report`` mirrors
    the value onto the outer resource so it stays visible in the result.
    """
    N = Naturals()
    XY = Ports({"x": N, "y": N})
    F = Ports({"c": N, "xy": XY})
    R = Ports({"xy": XY, "xy_report": XY})

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
