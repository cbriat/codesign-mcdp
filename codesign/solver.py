"""
Solver: Kleene fixed-point iteration for MCDPs.

Given a Loop DP, compute its least fixed point by repeatedly applying

    Phi(A) = Min(  union_{r in A}  h_inner( f_outer x {axis: r[axis]} ) ∩ up(r)  )

starting from A_0 = {⊥_R}. The sequence is monotone-increasing in the
antichain order, and either converges to a finite antichain (the minimal
resources) or sends one of its points to ⊤ (a proof of infeasibility).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from .antichains import Antichain
from .dp import DesignProblem


@dataclass
class SolveResult:
    """Outcome of a co-design solve.

    Attributes
    ----------
    antichain : the (approximate) least fixed point: minimal resource bundles
    iterations : how many Kleene steps were taken (0 for non-looped DPs)
    converged : whether the iteration converged within max_iter
    feasible : False iff every minimal resource hit the top of its poset
    trace : the sequence of antichains during the iteration (if requested)
    """

    antichain: Antichain
    iterations: int = 0
    converged: bool = True
    feasible: bool = True
    trace: List[Antichain] = field(default_factory=list)

    def __repr__(self) -> str:
        head = f"SolveResult(iters={self.iterations}, "
        head += f"converged={self.converged}, feasible={self.feasible})\n"
        head += f"  {self.antichain}"
        return head


# ---------------------------------------------------------------------------
# Core Kleene iteration for a Loop DP
# ---------------------------------------------------------------------------


def kleene_loop(
    loop_dp,
    f_outer: Mapping | None,
    max_iter: int = 200,
    record_trace: bool = False,
    trace_out: Optional[list] = None,
    info_out: Optional[dict] = None,
) -> Antichain:
    """Compute the loop's antichain at outer functionality ``f_outer``.

    Returns the antichain itself; ``solve`` wraps this with metadata. If
    ``trace_out`` is supplied (a list), every step's antichain is appended.
    If ``info_out`` is supplied (a dict), summary stats are written there:
    ``info_out['iterations']`` and ``info_out['converged']``.
    """
    inner = loop_dp.inner
    axis = loop_dp.axis
    R_loop = inner.R

    # Seed: a singleton antichain at bottom of R_loop.
    A = Antichain.singleton(R_loop, R_loop.bottom())
    if trace_out is not None:
        trace_out.append(A)

    # Divergence cap: any numeric resource exceeding this is treated as ⊤.
    # The Kleene iteration is increasing, so once a value exceeds a physically
    # absurd magnitude, the only finite consistent answer is infeasibility.
    DIVERGENCE_CAP = 1e30

    def cap_point(p):
        out = {}
        for k, v in p.items():
            if isinstance(v, (int, float)) and v > DIVERGENCE_CAP:
                out[k] = float("inf")
            else:
                out[k] = v
        return out

    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations += 1
        next_pieces: List[Antichain] = []
        for r in A:
            # Build inner functionality: outer inputs plus the loop axis value.
            f_inner = {}
            if f_outer is not None:
                f_inner.update(f_outer)
            f_inner[axis] = r[axis]
            try:
                a_r = inner.h(f_inner)
            except (OverflowError, ValueError, ZeroDivisionError):
                # Treat numerical blow-up as infeasibility on this branch.
                a_r = Antichain.singleton(R_loop, R_loop.top())
            # Cap any numeric components that already diverged.
            capped_points = [cap_point(p) for p in a_r.points]
            a_r = Antichain.from_set(R_loop, capped_points)
            a_r_above = a_r.filter_above(r)
            next_pieces.append(a_r_above)
        A_next = Antichain.union_min(R_loop, next_pieces)

        if trace_out is not None:
            trace_out.append(A_next)

        if A_next.is_empty():
            # No feasible extension; mark infeasible.
            A = Antichain.singleton(R_loop, R_loop.top())
            converged = True
            break

        if A_next.eq(A):
            A = A_next
            converged = True
            break

        # If every point reached the loop axis top, the system is infeasible.
        all_axis_top = all(
            inner.R.components[axis].is_top(p[axis]) for p in A_next
        )
        if all_axis_top:
            A = A_next
            converged = True
            break

        A = A_next

    # Record iteration stats.
    if info_out is not None:
        info_out["iterations"] = iterations
        info_out["converged"] = converged

    # Project onto outer R by dropping the looped axis.
    loop_axis_poset = inner.R.components[axis]
    all_loop_top = all(loop_axis_poset.is_top(p[axis]) for p in A) if len(A) else True

    if loop_dp.R.__class__.__name__ == "_Unit":
        # Empty outer R: a singleton if feasible, otherwise an antichain
        # marked at the top to signal infeasibility.
        if all_loop_top:
            return Antichain.empty(loop_dp.R)
        return Antichain.singleton(loop_dp.R, None)

    if all_loop_top:
        # All branches diverged: project as a single ⊤ point on the outer R.
        return Antichain.singleton(loop_dp.R, loop_dp.R.top())

    projected_points = []
    for p in A.points:
        # Drop points where the loop axis itself blew up; they're infeasible.
        if loop_axis_poset.is_top(p[axis]):
            continue
        projected_points.append({k: v for k, v in p.items() if k != axis})
    return Antichain.from_set(loop_dp.R, projected_points)


# ---------------------------------------------------------------------------
# Top-level solve
# ---------------------------------------------------------------------------


def solve(
    dp: DesignProblem,
    functionality: Mapping | None = None,
    max_iter: int = 200,
    record_trace: bool = False,
) -> SolveResult:
    """Solve a (possibly composite) design problem at the given functionality.

    For DPs without a top-level loop, this calls ``dp.h(functionality)``.
    For Loop DPs, it runs the Kleene iteration explicitly.
    """
    from .composition import Loop

    if functionality is None and hasattr(dp.F, "bottom"):
        try:
            functionality = dp.F.bottom()
        except Exception:
            functionality = None

    trace: list = [] if record_trace else None

    if isinstance(dp, Loop):
        info: dict = {}
        antichain = kleene_loop(
            dp, functionality, max_iter=max_iter, trace_out=trace, info_out=info
        )
        iters = info.get("iterations", 0)
        converged = info.get("converged", True)
    else:
        antichain = dp.h(functionality)
        iters = 0
        converged = True

    feasible = not antichain.has_any_top() and not antichain.is_empty()
    return SolveResult(
        antichain=antichain,
        iterations=iters,
        converged=converged,
        feasible=feasible,
        trace=trace or [],
    )


# ---------------------------------------------------------------------------
# Cost minimization on top of an antichain
# ---------------------------------------------------------------------------


def minimize_cost(result: SolveResult, cost_fn) -> Optional[Mapping]:
    """Pick the best point from a solved antichain according to a scalar cost.

    The antichain already gives the Pareto-minimal tradeoffs; flattening it to
    a single design typically means defining a scalar objective like
    ``cost_fn = lambda r: 0.5*r['weight'] + r['cost']`` and minimizing.
    Returns None when no feasible point is available.
    """
    if not result.feasible or not len(result.antichain):
        return None
    best = None
    best_cost = float("inf")
    for r in result.antichain:
        c = cost_fn(r)
        if c < best_cost:
            best_cost = c
            best = r
    return best
