"""
Solver: Kleene fixed-point iteration for MCDPs.

Given a Loop DP, compute its least fixed point by repeatedly applying

    Phi(A) = Min(  union_{r in A}  h_inner( f_outer x {axis: r[axis]} ) ∩ up(r)  )

starting from A_0 = {⊥_R}. The sequence is monotone-increasing in the
antichain order, and either converges to a finite antichain (the minimal
resources) or sends one of its points to ⊤ (a proof of infeasibility).

The solver supports three observability features:

- ``trace=True`` collects a structured :class:`TraceEntry` per iteration
  on ``result.trace`` for programmatic inspection or plotting after the
  fact.
- ``verbose=1`` prints a one-line summary at the end; ``verbose=2`` also
  prints a per-iteration progress line.
- ``on_iteration=callable`` is invoked with each :class:`TraceEntry` as
  it is produced, for live plotting, custom logging, etc.

The ``status`` field on :class:`SolveResult` distinguishes the three
outcomes of an iteration: ``"converged"``, ``"max_iter"``, and
``"diverged"``. ``feasible`` is kept as a separate field because
infeasibility (the antichain converged to ⊤) is logically independent
from the solver's termination reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional

from .antichains import Antichain
from .dp import DesignProblem


# ---------------------------------------------------------------------------
# Trace data structure
# ---------------------------------------------------------------------------


@dataclass
class TraceEntry:
    """One step of a Kleene iteration.

    Attributes
    ----------
    iteration : int
        0 = the seed antichain, 1 = after the first Kleene step, and so on.
    antichain : Antichain
        A snapshot of the antichain at this step.
    n_points : int
        Convenience: ``len(antichain)``.
    delta : float | None
        Convergence indicator. For Reals-valued posets, the maximum
        absolute change of any port value between this iteration and
        the previous one. For discrete/mixed posets, ``1.0`` if the
        antichain changed and ``0.0`` if it did not. ``None`` at
        iteration 0 (no previous state).
    elapsed_ms : float
        Wall time spent on this iteration alone, in milliseconds.
    """

    iteration: int
    antichain: Antichain
    n_points: int
    delta: Optional[float]
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Solve result
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Outcome of a co-design solve.

    Attributes
    ----------
    antichain : Antichain
        The (approximate) least fixed point: minimal resource bundles.
    iterations : int
        Number of Kleene steps taken (0 for non-looped DPs).
    status : str
        One of ``"converged"``, ``"max_iter"``, ``"diverged"``. This
        describes the solver's termination reason; orthogonal to
        ``feasible``.
    feasible : bool
        ``False`` iff every minimal resource hit ⊤ on some axis. A
        converged solve can still be infeasible (the antichain converges
        cleanly to ⊤) and a max-iter solve might still be making
        progress toward a feasible point.
    trace : list[TraceEntry] | None
        Structured per-iteration record, present iff ``trace=True`` was
        passed. ``None`` (not an empty list) when tracing was not
        requested.
    converged : bool
        Backward-compatibility alias for ``status == "converged"``.
    """

    antichain: Antichain
    iterations: int = 0
    status: str = "converged"
    feasible: bool = True
    trace: Optional[List[TraceEntry]] = None

    @property
    def converged(self) -> bool:
        """Backward-compatibility shim: True iff status is 'converged'."""
        return self.status == "converged"

    def __repr__(self) -> str:
        head = f"SolveResult(iters={self.iterations}, "
        head += f"status={self.status!r}, feasible={self.feasible})\n"
        head += f"  {self.antichain}"
        return head


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _numerical_delta(a_prev: Antichain, a_curr: Antichain) -> Optional[float]:
    """Maximum absolute change across all matching numeric components.

    Returns None if the antichains have different sizes (so dimensions
    don't line up) or if no numeric components could be compared. The
    set-equality delta is computed separately by the caller.
    """
    prev_pts = list(a_prev.points)
    curr_pts = list(a_curr.points)
    if len(prev_pts) != len(curr_pts) or not prev_pts:
        return None
    max_d = 0.0
    saw_numeric = False
    for p, q in zip(prev_pts, curr_pts):
        for k in p:
            if k not in q:
                continue
            vp, vq = p[k], q[k]
            if isinstance(vp, (int, float)) and isinstance(vq, (int, float)):
                try:
                    d = abs(float(vq) - float(vp))
                    if d != d:  # NaN guard
                        continue
                    if d > max_d:
                        max_d = d
                    saw_numeric = True
                except (OverflowError, ValueError):
                    continue
    return max_d if saw_numeric else None


def _compute_delta(a_prev: Optional[Antichain], a_curr: Antichain) -> Optional[float]:
    """Return a numeric delta if comparable, else the set-equality signal
    (1.0 changed, 0.0 same), or None on the very first iteration."""
    if a_prev is None:
        return None
    num = _numerical_delta(a_prev, a_curr)
    if num is not None:
        return num
    return 0.0 if a_curr.eq(a_prev) else 1.0


def _has_any_diverged(a: Antichain, cap: float) -> bool:
    """Return True if any numeric component of any point exceeds ``cap``
    (without yet being normalised to ⊤). Used to flag a divergence
    status separately from clean infeasibility."""
    for p in a.points:
        for v in p.values():
            if isinstance(v, (int, float)) and v >= cap:
                return True
    return False


def _verbose_iter_line(entry: TraceEntry) -> str:
    delta_str = "    -    " if entry.delta is None else f"{entry.delta:.3e}"
    return (
        f"[solve] iter {entry.iteration:>4}: "
        f"|A|={entry.n_points:<3} delta={delta_str}  "
        f"t={entry.elapsed_ms:.2f}ms"
    )


# ---------------------------------------------------------------------------
# Core Kleene iteration for a Loop DP
# ---------------------------------------------------------------------------


# Divergence cap: any numeric resource exceeding this is treated as ⊤.
# Surfaced as a module-level constant so the uncertainty layer can use it.
DIVERGENCE_CAP = 1e30


def kleene_loop(
    loop_dp,
    f_outer: Mapping | None,
    max_iter: int = 200,
    *,
    trace: bool = False,
    verbose: int = 0,
    on_iteration: Optional[Callable[[TraceEntry], None]] = None,
    info_out: Optional[dict] = None,
    # Legacy parameters for backward compatibility:
    record_trace: bool = False,
    trace_out: Optional[list] = None,
) -> Antichain:
    """Compute the loop's antichain at outer functionality ``f_outer``.

    Returns the antichain itself; :func:`solve` wraps this with metadata.

    Parameters
    ----------
    loop_dp : Loop
        The loop DP to iterate.
    f_outer : Mapping or None
        Values for the outer F ports (passed through to inner.h).
    max_iter : int
        Cap on Kleene steps. Reaching this cap sets ``status='max_iter'``.
    trace : bool
        If True, collects :class:`TraceEntry` objects (one per iteration
        plus the seed) and writes them to ``info_out['trace']``.
    verbose : int
        0 silent, 1 final summary line, 2 per-iteration progress feed.
    on_iteration : callable, optional
        Called with each :class:`TraceEntry` as it is produced (including
        the seed at iteration 0).
    info_out : dict, optional
        If supplied, ``info_out`` receives ``iterations``, ``status``, and
        (if ``trace=True``) ``trace``.
    record_trace, trace_out : legacy
        If ``record_trace=True`` and ``trace_out`` is a list, every step's
        antichain is appended (older two-call interface).
    """
    inner = loop_dp.inner
    axis = loop_dp.axis
    R_loop = inner.R

    # Seed.
    A = Antichain.singleton(R_loop, R_loop.bottom())

    legacy_trace_active = record_trace and (trace_out is not None)
    if legacy_trace_active:
        trace_out.append(A)

    structured_trace: List[TraceEntry] = [] if trace else None

    # Emit the seed entry.
    seed_entry = TraceEntry(
        iteration=0, antichain=A, n_points=len(A),
        delta=None, elapsed_ms=0.0,
    )
    if structured_trace is not None:
        structured_trace.append(seed_entry)
    if on_iteration is not None:
        on_iteration(seed_entry)
    if verbose >= 2:
        print(_verbose_iter_line(seed_entry))

    def cap_point(p):
        out = {}
        for k, v in p.items():
            if isinstance(v, (int, float)) and v > DIVERGENCE_CAP:
                out[k] = float("inf")
            else:
                out[k] = v
        return out

    status = "max_iter"  # default; overridden if we break early
    iterations = 0
    overall_t0 = time.perf_counter()
    for it in range(max_iter):
        iterations += 1
        t0 = time.perf_counter()
        next_pieces: List[Antichain] = []
        for r in A:
            f_inner = {}
            if f_outer is not None:
                f_inner.update(f_outer)
            f_inner[axis] = r[axis]
            try:
                a_r = inner.h(f_inner)
            except (OverflowError, ValueError, ZeroDivisionError):
                a_r = Antichain.singleton(R_loop, R_loop.top())
            capped_points = [cap_point(p) for p in a_r.points]
            a_r = Antichain.from_set(R_loop, capped_points)
            a_r_above = a_r.filter_above(r)
            next_pieces.append(a_r_above)
        A_next = Antichain.union_min(R_loop, next_pieces)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        if legacy_trace_active:
            trace_out.append(A_next)

        delta = _compute_delta(A, A_next)
        entry = TraceEntry(
            iteration=iterations, antichain=A_next, n_points=len(A_next),
            delta=delta, elapsed_ms=elapsed_ms,
        )
        if structured_trace is not None:
            structured_trace.append(entry)
        if on_iteration is not None:
            on_iteration(entry)
        if verbose >= 2:
            print(_verbose_iter_line(entry))

        if A_next.is_empty():
            A = Antichain.singleton(R_loop, R_loop.top())
            status = "converged"
            break

        if A_next.eq(A):
            A = A_next
            status = "converged"
            break

        # If every point reached the loop axis top, the system is infeasible.
        all_axis_top = all(
            inner.R.components[axis].is_top(p[axis]) for p in A_next
        )
        if all_axis_top:
            A = A_next
            status = "converged"
            break

        # Divergence guard: if any numeric component crosses the cap, flag it.
        if _has_any_diverged(A_next, DIVERGENCE_CAP):
            A = A_next
            status = "diverged"
            break

        A = A_next

    total_ms = (time.perf_counter() - overall_t0) * 1e3

    # Verbose summary.
    if verbose >= 1:
        feasible_now = (
            not A.has_any_top() if hasattr(A, "has_any_top") else False
        )
        n = len(A) if hasattr(A, "__len__") else 0
        print(
            f"[solve] {status}: {iterations} iters, "
            f"|A|={n}, total={total_ms:.1f}ms, feasible={feasible_now}"
        )

    if info_out is not None:
        info_out["iterations"] = iterations
        info_out["status"] = status
        if structured_trace is not None:
            info_out["trace"] = structured_trace

    # Project onto outer R by dropping the looped axis.
    loop_axis_poset = inner.R.components[axis]
    all_loop_top = all(loop_axis_poset.is_top(p[axis]) for p in A) if len(A) else True

    if loop_dp.R.__class__.__name__ == "_Unit":
        if all_loop_top:
            return Antichain.empty(loop_dp.R)
        return Antichain.singleton(loop_dp.R, None)

    if all_loop_top:
        return Antichain.singleton(loop_dp.R, loop_dp.R.top())

    projected_points = []
    for p in A.points:
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
    *,
    trace: bool = False,
    verbose: int = 0,
    on_iteration: Optional[Callable[[TraceEntry], None]] = None,
    # Legacy alias:
    record_trace: bool = False,
    # Uncertainty (lazy import inside to avoid circular dependency):
    uncertainty: Optional[List[str]] = None,
    n_samples: int = 1000,
    rng_seed: Optional[int] = None,
):
    """Solve a (possibly composite) design problem at the given functionality.

    Parameters
    ----------
    dp : DesignProblem
        The design problem.
    functionality : Mapping or None
        Outer F values.
    max_iter : int
        Max Kleene iterations.
    trace : bool, keyword-only
        Collect a structured trace on ``result.trace``.
    verbose : int, keyword-only
        Live printing level: 0 silent, 1 summary, 2 per-iteration.
    on_iteration : callable, keyword-only
        Optional callback receiving each :class:`TraceEntry`.
    record_trace : bool, keyword-only
        Backward-compatibility alias for ``trace`` (older code may pass this).
    uncertainty : list[str] or None, keyword-only
        If given, dispatch to the uncertainty solver. Allowed labels:
        ``"worst_case"``, ``"mean"``, ``"p95"``, ``"cvar95"``,
        ``"samples"``. Returns an :class:`~codesign.uncertainty.UncertaintyResult`
        instead of a plain :class:`SolveResult`.
    n_samples : int, keyword-only
        Monte Carlo sample count for stochastic uncertainty.
    rng_seed : int or None, keyword-only
        Optional seed for reproducibility under stochastic uncertainty.
    """
    # Dispatch to the uncertainty solver if requested.
    if uncertainty is not None:
        from .uncertainty import solve_with_uncertainty
        return solve_with_uncertainty(
            dp, functionality, uncertainty,
            n_samples=n_samples, rng_seed=rng_seed,
            max_iter=max_iter, verbose=verbose,
        )

    from .composition import Loop

    if functionality is None and hasattr(dp.F, "bottom"):
        try:
            functionality = dp.F.bottom()
        except Exception:
            functionality = None

    do_trace = trace or record_trace

    if isinstance(dp, Loop):
        info: dict = {}
        antichain = kleene_loop(
            dp, functionality, max_iter=max_iter,
            trace=do_trace, verbose=verbose, on_iteration=on_iteration,
            info_out=info,
        )
        iters = info.get("iterations", 0)
        status = info.get("status", "converged")
        trace_data = info.get("trace") if do_trace else None
    else:
        # Non-looped DP: single evaluation, no iteration.
        t0 = time.perf_counter()
        antichain = dp.h(functionality)
        ms = (time.perf_counter() - t0) * 1e3
        iters = 0
        status = "converged"
        if do_trace:
            seed = TraceEntry(
                iteration=0,
                antichain=Antichain.singleton(dp.R, dp.R.bottom())
                if hasattr(dp.R, "bottom") else antichain,
                n_points=0, delta=None, elapsed_ms=0.0,
            )
            final = TraceEntry(
                iteration=1, antichain=antichain, n_points=len(antichain),
                delta=None, elapsed_ms=ms,
            )
            trace_data = [seed, final]
            if on_iteration is not None:
                on_iteration(seed)
                on_iteration(final)
        else:
            trace_data = None
        if verbose >= 1:
            print(
                f"[solve] non-loop DP evaluated in {ms:.2f}ms, "
                f"|A|={len(antichain)}"
            )

    feasible = not antichain.has_any_top() and not antichain.is_empty()
    return SolveResult(
        antichain=antichain,
        iterations=iters,
        status=status,
        feasible=feasible,
        trace=trace_data,
    )


# ---------------------------------------------------------------------------
# Cost minimization on top of an antichain
# ---------------------------------------------------------------------------


def minimize_cost(result, cost_fn) -> Optional[Mapping]:
    """Pick the best point from a solved antichain according to a scalar cost.

    Accepts a :class:`SolveResult` or any object with ``feasible`` and
    ``antichain`` attributes.
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
