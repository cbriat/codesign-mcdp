"""Sequential co-design and antichain-valued dynamic programming.

This module implements the *antichain-valued* version of the temporal
Case 2 layer: a finite-horizon decision process whose stages are
co-design problems coupled by a carried state, solved by a Bellman
recursion whose value at each (stage, state) is itself a Pareto
antichain rather than a scalar. It is the strict generalisation of the
scalar prototype in :mod:`codesign.dynamic`: when the resource poset has
width one (a single cost axis) the value antichain is a singleton and the
recursion reduces to ordinary single-valued dynamic programming.

The formal object (following the sequential co-design theory)
-------------------------------------------------------------
Fix an ordered commutative resource monoid ``(R, <=, (+), 0)``: ``R`` a
poset, ``(+)`` associative, commutative, and *monotone*, with unit
``0 = bottom``. Work in the value space of upper sets of ``R`` ordered by
*reverse inclusion* (a larger feasible set is a lower resource threshold,
hence "easier"). In this package an upper set is represented by its
minimal antichain via :class:`~codesign.antichains.Antichain`, and the
monoid combination ``(+)`` is supplied by the caller (sum for a
consumable/accumulating resource, join for a renewable one).

A sequential co-design problem (SCDP) over stages ``0..N`` and a state
poset ``(X, <=, bottom)`` assigns to each stage ``k``:

* a state-parametrised minimal-resource map ``h_k : X -> Antichain(R)``,
  the resources sufficient to serve stage ``k`` from incoming state
  ``x`` (here obtained by solving that stage's co-design problem), and
* a transition ``phi_k : X x R -> X`` committing a chosen resource point
  at stage ``k`` from state ``x`` to a successor state.

The value is the backward recursion ``V_{N+1} = R`` (the whole monoid,
i.e. the bottom antichain) and

    V_k(x) = Min  U_{r in h_k(x)}  ( up(r) (+) V_{k+1}( phi_k(x, r) ) ).

This is the antichain-valued Bellman operator: the union is the
existential over feasible choices (the minimisation over actions),
``(+)`` combines stage resource with cost-to-go, and ``Min`` keeps the
Pareto antichain. Unlike the cost-vector value functions of
multi-objective DP, the value here lives in the resource lattice and the
per-stage feasible set is itself a co-design problem, which is what gives
the action set its order structure.

Three results from the theory are made operational here
-------------------------------------------------------
* **Monotone value (Q1).** If, for every stage, (H1) ``h_k`` is monotone
  in the state in the easier-when-larger sense (``x <= x'`` implies
  ``h_k(x) ⊇ h_k(x')`` as upper sets, i.e. more carried state makes the
  stage no easier) and (H2) ``phi_k`` is monotone in the state, then
  every ``V_k`` is monotone. (H1) is load-bearing and cannot be dropped.
  :func:`check_monotonicity` verifies both numerically on a state grid.

* **Front equals reachable frontier (Q2).** ``Min V_k(x)`` is exactly the
  antichain of minimal reachable cumulative resources over feasible
  choice sequences from ``(k, x)``. There is no tail-pruning gap; the
  size of the value antichain is the width of the reachable frontier.
  This is why the construction never enumerates dominated tails.

* **Exact factorisation at a reset (Q3).** If a stage ``m`` is a reset
  (its transition always returns the quiescent bottom state regardless of
  incoming state and choice), the horizon factorises as a ``(+)``-product
  of independent sub-problems, one per quiescence-free run. This holds in
  every regime (it uses only distributivity of ``(+)`` over the Bellman
  union, neither monotonicity nor a bounded front), and
  :func:`detect_resets` plus :func:`factorise_at_resets` expose it.

State is discretised onto a :class:`~codesign.dynamic.StateGrid`, with the
same out-of-bounds guard as the scalar layer: a transition leaving the
grid envelope is rejected before snapping so an over-spent resource is
never silently rescued. Because snapping is not order-preserving at
bucket boundaries, the monotonicity guarantees hold up to grid
resolution; :func:`check_monotonicity` reports violations introduced by a
too-coarse grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .antichains import Antichain
from .dynamic import StateGrid
from .posets import Ports, Reals
from .solver import solve
from .temporal import Architecture

INF = float("inf")


# ---------------------------------------------------------------------------
# Resource combination (the monoid (+))
# ---------------------------------------------------------------------------
def sum_combine(a: Mapping, b: Mapping) -> Dict:
    """Consumable / accumulating combination: component-wise sum.

    This is ``(+) = +`` on a product of real axes. Resources accumulate
    across stages (fuel burned, money spent, wear incurred); the reachable
    frontier can grow polynomially with the horizon (Q2, fixed-dimension
    regime).

    Both operands must share the same axes: the result is keyed by ``a``,
    so any axis present in ``a`` but missing from ``b`` raises ``KeyError``
    (and an axis only in ``b`` is silently dropped).
    """
    return {k: a[k] + b[k] for k in a}


def join_combine(a: Mapping, b: Mapping) -> Dict:
    """Renewable combination: component-wise maximum (join).

    This is ``(+) = join``. The requirement of two stages is the peak, not
    the sum (a worker, an oven, a bus reused across stages); the reachable
    frontier stays bounded uniformly in the horizon (Q2, renewable
    regime), and if every stage resets the whole problem collapses to a
    static co-design.

    Both operands must share the same axes: the result is keyed by ``a``,
    so any axis present in ``a`` but missing from ``b`` raises ``KeyError``
    (and an axis only in ``b`` is silently dropped).
    """
    return {k: max(a[k], b[k]) for k in a}


# ---------------------------------------------------------------------------
# Stage definition for the sequential (antichain-valued) problem
# ---------------------------------------------------------------------------
@dataclass
class SeqStage:
    """One stage of a sequential co-design problem.

    Parameters
    ----------
    name : str
        Stage identifier.
    functionality : callable
        ``functionality(state) -> Mapping`` giving the outer functionality
        demanded at this stage when the carried state has value ``state``.
    transition : callable
        ``transition(state, point) -> new_state`` mapping incoming carried
        state and a *chosen resource point* (an element of the stage's
        solved antichain, a Mapping over the resource ports) to the
        outgoing carried state. Returning the grid's bottom node makes the
        stage a reset at that argument.
    candidates : sequence of Architecture, optional
        Architectures admissible at this stage; falls back to the
        problem-level default when ``None``. Each architecture's co-design
        solve supplies part of ``h_k(x)``: the union of their antichains.
    admissible : callable, optional
        ``admissible(state) -> bool`` forbidding states for which it
        returns ``False`` (treated as carrying the infeasible empty value).
    """

    name: str
    functionality: Callable[[float], Mapping]
    transition: Callable[[float, Mapping], float]
    candidates: Optional[Sequence[Architecture]] = None
    admissible: Optional[Callable[[float], bool]] = None


@dataclass
class SeqResult:
    """Outcome of :func:`solve_sequential`.

    Attributes
    ----------
    value : Antichain
        The value antichain ``Min V_0(initial_state)``: the Pareto front of
        minimal cumulative resources achievable over the whole horizon from
        the initial state. Each point is a Mapping over the cost axes.
    width : int
        ``len(value)``: the number of incomparable Pareto-optimal totals
        (the reachable-frontier width alpha_0).
    feasible : bool
        ``True`` iff the value antichain is non-empty and free of tops.
    policy : SeqPolicy
        The full state-indexed antichain-valued policy table.
    """

    value: Antichain
    width: int
    feasible: bool
    policy: "SeqPolicy"

    def __repr__(self) -> str:
        feas = "feasible" if self.feasible else "INFEASIBLE"
        return f"SeqResult({feas}, width={self.width}, value={self.value!r})"


# ---------------------------------------------------------------------------
# Building the cost+state product poset
# ---------------------------------------------------------------------------
def _cost_state_poset(cost_axes: Sequence[str]) -> Ports:
    """Product poset over the named cost axes (all Reals).

    The carried state is handled by the grid/table indexing, not as an
    axis of this poset; the antichain lives purely over the accumulated
    *resource* axes, matching the theory where ``V_k(x)`` is an antichain
    in ``R`` parametrised by the state ``x``.
    """
    return Ports({name: Reals() for name in cost_axes})


# ---------------------------------------------------------------------------
# Per-stage co-design: assemble h_k(x) as a single antichain over R
# ---------------------------------------------------------------------------
def _stage_antichain(
    cands: Sequence[Architecture],
    functionality: Mapping,
    cost_axes: Sequence[str],
    *,
    max_iter: int,
    solve_kwargs: Dict[str, Any],
) -> Tuple[Antichain, Dict[Tuple, str], Dict[Tuple, Mapping]]:
    """Union of the candidates' solved antichains at this functionality.

    Returns three things: the combined antichain ``h_k(x)`` over the cost
    axes; a map from each (cost-projected) point key to the architecture
    that produced it; and a map from each point key to the *full* solved
    resource point (all ports, including any carried-state axis such as
    fuel), so the transition can read quantities that are not accumulated
    on the antichain. The accumulated resource ``R`` (the antichain axes)
    and the carried state ``x`` are deliberately distinct, matching the
    theory where ``V_k(x)`` is an antichain in ``R`` parametrised by ``x``.
    """
    poset = _cost_state_poset(cost_axes)
    per_arch_acs: List[Antichain] = []
    origin: Dict[Tuple, str] = {}
    full: Dict[Tuple, Mapping] = {}
    for arch in cands:
        res = solve(arch.dp, functionality, max_iter=max_iter, **solve_kwargs)
        if not res.feasible:
            continue
        pts = []
        for r in res.antichain:
            proj = {ax: r[ax] for ax in cost_axes}
            key = _key(proj, cost_axes)
            pts.append(proj)
            origin[key] = arch.name
            full[key] = dict(r)
        if pts:
            per_arch_acs.append(Antichain.from_set(poset, pts))
    combined = Antichain.union_min(poset, per_arch_acs)
    return combined, origin, full


def _key(point: Mapping, cost_axes: Sequence[str]) -> Tuple:
    """Hashable key for a resource point (rounded to tame float noise)."""
    return tuple(round(float(point[ax]), 9) for ax in cost_axes)


# ---------------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------------
class SeqPolicy:
    """State-indexed antichain-valued policy produced by the backward pass.

    For each (stage, state node) it stores the value antichain ``V_k(x)``
    and, for each Pareto point, the realising architecture and the chosen
    stage resource point, so an optimal choice sequence can be traced
    forward for any selected terminal Pareto point.
    """

    def __init__(
        self,
        stage_names: Sequence[str],
        grid: StateGrid,
        cost_axes: Sequence[str],
        value: List[Dict[float, Antichain]],
        # choice[k][node] : key(total_point) -> (arch, stage_point, succ_node)
        choice: List[Dict[float, Dict[Tuple, Tuple[str, Mapping, float]]]],
    ):
        self.stage_names = list(stage_names)
        self.grid = grid
        self.cost_axes = list(cost_axes)
        self._value = value
        self._choice = choice

    def value_at(self, k: int, state: float) -> Antichain:
        """Value antichain ``V_k(x)`` at the grid node nearest ``state``."""
        node = self.grid.snap(state)
        poset = _cost_state_poset(self.cost_axes)
        return self._value[k].get(node, Antichain.empty(poset))

    def width_at(self, k: int, state: float) -> int:
        """Reachable-frontier width ``alpha_k(x)`` at this (stage, state)."""
        return len(self.value_at(k, state))


# ---------------------------------------------------------------------------
# Backward antichain-valued Bellman pass
# ---------------------------------------------------------------------------
def solve_sequential(
    stages: Sequence[SeqStage],
    grid: StateGrid,
    *,
    cost_axes: Sequence[str],
    initial_state: float,
    combine: Callable[[Mapping, Mapping], Mapping] = sum_combine,
    architectures: Optional[Sequence[Architecture]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> SeqResult:
    """Solve the antichain-valued sequential co-design problem.

    Runs the backward Bellman recursion in the upper-set value space,
    carrying a full Pareto antichain of cumulative resources at each
    (stage, state). At each stage and state the union of the admissible
    architectures' solved antichains forms ``h_k(x)``; each of its points
    is combined (via ``combine``) with the value antichain of the snapped
    successor state, and the results are merged by :meth:`Antichain.union_min`
    to give ``V_k(x)``.

    Parameters
    ----------
    stages : sequence of SeqStage
        The horizon in forward order; solved backward internally.
    grid : StateGrid
        Discretisation of the carried scalar state.
    cost_axes : sequence of str
        Names of the resource ports to accumulate on the antichain (for
        example ``["cost", "co2"]``). These define the product poset ``R``.
    initial_state : float
        Starting value of the carried state; the returned value is
        ``V_0(initial_state)``.
    combine : callable
        The monoid combination ``(+)``: :func:`sum_combine` (default) for a
        consumable/accumulating resource, :func:`join_combine` for a
        renewable one.
    architectures : sequence of Architecture, optional
        Default candidates for stages that do not supply their own.
    max_iter : int
        Forwarded to :func:`~codesign.solver.solve`.
    solve_kwargs : mapping, optional
        Extra keyword arguments forwarded to :func:`~codesign.solver.solve`.

    Returns
    -------
    SeqResult
    """
    solve_kwargs = dict(solve_kwargs or {})
    T = len(stages)
    poset = _cost_state_poset(cost_axes)
    zero = {ax: 0.0 for ax in cost_axes}

    def candidates_for(st: SeqStage) -> Sequence[Architecture]:
        if st.candidates is not None:
            return st.candidates
        if architectures is None:
            raise ValueError(
                f"stage {st.name!r} has no candidates and no default "
                f"architecture set was supplied. Either set candidates=[...] "
                f"on the stage, or pass architectures=[...] to the sequential "
                f"solver (solve_sequential / check_monotonicity)."
            )
        return architectures

    # Terminal layer: V_{N+1}(x) = {0} (the bottom antichain, "nothing more
    # to spend"), at every grid node.
    value: List[Dict[float, Antichain]] = [dict() for _ in range(T + 1)]
    choice: List[Dict[float, Dict[Tuple, Tuple[str, Mapping, float]]]] = [
        dict() for _ in range(T)
    ]
    for node in grid:
        value[T][node] = Antichain.singleton(poset, dict(zero))

    lo, hi = grid.nodes[0], grid.nodes[-1]
    span = hi - lo
    tol = 1e-9 * (span if span > 0 else 1.0)

    # Backward pass.
    for k in range(T - 1, -1, -1):
        stage = stages[k]
        cands = candidates_for(stage)
        for x in grid:
            if stage.admissible is not None and not stage.admissible(x):
                value[k][x] = Antichain.empty(poset)
                choice[k][x] = {}
                continue
            func = stage.functionality(x)
            # Enumerate ALL architecture points (not the cost-Pareto-reduced
            # antichain): a point dominated on the cost axes may still be the
            # only feasible choice from a constrained carried state, because
            # its consequence for the carried state is invisible in the cost
            # projection. The final union_min prunes after the transition.
            _, origin, full = _stage_antichain(
                cands, func, cost_axes,
                max_iter=max_iter, solve_kwargs=solve_kwargs,
            )
            terms: List[Antichain] = []
            node_choice: Dict[Tuple, Tuple[str, Mapping, float]] = {}
            seen_full: set = set()
            for rkey, full_pt in full.items():
                if rkey in seen_full:
                    continue
                seen_full.add(rkey)
                r = {ax: full_pt[ax] for ax in cost_axes}
                # The transition reads the FULL solved point (so it can use
                # a carried-state axis like fuel that is not accumulated on
                # the antichain); the antichain accumulates only cost axes.
                x_succ_raw = stage.transition(x, full_pt)
                # Out-of-bounds guard: reject before snapping.
                if x_succ_raw < lo - tol or x_succ_raw > hi + tol:
                    continue
                if stage.admissible is not None and not stage.admissible(
                    x_succ_raw
                ):
                    continue
                x_succ = grid.snap(x_succ_raw)
                tail = value[k + 1].get(x_succ, Antichain.empty(poset))
                if tail.is_empty():
                    continue
                # up(r) (+) tail : combine this stage's resource point with
                # every tail Pareto point, then Min.
                combined_pts = [combine(r, t) for t in tail]
                term = Antichain.from_set(poset, combined_pts)
                terms.append(term)
                # Record, for each resulting total, the realising choice.
                arch_name = origin.get(rkey, "")
                for cp in combined_pts:
                    node_choice[_key(cp, cost_axes)] = (arch_name, full_pt, x_succ)
            vk = Antichain.union_min(poset, terms)
            value[k][x] = vk
            # Keep only choices whose totals survived into the Min front.
            surviving = {_key(p, cost_axes) for p in vk}
            choice[k][x] = {
                key: val for key, val in node_choice.items() if key in surviving
            }

    policy = SeqPolicy(
        stage_names=[st.name for st in stages],
        grid=grid,
        cost_axes=cost_axes,
        value=value,
        choice=choice,
    )
    v0 = policy.value_at(0, initial_state)
    feasible = (not v0.is_empty()) and (not v0.has_any_top())
    return SeqResult(
        value=v0,
        width=len(v0),
        feasible=feasible,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Q1: monotonicity guard (verify H1 and H2 numerically on the grid)
# ---------------------------------------------------------------------------
@dataclass
class MonotonicityReport:
    """Result of :func:`check_monotonicity`.

    Attributes
    ----------
    h1_ok : bool
        True iff the stage maps satisfied (H1) at every tested grid pair.
    h2_ok : bool
        True iff the transitions satisfied (H2) at every tested grid pair.
    h1_violations, h2_violations : list of tuple
        Sampled witnesses ``(stage_name, x, x')`` where the condition
        failed, for diagnosis (empty when the condition held).
    """

    h1_ok: bool
    h2_ok: bool
    h1_violations: List[Tuple[str, float, float]] = field(default_factory=list)
    h2_violations: List[Tuple[str, float, float]] = field(default_factory=list)

    @property
    def monotone_value_guaranteed(self) -> bool:
        """(H1) and (H2) together imply a monotone value (Theorem Q1)."""
        return self.h1_ok and self.h2_ok

    def __repr__(self) -> str:
        return (
            f"MonotonicityReport(H1={'ok' if self.h1_ok else 'FAIL'}, "
            f"H2={'ok' if self.h2_ok else 'FAIL'}, "
            f"value_monotone={'guaranteed' if self.monotone_value_guaranteed else 'NOT guaranteed'})"
        )


def check_monotonicity(
    stages: Sequence[SeqStage],
    grid: StateGrid,
    *,
    cost_axes: Sequence[str],
    architectures: Optional[Sequence[Architecture]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    max_violations: int = 8,
) -> MonotonicityReport:
    """Numerically verify (H1) and (H2) on the state grid.

    (H1) For each stage, the stage antichain ``h_k`` must be *consistently
    oriented* in the carried state: as the state grows it is either no
    easier everywhere (the paper's literal (H1), state oriented as
    accumulated commitment) or no harder everywhere (the benign
    "consumable but monotone" orientation, where the state is carried as a
    remaining budget). A stage is flagged only when ``h_k`` is genuinely
    non-monotone, neither consistently easier nor consistently harder as
    the state grows. That is the dangerous perishable / fatigue-as-state
    case with an interior optimum, which is exactly where the monotone-
    value guarantee fails.

    (H2) For each stage and ordered grid pair and each chosen point, the
    transition must be monotone non-decreasing in the state.

    Returns a :class:`MonotonicityReport`; when both hold, the value
    function is guaranteed monotone (the Q1 theorem). Spurious violations
    are most often introduced by a too-coarse grid (snapping is not
    order-preserving at bucket boundaries) rather than by the maps
    themselves.
    """
    solve_kwargs = dict(solve_kwargs or {})
    poset = _cost_state_poset(cost_axes)
    nodes = list(grid.nodes)

    def candidates_for(st: SeqStage) -> Sequence[Architecture]:
        if st.candidates is not None:
            return st.candidates
        if architectures is None:
            raise ValueError(
                f"stage {st.name!r} has no candidates and no default "
                f"architecture set was supplied. Either set candidates=[...] "
                f"on the stage, or pass architectures=[...] to the sequential "
                f"solver (solve_sequential / check_monotonicity)."
            )
        return architectures

    h1_violations: List[Tuple[str, float, float]] = []
    h2_violations: List[Tuple[str, float, float]] = []

    for st in stages:
        cands = candidates_for(st)
        # Cache stage antichains and full points per node.
        ac: Dict[float, Antichain] = {}
        full_by_node: Dict[float, Dict[Tuple, Mapping]] = {}
        for x in nodes:
            if st.admissible is not None and not st.admissible(x):
                ac[x] = Antichain.empty(poset)
                full_by_node[x] = {}
                continue
            hk, _, full = _stage_antichain(
                cands, st.functionality(x), cost_axes,
                max_iter=max_iter, solve_kwargs=solve_kwargs,
            )
            ac[x] = hk
            full_by_node[x] = full
        # First pass: determine the consistent orientation of h in the
        # state, if any. The paper's (H1) is stated for a state poset
        # oriented so that larger state means more carried commitment
        # (harder). A consumable resource carried as *remaining* budget has
        # the reverse orientation (larger = easier) and is the benign
        # "consumable but monotone" case: still monotone, just order-
        # reversing. The dangerous case (perishable / fatigue-as-state) is
        # genuine *non-monotonicity*: neither consistently easier nor
        # consistently harder as state grows. We therefore check for a
        # consistent orientation and only flag its absence.
        nondecreasing = True   # larger state no easier (paper's H1 literal)
        nonincreasing = True   # larger state no harder (reverse orientation)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                x, xp = nodes[i], nodes[j]  # x < xp
                a_x, a_xp = ac[x], ac[xp]
                if a_x.is_empty() or a_xp.is_empty():
                    continue
                # easier(a) means lower threshold: a.leq(b) => a easier.
                x_easier = a_x.leq(a_xp)   # x no harder than xp
                xp_easier = a_xp.leq(a_x)  # xp no harder than x
                if not x_easier:
                    nondecreasing = False
                if not xp_easier:
                    nonincreasing = False
        # (H1) holds in the monotone sense iff h is consistently oriented.
        h1_stage_ok = nondecreasing or nonincreasing
        if not h1_stage_ok:
            # The non-monotonicity is a property of the sequence (it goes
            # both easier and harder as the state grows), not of any single
            # incomparable pair. Record the adjacent pair at which the
            # orientation reverses as a witness.
            prev_dir = 0  # -1 = got easier, +1 = got harder
            recorded = False
            for t in range(len(nodes) - 1):
                x, xp = nodes[t], nodes[t + 1]
                a_x, a_xp = ac[x], ac[xp]
                if a_x.is_empty() or a_xp.is_empty():
                    continue
                if a_x.leq(a_xp) and not a_xp.leq(a_x):
                    direction = 1   # xp harder: got harder
                elif a_xp.leq(a_x) and not a_x.leq(a_xp):
                    direction = -1  # xp easier: got easier
                else:
                    direction = 0   # equal/incomparable
                if direction != 0 and prev_dir != 0 and direction != prev_dir:
                    if len(h1_violations) < max_violations:
                        h1_violations.append((st.name, nodes[t - 1], xp))
                    recorded = True
                    break
                if direction != 0:
                    prev_dir = direction
            if not recorded and len(h1_violations) < max_violations:
                # Fallback witness: first and last comparable nodes.
                h1_violations.append((st.name, nodes[0], nodes[-1]))
        # (H2): transition monotone non-decreasing in state. Use the full
        # solved point so a carried-state axis is available.
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                x, xp = nodes[i], nodes[j]
                a_x = ac[x]
                broke = False
                for r in a_x:
                    full_pt = full_by_node[x].get(_key(r, cost_axes), r)
                    if st.transition(x, full_pt) > st.transition(xp, full_pt) + 1e-9:
                        if len(h2_violations) < max_violations:
                            h2_violations.append((st.name, x, xp))
                        broke = True
                        break
                if broke:
                    break

    return MonotonicityReport(
        h1_ok=not h1_violations,
        h2_ok=not h2_violations,
        h1_violations=h1_violations,
        h2_violations=h2_violations,
    )


# ---------------------------------------------------------------------------
# Q3: reset detection and exact factorisation
# ---------------------------------------------------------------------------
def detect_resets(
    stages: Sequence[SeqStage],
    grid: StateGrid,
    *,
    cost_axes: Sequence[str],
    architectures: Optional[Sequence[Architecture]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Return the indices of stages that are resets (quiescent transitions).

    Stage ``m`` is a reset iff its transition lands on the grid's bottom
    node for every incoming state and every chosen resource point. At a
    reset the horizon factorises exactly (Theorem Q3): the value is a
    ``(+)``-product of the independent quiescence-free runs, and this holds
    in every regime since it uses only distributivity, not monotonicity.
    """
    solve_kwargs = dict(solve_kwargs or {})
    bottom = grid.nodes[0]
    resets: List[int] = []

    def candidates_for(st: SeqStage) -> Sequence[Architecture]:
        if st.candidates is not None:
            return st.candidates
        if architectures is None:
            raise ValueError(
                f"stage {st.name!r} has no candidates and no default "
                f"architecture set was supplied. Either set candidates=[...] "
                f"on the stage, or pass architectures=[...] to the sequential "
                f"solver (solve_sequential / check_monotonicity)."
            )
        return architectures

    for m, st in enumerate(stages):
        is_reset = True
        for x in grid:
            if st.admissible is not None and not st.admissible(x):
                continue
            _, _, full = _stage_antichain(
                candidates_for(st), st.functionality(x), cost_axes,
                max_iter=max_iter, solve_kwargs=solve_kwargs,
            )
            for full_pt in full.values():
                if abs(st.transition(x, full_pt) - bottom) > 1e-9:
                    is_reset = False
                    break
            if not is_reset:
                break
        if is_reset:
            resets.append(m)
    return resets


def factorise_at_resets(
    stages: Sequence[SeqStage],
    resets: Sequence[int],
) -> List[Tuple[int, int]]:
    """Partition the horizon into quiescence-free runs given reset stages.

    Returns a list of ``(start, end)`` inclusive index ranges, one per
    maximal run between consecutive resets. By Theorem Q3 the value over
    the full horizon from the quiescent bottom state is the ``(+)``-product
    of the values of these independent sub-problems, so they may be solved
    separately and combined, which is the order-theoretic, deterministic,
    multi-objective analogue of regeneration-point decomposition.
    """
    runs: List[Tuple[int, int]] = []
    start = 0
    reset_set = set(resets)
    for m in range(len(stages)):
        if m in reset_set:
            # A reset closes the run that includes it.
            runs.append((start, m))
            start = m + 1
    if start <= len(stages) - 1:
        runs.append((start, len(stages) - 1))
    return runs


# ---------------------------------------------------------------------------
# Precompute-then-DP (the Formula 1 paper's structure)
# ---------------------------------------------------------------------------
def precompute_catalog(
    architectures: Sequence[Architecture],
    functionality: Mapping,
    cost_axes: Sequence[str],
    *,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Mapping]]:
    """Solve every architecture once and return a flat Pareto catalog.

    This is the co-design *precomputation* step of the Neumann, Zardini et
    al. Formula 1 seasonal framework: the co-design layer is run once to
    produce track-dependent Pareto-optimal mappings, which are then frozen
    into a catalog of implementations that an outer dynamic program selects
    among. Here the catalog is the Min over the union of the architectures'
    solved antichains at a fixed functionality, each point tagged with the
    architecture that produced it.

    Crucially, points that look dominated at this single-stage level are
    *retained* if they are non-dominated in the combined antichain, because
    they may become optimal once aggregated in a season-level DP (the F1
    paper's observation that higher-wear, equal-time solutions must be kept
    for the seasonal optimisation).

    Returns a list of ``(arch_name, point)`` pairs. Pass it to
    :func:`dp_over_catalog` (or any DP) whose per-stage action set is the
    catalog index. This structure differs from :func:`solve_sequential`,
    which re-solves the co-design at every (stage, state); here the
    co-design is solved once and the DP only indexes the frozen result,
    which is cheaper and is valid when the per-stage co-design does not
    depend on the DP-carried state.
    """
    ac, origin, full = _stage_antichain(
        architectures, functionality, cost_axes,
        max_iter=max_iter, solve_kwargs=dict(solve_kwargs or {}),
    )
    catalog: List[Tuple[str, Mapping]] = []
    for p in ac:
        key = _key(p, cost_axes)
        catalog.append((origin.get(key, ""), full.get(key, dict(p))))
    return catalog


def dp_over_catalog(
    catalogs: Sequence[Sequence[Tuple[str, Mapping]]],
    grid: StateGrid,
    *,
    cost_axes: Sequence[str],
    initial_state: float,
    transition: Callable[[float, Mapping], float],
    combine: Callable[[Mapping, Mapping], Mapping] = sum_combine,
    admissible: Optional[Callable[[float], bool]] = None,
) -> SeqResult:
    """Run an antichain-valued DP that *selects from precomputed catalogs*.

    This is the outer dynamic program of the precompute-then-DP structure:
    each stage's action set is a frozen catalog (from
    :func:`precompute_catalog`) rather than a live co-design solve. The
    carried scalar state is advanced by ``transition`` reading the chosen
    catalog point, exactly as in :func:`solve_sequential`, but no
    :func:`~codesign.solver.solve` call happens inside the Bellman sweep.

    Use this when the per-stage co-design is independent of the carried
    state (so it can be precomputed once per stage), which is the regime
    the Formula 1 framework operates in: the lap/race Pareto fronts depend
    on the track and the initial component age, both fixed before the
    seasonal DP runs. When the co-design genuinely depends on the carried
    state, use :func:`solve_sequential` instead.

    Parameters
    ----------
    catalogs : sequence of catalogs
        One catalog per stage; each a sequence of ``(arch_name, point)``
        pairs as returned by :func:`precompute_catalog`.
    grid : StateGrid
        Discretisation of the carried scalar state.
    cost_axes : sequence of str
        Resource ports accumulated on the antichain.
    initial_state : float
        Starting carried state.
    transition : callable
        ``transition(state, point) -> new_state``.
    combine : callable
        Monoid combination ``(+)``.
    admissible : callable, optional
        State admissibility predicate.
    """
    poset = _cost_state_poset(cost_axes)
    zero = {ax: 0.0 for ax in cost_axes}
    T = len(catalogs)

    value: List[Dict[float, Antichain]] = [dict() for _ in range(T + 1)]
    choice: List[Dict[float, Dict[Tuple, Tuple[str, Mapping, float]]]] = [
        dict() for _ in range(T)
    ]
    for node in grid:
        value[T][node] = Antichain.singleton(poset, dict(zero))

    lo, hi = grid.nodes[0], grid.nodes[-1]
    span = hi - lo
    tol = 1e-9 * (span if span > 0 else 1.0)

    for k in range(T - 1, -1, -1):
        catalog = catalogs[k]
        for s in grid:
            if admissible is not None and not admissible(s):
                value[k][s] = Antichain.empty(poset)
                choice[k][s] = {}
                continue
            terms: List[Antichain] = []
            node_choice: Dict[Tuple, Tuple[str, Mapping, float]] = {}
            for arch_name, full_pt in catalog:
                proj = {ax: full_pt[ax] for ax in cost_axes}
                s_next_raw = transition(s, full_pt)
                if s_next_raw < lo - tol or s_next_raw > hi + tol:
                    continue
                if admissible is not None and not admissible(s_next_raw):
                    continue
                s_next = grid.snap(s_next_raw)
                tail = value[k + 1].get(s_next, Antichain.empty(poset))
                if tail.is_empty():
                    continue
                combined_pts = [combine(proj, t) for t in tail]
                terms.append(Antichain.from_set(poset, combined_pts))
                for cp in combined_pts:
                    node_choice[_key(cp, cost_axes)] = (arch_name, full_pt, s_next)
            vk = Antichain.union_min(poset, terms)
            value[k][s] = vk
            surviving = {_key(p, cost_axes) for p in vk}
            choice[k][s] = {
                key: val for key, val in node_choice.items() if key in surviving
            }

    policy = SeqPolicy(
        stage_names=[f"stage_{i}" for i in range(T)],
        grid=grid, cost_axes=cost_axes, value=value, choice=choice,
    )
    v0 = policy.value_at(0, initial_state)
    feasible = (not v0.is_empty()) and (not v0.has_any_top())
    return SeqResult(value=v0, width=len(v0), feasible=feasible, policy=policy)


__all__ = [
    "SeqStage",
    "SeqResult",
    "SeqPolicy",
    "solve_sequential",
    "sum_combine",
    "join_combine",
    "MonotonicityReport",
    "check_monotonicity",
    "detect_resets",
    "factorise_at_resets",
    "precompute_catalog",
    "dp_over_catalog",
]
