"""Vector-state sequential co-design: the general carried-state DP.

This module generalises :mod:`codesign.sequential` from a single carried
scalar to a full carried **state vector**. It is the framework needed for
problems whose stage-to-stage information is structured rather than a
lone number: the Formula 1 seasonal problem carries two battery wear
levels and a regulatory flag; a reconfigurable robot carries per-module
wear plus a shared energy budget; a self-assembly protocol carries the
concentrations of several intermediates. Each of these is a state vector,
handled here by a :class:`~codesign.state.VectorStateGrid`.

Everything else matches the antichain-valued sequential layer. The value
at each (stage, state vector) is a Pareto antichain of cumulative
resource totals over the named cost axes; the Bellman ``min`` is
:meth:`~codesign.antichains.Antichain.union_min`; the per-stage
map ``h_k`` is assembled by solving each admissible architecture's
co-design problem at the stage's (state-dependent) functionality; and the
transition advances the carried vector, read from the full solved point.
The carried state vector is kept distinct from the accumulated cost axes,
exactly as in the scalar case.

The monotonicity results carry over verbatim with the product order of
the vector grid substituted for the scalar order:

* (H1) the stage map is monotone in the carried state vector (more
  carried state, in the product order, is no easier), and
* (H2) the transition is monotone in the carried state vector.

:func:`check_vector_monotonicity` verifies both on the product grid,
using the same orientation-aware logic as the scalar guard so a
consumable-but-monotone axis is accepted while a genuinely non-monotone
(perishable) one is flagged.

A single-axis :class:`~codesign.state.VectorStateGrid` reproduces the
scalar :mod:`codesign.sequential` behaviour, so this module supersedes it
without contradicting it; ``sequential`` remains as the ergonomic
scalar-only entry point.
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
from .posets import Ports, Reals
from .solver import solve
from .state import StateVec, VectorStateGrid, state_as_dict
from .temporal import Architecture

INF = float("inf")


# ---------------------------------------------------------------------------
# Resource combination (the monoid (+)); same semantics as sequential.py
# ---------------------------------------------------------------------------
def sum_combine(a: Mapping, b: Mapping) -> Dict:
    """Consumable / accumulating combination: component-wise sum."""
    return {k: a[k] + b[k] for k in a}


def join_combine(a: Mapping, b: Mapping) -> Dict:
    """Renewable combination: component-wise maximum (join)."""
    return {k: max(a[k], b[k]) for k in a}


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------
@dataclass
class VecStage:
    """One stage of a vector-state sequential co-design problem.

    Parameters
    ----------
    name : str
        Stage identifier.
    functionality : callable
        ``functionality(state_vec) -> Mapping`` giving the outer
        functionality demanded at this stage as a function of the incoming
        carried state vector (a :data:`~codesign.state.StateVec`). Use
        :func:`~codesign.state.state_get` / :func:`~codesign.state.state_as_dict`
        to read axes.
    transition : callable
        ``transition(state_vec, point) -> Mapping[str, Any]`` mapping the
        incoming state vector and the chosen full resource point to the
        outgoing state as a plain ``{axis_name: value}`` mapping, which the
        solver snaps onto the grid. Returning the grid bottom makes the
        stage a reset.
    candidates : sequence of Architecture, optional
        Architectures admissible at this stage; falls back to the
        problem-level default when ``None``.
    admissible : callable, optional
        ``admissible(state_vec) -> bool`` forbidding states for which it
        returns ``False``.
    """

    name: str
    functionality: Callable[[StateVec], Mapping]
    transition: Callable[[StateVec, Mapping], Mapping[str, Any]]
    candidates: Optional[Sequence[Architecture]] = None
    admissible: Optional[Callable[[StateVec], bool]] = None


@dataclass
class VecResult:
    """Outcome of :func:`solve_vector_sequential`."""

    value: Antichain
    width: int
    feasible: bool
    policy: "VecPolicy"

    def __repr__(self) -> str:
        feas = "feasible" if self.feasible else "INFEASIBLE"
        return f"VecResult({feas}, width={self.width}, value={self.value!r})"


# ---------------------------------------------------------------------------
# Cost poset and per-stage antichain assembly (shared shape with sequential)
# ---------------------------------------------------------------------------
def _cost_poset(cost_axes: Sequence[str]) -> Ports:
    return Ports({name: Reals() for name in cost_axes})


def _key(point: Mapping, cost_axes: Sequence[str]) -> Tuple:
    return tuple(round(float(point[ax]), 9) for ax in cost_axes)


def _stage_antichain(
    cands: Sequence[Architecture],
    functionality: Mapping,
    cost_axes: Sequence[str],
    *,
    max_iter: int,
    solve_kwargs: Dict[str, Any],
) -> Tuple[Antichain, Dict[Tuple, str], Dict[Tuple, Mapping]]:
    """Union of candidates' solved antichains; keep origin and full points."""
    poset = _cost_poset(cost_axes)
    per_arch: List[Antichain] = []
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
            per_arch.append(Antichain.from_set(poset, pts))
    return Antichain.union_min(poset, per_arch), origin, full


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class VecPolicy:
    """State-vector-indexed antichain-valued policy."""

    def __init__(
        self,
        stage_names: Sequence[str],
        grid: VectorStateGrid,
        cost_axes: Sequence[str],
        value: List[Dict[StateVec, Antichain]],
        choice: List[Dict[StateVec, Dict[Tuple, Tuple[str, Mapping, StateVec]]]],
    ):
        self.stage_names = list(stage_names)
        self.grid = grid
        self.cost_axes = list(cost_axes)
        self._value = value
        self._choice = choice

    def value_at(self, k: int, state: StateVec) -> Antichain:
        poset = _cost_poset(self.cost_axes)
        node = self.grid.snap(state_as_dict(state))
        return self._value[k].get(node, Antichain.empty(poset))

    def width_at(self, k: int, state: StateVec) -> int:
        return len(self.value_at(k, state))

    def best_action_at(
        self, k: int, state: StateVec
    ) -> Optional[Tuple[str, Mapping, StateVec]]:
        """Return one realising ``(arch, full_point, succ_state)`` at (k, state).

        Picks the choice behind a minimal-first cost axis, useful for a
        myopic roll-out. Returns ``None`` if no finite choice exists.
        """
        node = self.grid.snap(state_as_dict(state))
        v = self._value[k].get(node)
        if v is None or v.is_empty():
            return None
        ch = self._choice[k].get(node, {})
        # Pick the front point minimal on the first cost axis.
        best_pt = min(v, key=lambda p: p[self.cost_axes[0]])
        return ch.get(_key(best_pt, self.cost_axes))


# ---------------------------------------------------------------------------
# Backward vector-state antichain-valued Bellman pass
# ---------------------------------------------------------------------------
def solve_vector_sequential(
    stages: Sequence[VecStage],
    grid: VectorStateGrid,
    *,
    cost_axes: Sequence[str],
    initial_state: Mapping[str, Any],
    combine: Callable[[Mapping, Mapping], Mapping] = sum_combine,
    architectures: Optional[Sequence[Architecture]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> VecResult:
    """Solve the vector-state antichain-valued sequential co-design problem.

    Identical in structure to :func:`codesign.sequential.solve_sequential`
    but carrying a full state vector on a
    :class:`~codesign.state.VectorStateGrid`. The transition returns a
    ``{axis: value}`` mapping snapped onto the product grid; out-of-bounds
    successors are rejected before snapping on every axis.

    Parameters
    ----------
    stages : sequence of VecStage
        Horizon in forward order (solved backward internally).
    grid : VectorStateGrid
        Product discretisation of the carried state vector.
    cost_axes : sequence of str
        Resource ports accumulated on the antichain (the poset ``R``).
    initial_state : mapping
        ``{axis: value}`` starting state; the value returned is
        ``V_0(initial_state)``.
    combine : callable
        Monoid combination ``(+)``: :func:`sum_combine` or
        :func:`join_combine`.
    architectures : sequence of Architecture, optional
        Default candidates for stages without their own.
    max_iter, solve_kwargs
        Forwarded to :func:`~codesign.solver.solve`.

    Returns
    -------
    VecResult
    """
    solve_kwargs = dict(solve_kwargs or {})
    T = len(stages)
    poset = _cost_poset(cost_axes)
    zero = {ax: 0.0 for ax in cost_axes}

    def candidates_for(st: VecStage) -> Sequence[Architecture]:
        if st.candidates is not None:
            return st.candidates
        if architectures is None:
            raise ValueError(
                f"stage {st.name!r} has no candidates and no default set"
            )
        return architectures

    value: List[Dict[StateVec, Antichain]] = [dict() for _ in range(T + 1)]
    choice: List[Dict[StateVec, Dict[Tuple, Tuple[str, Mapping, StateVec]]]] = [
        dict() for _ in range(T)
    ]
    for node in grid.nodes():
        value[T][node] = Antichain.singleton(poset, dict(zero))

    for k in range(T - 1, -1, -1):
        stage = stages[k]
        cands = candidates_for(stage)
        for x in grid.nodes():
            if stage.admissible is not None and not stage.admissible(x):
                value[k][x] = Antichain.empty(poset)
                choice[k][x] = {}
                continue
            func = stage.functionality(x)
            # Assemble h_k over ALL architecture points, not just the
            # cost-Pareto-reduced antichain. A point that is dominated on
            # the cost axes may still be the only feasible choice from a
            # constrained carried state (for example a higher-cost
            # morphology that spares a worn module), because its
            # consequence for the carried state is not visible in the cost
            # projection. Reducing to the cost antichain before the
            # transition would wrongly discard it. We therefore enumerate
            # the full solved points and let the final union_min prune only
            # after the carried-state transition has been applied.
            _, origin, full = _stage_antichain(
                cands, func, cost_axes,
                max_iter=max_iter, solve_kwargs=solve_kwargs,
            )
            terms: List[Antichain] = []
            node_choice: Dict[Tuple, Tuple[str, Mapping, StateVec]] = {}
            seen_full: set = set()
            for rkey, full_pt in full.items():
                if rkey in seen_full:
                    continue
                seen_full.add(rkey)
                r = {ax: full_pt[ax] for ax in cost_axes}
                succ_raw = stage.transition(x, full_pt)
                if not grid.in_bounds(succ_raw):
                    continue
                x_succ = grid.snap(succ_raw)
                if stage.admissible is not None and not stage.admissible(x_succ):
                    continue
                tail = value[k + 1].get(x_succ, Antichain.empty(poset))
                if tail.is_empty():
                    continue
                combined_pts = [combine(r, t) for t in tail]
                terms.append(Antichain.from_set(poset, combined_pts))
                arch_name = origin.get(rkey, "")
                for cp in combined_pts:
                    node_choice[_key(cp, cost_axes)] = (arch_name, full_pt, x_succ)
            vk = Antichain.union_min(poset, terms)
            value[k][x] = vk
            surviving = {_key(p, cost_axes) for p in vk}
            choice[k][x] = {
                key: val for key, val in node_choice.items() if key in surviving
            }

    policy = VecPolicy(
        stage_names=[st.name for st in stages],
        grid=grid,
        cost_axes=cost_axes,
        value=value,
        choice=choice,
    )
    v0 = policy.value_at(0, grid.snap(dict(initial_state)))
    feasible = (not v0.is_empty()) and (not v0.has_any_top())
    return VecResult(value=v0, width=len(v0), feasible=feasible, policy=policy)


# ---------------------------------------------------------------------------
# Vector monotonicity guard (H1/H2 over the product order)
# ---------------------------------------------------------------------------
@dataclass
class VectorMonotonicityReport:
    """Result of :func:`check_vector_monotonicity`."""

    h1_ok: bool
    h2_ok: bool
    h1_violations: List[Tuple[str, StateVec, StateVec]] = field(default_factory=list)
    h2_violations: List[Tuple[str, StateVec, StateVec]] = field(default_factory=list)

    @property
    def monotone_value_guaranteed(self) -> bool:
        return self.h1_ok and self.h2_ok

    def __repr__(self) -> str:
        return (
            f"VectorMonotonicityReport(H1={'ok' if self.h1_ok else 'FAIL'}, "
            f"H2={'ok' if self.h2_ok else 'FAIL'}, "
            f"value_monotone="
            f"{'guaranteed' if self.monotone_value_guaranteed else 'NOT guaranteed'})"
        )


def check_vector_monotonicity(
    stages: Sequence[VecStage],
    grid: VectorStateGrid,
    *,
    cost_axes: Sequence[str],
    architectures: Optional[Sequence[Architecture]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    max_violations: int = 8,
    max_pairs: int = 4000,
) -> VectorMonotonicityReport:
    """Verify (H1) and (H2) over the vector grid's product order.

    For every ordered pair of grid states ``x <= x'`` (in the product
    order), (H1) requires the stage antichain to be consistently oriented
    (no easier everywhere, or no harder everywhere) and (H2) requires the
    transition to be monotone in the state. The logic mirrors the scalar
    guard in :mod:`codesign.sequential`: a consistently oriented stage
    (including the benign consumable-but-monotone orientation) passes,
    while a genuinely non-monotone (perishable) stage is flagged.

    Because the product grid can be large, at most ``max_pairs`` ordered
    comparable pairs per stage are sampled; the report is exact when the
    grid is small enough that the cap is not reached.
    """
    solve_kwargs = dict(solve_kwargs or {})
    poset = _cost_poset(cost_axes)
    nodes = list(grid.nodes())

    def candidates_for(st: VecStage) -> Sequence[Architecture]:
        if st.candidates is not None:
            return st.candidates
        if architectures is None:
            raise ValueError(
                f"stage {st.name!r} has no candidates and no default set"
            )
        return architectures

    h1_violations: List[Tuple[str, StateVec, StateVec]] = []
    h2_violations: List[Tuple[str, StateVec, StateVec]] = []

    for st in stages:
        cands = candidates_for(st)
        ac: Dict[StateVec, Antichain] = {}
        full_by_node: Dict[StateVec, Dict[Tuple, Mapping]] = {}
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

        # Enumerate comparable ordered pairs x < x' up to the cap.
        pairs: List[Tuple[StateVec, StateVec]] = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                a, b = nodes[i], nodes[j]
                if grid.leq(a, b) and not grid.leq(b, a):  # a strictly below b
                    pairs.append((a, b))
                    if len(pairs) >= max_pairs:
                        break
            if len(pairs) >= max_pairs:
                break

        # (H1): consistent orientation across comparable pairs.
        nondecreasing = True   # larger state no easier
        nonincreasing = True   # larger state no harder
        for a, b in pairs:
            aa, ab = ac[a], ac[b]
            if aa.is_empty() or ab.is_empty():
                continue
            a_easier = aa.leq(ab)   # a no harder than b
            b_easier = ab.leq(aa)   # b no harder than a
            if not a_easier:
                nondecreasing = False
            if not b_easier:
                nonincreasing = False
        if not (nondecreasing or nonincreasing):
            for a, b in pairs:
                aa, ab = ac[a], ac[b]
                if aa.is_empty() or ab.is_empty():
                    continue
                if not aa.leq(ab) and not ab.leq(aa):
                    if len(h1_violations) < max_violations:
                        h1_violations.append((st.name, a, b))
                    break
            else:
                if len(h1_violations) < max_violations and pairs:
                    h1_violations.append((st.name, pairs[0][0], pairs[0][1]))

        # (H2): transition monotone in the state vector (product order).
        for a, b in pairs:
            broke = False
            for r in ac[a]:
                full_pt = full_by_node[a].get(_key(r, cost_axes), r)
                sa = grid.snap(st.transition(a, full_pt))
                sb = grid.snap(st.transition(b, full_pt))
                if not grid.leq(sa, sb):
                    if len(h2_violations) < max_violations:
                        h2_violations.append((st.name, a, b))
                    broke = True
                    break
            if broke:
                break

    return VectorMonotonicityReport(
        h1_ok=not h1_violations,
        h2_ok=not h2_violations,
        h1_violations=h1_violations,
        h2_violations=h2_violations,
    )


__all__ = [
    "VecStage",
    "VecResult",
    "VecPolicy",
    "solve_vector_sequential",
    "sum_combine",
    "join_combine",
    "VectorMonotonicityReport",
    "check_vector_monotonicity",
]
