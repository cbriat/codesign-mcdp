"""Dynamically changing architectures found by co-design (Case 2).

This module implements a finite-horizon dynamic program whose per-stage
*decision is which architecture to instantiate*, and whose per-stage
*cost is itself the solution of a co-design problem*. A scalar state is
carried from one stage to the next, which lets the model express
resources that deplete or accumulate over time: fuel burned, battery
charge consumed, cumulative wear, inventory drawn down. The architecture
chosen at each stage may differ, so the optimal policy is a *schedule of
architectures* threaded through a changing internal state.

This is deliberately *not* "a co-design problem in which a DP is solved".
The DP is the outer object; co-design is invoked once per (stage, state,
architecture) to score a candidate. The Bellman recursion is

    V_t(s) = min over architectures a of
                 [ stage_cost(t, s, a) + V_{t+1}( transition(t, s, a) ) ]

where ``stage_cost`` runs :func:`~codesign.solver.solve` on architecture
``a`` under a functionality that may depend on the current state ``s``,
and ``transition`` updates the carried resource using a quantity read off
the solved antichain (for example, the fuel a configuration burns this
stage). Terminal states are scored by an optional terminal cost.

Scope and a deliberate simplification
--------------------------------------
The carried state here is a **scalar** (or a short fixed tuple), and the
value function ``V_t`` is therefore scalar-valued: at each (stage, state)
the cost is a single number, the best achievable from here on. This is
the prototype layer. The harder, antichain-valued version, where the
value function is itself a Pareto front over (cost, end-state) and the
Bellman ``min`` becomes an antichain union-and-minimise, is a planned
extension; the interfaces below (notably :class:`Stage` and
:class:`DynamicResult`) are shaped so that upgrade localises to the
backward pass rather than rippling through the API.

State is handled by discretisation onto a user-supplied grid, which keeps
the prototype honest about the fact that a continuous carried resource
must be bucketed for tabular DP. The grid is explicit in the API so the
discretisation is never hidden from the caller.

The structure mirrors :mod:`codesign.temporal`: an :class:`Architecture`
there and here plays the same role, a :class:`Stage` is the dynamic analogue
of an ``Epoch``, and :func:`solve_dynamic` is the backward-DP analogue of
that module's forward Viterbi pass.
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

from .solver import minimize_cost, solve
from .temporal import Architecture  # shared decision object

INF = float("inf")

# A scalar cost on a resource point; lower is better.
CostFn = Callable[[Mapping], float]


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    """One step of the finite-horizon dynamic program.

    A stage bundles everything needed to score and advance every
    candidate architecture at this point in time, given the carried
    state.

    Parameters
    ----------
    name : str
        Identifier for the stage.
    functionality : callable
        ``functionality(state) -> Mapping`` returns the outer
        functionality demanded at this stage when the carried resource
        has value ``state``. State-dependence is the whole point: a
        survey stage might demand more coverage when more battery remains,
        or a mission leg might shorten as fuel runs low. A state-
        independent stage simply ignores its argument.
    candidates : sequence of Architecture, optional
        Architectures admissible at this stage. Falls back to the
        schedule-level default when ``None``.
    transition : callable
        ``transition(state, point) -> new_state`` maps the incoming
        carried resource and the *chosen resource point* (from the solved
        antichain) to the outgoing carried resource. This is where a
        depleting resource is decremented, for example
        ``new = state - point["fuel_burned"]``. The returned value is
        snapped to the nearest grid node by the solver.
    admissible : callable, optional
        ``admissible(state) -> bool``. When supplied, states for which it
        returns ``False`` are treated as forbidden (infinite value); use
        it to forbid negative fuel or over-full inventory rather than
        encoding the bound inside every cost function.
    """

    name: str
    functionality: Callable[[float], Mapping]
    transition: Callable[[float, Mapping], float]
    candidates: Optional[Sequence[Architecture]] = None
    admissible: Optional[Callable[[float], bool]] = None


@dataclass
class StageResult:
    """Per-stage record along a rolled-out optimal policy."""

    stage: str
    architecture: str
    state_in: float
    state_out: float
    stage_cost: float
    feasible: bool
    point: Optional[Mapping]
    tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class DynamicResult:
    """Outcome of :func:`solve_dynamic` rolled out from an initial state.

    Attributes
    ----------
    stages : list of StageResult
        Per-stage record of the optimal policy from the initial state.
    total_cost : float
        Optimal cost-to-go from the initial state (running plus terminal).
    feasible : bool
        ``True`` iff a finite-cost policy exists from the initial state
        and every stage along it was satisfiable.
    policy : DynamicPolicy
        The full state-indexed policy table, queryable at any
        (stage, state) for closed-loop use, not just the rolled-out path.
    """

    stages: List[StageResult]
    total_cost: float
    feasible: bool
    policy: "DynamicPolicy"

    @property
    def schedule(self) -> List[str]:
        """Architecture chosen at each stage along the rolled-out path."""
        return [s.architecture for s in self.stages]

    def __repr__(self) -> str:
        feas = "feasible" if self.feasible else "INFEASIBLE"
        arcs = " -> ".join(self.schedule)
        return (
            f"DynamicResult({feas}, cost={self.total_cost:.4g}, "
            f"[{arcs}])"
        )


# ---------------------------------------------------------------------------
# State grid
# ---------------------------------------------------------------------------
class StateGrid:
    """A 1-D discretisation of the carried scalar resource.

    Tabular dynamic programming needs a finite state set, so a continuous
    carried resource (fuel, charge, wear) is bucketed onto an explicit
    grid of nodes. Transitions that land between nodes are snapped to the
    nearest node. Keeping the grid an explicit object, rather than hiding
    it inside the solver, makes the discretisation visible and lets the
    caller trade accuracy against cost directly.
    """

    def __init__(self, nodes: Sequence[float]):
        if not nodes:
            raise ValueError(
                "StateGrid needs at least one node, got an empty sequence. "
                "Pass the grid node values explicitly, e.g. "
                "StateGrid([0.0, 1.0, 2.0]), or use StateGrid.linspace(lo, hi, n)."
            )
        self.nodes: List[float] = sorted(float(n) for n in nodes)

    @classmethod
    def linspace(cls, lo: float, hi: float, n: int) -> "StateGrid":
        """Build an evenly spaced grid of ``n`` nodes on ``[lo, hi]``."""
        if n < 1:
            raise ValueError(
                f"StateGrid.linspace needs at least one node, got n={n}. "
                f"Pass n >= 1 (n=1 yields a single node at lo)."
            )
        if n == 1:
            return cls([lo])
        step = (hi - lo) / (n - 1)
        return cls([lo + i * step for i in range(n)])

    def snap(self, value: float) -> float:
        """Return the grid node nearest to ``value``."""
        best = self.nodes[0]
        best_d = abs(value - best)
        for node in self.nodes[1:]:
            d = abs(value - node)
            if d < best_d:
                best_d = d
                best = node
        return best

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------------
class DynamicPolicy:
    """A solved, state-indexed policy: best architecture per (stage, state).

    Produced by :func:`solve_dynamic`. Beyond the single rolled-out path,
    the full table supports closed-loop control: at run time the true
    realised state may differ from the nominal roll-out (a leg burned more
    fuel than modelled), and the policy can be re-queried at the actual
    state.
    """

    def __init__(
        self,
        stage_names: Sequence[str],
        grid: StateGrid,
        # value[t][state_node] -> cost-to-go
        value: List[Dict[float, float]],
        # action[t][state_node] -> (arch_name, point, stage_cost, state_out)
        action: List[Dict[float, Tuple[str, Optional[Mapping], float, float]]],
    ):
        self.stage_names = list(stage_names)
        self.grid = grid
        self._value = value
        self._action = action

    def cost_to_go(self, t: int, state: float) -> float:
        """Optimal remaining cost from stage ``t`` at ``state``."""
        node = self.grid.snap(state)
        return self._value[t].get(node, INF)

    def action_at(
        self, t: int, state: float
    ) -> Optional[Tuple[str, Optional[Mapping], float, float]]:
        """Best ``(arch, point, stage_cost, state_out)`` at (``t``, ``state``).

        Returns ``None`` if no finite-cost action exists from here.
        """
        node = self.grid.snap(state)
        act = self._action[t].get(node)
        if act is None or act[2] == INF:
            return None
        return act


# ---------------------------------------------------------------------------
# Backward dynamic program
# ---------------------------------------------------------------------------
def _stage_cost_and_point(
    arch: Architecture,
    functionality: Mapping,
    cost_fn: CostFn,
    *,
    max_iter: int,
    solve_kwargs: Dict[str, Any],
) -> Tuple[float, Optional[Mapping]]:
    """Solve one architecture under one functionality; return (cost, point)."""
    result = solve(arch.dp, functionality, max_iter=max_iter, **solve_kwargs)
    if not result.feasible:
        return INF, None
    best = minimize_cost(result, cost_fn)
    if best is None:
        return INF, None
    return float(cost_fn(best)), best


def solve_dynamic(
    stages: Sequence[Stage],
    grid: StateGrid,
    *,
    cost_fn: CostFn,
    architectures: Optional[Sequence[Architecture]] = None,
    terminal_cost: Optional[Callable[[float], float]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    cache: bool = True,
) -> DynamicPolicy:
    """Solve the finite-horizon architecture DP, returning a full policy.

    Runs the standard backward Bellman pass over the (stage, state)
    lattice. At each (stage, state) and for each admissible architecture
    it solves the co-design problem for that architecture at the stage's
    state-dependent functionality, reads the cheapest feasible point,
    advances the carried state through the stage transition, and adds the
    discretised cost-to-go of the successor state. The minimising
    architecture defines the policy at that (stage, state).

    Parameters
    ----------
    stages : sequence of Stage
        The horizon, in forward order. The DP is solved backward
        internally.
    grid : StateGrid
        Discretisation of the carried scalar resource.
    cost_fn : callable
        Scalar cost on a resource point; lower is better.
    architectures : sequence of Architecture, optional
        Default candidates for stages that do not supply their own.
    terminal_cost : callable, optional
        ``terminal_cost(state) -> float`` scoring the state left after the
        final stage (for example, a reward for leftover fuel encoded as a
        negative cost, or a penalty for unmet end conditions). Defaults to
        zero everywhere.
    max_iter : int
        Forwarded to :func:`~codesign.solver.solve`.
    solve_kwargs : mapping, optional
        Extra keyword arguments forwarded to :func:`~codesign.solver.solve`.
    cache : bool
        Cache co-design solves keyed by (architecture name, functionality)
        so that identical (stage, state) functionalities are not re-solved.
        Safe whenever ``functionality(state)`` is deterministic.

    Returns
    -------
    DynamicPolicy
        The state-indexed optimal policy. Roll it out from a concrete
        initial state with :func:`rollout`.
    """
    solve_kwargs = dict(solve_kwargs or {})
    T = len(stages)
    term = terminal_cost or (lambda s: 0.0)

    def candidates_for(st: Stage) -> Sequence[Architecture]:
        if st.candidates is not None:
            return st.candidates
        if architectures is None:
            raise ValueError(
                f"stage {st.name!r} has no candidates and no default "
                f"architecture set was supplied. Either set candidates=[...] "
                f"on the stage, or pass architectures=[...] to solve_dynamic()."
            )
        return architectures

    # Optional memoisation of co-design solves. The functionality mapping
    # is reduced to a hashable key; if values are unhashable the cache is
    # skipped for that call rather than failing.
    solve_cache: Dict[Tuple[str, Tuple], Tuple[float, Optional[Mapping]]] = {}

    def scored(arch: Architecture, func: Mapping) -> Tuple[float, Optional[Mapping]]:
        if cache:
            try:
                key = (arch.name, tuple(sorted(func.items())))
            except TypeError:
                key = None
            if key is not None and key in solve_cache:
                return solve_cache[key]
            res = _stage_cost_and_point(
                arch, func, cost_fn, max_iter=max_iter, solve_kwargs=solve_kwargs
            )
            if key is not None:
                solve_cache[key] = res
            return res
        return _stage_cost_and_point(
            arch, func, cost_fn, max_iter=max_iter, solve_kwargs=solve_kwargs
        )

    # Terminal value layer: cost-to-go after the last stage is the
    # terminal cost of the resulting state.
    value: List[Dict[float, float]] = [dict() for _ in range(T + 1)]
    action: List[Dict[float, Tuple[str, Optional[Mapping], float, float]]] = [
        dict() for _ in range(T)
    ]
    for node in grid:
        value[T][node] = float(term(node))

    # Backward pass.
    for t in range(T - 1, -1, -1):
        stage = stages[t]
        cands = candidates_for(stage)
        for s in grid:
            if stage.admissible is not None and not stage.admissible(s):
                value[t][s] = INF
                action[t][s] = ("", None, INF, s)
                continue
            func = stage.functionality(s)
            best_total = INF
            best_act: Tuple[str, Optional[Mapping], float, float] = ("", None, INF, s)
            for arch in cands:
                sc, point = scored(arch, func)
                if sc == INF or point is None:
                    continue
                s_next_raw = stage.transition(s, point)
                # Guard against the snap-masking hazard: a transition that
                # lands outside the grid envelope (for example negative
                # fuel) must not be rescued by snapping back to the nearest
                # in-range node. Reject it before snapping. The small
                # tolerance admits transitions that land just past a
                # boundary node by rounding error.
                lo, hi = grid.nodes[0], grid.nodes[-1]
                span = hi - lo
                tol = 1e-9 * (span if span > 0 else 1.0)
                if s_next_raw < lo - tol or s_next_raw > hi + tol:
                    continue
                # Honour an explicit admissibility predicate on the raw
                # successor state too, so callers can forbid regions that
                # lie inside the grid envelope.
                if stage.admissible is not None and not stage.admissible(s_next_raw):
                    continue
                s_next = grid.snap(s_next_raw)
                cost_to_go = value[t + 1].get(s_next, INF)
                if cost_to_go == INF:
                    continue
                total = sc + cost_to_go
                if total < best_total:
                    best_total = total
                    best_act = (arch.name, point, sc, s_next)
            value[t][s] = best_total
            action[t][s] = best_act

    return DynamicPolicy(
        stage_names=[st.name for st in stages],
        grid=grid,
        value=value,
        action=action,
    )


def rollout(
    policy: DynamicPolicy,
    stages: Sequence[Stage],
    initial_state: float,
    *,
    arch_lookup: Optional[Mapping[str, Architecture]] = None,
) -> DynamicResult:
    """Roll a solved policy forward from a concrete initial state.

    Follows the policy's chosen architecture at each stage, advancing the
    *snapped* carried state, and assembles a per-stage record plus the
    total cost-to-go from the initial state.

    Parameters
    ----------
    policy : DynamicPolicy
        A policy produced by :func:`solve_dynamic`.
    stages : sequence of Stage
        The same stage list passed to :func:`solve_dynamic`.
    initial_state : float
        The starting value of the carried resource.
    arch_lookup : mapping, optional
        Maps architecture name to :class:`Architecture` so tags can be
        attached to each stage record. Built automatically from the stage
        candidate lists when omitted.
    """
    if arch_lookup is None:
        arch_lookup = {}
        for st in stages:
            for a in st.candidates or []:
                arch_lookup[a.name] = a

    state = policy.grid.snap(initial_state)
    total = policy.cost_to_go(0, state)
    feasible = total != INF

    records: List[StageResult] = []
    for t, st in enumerate(stages):
        act = policy.action_at(t, state)
        if act is None:
            records.append(
                StageResult(
                    stage=st.name,
                    architecture="",
                    state_in=state,
                    state_out=state,
                    stage_cost=INF,
                    feasible=False,
                    point=None,
                )
            )
            feasible = False
            break
        arch_name, point, sc, s_out = act
        tags = dict(arch_lookup.get(arch_name, Architecture(arch_name, None)).tags) \
            if arch_name in arch_lookup else {}
        records.append(
            StageResult(
                stage=st.name,
                architecture=arch_name,
                state_in=state,
                state_out=s_out,
                stage_cost=sc,
                feasible=True,
                point=point,
                tags=tags,
            )
        )
        state = s_out

    return DynamicResult(
        stages=records,
        total_cost=total,
        feasible=feasible,
        policy=policy,
    )


def solve_and_rollout(
    stages: Sequence[Stage],
    grid: StateGrid,
    initial_state: float,
    *,
    cost_fn: CostFn,
    architectures: Optional[Sequence[Architecture]] = None,
    terminal_cost: Optional[Callable[[float], float]] = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    cache: bool = True,
) -> DynamicResult:
    """Convenience wrapper: solve the policy, then roll it out once.

    Equivalent to calling :func:`solve_dynamic` and then :func:`rollout`
    from ``initial_state``. Use the two-step form directly when the policy
    will be queried at multiple initial states or in closed loop.
    """
    policy = solve_dynamic(
        stages,
        grid,
        cost_fn=cost_fn,
        architectures=architectures,
        terminal_cost=terminal_cost,
        max_iter=max_iter,
        solve_kwargs=solve_kwargs,
        cache=cache,
    )
    return rollout(policy, stages, initial_state)


__all__ = [
    "Stage",
    "StageResult",
    "DynamicResult",
    "StateGrid",
    "DynamicPolicy",
    "solve_dynamic",
    "rollout",
    "solve_and_rollout",
]
