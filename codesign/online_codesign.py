"""Online feedback co-design: closed-loop, measurement-driven re-solving.

The temporal layers so far are open-loop planners: fix the horizon,
environment sequence, and transitions up front, solve the whole dynamic
program offline, and read out a policy. This module adds the closed-loop
counterpart. At each real control step it senses the current measured
state of the process, the current requirement, and the current
environment, re-solves the co-design at those live conditions, applies
the first decision, and repeats. Because reality is read back from a
measurement rather than propagated from a nominal transition, the loop
corrects for the mismatch between plan and execution, which is the whole
point of feedback.

This is the co-design instance of *control co-design* (CCD) in its nested
form (an outer design loop wrapping an inner solve), specialised to a
receding-horizon, measurement-in-the-loop controller. Here we implement
the myopic variant (re-solve a single static co-design at the current
conditions each step); a receding-horizon lookahead that plans several
steps ahead with :mod:`codesign.vector_dp` and commits only the first is
a natural extension and is noted where it slots in.

The model is assumed known: measurements update the *carried state* and
the *conditions*, not the co-design model itself. Learning the model
online from measurements (adaptive co-design in the fullest sense) layers
on top of this loop using :mod:`codesign.online` and is deliberately left
for a later increment.

Structure of one step
---------------------
1. **Sense.** A user-supplied ``sensor`` returns the measured state
   (whatever the process actually is now: realised wear, actual charge),
   which may differ from any nominal prediction.
2. **Condition.** A ``requirement`` callable returns the functionality
   demanded now, and an ``environment`` callable returns exogenous
   parameters; both may depend on the measured state and the step index.
3. **Re-solve.** Each admissible architecture's co-design problem is
   solved at the live requirement; the cheapest feasible point (by a
   user cost) across architectures is selected.
4. **Apply.** The chosen architecture and point are committed; a
   ``plant`` callable advances the true process (a simulator here, the
   real world in deployment), returning the next measured state.
5. **Log.** A :class:`ControlStep` records the decision, the conditions,
   and the outcome, giving the audit trail a closed-loop operator needs.

The loop never assumes the plant follows the nominal model. That
separation, planning against a model but stepping the true (possibly
divergent) plant, is what makes this feedback rather than open-loop
replay.
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
from .temporal import Architecture

INF = float("inf")

# Types for the callables the loop is parameterised by.
Sensor = Callable[[int, Any], Any]                     # (step, prev_state) -> measured state
Requirement = Callable[[int, Any], Mapping]            # (step, state) -> functionality
Environment = Callable[[int, Any], Mapping[str, Any]]  # (step, state) -> exogenous params
Plant = Callable[[int, Any, str, Mapping], Any]        # (step, state, arch, point) -> next state
CostFn = Callable[[Mapping], float]


@dataclass
class ControlStep:
    """Audit record of one closed-loop control step."""

    step: int
    measured_state: Any
    requirement: Mapping
    environment: Mapping
    architecture: str
    point: Optional[Mapping]
    cost: float
    feasible: bool

    def __repr__(self) -> str:
        feas = "ok" if self.feasible else "INFEASIBLE"
        return (
            f"ControlStep(t={self.step}, arch={self.architecture!r}, "
            f"cost={self.cost:.4g}, {feas})"
        )


@dataclass
class OnlineCoDesignResult:
    """Trajectory produced by :func:`run_online_codesign`."""

    steps: List[ControlStep]
    total_cost: float
    feasible: bool

    @property
    def schedule(self) -> List[str]:
        return [s.architecture for s in self.steps]

    def __repr__(self) -> str:
        feas = "feasible" if self.feasible else "INFEASIBLE (a step failed)"
        return (
            f"OnlineCoDesignResult({feas}, steps={len(self.steps)}, "
            f"total_cost={self.total_cost:.4g})"
        )


# ---------------------------------------------------------------------------
# One myopic re-solve at the live conditions
# ---------------------------------------------------------------------------
def resolve_at(
    architectures: Sequence[Architecture],
    functionality: Mapping,
    cost_fn: CostFn,
    *,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Mapping], float]:
    """Solve every architecture at ``functionality`` and pick the cheapest.

    Returns ``(arch_name, point, cost)``; ``("", None, inf)`` when no
    architecture is feasible at these conditions. This is the co-design
    solve performed once per control step in the myopic loop.
    """
    solve_kwargs = dict(solve_kwargs or {})
    best_name, best_point, best_cost = "", None, INF
    for arch in architectures:
        res = solve(arch.dp, functionality, max_iter=max_iter, **solve_kwargs)
        if not res.feasible:
            continue
        pt = minimize_cost(res, cost_fn)
        if pt is None:
            continue
        c = float(cost_fn(pt))
        if c < best_cost:
            best_name, best_point, best_cost = arch.name, pt, c
    return best_name, best_point, best_cost


# ---------------------------------------------------------------------------
# The closed-loop controller
# ---------------------------------------------------------------------------
def run_online_codesign(
    architectures: Sequence[Architecture],
    *,
    n_steps: int,
    sensor: Sensor,
    requirement: Requirement,
    plant: Plant,
    cost_fn: CostFn,
    environment: Optional[Environment] = None,
    initial_state: Any = None,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> OnlineCoDesignResult:
    """Run the closed-loop, measurement-driven co-design loop.

    At each of ``n_steps`` steps the loop senses the measured state, reads
    the live requirement (and environment), re-solves the co-design over
    the admissible architectures, applies the cheapest feasible choice by
    stepping the ``plant``, and logs the outcome. The plant is stepped
    with the *true* process, so divergence from any nominal model is
    absorbed by the next sensing step, which is what makes this a feedback
    loop rather than open-loop replay.

    Parameters
    ----------
    architectures : sequence of Architecture
        The admissible configurations, re-evaluated every step.
    n_steps : int
        Number of closed-loop steps to run.
    sensor : callable
        ``sensor(step, prev_state) -> measured_state``. On the first step
        ``prev_state`` is ``initial_state``. In simulation this typically
        returns ``prev_state`` (optionally with measurement noise); in
        deployment it reads the real sensor.
    requirement : callable
        ``requirement(step, measured_state) -> functionality`` giving the
        live demand, which may change over time or with the state.
    plant : callable
        ``plant(step, measured_state, arch_name, point) -> next_state``
        advancing the true process under the applied decision. This is the
        simulator (or the real world). It need not agree with any nominal
        transition used for planning.
    cost_fn : callable
        Scalar cost on a resource point; the cheapest feasible point
        across architectures is selected each step.
    environment : callable, optional
        ``environment(step, measured_state) -> params`` for exogenous
        conditions. Recorded in the log; fold into ``requirement`` if it
        should influence the solve (for example a temperature that changes
        the demand). Defaults to empty.
    initial_state : any
        The state handed to the first ``sensor`` call.
    max_iter, solve_kwargs
        Forwarded to :func:`~codesign.solver.solve`.

    Returns
    -------
    OnlineCoDesignResult
    """
    solve_kwargs = dict(solve_kwargs or {})
    env_fn = environment or (lambda step, state: {})

    steps: List[ControlStep] = []
    total = 0.0
    all_ok = True
    state = initial_state

    for t in range(n_steps):
        # 1. Sense.
        measured = sensor(t, state)
        # 2. Conditions.
        func = requirement(t, measured)
        env = env_fn(t, measured)
        # 3. Re-solve at the live conditions.
        arch_name, point, cost = resolve_at(
            architectures, func, cost_fn,
            max_iter=max_iter, solve_kwargs=solve_kwargs,
        )
        feasible = point is not None
        all_ok = all_ok and feasible
        steps.append(ControlStep(
            step=t, measured_state=measured, requirement=dict(func),
            environment=dict(env), architecture=arch_name,
            point=point, cost=(cost if feasible else INF), feasible=feasible,
        ))
        if feasible:
            total += cost
            # 4. Apply: step the true plant under the chosen decision.
            state = plant(t, measured, arch_name, point)
        else:
            # No feasible configuration at these conditions. Hold the state
            # and let the next sensing step re-evaluate; the loop does not
            # abort, so the operator sees every gap in the trajectory.
            state = measured

    return OnlineCoDesignResult(
        steps=steps,
        total_cost=total,
        feasible=all_ok,
    )


__all__ = [
    "ControlStep",
    "OnlineCoDesignResult",
    "resolve_at",
    "run_online_codesign",
    "Sensor",
    "Requirement",
    "Environment",
    "Plant",
]
