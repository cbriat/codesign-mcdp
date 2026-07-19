"""Architecture switching across a changing environment (Case 1).

This module addresses the first temporal co-design problem: a single
system whose *best architecture changes over time* because the
environment, the mission, or the available resources change. At each
epoch one of several candidate architectures (each a fully specified
:class:`~codesign.system.System` or design problem) is the active one;
the task is to choose, for every epoch, which architecture to run.

The canonical example is an organism that switches carbon source and
the associated metabolic architecture: a glucose-regime network is one
architecture, an acetate-regime network another, and the environment
dictates which substrate is available in each epoch. Engineering
analogues include a vehicle that reconfigures between a high-power and
a high-efficiency drivetrain, or a sensor network that swaps topology
between a survey phase and a tracking phase.

If switching were free and stateless this would be a trivial loop over
epochs: solve each epoch independently and keep the cheapest feasible
architecture. The module earns its place by handling the coupling that
makes the schedule non-trivial:

* a **switching cost** charged whenever the active architecture changes
  between consecutive epochs (teardown, re-tooling, re-acclimation);
* **hysteresis**, a margin by which a challenger architecture must beat
  the incumbent before a switch is taken, which suppresses chattering
  between two near-equal options.

With either coupling present the optimal schedule is no longer
epoch-local and is found here by a small dynamic program over the
discrete choice of architecture per epoch (see
:func:`solve_schedule`). The per-epoch cost of an architecture is
obtained by solving its co-design problem with the ordinary
:func:`~codesign.solver.solve`, so this layer sits cleanly on top of the
existing solver without modifying it.

The design intentionally mirrors the interface of the dynamic-program
layer in :mod:`codesign.dynamic`: an ``Epoch`` plays the role of a DP
stage, an architecture plays the role of a decision, and
``solve_schedule`` is a Viterbi-style shortest path through the
epoch/architecture lattice. That symmetry is deliberate so the two
temporal layers compose and read consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .solver import minimize_cost, solve

# ---------------------------------------------------------------------------
# Cost convention
# ---------------------------------------------------------------------------
# A scalar cost function maps a single resource point (a Mapping from R
# port name to value) to a float. Lower is better. Infeasible epochs are
# represented by +inf so they are never selected unless nothing else is
# available, in which case the whole schedule is reported infeasible.
CostFn = Callable[[Mapping], float]

INF = float("inf")


@dataclass
class Architecture:
    """A named candidate configuration that can be active in an epoch.

    Parameters
    ----------
    name : str
        Identifier used in results, plots, and switching bookkeeping.
    dp : Any
        A design problem accepted by :func:`~codesign.solver.solve`
        (typically a built :class:`~codesign.system.System`). It may be
        shared across epochs or be epoch-specific; see
        :attr:`Epoch.candidates`.
    tags : mapping, optional
        Free-form metadata (for example the substrate an organism
        consumes, or the drivetrain mode). Not interpreted here; carried
        through to results for the caller's convenience.
    """

    name: str
    dp: Any
    tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Epoch:
    """One environment regime over which a single architecture is active.

    Parameters
    ----------
    name : str
        Identifier for the epoch (for example ``"glucose"`` or
        ``"phase_3"``).
    functionality : mapping
        The outer functionality demanded during this epoch, passed
        straight to :func:`~codesign.solver.solve`. This is where the
        changing environment enters: different epochs impose different
        requirements.
    candidates : sequence of Architecture, optional
        The architectures admissible in this epoch. When ``None`` the
        schedule-level default set is used (see
        :func:`solve_schedule`). Supplying a per-epoch list models the
        common case where some architectures are simply unavailable in
        some regimes (no glucose-metabolism when there is no glucose).
    duration : float
        A weight applied to the epoch's running cost, letting unequal
        epoch lengths be expressed without rescaling the cost function.
    """

    name: str
    functionality: Mapping
    candidates: Optional[Sequence[Architecture]] = None
    duration: float = 1.0


@dataclass
class EpochResult:
    """Per-epoch outcome within a solved schedule."""

    epoch: str
    architecture: str
    running_cost: float
    switch_cost: float
    feasible: bool
    point: Optional[Mapping]
    tags: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return self.running_cost + self.switch_cost


@dataclass
class ScheduleResult:
    """Outcome of :func:`solve_schedule`.

    Attributes
    ----------
    epochs : list of EpochResult
        One entry per epoch, in order, recording the chosen architecture
        and its decomposed cost.
    total_cost : float
        Sum over epochs of running plus switching cost.
    feasible : bool
        ``True`` iff every epoch was satisfiable by its chosen
        architecture.
    n_switches : int
        Number of epoch boundaries at which the active architecture
        changed.
    """

    epochs: List[EpochResult]
    total_cost: float
    feasible: bool
    n_switches: int

    @property
    def schedule(self) -> List[str]:
        """The chosen architecture name per epoch, in order."""
        return [e.architecture for e in self.epochs]

    def __repr__(self) -> str:
        feas = "feasible" if self.feasible else "INFEASIBLE"
        arcs = " -> ".join(self.schedule)
        return (
            f"ScheduleResult({feas}, cost={self.total_cost:.4g}, "
            f"switches={self.n_switches}, [{arcs}])"
        )


# ---------------------------------------------------------------------------
# Per-epoch evaluation
# ---------------------------------------------------------------------------
def _epoch_running_cost(
    arch: Architecture,
    epoch: Epoch,
    cost_fn: CostFn,
    *,
    max_iter: int,
    solve_kwargs: Dict[str, Any],
) -> Tuple[float, Optional[Mapping]]:
    """Solve one architecture in one epoch and return (cost, point).

    The cost is the epoch duration times the scalar cost of the cheapest
    feasible resource bundle. An infeasible solve yields ``(inf, None)``
    so the schedule optimiser avoids it when alternatives exist.
    """
    result = solve(arch.dp, epoch.functionality, max_iter=max_iter, **solve_kwargs)
    if not result.feasible:
        return INF, None
    best = minimize_cost(result, cost_fn)
    if best is None:
        return INF, None
    return epoch.duration * float(cost_fn(best)), best


# ---------------------------------------------------------------------------
# Schedule optimisation (Viterbi over the epoch/architecture lattice)
# ---------------------------------------------------------------------------
def solve_schedule(
    epochs: Sequence[Epoch],
    architectures: Optional[Sequence[Architecture]] = None,
    *,
    cost_fn: CostFn,
    switch_cost: float | Callable[[Architecture, Architecture], float] = 0.0,
    hysteresis: float = 0.0,
    max_iter: int = 200,
    solve_kwargs: Optional[Dict[str, Any]] = None,
) -> ScheduleResult:
    """Choose an architecture per epoch, minimising running + switching cost.

    The optimisation is an exact dynamic program (a Viterbi pass) over
    the lattice whose nodes are (epoch, architecture) pairs. The running
    cost of each node is obtained by solving that architecture's
    co-design problem at the epoch's functionality; the transition cost
    between consecutive epochs is the switching cost (zero when the
    architecture is unchanged). With ``switch_cost == 0`` and
    ``hysteresis == 0`` the result reduces to the epoch-local greedy
    choice, as expected.

    Parameters
    ----------
    epochs : sequence of Epoch
        The ordered environment regimes.
    architectures : sequence of Architecture, optional
        Default candidate set used for any epoch that does not supply its
        own ``candidates``. Required if any epoch leaves ``candidates``
        as ``None``.
    cost_fn : callable
        Maps a resource point to a scalar; lower is better.
    switch_cost : float or callable
        Cost charged when the active architecture changes between two
        consecutive epochs. A float applies uniformly; a callable
        ``f(prev_arch, next_arch) -> float`` lets the penalty depend on
        which transition is taken (for example, re-acclimating from
        acetate to glucose may differ from the reverse).
    hysteresis : float
        Extra margin (in cost units) a challenger must beat the incumbent
        by before a switch is preferred at equal switching cost. Models
        sticky preferences and suppresses chattering. Applied as a tie
        break, it never makes a strictly worse schedule look better than
        the optimum by more than ``hysteresis`` per retained boundary.
    max_iter : int
        Forwarded to :func:`~codesign.solver.solve` for each epoch solve.
    solve_kwargs : mapping, optional
        Extra keyword arguments forwarded to
        :func:`~codesign.solver.solve` (for example ``uncertainty``).

    Returns
    -------
    ScheduleResult
    """
    if not epochs:
        return ScheduleResult(epochs=[], total_cost=0.0, feasible=True, n_switches=0)

    solve_kwargs = dict(solve_kwargs or {})

    # Resolve each epoch's candidate list, falling back to the default.
    def candidates_for(ep: Epoch) -> Sequence[Architecture]:
        if ep.candidates is not None:
            return ep.candidates
        if architectures is None:
            raise ValueError(
                f"epoch {ep.name!r} has no candidates and no default "
                f"architecture set was supplied. Either set candidates=[...] "
                f"on the epoch, or pass architectures=[...] to solve_schedule()."
            )
        return architectures

    switch_fn = (
        switch_cost
        if callable(switch_cost)
        else (lambda a, b: 0.0 if a.name == b.name else float(switch_cost))
    )

    # Pre-compute running cost and chosen point for every (epoch, arch).
    running: List[Dict[str, Tuple[float, Optional[Mapping]]]] = []
    cand_by_epoch: List[Sequence[Architecture]] = []
    for ep in epochs:
        cands = candidates_for(ep)
        cand_by_epoch.append(cands)
        per_arch: Dict[str, Tuple[float, Optional[Mapping]]] = {}
        for arch in cands:
            per_arch[arch.name] = _epoch_running_cost(
                arch, ep, cost_fn, max_iter=max_iter, solve_kwargs=solve_kwargs
            )
        running.append(per_arch)

    # Viterbi forward pass. State is the active architecture name.
    # best[i][name] = (min cumulative cost to end epoch i with `name`
    #                  active, back-pointer to previous arch name).
    arch_lookup: Dict[str, Architecture] = {
        a.name: a for cands in cand_by_epoch for a in cands
    }

    best: List[Dict[str, Tuple[float, Optional[str]]]] = []
    first: Dict[str, Tuple[float, Optional[str]]] = {}
    for arch in cand_by_epoch[0]:
        rc, _pt = running[0][arch.name]
        first[arch.name] = (rc, None)
    best.append(first)

    for i in range(1, len(epochs)):
        layer: Dict[str, Tuple[float, Optional[str]]] = {}
        for arch in cand_by_epoch[i]:
            rc, _pt = running[i][arch.name]
            best_prev_cost = INF
            best_prev_name: Optional[str] = None
            for prev_name, (prev_cost, _bp) in best[i - 1].items():
                if prev_cost == INF:
                    continue
                prev_arch = arch_lookup[prev_name]
                trans = switch_fn(prev_arch, arch)
                # Hysteresis biases toward keeping the incumbent: a switch
                # must overcome the margin to be chosen at equal cost.
                bias = 0.0 if prev_name == arch.name else hysteresis
                cand_cost = prev_cost + trans + bias
                if cand_cost < best_prev_cost:
                    best_prev_cost = cand_cost
                    best_prev_name = prev_name
            layer[arch.name] = (best_prev_cost + rc, best_prev_name)
        best.append(layer)

    # Terminal: pick the cheapest end state, then back-trace.
    last = best[-1]
    end_name = min(last, key=lambda n: last[n][0])
    end_cost = last[end_name][0]

    chosen: List[str] = [end_name]
    for i in range(len(epochs) - 1, 0, -1):
        _cost, bp = best[i][chosen[-1]]
        chosen.append(bp if bp is not None else chosen[-1])
    chosen.reverse()

    # Reconstruct decomposed per-epoch results.
    epoch_results: List[EpochResult] = []
    n_switches = 0
    prev_name: Optional[str] = None
    total = 0.0
    all_feasible = True
    for i, ep in enumerate(epochs):
        name = chosen[i]
        rc, pt = running[i][name]
        feas = rc != INF and pt is not None
        all_feasible = all_feasible and feas
        if prev_name is not None and prev_name != name:
            sc = switch_fn(arch_lookup[prev_name], arch_lookup[name])
            n_switches += 1
        else:
            sc = 0.0
        total += (0.0 if rc == INF else rc) + sc
        epoch_results.append(
            EpochResult(
                epoch=ep.name,
                architecture=name,
                running_cost=rc,
                switch_cost=sc,
                feasible=feas,
                point=pt,
                tags=dict(arch_lookup[name].tags),
            )
        )
        prev_name = name

    # If the optimiser was forced through an infeasible node the reported
    # cost is dominated by +inf running costs; surface that as infeasible
    # while still returning the (best-effort) schedule for inspection.
    feasible = all_feasible and end_cost != INF
    return ScheduleResult(
        epochs=epoch_results,
        total_cost=total,
        feasible=feasible,
        n_switches=n_switches,
    )


__all__ = [
    "Architecture",
    "Epoch",
    "EpochResult",
    "ScheduleResult",
    "solve_schedule",
]
