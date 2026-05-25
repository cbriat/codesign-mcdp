"""
Design problems: the core abstraction.

A DesignProblem is a relation between a functionality space F and a resource
space R. It exposes one method, ``h(f) -> Antichain``, that returns the set of
minimal resources needed to deliver functionality f, an antichain in R.

Several concrete primitives are provided:

    AlgebraicDP   : R is determined by closed-form (monotone) formulas in f.
    FunctionDP    : wrap a user-supplied Python function f -> Antichain.
    CatalogDP     : choose from a discrete catalog of implementations.
    ConstraintDP  : feasibility predicate plus a cost function (lifted to Min).
    ODE_DP        : derive a monotone steady-state relation from an ODE.
    UncertainDP   : pessimistic + optimistic bracket around a nominal DP.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Sequence

from .antichains import Antichain
from .posets import Ports, Poset, Reals


class DesignProblem(ABC):
    """A monotone design problem: F -> antichain in R."""

    F: Poset
    R: Poset
    name: str = "dp"

    @abstractmethod
    def h(self, f) -> Antichain:
        """Return the minimal resources needed to deliver functionality f."""

    def __repr__(self) -> str:
        return f"DP({self.name}: {self.F.name} -> A[{self.R.name}])"


# ---------------------------------------------------------------------------
# AlgebraicDP
# ---------------------------------------------------------------------------


class AlgebraicDP(DesignProblem):
    """DP where each resource is a closed-form function of the functionality.

    ``equations`` maps each resource name to a callable f_dict -> value (or a
    constant number). The result is a singleton antichain because the relation
    is functional.

    Note: monotonicity is the user's responsibility; the framework relies on
    it but does not verify it.
    """

    def __init__(
        self,
        F: Poset,
        R: Poset,
        equations: Mapping[str, Callable[[Mapping[str, Any]], Any]],
        name: str = "algebraic",
    ):
        if not isinstance(R, Ports):
            raise TypeError("AlgebraicDP requires R to be a Ports")
        missing = set(R.keys()) - set(equations)
        if missing:
            raise ValueError(f"missing equations for resources: {missing}")
        self.F = F
        self.R = R
        self.equations = dict(equations)
        self.name = name

    def h(self, f) -> Antichain:
        result = {}
        for k, expr in self.equations.items():
            result[k] = expr(f) if callable(expr) else expr
        return Antichain.singleton(self.R, result)


# ---------------------------------------------------------------------------
# FunctionDP
# ---------------------------------------------------------------------------


class FunctionDP(DesignProblem):
    """Wrap an arbitrary function ``f -> Antichain`` or ``f -> point/iterable``.

    The user guarantees Scott-continuity / monotonicity.
    """

    def __init__(
        self,
        F: Poset,
        R: Poset,
        h_fn: Callable[[Any], Any],
        name: str = "function",
    ):
        self.F = F
        self.R = R
        self._h_fn = h_fn
        self.name = name

    def h(self, f) -> Antichain:
        result = self._h_fn(f)
        if isinstance(result, Antichain):
            return result
        if isinstance(result, dict):
            return Antichain.singleton(self.R, result)
        if not hasattr(result, "__iter__"):
            return Antichain.singleton(self.R, result)
        return Antichain.from_set(self.R, result)


# ---------------------------------------------------------------------------
# CatalogDP
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """One implementation in a catalog.

    ``provides`` is the functionality this implementation can deliver (it can
    fulfil any f <= provides). ``costs`` is the resource vector it costs.
    """

    provides: Mapping[str, Any]
    costs: Mapping[str, Any]
    name: str = ""


class CatalogDP(DesignProblem):
    """Pick the cheapest implementation(s) able to deliver functionality f.

    Each catalog entry can fulfil functionality f if its ``provides`` vector
    dominates f component-wise. Among all entries that work, we take the Min
    of their costs to form the resulting antichain.
    """

    def __init__(
        self,
        F: Ports,
        R: Ports,
        catalog: Sequence,
        name: str = "catalog",
    ):
        if not isinstance(F, Ports) or not isinstance(R, Ports):
            raise TypeError("CatalogDP requires Ports F and R")
        self.F = F
        self.R = R
        self.name = name
        self.catalog: List[CatalogEntry] = []
        for entry in catalog:
            if isinstance(entry, CatalogEntry):
                self.catalog.append(entry)
            else:
                self.catalog.append(
                    CatalogEntry(
                        provides=entry["provides"],
                        costs=entry["costs"],
                        name=entry.get("name", ""),
                    )
                )

    def h(self, f) -> Antichain:
        feasible_costs = []
        for entry in self.catalog:
            if self.F.leq(f, entry.provides):
                feasible_costs.append(dict(entry.costs))
        if not feasible_costs:
            return Antichain.singleton(self.R, self.R.top())
        return Antichain.from_set(self.R, feasible_costs)


# ---------------------------------------------------------------------------
# ConstraintDP
# ---------------------------------------------------------------------------


class ConstraintDP(DesignProblem):
    """DP defined by a feasibility predicate and an explicit cost function.

    Iterates a user-provided sampler over the implementation space; the result
    is Min over all feasible implementations.
    """

    def __init__(
        self,
        F: Poset,
        R: Ports,
        sampler: Callable[[Any], Iterable[Any]],
        feasible: Callable[[Any, Any], bool],
        cost: Callable[[Any], Mapping[str, Any]],
        name: str = "constraint",
    ):
        self.F = F
        self.R = R
        self.sampler = sampler
        self.feasible = feasible
        self.cost = cost
        self.name = name

    def h(self, f) -> Antichain:
        feasible_costs = []
        for impl in self.sampler(f):
            if self.feasible(impl, f):
                feasible_costs.append(dict(self.cost(impl)))
        if not feasible_costs:
            return Antichain.singleton(self.R, self.R.top())
        return Antichain.from_set(self.R, feasible_costs)


# ---------------------------------------------------------------------------
# ODE_DP: monotone relation derived from an ODE's steady-state or final state
# ---------------------------------------------------------------------------


class ODE_DP(DesignProblem):
    """Derive a relation from a (typically scalar) ODE.

    The user supplies dx/dt = rhs(x, t, f) along with a method for extracting
    the resource(s) from the trajectory: either ``steady_state`` (fixed point
    of rhs) or ``final_value`` (integrate to t_end and read off x).
    """

    def __init__(
        self,
        F: Poset,
        R: Ports,
        rhs: Callable[[Any, float, Any], Any],
        extract: Callable[[Any], Mapping[str, Any]],
        mode: str = "final_value",
        t_end: float = 10.0,
        n_steps: int = 200,
        x0_fn: Callable[[Any], Any] | None = None,
        name: str = "ode",
    ):
        self.F = F
        self.R = R
        self.rhs = rhs
        self.extract = extract
        self.mode = mode
        self.t_end = t_end
        self.n_steps = n_steps
        self.x0_fn = x0_fn or (lambda f: 0.0)
        self.name = name

    def _simulate(self, f) -> Any:
        x = self.x0_fn(f)
        dt = self.t_end / self.n_steps
        t = 0.0
        for _ in range(self.n_steps):
            dx = self.rhs(x, t, f)
            if isinstance(x, (int, float)):
                x = x + dx * dt
            else:
                x = [xi + dxi * dt for xi, dxi in zip(x, dx)]
            t += dt
        return x

    def _steady_state(self, f) -> Any:
        x = self.x0_fn(f)
        for _ in range(64):
            r = self.rhs(x, 0.0, f)
            if abs(r) < 1e-9:
                return x
            eps = 1e-5 * (1.0 + abs(x))
            r2 = self.rhs(x + eps, 0.0, f)
            denom = (r2 - r) / eps
            if abs(denom) < 1e-12:
                break
            x = x - r / denom
        return x

    def h(self, f) -> Antichain:
        if self.mode == "steady_state":
            x = self._steady_state(f)
        else:
            x = self._simulate(f)
        return Antichain.singleton(self.R, dict(self.extract(x)))


# ---------------------------------------------------------------------------
# UncertainDP
# ---------------------------------------------------------------------------


class UncertainDP(DesignProblem):
    """Bracket a nominal DP with cheaper (optimistic) and dearer (pessimistic)
    bounds, returning whichever was requested.

    The paper's Sec. VII approximates a non-finitely-representable antichain
    by a lower bound h^L <= h <= h^U. Solving with h^L gives an optimistic
    Pareto front, with h^U a pessimistic one. The true solution sits between.
    """

    def __init__(
        self,
        F: Poset,
        R: Poset,
        lower: DesignProblem,
        upper: DesignProblem,
        mode: str = "upper",
        name: str = "uncertain",
    ):
        if mode not in ("lower", "upper"):
            raise ValueError("mode must be 'lower' or 'upper'")
        self.F = F
        self.R = R
        self.lower = lower
        self.upper = upper
        self.mode = mode
        self.name = name

    def h(self, f) -> Antichain:
        return self.upper.h(f) if self.mode == "upper" else self.lower.h(f)

    def with_mode(self, mode: str) -> "UncertainDP":
        return UncertainDP(self.F, self.R, self.lower, self.upper, mode, self.name)
