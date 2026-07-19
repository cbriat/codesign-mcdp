"""General carried state for temporal co-design: scalars through vectors.

The scalar dynamic program in :mod:`codesign.dynamic` and the
antichain-valued one in :mod:`codesign.sequential` each carry a single
number between stages. Realistic problems carry more: the Formula 1
seasonal co-design of Neumann, Zardini, and colleagues carries a state
vector of two battery wear levels plus a discrete regulatory-penalty
flag; a reconfigurable robot carries per-module wear and a remaining
energy budget; a self-assembly protocol carries the concentrations of
several intermediate species. This module provides the general carried
state those problems need: a **vector state grid** over several named
axes, each with its own discretisation, together with the product order
that the monotonicity results are stated over.

The design deliberately subsumes the scalar case. A one-axis
:class:`VectorStateGrid` behaves exactly like the scalar
:class:`~codesign.dynamic.StateGrid`, so the vector layer is a strict
generalisation rather than a parallel track.

Two kinds of axis are supported, because realistic state has both:

* a **continuous axis** (:class:`ContinuousAxis`), a bucketed real
  interval for quantities like wear, charge, or concentration, snapped to
  the nearest node with the same direction-aware and out-of-bounds
  handling as the scalar grid; and
* a **discrete axis** (:class:`DiscreteAxis`), a finite set of labelled
  values for quantities like a regulatory flag, an active-module set, or
  an operating regime, with an explicit order supplied by the caller so
  the monotonicity machinery has a partial order to check against.

The product order over the axes is component-wise: one state vector is
below another iff it is below on every axis. That is the order the
carried-state hypotheses (H1)/(H2) of the sequential theory are phrased
in, so the vector grid plugs straight into the monotonicity guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# A state vector is an immutable mapping from axis name to that axis's
# value. We represent it as a tuple of (name, value) pairs sorted by name,
# which is hashable (usable as a dict key in the DP table) and comparable.
StateVec = Tuple[Tuple[str, Any], ...]


def make_state(**axis_values: Any) -> StateVec:
    """Build a canonical state vector from keyword axis values.

    ``make_state(fuel=12.0, flag=0)`` returns the hashable canonical form
    used as a key in the dynamic-program table. Axis order is normalised
    so two states with the same contents compare and hash equal.
    """
    return tuple(sorted(axis_values.items()))


def state_get(state: StateVec, axis: str) -> Any:
    """Read one axis value from a state vector."""
    for k, v in state:
        if k == axis:
            return v
    raise KeyError(
        f"state vector has no axis {axis!r}; it carries axes "
        f"{[k for k, _ in state]}"
    )


def state_as_dict(state: StateVec) -> Dict[str, Any]:
    """Return a plain dict view of a state vector."""
    return {k: v for k, v in state}


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
class Axis:
    """Base class for a single state axis."""

    name: str

    def nodes(self) -> Sequence[Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def snap(self, value: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def in_bounds(self, value: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def leq(self, a: Any, b: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ContinuousAxis(Axis):
    """A bucketed real interval, the vector analogue of the scalar grid.

    Parameters
    ----------
    name : str
        Axis name (for example ``"charge"`` or ``"wear1"``).
    lo, hi : float
        Interval bounds.
    n : int
        Number of evenly spaced nodes on ``[lo, hi]``.
    increasing_is_larger : bool
        Orientation of the axis for the product order. ``True`` (default)
        means a numerically larger value is higher in the order (more
        charge, more budget). Set ``False`` when the natural "more
        committed" direction is decreasing (for example remaining slack).
        The monotonicity guard uses this to interpret the state order.
    """

    name: str
    lo: float
    hi: float
    n: int
    increasing_is_larger: bool = True
    _nodes: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(
                f"ContinuousAxis {self.name!r} needs at least one node, got "
                f"n={self.n}. Pass n >= 1 (n=1 yields a single node at "
                f"lo={self.lo})."
            )
        if self.n == 1:
            self._nodes = [float(self.lo)]
        else:
            step = (self.hi - self.lo) / (self.n - 1)
            self._nodes = [self.lo + i * step for i in range(self.n)]

    def nodes(self) -> Sequence[float]:
        return self._nodes

    def snap(self, value: float) -> float:
        best = self._nodes[0]
        best_d = abs(value - best)
        for node in self._nodes[1:]:
            d = abs(value - node)
            if d < best_d:
                best_d = d
                best = node
        return best

    def in_bounds(self, value: float) -> bool:
        span = self.hi - self.lo
        tol = 1e-9 * (span if span > 0 else 1.0)
        return (self.lo - tol) <= value <= (self.hi + tol)

    def leq(self, a: float, b: float) -> bool:
        return a <= b if self.increasing_is_larger else b <= a


@dataclass
class DiscreteAxis(Axis):
    """A finite labelled axis with a caller-supplied order.

    Parameters
    ----------
    name : str
        Axis name (for example ``"penalty_flag"`` or ``"regime"``).
    values : sequence
        The admissible labels (any hashable values).
    order : sequence, optional
        A chain giving the partial order, least first: ``order[i] <=
        order[j]`` for ``i <= j``. Labels omitted from ``order`` (or when
        ``order`` is ``None``) are treated as mutually incomparable, which
        the monotonicity guard reads as "no order to exploit on this
        axis". Supplying an order lets the (H1)/(H2) checks reason about
        the axis (for example a monotone regulatory flag 0 <= 1).
    """

    name: str
    values: Sequence[Hashable]
    order: Optional[Sequence[Hashable]] = None
    _rank: Dict[Hashable, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(
                f"DiscreteAxis {self.name!r} needs at least one value, got an "
                f"empty sequence. Pass the admissible labels, e.g. "
                f"DiscreteAxis({self.name!r}, values=[0, 1])."
            )
        if self.order is not None:
            self._rank = {v: i for i, v in enumerate(self.order)}

    def nodes(self) -> Sequence[Hashable]:
        return list(self.values)

    def snap(self, value: Hashable) -> Hashable:
        # Discrete labels are used verbatim; snapping is identity when the
        # value is admissible.
        return value

    def in_bounds(self, value: Hashable) -> bool:
        return value in self.values

    def leq(self, a: Hashable, b: Hashable) -> bool:
        if a == b:
            return True
        if self.order is None:
            return False  # incomparable distinct labels
        ra, rb = self._rank.get(a), self._rank.get(b)
        if ra is None or rb is None:
            return False
        return ra <= rb


# ---------------------------------------------------------------------------
# Vector state grid
# ---------------------------------------------------------------------------
class VectorStateGrid:
    """A product grid over several named axes, carrying a full state vector.

    The grid enumerates the Cartesian product of its axes' nodes as the DP
    state set, snaps a proposed successor vector axis-by-axis, checks
    bounds axis-by-axis, and exposes the component-wise product order used
    by the monotonicity results. A single-axis grid reproduces the scalar
    :class:`~codesign.dynamic.StateGrid`.
    """

    def __init__(self, axes: Sequence[Axis]):
        if not axes:
            raise ValueError(
                "VectorStateGrid needs at least one axis, got an empty "
                "sequence. Pass one or more Axis objects, e.g. "
                "VectorStateGrid([ContinuousAxis('charge', 0.0, 1.0, 5)])."
            )
        names = [ax.name for ax in axes]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"VectorStateGrid axis names must be unique, but {dupes} "
                f"appear more than once in {names}. Give each axis a distinct "
                f"name."
            )
        self.axes: List[Axis] = list(axes)
        self.axis_by_name: Dict[str, Axis] = {ax.name: ax for ax in axes}

    @classmethod
    def scalar(cls, name: str, lo: float, hi: float, n: int,
               increasing_is_larger: bool = True) -> "VectorStateGrid":
        """Convenience: a one-axis continuous grid (the scalar case)."""
        return cls([ContinuousAxis(name, lo, hi, n, increasing_is_larger)])

    def nodes(self) -> Iterator[StateVec]:
        """Iterate over every state vector in the product grid."""
        axis_nodes = [[(ax.name, v) for v in ax.nodes()] for ax in self.axes]
        for combo in product(*axis_nodes):
            yield tuple(sorted(combo))

    def snap(self, vector: Mapping[str, Any]) -> StateVec:
        """Snap a proposed successor vector to the nearest grid node."""
        snapped = []
        for ax in self.axes:
            snapped.append((ax.name, ax.snap(vector[ax.name])))
        return tuple(sorted(snapped))

    def in_bounds(self, vector: Mapping[str, Any]) -> bool:
        """True iff every axis value is within that axis's envelope."""
        return all(ax.in_bounds(vector[ax.name]) for ax in self.axes)

    def leq(self, a: StateVec, b: StateVec) -> bool:
        """Component-wise product order: a <= b iff a[k] <= b[k] for all k."""
        da, db = state_as_dict(a), state_as_dict(b)
        return all(ax.leq(da[ax.name], db[ax.name]) for ax in self.axes)

    def bottom(self) -> StateVec:
        """The least state vector (each axis at its order-minimum node)."""
        comps = []
        for ax in self.axes:
            ns = list(ax.nodes())
            least = ns[0]
            for v in ns[1:]:
                if ax.leq(v, least) and not ax.leq(least, v):
                    least = v
            comps.append((ax.name, least))
        return tuple(sorted(comps))

    def __len__(self) -> int:
        total = 1
        for ax in self.axes:
            total *= len(list(ax.nodes()))
        return total


__all__ = [
    "StateVec",
    "make_state",
    "state_get",
    "state_as_dict",
    "Axis",
    "ContinuousAxis",
    "DiscreteAxis",
    "VectorStateGrid",
]
