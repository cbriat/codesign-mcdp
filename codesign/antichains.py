"""
Antichains: sets of mutually incomparable elements of a poset.

In the MCDP framework, the map h_dp associated with a design problem maps each
functionality f to an antichain of minimal resources in R. Antichains
generalize the notion of a Pareto front to arbitrary partial orders.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List

from .posets import Poset


class Antichain:
    """A finite set of mutually incomparable poset elements.

    Construction normalises: incomparable points are kept, dominated ones are
    discarded. This makes the constructor idempotent under Min.
    """

    __slots__ = ("poset", "_points")

    def __init__(self, poset: Poset, points: Iterable = ()):
        self.poset = poset
        self._points: List = []
        for p in points:
            self._insert(p)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def of_bottom(cls, poset: Poset) -> "Antichain":
        """The singleton antichain {⊥}. Seeds the Kleene iteration."""
        return cls(poset, [poset.bottom()])

    @classmethod
    def empty(cls, poset: Poset) -> "Antichain":
        """The empty antichain (no feasible solutions found yet)."""
        return cls(poset, [])

    @classmethod
    def singleton(cls, poset: Poset, point) -> "Antichain":
        return cls(poset, [point])

    @classmethod
    def from_set(cls, poset: Poset, points: Iterable) -> "Antichain":
        """Take Min of an arbitrary subset, yielding an antichain."""
        return cls(poset, points)

    def _insert(self, x) -> None:
        # Add x only if not dominated; remove dominated incumbents.
        new_points = []
        for p in self._points:
            if self.poset.leq(p, x) and not self.poset.eq(p, x):
                return  # p strictly dominates x
            if self.poset.eq(p, x):
                return  # duplicate
            if self.poset.leq(x, p) and not self.poset.eq(x, p):
                continue  # x strictly dominates p; drop it
            new_points.append(p)
        new_points.append(x)
        self._points = new_points

    # ------------------------------------------------------------------ #
    # Container interface
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator:
        return iter(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __bool__(self) -> bool:
        return bool(self._points)

    @property
    def points(self) -> List:
        return list(self._points)

    # ------------------------------------------------------------------ #
    # Ordering on antichains
    # ------------------------------------------------------------------ #

    def leq(self, other: "Antichain") -> bool:
        """Antichain order: self <= other iff up(self) supseteq up(other).

        Concretely: for every point in `other`, there's a point in `self` that
        dominates it (i.e., is <= to it in the underlying poset).
        """
        for b in other._points:
            if not any(self.poset.leq(a, b) for a in self._points):
                return False
        return True

    def eq(self, other: "Antichain") -> bool:
        # Equality of antichains compares point sets up to permutation under eq.
        if len(self._points) != len(other._points):
            return False
        used = [False] * len(other._points)
        for a in self._points:
            matched = False
            for j, b in enumerate(other._points):
                if not used[j] and self.poset.eq(a, b):
                    used[j] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Min and upper-closure-flavoured operations
    # ------------------------------------------------------------------ #

    @classmethod
    def union_min(cls, poset: Poset, antichains: Iterable["Antichain"]) -> "Antichain":
        """Min of the union of several antichains.

        This implements Min(union_i A_i), the operation used by the composition
        rules and the Kleene step for loops.
        """
        result = cls.empty(poset)
        for a in antichains:
            for p in a._points:
                result._insert(p)
        return result

    def filter_above(self, lower) -> "Antichain":
        """Restrict to points x with x >= lower (in the underlying poset).

        Implements the "intersect with up(lower)" operation appearing in
        Phi_{f_1}(A) = U_{r in A} h_{f_1}(r) ∩ up(r).
        """
        kept = [x for x in self._points if self.poset.leq(lower, x)]
        return Antichain(self.poset, kept)

    def has_any_top(self) -> bool:
        """True if any point in the antichain hits the top (infeasibility)."""
        any_top = getattr(self.poset, "any_top", None)
        if any_top is not None:
            return any(any_top(p) for p in self._points)
        return any(self.poset.is_top(p) for p in self._points)

    def is_empty(self) -> bool:
        return len(self._points) == 0

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        if not self._points:
            return "Antichain[∅]"
        formatted = [self.poset.format(p) for p in self._points]
        return "Antichain[" + ", ".join(formatted) + "]"
