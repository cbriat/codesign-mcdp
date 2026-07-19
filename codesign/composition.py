"""
Composition: series, parallel, loop.

The three operators that close the class of co-design problems under
composition, as in Censi's paper:

    series(dp1, dp2)  : F1 -> R1 = F2 -> R2, with the constraint r1 <= f2.
    par(dp1, dp2)     : F1 x F2 -> R1 x R2, side-by-side.
    loop(dp, axis)    : feed back one resource as one functionality.

The constructions preserve monotonicity (Censi, Thm. 3): chaining monotone
DPs yields a monotone DP, so the resulting composite supports the same solve
routine. ``loop`` triggers a Kleene fixed-point iteration via the solver.
"""
from __future__ import annotations

from typing import Mapping

from .antichains import Antichain
from .dp import DesignProblem
from .posets import Ports, Poset


# ---------------------------------------------------------------------------
# Series composition
# ---------------------------------------------------------------------------


class Series(DesignProblem):
    """Series composition: dp1 then dp2, with r1 <= f2 forced by construction.

    The relation is:
        h(f) = Min { h2(f2) : exists r1 in h1(f), r1 <= f2 }
    which simplifies (when h2 is monotone) to h(f) = Min(up(h2(h1(f)))).

    Each resource point produced by ``dp1`` is fed straight in as a
    functionality of ``dp2``, so every port ``dp2`` requires must be produced
    by ``dp1``: ``set(dp2.F.keys()) <= set(dp1.R.keys())`` when both interfaces
    are :class:`~codesign.posets.Ports`. This is checked upfront in
    :meth:`__init__` -- a mismatch raises a :class:`ValueError` naming the
    missing ports, rather than surfacing as a bare ``KeyError`` deep inside a
    later solve. Extra resource ports on ``dp1`` (not consumed by ``dp2``) are
    permitted: :meth:`h` simply ignores them, so the check is a subset, not an
    equality.
    """

    def __init__(self, dp1: DesignProblem, dp2: DesignProblem, name: str | None = None):
        if isinstance(dp1.R, Ports) and isinstance(dp2.F, Ports):
            r1_keys = set(dp1.R.keys())
            f2_keys = set(dp2.F.keys())
            missing = f2_keys - r1_keys
            if missing:
                raise ValueError(
                    f"Series({dp1.name!r}, {dp2.name!r}): interface mismatch. "
                    f"The connection feeds {dp1.name!r}'s resource ports in as "
                    f"{dp2.name!r}'s functionality, so every port {dp2.name!r} "
                    f"requires must be produced by {dp1.name!r}. "
                    f"{dp2.name!r} consumes functionality ports "
                    f"{sorted(f2_keys)} but {dp1.name!r} produces resource ports "
                    f"{sorted(r1_keys)}; missing on the resource side: "
                    f"{sorted(missing)}. Either add {sorted(missing)} to "
                    f"{dp1.name!r}'s R, or rename {dp2.name!r}'s F ports to "
                    f"match {dp1.name!r}'s R ports."
                )
        self.dp1 = dp1
        self.dp2 = dp2
        self.F = dp1.F
        self.R = dp2.R
        self.name = name or f"series({dp1.name},{dp2.name})"

    def h(self, f) -> Antichain:
        a1 = self.dp1.h(f)
        if a1.has_any_top() or a1.is_empty():
            return Antichain.singleton(self.R, self.R.top())
        return Antichain.union_min(self.R, [self.dp2.h(r1) for r1 in a1])


# ---------------------------------------------------------------------------
# Parallel composition
# ---------------------------------------------------------------------------


class Parallel(DesignProblem):
    """Parallel composition: independent dp1 and dp2 stacked side by side.

    F = F1 x F2 (Ports concatenation), R = R1 x R2. The combined
    antichain is the Cartesian product of individual antichains.
    """

    def __init__(self, dp1: DesignProblem, dp2: DesignProblem, name: str | None = None):
        if not isinstance(dp1.F, Ports) or not isinstance(dp2.F, Ports):
            raise TypeError("Parallel composition needs Ports F on both sides")
        if not isinstance(dp1.R, Ports) or not isinstance(dp2.R, Ports):
            raise TypeError("Parallel composition needs Ports R on both sides")
        f_overlap = set(dp1.F.keys()) & set(dp2.F.keys())
        r_overlap = set(dp1.R.keys()) & set(dp2.R.keys())
        if f_overlap:
            raise ValueError(f"functionality names clash: {f_overlap}")
        if r_overlap:
            raise ValueError(f"resource names clash: {r_overlap}")
        self.dp1 = dp1
        self.dp2 = dp2
        self.F = Ports({**dp1.F.components, **dp2.F.components})
        self.R = Ports({**dp1.R.components, **dp2.R.components})
        self.name = name or f"par({dp1.name},{dp2.name})"
        self._f1_keys = set(dp1.F.keys())
        self._f2_keys = set(dp2.F.keys())

    def _split(self, f: Mapping) -> tuple[dict, dict]:
        f1 = {k: f[k] for k in self._f1_keys}
        f2 = {k: f[k] for k in self._f2_keys}
        return f1, f2

    def h(self, f) -> Antichain:
        f1, f2 = self._split(f)
        a1 = self.dp1.h(f1)
        a2 = self.dp2.h(f2)
        products = []
        for r1 in a1:
            for r2 in a2:
                products.append({**r1, **r2})
        return Antichain.from_set(self.R, products)


# ---------------------------------------------------------------------------
# Loop composition
# ---------------------------------------------------------------------------


class Loop(DesignProblem):
    """Feedback composition: feed one resource back as one functionality.

    Given an inner DP with F = F_outer x {axis: P} and R = R_outer x {axis: P},
    the loop closes the named ``axis``: the resource produced is required to
    be >= the functionality consumed (Censi, Def. 16).

    solve(loop_dp, f_outer) returns the antichain ``lfp(Phi)``.
    """

    def __init__(
        self,
        inner: DesignProblem,
        axis: str,
        name: str | None = None,
    ):
        if not isinstance(inner.F, Ports):
            raise TypeError("Loop needs a Ports functionality space")
        if not isinstance(inner.R, Ports):
            raise TypeError("Loop needs a Ports resource space")
        if axis not in inner.F.components or axis not in inner.R.components:
            raise ValueError(
                f"axis '{axis}' must appear in both F and R of the inner DP"
            )
        self.inner = inner
        self.axis = axis
        outer_F_components = {
            k: p for k, p in inner.F.components.items() if k != axis
        }
        outer_R_components = {
            k: p for k, p in inner.R.components.items() if k != axis
        }
        if outer_F_components:
            self.F = Ports(outer_F_components)
        else:
            self.F = _Unit()
        self.R = Ports(outer_R_components) if outer_R_components else _Unit()
        self.name = name or f"loop({inner.name}, axis={axis})"

    def h(self, f_outer) -> Antichain:
        # Defer to the solver to avoid an import cycle at module load.
        from .solver import kleene_loop

        return kleene_loop(self, f_outer)


# ---------------------------------------------------------------------------
# Unit poset for empty-outer loops
# ---------------------------------------------------------------------------


class _Unit(Poset):
    """Trivial single-element poset, used as F/R when the outer interface is empty."""

    name = "Unit"

    def leq(self, a, b) -> bool:
        return True

    def bottom(self):
        return None

    def top(self):
        return None

    def is_top(self, x) -> bool:
        return False

    def is_bottom(self, x) -> bool:
        return True

    def format(self, x) -> str:
        return "*"


# ---------------------------------------------------------------------------
# Lowercase aliases matching the paper's notation
# ---------------------------------------------------------------------------


def series(dp1: DesignProblem, dp2: DesignProblem, name: str | None = None) -> Series:
    return Series(dp1, dp2, name=name)


def par(dp1: DesignProblem, dp2: DesignProblem, name: str | None = None) -> Parallel:
    return Parallel(dp1, dp2, name=name)


def loop(inner: DesignProblem, axis: str, name: str | None = None) -> Loop:
    return Loop(inner, axis, name=name)
