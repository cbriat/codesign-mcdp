"""
Posets: partially ordered sets used as functionality and resource spaces.

Following Censi's theory, every functionality space F and resource space R is
a partially ordered set (poset), often a Complete Partial Order (CPO) which
adds a bottom element and suprema of directed sets. Resources are typically
chains (totally ordered, like R+ or N+ extended with a top element), and
multi-dimensional resources are products of chains.

The core operations every Poset must support:
    leq(a, b)        : is a <= b ?
    bottom()         : the least element (used to seed Kleene iteration)
    top()            : the greatest element (signals infeasibility)
    is_top(x), is_bottom(x)
    join(a, b)       : least upper bound (supremum), used to combine constraints

The four poset implementations shipped here cover essentially every MCDP
modelling need:

- :class:`Reals` and :class:`Naturals` are the scalar chains. Use them for
  any individual port that is a real or integer-valued quantity (mass, cost,
  count, energy, ...).
- :class:`Ports` is the named product. It bundles several scalar (or
  recursively named) posets into a typed dict-like schema. Every design
  problem's F and R is a :class:`Ports`.
- :class:`Discrete` is a finite poset over an explicit element list with
  user-supplied ordering, useful for catalog choice or enumerated states.

A backward-compatible alias ``NamedProduct = Ports`` is exported from the
package; existing code using ``NamedProduct`` continues to work unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
import math


# ===========================================================================
# Abstract base
# ===========================================================================


class Poset(ABC):
    """Abstract partially ordered set.

    Subclasses must override :meth:`leq`, :meth:`bottom`, and :meth:`top`.
    Defaults are provided for :meth:`is_top`, :meth:`is_bottom`, :meth:`eq`,
    :meth:`lt`, :meth:`comparable`, :meth:`join`, and :meth:`format` and
    can be overridden for efficiency or special behaviour (e.g. handling
    ``+inf`` cleanly).
    """

    name: str = "Poset"

    @abstractmethod
    def leq(self, a, b) -> bool:
        """Return True iff ``a <= b`` in this poset."""

    @abstractmethod
    def bottom(self):
        """The least element of the poset (used to seed Kleene iteration)."""

    @abstractmethod
    def top(self):
        """The greatest element; conventionally signals infeasibility."""

    def is_top(self, x) -> bool:
        # Default: x is top iff top <= x <= top. Subclasses override for speed.
        return self.leq(self.top(), x) and self.leq(x, self.top())

    def is_bottom(self, x) -> bool:
        return self.leq(x, self.bottom()) and self.leq(self.bottom(), x)

    def eq(self, a, b) -> bool:
        """Anti-symmetric equality derived from leq."""
        return self.leq(a, b) and self.leq(b, a)

    def lt(self, a, b) -> bool:
        """Strict less-than: a <= b but not a == b."""
        return self.leq(a, b) and not self.eq(a, b)

    def comparable(self, a, b) -> bool:
        """True iff one of a, b dominates the other."""
        return self.leq(a, b) or self.leq(b, a)

    def join(self, a, b):
        """Least upper bound of {a, b}.

        Default falls back to max-under-leq for chains. For incomparable
        elements in a non-lattice poset, override this in the subclass.
        """
        if self.leq(a, b):
            return b
        if self.leq(b, a):
            return a
        raise NotImplementedError(
            f"join not defined for incomparable elements in {self.name}"
        )

    def format(self, x) -> str:
        """Human-readable string for an element. Used in Antichain printing."""
        return repr(x)


# ===========================================================================
# Chain posets: R+, N+, both extended with a top element
# ===========================================================================


@dataclass
class Reals(Poset):
    """Non-negative reals with an added top (+inf), forming a CPO.

    The natural order ``<=`` matches the partial order. Bottom is 0, top is
    +inf. ``+inf`` is used to mark "infeasible" results during Kleene
    iteration: an iterate that crosses ``+inf`` in any component represents
    a design that no longer satisfies its constraints.

    The ``unit`` field is metadata: it travels through composition so that
    ``format()`` can print values like ``0.56 kg`` rather than just ``0.56``.
    The unit is not used in any algebraic check; it is purely for display
    and documentation.
    """

    unit: str = ""
    name: str = field(default="R+")

    def __post_init__(self):
        # If the user gave a unit but accepted the default name, decorate the
        # name with the unit so debug prints stay informative.
        if self.unit and self.name == "R+":
            self.name = f"R+[{self.unit}]"

    def leq(self, a: float, b: float) -> bool:
        return a <= b

    def bottom(self) -> float:
        return 0.0

    def top(self) -> float:
        return math.inf

    def is_top(self, x) -> bool:
        # Cheaper than the default leq-based check.
        return math.isinf(x) and x > 0

    def join(self, a: float, b: float) -> float:
        # Two reals always have a max; no need for the abstract base's check.
        return max(a, b)

    def format(self, x) -> str:
        if math.isinf(x):
            return "⊤"
        return f"{x:.4g}{(' ' + self.unit) if self.unit else ''}"


@dataclass
class Naturals(Poset):
    """Non-negative integers with an added top, forming a CPO.

    Mirrors :class:`Reals` for integer-valued quantities like part counts,
    bit budgets, and so on. ``+inf`` is the top element here too, even
    though it isn't an integer; the iteration treats it as the infeasibility
    marker the same way.
    """

    unit: str = ""
    name: str = field(default="N+")

    def __post_init__(self):
        # Mirror Reals: if a unit was given but the default name kept, fold
        # the unit into the name so debug prints stay informative.
        if self.unit and self.name == "N+":
            self.name = f"N+[{self.unit}]"

    def leq(self, a, b) -> bool:
        # Handle the top (+inf) carefully: top is greater than every finite
        # natural, and only equal to itself.
        if a == math.inf:
            return b == math.inf
        if b == math.inf:
            return True
        return int(a) <= int(b)

    def bottom(self) -> int:
        return 0

    def top(self):
        return math.inf

    def is_top(self, x) -> bool:
        return x == math.inf

    def join(self, a, b):
        if a == math.inf or b == math.inf:
            return math.inf
        return max(int(a), int(b))

    def format(self, x) -> str:
        # Mirror Reals.format, but honour the integer nature of naturals so
        # a value like 3.0 prints "3" rather than "3.0", and append the unit
        # (previously ``unit=`` was accepted but silently ignored here).
        if math.isinf(x):
            return "⊤"
        return f"{int(x)}{(' ' + self.unit) if self.unit else ''}"


# ===========================================================================
# Product poset: named bundle of typed components
# ===========================================================================


class Ports(Poset):
    """A typed bundle of named ports, used as the F or R of a design problem.

    Mathematically this is a product poset: an element is a tuple where each
    coordinate lives in its own component poset. We address coordinates by
    string name (a dict) rather than by position (a tuple), so a design's
    inputs read naturally as ``{"capacity": 3.6e6, "endurance": 300.0}``
    instead of ``(3.6e6, 300.0)``. Names also make composition checks
    structural (Series requires ``dp1.R`` and ``dp2.F`` to have the same
    keys), make the constraint DSL possible (operator overloading looks up
    ports by name), and let units travel through the type system.

    The partial order is component-wise:

        x <= y  iff  x[k] <= y[k]  for every component name k.

    Bottom and top are obtained by taking the bottom and top of every
    component separately. ``is_top(x)`` is True iff *every* component is
    top; ``any_top(x)`` is True iff *at least one* component is top. The
    distinction matters for infeasibility tests in the Kleene iteration.

    Example:

        F = Ports({"capacity":    Reals(unit="J"),
                   "endurance":   Reals(unit="s")})
        R = Ports({"battery_mass": Reals(unit="kg")})

    The legacy name :class:`NamedProduct` is preserved as an alias in the
    top-level package, so existing imports keep working.
    """

    def __init__(self, components: Mapping[str, Poset]):
        if not components:
            raise ValueError(
                "Ports requires at least one component, got an empty mapping. "
                "Pass a non-empty {name: poset} dict, e.g. "
                "Ports({'mass': Reals(unit='kg')})."
            )
        # Copy into a fresh dict so later caller-side mutation of the
        # argument cannot corrupt this Ports instance.
        self.components: dict[str, Poset] = dict(components)
        # Display name like "capacity:R+[J] × endurance:R+[s]".
        self.name = "×".join(f"{k}:{p.name}" for k, p in self.components.items())

    def keys(self):
        """Iterate over the port names. Convenient for ``set(F.keys())``."""
        return self.components.keys()

    # ----- Order, bottom, top -----

    def leq(self, a: Mapping, b: Mapping) -> bool:
        """Component-wise leq: True iff every coordinate satisfies leq."""
        for k, p in self.components.items():
            if not p.leq(a[k], b[k]):
                return False
        return True

    def bottom(self) -> dict:
        """The all-bottoms element. Seeds the Kleene iteration."""
        return {k: p.bottom() for k, p in self.components.items()}

    def top(self) -> dict:
        """The all-tops element. Signals total infeasibility."""
        return {k: p.top() for k, p in self.components.items()}

    def is_top(self, x) -> bool:
        """Strict infeasibility: every coordinate must hit top."""
        return all(p.is_top(x[k]) for k, p in self.components.items())

    def any_top(self, x) -> bool:
        """Weak infeasibility marker: any single coordinate at top is enough.

        Used by the antichain machinery to decide that a candidate design
        is unsalvageable, since a numerical blow-up in one resource is
        sufficient to disqualify the whole point.
        """
        return any(p.is_top(x[k]) for k, p in self.components.items())

    def join(self, a, b) -> dict:
        """Component-wise least upper bound."""
        return {k: p.join(a[k], b[k]) for k, p in self.components.items()}

    # ----- Display and construction helpers -----

    def format(self, x) -> str:
        """Pretty-print like ``(mass=0.56 kg, cost=$120)``."""
        parts = [f"{k}={p.format(x[k])}" for k, p in self.components.items()]
        return "(" + ", ".join(parts) + ")"

    def make(self, **kwargs):
        """Build an element with keyword arguments, validating completeness.

        Raises ValueError if any component is missing or any extra key is
        supplied. Use this when you want a clean error message rather than
        a KeyError deep inside the solver.
        """
        missing = set(self.components) - set(kwargs)
        if missing:
            raise ValueError(
                f"Ports.make is missing component(s) {sorted(missing)}. This "
                f"Ports declares {sorted(self.components)}; supply a value for "
                f"every one. Add {sorted(missing)} to the keyword arguments."
            )
        extra = set(kwargs) - set(self.components)
        if extra:
            raise ValueError(
                f"Ports.make got unknown component(s) {sorted(extra)}. This "
                f"Ports declares {sorted(self.components)}; only those keys are "
                f"accepted. Remove {sorted(extra)} or correct the name(s)."
            )
        return dict(kwargs)


# Backward-compatible alias. Existing code that imports NamedProduct still
# works; new code should prefer Ports. The library's own internal modules
# have all been migrated to Ports.
NamedProduct = Ports


# ===========================================================================
# Discrete poset
# ===========================================================================


class Discrete(Poset):
    """Discrete poset over an explicit collection of elements.

    Useful when the design space is enumerated (a list of part numbers, a
    set of operating modes) and the order is defined by an explicit
    predicate rather than a numeric inequality. The default order is
    equality: nothing is less than anything else but itself.

    No canonical bottom or top exists for an arbitrary discrete poset, so
    callers must supply them by overriding or by wrapping in a structure
    that adds them explicitly.
    """

    def __init__(self, elements: Iterable[Any], leq_fn=None, name: str = "D"):
        self.elements = list(elements)
        # Default: discrete order. Subclasses can override with a richer
        # predicate (e.g. lexicographic, refinement order, ...).
        self._leq = leq_fn or (lambda a, b: a == b)
        self.name = name

    def leq(self, a, b) -> bool:
        return self._leq(a, b)

    def bottom(self):
        raise ValueError(
            f"Discrete poset {self.name!r} has no canonical bottom element. A "
            f"discrete order over {len(self.elements)} element(s) has no least "
            f"element in general; subclass Discrete and override bottom(), or "
            f"wrap it in a poset that adds an explicit bottom."
        )

    def top(self):
        raise ValueError(
            f"Discrete poset {self.name!r} has no canonical top element. A "
            f"discrete order over {len(self.elements)} element(s) has no "
            f"greatest element in general; subclass Discrete and override "
            f"top(), or wrap it in a poset that adds an explicit top."
        )
