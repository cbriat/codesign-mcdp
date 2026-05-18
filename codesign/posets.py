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
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
import math


class Poset(ABC):
    """Abstract partially ordered set."""

    name: str = "Poset"

    @abstractmethod
    def leq(self, a, b) -> bool:
        """Return True iff a <= b in this poset."""

    @abstractmethod
    def bottom(self):
        """The least element of the poset."""

    @abstractmethod
    def top(self):
        """The greatest element; conventionally signals infeasibility."""

    def is_top(self, x) -> bool:
        return self.leq(self.top(), x) and self.leq(x, self.top())

    def is_bottom(self, x) -> bool:
        return self.leq(x, self.bottom()) and self.leq(self.bottom(), x)

    def eq(self, a, b) -> bool:
        return self.leq(a, b) and self.leq(b, a)

    def lt(self, a, b) -> bool:
        return self.leq(a, b) and not self.eq(a, b)

    def comparable(self, a, b) -> bool:
        return self.leq(a, b) or self.leq(b, a)

    def join(self, a, b):
        """Least upper bound of {a, b}. Default impl: max under leq."""
        if self.leq(a, b):
            return b
        if self.leq(b, a):
            return a
        raise NotImplementedError(
            f"join not defined for incomparable elements in {self.name}"
        )

    def format(self, x) -> str:
        return repr(x)


# ---------------------------------------------------------------------------
# Chain posets: R+, N+, both extended with a top element
# ---------------------------------------------------------------------------


@dataclass
class Reals(Poset):
    """Non-negative reals with an added top (+inf), forming a CPO.

    The natural order <= matches the partial order. Bottom is 0, top is +inf.
    +inf is used to mark "infeasible" results during Kleene iteration.
    """

    unit: str = ""
    name: str = field(default="R+")

    def __post_init__(self):
        if self.unit and self.name == "R+":
            self.name = f"R+[{self.unit}]"

    def leq(self, a: float, b: float) -> bool:
        return a <= b

    def bottom(self) -> float:
        return 0.0

    def top(self) -> float:
        return math.inf

    def is_top(self, x) -> bool:
        return math.isinf(x) and x > 0

    def join(self, a: float, b: float) -> float:
        return max(a, b)

    def format(self, x) -> str:
        if math.isinf(x):
            return "⊤"
        return f"{x:.4g}{(' ' + self.unit) if self.unit else ''}"


@dataclass
class Naturals(Poset):
    """Non-negative integers with an added top, forming a CPO."""

    unit: str = ""
    name: str = field(default="N+")

    def leq(self, a, b) -> bool:
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


# ---------------------------------------------------------------------------
# Product poset: a tuple of named components, each its own poset
# ---------------------------------------------------------------------------


class NamedProduct(Poset):
    """Product of named posets. An element is a dict {name: value}.

    The partial order is component-wise: x <= y iff x[k] <= y[k] for all keys k.
    """

    def __init__(self, components: Mapping[str, Poset]):
        if not components:
            raise ValueError("NamedProduct requires at least one component")
        self.components: dict[str, Poset] = dict(components)
        self.name = "×".join(f"{k}:{p.name}" for k, p in self.components.items())

    def keys(self):
        return self.components.keys()

    def leq(self, a: Mapping, b: Mapping) -> bool:
        for k, p in self.components.items():
            if not p.leq(a[k], b[k]):
                return False
        return True

    def bottom(self) -> dict:
        return {k: p.bottom() for k, p in self.components.items()}

    def top(self) -> dict:
        return {k: p.top() for k, p in self.components.items()}

    def is_top(self, x) -> bool:
        return all(p.is_top(x[k]) for k, p in self.components.items())

    def any_top(self, x) -> bool:
        return any(p.is_top(x[k]) for k, p in self.components.items())

    def join(self, a, b) -> dict:
        return {k: p.join(a[k], b[k]) for k, p in self.components.items()}

    def format(self, x) -> str:
        parts = [f"{k}={p.format(x[k])}" for k, p in self.components.items()]
        return "(" + ", ".join(parts) + ")"

    def make(self, **kwargs):
        """Convenience constructor for elements with keyword arguments."""
        missing = set(self.components) - set(kwargs)
        if missing:
            raise ValueError(f"missing components: {missing}")
        extra = set(kwargs) - set(self.components)
        if extra:
            raise ValueError(f"unknown components: {extra}")
        return dict(kwargs)


# ---------------------------------------------------------------------------
# Discrete poset
# ---------------------------------------------------------------------------


class Discrete(Poset):
    """Discrete poset over an explicit collection of elements."""

    def __init__(self, elements: Iterable[Any], leq_fn=None, name: str = "D"):
        self.elements = list(elements)
        self._leq = leq_fn or (lambda a, b: a == b)
        self.name = name

    def leq(self, a, b) -> bool:
        return self._leq(a, b)

    def bottom(self):
        raise ValueError("Discrete poset has no canonical bottom; provide one")

    def top(self):
        raise ValueError("Discrete poset has no canonical top; provide one")
