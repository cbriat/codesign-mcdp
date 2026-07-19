"""
A thin MCDPL-style declarative front-end.

The paper's Fig. 48 shows a concrete syntax along the lines of:

    mcdp {
        provides endurance [s]
        provides extra_payload [kg]
        requires mass [kg]

        sub battery = mcdp {
            provides capacity [J]
            requires mass [kg]
            mass >= capacity * 0.00000055
        }
        ...
    }

This module gives the same shape in pure Python: open a builder, declare
``provides`` (functionalities) and ``requires`` (resources), add constraint
equations as plain lambdas, optionally close one or more loops, and emit a
``DesignProblem`` you can hand to ``solve``. It is a notation layer over the
operators already exposed by ``codesign.dp`` and ``codesign.composition``;
nothing it does is magic.

Example
-------

    from codesign.mcdpl import MCDP

    with MCDP("battery") as m:
        m.provides("capacity", unit="J")
        m.requires("mass", unit="kg")
        m.constraint("mass", lambda f: f["capacity"] / 1.8e6)
    battery = m.build()

    print(solve(battery, {"capacity": 3.6e6}))
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from .antichains import Antichain
from .composition import Loop
from .dp import AlgebraicDP, DesignProblem, FunctionDP
from .posets import Ports, Reals


class MCDP:
    """Builder for a monotone design problem in MCDPL-like notation.

    Use as a context manager (or just call methods directly), declare each
    functionality with ``provides`` and each resource with ``requires``, add
    constraint equations with ``constraint`` (closed form) or ``rule`` (a
    multi-valued antichain-returning function), and call ``build`` to get a
    plain ``DesignProblem``.
    """

    def __init__(self, name: str = "mcdp"):
        self.name = name
        self._provides: Dict[str, Any] = {}     # name -> poset
        self._requires: Dict[str, Any] = {}     # name -> poset
        self._constraints: Dict[str, Callable] = {}  # resource_name -> closed-form
        self._rule: Optional[Callable] = None   # antichain-valued override
        self._loops: List[str] = []             # axis names to close

    # -- context manager sugar -------------------------------------------

    def __enter__(self) -> "MCDP":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # No side effects on context exit; build() must be called explicitly.
        # Suppresses no exceptions.
        return None

    # -- declarations ----------------------------------------------------

    def provides(self, name: str, *, unit: str = "", poset=None) -> "MCDP":
        """Declare a functionality. Defaults to ``Reals(unit=unit)``."""
        self._provides[name] = poset if poset is not None else Reals(unit=unit)
        return self

    def requires(self, name: str, *, unit: str = "", poset=None) -> "MCDP":
        """Declare a resource. Defaults to ``Reals(unit=unit)``."""
        self._requires[name] = poset if poset is not None else Reals(unit=unit)
        return self

    def constraint(self, resource: str, fn: Callable[[Mapping], Any]) -> "MCDP":
        """Closed-form equation: ``resource >= fn(functionality_dict)``.

        Multiple calls for the same resource take the max (join), matching
        the MCDPL semantics where multiple ``>=`` clauses are all enforced.
        """
        if resource in self._constraints:
            prev = self._constraints[resource]
            self._constraints[resource] = lambda f, a=prev, b=fn: max(a(f), b(f))
        else:
            self._constraints[resource] = fn
        return self

    def rule(self, fn: Callable[[Mapping], Antichain]) -> "MCDP":
        """Hand-write the full h: functionality -> Antichain.

        Overrides any closed-form ``constraint`` calls. Use when the
        relation is multi-valued or branchy.
        """
        self._rule = fn
        return self

    def loop_on(self, axis: str) -> "MCDP":
        """Close a feedback loop on a name that appears in both provides
        and requires. The axis is projected out of the final F/R.

        The axis must be declared on *both* sides: with :meth:`provides` (as
        the fed-back functionality) and with :meth:`requires` (as the produced
        resource). If either side is missing, a :class:`ValueError` naming the
        missing side and the current declarations is raised. Because the axis
        is projected out of the closed loop's F/R, mirror it into a separate
        resource if you still want its value reported (see the ``report_mass``
        mirror in ``examples/06``)."""
        if axis not in self._provides:
            raise ValueError(
                f"loop_on({axis!r}) needs {axis!r} declared with provides() "
                f"too. A loop closes a feedback axis that must appear on BOTH "
                f"sides: provides() (the fed-back functionality) and requires() "
                f"(the produced resource). Currently provides() = "
                f"{sorted(self._provides)} and requires() = "
                f"{sorted(self._requires)}. Add m.provides({axis!r}, unit=...) "
                f"before calling loop_on({axis!r})."
            )
        if axis not in self._requires:
            raise ValueError(
                f"loop_on({axis!r}) needs {axis!r} declared with requires() "
                f"too. A loop closes a feedback axis that must appear on BOTH "
                f"sides: provides() (the fed-back functionality) and requires() "
                f"(the produced resource). Currently provides() = "
                f"{sorted(self._provides)} and requires() = "
                f"{sorted(self._requires)}. Add m.requires({axis!r}, unit=...) "
                f"before calling loop_on({axis!r}). Note loop_on() projects "
                f"{axis!r} out of the closed loop's F/R; if you still want its "
                f"value reported, mirror it into a separate resource "
                f"(a requires()/constraint() pair, like the ``report_mass`` "
                f"mirror in examples/06)."
            )
        self._loops.append(axis)
        return self

    # -- emit ------------------------------------------------------------

    def build(self) -> DesignProblem:
        """Produce a plain DesignProblem (with all declared loops closed)."""
        if not self._provides:
            raise ValueError(
                f"MCDP {self.name!r}: at least one provides() is required "
                f"before build(). A design problem needs a functionality "
                f"space. Declare one with m.provides('name', unit=...)."
            )
        if not self._requires:
            raise ValueError(
                f"MCDP {self.name!r}: at least one requires() is required "
                f"before build(). A design problem needs a resource space. "
                f"Declare one with m.requires('name', unit=...)."
            )

        F = Ports(dict(self._provides))
        R = Ports(dict(self._requires))

        if self._rule is not None:
            inner: DesignProblem = FunctionDP(
                F=F, R=R, h_fn=self._rule, name=self.name
            )
        else:
            # Verify every required resource has a constraint.
            missing = set(self._requires) - set(self._constraints)
            if missing:
                raise ValueError(
                    f"MCDP {self.name!r}: resource port(s) {sorted(missing)} "
                    f"declared via requires() have no constraint() or rule() "
                    f"defining them. Every resource needs an equation. Add "
                    f"m.constraint({sorted(missing)[0]!r}, lambda f: ...) for "
                    f"each, or supply a full m.rule(...)."
                )
            inner = AlgebraicDP(
                F=F, R=R,
                equations=dict(self._constraints),
                name=self.name,
            )

        # Close each declared loop in order. Loop drops the axis from F and R.
        dp: DesignProblem = inner
        for axis in self._loops:
            dp = Loop(dp, axis=axis)
        return dp
