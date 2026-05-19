"""
Operator-overloaded constraint syntax for the :class:`System` builder.

This module provides three pieces:

- :class:`Port` represents a named handle into a subsystem F/R port or an
  outer F/R name. Ports support arithmetic operators (returning :class:`Expr`
  trees) and, for constrainable ports (module F, outer R), the ``>=`` operator
  registers a constraint with the parent :class:`~codesign.system.System`.
- :class:`Expr` and its subclasses form a tiny algebraic AST (Add, Sub, Mul,
  Div, Pow, Neg, Func). :func:`compile_expr` walks the tree and produces a
  callable equivalent to the legacy lambda-style constraint demand.
- :class:`ModuleHandle` is what :meth:`System.add` returns. Attribute access
  on the handle (``battery.capacity``) yields a :class:`Port` of the
  appropriate kind.

The result is that

.. code-block:: python

    battery.capacity >= (actuator.power + extra_power) * endurance

is equivalent to, but cleaner than,

.. code-block:: python

    sys.constrain("battery.capacity",
                  lambda x: (x["actuator.power"] + x["extra_power"]) * x["endurance"])

Both styles compile to the same internal constraint list and produce
identical results from :func:`~codesign.solver.solve`.
"""
from __future__ import annotations

import math as _math
from typing import Any, Callable, Dict, Optional


# ===========================================================================
# Expression tree
# ===========================================================================


class Expr:
    """Base class for all nodes in the constraint expression DSL.

    Arithmetic operators (``+``, ``-``, ``*``, ``/``, ``**``, unary ``-``)
    on an :class:`Expr` produce a new :class:`Expr` rather than evaluating
    numerically. The result is a tree that
    :func:`compile_expr` later turns into a callable.
    """

    # ---- Arithmetic ----
    def __add__(self, other): return Add(self, _to_expr(other))
    def __radd__(self, other): return Add(_to_expr(other), self)
    def __sub__(self, other): return Sub(self, _to_expr(other))
    def __rsub__(self, other): return Sub(_to_expr(other), self)
    def __mul__(self, other): return Mul(self, _to_expr(other))
    def __rmul__(self, other): return Mul(_to_expr(other), self)
    def __truediv__(self, other): return Div(self, _to_expr(other))
    def __rtruediv__(self, other): return Div(_to_expr(other), self)
    def __pow__(self, other): return Pow(self, _to_expr(other))
    def __neg__(self): return Neg(self)

    # ---- Constraint registration ----
    def __ge__(self, other):
        raise TypeError(
            "Only F ports and outer R ports can be on the LHS of a >= "
            "constraint. Got an expression: " + self.pretty()
        )

    def __le__(self, other):
        raise TypeError(
            "MCDP constraints are written as 'target >= demand', not "
            "'demand <= target'. Use >= with the F-port or outer-R-port "
            "on the LHS."
        )

    # ---- Guard against use in boolean context ----
    def __bool__(self):
        raise TypeError(
            "An expression has no truth value. If you meant to write a "
            "constraint, write 'lhs >= rhs' as a statement rather than "
            "inside a conditional."
        )

    # ---- Display ----
    def __repr__(self) -> str:
        return self.pretty()

    def pretty(self) -> str:  # pragma: no cover
        raise NotImplementedError


def _to_expr(x: Any) -> "Expr":
    """Coerce numbers to Const, leave Exprs alone, reject everything else."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return Const(x)
    raise TypeError(
        f"Cannot use {type(x).__name__} in a constraint expression "
        f"(got {x!r}); only numbers and Exprs are accepted."
    )


# ---- Leaves ----


class Const(Expr):
    """A literal numeric value."""

    def __init__(self, value: float):
        self.value = value

    def pretty(self) -> str:
        return f"{self.value!r}"


class Port(Expr):
    """A handle to a named port of a subsystem or to an outer F/R name.

    ``kind`` is one of:

    - ``"module_f"``: subsystem F port. Can be the LHS of a ``>=`` constraint.
      Cannot appear in a demand expression (it is determined by constraints,
      not by its current value).
    - ``"module_r"``: subsystem R port. Can appear in demand expressions.
      Cannot be on the LHS of ``>=`` (it is set by the module's ``h``).
    - ``"outer_f"``: an outer functionality. Can appear in demand expressions.
      Cannot be on the LHS of ``>=`` (it is the system's input).
    - ``"outer_r"``: an outer resource. Can be the LHS of a ``>=`` constraint.
      Cannot appear in demand expressions (it is the system's output).
    """

    __slots__ = ("_system", "_module_name", "_port_name", "_kind")

    def __init__(
        self,
        system: Any,
        module_name: Optional[str],
        port_name: str,
        kind: str,
    ):
        self._system = system
        self._module_name = module_name
        self._port_name = port_name
        self._kind = kind

    @property
    def full_name(self) -> str:
        if self._kind in ("module_f", "module_r"):
            return f"{self._module_name}.{self._port_name}"
        return self._port_name

    @property
    def kind(self) -> str:
        return self._kind

    def __ge__(self, other):
        if self._kind == "module_f":
            target = self.full_name
        elif self._kind == "outer_r":
            target = self._port_name
        elif self._kind == "module_r":
            raise TypeError(
                f"Cannot externally constrain a module R port "
                f"({self.full_name!r}). R ports are determined by the "
                f"module's own h(); you can only constrain F ports."
            )
        elif self._kind == "outer_f":
            raise TypeError(
                f"Cannot constrain an outer F port ({self._port_name!r}). "
                f"Outer F ports are inputs to the system; they take values "
                f"from solve()'s functionality argument."
            )
        else:  # pragma: no cover
            raise AssertionError(f"unknown port kind {self._kind!r}")

        rhs = _to_expr(other)
        self._system._register_expr_constraint(target, rhs)
        return _RegisteredConstraint(self, rhs)

    def pretty(self) -> str:
        return self.full_name


class _RegisteredConstraint:
    """Returned by ``port >= expr`` so the statement has a value. Falsy
    use raises a clear error to catch ``if port >= x:`` mistakes."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs: "Port", rhs: "Expr"):
        self.lhs = lhs
        self.rhs = rhs

    def __bool__(self):
        raise TypeError(
            "A registered constraint is not a boolean. The line "
            "'lhs >= rhs' has already registered the constraint as a "
            "side effect; you cannot use it in a conditional."
        )

    def __repr__(self) -> str:
        return f"<Constraint: {self.lhs.pretty()} >= {self.rhs.pretty()}>"


# ---- Binary operators ----


class _BinOp(Expr):
    op_symbol: str = "?"

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def pretty(self) -> str:
        return f"({self.left.pretty()} {self.op_symbol} {self.right.pretty()})"


class Add(_BinOp): op_symbol = "+"
class Sub(_BinOp): op_symbol = "-"
class Mul(_BinOp): op_symbol = "*"
class Div(_BinOp): op_symbol = "/"
class Pow(_BinOp): op_symbol = "**"


class Neg(Expr):
    def __init__(self, inner: Expr):
        self.inner = inner

    def pretty(self) -> str:
        return f"-{self.inner.pretty()}"


# ---- Function application ----


class Func(Expr):
    """A unary function applied to an expression (sqrt, exp, log, ...)."""

    def __init__(self, fn: Callable[[float], float], inner: Expr, name: str = ""):
        self.fn = fn
        self.inner = inner
        self.name = name or getattr(fn, "__name__", "f")

    def pretty(self) -> str:
        return f"{self.name}({self.inner.pretty()})"


def sqrt(x) -> Expr:
    """Apply ``math.sqrt`` to an expression (preserves the expression tree)."""
    return Func(_math.sqrt, _to_expr(x), name="sqrt")


def exp(x) -> Expr:
    """Apply ``math.exp`` to an expression."""
    return Func(_math.exp, _to_expr(x), name="exp")


def log(x) -> Expr:
    """Apply ``math.log`` to an expression."""
    return Func(_math.log, _to_expr(x), name="log")


# ===========================================================================
# Compilation: Expr -> Callable[[ctx], value]
# ===========================================================================


def compile_expr(expr: Expr) -> Callable[[Dict[str, Any]], Any]:
    """Compile an expression tree to a closure taking a context dict.

    The context dict carries the outer F values (bare names) and the
    subsystem R port values (dotted names like ``"battery.mass"``). The
    returned closure evaluates the expression numerically.
    """
    if isinstance(expr, Const):
        v = expr.value
        return lambda ctx, _v=v: _v

    if isinstance(expr, Port):
        if expr.kind == "outer_f":
            k = expr._port_name
            return lambda ctx, _k=k: ctx[_k]
        if expr.kind == "module_r":
            k = f"{expr._module_name}.{expr._port_name}"
            return lambda ctx, _k=k: ctx[_k]
        if expr.kind == "module_f":
            raise ValueError(
                f"Module F port {expr.full_name!r} cannot appear inside a "
                f"demand expression. F ports are targets of constraints, "
                f"not values; use the module's R ports on the RHS instead."
            )
        if expr.kind == "outer_r":
            raise ValueError(
                f"Outer R port {expr.full_name!r} cannot appear inside a "
                f"demand expression. Outer R ports are constraint targets, "
                f"not values."
            )
        raise AssertionError(f"unknown port kind {expr.kind!r}")  # pragma: no cover

    if isinstance(expr, Add):
        l, r = compile_expr(expr.left), compile_expr(expr.right)
        return lambda ctx: l(ctx) + r(ctx)
    if isinstance(expr, Sub):
        l, r = compile_expr(expr.left), compile_expr(expr.right)
        return lambda ctx: l(ctx) - r(ctx)
    if isinstance(expr, Mul):
        l, r = compile_expr(expr.left), compile_expr(expr.right)
        return lambda ctx: l(ctx) * r(ctx)
    if isinstance(expr, Div):
        l, r = compile_expr(expr.left), compile_expr(expr.right)
        return lambda ctx: l(ctx) / r(ctx)
    if isinstance(expr, Pow):
        l, r = compile_expr(expr.left), compile_expr(expr.right)
        return lambda ctx: l(ctx) ** r(ctx)
    if isinstance(expr, Neg):
        i = compile_expr(expr.inner)
        return lambda ctx: -i(ctx)
    if isinstance(expr, Func):
        i = compile_expr(expr.inner)
        fn = expr.fn
        return lambda ctx: fn(i(ctx))

    raise TypeError(f"unsupported Expr node: {type(expr).__name__}")  # pragma: no cover


# ===========================================================================
# ModuleHandle
# ===========================================================================


class ModuleHandle:
    """A handle on a subsystem added to a :class:`~codesign.system.System`.

    Attribute access (``handle.port_name``) returns a :class:`Port` of kind
    ``"module_f"`` or ``"module_r"`` depending on which side of the underlying
    DP the port lives on. F and R port names must be disjoint within a
    single subsystem (this is enforced at handle construction).

    The handle is what :meth:`System.add` returns. The old usage that ignored
    the return value continues to work; the handle is just additional sugar.
    """

    def __init__(self, system: Any, name: str, dp: Any):
        # bypass __setattr__ if one is defined later
        object.__setattr__(self, "_system", system)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_dp", dp)

        f_ports = set(dp.F.components)
        r_ports = set(dp.R.components)
        overlap = f_ports & r_ports
        if overlap:
            raise ValueError(
                f"Subsystem {name!r} has port names that appear in both F "
                f"and R: {sorted(overlap)}. For the operator-overloaded "
                f"syntax, F and R port names within a single subsystem must "
                f"be disjoint. Rename one side (e.g. add a suffix like "
                f"_in/_out)."
            )
        object.__setattr__(self, "_f_ports", f_ports)
        object.__setattr__(self, "_r_ports", r_ports)

    def __getattr__(self, port_name: str):
        if port_name.startswith("_"):
            raise AttributeError(port_name)
        if port_name in self._f_ports:
            return Port(self._system, self._name, port_name, "module_f")
        if port_name in self._r_ports:
            return Port(self._system, self._name, port_name, "module_r")
        raise AttributeError(
            f"Subsystem {self._name!r} has no port named {port_name!r}. "
            f"F ports: {sorted(self._f_ports)}. "
            f"R ports: {sorted(self._r_ports)}."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def dp(self) -> Any:
        return self._dp

    def __repr__(self) -> str:
        return (
            f"<ModuleHandle {self._name!r}: "
            f"F={sorted(self._f_ports)} R={sorted(self._r_ports)}>"
        )
