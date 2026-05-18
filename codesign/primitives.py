"""
Reusable building-block DPs for everyday co-design plumbing.

These wrap common patterns like addition, multiplication, scaling, identity.
They correspond to the "degenerate monotone design problems" the paper uses
to glue diagrams together (Fig. 26, Fig. 35).
"""
from __future__ import annotations

from .dp import AlgebraicDP
from .posets import NamedProduct, Reals


def adder(in_names: list[str], out_name: str, poset: Reals | None = None) -> AlgebraicDP:
    """Sum several scalar inputs into one scalar output."""
    p = poset or Reals()
    F = NamedProduct({n: p for n in in_names})
    R = NamedProduct({out_name: p})
    return AlgebraicDP(
        F=F,
        R=R,
        equations={out_name: lambda f, ns=in_names: sum(f[n] for n in ns)},
        name=f"add[{','.join(in_names)}->{out_name}]",
    )


def multiplier(
    in_a: str, in_b: str, out_name: str, poset: Reals | None = None
) -> AlgebraicDP:
    """Multiply two scalar inputs (e.g. current * voltage = power)."""
    p = poset or Reals()
    F = NamedProduct({in_a: p, in_b: p})
    R = NamedProduct({out_name: p})
    return AlgebraicDP(
        F=F,
        R=R,
        equations={out_name: lambda f, a=in_a, b=in_b: f[a] * f[b]},
        name=f"mul[{in_a}*{in_b}->{out_name}]",
    )


def scale(in_name: str, out_name: str, factor: float, poset: Reals | None = None) -> AlgebraicDP:
    """Multiply by a constant (e.g. capacity * one_over_alpha = mass)."""
    p = poset or Reals()
    F = NamedProduct({in_name: p})
    R = NamedProduct({out_name: p})
    return AlgebraicDP(
        F=F,
        R=R,
        equations={out_name: lambda f, k=factor, n=in_name: f[n] * k},
        name=f"scale[{in_name}*{factor}->{out_name}]",
    )


def constant(out_name: str, value: float, poset: Reals | None = None) -> AlgebraicDP:
    """A DP that ignores its (trivial) functionality and emits a constant."""
    p = poset or Reals()
    F = NamedProduct({"_": p})
    R = NamedProduct({out_name: p})
    return AlgebraicDP(
        F=F,
        R=R,
        equations={out_name: lambda f, v=value: v},
        name=f"const[{out_name}={value}]",
    )


def identity(name: str, poset: Reals | None = None) -> AlgebraicDP:
    """Pass a single named scalar through unchanged."""
    p = poset or Reals()
    F = NamedProduct({name: p})
    R = NamedProduct({name: p})
    return AlgebraicDP(
        F=F,
        R=R,
        equations={name: lambda f, n=name: f[n]},
        name=f"id[{name}]",
    )
