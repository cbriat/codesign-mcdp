"""
codesign: a Python library for Monotone Co-Design Problems (MCDPs).

Implementation of the framework from Andrea Censi's "A Mathematical Theory of
Co-Design" (arXiv:1512.08055). Users define design problems as relations
between functionality and resource posets, compose them with series/par/loop,
and solve the resulting MCDP via Kleene fixed-point iteration.

Quick example:

    from codesign import Reals, NamedProduct, AlgebraicDP, solve

    F = NamedProduct({"capacity": Reals(unit="J")})
    R = NamedProduct({"mass": Reals(unit="kg")})
    battery = AlgebraicDP(F, R, {"mass": lambda f: f["capacity"] / 1.8e6})
    print(solve(battery, {"capacity": 3.6e6}))
"""

__version__ = "0.1.0"

from .antichains import Antichain
from .composition import Loop, Parallel, Series, loop, par, series
from .dp import (
    AlgebraicDP,
    CatalogDP,
    CatalogEntry,
    ConstraintDP,
    DesignProblem,
    FunctionDP,
    ODE_DP,
    UncertainDP,
)
from .mcdpl import MCDP
from .module import Module
from .posets import Discrete, NamedProduct, Naturals, Poset, Reals
from .primitives import adder, constant, identity, multiplier, scale
from .solver import SolveResult, kleene_loop, minimize_cost, solve
from .sugar import Expr, ModuleHandle, Port, exp, log, sqrt
from .system import System

__all__ = [
    "Reals", "Naturals", "Discrete", "NamedProduct", "Poset",
    "Antichain",
    "DesignProblem", "AlgebraicDP", "FunctionDP", "CatalogDP", "CatalogEntry",
    "ConstraintDP", "ODE_DP", "UncertainDP",
    "Series", "Parallel", "Loop", "series", "par", "loop",
    "adder", "multiplier", "scale", "constant", "identity",
    "solve", "minimize_cost", "kleene_loop", "SolveResult",
    "MCDP", "System", "Module",
    "Expr", "ModuleHandle", "Port", "sqrt", "exp", "log",
]
