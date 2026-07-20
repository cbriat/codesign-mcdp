"""
codesign: a Python library for Monotone Co-Design Problems (MCDPs).

Implementation of the framework from Andrea Censi's "A Mathematical Theory of
Co-Design" (arXiv:1512.08055). Users define design problems as relations
between functionality and resource posets, compose them with series/par/loop,
and solve the resulting MCDP via Kleene fixed-point iteration.

Quick example:

    from codesign import Reals, Ports, AlgebraicDP, solve

    F = Ports({"capacity": Reals(unit="J")})
    R = Ports({"mass": Reals(unit="kg")})
    battery = AlgebraicDP(F, R, {"mass": lambda f: f["capacity"] / 1.8e6})
    print(solve(battery, {"capacity": 3.6e6}))
"""

__version__ = "0.2.1"

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
from .posets import Discrete, NamedProduct, Naturals, Ports, Poset, Reals
from .primitives import adder, constant, identity, multiplier, scale
from .solver import SolveResult, TraceEntry, kleene_loop, minimize_cost, solve
from .sugar import Expr, ModuleHandle, Port, exp, log, sqrt
from .system import System
from .uncertainty import (
    Box,
    Circle,
    Disk,
    Ellipsoid,
    GaussianCopula,
    Independence,
    Stochastic,
    UncertaintyResult,
    UncertaintySet,
    solve_with_uncertainty,
)
from .online import (
    GaussianProcessEvaluator,
    LinearParametricEvaluator,
    LipschitzEvaluator,
    MonotonicityEvaluator,
    OnlineResult,
    OptimisticEvaluator,
    solve_online,
)
from .temporal import (
    Architecture,
    Epoch,
    EpochResult,
    ScheduleResult,
    solve_schedule,
)
from .dynamic import (
    DynamicPolicy,
    DynamicResult,
    Stage,
    StageResult,
    StateGrid,
    rollout,
    solve_and_rollout,
    solve_dynamic,
)
from .sequential import (
    MonotonicityReport,
    SeqPolicy,
    SeqResult,
    SeqStage,
    check_monotonicity,
    detect_resets,
    dp_over_catalog,
    factorise_at_resets,
    join_combine,
    precompute_catalog,
    solve_sequential,
    sum_combine,
)
from .state import (
    Axis,
    ContinuousAxis,
    DiscreteAxis,
    StateVec,
    VectorStateGrid,
    make_state,
    state_as_dict,
    state_get,
)
from .vector_dp import (
    VecPolicy,
    VecResult,
    VecStage,
    VectorMonotonicityReport,
    check_vector_monotonicity,
    solve_vector_sequential,
)
from .online_codesign import (
    ControlStep,
    OnlineCoDesignResult,
    resolve_at,
    run_online_codesign,
)
from . import viz
from . import diagram
from .diagram import draw_system

__all__ = [
    "Reals", "Naturals", "Discrete", "Ports", "NamedProduct", "Poset",
    "Antichain",
    "DesignProblem", "AlgebraicDP", "FunctionDP", "CatalogDP", "CatalogEntry",
    "ConstraintDP", "ODE_DP", "UncertainDP",
    "Series", "Parallel", "Loop", "series", "par", "loop",
    "adder", "multiplier", "scale", "constant", "identity",
    "solve", "minimize_cost", "kleene_loop", "SolveResult", "TraceEntry",
    "MCDP", "System", "Module",
    "Expr", "ModuleHandle", "Port", "sqrt", "exp", "log",
    "Box", "Disk", "Circle", "Ellipsoid",
    "Stochastic", "GaussianCopula", "Independence",
    "UncertaintySet", "UncertaintyResult", "solve_with_uncertainty",
    "OptimisticEvaluator", "MonotonicityEvaluator", "LipschitzEvaluator",
    "LinearParametricEvaluator", "GaussianProcessEvaluator",
    "OnlineResult", "solve_online",
    "Architecture", "Epoch", "EpochResult", "ScheduleResult", "solve_schedule",
    "Stage", "StageResult", "DynamicResult", "StateGrid", "DynamicPolicy",
    "solve_dynamic", "rollout", "solve_and_rollout",
    "SeqStage", "SeqResult", "SeqPolicy", "solve_sequential",
    "sum_combine", "join_combine", "MonotonicityReport",
    "check_monotonicity", "detect_resets", "factorise_at_resets",
    "precompute_catalog", "dp_over_catalog",
    "StateVec", "make_state", "state_get", "state_as_dict",
    "Axis", "ContinuousAxis", "DiscreteAxis", "VectorStateGrid",
    "VecStage", "VecResult", "VecPolicy", "solve_vector_sequential",
    "VectorMonotonicityReport", "check_vector_monotonicity",
    "ControlStep", "OnlineCoDesignResult", "resolve_at", "run_online_codesign",
    "draw_system",
]
