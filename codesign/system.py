"""
System: modular MCDP composition with named subsystems.

Each subsystem (battery, actuator, sensor, ...) is its own ``DesignProblem``
with its own functionality and resource posets. A ``System`` assembles
several subsystems by

    1. declaring outer functionalities (``provides``) and outer resources
       (``requires``) that the system as a whole exposes;
    2. adding subsystems by name (``add``);
    3. attaching connection constraints (``constrain``) of the form

           target_port >= demand_expression(ctx)

       where ``target_port`` is either ``"module.f_port"`` (a subsystem's
       required functionality) or an outer R name, and ``demand_expression``
       is a plain Python function of a dict ``ctx`` holding the outer F
       values plus every subsystem R port under its dotted name
       (``"battery.mass"``, ``"actuator.power"``).

The ``build()`` method emits a single ``DesignProblem`` whose Kleene
iteration closes a feedback loop over the bundle of every subsystem R.
The result composes like any other DP and can itself be used as a
subsystem in a larger System.

Notes
-----
* The loop axis is internal and named ``__modules__``. Users do not see it.
* When a subsystem returns a multi-valued antichain (e.g. CatalogDP), the
  System takes the Cartesian product across subsystems and lets the Min in
  the outer antichain prune dominated combinations. This can blow up if
  every subsystem is multi-valued; for engineering models it is rarely a
  bottleneck.
* Multiple ``constrain(target, ...)`` calls with the same target are
  joined: the effective demand is the maximum (the join in Reals/Naturals).
* Module F ports that have no constraint declared are an error. Outer R
  ports without a constraint are also an error.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional

from .antichains import Antichain
from .composition import Loop
from .dp import DesignProblem, FunctionDP
from .posets import Ports, Poset, Reals
from .sugar import Expr, ModuleHandle, Port, compile_expr


# Internal name for the loop axis bundling every subsystem's R. Users
# should never need to type this directly.
_MODULES_AXIS = "__modules__"


def _target_to_string(target) -> str:
    """Normalise a constraint target to its string form.

    Accepts either a plain string (legacy form) or a Port (operator-overloaded
    form). Raises ``TypeError`` otherwise.
    """
    if isinstance(target, str):
        return target
    if isinstance(target, Port):
        if target.kind == "module_f":
            return target.full_name
        if target.kind == "outer_r":
            return target._port_name
        if target.kind == "module_r":
            raise TypeError(
                f"Cannot constrain a module R port ({target.full_name!r}). "
                f"R ports are determined by the module's h, not externally."
            )
        if target.kind == "outer_f":
            raise TypeError(
                f"Cannot constrain an outer F port ({target._port_name!r}). "
                f"Outer F ports are system inputs, not constraint targets."
            )
    raise TypeError(
        f"constraint target must be a string or a Port (got "
        f"{type(target).__name__})"
    )


class System:
    """Modular MCDP composition with named subsystems.

    Build with ``provides``/``requires``/``add``/``constrain`` and emit a
    plain ``DesignProblem`` with ``build()``. The result is solved with
    the ordinary ``solve`` function and can be nested inside another
    System.
    """

    def __init__(self, name: str = "system"):
        self.name = name
        self._outer_F: Dict[str, Poset] = {}
        self._outer_R: Dict[str, Poset] = {}
        self._modules: Dict[str, DesignProblem] = {}
        # constraint entries: (target_string, demand_callable)
        self._constraints: List[tuple] = []

    # ------------------------------------------------------------------ #
    # Context manager sugar (optional)
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "System":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    # ------------------------------------------------------------------ #
    # Declarations
    # ------------------------------------------------------------------ #
    def provides(self, name: str, *, unit: str = "", poset: Poset = None) -> Port:
        """Declare an outer functionality.

        Returns a :class:`~codesign.sugar.Port` handle that can be used in
        operator-overloaded constraint expressions. For example::

            endurance = sys.provides("endurance", unit="s")
            battery.capacity >= endurance * actuator.power

        The legacy usage that discards the return value continues to work;
        constraints can still be added with the string-based
        :meth:`constrain` form.
        """
        if name in self._outer_F:
            raise ValueError(f"provides({name}) already declared")
        self._outer_F[name] = poset if poset is not None else Reals(unit=unit)
        return Port(self, None, name, "outer_f")

    def requires(self, name: str, *, unit: str = "", poset: Poset = None) -> Port:
        """Declare an outer resource.

        Returns a :class:`~codesign.sugar.Port` that can be the LHS of a
        ``>=`` constraint, for example::

            total_mass = sys.requires("total_mass", unit="kg")
            total_mass >= battery.mass + payload
        """
        if name in self._outer_R:
            raise ValueError(f"requires({name}) already declared")
        self._outer_R[name] = poset if poset is not None else Reals(unit=unit)
        return Port(self, None, name, "outer_r")

    def add(self, module_name: str, dp: DesignProblem) -> ModuleHandle:
        """Add a subsystem under a unique name.

        Returns a :class:`~codesign.sugar.ModuleHandle`. Attribute access on
        the handle (``handle.port_name``) yields a
        :class:`~codesign.sugar.Port` that can participate in constraint
        expressions.
        """
        if module_name in self._modules:
            raise ValueError(f"module name {module_name!r} already in use")
        if "." in module_name:
            raise ValueError("module name may not contain '.' (used for port refs)")
        if module_name == _MODULES_AXIS:
            raise ValueError(f"module name {_MODULES_AXIS!r} is reserved")
        if not isinstance(dp.F, Ports) or not isinstance(dp.R, Ports):
            raise ValueError(
                f"subsystem {module_name!r} must have Ports F and R "
                f"(got F={type(dp.F).__name__}, R={type(dp.R).__name__})"
            )
        self._modules[module_name] = dp
        return ModuleHandle(self, module_name, dp)

    def constrain(self, target, demand) -> "System":
        """Add a constraint of the form ``target >= demand``.

        ``target`` may be:

        - a string of the form ``"module.f_port"`` or an outer R name, or
        - a :class:`~codesign.sugar.Port` (typically obtained from a
          :class:`~codesign.sugar.ModuleHandle` or from :meth:`provides` / :meth:`requires`).

        ``demand`` may be:

        - a callable taking a ``ctx`` dict (the legacy lambda form), or
        - an :class:`~codesign.sugar.Expr` (the operator-overloaded form).

        Both forms compile to the same internal callable. Multiple
        ``constrain`` calls on the same target are joined with ``max``.
        """
        target_str = _target_to_string(target)
        if isinstance(demand, Expr):
            fn = compile_expr(demand)
            self._constraints.append((target_str, fn, demand))
        elif callable(demand):
            self._constraints.append((target_str, demand, None))
        else:
            raise TypeError(
                f"demand must be a callable or an Expr (got "
                f"{type(demand).__name__})"
            )
        return self

    def _register_expr_constraint(self, target: str, rhs_expr: Expr) -> None:
        """Internal hook called by Port.__ge__ to register a constraint."""
        self._constraints.append((target, compile_expr(rhs_expr), rhs_expr))

    # Convenience aliases mirroring MCDPL spelling.
    sub = add
    eq = constrain

    # ------------------------------------------------------------------ #
    # Internal validation
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if not self._modules and not self._outer_R:
            raise ValueError(
                "System has no subsystems and no outer resources; nothing to solve"
            )
        if not self._outer_R:
            raise ValueError("System must declare at least one requires()")

        # Bucket constraints by target.
        f_targets: Dict[tuple, List[Callable]] = {}
        r_targets: Dict[str, List[Callable]] = {}
        for target, fn, _expr in self._constraints:
            if "." in target:
                mod, port = target.split(".", 1)
                if mod not in self._modules:
                    raise ValueError(
                        f"constraint targets unknown module {mod!r} "
                        f"(known: {sorted(self._modules)})"
                    )
                if port not in self._modules[mod].F.components:
                    raise ValueError(
                        f"module {mod!r} has no F port {port!r} "
                        f"(ports: {list(self._modules[mod].F.components)})"
                    )
                f_targets.setdefault((mod, port), []).append(fn)
            else:
                if target not in self._outer_R:
                    raise ValueError(
                        f"constraint target {target!r} is not an outer R "
                        f"and is not in 'module.port' form"
                    )
                r_targets.setdefault(target, []).append(fn)

        # Every subsystem F port and every outer R must have at least one
        # constraint; otherwise the system is under-determined.
        for mod_name, mod in self._modules.items():
            for port in mod.F.components:
                if (mod_name, port) not in f_targets:
                    raise ValueError(
                        f"subsystem {mod_name!r} F port {port!r} has no "
                        "constraint; add one with constrain("
                        f"\"{mod_name}.{port}\", ...)"
                    )
        for r_name in self._outer_R:
            if r_name not in r_targets:
                raise ValueError(
                    f"outer R {r_name!r} has no constraint; add one with "
                    f"constrain(\"{r_name}\", ...)"
                )

        self._f_targets = f_targets
        self._r_targets = r_targets

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self) -> DesignProblem:
        """Produce a plain DesignProblem; can then be passed to ``solve``."""
        self._validate()
        modules = dict(self._modules)
        f_targets = self._f_targets
        r_targets = self._r_targets

        # Build the module-R bundle poset (axis of the internal loop).
        module_R_bundle = Ports({
            mod_name: mod.R for mod_name, mod in modules.items()
        })

        # When the System has subsystems, the internal loop's feedback
        # axis is a bundle of every subsystem's R ports: the solver
        # iterates an estimate of all module outputs until it reaches a
        # fixed point. With no subsystems the System is purely algebraic
        # (outer F straight to outer R), so the loop axis is skipped and
        # the result is a flat FunctionDP.
        if modules:
            inner_F_components = dict(self._outer_F)
            inner_F_components[_MODULES_AXIS] = module_R_bundle
            inner_F = Ports(inner_F_components)

            inner_R_components = dict(self._outer_R)
            inner_R_components[_MODULES_AXIS] = module_R_bundle
            inner_R = Ports(inner_R_components)
        else:
            inner_F = (
                Ports(dict(self._outer_F))
                if self._outer_F else Ports({"_": Reals()})
            )
            inner_R = Ports(dict(self._outer_R))

        outer_F_keys = list(self._outer_F)
        outer_R_keys = list(self._outer_R)

        def _ctx_with_modules(f_in: Mapping, modules_R: Mapping) -> Dict[str, Any]:
            """Build the context dict for demand expression evaluation."""
            ctx: Dict[str, Any] = {}
            for k in outer_F_keys:
                ctx[k] = f_in[k]
            if modules_R is not None:
                for mod_name, mod in modules.items():
                    mr = modules_R[mod_name]
                    for port_name in mod.R.components:
                        ctx[f"{mod_name}.{port_name}"] = mr[port_name]
            return ctx

        def _eval_max(fns: List[Callable], ctx: Mapping) -> Any:
            """Evaluate all demand fns in this constraint and take the join."""
            value = None
            for fn in fns:
                v = fn(ctx)
                if value is None:
                    value = v
                else:
                    # Multiple demands on one port are combined by their
                    # join, which on a totally ordered (chain) poset such
                    # as the reals is the maximum. Callers needing a join
                    # on a richer poset should pre-combine the demands;
                    # the max fallback below covers the common chain case.
                    try:
                        value = max(value, v)
                    except TypeError:
                        # Non-orderable values: keep the most recent and
                        # leave correctness to the caller.
                        value = v
            return value

        # --- Inner h ----------------------------------------------------- #
        def inner_h(f_in):
            # Snapshot module R estimate (from the loop axis).
            modules_R = f_in.get(_MODULES_AXIS) if modules else None
            ctx_loop = _ctx_with_modules(f_in, modules_R)

            # For each subsystem, compute its f from constraint demands,
            # then call h. Collect the antichain.
            module_antichains: Dict[str, Antichain] = {}
            for mod_name, mod in modules.items():
                f_M = {}
                for port_name in mod.F.components:
                    fns = f_targets[(mod_name, port_name)]
                    f_M[port_name] = _eval_max(fns, ctx_loop)
                try:
                    a_M = mod.h(f_M)
                except (OverflowError, ValueError, ZeroDivisionError):
                    a_M = Antichain.singleton(mod.R, mod.R.top())
                if a_M.is_empty():
                    # No feasible point for this subsystem; the whole system
                    # is infeasible at this loop iterate. Signal with a
                    # top antichain so the outer iteration can detect it.
                    a_M = Antichain.singleton(mod.R, mod.R.top())
                module_antichains[mod_name] = a_M

            # Take the Cartesian product over the subsystems' antichains.
            # Each combination picks one resource point from every
            # subsystem, forming a candidate full module-R bundle; the
            # outer R values are then evaluated from the constraint
            # expressions against that bundle.
            candidate_points: List[Dict[str, Any]] = []
            module_names = list(modules.keys())
            antichain_lists = [list(module_antichains[m]) for m in module_names]

            for combo in itertools.product(*antichain_lists) if module_names else [()]:
                modules_R_value: Dict[str, Any] = {}
                for m_name, r_val in zip(module_names, combo):
                    modules_R_value[m_name] = r_val

                ctx_combo = _ctx_with_modules(f_in, modules_R_value)

                outer_r_values: Dict[str, Any] = {}
                for r_name in outer_R_keys:
                    fns = r_targets[r_name]
                    outer_r_values[r_name] = _eval_max(fns, ctx_combo)

                out_point: Dict[str, Any] = dict(outer_r_values)
                if module_names:
                    out_point[_MODULES_AXIS] = modules_R_value
                candidate_points.append(out_point)

            return Antichain.from_set(inner_R, candidate_points)

        inner = FunctionDP(
            F=inner_F, R=inner_R, h_fn=inner_h, name=f"{self.name}_inner"
        )
        if not modules:
            inner._codesign_modules = {}
            inner._codesign_constraints = list(self._constraints)
            return inner
        result = Loop(inner, axis=_MODULES_AXIS)
        # Attach a reference to the modules dict so the uncertainty solver
        # (and other tooling) can find the Module instances after build().
        result._codesign_modules = dict(modules)
        result._codesign_constraints = list(self._constraints)
        return result

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        lines = [f"System({self.name!r}):"]
        if self._outer_F:
            lines.append("  provides:")
            for k, p in self._outer_F.items():
                lines.append(f"    {k}: {p.name}")
        if self._outer_R:
            lines.append("  requires:")
            for k, p in self._outer_R.items():
                lines.append(f"    {k}: {p.name}")
        if self._modules:
            lines.append("  subsystems:")
            for k, m in self._modules.items():
                f_ports = ", ".join(m.F.components.keys())
                r_ports = ", ".join(m.R.components.keys())
                lines.append(f"    {k}: ({f_ports}) -> ({r_ports})")
        if self._constraints:
            lines.append("  constraints:")
            for target, _fn, expr in self._constraints:
                if expr is not None:
                    lines.append(f"    {target} >= {expr.pretty()}")
                else:
                    lines.append(f"    {target} >= <lambda>")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Block-diagram rendering (Level-1 visual)
    # ------------------------------------------------------------------ #
    def draw_diagram(self, **kwargs):
        """Render this system as a GraphViz block diagram.

        Returns a :class:`graphviz.Digraph` object. See
        :func:`codesign.diagram.draw_system` for the full list of
        keyword arguments (``rankdir``, ``show_ports``,
        ``highlight_cycles``, ``graph_attrs``, ``name``). Display
        inline in a Jupyter notebook, or save to disk with
        ``.render(filename, format="svg")``.

        Raises ImportError if the ``graphviz`` package is not
        installed.
        """
        from .diagram import draw_system
        return draw_system(self, **kwargs)
