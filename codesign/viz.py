"""
Visualisation helpers for codesign results.

This module provides ready-made plotting and graph functions for the
common diagnostics you reach for when working with co-design:

- :func:`plot_antichain`: scatter the resource Pareto front (2D or 3D)
  with the dominated region optionally shaded, so you can see at a
  glance which trade-offs the system actually offers.
- :func:`plot_convergence`: semilog plot of the Kleene-iteration delta
  on a :class:`~codesign.solver.SolveResult`'s trace, the standard
  diagnostic for "is the solver behaving."
- :func:`plot_uncertainty`: histogram of MC samples from an
  :class:`~codesign.uncertainty.UncertaintyResult`, with summary lines
  for nominal, mean, p95, CVaR95, and worst-case overlaid.
- :func:`to_dot`: emit a GraphViz dot string for the structure of a DP
  (Series/Parallel/Loop trees, or for a System-built DP, the subsystem
  and constraint graph).

Matplotlib is imported lazily on the plotting functions so the rest of
the package stays usable without it.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "codesign.viz.plot_* functions require matplotlib. "
            "Install with `pip install matplotlib`."
        ) from e


def _extract_points(antichain, axes: Sequence[str]) -> List[Tuple[float, ...]]:
    """Pull numeric coordinates from each point of the antichain."""
    out: List[Tuple[float, ...]] = []
    for p in antichain.points:
        try:
            row = tuple(float(p[k]) for k in axes)
        except (KeyError, TypeError, ValueError):
            continue
        if any(r != r or abs(r) == float("inf") for r in row):
            continue
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# plot_antichain (2D and 3D)
# ---------------------------------------------------------------------------


def plot_antichain(
    result,
    axes: Sequence[str],
    *,
    ax=None,
    title: Optional[str] = None,
    shade_dominated: bool = True,
    point_size: float = 60.0,
    label: Optional[str] = None,
):
    """Scatter the antichain on the chosen axes.

    Accepts a :class:`~codesign.solver.SolveResult`, an
    :class:`~codesign.uncertainty.UncertaintyResult` (uses its
    worst-case antichain), or an :class:`~codesign.antichains.Antichain`
    directly. The number of ``axes`` selects 2D or 3D rendering.

    Parameters
    ----------
    result : SolveResult or UncertaintyResult or Antichain
        The object to plot.
    axes : sequence of str
        Two or three R-port names to use for the axes.
    ax : matplotlib axes, optional
        Existing axes to draw into. A new figure is created if omitted.
    title : str, optional
        Plot title. Defaults to a reasonable summary.
    shade_dominated : bool
        2D only. Shade the upper-right (Pareto-dominated) region of each
        front point, so the un-dominated frontier is visible at a glance.
    point_size : float
        Matplotlib marker size for the antichain points.
    label : str, optional
        Legend label for the scattered points.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the antichain was drawn into.
    """
    plt = _require_matplotlib()

    # Resolve to an antichain.
    if hasattr(result, "antichain"):
        antichain = result.antichain
    elif hasattr(result, "worst_case") and result.worst_case is not None:
        antichain = result.worst_case.antichain
    elif hasattr(result, "points"):
        antichain = result
    else:
        raise TypeError(
            f"plot_antichain: result must be a SolveResult, "
            f"UncertaintyResult, or Antichain, got {type(result).__name__}."
        )

    if len(axes) not in (2, 3):
        raise ValueError(
            f"plot_antichain: axes must be a sequence of 2 or 3 R-port names, "
            f"got {len(axes)} ({list(axes)}). Antichains are plotted in 2-D "
            f"or 3-D."
        )

    pts = _extract_points(antichain, axes)
    if not pts:
        raise ValueError(
            "plot_antichain: no plottable points (the antichain is empty, "
            "infeasible, or its values are non-numeric on the chosen axes)."
        )

    if len(axes) == 2:
        created = ax is None
        if created:
            _, ax = plt.subplots(figsize=(7, 5))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if shade_dominated:
            # Shade the Pareto-dominated (upper-right) region as a single
            # clean monotone staircase, so the frontier reads at a glance.
            # The antichain points form the lower-left frontier; every point
            # dominates the quadrant up and to the right of it.
            order = sorted(pts, key=lambda r: r[0])
            x_min = min(xs); y_min = min(ys)
            x_max = max(xs); y_max = max(ys)
            pad_x = (x_max - x_min) * 0.15 if x_max > x_min else max(1.0, abs(x_max))
            pad_y = (y_max - y_min) * 0.15 if y_max > y_min else max(1.0, abs(y_max))
            xlo = x_min - pad_x
            xhi = x_max + pad_x
            ylo = y_min - pad_y
            yhi = y_max + pad_y
            # Trace the boundary: down the left edge, then a descending
            # staircase across the points, then up the right edge.
            step_x = [order[0][0], order[0][0]]
            step_y = [yhi, order[0][1]]
            for i in range(1, len(order)):
                xi, yi = order[i]
                prev_y = order[i - 1][1]
                step_x.extend([xi, xi])       # horizontal run at prev level
                step_y.extend([prev_y, yi])   # then step down to this point
            step_x.append(xhi); step_y.append(order[-1][1])
            step_x.append(xhi); step_y.append(yhi)
            ax.fill(step_x, step_y, color="C3", alpha=0.12, linewidth=0,
                    zorder=1, label="dominated region")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
        ax.scatter(xs, ys, s=point_size, c="C3", zorder=4,
                   edgecolors="black", linewidths=0.7, label=label)
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_title(title or f"Pareto front: {axes[0]} vs {axes[1]}")
        ax.grid(True, alpha=0.3)
        # Only show a legend when the caller named the series; otherwise a
        # lone "dominated region" entry just adds clutter.
        if label is not None:
            ax.legend(loc="best", framealpha=0.9)
        if created:
            ax.figure.tight_layout()
        return ax

    # 3D
    created = ax is None
    if created:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    ax.scatter(xs, ys, zs, s=point_size, c="C3",
               edgecolors="black", linewidths=0.5, label=label)
    # Give the tick/axis labels room so they don't run into the panes.
    ax.set_xlabel(axes[0], labelpad=8)
    ax.set_ylabel(axes[1], labelpad=8)
    ax.set_zlabel(axes[2], labelpad=8)
    ax.set_title(title or f"Pareto front: {axes[0]}, {axes[1]}, {axes[2]}")
    if label is not None:
        ax.legend(loc="upper left")
    if created:
        # A little extra margin keeps the 3D axis labels off the figure edge.
        ax.figure.subplots_adjust(left=0.02, right=0.94, bottom=0.06, top=0.92)
    return ax


# ---------------------------------------------------------------------------
# plot_convergence
# ---------------------------------------------------------------------------


def plot_convergence(
    result_or_trace,
    *,
    ax=None,
    title: str = "Kleene-iteration convergence",
    floor: float = 1e-18,
    label: Optional[str] = None,
):
    """Semilog plot of per-iteration delta from a trace.

    Accepts a :class:`SolveResult` with ``trace`` populated (call
    ``solve(..., trace=True)``) or the trace list directly.

    Zero deltas are clamped to ``floor`` so log scales render cleanly.
    """
    plt = _require_matplotlib()
    if hasattr(result_or_trace, "trace") and result_or_trace.trace is not None:
        trace = result_or_trace.trace
    else:
        trace = result_or_trace

    iters: List[int] = []
    deltas: List[float] = []
    for e in trace:
        if e.delta is None:
            continue
        iters.append(e.iteration)
        d = e.delta
        deltas.append(max(d, floor) if isinstance(d, (int, float)) and d > 0 else floor)

    if not iters:
        raise ValueError("plot_convergence: trace has no numeric deltas.")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(iters, deltas, marker="o", markersize=4, label=label)
    ax.set_xlabel("Kleene iteration")
    ax.set_ylabel("delta (max absolute change)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if label is not None:
        ax.legend(loc="best")
    if created:
        ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# plot_uncertainty (MC histogram with summary lines)
# ---------------------------------------------------------------------------


def plot_uncertainty(
    result,
    port: str,
    *,
    ax=None,
    bins: int = 30,
    title: Optional[str] = None,
    nominal: Optional[float] = None,
    show_summaries: bool = True,
    xlabel: Optional[str] = None,
    legend_loc: str = "best",
):
    """Histogram of Monte Carlo samples on a chosen R port.

    Pass an :class:`~codesign.uncertainty.UncertaintyResult` that was
    produced with ``"samples"`` in the requested summaries (so the raw
    antichains are available). Vertical lines mark each requested
    summary (mean, p95, cvar95, worst_case), plus the nominal value if
    supplied.

    For multi-point antichains, the first point is used (the typical
    case is a single-output system like the drone in example 12).

    Parameters
    ----------
    xlabel : str, optional
        Human-readable x-axis label (e.g. ``"total cost (USD)"``). When
        omitted, the raw ``port`` name is used.
    legend_loc : str, optional
        Matplotlib legend location for the summary lines. Defaults to
        ``"best"`` so the legend avoids the tallest bars; pass e.g.
        ``"upper right"`` to pin it.
    """
    plt = _require_matplotlib()

    if not hasattr(result, "samples") or result.samples is None:
        raise ValueError(
            "plot_uncertainty: the UncertaintyResult has no `samples`. "
            "Add 'samples' to the uncertainty list when calling solve()."
        )

    values: List[float] = []
    for a in result.samples:
        if a.is_empty() or a.has_any_top():
            continue
        first = next(iter(a.points))
        v = first.get(port)
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
            values.append(float(v))
    if not values:
        raise ValueError(
            f"plot_uncertainty: no finite values on port {port!r}."
        )

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4.5))
    # Keep edges light so a narrow, many-binned distribution doesn't smear
    # into a solid black block.
    ax.hist(values, bins=bins, alpha=0.55, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel if xlabel is not None else port)
    ax.set_ylabel("samples")

    if show_summaries:
        if nominal is not None:
            ax.axvline(nominal, color="gray", linestyle=":",
                       label=f"nominal {nominal:.4g}")
        if result.mean is not None and port in result.mean:
            ax.axvline(result.mean[port], color="green", linestyle="-",
                       label=f"mean {result.mean[port]:.4g}")
        if result.p95 is not None and port in result.p95:
            ax.axvline(result.p95[port], color="orange", linestyle="-",
                       label=f"p95 {result.p95[port]:.4g}")
        if result.cvar95 is not None and port in result.cvar95:
            ax.axvline(result.cvar95[port], color="red", linestyle="-",
                       label=f"CVaR95 {result.cvar95[port]:.4g}")
        if result.worst_case is not None:
            wc_pts = list(result.worst_case.antichain.points)
            if wc_pts and port in wc_pts[0]:
                wc = wc_pts[0][port]
                ax.axvline(wc, color="black", linestyle="--",
                           label=f"worst case {wc:.4g}")
        ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)

    ax.set_title(title or f"MC distribution of {port}")
    if created:
        ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# to_dot: GraphViz string for DP structure
# ---------------------------------------------------------------------------


def _node_id(prefix: str, counter: List[int]) -> str:
    counter[0] += 1
    return f"{prefix}{counter[0]}"


def _short_label(s: str, max_len: int = 32) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "..."


def _dot_graph_id(name: str) -> str:
    """Return ``name`` usable as a dot graph ID: bare if already a valid identifier, else double-quoted with quotes/backslashes escaped."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return name
    escaped = name.replace("\\", r"\\").replace('"', r"\"")
    return f'"{escaped}"'


def to_dot(dp, *, name: str = "codesign") -> str:
    """Emit a GraphViz dot string describing the DP's structure.

    Three cases are handled:

    - **System-built DPs** (a :class:`Loop` carrying ``_codesign_modules``):
      render the subsystems as nodes with their F and R port lists, plus
      the constraint edges if they have been attached.
    - **Composition trees** (Series, Parallel, Loop): recursively render
      the nested DPs as nested boxes.
    - **Leaves** (single primitive DPs and Modules): render as a single
      box with the F and R port lists.

    The output is a string; save it to a file and run ``dot -Tpng`` (or
    ``-Tsvg``) to render. Returning a string keeps this module free of
    the graphviz package dependency.
    """
    counter = [0]
    lines: List[str] = [f"digraph {_dot_graph_id(name)} {{"]
    lines.append("  rankdir=LR;")
    lines.append('  node [shape=box, style="rounded,filled", '
                 'fillcolor="#f6f6f6", fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=10];')

    _to_dot_recurse(dp, lines, counter, parent=None, edge_label=None)
    lines.append("}")
    return "\n".join(lines)


def _module_label(mod, mod_name: Optional[str] = None) -> str:
    f_ports = ", ".join(mod.F.components.keys()) if hasattr(mod.F, "components") else "?"
    r_ports = ", ".join(mod.R.components.keys()) if hasattr(mod.R, "components") else "?"
    title = mod_name or mod.__class__.__name__
    return f"{title}\\nF: {f_ports}\\nR: {r_ports}"


def _to_dot_recurse(dp, lines: List[str], counter: List[int],
                    parent: Optional[str], edge_label: Optional[str]) -> str:
    """Emit dot for ``dp``, returning the id of its main node."""
    # System-built DP case: a Loop carrying _codesign_modules
    modules = getattr(dp, "_codesign_modules", None)
    constraints = getattr(dp, "_codesign_constraints", None)
    if modules:
        cluster_id = _node_id("cluster_sys", counter)
        lines.append(f"  subgraph {cluster_id} {{")
        lines.append('    label="System"; style="rounded,dashed"; color="#888888";')
        node_ids: dict = {}
        for mod_name, mod in modules.items():
            nid = _node_id("mod", counter)
            label = _module_label(mod, mod_name).replace('"', r'\"')
            lines.append(f'    {nid} [label="{label}", fillcolor="#e8f0fe"];')
            node_ids[mod_name] = nid
        # Constraint edges, if available.
        if constraints:
            for target, _fn, expr in constraints:
                # target is something like "module.port" or "outer.port"
                lhs = target.split(".")[0] if "." in target else target
                if lhs not in node_ids:
                    continue
                # Sources: any module mentioned in the expression. Fall back
                # to a single arrow with no source if expression structure
                # isn't introspectable.
                rhs_label = expr.pretty() if expr is not None else "<lambda>"
                rhs_label = _short_label(rhs_label).replace('"', r'\"')
                src_modules = _extract_module_refs(expr)
                if src_modules:
                    for src_mod in src_modules:
                        if src_mod in node_ids:
                            lines.append(
                                f"    {node_ids[src_mod]} -> {node_ids[lhs]} "
                                f'[label="{rhs_label}", color="#555"];'
                            )
                else:
                    # Just label the LHS node with the constraint as a comment.
                    pass
        lines.append("  }")
        return cluster_id

    # Composition operators: introspect by class name.
    # Series/Parallel store their two children as ``dp1``/``dp2`` (see
    # codesign.composition); older code read ``left``/``right``, which do
    # not exist and raised AttributeError. Fall back to left/right just in
    # case a future variant uses those names.
    cls = type(dp).__name__
    if cls == "Series":
        left = getattr(dp, "dp1", None)
        if left is None:
            left = getattr(dp, "left", None)
        right = getattr(dp, "dp2", None)
        if right is None:
            right = getattr(dp, "right", None)
        n_left = _to_dot_recurse(left, lines, counter, parent=None, edge_label=None)
        n_right = _to_dot_recurse(right, lines, counter, parent=None, edge_label=None)
        lines.append(f'  {n_left} -> {n_right} [label="series"];')
        if parent is not None:
            suffix = f' [label="{edge_label}"]' if edge_label else ""
            lines.append(f"  {parent} -> {n_left}{suffix};")
        return n_left
    if cls == "Parallel":
        left = getattr(dp, "dp1", None)
        if left is None:
            left = getattr(dp, "left", None)
        right = getattr(dp, "dp2", None)
        if right is None:
            right = getattr(dp, "right", None)
        n_left = _to_dot_recurse(left, lines, counter, parent=None, edge_label=None)
        n_right = _to_dot_recurse(right, lines, counter, parent=None, edge_label=None)
        # No edge between them, but bracket them with a synthetic node:
        bracket = _node_id("par", counter)
        lines.append(f'  {bracket} [label="parallel", shape=plaintext];')
        lines.append(f"  {bracket} -> {n_left} [style=dotted];")
        lines.append(f"  {bracket} -> {n_right} [style=dotted];")
        if parent is not None:
            lines.append(f"  {parent} -> {bracket};")
        return bracket
    if cls == "Loop":
        n_inner = _to_dot_recurse(dp.inner, lines, counter, parent=None, edge_label=None)
        # Self-loop on the axis.
        axis = getattr(dp, "axis", "")
        lines.append(
            f'  {n_inner} -> {n_inner} '
            f'[label="loop on {axis}", color="#cc3333", style=dashed];'
        )
        if parent is not None:
            lines.append(f"  {parent} -> {n_inner};")
        return n_inner

    # Leaf DP.
    leaf = _node_id("leaf", counter)
    label = _module_label(dp).replace('"', r'\"')
    lines.append(f'  {leaf} [label="{label}"];')
    return leaf


def _extract_module_refs(expr) -> List[str]:
    """Walk an Expr tree and collect any module names referenced as ports."""
    if expr is None:
        return []
    names: List[str] = []

    def walk(e):
        # Port handles carry .kind and ._module_name when from a module.
        kind = getattr(e, "kind", None)
        if kind in ("module_f", "module_r"):
            mod = getattr(e, "_module_name", None) or getattr(e, "module_name", None)
            if mod and mod not in names:
                names.append(mod)
            return
        # BinOp children: .left and .right
        left = getattr(e, "left", None)
        right = getattr(e, "right", None)
        if left is not None or right is not None:
            if left is not None:
                walk(left)
            if right is not None:
                walk(right)
            return
        # UnaryOp / function: .arg or .args
        arg = getattr(e, "arg", None)
        if arg is not None:
            walk(arg)
            return
        args = getattr(e, "args", None) or getattr(e, "children", None)
        if args:
            for c in args:
                walk(c)

    try:
        walk(expr)
    except Exception:
        return []
    return names
