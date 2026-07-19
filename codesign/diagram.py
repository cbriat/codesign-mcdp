"""
Block-diagram rendering for codesign Systems.

This module turns an MCDP :class:`~codesign.system.System` into a
GraphViz block diagram of the kind a process engineer or controls
engineer expects to see when reading a Simulink or SuperPro Designer
flowsheet:

- One box per subsystem, with its F ports (functionalities, inputs)
  listed on the left and R ports (resources, outputs) listed on the
  right. The box header carries the subsystem's name and class.
- Outer F and outer R appear as separate boxes on the left and right
  margins of the diagram respectively, so the diagram has visible
  inputs and outputs.
- Constraint wiring is rendered as port-to-port edges whenever the
  constraint was written in the operator-overloaded form
  (``module1.r_port >= module2.r_port * ...``); the connection
  resolves to the individual ports rather than just the modules.
- Lambda-based constraints (where the demand callable cannot be
  introspected) are rendered with a dashed edge from a small "lambda"
  marker, so the diagram does not silently omit them.
- Strongly-connected components of size > 1 are detected and the
  edges within them are coloured red, so the Kleene-iteration cycle
  is visible at a glance.

The output is a :class:`graphviz.Digraph` object that can be rendered
to SVG / PDF / PNG via its ``.render(format=...)`` method, or
displayed inline in a Jupyter notebook (Digraph implements
``_repr_svg_``). The graphviz Python package and the ``dot`` binary
are both required at render time; both are widely available
(``pip install graphviz`` plus ``apt-get install graphviz`` or
equivalent on the host system).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Visual palette (matched to the report and deck)
# ---------------------------------------------------------------------------

_NAVY   = "#0F2A44"  # outer R, headers
_TEAL   = "#0E7C7B"  # module accent, outer F
_TEAL2  = "#17A2B8"
_OFF    = "#F8FAFB"
_PORT   = "#FFFFFF"
_BORDER = "#D1D5DB"
_TEXT   = "#1F2937"
_CYCLE  = "#B45309"  # cycle edges, amber-rust


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_system(system, *, name: Optional[str] = None,
                rankdir: str = "LR",
                show_ports: bool = True,
                highlight_cycles: bool = True,
                graph_attrs: Optional[Mapping[str, str]] = None):
    """Render a :class:`System` as a graphviz block diagram.

    Parameters
    ----------
    system : codesign.System or a built System DP
        The model to render. Accepts either a live ``System`` (not yet
        built), or the ``DesignProblem`` returned by ``System.build()``
        (which carries the same modules and constraints on its
        ``_codesign_modules`` / ``_codesign_constraints`` attributes).
    name : str, optional
        Diagram title; defaults to ``system.name`` when available.
    rankdir : str, optional
        GraphViz rank direction. ``"LR"`` (default) gives a horizontal
        left-to-right flow that matches the way functionality
        propagates inward and resources outward. ``"TB"`` is useful
        for tall systems with many modules.
    show_ports : bool, optional
        Render the F and R port names inside each module box
        (default). When ``False``, only the module name is shown and
        edges attach to the module body; useful for high-level
        architecture views.
    highlight_cycles : bool, optional
        Detect strongly-connected components of size > 1 in the
        module-level constraint graph and colour the edges within
        them. Off-cycle edges are drawn in muted grey.
    graph_attrs : dict, optional
        Extra GraphViz graph attributes (e.g. ``{"ranksep": "0.6"}``).

    Returns
    -------
    graphviz.Digraph
        The rendered diagram. Call ``.render(filename, format="svg")``
        to write to disk, or display in a Jupyter notebook (the
        ``Digraph`` object implements ``_repr_svg_``).
    """
    try:
        import graphviz
    except ImportError as e:
        raise ImportError(
            "codesign.diagram requires the graphviz Python package. "
            "Install with `pip install graphviz`. The `dot` binary "
            "must also be on PATH (apt-get install graphviz, brew "
            "install graphviz, or equivalent)."
        ) from e

    spec = _extract_spec(system)
    diagram_name = name or spec.name or "system"

    dot = graphviz.Digraph(name=diagram_name, format="svg")
    dot.attr(rankdir=rankdir, bgcolor=_OFF,
             fontname="Helvetica", fontsize="11",
             labelloc="t", label=diagram_name, fontcolor=_NAVY,
             splines="spline", nodesep="0.4", ranksep="0.7")
    if graph_attrs:
        dot.attr(**dict(graph_attrs))
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9",
             color="#555555", arrowsize="0.7")

    # ---- Cycle detection on the module-level constraint graph ---------
    cycle_modules: Set[str] = set()
    if highlight_cycles:
        cycle_modules = _find_cycle_modules(spec)

    # ---- Outer F nodes (inputs on the left) ---------------------------
    for fname, _poset in spec.outer_F.items():
        nid = f"outer_f__{fname}"
        unit = _unit_of(_poset)
        label_text = fname if not unit else f"{fname}\n[{unit}]"
        dot.node(nid, label=label_text,
                 shape="ellipse", style="filled",
                 fillcolor=_TEAL, fontcolor=_PORT,
                 color=_TEAL, penwidth="1")

    # ---- Module nodes with port-level labels --------------------------
    for mod_name, mod in spec.modules.items():
        label = _module_html_label(mod_name, mod, show_ports=show_ports)
        in_cycle = mod_name in cycle_modules
        accent = _CYCLE if in_cycle else _TEAL
        dot.node(mod_name, label=f"<{label}>",
                 shape="plain", color=accent)

    # ---- Outer R nodes (outputs on the right) -------------------------
    for rname, _poset in spec.outer_R.items():
        nid = f"outer_r__{rname}"
        unit = _unit_of(_poset)
        label_text = rname if not unit else f"{rname}\n[{unit}]"
        dot.node(nid, label=label_text,
                 shape="ellipse", style="filled",
                 fillcolor=_NAVY, fontcolor=_PORT,
                 color=_NAVY, penwidth="1")

    # ---- Edges from constraints ---------------------------------------
    # A lambda marker is added lazily, only if needed.
    _lambda_marker_added = [False]

    def _ensure_lambda_marker():
        if not _lambda_marker_added[0]:
            dot.node("lambda_marker",
                     label="λ", shape="circle",
                     fillcolor=_OFF, color=_BORDER,
                     style="filled,dashed",
                     fontcolor=_TEXT, fontsize="14",
                     width="0.3", height="0.3", fixedsize="true")
            _lambda_marker_added[0] = True

    for target, _fn, expr in spec.constraints:
        # Destination node id and port anchor.
        if "." in target:
            dst_mod, dst_port = target.split(".", 1)
            if dst_mod not in spec.modules:
                continue
            dst_id = dst_mod
            dst_anchor = f":f_{_sanitize(dst_port)}" if show_ports else ""
        else:
            # Outer R.
            if target not in spec.outer_R:
                continue
            dst_id = f"outer_r__{target}"
            dst_anchor = ""

        # Source ports: walk the expression tree to find Port leaves.
        src_ports = _collect_port_refs(expr)
        if not src_ports and expr is None:
            # Lambda-based constraint: attach a single edge from the
            # marker node so the user sees it exists.
            _ensure_lambda_marker()
            edge_color = _CYCLE if _edge_in_cycle(
                "lambda_marker", target, spec, cycle_modules) else "#777777"
            dot.edge(f"lambda_marker", f"{dst_id}{dst_anchor}",
                     style="dashed", color=edge_color)
            continue

        # Deduplicate (src, src_port) pairs so that an expression that
        # references the same port twice does not produce a double edge.
        seen: Set[Tuple[str, str]] = set()
        for p in src_ports:
            key = (p.module or "__outer__", p.name)
            if key in seen:
                continue
            seen.add(key)

            if p.kind == "outer_f":
                src_id = f"outer_f__{p.name}"
                src_anchor = ""
            elif p.kind == "module_r":
                src_id = p.module
                src_anchor = f":r_{_sanitize(p.name)}" if show_ports else ""
            elif p.kind == "module_f":
                # Unusual: a module F port appearing in another module's
                # demand. Treat as the module's input.
                src_id = p.module
                src_anchor = f":f_{_sanitize(p.name)}" if show_ports else ""
            elif p.kind == "outer_r":
                # Should not happen on the RHS; skip.
                continue
            else:
                continue

            # Determine if this edge is inside the cycle.
            edge_in_cycle = (
                p.module in cycle_modules
                and target.split(".", 1)[0] in cycle_modules
            )
            edge_color = _CYCLE if edge_in_cycle else "#555555"
            edge_pen = "1.5" if edge_in_cycle else "1.0"

            dot.edge(f"{src_id}{src_anchor}",
                     f"{dst_id}{dst_anchor}",
                     color=edge_color, penwidth=edge_pen)

    return dot


# ---------------------------------------------------------------------------
# Internal extraction: take a System or built-System DP and pull out
# a uniform "spec" carrying the data needed to render it.
# ---------------------------------------------------------------------------


class _Spec:
    __slots__ = ("name", "modules", "outer_F", "outer_R", "constraints")

    def __init__(self, name, modules, outer_F, outer_R, constraints):
        self.name = name
        self.modules = modules            # dict name -> DesignProblem
        self.outer_F = outer_F            # dict name -> Poset
        self.outer_R = outer_R            # dict name -> Poset
        self.constraints = constraints    # list of (target, fn, expr)


def _extract_spec(system) -> _Spec:
    """Pull a uniform model spec from a live System or a built DP."""
    if hasattr(system, "_modules") and hasattr(system, "_constraints"):
        # Live System (not yet built).
        return _Spec(
            name=getattr(system, "name", None),
            modules=dict(system._modules),
            outer_F=dict(system._outer_F),
            outer_R=dict(system._outer_R),
            constraints=list(system._constraints),
        )
    # Built DP from System.build(): looks like a Loop with extra
    # _codesign_* attributes.
    modules = getattr(system, "_codesign_modules", None)
    constraints = getattr(system, "_codesign_constraints", None)
    if modules is None or constraints is None:
        raise TypeError(
            "draw_system expects a System or a built System DP; got "
            f"{type(system).__name__} with no _codesign_modules"
        )
    # Outer F/R aren't preserved on the built DP, but they are the
    # outer F/R of the Loop's inner DP minus the modules axis.
    outer_F: Dict[str, Any] = {}
    outer_R: Dict[str, Any] = {}
    inner_F = getattr(getattr(system, "F", None), "components", {})
    inner_R = getattr(getattr(system, "R", None), "components", {})
    for k, p in inner_F.items():
        outer_F[k] = p
    for k, p in inner_R.items():
        outer_R[k] = p
    return _Spec(
        name=getattr(system, "name", None),
        modules=dict(modules),
        outer_F=outer_F,
        outer_R=outer_R,
        constraints=list(constraints),
    )


# ---------------------------------------------------------------------------
# Cycle detection: Tarjan's strongly-connected-components on the
# module-level graph induced by constraint sources/targets.
# ---------------------------------------------------------------------------


def _find_cycle_modules(spec: _Spec) -> Set[str]:
    """Return the set of module names that participate in any cycle."""
    # Build adjacency: edge src_mod -> dst_mod whenever a constraint
    # has dst on dst_mod and src is a module port of src_mod.
    adj: Dict[str, Set[str]] = {m: set() for m in spec.modules}
    for target, _fn, expr in spec.constraints:
        if "." not in target:
            continue
        dst_mod = target.split(".", 1)[0]
        if dst_mod not in spec.modules:
            continue
        for p in _collect_port_refs(expr):
            if p.kind in ("module_r", "module_f") and p.module in spec.modules:
                if p.module != dst_mod:
                    adj.setdefault(p.module, set()).add(dst_mod)

    # Tarjan's SCC.
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    sccs: List[Set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp: Set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in adj:
        if v not in index:
            strongconnect(v)

    out: Set[str] = set()
    for comp in sccs:
        if len(comp) > 1:
            out.update(comp)
    return out


def _edge_in_cycle(src: str, target: str, spec: _Spec,
                   cycle_modules: Set[str]) -> bool:
    if "." not in target:
        return False
    dst_mod = target.split(".", 1)[0]
    return src in cycle_modules and dst_mod in cycle_modules


# ---------------------------------------------------------------------------
# Expression walking: collect Port leaves
# ---------------------------------------------------------------------------


class _PortRef:
    __slots__ = ("module", "name", "kind")

    def __init__(self, module: Optional[str], name: str, kind: str):
        self.module = module
        self.name = name
        self.kind = kind

    def __repr__(self):
        return f"PortRef({self.module!r}, {self.name!r}, {self.kind!r})"


def _collect_port_refs(expr) -> List[_PortRef]:
    """Walk an Expr tree and return all Port references encountered."""
    if expr is None:
        return []
    out: List[_PortRef] = []

    def walk(e: Any) -> None:
        kind = getattr(e, "kind", None) or getattr(e, "_kind", None)
        if kind in ("module_f", "module_r", "outer_f", "outer_r"):
            mod = getattr(e, "_module_name", None)
            port = getattr(e, "_port_name", None) or getattr(e, "name", None)
            if port is not None:
                out.append(_PortRef(mod, port, kind))
            return
        # Walk children. Handles _BinOp (left/right), Neg (inner),
        # Func (inner), plus future operators with the same shape.
        for attr in ("left", "right", "inner", "arg"):
            child = getattr(e, attr, None)
            if child is not None:
                walk(child)

    walk(expr)
    return out


# ---------------------------------------------------------------------------
# Module label rendering: HTML-like table with F left, R right.
# ---------------------------------------------------------------------------


def _module_html_label(mod_name: str, mod, *, show_ports: bool) -> str:
    """Build a GraphViz HTML-like label for a module box.

    The label is a small two-cell table: title bar across the top
    (module name and class), then one row with two cells, the left
    cell listing F ports, the right cell listing R ports. Each port
    is wrapped in its own ``<TD PORT="...">`` so that edges can
    attach to specific ports rather than the box as a whole.
    """
    cls_name = type(mod).__name__
    title = f"{mod_name}  <FONT POINT-SIZE=\"9\" COLOR=\"#6B7280\">({cls_name})</FONT>"

    if not show_ports:
        # Compact view: just the module name.
        return (
            f'<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
            f'CELLPADDING="6" BGCOLOR="white">'
            f'<TR><TD BGCOLOR="{_TEAL}"><FONT COLOR="white"><B>{title}</B></FONT></TD></TR>'
            f'</TABLE>'
        )

    f_ports = list(getattr(mod.F, "components", {}).keys()) \
              if hasattr(mod.F, "components") else []
    r_ports = list(getattr(mod.R, "components", {}).keys()) \
              if hasattr(mod.R, "components") else []

    def port_rows(ports: List[str], prefix: str, side: str) -> str:
        if not ports:
            return (f'<TR><TD ALIGN="{side}"><FONT COLOR="#9CA3AF">'
                    f'<I>none</I></FONT></TD></TR>')
        rows = []
        for p in ports:
            anchor = f"{prefix}_{_sanitize(p)}"
            color = "#0E7C7B" if prefix == "f" else "#0F2A44"
            rows.append(
                f'<TR><TD PORT="{anchor}" ALIGN="{side}" '
                f'CELLPADDING="3" BGCOLOR="white">'
                f'<FONT COLOR="{color}" POINT-SIZE="9">{p}</FONT></TD></TR>'
            )
        return "".join(rows)

    f_label = ('<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
               'CELLPADDING="0" BGCOLOR="white">'
               '<TR><TD BGCOLOR="#E8F4F3" ALIGN="left">'
               '<FONT COLOR="#0E7C7B" POINT-SIZE="8"><B>F</B></FONT></TD></TR>'
               + port_rows(f_ports, "f", "left") +
               '</TABLE>')
    r_label = ('<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
               'CELLPADDING="0" BGCOLOR="white">'
               '<TR><TD BGCOLOR="#E8EEF5" ALIGN="right">'
               '<FONT COLOR="#0F2A44" POINT-SIZE="8"><B>R</B></FONT></TD></TR>'
               + port_rows(r_ports, "r", "right") +
               '</TABLE>')

    return (
        f'<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
        f'CELLPADDING="6" BGCOLOR="white">'
        f'<TR><TD COLSPAN="2" BGCOLOR="{_TEAL}">'
        f'<FONT COLOR="white"><B>{title}</B></FONT></TD></TR>'
        f'<TR><TD CELLPADDING="0">{f_label}</TD>'
        f'<TD CELLPADDING="0">{r_label}</TD></TR>'
        f'</TABLE>'
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _unit_of(poset: Any) -> str:
    """Best-effort extraction of a unit string from a Poset."""
    return getattr(poset, "unit", "") or ""


def _sanitize(s: str) -> str:
    """Convert a port name into a GraphViz-safe port anchor."""
    return "".join(ch if ch.isalnum() else "_" for ch in s)
