"""
Example 25: replicating the synthetic benchmarks of the online co-design paper.

Paper
-----
Meshal Alharbi, Munther A. Dahleh & Gioele Zardini, "Compositional Online
Learning for Multi-Objective System Co-Design" (arXiv:2604.22624, 2026).

The paper studies online multi-objective decision-making in *monotone
co-design*: functionalities and resources are partially ordered, and the
agent must recover the target-feasible antichain of non-dominated resources
using as few expensive evaluations as possible. Its engine is *Algorithm 1*
(Rejection Sampler with Optimistic Evaluators): draw a candidate from a
low-discrepancy base measure, compute history-dependent *optimistic* bounds
on its resource and functionality, and skip (reject) it without evaluation
when either

  (13)  req_opt(i, H) in up(Anti_f(H))      -- optimistic resource already
                                                dominated by the incumbent, or
  (14)  not (f <= prov_opt(i, H))            -- optimistic functionality can
                                                never meet the target f.

Only survivors are actually queried.

What this example replicates
----------------------------
The two *synthetic* benchmark families of Section VII-B, which isolate the
value of structural optimism. Each is a resource map ``g : [0,1]^d ->
[0,1]^m`` (``d=3, m=2`` monotone; ``d=4, m=2`` Lipschitz), and the object to
recover is the non-dominated (minimal) antichain of resources.

- **E1 / Monotone** (Table I): ``g_j(x) = sum_k w_{j,k} 1{x >= t_k}`` with
  thresholds ``t_k ~ U([0,1]^3)`` and non-negative per-output-normalized
  weights (``sum_k w_{j,k} = 1``). Every ``g_j`` is monotone; the optimistic
  evaluator is the monotone join bound of eqs. (23)-(24)
  (``codesign.online.MonotonicityEvaluator``).
- **E2 / Lipschitz** (Table II): ``g_j(x) = phi([Ax+b]_j)`` with
  ``||A||_2 = L = 2`` and ``phi`` the unit-slope triangle wave
  (``|phi'| = 1``), so ``g`` is ``L``-Lipschitz in the l2 norm. The
  optimistic evaluator is the Lipschitz cone bound of eq. (25)
  (``codesign.online.LipschitzEvaluator`` with ``L = 2``).

We use the library's evaluators for the optimistic bounds; their ``bound()``
lower value reproduces eqs. (23)-(25) exactly (the monotone bound is the
component-wise max over queried predecessors ``x_j <= x_i`` = the join of
eq. 24; the Lipschitz bound is ``max_j max(0, r_k(j) - L*||x_i - x_j||_2)``
= the join of eq. 25 clipped at the bottom). The *upper* value of the same
``bound()`` supplies ``prov_opt`` for condition (14). Algorithm 1 itself --
the Halton base measure, the two elimination conditions, and the forced-
acceptance knob ``delta`` -- is implemented here, because the library's
``solve_online`` *scans* a candidate list with an upper-confidence picker
whereas the paper *samples* a base measure with rejection.

Deviation from the paper's literal text, and why (be honest)
------------------------------------------------------------
The paper says the target functionality is "fixed and satisfied by
construction, so the learning problem reduces to recovering the non-
dominated antichain of a resource map g". Taken literally this is
degenerate for the MONOTONE family: a monotone ``g`` is minimized at the
bottom of the box, so its unconstrained minimal antichain collapses to a
single point ``g(0) = (0,0)`` -- reachable by ~half the domain -- and there
is nothing to learn (elimination never fires; every method converges in one
step). The LIPSCHITZ family has no such problem because the triangle-wave
``g`` is non-monotone, so its minimal antichain is a genuine 2-D trade-off.

We therefore keep the Lipschitz benchmark as the literal unconstrained
minimization, and for the monotone benchmark make the functionality target
*binding*: we recover ``FixFunMinRes(f)`` = the minimal antichain of ``g``
restricted to the feasible upper set ``{x : g(x) >= f}`` (here ``prov = req
= g``, the single expensive 2-D map, and ``f`` is a fixed fraction of the
per-coordinate range, so the feasible region is non-empty -- "satisfied by
construction"). This is the standard co-design query, it exercises BOTH
elimination conditions (13) and (14), and it makes the monotone benchmark
non-degenerate. This is the minimal deviation needed; it is documented here
and does not affect the Lipschitz family.

What is NOT replicated
----------------------
- **E3 / Intermodal mobility** (Sec. VII-C1, Tables III-IV, Fig. 4): the
  expensive block is a large multi-commodity-flow LP from Zardini, Lanzetti,
  Censi, Frazzoli & Pavone, *IEEE TNSE* 2022 [ref 39]; that model is not in
  this repository.
- **E4 / Heterogeneous multi-robot** (Sec. VII-C2, Fig. 5): the expensive
  block is a planner-executor physics simulation from Stralz, Alharbi, Huang
  & Zardini, arXiv:2604.21894 [ref 10]; that simulator is not in this
  repository.
There is no public code/data release for the paper and the exact seeds,
discretization grid, and atom count ``K`` are unpublished, so this
replication is *statistical, not bit-exact*: the precise table entries
cannot be reproduced. We validate the paper's reproducible *qualitative*
claims (VALIDATION below) and print PASS/FAIL, quoting the paper's numbers
in comments.

Metric
------
Cumulative hypervolume difference (lower is better), summed over iterations
as in the paper's tables: ``sum_{t=1}^N [ HV(A_ref) - HV(A_hat_t) ]`` where
``A_hat_t`` is the incumbent antichain after ``t`` expensive queries, ``HV``
is the exact 2-D hypervolume w.r.t. reference point ``(1,1)``, and ``A_ref``
is the minimal antichain of the full feasible candidate pool (the best any
method can reach, so HVD >= 0 and HVD -> 0 on convergence). A queried
implementation that fails the target is assigned the worst resource (paper's
"target functionality" adaptation), i.e. it contributes nothing to the
incumbent. Our absolute numbers differ from the paper's (different
reference/normalization, grid, K, seeds); only the relative claims matter.

Baselines
---------
- **Halton**: the same Halton proposals as Ours but no elimination.
- **Random**: uniform draws from the shared discretized pool.
- **EA panel** (NSGA-III / MOEA/D / RVEA via pymoo): optional, gated behind
  ``import pymoo``; skipped with a one-line notice if pymoo is absent
  (``pip install pymoo`` to enable). pymoo is intentionally NOT a project
  dependency.

Run
---
    python -m examples.25_online_paper_benchmarks

Expected output: two Table-I / Table-II-style panels (per-instance mean
cumulative HVD for Ours / Halton / Random, plus EAs if pymoo is installed),
PASS/FAIL verdicts for the reproducible claims, and two figures saved to
``outputs/`` (mean HVD vs. iteration, log-y). Defaults are trimmed for a
< ~2-3 min run; module-level constants below crank up to paper scale.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from codesign import Ports, Reals
from codesign.antichains import Antichain
from codesign.online import LipschitzEvaluator, MonotonicityEvaluator

# ===========================================================================
# Scale constants.  Defaults are trimmed for a quick (< ~2-3 min) run.
# For a faithful (still statistical, not bit-exact) replication set:
#   RUNS = 100, ITERS_MONO = 4000, ITERS_LIP = 2000
# exactly as in Tables I and II (this takes substantially longer).
# ===========================================================================
N_INSTANCES = 8            # M1..M8 / L1..L8 in the paper (fixed at 8)
RUNS = 5                   # paper: 100 independent runs, averaged
ITERS_MONO = 250           # paper Table I: 4000 iterations
ITERS_LIP = 250            # paper Table II: 2000 iterations
POOL_FACTOR = 8            # discretized pool size = POOL_FACTOR * budget
K_ATOMS = 15               # monotone step atoms per output (K is UNPUBLISHED)
LIPSCHITZ_L = 2.0          # paper: ||A||_2 = L = 2
MONO_TARGET_ALPHA = 0.35   # monotone target f = alpha * per-coord range of g
DELTA = 0.05               # paper Sec. VII-B: forced-acceptance prob. = 0.05
REF_POINT = (1.0, 1.0)     # hypervolume reference (resources in [0,1]^2)
BASE_SEED = 20260424       # arXiv id date; deterministic instances/runs

# Paper reference numbers (used in the validation comments below):
#   Table I  (Monotone, OURS row):  8.81e1 .. 3.46e1  (best on 7/8 instances)
#   Table II (Lipschitz, OURS row): 5.89e1 .. 4.20e1  (best on 5/8, 2nd on 2/3)

RC = ("r0", "r1")
_R_POSET = Ports({"r0": Reals(), "r1": Reals()})


# ===========================================================================
# Exact 2-D hypervolume and Pareto machinery (minimization).
# ===========================================================================


def pareto_front(points: Sequence[Tuple[float, float]]
                 ) -> List[Tuple[float, float]]:
    """Non-dominated (minimization) subset, as an antichain. O(n log n)."""
    if not points:
        return []
    pts = sorted(set(points))
    front: List[Tuple[float, float]] = []
    best_y = math.inf
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def pareto_insert(front: List[Tuple[float, float]],
                  r: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Insert resource point ``r`` into a minimization antichain."""
    x, y = r
    for px, py in front:
        if px <= x and py <= y:
            return front  # r is dominated (or a duplicate)
    kept = [(px, py) for px, py in front if not (x <= px and y <= py)]
    kept.append((x, y))
    return kept


def hv2d(front: Sequence[Tuple[float, float]],
         ref: Tuple[float, float]) -> float:
    """Exact 2-D hypervolume of a minimization antichain w.r.t. ``ref``.

    HV = Lebesgue measure of { r | exists a in front: a <= r <= ref }.
    Sort the antichain by x ascending (y then descends) and sum vertical
    strips: strip [x_i, x_{i+1}) has height (ref_y - y_i).
    """
    rx, ry = ref
    pts = [(x, y) for x, y in front if x <= rx and y <= ry]
    if not pts:
        return 0.0
    pts = pareto_front(pts)
    pts.sort(key=lambda p: p[0])
    hv = 0.0
    n = len(pts)
    for i in range(n):
        x, y = pts[i]
        next_x = pts[i + 1][0] if i + 1 < n else rx
        hv += (next_x - x) * (ry - y)
    return hv


def _dominated_by_incumbent(lo: Dict[str, float],
                            front: Sequence[Tuple[float, float]]) -> bool:
    """Condition (13): optimistic lower bound ``lo`` in up(Anti_f(H)).

    Mirrors ``codesign.online._is_dominated_by_incumbent`` against our
    plain (x, y) incumbent front.
    """
    lx, ly = lo[RC[0]], lo[RC[1]]
    for px, py in front:
        if px <= lx and py <= ly:
            return True
    return False


# ===========================================================================
# Generative map families (exactly per Section VII-B).
# ===========================================================================


def make_monotone_map(seed: int, k_atoms: int = K_ATOMS
                      ) -> Callable[[np.ndarray], np.ndarray]:
    """E1: g_j(x) = sum_k w_{j,k} 1{x >= t_k}, t_k ~ U([0,1]^3), w >= 0,
    sum_k w_{j,k} = 1.  Monotone nondecreasing, image in [0,1]^2."""
    rng = np.random.default_rng(seed)
    d, m = 3, 2
    thresholds = rng.random((k_atoms, d))
    weights = rng.random((m, k_atoms))
    weights /= weights.sum(axis=1, keepdims=True)

    def g(X: np.ndarray) -> np.ndarray:
        ind = np.all(X[:, None, :] >= thresholds[None, :, :], axis=2)
        return ind.astype(float) @ weights.T

    return g


def _triangle_wave(t: np.ndarray) -> np.ndarray:
    """Unit-slope triangle wave phi(t) in [0,1], |phi'| = 1 a.e."""
    mod = np.mod(t, 2.0)
    return np.where(mod <= 1.0, mod, 2.0 - mod)


def make_lipschitz_map(seed: int, L: float = LIPSCHITZ_L
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """E2: g_j(x) = phi([Ax + b]_j) with ||A||_2 = L, phi triangle wave.
    L-Lipschitz in l2, image in [0,1]^2."""
    rng = np.random.default_rng(seed)
    d, m = 4, 2
    A = rng.standard_normal((m, d))
    A *= L / np.linalg.norm(A, 2)
    b = rng.uniform(0.0, 2.0, size=m)

    def g(X: np.ndarray) -> np.ndarray:
        return _triangle_wave(X @ A.T + b[None, :])

    return g


def monotone_target(g: Callable[[np.ndarray], np.ndarray], d: int,
                    alpha: float) -> np.ndarray:
    """Fixed functionality target f = alpha * (per-coordinate range of g),
    computed on a deterministic reference sample so f is run-independent."""
    sample = g(halton_pool(d, 5000, seed=BASE_SEED + 99))
    return alpha * sample.max(axis=0)


# ===========================================================================
# Halton base measure over the discretized candidate pool.
# ===========================================================================


def halton_pool(d: int, size: int, seed: int) -> np.ndarray:
    """Scrambled Halton low-discrepancy sample in [0,1]^d (the base measure
    mu and the shared discretized candidate grid)."""
    return qmc.Halton(d=d, scramble=True, seed=seed).random(size)


# ===========================================================================
# Method runners.  Each returns (cumulative_HVD, per-iteration HVD curve).
# ===========================================================================


def _singleton_antichain(r: Tuple[float, float]) -> Antichain:
    return Antichain(_R_POSET, [{"r0": float(r[0]), "r1": float(r[1])}])


def run_ours(pool: np.ndarray, res: np.ndarray, feas: np.ndarray,
             cand: List[Dict[str, float]], evaluator,
             f: Optional[np.ndarray], budget: int, hv_star: float,
             ref: Tuple[float, float], delta: float,
             rng: np.random.Generator) -> Tuple[float, List[float]]:
    """Paper Algorithm 1: rejection sampler with optimistic evaluators.

    Draw proposals from the Halton pool in low-discrepancy order; reject a
    proposal whose optimistic functionality can never meet f (condition 14)
    or whose optimistic resource is already dominated (condition 13). With
    probability ``delta`` force-accept (forced exploration, guarding against
    an erroneous certificate). Only accepted proposals consume the budget.
    """
    evaluator.reset()
    front: List[Tuple[float, float]] = []
    cum = 0.0
    curve: List[float] = []
    ptr = 0
    accepted = 0
    constrained = f is not None
    while accepted < budget and ptr < len(pool):
        i = ptr
        ptr += 1
        if front and rng.random() >= delta:
            lo, hi = evaluator.bound(cand[i])
            if constrained and (hi[RC[0]] < f[0] or hi[RC[1]] < f[1]):
                continue  # (14): optimistic functionality cannot reach target
            if _dominated_by_incumbent(lo, front):
                continue  # (13): optimistic resource already dominated
        # --- accept: query the expensive block ---
        r = (float(res[i, 0]), float(res[i, 1]))
        evaluator.observe(i, cand[i], _singleton_antichain(r))
        accepted += 1
        if feas[i]:
            front = pareto_insert(front, r)
        hvd = hv_star - hv2d(front, ref)
        cum += hvd
        curve.append(hvd)
    # Defensive padding if the pool was exhausted before budget (POOL_FACTOR
    # is chosen so this does not happen at default scale).
    while len(curve) < budget:
        pad = curve[-1] if curve else hv_star
        cum += pad
        curve.append(pad)
    return cum, curve


def run_ordered(res: np.ndarray, feas: np.ndarray, order: np.ndarray,
                budget: int, hv_star: float, ref: Tuple[float, float]
                ) -> Tuple[float, List[float]]:
    """Baseline with no elimination: query ``order[:budget]`` in sequence.

    ``order = arange(budget)`` gives the Halton baseline (native low-
    discrepancy order); a random permutation gives the uniform-random one.
    Infeasible queries contribute nothing (worst-resource assignment).
    """
    front: List[Tuple[float, float]] = []
    cum = 0.0
    curve: List[float] = []
    for t in range(budget):
        i = int(order[t])
        if feas[i]:
            front = pareto_insert(front, (float(res[i, 0]), float(res[i, 1])))
        hvd = hv_star - hv2d(front, ref)
        cum += hvd
        curve.append(hvd)
    return cum, curve


# ===========================================================================
# Optional EA panel (pymoo).  Gated; skipped cleanly if pymoo is absent.
# ===========================================================================

_EA_SKIP_NOTICE_SHOWN = False
EA_NAMES = ("NSGA-III", "MOEA/D", "RVEA")


def run_ea_panel(g: Callable[[np.ndarray], np.ndarray], d: int,
                 f: Optional[np.ndarray], budget: int, hv_star: float,
                 ref: Tuple[float, float], seed: int
                 ) -> Optional[Dict[str, Tuple[float, List[float]]]]:
    """Run the pymoo EA panel over the continuous [0,1]^d box, minimizing g.

    Returns {name: (cum_HVD, curve)} or None if pymoo is unavailable. The
    cumulative archive is scored exactly like Ours (paper's "cumulative
    evaluation"); infeasible points (g not >= f) contribute nothing. HVD is
    clipped at 0 because a continuous EA may reach points slightly better
    than the discretized pool's reference front.

    NOTE: this path runs only when pymoo is installed; pymoo is not a
    project dependency and is absent in the reference test environment.
    """
    global _EA_SKIP_NOTICE_SHOWN
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.algorithms.moo.rvea import RVEA
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except Exception:
        if not _EA_SKIP_NOTICE_SHOWN:
            print("  [EA panel skipped: pymoo not installed "
                  "(`pip install pymoo` to enable NSGA-III / MOEA/D / RVEA)]")
            _EA_SKIP_NOTICE_SHOWN = True
        return None

    n_constr = 0 if f is None else 2

    class _GProblem(Problem):
        def __init__(self):
            super().__init__(n_var=d, n_obj=2, n_ieq_constr=n_constr,
                             xl=0.0, xu=1.0)
            self.trace: List[Tuple[float, float, bool]] = []

        def _evaluate(self, X, out, *args, **kwargs):
            F = g(np.atleast_2d(X))
            out["F"] = F
            if f is not None:
                out["G"] = f[None, :] - F  # <= 0 means feasible
            for row in F:
                ok = True if f is None else bool(np.all(row >= f))
                self.trace.append((float(row[0]), float(row[1]), ok))

    pop = 20
    n_gen = max(2, budget // pop)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop - 1)
    algos = {
        "NSGA-III": lambda: NSGA3(pop_size=pop, ref_dirs=ref_dirs),
        "MOEA/D": lambda: MOEAD(ref_dirs=ref_dirs, n_neighbors=min(15, pop)),
        "RVEA": lambda: RVEA(pop_size=pop, ref_dirs=ref_dirs),
    }
    results: Dict[str, Tuple[float, List[float]]] = {}
    for name, factory in algos.items():
        prob = _GProblem()
        try:
            minimize(prob, factory(), ("n_gen", n_gen), seed=seed, verbose=False)
        except Exception as exc:
            print(f"  [EA {name} failed: {exc}]")
            continue
        front: List[Tuple[float, float]] = []
        cum = 0.0
        curve: List[float] = []
        for x, y, ok in prob.trace[:budget]:
            if ok:
                front = pareto_insert(front, (x, y))
            hvd = max(0.0, hv_star - hv2d(front, ref))
            cum += hvd
            curve.append(hvd)
        while len(curve) < budget:
            pad = curve[-1] if curve else hv_star
            cum += pad
            curve.append(pad)
        results[name] = (cum, curve)
    return results


# ===========================================================================
# Experiment driver for one family.
# ===========================================================================


def run_family(family: str, make_map: Callable[[int], Callable], d: int,
               feature_names: List[str], make_evaluator: Callable,
               constrained: bool, budget: int, with_ea: bool
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run all instances x runs for one family.

    Returns (per_instance_means, mean_curves).
    """
    pool_size = POOL_FACTOR * budget
    methods = ["Ours", "Halton", "Random"]
    per_inst: Dict[str, List[List[float]]] = {m: [] for m in methods}
    curves: Dict[str, List[List[float]]] = {m: [] for m in methods}

    for inst in range(N_INSTANCES):
        g = make_map(BASE_SEED + inst)
        f = monotone_target(g, d, MONO_TARGET_ALPHA) if constrained else None
        inst_scores: Dict[str, List[float]] = {m: [] for m in methods}

        for run in range(RUNS):
            rng = np.random.default_rng(BASE_SEED + 977 * inst + run)
            pool = halton_pool(d, pool_size, seed=BASE_SEED + 31 * inst + run)
            res = g(pool)
            feas = (np.ones(len(pool), dtype=bool) if f is None
                    else np.all(res >= f[None, :], axis=1))
            ref = REF_POINT
            hv_star = hv2d(pareto_front(
                [(float(a), float(b)) for a, b in res[feas]]), ref)
            cand = [{name: float(pool[i, j])
                     for j, name in enumerate(feature_names)}
                    for i in range(len(pool))]

            evaluator = make_evaluator(feature_names)
            c_ours, cv_ours = run_ours(pool, res, feas, cand, evaluator, f,
                                       budget, hv_star, ref, DELTA, rng)
            c_hal, cv_hal = run_ordered(res, feas, np.arange(budget),
                                        budget, hv_star, ref)
            perm = rng.permutation(len(pool))
            c_rnd, cv_rnd = run_ordered(res, feas, perm, budget, hv_star, ref)

            for m, c, cv in (("Ours", c_ours, cv_ours),
                             ("Halton", c_hal, cv_hal),
                             ("Random", c_rnd, cv_rnd)):
                inst_scores[m].append(c)
                curves[m].append(cv)

            if with_ea:
                ea = run_ea_panel(g, d, f, budget, hv_star, ref,
                                  seed=BASE_SEED + inst * 100 + run)
                if ea:
                    for ea_name, (c, cv) in ea.items():
                        if ea_name not in per_inst:
                            per_inst[ea_name] = [[] for _ in range(inst)]
                            curves[ea_name] = []
                        inst_scores.setdefault(ea_name, []).append(c)
                        curves[ea_name].append(cv)

        for m in list(per_inst.keys()):
            per_inst[m].append(inst_scores.get(m, []))

    means = {m: np.array([np.mean(s) if s else np.nan for s in rows])
             for m, rows in per_inst.items()}
    mean_curves = {m: (np.mean(np.array(cv), axis=0) if cv else None)
                   for m, cv in curves.items()}
    return means, mean_curves


# ===========================================================================
# Reporting.
# ===========================================================================


def print_panel(title: str, table_ref: str, prefix: str,
                means: Dict[str, np.ndarray]) -> None:
    print(f"\n{title}")
    print(f"  (cumulative HVD, lower is better; {table_ref})")
    header = "  Method    " + "".join(
        f"{prefix + str(i + 1):>10}" for i in range(N_INSTANCES))
    print(header)
    order = ["Ours", "Halton", "Random"] + [n for n in EA_NAMES if n in means]
    for m in order:
        if m not in means:
            continue
        cells = "".join(f"{v:>10.2e}" if np.isfinite(v) else f"{'--':>10}"
                        for v in means[m])
        print(f"  {m:<9} {cells}")


def make_figure(family: str, mean_curves: Dict[str, np.ndarray],
                out_dir: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [figure skipped: matplotlib not installed]")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"Ours": ("C3", "-", 2.2), "Halton": ("C0", "--", 1.5),
              "Random": ("C7", ":", 1.5), "NSGA-III": ("C2", "-.", 1.2),
              "MOEA/D": ("C1", "-.", 1.2), "RVEA": ("C4", "-.", 1.2)}
    for m, curve in mean_curves.items():
        if curve is None:
            continue
        color, ls, lw = styles.get(m, ("C5", "-", 1.2))
        x = np.arange(1, len(curve) + 1)
        y = np.clip(curve, 1e-6, None)  # floor for log axis
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=m)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration (expensive evaluations)")
    ax.set_ylabel("Hypervolume difference  HV(A_ref) - HV(A_hat)")
    ax.set_title(f"Online co-design synthetic benchmark: {family}\n"
                 f"(mean over {N_INSTANCES} instances x {RUNS} runs)")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, f"25_online_{family.lower()}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ===========================================================================
# Validation of the paper's reproducible claims.
# ===========================================================================


def validate(mono: Dict[str, np.ndarray], lip: Dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 74)
    print("VALIDATION  (reproducible qualitative claims; absolute numbers are"
          " NOT\nbit-reproducible -- unpublished seeds / grid / K)")
    print("=" * 74)
    verdicts: List[bool] = []

    # Claim (a): Ours beats Halton on ALL monotone instances.
    #   Paper Table I OURS row 8.81e1..3.46e1 is best on 7/8 and below HALTON
    #   (3.48e2..1.82e2) on all 8 instances.
    a_wins = int(np.sum(mono["Ours"] < mono["Halton"]))
    a_pass = a_wins == N_INSTANCES
    verdicts.append(a_pass)
    print("\n(a) Ours < Halton on ALL monotone instances")
    print("    paper: Table I, OURS below HALTON on 8/8")
    print(f"    ours : Ours beats Halton on {a_wins}/{N_INSTANCES}  "
          f"-> {'PASS' if a_pass else 'FAIL'}")

    # Claim (b): Ours beats Halton on MOST Lipschitz instances.
    #   Paper Table II OURS row 5.89e1..4.20e1 below HALTON on all 8.
    b_wins = int(np.sum(lip["Ours"] < lip["Halton"]))
    b_need = math.ceil(0.75 * N_INSTANCES)  # "most" >= 6/8
    b_pass = b_wins >= b_need
    verdicts.append(b_pass)
    print("\n(b) Ours < Halton on MOST Lipschitz instances")
    print("    paper: Table II, OURS below HALTON on 8/8")
    print(f"    ours : Ours beats Halton on {b_wins}/{N_INSTANCES} "
          f"(need >= {b_need}) -> {'PASS' if b_pass else 'FAIL'}")

    # Claim (c): EA panel (pymoo). Ours best-or-second on most instances, and
    #   MOEA/D shows higher across-instance variance than Ours on Lipschitz.
    has_ea = any(n in lip for n in EA_NAMES)
    if not has_ea:
        print("\n(c) [SKIPPED] EA claims need pymoo (not installed).")
        print("    paper: Ours best-or-2nd on most instances; MOEA/D high"
              " variance\n    (Table II strong on L2/L5, poor on L8).")
    else:
        all_methods = ["Ours", "Halton", "Random"] + \
            [n for n in EA_NAMES if n in lip]
        rank_ok, total = 0, 0
        for means in (mono, lip):
            for j in range(N_INSTANCES):
                col = sorted((means[m][j], m) for m in all_methods
                             if m in means and np.isfinite(means[m][j]))
                rank = [m for _, m in col].index("Ours")
                total += 1
                rank_ok += int(rank <= 1)
        c1 = rank_ok >= math.ceil(0.6 * total)
        ours_var = float(np.nanvar(lip["Ours"]))
        moead_var = float(np.nanvar(lip["MOEA/D"])) if "MOEA/D" in lip else 0.0
        c2 = moead_var > ours_var
        c_pass = c1 and c2
        verdicts.append(c_pass)
        print("\n(c) Ours best-or-2nd on most instances AND MOEA/D across-"
              "instance\n    variance > Ours variance on Lipschitz")
        print("    paper: Ours consistently best/2nd; MOEA/D high variance")
        print(f"    ours : best-or-2nd on {rank_ok}/{total}; "
              f"var(MOEA/D)={moead_var:.2e} vs var(Ours)={ours_var:.2e}  "
              f"-> {'PASS' if c_pass else 'FAIL'}")

    print("\n" + "-" * 74)
    print(f"VERDICT: {sum(verdicts)}/{len(verdicts)} evaluated claims PASS")
    print("-" * 74)


# ===========================================================================
# Main.
# ===========================================================================


def main() -> None:
    t0 = time.time()
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print("Online co-design synthetic benchmarks  (arXiv:2604.22624, Sec."
          " VII-B)")
    print("=" * 74)
    print(f"Config: {N_INSTANCES} instances x {RUNS} runs;  "
          f"monotone budget={ITERS_MONO}, lipschitz budget={ITERS_LIP};  "
          f"K={K_ATOMS}, L={LIPSCHITZ_L}, delta={DELTA}")
    print("(paper scale: 100 runs, 4000 / 2000 iterations -- see constants)")

    mono_means, mono_curves = run_family(
        "Monotone", make_monotone_map, d=3,
        feature_names=["x0", "x1", "x2"],
        make_evaluator=lambda ff: MonotonicityEvaluator(ff, list(RC)),
        constrained=True, budget=ITERS_MONO, with_ea=True)
    print_panel("TABLE I  -- Monotone problems (E1), FixFunMinRes(f)",
                "paper Table I OURS row: 8.81e1 .. 3.46e1", "M", mono_means)

    lip_means, lip_curves = run_family(
        "Lipschitz", make_lipschitz_map, d=4,
        feature_names=["x0", "x1", "x2", "x3"],
        make_evaluator=lambda ff: LipschitzEvaluator(ff, list(RC),
                                                     L=LIPSCHITZ_L),
        constrained=False, budget=ITERS_LIP, with_ea=True)
    print_panel("TABLE II -- Lipschitz problems (E2), pure minimization",
                "paper Table II OURS row: 5.89e1 .. 4.20e1", "L", lip_means)

    validate(mono_means, lip_means)

    p1 = make_figure("Monotone", mono_curves, out_dir)
    p2 = make_figure("Lipschitz", lip_curves, out_dir)
    print("\nFigures:")
    for p in (p1, p2):
        if p:
            print(f"  {p}")

    print(f"\nTotal runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
