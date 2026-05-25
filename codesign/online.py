"""
Compositional online learning for MCDPs.

A faithful (and deliberately simplified) port of the elimination-based
solver from Alharbi, Dahleh & Zardini, "Compositional Online Learning
for Co-Design" (arXiv:2604.22624). The idea: when a design problem has
many discrete candidates (catalog entries, robot types, component
families), evaluating every one is wasteful. Instead, maintain
history-dependent bounds on the inner-solve output for each candidate,
and only evaluate the candidate that the bounds say is "most promising"
under an upper-confidence rule. Candidates whose lower bound on the
antichain output is already dominated by the incumbent are eliminated
without ever being evaluated.

This module provides three concrete evaluators that encode three
different prior beliefs about how a candidate's "features" relate to
its output antichain:

- :class:`MonotonicityEvaluator` assumes the output is monotone in
  the features. Given an observation at features f0, candidates with
  features >= f0 are bounded below by the observation; candidates with
  features <= f0 are bounded above by it.
- :class:`LipschitzEvaluator` assumes the output is Lipschitz in the
  features with a user-supplied constant L. The bounds tighten by
  L * ||f - f_observed|| around every observation.
- :class:`LinearParametricEvaluator` fits a running least-squares
  linear model to observed output values and bounds new queries using
  a confidence band around the regressor.

The driver :func:`solve_online` is :func:`~codesign.solver.solve`'s
budgeted online cousin: it evaluates at most ``budget`` candidates,
returning the antichain of resource costs across the surviving ones
together with diagnostics about the elimination process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from .antichains import Antichain
from .posets import Ports, Poset


# ===========================================================================
# Helpers
# ===========================================================================


def _import_numpy():
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise ImportError(
            "The online learning module requires numpy. "
            "Install with `pip install numpy`."
        ) from e


def _feature_vector(candidate: Mapping[str, Any],
                    feature_names: Sequence[str]) -> List[float]:
    out = []
    for name in feature_names:
        v = candidate.get(name)
        if v is None:
            raise KeyError(
                f"candidate {candidate.get('name', '<unnamed>')!r} is missing "
                f"feature {name!r}"
            )
        out.append(float(v))
    return out


def _antichain_min_components(a: Antichain) -> Dict[str, float]:
    """Component-wise minimum across all points of an antichain.

    Used as a scalar proxy when we need to combine antichains into a
    comparable feature vector. For typical single-point antichains this
    just unpacks the unique point.
    """
    out: Dict[str, float] = {}
    for p in a.points:
        for k, v in p.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                if k not in out or v < out[k]:
                    out[k] = float(v)
    return out


def _antichain_dominates(better: Antichain, worse: Antichain) -> bool:
    """``better`` Pareto-dominates ``worse`` (in the upset sense).

    True iff for every point of ``worse``, there's a point of ``better``
    that is <=-dominated by it. (Smaller resources are better.)
    """
    return better.leq(worse)


# ===========================================================================
# Optimistic evaluators
# ===========================================================================


@dataclass
class _Observation:
    candidate_id: int
    features: List[float]
    antichain: Antichain
    summary: Dict[str, float]  # min-component summary for fast comparisons


class OptimisticEvaluator:
    """Base class: history-dependent bounds on the inner solve's output.

    Subclasses implement :meth:`bound` to return a ``(lower, upper)``
    pair of dicts mapping each numeric R component to its current
    lower and upper bound at the queried feature point. Lower bound is
    the smallest resource value still possible; upper bound is the
    largest. Tighter bounds prune more candidates.

    The base class maintains the observation history and supplies a
    default fallback: lower = 0, upper = +inf. Subclasses override
    :meth:`bound` to refine these.
    """

    def __init__(self, features: Sequence[str], r_components: Sequence[str]):
        self.features = list(features)
        self.r_components = list(r_components)
        self._obs: List[_Observation] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Forget every observation."""
        self._obs = []

    def observe(self, candidate_id: int, candidate: Mapping[str, Any],
                antichain: Antichain) -> None:
        """Record that the inner solve at ``candidate`` returned ``antichain``."""
        feat = _feature_vector(candidate, self.features)
        summary = _antichain_min_components(antichain)
        self._obs.append(_Observation(
            candidate_id=candidate_id,
            features=feat,
            antichain=antichain,
            summary=summary,
        ))

    def bound(self, candidate: Mapping[str, Any]
              ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Default fallback bound: (0, +inf) for each R component.

        Subclasses override to tighten.
        """
        lo = {k: 0.0 for k in self.r_components}
        hi = {k: float("inf") for k in self.r_components}
        return lo, hi

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"features={self.features}, "
                f"r={self.r_components}, "
                f"n_obs={len(self._obs)})")


class MonotonicityEvaluator(OptimisticEvaluator):
    """Bounds from monotonicity alone.

    Assumes the inner solve's output is component-wise monotone in the
    features (i.e. larger feature values produce no-smaller resource
    costs). Given an observation at feature point f0 with antichain min
    R0, then:

    - Any candidate with features >= f0 has resources >= R0 (lower bound).
    - Any candidate with features <= f0 has resources <= R0 (upper bound).
    """

    def bound(self, candidate):
        feat = _feature_vector(candidate, self.features)
        lo = {k: 0.0 for k in self.r_components}
        hi = {k: float("inf") for k in self.r_components}
        for obs in self._obs:
            geq = all(feat[i] >= obs.features[i] for i in range(len(feat)))
            leq = all(feat[i] <= obs.features[i] for i in range(len(feat)))
            for k in self.r_components:
                v = obs.summary.get(k)
                if v is None:
                    continue
                if geq and v > lo[k]:
                    lo[k] = v
                if leq and v < hi[k]:
                    hi[k] = v
        return lo, hi


class LipschitzEvaluator(OptimisticEvaluator):
    """Bounds from a Lipschitz assumption on the candidate-to-resource map.

    Assumes ``|h(c1) - h(c2)| <= L * ||features(c1) - features(c2)||``
    (Euclidean distance in feature space, per R component). Each
    observation tightens the bound everywhere by a cone of slope L.

    ``L`` is either a single positive float (same Lipschitz constant for
    every R component) or a dict mapping R component name to its own L.
    """

    def __init__(self, features, r_components, L):
        super().__init__(features, r_components)
        if isinstance(L, (int, float)):
            self.L: Dict[str, float] = {k: float(L) for k in r_components}
        else:
            self.L = {k: float(L[k]) for k in r_components}

    def bound(self, candidate):
        feat = _feature_vector(candidate, self.features)
        lo = {k: 0.0 for k in self.r_components}
        hi = {k: float("inf") for k in self.r_components}
        if not self._obs:
            return lo, hi
        for obs in self._obs:
            d = math.sqrt(sum(
                (feat[i] - obs.features[i]) ** 2 for i in range(len(feat))
            ))
            for k in self.r_components:
                v = obs.summary.get(k)
                if v is None:
                    continue
                lower_k = max(0.0, v - self.L[k] * d)
                upper_k = v + self.L[k] * d
                if lower_k > lo[k]:
                    lo[k] = lower_k
                if upper_k < hi[k]:
                    hi[k] = upper_k
        return lo, hi


class LinearParametricEvaluator(OptimisticEvaluator):
    """Bounds from a running least-squares linear model.

    Assumes each R component is approximately linear in the features:
    ``h_k(c) approx a_k + b_k . features(c)``. After a few observations,
    fit the coefficients by ordinary least squares; the bound at a query
    point is the prediction +/- ``confidence`` * sigma_k, where sigma_k
    is the residual standard deviation.

    Falls back to the default (0, +inf) bound while we have fewer than
    ``min_obs`` observations.
    """

    def __init__(self, features, r_components, confidence=2.0, min_obs=3):
        super().__init__(features, r_components)
        self.confidence = float(confidence)
        self.min_obs = int(min_obs)

    def bound(self, candidate):
        feat = _feature_vector(candidate, self.features)
        lo = {k: 0.0 for k in self.r_components}
        hi = {k: float("inf") for k in self.r_components}
        if len(self._obs) < self.min_obs:
            return lo, hi
        np = _import_numpy()
        X = np.array([[1.0] + list(o.features) for o in self._obs])
        for k in self.r_components:
            y = np.array([o.summary.get(k, float("nan")) for o in self._obs])
            mask = np.isfinite(y)
            if mask.sum() < self.min_obs:
                continue
            Xk = X[mask]
            yk = y[mask]
            try:
                coef, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
            except np.linalg.LinAlgError:
                continue
            pred = float(coef[0] + sum(coef[i + 1] * feat[i]
                                       for i in range(len(feat))))
            residuals = yk - Xk @ coef
            if residuals.size > Xk.shape[1]:
                sigma = float(np.sqrt(
                    (residuals ** 2).sum() / (residuals.size - Xk.shape[1])
                ))
            else:
                sigma = float(np.abs(residuals).max() if residuals.size else 0.0)
            lower_k = max(0.0, pred - self.confidence * sigma)
            upper_k = pred + self.confidence * sigma
            if lower_k > lo[k]:
                lo[k] = lower_k
            if upper_k < hi[k]:
                hi[k] = upper_k
        return lo, hi


# ===========================================================================
# Online solver
# ===========================================================================


@dataclass
class OnlineResult:
    """Outcome of an online elimination solve.

    Attributes
    ----------
    antichain : Antichain
        Min over the resource costs of every evaluated, surviving candidate.
    n_evaluated : int
        Number of inner-solve calls actually made.
    n_eliminated : int
        Number of candidates pruned without evaluation.
    n_candidates : int
        Total candidates considered at start.
    history : list[dict]
        Per-iteration log: ``{'pick', 'antichain', 'eliminated_now',
        'remaining'}``. The order in which candidates were chosen and the
        elimination cascade are reconstructible from this.
    evaluated_ids : list[int]
        Indices (into the original candidates list) actually evaluated.
    eliminated_ids : list[int]
        Indices pruned by the bound check.
    incumbent_ids : list[int]
        Indices whose evaluation contributed a point to the final antichain.
    """

    antichain: Antichain
    n_evaluated: int
    n_eliminated: int
    n_candidates: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    evaluated_ids: List[int] = field(default_factory=list)
    eliminated_ids: List[int] = field(default_factory=list)
    incumbent_ids: List[int] = field(default_factory=list)

    def __repr__(self):
        return (f"OnlineResult(\n"
                f"  antichain={self.antichain},\n"
                f"  n_evaluated={self.n_evaluated}, "
                f"n_eliminated={self.n_eliminated}, "
                f"n_candidates={self.n_candidates}\n"
                f")")


def _ucb_score(lower_bound: Mapping[str, float],
               r_components: Sequence[str]) -> float:
    """Lower-confidence-bound score for picking the next candidate.

    For minimisation problems the most "optimistic" candidate is the one
    whose lower bound is smallest, i.e. the one most likely to improve
    the incumbent. We return the sum of finite lower-bound components
    (cheap, basis-aware scoring that matches the antichain comparisons).
    """
    s = 0.0
    for k in r_components:
        v = lower_bound.get(k, 0.0)
        if math.isfinite(v):
            s += v
    return s


def _is_dominated_by_incumbent(
    lower_bound: Mapping[str, float],
    incumbent: Antichain,
) -> bool:
    """True if the candidate's optimistic lower bound is already dominated.

    The candidate cannot beat ``incumbent`` if for some point ``p`` in
    the incumbent, every R component's lower bound is >= the
    corresponding ``p`` value. The candidate is then provably suboptimal
    and can be eliminated.
    """
    if not incumbent.points:
        return False
    for p in incumbent.points:
        worse_or_equal = True
        for k, v in p.items():
            if not isinstance(v, (int, float)):
                worse_or_equal = False
                break
            lo = lower_bound.get(k, 0.0)
            if lo < v:
                worse_or_equal = False
                break
        if worse_or_equal:
            return True
    return False


def solve_online(
    candidate_fn: Callable[[Mapping[str, Any]], Any],
    functionality: Optional[Mapping],
    *,
    candidates: Sequence[Mapping[str, Any]],
    evaluator: OptimisticEvaluator,
    budget: Optional[int] = None,
    max_iter: int = 200,
    verbose: int = 0,
) -> OnlineResult:
    """Solve a co-design problem online with elimination-based pruning.

    For each entry in ``candidates``, ``candidate_fn(entry)`` should
    build a fresh design problem; the solver then runs the standard
    Kleene iteration at ``functionality`` on that DP. The evaluator's
    bounds are used to skip candidates that are provably suboptimal
    without running their inner solve.

    Parameters
    ----------
    candidate_fn : callable
        ``candidate_fn(candidate_dict) -> DP``. The DP is solved with
        the standard :func:`~codesign.solver.solve`. Catalog-style
        problems work well here: the candidate dict carries the
        catalog entry's features.
    functionality : dict or None
        The outer F vector, passed to every inner solve unchanged.
    candidates : list of dict
        Discrete set of candidate parameterisations. Each candidate's
        keys must include every name in ``evaluator.features``.
    evaluator : OptimisticEvaluator
        The bound machinery. Reset and populated as the solve proceeds.
    budget : int or None
        Maximum number of inner solves to run. ``None`` means unlimited
        (but elimination still prunes wherever bounds prove it).
    max_iter : int
        Forwarded to each inner solve.
    verbose : int
        0 silent, 1 final summary, 2 per-iteration trace.

    Returns
    -------
    OnlineResult
    """
    from .solver import solve  # avoid circular import

    evaluator.reset()

    n_total = len(candidates)
    if budget is None:
        budget = n_total

    R: Optional[Poset] = None
    incumbent: Optional[Antichain] = None
    remaining: List[int] = list(range(n_total))
    history: List[Dict[str, Any]] = []
    evaluated_ids: List[int] = []
    eliminated_ids: List[int] = []
    incumbent_ids: List[int] = []

    while remaining and len(evaluated_ids) < budget:
        # ---- 1. Compute lower bounds, eliminate anything already dominated.
        if incumbent is not None and incumbent.points:
            still_in: List[int] = []
            eliminated_now: List[int] = []
            for cid in remaining:
                lo, _hi = evaluator.bound(candidates[cid])
                if _is_dominated_by_incumbent(lo, incumbent):
                    eliminated_now.append(cid)
                else:
                    still_in.append(cid)
            remaining = still_in
            eliminated_ids.extend(eliminated_now)
            if not remaining:
                break

        # ---- 2. Pick the most promising remaining candidate by UCB on lo.
        best_cid = remaining[0]
        best_score = float("inf")
        for cid in remaining:
            lo, _hi = evaluator.bound(candidates[cid])
            s = _ucb_score(lo, evaluator.r_components)
            if s < best_score:
                best_score = s
                best_cid = cid
        remaining.remove(best_cid)

        # ---- 3. Run the inner solve.
        dp = candidate_fn(candidates[best_cid])
        if R is None:
            R = dp.R
            incumbent = Antichain.empty(R)
        inner_result = solve(dp, functionality, max_iter=max_iter, verbose=0)
        evaluated_ids.append(best_cid)
        evaluator.observe(best_cid, candidates[best_cid], inner_result.antichain)

        # ---- 4. Merge into incumbent (Min of union).
        new_incumbent = Antichain.union_min(R, [incumbent, inner_result.antichain])
        incumbent = new_incumbent

        if verbose >= 2:
            print(f"[online] eval #{len(evaluated_ids)} cid={best_cid} "
                  f"|incumbent|={len(incumbent)} remaining={len(remaining)}")

        history.append({
            "pick": best_cid,
            "antichain": incumbent,
            "remaining": len(remaining),
            "evaluated": len(evaluated_ids),
            "eliminated": len(eliminated_ids),
        })

    # The remaining (unevaluated, undominated) candidates ARE eliminated
    # for the purposes of the antichain (we never spent budget on them);
    # they are kept around as "unexplored" for the diagnostic.
    n_eval = len(evaluated_ids)
    n_elim = len(eliminated_ids)

    if verbose >= 1:
        unexplored = len(remaining)
        print(f"[online] done: evaluated {n_eval}/{n_total}, "
              f"eliminated {n_elim}, unexplored (budget hit) {unexplored}, "
              f"|antichain|={len(incumbent) if incumbent else 0}")

    if incumbent is None:
        # Pathological: candidates list was empty.
        return OnlineResult(
            antichain=Antichain.empty(Ports({})),
            n_evaluated=0, n_eliminated=0, n_candidates=0,
            history=history, evaluated_ids=evaluated_ids,
            eliminated_ids=eliminated_ids, incumbent_ids=incumbent_ids,
        )

    # Identify which evaluated candidates contributed a point to the final
    # antichain. An observation contributes if its antichain shares any
    # min-point with the incumbent.
    for obs in evaluator._obs:
        for p in obs.antichain.points:
            if any(R.eq(p, q) for q in incumbent.points):
                incumbent_ids.append(obs.candidate_id)
                break

    return OnlineResult(
        antichain=incumbent,
        n_evaluated=n_eval,
        n_eliminated=n_elim,
        n_candidates=n_total,
        history=history,
        evaluated_ids=evaluated_ids,
        eliminated_ids=eliminated_ids,
        incumbent_ids=incumbent_ids,
    )
