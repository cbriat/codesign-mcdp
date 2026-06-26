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


class GaussianProcessEvaluator(OptimisticEvaluator):
    """Bounds from a Gaussian process surrogate with an RBF kernel.

    For each R component k we fit a zero-mean GP

        h_k(c) ~ GP(0, sigma_f^2 RBF(c, c'; ell))

    on observed (features, summary) pairs. The bound at an unobserved
    query is ``mean +/- confidence * sigma``. The implementation is
    pure-numpy (no scikit-learn dependency); for typical small N (a
    few dozen observations) and a few-dimensional feature space the
    cost of refitting on every ``bound`` call is negligible.

    Hyperparameters
    ---------------
    length_scale : float
        Kernel length scale, applied uniformly across features. Tune to
        the typical "smoothness" of the response surface relative to
        the normalised feature span; a starting value of 0.3 is
        appropriate for features in roughly [0, 1].
    sigma_f : float
        Kernel signal amplitude. The default of 1.0 is rescaled
        per-output by the empirical standard deviation of the
        observations, so users rarely need to tune this directly.
    noise : float
        Observation noise variance (jitter). Stabilises the Cholesky
        factor and accommodates a small amount of unexplained variation
        in the inner-solve output. Default ``1e-3`` works in practice;
        increase to 1e-2 if observations look genuinely noisy.
    confidence : float
        Multiplier on the predictive standard deviation. The
        ``confidence=2.0`` default corresponds to a roughly 95%
        coverage band assuming Gaussian residuals.
    min_obs : int
        Minimum number of observations before the GP returns non-trivial
        bounds. Below this the evaluator returns the fallback ``(0, inf)``.
    """

    def __init__(self, features, r_components,
                 length_scale=0.3, sigma_f=1.0, noise=1e-3,
                 confidence=2.0, min_obs=3):
        super().__init__(features, r_components)
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.noise = float(noise)
        self.confidence = float(confidence)
        self.min_obs = int(min_obs)
        # Cache the last fit so successive bound() calls in the same
        # solve iteration don't refit redundantly.
        self._fit_cache_n = -1
        self._fit_cache: Dict[str, Tuple[Any, Any, Any, float, float]] = {}

    def _rbf(self, X1, X2):
        np = _import_numpy()
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1 . x2
        sq1 = (X1 ** 2).sum(axis=1)[:, None]
        sq2 = (X2 ** 2).sum(axis=1)[None, :]
        d2 = sq1 + sq2 - 2.0 * X1 @ X2.T
        d2 = np.maximum(d2, 0.0)
        return self.sigma_f ** 2 * np.exp(-0.5 * d2 / self.length_scale ** 2)

    def _refit(self):
        """Cache Cholesky factor and alpha vector for every R component."""
        np = _import_numpy()
        n = len(self._obs)
        if n == self._fit_cache_n and self._fit_cache:
            return
        X = np.array([o.features for o in self._obs], dtype=float)
        # Common kernel matrix.
        K = self._rbf(X, X) + self.noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Bump jitter and retry once.
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        self._fit_cache = {}
        for k in self.r_components:
            y = np.array([o.summary.get(k, float("nan")) for o in self._obs],
                         dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < self.min_obs:
                continue
            # Zero-mean GP: centre y at empirical mean, scale by std.
            y_mean = float(y[mask].mean())
            y_std = float(y[mask].std())
            if y_std < 1e-12:
                y_std = 1.0
            y_centred = (y - y_mean) / y_std
            # Refit just this output with possibly-masked observations.
            if mask.sum() < n:
                Xk = X[mask]
                Kk = self._rbf(Xk, Xk) + self.noise * np.eye(mask.sum())
                try:
                    Lk = np.linalg.cholesky(Kk)
                except np.linalg.LinAlgError:
                    Lk = np.linalg.cholesky(Kk + 1e-6 * np.eye(mask.sum()))
                alpha_k = np.linalg.solve(Lk.T,
                                          np.linalg.solve(Lk, y_centred[mask]))
                self._fit_cache[k] = (Xk, alpha_k, Lk, y_mean, y_std)
            else:
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centred))
                self._fit_cache[k] = (X, alpha, L, y_mean, y_std)
        self._fit_cache_n = n

    def observe(self, candidate_id, candidate, antichain):
        super().observe(candidate_id, candidate, antichain)
        # Invalidate cache; will refit lazily on next bound().
        self._fit_cache_n = -1

    def bound(self, candidate):
        np = _import_numpy()
        feat = np.array([_feature_vector(candidate, self.features)],
                        dtype=float)
        lo = {k: 0.0 for k in self.r_components}
        hi = {k: float("inf") for k in self.r_components}
        if len(self._obs) < self.min_obs:
            return lo, hi
        self._refit()
        for k in self.r_components:
            if k not in self._fit_cache:
                continue
            Xk, alpha, L, y_mean, y_std = self._fit_cache[k]
            k_star = self._rbf(feat, Xk)
            mu_centred = float((k_star @ alpha)[0])
            # Predictive variance: k(x*, x*) - k_star K^{-1} k_star^T
            v = np.linalg.solve(L, k_star.T)
            var = self.sigma_f ** 2 - float((v * v).sum())
            sigma_centred = float(np.sqrt(max(var, 0.0)))
            # Rescale to the original output scale.
            pred = mu_centred * y_std + y_mean
            sigma = sigma_centred * y_std
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


def _picker_lcb(lo: Mapping[str, float], hi: Mapping[str, float],
                r_components: Sequence[str], **_kwargs) -> float:
    """Lower-confidence-bound: sum of finite lower-bound components.

    For minimisation the most optimistic candidate has the smallest
    lower bound. This is the default strategy and matches the
    behaviour of the original solver.
    """
    s = 0.0
    for k in r_components:
        v = lo.get(k, 0.0)
        if math.isfinite(v):
            s += v
    return s


def _picker_ucb(lo: Mapping[str, float], hi: Mapping[str, float],
                r_components: Sequence[str], *, kappa: float = 0.5,
                **_kwargs) -> float:
    """Lower bound minus an exploration bonus weighted by uncertainty.

    The bonus ``kappa * (hi - lo)`` favours candidates whose bounds are
    still wide. With ``kappa = 0`` this collapses to pure LCB; with
    large ``kappa`` it approaches pure exploration. ``kappa = 0.5`` is
    a reasonable default for normalised-output problems; larger values
    are appropriate when the response surface has many local optima.
    """
    s = 0.0
    for k in r_components:
        lk = lo.get(k, 0.0)
        hk = hi.get(k, float("inf"))
        if not math.isfinite(lk):
            continue
        width = (hk - lk) if math.isfinite(hk) else 0.0
        s += lk - kappa * width
    return s


def _picker_random(lo: Mapping[str, float], hi: Mapping[str, float],
                   r_components: Sequence[str], *, rng=None,
                   **_kwargs) -> float:
    """Uniform-random picker: returns a uniform sample as the score.

    Useful as a baseline for comparing the value of structural priors.
    """
    if rng is None:
        import random as _random
        return _random.random()
    return rng.random()


_PICKERS = {
    "lcb":    _picker_lcb,
    "ucb":    _picker_ucb,
    "random": _picker_random,
}


def _resolve_picker(picker):
    """Map a picker spec (string, callable, or None) to a callable.

    Returns ``(score_fn, kwargs)`` where ``kwargs`` is a dict of
    strategy-specific keyword arguments. If ``picker`` is a tuple
    ``(name, kwargs)``, the kwargs are forwarded to the score function;
    e.g. ``picker=("ucb", {"kappa": 1.0})``.
    """
    if picker is None or (isinstance(picker, str) and picker == "lcb"):
        return _picker_lcb, {}
    if callable(picker):
        return picker, {}
    if isinstance(picker, str):
        if picker not in _PICKERS:
            raise ValueError(
                f"unknown picker {picker!r}; "
                f"choose from {sorted(_PICKERS)} or pass a callable"
            )
        return _PICKERS[picker], {}
    if isinstance(picker, tuple) and len(picker) == 2:
        name, kwargs = picker
        if name not in _PICKERS:
            raise ValueError(f"unknown picker {name!r}")
        return _PICKERS[name], dict(kwargs)
    raise TypeError(f"picker must be a string, tuple, or callable, got {type(picker).__name__}")


# Backwards-compatible alias for the original score function.
def _ucb_score(lower_bound: Mapping[str, float],
               r_components: Sequence[str]) -> float:
    """Deprecated alias for :func:`_picker_lcb`; kept for backwards compat."""
    return _picker_lcb(lower_bound, {}, r_components)


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


def _farthest_point_seeds(candidates: Sequence[Mapping[str, Any]],
                          features: Sequence[str],
                          n_seeds: int) -> List[int]:
    """Pick ``n_seeds`` candidate indices that are mutually far apart.

    Greedy farthest-point heuristic on the (unnormalised) feature
    Euclidean metric: start with the candidate closest to the centroid,
    then repeatedly add the candidate maximising the minimum distance
    to the already-picked set. Used by the integer form of the
    ``warm_start`` argument to produce diverse seed observations.
    """
    np = _import_numpy()
    if n_seeds <= 0 or not candidates:
        return []
    n_seeds = min(n_seeds, len(candidates))
    X = np.array([_feature_vector(c, features) for c in candidates],
                 dtype=float)
    centroid = X.mean(axis=0)
    d_to_centroid = np.linalg.norm(X - centroid, axis=1)
    picked = [int(np.argmin(d_to_centroid))]
    while len(picked) < n_seeds:
        d = np.full(len(candidates), np.inf)
        for p in picked:
            di = np.linalg.norm(X - X[p], axis=1)
            d = np.minimum(d, di)
        d[picked] = -1.0  # exclude already-picked
        picked.append(int(np.argmax(d)))
    return picked


def solve_online(
    candidate_fn: Callable[[Mapping[str, Any]], Any],
    functionality: Optional[Mapping],
    *,
    candidates: Sequence[Mapping[str, Any]],
    evaluator: OptimisticEvaluator,
    budget: Optional[int] = None,
    max_iter: int = 200,
    verbose: int = 0,
    warm_start: Optional[Any] = None,
    picker: Any = "lcb",
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
    warm_start : None, int, or list of int, optional
        Seed observations to populate the evaluator before the picker
        takes over. ``None`` (default) means no warm-start, matching
        the original behaviour. An integer ``n`` triggers a greedy
        farthest-point sampling of ``n`` mutually distant candidates,
        useful for evaluators (notably :class:`MonotonicityEvaluator`)
        that need observations spread across the feature space before
        their bounds become informative. A list of integer indices
        evaluates exactly those candidates as the seeds, intended for
        manually specified corner runs.
    picker : str, tuple, or callable, optional
        The candidate-selection strategy. Built-in options:
        ``"lcb"`` (default) minimises the sum of lower-bound
        components, equivalent to pure exploitation of the optimistic
        estimate; ``"ucb"`` adds an exploration bonus ``-kappa * (hi -
        lo)`` summed over R components; ``"random"`` picks uniformly
        at random. To tune the exploration weight pass a tuple, e.g.
        ``picker=("ucb", {"kappa": 1.0})``. A custom callable receives
        ``(lo, hi, r_components, **kwargs)`` and returns a score to
        minimise.

    Returns
    -------
    OnlineResult
    """
    from .solver import solve  # avoid circular import

    evaluator.reset()
    score_fn, picker_kwargs = _resolve_picker(picker)

    n_total = len(candidates)
    if budget is None:
        budget = n_total

    # Resolve warm-start spec to a concrete list of indices.
    warm_ids: List[int] = []
    if warm_start is not None:
        if isinstance(warm_start, int):
            warm_ids = _farthest_point_seeds(
                candidates, evaluator.features, warm_start)
        else:
            warm_ids = [int(i) for i in warm_start]
        # Truncate so the warm-start seeds cannot exceed the budget.
        warm_ids = warm_ids[:budget]

    R: Optional[Poset] = None
    incumbent: Optional[Antichain] = None
    remaining: List[int] = list(range(n_total))
    history: List[Dict[str, Any]] = []
    evaluated_ids: List[int] = []
    eliminated_ids: List[int] = []
    incumbent_ids: List[int] = []

    # ---- Warm-start: evaluate the seed candidates first, in given order.
    for cid in warm_ids:
        if cid not in remaining:
            continue
        if len(evaluated_ids) >= budget:
            break
        remaining.remove(cid)
        dp = candidate_fn(candidates[cid])
        if R is None:
            R = dp.R
            incumbent = Antichain.empty(R)
        inner_result = solve(dp, functionality, max_iter=max_iter, verbose=0)
        evaluated_ids.append(cid)
        evaluator.observe(cid, candidates[cid], inner_result.antichain)
        incumbent = Antichain.union_min(R, [incumbent, inner_result.antichain])
        if verbose >= 2:
            print(f"[online] warm-start cid={cid} "
                  f"|incumbent|={len(incumbent)} remaining={len(remaining)}")
        history.append({
            "pick": cid, "antichain": incumbent,
            "remaining": len(remaining),
            "evaluated": len(evaluated_ids),
            "eliminated": len(eliminated_ids),
            "phase": "warm_start",
        })

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

        # ---- 2. Pick the most promising remaining candidate by picker.
        best_cid = remaining[0]
        best_score = float("inf")
        for cid in remaining:
            lo, hi = evaluator.bound(candidates[cid])
            s = score_fn(lo, hi, evaluator.r_components, **picker_kwargs)
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
            "phase": "picker",
        })

    # The remaining (unevaluated, undominated) candidates are eliminated
    # for the purposes of the antichain (no budget was ever spent on them);
    # they are retained as "unexplored" for the diagnostic.
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
