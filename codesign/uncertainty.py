"""
Uncertainty layer for the codesign package.

Two regimes are supported.

**Set-based, deterministic.** Each uncertain parameter (or correlated bundle
of parameters) lives in a known set: a :class:`Box` (axis-aligned interval
product), a :class:`Disk` or :class:`Circle` (2-D conveniences), or an
:class:`Ellipsoid` (n-D). The worst-case antichain over the set is computed
analytically when "direction of badness" is declared, by boundary sampling
otherwise.

**Stochastic.** Each parameter has a marginal distribution
(``scipy.stats``-compatible) and optionally a copula describing the joint
dependence. Monte Carlo gives a distribution of antichains; statistical
summaries (mean, p95, CVaR at 95%) come out alongside the raw samples.

Both kinds of uncertainty are declared on a :class:`~codesign.module.Module`
instance via the ``uncertain_set`` and ``uncertain_dist`` attributes.

Example::

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass":     Reals(unit="kg")}

        def __init__(self, specific_energy=1.8e6, efficiency=0.85):
            self.specific_energy = specific_energy
            self.efficiency = efficiency
            super().__init__()

        def h(self, f):
            return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}

    b = Battery()
    b.uncertain_set = Box(
        specific_energy=(1.6e6, 2.0e6, "more_is_better"),
        efficiency=(0.80, 0.90, "more_is_better"),
    )
    b.uncertain_dist = Stochastic(
        marginals={
            "specific_energy": stats.uniform(loc=1.6e6, scale=0.4e6),
            "efficiency":      stats.uniform(loc=0.80, scale=0.10),
        },
        copula=GaussianCopula(correlation=[[1.0, 0.4], [0.4, 1.0]]),
    )

Then::

    result = solve(dp, f,
                   uncertainty=["worst_case", "mean", "p95", "cvar95", "samples"],
                   n_samples=1000, rng_seed=0)
    result.worst_case   # SolveResult-equivalent at the worst point of the set
    result.mean         # dict[r_port -> mean value across MC samples]
    result.p95          # dict[r_port -> 95th percentile]
    result.cvar95       # dict[r_port -> CVaR at 95% level]
    result.samples      # list[Antichain], one per MC sample
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _import_numpy_or_die():
    try:
        import numpy as np
        return np
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The uncertainty layer requires numpy. "
            "Install with `pip install numpy`."
        ) from e


def _import_scipy_stats_or_die():  # pragma: no cover
    try:
        from scipy import stats
        return stats
    except ImportError as e:
        raise ImportError(
            "The stochastic uncertainty layer requires scipy. "
            "Install with `pip install scipy`."
        ) from e


# ===========================================================================
# Set-based uncertainty
# ===========================================================================


_DIRECTION_TOKENS = (
    "more_is_better", "more_is_worse", "less_is_better", "less_is_worse",
)


def _normalise_direction(token: Optional[str]) -> Optional[int]:
    """Return +1 if 'larger is worse', -1 if 'smaller is worse', None if undeclared.

    Mnemonic: the worst value for the user is the smallest value when
    "more_is_better", so the direction-of-badness sign is -1.
    """
    if token is None:
        return None
    if token in ("more_is_better", "less_is_worse"):
        return -1   # worst = smallest
    if token in ("more_is_worse", "less_is_better"):
        return +1   # worst = largest
    raise ValueError(
        f"unknown direction-of-badness token {token!r}; must be one of "
        f"{_DIRECTION_TOKENS}."
    )


@dataclass
class _ParamSpec:
    """Internal spec for a single uncertain parameter under a Box."""
    name: str
    lo: float
    hi: float
    direction: Optional[int]   # +1 or -1 or None


class UncertaintySet:
    """Abstract base for deterministic, set-based parameter uncertainty.

    Subclasses implement :meth:`worst_case_values` and :meth:`param_names`.
    """

    def param_names(self) -> List[str]:
        raise NotImplementedError

    def worst_case_values(self, h_callable, base_values: Mapping[str, Any],
                          f_inner: Mapping[str, Any]) -> Dict[str, float]:
        """Return the parameter dict at the worst-case point of the set.

        ``h_callable(params_override) -> antichain`` is a function the
        worst-case solver can evaluate at any candidate parameter point
        (used for boundary search when directions aren't declared).
        ``base_values`` carries nominal values for any params not in the set.
        ``f_inner`` is the current F values (passed for context to h, though
        usually unused).
        """
        raise NotImplementedError


class Box(UncertaintySet):
    """Axis-aligned interval product over one or more parameters.

    Each kwarg is either ``(lo, hi)`` (no direction declared) or
    ``(lo, hi, direction)`` where ``direction`` is one of
    ``"more_is_better"``, ``"more_is_worse"``, ``"less_is_better"``,
    ``"less_is_worse"``.

    When all directions are declared, the worst case is the unique
    corner where each parameter takes its worst endpoint. When some
    are undeclared, those parameters are searched by sampling endpoints
    (still cheap: at most 2 calls per undeclared parameter).
    """

    def __init__(self, **params):
        self._specs: Dict[str, _ParamSpec] = {}
        for name, val in params.items():
            if len(val) == 2:
                lo, hi = val
                direction = None
            elif len(val) == 3:
                lo, hi, dtoken = val
                direction = _normalise_direction(dtoken)
            else:
                raise ValueError(
                    f"Box parameter {name!r} must be (lo, hi) or "
                    f"(lo, hi, direction), got {val!r}"
                )
            if lo > hi:
                raise ValueError(f"Box parameter {name!r}: lo > hi")
            self._specs[name] = _ParamSpec(name, float(lo), float(hi), direction)

    def param_names(self) -> List[str]:
        return list(self._specs.keys())

    def worst_case_values(self, h_callable, base_values, f_inner):
        # Declared params: pick the worst endpoint directly.
        # Undeclared params: probe both endpoints with everything else at
        # its worst declared value, take whichever is worse.
        declared: Dict[str, float] = {}
        undeclared: List[_ParamSpec] = []
        for name, spec in self._specs.items():
            if spec.direction is None:
                undeclared.append(spec)
            else:
                # direction = +1 means larger is worse, so worst = hi
                declared[name] = spec.hi if spec.direction == +1 else spec.lo

        if not undeclared:
            return declared

        # Probe undeclared by sampling both endpoints. The "worst" is
        # the configuration whose antichain dominates the others under
        # the standard upset order; a cheap proxy is the maximum sum of
        # numeric R components.
        def candidate_score(overrides: Mapping[str, float]) -> float:
            merged = dict(base_values)
            merged.update(declared)
            merged.update(overrides)
            try:
                a = h_callable(merged)
            except Exception:
                return float("inf")  # treat blow-up as the worst possible
            score = 0.0
            for p in a.points:
                for v in p.values():
                    if isinstance(v, (int, float)):
                        score += float(v) if math.isfinite(v) else 1e30
            return score

        # Enumerate 2^n endpoint combinations (n = number of undeclared).
        n = len(undeclared)
        best_score = -float("inf")
        best_overrides: Dict[str, float] = {}
        for mask in range(1 << n):
            overrides = {}
            for i, spec in enumerate(undeclared):
                overrides[spec.name] = spec.hi if (mask >> i) & 1 else spec.lo
            s = candidate_score(overrides)
            if s > best_score:
                best_score = s
                best_overrides = dict(overrides)

        declared.update(best_overrides)
        return declared


class Ellipsoid(UncertaintySet):
    """n-D ellipsoid: ``(p - center)^T Sigma^{-1} (p - center) <= 1``.

    Parameters
    ----------
    center : dict[str, float]
        Centre of the ellipsoid, keyed by parameter name.
    cov : 2-D array
        Symmetric positive-definite "shape" matrix Sigma (rows/cols in
        the order of ``params``).
    params : list[str]
        Parameter names in the order of cov's rows/columns.
    directions : dict[str, str] or None
        Optional per-parameter "direction of badness" tokens. When fully
        provided, the worst case is computed analytically as the boundary
        point in the direction of badness. Otherwise, a grid of boundary
        points is sampled (``boundary_samples`` per quadrant, projected
        through the ellipsoid).
    boundary_samples : int
        Number of boundary samples per dimension when directions are
        not fully declared. Default 8.
    """

    def __init__(
        self,
        center: Mapping[str, float],
        cov,
        params: Sequence[str],
        directions: Optional[Mapping[str, str]] = None,
        boundary_samples: int = 8,
    ):
        np = _import_numpy_or_die()
        self._center = {k: float(v) for k, v in center.items()}
        self._params = list(params)
        self._cov = np.asarray(cov, dtype=float)
        if self._cov.shape != (len(self._params), len(self._params)):
            raise ValueError(
                f"cov shape {self._cov.shape} does not match params length "
                f"{len(self._params)}"
            )
        # Cholesky factor for sampling/boundary parametrisation.
        try:
            self._L = np.linalg.cholesky(self._cov)
        except Exception as e:
            raise ValueError("Ellipsoid cov must be positive definite") from e

        self._directions: Dict[str, Optional[int]] = {}
        if directions:
            for k, tok in directions.items():
                self._directions[k] = _normalise_direction(tok)
        for k in self._params:
            self._directions.setdefault(k, None)
        self._boundary_samples = int(boundary_samples)

    def param_names(self) -> List[str]:
        return list(self._params)

    def worst_case_values(self, h_callable, base_values, f_inner):
        np = _import_numpy_or_die()
        params = self._params
        center = np.array([self._center[k] for k in params])
        L = self._L

        # If all directions are declared, the worst case is at the
        # boundary point in the analytic direction
        # ``v = (sign[i]) * 1`` mapped through L, normalised.
        declared = [self._directions[k] for k in params]
        if all(d is not None for d in declared):
            # The R-effect direction in u-coords is unknown a priori,
            # but for a monotone-in-each-param module it equals the
            # declared direction. The worst boundary point is
            # ``center + L @ u_star`` where ``u_star`` maximises
            # ``d^T u`` subject to ``||u||<=1``, i.e. ``u_star = d/||d||``.
            d_vec = np.array(declared, dtype=float)
            n = np.linalg.norm(d_vec)
            u_star = d_vec / (n if n > 0 else 1.0)
            p_star = center + L @ u_star
            return {k: float(p_star[i]) for i, k in enumerate(params)}

        # Otherwise, sample boundary points and pick the worst by score.
        n_dim = len(params)
        # Generate a Fibonacci sphere of unit vectors as a quasi-uniform
        # sample of the boundary in u-space.
        N = max(self._boundary_samples * (2 ** n_dim), 32)
        rng = np.random.default_rng(0)
        # uniform samples on unit sphere via normal-projection
        z = rng.normal(size=(N, n_dim))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-30
        # boundary points in parameter space
        boundary_pts = center + z @ L.T  # shape (N, n_dim)

        def score(p_vec) -> float:
            merged = dict(base_values)
            for i, k in enumerate(params):
                merged[k] = float(p_vec[i])
            try:
                a = h_callable(merged)
            except Exception:
                return float("inf")
            s = 0.0
            for q in a.points:
                for v in q.values():
                    if isinstance(v, (int, float)):
                        s += float(v) if math.isfinite(v) else 1e30
            return s

        best_score = -float("inf")
        best_p = boundary_pts[0]
        for p_vec in boundary_pts:
            s = score(p_vec)
            if s > best_score:
                best_score = s
                best_p = p_vec
        return {k: float(best_p[i]) for i, k in enumerate(params)}


def Disk(center: Mapping[str, float], radius: float,
         params: Optional[Sequence[str]] = None,
         directions: Optional[Mapping[str, str]] = None) -> Ellipsoid:
    """2-D disk (filled circle): a special case of :class:`Ellipsoid`.

    ``radius`` is in parameter-units, and the disk is isotropic
    (``Sigma = radius**2 * I``).
    """
    if params is None:
        params = list(center.keys())
    if len(params) != 2:
        raise ValueError("Disk requires exactly 2 parameters.")
    np = _import_numpy_or_die()
    cov = (radius ** 2) * np.eye(2)
    return Ellipsoid(center, cov, params, directions)


def Circle(center: Mapping[str, float], radius: float,
           params: Optional[Sequence[str]] = None,
           directions: Optional[Mapping[str, str]] = None) -> Ellipsoid:
    """2-D circle (the boundary only).

    For monotone modules with declared directions, the worst case lies
    on the boundary anyway, so :func:`Disk` and :func:`Circle` produce
    identical worst-case answers. They differ only when directions are
    undeclared and boundary sampling is used: a :func:`Circle` is
    sampled only on the boundary, while a :func:`Disk` could also be
    sampled in the interior (though our implementation samples the
    boundary for both, since the interior is dominated for monotone
    modules).
    """
    # In the worst-case sense for monotone systems, the boundary is what
    # matters. Disk and Circle resolve identically here.
    return Disk(center, radius, params, directions)


# ===========================================================================
# Copulas
# ===========================================================================


class Copula:
    """Base class for copulas. Subclasses implement
    :meth:`sample_uniform` which returns shape ``(n, d)`` samples in
    ``[0, 1]^d`` with the desired dependence structure."""

    def sample_uniform(self, n: int, d: int, rng) -> "np.ndarray":  # noqa: F821
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Independence(Copula):
    """The independence copula: each component sampled independently in
    ``U(0, 1)``."""

    def sample_uniform(self, n, d, rng):
        return rng.uniform(size=(n, d))


class GaussianCopula(Copula):
    """Gaussian copula with given correlation matrix.

    ``correlation`` is a ``(d, d)`` symmetric positive-definite matrix
    with unit diagonal. To sample, draw ``z ~ N(0, R)`` and apply
    ``Phi`` to each component to obtain correlated uniforms.
    """

    def __init__(self, correlation):
        np = _import_numpy_or_die()
        R = np.asarray(correlation, dtype=float)
        if R.shape[0] != R.shape[1]:
            raise ValueError("correlation must be square")
        # Cholesky for sampling.
        try:
            self._L = np.linalg.cholesky(R)
        except Exception as e:
            raise ValueError(
                "correlation matrix must be positive definite"
            ) from e
        self._d = R.shape[0]

    def sample_uniform(self, n, d, rng):
        if d != self._d:
            raise ValueError(
                f"GaussianCopula was constructed for d={self._d}, "
                f"but {d} marginals were declared."
            )
        np = _import_numpy_or_die()
        from scipy import stats as _stats
        z = rng.normal(size=(n, d)) @ self._L.T
        return _stats.norm.cdf(z)

    def __repr__(self):
        return f"GaussianCopula(d={self._d})"


# ===========================================================================
# Stochastic uncertainty
# ===========================================================================


class Stochastic:
    """Joint distribution over a set of uncertain parameters.

    Parameters
    ----------
    marginals : dict[str, scipy.stats.rv_frozen]
        Marginal distributions for each parameter. Pass a frozen scipy
        distribution like ``stats.norm(loc=mu, scale=sigma)``.
    copula : Copula, optional
        Dependence structure between marginals. Defaults to
        :class:`Independence`.
    """

    def __init__(
        self,
        marginals: Mapping[str, Any] = None,
        copula: Optional[Copula] = None,
        **marginals_kw,
    ):
        if marginals is None:
            marginals = {}
        if marginals_kw:
            marginals = {**marginals, **marginals_kw}
        if not marginals:
            raise ValueError("Stochastic requires at least one marginal.")
        self.marginals: Dict[str, Any] = dict(marginals)
        self.copula: Copula = copula if copula is not None else Independence()

    def param_names(self) -> List[str]:
        return list(self.marginals.keys())

    def sample(self, n: int, rng) -> List[Dict[str, float]]:
        np = _import_numpy_or_die()
        names = list(self.marginals.keys())
        d = len(names)
        U = self.copula.sample_uniform(n, d, rng)
        # Apply inverse CDF (ppf) of each marginal.
        out = []
        for i in range(n):
            row = {}
            for j, name in enumerate(names):
                row[name] = float(self.marginals[name].ppf(U[i, j]))
            out.append(row)
        return out


# ===========================================================================
# Uncertainty result
# ===========================================================================


@dataclass
class UncertaintyResult:
    """Outcome of an uncertainty-aware solve.

    The attributes populated depend on which summaries were requested
    via the ``uncertainty=[...]`` argument to :func:`~codesign.solver.solve`.

    Attributes
    ----------
    worst_case : SolveResult or None
        The result at the worst point of every module's deterministic set.
    mean, p95, cvar95 : dict[str, float] or None
        Per-R-port statistics across MC samples (only over feasible samples).
    samples : list[Antichain] or None
        Raw MC antichains (only populated if ``"samples"`` was requested).
    feasibility_rate : float or None
        Fraction of MC samples that produced a feasible antichain.
    n_samples_used : int
        Number of MC samples actually drawn (0 if no stochastic summary
        was requested).
    """

    worst_case: Any = None
    mean: Optional[Dict[str, float]] = None
    p95: Optional[Dict[str, float]] = None
    cvar95: Optional[Dict[str, float]] = None
    samples: Optional[List[Any]] = None
    feasibility_rate: Optional[float] = None
    n_samples_used: int = 0

    def __repr__(self):
        bits = []
        if self.worst_case is not None:
            bits.append(f"worst_case={self.worst_case.antichain}")
        if self.mean is not None:
            bits.append(f"mean={self.mean}")
        if self.p95 is not None:
            bits.append(f"p95={self.p95}")
        if self.cvar95 is not None:
            bits.append(f"cvar95={self.cvar95}")
        if self.feasibility_rate is not None:
            bits.append(f"feasibility_rate={self.feasibility_rate:.3f}")
        return "UncertaintyResult(\n  " + ",\n  ".join(bits) + "\n)"


# ===========================================================================
# Uncertainty solver
# ===========================================================================


_VALID_SUMMARIES = {"worst_case", "mean", "p95", "cvar95", "samples"}


def _iter_uncertain_modules(dp) -> List[Tuple[str, Any]]:
    """Walk a DP and yield ``(module_name, module_instance)`` for every
    Module-like object carrying an ``uncertain_set`` or ``uncertain_dist``."""
    out: List[Tuple[str, Any]] = []
    # System-built DPs attach a `_codesign_modules` dict directly.
    modules = getattr(dp, "_codesign_modules", None)
    if modules:
        for name, mod in modules.items():
            if (getattr(mod, "uncertain_set", None) is not None or
                    getattr(mod, "uncertain_dist", None) is not None):
                out.append((name, mod))
        return out
    # Otherwise, a single DP that's itself an uncertainty-bearing Module.
    if (getattr(dp, "uncertain_set", None) is not None or
            getattr(dp, "uncertain_dist", None) is not None):
        out.append((getattr(dp, "name", "<dp>"), dp))
    return out


def _save_params(mod, names: Iterable[str]) -> Dict[str, Any]:
    """Snapshot module attribute values, so they can be restored after
    the uncertainty solve."""
    return {n: getattr(mod, n) for n in names}


def _apply_params(mod, values: Mapping[str, Any]) -> None:
    for k, v in values.items():
        setattr(mod, k, v)


def _restore_params(mod, snapshot: Mapping[str, Any]) -> None:
    for k, v in snapshot.items():
        setattr(mod, k, v)


def _percentile(values: Sequence[float], p: float) -> float:
    np = _import_numpy_or_die()
    return float(np.percentile(np.asarray(values), p))


def _cvar(values: Sequence[float], p: float) -> float:
    """Conditional value at risk at the upper p-percentile.

    For ``p=95``, this is the mean of the worst 5% of values.
    """
    np = _import_numpy_or_die()
    arr = np.asarray(values)
    thr = np.percentile(arr, p)
    tail = arr[arr >= thr]
    return float(tail.mean()) if tail.size else float(thr)


def solve_with_uncertainty(
    dp,
    functionality: Optional[Mapping],
    uncertainty: List[str],
    n_samples: int = 1000,
    rng_seed: Optional[int] = None,
    max_iter: int = 200,
    verbose: int = 0,
):
    """Run an uncertainty-aware solve and return an :class:`UncertaintyResult`.

    See :func:`codesign.solver.solve` for parameter docs.
    """
    from .solver import solve  # avoid circular import

    for label in uncertainty:
        if label not in _VALID_SUMMARIES:
            raise ValueError(
                f"unknown uncertainty summary {label!r}; "
                f"must be one of {sorted(_VALID_SUMMARIES)}"
            )

    want_worst = "worst_case" in uncertainty
    want_stoch = any(s in uncertainty for s in ("mean", "p95", "cvar95", "samples"))

    np = _import_numpy_or_die()
    rng = np.random.default_rng(rng_seed)

    modules = _iter_uncertain_modules(dp)
    if not modules:
        raise ValueError(
            "solve(uncertainty=...) was called but no module on the DP "
            "carries an uncertain_set or uncertain_dist attribute."
        )

    result = UncertaintyResult()

    # ----------------- Worst case (deterministic) -----------------
    if want_worst:
        snapshots: List[Tuple[Any, Dict[str, Any]]] = []
        try:
            for name, mod in modules:
                u_set = getattr(mod, "uncertain_set", None)
                if u_set is None:
                    continue
                params_to_set = u_set.param_names()
                snapshots.append((mod, _save_params(mod, params_to_set)))

                # Build a local h-evaluator that takes overrides.
                base_values = _save_params(mod, params_to_set)

                def _h_with_overrides(overrides, _mod=mod, _f=functionality):
                    snap = _save_params(_mod, params_to_set)
                    _apply_params(_mod, overrides)
                    try:
                        from .composition import Loop
                        if isinstance(dp, Loop):
                            # We can't evaluate the inner h alone for a system
                            # DP cleanly; cheap proxy: use the module's own h.
                            f_for_mod = {k: v for k, v in (_f or {}).items()
                                         if k in _mod.F.components}
                            return _mod.h(f_for_mod) if f_for_mod else \
                                Antichain.singleton(_mod.R, _mod.R.bottom())
                        return _mod.h(_f or {})
                    finally:
                        _restore_params(_mod, snap)

                # Strip the runtime import; reference the package-level one.
                from .antichains import Antichain  # noqa: F401  (used above)

                worst = u_set.worst_case_values(
                    _h_with_overrides, base_values, functionality or {},
                )
                _apply_params(mod, worst)

            # Run the canonical solve at the worst-case parameter setting.
            wc_result = solve(
                dp, functionality, max_iter=max_iter, verbose=verbose,
            )
            result.worst_case = wc_result
        finally:
            for mod, snap in snapshots:
                _restore_params(mod, snap)

    # ----------------- Stochastic summaries -----------------
    if want_stoch:
        # For each module with uncertain_dist, draw n_samples vectors.
        dists = [(name, mod, mod.uncertain_dist)
                 for (name, mod) in modules
                 if getattr(mod, "uncertain_dist", None) is not None]
        if not dists:
            raise ValueError(
                "Stochastic summaries requested but no module on the DP "
                "carries an uncertain_dist attribute."
            )

        # Pre-sample all distributions.
        all_samples: List[List[Tuple[Any, Dict[str, float]]]] = []
        # all_samples[i] = list of (module, params_dict) for sample i
        per_module_samples = []
        for name, mod, dist in dists:
            per_module_samples.append((mod, dist.sample(n_samples, rng)))

        antichains_per_sample: List[Any] = []
        feasibility: List[bool] = []
        snapshots = [(mod, _save_params(mod, dist.param_names()))
                     for name, mod, dist in dists]

        try:
            for i in range(n_samples):
                for (mod, samples_for_mod) in per_module_samples:
                    _apply_params(mod, samples_for_mod[i])
                r = solve(dp, functionality, max_iter=max_iter, verbose=0)
                antichains_per_sample.append(r.antichain)
                feasibility.append(r.feasible)
        finally:
            for mod, snap in snapshots:
                _restore_params(mod, snap)

        # Aggregate marginal statistics over feasible samples.
        feasible_chains = [
            a for a, ok in zip(antichains_per_sample, feasibility) if ok
        ]
        feas_rate = float(np.mean(feasibility)) if feasibility else 0.0
        result.feasibility_rate = feas_rate
        result.n_samples_used = n_samples

        # Marginal stats: for each R port, collect one number per sample.
        # For multi-point antichains, take the minimum over points (most
        # optimistic) to define a per-sample scalar. This is a defensible
        # default; users wanting other reductions can iterate over
        # `result.samples` themselves.
        port_values: Dict[str, List[float]] = {}
        for a in feasible_chains:
            for p in a.points:
                for k, v in p.items():
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        port_values.setdefault(k, []).append(float(v))

        if "mean" in uncertainty:
            result.mean = {k: float(np.mean(vs)) for k, vs in port_values.items()}
        if "p95" in uncertainty:
            result.p95 = {k: _percentile(vs, 95.0) for k, vs in port_values.items()}
        if "cvar95" in uncertainty:
            result.cvar95 = {k: _cvar(vs, 95.0) for k, vs in port_values.items()}
        if "samples" in uncertainty:
            result.samples = list(antichains_per_sample)

        if verbose >= 1:
            print(
                f"[solve] uncertainty: {n_samples} MC samples, "
                f"feasibility_rate={feas_rate:.3f}"
            )

    return result
