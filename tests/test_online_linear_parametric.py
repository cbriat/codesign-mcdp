"""Tests for the certified LinearParametricEvaluator.

The evaluator was reimplemented (arXiv:2604.22624, Section V-C3, eq. 26-28,
Lemma V.5) as a confidence polytope over the unknown linear parameters plus
one LP per resource coordinate, replacing the old OLS +/- confidence*sigma
band. The old band was *not* a guaranteed lower bound and could wrongly
eliminate an optimal candidate; the new construction is certified.

Coverage:
  (a) optimism guarantee: on data from a genuine linear map the bound never
      exceeds the true resource (property test over random instances);
  (b) old-failure-mode prevention: a paper-style case where the OLS band
      over-eliminates while the certified bound stays <= the true value;
  (c) degenerate cases: no observations, under-determined (unbounded)
      polytope, finite prior box, infeasible LP on noiseless-but-inconsistent
      data, bounded-noise band, and the deprecated `confidence` kwarg;
  (d) integration: solve_online with the certified evaluator recovers the
      same Pareto set as the no-pruning baseline (evaluator=None analogue).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from codesign import (
    AlgebraicDP,
    Ports,
    Reals,
    solve,
    solve_online,
)
from codesign.antichains import Antichain
from codesign.online import LinearParametricEvaluator, OptimisticEvaluator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

R2 = Ports({"r": Reals()})


def _ac(value):
    return Antichain(R2, [{"r": float(value)}])


def _ols_band_lower(obs_feats, obs_vals, query, confidence):
    """Replicate the OLD OLS +/- confidence*sigma lower bound.

    Mirrors the previous LinearParametricEvaluator.bound implementation so
    the test can demonstrate its (unsafe) over-elimination behaviour.
    """
    X = np.array([[1.0] + list(f) for f in obs_feats])
    y = np.array(obs_vals, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = float(coef[0] + sum(coef[i + 1] * query[i] for i in range(len(query))))
    residuals = y - X @ coef
    if residuals.size > X.shape[1]:
        sigma = float(np.sqrt((residuals ** 2).sum() / (residuals.size - X.shape[1])))
    else:
        sigma = float(np.abs(residuals).max() if residuals.size else 0.0)
    return max(0.0, pred - confidence * sigma)


# --------------------------------------------------------------------------- #
# (a) Optimism guarantee: property test over random linear instances
# --------------------------------------------------------------------------- #


def test_optimism_property_random_linear_instances():
    """On genuinely-linear data the certified bound never exceeds the truth."""
    rng = np.random.default_rng(20260719)
    features = ["f0", "f1", "f2"]
    r_components = ["r"]
    n_instances = 40
    violations = 0

    for _ in range(n_instances):
        # Random non-negative-valued affine map on the sampled region.
        theta = rng.uniform(0.5, 5.0, size=len(features) + 1)  # intercept + coefs

        def truth(feat):
            return float(theta[0] + sum(theta[i + 1] * feat[i]
                                        for i in range(len(feat))))

        ev = LinearParametricEvaluator(features, r_components, min_obs=2)
        # Enough well-spread observations to (over-)determine theta.
        n_obs = rng.integers(len(features) + 1, len(features) + 6)
        for j in range(int(n_obs)):
            fj = rng.uniform(0.0, 4.0, size=len(features))
            cand = {name: float(fj[i]) for i, name in enumerate(features)}
            ev.observe(j, cand, _ac(truth(fj)))

        # Fresh queries: the certified lower bound must be <= true value.
        for _q in range(5):
            fq = rng.uniform(0.0, 4.0, size=len(features))
            cand = {name: float(fq[i]) for i, name in enumerate(features)}
            lo, hi = ev.bound(cand)
            true_val = truth(fq)
            if lo["r"] > true_val + 1e-6:
                violations += 1
            assert hi["r"] == float("inf")  # no certified upper bound

    assert violations == 0, f"{violations} optimism violations across instances"


def test_optimism_exact_recovery_at_full_rank():
    """With rank(Phi_H) = p the bound recovers the true value (Remark V.4)."""
    features = ["f0", "f1"]
    ev = LinearParametricEvaluator(features, ["r"], min_obs=2)
    # r = 2 + 3 f0 + 1 f1
    def truth(a, b):
        return 2.0 + 3.0 * a + 1.0 * b
    for j, (a, b) in enumerate([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 3.0)]):
        ev.observe(j, {"f0": a, "f1": b}, _ac(truth(a, b)))
    for (a, b) in [(1.5, 1.5), (4.0, 0.0), (0.0, 5.0)]:
        lo, _ = ev.bound({"f0": a, "f1": b})
        assert lo["r"] == pytest.approx(truth(a, b), abs=1e-6)


# --------------------------------------------------------------------------- #
# (b) The OLD OLS-band failure mode is prevented
# --------------------------------------------------------------------------- #


def test_old_ols_band_would_over_eliminate_certified_does_not():
    """Paper-style case: OLS+band lower bound exceeds the true value.

    True resource is r = 10 f1 + 10 f2, but the evaluator is only given f1
    (a mis-specified feature set). Observations at equal f1 but different f2
    then look "noisy" to the OLS fit. At the query f1 = 0 (a candidate whose
    true resource is 0) the OLS band predicts a strictly positive lower
    bound -> it would eliminate an optimal candidate. The certified bound,
    seeing inconsistent exact-equality constraints, degrades safely to a
    lower bound <= the true value.
    """
    def truth(f1, f2):
        return 10.0 * f1 + 10.0 * f2

    obs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
    obs_feats = [(f1,) for (f1, _f2) in obs]          # evaluator sees only f1
    obs_vals = [truth(f1, f2) for (f1, f2) in obs]

    query = (0.0,)          # candidate with f1 = 0; its true resource is 0.0
    true_at_query = truth(0.0, 0.0)  # = 0.0

    # OLD behaviour: with a modest confidence the band lower bound is > true.
    confidence = 0.5
    old_lower = _ols_band_lower(obs_feats, obs_vals, query, confidence)
    assert old_lower > true_at_query + 0.5, (
        "test premise broken: OLS band should over-shoot the true value"
    )

    # NEW behaviour: certified bound stays at or below the true value.
    ev = LinearParametricEvaluator(["f1"], ["r"], min_obs=3)
    for j, ((f1,), v) in enumerate(zip(obs_feats, obs_vals)):
        ev.observe(j, {"f1": f1}, _ac(v))
    lo, _ = ev.bound({"f1": 0.0})
    assert lo["r"] <= true_at_query + 1e-6, (old_lower, lo["r"])


# --------------------------------------------------------------------------- #
# (c) Degenerate cases
# --------------------------------------------------------------------------- #


def test_no_observations_returns_trivial_bound():
    ev = LinearParametricEvaluator(["f0", "f1"], ["r"], min_obs=3)
    lo, hi = ev.bound({"f0": 1.0, "f1": 2.0})
    assert lo == {"r": 0.0}
    assert hi == {"r": float("inf")}


def test_underdetermined_unbounded_polytope_falls_back_to_prior_box():
    # One observation, 2 features -> polytope unbounded with the default
    # (open) prior box: the LP is unbounded, bound degrades to trivial.
    ev = LinearParametricEvaluator(["f0", "f1"], ["r"], min_obs=1)
    ev.observe(0, {"f0": 1.0, "f1": 1.0}, _ac(6.0))
    lo, _ = ev.bound({"f0": 5.0, "f1": 5.0})
    assert lo["r"] == 0.0  # unbounded LP -> safe trivial lower bound


def test_finite_prior_box_bounds_underdetermined_fit():
    # Same under-determined data, but a finite prior box on the parameters
    # makes the LP bounded and yields a non-trivial (still valid) lower bound.
    ev = LinearParametricEvaluator(
        ["f0", "f1"], ["r"], min_obs=1, prior_box=(0.0, 3.0),
    )
    ev.observe(0, {"f0": 1.0, "f1": 1.0}, _ac(6.0))  # theta0 + theta1 + theta2 = 6
    lo, _ = ev.bound({"f0": 0.0, "f1": 0.0})  # objective = intercept theta0
    # theta0 in [0, 3], and theta0 = 6 - theta1 - theta2 >= 6 - 3 - 3 = 0.
    assert 0.0 <= lo["r"] <= 3.0 + 1e-6


def test_prior_box_dict_per_component():
    ev = LinearParametricEvaluator(
        ["f0"], ["r"], min_obs=1, prior_box={"r": (-10.0, 10.0)},
    )
    ev.observe(0, {"f0": 1.0}, _ac(4.0))
    lo, _ = ev.bound({"f0": 2.0})
    assert np.isfinite(lo["r"])


def test_infeasible_noiseless_data_handled_gracefully():
    # Two observations with identical features but different resources are
    # inconsistent under the exact-equality (noiseless) model -> LP infeasible.
    # Must not crash; must degrade to a valid (<=truth) lower bound.
    ev = LinearParametricEvaluator(["f0"], ["r"], min_obs=2)
    ev.observe(0, {"f0": 1.0}, _ac(3.0))
    ev.observe(1, {"f0": 1.0}, _ac(9.0))  # inconsistent with obs 0
    lo, hi = ev.bound({"f0": 1.0})
    assert lo["r"] == 0.0
    assert hi["r"] == float("inf")


def test_bounded_noise_band_absorbs_inconsistency():
    # With noise_bound large enough to cover the spread, the same data is
    # feasible and produces a finite bound.
    ev = LinearParametricEvaluator(["f0"], ["r"], min_obs=2, noise_bound=5.0)
    ev.observe(0, {"f0": 1.0}, _ac(3.0))
    ev.observe(1, {"f0": 1.0}, _ac(9.0))
    lo, _ = ev.bound({"f0": 1.0})
    # Feasible: intercept in [9-5, 3+5] = [4, 8] (slope free); bound is finite.
    assert np.isfinite(lo["r"])


def test_negative_noise_bound_rejected():
    with pytest.raises(ValueError, match="noise_bound must be non-negative"):
        LinearParametricEvaluator(["f0"], ["r"], noise_bound=-1.0)


def test_bad_prior_box_entry_rejected():
    ev = LinearParametricEvaluator(["f0"], ["r"], min_obs=1, prior_box=3.0)
    ev.observe(0, {"f0": 1.0}, _ac(4.0))
    with pytest.raises(TypeError, match="prior_box entry"):
        ev.bound({"f0": 2.0})


def test_deprecated_confidence_kwarg_warns_but_works():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = LinearParametricEvaluator(["f0"], ["r"], confidence=3.0, min_obs=2)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "confidence" in str(caught[0].message)
    # Still functions identically to the no-confidence construction.
    for j, (a, v) in enumerate([(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]):
        ev.observe(j, {"f0": a}, _ac(v))  # r = 1 + 2 f0
    lo, _ = ev.bound({"f0": 3.0})
    assert lo["r"] == pytest.approx(7.0, abs=1e-6)


def test_no_confidence_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LinearParametricEvaluator(["f0"], ["r"])
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


# --------------------------------------------------------------------------- #
# (d) Integration: full Pareto recovery vs the no-pruning baseline
# --------------------------------------------------------------------------- #


def _linear_catalog():
    """Catalog whose inner-solve resource is affine in the features."""
    F = Ports({"load": Reals()})
    Rp = Ports({"total_cost": Reals(unit="USD"), "total_mass": Reals(unit="kg")})

    def make_dp(entry):
        # total_cost = 5 + 2*size + 3*power ; total_mass = 1 + 4*size
        # (both independent of the functionality request here, so the inner
        #  solve returns a single affine point in the features).
        cost = 5.0 + 2.0 * entry["size"] + 3.0 * entry["power"]
        mass = 1.0 + 4.0 * entry["size"]
        return AlgebraicDP(F, Rp, {
            "total_cost": lambda f, c=cost: c,
            "total_mass": lambda f, m=mass: m,
        })

    rng = np.random.default_rng(7)
    candidates = []
    for i in range(30):
        candidates.append({
            "name": f"c{i}",
            "size": float(rng.uniform(0.0, 10.0)),
            "power": float(rng.uniform(0.0, 10.0)),
        })
    return make_dp, {"load": 1.0}, candidates


def test_integration_recovers_full_pareto_vs_baseline():
    make_dp, f, candidates = _linear_catalog()
    features = ["size", "power"]
    r_components = ["total_cost", "total_mass"]

    # Baseline: base evaluator has trivial (0, inf) bounds -> never eliminates,
    # so it evaluates the whole catalog and yields the exact Pareto set. This
    # is the "evaluator=None" analogue for the required comparison.
    baseline_ev = OptimisticEvaluator(features, r_components)
    res_base = solve_online(make_dp, f, candidates=candidates,
                            evaluator=baseline_ev)

    # Certified linear-parametric evaluator with a warm start so the polytope
    # is determined early.
    lp_ev = LinearParametricEvaluator(features, r_components, min_obs=3)
    res_lp = solve_online(make_dp, f, candidates=candidates,
                          evaluator=lp_ev, warm_start=4)

    def to_set(ac):
        return sorted(
            (round(p["total_cost"], 6), round(p["total_mass"], 6))
            for p in ac.points
        )

    assert to_set(res_lp.antichain) == to_set(res_base.antichain), (
        to_set(res_lp.antichain), to_set(res_base.antichain),
    )
    # The certified bound is safe: it must never miss a true Pareto point.
    # (It may or may not prune, but correctness is the invariant.)
    assert res_lp.n_evaluated + res_lp.n_eliminated <= len(candidates)


def test_integration_certified_prunes_on_linear_catalog():
    """With an informative linear fit the evaluator prunes some candidates."""
    make_dp, f, candidates = _linear_catalog()
    features = ["size", "power"]
    r_components = ["total_cost", "total_mass"]
    lp_ev = LinearParametricEvaluator(features, r_components, min_obs=3)
    res = solve_online(make_dp, f, candidates=candidates,
                       evaluator=lp_ev, warm_start=4)
    # Some elimination should occur once the polytope is well determined.
    assert res.n_eliminated >= 1
