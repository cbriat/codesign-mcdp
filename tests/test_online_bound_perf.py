"""Equivalence regression test for the vectorized evaluator bounds.

`MonotonicityEvaluator.bound()` and `LipschitzEvaluator.bound()` were
optimized from a per-call Python loop over the whole observation history
(O(history) per call, making Algorithm-1-style loops O(budget^2)) to a
numpy-vectorized full-history rescan (still O(history) per call, but with a
tiny constant and no Python-level per-observation loop). A pure-Python
fallback is kept for when numpy is unavailable.

This is a pure performance change: the returned bounds must be IDENTICAL.
These tests pin that by comparing the (numpy) fast path against an
independent naive reference implementation embedded below -- the exact
pre-optimization algorithm -- over hundreds of random scenarios plus
explicit edge cases (empty history, single/duplicate observations,
boundary queries that hit an observation exactly, missing summary
components, and reset() mid-stream). The pure-Python fallback is also
checked against the same reference.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

from codesign.antichains import Antichain
from codesign.online import LipschitzEvaluator, MonotonicityEvaluator
from codesign.posets import Ports, Reals


# --------------------------------------------------------------------------- #
# Naive reference: the exact pre-optimization bound() algorithms.
# --------------------------------------------------------------------------- #

def mono_bound_ref(obs_list, feat, r_components):
    lo = {k: 0.0 for k in r_components}
    hi = {k: float("inf") for k in r_components}
    for obs in obs_list:
        geq = all(feat[i] >= obs.features[i] for i in range(len(feat)))
        leq = all(feat[i] <= obs.features[i] for i in range(len(feat)))
        for k in r_components:
            v = obs.summary.get(k)
            if v is None:
                continue
            if geq and v > lo[k]:
                lo[k] = v
            if leq and v < hi[k]:
                hi[k] = v
    return lo, hi


def lip_bound_ref(obs_list, feat, r_components, L):
    lo = {k: 0.0 for k in r_components}
    hi = {k: float("inf") for k in r_components}
    if not obs_list:
        return lo, hi
    for obs in obs_list:
        d = math.sqrt(sum(
            (feat[i] - obs.features[i]) ** 2 for i in range(len(feat))
        ))
        for k in r_components:
            v = obs.summary.get(k)
            if v is None:
                continue
            lower_k = max(0.0, v - L[k] * d)
            upper_k = v + L[k] * d
            if lower_k > lo[k]:
                lo[k] = lower_k
            if upper_k < hi[k]:
                hi[k] = upper_k
    return lo, hi


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _R(r_components):
    return Ports({k: Reals() for k in r_components})


def _cand(features, vec):
    return {f: v for f, v in zip(features, vec)}


def _assert_equal(lo_new, hi_new, lo_ref, hi_ref):
    # The vectorized and pure-Python paths agree to machine epsilon, but the
    # numpy pairwise-summed distance can differ from the scalar left-to-right
    # sum by 1-2 ULP, which is ~1e-11 on bound values of magnitude ~1e4. Use a
    # relative tolerance (with a small absolute floor for values near zero).
    for k in lo_ref:
        for a, b in ((lo_new[k], lo_ref[k]), (hi_new[k], hi_ref[k])):
            if math.isinf(a) or math.isinf(b):
                assert a == b, f"inf mismatch {a} vs {b} at {k}"
            else:
                assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (
                    f"dev {abs(a - b)} ({a} vs {b}) at {k}")
    assert set(lo_new) == set(lo_ref)


def _rand_point(r_components, rng):
    """Single antichain point; a component set to +inf is dropped from the
    min-summary, exercising the missing-component (`v is None`) path."""
    pt = {}
    for k in r_components:
        pt[k] = float("inf") if rng.random() < 0.15 else rng.uniform(-50, 200)
    return pt


# --------------------------------------------------------------------------- #
# Randomized equivalence property test
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(220))
def test_bound_matches_naive_reference(seed):
    rng = random.Random(seed)
    nf = rng.randint(1, 3)
    nr = rng.randint(1, 3)
    features = [f"f{i}" for i in range(nf)]
    r_components = [f"r{i}" for i in range(nr)]
    R = _R(r_components)
    n_obs = rng.randint(0, 30)

    if rng.random() < 0.5:
        ev = MonotonicityEvaluator(features, r_components)

        def ref(obs, q):
            return mono_bound_ref(obs, q, r_components)
    else:
        if rng.random() < 0.5:
            Lval = rng.choice([0.0, rng.uniform(0.1, 500)])
            Ld = {k: float(Lval) for k in r_components}
        else:
            Lval = {k: rng.choice([0.0, rng.uniform(0.1, 500)])
                    for k in r_components}
            Ld = {k: float(Lval[k]) for k in r_components}
        ev = LipschitzEvaluator(features, r_components, Lval)

        def ref(obs, q):
            return lip_bound_ref(obs, q, r_components, Ld)

    rng_scale = rng.choice([1.0, 5.0, 100.0])
    obs_feats = []
    for i in range(n_obs):
        vec = [rng.uniform(-rng_scale, rng_scale) for _ in range(nf)]
        obs_feats.append(vec)
        ev.observe(i, _cand(features, vec),
                   Antichain(R, [_rand_point(r_components, rng)]))

    queries = [[rng.uniform(-rng_scale, rng_scale) for _ in range(nf)]
               for _ in range(5)]
    # Boundary queries that land exactly on an observation.
    if obs_feats:
        queries += [list(obs_feats[rng.randrange(len(obs_feats))])
                    for _ in range(3)]

    for q in queries:
        lo_new, hi_new = ev.bound(_cand(features, q))
        lo_ref, hi_ref = ref(ev._obs, q)
        _assert_equal(lo_new, hi_new, lo_ref, hi_ref)


# --------------------------------------------------------------------------- #
# Explicit edge cases
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["mono", "lip"])
def test_edge_cases(kind):
    features, r_components = ["f0"], ["r0"]
    R = _R(r_components)
    if kind == "mono":
        ev = MonotonicityEvaluator(features, r_components)

        def ref(obs, q):
            return mono_bound_ref(obs, q, r_components)
    else:
        ev = LipschitzEvaluator(features, r_components, 3.0)

        def ref(obs, q):
            return lip_bound_ref(obs, q, r_components, {"r0": 3.0})

    # Empty history -> trivial bound.
    lo, hi = ev.bound({"f0": 1.0})
    assert lo == {"r0": 0.0} and hi == {"r0": float("inf")}

    # Single observation.
    ev.observe(0, {"f0": 0.5}, Antichain(R, [{"r0": 10.0}]))
    _assert_equal(*ev.bound({"f0": 0.5}), *ref(ev._obs, [0.5]))
    _assert_equal(*ev.bound({"f0": 2.0}), *ref(ev._obs, [2.0]))

    # Duplicate observation.
    ev.observe(1, {"f0": 0.5}, Antichain(R, [{"r0": 10.0}]))
    _assert_equal(*ev.bound({"f0": 0.7}), *ref(ev._obs, [0.7]))

    # reset() mid-stream clears both history and buffers.
    ev.reset()
    assert ev._obs == [] and ev._n == 0
    lo, hi = ev.bound({"f0": 1.0})
    assert lo == {"r0": 0.0} and hi == {"r0": float("inf")}
    ev.observe(0, {"f0": 2.0}, Antichain(R, [{"r0": 5.0}]))
    _assert_equal(*ev.bound({"f0": 3.0}), *ref(ev._obs, [3.0]))


@pytest.mark.parametrize("kind", ["mono", "lip"])
def test_pure_python_fallback_matches(kind):
    """The numpy-absent fallback path must equal the vectorized path."""
    features, r_components = ["f0", "f1"], ["r0", "r1"]
    R = _R(r_components)
    rng = random.Random(7)
    if kind == "mono":
        ev_np = MonotonicityEvaluator(features, r_components)
        ev_py = MonotonicityEvaluator(features, r_components)
    else:
        ev_np = LipschitzEvaluator(features, r_components, {"r0": 2.0, "r1": 5.0})
        ev_py = LipschitzEvaluator(features, r_components, {"r0": 2.0, "r1": 5.0})
    ev_py._np = None  # force pure-Python bound()

    for i in range(25):
        vec = [rng.uniform(0, 10), rng.uniform(0, 10)]
        pt = [{"r0": rng.uniform(0, 100), "r1": rng.uniform(0, 100)}]
        for ev in (ev_np, ev_py):
            ev.observe(i, _cand(features, vec), Antichain(R, pt))
        for _ in range(3):
            q = _cand(features, [rng.uniform(0, 10), rng.uniform(0, 10)])
            _assert_equal(*ev_np.bound(q), *ev_py.bound(q))
