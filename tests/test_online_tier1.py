"""
Smoke test for the Tier-1 online-solver additions:

- warm_start (list of indices and integer farthest-point seeds)
- picker strategies (lcb, ucb, random) including the tuple form
- GaussianProcessEvaluator

Each subtest verifies the feature works mechanically and (where
applicable) measures the effect on Pareto recovery against the
example 16 grid. The goal is to confirm the implementation, not to
benchmark the strategies; the report-grade numbers come from running
the dedicated example.
"""
from __future__ import annotations

import math
import os
import random
import sys

# Make codesign importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

# The online tier-1 tests reuse the example 16 model, which depends on
# numpy. Skip the whole module cleanly when numpy is unavailable (for
# example on a bare install without the 'online' or 'dev' extra).
try:
    import pytest
    pytest.importorskip("numpy")
except ImportError:  # pytest itself not installed (script-mode run)
    import importlib.util as _ilu
    if _ilu.find_spec("numpy") is None:
        print("numpy not installed; skipping online tier-1 tests")
        sys.exit(0)

# Re-use the example 16 model and grid wholesale.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ex16",
    os.path.join(os.path.dirname(__file__), "..", "examples", "16_online_doe.py"),
)
ex16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex16)

from codesign import (
    GaussianProcessEvaluator,
    LinearParametricEvaluator,
    LipschitzEvaluator,
    MonotonicityEvaluator,
    solve_online,
)


# ---------------------------------------------------------------------------
# Setup: build the grid and compute the true Pareto for comparison.
# ---------------------------------------------------------------------------


candidates = ex16.make_grid()
_all_results, true_pareto = ex16.exhaustive_baseline(candidates)
true_classes = {
    (round(p["cogs_per_g"], 2), round(p["footprint_m2"], 1))
    for p in true_pareto
}


def recovery(res):
    """Return the count of true-Pareto (cogs, footprint) classes that
    appear among the incumbent candidate set."""
    recovered = set()
    for i in res.incumbent_ids:
        out = ex16.simulate_run(
            candidates[i]["T_C"], candidates[i]["pH"],
            candidates[i]["glucose_mm"], candidates[i]["feed_start_day"],
        )
        recovered.add((round(out["cogs_per_g"], 2),
                       round(out["footprint_m2"], 1)))
    return len(true_classes & recovered), len(true_classes)


r_components = ["cogs_per_g", "footprint_m2"]
norm_features = ["T_norm", "pH_norm", "glu_norm", "feed_norm"]
F = {"target_titer": ex16.TARGET_TITER_G_L}


# ---------------------------------------------------------------------------
# Test 1. warm_start via explicit indices (manually picked corner runs).
# ---------------------------------------------------------------------------


# Pick four corners of the 4D normalised grid: (T=37, pH=7.05, glu=8, feed=2)
# and three opposite extremes. These are the "experienced scientist" seeds
# that the Monotonicity evaluator needs to become useful.
corner_picks = []
target_corners = [
    (37, 7.0, 9, 2),
    (33, 7.0, 9, 4),
    (37, 6.9, 5, 4),
    (33, 7.3, 13, 2),
]
for tgt in target_corners:
    for i, c in enumerate(candidates):
        if (c["T_C"], c["pH"], c["glucose_mm"], c["feed_start_day"]) == tgt:
            corner_picks.append(i)
            break

assert len(corner_picks) == 4, "expected 4 corner candidates to be present"

mono_ev = MonotonicityEvaluator(
    features=["pH_distance", "glucose_extremity", "feed_delay"],
    r_components=["cogs_per_g"],
)
res_warm = solve_online(
    ex16.make_dp, F,
    candidates=candidates,
    evaluator=mono_ev,
    budget=25,
    warm_start=corner_picks,
)
assert res_warm.n_evaluated == 25, f"expected 25 evals, got {res_warm.n_evaluated}"
# The first four picks should be exactly the corner indices.
assert res_warm.evaluated_ids[:4] == corner_picks, \
    f"warm-start order broken: {res_warm.evaluated_ids[:4]} vs {corner_picks}"
n_rec_warm, n_total = recovery(res_warm)
print(f"  T1 explicit warm-start: evals={res_warm.n_evaluated}, "
      f"recovery {n_rec_warm}/{n_total}")
# Without warm-start the Monotonicity evaluator scored 0/4 in the example.
# With warm-start it should beat that.
assert n_rec_warm >= 1, "warm-start should rescue Monotonicity from 0% recovery"


# ---------------------------------------------------------------------------
# Test 2. warm_start via integer (farthest-point heuristic).
# ---------------------------------------------------------------------------


lp_ev = LinearParametricEvaluator(
    features=norm_features, r_components=r_components,
    confidence=2.5, min_obs=10,
)
res_seeds = solve_online(
    ex16.make_dp, F,
    candidates=candidates,
    evaluator=lp_ev,
    budget=25,
    warm_start=6,
)
assert res_seeds.n_evaluated == 25
# Confirm the first 6 picks are mutually distant in the normalised feature
# space (each pair separated by at least sqrt(0.5) on average).
import numpy as np
seed_feats = np.array([
    [candidates[i][f] for f in norm_features]
    for i in res_seeds.evaluated_ids[:6]
])
dists = [
    np.linalg.norm(seed_feats[a] - seed_feats[b])
    for a in range(6) for b in range(a + 1, 6)
]
mean_d = float(np.mean(dists))
print(f"  T2 farthest-point seeds: mean pairwise distance = {mean_d:.3f}")
assert mean_d > 0.6, f"farthest-point seeds too close together: mean {mean_d:.3f}"


# ---------------------------------------------------------------------------
# Test 3. Picker strategies: lcb (default), ucb with kappa, random.
# ---------------------------------------------------------------------------


# All three should run to completion without error and produce the
# expected number of evaluations.
for picker_spec in ["lcb", "ucb", ("ucb", {"kappa": 1.5}), "random"]:
    lp_ev = LinearParametricEvaluator(
        features=norm_features, r_components=r_components,
        confidence=2.5, min_obs=10,
    )
    res = solve_online(
        ex16.make_dp, F,
        candidates=candidates,
        evaluator=lp_ev,
        budget=15,
        picker=picker_spec,
    )
    n_rec, _ = recovery(res)
    label = picker_spec if isinstance(picker_spec, str) else picker_spec[0]
    print(f"  T3 picker={label!r:>20}: evals={res.n_evaluated}, "
          f"recovery {n_rec}/{n_total}")
    assert res.n_evaluated == 15


# ---------------------------------------------------------------------------
# Test 4. GaussianProcessEvaluator returns sensible bounds and finishes.
# ---------------------------------------------------------------------------


gp_ev = GaussianProcessEvaluator(
    features=norm_features, r_components=r_components,
    length_scale=0.35, sigma_f=1.0, noise=1e-3,
    confidence=2.0, min_obs=5,
)
# Before any observation the bound should be the trivial fallback.
lo_pre, hi_pre = gp_ev.bound(candidates[0])
assert lo_pre["cogs_per_g"] == 0.0
assert math.isinf(hi_pre["cogs_per_g"])

res_gp = solve_online(
    ex16.make_dp, F,
    candidates=candidates,
    evaluator=gp_ev,
    budget=40,
)
n_rec_gp, _ = recovery(res_gp)
print(f"  T4 GP evaluator (budget 40): evals={res_gp.n_evaluated}, "
      f"recovery {n_rec_gp}/{n_total}")
assert res_gp.n_evaluated <= 40
assert n_rec_gp >= 1, "GP should recover at least one Pareto class"

# After observations the GP should produce finite upper bounds.
lo_post, hi_post = gp_ev.bound(candidates[0])
assert math.isfinite(hi_post["cogs_per_g"]), \
    "GP upper bound should be finite after observations"


# ---------------------------------------------------------------------------
# Test 5. Combined: GP + warm-start + ucb picker.
# ---------------------------------------------------------------------------


gp_combined = GaussianProcessEvaluator(
    features=norm_features, r_components=r_components,
    length_scale=0.35, sigma_f=1.0, noise=1e-3,
    confidence=2.0, min_obs=5,
)
res_combined = solve_online(
    ex16.make_dp, F,
    candidates=candidates,
    evaluator=gp_combined,
    budget=40,
    warm_start=8,
    picker=("ucb", {"kappa": 0.5}),
)
n_rec_combined, _ = recovery(res_combined)
print(f"  T5 GP + 8-seed warm-start + UCB(kappa=0.5): "
      f"evals={res_combined.n_evaluated}, recovery {n_rec_combined}/{n_total}")

# Phase markers in history should distinguish warm-start vs picker iterations.
phases = {h.get("phase") for h in res_combined.history}
assert phases == {"warm_start", "picker"}, f"unexpected phases: {phases}"


# ---------------------------------------------------------------------------
# Test 6. Invalid picker spec raises a clear error.
# ---------------------------------------------------------------------------


try:
    solve_online(
        ex16.make_dp, F,
        candidates=candidates[:5],
        evaluator=LinearParametricEvaluator(norm_features, r_components),
        budget=2,
        picker="nonexistent_strategy",
    )
except ValueError as e:
    print(f"  T6 invalid picker spec: ValueError raised correctly")
    assert "unknown picker" in str(e)
else:
    raise AssertionError("expected ValueError for bad picker spec")


print("\nAll Tier-1 online-solver tests passed.")
