# Release Polish Tracker — codesign-mcdp

**Purpose:** memory + tracking file for the multi-session effort to polish this repo for public GitHub release.
**Read this first** if continuing in a new session.

## Ground rules (user-set, do not violate)
- Orchestrator: Fable (Claude) — orchestration ONLY (dispatch, commits, tracker updates).
- All actual work is done by **Opus agents** (`model: "opus"` in Agent tool).
- Using Fable for anything other than orchestration requires explicit user approval.
- User wants structured, interactive reports — no walls of text.
- When unsure: ASK the user, never decide alone.

## User decisions (2026-07-19)
| Topic | Decision |
|---|---|
| Git workflow | Branch `release-polish`, one commit per work package (Fable commits, agents only edit) |
| Repo hygiene | `.gitignore` + untrack `graphify-out/` and LaTeX build artifacts; keep manual `.tex`, `Makefile`, `README.md`, final `.pdf` |
| GitHub URL | `https://github.com/cbriat/codesign-mcdp` |
| Notebooks | Regenerate + re-execute via `build_notebooks.py` after examples are polished (Phase 2) |

## Goal
Public-release quality:
1. Examples polished; **all figures clear — no text overlapping axes/plots**, good layout.
2. Manual (docs/manual/codesign-mcdp-manual.tex) crystal clear and professional; PDF rebuilt.
3. Core library docstrings/README/CHANGELOG/pyproject professional; repo hygiene done.
4. Notebooks regenerated to match polished examples.
5. Tests still pass.

## Work packages

### Phase 1 (parallel Opus agents)
| WP | Scope (files agent may edit) | Status |
|---|---|---|
| A — Packaging & core docs | README.md, CHANGELOG.md, pyproject.toml (+urls), .gitignore, untrack artifacts, codesign/*.py docstrings (NO behavior changes) | ✅ committed 3ebbf53; follow-up running (11 docstring fixes) |
| B — Plot infrastructure | codesign/viz.py, codesign/diagram.py only | ✅ committed 28adda8; follow-up running (to_dot dp1/dp2 bug) |
| C — Examples 01–08 | examples/01…08*.py only | ✅ done, committed 2d3e2e6 |
| D — Examples 09–16 | examples/09…16*.py only | ✅ done, committed c1a1a09 |
| E — Examples 17–23 | examples/17_car_codesign.py, 17_full_car.py, 18…23*.py only | ✅ committed 073846a |
| F — Manual | docs/manual/*.tex, Makefile, docs/manual/README.md; rebuild PDF | ✅ committed e46e126 |

### Phase 1.5
| WP | Scope | Status |
|---|---|---|
| G — Finish 17_full_car | **User decision:** finish it + renumber to examples/24_car_catalog_codesign.py; all new numbers must be INTERNET-SOURCED (no invented values), sources cited in comments | ✅ committed 17d5dc1 (rename itself was swept into 10d63d1). All 12 architectures feasible, sane magnitudes, 0.87s, 65 tests pass. SOURCES block in module docstring. |
| I — API error-handling | **User approved all 3:** ODE_DP dict-state clear error, loop_on clearer ValueError, Series upfront port-mismatch check. codesign/ + tests/ + CHANGELOG. No algorithm changes. **User quality bar:** every message = what went wrong + expected (with offending names interpolated) + how to fix; right exception type; earliest raise point; tests assert message content. | ✅ committed 10d63d1 (65 tests pass, was 55). Note: loop_on lives in mcdpl.py not sugar.py. Series check is subset (dp2.F ⊆ dp1.R), matching h() exactly. Left alone: ODE_DP invalid-mode silent fallthrough (4th fix, unapproved). |

## User decisions (2026-07-19, round 2)
| Topic | Decision |
|---|---|
| CI | Full coverage: all 24 examples + full pytest on the whole 3.9–3.12 matrix |
| Notebook 24 | Author NB_24 in build_notebooks.py style, execute with the rest |
| ODE_DP mode | Validate (4th error fix) |
| Error messages | FULL audit of every raise in codesign/ against the quality bar |
| 17_full_car | Finish + renumber to 24, numbers internet-sourced only (done, 17d5dc1) |

### Phase 2 (after Phase 1 committed)
| WP | Scope | Status |
|---|---|---|
| J — CI broaden | .github/workflows/test.yml: all 24 examples + full pytest | ✅ committed 906696b. Loop sanity-run locally: all 24 OK. Est. +~2min/leg CI. |
| K — Full error audit | Every raise in codesign/ to quality bar + ODE_DP mode validation + tests | ✅ committed 21fde5d. 112 raises audited: 75 OK, 41 upgraded, 0 type changes (2 KeyErrors kept deliberately). +32 tests → 97 pass. |
| V — Verification | Re-run ALL 24 examples + tests, visually re-verify every figure (esp. after WP-B viz changes), fix residual overlaps, check CI python matrix covers <3.12, verify no stale 17_full_car refs | ✅ committed b99e90c. 65 tests pass; all 24 examples exit 0, zero warnings; ALL 10 PNGs re-viewed clean; plot_antichain staircase verified correct; no stale 17_full_car refs. |
| H — Notebooks | Author NB_24 + regenerate + execute ALL notebooks via build_notebooks.py. MUST run after WP-K commits (notebooks execute against final library). | ✅ committed f4e204b. All 24 execute clean (37s), figures verified. NOTE: notebooks must be built WITHOUT MPLBACKEND=Agg (Agg strips embedded figures); needs `graphviz` pip package for NB_17. |

## Rules for agents (recorded so future sessions replicate)
- Agents edit ONLY files in their WP scope. Issues found outside scope → report back, do not fix.
- Figure inspection: render to PNG (MPLBACKEND=Agg) in scratchpad, view the PNG, fix, re-render, re-view. A figure is done only when visually verified.
- No API/numerics changes; polish = clarity, docs, figure quality.
- Agents do NOT run `git commit` (Fable commits per WP to avoid index races).

## Commit log
| Commit | WP | Notes |
|---|---|---|
| c1a1a09 | D | Examples 09–16. Changed: 09, 12, 13, 14, 16 (10/11/15 needed nothing). All 4 figures (13→2, 14→2) visually verified clean. |
| 2d3e2e6 | C | Examples 01–08, all 8 changed. Removed ~50-line dead `make_problem()` in 02, sys.path hack in 01, "1 pts" grammar in 05. 3 figures (05) verified. Convention: run via `python -m examples.NN`. |
| 28adda8 | B | viz.py: fixed broken dominated-region shading (spikes→staircase), legends, tight_layout, `xlabel=`/`legend_loc=` kwargs on plot_uncertainty, **py<3.12 f-string SyntaxError fix** (package was unimportable on 3.10!). diagram.py needed nothing. 55 tests pass. |
| 3ebbf53 | A | .gitignore(+graphify-out), pyproject urls (cbriat→corentinbriat!), README badges+fixes, CHANGELOG Keep-a-Changelog cleanup, __init__ NamedProduct→Ports. Nothing needed untracking. 55 tests pass. |
| 073846a | E | Examples 17–23: sys.path hacks removed, Run/Expected docstring notes, F1 figures (3) verified professional. tests/test_formula1.py 9 passed. |
| 35c4d36 | A2 | 11/11 docstring-scan fixes applied (Portss, Sphinx roles, hits, catalog, _repr_svg_). 55 tests pass. |
| 0e1bf3d | B2 | to_dot crash on raw Series/Parallel confirmed + fixed (dp1/dp2), render-verified. 55 tests pass. |
| e46e126 | F | Manual: 5 genuinely broken snippets fixed (incl. dict-state ODE crash, loop_on ValueError), 10× fixed-point typo (0.49→0.049 kg), stale counts/URLs, PDF rebuilt clean (41 pp, TeX Live 2025). All refs verified. |
| 10d63d1 | I | 3 validation fixes (ODE_DP dict state, loop_on, Series subset check) + 10 message-content tests. 65 pass. (Also swept in the 17_full_car→24 git mv staged by WP-G.) |
| 17d5dc1 | G | 24_car_catalog_codesign.py completed: 22-subsystem System, 2 Kleene loops, 12/12 architectures feasible, internet-sourced data w/ SOURCES block. |
| b99e90c | V | Verification: 24/24 examples clean, 10/10 figures re-viewed OK; 13_microgrid xlabel fix. |
| 906696b | J | CI: full pytest + all 24 examples on 3.9–3.12 matrix. |
| 21fde5d | K | Full error audit: 112 raises, 41 upgraded, ODE_DP mode validated, +32 tests → 97 pass. |
| f4e204b | H | Notebooks: NB_24 authored, all 24 regenerated + figure-verified. |

## Open issues / cross-scope findings
- Graph health note from graphify run: 132 dangling edges (benign, `Any`-type placeholders) — no action needed for release.
- **For WP-B (viz.py), from WP-D:** `plot_uncertainty` uses raw R-port name (`total_cost_usd`) as x-label — no human-readable label/units, not overridable from example. Legend hardcoded top-left, could collide with leftmost mode. Consider optional `xlabel=`/`legend_loc=` kwargs.
- **Awaiting user decision:** duplicate example number 17 (17_car_codesign.py + 17_full_car.py) — renumber, merge, or leave?
- **Forwarded to WP-E:** examples 17–23 still carry the `sys.path.insert` hack; asked WP-E to remove it and match the 01–08 docstring style (`python -m examples.NN` convention). → DONE in 073846a.
- **17_full_car.py — RESOLVED by user:** finish + renumber to 24_car_catalog_codesign.py, numbers must be sourced from the internet (never invented). → WP-G dispatched.
- ~~In follow-up (WP-A resumed): 11 docstring fixes~~ → DONE, committed 35c4d36 (11/11 applied, 55 tests pass).
- ~~In follow-up (WP-B resumed): to_dot dp.left/right bug~~ → CONFIRMED real crash (AttributeError on raw Series/Parallel trees), fixed in 0e1bf3d, render-verified, 55 tests pass.
- **From WP-B:** pre-existing, unfixed: `to_dot(name=...)` interpolates name verbatim into `digraph {name} {` — invalid dot if name has spaces/parens. Minor; Phase-2 candidate.
- **From WP-F, API awkwardness (library-code candidates, need user approval to touch behavior):** (1) ODE_DP silently crashes on dict state — wants clearer error; (2) MCDP.loop_on needs axis in both provides+requires — non-obvious, wants better error; (3) Series doesn't validate dp1.R == dp2.F upfront (fails later with bare KeyError); (4) two "drone" models converge to 0.5492 vs 0.5602 kg — both cited in manual, could confuse readers.
- **From WP-A, for user at release time:** pyproject author email is corentin@briat.info (vs protonmail contact) — confirm intended. `Development Status :: 3 - Alpha` classifier — bump to Beta? Verify arXiv:2604.22624 resolves.
- **From WP-E:** F1 figure PNGs write to CWD (repo root) when run via `-m` — gitignored, but maybe examples should write into an outputs/ dir (like 13_microgrid) for consistency. Phase-2 candidate.
- **From WP-V — CI coverage gap:** matrix (3.9–3.12) is fine, but workflow only runs examples 01–04+06–08 and test_smoke, not full pytest — that's how the py<3.12 SyntaxError shipped. Proposal: run all 24 examples + `pytest tests/ -q`. AWAITING USER.
- **From WP-V — no notebook for example 24:** notebooks are bespoke hand-authored prose (not mechanical), set stops at 23. Author NB_24? AWAITING USER.
- (agents append findings here via Fable)

## Status: RELEASED — merged to main and pushed 2026-07-19

15 commits on `release-polish`, merged (--no-ff, merge commit 387acc0) into main and PUSHED to github.com/**cbriat**/codesign-mcdp (both branches). Suite: 97 tests pass. All 24 examples clean, figures verified, manual rebuilt (41 pp), notebooks regenerated incl. NB_24, CI full coverage.

### Final user decisions (round 3, 2026-07-19)
| Topic | Decision |
|---|---|
| GitHub account | **cbriat** (NOT corentinbriat — earlier guess was wrong; all URL rewrites reverted in 8ead056) |
| Publish | Merge + push main directly (no PR) — DONE |
| Author email | corentin@briat.info confirmed correct |
| Classifier | 4 - Beta (done, 8ead056) |
| arXiv:2604.22624 | Confirmed correct by user |
| RELEASE_TRACKER.md | Stays untracked (not in public repo) |

## Phase 3 (post-release): replicate arXiv:2604.22624 examples
**User request (2026-07-19):** check whether the examples of arXiv:2604.22624 (Alharbi, Dahleh & Zardini — basis of codesign/online.py) are implemented in examples/; implement the missing ones and VALIDATE by comparing to the paper's reported results. Fable coordinates, Opus agents work.

**P1 findings (2026-07-19):** Paper = "Compositional Online Learning for Multi-Objective System Co-Design". NONE of its 4 experiments implemented here. E1 (synthetic monotone, Table I) + E2 (synthetic Lipschitz, Table II) = statistically reproducible (seeds/grid unpublished → validate orderings/margins, not exact digits). E3 (AMoD mobility) + E4 (multi-robot) = BLOCKED on external models (Zardini TNSE 2022 LP; arXiv:2604.21894 sim); paper has no code/data repo. CORRECTNESS FINDING: LinearParametricEvaluator (OLS±band) is not a valid optimistic evaluator — can wrongly eliminate optima; paper's version is a certified confidence-polytope LP (eq 27). Paper text saved at scratchpad/paper.txt.

**User decisions (round 4):** one combined example 25; pymoo EA panel optional-gated; E3/E4 documented as blocked (no fakes); LinearParametricEvaluator fixed to certified version. Work on branch `paper-replication`.

| WP | Scope | Status |
|---|---|---|
| P1 — Gap analysis | Read paper, audit coverage, replicability, plan | ✅ done (findings above) |
| P2a — examples/25_online_paper_benchmarks.py | E1+E2 map families, Algorithm-1 loop (Halton/rejection/δ) in-example, 2-D HVD, Halton+random baselines, optional pymoo EAs, PASS/FAIL validation vs paper claims, CI loop entry | ✅ committed ba1152a. Claims (a),(b) PASS 8/8+8/8; (c) skipped (no pymoo). Documented deviation: monotone family as FixFunMinRes(f) query (α=0.35) — literal unconstrained min is degenerate for monotone maps. 5.3s default, figures verified. |
| P2b — Certified LinearParametricEvaluator | eq 27 confidence-polytope LP via scipy linprog, API-compatible, property tests (optimism guarantee), consumer check | ✅ committed 2bbe4f8. PROOF the old bug was real: example 14 missed Pareto point r033, now recovered (antichain 4→5). Example 16 (non-linear model): degrades safely to no-info (0/4) instead of wrong eliminations. Suite 112 pass. |
| P2c — Manual/example prose sync | Manual §online passages stale (still describe OLS band + the now-fixed wrongly-eliminate risk + "empirically close" claim); example 14/16 prose check; PDF rebuild | ✅ committed 1e0cf0d (42 pp PDF) + 1326bd5 (confidence= kwargs dropped, zero-warning verified with -W error). |
| P4 — Perf: MonotonicityEvaluator | **User approved:** optimize bound() from O(history) per call (O(budget²) total, blocks paper-scale 4000-iter runs). Must produce IDENTICAL results (property-tested), benchmarked before/after. | queued — dispatch AFTER P3 lands (avoid editing online.py mid-validation) |
| P5 — Notebooks NB_14/16/25 | NB_14/16 embed stale outputs after evaluator change; **user approved NB_25** (paper claims, HVD curves, PASS/FAIL panel, NB_17–24 style). Regen WITHOUT MPLBACKEND=Agg. | queued — LAST, after P4 (notebooks execute against final library) |
| P3 — Independent validation | Code-review vs paper, verify degeneracy claim, scaled runs (2 seeds), pymoo installed env-only for claim (c), honesty audit | ✅ done + fix committed c1d37d2. **VERDICT: claims (a),(b) reproduce ROBUSTLY** (Ours<Halton 8/8 both families both seeds; Ours best-of-all on every monotone instance — stronger than paper). **Claim (c) does NOT reproduce** at scale: EAs match/beat Ours on Lipschitz; MOEA/D low-variance, no blowups. Hypotheses: continuous-EA vs discrete-pool asymmetry, smaller budget, untuned EAs. Defect found+fixed: MOEA/D silently absent from monotone panel (pymoo constraint rejection). Degeneracy claim independently confirmed. Honest (c) wording committed c633a14 — prints "NOT REPRODUCED", families separated, blowup-based test, docstring Validation status note. |
| P4 — Perf: evaluators | O(budget²) fix: vectorize bound() over observation history (numpy path + pure-Python fallback), equivalence property-tested, benchmarked N=500/2000/4000 | ✅ committed ab594db. 6–17× speedup (N=4000: 6-8s→0.5-0.7s). Bit-exact: 3169 scenarios, deviation 0.0. Suite 336 pass (+224 regression tests). Honest note: constant-factor win, still O(budget²) asymptotically. |
| P5 — Notebooks | NB_14/16 stale outputs+prose (old OLS-band story), author NB_25 (incl. honest (c) NOT REPRODUCED), regen all 25, no MPLBACKEND=Agg | ✅ committed 7c2cead. 25/25 clean, 0 errors/warnings, figures verified (NB_25's own figure visually shows MOEA/D beating Ours on Lipschitz — corroborates the NOT REPRODUCED verdict). |

### Leftovers — ALL THREE user-approved (round 5)
| WP | Task | Status |
|---|---|---|
| L1 — to_dot name sanitization | codesign/viz.py + tests | ✅ committed edb8144 (conditional quoting, lossless; default byte-identical) |
| L2 — F1 PNGs → outputs/ | examples/23 (+ NB_23 regen if paths named) | ✅ committed 3f5b151 (NB_23 never names paths — untouched) |
| L3 — Unify drone models | 11/12 nominal must reduce exactly to canonical 01/06/07 model (0.5492 kg); manual numbers + PDF; NB regen; uncertainty spreads must stay meaningful | ✅ committed 8e9b021 (2.0e6×0.9=1.8e6 exactly; spreads preserved) |

## Status: PHASE 3 SHIPPED — merged to main (b80cd66) and pushed 2026-07-19

Suite: 336 tests pass. 25 examples, 25 notebooks, manual 42 pp. The paper's core claims (a)/(b) replicate robustly; claim (c) honestly reported NOT REPRODUCED (see example 25 docstring). Certified evaluator fixed a real over-elimination bug (ex 14 had been missing a Pareto point). 6–17× evaluator speedup.

## Phase 4: full manual expansion (branch manual-expansion)
**User decision (2026-07-19):** manual becomes a FULL REFERENCE — every public symbol at reference-card depth (signature + runnable snippet + parameter notes + errors raised) + worked-analysis section for every example 01–25. Expected growth ~60–80 pp.

| WP | Scope | Status |
|---|---|---|
| M0 — Audit & architecture | API inventory, coverage map, manual structure design, fragment protocol (scratchpad/wpm/PROTOCOL.md), ~9 writer package manifests | ✅ done. 105 symbols: 12 FULL / 73 PARTIAL / 20 NONE. Structure: new \section{API Reference} before §Worked Examples; example subsections get Model/Code/Output/Reading beats. Est. manual 42→~100–110 pp. **User green-lit full plan.** |
| M1–M9 — Writers | 6 API packages (A: posets/antichains, B: dp/primitives/composition/solver, C: builders/diagram/viz, D: uncertainty/online, E: temporal/dynamic/sequential, F: state/vector/online_codesign) + 3 example packages (EX1 01–08, EX2 09–17, EX3 18–25); fragments to scratchpad/wpm/fragments/; every snippet must RUN | 🟡 all 9 dispatched in parallel |
| M-int — Integration | merge fragments into .tex, consistency pass, labels/refs, PDF rebuild, page-count report | ✅ committed d761bf2. **113 pages**, 0 errors/undefined refs/dup labels. API Reference at .tex L1894–5594. All stale text fixed. |
| M10 — Writer-found fixes | **User approved all:** to_dot .inner edges, Naturals unit format, CatalogDP guards, detect_resets consistency + 4 docstring cleanups + manual card sync + PDF rebuild | ✅ committed 38f7518. All 4 fixes + 4 cleanups applied, nothing deferred. 343 tests pass. |

**User decisions (round 6):** all 4 library fixes + all cleanups approved; publish = merge manual-expansion → main + push (after M10 lands).

## Phase 5: go-public preparation (branch release-0.2.0)
**User approved (2026-07-19):** fresh-user preflight + version 0.2.0 + CITATION.cff + Citing sections (README & manual) + repo description text.

| WP | Scope | Status |
|---|---|---|
| R1 — Fresh-user preflight | clean clone → fresh venvs → README install path, wheel build+install, tests, examples, notebooks scan, README link audit, hygiene sweep (personal paths!), no-extras degradation | ✅ READY-WITH-NITS: everything passed except (1) 3 test failures on py3.13+numpy2.5 (float tol — FIXED c79d758) and (2) version 0.1.0 stragglers (FIXED in R2). Hygiene clean, no personal paths, README snippet byte-exact from wheel. |
| R2 — Release prep | pyproject 0.2.0, CHANGELOG cut to [0.2.0], CITATION.cff (+Censi ref), Citing sections in README + manual (PDF rebuild), recommended GitHub description/topics | ✅ committed f6932ed. CITATION.cff cffconvert-validated. CI matrix +3.13 (user approved). |

## CI RED on d0441bd → FIXED in a926193 (pushed; watch the rerun)
All 5 legs failed with ONE identical test: test_viz_smoke "invalid dot" — the test helper swallowed graphviz.ExecutableNotFound (runners have the python package but no dot binary) as "invalid dot" instead of falling through to the structural check. Test-only fix, verified pre/post in a dot-less venv with CI's numpy 2.2.6; suite 343. Positive signal from the red run: the py3.13 float-tolerance fix passed on the real 3.13 leg. Deliberate: did NOT add apt-get graphviz to CI — the structural fallback is the intended dot-less path.
**Unconfirmed flag from WP-R3 (do not act without verification):** claims `X | Y` unions in temporal/primitives/solver lack future-import and "would raise on 3.9" — EMPIRICALLY CONTRADICTED by the 3.9 CI leg passing 342 tests including temporal ones; likely mis-grep (quoted annotations or future import present). Re-check only if 3.9 ever breaks on import.

## Status: v0.2.0 SHIPPED — merged to main (d0441bd), tag v0.2.0 pushed 2026-07-19

### GitHub About box — paste this (from R2):
**Description:** A from-scratch Python library for Monotone Co-Design Problems (MCDPs) after Censi (2015): compose design problems over functionality/resource posets with series/parallel/feedback, solve Pareto fronts by Kleene fixed-point iteration. Ships uncertainty, online learning, temporal & sequential co-design, 25 worked examples, and a 113-page manual.
**Topics:** co-design, mcdp, pareto-optimization, fixed-point-iteration, poset, lattice-theory, monotone, python, systems-engineering, optimization, design-automation, operations-research

### User go-public checklist (only you can do these)
1. ~~GitHub Actions green~~ ✅ CONFIRMED: a926193 green on all 5 legs (2026-07-20). Tag v0.2.0 moved to a926193 and re-pushed.
2. Paste the About description + topics above.
3. Create the Release page from tag v0.2.0 (release notes: CHANGELOG [0.2.0] section).
4. Post-publication: click through the external URLs listed in R1's report (badges, repo links).
5. Flip repository visibility to public.

## Status: PHASE 4 SHIPPED — merged to main (b403cb1) and pushed 2026-07-19
Manual: 42 → **113 pages** (full API reference: 105 symbol cards / 20 modules, every snippet executed; worked analyses for all 25 examples). 4 library bugs found by documentation fact-checking, all fixed. Suite: **343 tests passing**.

**Writer status / findings (for M-int and follow-ups):**
- M1 PKG_A ✅ — ref_posets.tex (6 sym) + ref_antichains.tex (1 sym, 11-method table); 7 snippets ran; compiles under real preamble (4 pp). FINDINGS: (a) existing manual L440 documents Antichain.union_min as instance method — it's a CLASSMETHOD union_min(cls, poset, antichains) (fragment corrects; integrator must drop the old wrong text); (b) LIBRARY BUG CANDIDATE: Naturals accepts unit= but format() ignores it (Reals shows units, Naturals doesn't) — needs user decision; (c) manual L429 points: tuple → actually list; (d) integrator note: \topp is math-mode only — check other fragments use $...$ around it.
- M2 PKG_B ✅ — ref_dp/primitives/composition/solver.tex (24 sym + method); 21/21 snippets pass; compiles (8 pp). FINDINGS: (a) LIBRARY VALIDATION CANDIDATE: CatalogDP silently accepts empty catalogue / duplicate entries (no guards; infeasible query returns ⊤ singleton) — needs user decision; (b) solve() never raises on non-convergence (status field instead) — documented explicitly, not a bug; (c) Loop projecting to _Unit prints Antichain[*] — documented around, not a bug.
- M7 EX1 ✅ — ex_01..08.tex, all runs verified, compiles. FINDINGS for M-int: (a) GLOBAL literate= needs {⊤}{{$\top$}}1 {×}{{$\times$}}1 {→}{{$\rightarrow$}}1 + extendedchars=true (fatal UTF-8 error otherwise; broadcast sent to all writers, M7 patched its 3 affected listings locally); (b) figure path convention: used ../images/kleene_trace_c4.png — integrator confirm; (c) ex05 writes PNGs to outputs/ but committed figures live in docs/images/ — divergence noted in prose.
- M8 EX2 ✅ — ex_09..17.tex, all 9 runs verified (14: 122/200 all-5; 16: LP 0/4 safe; drone numbers all match post-L3 reality), compiles clean. Manifest filename mismatch noted (PKG_EX2_examples_09_17.md). Ex 15/13 flagged HEAVY but actually fast (~0.03s/1s) — only 14 (~17s) and 16 (~26s) are slow.
- M3 PKG_C ✅ — 6 fragments (14 sym + 19 methods), all snippets ran, compiles (7 pp). FINDINGS: (a) LIBRARY BUG CANDIDATE: to_dot's _extract_module_refs walks .left/.right/.arg/.args but not .inner (Func/Neg operands) → constraints like sqrt(actuator.power) silently DROP the edge in to_dot output; draw_system handles .inner correctly. Latent emitter inconsistency — needs user decision; (b) module-less System renders as bare leaf (cosmetic, documented).
- M4 PKG_D ✅ — ref_uncertainty (10 sym) + ref_online (7 sym), 17/17 snippets ran, compiles clean. FINDINGS (docstring-cleanup candidates): (a) OnlineResult docstring lists stale history keys ({pick,antichain,eliminated_now,remaining} vs actual {pick,antichain,remaining,evaluated,eliminated,phase}); (b) picker named "lcb" in code but some docstrings/manual prose say "UCB"; (c) dead helper _import_scipy_stats_or_die unused (scipy imported inline).
- M9 EX3 ✅ — ex_18..25.tex, all 8 runs verified, compiles (10 pp). 24/25 first-ever coverage; 25 quotes the honest NOT-REPRODUCED verdict verbatim. Integrator notes: needs sibling labels ref:online:{monotonicityevaluator,lipschitzevaluator} (M4 wrote those cards — verify exact label spelling), plus M5/M6-owned ref:temporal/dynamic/sequential/state/vector_dp/online_codesign cards. Also: docs/manual PDF shows as modified in git status on this branch — check/restore at integration (writers were read-only; likely a stray make from an earlier phase).
- M6 PKG_F ✅ (after session-limit interruption + resume) — ref_state (8) + ref_vector_dp (6) + ref_online_codesign (4), 20/20 snippets ran incl. verbatim upgraded-error reproductions, compiles (9 pp). All sibling-linked labels minted exactly. FINDINGS: run_online_codesign never raises on infeasible step (ControlStep with cost=inf; documented); Axis is plain class not ABC (instantiates, fails on call); ContinuousAxis.snap ties snap down.
- M5 PKG_E ✅ (after session-limit kill + resume) — ref_temporal (5) + ref_dynamic (8) + ref_sequential (12), 24/24 snippets ran, compiles (13 pp). First-ever docs for most cards. FINDINGS: detect_resets silently treats no-candidates stage as vacuous reset (inconsistent with sibling solvers' upgraded ValueError); sum/join_combine can KeyError on mismatched axes; rollout never raises on infeasibility (documented).
- **ALL 9 WRITERS DONE.** M-int integration dispatched (merge 45 fragments, global literate fix, stale-text removal, label integrity, 2-pass build, expect ~100-110 pp).

### Ideas parked for future sessions
- Compositional Algorithm 2 (multigraph bound propagation) — would enable E3/E4-style demos on synthetic blocks; a real library project.
- Halton/rejection/δ sampling as a first-class solve_online mode (currently example-level only).
- Sub-linear-per-call dominance structures for evaluators (current fix is constant-factor).

**Publish decision (round 5):** when everything lands → merge paper-replication into main (--no-ff) + push both branches (same as release round).

## Session log
- **2026-07-19 (session 1):** graphify graph built. User requested release polish. Decisions taken (tables above). Branch created. Phase 1 (A–F) + follow-ups (A2 docstrings, B2 to_dot bug) done. User decisions: 17_full_car → finished as 24 with internet-sourced numbers (G); 3+1 error-handling fixes (I, K); full error audit (K, 41 raises upgraded); CI full coverage (J); verification pass (V); notebooks + NB_24 (H). All committed; see commit log.
