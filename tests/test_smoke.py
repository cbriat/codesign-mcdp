"""Quick smoke test verifying posets, antichains, and basic DPs."""
import sys
sys.path.insert(0, "/home/claude")

from codesign import (
    Reals, Naturals, Ports,
    Antichain,
    AlgebraicDP, FunctionDP, CatalogDP, CatalogEntry,
    series, par, loop,
    solve,
    System, Module,
    Box, Stochastic, GaussianCopula,
    LipschitzEvaluator, MonotonicityEvaluator,
    LinearParametricEvaluator, solve_online,
)


def test_posets():
    print("=== Posets ===")
    r = Reals(unit="kg")
    assert r.leq(1.0, 2.0)
    assert not r.leq(2.0, 1.0)
    assert r.is_top(float("inf"))
    print(f"Reals format: {r.format(1.5)}, top: {r.format(r.top())}")

    np_p = Ports({"a": Reals(), "b": Reals()})
    x = np_p.make(a=1.0, b=2.0)
    y = np_p.make(a=1.5, b=2.5)
    z = np_p.make(a=1.5, b=1.0)
    assert np_p.leq(x, y)
    assert not np_p.leq(x, z)
    assert not np_p.leq(z, x)
    print(f"Product order works: {np_p.format(x)} <= {np_p.format(y)}")


def test_antichain():
    print("\n=== Antichains ===")
    p = Ports({"cost": Reals(), "weight": Reals()})
    pts = [
        {"cost": 100.0, "weight": 200.0},
        {"cost": 200.0, "weight": 100.0},
        {"cost": 150.0, "weight": 150.0},
        {"cost": 1000.0, "weight": 1000.0},
    ]
    a = Antichain.from_set(p, pts)
    print(f"Pareto front of 4 points: {a}")
    assert len(a) == 3


def test_algebraic_dp():
    print("\n=== AlgebraicDP ===")
    F = Ports({"capacity": Reals(unit="J")})
    R = Ports({"mass": Reals(unit="kg")})
    battery = AlgebraicDP(F, R, {"mass": lambda f: f["capacity"] / 1.8e6})
    res = solve(battery, {"capacity": 3.6e6})
    print(f"Battery for 3.6 MJ: {res.antichain}")
    pt = list(res.antichain)[0]
    assert abs(pt["mass"] - 2.0) < 1e-9


def test_catalog_dp():
    print("\n=== CatalogDP ===")
    F = Ports({"torque": Reals(), "speed": Reals()})
    R = Ports({"cost": Reals(), "weight": Reals()})
    catalog = [
        CatalogEntry(provides={"torque": 1.0, "speed": 100.0}, costs={"cost": 50.0, "weight": 100.0}, name="m1"),
        CatalogEntry(provides={"torque": 2.0, "speed": 150.0}, costs={"cost": 90.0, "weight": 200.0}, name="m2"),
        CatalogEntry(provides={"torque": 3.0, "speed": 200.0}, costs={"cost": 60.0, "weight": 300.0}, name="m3"),
    ]
    motors = CatalogDP(F, R, catalog)
    res = solve(motors, {"torque": 0.5, "speed": 50.0})
    print(f"Motors for low load: {res.antichain}")
    res2 = solve(motors, {"torque": 2.5, "speed": 180.0})
    print(f"Motors for high load: {res2.antichain}")


def test_series():
    print("\n=== Series composition ===")
    F1 = Ports({"capacity": Reals()})
    R1 = Ports({"mass": Reals()})
    battery = AlgebraicDP(F1, R1, {"mass": lambda f: f["capacity"] / 1.8e6})

    F2 = Ports({"mass": Reals()})
    R2 = Ports({"cost": Reals()})
    pricing = AlgebraicDP(F2, R2, {"cost": lambda f: f["mass"] * 10.0})

    chain = series(battery, pricing)
    res = solve(chain, {"capacity": 3.6e6})
    print(f"Battery + pricing for 3.6 MJ: {res.antichain}")


def test_parallel():
    print("\n=== Parallel composition ===")
    F1 = Ports({"capacity": Reals()})
    R1 = Ports({"mass": Reals()})
    battery = AlgebraicDP(F1, R1, {"mass": lambda f: f["capacity"] / 1.8e6})

    F2 = Ports({"lift": Reals()})
    R2 = Ports({"power": Reals()})
    actuator = AlgebraicDP(F2, R2, {"power": lambda f: 10.0 * f["lift"] ** 2})

    combo = par(battery, actuator)
    res = solve(combo, {"capacity": 3.6e6, "lift": 5.0})
    print(f"Battery+actuator: {res.antichain}")


def test_loop_simple():
    print("\n=== Simple loop ===")
    # f -> r where r = 2*sqrt(f) + 1, then close r <= f.
    # Fixed-point: c = 2*sqrt(c) + 1, which gives c = 3 + 2*sqrt(2) ~ 5.83.
    F = Ports({"capacity": Reals()})
    R = Ports({"capacity": Reals()})

    def h_fn(f):
        c = f["capacity"]
        return {"capacity": 2.0 * (c ** 0.5) + 1.0}

    inner = FunctionDP(F, R, h_fn)
    looped = loop(inner, axis="capacity")
    res = solve(looped, None, record_trace=True)
    print(f"Loop result (expect ~5.83): {res.antichain}")
    print(f"Iterations: {res.iterations}")


def test_system():
    """Two modules wired via System: a producer feeding a consumer
    in a feedback loop. The producer doubles its input; the consumer
    halves its input plus 1. Closing the loop produces fixed point x=2.
    """
    print("\n=== System (modular composition) ===")
    producer = AlgebraicDP(
        F=Ports({"in": Reals()}),
        R=Ports({"out": Reals()}),
        equations={"out": lambda f: 0.5 * f["in"] + 1.0},
        name="producer",
    )
    consumer = AlgebraicDP(
        F=Ports({"need": Reals()}),
        R=Ports({"cost": Reals()}),
        equations={"cost": lambda f: f["need"]},
        name="consumer",
    )

    sys = System("mini")
    sys.provides("driver", unit="x")
    sys.requires("price", unit="$")
    sys.add("producer", producer)
    sys.add("consumer", consumer)
    sys.constrain("producer.in", lambda x: x["driver"] + x["consumer.cost"])
    sys.constrain("consumer.need", lambda x: x["producer.out"])
    sys.constrain("price", lambda x: x["consumer.cost"])

    dp = sys.build()
    res = solve(dp, {"driver": 1.0}, max_iter=100)
    price = list(res.antichain.points)[0]["price"]
    # Fixed point: price = 0.5 * (driver + price) + 1 -> price = driver + 2 = 3.0
    print(f"System fixed point (expect 3.0): price = {price:.4f}, "
          f"iters = {res.iterations}, feasible = {res.feasible}")
    assert abs(price - 3.0) < 1e-3, f"got {price}"
    assert res.feasible


def test_module_and_operator_dsl():
    """Same model as test_system, but built with Module subclasses and
    the operator-overloaded constraint syntax. Should produce the same
    result, confirming the new sugar layer is consistent with the
    string-based API."""
    print("\n=== Module + operator DSL ===")

    class Producer(Module):
        F = {"in_signal": Reals()}
        R = {"out_signal": Reals()}
        def h(self, f):
            return {"out_signal": 0.5 * f["in_signal"] + 1.0}

    class Consumer(Module):
        F = {"need": Reals()}
        R = {"cost": Reals()}
        def h(self, f):
            return {"cost": f["need"]}

    sys = System("mini2")
    driver = sys.provides("driver", unit="x")
    price = sys.requires("price", unit="$")
    p = sys.add("producer", Producer())
    c = sys.add("consumer", Consumer())

    p.in_signal >= driver + c.cost
    c.need      >= p.out_signal
    price       >= c.cost

    dp = sys.build()
    res = solve(dp, {"driver": 1.0}, max_iter=100)
    price_val = list(res.antichain.points)[0]["price"]
    print(f"Operator DSL fixed point (expect 3.0): price = {price_val:.4f}, "
          f"iters = {res.iterations}, feasible = {res.feasible}")
    assert abs(price_val - 3.0) < 1e-3, f"got {price_val}"
    assert res.feasible


def test_dsl_type_errors():
    """The DSL should refuse silly mistakes loudly."""
    print("\n=== DSL type-error guards ===")

    class M(Module):
        F = {"x": Reals()}
        R = {"y": Reals()}
        def h(self, f):
            return {"y": f["x"]}

    sys = System("g")
    sys.requires("z")
    z = sys.requires  # placeholder
    a = sys.provides("a")
    m = sys.add("m", M())

    # outer F can't be constrained
    caught = False
    try:
        a >= 1
    except TypeError as e:
        caught = True
        assert "outer F" in str(e), str(e)
    assert caught, "expected TypeError when constraining an outer F port"

    # module R can't be constrained externally
    caught = False
    try:
        m.y >= 1
    except TypeError as e:
        caught = True
        assert "module R" in str(e), str(e)
    assert caught, "expected TypeError when constraining a module R port"

    # using a module F port as a value in an expression should fail at compile
    caught = False
    try:
        sys.requires("w")
        # Build will fail because w has no constraint, but also because the
        # expression uses an F port as a value:
        from codesign.sugar import compile_expr
        compile_expr(m.x + 1)
    except ValueError as e:
        caught = True
        assert "F port" in str(e), str(e)
    assert caught, "expected ValueError when an F port appears in a demand"

    print("All DSL guard checks passed.")


def test_solver_trace_and_status():
    """The solver should expose status / trace / verbose / on_iteration."""
    print("\n=== Solver trace and status ===")

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass": Reals(unit="kg")}
        def h(self, f):
            return {"mass": f["capacity"] / 1.8e6}

    class Actuator(Module):
        F = {"lift_force": Reals(unit="N")}
        R = {"power": Reals(unit="W")}
        def h(self, f):
            return {"power": 10.0 * f["lift_force"] ** 2}

    sys = System("drone")
    endurance = sys.provides("endurance", unit="s")
    extra_p   = sys.provides("extra_power", unit="W")
    extra_pl  = sys.provides("extra_payload", unit="kg")
    total_m   = sys.requires("total_mass", unit="kg")
    b = sys.add("battery",  Battery())
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_p) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_pl)
    total_m       >= b.mass + extra_pl
    drone = sys.build()

    # Trace + callback
    captured = []
    res = solve(
        drone,
        {"endurance": 60.0, "extra_power": 1.0, "extra_payload": 0.10},
        trace=True, max_iter=50,
        on_iteration=captured.append,
    )
    assert res.status == "converged"
    assert res.feasible is True
    assert res.trace is not None and len(res.trace) >= 2
    assert len(captured) == len(res.trace)
    # delta is None at iteration 0, a numeric float after that
    assert res.trace[0].delta is None
    assert isinstance(res.trace[1].delta, float)
    # Backwards-compatibility alias
    assert res.converged is True

    # Max-iter case: deliberately too few iterations
    res2 = solve(
        drone,
        {"endurance": 60.0, "extra_power": 1.0, "extra_payload": 0.10},
        max_iter=3,
    )
    assert res2.status == "max_iter"
    assert res2.iterations == 3

    print(f"status={res.status}, iters={res.iterations}, trace_len={len(res.trace)}")
    print(f"early-cut status={res2.status} after {res2.iterations} iters")


def test_uncertainty_box():
    """Box uncertainty: worst case is at the declared-worst corner."""
    print("\n=== Uncertainty: Box (set-based) ===")

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass":     Reals(unit="kg")}
        def __init__(self, specific_energy=1.8e6, efficiency=0.85):
            self.specific_energy = specific_energy
            self.efficiency = efficiency
            super().__init__()
        def h(self, f):
            return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}

    class Actuator(Module):
        F = {"lift_force": Reals(unit="N")}
        R = {"power":      Reals(unit="W")}
        def h(self, f):
            return {"power": 10.0 * f["lift_force"] ** 2}

    b = Battery()
    b.uncertain_set = Box(
        specific_energy=(1.6e6, 2.0e6, "more_is_better"),
        efficiency=(0.80, 0.90, "more_is_better"),
    )

    sys = System("drone")
    endurance = sys.provides("endurance", unit="s")
    extra_p   = sys.provides("extra_power", unit="W")
    extra_pl  = sys.provides("extra_payload", unit="kg")
    total_m   = sys.requires("total_mass", unit="kg")
    b_h = sys.add("battery",  b)
    a_h = sys.add("actuator", Actuator())
    b_h.capacity    >= (a_h.power + extra_p) * endurance
    a_h.lift_force  >= 9.81 * (b_h.mass + extra_pl)
    total_m         >= b_h.mass + extra_pl
    drone = sys.build()

    f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}

    nominal = solve(drone, f, max_iter=200)
    nominal_mass = list(nominal.antichain.points)[0]["total_mass"]

    res = solve(drone, f, uncertainty=["worst_case"], max_iter=200)
    worst_mass = list(res.worst_case.antichain.points)[0]["total_mass"]

    # The worst-case mass should be strictly larger than the nominal.
    assert worst_mass > nominal_mass, (worst_mass, nominal_mass)

    # And the battery's parameters should have been restored to their
    # nominal values after the solve.
    assert b.specific_energy == 1.8e6, b.specific_energy
    assert b.efficiency == 0.85, b.efficiency

    print(f"nominal mass={nominal_mass:.4f} kg, worst-case mass={worst_mass:.4f} kg")


def test_uncertainty_stochastic():
    """Stochastic uncertainty with a Gaussian copula: summaries are sane."""
    print("\n=== Uncertainty: Stochastic (MC) ===")
    from scipy import stats

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass":     Reals(unit="kg")}
        def __init__(self, specific_energy=1.8e6, efficiency=0.85):
            self.specific_energy = specific_energy
            self.efficiency = efficiency
            super().__init__()
        def h(self, f):
            return {"mass": f["capacity"] / (self.specific_energy * self.efficiency)}

    class Actuator(Module):
        F = {"lift_force": Reals(unit="N")}
        R = {"power":      Reals(unit="W")}
        def h(self, f):
            return {"power": 10.0 * f["lift_force"] ** 2}

    b = Battery()
    b.uncertain_dist = Stochastic(
        marginals={
            "specific_energy": stats.uniform(loc=1.6e6, scale=0.4e6),
            "efficiency":      stats.uniform(loc=0.80, scale=0.10),
        },
        copula=GaussianCopula(correlation=[[1.0, 0.4], [0.4, 1.0]]),
    )

    sys = System("drone")
    endurance = sys.provides("endurance", unit="s")
    extra_p   = sys.provides("extra_power", unit="W")
    extra_pl  = sys.provides("extra_payload", unit="kg")
    total_m   = sys.requires("total_mass", unit="kg")
    b_h = sys.add("battery",  b)
    a_h = sys.add("actuator", Actuator())
    b_h.capacity    >= (a_h.power + extra_p) * endurance
    a_h.lift_force  >= 9.81 * (b_h.mass + extra_pl)
    total_m         >= b_h.mass + extra_pl
    drone = sys.build()

    f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}

    res = solve(
        drone, f,
        uncertainty=["mean", "p95", "cvar95"],
        n_samples=200, rng_seed=0,
    )

    mean = res.mean["total_mass"]
    p95 = res.p95["total_mass"]
    cvar = res.cvar95["total_mass"]
    # Sanity: mean < p95 < cvar95 (since cvar is the mean of the worst tail)
    assert mean < p95 < cvar, (mean, p95, cvar)
    assert res.feasibility_rate > 0.95
    print(f"mean={mean:.4f}, p95={p95:.4f}, cvar95={cvar:.4f}, "
          f"feas={res.feasibility_rate:.2f}")


def test_warm_start():
    """Warm-start: solving with start_from=prev should accept the prior
    inner antichain and converge in fewer iterations on average."""
    print("\n=== Warm start across a parameter sweep ===")
    import numpy as np

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass": Reals(unit="kg")}
        def h(self, f):
            return {"mass": f["capacity"] / 1.8e6}

    class Actuator(Module):
        F = {"lift_force": Reals(unit="N")}
        R = {"power": Reals(unit="W")}
        def h(self, f):
            return {"power": 10.0 * f["lift_force"] ** 2}

    sys = System("drone")
    endurance = sys.provides("endurance", unit="s")
    extra_p   = sys.provides("extra_power", unit="W")
    extra_pl  = sys.provides("extra_payload", unit="kg")
    total_m   = sys.requires("total_mass", unit="kg")
    b = sys.add("battery",  Battery())
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_p) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_pl)
    total_m       >= b.mass + extra_pl
    drone = sys.build()

    endurances = np.linspace(60.0, 300.0, 30)

    cold_iters = 0
    cold_masses = []
    for L in endurances:
        f = {"endurance": float(L), "extra_payload": 0.5, "extra_power": 5.0}
        r = solve(drone, f, max_iter=200)
        cold_iters += r.iterations
        cold_masses.append(list(r.antichain.points)[0]["total_mass"])

    warm_iters = 0
    warm_masses = []
    prev = None
    for L in endurances:
        f = {"endurance": float(L), "extra_payload": 0.5, "extra_power": 5.0}
        r = solve(drone, f, max_iter=200, start_from=prev)
        warm_iters += r.iterations
        warm_masses.append(list(r.antichain.points)[0]["total_mass"])
        prev = r

    # Same answer up to FP noise, warm not slower than cold.
    for cm, wm in zip(cold_masses, warm_masses):
        assert abs(cm - wm) < 1e-6, (cm, wm)
    assert warm_iters <= cold_iters

    # Also: start_from with an Antichain directly works.
    r_first = solve(drone,
                    {"endurance": 100.0, "extra_power": 5.0, "extra_payload": 0.5},
                    max_iter=200, trace=True)
    inner = r_first._inner_antichain
    r_again = solve(drone,
                    {"endurance": 100.0, "extra_power": 5.0, "extra_payload": 0.5},
                    max_iter=200, start_from=inner)
    assert r_again.iterations <= 2, r_again.iterations  # immediately fixed

    print(f"cold={cold_iters} warm={warm_iters} "
          f"(speedup {cold_iters/max(warm_iters,1):.2f}x)")


def test_viz_smoke():
    """Visualisation helpers should at least run without errors."""
    print("\n=== Visualisation smoke ===")
    from codesign import viz
    from codesign.antichains import Antichain
    from codesign.posets import Ports

    class Battery(Module):
        F = {"capacity": Reals(unit="J")}
        R = {"mass": Reals(unit="kg")}
        def h(self, f):
            return {"mass": f["capacity"] / 1.8e6}

    class Actuator(Module):
        F = {"lift_force": Reals(unit="N")}
        R = {"power": Reals(unit="W")}
        def h(self, f):
            return {"power": 10.0 * f["lift_force"] ** 2}

    sys = System("drone")
    endurance = sys.provides("endurance", unit="s")
    extra_p   = sys.provides("extra_power", unit="W")
    extra_pl  = sys.provides("extra_payload", unit="kg")
    total_m   = sys.requires("total_mass", unit="kg")
    b = sys.add("battery",  Battery())
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_p) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_pl)
    total_m       >= b.mass + extra_pl
    drone = sys.build()

    # to_dot should produce a dot string with all four wires named.
    dot = viz.to_dot(drone)
    assert "digraph" in dot
    assert "battery" in dot and "actuator" in dot
    # cyclic-or-not, both modules should appear as nodes
    assert dot.count("[label=") >= 2

    # plot functions should at least import matplotlib lazily; we skip if
    # it isn't available, so the test passes on minimal installations.
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
        matplotlib.use("Agg", force=True)
    except ImportError:
        print("(matplotlib not installed; skipping plot tests)")
        return

    r = solve(drone,
              {"endurance": 300.0, "extra_power": 5.0, "extra_payload": 0.5},
              max_iter=200, trace=True)
    ax = viz.plot_convergence(r)
    assert ax is not None
    plt.close()

    # plot_antichain on a handcrafted 2D antichain
    R = Ports({"cost": Reals(), "weight": Reals()})
    pts = [{"cost": 100, "weight": 5},
           {"cost": 80,  "weight": 8},
           {"cost": 60,  "weight": 12}]
    a2 = Antichain.from_set(R, pts)
    ax = viz.plot_antichain(a2, axes=["cost", "weight"])
    assert ax is not None
    plt.close()
    print("viz smoke ok")


def test_online_solver():
    """The online elimination solver should agree with exhaustive solve.

    Builds a small catalogue, runs exhaustive then online with each of
    the three evaluators, and checks: (a) Lipschitz and Monotonicity
    find the same antichain, (b) they evaluate strictly fewer
    candidates than the catalogue size when bounds bite.
    """
    print("\n=== Online elimination solver ===")
    import math, random

    F = Ports({"target_throughput": Reals(unit="pkg/h")})
    R = Ports({"total_cost": Reals(unit="USD")})

    def make_dp(robot):
        cap = robot["speed"] * robot["payload"]
        uc = robot["unit_cost"]
        return AlgebraicDP(F, R, {
            "total_cost": lambda f, c=cap, u=uc: (f["target_throughput"] / c) * u,
        })

    rng = random.Random(0)
    candidates = []
    for i in range(60):
        s = rng.uniform(5, 30)
        p = rng.uniform(1, 20)
        c = rng.uniform(500, 5000)
        candidates.append({
            "name": f"r{i}", "speed": s, "payload": p, "unit_cost": c,
            "cost_per_capacity": c / (s * p),
        })
    f = {"target_throughput": 100.0}

    # Exhaustive baseline
    costs = []
    for c in candidates:
        pt = list(solve(make_dp(c), f).antichain.points)[0]
        costs.append((pt["total_cost"], c["name"]))
    costs.sort()
    opt = costs[0][0]

    # Monotonicity with the derived monotone feature: same optimum,
    # strictly fewer evaluations.
    ev = MonotonicityEvaluator(features=["cost_per_capacity"],
                               r_components=["total_cost"])
    res_mono = solve_online(make_dp, f, candidates=candidates, evaluator=ev)
    found_mono = list(res_mono.antichain.points)[0]["total_cost"]
    assert abs(found_mono - opt) < 1e-6, (found_mono, opt)
    assert res_mono.n_evaluated < len(candidates), res_mono.n_evaluated
    assert res_mono.n_evaluated + res_mono.n_eliminated == len(candidates)

    # Lipschitz with sufficient L: same optimum.
    ev = LipschitzEvaluator(features=["speed", "payload", "unit_cost"],
                            r_components=["total_cost"], L=500.0)
    res_lip = solve_online(make_dp, f, candidates=candidates, evaluator=ev)
    found_lip = list(res_lip.antichain.points)[0]["total_cost"]
    assert abs(found_lip - opt) < 1e-6, (found_lip, opt)
    assert res_lip.n_evaluated <= len(candidates)

    # LinearParametric ran and didn't crash; correctness is best-effort.
    ev = LinearParametricEvaluator(features=["speed", "payload", "unit_cost"],
                                   r_components=["total_cost"],
                                   confidence=3.0, min_obs=4)
    res_par = solve_online(make_dp, f, candidates=candidates, evaluator=ev)
    assert res_par.n_evaluated >= 1
    assert len(res_par.antichain) >= 1

    print(f"exhaustive opt={opt:.2f}, "
          f"mono {res_mono.n_evaluated}ev, "
          f"lip {res_lip.n_evaluated}ev, "
          f"linpar {res_par.n_evaluated}ev")


def test_diagram_smoke():
    """Mechanical check that the block-diagram renderer produces a
    well-formed Digraph for both a simple feedback System and a
    System with lambda-based outer-R constraints.

    Skips silently if the optional graphviz Python package is not
    installed; the rest of the suite remains usable.
    """
    print("\n=== Diagram renderer ===")
    try:
        import graphviz  # noqa: F401
    except ImportError:
        print("  graphviz package not installed; skipped")
        return

    from codesign.diagram import (
        draw_system, _extract_spec, _find_cycle_modules,
        _collect_port_refs,
    )

    # Build a 2-module feedback System: an actuator/battery loop where
    # battery.mass affects actuator demand and actuator.power affects
    # battery capacity. This is the smallest case that exercises cycle
    # detection.
    sys_obj = System("loop_smoke")
    end = sys_obj.provides("endurance", unit="s")
    tot = sys_obj.requires("total_mass", unit="kg")

    actuator = sys_obj.add("actuator", AlgebraicDP(
        F=Ports({"lift_force": Reals(unit="N")}),
        R=Ports({"power": Reals(unit="W"),
                 "mass":  Reals(unit="kg")}),
        equations={
            "power": lambda f: f["lift_force"] * 4.0,
            "mass":  lambda f: f["lift_force"] * 0.02,
        },
    ))
    battery = sys_obj.add("battery", AlgebraicDP(
        F=Ports({"capacity": Reals(unit="Wh")}),
        R=Ports({"mass": Reals(unit="kg")}),
        equations={"mass": lambda f: f["capacity"] * 0.005},
    ))

    actuator.lift_force >= 10.0 + battery.mass
    battery.capacity    >= end * actuator.power / 3600.0
    tot                 >= actuator.mass + battery.mass

    # Sanity-check the extracted spec and cycle detection BEFORE the
    # actual render.
    spec = _extract_spec(sys_obj)
    assert set(spec.modules) == {"actuator", "battery"}
    assert set(spec.outer_F) == {"endurance"}
    assert set(spec.outer_R) == {"total_mass"}
    cyc = _find_cycle_modules(spec)
    assert cyc == {"actuator", "battery"}, \
        f"expected actuator/battery cycle, got {cyc}"

    # Render. The Digraph should be non-empty and reference both
    # module names plus the outer F/R.
    dot = sys_obj.draw_diagram()
    src = dot.source
    assert "actuator" in src and "battery" in src
    assert "outer_f__endurance" in src
    assert "outer_r__total_mass" in src
    # Cycle highlight: the cycle colour string should appear in the
    # rendered source.
    assert "#B45309" in src, "cycle edges should be coloured"

    # Now exercise the lambda-based path. A constraint with a callable
    # (no expr) should produce a "lambda_marker" node and a dashed
    # edge.
    sys2 = System("lam_smoke")
    sys2.provides("input_x", unit="")
    sys2.requires("output_y", unit="")
    sys2.constrain("output_y", lambda x: 1.0)  # callable, no expr
    dot2 = sys2.draw_diagram()
    src2 = dot2.source
    assert "lambda_marker" in src2, \
        "lambda-based constraint should add a marker node"
    assert "dashed" in src2

    # Port-reference walker basic check.
    feat_port = actuator.power
    refs = _collect_port_refs(feat_port * 2.0)
    assert any(r.module == "actuator" and r.name == "power" for r in refs)

    print(f"  feedback diagram: {len(dot.body)} body lines, "
          f"cycle modules = {sorted(cyc)}")
    print(f"  lambda diagram: marker present, "
          f"{len(dot2.body)} body lines")


if __name__ == "__main__":
    test_posets()
    test_antichain()
    test_algebraic_dp()
    test_catalog_dp()
    test_series()
    test_parallel()
    test_loop_simple()
    test_system()
    test_module_and_operator_dsl()
    test_dsl_type_errors()
    test_solver_trace_and_status()
    test_uncertainty_box()
    test_uncertainty_stochastic()
    test_warm_start()
    test_viz_smoke()
    test_online_solver()
    test_diagram_smoke()
    print("\nAll smoke tests passed.")
