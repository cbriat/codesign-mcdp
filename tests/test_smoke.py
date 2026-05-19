"""Quick smoke test verifying posets, antichains, and basic DPs."""
import sys
sys.path.insert(0, "/home/claude")

from codesign import (
    Reals, Naturals, NamedProduct,
    Antichain,
    AlgebraicDP, FunctionDP, CatalogDP, CatalogEntry,
    series, par, loop,
    solve,
    System, Module,
)


def test_posets():
    print("=== Posets ===")
    r = Reals(unit="kg")
    assert r.leq(1.0, 2.0)
    assert not r.leq(2.0, 1.0)
    assert r.is_top(float("inf"))
    print(f"Reals format: {r.format(1.5)}, top: {r.format(r.top())}")

    np_p = NamedProduct({"a": Reals(), "b": Reals()})
    x = np_p.make(a=1.0, b=2.0)
    y = np_p.make(a=1.5, b=2.5)
    z = np_p.make(a=1.5, b=1.0)
    assert np_p.leq(x, y)
    assert not np_p.leq(x, z)
    assert not np_p.leq(z, x)
    print(f"Product order works: {np_p.format(x)} <= {np_p.format(y)}")


def test_antichain():
    print("\n=== Antichains ===")
    p = NamedProduct({"cost": Reals(), "weight": Reals()})
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
    F = NamedProduct({"capacity": Reals(unit="J")})
    R = NamedProduct({"mass": Reals(unit="kg")})
    battery = AlgebraicDP(F, R, {"mass": lambda f: f["capacity"] / 1.8e6})
    res = solve(battery, {"capacity": 3.6e6})
    print(f"Battery for 3.6 MJ: {res.antichain}")
    pt = list(res.antichain)[0]
    assert abs(pt["mass"] - 2.0) < 1e-9


def test_catalog_dp():
    print("\n=== CatalogDP ===")
    F = NamedProduct({"torque": Reals(), "speed": Reals()})
    R = NamedProduct({"cost": Reals(), "weight": Reals()})
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
    F1 = NamedProduct({"capacity": Reals()})
    R1 = NamedProduct({"mass": Reals()})
    battery = AlgebraicDP(F1, R1, {"mass": lambda f: f["capacity"] / 1.8e6})

    F2 = NamedProduct({"mass": Reals()})
    R2 = NamedProduct({"cost": Reals()})
    pricing = AlgebraicDP(F2, R2, {"cost": lambda f: f["mass"] * 10.0})

    chain = series(battery, pricing)
    res = solve(chain, {"capacity": 3.6e6})
    print(f"Battery + pricing for 3.6 MJ: {res.antichain}")


def test_parallel():
    print("\n=== Parallel composition ===")
    F1 = NamedProduct({"capacity": Reals()})
    R1 = NamedProduct({"mass": Reals()})
    battery = AlgebraicDP(F1, R1, {"mass": lambda f: f["capacity"] / 1.8e6})

    F2 = NamedProduct({"lift": Reals()})
    R2 = NamedProduct({"power": Reals()})
    actuator = AlgebraicDP(F2, R2, {"power": lambda f: 10.0 * f["lift"] ** 2})

    combo = par(battery, actuator)
    res = solve(combo, {"capacity": 3.6e6, "lift": 5.0})
    print(f"Battery+actuator: {res.antichain}")


def test_loop_simple():
    print("\n=== Simple loop ===")
    # f -> r where r = 2*sqrt(f) + 1, then close r <= f.
    # Fixed-point: c = 2*sqrt(c) + 1, which gives c = 3 + 2*sqrt(2) ~ 5.83.
    F = NamedProduct({"capacity": Reals()})
    R = NamedProduct({"capacity": Reals()})

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
        F=NamedProduct({"in": Reals()}),
        R=NamedProduct({"out": Reals()}),
        equations={"out": lambda f: 0.5 * f["in"] + 1.0},
        name="producer",
    )
    consumer = AlgebraicDP(
        F=NamedProduct({"need": Reals()}),
        R=NamedProduct({"cost": Reals()}),
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
    print("\nAll smoke tests passed.")
