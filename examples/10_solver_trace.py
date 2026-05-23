"""
Solver trace, verbose printing, and on_iteration callback.

This example exercises the solver observability features introduced
alongside the uncertainty layer:

- ``trace=True``: collect a structured per-iteration record on
  ``result.trace`` (a list of :class:`TraceEntry`).
- ``verbose=0|1|2``: silent / final summary / per-iteration progress.
- ``on_iteration=callable``: a callback receiving each :class:`TraceEntry`
  as it is produced, useful for live plotting or custom logging.
- ``result.status``: ``"converged"``, ``"max_iter"``, or ``"diverged"``,
  distinct from ``result.feasible``.
"""
from __future__ import annotations

from codesign import Module, Reals, System, solve


# ---------------------------------------------------------------------------
# Toy two-subsystem drone (same model as example 7).
# ---------------------------------------------------------------------------


class Battery(Module):
    F = {"capacity": Reals(unit="J")}
    R = {"mass":     Reals(unit="kg")}
    def h(self, f):
        return {"mass": f["capacity"] / 1.8e6}


class Actuator(Module):
    F = {"lift_force": Reals(unit="N")}
    R = {"power":      Reals(unit="W")}
    def h(self, f):
        return {"power": 10.0 * f["lift_force"] ** 2}


def make_drone():
    sys = System("drone")
    endurance     = sys.provides("endurance",     unit="s")
    extra_payload = sys.provides("extra_payload", unit="kg")
    extra_power   = sys.provides("extra_power",   unit="W")
    total_mass    = sys.requires("total_mass",    unit="kg")
    b = sys.add("battery",  Battery())
    a = sys.add("actuator", Actuator())
    b.capacity    >= (a.power + extra_power) * endurance
    a.lift_force  >= 9.81 * (b.mass + extra_payload)
    total_mass    >= b.mass + extra_payload
    return sys.build()


if __name__ == "__main__":
    drone = make_drone()
    f = {"endurance": 300.0, "extra_payload": 0.5, "extra_power": 5.0}

    # ----- verbose=0 (silent) -----
    print("# 1. Silent (verbose=0, default)")
    print("-" * 50)
    r = solve(drone, f)
    print(f"   status={r.status}, iters={r.iterations}, "
          f"feasible={r.feasible}, mass={r.antichain}")
    print()

    # ----- verbose=1 (final summary) -----
    print("# 2. Final summary (verbose=1)")
    print("-" * 50)
    _ = solve(drone, f, verbose=1)
    print()

    # ----- verbose=2 (per-iteration feed) -----
    print("# 3. Per-iteration feed (verbose=2)")
    print("-" * 50)
    _ = solve(drone, f, verbose=2, max_iter=20)
    print()

    # ----- trace=True (collect, no printing) -----
    print("# 4. Structured trace (trace=True)")
    print("-" * 50)
    r = solve(drone, f, trace=True, max_iter=100)
    print(f"   collected {len(r.trace)} trace entries")
    print(f"   first 5 deltas: {[e.delta for e in r.trace[:5]]}")
    print(f"   final delta: {r.trace[-1].delta}")
    print(f"   total wall time (sum of per-iter): "
          f"{sum(e.elapsed_ms for e in r.trace):.2f} ms")
    print()

    # ----- on_iteration callback -----
    print("# 5. on_iteration callback (custom log line)")
    print("-" * 50)

    def my_logger(entry):
        # Print only every 5th iteration to keep things tidy.
        if entry.iteration % 5 == 0:
            d = "    -    " if entry.delta is None else f"{entry.delta:.3e}"
            print(f"   callback: iter {entry.iteration:>3}, |A|="
                  f"{entry.n_points}, delta={d}")

    _ = solve(drone, f, on_iteration=my_logger, max_iter=100)
    print()

    # ----- max_iter: status distinguishes from infeasibility -----
    print("# 6. Status field (max_iter vs converged vs feasible)")
    print("-" * 50)
    r_short = solve(drone, f, max_iter=3)
    print(f"   max_iter=3:  status={r_short.status!r}, "
          f"feasible={r_short.feasible}, iters={r_short.iterations}")
    r_ok = solve(drone, f, max_iter=200)
    print(f"   max_iter=200: status={r_ok.status!r}, "
          f"feasible={r_ok.feasible}, iters={r_ok.iterations}")
    # A genuinely infeasible case: tiny capacity, huge power demand
    r_inf = solve(
        drone,
        {"endurance": 1800.0, "extra_payload": 1.0, "extra_power": 10.0},
        max_iter=200,
    )
    print(f"   infeasible:   status={r_inf.status!r}, "
          f"feasible={r_inf.feasible}, iters={r_inf.iterations}")

    print()
    print("Note: max_iter and infeasible are different conditions and the")
    print("status field distinguishes them cleanly. With only the feasible")
    print("flag you couldn't tell whether bumping max_iter would help.")
