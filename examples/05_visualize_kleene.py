"""
Visualize the Kleene fixed-point iteration on the Sec. VI-D integer
example, reproducing the structure of Fig. 36 in the paper. Each panel
shows one iterate S_k of the antichain in N x N. The seed is {(0, 0)}
and the iteration converges in a handful of steps.

Run:  python -m examples.05_visualize_kleene
Produces PNG figures in an ``outputs/`` directory at the repository root.
"""
from __future__ import annotations

import importlib
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Module name starts with a digit, so use importlib rather than `from x import y`.
_integer_example = importlib.import_module("examples.02_integer_optimization")
make_looped = _integer_example.make_looped

from codesign import solve


def plot_trace(c_value: int, out_path: str):
    looped = make_looped(c_value)
    result = solve(looped, {"c": c_value}, max_iter=50, record_trace=True)
    trace = result.trace

    n = len(trace)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = [axes] if rows * cols == 1 else (axes.flat if rows > 1 else axes)
    axes = list(axes)

    # Pick axis bounds wide enough to show convergence comfortably.
    max_xy = 1
    for entry in trace:
        for p in entry.antichain.points:
            x = p["xy"]["x"]
            y = p["xy"]["y"]
            if x != float("inf"):
                max_xy = max(max_xy, int(x))
            if y != float("inf"):
                max_xy = max(max_xy, int(y))
    bound = max_xy + 3

    for k, entry in enumerate(trace):
        A = entry.antichain
        ax = axes[k]
        xs, ys = [], []
        for p in A.points:
            x = p["xy"]["x"]
            y = p["xy"]["y"]
            if x == float("inf") or y == float("inf"):
                continue
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=60, c="C3", zorder=3)
        # Light grid + diagonal x+y = const lines.
        ax.set_xlim(-0.5, bound)
        ax.set_ylim(-0.5, bound)
        ax.set_xticks(range(0, bound + 1, max(1, bound // 6)))
        ax.set_yticks(range(0, bound + 1, max(1, bound // 6)))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_aspect("equal")
        npts = len(A.points)
        ax.set_title(f"$S_{{{k}}}$  ({npts} pt{'s' if npts != 1 else ''})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Hide unused axes.
    for k in range(len(trace), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Kleene ascent on  x + y >= ceil(sqrt(x)) + ceil(sqrt(y)) + {c_value}\n"
        f"converged in {result.iterations} iterations, "
        f"|M(c={c_value})| = {len(result.antichain.points)} minimal points",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {out_path}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Plotting Kleene iteration traces into {out_dir} ...")
    for c in (1, 4, 8):
        out = os.path.join(out_dir, f"kleene_trace_c{c}.png")
        plot_trace(c, out)
