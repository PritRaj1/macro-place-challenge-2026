from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from macro_place.loader import load_benchmark_from_dir
from submissions.pritraj1 import legalize_graph


def count_overlaps(p, s, gap=0.05):
    half = s / 2
    c = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if (
                abs(p[i, 0] - p[j, 0]) < half[i, 0] + half[j, 0] + gap
                and abs(p[i, 1] - p[j, 1]) < half[i, 1] + half[j, 1] + gap
            ):
                c += 1

    return c


def plot_pair(before, after, sizes, cw, ch, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, p, name in zip(axes, [before, after], ["before", "after"]):
        ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, linewidth=2))
        for i in range(len(p)):
            w, h = sizes[i]
            ax.add_patch(
                Rectangle(
                    (p[i, 0] - w / 2, p[i, 1] - h / 2),
                    w,
                    h,
                    facecolor="steelblue",
                    edgecolor="black",
                    alpha=0.7,
                )
            )

        ax.set_xlim(0, cw)
        ax.set_ylim(0, ch)
        ax.set_aspect("equal")
        ax.set_title(f"ibm1 {name}  overlaps={count_overlaps(p, sizes)}")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


bench, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
n = bench.num_hard_macros
pos = bench.macro_positions[:n].numpy().astype(np.float64).copy()
sizes = bench.macro_sizes[:n].numpy().astype(np.float64)
movable = bench.get_movable_mask()[:n].numpy()
cw, ch = float(bench.canvas_width), float(bench.canvas_height)

print(f"input overlaps: {count_overlaps(pos, sizes)}")
legal = legalize_graph(pos, sizes, movable, cw, ch, gap=0.05, prefer_displacement=True)
print(f"output overlaps: {count_overlaps(legal, sizes)}")
print(f"mean displacement: {np.mean(np.linalg.norm(legal - pos, axis=1)):.4f}")

half = sizes / 2
in_bounds = (
    np.all(legal[:, 0] >= half[:, 0] - 1e-6)
    and np.all(legal[:, 0] <= cw - half[:, 0] + 1e-6)
    and np.all(legal[:, 1] >= half[:, 1] - 1e-6)
    and np.all(legal[:, 1] <= ch - half[:, 1] + 1e-6)
)
fixed_ok = np.allclose(legal[~movable], pos[~movable], atol=1e-3)
print(f"in bounds: {in_bounds}")
print(f"fixed ok:  {fixed_ok}")

out = Path("gifs")
out.mkdir(exist_ok=True)
plot_pair(pos, legal, sizes, cw, ch, out / "legalize_ibm1.png")

# Put all movable macros at center
pos_c = pos.copy()
pos_c[movable, 0] = cw / 2
pos_c[movable, 1] = ch / 2
legal_c = legalize_graph(pos_c, sizes, movable, cw, ch)
print(f"centered input overlaps: {count_overlaps(pos_c, sizes)}")
print(f"centered output overlaps: {count_overlaps(legal_c, sizes)}")
plot_pair(pos_c, legal_c, sizes, cw, ch, out / "legalize_ibm1_centered.png")

if (
    count_overlaps(legal, sizes) != 0
    or count_overlaps(legal_c, sizes) != 0
    or not in_bounds
    or not fixed_ok
):
    print("FAIL")
else:
    print("PASS")
