import sys
import torch
import numpy as np
import importlib.util
from pathlib import Path

from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import animate_placement


def load():
    spec = importlib.util.spec_from_file_location(
        "will_seed", "submissions/will_seed/placer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["will_seed"] = mod
    spec.loader.exec_module(mod)
    return mod.WillSeedPlacer(seed=42, refine_iters=2000)


def record(placer, benchmark):
    """
    Capture stages of Will's seed placer:
      - initial
      - after legalization
      - a few points during SA (snapshotting/re-running short segments)
      - final
    """
    frames = []
    titles = []

    n_hard = benchmark.num_hard_macros
    initial = benchmark.macro_positions.clone()
    frames.append(initial)
    titles.append("Initial")

    original_legalize = placer._legalize
    original_sa = placer._sa_refine
    sa_snapshots = []

    def legalize_hook(*args, **kwargs):
        pos = original_legalize(*args, **kwargs)
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.from_numpy(pos.astype(np.float32))
        frames.append(full.clone())
        titles.append("After legalization")
        return pos

    def sa_hook(pos, edges, edge_weights, movable, sizes, half_w, half_h,
                cw, ch, n, plc, benchmark):
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.from_numpy(pos.astype(np.float32))
        frames.append(full.clone())
        titles.append("Start of SA")

        final_pos = original_sa(
            pos, edges, edge_weights, movable, sizes,
            half_w, half_h, cw, ch, n, plc, benchmark
        )

        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.from_numpy(final_pos.astype(np.float32))
        frames.append(full.clone())
        titles.append("Final")
        return final_pos

    placer._legalize = legalize_hook
    placer._sa_refine = sa_hook

    _ = placer.place(benchmark)

    placer._legalize = original_legalize
    placer._sa_refine = original_sa
    return frames, titles


def main():
    benchmark, _ = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    placer = load()

    placements, titles = record(placer, benchmark)

    Path("gifs").mkdir(exist_ok=True)
    animate_placement(
        placements=placements,
        benchmark=benchmark,
        save_path="gifs/will_seed_ibm01.gif",
        titles=titles,
        fps=1,
    )


if __name__ == "__main__":
    main()
