import importlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import animate_placement, visualize_placement


def load_placer(import_path: str, class_name: str, **kwargs):
    mod = importlib.import_module(import_path)
    placer_cls = getattr(mod, class_name)
    return placer_cls(**kwargs), mod


class PlacementRecorder:
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.n_hard = benchmark.num_hard_macros
        self.frames = []
        self.titles = []
        self.add_frame(benchmark.macro_positions, "Initial State")

    def add_frame(self, pos, title: str):
        full = self.benchmark.macro_positions.clone()
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos.astype(np.float32))

        # If pos only contains hard macros (shape: [n_hard, 2]), insert into the first n_hard
        if pos.shape[0] == self.n_hard:
            full[: self.n_hard] = pos.clone()
        else:
            full = pos.clone()

        self.frames.append(full)
        self.titles.append(title)


@contextmanager
def patch_methods(patches):
    originals = []
    for obj, attr, new_func in patches:
        originals.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, new_func)
    try:
        yield
    finally:
        for obj, attr, orig_func in originals:
            setattr(obj, attr, orig_func)


def record_placer(placer_type: str, placer, benchmark, mod, langevin_interval: int = 1):
    rec = PlacementRecorder(benchmark)
    patches = []

    # Patch Will's _legalize and _sa_refine
    if placer_type == "will":
        orig_legalize = placer._legalize
        orig_sa = placer._sa_refine

        def legalize_hook(*args, **kwargs):
            pos = orig_legalize(*args, **kwargs)
            rec.add_frame(pos, "After Legalization")
            return pos

        def sa_hook(pos, *args, **kwargs):
            rec.add_frame(pos, "Start of SA")
            final_pos = orig_sa(pos, *args, **kwargs)
            rec.add_frame(final_pos, "Final State")
            return final_pos

        patches = [
            (placer, "_legalize", legalize_hook),
            (placer, "_sa_refine", sa_hook),
        ]

    # Patch BoltzmannPlacer.optimize and legalize_graph
    elif placer_type == "pritraj":
        orig_legalize = mod.legalize_graph
        num_steps = placer.langevin_steps

        def langevin_callback(*args, **kwargs):
            step = kwargs.get("step")
            pos = kwargs.get("pos")
            temperature = kwargs.get(
                "temperature",
                None,
            )

            if pos is None and len(args) >= 2:
                step = args[0]
                pos = args[1]

                if len(args) >= 3:
                    temperature = args[2]

            if step is None:
                step = 0
            if step % langevin_interval != 0 and step != num_steps - 1:
                return

            if isinstance(pos, np.ndarray):
                pos_tensor = torch.from_numpy(pos.astype(np.float32))
            else:
                pos_tensor = pos.detach().cpu()

            if temperature is None:
                title = f"Langevin Step {step + 1}/{num_steps}"
            else:
                title = f"Langevin Step {step + 1}/{num_steps} (T={temperature:.3g})"

            rec.add_frame(pos_tensor, title)

        placer._placement_callback = langevin_callback

        def legalize_hook(pos, *args, **kwargs):
            rec.add_frame(pos, "Pre-Legalization (Global Done)")
            legal_pos = orig_legalize(pos, *args, **kwargs)
            rec.add_frame(legal_pos, "Final Legalized State")
            return legal_pos

        patches = [
            (mod, "legalize_graph", legalize_hook),
        ]

    else:
        raise ValueError(f"Unknown placer_type: {placer_type}")

    original_callback = getattr(placer, "_placement_callback", None)

    # Restore og callback after placement.
    try:
        with patch_methods(patches):
            final_full_pos = placer.place(benchmark)
    finally:
        if placer_type == "pritraj":
            placer._placement_callback = original_callback

    rec.add_frame(
        final_full_pos,
        "Final State (Stdcells Unoptimized)",
    )

    return (
        rec.frames,
        rec.titles,
    )


def main():
    benchmark_path = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    benchmark, _ = load_benchmark_from_dir(benchmark_path)

    Path("gifs").mkdir(exist_ok=True)

    configs = [
        {
            "type": "pritraj",
            "import_path": "submissions.pritraj1.placer",
            "class_name": "PritRajPlacer",
            "kwargs": {"seed": 42, "langevin_steps": 1000, "fast_mode": True},
            "output_gif": "gifs/pritraj_ibm01.gif",
            "output_png": "gifs/pritraj_ibm01.png",
            "fps": 1,
        },
        {
            "type": "will",
            "import_path": "submissions.will_seed.placer",
            "class_name": "WillSeedPlacer",
            "kwargs": {"seed": 42, "refine_iters": 1000},
            "output_gif": "gifs/will_seed_ibm01.gif",
            "output_png": "gifs/will_seed_ibm01.png",
            "fps": 1,
        },
    ]

    for cfg in configs:
        placer, mod = load_placer(
            cfg["import_path"], cfg["class_name"], **cfg["kwargs"]
        )

        frames, titles = record_placer(
            placer_type=cfg["type"], placer=placer, benchmark=benchmark, mod=mod
        )

        animate_placement(
            placements=frames,
            benchmark=benchmark,
            save_path=cfg["output_gif"],
            titles=titles,
            fps=cfg["fps"],
        )
        print(f"Saved: {cfg['output_gif']}")

        final_frame = frames[-1]
        visualize_placement(
            placement=final_frame,
            benchmark=benchmark,
            save_path=cfg["output_png"],
        )
        print(f"Saved PNG: {cfg['output_png']}")


if __name__ == "__main__":
    main()
