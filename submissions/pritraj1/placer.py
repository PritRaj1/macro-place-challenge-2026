from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

from ._langevin import BoltzmannPlacer
from ._legalize import legalize_graph


def _load_plc(name):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d
            / "netlist"
            / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc

    return None


def _extract_hypergraph_nets(benchmark, plc) -> list[list[int]]:
    """Extracts lists of connected macro indices."""
    if plc is None:
        return []

    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    nets = []
    for driver, sinks in plc.nets.items():
        macro_set = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macro_set.add(name_to_bidx[parent])

        # Hypernets require at least 2 distinct hard macros
        if len(macro_set) >= 2:
            nets.append(list(macro_set))

    return nets


class PritRajPlacer:
    def __init__(
        self, seed: int = 42, langevin_steps: int = 600, fast_mode: bool = False
    ):
        self.seed = seed
        self.langevin_steps = langevin_steps
        self.fast_mode = fast_mode
        self._placement_callback = None

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        # Extract netlist hypergraph topology and initial center positions
        plc = _load_plc(benchmark.name)
        nets = _extract_hypergraph_nets(benchmark, plc)
        pos_init = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

        # Smooth Energy-Based Langevin Global Placement
        if len(nets) > 0:
            placer = BoltzmannPlacer(
                sizes=sizes_np,
                nets=nets,
                canvas_width=cw,
                canvas_height=ch,
                gap=0.01, # smaller gap permitted for global placer
            )
            global_pos = placer.optimize(
                pos_init=pos_init,
                movable=movable,
                num_steps=self.langevin_steps,
                lr=0.005,
                density_weight=30.0,
                temp_start=1.0,
                temp_end=0.001,
                callback=self._placement_callback,
            )
        else:
            global_pos = pos_init

        # LP + Greedy Zero-Overlap Legalizer
        legal_pos = legalize_graph(
            pos=global_pos,
            sizes=sizes_np,
            movable=movable,
            canvas_width=cw,
            canvas_height=ch,
            prefer_displacement=True,
        )

        # Soft Macro Co-Optimization
        if plc is not None and not self.fast_mode:
            for bidx, module_idx in enumerate(plc.hard_macro_indices):
                module = plc.modules_w_pins[module_idx]
                new_x, new_y = legal_pos[bidx][0], legal_pos[bidx][1]
                module.set_pos(new_x, new_y)

            # Force-directed placement to re-align soft macros around updated hard macros
            canvas_size = max(cw, ch)
            plc.optimize_stdcells(
                use_current_loc=False,
                move_stdcells=True,
                move_macros=False,
                log_scale_conns=False,
                use_sizes=False,
                io_factor=1.0,
                num_steps=[10, 10, 10],
                max_move_distance=[canvas_size / 20] * 3,
                attract_factor=[100, 1.0e-3, 1.0e-5],
                repel_factor=[0, 1.0e6, 1.0e7],
            )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)

        # Update soft macro positions tensor if plc was available
        if (
            plc is not None
            and hasattr(plc, "soft_macro_indices")
            and not self.fast_mode
        ):
            for bidx, module_idx in enumerate(plc.soft_macro_indices):
                pos = plc.modules_w_pins[module_idx].get_pos()
                full_pos[n_hard + bidx] = torch.tensor(pos, dtype=torch.float32)

        return full_pos
