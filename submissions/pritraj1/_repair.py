import numpy as np
import torch

from ._langevin import BoltzmannPlacer


class GreedyRepair:
    def __init__(self, placer: BoltzmannPlacer, gap: float = 0.1):
        """Local legality-preserving repair; accept only if energy decreases."""
        self.placer = placer
        self.sizes = placer.sizes
        self.half_sizes = placer.half_sizes
        self.gap = gap
        self.cw = placer.cw
        self.ch = placer.ch

    def _get_valid_candidates(
        self, candidate_pos: torch.Tensor, current_pos: torch.Tensor, macro_idx: int
    ) -> torch.Tensor:
        """Feasibility/overlap check for K candidates against N-1 other macros."""
        # Exclude current macro from collision check
        mask = torch.ones(
            current_pos.shape[0], dtype=torch.bool, device=current_pos.device
        )
        mask[macro_idx] = False

        other_pos = current_pos[mask]
        other_half = self.half_sizes[mask]
        my_half = self.half_sizes[macro_idx]

        # Broadcast to [K, N-1, 2]
        dist = torch.abs(candidate_pos.unsqueeze(1) - other_pos.unsqueeze(0))
        req_dist = (
            my_half.unsqueeze(0).unsqueeze(0) + other_half.unsqueeze(0) + self.gap
        )

        # 2D Bounding box collision check
        overlap_x = dist[:, :, 0] < req_dist[:, :, 0]
        overlap_y = dist[:, :, 1] < req_dist[:, :, 1]
        overlap_2d = overlap_x & overlap_y  # [K, N-1]

        # Valid if NO overlaps across all N-1 macros
        valid = ~torch.any(overlap_2d, dim=1)
        return valid

    def _score(self, pos: torch.Tensor) -> torch.Tensor:
        wl = self.placer.wirelength_score(pos)
        dens = self.placer.density_score(pos)
        cong = self.placer.congestion_score(pos)
        return wl + 0.5 * dens + 0.5 * cong

    def repair(
        self,
        legal_pos_np: np.ndarray,
        movable: np.ndarray,
        iters: int = 5,
        K: int = 64,
        search_radius: float = 10.0,
    ) -> np.ndarray:
        pos = torch.tensor(legal_pos_np, dtype=torch.float32, device=self.placer.device)
        movable_t = torch.tensor(movable, dtype=torch.bool, device=self.placer.device)
        movable_indices = torch.where(movable_t)[0]
        if len(movable_indices) == 0:
            return legal_pos_np

        current = self._score(pos).item()
        canvas = torch.tensor([self.cw, self.ch], device=pos.device)

        for _ in range(iters):
            order = movable_indices[torch.randperm(len(movable_indices))]
            for idx in order:
                idx_i = int(idx.item())
                half = self.half_sizes[idx_i]
                noise = torch.randn((K, 2), device=pos.device) * search_radius
                candidates = pos[idx_i].unsqueeze(0) + noise
                candidates = torch.max(candidates, half)
                candidates = torch.min(candidates, canvas - half)

                valid = self._get_valid_candidates(candidates, pos, idx_i)
                valid_candidates = candidates[valid]
                if valid_candidates.numel() == 0:
                    continue

                # Evaluate all valid candidates; keep best energy
                best_cand = None
                best_score = current
                for cand in valid_candidates:
                    temp = pos.clone()
                    temp[idx_i] = cand
                    s = self._score(temp).item()
                    if s < best_score:
                        best_score = s
                        best_cand = cand

                if best_cand is not None:
                    pos[idx_i] = best_cand
                    current = best_score

        return pos.detach().cpu().numpy().astype(np.float64)
