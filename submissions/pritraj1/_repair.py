import numpy as np
import torch

from ._langevin import BoltzmannPlacer


class GreedyRepair:
    def __init__(self, placer: BoltzmannPlacer, gap: float = 0.1):
        """After legalizing, (which is ambivalent to proxy cost),
        this greedily moves macros with accept/reject on proxy scoring"""
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

        current_wl = self.placer.wirelength_score(pos).item()

        for _ in range(iters):
            indices = movable_indices[
                torch.randperm(len(movable_indices))
            ]  # Shuffle order to prevent bias

            for idx in indices:
                noise = torch.randn((K, 2), device=pos.device) * search_radius
                candidates = pos[idx].unsqueeze(0) + noise
                candidates = torch.clamp(
                    candidates,
                    self.half_sizes[idx],
                    torch.tensor([self.cw, self.ch], device=pos.device)
                    - self.half_sizes[idx],
                )  # Clamp inside canvas

                # Filter overlaps
                valid_mask = self._get_valid_candidates(candidates, pos, idx)
                valid_candidates = candidates[valid_mask]
                if len(valid_candidates) == 0:
                    continue

                # Score using placer's proxy
                best_candidate = None
                best_wl = current_wl

                for cand in valid_candidates:
                    temp_pos = pos.clone()
                    temp_pos[idx] = cand
                    cand_wl = self.placer.wirelength_score(temp_pos).item()

                    if cand_wl < best_wl:
                        best_wl = cand_wl
                        best_candidate = cand

                # Accept/reject
                if best_candidate is not None:
                    pos[idx] = best_candidate
                    current_wl = best_wl

        return pos.detach().cpu().numpy().astype(np.float64)
