from __future__ import annotations

import math

import numpy as np
import torch


class BoltzmannPlacer:
    """EBM-based global placer using ula sampling.

    E(P) = Wirelength_Energy(P)
         + Density_Energy(P)
         + Congestion_Energy(P).
    """

    def __init__(
        self,
        sizes: np.ndarray,
        nets: list[list[int]],
        net_weights: list[float],  # proxy weights
        canvas_width: float,  # canvas boundary
        canvas_height: float,
        grid_col: int = 10,  # proxy grid density
        grid_row: int = 10,
        gap: float = 0.1,  # min gap
        congestion_smoothing: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        print(f"Using device: {device}")
        self.device = torch.device(device)
        self.cw = float(canvas_width)
        self.ch = float(canvas_height)
        self.sizes = torch.tensor(sizes, dtype=torch.float32, device=self.device)
        self.n = len(sizes)
        self.half_sizes = self.sizes / 2.0
        self.gap = gap

        # Eval congenstion and density on grid
        self.grid_col = grid_col
        self.grid_row = grid_row
        self.congestion_smoothing = congestion_smoothing

        grid_x = (
            torch.arange(self.grid_col, dtype=torch.float32, device=self.device) + 0.5
        ) * (self.cw / self.grid_col)
        grid_y = (
            torch.arange(self.grid_row, dtype=torch.float32, device=self.device) + 0.5
        ) * (self.ch / self.grid_row)
        self.grid_x, self.grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
        self.grid_area = (self.cw / self.grid_col) * (self.ch / self.grid_row)

        # Pad for batching
        max_net_len = max(len(net) for net in nets) if nets else 0
        padded_nets = []
        net_masks = []

        for net in nets:
            if len(net) <= 1:
                continue
            pad_len = max_net_len - len(net)
            padded_nets.append(net + [0] * pad_len)
            net_masks.append([1.0] * len(net) + [0.0] * pad_len)

        if padded_nets:
            self.net_indices = torch.tensor(
                padded_nets, dtype=torch.long, device=self.device
            )
            self.net_masks = torch.tensor(
                net_masks, dtype=torch.float32, device=self.device
            )
            self.net_weights = torch.tensor(
                net_weights, dtype=torch.float32, device=self.device
            )
        else:
            self.net_indices = None
            self.net_masks = None

    def _abu_score(
        self, grid_tensor: torch.Tensor, top_percent: float = 0.1
    ) -> torch.Tensor:
        """Grads flow back through the top K elements."""
        flat_grid = grid_tensor.flatten()
        k = max(1, math.floor(flat_grid.numel() * top_percent))

        if flat_grid.numel() < 10:
            return flat_grid.mean()

        top_k_vals, _ = torch.topk(flat_grid, k)
        return top_k_vals.mean()

    def density_score(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Calculates smooth grid ABU-based density, differentiable via top-k.
        """
        smoothing = self.congestion_smoothing

        # Macro bounding boxes
        min_x = (
            (pos[:, 0] - self.half_sizes[:, 0] - self.gap).unsqueeze(-1).unsqueeze(-1)
        )
        max_x = (
            (pos[:, 0] + self.half_sizes[:, 0] + self.gap).unsqueeze(-1).unsqueeze(-1)
        )
        min_y = (
            (pos[:, 1] - self.half_sizes[:, 1] - self.gap).unsqueeze(-1).unsqueeze(-1)
        )
        max_y = (
            (pos[:, 1] + self.half_sizes[:, 1] + self.gap).unsqueeze(-1).unsqueeze(-1)
        )

        gx = self.grid_x.unsqueeze(0)
        gy = self.grid_y.unsqueeze(0)

        # Smooth inclusion masking
        inside_x = torch.sigmoid((gx - min_x) / smoothing) * torch.sigmoid(
            (max_x - gx) / smoothing
        )
        inside_y = torch.sigmoid((gy - min_y) / smoothing) * torch.sigmoid(
            (max_y - gy) / smoothing
        )

        # Area footprint per macro per grid cell
        macro_footprint = inside_x * inside_y
        grid_density = macro_footprint.sum(dim=0) / self.grid_area

        # Proxy evaluates Top 10%, halved
        e_density = self._abu_score(grid_density, 0.1) * 0.5
        return e_density

    def wirelength_score(self, pos: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
        """
        Return score of weighted-avg wirelength

        Smooths bounding-box HPWL = (max(X) - min(X)) using ePlace / DREAMPlace method.
        """
        if self.net_indices is None:
            return torch.tensor(0.0, device=self.device)

        net_coords = pos[self.net_indices]

        # WA Max Boundary
        max_coords = torch.where(
            self.net_masks.unsqueeze(-1) > 0,
            net_coords,
            torch.full_like(net_coords, -1e9),
        )
        shift_max = max_coords.max(dim=1, keepdim=True).values.detach()
        exp_pos = torch.exp(
            (net_coords - shift_max) / gamma
        ) * self.net_masks.unsqueeze(-1)
        wa_max = (net_coords * exp_pos).sum(dim=1) / (exp_pos.sum(dim=1) + 1e-8)

        # WA Min Boundary
        min_coords = torch.where(
            self.net_masks.unsqueeze(-1) > 0,
            -net_coords,
            torch.full_like(net_coords, -1e9),
        )
        shift_min = min_coords.max(dim=1, keepdim=True).values.detach()
        exp_neg = torch.exp(
            (-net_coords - shift_min) / gamma
        ) * self.net_masks.unsqueeze(-1)
        wa_min = (net_coords * exp_neg).sum(dim=1) / (exp_neg.sum(dim=1) + 1e-8)

        # Mul by Proxy Net Weights
        hpwl_per_net_per_dim = wa_max - wa_min
        hpwl_per_net = hpwl_per_net_per_dim.sum(dim=-1)  # Sum X and Y
        weighted_hpwl = hpwl_per_net * self.net_weights
        return weighted_hpwl.sum()

    def congestion_score(
        self,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return approx smooth spatial routing-demand congestion energy.

        E_congestion = mean(max(congestion - capacity, 0)^2)

        Args:
            pos: centres [N, 2]

        Returns:
            Scalar congestion energy.
        """
        if self.net_indices is None:
            return torch.tensor(0.0, dtype=pos.dtype, device=self.device)

        net_coords = pos[self.net_indices]
        valid_mask = self.net_masks > 0

        # Smooth net bounding-box boundaries.
        masked_x = torch.where(
            valid_mask, net_coords[:, :, 0], torch.full_like(net_coords[:, :, 0], -1e9)
        )
        masked_neg_x = torch.where(
            valid_mask, net_coords[:, :, 0], torch.full_like(net_coords[:, :, 0], -1e9)
        )
        masked_y = torch.where(
            valid_mask, net_coords[:, :, 1], torch.full_like(net_coords[:, :, 1], -1e9)
        )
        masked_neg_y = torch.where(
            valid_mask, -net_coords[:, :, 1], torch.full_like(net_coords[:, :, 1], -1e9)
        )

        smoothing = self.congestion_smoothing
        max_x = smoothing * torch.logsumexp(masked_x / smoothing, dim=1)
        min_x = -smoothing * torch.logsumexp(masked_neg_x / smoothing, dim=1)
        max_y = smoothing * torch.logsumexp(masked_y / smoothing, dim=1)
        min_y = -smoothing * torch.logsumexp(masked_neg_y / smoothing, dim=1)

        # Grid segment receives demand when inside the bounding box of a net.
        gx = self.grid_x.unsqueeze(0)
        gy = self.grid_y.unsqueeze(0)
        min_x = min_x.unsqueeze(-1).unsqueeze(-1)
        max_x = max_x.unsqueeze(-1).unsqueeze(-1)
        min_y = min_y.unsqueeze(-1).unsqueeze(-1)
        max_y = max_y.unsqueeze(-1).unsqueeze(-1)

        inside_x = torch.sigmoid((gx - min_x) / smoothing) * torch.sigmoid(
            (max_x - gx) / smoothing
        )
        inside_y = torch.sigmoid((gy - min_y) / smoothing) * torch.sigmoid(
            (max_y - gy) / smoothing
        )
        net_demand = inside_x * inside_y
        congestion_grid = net_demand.sum(dim=0)

        # Proxy evaluates Top 5%
        e_congestion = self._abu_score(congestion_grid, 0.05)
        return e_congestion

    def optimize(
        self,
        pos_init: np.ndarray,
        movable: np.ndarray,
        num_steps: int = 500,
        lr: float = 0.05,
        abu_weight: float = 1.0,
        temp_start: float = 1.0,
        temp_end: float = 0.001,
        gamma: float = 2.0,
        callback=None,
    ) -> np.ndarray:
        """Stochastic Langevin optimization.

        Args:
            pos_init: [N, 2] initial macro centers.
            movable: Boolean array of shape [N]; False indicates fixed/pinned macros.
            num_steps: Total sampling time steps.
            lr: Step size multiplying the energy gradient.
            abu_weight: Scaling weight balancing ABU overflow penalty vs wirelength energy.
            temp_start: Initial thermal noise scale (high exploration / tunneling).
            temp_end: Final thermal noise scale (pure grad descent).
            gamma: Smoothing parameter for weighted-average wirelength.
            callback: Optional placement callback called after each step.

        Returns:
            Array of shape [N, 2] containing globally optimized positions.
        """
        pos = torch.tensor(
            pos_init,
            dtype=torch.float32,
            device=self.device,
        )

        movable_mask = torch.tensor(
            movable,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(-1)

        low = self.half_sizes
        high = (
            torch.tensor(
                [self.cw, self.ch],
                dtype=torch.float32,
                device=self.device,
            )
            - self.half_sizes
        )

        # Keep the original fixed macro locations unchanged.
        fixed_pos = pos.clone()

        for step in range(num_steps):
            decay = step / max(num_steps - 1, 1)  # temp annealing.

            if temp_start == 0.0 or temp_end == 0.0 and step == num_steps - 1:
                T_t = 0.0
            else:
                T_t = temp_start * (temp_end / temp_start) ** decay

            pos_req = pos.detach().requires_grad_(True)
            e_wl = self.wirelength_score(pos_req, gamma=gamma)
            e_den = self.density_score(pos_req)
            e_cong = self.congestion_score(pos_req)
            total_energy = e_wl + abu_weight * e_den + abu_weight * e_cong

            grad = torch.autograd.grad(total_energy, pos_req)[0]

            noise = torch.randn_like(pos_req)
            update = -lr * grad + np.sqrt(2.0 * lr * T_t) * noise
            candidate = pos_req.detach() + update

            # Clamp only movable macros.
            candidate = torch.clamp(candidate, low, high)
            pos = torch.where(
                movable_mask,
                candidate,
                fixed_pos,
            )

            if callback is not None:
                callback(step, pos.detach().cpu().numpy(), total_energy.detach().item())

        return pos.detach().cpu().numpy().astype(np.float64)
