from __future__ import annotations

import numpy as np
import torch


class BoltzmannPlacer:
    """EBM-based global placer using ula sampling.

    E(P) = Wirelength_Energy(P)
         + Density_Energy(P)
         + Congestion_Energy(P).

    Attributes:
        device (torch.device): CUDA or CPU compute device.
        cw (float): Canvas width boundary.
        ch (float): Canvas height boundary.
        sizes (torch.Tensor): Tensor of shape [N, 2] containing (width, height).
        half_sizes (torch.Tensor): Tensor of shape [N, 2] containing (w/2, h/2).
        sigmas (torch.Tensor): Standard deviations of Gaussian [N, 2].
        var_sum (torch.Tensor): Pairwise variance sum matrix [N, N, 2] for Gaussian.
        net_indices (torch.Tensor): Padded tensor [M, K] mapping nets to macro node IDs.
        net_masks (torch.Tensor): Tensor [M, K] masking padding elements in nets.
        grid_x (torch.Tensor): Congestion grid X coordinates.
        grid_y (torch.Tensor): Congestion grid Y coordinates.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        nets: list[list[int]],
        canvas_width: float,
        canvas_height: float,
        gap: float = 0.1,
        congestion_grid: tuple[int, int] = (32, 32),
        congestion_capacity: float = 1.0,
        congestion_smoothing: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        print(f"Using device: {device}")
        self.device = torch.device(device)
        self.cw = float(canvas_width)
        self.ch = float(canvas_height)
        self.sizes = torch.tensor(
            sizes,
            dtype=torch.float32,
            device=self.device,
        )
        self.n = len(sizes)
        self.gap = gap

        self.half_sizes = self.sizes / 2.0
        self.sigmas = (self.half_sizes + gap) / np.sqrt(2.0)

        # Pairwise var sums: var_sum[i, j, dim] = sigmas[i]^2 + sigmas[j]^2
        sigmas_sq = self.sigmas**2
        self.var_sum = sigmas_sq.unsqueeze(1) + sigmas_sq.unsqueeze(0)

        # Congestion grid
        self.congestion_nx = congestion_grid[0]
        self.congestion_ny = congestion_grid[1]
        self.congestion_capacity = congestion_capacity
        self.congestion_smoothing = congestion_smoothing

        grid_x = (
            torch.arange(
                self.congestion_nx,
                dtype=torch.float32,
                device=self.device,
            )
            + 0.5
        ) * (self.cw / self.congestion_nx)

        grid_y = (
            torch.arange(
                self.congestion_ny,
                dtype=torch.float32,
                device=self.device,
            )
            + 0.5
        ) * (self.ch / self.congestion_ny)

        self.grid_x, self.grid_y = torch.meshgrid(
            grid_x,
            grid_y,
            indexing="xy",
        )

        # Pad for batching
        max_net_len = max(len(net) for net in nets) if nets else 0
        padded_nets = []
        net_masks = []

        for net in nets:
            if len(net) <= 1:
                continue

            pad_len = max_net_len - len(net)

            padded_nets.append(net + [0] * pad_len)

            mask = [1.0] * len(net) + [0.0] * pad_len

            net_masks.append(mask)

        if padded_nets:
            self.net_indices = torch.tensor(
                padded_nets,
                dtype=torch.long,
                device=self.device,
            )

            self.net_masks = torch.tensor(
                net_masks,
                dtype=torch.float32,
                device=self.device,
            )

        else:
            self.net_indices = None
            self.net_masks = None

    def density_score(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Return smooth Gaussian overlap energy.

        E_density = 0.5 * sum_ij exp(-0.5 * norm_distance^2)
        Diagonal is masked so that a macro does not repel itself.

        Args:
            pos: centres [N, 2]

        Returns:
            Scalar density energy.
        """
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)

        norm_dist_sq = (diff**2) / self.var_sum

        total_dist_sq = norm_dist_sq.sum(dim=-1)

        # Gaussian overlap energy
        overlap_weight = torch.exp(-0.5 * total_dist_sq)

        # Mask out self-overlap on diagonal
        overlap = overlap_weight * (
            1.0
            - torch.eye(
                self.n,
                device=self.device,
                dtype=pos.dtype,
            )
        )

        # Each pair appears twice: (i, j) and (j, i)
        return 0.5 * overlap.sum()

    def wirelength_score(
        self,
        pos: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Return score of weighted-avg wirelength

        Smooths bounding-box HPWL = (max(X) - min(X)) using ePlace / DREAMPlace method.
        """
        e_wl = torch.tensor(
            0.0,
            device=self.device,
        )

        if self.net_indices is not None:
            net_coords = pos[self.net_indices]

            # WA Max Boundary
            max_coords = torch.where(
                self.net_masks.unsqueeze(-1) > 0,
                net_coords,
                torch.full_like(
                    net_coords,
                    -1e9,  # Invalid padded receive -ve inf.
                ),
            )

            shift_max = max_coords.max(
                dim=1,
                keepdim=True,
            ).values.detach()

            exp_pos = torch.exp(
                (net_coords - shift_max) / gamma
            ) * self.net_masks.unsqueeze(-1)
            wa_max = (net_coords * exp_pos).sum(dim=1) / (exp_pos.sum(dim=1) + 1e-8)

            # WA Min Boundary
            min_coords = torch.where(
                self.net_masks.unsqueeze(-1) > 0,
                -net_coords,
                torch.full_like(
                    net_coords,
                    -1e9,  # Invalid padded receive -ve inf.
                ),
            )

            shift_min = min_coords.max(
                dim=1,
                keepdim=True,
            ).values.detach()

            exp_neg = torch.exp(
                (-net_coords - shift_min) / gamma
            ) * self.net_masks.unsqueeze(-1)

            wa_min = (net_coords * exp_neg).sum(dim=1) / (exp_neg.sum(dim=1) + 1e-8)

            # Total HPWL = sum(Max - Min) across all nets and dims
            e_wl = (wa_max - wa_min).sum()

        return e_wl

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
            return torch.tensor(
                0.0,
                dtype=pos.dtype,
                device=self.device,
            )

        net_coords = pos[self.net_indices]

        # Smooth net bounding-box boundaries.
        valid_mask = self.net_masks > 0
        masked_x = torch.where(
            valid_mask,
            net_coords[:, :, 0],
            torch.full_like(
                net_coords[:, :, 0],
                -1e9,
            ),
        )

        masked_neg_x = torch.where(
            valid_mask,
            -net_coords[:, :, 0],
            torch.full_like(
                net_coords[:, :, 0],
                -1e9,
            ),
        )

        masked_y = torch.where(
            valid_mask,
            net_coords[:, :, 1],
            torch.full_like(
                net_coords[:, :, 1],
                -1e9,
            ),
        )

        masked_neg_y = torch.where(
            valid_mask,
            -net_coords[:, :, 1],
            torch.full_like(
                net_coords[:, :, 1],
                -1e9,
            ),
        )

        smoothing = self.congestion_smoothing

        max_x = smoothing * torch.logsumexp(
            masked_x / smoothing,
            dim=1,
        )
        min_x = -smoothing * torch.logsumexp(
            masked_neg_x / smoothing,
            dim=1,
        )

        max_y = smoothing * torch.logsumexp(
            masked_y / smoothing,
            dim=1,
        )
        min_y = -smoothing * torch.logsumexp(
            masked_neg_y / smoothing,
            dim=1,
        )

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
        congestion = net_demand.sum(dim=0)

        # Penalize only regions whose demand exceeds capacity.
        overflow = torch.relu(congestion - self.congestion_capacity)
        e_congestion = overflow.square().mean()
        return e_congestion

    def optimize(
        self,
        pos_init: np.ndarray,
        movable: np.ndarray,
        num_steps: int = 500,
        lr: float = 0.05,
        density_weight: float = 10.0,
        congestion_weight: float = 1.0,
        temp_start: float = 1.0,
        temp_end: float = 0.001,
        gamma: float = 2.0,
        callback=None,
    ) -> np.ndarray:
        """Stochastic Langevin optimization.

        P_{t+1} = P_t - lr * grad(E) + sqrt(2 * lr * T_t) * Noise

        This is ULA using the Boltzmann score:
            score(P) = grad log p(P) = -grad(E) / T
        with score-step-size epsilon = lr * T.

        Args:
            pos_init: [N, 2] initial macro centers.
            movable: Boolean array of shape [N]; False indicates fixed/pinned macros.
            num_steps: Total sampling time steps.
            lr: Step size multiplying the energy gradient.
            density_weight: Scaling weight balancing overlap energy vs wirelength energy.
            congestion_weight: Scaling weight balancing routing congestion energy.
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
            decay = step / max(
                num_steps - 1,
                1,
            )  # temp annealing.

            if temp_start == 0.0 or temp_end == 0.0 and step == num_steps - 1:
                T_t = 0.0
            else:
                T_t = temp_start * (temp_end / temp_start) ** decay

            pos_req = pos.detach().requires_grad_(True)
            e_density = self.density_score(pos_req)
            e_wl = self.wirelength_score(
                pos_req,
                gamma=gamma,
            )
            e_congestion = self.congestion_score(pos_req)

            total_energy = (
                e_wl + density_weight * e_density + congestion_weight * e_congestion
            )

            grad = torch.autograd.grad(
                total_energy,
                pos_req,
            )[0]

            noise = torch.randn_like(pos_req)
            update = -lr * grad + np.sqrt(2.0 * lr * T_t) * noise
            candidate = pos_req.detach() + update

            # Clamp only movable macros.
            candidate = torch.clamp(
                candidate,
                low,
                high,
            )
            pos = torch.where(
                movable_mask,
                candidate,
                fixed_pos,
            )

            if callback is not None:
                callback(
                    step,
                    pos.detach().cpu().numpy(),
                    total_energy.detach().item(),
                )

        return pos.detach().cpu().numpy().astype(np.float64)
