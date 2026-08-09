from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


class BoltzmannPlacer:
    """EBM-based stochastic global placer using annealed Langevin dynamics.

    E(P) = E_wirelength(P) + density_weight * E_density(P)

    where:
        - E_wirelength is a smooth weighted-average approx of HPWL.
        - E_density is a pairwise Gaussian-overlap energy.

    The class performs projected, annealed Langevin updates:

    P[t+1] = P[t] - lr * grad(E) + sqrt(2 * lr * T[t]) * noise

    Fixed macros participate in the energy but are not moved.

    Args:
        sizes: np array of shape [N, 2], containing (width, height).
        nets: List of list of macro/node indices.
        canvas_width: width of the placement canvas.
        canvas_height: height of the placement canvas.
        gap: Gaussian interaction scale added to each macro half-size.
        device: PyTorch device, e.g. "cuda" or "cpu".
    """

    def __init__(
        self,
        sizes: np.ndarray,
        nets: list[list[int]],
        canvas_width: float,
        canvas_height: float,
        gap: float = 0.1,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")
        self.device = torch.device(device)

        sizes = np.asarray(sizes, dtype=np.float32)
        self.cw = float(canvas_width)
        self.ch = float(canvas_height)

        self.sizes = torch.as_tensor(
            sizes,
            dtype=torch.float32,
            device=self.device,
        )

        self.n = sizes.shape[0]
        self.gap = float(gap)
        self.half_sizes = self.sizes / 2.0
        self.sigmas = (self.half_sizes + self.gap) / np.sqrt(2.0)
        sigmas_sq = self.sigmas.square()

        # var_sum[i, j, d] = sigma[i, d]^2 + sigma[j, d]^2
        self.var_sum = sigmas_sq.unsqueeze(1) + sigmas_sq.unsqueeze(0)
        self.var_sum = self.var_sum.clamp_min(1e-8)
        valid_nets: list[list[int]] = []

        for net in nets:
            if len(net) <= 1:
                continue

            net = list(net)

            for node in net:
                if node < 0 or node >= self.n:
                    raise ValueError(
                        f"Net contains invalid node index {node}; "
                        f"valid range is [0, {self.n - 1}]."
                    )

            valid_nets.append(net)

        self.num_nets = len(valid_nets)

        if valid_nets:
            max_net_len = max(len(net) for net in valid_nets)
            padded_nets = []
            net_masks = []

            for net in valid_nets:
                pad_len = max_net_len - len(net)
                padded_nets.append(net + [0] * pad_len)
                net_masks.append([1.0] * len(net) + [0.0] * pad_len)

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

        # Diagonal mask used by density energy.
        if self.n > 0:
            self.nonself_mask = 1.0 - torch.eye(
                self.n,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.nonself_mask = None

        self.density_nx = 64
        self.density_ny = 64

        self.grid_x = (
            torch.arange(
                self.density_nx,
                dtype=torch.float32,
                device=self.device,
            )
            + 0.5
        ) * (self.cw / self.density_nx)

        self.grid_y = (
            torch.arange(
                self.density_ny,
                dtype=torch.float32,
                device=self.device,
            )
            + 0.5
        ) * (self.ch / self.density_ny)

        grid_x, grid_y = torch.meshgrid(
            self.grid_x,
            self.grid_y,
            indexing="xy",
        )

        self.density_grid = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        )

        self.target_density = self.n / (self.density_nx * self.density_ny)

    def density_score(
        self,
        pos: torch.Tensor,
        short_range_weight: float = 0.25,
        canvas_density_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Density energy combining:

        1. Pairwise Gaussian repulsion 0.5 * sum_{i != j} E_ij.
        2. Canvas-aware density matching.

        Args:
            pos: centres, shape [N, 2].
            short_range_weight: strength of short-range pairwise Gaussian.
            canvas_density_weight: weight of canvas occupancy penalty.

        Returns:
            Scalar density energy.
        """
        if self.n <= 1:
            pairwise_energy = torch.zeros(
                (),
                dtype=pos.dtype,
                device=pos.device,
            )

        else:
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            norm_dist_sq = diff.square() / self.var_sum
            total_dist_sq = norm_dist_sq.sum(dim=-1)
            overlap = torch.exp(-0.5 * total_dist_sq)

            # Stronger short-range interaction.
            short_var_sum = (self.var_sum * 0.25).clamp_min(1e-8)
            short_dist_sq = (diff.square() / short_var_sum).sum(dim=-1)
            short_overlap = torch.exp(-0.5 * short_dist_sq)
            overlap = overlap + short_range_weight * short_overlap
            overlap = overlap * self.nonself_mask
            pairwise_energy = 0.5 * overlap.sum()

        diff_grid = pos[:, None, None, :] - self.density_grid[None, :, :, :]
        sigma = self.sigmas[:, None, None, :].clamp_min(1e-4)
        exponent = -0.5 * (diff_grid.square() / sigma.square()).sum(dim=-1)

        gaussian_mass = torch.exp(exponent)
        gaussian_mass = gaussian_mass / gaussian_mass.sum(
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(1e-8)
        density = gaussian_mass.sum(dim=0)

        # Penalize deviation from uniform density.
        target = torch.full_like(
            density,
            self.target_density,
        )
        density_error = density - target
        canvas_energy = 0.5 * (density_error.square().mean())
        return pairwise_energy + canvas_density_weight * canvas_energy

    def wirelength_score(
        self,
        pos: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Smooth weighted-average HPWL.

        Approx HPWL = max(x) - min(x)

        Args:
            pos: centres of shape [N, 2].
            gamma: Smoothing temperature.

        Returns:
            Scalar wirelength energy.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive.")

        if self.net_indices is None:
            return torch.zeros(
                (),
                dtype=pos.dtype,
                device=pos.device,
            )

        net_coords = pos[self.net_indices]
        mask = self.net_masks.unsqueeze(-1)

        # Invalid padded entries receive a very negative value.
        neg_inf = torch.finfo(pos.dtype).min
        max_logits = torch.where(
            mask > 0,
            net_coords / gamma,
            torch.full_like(
                net_coords,
                neg_inf,
            ),
        )

        max_weights = torch.softmax(
            max_logits,
            dim=1,
        )
        max_weights = max_weights * mask
        max_weights = max_weights / max_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)

        wa_max = (net_coords * max_weights).sum(dim=1)
        min_logits = torch.where(
            mask > 0,
            -net_coords / gamma,
            torch.full_like(
                net_coords,
                neg_inf,
            ),
        )

        min_weights = torch.softmax(
            min_logits,
            dim=1,
        )
        min_weights = min_weights * mask
        min_weights = min_weights / min_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
        wa_min = (net_coords * min_weights).sum(dim=1)

        # Sum x/y wirelength over all nets.
        wirelength = (wa_max - wa_min).sum()
        return wirelength

    def energy(
        self,
        pos: torch.Tensor,
        density_weight: float = 10.0,
        gamma: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, wirelength, and density energies."""
        e_density = self.density_score(pos)
        e_wirelength = self.wirelength_score(
            pos,
            gamma=gamma,
        )

        total = e_wirelength + density_weight * e_density
        return (
            total,
            e_wirelength,
            e_density,
        )

    def optimize(
        self,
        pos_init: np.ndarray,
        movable: np.ndarray,
        num_steps: int = 500,
        lr: float = 0.05,
        density_weight: float = 10.0,
        gamma: float = 2.0,
        temp_start: float = 1.0,
        temp_end: float = 0.001,
        gradient_clip: float | None = None,
        noise_scale: float = 1.0,
        verbose: bool = False,
        log_every: int = 50,
        callback: Callable | None = None,
    ) -> np.ndarray:
        """Optimize macro positions with annealed Langevin dynamics.

        P[t+1] = P[t] - lr * grad(E) + sqrt(2 * lr * T[t]) * noise


        Args:
            pos_init: init center coordinates, shape [N, 2].
            movable: True = movable.
            num_steps: Langevin iterations.
            lr: step size.
            density_weight: density energy weighting.
            gamma: wirelength smoothing parameter.
            temp_start: init Langevin temperature.
            temp_end: final Langevin temperature.
            gradient_clip: None disables clipping.
            noise_scale: multiplier on Langevin noise.
            verbose: print diagnostics.
            log_every: print every N iterations.

        Returns:
            numpy array of centres, shape [N, 2].
        """
        pos = torch.as_tensor(
            pos_init,
            dtype=torch.float32,
            device=self.device,
        ).clone()

        movable_mask = torch.as_tensor(
            movable,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(-1)

        low = self.half_sizes
        canvas = torch.tensor(
            [self.cw, self.ch],
            dtype=torch.float32,
            device=self.device,
        )

        high = canvas - self.half_sizes

        # Keep fixed macros where they started.
        fixed_pos = pos.clone()
        projected = torch.minimum(
            torch.maximum(pos, low),
            high,
        )
        pos = torch.where(
            movable_mask,
            projected,
            fixed_pos,
        )

        # Avoid log(0) when temp_end == 0.
        schedule_temp_end = max(temp_end, 1e-12)

        for step in range(num_steps):
            if num_steps == 1:
                decay = 1.0
            else:
                decay = step / float(num_steps - 1)

            # Geometric annealing.
            if temp_start == 0.0:
                temperature = 0.0

            elif temp_end == 0.0:
                temperature = temp_start * (schedule_temp_end / temp_start) ** decay

                # Zero on final
                if step == num_steps - 1:
                    temperature = 0.0

            else:
                temperature = temp_start * (temp_end / temp_start) ** decay

            pos_req = pos.detach().requires_grad_(True)
            e_total, e_wl, e_density = self.energy(
                pos_req,
                density_weight=density_weight,
                gamma=gamma,
            )

            grad = torch.autograd.grad(
                e_total,
                pos_req,
                create_graph=False,
                retain_graph=False,
            )[0]

            grad = torch.where(
                movable_mask,
                grad,
                torch.zeros_like(grad),
            )

            if gradient_clip is not None:
                grad_norm = torch.linalg.vector_norm(grad)

                clip_scale = torch.clamp(
                    gradient_clip / grad_norm.clamp_min(1e-12),
                    max=1.0,
                )

                grad = grad * clip_scale

            noise = torch.randn_like(pos_req)
            noise = noise * movable_mask
            noise_amplitude = noise_scale * np.sqrt(
                max(
                    0.0,
                    2.0 * lr * temperature,
                )
            )

            update = -lr * grad + noise_amplitude * noise
            candidate = pos_req.detach() + update
            candidate = torch.minimum(
                torch.maximum(
                    candidate,
                    low,
                ),
                high,
            )

            pos = torch.where(
                movable_mask,
                candidate,
                fixed_pos,
            )

            if callback is not None:
                callback(
                    step=step,
                    pos=pos,
                    temperature=temperature,
                    energy=e_total,
                    wirelength=e_wl,
                    density=e_density,
                )

            if verbose and (step % log_every == 0 or step == num_steps - 1):
                grad_norm_value = torch.linalg.vector_norm(grad).detach().item()

                movable_pos = pos[movable_mask.expand_as(pos)].reshape(-1, 2)
                if movable_pos.numel() > 0:
                    min_xy = movable_pos.min(dim=0).values
                    max_xy = movable_pos.max(dim=0).values
                    min_x = min_xy[0].item()
                    min_y = min_xy[1].item()
                    max_x = max_xy[0].item()
                    max_y = max_xy[1].item()
                else:
                    min_x = min_y = max_x = max_y = 0.0

                print(
                    f"[{step + 1:4d}/{num_steps}] "
                    f"T={temperature:.5g} "
                    f"E={e_total.item():.5g} "
                    f"WL={e_wl.item():.5g} "
                    f"D={e_density.item():.5g} "
                    f"|grad|={grad_norm_value:.5g} "
                    f"bbox=({min_x:.2f},{min_y:.2f})"
                    f"-({max_x:.2f},{max_y:.2f})"
                )

        return pos.detach().cpu().numpy().astype(np.float64)
