from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numba import njit
from scipy.optimize import linprog


def _build_graph(
    pos: np.ndarray,
    sizes: np.ndarray,
    gap: float = 0.05,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    From macro centers, build pairwise relative-order graph of separation constraints
    (each pair contributes one horizontal or vertical separation edge)


    Returns:
        h_edges: list of (i, j, min_sep) meaning x_j >= x_i + min_sep  ("i left of j")
        v_edges: list of (i, j, min_sep) meaning y_j >= y_i + min_sep  ("i below j")
    """
    n = pos.shape[0]
    half = sizes / 2.0

    # Upper triangle to avoid duplicate pairs
    i_indices, j_indices = np.triu_indices(n, k=1)
    dx = pos[j_indices, 0] - pos[i_indices, 0]
    dy = pos[j_indices, 1] - pos[i_indices, 1]

    sep_x = half[i_indices, 0] + half[j_indices, 0] + gap
    sep_y = half[i_indices, 1] + half[j_indices, 1] + gap

    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)

    # Overlap condition
    overlapping = (abs_dx < sep_x) & (abs_dy < sep_y)

    h_edges: List[Tuple[int, int, float]] = []
    v_edges: List[Tuple[int, int, float]] = []

    if not np.any(overlapping):
        return h_edges, v_edges

    # Filter overlapping pairs only
    i_overlap = i_indices[overlapping]
    j_overlap = j_indices[overlapping]
    dx_overlap = dx[overlapping]
    dy_overlap = dy[overlapping]
    sep_x_overlap = sep_x[overlapping]
    sep_y_overlap = sep_y[overlapping]

    # Add an edge if macros overlap on both axes
    overlap_x = sep_x_overlap - abs_dx[overlapping]
    overlap_y = sep_y_overlap - abs_dy[overlapping]
    use_x = overlap_x <= overlap_y  # prefer axis that needs less movement to resolve

    for idx in range(len(i_overlap)):
        i, j = i_overlap[idx], j_overlap[idx]
        if use_x[idx]:
            if dx_overlap[idx] >= 0:
                h_edges.append((i, j, float(sep_x_overlap[idx])))  # i left of j
            else:
                h_edges.append((j, i, float(sep_x_overlap[idx])))  # j left of i
        else:
            if dy_overlap[idx] >= 0:
                v_edges.append((i, j, float(sep_y_overlap[idx])))  # i below j
            else:
                v_edges.append((j, i, float(sep_y_overlap[idx])))  # j below i

    return h_edges, v_edges


def _bottom_compact(
    n: int,
    edges: List[Tuple[int, int, float]],
    low: np.ndarray,
    high: np.ndarray,
    fixed: np.ndarray,
    fixed_vals: np.ndarray,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """
    min Σ coord_i  (bottom / left compact)
    Returns: c, bounds, A_ub, b_ub
    """
    c = np.ones(n)
    bounds = []
    for i in range(n):
        if fixed[i]:
            v = float(np.clip(fixed_vals[i], low[i], high[i]))
            bounds.append((v, v))
        else:
            bounds.append((float(low[i]), float(high[i])))

    # coord_i - coord_j <= -w  <->  coord_j >= coord_i + w
    A_rows: List[np.ndarray] = []
    b_ub: List[float] = []
    for i, j, w in edges:
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = -1.0
        A_rows.append(row)
        b_ub.append(float(-w))

    A_ub = np.asarray(A_rows, dtype=float) if A_rows else np.zeros((0, n))
    return c, bounds, A_ub, np.asarray(b_ub, dtype=float)


def _min_displacement(
    n: int,
    edges: List[Tuple[int, int, float]],
    low: np.ndarray,
    high: np.ndarray,
    fixed: np.ndarray,
    fixed_vals: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """
    min Σ |coord_i - target_i|
    vars: coord[0:n], t[0:n] with t_i >= |coord_i - target_i|
    Returns: c, bounds, A_ub, b_ub
    """
    c = np.zeros(2 * n)
    c[n:] = 1.0

    bounds = []
    for i in range(n):
        if fixed[i]:
            v = float(np.clip(fixed_vals[i], low[i], high[i]))
            bounds.append((v, v))
        else:
            bounds.append((float(low[i]), float(high[i])))
    bounds.extend([(0.0, None)] * n)

    A_rows: List[np.ndarray] = []
    b_ub: List[float] = []

    for i in range(n):
        # t_i >= coord_i - target_i  ->  coord_i - t_i <= target_i
        row = np.zeros(2 * n)
        row[i] = 1.0
        row[n + i] = -1.0
        A_rows.append(row)
        b_ub.append(float(target[i]))

        # t_i >= target_i - coord_i  ->  -coord_i - t_i <= -target_i
        row = np.zeros(2 * n)
        row[i] = -1.0
        row[n + i] = -1.0
        A_rows.append(row)
        b_ub.append(float(-target[i]))

    for i, j, w in edges:
        row = np.zeros(2 * n)
        row[i] = 1.0
        row[j] = -1.0
        A_rows.append(row)
        b_ub.append(float(-w))

    A_ub = np.asarray(A_rows, dtype=float) if A_rows else np.zeros((0, 2 * n))
    return c, bounds, A_ub, np.asarray(b_ub, dtype=float)


def _solve_axis_lp(
    n: int,
    edges: List[Tuple[int, int, float]],
    low: np.ndarray,
    high: np.ndarray,
    fixed: np.ndarray,
    fixed_vals: np.ndarray,
    target: np.ndarray | None,
) -> np.ndarray | None:
    """
    Single-axis 1D linear programming solution (using scipy HiGHS).

    If target is given:
        min Σ |coord_i - target_i|
    else:
        min Σ coord_i   (bottom-left / bottom compact)

    Constraints:
        coord_j >= coord_i + w
        low <= coord <= high
        fixed nodes pinned
    """
    if target is not None:
        c, bounds, A_ub, b_ub = _min_displacement(
            n, edges, low, high, fixed, fixed_vals, target
        )
    else:
        c, bounds, A_ub, b_ub = _bottom_compact(n, edges, low, high, fixed, fixed_vals)

    if A_ub.shape[0] == 0:
        out = target.copy() if target is not None else low.copy()
        out = np.clip(out, low, high)
        out[fixed] = np.clip(fixed_vals[fixed], low[fixed], high[fixed])
        return out

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None

    return res.x[:n].copy()


@njit(fastmath=True)
def _find_best(
    target_x: float,
    target_y: float,
    hw_i: float,
    hh_i: float,
    cand_arr: np.ndarray,
    legal_pos: np.ndarray,
    half_sizes: np.ndarray,
    placed_idxs: np.ndarray,
    cw: float,
    ch: float,
    gap: float,
) -> np.ndarray:
    """
    JIT-compiled candidate search kernel.
    Returns the closest valid position (cx, cy).
    """
    num_placed = len(placed_idxs)
    num_cands = len(cand_arr)

    best_dist = 1e18
    best_x = min(max(target_x, hw_i), cw - hw_i)
    best_y = min(max(target_y, hh_i), ch - hh_i)

    for c in range(num_cands):
        cx = cand_arr[c, 0]
        cy = cand_arr[c, 1]

        # Bounds check
        if cx < hw_i - 1e-5 or cx > cw - hw_i + 1e-5:
            continue
        if cy < hh_i - 1e-5 or cy > ch - hh_i + 1e-5:
            continue

        # Early prune by distance
        d = (cx - target_x) ** 2 + (cy - target_y) ** 2
        if d >= best_dist:
            continue

        # Overlap check against placed macros
        collision = False
        for k in range(num_placed):
            p_idx = placed_idxs[k]
            dx = abs(cx - legal_pos[p_idx, 0])
            dy = abs(cy - legal_pos[p_idx, 1])
            req_x = hw_i + half_sizes[p_idx, 0] + gap - 1e-5
            req_y = hh_i + half_sizes[p_idx, 1] + gap - 1e-5

            if dx < req_x and dy < req_y:
                collision = True
                break

        if not collision:
            best_dist = d
            best_x = cx
            best_y = cy

    return np.array([best_x, best_y], dtype=np.float64)


def _greedy_legalize(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float,
) -> np.ndarray:
    """
    Final 2D candidate-point legalizer that guarantees no overlaps.
    Places macros in descending order of size onto gaps.
    """
    n = pos.shape[0]
    half = sizes / 2.0
    legal = pos.copy()

    # Lock fixed macros in place and sort movable macros by area
    placed = ~movable.copy()
    movable_indices = np.where(movable)[0]
    order = sorted(movable_indices, key=lambda i: -sizes[i, 0] * sizes[i, 1])

    for idx in order:
        hw_i, hh_i = half[idx]
        target_x, target_y = pos[idx]
        placed_idxs = np.where(placed)[0]

        # Already legal?
        if len(placed_idxs) > 0:
            dx = np.abs(legal[idx, 0] - legal[placed_idxs, 0])
            dy = np.abs(legal[idx, 1] - legal[placed_idxs, 1])
            req_x = hw_i + half[placed_idxs, 0] + gap - 1e-5
            req_y = hh_i + half[placed_idxs, 1] + gap - 1e-5

            if not np.any((dx < req_x) & (dy < req_y)):
                placed[idx] = True
                continue

        # Generate candidate positions (canvas corners + edges of existing placed macros)
        target_x_clamped = min(max(target_x, hw_i), cw - hw_i)
        target_y_clamped = min(max(target_y, hh_i), ch - hh_i)
        cand_x = [hw_i, cw - hw_i, target_x_clamped]
        cand_y = [hh_i, ch - hh_i, target_y_clamped]

        for j in placed_idxs:
            px, py = legal[j]
            hx, hy = half[j]
            cand_x.extend([px - hx - hw_i - gap, px + hx + hw_i + gap])
            cand_y.extend([py - hy - hh_i - gap, py + hy + hh_i + gap])

        valid_x = [x for x in cand_x if hw_i - 1e-5 <= x <= cw - hw_i + 1e-5]
        valid_y = [y for y in cand_y if hh_i - 1e-5 <= y <= ch - hh_i + 1e-5]

        full_grid = np.array(
            [(x, y) for x in valid_x for y in valid_y], dtype=np.float64
        )

        best_p = _find_best(
            target_x,
            target_y,
            hw_i,
            hh_i,
            full_grid,
            legal,
            half,
            placed_idxs,
            cw,
            ch,
            gap,
        )

        legal[idx] = best_p
        placed[idx] = True

    return legal


def legalize_graph(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    gap: float = 0.1,
    prefer_displacement: bool = True,
    max_iters: int = 3,
) -> np.ndarray:
    """
    Legalize hard-macro centers via separation constraints + iterative LP.

      - prefer_displacement=True  -> minimize Σ|coord_i - pos_i|
      - prefer_displacement=False -> minimize Σ coord_i (compact to origin side)

    Args:
        pos:                 [n, 2] macro centers (not modified)
        sizes:               [n, 2] (width, height)
        movable:             [n] bool; False -> fixed
        canvas_width:        canvas width in the same units as pos
        canvas_height:       canvas height
        gap:                 extra clearance added to half-size separations
        prefer_displacement: use min-displacement objective when True
        max_iters:           resolving along one axis can introduce new overlaps along the other, so iteratively refine.


    Returns:
        legal_pos [n, 2] with zero hard-macro overlaps (up to numerical tol),
        inside the canvas, fixed macros unchanged.
    """
    pos = np.asarray(pos, dtype=np.float64).copy()
    sizes = np.asarray(sizes, dtype=np.float64)
    movable = np.asarray(movable, dtype=bool)
    n = pos.shape[0]
    half = sizes / 2.0
    fixed = ~movable

    low_x = half[:, 0].copy()
    high_x = canvas_width - half[:, 0]
    low_y = half[:, 1].copy()
    high_y = canvas_height - half[:, 1]
    fixed_x = np.clip(pos[:, 0], low_x, high_x)
    fixed_y = np.clip(pos[:, 1], low_y, high_y)

    current_pos = pos.copy()

    # LP passes to un-clutter overlapping groups
    for _ in range(max_iters):
        h_edges, v_edges = _build_graph(current_pos, sizes, gap=gap)

        if not h_edges and not v_edges:
            break

        target_x = current_pos[:, 0] if prefer_displacement else None
        target_y = current_pos[:, 1] if prefer_displacement else None

        xs = _solve_axis_lp(n, h_edges, low_x, high_x, fixed, fixed_x, target_x)
        ys = _solve_axis_lp(n, v_edges, low_y, high_y, fixed, fixed_y, target_y)

        if xs is None or ys is None:
            break

        current_pos[:, 0] = np.clip(xs, low_x, high_x)
        current_pos[:, 1] = np.clip(ys, low_y, high_y)

    # Final pass to guarantee no overlaps
    return _greedy_legalize(
        current_pos, sizes, movable, canvas_width, canvas_height, gap
    )
