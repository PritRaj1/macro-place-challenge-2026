from __future__ import annotations

from typing import List, Tuple

import numpy as np
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
    h_edges: List[Tuple[int, int, float]] = []
    v_edges: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            sep_x = half[i, 0] + half[j, 0] + gap
            sep_y = half[i, 1] + half[j, 1] + gap

            if abs(dx) >= abs(dy):  # Prefer axis with clearer separation
                if dx >= 0:
                    h_edges.append((i, j, sep_x))  # i left of j
                else:
                    h_edges.append((j, i, sep_x))  # j left of i
            else:
                if dy >= 0:
                    v_edges.append((i, j, sep_y))  # i below j
                else:
                    v_edges.append((j, i, sep_y))  # j below i

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
        # t_i >= coord_i - target_i  →  coord_i - t_i <= target_i
        row = np.zeros(2 * n)
        row[i] = 1.0
        row[n + i] = -1.0
        A_rows.append(row)
        b_ub.append(float(target[i]))

        # t_i >= target_i - coord_i  →  -coord_i - t_i <= -target_i
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


def _sequential(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float,
) -> np.ndarray:
    """Sequential fallback from Will's seed"""
    n = pos.shape[0]
    half = sizes / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap
    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            hit = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
            hit[idx] = False
            if not hit.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_p = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, 150):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue

                    cx = float(
                        np.clip(
                            pos[idx, 0] + dxm * step, half[idx, 0], cw - half[idx, 0]
                        )
                    )
                    cy = float(
                        np.clip(
                            pos[idx, 1] + dym * step, half[idx, 1], ch - half[idx, 1]
                        )
                    )
                    if placed.any():
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        hit = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
                        hit[idx] = False
                        if hit.any():
                            continue

                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    return legal


def legalize_graph(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    gap: float = 0.05,
    prefer_displacement: bool = True,
) -> np.ndarray:
    """
    Legalize hard-macro centers via separation constraints + LP.

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

    Returns:
        legal_pos [n, 2] with zero hard-macro overlaps (up to numerical tol),
        inside the canvas, fixed macros unchanged.
    """
    pos = np.asarray(pos, dtype=np.float64)
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

    h_edges, v_edges = _build_graph(pos, sizes, gap=gap)

    target_x = pos[:, 0] if prefer_displacement else None
    target_y = pos[:, 1] if prefer_displacement else None

    xs = _solve_axis_lp(n, h_edges, low_x, high_x, fixed, fixed_x, target_x)
    ys = _solve_axis_lp(n, v_edges, low_y, high_y, fixed, fixed_y, target_y)

    # Fallback
    if xs is None or ys is None:
        return _sequential(pos, sizes, movable, canvas_width, canvas_height, gap)

    legal = pos.copy()
    legal[:, 0] = np.clip(xs, low_x, high_x)
    legal[:, 1] = np.clip(ys, low_y, high_y)
    return legal
