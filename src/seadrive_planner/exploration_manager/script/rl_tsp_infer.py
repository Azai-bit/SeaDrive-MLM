#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _classical_mds_from_distance(dmat: np.ndarray, out_dim: int = 2) -> np.ndarray:
    # dmat: symmetric distance matrix, shape (n, n)
    n = dmat.shape[0]
    d2 = np.square(dmat)
    j = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    b = -0.5 * j.dot(d2).dot(j)

    # Eigen decomposition (symmetric matrix)
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    keep = min(out_dim, n)
    lam = np.maximum(eigvals[:keep], 0.0)
    vec = eigvecs[:, :keep]
    x = vec * np.sqrt(lam + 1e-12)

    if keep < out_dim:
        pad = np.zeros((n, out_dim - keep), dtype=np.float64)
        x = np.concatenate([x, pad], axis=1)

    return x.astype(np.float32)


def _load_actor(model_root: str, n_nodes: int, w2: float, use_cuda: bool):
    import torch

    project_root = os.path.abspath(model_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from model import DRL4TSP
    from tasks import motsp

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Prefer model trained on 20 for small n, 40 for larger n
    root20 = os.path.join(project_root, "tsp_transfer_100run_500000_5epoch_20city", "20")
    root40 = os.path.join(project_root, "tsp_transfer_100run_500000_5epoch_40city", "40")
    base_dir = root20 if (n_nodes <= 20 and os.path.isdir(root20)) else root40
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"No model dir found: {root20} or {root40}")

    w2 = min(max(float(w2), 0.0), 1.0)
    w1 = 1.0 - w2
    folder = f"w_{w1:0.2f}_{w2:0.2f}"
    actor_path = os.path.join(base_dir, folder, "actor.pt")
    if not os.path.isfile(actor_path):
        # fallback to balanced weight
        actor_path = os.path.join(base_dir, "w_0.50_0.50", "actor.pt")
        if not os.path.isfile(actor_path):
            raise FileNotFoundError(f"actor checkpoint not found in {base_dir}")

    actor = DRL4TSP(
        4,
        1,
        128,
        None,
        motsp.update_mask,
        1,
        0.1,
    ).to(device)

    state = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state)
    actor.eval()
    return actor, device


def _load_frontier_features(frontier_file: str, n: int) -> np.ndarray:
    # file format:
    # line1: n
    # next n lines: x y z
    with open(frontier_file, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        m = int(first)
        rows = []
        for _ in range(m):
            line = f.readline()
            if not line:
                break
            vals = [float(v) for v in line.strip().split()]
            rows.append(vals)

    arr = np.array(rows, dtype=np.float32)
    if arr.shape[0] != n or arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(
            f"invalid frontier feature shape {arr.shape}, expect ({n}, >=2)")

    if arr.shape[1] == 2:
        z = np.zeros((n, 1), dtype=np.float32)
        arr = np.concatenate([arr, z], axis=1)

    arr = arr[:, :3]
    return arr


def infer_tour(
    cost_mat: np.ndarray,
    model_root: str,
    w2: float = 0.5,
    use_cuda: bool = False,
    frontier_xyz: np.ndarray = None):
    import torch

    c = np.asarray(cost_mat, dtype=np.float32)
    n = c.shape[0]
    if c.ndim != 2 or c.shape[1] != n:
        raise ValueError("cost matrix must be square")

    # Build 4D static features for RL_TSP_4static.
    # Prefer real geometric frontier features when provided.
    if frontier_xyz is not None:
        xyz = np.asarray(frontier_xyz, dtype=np.float32)
        if xyz.shape != (n, 3):
            raise ValueError(f"frontier_xyz shape mismatch: {xyz.shape}, expect ({n}, 3)")
        x = _minmax_norm(xyz[:, 0])
        y = _minmax_norm(xyz[:, 1])
        z = _minmax_norm(xyz[:, 2])
        geom_feat = z
    else:
        # Fallback: pseudo coordinates from symmetric distance matrix.
        c_sym = 0.5 * (c + c.T)
        np.fill_diagonal(c_sym, 0.0)
        c_sym = np.maximum(c_sym, 0.0)
        xy = _classical_mds_from_distance(c_sym, out_dim=2)
        x = _minmax_norm(xy[:, 0])
        y = _minmax_norm(xy[:, 1])
        geom_feat = _minmax_norm(np.sqrt(np.maximum(x * x + y * y, 0.0)))

    row_mean = _minmax_norm(np.mean(c, axis=1))
    # Keep original 4D input size required by model:
    # [geom_x, geom_y, geom_z_or_radius, row_mean]
    static_np = np.stack([x, y, geom_feat, row_mean], axis=0)[None, ...]  # (1, 4, n)
    dynamic_np = np.zeros((1, 1, n), dtype=np.float32)

    actor, device = _load_actor(model_root, n, w2, use_cuda)

    static = torch.from_numpy(static_np).to(device)
    dynamic = torch.from_numpy(dynamic_np).to(device)

    with torch.no_grad():
        tour_idx, _ = actor(static, dynamic, None)

    tour = tour_idx[0].detach().cpu().numpy().astype(int).tolist()

    # Ensure valid permutation with de-duplication and padding
    seen = set()
    clean = []
    for i in tour:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            clean.append(i)
    for i in range(n):
        if i not in seen:
            clean.append(i)
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--model_root", required=True, type=str)
    parser.add_argument("--w2", default=0.5, type=float)
    parser.add_argument("--use_cuda", default=0, type=int)
    parser.add_argument("--frontier_file", default="", type=str)
    args = parser.parse_args()

    with open(args.cost_file, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        n = int(first)
        rows = []
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            vals = [float(v) for v in line.strip().split()]
            rows.append(vals)

    mat = np.array(rows, dtype=np.float32)
    if mat.shape != (n, n):
        raise RuntimeError(f"invalid matrix shape {mat.shape}, expect {(n, n)}")

    frontier_xyz = None
    if args.frontier_file:
        frontier_xyz = _load_frontier_features(args.frontier_file, n)

    tour = infer_tour(
        mat,
        args.model_root,
        w2=args.w2,
        use_cuda=bool(args.use_cuda),
        frontier_xyz=frontier_xyz)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for i in tour:
            f.write(f"{int(i)}\n")


if __name__ == "__main__":
    main()
