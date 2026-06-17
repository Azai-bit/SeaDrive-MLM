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


def _load_cost(cost_file: str) -> np.ndarray:
    with open(cost_file, "r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        rows = []
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            rows.append([float(v) for v in line.strip().split()])
    m = np.array(rows, dtype=np.float32)
    if m.shape != (n, n):
        raise RuntimeError(f"invalid matrix shape {m.shape}, expect {(n, n)}")
    return m


def _load_frontier(frontier_file: str, n: int) -> np.ndarray:
    with open(frontier_file, "r", encoding="utf-8") as f:
        m = int(f.readline().strip())
        rows = []
        for _ in range(m):
            line = f.readline()
            if not line:
                break
            rows.append([float(v) for v in line.strip().split()])
    arr = np.array(rows, dtype=np.float32)
    if arr.shape[0] != n or arr.shape[1] < 2:
        raise RuntimeError(f"invalid frontier shape {arr.shape}, expect ({n}, >=2)")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((n, 1), dtype=np.float32)], axis=1)
    return arr.astype(np.float32)


def _resolve_actor_ckpt(actor_ckpt: str, model_root: str) -> str:
    if actor_ckpt:
        return actor_ckpt
    if model_root:
        cand = os.path.join(model_root, "actor.pt")
        if os.path.isfile(cand):
            return cand
        subdirs = [
            os.path.join(model_root, d)
            for d in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, d))
        ] if os.path.isdir(model_root) else []
        subdirs.sort(reverse=True)
        for sd in subdirs:
            cand2 = os.path.join(sd, "actor.pt")
            if os.path.isfile(cand2):
                return cand2
    raise FileNotFoundError("Cannot resolve actor checkpoint. Provide --actor_ckpt or valid --model_root")


def _load_actor(actor_ckpt: str, use_cuda: bool = False):
    import torch

    root = os.path.abspath(os.path.join(os.path.dirname(actor_ckpt), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    if os.path.dirname(actor_ckpt) not in sys.path:
        sys.path.insert(0, os.path.dirname(actor_ckpt))

    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    rl_root = os.path.join(ws_root, "RL_TSP_4static")
    if rl_root not in sys.path:
        sys.path.insert(0, rl_root)

    from model import DRL4TSP
    from tasks import atsp_open_explore as atsp

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    state = torch.load(actor_ckpt, map_location=device)
    static_size = int(state["static_encoder.conv.weight"].shape[1])
    hidden_size = int(state["static_encoder.conv.weight"].shape[0])
    actor = DRL4TSP(static_size, 1, hidden_size, None, atsp.update_mask, 1, 0.1).to(device)
    actor.load_state_dict(state)
    actor.eval()
    return actor, device, static_size


def _build_static_features(cost: np.ndarray, frontier_feat: np.ndarray, static_size: int) -> np.ndarray:
    from tasks import atsp_open_explore as atsp

    n = cost.shape[0]
    xyz = np.zeros((n, 3), dtype=np.float32)
    yaw = np.zeros((n,), dtype=np.float32)
    gain = np.ones((n,), dtype=np.float32)
    start_cost = np.zeros((n,), dtype=np.float32)

    if frontier_feat is not None:
        if frontier_feat.shape[0] != n or frontier_feat.shape[1] < 3:
            raise ValueError(f"frontier feature shape mismatch {frontier_feat.shape}, expect ({n}, >=3)")
        xyz = frontier_feat[:, :3].astype(np.float32)
        if frontier_feat.shape[1] >= 4:
            yaw = frontier_feat[:, 3].astype(np.float32)
        if frontier_feat.shape[1] >= 5:
            gain = frontier_feat[:, 4].astype(np.float32)
        if frontier_feat.shape[1] >= 6:
            start_cost = frontier_feat[:, 5].astype(np.float32)

    if not np.any(xyz):
        c_sym = 0.5 * (cost + cost.T)
        np.fill_diagonal(c_sym, 0.0)
        eigvals, eigvecs = np.linalg.eigh(-0.5 * (np.eye(n) - 1.0 / n).dot(c_sym ** 2).dot(np.eye(n) - 1.0 / n))
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        lam = np.maximum(eigvals[:2], 0.0)
        xy = eigvecs[:, :2] * np.sqrt(lam + 1e-12)
        xyz = np.concatenate([xy.astype(np.float32), np.zeros((n, 1), dtype=np.float32)], axis=1)

    if not np.any(start_cost):
        start_cost = _minmax_norm(np.mean(cost, axis=1))

    return atsp.build_static_features(xyz, yaw, gain, start_cost, cost, static_size)[None, ...]


def infer_tour(cost: np.ndarray, frontier_feat: np.ndarray, actor_ckpt: str, use_cuda: bool = False):
    import torch

    n = cost.shape[0]
    actor, device, static_size = _load_actor(actor_ckpt, use_cuda=use_cuda)
    static_np = _build_static_features(cost, frontier_feat, static_size)
    dynamic_np = np.zeros((1, 1, n), dtype=np.float32)
    static = torch.from_numpy(static_np).to(device)
    dynamic = torch.from_numpy(dynamic_np).to(device)

    with torch.no_grad():
        tour_idx, _ = actor(static, dynamic, None)

    tour = tour_idx[0].detach().cpu().numpy().astype(int).tolist()
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
    p = argparse.ArgumentParser()
    p.add_argument("--cost_file", required=True, type=str)
    p.add_argument("--frontier_file", default="", type=str)
    p.add_argument("--actor_ckpt", default="", type=str)
    p.add_argument("--model_root", default="", type=str)
    p.add_argument("--w2", default=0.5, type=float)
    p.add_argument("--use_cuda", default=0, type=int)
    p.add_argument("--output_file", required=True, type=str)
    args = p.parse_args()

    cost = _load_cost(args.cost_file)
    frontier_feat = _load_frontier(args.frontier_file, cost.shape[0]) if args.frontier_file else None

    ckpt = _resolve_actor_ckpt(args.actor_ckpt, args.model_root)
    tour = infer_tour(cost, frontier_feat, ckpt, use_cuda=bool(args.use_cuda))

    with open(args.output_file, "w", encoding="utf-8") as f:
        for i in tour:
            f.write(f"{int(i)}\n")


if __name__ == "__main__":
    main()
