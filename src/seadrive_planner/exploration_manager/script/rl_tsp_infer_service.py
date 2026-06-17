#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from typing import List, Tuple

import numpy as np
import rospy

from exploration_manager.srv import RLTSPInfer, RLTSPInferResponse


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _resolve_actor_ckpt(model_root: str) -> str:
    cand = os.path.join(model_root, "actor.pt")
    if os.path.isfile(cand):
        return cand
    if not os.path.isdir(model_root):
        raise FileNotFoundError("model_root does not exist")
    subdirs = [
        os.path.join(model_root, d)
        for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ]
    subdirs.sort(reverse=True)
    for sd in subdirs:
        cand2 = os.path.join(sd, "actor.pt")
        if os.path.isfile(cand2):
            return cand2
    raise FileNotFoundError("Cannot find actor.pt under model_root")


class RLTSPInferServer:
    def __init__(self):
        self.model_root = rospy.get_param("~model_root", "")
        self.use_cuda = bool(rospy.get_param("~use_cuda", False))
        self.service_name = rospy.get_param("~service_name", "/rl_tsp_infer")

        if not self.model_root:
            # Fallback to the exploration node parameter so model path can be
            # configured at a single place in launch files.
            for key in [
                "/exploration_node/exploration/rl_tsp_root",
                "/exploration/rl_tsp_root",
                "exploration/rl_tsp_root",
            ]:
                if rospy.has_param(key):
                    self.model_root = str(rospy.get_param(key, ""))
                    if self.model_root:
                        rospy.loginfo("RL-TSP service fallback model_root from %s", key)
                        break

        if not self.model_root:
            raise RuntimeError("~model_root is empty and fallback exploration/rl_tsp_root not found")

        actor_ckpt = _resolve_actor_ckpt(self.model_root)
        self.actor, self.device, self.static_size = self._load_actor(actor_ckpt, self.use_cuda)
        rospy.loginfo("RL-TSP service ready. ckpt=%s static_size=%d device=%s",
                      actor_ckpt, self.static_size, str(self.device))

        self.srv = rospy.Service(self.service_name, RLTSPInfer, self.handle_request)

    def _load_actor(self, actor_ckpt: str, use_cuda: bool = False):
        import torch

        ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        rl_root = os.path.join(ws_root, "RL_TSP_4static")
        if rl_root not in sys.path:
            sys.path.insert(0, rl_root)

        from model import DRL4TSP
        from tasks import atsp_open_explore as atsp

        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        # Always load checkpoint tensors on CPU first, then move model once.
        # This avoids mixed-device edge cases across different torch/cuda setups.
        state = torch.load(actor_ckpt, map_location="cpu")
        static_size = int(state["static_encoder.conv.weight"].shape[1])
        hidden_size = int(state["static_encoder.conv.weight"].shape[0])

        actor = DRL4TSP(static_size, 1, hidden_size, None, atsp.update_mask, 1, 0.1)
        # Backward compatibility: older checkpoints may not contain x0.
        load_ret = actor.load_state_dict(state, strict=False)
        missing = set(load_ret.missing_keys)
        unexpected = set(load_ret.unexpected_keys)
        allowed_missing = {"x0"}
        real_missing = missing - allowed_missing
        if real_missing or unexpected:
            raise RuntimeError(
                "Checkpoint mismatch. missing=%s unexpected=%s" %
                (sorted(real_missing), sorted(unexpected))
            )
        if missing:
            rospy.logwarn("RL-TSP ckpt missing keys tolerated: %s", sorted(missing))
        actor = actor.to(device)
        actor.eval()

        # Verify the model is really on the target device.
        param_dev = next(actor.parameters()).device
        if param_dev != device:
            rospy.logwarn("RL-TSP actor on %s, expected %s. Forcing move.", str(param_dev), str(device))
            actor = actor.to(device)

        return actor, device, static_size

    def _build_static_features(self, cost: np.ndarray, frontier_feat: np.ndarray) -> np.ndarray:
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
            eye = np.eye(n, dtype=np.float32)
            h = eye - (1.0 / float(n))
            b = -0.5 * h.dot(c_sym ** 2).dot(h)
            eigvals, eigvecs = np.linalg.eigh(b)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            lam = np.maximum(eigvals[:2], 0.0)
            xy = eigvecs[:, :2] * np.sqrt(lam + 1e-12)
            xyz = np.concatenate([xy.astype(np.float32), np.zeros((n, 1), dtype=np.float32)], axis=1)

        if not np.any(start_cost):
            start_cost = _minmax_norm(np.mean(cost, axis=1))

        return atsp.build_static_features(xyz, yaw, gain, start_cost, cost, self.static_size)[None, ...]

    def _infer_tour(self, cost: np.ndarray, frontier_feat: np.ndarray) -> List[int]:
        import torch

        n = cost.shape[0]
        static_np = self._build_static_features(cost, frontier_feat)
        dynamic_np = np.zeros((1, 1, n), dtype=np.float32)
        # Use the actor parameter device as the source of truth.
        model_device = next(self.actor.parameters()).device
        if model_device != self.device:
            rospy.logwarn_throttle(5.0, "RL-TSP device mismatch detected: actor=%s self.device=%s",
                                   str(model_device), str(self.device))
            self.device = model_device

        static = torch.from_numpy(static_np).to(model_device)
        dynamic = torch.from_numpy(dynamic_np).to(model_device)

        with torch.no_grad():
            tour_idx, _ = self.actor(static, dynamic, None)

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

    def _parse_request(self, req: RLTSPInfer) -> Tuple[np.ndarray, np.ndarray]:
        n = int(req.num_nodes)
        if n <= 1:
            raise ValueError("num_nodes must be > 1")

        cm = np.asarray(req.cost_matrix, dtype=np.float32)
        if cm.size != n * n:
            raise ValueError(f"cost_matrix size mismatch: {cm.size} vs {n*n}")
        cost = cm.reshape((n, n))

        ff = np.asarray(req.frontier_feat, dtype=np.float32)
        if ff.size == 0:
            frontier = np.zeros((n, 6), dtype=np.float32)
        else:
            if ff.size % n != 0:
                raise ValueError(f"frontier_feat size {ff.size} not divisible by n={n}")
            cols = ff.size // n
            if cols < 3:
                raise ValueError("frontier_feat columns must be >= 3")
            frontier = ff.reshape((n, cols))
        return cost, frontier

    def handle_request(self, req: RLTSPInfer) -> RLTSPInferResponse:
        t0 = time.time()
        try:
            cost, frontier = self._parse_request(req)
            tour = self._infer_tour(cost, frontier)
            dt_ms = float((time.time() - t0) * 1000.0)
            return RLTSPInferResponse(True, tour, "ok", dt_ms)
        except Exception as e:
            dt_ms = float((time.time() - t0) * 1000.0)
            rospy.logwarn("RL-TSP infer failed: %s", str(e))
            return RLTSPInferResponse(False, [], str(e), dt_ms)


def main():
    rospy.init_node("rl_tsp_infer_service")
    RLTSPInferServer()
    rospy.spin()


if __name__ == "__main__":
    main()
