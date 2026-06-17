#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import shutil
import time
from typing import Optional, Tuple


def _read_matrix_size(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        return int(first)
    except Exception:
        return None


def _is_cost_file_consistent(path: str) -> Tuple[bool, int]:
    n = _read_matrix_size(path)
    if n is None or n <= 1:
        return False, -1

    rows = 0
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            s = line.strip()
            if not s:
                continue
            cols = len(s.split())
            if cols != n:
                return False, n
            rows += 1

    return rows == n, n


def _is_frontier_file_consistent(path: str, n_expected: int) -> bool:
    n = _read_matrix_size(path)
    if n is None or n != n_expected:
        return False

    rows = 0
    col_ref = None
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                return False
            if col_ref is None:
                col_ref = len(parts)
            elif len(parts) != col_ref:
                return False
            rows += 1

    return rows == n_expected


def _is_snapshot_pair_consistent(cost_path: str, frontier_path: str, num_nodes_filter: int) -> Tuple[bool, int]:
    ok_cost, n = _is_cost_file_consistent(cost_path)
    if not ok_cost:
        return False, -1
    if num_nodes_filter > 0 and n != num_nodes_filter:
        return False, n
    if not _is_frontier_file_consistent(frontier_path, n):
        return False, n
    return True, n


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sample_prefix(idx: int) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"sample_{idx:06d}_{ts}"


def _resolve_input_paths(tsp_dir: str) -> Tuple[str, str, str]:
    cost_file = os.path.join(tsp_dir, "rl_cost.txt")
    frontier_file = os.path.join(tsp_dir, "rl_frontier_feat.txt")
    tour_file = os.path.join(tsp_dir, "rl_tour.txt")
    return cost_file, frontier_file, tour_file


def _write_index_header(index_path: str):
    if os.path.exists(index_path):
        return
    with open(index_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "timestamp", "session_id", "step_id", "num_nodes", "cost_file", "frontier_file", "tour_file",
            "cost_sha1", "frontier_sha1", "tour_sha1"
        ])


def _append_index(index_path: str, row):
    with open(index_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def main():
    p = argparse.ArgumentParser("Record online RL snapshots from tsp_dir files")
    p.add_argument("--tsp_dir", type=str, default=os.path.join(os.getcwd(), "tmp", "lkh_tsp_solver", "resource"))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_nodes", type=int, default=0)
    p.add_argument("--session_id", type=str, default="")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "index.csv")
    _write_index_header(index_path)

    cost_in, frontier_in, tour_in = _resolve_input_paths(args.tsp_dir)
    print("[info] monitoring:", cost_in, frontier_in, tour_in)
    print("[info] output_dir:", args.output_dir)

    last_cost_mtime = -1.0
    last_pair_hash = ""
    sample_id = 0
    session_id = args.session_id if args.session_id else time.strftime("%Y%m%d_%H%M%S", time.localtime())

    while True:
        if args.max_samples > 0 and sample_id >= args.max_samples:
            print("[done] reached max_samples", args.max_samples)
            return

        if not (os.path.isfile(cost_in) and os.path.isfile(frontier_in) and os.path.isfile(tour_in)):
            time.sleep(args.interval)
            continue

        try:
            cost_mtime = os.path.getmtime(cost_in)
            if cost_mtime <= last_cost_mtime:
                time.sleep(args.interval)
                continue

            # The planner writes files asynchronously. Retry a few times to avoid
            # copying half-written snapshots.
            ok_pair = False
            n_cost = -1
            for _ in range(4):
                ok_pair, n_cost = _is_snapshot_pair_consistent(cost_in, frontier_in, args.num_nodes)
                if ok_pair:
                    break
                time.sleep(max(0.01, min(0.05, args.interval * 0.5)))
            if not ok_pair:
                time.sleep(args.interval)
                continue

            cost_hash = _sha1(cost_in)
            frontier_hash = _sha1(frontier_in)
            tour_hash = _sha1(tour_in)
            pair_hash = cost_hash + ":" + frontier_hash + ":" + tour_hash
            if pair_hash == last_pair_hash:
                last_cost_mtime = cost_mtime
                time.sleep(args.interval)
                continue

            prefix = _sample_prefix(sample_id)
            cost_out = os.path.join(args.output_dir, prefix + "_cost.txt")
            frontier_out = os.path.join(args.output_dir, prefix + "_frontier.txt")
            tour_out = os.path.join(args.output_dir, prefix + "_tour.txt")
            shutil.copy2(cost_in, cost_out)
            shutil.copy2(frontier_in, frontier_out)
            shutil.copy2(tour_in, tour_out)

            _append_index(index_path, [
                sample_id,
                round(time.time(), 3),
                session_id,
                sample_id,
                n_cost,
                os.path.basename(cost_out),
                os.path.basename(frontier_out),
                os.path.basename(tour_out),
                cost_hash,
                frontier_hash,
                tour_hash,
            ])

            sample_id += 1
            last_cost_mtime = cost_mtime
            last_pair_hash = pair_hash
            print(f"[saved] id={sample_id:06d} n={n_cost} cost={os.path.basename(cost_out)}")

        except KeyboardInterrupt:
            print("\n[done] interrupted by user")
            return
        except Exception as e:
            print("[warn] capture failed:", e)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
