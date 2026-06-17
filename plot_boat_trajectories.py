#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bisect
import csv
import glob
import math
import os

import matplotlib.pyplot as plt


MOTION_START_DISTANCE_M = 0.2
GOAL_RECORD_TYPES = {"goal", "goal_point", "target_point", "waypoint"}
FONT_SIZE_BASE = 14
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 13
FONT_SIZE_LEGEND = 13
FONT_SIZE_ANNOTATION = 12
FONT_SIZE_COLLISION = 14
PLOT_X_MIN = -480.0
PLOT_X_MAX = -420.0
PLOT_Y_MIN = 220.0
PLOT_Y_MAX = 280.0
LABEL_CANDIDATE_OFFSETS_PT = [
    (4, 4),
    (8, 8),
    (-8, 8),
    (8, -8),
    (-8, -8),
    (14, 0),
    (-14, 0),
    (0, 14),
    (0, -14),
    (18, 10),
    (-18, 10),
    (18, -10),
    (-18, -10),
    (24, 0),
    (-24, 0),
    (0, 24),
    (0, -24),
]


def _annotate_without_overlap(
    ax,
    renderer,
    placed_bboxes,
    text,
    xy,
    color,
    fontsize,
    force_place=False,
):
    bbox_style = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75)

    for off in LABEL_CANDIDATE_OFFSETS_PT:
        ann = ax.annotate(
            text,
            xy,
            xytext=off,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            bbox=bbox_style,
            annotation_clip=True,
        )
        ann.set_clip_on(True)
        ann.set_clip_box(ax.bbox)
        bb = ann.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
        if any(bb.overlaps(pb) for pb in placed_bboxes):
            ann.remove()
            continue
        placed_bboxes.append(bb)
        return ann

    if force_place:
        off = LABEL_CANDIDATE_OFFSETS_PT[-1]
        ann = ax.annotate(
            text,
            xy,
            xytext=off,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            bbox=bbox_style,
            annotation_clip=True,
        )
        ann.set_clip_on(True)
        ann.set_clip_box(ax.bbox)
        bb = ann.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
        placed_bboxes.append(bb)
        return ann
    return None


def _find_motion_start_time(track, distance_threshold_m=MOTION_START_DISTANCE_M):
    ts = track.get("t", [])
    xs = track.get("x", [])
    ys = track.get("y", [])
    if not ts or not xs or not ys:
        return None

    x0 = xs[0]
    y0 = ys[0]
    for i in range(1, min(len(ts), len(xs), len(ys))):
        if math.hypot(xs[i] - x0, ys[i] - y0) >= max(0.0, distance_threshold_m):
            return ts[i]
    return ts[0]


def load_tracks(csv_path):
    tracks = {}
    constraints = {}
    all_stamps = []
    prev_raw_stamp = None
    rollover_offset = 0.0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = (row.get("record_type") or "track").strip().lower()
            group = (row.get("group") or "").strip()
            boat = (row.get("boat") or "").strip()
            if not boat:
                continue
            try:
                stamp_raw = float(row.get("stamp_s", "nan"))
                x = float(row.get("x", "nan"))
                y = float(row.get("y", "nan"))
            except Exception:
                continue
            try:
                yaw = float(row.get("yaw", "nan"))
            except Exception:
                yaw = float("nan")

            if record_type == "constraint_point":
                key = group or boat
                constraints.setdefault(key, {"x": [], "y": []})
                constraints[key]["x"].append(x)
                constraints[key]["y"].append(y)
                continue
            if record_type in GOAL_RECORD_TYPES:
                continue

            # Some rosbag/sim sessions may reset /clock to a smaller value.
            # Unwrap timestamps to keep timeline monotonic for annotations.
            if prev_raw_stamp is not None and stamp_raw < (prev_raw_stamp - 0.1):
                rollover_offset += prev_raw_stamp
            stamp_s = stamp_raw + rollover_offset
            prev_raw_stamp = stamp_raw

            tracks.setdefault(boat, {"t": [], "x": [], "y": [], "yaw": []})
            tracks[boat]["t"].append(stamp_s)
            tracks[boat]["x"].append(x)
            tracks[boat]["y"].append(y)
            tracks[boat]["yaw"].append(yaw)
            all_stamps.append(stamp_s)

    if all_stamps:
        t0 = min(all_stamps)
        myboat_track = tracks.get("myboat")
        motion_t0 = _find_motion_start_time(myboat_track) if myboat_track else None
        if motion_t0 is not None:
            t0 = motion_t0
        for data in tracks.values():
            data["t"] = [max(0.0, t - t0) for t in data["t"]]

    return tracks, constraints


def _first_track_stamp(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = (row.get("record_type") or "track").strip().lower()
            if record_type == "constraint_point" or record_type in GOAL_RECORD_TYPES:
                continue
            try:
                stamp_s = float(row.get("stamp_s", "nan"))
                float(row.get("x", "nan"))
                float(row.get("y", "nan"))
                return stamp_s
            except Exception:
                continue
    return None


def _safe_stamp_label(stamp_s):
    if stamp_s is None or not math.isfinite(stamp_s):
        return "tunknown"
    text = "%.3f" % float(stamp_s)
    return "t" + text.replace("-", "m").replace(".", "_")


def _plot_base_name(csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return "%s_%s" % (stem, _safe_stamp_label(_first_track_stamp(csv_path)))


def _pick_label_indices(ts, interval_s, max_labels_per_track):
    if not ts:
        return []
    indices = [0]
    next_t = interval_s
    for i, t in enumerate(ts):
        if t >= next_t:
            indices.append(i)
            next_t += interval_s
        if len(indices) >= max_labels_per_track:
            break
    if (len(ts) - 1) not in indices:
        indices.append(len(ts) - 1)
    return sorted(set(indices))


def _interp_xy(ts, xs, ys, t):
    if not ts or t < ts[0] or t > ts[-1]:
        return None

    idx = bisect.bisect_left(ts, t)
    if idx < len(ts) and abs(ts[idx] - t) <= 1e-9:
        return (xs[idx], ys[idx])
    if idx == 0:
        return (xs[0], ys[0])
    if idx >= len(ts):
        return (xs[-1], ys[-1])

    t0 = ts[idx - 1]
    t1 = ts[idx]
    if abs(t1 - t0) <= 1e-9:
        return (xs[idx], ys[idx])

    ratio = (t - t0) / (t1 - t0)
    x = xs[idx - 1] + ratio * (xs[idx] - xs[idx - 1])
    y = ys[idx - 1] + ratio * (ys[idx] - ys[idx - 1])
    return (x, y)


def _find_interval_collision(a, b, own_start, own_end, target_start, target_end, threshold_m):
    dt = b - a
    if dt < 0.0:
        return None

    rx0 = own_start[0] - target_start[0]
    ry0 = own_start[1] - target_start[1]
    threshold_sq = threshold_m * threshold_m
    start_dist_sq = rx0 * rx0 + ry0 * ry0
    if start_dist_sq < threshold_sq:
        return a
    if dt <= 1e-9:
        return None

    rx1 = own_end[0] - target_end[0]
    ry1 = own_end[1] - target_end[1]
    vx = (rx1 - rx0) / dt
    vy = (ry1 - ry0) / dt

    qa = vx * vx + vy * vy
    qb = 2.0 * (rx0 * vx + ry0 * vy)
    qc = start_dist_sq - threshold_sq

    if qa <= 1e-12:
        return a if qc < 0.0 else None

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(max(0.0, disc))
    s0 = (-qb - sqrt_disc) / (2.0 * qa)
    s1 = (-qb + sqrt_disc) / (2.0 * qa)
    enter_s = min(s0, s1)
    exit_s = max(s0, s1)
    if exit_s < 0.0 or enter_s > dt:
        return None

    if enter_s <= 0.0 <= exit_s:
        return a
    if 0.0 <= enter_s <= dt:
        return a + enter_s
    return None


def find_first_collision(tracks, own_boat="myboat", threshold_m=2.0):
    own_track = tracks.get(own_boat)
    if not own_track or not own_track.get("t"):
        return None

    earliest = None
    own_ts = own_track["t"]
    own_xs = own_track["x"]
    own_ys = own_track["y"]

    for boat, target_track in tracks.items():
        if boat == own_boat or not target_track.get("t"):
            continue

        target_ts = target_track["t"]
        target_xs = target_track["x"]
        target_ys = target_track["y"]
        overlap_start = max(own_ts[0], target_ts[0])
        overlap_end = min(own_ts[-1], target_ts[-1])
        if overlap_start > overlap_end:
            continue

        candidate_ts = sorted(
            set(t for t in own_ts if overlap_start <= t <= overlap_end)
            | set(t for t in target_ts if overlap_start <= t <= overlap_end)
            | {overlap_start, overlap_end}
        )
        if len(candidate_ts) == 1:
            t = candidate_ts[0]
            own_xy = _interp_xy(own_ts, own_xs, own_ys, t)
            target_xy = _interp_xy(target_ts, target_xs, target_ys, t)
            if own_xy and target_xy and math.dist(own_xy, target_xy) < threshold_m:
                earliest = {
                    "boat": boat,
                    "t": t,
                    "myboat_xy": own_xy,
                    "target_xy": target_xy,
                    "point": own_xy,
                }
            continue

        for a, b in zip(candidate_ts, candidate_ts[1:]):
            own_start = _interp_xy(own_ts, own_xs, own_ys, a)
            own_end = _interp_xy(own_ts, own_xs, own_ys, b)
            target_start = _interp_xy(target_ts, target_xs, target_ys, a)
            target_end = _interp_xy(target_ts, target_xs, target_ys, b)
            if not own_start or not own_end or not target_start or not target_end:
                continue

            collision_t = _find_interval_collision(
                a,
                b,
                own_start,
                own_end,
                target_start,
                target_end,
                threshold_m,
            )
            if collision_t is None:
                continue

            own_xy = _interp_xy(own_ts, own_xs, own_ys, collision_t)
            target_xy = _interp_xy(target_ts, target_xs, target_ys, collision_t)
            if not own_xy or not target_xy:
                continue

            hit = {
                "boat": boat,
                "t": collision_t,
                "myboat_xy": own_xy,
                "target_xy": target_xy,
                "point": own_xy,
            }
            if earliest is None or collision_t < earliest["t"]:
                earliest = hit
            break

    return earliest


def plot_tracks(
    tracks,
    constraints,
    out_png,
    title,
    label_interval_s=2.0,
    max_labels_per_track=12,
    min_label_gap_m=1.0,
    collision_distance_m=2.0,
):
    if not tracks:
        raise RuntimeError("no_tracks_found")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
        }
    )

    colors = {
        "myboat": "#1f77b4",
        "target_boat": "#d62728",
        "target_boat_2": "#2ca02c",
        "target_boat_3": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.92)
    constraint_styles = {
        "channel_left": {"color": "#6b7280", "marker": "s", "label": "channel_left"},
        "channel_right": {"color": "#4b5563", "marker": "s", "label": "channel_right"},
    }
    for group, data in sorted(constraints.items()):
        xs = data.get("x", [])
        ys = data.get("y", [])
        if not xs or not ys:
            continue
        style = constraint_styles.get(group, {"color": "#6b7280", "marker": ".", "label": group})
        ax.plot(xs, ys, linestyle="--", linewidth=1.2, color=style["color"], alpha=0.65)
        ax.scatter(xs, ys, marker=style["marker"], s=28, color=style["color"], alpha=0.85, label=style["label"])

    label_requests = []
    for boat, data in tracks.items():
        ts = data.get("t", [])
        xs = data["x"]
        ys = data["y"]
        if not xs:
            continue
        c = colors.get(boat, None)
        ax.plot(xs, ys, linewidth=2.0, label=boat, color=c)
        ax.scatter([xs[0]], [ys[0]], marker="o", s=35, color=c)
        ax.scatter([xs[-1]], [ys[-1]], marker="x", s=50, color=c)

        last_label_xy = None
        picked = _pick_label_indices(ts, label_interval_s, max_labels_per_track)
        for idx in picked:
            if idx >= len(xs) or idx >= len(ys) or idx >= len(ts):
                continue
            is_last = idx == (len(xs) - 1)
            if last_label_xy is not None:
                dx = xs[idx] - last_label_xy[0]
                dy = ys[idx] - last_label_xy[1]
                if (dx * dx + dy * dy) ** 0.5 < min_label_gap_m and not is_last:
                    continue
            label = "%.1fs" % ts[idx]
            label_requests.append(
                {
                    "t": float(ts[idx]),
                    "label": label,
                    "x": float(xs[idx]),
                    "y": float(ys[idx]),
                    "color": c,
                    "force": bool(is_last),
                }
            )
            last_label_xy = (xs[idx], ys[idx])

    collision = find_first_collision(tracks, threshold_m=collision_distance_m)
    if collision is not None:
        cx, cy = collision["point"]
        ax.scatter([cx], [cy], marker="*", s=180, color="#b22222", zorder=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_bboxes = []

    label_requests.sort(key=lambda d: (0 if d["force"] else 1, d["t"]))
    for req in label_requests:
        _annotate_without_overlap(
            ax,
            renderer,
            placed_bboxes,
            text=req["label"],
            xy=(req["x"], req["y"]),
            color=req["color"],
            fontsize=FONT_SIZE_ANNOTATION,
            force_place=req["force"],
        )

    if collision is not None:
        cx, cy = collision["point"]
        ann = ax.annotate(
            "collision",
            (cx, cy),
            xytext=(-40, -22),
            textcoords="offset points",
            fontsize=FONT_SIZE_COLLISION,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
            annotation_clip=True,
        )
        ann.set_clip_on(True)
        ann.set_clip_box(ax.bbox)
        bb = ann.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
        if any(bb.overlaps(pb) for pb in placed_bboxes):
            ann.remove()
            _annotate_without_overlap(
                ax,
                renderer,
                placed_bboxes,
                text="collision",
                xy=(cx, cy),
                color="black",
                fontsize=FONT_SIZE_COLLISION,
                force_place=True,
            )
        else:
            placed_bboxes.append(bb)

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _unwrap_angles(angles):
    if not angles:
        return []
    out = [angles[0]]
    offset = 0.0
    prev = angles[0]
    for angle in angles[1:]:
        delta = angle - prev
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        out.append(angle + offset)
        prev = angle
    return out


def plot_myboat_yaw(tracks, out_png, title):
    myboat = tracks.get("myboat")
    if not myboat:
        raise RuntimeError("no_myboat_track_found")

    ts = myboat.get("t", [])
    yaws = myboat.get("yaw", [])
    samples = [
        (float(t), float(yaw))
        for t, yaw in zip(ts, yaws)
        if yaw is not None and math.isfinite(float(yaw))
    ]
    if not samples:
        raise RuntimeError("no_myboat_yaw_found")

    plot_ts = [item[0] for item in samples]
    plot_yaws_deg = [math.degrees(yaw) for yaw in _unwrap_angles([item[1] for item in samples])]

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.88)
    ax.plot(plot_ts, plot_yaws_deg, linewidth=2.0, color="#1f77b4")
    ax.scatter([plot_ts[0]], [plot_yaws_deg[0]], marker="o", s=35, color="#1f77b4")
    ax.scatter([plot_ts[-1]], [plot_yaws_deg[-1]], marker="x", s=50, color="#1f77b4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("myboat yaw (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _iter_csv_files(input_dir, pattern):
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def process_csv(
    csv_path,
    out_dir=None,
    title=None,
    out_png=None,
    label_interval_s=2.0,
    max_labels_per_track=3,
    min_label_gap_m=5.0,
    collision_distance_m=2.0,
):
    tracks, constraints = load_tracks(csv_path)
    if not tracks:
        raise RuntimeError("no_tracks_found")

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    base_title = title or csv_stem
    tracks_title = title or (csv_stem + " Trajectories")
    yaw_title = base_title + " Myboat Yaw"
    if out_png:
        tracks_png = out_png
        yaw_png = os.path.splitext(out_png)[0] + "_myboat_yaw.png"
    else:
        plot_dir = out_dir or os.path.join(os.path.dirname(csv_path), "plot")
        base_name = _plot_base_name(csv_path)
        tracks_png = os.path.join(plot_dir, base_name + "_tracks.png")
        yaw_png = os.path.join(plot_dir, base_name + "_myboat_yaw.png")

    plot_tracks(
        tracks,
        constraints,
        tracks_png,
        tracks_title,
        label_interval_s=max(0.1, label_interval_s),
        max_labels_per_track=max(2, max_labels_per_track),
        min_label_gap_m=max(0.0, min_label_gap_m),
        collision_distance_m=max(0.0, collision_distance_m),
    )
    plot_myboat_yaw(tracks, yaw_png, yaw_title)
    return tracks_png, yaw_png


def main():
    parser = argparse.ArgumentParser(description="Plot myboat/target trajectories from CSV files")
    parser.add_argument(
        "path",
        nargs="?",
        help="input CSV path or directory. Directory mode writes images to <directory>/plot",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="input CSV path",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="input directory containing CSV files",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="CSV filename pattern used in directory mode",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="maximum number of CSV files to read in directory mode; <=0 means all",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output PNG path for single CSV mode",
    )
    parser.add_argument(
        "--title",
        default="Boat Trajectories",
        help="figure title",
    )
    parser.add_argument(
        "--label-interval",
        type=float,
        default=2.0,
        help="seconds between time labels on each trajectory",
    )
    parser.add_argument(
        "--max-labels-per-track",
        type=int,
        default=3,
        help="max number of time labels for each trajectory",
    )
    parser.add_argument(
        "--min-label-gap",
        type=float,
        default=5.0,
        help="minimum spatial gap (meters) between two labels on one trajectory",
    )
    parser.add_argument(
        "--collision-distance",
        type=float,
        default=2.0,
        help="distance threshold (meters) used to mark the first collision point",
    )
    args = parser.parse_args()

    input_path = args.dir or args.csv or args.path or os.path.join(os.getcwd(), "tmp", "boat_tracks.csv")
    if os.path.isdir(input_path):
        csv_files = _iter_csv_files(input_path, args.pattern)
        if not csv_files:
            raise RuntimeError("no_csv_files_found: %s" % input_path)
        saved_count = 0
        for csv_path in csv_files:
            if args.max_files > 0 and saved_count >= args.max_files:
                break
            try:
                tracks_png, yaw_png = process_csv(
                    csv_path,
                    out_dir=os.path.join(os.path.dirname(csv_path), "plot"),
                    title=None,
                    label_interval_s=args.label_interval,
                    max_labels_per_track=args.max_labels_per_track,
                    min_label_gap_m=args.min_label_gap,
                    collision_distance_m=args.collision_distance,
                )
            except RuntimeError as exc:
                print("skipped: %s (%s)" % (csv_path, exc))
                continue
            print("saved:", tracks_png)
            print("saved:", yaw_png)
            saved_count += 1
        print("processed_csv:", saved_count)
        return

    tracks_png, yaw_png = process_csv(
        input_path,
        out_png=args.out,
        title=args.title,
        label_interval_s=args.label_interval,
        max_labels_per_track=args.max_labels_per_track,
        min_label_gap_m=args.min_label_gap,
        collision_distance_m=args.collision_distance,
    )
    print("saved:", tracks_png)
    print("saved:", yaw_png)


if __name__ == "__main__":
    main()
