#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from plot_boat_trajectories import (
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    PLOT_X_MAX,
    PLOT_X_MIN,
    PLOT_Y_MAX,
    PLOT_Y_MIN,
    _plot_base_name,
    load_tracks,
)


DEFAULT_COLORS = {
    "myboat": "#1f77b4",
    "target_boat": "#d62728",
    "target_boat_2": "#2ca02c",
    "target_boat_3": "#ff7f0e",
}


MOTION_START_DISTANCE_M = 0.2


def _is_numeric_boat_name(value):
    text = (value or "").strip()
    if not text:
        return False
    try:
        float(text)
    except Exception:
        return False
    return True


def _float_or_none(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _find_motion_start_raw_time(myboat_samples, distance_threshold_m=MOTION_START_DISTANCE_M):
    if not myboat_samples:
        return None
    t0, x0, y0 = myboat_samples[0]
    for stamp_s, x, y in myboat_samples[1:]:
        if math.hypot(x - x0, y - y0) >= max(0.0, distance_threshold_m):
            return stamp_s
    return t0


def load_vlm_instruction_events(csv_path):
    events = []
    track_stamps = []
    myboat_samples = []
    if not os.path.exists(csv_path):
        return events

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            boat = (row.get("boat") or "").strip()
            record_type = (row.get("record_type") or "track").strip().lower()
            stamp_s = _float_or_none(row.get("stamp_s"))
            if stamp_s is None:
                continue

            if _is_numeric_boat_name(boat):
                events.append(
                    {
                        "t_raw": stamp_s,
                        "call": boat,
                        "x_cmd": (row.get("x") or "").strip(),
                        "y_cmd": (row.get("y") or "").strip(),
                    }
                )
                continue

            if record_type != "track":
                continue
            x = _float_or_none(row.get("x"))
            y = _float_or_none(row.get("y"))
            if x is None or y is None:
                continue
            track_stamps.append(stamp_s)
            if boat == "myboat":
                myboat_samples.append((stamp_s, x, y))

    if not events or not track_stamps:
        return []

    t0 = min(track_stamps)
    motion_t0 = _find_motion_start_raw_time(myboat_samples)
    if motion_t0 is not None:
        t0 = motion_t0

    normalized = []
    for event in events:
        normalized.append(
            {
                "t": max(0.0, event["t_raw"] - t0),
                "call": event["call"],
                "x_cmd": event["x_cmd"],
                "y_cmd": event["y_cmd"],
            }
        )
    normalized.sort(key=lambda item: (item["t"], float(item["call"])))
    return normalized


def _read_goal_points_from_csv(csv_path):
    goals = []
    if not os.path.exists(csv_path):
        return goals

    goal_record_types = {"goal", "goal_point", "target_point", "waypoint"}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = (row.get("record_type") or "").strip().lower()
            group = (row.get("group") or "").strip().lower()
            boat = (row.get("boat") or "").strip()
            if record_type not in goal_record_types and group not in goal_record_types:
                continue
            x = _float_or_none(row.get("x"))
            y = _float_or_none(row.get("y"))
            if x is None or y is None:
                continue
            label = boat or group or "goal"
            goals.append((label, x, y))
    return goals


def _infer_goal_points_from_results(csv_path):
    results_csv = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "results.csv")
    if not os.path.exists(results_csv):
        return []

    csv_abs = os.path.abspath(csv_path)
    csv_base = os.path.basename(csv_path)
    with open(results_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory_csv = (row.get("trajectory_csv") or "").strip()
            if not trajectory_csv:
                continue
            candidates = {trajectory_csv, os.path.basename(trajectory_csv)}
            if os.path.isabs(trajectory_csv):
                candidates.add(os.path.abspath(trajectory_csv))
            else:
                candidates.add(os.path.abspath(os.path.join(os.path.dirname(results_csv), trajectory_csv)))
            if csv_abs not in candidates and csv_base not in candidates:
                continue
            x = _float_or_none(row.get("ego_goal_x"))
            y = _float_or_none(row.get("ego_goal_y"))
            if x is not None and y is not None:
                return [("goal", x, y)]
    return []


def load_goal_points(csv_path, goal_x=None, goal_y=None):
    goals = []
    if goal_x is not None and goal_y is not None:
        goals.append(("goal", float(goal_x), float(goal_y)))
    goals.extend(_read_goal_points_from_csv(csv_path))
    if not goals:
        goals.extend(_infer_goal_points_from_results(csv_path))

    unique = []
    seen = set()
    for label, x, y in goals:
        key = (label, round(float(x), 6), round(float(y), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, float(x), float(y)))
    return unique


def _target_boats(tracks, own_boat):
    return sorted(boat for boat in tracks.keys() if boat != own_boat and tracks[boat].get("x"))


def _animation_times(tracks, own_boat):
    own_track = tracks.get(own_boat)
    if own_track and own_track.get("t"):
        return list(own_track["t"])
    all_ts = sorted({t for data in tracks.values() for t in data.get("t", [])})
    return all_ts


def _playback_frame_times(source_times, output_fps, speed_multiplier):
    if not source_times:
        return []
    start_t = float(source_times[0])
    end_t = float(source_times[-1])
    if end_t <= start_t:
        return [start_t]

    fps = max(1, int(output_fps))
    speed = max(0.1, float(speed_multiplier))
    sim_step_s = speed / float(fps)
    frame_count = max(1, int(math.ceil((end_t - start_t) / sim_step_s)))
    frame_times = [start_t + i * sim_step_s for i in range(frame_count)]
    frame_times[-1] = end_t
    return frame_times


def _sample_index_at_time(ts, t):
    if not ts:
        return -1
    lo = 0
    hi = len(ts) - 1
    if t <= ts[lo]:
        return lo
    if t >= ts[hi]:
        return hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(0, hi)


def _interp_value(ts, values, t):
    if not ts or not values:
        return None
    idx = _sample_index_at_time(ts, t)
    if idx < 0:
        return None
    if idx >= len(ts) - 1 or t <= ts[idx]:
        return float(values[idx])

    t0 = float(ts[idx])
    t1 = float(ts[idx + 1])
    if t1 <= t0:
        return float(values[idx])
    ratio = (float(t) - t0) / (t1 - t0)
    return float(values[idx]) + ratio * (float(values[idx + 1]) - float(values[idx]))


def _track_xy_at_time(data, t):
    x = _interp_value(data.get("t", []), data.get("x", []), t)
    y = _interp_value(data.get("t", []), data.get("y", []), t)
    if x is None or y is None:
        return None
    return x, y


def _track_path_until(data, t, start_t=None):
    ts = data.get("t", [])
    xs = data.get("x", [])
    ys = data.get("y", [])
    if not ts or not xs or not ys:
        return [], []

    plot_xs = []
    plot_ys = []
    if start_t is not None and ts[0] < start_t < t:
        start_xy = _track_xy_at_time(data, start_t)
        if start_xy is not None:
            plot_xs.append(start_xy[0])
            plot_ys.append(start_xy[1])

    for sample_t, x, y in zip(ts, xs, ys):
        if start_t is not None and sample_t < start_t:
            continue
        if sample_t > t:
            break
        plot_xs.append(x)
        plot_ys.append(y)

    end_xy = _track_xy_at_time(data, t)
    if end_xy is not None:
        if not plot_xs or abs(plot_xs[-1] - end_xy[0]) > 1e-9 or abs(plot_ys[-1] - end_xy[1]) > 1e-9:
            plot_xs.append(end_xy[0])
            plot_ys.append(end_xy[1])

    return plot_xs, plot_ys


def _set_line_data(line, xs, ys):
    line.set_data(list(xs), list(ys))


def _format_speed_multiplier(speed_multiplier):
    value = float(speed_multiplier)
    if abs(value - round(value)) <= 1e-9:
        return "%dX" % int(round(value))
    return ("%.2f" % value).rstrip("0").rstrip(".") + "X"


def _active_vlm_event(vlm_events, t):
    active = None
    for event in vlm_events:
        if event["t"] > t:
            break
        active = event
    return active


def _format_vlm_event(event):
    if event is None:
        return "VLM: --"
    return "VLM #%s @ %.1fs\nx: %s\ny: %s" % (
        event["call"],
        event["t"],
        event["x_cmd"] or "--",
        event["y_cmd"] or "--",
    )


def _optimize_gif(gif_path, colors=128):
    colors = max(2, min(256, int(colors)))
    try:
        from PIL import Image, ImageSequence
    except Exception as exc:
        print("gif_optimize: skipped (PIL unavailable: %s)" % exc)
        return

    if not os.path.exists(gif_path):
        return

    original_size = os.path.getsize(gif_path)
    tmp_path = gif_path + ".opt.gif"
    try:
        im = Image.open(gif_path)
        loop = im.info.get("loop", 0)
        default_duration = im.info.get("duration", 100)
        durations = []
        frames = []
        for frame in ImageSequence.Iterator(im):
            durations.append(frame.info.get("duration", default_duration))
            rgb = frame.convert("RGB")
            frames.append(rgb.quantize(colors=colors, method=Image.MEDIANCUT))
        if not frames:
            return
        frames[0].save(
            tmp_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            optimize=True,
            disposal=2,
        )
        optimized_size = os.path.getsize(tmp_path)
        if optimized_size < original_size:
            os.replace(tmp_path, gif_path)
            print(
                "gif_optimize: %.1fMB -> %.1fMB (colors=%d)"
                % (original_size / 1048576.0, optimized_size / 1048576.0, colors)
            )
        else:
            os.remove(tmp_path)
            print("gif_optimize: kept original (optimized copy was not smaller)")
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print("gif_optimize: skipped (%s)" % exc)


def animate_tracks(
    tracks,
    constraints,
    goal_points,
    out_gif,
    title,
    own_boat="myboat",
    vlm_events=None,
    fps=10,
    speed_multiplier=4.0,
    trail_seconds=0.0,
    dpi=80,
    optimize_gif=False,
    gif_colors=128,
):
    if not tracks:
        raise RuntimeError("no_tracks_found")
    vlm_events = vlm_events or []

    source_times = _animation_times(tracks, own_boat)
    if not source_times:
        raise RuntimeError("no_track_samples_found")
    writer_fps = max(1, int(fps))
    frame_times = _playback_frame_times(source_times, writer_fps, speed_multiplier)
    source_duration_s = max(0.0, float(source_times[-1]) - float(source_times[0]))
    gif_duration_s = float(len(frame_times)) / float(writer_fps)

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

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
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
        ax.plot(xs, ys, linestyle="--", linewidth=1.2, color=style["color"], alpha=0.55)
        ax.scatter(xs, ys, marker=style["marker"], s=22, color=style["color"], alpha=0.75, label=style["label"])

    for label, x, y in goal_points:
        ax.scatter([x], [y], marker="*", s=180, color="#9467bd", edgecolors="black", linewidths=0.7, zorder=8, label=label)
        ax.annotate(
            label,
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=12,
            color="#5b2a86",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
        )

    full_lines = {}
    trail_lines = {}
    markers = {}
    for boat in [own_boat] + _target_boats(tracks, own_boat):
        data = tracks.get(boat)
        if not data or not data.get("x"):
            continue
        color = DEFAULT_COLORS.get(boat, None)
        line_width = 2.4 if boat == own_boat else 2.0
        alpha = 0.28 if boat == own_boat else 0.22
        (full_line,) = ax.plot([], [], linewidth=line_width, color=color, alpha=alpha, label=boat)
        (trail_line,) = ax.plot([], [], linewidth=line_width, color=color, alpha=0.95)
        marker_style = "o" if boat == own_boat else "^"
        (marker,) = ax.plot([], [], marker=marker_style, markersize=8, color=color, linestyle="None")
        full_lines[boat] = full_line
        trail_lines[boat] = trail_line
        markers[boat] = marker

    time_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d1d5db", alpha=0.86),
    )
    speed_text = ax.text(
        0.98,
        0.03,
        _format_speed_multiplier(speed_multiplier),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=24,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
    )
    vlm_text = None
    if vlm_events:
        vlm_text = ax.text(
            0.98,
            0.97,
            "VLM: --",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d1d5db", alpha=0.86),
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    artists = list(full_lines.values()) + list(trail_lines.values()) + list(markers.values()) + [time_text, speed_text]
    if vlm_text is not None:
        artists.append(vlm_text)

    def init():
        for line in full_lines.values():
            _set_line_data(line, [], [])
        for line in trail_lines.values():
            _set_line_data(line, [], [])
        for marker in markers.values():
            _set_line_data(marker, [], [])
        time_text.set_text("")
        speed_text.set_text(_format_speed_multiplier(speed_multiplier))
        if vlm_text is not None:
            vlm_text.set_text("VLM: --")
        return artists

    def update(frame_idx):
        t = frame_times[frame_idx]
        for boat, data in tracks.items():
            if boat not in full_lines:
                continue
            ts = data.get("t", [])
            xs = data.get("x", [])
            ys = data.get("y", [])
            idx = _sample_index_at_time(ts, t)
            if idx < 0:
                _set_line_data(full_lines[boat], [], [])
                _set_line_data(trail_lines[boat], [], [])
                _set_line_data(markers[boat], [], [])
                continue

            trail_start_t = None
            if trail_seconds and trail_seconds > 0.0:
                trail_start_t = max(ts[0], t - trail_seconds)
            full_xs, full_ys = _track_path_until(data, t)
            trail_xs, trail_ys = _track_path_until(data, t, start_t=trail_start_t)
            current_xy = _track_xy_at_time(data, t)
            _set_line_data(full_lines[boat], full_xs, full_ys)
            _set_line_data(trail_lines[boat], trail_xs, trail_ys)
            if current_xy is None:
                _set_line_data(markers[boat], [], [])
            else:
                _set_line_data(markers[boat], [current_xy[0]], [current_xy[1]])
        time_text.set_text("t = %.1fs" % float(t))
        if vlm_text is not None:
            vlm_text.set_text(_format_vlm_event(_active_vlm_event(vlm_events, t)))
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        init_func=init,
        interval=max(1, int(round(1000.0 / float(writer_fps)))),
        blit=True,
    )

    out_dir = os.path.dirname(out_gif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ani.save(out_gif, writer=animation.PillowWriter(fps=writer_fps))
    plt.close(fig)
    if optimize_gif:
        _optimize_gif(out_gif, colors=gif_colors)
    print(
        "playback: source=%.3fs gif=%.3fs speed=%s fps=%d frames=%d"
        % (
            source_duration_s,
            gif_duration_s,
            _format_speed_multiplier(speed_multiplier),
            writer_fps,
            len(frame_times),
        )
    )


def process_csv(
    csv_path,
    out_gif=None,
    out_dir=None,
    title=None,
    goal_x=None,
    goal_y=None,
    own_boat="myboat",
    vlm_events=None,
    fps=10,
    speed_multiplier=4.0,
    trail_seconds=0.0,
    dpi=80,
    optimize_gif=False,
    gif_colors=128,
):
    tracks, constraints = load_tracks(csv_path)
    if not tracks:
        raise RuntimeError("no_tracks_found")

    if out_gif:
        gif_path = out_gif
    else:
        plot_dir = out_dir or os.path.join(os.path.dirname(csv_path), "plot")
        gif_path = os.path.join(plot_dir, _plot_base_name(csv_path) + "_tracks.gif")

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    goal_points = load_goal_points(csv_path, goal_x=goal_x, goal_y=goal_y)
    if vlm_events is None:
        vlm_events = load_vlm_instruction_events(csv_path)
    animate_tracks(
        tracks,
        constraints,
        goal_points,
        gif_path,
        title or (csv_stem + " Trajectory Animation"),
        own_boat=own_boat,
        vlm_events=vlm_events,
        fps=fps,
        speed_multiplier=speed_multiplier,
        trail_seconds=trail_seconds,
        dpi=dpi,
        optimize_gif=optimize_gif,
        gif_colors=gif_colors,
    )
    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Animate myboat/target trajectories from a CSV and save a GIF")
    parser.add_argument(
        "path",
        nargs="?",
        help="input trajectory CSV path",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="input trajectory CSV path",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output GIF path",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="animation title",
    )
    parser.add_argument(
        "--goal-x",
        type=float,
        default=None,
        help="goal/target point x coordinate",
    )
    parser.add_argument(
        "--goal-y",
        type=float,
        default=None,
        help="goal/target point y coordinate",
    )
    parser.add_argument(
        "--own-boat",
        default="myboat",
        help="own boat name in CSV",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="output GIF frames per second",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=4.0,
        help="GIF playback speed multiplier shown in the lower-right corner",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=0.0,
        help="highlight only the latest N seconds as a bright trail; <=0 keeps all bright",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=80,
        help="output figure DPI",
    )
    parser.add_argument(
        "--gif-colors",
        type=int,
        default=128,
        help="color count used by GIF optimization; 256 keeps more quality, 64 makes smaller files",
    )
    parser.add_argument(
        "--optimize-gif",
        action="store_true",
        help="try Pillow GIF post-processing optimization after saving",
    )
    args = parser.parse_args()

    if (args.goal_x is None) != (args.goal_y is None):
        raise RuntimeError("--goal-x and --goal-y must be provided together")

    csv_path = args.csv or args.path or os.path.join(os.getcwd(), "tmp", "boat_tracks.csv")
    gif_path = process_csv(
        csv_path,
        out_gif=args.out,
        title=args.title,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        own_boat=args.own_boat,
        fps=max(1, args.fps),
        speed_multiplier=max(0.1, args.speed_multiplier),
        trail_seconds=max(0.0, args.trail_seconds),
        dpi=max(40, args.dpi),
        optimize_gif=args.optimize_gif,
        gif_colors=max(2, min(256, args.gif_colors)),
    )
    print("saved:", gif_path)


if __name__ == "__main__":
    main()
