#!/usr/bin/env python3
"""Build a HuggingFace-style SFT dataset from successful Monte Carlo VLA logs."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


INPUT_START_RE = re.compile(r"^===== LLM INPUT call_id=(\d+) =====$")
INPUT_END_RE = re.compile(r"^===== END LLM INPUT call_id=(\d+) =====$")
OUTPUT_START_RE = re.compile(r"^===== LLM OUTPUT call_id=(\d+) =====$")
OUTPUT_END_RE = re.compile(r"^===== END LLM OUTPUT call_id=(\d+) =====$")


@dataclass(frozen=True)
class LlmPair:
    call_id: int
    prompt: str
    response: str
    input_meta: dict[str, Any]
    output_meta: dict[str, Any]


@dataclass(frozen=True)
class SnapshotAssets:
    radar_image_path: Path | None
    bev_image_path: Path | None
    visualization_image_path: Path | None
    video_frame_paths: list[Path]
    radar_diag: dict[str, Any]
    bev_diag: dict[str, Any]
    input_bev_diag: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select SUCCESS trials under tmp/monte_carlo and convert their "
            "LLM input/output logs into a HuggingFace JSONL SFT dataset."
        )
    )
    parser.add_argument(
        "sources",
        nargs="*",
        default=[Path("tmp/monte_carlo")],
        type=Path,
        help=(
            "Monte Carlo root or one/more run directories. A run directory must "
            "contain results.csv. Default: tmp/monte_carlo"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tmp/hf_success_dataset"),
        help="Output dataset directory. Default: tmp/hf_success_dataset",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help="Fraction of samples written to validation.jsonl. Default: 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for train/validation split. Default: 42",
    )
    parser.add_argument(
        "--format",
        choices=("messages", "instruction"),
        default="messages",
        help=(
            "Dataset row format. messages is recommended for chat SFT. "
            "Default: messages"
        ),
    )
    parser.add_argument(
        "--clean-assistant-json",
        action="store_true",
        help=(
            "Keep only the first JSON object found in assistant raw_output. "
            "Useful when logs contain extra text."
        ),
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help=(
            "Add VL image entries to user messages when matching snapshots are "
            "available in vla_trial_N_events.txt."
        ),
    )
    parser.add_argument(
        "--vl-image-source",
        choices=("bev_video", "bev", "radar"),
        default="bev_video",
        help=(
            "VL image source. bev_video embeds the four most recent clean LiDAR "
            "BEV frames as a video; bev embeds one BEV PNG; radar is kept for "
            "compatibility. Default: bev_video"
        ),
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy matched images into output_dir/images and use relative paths.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Skip samples that do not have a matched image.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a SUCCESS trial has no parseable LLM input/output pairs.",
    )
    return parser.parse_args()


def discover_run_dirs(sources: Iterable[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for source in sources:
        if (source / "results.csv").is_file():
            run_dirs.append(source)
            continue
        if not source.is_dir():
            continue
        run_dirs.extend(
            child
            for child in sorted(source.iterdir())
            if child.is_dir() and (child / "results.csv").is_file()
        )
    return sorted(dict.fromkeys(run_dirs))


def read_success_trials(run_dir: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with (run_dir / "results.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("result", "").strip().upper() != "SUCCESS":
                continue
            trial_raw = row.get("trial", "").strip()
            if not trial_raw:
                continue
            rows[int(trial_raw)] = row
    return rows


def parse_meta_lines(lines: list[str]) -> tuple[dict[str, Any], str]:
    meta: dict[str, Any] = {}
    prompt_start = -1
    for index, line in enumerate(lines):
        if line == "prompt:":
            prompt_start = index + 1
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            meta[key] = parse_scalar(value)
    prompt = "\n".join(lines[prompt_start:]).strip() if prompt_start >= 0 else ""
    return meta, prompt


def parse_output_lines(lines: list[str]) -> tuple[dict[str, Any], str]:
    meta: dict[str, Any] = {}
    raw_start = -1
    for index, line in enumerate(lines):
        if line == "raw_output:":
            raw_start = index + 1
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            meta[key] = parse_scalar(value)
    response = "\n".join(lines[raw_start:]).strip() if raw_start >= 0 else ""
    return meta, response


def parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_txt_log(path: Path) -> list[LlmPair]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    inputs: dict[int, tuple[dict[str, Any], str]] = {}
    outputs: dict[int, tuple[dict[str, Any], str]] = {}
    index = 0
    while index < len(lines):
        input_match = INPUT_START_RE.match(lines[index])
        output_match = OUTPUT_START_RE.match(lines[index])
        if input_match:
            call_id = int(input_match.group(1))
            block, index = collect_block(lines, index + 1, INPUT_END_RE)
            meta, prompt = parse_meta_lines(block)
            if prompt:
                inputs[call_id] = (meta, prompt)
            continue
        if output_match:
            call_id = int(output_match.group(1))
            block, index = collect_block(lines, index + 1, OUTPUT_END_RE)
            meta, response = parse_output_lines(block)
            if response:
                outputs[call_id] = (meta, response)
            continue
        index += 1

    return build_pairs(inputs, outputs)


def collect_block(
    lines: list[str], start_index: int, end_re: re.Pattern[str]
) -> tuple[list[str], int]:
    block: list[str] = []
    index = start_index
    while index < len(lines):
        if end_re.match(lines[index]):
            return block, index + 1
        block.append(lines[index])
        index += 1
    return block, index


def parse_jsonl_log(path: Path) -> list[LlmPair]:
    inputs: dict[int, tuple[dict[str, Any], str]] = {}
    outputs: dict[int, tuple[dict[str, Any], str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL event: {exc}") from exc

            call_id = event.get("call_id")
            if call_id is None:
                continue
            call_id = int(call_id)
            event_type = event.get("event")
            if event_type == "llm_input":
                prompt = str(event.get("prompt", "")).strip()
                meta = {k: v for k, v in event.items() if k != "prompt"}
                if prompt:
                    inputs[call_id] = (meta, prompt)
            elif event_type == "llm_output":
                response = str(event.get("raw_output", "")).strip()
                meta = {k: v for k, v in event.items() if k != "raw_output"}
                if response:
                    outputs[call_id] = (meta, response)
    return build_pairs(inputs, outputs)


def build_pairs(
    inputs: dict[int, tuple[dict[str, Any], str]],
    outputs: dict[int, tuple[dict[str, Any], str]],
) -> list[LlmPair]:
    pairs: list[LlmPair] = []
    for call_id in sorted(set(inputs) & set(outputs)):
        input_meta, prompt = inputs[call_id]
        output_meta, response = outputs[call_id]
        pairs.append(
            LlmPair(
                call_id=call_id,
                prompt=prompt,
                response=response,
                input_meta=input_meta,
                output_meta=output_meta,
            )
        )
    return pairs


def first_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if end:
            break
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def find_trial_log(run_dir: Path, trial: int) -> Path | None:
    for suffix in ("txt", "jsonl"):
        candidate = run_dir / f"vla_trial_{trial}_io.{suffix}"
        if candidate.is_file():
            return candidate
    return None


def parse_event_blocks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    events: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("===== VLA EVENT "):
            index += 1
            continue
        index += 1
        block: list[str] = []
        while index < len(lines) and not lines[index].startswith("===== END VLA EVENT "):
            block.append(lines[index])
            index += 1
        try:
            event = json.loads("\n".join(block))
        except json.JSONDecodeError:
            event = {}
        if event:
            events.append(event)
        index += 1
    return events


def resolve_logged_image_path(run_dir: Path, logged_path: str) -> Path | None:
    if not logged_path:
        return None
    path = Path(logged_path)
    if path.is_file():
        return path
    fallback = run_dir / "vla_snapshots" / path.name
    if fallback.is_file():
        return fallback
    return None


def load_trial_image_map(run_dir: Path, trial: int) -> dict[int, Path]:
    events_path = run_dir / f"vla_trial_{trial}_events.txt"
    image_map: dict[int, Path] = {}
    for event in parse_event_blocks(events_path):
        if event.get("event") != "snapshot_saved":
            continue
        call_id = (
            event.get("call_id")
            or event.get("planned_llm_call_index")
            or event.get("snapshot_seq")
        )
        image_path = resolve_logged_image_path(run_dir, str(event.get("image_path", "")))
        if call_id is not None and image_path is not None:
            image_map[int(call_id)] = image_path
    return image_map


def load_trial_snapshot_assets(run_dir: Path, trial: int) -> dict[int, SnapshotAssets]:
    events_path = run_dir / f"vla_trial_{trial}_events.txt"
    asset_map: dict[int, SnapshotAssets] = {}
    clean_bev_history: list[Path] = []
    for event in parse_event_blocks(events_path):
        if event.get("event") != "snapshot_saved":
            continue
        call_id = (
            event.get("call_id")
            or event.get("planned_llm_call_index")
            or event.get("snapshot_seq")
        )
        if call_id is None:
            continue
        radar_image_path = resolve_logged_image_path(
            run_dir, str(event.get("radar_image_path", ""))
        )
        visualization_image_path = resolve_logged_image_path(
            run_dir, str(event.get("image_path", ""))
        )
        bev_image_path = resolve_logged_image_path(
            run_dir, str(event.get("input_image_path", ""))
        )
        if bev_image_path is None:
            bev_image_path = visualization_image_path
        if bev_image_path is not None:
            clean_bev_history.append(bev_image_path)
        event_video_paths = []
        raw_event_video = event.get("bev_video_frame_paths", [])
        if isinstance(raw_event_video, list):
            event_video_paths = [
                path
                for path in (
                    resolve_logged_image_path(run_dir, str(raw_path))
                    for raw_path in raw_event_video
                )
                if path is not None
            ]
        asset_map[int(call_id)] = SnapshotAssets(
            radar_image_path=radar_image_path,
            bev_image_path=bev_image_path,
            visualization_image_path=visualization_image_path,
            video_frame_paths=(event_video_paths[-4:] if event_video_paths else clean_bev_history[-4:]),
            radar_diag=event.get("radar_image_diag", {})
            if isinstance(event.get("radar_image_diag"), dict)
            else {},
            bev_diag=event.get("bev_diag", {})
            if isinstance(event.get("bev_diag"), dict)
            else {},
            input_bev_diag=event.get("input_bev_diag", {})
            if isinstance(event.get("input_bev_diag"), dict)
            else {},
        )
    return asset_map


def pair_image_path(run_dir: Path, pair: LlmPair, image_map: dict[int, Path]) -> Path | None:
    if pair.call_id in image_map:
        return image_map[pair.call_id]
    for key in ("event_image_path", "image_path"):
        image_path = resolve_logged_image_path(run_dir, str(pair.input_meta.get(key, "")))
        if image_path is not None:
            return image_path
    image_diag = pair.input_meta.get("image_diag")
    if isinstance(image_diag, dict):
        image_path = resolve_logged_image_path(
            run_dir,
            str(image_diag.get("event_image_path") or image_diag.get("path") or ""),
        )
        if image_path is not None:
            return image_path
    return None


def pair_snapshot_assets(
    run_dir: Path,
    pair: LlmPair,
    asset_map: dict[int, SnapshotAssets],
    image_map: dict[int, Path],
) -> SnapshotAssets:
    if pair.call_id in asset_map:
        assets = asset_map[pair.call_id]
        raw_pair_video = pair.input_meta.get("bev_video_frame_paths", [])
        if isinstance(raw_pair_video, list):
            pair_video_paths = [
                path
                for path in (
                    resolve_logged_image_path(run_dir, str(raw_path))
                    for raw_path in raw_pair_video
                )
                if path is not None
            ]
            if pair_video_paths:
                return SnapshotAssets(
                    radar_image_path=assets.radar_image_path,
                    bev_image_path=assets.bev_image_path,
                    visualization_image_path=assets.visualization_image_path,
                    video_frame_paths=pair_video_paths[-4:],
                    radar_diag=assets.radar_diag,
                    bev_diag=assets.bev_diag,
                    input_bev_diag=assets.input_bev_diag,
                )
        return assets
    fallback_visualization_path = pair_image_path(run_dir, pair, image_map)
    fallback_bev_path = resolve_logged_image_path(
        run_dir, str(pair.input_meta.get("event_input_image_path", ""))
    )
    if fallback_bev_path is None:
        fallback_bev_path = resolve_logged_image_path(
            run_dir, str(pair.input_meta.get("input_image_path", ""))
        )
    if fallback_bev_path is None:
        fallback_bev_path = fallback_visualization_path
    raw_pair_video = pair.input_meta.get("bev_video_frame_paths", [])
    fallback_video_paths = []
    if isinstance(raw_pair_video, list):
        fallback_video_paths = [
            path
            for path in (
                resolve_logged_image_path(run_dir, str(raw_path))
                for raw_path in raw_pair_video
            )
            if path is not None
        ]
    radar_image_path = resolve_logged_image_path(
        run_dir, str(pair.input_meta.get("event_radar_image_path", ""))
    )
    if radar_image_path is None:
        radar_image_path = resolve_logged_image_path(
            run_dir, str(pair.input_meta.get("radar_image_path", ""))
        )
    return SnapshotAssets(
        radar_image_path=radar_image_path,
        bev_image_path=fallback_bev_path,
        visualization_image_path=fallback_visualization_path,
        video_frame_paths=(
            fallback_video_paths[-4:]
            if fallback_video_paths
            else ([fallback_bev_path] if fallback_bev_path is not None else [])
        ),
        radar_diag={},
        bev_diag={},
        input_bev_diag={},
    )


def file_data_url(path: Path) -> str:
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"empty image file: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def dataset_image_path(
    output_dir: Path,
    run_dir: Path,
    trial: int,
    call_id: int,
    image_path: Path,
    copy_images: bool,
    label: str = "",
) -> str:
    if not copy_images:
        return str(image_path)

    suffix = image_path.suffix or ".png"
    label_part = f"_{label}" if label else ""
    destination = (
        output_dir
        / "images"
        / run_dir.name
        / f"vla_trial_{trial}_call_{call_id:03d}{label_part}{suffix}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(image_path, destination)
    return str(destination.relative_to(output_dir))


def prompt_for_vl_image_source(prompt: str, vl_image_source: str) -> str:
    if vl_image_source == "bev_video":
        text = prompt.strip()
        replacement = (
            "Scene input: four recent clean LiDAR BEV frames as a short video, ordered oldest to newest.\n"
            "Analyze the visual trend across frames: target motion direction, closing/separating trend, and collision risk.\n"
            "BEV convention: own vessel is the red dot at image center; forward is up; aft is down; port is left; starboard is right. Green points are LiDAR returns. Frames are clean pointcloud projections without grid, boxes, or red-arc overlays.\n"
        )
        text = re.sub(
            r"Scene input:.*?(?=Fields:|COLREGS:)",
            replacement,
            text,
            count=1,
            flags=re.S,
        )
        text = re.sub(
            r"Fields: temporal_risk_tokens summarize each track over time;.*?\n",
            "",
            text,
            count=1,
            flags=re.S,
        )
        text = text.replace(
            "COLREGS: follow standard maritime encounter rules using the track input.",
            "COLREGS: follow standard maritime encounter rules using the BEV video trend.",
        )
        text = text.replace(
            "Course actions: KEEP_COURSE, TURN_STARBOARD, TURN_PORT.\n"
            "Speed actions: SLOW_DOWN, SPEED_UP, EMERGENCY_STOP.\n"
            "Return only JSON: {\"confidence\":number,\"reasoning\":string,\"course_action\":course_action,\"speed_action\":speed_action}.\n",
            "JSON enum rules:\n"
            "course_action MUST be exactly one of [\"KEEP_COURSE\",\"TURN_STARBOARD\",\"TURN_PORT\"]. Never put SPEED_UP, SLOW_DOWN, or EMERGENCY_STOP in course_action.\n"
            "speed_action MUST be exactly one of [\"SLOW_DOWN\",\"SPEED_UP\",\"EMERGENCY_STOP\"]. Never put KEEP_COURSE, TURN_STARBOARD, or TURN_PORT in speed_action.\n"
            "Return only JSON: {\"confidence\":number,\"reasoning\":string,\"course_action\":\"KEEP_COURSE|TURN_STARBOARD|TURN_PORT\",\"speed_action\":\"SLOW_DOWN|SPEED_UP|EMERGENCY_STOP\"}.\n",
        )
        text = text.replace(
            "briefly describe the BEV scene first",
            "briefly describe the BEV trend first",
        )
        return text
    if vl_image_source != "radar":
        return prompt.strip()
    text = prompt.strip()
    text = text.replace(
        "Scene input: LiDAR BEV image",
        "Scene input: losslessly packed radar PointCloud2 byte-image",
    )
    text = text.replace(
        "BEV convention: own vessel is at bottom center; forward is up; port is left; "
        "starboard is right; grid spacing is 5 meters with stronger 10-meter lines.\n",
        "Radar image convention: PNG pixels losslessly store the raw PointCloud2.data "
        "byte stream; use it together with the track tokens.\n",
    )
    text = text.replace(
        "BEV convention: own vessel is the red dot at image center; forward is up; aft is down; "
        "port is left; starboard is right; grid spacing is 5 meters with stronger 10-meter lines.\n",
        "Radar image convention: PNG pixels losslessly store the raw PointCloud2.data "
        "byte stream; use it together with the track tokens.\n",
    )
    text = text.replace(
        "briefly describe the BEV scene first",
        "briefly describe the radar/track scene first",
    )
    return text


def row_for_pair(
    run_dir: Path,
    trial: int,
    result_row: dict[str, str],
    log_path: Path,
    pair: LlmPair,
    row_format: str,
    clean_assistant_json: bool,
    output_dir: Path,
    radar_image_path: Path | None,
    bev_image_path: Path | None,
    visualization_image_path: Path | None,
    video_frame_paths: list[Path],
    radar_diag: dict[str, Any],
    bev_diag: dict[str, Any],
    input_bev_diag: dict[str, Any],
    copy_images: bool,
    vl_image_source: str,
) -> dict[str, Any]:
    response = (
        first_json_object(pair.response) if clean_assistant_json else pair.response.strip()
    )
    response_json = parse_json_object(response)
    decision = {
        "course_action": response_json.get("course_action", ""),
        "speed_action": response_json.get("speed_action", ""),
        "confidence": response_json.get("confidence", ""),
        "reasoning": response_json.get("reasoning", ""),
    }
    decision = {key: value for key, value in decision.items() if value != ""}
    row_bev_image_path = (
        dataset_image_path(
            output_dir=output_dir,
            run_dir=run_dir,
            trial=trial,
            call_id=pair.call_id,
            image_path=bev_image_path,
            copy_images=copy_images,
            label="input" if visualization_image_path is not None and visualization_image_path != bev_image_path else "",
        )
        if bev_image_path is not None
        else ""
    )
    row_visualization_image_path = (
        dataset_image_path(
            output_dir=output_dir,
            run_dir=run_dir,
            trial=trial,
            call_id=pair.call_id,
            image_path=visualization_image_path,
            copy_images=copy_images,
            label="debug" if bev_image_path is not None and visualization_image_path != bev_image_path else "",
        )
        if visualization_image_path is not None
        else row_bev_image_path
    )
    row_input_image = ""
    row_video_frames: list[str] = []
    row_video_frame_paths: list[str] = []
    row_radar_image_path = str(radar_image_path) if radar_image_path is not None else ""
    if vl_image_source == "bev_video" and len(video_frame_paths) >= 4:
        row_video_frames = [file_data_url(path) for path in video_frame_paths[-4:]]
        row_video_frame_paths = [str(path) for path in video_frame_paths[-4:]]
    elif vl_image_source == "radar" and radar_image_path is not None:
        row_input_image = file_data_url(radar_image_path)
    elif vl_image_source == "bev" and bev_image_path is not None:
        row_input_image = file_data_url(bev_image_path)
    prompt = prompt_for_vl_image_source(pair.prompt, vl_image_source)
    metadata = sanitize_json_value({
        "run_id": run_dir.name,
        "trial": trial,
        "call_id": pair.call_id,
        "scenario": result_row.get("scenario", ""),
        "mode": result_row.get("mode", ""),
        "seed": result_row.get("seed", ""),
        "result": result_row.get("result", ""),
        "reason": result_row.get("reason", ""),
        "duration_s": result_row.get("duration_s", ""),
        "source_log": str(log_path),
        "vl_image_source": vl_image_source,
        "input_image_kind": (
            "lidar_pointcloud_bev_video_base64_frames"
            if vl_image_source == "bev_video"
            else (
                "radar_pointcloud_raw_png_base64"
                if vl_image_source == "radar"
                else "lidar_pointcloud_bev_png_base64"
            )
        ),
        "video_frame_paths": row_video_frame_paths,
        "radar_image_path": row_radar_image_path,
        "bev_image_path": row_bev_image_path,
        "visualization_image_path": row_visualization_image_path,
        "image_path": row_visualization_image_path,
        "radar_image_diag": radar_diag,
        "bev_diag": bev_diag,
        "input_bev_diag": input_bev_diag,
        "decision": decision,
        "input_meta": pair.input_meta,
        "output_meta": pair.output_meta,
    })
    if row_format == "instruction":
        return {
            "instruction": prompt,
            "input": "",
            "output": response,
            "decision": decision,
            "metadata": metadata,
        }
    user_content: str | list[dict[str, str]]
    if row_video_frames:
        user_content = [
            {"type": "video", "video": row_video_frames},
            {"type": "text", "text": prompt},
        ]
    elif row_input_image:
        user_content = [
            {"type": "image", "image": row_input_image},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    row = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ],
        "decision": decision,
        "metadata": metadata,
    }
    if row_video_frames:
        row["videos"] = [row_video_frames]
    if row_input_image:
        row["images"] = [row_input_image]
    return row


def has_required_actions(pair: LlmPair, clean_assistant_json: bool) -> bool:
    response = (
        first_json_object(pair.response) if clean_assistant_json else pair.response.strip()
    )
    response_json = parse_json_object(response)
    return bool(response_json.get("course_action")) and bool(
        response_json.get("speed_action")
    )


def parse_log(path: Path) -> list[LlmPair]:
    if path.suffix == ".jsonl":
        return parse_jsonl_log(path)
    return parse_txt_log(path)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def write_readme(output_dir: Path, stats: dict[str, Any], row_format: str) -> None:
    load_line = (
        'load_dataset("json", data_files={"train": "train.jsonl", '
        '"validation": "validation.jsonl"})'
        if stats["validation_samples"]
        else 'load_dataset("json", data_files="train.jsonl")'
    )
    readme = f"""# Monte Carlo SUCCESS VLA SFT Dataset

This dataset was generated from SUCCESS trials in Monte Carlo VLA logs.

- row_format: {row_format}
- vl_image_source: {stats["vl_image_source"]}
- run_dirs: {stats["run_dirs"]}
- success_trials: {stats["success_trials"]}
- train_samples: {stats["train_samples"]}
- validation_samples: {stats["validation_samples"]}
- skipped_success_trials: {stats["skipped_success_trials"]}

Load with:

```python
from datasets import load_dataset

dataset = {load_line}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.validation_ratio < 0 or args.validation_ratio >= 1:
        print("--validation-ratio must be in [0, 1).", file=sys.stderr)
        return 2

    run_dirs = discover_run_dirs(args.sources)
    if not run_dirs:
        print("No run directories with results.csv found.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    success_trials = 0
    skipped_success_trials: list[str] = []
    samples_without_images = 0
    samples_without_required_actions = 0

    for run_dir in run_dirs:
        success_rows = read_success_trials(run_dir)
        success_trials += len(success_rows)
        for trial, result_row in sorted(success_rows.items()):
            log_path = find_trial_log(run_dir, trial)
            if log_path is None:
                skipped_success_trials.append(f"{run_dir.name}/trial_{trial}: missing io log")
                continue
            pairs = parse_log(log_path)
            if not pairs:
                skipped_success_trials.append(f"{run_dir.name}/trial_{trial}: no llm pairs")
                continue
            image_map = load_trial_image_map(run_dir, trial) if args.include_images else {}
            asset_map = (
                load_trial_snapshot_assets(run_dir, trial) if args.include_images else {}
            )
            for pair in pairs:
                assets = (
                    pair_snapshot_assets(run_dir, pair, asset_map, image_map)
                    if args.include_images
                    else SnapshotAssets(None, None, None, [], {}, {}, {})
                )
                if args.vl_image_source == "bev_video":
                    input_image_path = (
                        assets.video_frame_paths[-1]
                        if len(assets.video_frame_paths) >= 4
                        else None
                    )
                elif args.vl_image_source == "radar":
                    input_image_path = assets.radar_image_path
                else:
                    input_image_path = assets.bev_image_path
                if args.include_images and input_image_path is None:
                    samples_without_images += 1
                    if args.require_images:
                        continue
                if not has_required_actions(pair, args.clean_assistant_json):
                    samples_without_required_actions += 1
                    continue
                rows.append(
                    row_for_pair(
                        run_dir=run_dir,
                        trial=trial,
                        result_row=result_row,
                        log_path=log_path,
                        pair=pair,
                        row_format=args.format,
                        clean_assistant_json=args.clean_assistant_json,
                        output_dir=args.output_dir,
                        radar_image_path=assets.radar_image_path,
                        bev_image_path=assets.bev_image_path,
                        visualization_image_path=assets.visualization_image_path,
                        video_frame_paths=assets.video_frame_paths,
                        radar_diag=assets.radar_diag,
                        bev_diag=assets.bev_diag,
                        input_bev_diag=assets.input_bev_diag,
                        copy_images=args.copy_images,
                        vl_image_source=args.vl_image_source,
                    )
                )

    if args.strict and skipped_success_trials:
        for skipped in skipped_success_trials:
            print(f"Skipped: {skipped}", file=sys.stderr)
        return 1
    if not rows:
        print("No dataset samples generated.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    validation_count = int(round(len(rows) * args.validation_ratio))
    validation_rows = rows[:validation_count]
    train_rows = rows[validation_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_jsonl(args.output_dir / "train.jsonl", train_rows)
    validation_count = 0
    validation_path = args.output_dir / "validation.jsonl"
    if validation_rows:
        validation_count = write_jsonl(validation_path, validation_rows)
    elif validation_path.exists():
        validation_path.unlink()

    stats = {
        "run_dirs": [str(path) for path in run_dirs],
        "success_trials": success_trials,
        "train_samples": train_count,
        "validation_samples": validation_count,
        "skipped_success_trials": skipped_success_trials,
        "include_images": args.include_images,
        "vl_image_source": args.vl_image_source,
        "copy_images": args.copy_images,
        "samples_without_images": samples_without_images,
        "samples_without_required_actions": samples_without_required_actions,
    }
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(args.output_dir, stats, args.format)

    print(f"Wrote {train_count} train samples to {args.output_dir / 'train.jsonl'}")
    if validation_count:
        print(
            f"Wrote {validation_count} validation samples to "
            f"{args.output_dir / 'validation.jsonl'}"
        )
    print(f"SUCCESS trials: {success_trials}; skipped: {len(skipped_success_trials)}")
    if args.include_images:
        print(f"Samples without matched images: {samples_without_images}")
    print(f"Samples without required actions: {samples_without_required_actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
