#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re
import ast
import io
import subprocess
import sys
import threading
import time
import base64
import mimetypes
import struct
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import deque

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR and THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
DEFAULT_TMP_DIR = os.path.join(os.getcwd(), "tmp")
DEFAULT_DINOV2_PROTOTYPES_DIR = os.path.normpath(
    os.path.join(THIS_DIR, "..", "resource", "dinov2_prototypes")
)

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool, String
from PIL import Image as PilImage, ImageDraw, PngImagePlugin

try:
    import numpy as np
    NP_IMPORT_ERROR = ""
except Exception as e:
    np = None
    NP_IMPORT_ERROR = str(e)

try:
    import cv2
    CV2_IMPORT_ERROR = ""
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = str(e)

try:
    from dinov2_ship_classifier import DinoV2ShipClassifier
    DINO_IMPORT_ERROR = ""
except Exception as e:
    DinoV2ShipClassifier = None
    DINO_IMPORT_ERROR = str(e)

class ColregsLlmDecisionNode:
    """VLA target-type classifier with image-track alignment and JSON-constrained output."""

    @staticmethod
    def _backend_name_zh(backend):
        b = (backend or "").strip().lower()
        if b == "dinov2":
            return "本地 DINOv2"
        if b == "ollama":
            return "本地 Ollama"
        if b == "gemini":
            return "Google Gemini 联网 API"
        if b == "http":
            return "HTTP 聊天接口（OpenAI 兼容）"
        return backend or "未知"

    @staticmethod
    def _call_reason_zh(reason):
        m = {
            "trigger_rising_edge": "风险触发上升沿（本事件仅调一次）",
            "semantic_pretrigger": "语义预触发提前调用",
            "trigger_generation_first_call": "本次trigger事件首次满足条件，补调一次",
            "periodic_recall": "周期再次调用",
            "init": "初始化",
            "skip_no_trigger": "未触发，跳过",
            "skip_trigger_not_armed": "尚未观测到 trigger 低电平基线，跳过",
            "skip_trigger_generation_changed": "触发状态已变化，本次调用作废",
            "skip_no_state": "尚无状态估计，跳过",
            "skip_no_image": "尚无可用图像，等待下一帧",
            "skip_no_pointcloud_bev": "尚无可用点云BEV，等待下一帧点云",
            "skip_pointcloud_bev_warming": "点云BEV仍在预热，等待点云稳定",
            "skip_target_history_warming": "目标历史仍在预热，等待轨迹稳定",
            "skip_no_fov_tracks": "尚无可用于FOV门控的track，跳过",
            "skip_tracks_not_in_fov": "track尚未稳定落入相机FOV，等待更合适图像",
            "skip_once_per_trigger_not_edge": "非上升沿（本事件只调一次），跳过",
            "skip_once_per_trigger_already_called": "本次trigger事件已调用过，跳过",
            "skip_interval_guard": "未到周期间隔，跳过",
            "skip_rising_edge_cooldown": "触发抖动冷却中，跳过",
            "skip_quota_guard": "触发频率过高，配额保护跳过",
            "skip_no_distance": "缺少点云目标距离，跳过",
            "skip_distance_gate": "目标距离过远，跳过",
            "skip_no_front_sensor_hazard": "传感器前方危险区内无目标，跳过",
            "skip_locked_after_success": "首次成功分类后已锁定，后续不再调用",
            "skip_image_buffer_warming": "图像缓冲区预热中，等待足够帧数",
            "skip_no_unknown_fov_track": "FOV内没有Unknown track，跳过",
            "fov_all_tracks_ready": "全部track已落入相机FOV，优先调用",
            "fov_partial_tracks_ready": "等待全部FOV超时，使用部分可见track图像调用",
            "fov_state_fallback_ready": "track未就绪，使用状态估计FOV兜底调用",
            "fov_soft_tracks_ready": "FOV门控软放行，使用当前track和图像调用",
        }
        return m.get(reason, reason)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _llm_result_summary_zh(self):
        """供话题 JSON 的人类可读一行结论。"""
        if not self._last_llm_called:
            return self._call_reason_zh(self._last_llm_call_reason)
        if not self.llm_enable:
            return "LLM 已关闭（仅用规则分类）"
        src = self._last_cls.get("source", "")
        if src == "llm":
            return "成功（VLM 决策有效）"
        if src == "dinov2":
            return "成功（DINOv2 分类有效）"
        if isinstance(src, str) and "fallback" in src:
            return "失败或解析错误，已用规则兜底"
        return "未知"

    @staticmethod
    def _mask_secret(s):
        if not s:
            return "(empty)"
        s = str(s)
        if len(s) <= 8:
            return "****"
        return "%s...%s" % (s[:4], s[-4:])

    def _append_llm_io_record(self, record):
        event = str((record or {}).get("event") or "")
        if event not in ("llm_input", "llm_output", "llm_error"):
            self._append_vla_event_record(record)
            return
        if not getattr(self, "llm_io_log_path", ""):
            return
        text = self._format_llm_io_record(record)
        if not text:
            return
        try:
            out_dir = os.path.dirname(self.llm_io_log_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.llm_io_log_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            rospy.logwarn_throttle(
                10.0,
                "[colregs_llm_decision] failed to write llm_io_log_path=%s: %s",
                self.llm_io_log_path,
                str(e)[:200],
            )

    def _default_vla_event_log_path(self):
        path = str(getattr(self, "vla_event_log_path", "") or "").strip()
        if path:
            return path
        io_path = str(getattr(self, "llm_io_log_path", "") or "").strip()
        if not io_path:
            return ""
        root, ext = os.path.splitext(io_path)
        if root.endswith("_io"):
            root = root[:-3]
        return "%s_events%s" % (root, ext or ".txt")

    def _append_vla_event_record(self, record):
        path = self._default_vla_event_log_path()
        if not path:
            return
        event = str((record or {}).get("event") or "event")
        text = (
            "\n===== VLA EVENT %s =====\n"
            "%s\n"
            "===== END VLA EVENT %s =====\n"
        ) % (
            event,
            json.dumps(record or {}, ensure_ascii=True, default=str, indent=2),
            event,
        )
        try:
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            rospy.logwarn_throttle(
                10.0,
                "[colregs_llm_decision] failed to write vla_event_log_path=%s: %s",
                path,
                str(e)[:200],
            )

    def _format_llm_io_record(self, record):
        event = str((record or {}).get("event") or "")
        if event == "llm_input":
            wall_time_s = float(record.get("wall_time_s", 0.0) or 0.0)
            return (
                "\n===== LLM INPUT call_id=%s =====\n"
                "wall_time: %s\n"
                "wall_time_s: %.6f\n"
                "ros_time_s: %.6f\n"
                "reason: %s\n"
                "model: %s\n"
                "prompt:\n%s\n"
                "===== END LLM INPUT call_id=%s =====\n"
            ) % (
                record.get("call_id", ""),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time_s)),
                wall_time_s,
                float(record.get("ros_time_s", 0.0) or 0.0),
                record.get("reason", ""),
                record.get("model", ""),
                str(record.get("prompt", "") or ""),
                record.get("call_id", ""),
            )
        if event == "llm_output":
            wall_time_s = float(record.get("wall_time_s", 0.0) or 0.0)
            return (
                "\n===== LLM OUTPUT call_id=%s =====\n"
                "wall_time: %s\n"
                "wall_time_s: %.6f\n"
                "ros_time_s: %.6f\n"
                "elapsed_ms: %.3f\n"
                "raw_output:\n%s\n"
                "===== END LLM OUTPUT call_id=%s =====\n"
            ) % (
                record.get("call_id", ""),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time_s)),
                wall_time_s,
                float(record.get("ros_time_s", 0.0) or 0.0),
                float(record.get("elapsed_ms", -1.0) or -1.0),
                str(record.get("raw_output", "") or ""),
                record.get("call_id", ""),
            )
        if event == "llm_error":
            wall_time_s = float(record.get("wall_time_s", 0.0) or 0.0)
            return (
                "\n===== LLM ERROR call_id=%s =====\n"
                "wall_time: %s\n"
                "wall_time_s: %.6f\n"
                "ros_time_s: %.6f\n"
                "elapsed_ms: %.3f\n"
                "exception_type: %s\n"
                "error:\n%s\n"
                "prompt:\n%s\n"
                "===== END LLM ERROR call_id=%s =====\n"
            ) % (
                record.get("call_id", ""),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time_s)),
                wall_time_s,
                float(record.get("ros_time_s", 0.0) or 0.0),
                float(record.get("elapsed_ms", -1.0) or -1.0),
                record.get("exception_type", ""),
                str(record.get("error", "") or ""),
                str(record.get("prompt", "") or ""),
                record.get("call_id", ""),
            )
        return ""

    def _terminal_dump_llm_input(self, call_id, call_reason, prompt, call_context=None):
        call_context = call_context if isinstance(call_context, dict) else {}
        record = {
            "event": "llm_input",
            "wall_time_s": time.time(),
            "ros_time_s": rospy.Time.now().to_sec(),
            "call_id": call_id,
            "reason": self._call_reason_zh(call_reason),
            "reason_key": call_reason,
            "backend": self._backend_name_zh(self.llm_backend),
            "backend_key": self.llm_backend,
            "model": self._last_llm_model_id or "",
            "vla_prompt_mode": self.vla_prompt_mode,
            "trigger": bool(self._trigger),
            "state_seq": self._state_seq,
            "state_snapshot": self._state_snapshot(),
            "image_input_enabled": bool(self._llm_image_input_active()),
            "image_source": self._last_llm_image_source or "",
            "image_data_url_len": int(getattr(self, "_last_llm_image_data_url_len", 0) or 0),
            "video_frame_paths": list(getattr(self, "_last_llm_video_frame_paths", []) or []),
            "video_frame_records": list(getattr(self, "_last_llm_video_frame_records", []) or []),
            "video_frame_count": len(getattr(self, "_last_llm_video_frame_paths", []) or []),
            "event_image_path": call_context.get("event_image_path", self._event_image_path or ""),
            "event_input_image_path": self._event_input_image_path or "",
            "bev_video_frame_paths": [
                str(rec.get("path"))
                for rec in self._recent_bev_video_frame_records()
                if isinstance(rec, dict) and rec.get("path")
            ],
            "fov_gate_diag": call_context.get("fov_gate_diag", getattr(self, "_last_fov_gate_diag", {})),
            "prompt": prompt,
        }
        self._append_llm_io_record(record)
        if not self.terminal_llm_io_enable:
            return
        rospy.loginfo(
            "\n[colregs_llm_decision] ===== LLM INPUT BEGIN =====\ncall_id=%d\nreason=%s\nbackend=%s\ntrigger=%s\nstate_seq=%d\nstate_snapshot=%s\nimage_input_enabled=%s\nimage_source=%s\nimage_data_url_len=%d\nvideo_frame_count=%d\nvideo_frame_paths=%s\nevent_image_path=%s\nprompt:\n%s\n[colregs_llm_decision] ===== LLM INPUT END =====",
            call_id,
            self._call_reason_zh(call_reason),
            self._backend_name_zh(self.llm_backend),
            "开" if self._trigger else "关",
            self._state_seq,
            json.dumps(self._state_snapshot(), ensure_ascii=True),
            "开" if self._llm_image_input_active() else "关",
            self._last_llm_image_source or "(none)",
            int(getattr(self, "_last_llm_image_data_url_len", 0) or 0),
            len(getattr(self, "_last_llm_video_frame_paths", []) or []),
            json.dumps(list(getattr(self, "_last_llm_video_frame_paths", []) or []), ensure_ascii=True),
            call_context.get("event_image_path", self._event_image_path or "(none)") or "(none)",
            prompt,
        )

    def _terminal_dump_llm_output(self, call_id, raw_out):
        record = {
            "event": "llm_output",
            "wall_time_s": time.time(),
            "ros_time_s": rospy.Time.now().to_sec(),
            "call_id": call_id,
            "backend": self._backend_name_zh(self.llm_backend),
            "backend_key": self.llm_backend,
            "model": self._last_llm_model_id or "",
            "elapsed_ms": self._last_llm_elapsed_ms,
            "raw_output": str(raw_out or ""),
        }
        self._append_llm_io_record(record)
        if not self.terminal_llm_io_enable:
            return
        rospy.loginfo(
            "\n[colregs_llm_decision] ===== LLM OUTPUT BEGIN =====\ncall_id=%d\nmodel=%s\nelapsed_ms=%.0f\nraw_output:\n%s\n[colregs_llm_decision] ===== LLM OUTPUT END =====",
            call_id,
            self._last_llm_model_id or "(unknown)",
            self._last_llm_elapsed_ms,
            str(raw_out or ""),
        )

    def _vla_image_diag(self):
        now = rospy.Time.now()
        msg = self._latest_image_msg
        latest = {}
        if msg is not None:
            age_s = None
            try:
                age_s = (now - self._latest_image_stamp).to_sec()
            except Exception:
                age_s = None
            latest = {
                "seen": True,
                "stamp_s": self._latest_image_stamp.to_sec(),
                "age_s": age_s,
                "width": int(getattr(msg, "width", 0) or 0),
                "height": int(getattr(msg, "height", 0) or 0),
                "encoding": str(getattr(msg, "encoding", "") or ""),
                "step": int(getattr(msg, "step", 0) or 0),
                "data_len": len(getattr(msg, "data", b"") or b""),
                "fresh": (age_s is not None and age_s <= max(0.0, self.vla_image_max_age_s)),
            }
        else:
            latest = {"seen": False}
        def _pointcloud_diag(cloud_msg, stamp, recv_time, first_recv_time, frame_count):
            if cloud_msg is None:
                return {"seen": False}
            cloud_age_s = None
            try:
                cloud_age_s = (now - stamp).to_sec()
            except Exception:
                cloud_age_s = None
            return {
                "seen": True,
                "stamp_s": stamp.to_sec(),
                "age_s": cloud_age_s,
                "recv_age_s": (
                    (now - recv_time).to_sec()
                    if getattr(recv_time, "to_sec", lambda: 0.0)() > 0.0
                    else None
                ),
                "since_first_recv_s": (
                    (now - first_recv_time).to_sec()
                    if getattr(first_recv_time, "to_sec", lambda: 0.0)() > 0.0
                    else None
                ),
                "frame_count": int(frame_count or 0),
                "width": int(getattr(cloud_msg, "width", 0) or 0),
                "height": int(getattr(cloud_msg, "height", 0) or 0),
                "point_step": int(getattr(cloud_msg, "point_step", 0) or 0),
                "row_step": int(getattr(cloud_msg, "row_step", 0) or 0),
                "data_len": len(getattr(cloud_msg, "data", b"") or b""),
                "fresh": (cloud_age_s is not None and cloud_age_s <= max(0.0, self.vla_bev_cloud_max_age_s)),
            }
        latest_cloud = _pointcloud_diag(
            getattr(self, "_latest_pointcloud_msg", None),
            getattr(self, "_latest_pointcloud_stamp", rospy.Time(0)),
            getattr(self, "_latest_pointcloud_recv_time", rospy.Time(0)),
            getattr(self, "_first_pointcloud_recv_time", rospy.Time(0)),
            getattr(self, "_pointcloud_frame_count", 0),
        )
        buffer_diag = {
            "duration_s": self.vla_image_buffer_duration_s,
            "max_frames": self.vla_image_buffer_max_frames,
            "frame_count": 0,
        }
        buf = list(getattr(self, "_image_buffer", []))
        if buf:
            try:
                newest = buf[-1]
                oldest = buf[0]
                newest_stamp = newest.get("stamp", rospy.Time(0))
                oldest_stamp = oldest.get("stamp", rospy.Time(0))
                newest_recv = newest.get("recv_time", rospy.Time(0))
                oldest_recv = oldest.get("recv_time", rospy.Time(0))
                buffer_diag.update(
                    {
                        "frame_count": len(buf),
                        "oldest_stamp_age_s": (now - oldest_stamp).to_sec(),
                        "newest_stamp_age_s": (now - newest_stamp).to_sec(),
                        "oldest_recv_age_s": (now - oldest_recv).to_sec(),
                        "newest_recv_age_s": (now - newest_recv).to_sec(),
                    }
                )
            except Exception:
                buffer_diag["frame_count"] = len(buf)
        return {
            "source": self.vla_image_source,
            "topic": self.vla_image_topic,
            "path": self.vla_image_path,
            "path_exists": bool(self.vla_image_path and os.path.isfile(os.path.expanduser(self.vla_image_path))),
            "url_configured": bool(self.vla_image_url),
            "max_age_s": self.vla_image_max_age_s,
            "max_bytes": self.vla_image_max_bytes,
            "event_image_path": self._event_image_path,
            "event_image_exists": bool(self._event_image_path and os.path.isfile(self._event_image_path)),
            "event_input_image_path": self._event_input_image_path,
            "event_input_image_exists": bool(
                self._event_input_image_path and os.path.isfile(self._event_input_image_path)
            ),
            "latest_image": latest,
            "latest_pointcloud": latest_cloud,
            "camera_video": {
                "enable": bool(self.vla_camera_video_enable),
                "active": bool(self._use_camera_video_input()),
                "frame_count": int(self.vla_camera_video_frame_count),
                "interval_s": float(self.vla_camera_video_interval_s),
                "window_s": float(self.vla_camera_video_window_s),
                "sampled_frame_count": len(getattr(self, "_camera_video_frames", []) or []),
                "event_frame_paths": list(getattr(self, "_event_camera_video_frame_paths", []) or []),
            },
            "bev": {
                "topic": self.vla_bev_pointcloud_topic,
                "active_topic": self._active_bev_topic() if self._use_bev_image() else "",
                "active_sensor": self._active_bev_sensor_name() if self._use_bev_image() else "",
                "max_age_s": self.vla_bev_cloud_max_age_s,
                "range_forward_m": self.vla_bev_range_forward_m,
                "range_backward_m": self.vla_bev_range_backward_m,
                "range_side_m": self.vla_bev_range_side_m,
                "z_min_m": self.vla_bev_z_min_m,
                "z_max_m": self.vla_bev_z_max_m,
                "max_points": self.vla_bev_max_points,
                "image_width": self.vla_bev_image_width,
                "image_height": self.vla_bev_image_height,
                "track_box_margin_m": self.vla_bev_track_box_margin_m,
                "video_enable": bool(self.vla_bev_video_enable),
                "video_frame_count": int(self.vla_bev_video_frame_count),
                "video_interval_s": float(self.vla_bev_video_interval_s),
                "video_window_s": float(self.vla_bev_video_window_s),
                "video_skip_initial_frames": int(self.vla_bev_video_skip_initial_frames),
                "fallback_to_camera": bool(self.vla_bev_fallback_to_camera),
                "warmup_s": self.vla_bev_warmup_s,
                "warmup_min_frames": self.vla_bev_warmup_min_frames,
                "target_history_warmup_s": self.vla_target_history_warmup_s,
                "target_history_warmup_min_samples": self.vla_target_history_warmup_min_samples,
                "target_history_warmup_min_ready_targets": self.vla_target_history_warmup_min_ready_targets,
            },
            "global_trajectory": {
                "active": bool(self._use_global_trajectory_image()),
                "source_mode": "ais_odom" if self._use_ais_trajectory_image() else "perception_track",
                "history_s": float(getattr(self, "vla_global_history_s", 0.0)),
                "image_width": int(getattr(self, "vla_global_image_width", 0)),
                "image_height": int(getattr(self, "vla_global_image_height", 0)),
                "min_span_m": float(getattr(self, "vla_global_min_span_m", 0.0)),
                "available": bool(self._global_trajectory_available_for_capture()) if self._use_global_trajectory_image() else False,
            },
            "prompt": {
                "track_input_enable": bool(getattr(self, "vla_prompt_track_input_enable", True)),
            },
            "buffer": buffer_diag,
        }

    def _use_pointcloud_bev_image(self):
        return str(getattr(self, "vla_image_source", "") or "").strip().lower() == "pointcloud_bev"

    def _use_bev_image(self):
        return self._use_pointcloud_bev_image()

    def _use_camera_image(self):
        return str(getattr(self, "vla_image_source", "") or "").strip().lower() == "camera"

    def _use_global_trajectory_image(self):
        return str(getattr(self, "vla_image_source", "") or "").strip().lower() in ("global", "ais")

    def _use_ais_trajectory_image(self):
        return str(getattr(self, "vla_image_source", "") or "").strip().lower() == "ais"

    def _use_camera_video_input(self):
        return bool(
            self._llm_image_input_active()
            and self._use_camera_image()
            and self.vla_camera_video_enable
        )

    def _use_visual_video_input(self):
        if not self._llm_image_input_active():
            return False
        return bool(
            (self._use_bev_image() and self.vla_bev_video_enable)
            or self._use_camera_video_input()
        )

    def _active_bev_sensor_name(self):
        return "lidar"

    def _bev_source_label(self, suffix="base64"):
        prefix = "pointcloud_bev"
        return prefix + ("_" + str(suffix) if suffix else "")

    def _active_bev_topic(self):
        return self.vla_bev_pointcloud_topic

    def _active_pointcloud_msg(self):
        return getattr(self, "_latest_pointcloud_msg", None)

    def _active_pointcloud_stamp(self):
        return getattr(self, "_latest_pointcloud_stamp", rospy.Time(0))

    def _active_pointcloud_recv_time(self):
        return getattr(self, "_latest_pointcloud_recv_time", rospy.Time(0))

    def _active_pointcloud_first_recv_time(self):
        return getattr(self, "_first_pointcloud_recv_time", rospy.Time(0))

    def _active_pointcloud_frame_count(self):
        return int(getattr(self, "_pointcloud_frame_count", 0) or 0)

    def _latest_pointcloud_available_for_capture(self):
        if not self._llm_image_input_active():
            return True
        msg = self._active_pointcloud_msg()
        if msg is None:
            return False
        try:
            age_s = (rospy.Time.now() - self._active_pointcloud_stamp()).to_sec()
        except Exception:
            return False
        return age_s <= max(0.0, float(self.vla_bev_cloud_max_age_s))

    def _global_trajectory_records(self, now_s=None):
        if now_s is None:
            try:
                now_s = rospy.Time.now().to_sec()
            except Exception:
                latest = 0.0
                for rec in list(getattr(self, "_ego_world_history", [])):
                    latest = max(latest, self._safe_float(rec.get("t"), 0.0) if isinstance(rec, dict) else 0.0)
                history_source = getattr(self, "_ais_target_histories", {}) if self._use_ais_trajectory_image() else getattr(self, "_target_histories", {})
                for hist in history_source.values():
                    for rec in list(hist):
                        latest = max(latest, self._safe_float(rec.get("t"), 0.0) if isinstance(rec, dict) else 0.0)
                now_s = latest
        else:
            now_s = float(now_s)
        cut_t = now_s - max(0.5, float(self.vla_global_history_s))
        ego = [
            dict(rec)
            for rec in list(getattr(self, "_ego_world_history", []))
            if isinstance(rec, dict)
            and self._safe_float(rec.get("t"), -1.0) >= cut_t
            and self._safe_float(rec.get("x"), None) is not None
            and self._safe_float(rec.get("y"), None) is not None
        ]
        targets = {}
        target_histories = (
            getattr(self, "_ais_target_histories", {})
            if self._use_ais_trajectory_image()
            else getattr(self, "_target_histories", {})
        )
        for tid, hist in target_histories.items():
            pts = [
                dict(rec)
                for rec in list(hist)
                if isinstance(rec, dict)
                and self._safe_float(rec.get("t"), -1.0) >= cut_t
                and self._safe_float(rec.get("world_x"), None) is not None
                and self._safe_float(rec.get("world_y"), None) is not None
            ]
            if pts:
                targets[str(tid)] = pts
        if (not self._use_ais_trajectory_image()) and (not targets) and getattr(self, "_state_target_history", None):
            pts = [
                dict(rec)
                for rec in list(self._state_target_history)
                if isinstance(rec, dict)
                and self._safe_float(rec.get("t"), -1.0) >= cut_t
                and self._safe_float(rec.get("world_x"), None) is not None
                and self._safe_float(rec.get("world_y"), None) is not None
            ]
            if pts:
                targets["track_1"] = pts
        return {"ego": ego, "targets": targets, "now_s": now_s, "cut_t": cut_t}

    def _global_trajectory_available_for_capture(self):
        if not self._llm_image_input_active():
            return True
        records = self._global_trajectory_records()
        if len(records.get("ego", [])) < 2:
            return False
        targets = records.get("targets", {})
        return any(len(pts) >= 2 for pts in targets.values())

    def _latest_ros_image_available_for_capture(self):
        if not self._llm_image_input_active():
            return True
        if self._use_global_trajectory_image():
            return self._global_trajectory_available_for_capture()
        if self._use_bev_image():
            if self.vla_bev_video_enable:
                if len(self._recent_bev_video_frame_records()) >= int(self.vla_bev_video_frame_count):
                    return True
                if not self.vla_bev_fallback_to_camera:
                    return False
            if self._latest_pointcloud_available_for_capture():
                return True
            if not self.vla_bev_fallback_to_camera:
                return False
        msg = self._latest_image_msg
        if msg is None:
            return False
        try:
            age_s = (rospy.Time.now() - self._latest_image_stamp).to_sec()
        except Exception:
            return False
        return age_s <= max(0.0, self.vla_image_max_age_s)

    def _vla_image_available_for_call(self):
        if not self._llm_image_input_active():
            return True
        if self._use_camera_video_input():
            return (
                len(self._valid_event_camera_video_records()) >= int(self.vla_camera_video_frame_count)
                or len(self._recent_camera_video_frame_records()) >= int(self.vla_camera_video_frame_count)
            )
        if self._event_image_path and os.path.isfile(self._event_image_path):
            return True
        if self._use_global_trajectory_image():
            return self._global_trajectory_available_for_capture()
        if self._use_bev_image():
            if self.vla_bev_video_enable:
                if len(self._recent_bev_video_frame_records()) >= int(self.vla_bev_video_frame_count):
                    return True
                if not self.vla_bev_fallback_to_camera:
                    return False
            if self._latest_pointcloud_available_for_capture():
                return True
            if not self.vla_bev_fallback_to_camera:
                return False
        if self.vla_image_path and os.path.isfile(os.path.expanduser(self.vla_image_path)):
            return True
        if self.vla_image_url:
            return True
        return self._latest_ros_image_available_for_capture()

    def _dump_llm_error(self, call_id, err_detail, prompt="", exception_type="", traceback_text=""):
        self._append_llm_io_record(
            {
                "event": "llm_error",
                "wall_time_s": time.time(),
                "ros_time_s": rospy.Time.now().to_sec(),
                "call_id": call_id,
                "backend": self._backend_name_zh(self.llm_backend),
                "backend_key": self.llm_backend,
                "model": self._last_llm_model_id or "",
                "elapsed_ms": self._last_llm_elapsed_ms,
                "reason": self._call_reason_zh(self._last_llm_call_reason),
                "reason_key": self._last_llm_call_reason,
                "vla_prompt_mode": self.vla_prompt_mode,
                "trigger": bool(self._trigger),
                "state_seq": self._state_seq,
                "state_snapshot": self._state_snapshot(),
                "image_source": self._last_llm_image_source or "",
                "image_diag": self._vla_image_diag(),
                "prompt": prompt,
                "exception_type": exception_type,
                "error": str(err_detail or ""),
                "traceback": traceback_text,
            }
        )

    def __init__(self):
        rospy.init_node("colregs_llm_decision_node")
        trigger_topic = rospy.get_param("~trigger_topic", "/collision/trigger")
        state_topic = rospy.get_param("~state_topic", "/collision/state_estimation")
        self.decision_topic = rospy.get_param("~decision_topic", "/collision/llm_decision")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/collision/llm_cmd_vel")
        self.ego_odom_topic = rospy.get_param("~ego_odom_topic", "/myboat/odom")
        self.ais_target_odom_topic = str(rospy.get_param("~ais_target_odom_topic", "/target_boat/odom")).strip()
        self.ais_target2_odom_topic = str(rospy.get_param("~ais_target2_odom_topic", "/target_boat_2/odom")).strip()
        self.ais_target3_odom_topic = str(rospy.get_param("~ais_target3_odom_topic", "/target_boat_3/odom")).strip()
        self.vla_goal_x = self._safe_float(rospy.get_param("~vla_goal_x", ""), None)
        self.vla_goal_y = self._safe_float(rospy.get_param("~vla_goal_y", ""), None)
        self.starboard_turn_rate = float(rospy.get_param("~starboard_turn_rate", -0.5))
        self.safe_forward_speed = float(rospy.get_param("~safe_forward_speed", 0.6))
        self.normal_forward_speed = float(rospy.get_param("~normal_forward_speed", 1.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.llm_enable = bool(rospy.get_param("~llm_enable", True))
        self.llm_backend = str(rospy.get_param("~llm_backend", "http")).strip().lower()
        self.ollama_cmd = rospy.get_param(
            "~ollama_command", ["ollama", "run", "ros_ai_agent", "--think=false"]
        )
        self.ollama_timeout = float(rospy.get_param("~ollama_timeout", 20.0))
        self.llm_update_interval_s = max(
            1.0, float(rospy.get_param("~llm_update_interval_s", 10.0))
        )
        self.llm_once_per_trigger = bool(rospy.get_param("~llm_once_per_trigger", True))
        api_key_env = str(rospy.get_param("~api_key_env", "DASHSCOPE_API_KEY"))
        self.api_url = str(
            rospy.get_param(
                "~api_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            )
        ).strip()
        self.api_model = str(rospy.get_param("~api_model", "qwen3-vl-2b-instruct-local")).strip()
        self.api_key = str(rospy.get_param("~api_key", "")).strip()
        if not self.api_key and api_key_env:
            self.api_key = (os.environ.get(api_key_env) or "").strip()
        self.api_timeout = float(rospy.get_param("~api_timeout", self.ollama_timeout))
        self.api_temperature = float(rospy.get_param("~api_temperature", 0.2))
        self.api_max_output_tokens = int(rospy.get_param("~api_max_output_tokens", 220))
        self.api_force_json_mode = bool(rospy.get_param("~api_force_json_mode", True))
        self.api_repeat_retry_enable = bool(rospy.get_param("~api_repeat_retry_enable", True))
        self.api_repeat_ngram_words = max(3, int(rospy.get_param("~api_repeat_ngram_words", 8)))
        self.api_repeat_ngram_min_count = max(2, int(rospy.get_param("~api_repeat_ngram_min_count", 3)))
        self.api_no_json_retry_chars = max(80, int(rospy.get_param("~api_no_json_retry_chars", 180)))
        self.dinov2_model_name = str(
            rospy.get_param("~dinov2_model_name", "facebook/dinov2-base")
        ).strip()
        self.dinov2_device = str(rospy.get_param("~dinov2_device", "")).strip()
        self.dinov2_prototypes_dir = str(
            rospy.get_param(
                "~dinov2_prototypes_dir",
                DEFAULT_DINOV2_PROTOTYPES_DIR,
            )
        ).strip()
        self.dinov2_min_similarity = float(rospy.get_param("~dinov2_min_similarity", 0.34))
        self.dinov2_similarity_margin = float(rospy.get_param("~dinov2_similarity_margin", 0.035))
        self.dinov2_heuristic_only = bool(rospy.get_param("~dinov2_heuristic_only", False))
        self.vla_image_topic = str(
            rospy.get_param(
                "~vla_image_topic", "/myboat/sensors/cameras/front_camera/image_raw"
            )
        ).strip()
        self.vla_image_path = str(rospy.get_param("~vla_image_path", "")).strip()
        self.vla_image_url = str(rospy.get_param("~vla_image_url", "")).strip()
        self.llm_use_image_input = bool(rospy.get_param("~llm_use_image_input", True))
        self.vla_image_source = str(rospy.get_param("~vla_image_source", "pointcloud_bev")).strip().lower()
        if self.vla_image_source == "bev":
            self.vla_image_source = "pointcloud_bev"
        if self.vla_image_source == "lidar_bev":
            self.vla_image_source = "pointcloud_bev"
        if self.vla_image_source not in ("camera", "pointcloud_bev", "global", "ais"):
            rospy.logwarn(
                "[colregs_llm_decision] unknown vla_image_source=%s, fallback to pointcloud_bev",
                self.vla_image_source,
            )
            self.vla_image_source = "pointcloud_bev"
        self.vla_global_history_s = max(
            0.5, float(rospy.get_param("~vla_global_history_s", 3.0))
        )
        self.vla_global_image_width = max(
            480, int(rospy.get_param("~vla_global_image_width", 1000))
        )
        self.vla_global_image_height = max(
            480, int(rospy.get_param("~vla_global_image_height", 1000))
        )
        self.vla_global_min_span_m = max(
            5.0, float(rospy.get_param("~vla_global_min_span_m", 30.0))
        )
        self.vla_bev_pointcloud_topic = str(
            rospy.get_param("~vla_bev_pointcloud_topic", "/myboat/sensors/lidar_wamv/points")
        ).strip()
        self.vla_bev_cloud_max_age_s = float(rospy.get_param("~vla_bev_cloud_max_age_s", 1.5))
        self.vla_bev_range_forward_m = max(
            5.0, float(rospy.get_param("~vla_bev_range_forward_m", 60.0))
        )
        self.vla_bev_range_backward_m = max(
            5.0, float(rospy.get_param("~vla_bev_range_backward_m", self.vla_bev_range_forward_m))
        )
        self.vla_bev_range_side_m = max(
            5.0, float(rospy.get_param("~vla_bev_range_side_m", 35.0))
        )
        self.vla_bev_z_min_m = float(rospy.get_param("~vla_bev_z_min_m", -1.5))
        self.vla_bev_z_max_m = float(rospy.get_param("~vla_bev_z_max_m", 4.0))
        self.vla_bev_max_points = max(1000, int(rospy.get_param("~vla_bev_max_points", 120000)))
        self.vla_bev_image_width = max(480, int(rospy.get_param("~vla_bev_image_width", 1000)))
        self.vla_bev_image_height = max(480, int(rospy.get_param("~vla_bev_image_height", 1000)))
        self.vla_bev_track_box_margin_m = max(
            0.0, float(rospy.get_param("~vla_bev_track_box_margin_m", 1.0))
        )
        self.vla_bev_video_enable = bool(rospy.get_param("~vla_bev_video_enable", True))
        self.vla_bev_video_frame_count = max(
            1, int(rospy.get_param("~vla_bev_video_frame_count", 4))
        )
        self.vla_bev_video_interval_s = max(
            0.05, float(rospy.get_param("~vla_bev_video_interval_s", 2.5))
        )
        self.vla_bev_video_skip_initial_frames = max(
            0, int(rospy.get_param("~vla_bev_video_skip_initial_frames", 9))
        )
        self.vla_bev_video_window_s = max(
            self.vla_bev_video_interval_s * max(1, self.vla_bev_video_frame_count - 1),
            float(rospy.get_param("~vla_bev_video_window_s", 7.5)),
        )
        self.vla_bev_video_buffer_max_frames = max(
            self.vla_bev_video_frame_count,
            int(rospy.get_param("~vla_bev_video_buffer_max_frames", 24)),
        )
        self.vla_bev_fallback_to_camera = bool(
            rospy.get_param("~vla_bev_fallback_to_camera", False)
        )
        self.vla_bev_warmup_s = max(0.0, float(rospy.get_param("~vla_bev_warmup_s", 2.0)))
        self.vla_bev_warmup_min_frames = max(
            1, int(rospy.get_param("~vla_bev_warmup_min_frames", 3))
        )
        self.vla_target_history_warmup_s = max(
            0.0, float(rospy.get_param("~vla_target_history_warmup_s", 2.0))
        )
        self.vla_target_history_warmup_min_samples = max(
            1, int(rospy.get_param("~vla_target_history_warmup_min_samples", 3))
        )
        self.vla_target_history_warmup_min_ready_targets = max(
            1, int(rospy.get_param("~vla_target_history_warmup_min_ready_targets", 1))
        )
        self.vla_prompt_track_input_enable = self._as_bool(
            rospy.get_param("~vla_prompt_track_input_enable", True)
        )
        self.vla_prompt_mode = str(rospy.get_param("~vla_prompt_mode", "full")).strip().lower()
        if self.vla_prompt_mode not in (
            "rgb_only_vlm",
            "track_overlay",
            "trajectory_tokens",
            "full",
        ):
            rospy.logwarn(
                "[colregs_llm_decision] unknown vla_prompt_mode=%s, fallback to full",
                self.vla_prompt_mode,
            )
            self.vla_prompt_mode = "full"
        self.vla_camera_hfov_deg = float(rospy.get_param("~vla_camera_hfov_deg", 90.0))
        self.llm_fov_gate_enable = bool(rospy.get_param("~llm_fov_gate_enable", True))
        self.llm_fov_gate_margin_deg = float(rospy.get_param("~llm_fov_gate_margin_deg", 6.0))
        self.llm_fov_gate_min_forward_m = float(rospy.get_param("~llm_fov_gate_min_forward_m", 1.0))
        self.llm_fov_gate_max_distance_m = float(rospy.get_param("~llm_fov_gate_max_distance_m", 20.0))
        self.llm_fov_gate_state_max_distance_m = float(
            rospy.get_param("~llm_fov_gate_state_max_distance_m", 20.0)
        )
        self.llm_fov_gate_min_visible_tracks = int(
            rospy.get_param("~llm_fov_gate_min_visible_tracks", 1)
        )
        self.llm_fov_gate_require_all_tracks = bool(
            rospy.get_param("~llm_fov_gate_require_all_tracks", True)
        )
        self.llm_fov_gate_partial_after_s = float(
            rospy.get_param("~llm_fov_gate_partial_after_s", 2.0)
        )
        self.llm_fov_gate_bypass_distance = bool(
            rospy.get_param("~llm_fov_gate_bypass_distance", True)
        )
        self.llm_fov_gate_state_fallback_enable = bool(
            rospy.get_param("~llm_fov_gate_state_fallback_enable", True)
        )
        self.llm_fov_gate_state_fallback_after_s = float(
            rospy.get_param("~llm_fov_gate_state_fallback_after_s", 0.0)
        )
        self.llm_fov_gate_soft_enable = bool(
            rospy.get_param("~llm_fov_gate_soft_enable", True)
        )
        self.llm_fov_gate_soft_after_s = float(
            rospy.get_param("~llm_fov_gate_soft_after_s", 0.0)
        )
        self.llm_fov_gate_soft_max_targets = max(
            1, int(rospy.get_param("~llm_fov_gate_soft_max_targets", 3))
        )
        self.llm_require_unknown_track_in_fov = bool(
            rospy.get_param("~llm_require_unknown_track_in_fov", True)
        )
        self.vla_image_max_age_s = float(rospy.get_param("~vla_image_max_age_s", 1.5))
        self.vla_image_max_bytes = int(rospy.get_param("~vla_image_max_bytes", 8 * 1024 * 1024))
        self.vla_image_buffer_duration_s = max(
            0.0, float(rospy.get_param("~vla_image_buffer_duration_s", 2.0))
        )
        self.vla_image_buffer_max_frames = max(
            1, int(rospy.get_param("~vla_image_buffer_max_frames", 30))
        )
        self.vla_camera_video_enable = bool(
            rospy.get_param("~vla_camera_video_enable", True)
        )
        self.vla_camera_video_frame_count = max(
            1,
            int(
                rospy.get_param(
                    "~vla_camera_video_frame_count",
                    self.vla_bev_video_frame_count,
                )
            ),
        )
        self.vla_camera_video_interval_s = max(
            0.05,
            float(
                rospy.get_param(
                    "~vla_camera_video_interval_s",
                    self.vla_bev_video_interval_s,
                )
            ),
        )
        self.vla_camera_video_window_s = max(
            self.vla_camera_video_interval_s * max(1, self.vla_camera_video_frame_count),
            float(
                rospy.get_param(
                    "~vla_camera_video_window_s",
                    self.vla_camera_video_interval_s * max(1, self.vla_camera_video_frame_count),
                )
            ),
        )
        self.vla_camera_video_buffer_max_frames = max(
            self.vla_camera_video_frame_count,
            int(rospy.get_param("~vla_camera_video_buffer_max_frames", 24)),
        )
        self.vla_annotated_snapshot_enable = bool(
            rospy.get_param("~vla_annotated_snapshot_enable", True)
        )
        self.vla_trial_index = str(rospy.get_param("~vla_trial_index", "")).strip()
        self.vla_snapshot_on_trigger = bool(rospy.get_param("~vla_snapshot_on_trigger", True))
        self.vla_snapshot_dir = str(rospy.get_param("~vla_snapshot_dir", DEFAULT_TMP_DIR)).strip()
        self.perception_targets_topic = str(
            rospy.get_param("~perception_targets_topic", "/collision/perception/targets")
        ).strip()
        self.risk_topic = str(rospy.get_param("~risk_topic", "/collision/risk")).strip()
        self.target_history_window_s = float(rospy.get_param("~target_history_window_s", 8.0))
        self.target_history_max_samples = int(rospy.get_param("~target_history_max_samples", 80))
        self.target_history_min_dt_s = float(rospy.get_param("~target_history_min_dt_s", 0.15))
        self.llm_control_delay_s = float(rospy.get_param("~llm_control_delay_s", 2.0))
        self.llm_future_point_dt_s = float(rospy.get_param("~llm_future_point_dt_s", 2.0))
        self.llm_trigger_distance_m = float(rospy.get_param("~llm_trigger_distance_m", 11.0))
        self.sensor_front_hazard_enable = self._as_bool(
            rospy.get_param("~sensor_front_hazard_enable", True)
        )
        self.sensor_front_hazard_range_m = float(
            rospy.get_param("~sensor_front_hazard_range_m", self.llm_trigger_distance_m)
        )
        self.sensor_front_hazard_bearing_deg = float(
            rospy.get_param("~sensor_front_hazard_bearing_deg", 45.0)
        )
        self.sensor_front_hazard_min_forward_m = float(
            rospy.get_param("~sensor_front_hazard_min_forward_m", 1.0)
        )
        self.llm_pretrigger_enable = bool(rospy.get_param("~llm_pretrigger_enable", True))
        self.llm_pretrigger_distance_m = float(
            rospy.get_param("~llm_pretrigger_distance_m", max(30.0, self.llm_trigger_distance_m))
        )
        self.llm_require_pointcloud_distance = bool(
            rospy.get_param("~llm_require_pointcloud_distance", True)
        )
        gemini_key_env = str(rospy.get_param("~gemini_api_key_env", "GEMINI_API_KEY"))
        self.gemini_api_base = str(
            rospy.get_param(
                "~gemini_api_base",
                "https://generativelanguage.googleapis.com",
            )
        ).rstrip("/")
        # 若遇 gemini_http_404，请用 ListModels 核对当前账号可用的 name（不含 models/ 前缀），例如 gemini-3.1-flash-lite。
        self.gemini_model = str(
            rospy.get_param("~gemini_model", "gemini-3.1-flash-lite")
        ).strip()
        self.gemini_api_key = str(rospy.get_param("~gemini_api_key", "")).strip()
        if not self.gemini_api_key and gemini_key_env:
            self.gemini_api_key = (os.environ.get(gemini_key_env) or "").strip()
        self.gemini_max_output_tokens = int(
            rospy.get_param("~gemini_max_output_tokens", 1024)
        )
        self._last_llm_remote_ok = False
        self._last_llm_model_id = ""
        self._last_llm_image_data_url_len = 0
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self.debug_topic = rospy.get_param("~debug_topic", "/collision/llm_debug")
        self.raw_output_topic = rospy.get_param("~raw_output_topic", "/collision/llm_raw_output")
        self.raw_output_max_chars = int(rospy.get_param("~raw_output_max_chars", 0))
        self.llm_terminal_preview_chars = int(rospy.get_param("~llm_terminal_preview_chars", 0))
        self.terminal_llm_io_enable = bool(rospy.get_param("~terminal_llm_io_enable", True))
        self.terminal_clean_mode = bool(rospy.get_param("~terminal_clean_mode", True))
        self.llm_io_log_path = str(rospy.get_param("~llm_io_log_path", "")).strip()
        self.vla_event_log_path = str(rospy.get_param("~vla_event_log_path", "")).strip()
        self.llm_gate_status_log_interval_s = float(
            rospy.get_param("~llm_gate_status_log_interval_s", 5.0)
        )
        self.vlm_check_topic = rospy.get_param("~vlm_check_topic", "/collision/vlm_check")
        self.vlm_check_enable = bool(rospy.get_param("~vlm_check_enable", True))
        self.vlm_check_raw_max_chars = int(rospy.get_param("~vlm_check_raw_max_chars", 0))
        self.terminal_status_enable = bool(rospy.get_param("~terminal_status_enable", False))
        self.terminal_status_interval_s = float(
            rospy.get_param("~terminal_status_interval_s", 1.0)
        )
        self.recovery_active_topic = rospy.get_param(
            "~recovery_active_topic", "/collision/recovery_active"
        )
        self.llm_call_requires_trigger = bool(
            rospy.get_param("~llm_call_requires_trigger", True)
        )
        self.llm_require_trigger_low_before_call = bool(
            rospy.get_param("~llm_require_trigger_low_before_call", True)
        )
        self.default_traj_duration_s = float(
            rospy.get_param("~default_traj_duration_s", 8.0)
        )
        self.default_max_linear_acc = float(
            rospy.get_param("~default_max_linear_acc", 0.20)
        )
        self.default_max_angular_acc = float(
            rospy.get_param("~default_max_angular_acc", 0.35)
        )
        self.default_avoid_speed = float(rospy.get_param("~default_avoid_speed", 0.56))
        self.default_avoid_turn_rate = float(
            rospy.get_param("~default_avoid_turn_rate", self.starboard_turn_rate)
        )
        self.avoid_turn_rate_limit = float(rospy.get_param("~avoid_turn_rate_limit", 0.35))
        self.default_min_avoid_speed = float(
            rospy.get_param("~default_min_avoid_speed", 0.20)
        )
        self.pending_safety_enable = bool(rospy.get_param("~pending_safety_enable", True))
        self.pending_safety_speed_scale = float(rospy.get_param("~pending_safety_speed_scale", 0.35))
        self.pending_safety_turn_rate = float(rospy.get_param("~pending_safety_turn_rate", 0.18))
        self.pending_safety_duration_s = float(rospy.get_param("~pending_safety_duration_s", 10.0))
        self.pending_safety_dcpa_m = float(rospy.get_param("~pending_safety_dcpa_m", 3.5))
        self.pending_safety_urgent_tcpa_s = float(rospy.get_param("~pending_safety_urgent_tcpa_s", 5.0))
        self.pending_safety_emergency_range_m = float(rospy.get_param("~pending_safety_emergency_range_m", 4.5))
        self.pending_safety_emergency_tcpa_s = float(rospy.get_param("~pending_safety_emergency_tcpa_s", 2.5))
        self.recovery_heading_kp = float(rospy.get_param("~recovery_heading_kp", 1.3))
        self.recovery_max_turn_rate = float(
            rospy.get_param("~recovery_max_turn_rate", 0.5)
        )
        self.recovery_heading_tolerance = float(
            rospy.get_param("~recovery_heading_tolerance", 0.10)
        )
        self.recovery_hold_s = float(rospy.get_param("~recovery_hold_s", 1.2))
        self.recovery_linear_speed = float(
            rospy.get_param("~recovery_linear_speed", self.normal_forward_speed)
        )
        self.recovery_enable_lane_recenter = bool(
            rospy.get_param("~recovery_enable_lane_recenter", True)
        )
        self.recovery_lateral_kp = float(rospy.get_param("~recovery_lateral_kp", 0.10))
        self.recovery_lateral_tolerance = float(
            rospy.get_param("~recovery_lateral_tolerance", 1.0)
        )
        self.recovery_left_turn_bias = float(
            rospy.get_param("~recovery_left_turn_bias", 0.03)
        )
        self.recovery_auto_when_tcpa_negative = bool(
            rospy.get_param("~recovery_auto_when_tcpa_negative", True)
        )
        self.recovery_tcpa_release_threshold = float(
            rospy.get_param("~recovery_tcpa_release_threshold", -1.0)
        )
        self.llm_async_enable = bool(rospy.get_param("~llm_async_enable", True))
        self.llm_disable_after_first_success = bool(
            rospy.get_param("~llm_disable_after_first_success", True)
        )
        self.llm_rising_edge_min_interval_s = float(
            rospy.get_param("~llm_rising_edge_min_interval_s", 20.0)
        )
        self.llm_max_calls_per_minute = int(
            rospy.get_param("~llm_max_calls_per_minute", 6)
        )
        self.llm_quota_cooldown_s = float(rospy.get_param("~llm_quota_cooldown_s", 60.0))
        self.decision_debug_hz = float(rospy.get_param("~decision_debug_hz", 1.0))
        self.cmd_log_enable = bool(rospy.get_param("~cmd_log_enable", False))
        self.cmd_log_interval_s = float(rospy.get_param("~cmd_log_interval_s", 5.0))

        self._trigger = False
        self._state = {}
        self._state_seq = 0
        self._last_llm_t = rospy.Time(0)
        self._last_llm_error = ""
        self._last_llm_elapsed_ms = -1.0
        self._last_llm_called = False
        self._last_llm_call_reason = "init"
        self._llm_call_seq = 0
        self._llm_call_timestamps = []
        self._llm_quota_block_until = rospy.Time(0)
        self._last_trigger_log_t = rospy.Time(0)
        self._prev_trigger_for_llm = False
        self._trigger_generation = 0
        self._trigger_seen_low = False
        self._last_llm_attempt_trigger_generation = -1
        self._last_snapshot_trigger_generation = -1
        self._bev_snapshot_seq = 0
        self._last_cls = {
            "confidence": 0.0,
            "reasoning": "No model result yet",
            "track_classifications": [],
            "trajectory_constraints": {},
        }
        self._last_llm_raw_preview = ""
        self._last_llm_raw_text = ""
        self._last_parse_diag = ""
        self._llm_inflight = False
        self._llm_pending_result = None
        self._llm_lock = threading.Lock()
        self._llm_locked_after_success = False
        self._llm_locked_call_id = -1
        self._latest_image_msg = None
        self._latest_image_stamp = rospy.Time(0)
        self._image_buffer = deque()
        self._latest_pointcloud_msg = None
        self._latest_pointcloud_stamp = rospy.Time(0)
        self._first_pointcloud_recv_time = rospy.Time(0)
        self._latest_pointcloud_recv_time = rospy.Time(0)
        self._pointcloud_frame_count = 0
        self._bev_video_frames = deque()
        self._bev_video_seq = 0
        self._camera_video_frames = deque()
        self._camera_video_seq = 0
        self._last_bev_video_frame_record_t = rospy.Time(0)
        self._last_camera_video_frame_record_t = rospy.Time(0)
        self._bev_video_first_stamp_s = None
        self._bev_video_seen_stamp_count = 0
        self._bev_video_last_seen_stamp_s = None
        self._last_llm_image_source = ""
        self._last_llm_video_frame_paths = []
        self._last_llm_video_frame_records = []
        self._event_camera_video_frame_paths = []
        self._event_camera_video_frame_records = []
        self._event_image_path = ""
        self._event_image_url = ""
        self._event_input_image_path = ""
        self._event_input_image_url = ""
        self._last_vlm_score_plot_path = ""
        self._last_vlm_frame_score_records = []
        self._last_vlm_snapshot_annotated = False
        self._last_vlm_snapshot_annotation_error = ""
        self._perception_targets = []
        self._perception_stamp = 0.0
        self._risk_state = {}
        self._risk_stamp = 0.0
        self._target_histories = {}
        self._ais_target_histories = {}
        self._state_target_history = deque()
        self._ego_world_history = deque()
        self._active_traj = None
        self._recovery_active = False
        self._pre_encounter_heading = None
        self._pre_encounter_pos = None
        self._last_lateral_error = None
        self._recovery_ok_since = None
        self._post_recovery_keep_course = False
        self._ego_pos_xy = None
        self._last_decision_debug_t = 0.0
        self._last_terminal_status_t = 0.0
        self._last_cmd_log_t = 0.0
        self._last_cmd_log_sig = None
        self._last_no_image_skip_log_t = rospy.Time(0)
        self._fov_gate_first_track_t = None
        self._last_fov_gate_skip_log_t = rospy.Time(0)
        self._last_gate_status_record_t = rospy.Time(0)
        self._diagnostic_snapshot_saved = False
        self._last_fov_gate_diag = {}
        self._last_fov_gate_call_reason = ""
        self._last_sensor_front_hazard_diag = {}
        self._last_cmd = Twist()
        self._last_cmd_stamp = rospy.Time.now()
        self._decision_pub = rospy.Publisher(self.decision_topic, String, queue_size=20)
        self._cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=20)
        self._recovery_active_pub = rospy.Publisher(
            self.recovery_active_topic, Bool, queue_size=20
        )
        self._debug_pub = rospy.Publisher(self.debug_topic, String, queue_size=20)
        self._raw_pub = rospy.Publisher(self.raw_output_topic, String, queue_size=20)
        self._vlm_check_pub = rospy.Publisher(self.vlm_check_topic, String, queue_size=20)
        self._dinov2_classifier = None
        if self.llm_backend == "dinov2":
            if DinoV2ShipClassifier is None:
                raise RuntimeError(
                    "llm_backend=dinov2 but dinov2_ship_classifier.py is not importable: %s"
                    % (DINO_IMPORT_ERROR or "unknown")
                )
            self._dinov2_classifier = DinoV2ShipClassifier(
                model_name=self.dinov2_model_name,
                prototypes_dir=self.dinov2_prototypes_dir,
                device=self.dinov2_device,
                min_similarity=self.dinov2_min_similarity,
                similarity_margin=self.dinov2_similarity_margin,
                heuristic_only=self.dinov2_heuristic_only,
            )
        rospy.Subscriber(trigger_topic, Bool, self._trigger_cb, queue_size=20)
        rospy.Subscriber(state_topic, String, self._state_cb, queue_size=20)
        rospy.Subscriber(self.ego_odom_topic, Odometry, self._ego_odom_cb, queue_size=20)
        for target_id, topic in (
            ("target_boat", self.ais_target_odom_topic),
            ("target_boat_2", self.ais_target2_odom_topic),
            ("target_boat_3", self.ais_target3_odom_topic),
        ):
            if topic:
                rospy.Subscriber(
                    topic,
                    Odometry,
                    lambda msg, tid=target_id: self._ais_target_odom_cb(tid, msg),
                    queue_size=20,
                )
        if self.vla_image_topic:
            rospy.Subscriber(self.vla_image_topic, Image, self._image_cb, queue_size=1)
        if self.vla_bev_pointcloud_topic:
            rospy.Subscriber(
                self.vla_bev_pointcloud_topic, PointCloud2, self._pointcloud_cb, queue_size=1
            )
        if self.perception_targets_topic:
            rospy.Subscriber(
                self.perception_targets_topic, String, self._perception_targets_cb, queue_size=20
            )
        if self.risk_topic:
            rospy.Subscriber(self.risk_topic, String, self._risk_cb, queue_size=20)

        rospy.loginfo(
            "[colregs_llm_decision] 话题 trigger=%s state=%s perception=%s risk=%s ego_odom=%s decision=%s cmd=%s recovery=%s",
            trigger_topic,
            state_topic,
            self.perception_targets_topic,
            self.risk_topic,
            self.ego_odom_topic,
            self.decision_topic,
            self.cmd_topic,
            self.recovery_active_topic,
        )
        rospy.loginfo(
            "[colregs_llm_decision] LLM开关=%s | 调用方式=%s（%s）",
            "开" if self.llm_enable else "关",
            self._backend_name_zh(self.llm_backend),
            self.llm_backend,
        )
        rospy.loginfo("[colregs_llm_decision] VLA prompt mode=%s", self.vla_prompt_mode)
        if self.llm_backend == "ollama":
            rospy.loginfo(
                "[colregs_llm_decision] Ollama 命令=%s | 超时=%.1f 秒",
                " ".join(self.ollama_cmd)
                if isinstance(self.ollama_cmd, list)
                else str(self.ollama_cmd),
                self.ollama_timeout,
            )
        elif self.llm_backend == "dinov2":
            rospy.loginfo(
                "[colregs_llm_decision] DINOv2 模型=%s | device=%s | prototypes=%s | heuristic_only=%s",
                self.dinov2_model_name,
                self.dinov2_device or "auto",
                self.dinov2_prototypes_dir,
                str(self.dinov2_heuristic_only),
            )
        elif self.llm_backend == "gemini":
            rospy.loginfo(
                "[colregs_llm_decision] Gemini 基址=%s | 模型=%s | 密钥=%s | 超时=%.1f 秒",
                self.gemini_api_base,
                self.gemini_model,
                self._mask_secret(self.gemini_api_key),
                self.api_timeout,
            )
        else:
            key_hint = "未设置" if not self.api_key else "已设置"
            rospy.loginfo(
                "[colregs_llm_decision] HTTP 地址=%s | 模型=%s | API密钥=%s | 超时=%.1f 秒",
                self.api_url or "（未配置）",
                self.api_model,
                key_hint,
                self.api_timeout,
            )
            rospy.loginfo(
                "[colregs_llm_decision] VLA 图像源: topic=%s | path=%s | url=%s | 最大图像字节=%d | 最大时延=%.2fs",
                self.vla_image_topic or "（未配置）",
                self.vla_image_path or "（未配置）",
                self.vla_image_url or "（未配置）",
                self.vla_image_max_bytes,
                self.vla_image_max_age_s,
            )
            rospy.loginfo(
                "[colregs_llm_decision] VLA 输入图模式=%s | LiDAR点云=%s | active=%s | 点云最大时延=%.2fs | 前向=%.1fm | 后向=%.1fm | 侧向=±%.1fm | fallback_camera=%s",
                self.vla_image_source,
                self.vla_bev_pointcloud_topic or "（未配置）",
                self._active_bev_topic() if self._use_bev_image() else ("ais_trajectory" if self._use_ais_trajectory_image() else ("global_trajectory" if self._use_global_trajectory_image() else "camera")),
                self.vla_bev_cloud_max_age_s,
                self.vla_bev_range_forward_m,
                self.vla_bev_range_backward_m,
                self.vla_bev_range_side_m,
                "开" if self.vla_bev_fallback_to_camera else "关",
            )
        rospy.loginfo(
            "[colregs_llm_decision] VLA 图像缓冲: duration=%.2fs | max_frames=%d",
            self.vla_image_buffer_duration_s,
            self.vla_image_buffer_max_frames,
        )
        rospy.loginfo(
            "[colregs_llm_decision] VLA 事件图标注: enable=%s | opencv=%s",
            "开" if self.vla_annotated_snapshot_enable else "关",
            "可用" if cv2 is not None and np is not None else ("不可用: %s" % (CV2_IMPORT_ERROR or "unknown")),
        )
        rospy.loginfo(
            "[colregs_llm_decision] 终端状态心跳=%s | 间隔=%.1fs",
            "开" if self.terminal_status_enable else "关",
            self.terminal_status_interval_s,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 图像输入给LLM=%s（关闭时仍保存触发截图到 tmp）",
            "开" if self._llm_image_input_active() else "关",
        )
        rospy.loginfo(
            "[colregs_llm_decision] 每触发仅调一次=%s | 周期间隔=%.1f 秒（仅当「每触发仅调一次」为假时生效）",
            "是" if self.llm_once_per_trigger else "否",
            self.llm_update_interval_s,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 调试日志=%s",
            "开" if self.debug_enable else "关",
        )
        rospy.loginfo(
            "[colregs_llm_decision] debug_topic=%s raw_output_topic=%s",
            self.debug_topic,
            self.raw_output_topic,
        )
        rospy.loginfo(
            "[colregs_llm_decision] LLM输入输出文件日志=%s",
            self.llm_io_log_path or "（未配置）",
        )
        rospy.loginfo(
            "[colregs_llm_decision] VLA事件文件日志=%s",
            self._default_vla_event_log_path() or "（未配置）",
        )
        rospy.loginfo(
            "[colregs_llm_decision] vlm_check_topic=%s enable=%s",
            self.vlm_check_topic,
            "开" if self.vlm_check_enable else "关",
        )
        rospy.loginfo(
            "[colregs_llm_decision] LLM原文输出长度限制 raw_output_max_chars=%d（<=0 表示不截断）",
            self.raw_output_max_chars,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 事件截图: enable=%s dir=%s",
            "开" if self.vla_snapshot_on_trigger else "关",
            self.vla_snapshot_dir,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 点云触发门槛: trigger<=%.1fm | pretrigger=%s<=%.1fm | require_pointcloud=%s | targets_topic=%s",
            self.llm_trigger_distance_m,
            "开" if self.llm_pretrigger_enable else "关",
            self.llm_pretrigger_distance_m,
            "是" if self.llm_require_pointcloud_distance else "否",
            self.perception_targets_topic or "（未配置）",
        )
        rospy.loginfo(
            "[colregs_llm_decision] 传感器前方危险门控: enable=%s | range<=%.1fm | |bearing|<=%.1f deg | forward>=%.1fm",
            "开" if self.sensor_front_hazard_enable else "关",
            self.sensor_front_hazard_range_m,
            self.sensor_front_hazard_bearing_deg,
            self.sensor_front_hazard_min_forward_m,
        )
        rospy.loginfo(
            "[colregs_llm_decision] VLM FOV门控: enable=%s | hfov=%.1f deg | margin=%.1f deg | forward>=%.1fm | track_distance<=%.1fm | state_distance<=%.1fm | min_visible=%d | require_all=%s | partial_after=%.1fs | soft=%s after=%.1fs max=%d | bypass_distance=%s",
            "开" if self.llm_fov_gate_enable else "关",
            self.vla_camera_hfov_deg,
            self.llm_fov_gate_margin_deg,
            self.llm_fov_gate_min_forward_m,
            self.llm_fov_gate_max_distance_m,
            self.llm_fov_gate_state_max_distance_m,
            max(1, self.llm_fov_gate_min_visible_tracks),
            "是" if self.llm_fov_gate_require_all_tracks else "否",
            self.llm_fov_gate_partial_after_s,
            "开" if self.llm_fov_gate_soft_enable else "关",
            self.llm_fov_gate_soft_after_s,
            self.llm_fov_gate_soft_max_targets,
            "是" if self.llm_fov_gate_bypass_distance else "否",
        )
        rospy.loginfo(
            "[colregs_llm_decision] 历史运动窗口: %.1fs | max_samples=%d | min_dt=%.2fs",
            self.target_history_window_s,
            self.target_history_max_samples,
            self.target_history_min_dt_s,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 终端LLM预览长度 llm_terminal_preview_chars=%d（<=0 表示全量）",
            self.llm_terminal_preview_chars,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 无触发不调 LLM=%s",
            "是" if self.llm_call_requires_trigger else "否",
        )
        rospy.loginfo(
            "[colregs_llm_decision] LLM防抖/配额保护: rising_edge_min_interval=%.1fs, max_calls_per_min=%d, quota_cooldown=%.1fs",
            self.llm_rising_edge_min_interval_s,
            self.llm_max_calls_per_minute,
            self.llm_quota_cooldown_s,
        )
        rospy.loginfo(
            "[colregs_llm_decision] 决策调试精简模式：full_diag 频率=%.2f Hz",
            max(0.1, self.decision_debug_hz),
        )
        rospy.loginfo(
            "[colregs_llm_decision] cmd_out 控制台日志: enable=%s interval=%.1fs（变化事件会立即打印）",
            "开" if self.cmd_log_enable else "关",
            max(0.5, self.cmd_log_interval_s),
        )
        self._append_llm_io_record(
            {
                "event": "vlm_session_start",
                "wall_time_s": time.time(),
                "ros_time_s": rospy.Time.now().to_sec(),
                "backend": self._backend_name_zh(self.llm_backend),
                "backend_key": self.llm_backend,
                "model": self.api_model if self.llm_backend == "http" else self.gemini_model if self.llm_backend == "gemini" else self.dinov2_model_name if self.llm_backend == "dinov2" else "ollama",
                "vla_prompt_mode": self.vla_prompt_mode,
                "llm_enable": bool(self.llm_enable),
                "llm_use_image_input": bool(self._llm_image_input_active()),
                "llm_call_requires_trigger": bool(self.llm_call_requires_trigger),
                "llm_once_per_trigger": bool(self.llm_once_per_trigger),
                "llm_io_log_path": self.llm_io_log_path,
                "vla_event_log_path": self._default_vla_event_log_path(),
                "vla_snapshot_on_trigger": bool(self.vla_snapshot_on_trigger),
                "vla_snapshot_dir": self.vla_snapshot_dir,
                "vla_image_topic": self.vla_image_topic,
                "vla_image_buffer_duration_s": self.vla_image_buffer_duration_s,
                "vla_image_buffer_max_frames": self.vla_image_buffer_max_frames,
                "vla_annotated_snapshot_enable": bool(self.vla_annotated_snapshot_enable),
                "vla_trial_index": self.vla_trial_index,
                "llm_fov_gate_enable": bool(self.llm_fov_gate_enable),
                "llm_fov_gate_require_all_tracks": bool(self.llm_fov_gate_require_all_tracks),
                "llm_fov_gate_partial_after_s": self.llm_fov_gate_partial_after_s,
                "llm_fov_gate_soft_enable": bool(self.llm_fov_gate_soft_enable),
                "llm_fov_gate_soft_after_s": self.llm_fov_gate_soft_after_s,
                "llm_fov_gate_soft_max_targets": self.llm_fov_gate_soft_max_targets,
                "llm_require_unknown_track_in_fov": bool(self.llm_require_unknown_track_in_fov),
                "llm_gate_status_log_interval_s": self.llm_gate_status_log_interval_s,
            }
        )

    def _publish_debug(self, event, extra=None):
        if not self.debug_enable:
            return
        raw_out = self._last_llm_raw_text or ""
        if self.raw_output_max_chars > 0:
            raw_out = raw_out[: self.raw_output_max_chars]
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "event": event,
            "trigger": self._trigger,
            "llm_called": self._last_llm_called,
            "llm_call_reason": self._last_llm_call_reason,
            "llm_error": self._last_llm_error,
            "llm_remote_ok": self._last_llm_remote_ok,
            "llm_model": self._last_llm_model_id,
            "llm_elapsed_ms": self._last_llm_elapsed_ms,
            "state_seq": self._state_seq,
            "parse_diag": self._last_parse_diag,
            "raw_output": raw_out,
            "raw_len": len(self._last_llm_raw_text or ""),
            "llm_image_source": self._last_llm_image_source,
            "perception_diag": self._perception_diag(),
        }
        if isinstance(extra, dict):
            payload.update(extra)
        self._debug_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))

    def _publish_vlm_check(self, event, extra=None):
        if not self.vlm_check_enable:
            return
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "event": str(event),
            "trigger": bool(self._trigger),
            "llm_call_id": int(self._llm_call_seq),
            "llm_model": self._last_llm_model_id,
            "llm_image_source": self._last_llm_image_source,
            "event_image_path": self._event_image_path,
            "event_image_url": self._event_image_url,
            "event_input_image_path": self._event_input_image_path,
            "event_input_image_url": self._event_input_image_url,
            "score_plot_path": self._last_vlm_score_plot_path,
            "snapshot_annotated": bool(self._last_vlm_snapshot_annotated),
        }
        if isinstance(extra, dict):
            payload.update(extra)
        self._vlm_check_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))

    def _ego_odom_cb(self, msg):
        p = msg.pose.pose.position
        x = float(p.x)
        y = float(p.y)
        yaw = self._yaw_from_quat(msg.pose.pose.orientation)
        stamp_s = msg.header.stamp.to_sec() if getattr(msg, "header", None) and msg.header.stamp else 0.0
        if stamp_s <= 0.0:
            stamp_s = rospy.Time.now().to_sec()
        self._ego_pos_xy = (x, y)
        self._append_ego_world_sample(stamp_s, x, y, yaw)

    def _ais_target_odom_cb(self, target_id, msg):
        p = msg.pose.pose.position
        yaw = self._yaw_from_quat(msg.pose.pose.orientation)
        stamp_s = msg.header.stamp.to_sec() if getattr(msg, "header", None) and msg.header.stamp else 0.0
        if stamp_s <= 0.0:
            stamp_s = rospy.Time.now().to_sec()
        self._append_ais_target_sample(str(target_id), stamp_s, float(p.x), float(p.y), yaw)

    def _trigger_cb(self, msg):
        prev = self._trigger
        self._trigger = bool(msg.data)
        if self._trigger != prev:
            self._trigger_generation += 1
        if not self._trigger:
            self._trigger_seen_low = True
        valid_rising_edge = self._trigger_seen_low and (not prev) and self._trigger
        if ((not prev) and self._trigger) and (not valid_rising_edge):
            if self.debug_enable:
                rospy.loginfo(
                    "[colregs_llm_decision] 忽略启动期 trigger=true：尚未观测到 trigger=false 基线，不允许调用分类后端"
                )
        if valid_rising_edge:
            self._event_image_path = ""
            self._event_image_url = ""
            self._event_input_image_path = ""
            self._event_input_image_url = ""
            self._event_camera_video_frame_paths = []
            self._event_camera_video_frame_records = []
            if self._use_global_trajectory_image():
                self._ego_world_history.clear()
                self._target_histories.clear()
                self._ais_target_histories.clear()
                self._state_target_history.clear()
            self._last_snapshot_trigger_generation = -1
            ego_heading = self._state.get("ego_heading_rad")
            if ego_heading is not None:
                self._pre_encounter_heading = float(ego_heading)
                self._pre_encounter_pos = self._ego_pos_xy
                self._recovery_active = False
                self._recovery_ok_since = None
                self._post_recovery_keep_course = False
                self._last_lateral_error = None
                if self.debug_enable:
                    rospy.loginfo(
                        "[colregs_llm_decision] 记录会遇前参考: heading=%.3f rad pos=%s",
                        self._pre_encounter_heading,
                        str(self._pre_encounter_pos),
                    )
        if prev and (not self._trigger):
            # trigger 解除后进入航向恢复阶段，恢复完成后再完全放权给 normal_cmd。
            if self._pre_encounter_heading is not None:
                self._recovery_active = True
                self._recovery_ok_since = None
                if self.debug_enable:
                    rospy.loginfo(
                        "[colregs_llm_decision] 会遇结束，进入回航向恢复阶段"
                    )
        if self.debug_enable and (self._trigger != prev):
            now = rospy.Time.now()
            dt = (now - self._last_trigger_log_t).to_sec()
            self._last_trigger_log_t = now
            rospy.loginfo(
                "[colregs_llm_decision] 【触发器】%s → %s | generation=%d | 距上次变化 %.3f 秒",
                "关" if not prev else "开",
                "关" if not self._trigger else "开",
                self._trigger_generation,
                dt if dt >= 0 else -1.0,
            )

    def _llm_call_block_reason(self, trigger_generation=None):
        if self.llm_disable_after_first_success and self._llm_locked_after_success:
            return "skip_locked_after_success"
        pretrigger_active = self._semantic_pretrigger_active()
        if self.llm_call_requires_trigger and (not self._trigger) and (not pretrigger_active):
            return "skip_no_trigger"
        if (
            self.llm_call_requires_trigger
            and self.llm_require_trigger_low_before_call
            and (not self._trigger_seen_low)
        ):
            return "skip_trigger_not_armed"
        if trigger_generation is not None and self._trigger_generation != int(trigger_generation):
            return "skip_trigger_generation_changed"
        return ""

    def _llm_call_allowed(self, trigger_generation=None):
        return self._llm_call_block_reason(trigger_generation) == ""

    def _llm_result_generation_valid(self, trigger_generation):
        if trigger_generation is None:
            return True
        try:
            return self._trigger_generation == int(trigger_generation)
        except Exception:
            return False

    @staticmethod
    def _llm_result_is_success(result_obj):
        if not isinstance(result_obj, dict):
            return False
        return str(result_obj.get("source", "")).strip().lower() in ("llm", "dinov2")

    def _lock_llm_after_success_if_needed(self, result_obj):
        if (not self.llm_disable_after_first_success) or (not self._llm_result_is_success(result_obj)):
            return
        if self._llm_locked_after_success:
            return
        self._llm_locked_after_success = True
        self._llm_locked_call_id = int(result_obj.get("debug_call_id", -1))
        if self.debug_enable:
            rospy.loginfo(
                "[colregs_llm_decision] 【LLM】首次成功分类已锁定，后续不再调用分类后端 | call_id=%d",
                self._llm_locked_call_id,
            )
        self._publish_vlm_check(
            "vlm_locked_after_success",
            {
                "llm_call_id": self._llm_locked_call_id,
                "lock_enabled": True,
            },
        )

    def _maybe_auto_start_recovery(self):
        if not self.recovery_auto_when_tcpa_negative:
            return
        if not self._trigger:
            return
        if self._recovery_active:
            return
        if self._pre_encounter_heading is None:
            return
        tcpa = self._state.get("tcpa_s")
        if tcpa is None:
            return
        try:
            tcpa_v = float(tcpa)
        except Exception:
            return
        if tcpa_v <= self.recovery_tcpa_release_threshold:
            self._recovery_active = True
            self._recovery_ok_since = None
            if self.debug_enable:
                rospy.loginfo(
                    "[colregs_llm_decision] trigger仍为true但tcpa=%.3f<=%.3f，自动进入恢复阶段",
                    tcpa_v,
                    self.recovery_tcpa_release_threshold,
                )

    def _prune_image_buffer(self, now=None):
        if not hasattr(self, "_image_buffer"):
            return
        if now is None:
            now = rospy.Time.now()
        duration = float(self.vla_image_buffer_duration_s)
        if duration > 0.0:
            cutoff_s = now.to_sec() - duration
            while self._image_buffer:
                recv_time = self._image_buffer[0].get("recv_time", rospy.Time(0))
                try:
                    recv_s = recv_time.to_sec()
                except Exception:
                    recv_s = 0.0
                if recv_s >= cutoff_s:
                    break
                self._image_buffer.popleft()
        while len(self._image_buffer) > max(1, int(self.vla_image_buffer_max_frames)):
            self._image_buffer.popleft()

    def _recent_camera_buffer_records(self, count=None):
        if not hasattr(self, "_image_buffer"):
            return []
        now = rospy.Time.now()
        self._prune_image_buffer(now)
        requested = int(count if count is not None else self.vla_camera_video_frame_count)
        requested = max(1, requested)
        frames = [
            rec
            for rec in list(getattr(self, "_image_buffer", []))
            if isinstance(rec, dict) and rec.get("msg") is not None
        ]
        if not frames:
            return []
        try:
            newest_t = frames[-1].get("recv_time", frames[-1].get("stamp", rospy.Time(0)))
            newest_age_s = max(0.0, (now - newest_t).to_sec())
            if newest_age_s > max(0.0, float(self.vla_image_max_age_s)):
                return []
        except Exception:
            return []
        return frames[-requested:]

    def _record_camera_video_frame_if_needed(self, msg, stamp, recv_time):
        if not (self.vla_camera_video_enable and self._use_camera_image()):
            return
        if msg is None:
            return
        try:
            last_t = getattr(self, "_last_camera_video_frame_record_t", rospy.Time(0))
            if last_t.to_sec() > 0.0:
                dt = (recv_time - last_t).to_sec()
                if dt < float(self.vla_camera_video_interval_s):
                    return
        except Exception:
            pass
        self._last_camera_video_frame_record_t = recv_time
        record = {
            "stamp": stamp,
            "recv_time": recv_time,
            "msg": msg,
        }
        self._camera_video_frames.append(record)
        self._prune_camera_video_frames(recv_time)

    def _prune_camera_video_frames(self, now=None):
        if not hasattr(self, "_camera_video_frames"):
            return
        if now is None:
            now = rospy.Time.now()
        window_s = float(self.vla_camera_video_window_s)
        if window_s > 0.0:
            cutoff_s = now.to_sec() - window_s
            while self._camera_video_frames:
                rec_t = self._camera_video_frames[0].get("recv_time", rospy.Time(0))
                try:
                    rec_s = rec_t.to_sec()
                except Exception:
                    rec_s = 0.0
                if rec_s >= cutoff_s:
                    break
                self._camera_video_frames.popleft()
        while len(self._camera_video_frames) > int(self.vla_camera_video_buffer_max_frames):
            self._camera_video_frames.popleft()

    def _recent_camera_video_frame_records(self, count=None):
        if not hasattr(self, "_camera_video_frames"):
            return []
        now = rospy.Time.now()
        self._prune_camera_video_frames(now)
        requested = max(1, int(count if count is not None else self.vla_camera_video_frame_count))
        frames = [
            rec
            for rec in list(getattr(self, "_camera_video_frames", []))
            if isinstance(rec, dict) and rec.get("msg") is not None
        ]
        if len(frames) < requested:
            return frames[-requested:]
        try:
            newest_t = frames[-1].get("recv_time", frames[-1].get("stamp", rospy.Time(0)))
            newest_age_s = max(0.0, (now - newest_t).to_sec())
            max_sample_age_s = max(
                0.0,
                float(self.vla_image_max_age_s),
                float(self.vla_camera_video_interval_s) + float(self.vla_image_max_age_s),
            )
            if newest_age_s > max_sample_age_s:
                return []
        except Exception:
            return []
        return frames[-requested:]

    def _image_buffer_ready_for_vlm(self):
        if not self._llm_image_input_active():
            return True, {"required_frames": 0, "frame_count": 0, "ready": True}
        if self._use_global_trajectory_image():
            records = self._global_trajectory_records()
            target_counts = {
                str(k): len(v)
                for k, v in (records.get("targets", {}) or {}).items()
            }
            ready = len(records.get("ego", [])) >= 2 and any(c >= 2 for c in target_counts.values())
            return ready, {
                "source": self.vla_image_source,
                "history_s": float(self.vla_global_history_s),
                "ego_point_count": len(records.get("ego", [])),
                "target_point_counts": target_counts,
                "ready": bool(ready),
            }
        if self._use_bev_image():
            latest_ready = self._latest_pointcloud_available_for_capture()
            frame_count = int(self._active_pointcloud_frame_count())
            since_first_s = None
            try:
                first_t = self._active_pointcloud_first_recv_time()
                if first_t.to_sec() > 0.0:
                    since_first_s = max(0.0, (rospy.Time.now() - first_t).to_sec())
            except Exception:
                since_first_s = None
            frames_ready = frame_count >= int(self.vla_bev_warmup_min_frames)
            time_ready = (
                float(self.vla_bev_warmup_s) <= 0.0
                or (since_first_s is not None and since_first_s >= float(self.vla_bev_warmup_s))
            )
            ready = latest_ready and frames_ready and time_ready
            sensor_name = self._active_bev_sensor_name()
            diag = {
                "source": self.vla_image_source,
                "sensor": sensor_name,
                "required_frames": int(self.vla_bev_warmup_min_frames),
                "frame_count": frame_count,
                "since_first_pointcloud_s": None if since_first_s is None else round(since_first_s, 3),
                "required_warmup_s": float(self.vla_bev_warmup_s),
                "latest_ready": bool(latest_ready),
                "frames_ready": bool(frames_ready),
                "time_ready": bool(time_ready),
                "ready": bool(ready),
                "pointcloud_topic": self._active_bev_topic(),
                "max_age_s": self.vla_bev_cloud_max_age_s,
            }
            msg = self._active_pointcloud_msg()
            if msg is not None:
                try:
                    diag["pointcloud_age_s"] = round(
                        max(0.0, (rospy.Time.now() - self._active_pointcloud_stamp()).to_sec()), 3
                    )
                except Exception:
                    pass
                diag["pointcloud_width"] = int(getattr(msg, "width", 0) or 0)
                diag["pointcloud_height"] = int(getattr(msg, "height", 0) or 0)
            return ready, diag
        now = rospy.Time.now()
        self._prune_image_buffer(now)
        frames = list(getattr(self, "_image_buffer", []))
        if self._use_camera_video_input():
            required_frames = max(1, int(self.vla_camera_video_frame_count))
            video_frames = self._recent_camera_video_frame_records(required_frames)
            frame_count = len(video_frames)
        else:
            required_frames = min(30, max(1, int(self.vla_image_buffer_max_frames)))
            frame_count = len(frames)
        diag = {
            "source": self.vla_image_source,
            "video_enable": bool(self._use_camera_video_input()),
            "required_frames": required_frames,
            "frame_count": frame_count,
            "raw_buffer_frame_count": len(frames),
            "camera_video_interval_s": float(getattr(self, "vla_camera_video_interval_s", 0.0)),
            "camera_video_window_s": float(getattr(self, "vla_camera_video_window_s", 0.0)),
            "duration_s": self.vla_image_buffer_duration_s,
            "max_frames": self.vla_image_buffer_max_frames,
            "ready": False,
        }
        diag_frames = video_frames if self._use_camera_video_input() else frames
        if diag_frames:
            try:
                oldest = diag_frames[0].get("recv_time", rospy.Time(0))
                newest = diag_frames[-1].get("recv_time", rospy.Time(0))
                diag["span_s"] = round(max(0.0, (newest - oldest).to_sec()), 3)
                diag["newest_age_s"] = round(max(0.0, (now - newest).to_sec()), 3)
            except Exception:
                pass
        ready = frame_count >= required_frames
        diag["ready"] = bool(ready)
        return ready, diag

    def _target_history_sample_for_stamp(self, target_id, stamp_s):
        hist = self._target_histories.get(str(target_id), deque())
        if (not hist) and str(target_id) == "track_1":
            hist = self._state_target_history
        if not hist:
            return None
        best = None
        best_dt = float("inf")
        for sample in hist:
            try:
                dt = abs(float(sample.get("t", 0.0)) - float(stamp_s))
            except Exception:
                continue
            if dt < best_dt:
                best = sample
                best_dt = dt
        max_dt = max(0.5, min(1.0, self.vla_image_buffer_duration_s + 0.25))
        if best is None or best_dt > max_dt:
            return None
        return best

    def _bearing_center_score(self, bearing_deg):
        if bearing_deg is None:
            return None
        half_angle = max(
            1.0,
            0.5 * float(self.vla_camera_hfov_deg) - max(0.0, float(self.llm_fov_gate_margin_deg)),
        )
        return 1.0 - self._clamp(abs(float(bearing_deg)) / half_angle, 0.0, 1.0)

    def _frame_target_center_score(self, frame, targets):
        if not isinstance(targets, list) or not targets:
            return None, 0, False
        try:
            stamp_s = frame.get("stamp", rospy.Time(0)).to_sec()
        except Exception:
            stamp_s = 0.0
        weighted = 0.0
        weight_sum = 0.0
        used = 0
        frame_specific = False
        for target in targets:
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("target_id", ""))
            bearing = self._safe_float(target.get("relative_angle_deg"), None)
            distance = self._safe_float(target.get("distance_m"), None)
            sample = self._target_history_sample_for_stamp(target_id, stamp_s)
            if sample is not None:
                bearing = self._safe_float(sample.get("bearing_deg"), bearing)
                distance = self._safe_float(sample.get("range_m"), distance)
                frame_specific = True
            score = self._bearing_center_score(bearing)
            if score is None:
                x_hint = self._safe_float(target.get("image_x_hint_01"), None)
                if x_hint is not None:
                    score = 1.0 - self._clamp(abs(float(x_hint) - 0.5) / 0.5, 0.0, 1.0)
            if score is None:
                continue
            weight = 1.0
            if distance is not None and distance > 0.0:
                # Nearer vessels matter more for the selected VLM frame.
                weight = 1.0 / max(0.25, math.sqrt(float(distance)))
            weighted += float(score) * weight
            weight_sum += weight
            used += 1
        if weight_sum <= 0.0:
            return None, 0, False
        return self._clamp(weighted / weight_sum, 0.0, 1.0), used, frame_specific

    def _score_image_buffer_frames(self, fov_gate_diag=None):
        now = rospy.Time.now()
        self._prune_image_buffer(now)
        frames = list(getattr(self, "_image_buffer", []))
        if not frames and self._latest_image_msg is not None:
            frames = [
                {
                    "stamp": self._latest_image_stamp,
                    "recv_time": now,
                    "msg": self._latest_image_msg,
                    "fallback_latest": True,
                }
            ]

        targets = []
        if isinstance(fov_gate_diag, dict):
            targets = fov_gate_diag.get("prompt_targets") or fov_gate_diag.get("targets") or []
        duration = max(1e-3, float(self.vla_image_buffer_duration_s or self.vla_image_max_age_s or 1.0))
        prelim = []
        any_frame_specific_center = False
        for idx, frame in enumerate(frames):
            stamp = frame.get("stamp", rospy.Time(0))
            recv_time = frame.get("recv_time", rospy.Time(0))
            try:
                stamp_age_s = max(0.0, (now - stamp).to_sec())
            except Exception:
                stamp_age_s = 0.0
            try:
                recv_age_s = max(0.0, (now - recv_time).to_sec())
            except Exception:
                recv_age_s = stamp_age_s
            age_score = 1.0 - self._clamp(recv_age_s / duration, 0.0, 1.0)
            center_score, center_used, frame_specific = self._frame_target_center_score(frame, targets)
            any_frame_specific_center = any_frame_specific_center or frame_specific
            prelim.append(
                {
                    "buffer_index": idx,
                    "stamp_s": round(stamp.to_sec(), 6) if hasattr(stamp, "to_sec") else 0.0,
                    "stamp_age_s": round(stamp_age_s, 3),
                    "recv_age_s": round(recv_age_s, 3),
                    "age_score": round(age_score, 4),
                    "center_score": None if center_score is None else round(center_score, 4),
                    "center_targets_used": int(center_used),
                    "center_score_frame_specific": bool(frame_specific),
                    "fallback_latest": bool(frame.get("fallback_latest", False)),
                    "_msg": frame.get("msg"),
                }
            )

        records = []
        for rec in prelim:
            center_score = rec.get("center_score")
            if any_frame_specific_center and center_score is not None:
                score = 0.45 * float(rec["age_score"]) + 0.55 * float(center_score)
            else:
                score = float(rec["age_score"])
            clean = dict(rec)
            clean.pop("_msg", None)
            clean["score"] = round(self._clamp(score, 0.0, 1.0), 4)
            records.append(clean)
        return prelim, records

    def _select_vlm_image_frame(self, fov_gate_diag=None):
        prelim, records = self._score_image_buffer_frames(fov_gate_diag=fov_gate_diag)
        if not prelim:
            return {"msg": None, "score_records": [], "selected_record": {}}
        best_i = max(range(len(records)), key=lambda i: float(records[i].get("score", 0.0)))
        selected_record = dict(records[best_i])
        selected_record["selected"] = True
        clean_records = []
        for i, rec in enumerate(records):
            out = dict(rec)
            out["selected"] = i == best_i
            clean_records.append(out)
        return {
            "msg": prelim[best_i].get("_msg"),
            "score_records": clean_records,
            "selected_record": selected_record,
        }

    def _snapshot_trial_token(self):
        raw = str(getattr(self, "vla_trial_index", "") or "").strip()
        if not raw and getattr(self, "llm_io_log_path", ""):
            m = re.search(r"vla_trial_([0-9]+)(?:_io)?", os.path.basename(self.llm_io_log_path))
            if m:
                raw = m.group(1)
        if raw:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
            if safe:
                if safe.isdigit():
                    return "trial_%s" % safe
                if safe.lower().startswith("trial_"):
                    return safe
                return "trial_%s" % safe
        return "call_%d" % max(1, int(getattr(self, "_llm_call_seq", 0) or 1))

    def _next_vlm_call_index_for_snapshot(self):
        try:
            return max(1, int(getattr(self, "_llm_call_seq", 0) or 0) + 1)
        except Exception:
            return 1

    def _unique_snapshot_filename(self, filename):
        base = os.path.join(self.vla_snapshot_dir, str(filename))
        if not os.path.exists(base):
            return base
        stem, ext = os.path.splitext(base)
        for i in range(2, 10000):
            candidate = "%s_%d%s" % (stem, i, ext or ".png")
            if not os.path.exists(candidate):
                return candidate
        return "%s_%d%s" % (stem, int(time.time() * 1000), ext or ".png")

    def _save_camera_video_frame_records(self, records, snapshot_token):
        os.makedirs(self.vla_snapshot_dir, exist_ok=True)
        self._camera_video_seq += 1
        saved = []
        for idx, rec in enumerate(records):
            msg = rec.get("msg") if isinstance(rec, dict) else None
            if msg is None:
                continue
            png = self._image_msg_to_png_bytes(msg)
            if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                raise ValueError("vla_camera_frame_too_large:%d" % len(png))
            path = self._unique_snapshot_filename(
                "camera_video_%s_seq_%03d_frame_%02d.png"
                % (snapshot_token, int(self._camera_video_seq), idx + 1)
            )
            with open(path, "wb") as f:
                f.write(png)
            stamp = rec.get("stamp", rospy.Time(0))
            recv_time = rec.get("recv_time", rospy.Time(0))
            saved.append(
                {
                    "path": path,
                    "stamp_s": round(stamp.to_sec(), 6) if hasattr(stamp, "to_sec") else 0.0,
                    "record_time_s": round(recv_time.to_sec(), 6) if hasattr(recv_time, "to_sec") else 0.0,
                    "frame_index": idx,
                    "image_bytes": len(png),
                    "source": "camera",
                }
            )
        return saved

    def _save_trigger_camera_video_snapshot(self, fov_gate_diag=None):
        required = int(self.vla_camera_video_frame_count)
        records = self._recent_camera_video_frame_records(required)
        if len(records) < required:
            self._append_llm_io_record(
                {
                    "event": "snapshot_missing",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": "camera_video",
                    "error": "camera_video_frame_count_insufficient",
                    "required_video_frame_count": required,
                    "available_frame_count": len(records),
                    "camera_video_interval_s": float(self.vla_camera_video_interval_s),
                    "camera_video_window_s": float(self.vla_camera_video_window_s),
                    "image_diag": self._vla_image_diag(),
                }
            )
            self._publish_vlm_check(
                "trigger_camera_video_snapshot_missing",
                {
                    "error": "camera_video_frame_count_insufficient",
                    "required_video_frame_count": required,
                    "available_frame_count": len(records),
                    "camera_video_interval_s": float(self.vla_camera_video_interval_s),
                    "camera_video_window_s": float(self.vla_camera_video_window_s),
                },
            )
            return ""
        try:
            call_index = self._next_vlm_call_index_for_snapshot()
            snapshot_token = "%s_call_%03d" % (self._snapshot_trial_token(), call_index)
            saved = self._save_camera_video_frame_records(records, snapshot_token)
            paths = [
                str(rec.get("path"))
                for rec in saved
                if isinstance(rec, dict) and rec.get("path")
            ]
            if len(paths) < required:
                raise ValueError("camera_video_saved_frame_count_insufficient")
            self._event_camera_video_frame_records = saved
            self._event_camera_video_frame_paths = paths
            self._event_image_path = paths[-1]
            self._event_image_url = "file://" + paths[-1]
            self._event_input_image_path = ""
            self._event_input_image_url = ""
            self._last_vlm_score_plot_path = ""
            self._last_vlm_frame_score_records = []
            self._last_vlm_snapshot_annotated = False
            self._last_vlm_snapshot_annotation_error = ""
            self._append_llm_io_record(
                {
                    "event": "snapshot_saved",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": "camera_video",
                    "image_path": paths[-1],
                    "image_url": self._event_image_url,
                    "input_video_frame_paths": paths,
                    "input_video_frame_count": len(paths),
                    "input_video_frame_records": saved,
                    "snapshot_token": snapshot_token,
                    "planned_llm_call_index": int(call_index),
                    "image_diag": self._vla_image_diag(),
                    "fov_gate_diag": fov_gate_diag if isinstance(fov_gate_diag, dict) else {},
                }
            )
            self._publish_vlm_check(
                "trigger_camera_video_snapshot_saved",
                {
                    "image_path": paths[-1],
                    "image_url": self._event_image_url,
                    "input_video_frame_paths": paths,
                    "input_video_frame_count": len(paths),
                    "snapshot_token": snapshot_token,
                    "planned_llm_call_index": int(call_index),
                },
            )
            return paths[-1]
        except Exception as e:
            self._append_llm_io_record(
                {
                    "event": "snapshot_failed",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": "camera_video",
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "image_diag": self._vla_image_diag(),
                }
            )
            self._publish_vlm_check(
                "trigger_camera_video_snapshot_failed",
                {"error": str(e)[:300], "image_diag": self._vla_image_diag()},
            )
            return ""

    def _save_vlm_score_plot(self, records, selected_buffer_index, snapshot_token):
        if not records:
            return ""
        try:
            os.makedirs(self.vla_snapshot_dir, exist_ok=True)
            recent = list(records)[-30:]
            path = os.path.join(self.vla_snapshot_dir, "vlm_score_%s.png" % str(snapshot_token))
            width, height = 1000, 430
            left, right, top, bottom = 60, 970, 28, 285
            img = PilImage.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width - 1, height - 1], outline=(210, 210, 210))
            for yv in (0.0, 0.5, 1.0):
                y = bottom - int((bottom - top) * yv)
                draw.line([left, y, right, y], fill=(225, 225, 225), width=1)
                draw.text((8, y - 7), "%.1f" % yv, fill=(70, 70, 70))
            draw.line([left, top, left, bottom], fill=(80, 80, 80), width=1)
            draw.line([left, bottom, right, bottom], fill=(80, 80, 80), width=1)
            draw.text((left, 8), "VLM frame score curve, recent %d frames" % len(recent), fill=(20, 20, 20))

            points = []
            n = len(recent)
            for i, rec in enumerate(recent):
                x = int((left + right) / 2) if n <= 1 else left + int((right - left) * i / float(n - 1))
                score = self._clamp(float(rec.get("score", 0.0)), 0.0, 1.0)
                y = bottom - int((bottom - top) * score)
                points.append((x, y, rec))
            for i in range(1, len(points)):
                draw.line([points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]], fill=(35, 115, 210), width=2)
            for x, y, rec in points:
                selected = rec.get("buffer_index") == selected_buffer_index
                color = (220, 45, 45) if selected else (35, 115, 210)
                r = 5 if selected else 3
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)

            score_items = [
                "%02d:%0.2f%s" % (
                    i,
                    float(rec.get("score", 0.0)),
                    "*" if rec.get("buffer_index") == selected_buffer_index else "",
                )
                for i, rec in enumerate(recent)
            ]
            draw.text((left, 305), "scores (* selected):", fill=(20, 20, 20))
            line = ""
            y = 328
            for item in score_items:
                piece = item + "  "
                if len(line) + len(piece) > 120:
                    draw.text((left, y), line, fill=(40, 40, 40))
                    y += 18
                    line = piece
                else:
                    line += piece
            if line:
                draw.text((left, y), line, fill=(40, 40, 40))
            img.save(path)
            return path
        except Exception as e:
            self._append_llm_io_record(
                {
                    "event": "vlm_score_plot_failed",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "error": str(e),
                    "exception_type": type(e).__name__,
                }
            )
            return ""

    def _annotation_visible_targets(self, fov_gate_diag):
        if not isinstance(fov_gate_diag, dict):
            return []
        targets = fov_gate_diag.get("targets") or []
        visible = []
        if isinstance(targets, list):
            for t in targets:
                if isinstance(t, dict) and bool(t.get("in_fov", False)):
                    visible.append(t)
        if visible:
            return visible
        prompt_targets = fov_gate_diag.get("prompt_targets") or []
        if isinstance(prompt_targets, list):
            return [t for t in prompt_targets if isinstance(t, dict)]
        return []

    def _image_msg_to_cv_bgr(self, msg):
        if cv2 is None or np is None:
            raise RuntimeError("opencv_unavailable:%s" % (CV2_IMPORT_ERROR or "unknown"))
        enc = str(msg.encoding or "").lower()
        w = int(msg.width)
        h = int(msg.height)
        step = int(msg.step) if int(msg.step) > 0 else 0
        if w <= 0 or h <= 0 or step <= 0:
            raise ValueError("vla_image_invalid_size_or_step")
        raw = np.frombuffer(bytes(msg.data or b""), dtype=np.uint8)
        if raw.size < step * h:
            raise ValueError("vla_image_data_too_short")

        if enc == "mono8":
            row_bytes = w
            if step < row_bytes:
                raise ValueError("vla_image_step_too_small")
            gray = raw[: step * h].reshape((h, step))[:, :row_bytes].copy()
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if enc in ("rgb8", "bgr8"):
            row_bytes = w * 3
            if step < row_bytes:
                raise ValueError("vla_image_step_too_small")
            arr = raw[: step * h].reshape((h, step))[:, :row_bytes].reshape((h, w, 3)).copy()
            if enc == "rgb8":
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        if enc in ("rgba8", "bgra8"):
            row_bytes = w * 4
            if step < row_bytes:
                raise ValueError("vla_image_step_too_small")
            arr = raw[: step * h].reshape((h, step))[:, :row_bytes].reshape((h, w, 4)).copy()
            if enc == "rgba8":
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        raise ValueError("vla_image_unsupported_encoding:%s" % enc)

    def _draw_label_box(self, img, x, y, lines, color):
        h, w = img.shape[:2]
        x = int(self._clamp(x, 4, max(4, w - 210)))
        y = int(self._clamp(y, 4, max(4, h - 56)))
        line_h = 18
        box_w = 205
        box_h = max(28, 8 + line_h * len(lines))
        x2 = min(w - 4, x + box_w)
        y2 = min(h - 4, y + box_h)
        cv2.rectangle(img, (x, y), (x2, y2), color, 2)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(img, (x, y), (x2, y2), color, 2)
        for i, text in enumerate(lines):
            cv2.putText(
                img,
                str(text)[:32],
                (x + 6, y + 18 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        return x, y, x2, y2

    def _annotate_vlm_image_bgr(self, img, fov_gate_diag=None, selected_record=None):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        stamp_age = ""
        if isinstance(selected_record, dict) and selected_record.get("stamp_age_s") is not None:
            stamp_age = " age=%.2fs" % float(selected_record.get("stamp_age_s"))
        cv2.putText(
            img,
            "VLM event%s" % stamp_age,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        ego_heading = self._safe_float((self._state or {}).get("ego_heading_rad"), None)
        if ego_heading is not None:
            start = (w // 2, h - 34)
            end = (w // 2, max(20, h - 120))
            cv2.arrowedLine(img, start, end, (0, 255, 255), 3, tipLength=0.22)
            cv2.putText(
                img,
                "own hdg %.1f deg" % math.degrees(ego_heading),
                (max(8, w // 2 - 95), h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        targets = self._annotation_visible_targets(fov_gate_diag)
        color_cycle = [(0, 255, 0), (255, 200, 0), (0, 180, 255), (255, 0, 255), (180, 255, 180)]
        for i, t in enumerate(targets[:12]):
            x_hint = self._safe_float(t.get("image_x_hint_01"), None)
            bearing = self._safe_float(t.get("relative_angle_deg"), None)
            distance = self._safe_float(t.get("distance_m"), None)
            if x_hint is None and bearing is not None:
                x_hint = self._image_x_hint_01(bearing)
            x = int(self._clamp(float(x_hint if x_hint is not None else 0.5), 0.0, 1.0) * (w - 1))
            y_anchor = int(h * 0.34)
            color = color_cycle[i % len(color_cycle)]
            cv2.line(img, (x, 44), (x, h - 44), color, 1)
            cv2.circle(img, (x, y_anchor), 9, color, 2)
            label_lines = [str(t.get("target_id", "track"))]
            detail = []
            if bearing is not None:
                detail.append("b=%.1fdeg" % bearing)
            if distance is not None:
                detail.append("d=%.1fm" % distance)
            if detail:
                label_lines.append(" ".join(detail))
            box_x = x + 12 if x < w * 0.65 else x - 210
            box_y = 56 + 48 * (i % 5)
            bx1, by1, bx2, by2 = self._draw_label_box(img, box_x, box_y, label_lines, color)
            cv2.arrowedLine(
                img,
                (bx1 if x < w * 0.65 else bx2, (by1 + by2) // 2),
                (x, y_anchor),
                color,
                1,
                tipLength=0.12,
            )
        return img

    def _latest_pointcloud_xyz(self):
        if np is None:
            raise RuntimeError("numpy_unavailable:%s" % (NP_IMPORT_ERROR or "unknown"))
        msg = self._active_pointcloud_msg()
        if msg is None:
            raise ValueError("pointcloud_unavailable")
        if not self._latest_pointcloud_available_for_capture():
            raise ValueError("pointcloud_stale")

        total = int(getattr(msg, "width", 0) or 0) * int(getattr(msg, "height", 0) or 0)
        sample_step = 1
        if total > int(self.vla_bev_max_points):
            sample_step = max(1, int(math.ceil(float(total) / float(self.vla_bev_max_points))))
        pts = []
        for i, p in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if sample_step > 1 and (i % sample_step) != 0:
                continue
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
            except Exception:
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if x < -float(self.vla_bev_range_backward_m) or x > float(self.vla_bev_range_forward_m):
                continue
            if abs(y) > self.vla_bev_range_side_m:
                continue
            if z < self.vla_bev_z_min_m or z > self.vla_bev_z_max_m:
                continue
            pts.append((x, y, z))
            if len(pts) >= int(self.vla_bev_max_points):
                break
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), {
                "sample_step": sample_step,
                "filtered_points": 0,
                "raw_points_est": total,
                "sensor": self._active_bev_sensor_name(),
                "topic": self._active_bev_topic(),
            }
        return np.asarray(pts, dtype=np.float32), {
            "sample_step": sample_step,
            "filtered_points": len(pts),
            "raw_points_est": total,
            "sensor": self._active_bev_sensor_name(),
            "topic": self._active_bev_topic(),
        }

    def _bev_render_targets(self, fov_gate_diag=None):
        targets = []
        if isinstance(fov_gate_diag, dict):
            targets = fov_gate_diag.get("prompt_targets") or fov_gate_diag.get("targets") or []
        if not isinstance(targets, list) or not targets:
            targets = self._build_targets_prompt_context()
        return [t for t in targets if isinstance(t, dict)]

    def _bev_target_size(self, target):
        sx = self._safe_float(target.get("size_x_m"), None)
        sy = self._safe_float(target.get("size_y_m"), None)
        size = target.get("size")
        if isinstance(size, dict):
            sx = sx if sx is not None else self._safe_float(size.get("x"), None)
            sy = sy if sy is not None else self._safe_float(size.get("y"), None)
        elif isinstance(size, (list, tuple)):
            if len(size) > 0:
                sx = sx if sx is not None else self._safe_float(size[0], None)
            if len(size) > 1:
                sy = sy if sy is not None else self._safe_float(size[1], None)
        sx = max(1.5, sx if sx is not None else 2.0)
        sy = max(1.5, sy if sy is not None else 2.0)
        margin = float(self.vla_bev_track_box_margin_m)
        return sx + 2.0 * margin, sy + 2.0 * margin

    def _render_pointcloud_bev_png_bytes(self, fov_gate_diag=None, debug_overlay=True):
        pts, pc_diag = self._latest_pointcloud_xyz()
        width = int(self.vla_bev_image_width)
        height = int(self.vla_bev_image_height)
        if debug_overlay:
            margin_l, margin_r, margin_t, margin_b = 72, 28, 48, 74
        else:
            margin_l, margin_r, margin_t, margin_b = 0, 0, 0, 0
        usable_w = max(100, width - margin_l - margin_r)
        usable_h = max(100, height - margin_t - margin_b)
        side = float(self.vla_bev_range_side_m)
        forward = float(self.vla_bev_range_forward_m)
        backward = float(self.vla_bev_range_backward_m)
        longitudinal_span = max(1e-6, forward + backward)
        scale = min(usable_w / max(1e-6, 2.0 * side), usable_h / longitudinal_span)
        center_u = margin_l + 0.5 * usable_w
        center_v = margin_t + 0.5 * usable_h
        top_v = center_v - forward * scale
        bottom_v = center_v + backward * scale
        left_u = center_u - side * scale
        right_u = center_u + side * scale

        def to_px(x_m, y_m):
            u = center_u - float(y_m) * scale
            v = center_v - float(x_m) * scale
            return int(round(u)), int(round(v))

        bg_color = (248, 250, 252) if debug_overlay else (0, 0, 0)
        img = PilImage.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        if debug_overlay:
            draw.rectangle([left_u, top_v, right_u, bottom_v], fill=(255, 255, 255), outline=(120, 130, 145))

            x_start = int(math.ceil(-backward / 5.0) * 5)
            x_stop = int(math.floor(forward / 5.0) * 5)
            for x_m in range(x_start, x_stop + 1, 5):
                _, v = to_px(x_m, 0.0)
                color = (205, 211, 220) if x_m % 10 else (170, 180, 192)
                draw.line([left_u, v, right_u, v], fill=color, width=1)
            lateral_max_i = int(math.floor(side))
            for y_m in range(-lateral_max_i, lateral_max_i + 1, 5):
                u, _ = to_px(0.0, y_m)
                color = (205, 211, 220) if y_m % 10 else (170, 180, 192)
                draw.line([u, top_v, u, bottom_v], fill=color, width=1)
        u0, v0 = to_px(0.0, 0.0)
        if debug_overlay:
            draw.line([u0, top_v, u0, bottom_v], fill=(42, 65, 92), width=2)

        if pts.shape[0] > 0:
            us = np.rint(center_u - pts[:, 1] * scale).astype(np.int32)
            vs = np.rint(center_v - pts[:, 0] * scale).astype(np.int32)
            point_radius_px = 2 if debug_overlay else 3
            mask = (
                (us >= point_radius_px)
                & (us < width - point_radius_px)
                & (vs >= point_radius_px)
                & (vs < height - point_radius_px)
            )
            us = us[mask]
            vs = vs[mask]
            zs = pts[:, 2][mask]
            arr = np.asarray(img).copy()
            denom = max(1e-6, float(self.vla_bev_z_max_m - self.vla_bev_z_min_m))
            zn = np.clip((zs - float(self.vla_bev_z_min_m)) / denom, 0.0, 1.0)
            colors = np.stack(
                [
                    (58 + 70 * zn).astype(np.uint8) if debug_overlay else (20 + 20 * zn).astype(np.uint8),
                    (96 + 70 * zn).astype(np.uint8) if debug_overlay else (190 + 65 * zn).astype(np.uint8),
                    (145 + 65 * zn).astype(np.uint8) if debug_overlay else (40 + 20 * zn).astype(np.uint8),
                ],
                axis=1,
            )
            for du in range(-point_radius_px, point_radius_px + 1):
                for dv in range(-point_radius_px, point_radius_px + 1):
                    if (du * du + dv * dv) > (point_radius_px * point_radius_px + 1):
                        continue
                    uu = us + du
                    vv = vs + dv
                    if debug_overlay:
                        arr[vv, uu] = np.minimum(arr[vv, uu], colors)
                    else:
                        arr[vv, uu] = np.maximum(arr[vv, uu], colors)
            img = PilImage.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)

        if debug_overlay:
            danger_radius_m = min(30.0, forward)
            danger_bbox = [
                center_u - danger_radius_m * scale,
                center_v - danger_radius_m * scale,
                center_u + danger_radius_m * scale,
                center_v + danger_radius_m * scale,
            ]
            draw.arc(danger_bbox, start=180, end=360, fill=(220, 38, 38), width=4)

        ego_radius_px = 7 if debug_overlay else 9
        draw.ellipse(
            [u0 - ego_radius_px - 1, v0 - ego_radius_px - 1, u0 + ego_radius_px + 1, v0 + ego_radius_px + 1],
            fill=(255, 240, 240) if not debug_overlay else (127, 29, 29),
        )
        draw.ellipse(
            [u0 - ego_radius_px, v0 - ego_radius_px, u0 + ego_radius_px, v0 + ego_radius_px],
            fill=(230, 36, 36),
        )
        rendered_targets = 0

        age_s = None
        try:
            age_s = max(0.0, (rospy.Time.now() - self._active_pointcloud_stamp()).to_sec())
        except Exception:
            pass

        crop_box = (
            max(0, int(math.floor(left_u))),
            max(0, int(math.floor(top_v))),
            min(width, int(math.ceil(right_u)) + 1),
            min(height, int(math.ceil(bottom_v)) + 1),
        )
        img = img.crop(crop_box)
        final_width, final_height = img.size

        out = io.BytesIO()
        img.save(out, format="PNG")
        diag = dict(pc_diag)
        diag.update(
            {
                "image_width": final_width,
                "image_height": final_height,
                "configured_image_width": width,
                "configured_image_height": height,
                "range_forward_m": forward,
                "range_backward_m": backward,
                "range_side_m": side,
                "ego_marker": "red_center_dot",
                "rendered_targets": rendered_targets,
                "pointcloud_age_s": age_s,
                "debug_overlay": bool(debug_overlay),
            }
        )
        return out.getvalue(), diag

    def _save_trigger_bev_image(self, fov_gate_diag=None):
        try:
            os.makedirs(self.vla_snapshot_dir, exist_ok=True)
            t0 = time.perf_counter()
            self._bev_snapshot_seq += 1
            call_index = self._next_vlm_call_index_for_snapshot()
            snapshot_token = "%s_call_%03d" % (self._snapshot_trial_token(), call_index)
            bev_video_frame_paths = []
            input_path = ""
            input_png_len = 0
            input_bev_diag = {}
            snapshot_source = "pointcloud_bev_clean"
            if self.vla_bev_video_enable:
                bev_video_records = self._recent_bev_video_frame_records()
                bev_video_frame_paths = [
                    str(rec.get("path"))
                    for rec in bev_video_records
                    if isinstance(rec, dict) and rec.get("path")
                ]
                if len(bev_video_frame_paths) < int(self.vla_bev_video_frame_count):
                    self._append_llm_io_record(
                        {
                            "event": "snapshot_missing",
                            "wall_time_s": time.time(),
                            "ros_time_s": rospy.Time.now().to_sec(),
                            "snapshot_source": source_prefix + "_video",
                            "error": "bev_video_frame_count_insufficient",
                            "required_video_frame_count": int(self.vla_bev_video_frame_count),
                            "input_video_frame_paths": bev_video_frame_paths,
                            "image_diag": self._vla_image_diag(),
                        }
                    )
                    self._publish_vlm_check(
                        "trigger_bev_snapshot_missing",
                        {
                            "error": "bev_video_frame_count_insufficient",
                            "required_video_frame_count": int(self.vla_bev_video_frame_count),
                            "input_video_frame_paths": bev_video_frame_paths,
                        },
                    )
                    return ""
                latest_record = bev_video_records[-1]
                input_path = str(latest_record.get("path") or "")
                input_bev_diag = dict(latest_record.get("diag") or {})
                input_png_len = int(os.path.getsize(input_path)) if input_path and os.path.isfile(input_path) else 0
                snapshot_source = source_prefix + "_video_4frames"
            else:
                input_png, input_bev_diag = self._render_pointcloud_bev_png_bytes(
                    fov_gate_diag=fov_gate_diag,
                    debug_overlay=False,
                )
                input_path = self._unique_snapshot_filename("bev_input_%s.png" % snapshot_token)
                with open(input_path, "wb") as f:
                    f.write(input_png)
                input_png_len = len(input_png)
            self._event_image_path = input_path
            self._event_image_url = "file://" + input_path
            self._event_input_image_path = "" if self.vla_bev_video_enable else input_path
            self._event_input_image_url = "" if self.vla_bev_video_enable else "file://" + input_path
            self._last_vlm_score_plot_path = ""
            self._last_vlm_frame_score_records = []
            self._last_vlm_snapshot_annotated = True
            self._last_vlm_snapshot_annotation_error = ""
            input_bev_diag["render_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            input_bev_diag["snapshot_seq"] = int(self._bev_snapshot_seq)
            input_bev_diag["planned_llm_call_index"] = int(call_index)
            self._append_llm_io_record(
                {
                    "event": "snapshot_saved",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": snapshot_source,
                    "image_path": input_path,
                    "image_url": self._event_image_url,
                    "image_bytes": input_png_len,
                    "input_image_path": self._event_input_image_path,
                    "input_image_url": self._event_input_image_url,
                    "input_image_bytes": 0 if self.vla_bev_video_enable else input_png_len,
                    "input_video_frame_paths": bev_video_frame_paths,
                    "input_video_frame_count": len(bev_video_frame_paths),
                    "bev_video_frame_paths": bev_video_frame_paths,
                    "snapshot_token": snapshot_token,
                    "snapshot_seq": int(self._bev_snapshot_seq),
                    "planned_llm_call_index": int(call_index),
                    "bev_diag": input_bev_diag,
                    "input_bev_diag": input_bev_diag,
                    "image_diag": self._vla_image_diag(),
                }
            )
            self._publish_vlm_check(
                "trigger_bev_snapshot_saved",
                {
                    "image_path": input_path,
                    "image_url": self._event_image_url,
                    "image_bytes": input_png_len,
                    "input_image_path": self._event_input_image_path,
                    "input_image_url": self._event_input_image_url,
                    "input_image_bytes": 0 if self.vla_bev_video_enable else input_png_len,
                    "input_video_frame_paths": bev_video_frame_paths,
                    "input_video_frame_count": len(bev_video_frame_paths),
                    "bev_video_frame_paths": bev_video_frame_paths,
                    "snapshot_token": snapshot_token,
                    "snapshot_seq": int(self._bev_snapshot_seq),
                    "planned_llm_call_index": int(call_index),
                    "bev_diag": input_bev_diag,
                    "input_bev_diag": input_bev_diag,
                },
            )
            return input_path
        except Exception as e:
            self._append_llm_io_record(
                {
                    "event": "snapshot_failed",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": source_prefix,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "image_diag": self._vla_image_diag(),
                }
            )
            self._publish_vlm_check(
                "trigger_bev_snapshot_failed",
                {"error": str(e)[:300], "image_diag": self._vla_image_diag()},
            )
            return ""

    def _annotated_snapshot_png_bytes(self, msg, fov_gate_diag=None, selected_record=None):
        if not self.vla_annotated_snapshot_enable:
            return self._image_msg_to_png_bytes(msg), False, ""
        try:
            img = self._image_msg_to_cv_bgr(msg)
            img = self._annotate_vlm_image_bgr(img, fov_gate_diag=fov_gate_diag, selected_record=selected_record)
            ok, enc = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("opencv_png_encode_failed")
            return enc.tobytes(), True, ""
        except Exception as e:
            return self._image_msg_to_png_bytes(msg), False, str(e)

    def _render_global_trajectory_png_bytes(self):
        records = self._global_trajectory_records()
        ego_pts = records.get("ego", []) or []
        target_map = records.get("targets", {}) or {}
        if len(ego_pts) < 2 or not any(len(v) >= 2 for v in target_map.values()):
            raise ValueError("global_trajectory_history_insufficient")

        width = int(self.vla_global_image_width)
        height = int(self.vla_global_image_height)
        margin_l, margin_r, margin_t, margin_b = 24, 24, 24, 24
        usable_w = max(100, width - margin_l - margin_r)
        usable_h = max(100, height - margin_t - margin_b)
        all_xy = [(float(rec["x"]), float(rec["y"])) for rec in ego_pts]
        for pts in target_map.values():
            all_xy.extend((float(rec["world_x"]), float(rec["world_y"])) for rec in pts)
        goal_xy = None
        if self.vla_goal_x is not None and self.vla_goal_y is not None:
            goal_xy = (float(self.vla_goal_x), float(self.vla_goal_y))
            all_xy.append(goal_xy)
        xs = [p[0] for p in all_xy]
        ys = [p[1] for p in all_xy]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(float(self.vla_global_min_span_m), max_x - min_x)
        span_y = max(float(self.vla_global_min_span_m), max_y - min_y)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        scale = min(usable_w / max(1e-6, span_x), usable_h / max(1e-6, span_y))
        world_w = usable_w / max(1e-6, scale)
        world_h = usable_h / max(1e-6, scale)
        min_x = cx - 0.5 * world_w
        max_x = cx + 0.5 * world_w
        min_y = cy - 0.5 * world_h
        max_y = cy + 0.5 * world_h

        def to_px(x, y):
            u = margin_l + (float(x) - min_x) * scale
            v = margin_t + (max_y - float(y)) * scale
            return int(round(u)), int(round(v))

        img = PilImage.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        def draw_star(x, y, outer_r=16, inner_r=7, color=(255, 36, 36)):
            pts = []
            for i in range(10):
                ang = -math.pi / 2.0 + i * math.pi / 5.0
                r = outer_r if i % 2 == 0 else inner_r
                pts.append((int(round(x + r * math.cos(ang))), int(round(y + r * math.sin(ang)))))
            draw.polygon(pts, fill=color, outline=color)

        def draw_polyline(points, color, width_px):
            if len(points) < 2:
                return
            pix = [to_px(p[0], p[1]) for p in points]
            for i in range(1, len(pix)):
                draw.line([pix[i - 1][0], pix[i - 1][1], pix[i][0], pix[i][1]], fill=color, width=width_px)
            start = pix[0]
            end = pix[-1]
            draw.ellipse([start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4], outline=color, width=2)
            draw.ellipse([end[0] - 7, end[1] - 7, end[0] + 7, end[1] + 7], fill=color, outline=color)
            if len(pix) >= 2:
                dx = end[0] - pix[-2][0]
                dy = end[1] - pix[-2][1]
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    ux, uy = dx / norm, dy / norm
                    tip = (end[0] + int(12 * ux), end[1] + int(12 * uy))
                    left = (end[0] - int(7 * ux - 5 * uy), end[1] - int(7 * uy + 5 * ux))
                    right = (end[0] - int(7 * ux + 5 * uy), end[1] - int(7 * uy - 5 * ux))
                    draw.polygon([tip, left, right], fill=color)

        draw_polyline([(float(rec["x"]), float(rec["y"])) for rec in ego_pts], (255, 36, 36), 6)
        colors = [(56, 189, 248), (52, 211, 153), (196, 181, 253), (251, 146, 60), (45, 212, 191), (244, 114, 182)]
        for idx, (tid, pts) in enumerate(sorted(target_map.items())):
            if len(pts) < 2:
                continue
            draw_polyline(
                [(float(rec["world_x"]), float(rec["world_y"])) for rec in pts],
                colors[idx % len(colors)],
                4,
            )
        if goal_xy is not None:
            gu, gv = to_px(goal_xy[0], goal_xy[1])
            draw_star(gu, gv)

        out = io.BytesIO()
        img.save(out, format="PNG")
        diag = {
            "history_s": float(self.vla_global_history_s),
            "ego_points": len(ego_pts),
            "target_points": {str(k): len(v) for k, v in target_map.items()},
            "goal_xy": None if goal_xy is None else [round(goal_xy[0], 3), round(goal_xy[1], 3)],
            "world_bounds": {
                "min_x": round(min_x, 3),
                "max_x": round(max_x, 3),
                "min_y": round(min_y, 3),
                "max_y": round(max_y, 3),
            },
        }
        return out.getvalue(), diag

    def _save_trigger_global_trajectory_image(self, fov_gate_diag=None):
        try:
            os.makedirs(self.vla_snapshot_dir, exist_ok=True)
            png, diag = self._render_global_trajectory_png_bytes()
            if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                raise ValueError("vla_global_image_too_large:%d" % len(png))
            snapshot_token = self._snapshot_trial_token()
            source_prefix = "ais_traj" if self._use_ais_trajectory_image() else "global_traj"
            path = self._unique_snapshot_filename("%s_input_%s.png" % (source_prefix, snapshot_token))
            with open(path, "wb") as f:
                f.write(png)
            self._event_image_path = path
            self._event_image_url = "file://" + path
            self._last_vlm_score_plot_path = ""
            self._last_vlm_frame_score_records = []
            self._last_vlm_snapshot_annotated = False
            self._last_vlm_snapshot_annotation_error = ""
            self._append_llm_io_record(
                {
                    "event": "trigger_global_trajectory_snapshot_saved",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": "ais_trajectory" if self._use_ais_trajectory_image() else "global_trajectory",
                    "event_image_path": path,
                    "input_image_bytes": len(png),
                    "global_trajectory_diag": diag,
                    "fov_gate_diag": fov_gate_diag or {},
                }
            )
            self._publish_vlm_check(
                "trigger_global_trajectory_snapshot_saved",
                {"event_image_path": path, "input_image_bytes": len(png), "global_trajectory_diag": diag},
            )
            return path
        except Exception as e:
            self._append_llm_io_record(
                {
                    "event": "trigger_global_trajectory_snapshot_failed",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "snapshot_source": "ais_trajectory" if self._use_ais_trajectory_image() else "global_trajectory",
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "image_diag": self._vla_image_diag(),
                }
            )
            self._publish_vlm_check(
                "trigger_global_trajectory_snapshot_failed",
                {"error": str(e)[:300], "image_diag": self._vla_image_diag()},
            )
            return ""

    def _save_trigger_snapshot_image(self, fov_gate_diag=None):
        if self._use_global_trajectory_image():
            return self._save_trigger_global_trajectory_image(fov_gate_diag=fov_gate_diag)
        if self._use_bev_image():
            path = self._save_trigger_bev_image(fov_gate_diag=fov_gate_diag)
            if path or not self.vla_bev_fallback_to_camera:
                return path
        if self._use_camera_video_input():
            return self._save_trigger_camera_video_snapshot(fov_gate_diag=fov_gate_diag)
        selected = self._select_vlm_image_frame(fov_gate_diag=fov_gate_diag)
        msg = selected.get("msg")
        frame_score_records = selected.get("score_records", [])
        selected_record = selected.get("selected_record", {})
        score_plot_path = ""
        if msg is None:
            self._append_llm_io_record(
                {
                    "event": "snapshot_missing",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "error": "no_latest_ros_image",
                    "image_diag": self._vla_image_diag(),
                    "frame_score_records": frame_score_records,
                }
            )
            self._publish_vlm_check(
                "trigger_snapshot_missing",
                {"error": "no_latest_ros_image"},
            )
            return ""
        try:
            os.makedirs(self.vla_snapshot_dir, exist_ok=True)
            png, annotated, annotation_error = self._annotated_snapshot_png_bytes(
                msg,
                fov_gate_diag=fov_gate_diag,
                selected_record=selected_record,
            )
            snapshot_token = self._snapshot_trial_token()
            path = self._unique_snapshot_filename("camera_snapshot_%s.png" % snapshot_token)
            with open(path, "wb") as f:
                f.write(png)
            score_plot_path = self._save_vlm_score_plot(
                frame_score_records,
                selected_record.get("buffer_index"),
                snapshot_token,
            )
            self._event_image_path = path
            self._event_image_url = "file://" + path
            self._event_input_image_path = ""
            self._event_input_image_url = ""
            self._last_vlm_score_plot_path = score_plot_path
            self._last_vlm_frame_score_records = frame_score_records[-30:]
            self._last_vlm_snapshot_annotated = bool(annotated)
            self._last_vlm_snapshot_annotation_error = annotation_error
            self._append_llm_io_record(
                {
                    "event": "snapshot_saved",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "image_path": path,
                    "image_url": self._event_image_url,
                    "image_bytes": len(png),
                    "snapshot_token": snapshot_token,
                    "selected_frame": selected_record,
                    "annotated": bool(annotated),
                    "annotation_error": annotation_error,
                    "score_plot_path": score_plot_path,
                    "frame_score_records": frame_score_records[-30:],
                    "image_diag": self._vla_image_diag(),
                }
            )
            self._publish_vlm_check(
                "trigger_snapshot_saved",
                {
                    "image_path": path,
                    "image_url": self._event_image_url,
                    "image_bytes": len(png),
                    "snapshot_token": snapshot_token,
                    "annotated": bool(annotated),
                    "annotation_error": annotation_error[:200] if annotation_error else "",
                    "selected_frame_score": selected_record.get("score"),
                    "selected_frame_age_s": selected_record.get("stamp_age_s"),
                    "score_plot_path": score_plot_path,
                },
            )
            return path
        except Exception as e:
            self._append_llm_io_record(
                {
                    "event": "snapshot_failed",
                    "wall_time_s": time.time(),
                    "ros_time_s": rospy.Time.now().to_sec(),
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "image_diag": self._vla_image_diag(),
                    "selected_frame": selected_record,
                    "annotated": bool(self._last_vlm_snapshot_annotated),
                    "annotation_error": self._last_vlm_snapshot_annotation_error,
                    "score_plot_path": score_plot_path,
                    "frame_score_records": frame_score_records[-30:],
                }
            )
            self._publish_vlm_check(
                "trigger_snapshot_failed",
                {"error": str(e)[:300]},
            )
            return ""

    def _save_diagnostic_snapshot_image(self, reason_key, fov_gate_diag=None):
        # Diagnostic images are intentionally disabled; only VLM event snapshots are persisted.
        return ""

    def _capture_gate_snapshot_if_needed(self, trigger_generation, reason, fov_gate_diag=None):
        if not self.vla_snapshot_on_trigger:
            return ""
        try:
            generation = int(trigger_generation)
        except Exception:
            generation = -1
        if generation < 0:
            return ""
        if self._last_snapshot_trigger_generation == generation and self._event_image_path:
            return self._event_image_path
        self._event_image_path = ""
        self._event_image_url = ""
        self._event_input_image_path = ""
        self._event_input_image_url = ""
        path = self._save_trigger_snapshot_image(fov_gate_diag=fov_gate_diag)
        self._last_snapshot_trigger_generation = generation
        self._publish_vlm_check(
            "trigger_gate_snapshot",
            {
                "trigger_generation": generation,
                "call_reason": str(reason),
                "saved": bool(self._event_image_path),
            },
        )
        return path

    def _append_vlm_gate_status(self, reason_key, extra=None, force=False):
        now = rospy.Time.now()
        if not force:
            elapsed = (now - self._last_gate_status_record_t).to_sec()
            if elapsed < max(0.5, float(self.llm_gate_status_log_interval_s)):
                return
        self._last_gate_status_record_t = now
        record = {
            "event": "vlm_gate_status",
            "wall_time_s": time.time(),
            "ros_time_s": now.to_sec(),
            "reason": self._call_reason_zh(reason_key),
            "reason_key": str(reason_key),
            "backend": self._backend_name_zh(self.llm_backend),
            "backend_key": self.llm_backend,
            "model": self._last_llm_model_id or self.api_model,
            "vla_prompt_mode": self.vla_prompt_mode,
            "trigger": bool(self._trigger),
            "trigger_generation": int(self._trigger_generation),
            "state_seq": int(self._state_seq),
            "state_snapshot": self._state_snapshot(),
            "image_diag": self._vla_image_diag(),
            "llm_called_count": int(self._llm_call_seq),
            "llm_inflight": bool(self._llm_inflight),
        }
        if isinstance(extra, dict):
            record.update(extra)
        self._append_llm_io_record(record)

    def _state_cb(self, msg):
        try:
            self._state = json.loads(msg.data)
            self._state_seq += 1
            now_s = rospy.Time.now().to_sec()
            rp = self._state.get("relative_position", {}) or {}
            rel_x = self._safe_float(rp.get("x"), None)
            rel_y = self._safe_float(rp.get("y"), None)
            if rel_x is not None and rel_y is not None:
                range_m = math.hypot(rel_x, rel_y)
                world_xy = self._relative_to_world_latest(rel_x, rel_y)
                self._append_history_sample(
                    self._state_target_history,
                    now_s,
                    rel_x,
                    rel_y,
                    range_m,
                    world_x=None if world_xy is None else world_xy[0],
                    world_y=None if world_xy is None else world_xy[1],
                )
        except Exception:
            self._state = {}

    def _perception_targets_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            targets = payload.get("targets") or []
            if not isinstance(targets, list):
                targets = []
            self._perception_targets = targets
            self._perception_stamp = float(payload.get("stamp", rospy.Time.now().to_sec()))

            seen_keys = set()
            stamp_s = float(self._perception_stamp)
            for i, t in enumerate(targets, start=1):
                if not isinstance(t, dict):
                    continue
                key = str(t.get("id", "target_%d" % i))
                rel_x = self._safe_float(t.get("rel_x"), None)
                rel_y = self._safe_float(t.get("rel_y"), None)
                range_m = self._safe_float(t.get("range_m"), None)
                if range_m is None and rel_x is not None and rel_y is not None:
                    range_m = math.hypot(rel_x, rel_y)
                if rel_x is None or rel_y is None:
                    continue
                hist = self._target_histories.get(key)
                if hist is None:
                    hist = deque()
                    self._target_histories[key] = hist
                world_xy = self._relative_to_world_latest(rel_x, rel_y)
                self._append_history_sample(
                    hist,
                    stamp_s,
                    rel_x,
                    rel_y,
                    range_m,
                    world_x=None if world_xy is None else world_xy[0],
                    world_y=None if world_xy is None else world_xy[1],
                )
                seen_keys.add(key)

            self._prune_target_histories(stamp_s, seen_keys)
        except Exception:
            self._perception_targets = []

    def _risk_cb(self, msg):
        try:
            self._risk_state = json.loads(msg.data)
            self._risk_stamp = float(self._risk_state.get("stamp", rospy.Time.now().to_sec()))
        except Exception:
            self._risk_state = {}

    @staticmethod
    def _safe_float(v, default_v=None):
        try:
            return float(v)
        except Exception:
            return default_v

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _norm_deg(a):
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    @staticmethod
    def _bearing_sector_label(bearing_deg):
        if bearing_deg is None:
            return "unknown"
        bearing_deg = float(bearing_deg)
        abs_b = abs(bearing_deg)
        if abs_b <= 15.0:
            return "ahead_center"
        if abs_b <= 60.0:
            return "ahead_port" if bearing_deg > 0.0 else "ahead_starboard"
        if abs_b <= 120.0:
            return "abeam_port" if bearing_deg > 0.0 else "abeam_starboard"
        if abs_b <= 165.0:
            return "astern_port" if bearing_deg > 0.0 else "astern_starboard"
        return "astern_center"

    @staticmethod
    def _image_region_hint(bearing_deg):
        if bearing_deg is None:
            return "unknown"
        bearing_deg = float(bearing_deg)
        if bearing_deg >= 35.0:
            return "far_left"
        if bearing_deg >= 10.0:
            return "left"
        if bearing_deg <= -35.0:
            return "far_right"
        if bearing_deg <= -10.0:
            return "right"
        return "center"

    def _image_x_hint_01(self, bearing_deg):
        if bearing_deg is None:
            return None
        hfov = max(1.0, float(self.vla_camera_hfov_deg))
        half = 0.5 * hfov
        x = 0.5 - (float(bearing_deg) / hfov)
        if float(bearing_deg) >= half:
            x = 0.0
        elif float(bearing_deg) <= -half:
            x = 1.0
        return round(self._clamp(x, 0.0, 1.0), 3)

    @staticmethod
    def _image_x_band(x01):
        if x01 is None:
            return "unknown"
        x01 = float(x01)
        if x01 <= 0.15:
            return "far_left_edge"
        if x01 <= 0.35:
            return "left_inner"
        if x01 < 0.65:
            return "center_band"
        if x01 < 0.85:
            return "right_inner"
        return "far_right_edge"

    def _semantic_perception_targets(self):
        out = []
        for t in self._perception_targets:
            if not isinstance(t, dict):
                continue
            if bool(t.get("semantic_candidate", True)):
                out.append(t)
        return out

    def _current_target_ids(self):
        ids = []
        for i, t in enumerate(self._semantic_perception_targets(), start=1):
            if not isinstance(t, dict):
                continue
            ids.append(str(t.get("id", "target_%d" % i)))
        if ids:
            return ids
        if isinstance(self._state, dict) and self._state:
            return ["track_1"]
        return []

    @staticmethod
    def _target_id_index(raw_tid):
        text = str(raw_tid or "").strip().lower()
        if not text:
            return None
        match = re.search(r"(?:^|[^0-9])(\d+)$", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _normalize_target_id(self, raw_tid, expected_ids=None):
        text = str(raw_tid or "").strip()
        if not text:
            return ""
        if expected_ids and text in expected_ids:
            return text

        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if expected_ids:
            for expected in expected_ids:
                expected_text = str(expected or "").strip()
                if not expected_text:
                    continue
                if normalized == expected_text.lower().replace("-", "_").replace(" ", "_"):
                    return expected_text

            raw_index = self._target_id_index(text)
            if raw_index is not None:
                for expected in expected_ids:
                    expected_text = str(expected or "").strip()
                    if self._target_id_index(expected_text) == raw_index:
                        return expected_text
            return ""

        if normalized.isdigit():
            return "track_%d" % int(normalized)
        raw_index = self._target_id_index(text)
        if raw_index is not None:
            return "track_%d" % raw_index
        return text

    @staticmethod
    def _canonical_vessel_type(v):
        s = str(v or "").strip().lower().replace("-", " ").replace("_", " ")
        if s in ("救生艇", "救援艇", "lifeboat", "rescue boat", "rescue vessel"):
            return "Lifeboat"
        if s in ("无人艇", "无人船", "usv", "unmanned boat", "unmanned vessel", "unmanned surface vessel"):
            return "USV"
        if s in ("渔船", "fishing", "fishing boat", "fishing vessel", "trawler"):
            return "Fishing"
        if s in ("small vessel", "small boat", "smallvessel", "generic small vessel"):
            return "SmallVessel"
        if s in ("unknown", "unk", "none", "null"):
            return "Unknown"
        return "Unknown"

    def _normalize_track_classifications(self, parsed, expected_ids=None):
        if expected_ids is None:
            expected_ids = self._current_target_ids()
        raw_items = parsed.get("track_classifications", []) if isinstance(parsed, dict) else []
        by_id = {}
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                target_id = self._normalize_target_id(item.get("target_id", ""), expected_ids)
                if not target_id:
                    continue
                assoc = self._safe_float(
                    item.get("association_confidence", item.get("confidence", parsed.get("confidence", 0.0))),
                    0.0,
                )
                normalized_item = {
                    "target_id": target_id,
                    "vessel_type": self._canonical_vessel_type(
                        item.get("vessel_type", item.get("ship_type", item.get("target_type", "UNKNOWN")))
                    ),
                    "association_confidence": round(self._clamp(float(assoc), 0.0, 1.0), 3),
                }
                by_id[target_id] = normalized_item
        out = []
        ordered_ids = expected_ids or list(by_id.keys())
        for target_id in ordered_ids:
            out.append(
                by_id.get(
                    target_id,
                    {
                        "target_id": target_id,
                        "vessel_type": "Unknown",
                        "association_confidence": 0.0,
                    },
                )
            )
        return out

    def _fallback_track_classifications(self):
        return [
            {
                "target_id": target_id,
                "vessel_type": "Unknown",
                "association_confidence": 0.0,
            }
            for target_id in self._current_target_ids()
        ]

    def _append_ego_world_sample(self, stamp_s, x, y, yaw):
        hist = getattr(self, "_ego_world_history", None)
        if hist is None:
            self._ego_world_history = deque()
            hist = self._ego_world_history
        if hist and (float(stamp_s) - float(hist[-1].get("t", 0.0))) < self.target_history_min_dt_s:
            return
        hist.append({"t": float(stamp_s), "x": float(x), "y": float(y), "yaw": float(yaw)})
        while len(hist) > max(5, self.target_history_max_samples):
            hist.popleft()
        cut_t = float(stamp_s) - max(1.0, self.target_history_window_s)
        while len(hist) > 2 and float(hist[0].get("t", 0.0)) < cut_t:
            hist.popleft()

    def _append_ais_target_sample(self, target_id, stamp_s, x, y, yaw):
        histories = getattr(self, "_ais_target_histories", None)
        if histories is None:
            self._ais_target_histories = {}
            histories = self._ais_target_histories
        hist = histories.get(str(target_id))
        if hist is None:
            hist = deque()
            histories[str(target_id)] = hist
        if hist and (float(stamp_s) - float(hist[-1].get("t", 0.0))) < self.target_history_min_dt_s:
            return
        hist.append(
            {
                "t": float(stamp_s),
                "world_x": float(x),
                "world_y": float(y),
                "yaw": float(yaw),
            }
        )
        while len(hist) > max(5, self.target_history_max_samples):
            hist.popleft()
        cut_t = float(stamp_s) - max(1.0, self.target_history_window_s)
        while len(hist) > 2 and float(hist[0].get("t", 0.0)) < cut_t:
            hist.popleft()

    def _latest_ego_world_sample(self):
        hist = getattr(self, "_ego_world_history", deque())
        if hist:
            return hist[-1]
        if self._ego_pos_xy is not None:
            yaw = self._safe_float((self._state or {}).get("ego_heading_rad"), 0.0)
            return {"t": rospy.Time.now().to_sec(), "x": self._ego_pos_xy[0], "y": self._ego_pos_xy[1], "yaw": yaw}
        return None

    def _relative_to_world_latest(self, rel_x, rel_y):
        ego = self._latest_ego_world_sample()
        if not isinstance(ego, dict):
            return None
        ex = self._safe_float(ego.get("x"), None)
        ey = self._safe_float(ego.get("y"), None)
        yaw = self._safe_float(ego.get("yaw"), None)
        if ex is None or ey is None or yaw is None:
            return None
        wx = ex + float(rel_x) * math.cos(yaw) - float(rel_y) * math.sin(yaw)
        wy = ey + float(rel_x) * math.sin(yaw) + float(rel_y) * math.cos(yaw)
        return wx, wy

    def _append_history_sample(self, hist, stamp_s, rel_x, rel_y, range_m, world_x=None, world_y=None):
        if hist and (stamp_s - float(hist[-1].get("t", 0.0))) < self.target_history_min_dt_s:
            return
        item = {
            "t": float(stamp_s),
            "rel_x": float(rel_x),
            "rel_y": float(rel_y),
            "range_m": float(range_m) if range_m is not None else math.hypot(rel_x, rel_y),
            "bearing_deg": math.degrees(math.atan2(rel_y, rel_x)),
        }
        if world_x is not None and world_y is not None:
            item["world_x"] = float(world_x)
            item["world_y"] = float(world_y)
        hist.append(item)
        while len(hist) > max(5, self.target_history_max_samples):
            hist.popleft()
        cut_t = stamp_s - max(1.0, self.target_history_window_s)
        while len(hist) > 2 and float(hist[0].get("t", 0.0)) < cut_t:
            hist.popleft()

    def _prune_target_histories(self, now_s, seen_keys):
        stale_cut = now_s - max(2.0, self.target_history_window_s * 2.0)
        to_del = []
        for k, hist in self._target_histories.items():
            while len(hist) > 2 and float(hist[0].get("t", 0.0)) < (now_s - self.target_history_window_s):
                hist.popleft()
            last_t = float(hist[-1].get("t", 0.0)) if hist else 0.0
            if (not hist) or ((k not in seen_keys) and last_t < stale_cut):
                to_del.append(k)
        for k in to_del:
            self._target_histories.pop(k, None)

    def _history_motion_summary(self, hist):
        if not hist or len(hist) < 3:
            return {
                "history_span_s": 0.0,
                "closing_speed_mps": None,
                "bearing_trend": "unknown",
                "collision_course_hint": "unknown",
                "target_turn_trend": "unknown",
            }

        h = list(hist)
        p0 = h[0]
        p1 = h[len(h) // 2]
        p2 = h[-1]
        dt02 = max(1e-3, float(p2["t"]) - float(p0["t"]))
        dt01 = max(1e-3, float(p1["t"]) - float(p0["t"]))
        dt12 = max(1e-3, float(p2["t"]) - float(p1["t"]))

        range_rate = (float(p2["range_m"]) - float(p0["range_m"])) / dt02
        closing_speed = -range_rate
        bearing_rate = self._norm_deg(float(p2["bearing_deg"]) - float(p0["bearing_deg"])) / dt02

        v1x = (float(p1["rel_x"]) - float(p0["rel_x"])) / dt01
        v1y = (float(p1["rel_y"]) - float(p0["rel_y"])) / dt01
        v2x = (float(p2["rel_x"]) - float(p1["rel_x"])) / dt12
        v2y = (float(p2["rel_y"]) - float(p1["rel_y"])) / dt12
        h1 = math.degrees(math.atan2(v1y, v1x))
        h2 = math.degrees(math.atan2(v2y, v2x))
        turn_rate = self._norm_deg(h2 - h1) / max(1e-3, 0.5 * (dt01 + dt12))

        if turn_rate > 3.0:
            turn_trend = "turning_port"
        elif turn_rate < -3.0:
            turn_trend = "turning_starboard"
        else:
            turn_trend = "steady"

        if bearing_rate > 1.0:
            bearing_trend = "moving_to_port"
        elif bearing_rate < -1.0:
            bearing_trend = "moving_to_starboard"
        else:
            bearing_trend = "constant_bearing"

        if closing_speed > 0.05 and bearing_trend == "constant_bearing":
            collision_hint = "high"
        elif closing_speed > 0.05:
            collision_hint = "medium"
        elif closing_speed < -0.05:
            collision_hint = "low"
        else:
            collision_hint = "uncertain"

        return {
            "history_span_s": round(float(dt02), 2),
            "closing_speed_mps": round(float(closing_speed), 3),
            "bearing_trend": bearing_trend,
            "collision_course_hint": collision_hint,
            "target_turn_trend": turn_trend,
        }

    def _history_warmup_diag_for_targets(self, fov_gate_diag=None):
        targets = []
        if isinstance(fov_gate_diag, dict):
            targets = fov_gate_diag.get("prompt_targets") or fov_gate_diag.get("targets") or []
        if not isinstance(targets, list) or not targets:
            targets = self._build_targets_prompt_context(include_state_fallback=False)
        min_span_s = float(self.vla_target_history_warmup_s)
        min_samples = int(self.vla_target_history_warmup_min_samples)
        target_diags = []
        ready_count = 0
        for idx, target in enumerate(targets):
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("target_id", target.get("id", "target_%d" % (idx + 1))))
            hist = self._target_histories.get(target_id)
            if (not hist) and target_id == "track_1":
                hist = self._state_target_history
            samples = len(hist) if hist is not None else 0
            span_s = 0.0
            if hist and len(hist) >= 2:
                try:
                    span_s = max(0.0, float(hist[-1].get("t", 0.0)) - float(hist[0].get("t", 0.0)))
                except Exception:
                    span_s = 0.0
            ready = samples >= min_samples and (min_span_s <= 0.0 or span_s >= min_span_s)
            if ready:
                ready_count += 1
            target_diags.append(
                {
                    "target_id": target_id,
                    "samples": int(samples),
                    "span_s": round(float(span_s), 3),
                    "ready": bool(ready),
                }
            )
        required_targets = min(
            max(1, int(self.vla_target_history_warmup_min_ready_targets)),
            max(1, len(target_diags)),
        )
        ready = (not target_diags) or ready_count >= required_targets
        return {
            "ready": bool(ready),
            "ready_count": int(ready_count),
            "required_ready_targets": int(required_targets),
            "target_count": int(len(target_diags)),
            "required_span_s": float(min_span_s),
            "required_samples": int(min_samples),
            "targets": target_diags[:12],
        }

    def _clearance_hint_m(self, rel_x, rel_y, range_m, rel_speed, collision_hint):
        clearance = 4.0
        if rel_y is not None and abs(float(rel_y)) < 3.0:
            clearance += 2.0
        if range_m is not None and float(range_m) < 20.0:
            clearance += 2.0
        if range_m is not None and float(range_m) < 12.0:
            clearance += 1.5
        if rel_speed is not None and float(rel_speed) > 0.8:
            clearance += 1.0
        if collision_hint == "high":
            clearance += 2.0
        elif collision_hint == "medium":
            clearance += 1.0
        if rel_x is not None and float(rel_x) < 8.0:
            clearance += 1.0
        return round(self._clamp(clearance, 4.0, 12.0), 2)

    def _project_relative_target(self, rel_x, rel_y, rvx, rvy, dt_s):
        if rel_x is None or rel_y is None:
            return None
        dt_s = max(0.0, float(dt_s))
        pred_x = float(rel_x) + float(rvx or 0.0) * dt_s
        pred_y = float(rel_y) + float(rvy or 0.0) * dt_s
        pred_range = math.hypot(pred_x, pred_y)
        pred_bearing = math.degrees(math.atan2(pred_y, pred_x)) if pred_range > 1e-6 else 0.0
        return {
            "t_s": round(dt_s, 2),
            "relative_forward_m": round(pred_x, 3),
            "relative_lateral_m": round(pred_y, 3),
            "distance_m": round(pred_range, 3),
            "relative_angle_deg": round(pred_bearing, 2),
            "image_region_hint": self._image_region_hint(pred_bearing),
            "image_x_hint_01": self._image_x_hint_01(pred_bearing),
        }

    def _target_future_checkpoints(self, rel_x, rel_y, rvx, rvy):
        base_dt = max(0.5, float(self.llm_future_point_dt_s))
        delay_dt = max(0.0, float(self.llm_control_delay_s))
        checkpoints = []
        for dt_s in (delay_dt, delay_dt + base_dt, delay_dt + 2.0 * base_dt):
            snap = self._project_relative_target(rel_x, rel_y, rvx, rvy, dt_s)
            if snap is not None:
                snap["image_x_band"] = self._image_x_band(snap.get("image_x_hint_01"))
                checkpoints.append(snap)
        return checkpoints

    def _cpa_metrics(self, rel_x, rel_y, rvx, rvy):
        if rel_x is None or rel_y is None:
            return None, None
        px = float(rel_x)
        py = float(rel_y)
        vx = float(rvx or 0.0)
        vy = float(rvy or 0.0)
        v2 = vx * vx + vy * vy
        if v2 < 1e-6:
            return math.hypot(px, py), float("inf")
        tcpa = -((px * vx + py * vy) / v2)
        cx = px + vx * tcpa
        cy = py + vy * tcpa
        return math.hypot(cx, cy), tcpa

    def _format_temporal_risk_token(self, label, rel_x, rel_y, rvx, rvy):
        if rel_x is None or rel_y is None:
            return "%s: a=NA, dcpa=NA, tcpa=NA" % label
        angle = math.degrees(math.atan2(float(rel_y), float(rel_x)))
        dcpa, tcpa = self._cpa_metrics(rel_x, rel_y, rvx, rvy)
        dcpa_text = "NA" if dcpa is None else "%.1fm" % float(dcpa)
        if tcpa is None:
            tcpa_text = "NA"
        elif math.isinf(float(tcpa)):
            tcpa_text = "inf"
        else:
            tcpa_text = "%.1fs" % float(tcpa)
        return "%s: a=%.0f, dcpa=%s, tcpa=%s" % (label, angle, dcpa_text, tcpa_text)

    def _history_sample_nearest_index(self, hist, stamp_s):
        if not hist:
            return None
        best_i = None
        best_dt = float("inf")
        for i, sample in enumerate(hist):
            try:
                dt = abs(float(sample.get("t", 0.0)) - float(stamp_s))
            except Exception:
                continue
            if dt < best_dt:
                best_dt = dt
                best_i = i
        return best_i

    def _history_velocity_at_index(self, hist, idx, fallback_vx=None, fallback_vy=None):
        if not hist or idx is None:
            return fallback_vx, fallback_vy
        if len(hist) < 2:
            return fallback_vx, fallback_vy
        i0 = max(0, min(int(idx), len(hist) - 1))
        if i0 > 0:
            a = hist[i0 - 1]
            b = hist[i0]
        else:
            a = hist[i0]
            b = hist[min(i0 + 1, len(hist) - 1)]
        dt = float(b.get("t", 0.0)) - float(a.get("t", 0.0))
        if abs(dt) < 1e-6:
            return fallback_vx, fallback_vy
        vx = (float(b.get("rel_x", 0.0)) - float(a.get("rel_x", 0.0))) / dt
        vy = (float(b.get("rel_y", 0.0)) - float(a.get("rel_y", 0.0))) / dt
        return vx, vy

    def _temporal_risk_tokens(self, hist, rel_x, rel_y, rvx, rvy):
        tokens = []
        h = list(hist) if hist else []
        now_s = float(h[-1].get("t", 0.0)) if h else 0.0
        for age_s in (6.0, 4.0, 2.0):
            label = "t-%d" % int(age_s)
            if not h:
                tokens.append("%s: a=NA, dcpa=NA, tcpa=NA" % label)
                continue
            idx = self._history_sample_nearest_index(h, now_s - age_s)
            if idx is None:
                tokens.append("%s: a=NA, dcpa=NA, tcpa=NA" % label)
                continue
            sample = h[idx]
            hvx, hvy = self._history_velocity_at_index(h, idx, rvx, rvy)
            tokens.append(
                self._format_temporal_risk_token(
                    label,
                    self._safe_float(sample.get("rel_x"), None),
                    self._safe_float(sample.get("rel_y"), None),
                    hvx,
                    hvy,
                )
            )
        tokens.append(self._format_temporal_risk_token("t0", rel_x, rel_y, rvx, rvy))
        return tokens

    def _planning_prompt_context(self):
        d = self._perception_diag()
        ego_speed = self._safe_float(d.get("ego_speed_mps"), 0.0)
        delay_s = max(0.0, float(self.llm_control_delay_s))
        travel_m = max(0.0, float(ego_speed)) * delay_s
        return {
            "control_delay_s": round(delay_s, 2),
            "ego_speed_mps": round(float(ego_speed), 3),
            "ego_delay_travel_m": round(float(travel_m), 3),
            "future_prediction_step_s": round(max(0.5, float(self.llm_future_point_dt_s)), 2),
            "recommended_insert_waypoint_count": 3,
            "minimum_insert_waypoint_count_for_avoidance": 2,
        }

    def _state_fallback_prompt_target(self):
        if not self._state:
            return None
        rel_p = self._state.get("relative_position", {}) or {}
        rel_v = self._state.get("relative_velocity", {}) or {}
        rel_x = self._safe_float(rel_p.get("x"), 0.0)
        rel_y = self._safe_float(rel_p.get("y"), 0.0)
        rvx = self._safe_float(rel_v.get("x"), 0.0)
        rvy = self._safe_float(rel_v.get("y"), 0.0)
        size_proxy = None
        pcd = self._state.get("pointcloud_tracking_debug", {}) or {}
        stats = pcd.get("pointcloud_stats", {}) if isinstance(pcd, dict) else {}
        if isinstance(stats, dict):
            size_proxy = self._safe_float(stats.get("chosen_cluster_points"), None)
        hist_summary = self._history_motion_summary(self._state_target_history)
        range_m = math.hypot(rel_x, rel_y)
        rel_speed = math.hypot(rvx, rvy)
        bearing_deg = math.degrees(math.atan2(rel_y, rel_x))
        image_x_hint_01 = self._image_x_hint_01(bearing_deg)
        future_checkpoints = self._target_future_checkpoints(rel_x, rel_y, rvx, rvy)
        temporal_risk_tokens = self._temporal_risk_tokens(
            self._state_target_history,
            rel_x,
            rel_y,
            rvx,
            rvy,
        )
        sensor_estimated_dcpa_m, sensor_estimated_tcpa_s = self._cpa_metrics(rel_x, rel_y, rvx, rvy)
        if sensor_estimated_dcpa_m is not None and not math.isfinite(float(sensor_estimated_dcpa_m)):
            sensor_estimated_dcpa_m = None
        if sensor_estimated_tcpa_s is not None and not math.isfinite(float(sensor_estimated_tcpa_s)):
            sensor_estimated_tcpa_s = None
        clearance_hint_m = self._clearance_hint_m(
            rel_x,
            rel_y,
            range_m,
            rel_speed,
            hist_summary["collision_course_hint"],
        )
        return {
            "target_id": "track_1",
            "distance_m": round(range_m, 3),
            "relative_forward_m": round(rel_x, 3),
            "relative_lateral_m": round(rel_y, 3),
            "relative_z_m": 0.65,
            "direction": "port_left" if rel_y > 0.0 else "starboard_right",
            "relative_angle_deg": round(bearing_deg, 2),
            "relative_sector": self._bearing_sector_label(bearing_deg),
            "image_region_hint": self._image_region_hint(bearing_deg),
            "image_x_hint_01": image_x_hint_01,
            "image_x_band": self._image_x_band(image_x_hint_01),
            "relative_velocity_forward_mps": round(rvx, 3),
            "relative_velocity_lateral_mps": round(rvy, 3),
            "relative_speed_mps": round(rel_speed, 3),
            "size": None if size_proxy is None else round(size_proxy, 3),
            "size_x_m": None,
            "size_y_m": None,
            "size_z_m": None,
            "point_count": None if size_proxy is None else int(round(size_proxy)),
            "future_relative_checkpoints": future_checkpoints,
            "temporal_risk_tokens": temporal_risk_tokens,
            "sensor_estimated_dcpa_m": None if sensor_estimated_dcpa_m is None else round(sensor_estimated_dcpa_m, 3),
            "sensor_estimated_tcpa_s": None if sensor_estimated_tcpa_s is None else round(sensor_estimated_tcpa_s, 3),
            "history_span_s": hist_summary["history_span_s"],
            "closing_speed_mps": hist_summary["closing_speed_mps"],
            "bearing_trend": hist_summary["bearing_trend"],
            "collision_course_hint": hist_summary["collision_course_hint"],
            "target_turn_trend": hist_summary["target_turn_trend"],
            "clearance_hint_m": clearance_hint_m,
            "source": "sensor_estimate_state_fallback",
            "track_input_source": "sensor_estimate",
        }

    def _build_targets_prompt_context(self, include_state_fallback=True):
        lines = []
        idx = 1
        for t in self._semantic_perception_targets():
            if not isinstance(t, dict):
                continue
            rel_x = self._safe_float(t.get("rel_x"), None)
            rel_y = self._safe_float(t.get("rel_y"), None)
            range_m = self._safe_float(t.get("range_m"), None)
            rvx = self._safe_float(t.get("rel_vx"), None)
            rvy = self._safe_float(t.get("rel_vy"), None)
            rel_speed = self._safe_float(t.get("relative_speed_mps"), None)
            if rel_speed is None and rvx is not None and rvy is not None:
                rel_speed = math.hypot(rvx, rvy)
            if range_m is None and rel_x is not None and rel_y is not None:
                range_m = math.hypot(rel_x, rel_y)
            bearing_deg = None
            direction = "unknown"
            relative_sector = "unknown"
            if rel_x is not None and rel_y is not None:
                bearing_deg = math.degrees(math.atan2(rel_y, rel_x))
                direction = "port_left" if rel_y > 0.0 else "starboard_right"
                relative_sector = self._bearing_sector_label(bearing_deg)
            size_proxy = self._safe_float(t.get("size_m2"), None)
            if size_proxy is None:
                size_proxy = self._safe_float(t.get("cluster_points"), None)
            size_x_m = self._safe_float(t.get("size_x_m"), None)
            size_y_m = self._safe_float(t.get("size_y_m"), None)
            size_z_m = self._safe_float(t.get("size_z_m"), None)
            rel_z_m = self._safe_float(t.get("rel_z"), None)
            point_count = self._safe_float(t.get("cluster_points"), None)
            image_x_hint_01 = self._image_x_hint_01(bearing_deg)
            future_checkpoints = self._target_future_checkpoints(rel_x, rel_y, rvx, rvy)
            key = str(t.get("id", "target_%d" % idx))
            hist = self._target_histories.get(key, deque())
            hist_summary = self._history_motion_summary(hist)
            temporal_risk_tokens = self._temporal_risk_tokens(hist, rel_x, rel_y, rvx, rvy)
            sensor_estimated_dcpa_m, sensor_estimated_tcpa_s = self._cpa_metrics(rel_x, rel_y, rvx, rvy)
            if sensor_estimated_dcpa_m is not None and not math.isfinite(float(sensor_estimated_dcpa_m)):
                sensor_estimated_dcpa_m = None
            if sensor_estimated_tcpa_s is not None and not math.isfinite(float(sensor_estimated_tcpa_s)):
                sensor_estimated_tcpa_s = None
            clearance_hint_m = self._clearance_hint_m(
                rel_x,
                rel_y,
                range_m,
                rel_speed,
                hist_summary["collision_course_hint"],
            )
            lines.append(
                {
                    "target_id": key,
                    "distance_m": None if range_m is None else round(range_m, 3),
                    "relative_forward_m": None if rel_x is None else round(rel_x, 3),
                    "relative_lateral_m": None if rel_y is None else round(rel_y, 3),
                    "relative_z_m": None if rel_z_m is None else round(rel_z_m, 3),
                    "direction": direction,
                    "relative_angle_deg": None if bearing_deg is None else round(bearing_deg, 2),
                    "relative_sector": relative_sector,
                    "image_region_hint": self._image_region_hint(bearing_deg),
                    "image_x_hint_01": image_x_hint_01,
                    "image_x_band": self._image_x_band(image_x_hint_01),
                    "relative_velocity_forward_mps": None if rvx is None else round(rvx, 3),
                    "relative_velocity_lateral_mps": None if rvy is None else round(rvy, 3),
                    "relative_speed_mps": None if rel_speed is None else round(rel_speed, 3),
                    "size": None if size_proxy is None else round(size_proxy, 3),
                    "size_x_m": None if size_x_m is None else round(size_x_m, 3),
                    "size_y_m": None if size_y_m is None else round(size_y_m, 3),
                    "size_z_m": None if size_z_m is None else round(size_z_m, 3),
                    "point_count": None if point_count is None else int(round(point_count)),
                    "future_relative_checkpoints": future_checkpoints,
                    "temporal_risk_tokens": temporal_risk_tokens,
                    "sensor_estimated_dcpa_m": None if sensor_estimated_dcpa_m is None else round(sensor_estimated_dcpa_m, 3),
                    "sensor_estimated_tcpa_s": None if sensor_estimated_tcpa_s is None else round(sensor_estimated_tcpa_s, 3),
                    "history_span_s": hist_summary["history_span_s"],
                    "closing_speed_mps": hist_summary["closing_speed_mps"],
                    "bearing_trend": hist_summary["bearing_trend"],
                    "collision_course_hint": hist_summary["collision_course_hint"],
                    "target_turn_trend": hist_summary["target_turn_trend"],
                    "clearance_hint_m": clearance_hint_m,
                    "track_input_source": "sensor_estimate",
                }
            )
            idx += 1

        # Fallback to current state target if perception targets are unavailable.
        if include_state_fallback and (not lines) and self._state:
            fallback = self._state_fallback_prompt_target()
            if fallback is not None:
                lines.append(fallback)

        return lines

    def _soft_fov_prompt_targets(self, targets, max_distance):
        if not isinstance(targets, list):
            return []
        candidates = []
        for idx, t in enumerate(targets):
            if not isinstance(t, dict):
                continue
            distance = self._safe_float(t.get("distance_m"), None)
            if distance is None or distance <= 0.0:
                continue
            if not math.isinf(float(max_distance)) and distance > float(max_distance):
                continue
            rel_forward = self._safe_float(t.get("relative_forward_m"), 0.0)
            bearing = self._safe_float(t.get("relative_angle_deg"), 0.0)
            behind_penalty = 10.0 if rel_forward is not None and rel_forward < 0.0 else 0.0
            side_penalty = 0.03 * abs(float(bearing or 0.0))
            candidates.append((distance + behind_penalty + side_penalty, idx, t))
        candidates.sort(key=lambda item: (item[0], item[1]))
        limit = max(1, int(self.llm_fov_gate_soft_max_targets))
        return [item[2] for item in candidates[:limit]]

    def _soft_fov_checked_targets(self, prompt_targets, checked_targets):
        if not isinstance(prompt_targets, list):
            return []
        checked_by_id = {}
        if isinstance(checked_targets, list):
            for item in checked_targets:
                if not isinstance(item, dict):
                    continue
                target_id = str(item.get("target_id", item.get("id", ""))).strip()
                if target_id:
                    checked_by_id[target_id] = item

        checked = []
        for t in prompt_targets:
            if not isinstance(t, dict):
                continue
            target_id = str(t.get("target_id", t.get("id", ""))).strip()
            base = dict(checked_by_id.get(target_id, {}))
            if not base:
                base = {
                    "target_id": target_id,
                    "distance_m": self._safe_float(t.get("distance_m"), None),
                    "relative_forward_m": self._safe_float(t.get("relative_forward_m"), None),
                    "relative_angle_deg": self._safe_float(t.get("relative_angle_deg"), None),
                    "image_x_hint_01": t.get("image_x_hint_01"),
                    "image_x_band": t.get("image_x_band"),
                    "reject_reasons": ["soft_prompt_only"],
                }
            base["geometric_in_fov"] = bool(base.get("in_fov", False))
            base["in_fov"] = True
            base["soft_fov"] = True
            base["source"] = "soft_fov"
            checked.append(base)
        return checked

    def _vla_fov_gate_diag(self):
        targets = self._build_targets_prompt_context(include_state_fallback=False)
        half_angle = max(
            1.0,
            0.5 * float(self.vla_camera_hfov_deg) - max(0.0, float(self.llm_fov_gate_margin_deg)),
        )
        min_forward = max(0.0, float(self.llm_fov_gate_min_forward_m))
        max_distance = float(self.llm_fov_gate_max_distance_m)
        if max_distance <= 0.0:
            max_distance = float("inf")

        checked = []
        prompt_targets = []
        visible_count = 0
        for t in targets:
            if not isinstance(t, dict):
                continue
            rel_forward = self._safe_float(t.get("relative_forward_m"), None)
            rel_angle = self._safe_float(t.get("relative_angle_deg"), None)
            distance = self._safe_float(t.get("distance_m"), None)
            reasons = []
            in_fov = True
            if rel_forward is None or rel_forward < min_forward:
                in_fov = False
                reasons.append("not_ahead")
            if rel_angle is None or abs(rel_angle) > half_angle:
                in_fov = False
                reasons.append("outside_hfov")
            if distance is None or distance > max_distance:
                in_fov = False
                reasons.append("distance_out")
            if in_fov:
                visible_count += 1
                prompt_targets.append(t)
            target_id = t.get("target_id", t.get("id", ""))
            checked.append(
                {
                    "target_id": target_id,
                    "distance_m": distance,
                    "relative_forward_m": rel_forward,
                    "relative_angle_deg": rel_angle,
                    "image_x_hint_01": t.get("image_x_hint_01"),
                    "image_x_band": t.get("image_x_band"),
                    "in_fov": bool(in_fov),
                    "reject_reasons": reasons,
                }
            )

        track_count = len(checked)
        all_in_fov = track_count > 0 and visible_count == track_count
        any_in_fov = visible_count > 0
        soft_prompt_targets = self._soft_fov_prompt_targets(targets, max_distance)
        soft_checked_targets = self._soft_fov_checked_targets(soft_prompt_targets, checked)
        state_range = self._safe_float((self._state or {}).get("range_m"), None)
        state_bearing = self._safe_float((self._state or {}).get("relative_bearing_deg"), None)
        state_max_distance = float(self.llm_fov_gate_state_max_distance_m)
        if state_max_distance <= 0.0:
            state_max_distance = float("inf")
        state_reasons = []
        state_in_fov = True
        if state_range is None or state_range > state_max_distance:
            state_in_fov = False
            state_reasons.append("state_distance_out")
        if state_bearing is None or abs(state_bearing) > half_angle:
            state_in_fov = False
            state_reasons.append("state_outside_hfov")
        state_prompt_target = self._state_fallback_prompt_target() if state_in_fov else None
        state_checked = None
        if isinstance(state_prompt_target, dict):
            state_checked = {
                "target_id": state_prompt_target.get("target_id", "track_1"),
                "distance_m": state_prompt_target.get("distance_m", state_range),
                "relative_forward_m": state_prompt_target.get("relative_forward_m"),
                "relative_angle_deg": state_prompt_target.get("relative_angle_deg", state_bearing),
                "image_x_hint_01": state_prompt_target.get("image_x_hint_01"),
                "image_x_band": state_prompt_target.get("image_x_band"),
                "in_fov": True,
                "reject_reasons": [],
                "source": "state_fallback",
            }
        return {
            "enabled": bool(self.llm_fov_gate_enable),
            "track_count": track_count,
            "visible_count": visible_count,
            "all_in_fov": bool(all_in_fov),
            "any_in_fov": bool(any_in_fov),
            "min_visible_tracks": max(1, int(self.llm_fov_gate_min_visible_tracks)),
            "half_angle_deg": round(half_angle, 3),
            "min_forward_m": round(min_forward, 3),
            "max_distance_m": None if math.isinf(max_distance) else round(max_distance, 3),
            "state_gate": {
                "range_m": state_range,
                "bearing_deg": state_bearing,
                "max_distance_m": None if math.isinf(state_max_distance) else round(state_max_distance, 3),
                "in_fov": bool(state_in_fov),
                "reject_reasons": state_reasons,
                "fallback_enabled": bool(self.llm_fov_gate_state_fallback_enable),
            },
            "state_prompt_target": state_prompt_target,
            "state_checked_target": state_checked,
            "soft_gate": {
                "enabled": bool(self.llm_fov_gate_soft_enable),
                "after_s": max(0.0, float(self.llm_fov_gate_soft_after_s)),
                "max_targets": max(1, int(self.llm_fov_gate_soft_max_targets)),
                "candidate_count": len(soft_prompt_targets),
            },
            "soft_prompt_targets": soft_prompt_targets,
            "soft_checked_targets": soft_checked_targets,
            "require_all_tracks": bool(self.llm_fov_gate_require_all_tracks),
            "targets": checked,
            "prompt_targets": prompt_targets if prompt_targets else targets,
        }

    def _vla_fov_gate_ready(self, now):
        diag = self._vla_fov_gate_diag()
        self._last_fov_gate_diag = diag
        self._last_fov_gate_call_reason = ""

        if not self.llm_fov_gate_enable:
            self._fov_gate_first_track_t = None
            return True, "", diag

        def _state_fallback_ready(wait_s=0.0):
            if not self.llm_fov_gate_state_fallback_enable:
                return False
            state_gate = diag.get("state_gate", {})
            if not (isinstance(state_gate, dict) and bool(state_gate.get("in_fov", False))):
                return False
            if float(wait_s) < max(0.0, float(self.llm_fov_gate_state_fallback_after_s)):
                return False
            state_target = diag.get("state_prompt_target")
            if not isinstance(state_target, dict):
                return False
            checked_target = diag.get("state_checked_target")
            if isinstance(checked_target, dict):
                diag["targets"] = [checked_target]
            diag["prompt_targets"] = [state_target]
            diag["visible_count"] = max(1, int(diag.get("visible_count", 0)))
            diag["any_in_fov"] = True
            diag["state_fallback_used"] = True
            return True

        def _soft_fov_ready(wait_s=0.0):
            if not self.llm_fov_gate_soft_enable:
                return False
            if float(wait_s) < max(0.0, float(self.llm_fov_gate_soft_after_s)):
                return False
            prompt_targets = diag.get("soft_prompt_targets")
            checked_targets = diag.get("soft_checked_targets")
            if not (isinstance(prompt_targets, list) and prompt_targets):
                return False
            diag["prompt_targets"] = prompt_targets
            if isinstance(checked_targets, list) and checked_targets:
                diag["targets"] = checked_targets
            diag["visible_count"] = max(1, int(diag.get("visible_count", 0)))
            diag["any_in_fov"] = True
            diag["soft_fov_used"] = True
            return True

        track_count = int(diag.get("track_count", 0))
        if track_count <= 0:
            if _state_fallback_ready(0.0):
                self._fov_gate_first_track_t = None
                self._last_fov_gate_call_reason = "fov_state_fallback_ready"
                return True, "fov_state_fallback_ready", diag
            if _soft_fov_ready(0.0):
                self._fov_gate_first_track_t = None
                self._last_fov_gate_call_reason = "fov_soft_tracks_ready"
                return True, "fov_soft_tracks_ready", diag
            self._fov_gate_first_track_t = None
            return False, "skip_no_fov_tracks", diag

        if self._fov_gate_first_track_t is None:
            self._fov_gate_first_track_t = now

        wait_s = 0.0
        try:
            wait_s = (now - self._fov_gate_first_track_t).to_sec()
        except Exception:
            wait_s = 0.0
        diag["wait_s"] = round(max(0.0, wait_s), 3)

        if int(diag.get("visible_count", 0)) < max(1, int(self.llm_fov_gate_min_visible_tracks)):
            if _state_fallback_ready(wait_s):
                self._last_fov_gate_call_reason = "fov_state_fallback_ready"
                return True, "fov_state_fallback_ready", diag
            if _soft_fov_ready(wait_s):
                self._last_fov_gate_call_reason = "fov_soft_tracks_ready"
                return True, "fov_soft_tracks_ready", diag
            return False, "skip_tracks_not_in_fov", diag

        if bool(diag.get("all_in_fov", False)):
            self._last_fov_gate_call_reason = "fov_all_tracks_ready"
            return True, "fov_all_tracks_ready", diag

        if (not self.llm_fov_gate_require_all_tracks) and bool(diag.get("any_in_fov", False)):
            self._last_fov_gate_call_reason = "fov_partial_tracks_ready"
            return True, "fov_partial_tracks_ready", diag

        if (
            bool(diag.get("any_in_fov", False))
            and float(self.llm_fov_gate_partial_after_s) >= 0.0
            and wait_s >= float(self.llm_fov_gate_partial_after_s)
        ):
            self._last_fov_gate_call_reason = "fov_partial_tracks_ready"
            return True, "fov_partial_tracks_ready", diag

        if _soft_fov_ready(wait_s):
            self._last_fov_gate_call_reason = "fov_soft_tracks_ready"
            return True, "fov_soft_tracks_ready", diag

        return False, "skip_tracks_not_in_fov", diag

    def _fov_unknown_track_gate_ready(self, fov_gate_diag):
        if not self.llm_require_unknown_track_in_fov:
            return True, {
                "enabled": False,
                "visible_ids": [],
                "unknown_visible_ids": [],
                "known_visible_ids": [],
            }

        targets = []
        if isinstance(fov_gate_diag, dict):
            raw_targets = fov_gate_diag.get("targets", [])
            if bool(fov_gate_diag.get("soft_fov_used", False)):
                raw_targets = fov_gate_diag.get("soft_checked_targets") or raw_targets
            if isinstance(raw_targets, list):
                targets = raw_targets

        visible_ids = []
        for t in targets:
            if not isinstance(t, dict) or not bool(t.get("in_fov", False)):
                continue
            target_id = str(t.get("target_id", t.get("id", ""))).strip()
            if target_id:
                visible_ids.append(target_id)

        expected_ids = self._current_target_ids()
        cls_by_id = {}
        last_cls = self._last_cls if isinstance(self._last_cls, dict) else {}
        raw_items = last_cls.get("track_classifications", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                raw_tid = str(item.get("target_id", "")).strip()
                if not raw_tid:
                    continue
                vessel_type = self._canonical_vessel_type(item.get("vessel_type", "Unknown"))
                if vessel_type == "Unknown":
                    continue
                cls_by_id[raw_tid] = vessel_type
                normalized_tid = self._normalize_target_id(raw_tid, expected_ids)
                if normalized_tid:
                    cls_by_id[normalized_tid] = vessel_type

        unknown_ids = []
        known_ids = []
        for target_id in visible_ids:
            vessel_type = cls_by_id.get(target_id)
            normalized_tid = self._normalize_target_id(target_id, expected_ids)
            if vessel_type is None and normalized_tid:
                vessel_type = cls_by_id.get(normalized_tid)
            if vessel_type is None or self._canonical_vessel_type(vessel_type) == "Unknown":
                unknown_ids.append(target_id)
            else:
                known_ids.append(target_id)

        diag = {
            "enabled": True,
            "visible_ids": visible_ids,
            "unknown_visible_ids": unknown_ids,
            "known_visible_ids": known_ids,
            "classification_source": last_cls.get("source", "none"),
        }
        return len(unknown_ids) > 0, diag

    def _llm_image_input_active(self):
        return bool(self.llm_use_image_input and self.llm_backend in ("http", "dinov2"))

    def _pointcloud_trigger_distance(self):
        vals = []
        for t in self._semantic_perception_targets():
            if not isinstance(t, dict):
                continue
            d = self._safe_float(t.get("range_m"), None)
            if d is not None and d > 0.0:
                vals.append(d)

        # fallback to state-based pointcloud range
        s = self._state or {}
        d_state = self._safe_float(s.get("range_m"), None)
        if d_state is not None and d_state > 0.0:
            vals.append(d_state)
        pcd = s.get("pointcloud_tracking_debug", {}) or {}
        d_sel = self._safe_float(pcd.get("selected_range_m"), None) if isinstance(pcd, dict) else None
        if d_sel is not None and d_sel > 0.0:
            vals.append(d_sel)

        if not vals:
            return None
        return min(vals)

    def _semantic_pretrigger_active(self, d_min=None):
        if not self.llm_pretrigger_enable:
            return False
        if self._trigger:
            return False
        if d_min is None:
            d_min = self._pointcloud_trigger_distance()
        if d_min is None:
            return False
        return float(d_min) <= max(0.1, float(self.llm_pretrigger_distance_m))

    def _image_cb(self, msg):
        recv_time = rospy.Time.now()
        self._latest_image_msg = msg
        stamp = msg.header.stamp
        try:
            if stamp is None or stamp.to_sec() <= 0.0:
                stamp = recv_time
        except Exception:
            stamp = recv_time
        self._latest_image_stamp = stamp
        self._image_buffer.append(
            {
                "stamp": stamp,
                "msg": msg,
                "recv_time": recv_time,
            }
        )
        self._prune_image_buffer(recv_time)
        self._record_camera_video_frame_if_needed(msg, stamp, recv_time)

    def _pointcloud_cb(self, msg):
        self._store_bev_pointcloud_msg(msg)

    def _store_bev_pointcloud_msg(self, msg):
        stamp = msg.header.stamp
        recv_time = rospy.Time.now()
        try:
            if stamp is None or stamp.to_sec() <= 0.0:
                stamp = recv_time
        except Exception:
            stamp = recv_time
        self._latest_pointcloud_msg = msg
        if int(getattr(self, "_pointcloud_frame_count", 0) or 0) <= 0:
            self._first_pointcloud_recv_time = recv_time
        self._latest_pointcloud_recv_time = recv_time
        self._pointcloud_frame_count = int(getattr(self, "_pointcloud_frame_count", 0) or 0) + 1
        self._latest_pointcloud_stamp = stamp
        if self._use_bev_image() and self._active_bev_sensor_name() == str(sensor_name):
            self._record_bev_video_frame_if_needed(recv_time)

    def _record_bev_video_frame_if_needed(self, now=None):
        if not (self._use_bev_image() and self.vla_bev_video_enable):
            return
        now = now or rospy.Time.now()
        stamp_s = float(self._active_pointcloud_stamp().to_sec())
        last_stamp_s = getattr(self, "_bev_video_last_seen_stamp_s", None)
        if last_stamp_s is None or abs(stamp_s - float(last_stamp_s)) > 1.0e-6:
            self._bev_video_seen_stamp_count = int(getattr(self, "_bev_video_seen_stamp_count", 0) or 0) + 1
            self._bev_video_last_seen_stamp_s = stamp_s
            if getattr(self, "_bev_video_first_stamp_s", None) is None:
                self._bev_video_first_stamp_s = stamp_s
        if int(getattr(self, "_bev_video_seen_stamp_count", 0) or 0) <= int(self.vla_bev_video_skip_initial_frames):
            return
        try:
            last_t = getattr(self, "_last_bev_video_frame_record_t", rospy.Time(0))
            if last_t.to_sec() > 0.0 and (now - last_t).to_sec() < self.vla_bev_video_interval_s:
                return
        except Exception:
            pass
        if not self._latest_pointcloud_available_for_capture():
            return
        try:
            os.makedirs(self.vla_snapshot_dir, exist_ok=True)
            png, diag = self._render_pointcloud_bev_png_bytes(debug_overlay=False)
            self._bev_video_seq += 1
            path = os.path.join(
                self.vla_snapshot_dir,
                "%s_bev_video_%s_%06d.png" % (
                    self._active_bev_sensor_name(),
                    self._snapshot_trial_token(),
                    self._bev_video_seq,
                ),
            )
            with open(path, "wb") as f:
                f.write(png)
            record = {
                "path": path,
                "url": "file://" + path,
                "stamp_s": stamp_s,
                "record_time_s": now.to_sec(),
                "seq": int(self._bev_video_seq),
                "diag": diag,
            }
            self._bev_video_frames.append(record)
            while len(self._bev_video_frames) > int(self.vla_bev_video_buffer_max_frames):
                self._bev_video_frames.popleft()
            self._last_bev_video_frame_record_t = now
        except Exception as e:
            if self.debug_enable:
                rospy.logwarn_throttle(
                    5.0,
                    "[colregs_llm_decision] failed to record BEV video frame: %s",
                    str(e)[:200],
                )

    @staticmethod
    def _png_chunk(tag, payload):
        head = struct.pack("!I", len(payload)) + tag + payload
        crc = zlib.crc32(tag)
        crc = zlib.crc32(payload, crc)
        return head + struct.pack("!I", crc & 0xFFFFFFFF)

    def _image_msg_to_png_bytes(self, msg):
        enc = str(msg.encoding or "").lower()
        w = int(msg.width)
        h = int(msg.height)
        if w <= 0 or h <= 0:
            raise ValueError("vla_image_invalid_size")
        raw = bytes(msg.data or b"")
        step = int(msg.step) if int(msg.step) > 0 else 0
        if step <= 0:
            raise ValueError("vla_image_invalid_step")

        if enc == "mono8":
            color_type = 0
            channels = 1
            row_bytes = w
            expected_step = row_bytes
            convert_row = None
        elif enc in ("rgb8", "bgr8"):
            color_type = 2
            channels = 3
            row_bytes = w * 3
            expected_step = row_bytes

            def convert_row(r):
                if enc == "rgb8":
                    return r
                out = bytearray(row_bytes)
                for i in range(0, row_bytes, 3):
                    out[i] = r[i + 2]
                    out[i + 1] = r[i + 1]
                    out[i + 2] = r[i]
                return bytes(out)

        elif enc in ("rgba8", "bgra8"):
            color_type = 6
            channels = 4
            row_bytes = w * 4
            expected_step = row_bytes

            def convert_row(r):
                if enc == "rgba8":
                    return r
                out = bytearray(row_bytes)
                for i in range(0, row_bytes, 4):
                    out[i] = r[i + 2]
                    out[i + 1] = r[i + 1]
                    out[i + 2] = r[i]
                    out[i + 3] = r[i + 3]
                return bytes(out)

        else:
            raise ValueError("vla_image_unsupported_encoding:%s" % enc)

        min_bytes = step * h
        if len(raw) < min_bytes:
            raise ValueError("vla_image_data_too_short")
        if step < expected_step:
            raise ValueError("vla_image_step_too_small")

        scan = bytearray()
        for y in range(h):
            base = y * step
            row = raw[base : base + expected_step]
            if convert_row is not None:
                row = convert_row(row)
            scan.append(0)
            scan.extend(row)

        ihdr = struct.pack("!IIBBBBB", w, h, 8, color_type, 0, 0, 0)
        comp = zlib.compress(bytes(scan), level=9)
        png = b"\x89PNG\r\n\x1a\n"
        png += self._png_chunk(b"IHDR", ihdr)
        png += self._png_chunk(b"IDAT", comp)
        png += self._png_chunk(b"IEND", b"")
        return png

    def _pointcloud_msg_to_lossless_png_bytes(self, msg):
        raw = bytes(getattr(msg, "data", b"") or b"")
        data_len = len(raw)
        point_step = int(getattr(msg, "point_step", 0) or 0)
        if point_step > 0 and point_step <= 8192:
            image_width = point_step
        else:
            image_width = min(4096, max(1, data_len))
        image_height = max(1, int(math.ceil(float(max(1, data_len)) / float(image_width))))
        padded_len = image_width * image_height
        payload = raw + (b"\x00" * max(0, padded_len - data_len))

        img = PilImage.frombytes("L", (image_width, image_height), payload)
        stamp = getattr(getattr(msg, "header", None), "stamp", rospy.Time(0))
        metadata = {
            "format": "sensor_msgs/PointCloud2.data.raw_bytes.v1",
            "data_len": data_len,
            "png_width": image_width,
            "png_height": image_height,
            "pointcloud_width": int(getattr(msg, "width", 0) or 0),
            "pointcloud_height": int(getattr(msg, "height", 0) or 0),
            "point_step": point_step,
            "row_step": int(getattr(msg, "row_step", 0) or 0),
            "is_bigendian": bool(getattr(msg, "is_bigendian", False)),
            "is_dense": bool(getattr(msg, "is_dense", False)),
            "frame_id": str(getattr(getattr(msg, "header", None), "frame_id", "") or ""),
            "stamp_s": float(stamp.to_sec()) if hasattr(stamp, "to_sec") else 0.0,
            "fields": [
                {
                    "name": str(getattr(field, "name", "") or ""),
                    "offset": int(getattr(field, "offset", 0) or 0),
                    "datatype": int(getattr(field, "datatype", 0) or 0),
                    "count": int(getattr(field, "count", 0) or 0),
                }
                for field in list(getattr(msg, "fields", []) or [])
            ],
        }
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("bev_format", metadata["format"])
        pnginfo.add_text(
            "bev_metadata_json",
            json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
        )
        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=pnginfo, compress_level=9)
        return out.getvalue(), metadata

    def _latest_pointcloud_lossless_png_bytes(self):
        msg = self._active_pointcloud_msg()
        if msg is None:
            raise ValueError("pointcloud_unavailable")
        if not self._latest_pointcloud_available_for_capture():
            raise ValueError("pointcloud_stale")
        return self._pointcloud_msg_to_lossless_png_bytes(msg)

    def _image_file_to_data_url(self, path):
        with open(path, "rb") as f:
            payload = f.read()
        if not payload:
            raise ValueError("vla_image_file_empty")
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png"
        b64 = base64.b64encode(payload).decode("ascii")
        return "data:%s;base64,%s" % (mime, b64)

    def _recent_bev_video_frame_records(self):
        if not (self._use_bev_image() and self.vla_bev_video_enable):
            return []
        self._record_bev_video_frame_if_needed(rospy.Time.now())
        now_s = rospy.Time.now().to_sec()
        min_s = now_s - float(self.vla_bev_video_window_s)
        records = [
            rec
            for rec in list(getattr(self, "_bev_video_frames", []))
            if isinstance(rec, dict)
            and rec.get("path")
            and os.path.isfile(str(rec.get("path")))
            and float(rec.get("record_time_s", rec.get("stamp_s", 0.0)) or 0.0) >= min_s
        ]
        if len(records) < int(self.vla_bev_video_frame_count):
            records = [
                rec
                for rec in list(getattr(self, "_bev_video_frames", []))
                if isinstance(rec, dict)
                and rec.get("path")
                and os.path.isfile(str(rec.get("path")))
            ]
        return records[-int(self.vla_bev_video_frame_count) :]

    def _encode_bev_frames_as_video_jpeg_url(self, frame_paths):
        encoded_frames = []
        for path in frame_paths:
            with PilImage.open(path) as img:
                out = io.BytesIO()
                img.convert("RGB").save(out, format="JPEG", quality=92)
                encoded_frames.append(base64.b64encode(out.getvalue()).decode("ascii"))
        return "data:video/jpeg;base64,%s" % ",".join(encoded_frames)

    def _build_vla_video_url(self):
        records = self._recent_bev_video_frame_records()
        if len(records) < int(self.vla_bev_video_frame_count):
            self._last_llm_video_frame_paths = []
            self._last_llm_video_frame_records = []
            return "", {}
        paths = [str(rec.get("path")) for rec in records]
        video_url = self._encode_bev_frames_as_video_jpeg_url(paths)
        frame_indices = list(range(len(paths)))
        duration = max(
            self.vla_bev_video_interval_s * max(0, len(paths) - 1),
            self.vla_bev_video_interval_s,
        )
        media_io_kwargs = {
            "video": {
                "fps": 1.0 / max(self.vla_bev_video_interval_s, 1.0e-6),
                "frames_indices": frame_indices,
                "total_num_frames": len(paths),
                "duration": duration,
                "do_sample_frames": False,
            }
        }
        self._last_llm_image_source = self._bev_source_label("video")
        self._last_llm_video_frame_paths = paths
        self._last_llm_video_frame_records = [
            {
                "path": str(rec.get("path")),
                "stamp_s": rec.get("stamp_s"),
                "record_time_s": rec.get("record_time_s"),
                "frame_index": idx,
            }
            for idx, rec in enumerate(records)
            if isinstance(rec, dict)
        ]
        return video_url, media_io_kwargs

    def _camera_video_media_io_kwargs(self, records, frame_count):
        times = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            t = self._safe_float(rec.get("record_time_s"), None)
            if t is None:
                t = self._safe_float(rec.get("stamp_s"), None)
            if t is not None:
                times.append(float(t))
        duration = 1.0
        if len(times) >= 2:
            duration = max(1.0e-3, times[-1] - times[0])
        fps = 1.0 if frame_count <= 1 else float(frame_count - 1) / max(duration, 1.0e-3)
        frame_indices = list(range(int(frame_count)))
        return {
            "video": {
                "fps": fps,
                "frames_indices": frame_indices,
                "total_num_frames": int(frame_count),
                "duration": duration,
                "do_sample_frames": False,
            }
        }

    def _valid_event_camera_video_records(self):
        required = int(self.vla_camera_video_frame_count)
        records = list(getattr(self, "_event_camera_video_frame_records", []) or [])
        if len(records) < required:
            return []
        valid = [
            rec
            for rec in records[-required:]
            if isinstance(rec, dict)
            and rec.get("path")
            and os.path.isfile(str(rec.get("path")))
        ]
        return valid if len(valid) >= required else []

    def _build_camera_video_url(self):
        required = int(self.vla_camera_video_frame_count)
        records = self._valid_event_camera_video_records()
        if not records:
            buffer_records = self._recent_camera_video_frame_records(required)
            if len(buffer_records) < required:
                self._last_llm_video_frame_paths = []
                self._last_llm_video_frame_records = []
                return "", {}
            snapshot_token = "%s_call_%03d" % (
                self._snapshot_trial_token(),
                max(1, int(getattr(self, "_llm_call_seq", 0) or 1)),
            )
            records = self._save_camera_video_frame_records(buffer_records, snapshot_token)
            self._event_camera_video_frame_records = records
            self._event_camera_video_frame_paths = [
                str(rec.get("path"))
                for rec in records
                if isinstance(rec, dict) and rec.get("path")
            ]
        if len(records) < required:
            self._last_llm_video_frame_paths = []
            self._last_llm_video_frame_records = []
            return "", {}
        paths = [str(rec.get("path")) for rec in records[-required:]]
        video_url = self._encode_bev_frames_as_video_jpeg_url(paths)
        media_io_kwargs = self._camera_video_media_io_kwargs(records[-required:], len(paths))
        self._last_llm_image_source = "camera_video"
        self._last_llm_video_frame_paths = paths
        self._last_llm_video_frame_records = [
            {
                "path": str(rec.get("path")),
                "stamp_s": rec.get("stamp_s"),
                "record_time_s": rec.get("record_time_s"),
                "frame_index": idx,
            }
            for idx, rec in enumerate(records[-required:])
            if isinstance(rec, dict)
        ]
        return video_url, media_io_kwargs

    def _build_vla_image_data_url(self, event_image_path=None):
        if self._use_global_trajectory_image() and self._event_image_path and os.path.isfile(self._event_image_path):
            data_url = self._image_file_to_data_url(self._event_image_path)
            self._last_llm_image_source = "ais_trajectory_base64" if self._use_ais_trajectory_image() else "global_trajectory_base64"
            self._event_image_url = "file://" + self._event_image_path
            return data_url
        if self._use_global_trajectory_image():
            png, _ = self._render_global_trajectory_png_bytes()
            if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                raise ValueError("vla_global_image_too_large:%d" % len(png))
            b64 = base64.b64encode(png).decode("ascii")
            self._last_llm_image_source = "ais_trajectory_base64" if self._use_ais_trajectory_image() else "global_trajectory_base64"
            return "data:image/png;base64,%s" % b64
        if self._use_bev_image() and self._event_input_image_path and os.path.isfile(self._event_input_image_path):
            data_url = self._image_file_to_data_url(self._event_input_image_path)
            self._last_llm_image_source = self._bev_source_label("clean_base64")
            self._event_input_image_url = "file://" + self._event_input_image_path
            return data_url
        path_override = str(event_image_path or "").strip()
        if path_override and os.path.isfile(path_override):
            data_url = self._image_file_to_data_url(path_override)
            self._last_llm_image_source = self._bev_source_label("base64") if self._use_bev_image() else "event_image_path"
            self._event_image_url = "file://" + path_override
            return data_url
        if self._event_image_path and os.path.isfile(self._event_image_path):
            data_url = self._image_file_to_data_url(self._event_image_path)
            self._last_llm_image_source = self._bev_source_label("base64") if self._use_bev_image() else "event_image_path"
            self._event_image_url = "file://" + self._event_image_path
            return data_url

        if self._use_bev_image() and self._latest_pointcloud_available_for_capture():
            png, _ = self._render_pointcloud_bev_png_bytes(
                fov_gate_diag=getattr(self, "_last_fov_gate_diag", {}),
                debug_overlay=False,
            )
            if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                raise ValueError("vla_image_too_large:%d" % len(png))
            b64 = base64.b64encode(png).decode("ascii")
            self._last_llm_image_source = self._bev_source_label("clean_base64")
            return "data:image/png;base64,%s" % b64
        if self._use_bev_image() and not self.vla_bev_fallback_to_camera:
            self._last_llm_image_source = ""
            raise ValueError("vla_%s_unavailable" % self.vla_image_source)

        now = rospy.Time.now()
        msg = self._latest_image_msg
        if msg is not None:
            age_s = (now - self._latest_image_stamp).to_sec()
            if age_s <= max(0.0, self.vla_image_max_age_s):
                png = self._image_msg_to_png_bytes(msg)
                if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                    raise ValueError("vla_image_too_large:%d" % len(png))
                b64 = base64.b64encode(png).decode("ascii")
                self._last_llm_image_source = "ros_image_topic"
                return "data:image/png;base64,%s" % b64

        if self.vla_image_path:
            path = os.path.expanduser(self.vla_image_path)
            if os.path.isfile(path):
                data_url = self._image_file_to_data_url(path)
                if self.vla_image_max_bytes > 0:
                    payload_len = len(data_url) * 3 // 4
                    if payload_len > self.vla_image_max_bytes:
                        raise ValueError("vla_image_too_large:%d" % payload_len)
                self._last_llm_image_source = "image_path"
                return data_url

        if self.vla_image_url:
            self._last_llm_image_source = "image_url"
            return self.vla_image_url

        self._last_llm_image_source = ""
        raise ValueError("vla_image_unavailable")

    def _build_backend_image_pil(self, event_image_path=None):
        if self._use_global_trajectory_image() and self._event_image_path and os.path.isfile(self._event_image_path):
            with PilImage.open(self._event_image_path) as img:
                self._last_llm_image_source = "ais_trajectory_base64" if self._use_ais_trajectory_image() else "global_trajectory_base64"
                self._event_image_url = "file://" + self._event_image_path
                return img.convert("RGB")
        if self._use_global_trajectory_image():
            png, _ = self._render_global_trajectory_png_bytes()
            if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                raise ValueError("vla_global_image_too_large:%d" % len(png))
            self._last_llm_image_source = "ais_trajectory_base64" if self._use_ais_trajectory_image() else "global_trajectory_base64"
            return PilImage.open(io.BytesIO(png)).convert("RGB")
        if self._use_bev_image() and self._event_input_image_path and os.path.isfile(self._event_input_image_path):
            with PilImage.open(self._event_input_image_path) as img:
                self._last_llm_image_source = self._bev_source_label("clean_base64")
                self._event_input_image_url = "file://" + self._event_input_image_path
                return img.convert("RGB")
        path_override = str(event_image_path or "").strip()
        if path_override and os.path.isfile(path_override):
            with PilImage.open(path_override) as img:
                self._last_llm_image_source = self._bev_source_label("base64") if self._use_bev_image() else "event_image_path"
                self._event_image_url = "file://" + path_override
                return img.convert("RGB")
        if self._event_image_path and os.path.isfile(self._event_image_path):
            with PilImage.open(self._event_image_path) as img:
                self._last_llm_image_source = self._bev_source_label("base64") if self._use_bev_image() else "event_image_path"
                self._event_image_url = "file://" + self._event_image_path
                return img.convert("RGB")

        if self._use_bev_image() and self._latest_pointcloud_available_for_capture():
            png, _ = self._render_pointcloud_bev_png_bytes(
                fov_gate_diag=getattr(self, "_last_fov_gate_diag", {}),
                debug_overlay=False,
            )
            if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                raise ValueError("vla_image_too_large:%d" % len(png))
            self._last_llm_image_source = self._bev_source_label("clean_base64")
            return PilImage.open(io.BytesIO(png)).convert("RGB")
        if self._use_bev_image() and not self.vla_bev_fallback_to_camera:
            self._last_llm_image_source = ""
            raise ValueError("vla_%s_unavailable" % self.vla_image_source)

        now = rospy.Time.now()
        msg = self._latest_image_msg
        if msg is not None:
            age_s = (now - self._latest_image_stamp).to_sec()
            if age_s <= max(0.0, self.vla_image_max_age_s):
                png = self._image_msg_to_png_bytes(msg)
                if self.vla_image_max_bytes > 0 and len(png) > self.vla_image_max_bytes:
                    raise ValueError("vla_image_too_large:%d" % len(png))
                self._last_llm_image_source = "ros_image_topic"
                return PilImage.open(io.BytesIO(png)).convert("RGB")

        if self.vla_image_path:
            path = os.path.expanduser(self.vla_image_path)
            if os.path.isfile(path):
                with PilImage.open(path) as img:
                    self._last_llm_image_source = "image_path"
                    return img.convert("RGB")

        self._last_llm_image_source = ""
        raise ValueError("vla_image_unavailable")

    def _run_dinov2_classifier(self, call_context=None):
        call_context = call_context if isinstance(call_context, dict) else {}
        image = self._build_backend_image_pil(call_context.get("event_image_path", ""))
        targets = call_context.get("targets")
        if not isinstance(targets, list):
            targets = self._build_targets_prompt_context()
        result = self._dinov2_classifier.classify(image, targets)
        if not isinstance(result, dict):
            raise RuntimeError("dinov2_invalid_result")
        result.setdefault("reasoning", "DINOv2原型匹配")
        result.setdefault("source", self._dinov2_classifier.runtime_backend)
        result.setdefault("trajectory_constraints", {})
        return result

    def _state_snapshot(self):
        s = self._state or {}
        return {
            "range_m": round(float(s.get("range_m", 0.0)), 3),
            "dcpa_m": round(float(s.get("dcpa_m", 0.0)), 3),
            "tcpa_s": round(float(s.get("tcpa_s", 0.0)), 3),
            "bearing_deg": round(float(s.get("relative_bearing_deg", 0.0)), 2),
            "ego_h": round(float(s.get("ego_heading_rad", 0.0)), 3),
            "tgt_h": round(float(s.get("target_heading_rad", 0.0)), 3),
        }

    @staticmethod
    def _norm_pi(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _fallback_classification(self):
        course_action = "KEEP_COURSE"
        speed_action = ""
        action = self._combine_colreg_action(course_action, speed_action)
        weights = self._colreg_weights_for_split(course_action, speed_action)
        weights.update(
            strength=0.5,
            keep_course_bias=1.0,
            speed_scale=1.0,
            clearance_scale=1.0,
            predictive_risk_scale=1.0,
            collision_penalty_scale=1.0,
            target_penalty_scale=1.0,
        )
        return {
            "confidence": 0.25,
            "reasoning": "No VLM signal; keep course",
            "track_classifications": self._fallback_track_classifications(),
            "course_action": course_action,
            "speed_action": speed_action,
            "colreg_action": action,
            "trajectory_constraints": {
                "duration_s": self._clamp(self.default_traj_duration_s, 1.0, 30.0),
                "target_linear_x": self.normal_forward_speed,
                "target_angular_z": 0.0,
                "max_linear_acc": self.default_max_linear_acc,
                "max_angular_acc": self.default_max_angular_acc,
                "min_linear_x": 0.0,
                "course_action": course_action,
                "speed_action": speed_action,
                "colreg_action": action,
                "colreg_weights": weights,
            },
            "source": "fallback",
        }

    def _pending_safety_classification(self):
        course_action = "KEEP_COURSE"
        speed_action = ""

        action = self._combine_colreg_action(course_action, speed_action)
        weights = self._colreg_weights_for_split(course_action, speed_action)
        weights.update(
            strength=0.5,
            keep_course_bias=1.0,
            speed_scale=1.0,
            clearance_scale=1.0,
            predictive_risk_scale=1.0,
            collision_penalty_scale=1.0,
            target_penalty_scale=1.0,
        )

        target_v = self.normal_forward_speed
        target_w = 0.0

        return {
            "confidence": 0.15,
            "reasoning": "Waiting for VLM; keep course",
            "track_classifications": self._fallback_track_classifications(),
            "course_action": course_action,
            "speed_action": speed_action,
            "colreg_action": action,
            "source": "pending_safety_pre_llm",
            "trajectory_constraints": {
                "duration_s": self._clamp(self.pending_safety_duration_s, 1.0, 30.0),
                "target_linear_x": target_v,
                "target_angular_z": target_w,
                "max_linear_acc": min(max(0.02, self.default_max_linear_acc), 0.16),
                "max_angular_acc": min(max(0.05, self.default_max_angular_acc), 0.28),
                "min_linear_x": 0.0,
                "course_action": course_action,
                "speed_action": speed_action,
                "colreg_action": action,
                "colreg_weights": weights,
            },
        }

    @staticmethod
    def _canonical_course_action(action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "STAND_ON": "KEEP_COURSE",
            "STAND_ON_KEEP_COURSE": "KEEP_COURSE",
            "MAINTAIN_COURSE": "KEEP_COURSE",
            "KEEP": "KEEP_COURSE",
            "GIVE_WAY": "TURN_STARBOARD",
            "ALTER_TO_STARBOARD": "TURN_STARBOARD",
            "STARBOARD": "TURN_STARBOARD",
            "RIGHT": "TURN_STARBOARD",
            "TURN_RIGHT": "TURN_STARBOARD",
            "RIGHT_TURN": "TURN_STARBOARD",
            "ALTER_TO_PORT": "TURN_PORT",
            "PORT": "TURN_PORT",
            "LEFT": "TURN_PORT",
            "TURN_LEFT": "TURN_PORT",
            "LEFT_TURN": "TURN_PORT",
        }
        text = aliases.get(text, text)
        allowed = {"KEEP_COURSE", "TURN_STARBOARD", "TURN_PORT"}
        return text if text in allowed else "KEEP_COURSE"

    @staticmethod
    def _canonical_speed_action(action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        if not text:
            return ""
        aliases = {
            "KEEP": "",
            "KEEP_SPEED": "",
            "MAINTAIN_SPEED": "",
            "SLOW": "SLOW_DOWN",
            "REDUCE_SPEED": "SLOW_DOWN",
            "DECELERATE": "SLOW_DOWN",
            "FAST": "SPEED_UP",
            "ACCELERATE": "SPEED_UP",
            "INCREASE_SPEED": "SPEED_UP",
            "STOP": "EMERGENCY_STOP",
            "EMERGENCY": "EMERGENCY_STOP",
            "FULL_STOP": "EMERGENCY_STOP",
        }
        text = aliases.get(text, text)
        allowed = {"", "SLOW_DOWN", "SPEED_UP", "EMERGENCY_STOP"}
        return text if text in allowed else "SLOW_DOWN"

    @staticmethod
    def _action_token(action):
        return str(action or "").strip().upper().replace("-", "_").replace(" ", "_")

    @classmethod
    def _action_field_diag(cls, raw_course_action, raw_speed_action):
        course_token = cls._action_token(raw_course_action)
        speed_token = cls._action_token(raw_speed_action)
        course_allowed = {
            "KEEP_COURSE",
            "STAND_ON",
            "STAND_ON_KEEP_COURSE",
            "MAINTAIN_COURSE",
            "KEEP",
            "GIVE_WAY",
            "ALTER_TO_STARBOARD",
            "STARBOARD",
            "RIGHT",
            "TURN_RIGHT",
            "RIGHT_TURN",
            "TURN_STARBOARD",
            "ALTER_TO_PORT",
            "PORT",
            "LEFT",
            "TURN_LEFT",
            "LEFT_TURN",
            "TURN_PORT",
        }
        speed_allowed = {
            "SLOW_DOWN",
            "SLOW",
            "REDUCE_SPEED",
            "DECELERATE",
            "SPEED_UP",
            "FAST",
            "ACCELERATE",
            "INCREASE_SPEED",
            "EMERGENCY_STOP",
            "STOP",
            "EMERGENCY",
            "FULL_STOP",
        }
        issues = []
        if course_token and course_token not in course_allowed:
            issues.append("invalid_course_action_raw=" + course_token)
        if speed_token and speed_token not in speed_allowed:
            issues.append("invalid_speed_action_raw=" + speed_token)
        if course_token in speed_allowed:
            issues.append("speed_enum_in_course_action")
        if speed_token in course_allowed:
            issues.append("course_enum_in_speed_action")
        return ",".join(issues)

    @classmethod
    def _split_colreg_action(cls, action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        parts = [p for p in re.split(r"[+/,;|]+", text) if p]
        course_action = ""
        speed_action = ""
        for part in parts or [text]:
            course = cls._canonical_course_action(part)
            speed = cls._canonical_speed_action(part)
            if part in ("KEEP_COURSE", "KEEP", "STAND_ON", "TURN_STARBOARD", "STARBOARD", "RIGHT", "TURN_RIGHT", "TURN_PORT", "PORT", "LEFT", "TURN_LEFT"):
                course_action = course
            elif part in ("SLOW_DOWN", "SLOW", "REDUCE_SPEED", "DECELERATE", "SPEED_UP", "FAST", "ACCELERATE", "INCREASE_SPEED", "EMERGENCY_STOP", "STOP", "EMERGENCY", "FULL_STOP"):
                speed_action = speed
        if not course_action:
            legacy = cls._canonical_colreg_action(text)
            if legacy in ("KEEP_COURSE", "TURN_STARBOARD", "TURN_PORT"):
                course_action = legacy
            else:
                course_action = "KEEP_COURSE"
        if not speed_action:
            legacy = cls._canonical_colreg_action(text)
            if legacy in ("SLOW_DOWN", "SPEED_UP", "EMERGENCY_STOP"):
                speed_action = legacy
            else:
                speed_action = ""
        return course_action, speed_action

    @staticmethod
    def _combine_colreg_action(course_action, speed_action):
        course = ColregsLlmDecisionNode._canonical_course_action(course_action)
        speed = ColregsLlmDecisionNode._canonical_speed_action(speed_action)
        return course if not speed else "%s+%s" % (course, speed)

    @staticmethod
    def _canonical_colreg_action(action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "STAND_ON": "KEEP_COURSE",
            "STAND_ON_KEEP_COURSE": "KEEP_COURSE",
            "MAINTAIN_COURSE": "KEEP_COURSE",
            "KEEP_SPEED": "KEEP_COURSE",
            "GIVE_WAY": "TURN_STARBOARD",
            "ALTER_TO_STARBOARD": "TURN_STARBOARD",
            "STARBOARD": "TURN_STARBOARD",
            "RIGHT": "TURN_STARBOARD",
            "TURN_RIGHT": "TURN_STARBOARD",
            "ALTER_TO_PORT": "TURN_PORT",
            "PORT": "TURN_PORT",
            "LEFT": "TURN_PORT",
            "TURN_LEFT": "TURN_PORT",
            "REDUCE_SPEED": "SLOW_DOWN",
            "DECELERATE": "SLOW_DOWN",
            "STOP": "EMERGENCY_STOP",
            "FULL_STOP": "EMERGENCY_STOP",
            "ACCELERATE": "SPEED_UP",
        }
        text = aliases.get(text, text)
        allowed = {
            "KEEP_COURSE",
            "TURN_STARBOARD",
            "TURN_PORT",
            "SLOW_DOWN",
            "SPEED_UP",
            "EMERGENCY_STOP",
        }
        return text if text in allowed else "TURN_STARBOARD"

    def _colreg_weights_for_action(self, action):
        action = self._canonical_colreg_action(action)
        weights = {
            "action": action,
            "strength": 1.0,
            "turn_bias": 0.0,
            "keep_course_bias": 0.0,
            "speed_scale": 1.0,
            "clearance_scale": 1.0,
            "predictive_risk_scale": 1.0,
            "collision_penalty_scale": 1.0,
            "target_penalty_scale": 1.0,
        }
        if action == "KEEP_COURSE":
            weights.update(
                strength=0.0,
                keep_course_bias=0.0,
                clearance_scale=1.0,
                predictive_risk_scale=1.0,
                collision_penalty_scale=1.0,
                target_penalty_scale=1.0,
            )
        elif action == "TURN_STARBOARD":
            weights.update(
                strength=1.55,
                turn_bias=-1.25,
                speed_scale=0.72,
                clearance_scale=1.55,
                predictive_risk_scale=1.55,
                collision_penalty_scale=1.45,
                target_penalty_scale=1.35,
            )
        elif action == "TURN_PORT":
            weights.update(
                strength=1.05,
                turn_bias=1.0,
                speed_scale=0.82,
                clearance_scale=1.25,
                predictive_risk_scale=1.20,
                collision_penalty_scale=1.15,
                target_penalty_scale=1.12,
            )
        elif action == "SLOW_DOWN":
            weights.update(
                strength=1.0,
                speed_scale=0.45,
                clearance_scale=1.20,
                predictive_risk_scale=1.15,
                collision_penalty_scale=1.10,
            )
        elif action == "SPEED_UP":
            weights.update(
                strength=0.9,
                speed_scale=1.15,
                clearance_scale=1.05,
                predictive_risk_scale=1.05,
            )
        elif action == "EMERGENCY_STOP":
            weights.update(
                strength=1.5,
                speed_scale=0.05,
                clearance_scale=1.45,
                predictive_risk_scale=1.45,
                collision_penalty_scale=1.35,
                target_penalty_scale=1.25,
            )
        return weights

    def _colreg_weights_for_split(self, course_action, speed_action):
        course_action = self._canonical_course_action(course_action)
        speed_action = self._canonical_speed_action(speed_action)
        if course_action == "KEEP_COURSE":
            speed_action = ""
        weights = {
            "action": self._combine_colreg_action(course_action, speed_action),
            "course_action": course_action,
            "speed_action": speed_action,
            "strength": 1.0,
            "turn_bias": 0.0,
            "keep_course_bias": 0.0,
            "speed_scale": 1.0,
            "clearance_scale": 1.0,
            "predictive_risk_scale": 1.0,
            "collision_penalty_scale": 1.0,
            "target_penalty_scale": 1.0,
        }
        if course_action == "KEEP_COURSE":
            weights.update(
                strength=0.0,
                keep_course_bias=0.0,
                clearance_scale=1.0,
                predictive_risk_scale=1.0,
                collision_penalty_scale=1.0,
                target_penalty_scale=1.0,
            )
        elif course_action == "TURN_STARBOARD":
            weights.update(
                strength=1.55,
                turn_bias=-1.25,
                clearance_scale=1.55,
                predictive_risk_scale=1.55,
                collision_penalty_scale=1.45,
                target_penalty_scale=1.35,
            )
        elif course_action == "TURN_PORT":
            weights.update(
                strength=1.05,
                turn_bias=1.0,
                clearance_scale=1.25,
                predictive_risk_scale=1.20,
                collision_penalty_scale=1.15,
                target_penalty_scale=1.12,
            )

        if speed_action == "SLOW_DOWN":
            weights.update(
                speed_scale=0.45,
                clearance_scale=max(float(weights.get("clearance_scale", 1.0)), 1.20),
                predictive_risk_scale=max(float(weights.get("predictive_risk_scale", 1.0)), 1.15),
                collision_penalty_scale=max(float(weights.get("collision_penalty_scale", 1.0)), 1.10),
            )
        elif speed_action == "SPEED_UP":
            weights.update(
                speed_scale=1.15,
                clearance_scale=max(float(weights.get("clearance_scale", 1.0)), 1.05),
                predictive_risk_scale=max(float(weights.get("predictive_risk_scale", 1.0)), 1.05),
            )
        elif speed_action == "EMERGENCY_STOP":
            weights.update(
                strength=max(float(weights.get("strength", 1.0)), 1.5),
                speed_scale=0.05,
                clearance_scale=max(float(weights.get("clearance_scale", 1.0)), 1.45),
                predictive_risk_scale=max(float(weights.get("predictive_risk_scale", 1.0)), 1.45),
                collision_penalty_scale=max(float(weights.get("collision_penalty_scale", 1.0)), 1.35),
                target_penalty_scale=max(float(weights.get("target_penalty_scale", 1.0)), 1.25),
            )
        return weights

    def _previous_vlm_action_context(self):
        last_cls = getattr(self, "_last_cls", {}) or {}
        if not isinstance(last_cls, dict):
            last_cls = {}
        constraints = last_cls.get("trajectory_constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        course_action = str(
            last_cls.get("course_action", constraints.get("course_action", "")) or ""
        ).strip()
        speed_action = str(
            last_cls.get("speed_action", constraints.get("speed_action", "")) or ""
        ).strip()
        if not (course_action or speed_action):
            return {"available": False}
        return {
            "available": True,
            "course_action": course_action,
            "speed_action": speed_action,
        }

    def _build_prompt(self, targets_override=None):
        targets = targets_override if isinstance(targets_override, list) else self._build_targets_prompt_context()
        target_ids = [
            str(t.get("target_id", "track_%d" % idx))
            for idx, t in enumerate(targets, start=1)
            if isinstance(t, dict)
        ]
        if not target_ids:
            target_ids = self._current_target_ids()
        vlm_targets = []
        for idx, t in enumerate(targets, start=1):
            if not isinstance(t, dict):
                continue
            target_id = t.get("target_id", "track_%d" % idx)
            if self.vla_prompt_mode == "rgb_only_vlm":
                # Deliberately omit track geometry in this ablation. Keep only stable ids
                # so downstream normalization can match model output to active tracks.
                item = {"target_id": target_id}
                vlm_targets.append(item)
            elif self.vla_prompt_mode == "track_overlay":
                item = {
                    "target_id": target_id,
                    "track_input_source": "sensor_estimate",
                    "relative_angle_deg": t.get("relative_angle_deg"),
                    "relative_sector": t.get("relative_sector"),
                    "image_region_hint": t.get("image_region_hint"),
                    "image_x_band": t.get("image_x_band"),
                    "image_x_hint_01": t.get("image_x_hint_01"),
                }
                vlm_targets.append(item)
            elif self.vla_prompt_mode == "trajectory_tokens":
                temporal_risk_tokens = t.get("temporal_risk_tokens")
                if not isinstance(temporal_risk_tokens, list):
                    temporal_risk_tokens = [
                        self._format_temporal_risk_token(
                            "t0",
                            t.get("relative_forward_m"),
                            t.get("relative_lateral_m"),
                            t.get("relative_velocity_forward_mps"),
                            t.get("relative_velocity_lateral_mps"),
                        )
                    ]
                item = {
                    "target_id": target_id,
                    "track_input_source": "sensor_estimate",
                    "temporal_risk_tokens": temporal_risk_tokens,
                    "sensor_estimated_dcpa_m": t.get("sensor_estimated_dcpa_m"),
                    "sensor_estimated_tcpa_s": t.get("sensor_estimated_tcpa_s"),
                }
                vlm_targets.append(item)
            else:
                item = {
                    "target_id": target_id,
                    "track_input_source": "sensor_estimate",
                    "distance_m": t.get("distance_m"),
                    "relative_forward_m": t.get("relative_forward_m"),
                    "relative_lateral_m": t.get("relative_lateral_m"),
                    "relative_z_m": t.get("relative_z_m"),
                    "relative_angle_deg": t.get("relative_angle_deg"),
                    "relative_sector": t.get("relative_sector"),
                    "image_region_hint": t.get("image_region_hint"),
                    "image_x_band": t.get("image_x_band"),
                    "image_x_hint_01": t.get("image_x_hint_01"),
                    "relative_velocity_forward_mps": t.get("relative_velocity_forward_mps"),
                    "relative_velocity_lateral_mps": t.get("relative_velocity_lateral_mps"),
                    "relative_speed_mps": t.get("relative_speed_mps"),
                    "sensor_estimated_dcpa_m": t.get("sensor_estimated_dcpa_m"),
                    "sensor_estimated_tcpa_s": t.get("sensor_estimated_tcpa_s"),
                    "size": t.get("size"),
                    "size_x_m": t.get("size_x_m"),
                    "size_y_m": t.get("size_y_m"),
                    "size_z_m": t.get("size_z_m"),
                    "point_count": t.get("point_count"),
                }
                vlm_targets.append(item)
        targets_json = json.dumps(vlm_targets, ensure_ascii=True)
        target_count = len(vlm_targets) if vlm_targets else len(target_ids)
        image_input_active = self._llm_image_input_active()
        if self._use_pointcloud_bev_image():
            image_desc = "LiDAR BEV image"
            bev_sensor_desc = "Green points are LiDAR returns"
        elif self._use_global_trajectory_image():
            image_desc = "AIS world-coordinate trajectory image" if self._use_ais_trajectory_image() else "global world-coordinate trajectory image"
            bev_sensor_desc = ""
        else:
            image_desc = "camera image"
            bev_sensor_desc = ""
        video_input_active = bool(self._use_visual_video_input())
        track_input_enable = bool(getattr(self, "vla_prompt_track_input_enable", True))
        active_ids_text = json.dumps(target_ids, ensure_ascii=True)
        track_input_text = (
            "Track input (sensor_estimate, not ground truth):\n" + targets_json + "\n"
            if track_input_enable
            else ""
        )
        if self._use_global_trajectory_image() and image_input_active:
            target_source_desc = (
                "target trajectories come from target odometry world positions"
                if self._use_ais_trajectory_image()
                else "target trajectories come from perception tracks converted to world coordinates"
            )
            return (
                "You are the decision module controlling the own unmanned surface vessel.\n"
                "In the input image, the red trajectory/vessel is the own vessel you control; it is never a target or obstacle. "
                "Every non-red trajectory/vessel is a target vessel, and the red five-point star is the goal point.\n"
                "Choose one COLREG action for the own vessel from the predefined options.\n"
                "Encounter summary: current target vessel count is %d. \n"
                "Global trajectory image convention: black-background world-coordinate plot; %s. "
                "The task is to choose the next safe action for the red trajectory only. "
                "Red trajectory is the own vessel; each other color represents a different target vessel; the red five-point star is the goal point. "
                "Choose a safe action that avoids target vessels while making progress toward the red star goal. "
                "Empty circles are older starts; filled dots and arrowheads are latest positions and motion direction over the last %.1f seconds. "
                "Image horizontal direction is world x and image vertical direction is world y. "
                "For the red own-vessel latest motion vector v=(dx,dy), port is the positive signed side n_port=(-dy,dx), "
                "starboard is the negative signed side n_starboard=(dy,-dx); classify a target by the sign of dot(target_latest_xy - own_latest_xy, n_port).\n"
                "JSON enum rules:\n"
                "course_action MUST be exactly one of [\"KEEP_COURSE\",\"TURN_STARBOARD\",\"TURN_PORT\"]. \n"
                "TURN_PORT means the own vessel turns left; TURN_STARBOARD means the own vessel turns right. "
                "If there are no risk in the front,KEEP_COURSE will autonomously lead to the goal.\n"
                "speed_action MUST be exactly one of [\"SLOW_DOWN\",\"SPEED_UP\",\"EMERGENCY_STOP\"].\n"
                "The selected course_action and speed_action command the own vessel only; never recommend what another vessel should do.\n"
                "Return only JSON: {\"confidence\":number,\"reasoning\":string,\"course_action\":\"KEEP_COURSE|TURN_STARBOARD|TURN_PORT\",\"speed_action\":\"SLOW_DOWN|SPEED_UP|EMERGENCY_STOP\"}.\n"
                "Output constraints: one-line JSON only; no markdown; no extra text; stop after the final }.\n"
                "reasoning must be <= 36 English words and start from own-vessel perspective: state target vessel count, "
                "name the greatest-threat target_id, briefly describe the trajectory scene, then state how the red own vessel should move safely toward the goal.\n"
                "Do not include field echo.\n"
                % (target_count, target_source_desc, float(self.vla_global_history_s))
            )
        if video_input_active:
            video_frame_count = (
                int(self.vla_bev_video_frame_count)
                if self._use_bev_image()
                else int(self.vla_camera_video_frame_count)
            )
            video_interval_s = (
                float(self.vla_bev_video_interval_s)
                if self._use_bev_image()
                else float(self.vla_camera_video_interval_s)
            )
            context_desc = (
                "Scene input: "
                + str(video_frame_count)
                + " recent clean "
                + image_desc
                + " frames as a short video, ordered oldest to newest, sampled about "
                + ("%.1f" % video_interval_s)
                + " seconds apart.\n"
                "Analyze the visual trend across frames: target motion direction, closing/separating trend, and collision risk.\n"
            )
            if track_input_enable:
                context_desc += track_input_text
        elif self.vla_prompt_mode == "rgb_only_vlm":
            context_desc = (
                "Scene input: " + image_desc + " only. Active target ids: "
                + active_ids_text
                + ".\n"
            )
        elif self.vla_prompt_mode == "track_overlay":
            if track_input_enable:
                context_desc = "Scene input: " + image_desc + " plus current track bearing hints.\n" + track_input_text
            else:
                context_desc = "Scene input: " + image_desc + " only. Active target ids: " + active_ids_text + ".\n"
        elif self.vla_prompt_mode == "trajectory_tokens":
            if track_input_enable and image_input_active:
                context_desc = (
                    "Scene input: " + image_desc + " plus track trajectory tokens.\n"
                    + track_input_text
                )
            elif track_input_enable:
                context_desc = (
                    "Scene input: track trajectory tokens.\n"
                    + track_input_text
                )
            elif image_input_active:
                context_desc = "Scene input: " + image_desc + " only. Active target ids: " + active_ids_text + ".\n"
            else:
                context_desc = "Scene input: no image and no track input. Active target ids: " + active_ids_text + ".\n"
        else:
            if track_input_enable:
                context_desc = "Scene input: " + image_desc + " plus structured tracks.\n" + track_input_text
            else:
                context_desc = "Scene input: " + image_desc + " only. Active target ids: " + active_ids_text + ".\n"
        context_desc += (
            "Encounter summary: current target vessel count is %d. "
            "In reasoning, explicitly state this count and name the greatest-threat target_id from the Track input or Active target ids.\n"
            % target_count
        )
        if video_input_active and self._use_bev_image():
            context_desc += (
                "BEV image coordinate convention: own vessel is the red dot at image center; image x increases to the right (starboard) and decreases to the left (port); image y increases downward (aft) and decreases upward (forward). " + bev_sensor_desc + ".\n"
            )
        elif video_input_active and self._use_camera_image():
            context_desc += (
                "Camera convention: frames are from the own vessel's forward-facing camera; visible vessels are other targets/obstacles, not the controlled vessel. "
                "Compare frames in oldest-to-newest order: a target that grows larger or moves toward image center is closing in front of own vessel, not approaching from behind.\n"
            )
        elif self._use_bev_image() and image_input_active:
            context_desc += (
                "BEV image coordinate convention: own vessel is the red dot at image center; image x increases to the right (starboard) and decreases to the left (port); image y increases downward (aft) and decreases upward (forward); grid spacing is 5 meters with stronger 10-meter lines.\n"
            )
        elif self._use_global_trajectory_image() and image_input_active:
            target_source_desc = "target trajectories come from target odometry world positions" if self._use_ais_trajectory_image() else "target trajectories come from perception tracks converted to world coordinates"
            context_desc += (
                "Global trajectory image convention: black-background world-coordinate plot, not an own-vessel-centered view; %s. The task is to choose the next safe action for the red trajectory only. Red trajectory is the own vessel; each other color represents a different target vessel; the red five-point star is the goal point. Choose a safe action that avoids target vessels while making progress toward the red star goal. Empty circles are older starts; filled dots and arrowheads are latest positions and motion direction over the last %.1f seconds. Image horizontal direction is world x and image vertical direction is world y. For the red own-vessel latest motion vector v=(dx,dy), port is the positive signed side n_port=(-dy,dx), starboard is the negative signed side n_starboard=(dy,-dx); classify a target by the sign of dot(target_latest_xy - own_latest_xy, n_port).\n"
                % (target_source_desc, float(self.vla_global_history_s))
            )
        elif self._use_camera_image() and image_input_active:
            context_desc += (
                "Camera convention: the image is from the own vessel's forward-facing camera; visible vessels are targets/obstacles, not the controlled vessel.\n"
            )
        previous_action_json = json.dumps(
            self._previous_vlm_action_context(),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        previous_action_line = (
            "Previous VLM action context:\n"
            + previous_action_json
            + "\n"
            "Use the previous action only for action continuity; change it when the current visual trend indicates a different risk.\n"
        )
        fields_line = (
            ""
            if video_input_active and not track_input_enable
            else "Fields: track_input_source=sensor_estimate. sensor_estimated_dcpa_m and sensor_estimated_tcpa_s are closest-point metrics computed from sensor-estimated tracks, not ground truth. temporal_risk_tokens summarize each track over time; a is relative angle in degrees, dcpa/tcpa inside those tokens are also sensor-estimated.\n"
        )
        colregs_line = (
            (
                "COLREGS: follow standard maritime encounter rules using the BEV video trend.\n"
                if self._use_bev_image()
                else "COLREGS: follow standard maritime encounter rules using the camera video trend.\n"
            )
            if video_input_active
            else "COLREGS: follow standard maritime encounter rules using the track input.\n"
        )
        reasoning_focus = (
            "visual trend"
            if video_input_active
            else ("BEV scene" if self._use_bev_image() else "camera scene")
        )
        return (
            "You are the decision module controlling the own unmanned surface vessel.\n"
            "Controlled actor: own vessel only. Other visible vessels are targets/obstacles and cannot execute your command.\n"
            "Choose one COLREG action for the own vessel from the predefined options.\n"
            + context_desc
            + previous_action_line
            + fields_line
            + colregs_line
            + "JSON enum rules:\n"
            "course_action MUST be exactly one of [\"KEEP_COURSE\",\"TURN_STARBOARD\",\"TURN_PORT\"]. \n"
            "TURN_PORT means the own vessel turns left/port; TURN_STARBOARD means the own vessel turns right/starboard. For obstacle-relative avoidance, a front-right/starboard target maps to TURN_PORT, and a front-left/port target maps to TURN_STARBOARD.\n"
            "speed_action MUST be exactly one of [\"SLOW_DOWN\",\"SPEED_UP\",\"EMERGENCY_STOP\"].\n"
            "The selected course_action and speed_action command the own vessel only; never recommend what another vessel should do.\n"
            "Return only JSON: {\"confidence\":number,\"reasoning\":string,\"course_action\":\"KEEP_COURSE|TURN_STARBOARD|TURN_PORT\",\"speed_action\":\"SLOW_DOWN|SPEED_UP|EMERGENCY_STOP\"}.\n"
            "Output constraints: one-line JSON only; no markdown; no extra text; stop after the final }.\n"
            "reasoning must be <= 36 English words and start from own-vessel perspective: state target vessel count, name the greatest-threat target_id, briefly describe the " + reasoning_focus + ", then state the own-vessel judgement.\n"
            "Do not include field echo.\n"
        )

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _default_traj_constraints(self):
        return {
            "duration_s": self._clamp(self.default_traj_duration_s, 1.0, 30.0),
            "target_linear_x": self._clamp(self.default_avoid_speed, 0.0, self.normal_forward_speed),
            "target_angular_z": self._clamp(self.default_avoid_turn_rate, -1.5, 1.5),
            "max_linear_acc": self._clamp(self.default_max_linear_acc, 0.02, 2.0),
            "max_angular_acc": self._clamp(self.default_max_angular_acc, 0.05, 4.0),
            "min_linear_x": self._clamp(self.default_min_avoid_speed, 0.0, 1.0),
        }

    def _normalize_traj_constraints(self, parsed):
        if not isinstance(parsed, dict):
            parsed = {}
        base = self._default_traj_constraints()

        def _safe_float(v, default_v):
            if v is None:
                return float(default_v)
            if isinstance(v, str) and (v.strip() == ""):
                return float(default_v)
            try:
                return float(v)
            except Exception:
                return float(default_v)

        out = dict(base)
        legacy_action = parsed.get(
            "colreg_action",
            parsed.get("colreg_compliant_action", parsed.get("action", "")),
        )
        legacy_course_action, legacy_speed_action = self._split_colreg_action(legacy_action)
        raw_course_action = parsed.get(
            "course_action",
            parsed.get("action_decision", parsed.get("maneuver_action", legacy_course_action)),
        )
        raw_speed_action = parsed.get(
            "speed_action",
            parsed.get("speed_decision", parsed.get("velocity_action", legacy_speed_action)),
        )
        action_diag = self._action_field_diag(raw_course_action, raw_speed_action)
        if action_diag:
            self._last_parse_diag = (self._last_parse_diag + ";" if self._last_parse_diag else "") + action_diag
        course_action = self._canonical_course_action(raw_course_action)
        speed_action = self._canonical_speed_action(raw_speed_action)
        if course_action == "KEEP_COURSE":
            speed_action = ""
        colreg_action = self._combine_colreg_action(course_action, speed_action)
        colreg_weights = self._colreg_weights_for_split(course_action, speed_action)
        user_weights = parsed.get("colreg_weights", {})
        if isinstance(user_weights, dict):
            for key in (
                "strength",
                "turn_bias",
                "keep_course_bias",
                "speed_scale",
                "clearance_scale",
                "predictive_risk_scale",
                "collision_penalty_scale",
                "target_penalty_scale",
            ):
                if key in user_weights:
                    colreg_weights[key] = self._clamp(
                        _safe_float(user_weights.get(key), colreg_weights[key]),
                        -2.0 if key == "turn_bias" else 0.0,
                        2.5,
                    )
        if speed_action == "EMERGENCY_STOP":
            out["target_linear_x"] = 0.0
            out["min_linear_x"] = 0.0
        elif speed_action == "SLOW_DOWN":
            out["target_linear_x"] = self._clamp(
                self.normal_forward_speed * float(colreg_weights.get("speed_scale", 0.45)),
                0.0,
                self.normal_forward_speed,
            )
        elif speed_action == "SPEED_UP":
            out["target_linear_x"] = self._clamp(
                self.normal_forward_speed * float(colreg_weights.get("speed_scale", 1.15)),
                0.0,
                self.normal_forward_speed,
            )

        if speed_action == "EMERGENCY_STOP":
            out["target_angular_z"] = 0.0
        elif course_action == "TURN_STARBOARD":
            out["target_angular_z"] = -abs(self.default_avoid_turn_rate)
        elif course_action == "TURN_PORT":
            out["target_angular_z"] = abs(self.default_avoid_turn_rate)
        else:
            out["target_angular_z"] = 0.0
        out["duration_s"] = self._clamp(
            _safe_float(parsed.get("duration_s", out["duration_s"]), out["duration_s"]),
            1.0,
            45.0,
        )
        out["target_linear_x"] = self._clamp(
            _safe_float(parsed.get("target_linear_x", out["target_linear_x"]), out["target_linear_x"]),
            0.0,
            self.normal_forward_speed,
        )
        out["target_angular_z"] = self._clamp(
            _safe_float(parsed.get("target_angular_z", out["target_angular_z"]), out["target_angular_z"]),
            -2.0,
            2.0,
        )
        out["max_linear_acc"] = self._clamp(
            _safe_float(parsed.get("max_linear_acc", out["max_linear_acc"]), out["max_linear_acc"]),
            0.02,
            3.0,
        )
        out["max_angular_acc"] = self._clamp(
            _safe_float(parsed.get("max_angular_acc", out["max_angular_acc"]), out["max_angular_acc"]),
            0.05,
            6.0,
        )
        out["min_linear_x"] = self._clamp(
            _safe_float(parsed.get("min_linear_x", out["min_linear_x"]), out["min_linear_x"]),
            0.0,
            1.2,
        )

        # Keep turn-rate bounded, but do not force a fixed turn side.
        out["target_angular_z"] = self._clamp(
            out["target_angular_z"], -abs(self.avoid_turn_rate_limit), abs(self.avoid_turn_rate_limit)
        )
        out["target_linear_x"] = max(out["target_linear_x"], out["min_linear_x"])
        out["course_action"] = course_action
        out["speed_action"] = speed_action
        out["colreg_action"] = colreg_action
        out["colreg_weights"] = colreg_weights
        return out

    def _start_async_llm_call(self, reason, trigger_generation, call_context=None):
        if self._llm_inflight:
            return False
        block_reason = self._llm_call_block_reason(trigger_generation)
        if block_reason:
            self._last_llm_called = False
            self._last_llm_call_reason = block_reason
            return False
        self._llm_inflight = True
        call_context = dict(call_context) if isinstance(call_context, dict) else {}

        def _worker():
            try:
                if not self._llm_call_allowed(trigger_generation):
                    return
                result = self._llm_classify(
                    reason,
                    trigger_generation=trigger_generation,
                    call_context=call_context,
                )
                with self._llm_lock:
                    if result is not None and self._llm_result_generation_valid(trigger_generation):
                        self._llm_pending_result = (
                            result,
                            None if trigger_generation is None else int(trigger_generation),
                        )
            except Exception as e:
                err = str(e)[:500]
                self._last_llm_error = err
                self._append_llm_io_record(
                    {
                        "event": "llm_worker_error",
                        "wall_time_s": time.time(),
                        "ros_time_s": rospy.Time.now().to_sec(),
                        "exception_type": type(e).__name__,
                        "error": err,
                        "reason": reason,
                        "trigger_generation": trigger_generation,
                        "event_image_path": call_context.get("event_image_path", self._event_image_path or ""),
                        "image_diag": self._vla_image_diag(),
                    }
                )
                rospy.logwarn_throttle(
                    5.0,
                    "[colregs_llm_decision] async LLM worker failed: %s",
                    err,
                )
            finally:
                with self._llm_lock:
                    self._llm_inflight = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True

    def _consume_pending_llm_result(self):
        with self._llm_lock:
            result = self._llm_pending_result
            self._llm_pending_result = None
        if isinstance(result, tuple) and len(result) == 2:
            result_obj, trigger_generation = result
            if result_obj is None:
                return
            if not self._llm_result_generation_valid(trigger_generation):
                self._publish_vlm_check(
                    "llm_result_dropped_generation_changed",
                    {
                        "llm_call_id": int(result_obj.get("debug_call_id", -1)),
                        "result_generation": trigger_generation,
                        "current_generation": int(self._trigger_generation),
                    },
                )
                return
            self._last_cls = result_obj
            self._lock_llm_after_success_if_needed(result_obj)
            self._activate_traj_plan()
            return
        if result is not None:
            self._last_cls = result
            self._lock_llm_after_success_if_needed(result)
            self._activate_traj_plan()

    def _sensor_front_hazard_diag(self):
        targets = self._build_targets_prompt_context(include_state_fallback=False)
        range_gate = max(0.1, float(self.sensor_front_hazard_range_m))
        bearing_gate = max(0.0, float(self.sensor_front_hazard_bearing_deg))
        min_forward = max(0.0, float(self.sensor_front_hazard_min_forward_m))
        checked = []
        hazard_targets = []
        for idx, t in enumerate(targets):
            if not isinstance(t, dict):
                continue
            target_id = str(t.get("target_id", t.get("id", "target_%d" % (idx + 1))))
            distance = self._safe_float(t.get("distance_m"), None)
            bearing = self._safe_float(t.get("relative_angle_deg"), None)
            rel_forward = self._safe_float(t.get("relative_forward_m"), None)
            if rel_forward is None and distance is not None and bearing is not None:
                rel_forward = float(distance) * math.cos(math.radians(float(bearing)))

            in_range = bool(distance is not None and distance > 0.0 and distance <= range_gate)
            in_bearing = bool(bearing is not None and abs(float(bearing)) <= bearing_gate)
            in_front = bool(rel_forward is not None and float(rel_forward) >= min_forward)
            hazard = bool(in_range and in_bearing and in_front)
            item = {
                "target_id": target_id,
                "distance_m": None if distance is None else round(float(distance), 3),
                "bearing_deg": None if bearing is None else round(float(bearing), 2),
                "relative_forward_m": None if rel_forward is None else round(float(rel_forward), 3),
                "in_range": in_range,
                "in_bearing": in_bearing,
                "in_front": in_front,
                "hazard": hazard,
            }
            checked.append(item)
            if hazard:
                hazard_targets.append(item)

        hazard_targets.sort(
            key=lambda item: (
                float("inf") if item.get("distance_m") is None else float(item.get("distance_m")),
                abs(float(item.get("bearing_deg") or 0.0)),
            )
        )
        nearest = hazard_targets[0] if hazard_targets else {}
        return {
            "enable": bool(self.sensor_front_hazard_enable),
            "hazard": bool(hazard_targets),
            "target_count": len(checked),
            "hazard_count": len(hazard_targets),
            "range_gate_m": round(range_gate, 3),
            "bearing_gate_deg": round(bearing_gate, 3),
            "min_forward_m": round(min_forward, 3),
            "greatest_threat_target_id": str(nearest.get("target_id", "")),
            "nearest_hazard_range_m": nearest.get("distance_m"),
            "nearest_hazard_bearing_deg": nearest.get("bearing_deg"),
            "checked_targets": checked[:12],
        }

    def _sensor_front_clear_classification(self, diag):
        return {
            "source": "sensor_front_clear",
            "confidence": 1.0,
            "reasoning": (
                "Sensor range/bearing gate reports no target ahead inside "
                "%.1fm and %.1f deg; release avoidance behavior."
                % (
                    float(self.sensor_front_hazard_range_m),
                    float(self.sensor_front_hazard_bearing_deg),
                )
            ),
            "track_classifications": [],
            "targets_json": [],
            "course_action": "KEEP_COURSE",
            "speed_action": "",
            "colreg_action": "KEEP_COURSE",
            "colreg_weights": {},
            "trajectory_constraints": {},
            "behavior_active": False,
            "force_keep_course": True,
            "sensor_front_hazard": False,
            "sensor_front_hazard_diag": diag,
            "debug_call_id": -1,
        }

    def _attach_traj_constraints(self, out_dict):
        constraints = out_dict.get("trajectory_constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        constraints = dict(constraints)
        constraints.setdefault("course_action", out_dict.get("course_action", ""))
        constraints.setdefault("speed_action", out_dict.get("speed_action", ""))
        constraints.setdefault("colreg_action", out_dict.get("colreg_action", ""))
        if isinstance(out_dict.get("colreg_weights"), dict):
            constraints.setdefault("colreg_weights", out_dict.get("colreg_weights"))
        out_dict["trajectory_constraints"] = self._normalize_traj_constraints(constraints)
        out_dict["course_action"] = out_dict["trajectory_constraints"].get("course_action", "TURN_STARBOARD")
        out_dict["speed_action"] = out_dict["trajectory_constraints"].get("speed_action", "SLOW_DOWN")
        out_dict["colreg_action"] = out_dict["trajectory_constraints"].get("colreg_action", "TURN_STARBOARD+SLOW_DOWN")
        out_dict["colreg_weights"] = out_dict["trajectory_constraints"].get("colreg_weights", {})
        if out_dict.get("source") == "llm":
            out_dict.pop("track_classifications", None)
        else:
            out_dict["track_classifications"] = self._normalize_track_classifications(out_dict)
        return out_dict

    def _extract_json(self, text):
        text = (text or "").strip()
        # Remove ANSI escapes and non-printable control chars that may appear in model output.
        text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
        text = "".join(ch for ch in text if (ch >= " " or ch in "\n\r\t"))
        # Remove common markdown code fences and leading "json" prefixes.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
        if not text:
            self._last_parse_diag = "empty_output"
            return None
        repeat_reason = self._raw_output_repeat_reason(text) if self.api_repeat_retry_enable else ""
        if repeat_reason:
            self._last_parse_diag = "degenerate_raw_output:" + repeat_reason
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                self._last_parse_diag = "json.loads(full_text_dict)"
                return obj
            if isinstance(obj, str):
                nested = obj.strip()
                try:
                    obj2 = json.loads(nested)
                    if isinstance(obj2, dict):
                        self._last_parse_diag = "json.loads(full_text_string_then_json)"
                        return obj2
                except Exception:
                    pass
                text = nested
                self._last_parse_diag = "json.loads(full_text_string_non_dict)"
            else:
                self._last_parse_diag = "json.loads(full_text_non_dict)"
                return None
        except Exception:
            pass
        # Try decoder raw_decode from first '{' position.
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\{", text):
            start = m.start()
            try:
                obj, _ = decoder.raw_decode(text[start:])
                if isinstance(obj, dict):
                    self._last_parse_diag = "raw_decode_from_first_left_brace"
                    return obj
            except Exception:
                continue
        # Fallback: extract first balanced JSON object
        start = text.find("{")
        if start < 0:
            self._last_parse_diag = "no_left_brace_found"
            return None
        depth = 0
        in_str = False
        escaped = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            self._last_parse_diag = "no_balanced_right_brace_found_or_truncated"
            return None
        obj_txt = text[start : end + 1]
        try:
            obj = json.loads(obj_txt)
            self._last_parse_diag = "balanced_brace_substring"
            return obj
        except Exception:
            try:
                obj = ast.literal_eval(obj_txt)
                if isinstance(obj, dict):
                    self._last_parse_diag = "python_literal_dict_after_balanced_brace"
                    return obj
            except Exception:
                pass
            self._last_parse_diag = "parse_failed_after_balanced_brace"
            return None

    def _normalize_llm_result(self, parsed, expected_ids=None):
        legacy_action = parsed.get(
            "colreg_action",
            parsed.get("colreg_compliant_action", parsed.get("action", "")),
        )
        legacy_course_action, legacy_speed_action = self._split_colreg_action(legacy_action)
        raw_course_action = parsed.get(
            "course_action",
            parsed.get("action_decision", parsed.get("maneuver_action", legacy_course_action)),
        )
        raw_speed_action = parsed.get(
            "speed_action",
            parsed.get("speed_decision", parsed.get("velocity_action", legacy_speed_action)),
        )
        action_diag = self._action_field_diag(raw_course_action, raw_speed_action)
        if action_diag:
            self._last_parse_diag = (self._last_parse_diag + ";" if self._last_parse_diag else "") + action_diag
        course_action = self._canonical_course_action(raw_course_action)
        speed_action = self._canonical_speed_action(raw_speed_action)
        colreg_action = self._combine_colreg_action(course_action, speed_action)
        constraints = parsed.get("trajectory_constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        constraints = dict(constraints)
        constraints.setdefault("course_action", course_action)
        constraints.setdefault("speed_action", speed_action)
        constraints.setdefault("colreg_action", colreg_action)
        if isinstance(parsed.get("colreg_weights"), dict):
            constraints.setdefault("colreg_weights", parsed.get("colreg_weights"))
        normalized = {
            "confidence": self._clamp(self._safe_float(parsed.get("confidence", 0.6), 0.6), 0.0, 1.0),
            "reasoning": str(parsed.get("reasoning", "LLM decision")),
            "course_action": course_action,
            "speed_action": speed_action,
            "colreg_action": colreg_action,
            "source": "llm",
            "debug_call_id": parsed.get("debug_call_id", -1),
        }
        normalized["trajectory_constraints"] = self._normalize_traj_constraints(constraints)
        normalized["course_action"] = normalized["trajectory_constraints"].get("course_action", course_action)
        normalized["speed_action"] = normalized["trajectory_constraints"].get("speed_action", speed_action)
        normalized["colreg_action"] = normalized["trajectory_constraints"].get("colreg_action", colreg_action)
        normalized["colreg_weights"] = normalized["trajectory_constraints"].get("colreg_weights", {})
        return normalized

    def _perception_diag(self):
        s = self._state or {}
        rv = s.get("relative_velocity", {}) or {}
        ev = s.get("ego_velocity", {}) or {}
        tv = s.get("target_velocity", {}) or {}
        rp = s.get("relative_position", {}) or {}

        rel_px = float(rp.get("x", 0.0))
        rel_py = float(rp.get("y", 0.0))
        rel_vx = float(rv.get("x", 0.0))
        rel_vy = float(rv.get("y", 0.0))
        evx = float(ev.get("x", 0.0))
        evy = float(ev.get("y", 0.0))
        tvx = float(tv.get("x", 0.0))
        tvy = float(tv.get("y", 0.0))

        range_m = max(1e-6, math.hypot(rel_px, rel_py))
        tcpa = float(s.get("tcpa_s", 1e9))
        dcpa = float(s.get("dcpa_m", 1e9))
        rb_deg = float(s.get("relative_bearing_deg", 0.0))

        ego_spd = math.hypot(evx, evy)
        tgt_spd = math.hypot(tvx, tvy)
        rel_speed = math.hypot(rel_vx, rel_vy)
        range_rate = (rel_px * rel_vx + rel_py * rel_vy) / range_m

        ego_course = math.atan2(evy, evx) if ego_spd > 0.2 else float(s.get("ego_heading_rad", 0.0))
        tgt_course = math.atan2(tvy, tvx) if tgt_spd > 0.2 else float(s.get("target_heading_rad", 0.0))
        course_diff_deg = abs(math.degrees(self._norm_pi(ego_course - tgt_course)))

        target_course_observable = tgt_spd > 0.2
        ego_course_observable = ego_spd > 0.2
        diagnosis = "ok"
        if not target_course_observable:
            diagnosis = "target_course_unobservable_from_perception"
        elif abs(range_rate) < 0.03:
            diagnosis = "range_rate_near_zero_unstable_for_action_decision"

        return {
            "range_m": round(range_m, 3),
            "dcpa_m": round(dcpa, 3),
            "tcpa_s": round(tcpa, 3),
            "relative_bearing_deg": round(rb_deg, 2),
            "range_rate_mps": round(range_rate, 3),
            "ego_speed_mps": round(ego_spd, 3),
            "target_speed_mps": round(tgt_spd, 3),
            "relative_speed_mps": round(rel_speed, 3),
            "course_diff_deg": round(course_diff_deg, 2),
            "target_ahead_sector": abs(rb_deg) <= 35.0,
            "same_course_like": course_diff_deg <= 22.5,
            "closing": range_rate <= -0.05,
            "own_faster": ego_spd >= (tgt_spd + 0.08),
            "overtaking_likely_geometry": self._is_strong_overtaking_case(),
            "ego_course_observable": ego_course_observable,
            "target_course_observable": target_course_observable,
            "ego_heading_rad": round(float(s.get("ego_heading_rad", 0.0)), 3),
            "target_heading_rad": round(float(s.get("target_heading_rad", 0.0)), 3),
            "diagnosis_hint": diagnosis,
        }

    def _perception_brief(self):
        d = self._perception_diag()
        return {
            "range_m": d["range_m"],
            "tcpa_s": d["tcpa_s"],
            "dcpa_m": d["dcpa_m"],
            "bearing_deg": d["relative_bearing_deg"],
            "range_rate_mps": d["range_rate_mps"],
            "course_diff_deg": d["course_diff_deg"],
            "closing": d["closing"],
            "target_course_observable": d["target_course_observable"],
            "diagnosis_hint": d["diagnosis_hint"],
        }

    def _terminal_status_debug(self, force=False):
        if not self.terminal_status_enable:
            return
        now_s = rospy.Time.now().to_sec()
        if (not force) and ((now_s - self._last_terminal_status_t) < max(0.2, self.terminal_status_interval_s)):
            return
        self._last_terminal_status_t = now_s

        d_min = self._pointcloud_trigger_distance()
        risk = self._risk_state if isinstance(self._risk_state, dict) else {}
        risk_stats = risk.get("pointcloud_stats", {}) if isinstance(risk, dict) else {}
        fov_pts = risk.get("fov_valid_points", risk_stats.get("fov_valid_points", "-"))
        on_thr = risk.get("fov_point_count_on_threshold", "-")
        off_thr = risk.get("fov_point_count_off_threshold", "-")
        manual_gate = risk.get("manual_trigger_enabled", "-")
        risk_trigger = risk.get("trigger", "-")
        risk_on = risk.get("trigger_on_candidate", "-")
        targets_n = len(self._perception_targets) if isinstance(self._perception_targets, list) else 0
        p_age = -1.0 if self._perception_stamp <= 0.0 else max(0.0, now_s - self._perception_stamp)
        r_age = -1.0 if self._risk_stamp <= 0.0 else max(0.0, now_s - self._risk_stamp)
        rospy.loginfo(
            "[colregs_llm_decision] 【STATUS】trigger=%s gen=%d seen_low=%s state_seq=%d targets=%d p_age=%.1fs risk=%s manual=%s risk_on=%s fov=%s on/off=%s/%s r_age=%.1fs d_min=%s gate=%.1f req_dist=%s inflight=%s last=%s",
            str(self._trigger),
            int(self._trigger_generation),
            str(self._trigger_seen_low),
            int(self._state_seq),
            int(targets_n),
            p_age,
            str(risk_trigger),
            str(manual_gate),
            str(risk_on),
            str(fov_pts),
            str(on_thr),
            str(off_thr),
            r_age,
            "-" if d_min is None else "%.2f" % float(d_min),
            float(self.llm_trigger_distance_m),
            str(self.llm_require_pointcloud_distance),
            str(self._llm_inflight),
            str(self._last_llm_call_reason),
        )

    def _should_attach_full_diag(self, now_s):
        period = 1.0 / max(0.1, self.decision_debug_hz)
        if (now_s - self._last_decision_debug_t) >= period:
            self._last_decision_debug_t = now_s
            return True
        # LLM 刚调用时，允许即时带一次 full_diag 便于定位误判。
        return bool(self._last_llm_called)

    def _is_strong_overtaking_case(self):
        s = self._state or {}
        rv = s.get("relative_velocity", {}) or {}
        ev = s.get("ego_velocity", {}) or {}
        tv = s.get("target_velocity", {}) or {}
        rp = s.get("relative_position", {}) or {}

        rel_px = float(rp.get("x", 0.0))
        rel_py = float(rp.get("y", 0.0))
        rel_vx = float(rv.get("x", 0.0))
        rel_vy = float(rv.get("y", 0.0))
        evx = float(ev.get("x", 0.0))
        evy = float(ev.get("y", 0.0))
        tvx = float(tv.get("x", 0.0))
        tvy = float(tv.get("y", 0.0))

        range_m = max(1e-6, math.hypot(rel_px, rel_py))
        tcpa = float(s.get("tcpa_s", 1e9))
        rb_deg = float(s.get("relative_bearing_deg", 0.0))

        ego_spd = math.hypot(evx, evy)
        tgt_spd = math.hypot(tvx, tvy)
        rel_speed = math.hypot(rel_vx, rel_vy)
        if rel_speed < 0.05 or tcpa <= 0.0:
            return False

        ego_course = math.atan2(evy, evx) if ego_spd > 0.2 else float(s.get("ego_heading_rad", 0.0))
        tgt_course = math.atan2(tvy, tvx) if tgt_spd > 0.2 else float(s.get("target_heading_rad", 0.0))
        course_diff_deg = abs(math.degrees(self._norm_pi(ego_course - tgt_course)))

        # Negative range rate means closing.
        range_rate = (rel_px * rel_vx + rel_py * rel_vy) / range_m

        target_ahead = abs(rb_deg) <= 35.0
        same_course = course_diff_deg <= 22.5
        own_faster = ego_spd >= (tgt_spd + 0.08)
        closing = range_rate <= -0.05

        return bool(target_ahead and same_course and own_faster and closing and tcpa < 180.0)

    def _activate_traj_plan(self):
        constraints = self._last_cls.get("trajectory_constraints") or {}
        plan = self._normalize_traj_constraints(constraints)
        plan["start_t"] = rospy.Time.now().to_sec()
        self._active_traj = plan

    def _slew(self, target, current, limit_per_sec, dt):
        if dt <= 1e-6:
            return current
        max_delta = max(0.0, limit_per_sec) * dt
        delta = target - current
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        return current + delta

    def _build_desired_cmd(self):
        desired = Twist()
        self._maybe_auto_start_recovery()

        if self._post_recovery_keep_course and (not self._recovery_active):
            desired.linear.x = self.normal_forward_speed
            desired.angular.z = 0.0
            return desired, self.default_max_linear_acc, self.default_max_angular_acc

        pending_safety_active = (
            self._llm_inflight
            and isinstance(self._last_cls, dict)
            and self._last_cls.get("source") == "pending_safety_pre_llm"
        )
        pretrigger_llm_active = (
            (not self._trigger)
            and isinstance(self._last_cls, dict)
            and self._last_cls.get("source") == "llm"
            and bool(self._active_traj)
            and self._canonical_course_action(self._last_cls.get("course_action", "")) != "KEEP_COURSE"
        )
        if isinstance(self._last_cls, dict) and self._last_cls.get("force_keep_course", False):
            self._active_traj = None
            desired.linear.x = self.normal_forward_speed
            desired.angular.z = 0.0
            return desired, self.default_max_linear_acc, self.default_max_angular_acc
        if (
            (not self._trigger)
            and (not self._recovery_active)
            and (not pending_safety_active)
            and (not pretrigger_llm_active)
        ):
            self._active_traj = None
            desired.linear.x = self.normal_forward_speed
            desired.angular.z = 0.0
            return desired, self.default_max_linear_acc, self.default_max_angular_acc

        if self._recovery_active:
            ego_heading = self._state.get("ego_heading_rad")
            if (ego_heading is None) or (self._pre_encounter_heading is None):
                self._recovery_active = False
                desired.linear.x = self.normal_forward_speed
                desired.angular.z = 0.0
                return desired, self.default_max_linear_acc, self.default_max_angular_acc

            yaw_err = self._norm_pi(float(self._pre_encounter_heading) - float(ego_heading))
            desired.linear.x = min(self.normal_forward_speed, self.recovery_linear_speed)

            lat_err = 0.0
            has_lat_err = False
            if (
                self.recovery_enable_lane_recenter
                and (self._pre_encounter_pos is not None)
                and (self._ego_pos_xy is not None)
            ):
                dx = self._ego_pos_xy[0] - self._pre_encounter_pos[0]
                dy = self._ego_pos_xy[1] - self._pre_encounter_pos[1]
                h0 = float(self._pre_encounter_heading)
                # Signed lateral error to the pre-encounter reference track.
                # >0: left of track, <0: right of track.
                lat_err = (-math.sin(h0) * dx) + (math.cos(h0) * dy)
                has_lat_err = True
                self._last_lateral_error = lat_err

            omega_cmd = self.recovery_heading_kp * yaw_err
            if has_lat_err:
                omega_cmd += (-self.recovery_lateral_kp * lat_err)
                if abs(lat_err) > self.recovery_lateral_tolerance:
                    omega_cmd += self.recovery_left_turn_bias

            desired.angular.z = self._clamp(
                omega_cmd,
                -abs(self.recovery_max_turn_rate),
                abs(self.recovery_max_turn_rate),
            )

            now_s = rospy.Time.now().to_sec()
            lateral_ok = True
            if has_lat_err:
                lateral_ok = abs(lat_err) <= self.recovery_lateral_tolerance

            if (abs(yaw_err) <= self.recovery_heading_tolerance) and lateral_ok:
                if self._recovery_ok_since is None:
                    self._recovery_ok_since = now_s
                elif (now_s - self._recovery_ok_since) >= self.recovery_hold_s:
                    self._recovery_active = False
                    self._recovery_ok_since = None
                    self._post_recovery_keep_course = True
                    desired.angular.z = 0.0
                    if self.debug_enable:
                        rospy.loginfo(
                            "[colregs_llm_decision] 回航向恢复完成，退出恢复阶段"
                        )
            else:
                self._recovery_ok_since = None

            return desired, self.default_max_linear_acc, self.default_max_angular_acc

        if self._active_traj is None:
            self._activate_traj_plan()

        plan = self._active_traj or {}
        start_t = float(plan.get("start_t", rospy.Time.now().to_sec()))
        elapsed = rospy.Time.now().to_sec() - start_t
        duration_s = float(plan.get("duration_s", self.default_traj_duration_s))

        if elapsed <= duration_s:
            desired.linear.x = float(plan.get("target_linear_x", self.safe_forward_speed))
            desired.angular.z = float(plan.get("target_angular_z", self.starboard_turn_rate))
            desired.linear.x = max(desired.linear.x, float(plan.get("min_linear_x", 0.0)))
        else:
            # 轨迹约束时段结束后，保持安全低速直行，直到 trigger 清除。
            desired.linear.x = self.safe_forward_speed
            desired.angular.z = 0.0

        max_lin_acc = float(plan.get("max_linear_acc", self.default_max_linear_acc))
        max_ang_acc = float(plan.get("max_angular_acc", self.default_max_angular_acc))
        return desired, max_lin_acc, max_ang_acc

    def _build_continuous_cmd(self):
        desired, max_lin_acc, max_ang_acc = self._build_desired_cmd()
        now = rospy.Time.now()
        dt = (now - self._last_cmd_stamp).to_sec()
        if dt <= 1e-4:
            dt = 1.0 / max(1.0, self.publish_rate)

        out = Twist()
        out.linear.x = self._slew(
            desired.linear.x,
            self._last_cmd.linear.x,
            max_lin_acc,
            dt,
        )
        out.angular.z = self._slew(
            desired.angular.z,
            self._last_cmd.angular.z,
            max_ang_acc,
            dt,
        )
        self._last_cmd = out
        self._last_cmd_stamp = now
        return out

    def _run_ollama(self, prompt):
        result = subprocess.run(
            self.ollama_cmd,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.ollama_timeout,
            check=True,
        )
        return result.stdout or ""

    def _post_http_chat(self, body):
        if not self.api_url:
            raise ValueError("api_url_empty")
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.api_timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RuntimeError("http_%d: %s" % (e.code, err_body or str(e))) from e
        except urllib.error.URLError as e:
            raise RuntimeError("http_url_error: %s" % (e,)) from e
        return json.loads(raw)

    @staticmethod
    def _chat_payload_text(payload):
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("http_no_choices: " + json.dumps(payload, ensure_ascii=True)[:300])
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if content is None:
            content = choices[0].get("text", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text")
                    if t:
                        text_parts.append(str(t))
            content = "".join(text_parts)
        return str(content or "")

    def _raw_output_repeat_reason(self, text):
        text = str(text or "").strip()
        if not text:
            return ""
        if len(text) >= int(self.api_no_json_retry_chars) and "{" not in text:
            return "long_output_without_json"
        words = re.findall(r"[A-Za-z0-9_]+", text.lower())
        n = int(self.api_repeat_ngram_words)
        if len(words) < n * int(self.api_repeat_ngram_min_count):
            return ""
        counts = {}
        for i in range(0, len(words) - n + 1):
            key = tuple(words[i : i + n])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= int(self.api_repeat_ngram_min_count):
                return "repeated_%dgram_%dx" % (n, counts[key])
        return ""

    def _post_http_chat_with_json_fallback(self, body):
        try:
            return self._post_http_chat(body)
        except RuntimeError as e:
            # Some OpenAI-compatible gateways reject response_format; retry once without it.
            if self.api_force_json_mode and "response_format" in str(e):
                body.pop("response_format", None)
                return self._post_http_chat(body)
            raise

    def _run_http_api(self, prompt, image_data_url="", video_url="", media_io_kwargs=None):
        media_io_kwargs = media_io_kwargs if isinstance(media_io_kwargs, dict) else {}
        if video_url:
            content = [
                {
                    "type": "video_url",
                    "video_url": {"url": str(video_url)},
                },
                {"type": "text", "text": prompt},
            ]
        elif image_data_url:
            content = []
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                }
            )
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        body = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.api_temperature,
        }
        if self.api_max_output_tokens > 0:
            body["max_tokens"] = int(self.api_max_output_tokens)
        if self.api_force_json_mode:
            body["response_format"] = {"type": "json_object"}
        if media_io_kwargs:
            body["media_io_kwargs"] = media_io_kwargs

        payload = self._post_http_chat_with_json_fallback(body)
        text = self._chat_payload_text(payload)
        repeat_reason = self._raw_output_repeat_reason(text) if self.api_repeat_retry_enable else ""
        if repeat_reason:
            retry_body = dict(body)
            retry_body["temperature"] = min(float(self.api_temperature), 0.05)
            if self.api_max_output_tokens > 0:
                retry_body["max_tokens"] = min(int(self.api_max_output_tokens), 96)
            else:
                retry_body["max_tokens"] = 96
            if self.api_force_json_mode:
                retry_body["response_format"] = {"type": "json_object"}
            if self.debug_enable:
                rospy.logwarn(
                    "[colregs_llm_decision] HTTP VLM raw output degenerated (%s), retrying once with stricter decoding",
                    repeat_reason,
                )
            retry_payload = self._post_http_chat_with_json_fallback(retry_body)
            retry_text = self._chat_payload_text(retry_payload)
            retry_reason = self._raw_output_repeat_reason(retry_text)
            if retry_reason:
                self._last_parse_diag = "degenerate_raw_output_after_retry:" + retry_reason
            return retry_text
        return text

    def _run_gemini_api(self, prompt):
        """Google AI Gemini generateContent (REST v1beta)."""
        if not self.gemini_api_key:
            raise ValueError("gemini_api_key_empty")
        model = self.gemini_model.strip()
        if model.startswith("models/"):
            model = model[len("models/") :]
        path = "v1beta/models/%s:generateContent" % model
        q = urllib.parse.urlencode({"key": self.gemini_api_key})
        url = "%s/%s?%s" % (self.gemini_api_base, path, q)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.api_temperature,
                "maxOutputTokens": int(self.gemini_max_output_tokens),
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.api_timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                pass
            raise RuntimeError("gemini_http_%d: %s" % (e.code, err_body or str(e))) from e
        except urllib.error.URLError as e:
            raise RuntimeError("gemini_url_error: %s" % (e,)) from e
        payload = json.loads(raw)
        err = payload.get("error")
        if err:
            raise RuntimeError(
                "gemini_api_error: %s" % (json.dumps(err, ensure_ascii=True)[:500],)
            )
        fb = payload.get("promptFeedback") or {}
        br = fb.get("blockReason")
        if br:
            raise RuntimeError("gemini_blocked: promptFeedback=%s" % (fb,))
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError("gemini_no_candidates: " + raw[:500])
        parts = (candidates[0].get("content") or {}).get("parts") or []
        texts = []
        for p in parts:
            if isinstance(p, dict) and p.get("text"):
                texts.append(str(p["text"]))
        out = "".join(texts)
        if not out.strip():
            raise RuntimeError("gemini_empty_text: " + raw[:400])
        return out

    def _llm_classify(self, call_reason, trigger_generation=None, call_context=None):
        call_context = dict(call_context) if isinstance(call_context, dict) else {}
        block_reason = self._llm_call_block_reason(trigger_generation)
        if block_reason:
            self._last_llm_called = False
            self._last_llm_call_reason = block_reason
            return None
        self._llm_call_seq += 1
        call_id = self._llm_call_seq
        t0 = time.perf_counter()
        self._last_llm_call_reason = call_reason
        self._last_llm_called = True

        if self.debug_enable and (not self.terminal_clean_mode):
            rospy.loginfo(
                "[colregs_llm_decision] 【LLM】准备第 %d 次调用 | 原因=%s | 方式=%s | 触发=%s | 状态序号=%d | 快照=%s",
                call_id,
                self._call_reason_zh(call_reason),
                self._backend_name_zh(self.llm_backend),
                "开" if self._trigger else "关",
                self._state_seq,
                json.dumps(self._state_snapshot(), ensure_ascii=True),
            )

        if not self.llm_enable:
            self._last_llm_error = "llm_disabled"
            self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out = self._fallback_classification()
            out["debug_call_id"] = call_id
            return self._attach_traj_constraints(out)
        prompt_targets = call_context.get("targets")
        expected_target_ids = [
            str(t.get("target_id", "track_%d" % idx))
            for idx, t in enumerate(prompt_targets, start=1)
            if isinstance(t, dict)
        ] if isinstance(prompt_targets, list) else None
        prompt = self._build_prompt(prompt_targets if isinstance(prompt_targets, list) else None)
        image_data_url = ""
        video_url = ""
        media_io_kwargs = {}
        raw_out = ""
        self._last_llm_remote_ok = False
        self._last_llm_model_id = ""
        self._last_llm_image_source = ""
        self._last_llm_image_data_url_len = 0
        self._last_llm_video_frame_paths = []
        try:
            block_reason = self._llm_call_block_reason(trigger_generation)
            if block_reason:
                self._last_llm_called = False
                self._last_llm_call_reason = block_reason
                return None
            if self.llm_backend == "dinov2":
                self._last_llm_model_id = self.dinov2_model_name
                self._terminal_dump_llm_input(
                    call_id,
                    call_reason,
                    "DINOv2 local prototype classification",
                    call_context=call_context,
                )
                dino_result = self._run_dinov2_classifier(call_context=call_context)
                raw_out = json.dumps(dino_result, ensure_ascii=True)
                self._last_llm_error = ""
                self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._last_llm_remote_ok = True
                self._last_llm_raw_text = raw_out
                self._last_llm_raw_preview = raw_out[:400]
                self._terminal_dump_llm_output(call_id, raw_out)
                self._raw_pub.publish(String(data=raw_out))
                dino_result["debug_call_id"] = call_id
                out = self._attach_traj_constraints(dino_result)
                if self.debug_enable:
                    self._publish_debug(
                        "dinov2_classification_ok",
                        {
                            "llm_call_id": call_id,
                            "track_classifications": out.get("track_classifications", []),
                            "backend_runtime": self._dinov2_classifier.runtime_backend,
                            "backend_status": self._dinov2_classifier.status,
                            "llm_image_source": self._last_llm_image_source,
                            "perception_diag": self._perception_diag(),
                        },
                    )
                self._publish_vlm_check(
                    "dinov2_ok",
                    {
                        "llm_call_id": call_id,
                        "image_source": self._last_llm_image_source,
                        "backend_runtime": self._dinov2_classifier.runtime_backend,
                        "backend_status": self._dinov2_classifier.status,
                    },
                )
                return out
            if self.llm_backend == "ollama":
                self._last_llm_model_id = "ollama"
                self._last_llm_image_source = "not_supported_by_backend"
                self._terminal_dump_llm_input(call_id, call_reason, prompt, call_context=call_context)
                raw_out = self._run_ollama(prompt)
            elif self.llm_backend == "gemini":
                self._last_llm_model_id = self.gemini_model
                self._last_llm_image_source = "not_supported_by_backend"
                self._terminal_dump_llm_input(call_id, call_reason, prompt, call_context=call_context)
                raw_out = self._run_gemini_api(prompt)
            else:
                self._last_llm_model_id = self.api_model
                if self._llm_image_input_active():
                    if self._use_bev_image() and self.vla_bev_video_enable:
                        video_url, media_io_kwargs = self._build_vla_video_url()
                        self._last_llm_image_data_url_len = len(video_url or "")
                        if not video_url:
                            raise ValueError("vla_bev_video_unavailable")
                    elif self._use_camera_video_input():
                        video_url, media_io_kwargs = self._build_camera_video_url()
                        self._last_llm_image_data_url_len = len(video_url or "")
                        if not video_url:
                            raise ValueError("vla_camera_video_unavailable")
                    else:
                        image_data_url = self._build_vla_image_data_url(
                            call_context.get("event_image_path", "")
                        )
                        self._last_llm_image_data_url_len = len(image_data_url or "")
                else:
                    self._last_llm_image_source = "disabled_by_flag"
                self._terminal_dump_llm_input(call_id, call_reason, prompt, call_context=call_context)
                raw_out = self._run_http_api(
                    prompt,
                    image_data_url,
                    video_url=video_url,
                    media_io_kwargs=media_io_kwargs,
                )
            self._last_llm_error = ""
            self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._last_llm_remote_ok = True
            self._last_llm_raw_text = str(raw_out or "")
            self._last_llm_raw_preview = (raw_out or "").strip().replace("\n", " ")[:400]
            self._terminal_dump_llm_output(call_id, raw_out)
            if not self._last_llm_raw_text.strip():
                err_detail = "llm_no_output: backend returned empty content"
                self._last_llm_error = err_detail
                self._last_llm_remote_ok = False
                self._dump_llm_error(
                    call_id,
                    err_detail,
                    prompt=prompt,
                    exception_type="NoOutputError",
                )
                rospy.logwarn(
                    "[colregs_llm_decision] 【LLM】远端无有效输出，已用预置兜底 | 方式=%s | 第 %d 次 | 用时=%.0f ms",
                    self._backend_name_zh(self.llm_backend),
                    call_id,
                    self._last_llm_elapsed_ms,
                )
                out = self._fallback_classification()
                out["source"] = "fallback_after_llm_no_output"
                out["debug_call_id"] = call_id
                return self._attach_traj_constraints(out)
            pub_raw = self._last_llm_raw_text
            if self.raw_output_max_chars > 0:
                pub_raw = pub_raw[: self.raw_output_max_chars]
            self._raw_pub.publish(String(data=pub_raw))
            if self.debug_enable and (not self.terminal_clean_mode):
                preview = (raw_out or "").strip().replace("\n", " ")
                if self.llm_terminal_preview_chars > 0:
                    preview = preview[: self.llm_terminal_preview_chars]
                rospy.loginfo(
                    "[colregs_llm_decision] 【LLM】远端调用成功 | 方式=%s | 第 %d 次 | 网络耗时=%.0f ms | 模型=%s | 返回长度=%d 字 | 预览=%s",
                    self._backend_name_zh(self.llm_backend),
                    call_id,
                    self._last_llm_elapsed_ms,
                    self._last_llm_model_id,
                    len(raw_out or ""),
                    preview,
                )
                self._publish_debug(
                    "llm_remote_ok",
                    {
                        "llm_call_id": call_id,
                        "raw_len": len(raw_out or ""),
                        "llm_image_source": self._last_llm_image_source,
                    },
                )
                raw_text = self._last_llm_raw_text
                if self.vlm_check_raw_max_chars > 0:
                    raw_text = raw_text[: self.vlm_check_raw_max_chars]
                self._publish_vlm_check(
                    "vlm_remote_ok",
                    {
                        "llm_call_id": call_id,
                        "image_source": self._last_llm_image_source,
                        "raw_output": raw_text,
                        "raw_len": len(self._last_llm_raw_text or ""),
                    },
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                err_detail = (e.stderr or str(e)).strip()
            else:
                err_detail = str(e)
            self._last_llm_error = err_detail
            self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._last_llm_remote_ok = False
            self._last_llm_raw_preview = ""
            self._last_llm_raw_text = ""
            self._dump_llm_error(
                call_id,
                err_detail,
                prompt=prompt,
                exception_type=type(e).__name__,
                traceback_text=traceback.format_exc(),
            )
            rospy.logwarn(
                "[colregs_llm_decision] 【LLM】调用失败，已用预置兜底 | 方式=%s | 用时=%.0f ms | 原因=%s",
                self._backend_name_zh(self.llm_backend),
                self._last_llm_elapsed_ms,
                err_detail[:400],
            )
            if self.llm_backend == "dinov2":
                self._publish_debug("dinov2_error", {"llm_call_id": call_id, "error": err_detail[:400]})
                return None
                return None
            out = self._fallback_classification()
            out["source"] = "fallback_after_llm_error"
            out["debug_call_id"] = call_id
            if self.debug_enable and (not self.terminal_clean_mode):
                rospy.logwarn(
                    "[colregs_llm_decision] 【LLM】第 %d 次：预置兜底（Ollama/进程错误）| 耗时=%.0f ms",
                    call_id,
                    self._last_llm_elapsed_ms,
                )
            self._publish_debug("llm_error_fallback", {"llm_call_id": call_id})
            return self._attach_traj_constraints(out)
        except (ValueError, RuntimeError, json.JSONDecodeError, OSError) as e:
            err_detail = str(e)
            self._last_llm_error = err_detail
            self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._last_llm_remote_ok = False
            self._last_llm_raw_preview = ""
            self._last_llm_raw_text = ""
            self._dump_llm_error(
                call_id,
                err_detail,
                prompt=prompt,
                exception_type=type(e).__name__,
                traceback_text=traceback.format_exc(),
            )
            rospy.logwarn(
                "[colregs_llm_decision] 【LLM】远端调用失败 | 方式=%s | 第 %d 次 | 用时=%.0f ms | 原因=%s",
                self._backend_name_zh(self.llm_backend),
                call_id,
                self._last_llm_elapsed_ms,
                err_detail[:300],
            )
            if self.llm_backend == "dinov2":
                self._publish_debug("dinov2_error", {"llm_call_id": call_id, "error": err_detail[:300]})
                return None
                return None
            out = self._fallback_classification()
            out["source"] = "fallback_after_llm_error"
            out["debug_call_id"] = call_id
            if self.debug_enable and (not self.terminal_clean_mode):
                rospy.logwarn(
                    "[colregs_llm_decision] 【LLM】第 %d 次：已兜底 | 耗时=%.0f ms | 详情=%s",
                    call_id,
                    self._last_llm_elapsed_ms,
                    err_detail[:200],
                )
            self._publish_debug("llm_error_fallback", {"llm_call_id": call_id})
            return self._attach_traj_constraints(out)

        parsed = self._extract_json(raw_out)
        if not isinstance(parsed, dict):
            self._last_llm_error = "parse_error: non-json output"
            self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._last_llm_remote_ok = True
            out = self._fallback_classification()
            out["source"] = "fallback_after_parse_error"
            out["debug_call_id"] = call_id
            if self.debug_enable and (not self.terminal_clean_mode):
                rospy.logwarn(
                    "[colregs_llm_decision] 【LLM】远端已成功但 JSON 解析失败，已用预置兜底 | 第 %d 次 | 用时=%.0f ms | 解析路径=%s | 原始片段=%s",
                    call_id,
                    self._last_llm_elapsed_ms,
                    self._last_parse_diag,
                    (raw_out or "").strip()[:240],
                )
                self._publish_debug(
                    "llm_parse_error_fallback",
                    {
                        "llm_call_id": call_id,
                        "raw_len": len(raw_out or ""),
                    },
                )
            return self._attach_traj_constraints(out)
        self._last_llm_error = ""
        self._last_llm_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        parsed["debug_call_id"] = call_id
        out = self._normalize_llm_result(parsed, expected_ids=expected_target_ids)
        if self.debug_enable or self.terminal_llm_io_enable:
            colreg_action = out.get("colreg_action", out.get("trajectory_constraints", {}).get("colreg_action", ""))
            course_action = out.get("course_action", out.get("trajectory_constraints", {}).get("course_action", ""))
            speed_action = out.get("speed_action", out.get("trajectory_constraints", {}).get("speed_action", ""))
            rospy.loginfo(
                "[colregs_llm_decision] 【LLM】决策完成（成功）| 方式=%s | 第 %d 次 | 总用时=%.0f ms | 模型=%s | course_action=%s | speed_action=%s | colreg_action=%s | 置信度=%.2f",
                self._backend_name_zh(self.llm_backend),
                call_id,
                self._last_llm_elapsed_ms,
                self._last_llm_model_id,
                course_action,
                speed_action,
                colreg_action,
                out["confidence"],
            )
            if self.debug_enable:
                self._publish_debug(
                    "llm_decision_ok",
                    {
                        "llm_call_id": call_id,
                        "course_action": course_action,
                        "speed_action": speed_action,
                        "colreg_action": colreg_action,
                        "perception_diag": self._perception_diag(),
                    },
                )
        return out

    def _maybe_update_classification(self):
        self._consume_pending_llm_result()
        now = rospy.Time.now()
        rising_edge = (not self._prev_trigger_for_llm) and self._trigger
        self._prev_trigger_for_llm = self._trigger
        d_min = self._pointcloud_trigger_distance()
        pretrigger_active = self._semantic_pretrigger_active(d_min)

        if not self._state:
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_no_state"
            self._append_vlm_gate_status("skip_no_state")
            if self.debug_enable:
                rospy.loginfo_throttle(
                    2.0,
                    "[colregs_llm_decision] 【LLM】跳过：尚无状态估计数据",
                )
            return

        sensor_front_diag = self._sensor_front_hazard_diag()
        self._last_sensor_front_hazard_diag = sensor_front_diag
        if self.sensor_front_hazard_enable and (not bool(sensor_front_diag.get("hazard", False))):
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_no_front_sensor_hazard"
            self._llm_locked_after_success = False
            self._llm_locked_call_id = -1
            self._last_llm_attempt_trigger_generation = -1
            self._last_cls = self._sensor_front_clear_classification(sensor_front_diag)
            self._append_vlm_gate_status(
                "skip_no_front_sensor_hazard",
                {"sensor_front_hazard_diag": sensor_front_diag},
            )
            if self.debug_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[colregs_llm_decision] 【LLM】跳过：前方危险区内无传感器目标 | gate=%s",
                    json.dumps(sensor_front_diag, ensure_ascii=True)[:700],
                )
            return

        if self.llm_call_requires_trigger and (not self._trigger) and (not pretrigger_active):
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_no_trigger"
            self._append_vlm_gate_status(
                "skip_no_trigger",
                {"sensor_front_hazard_diag": sensor_front_diag},
            )
            return

        if self.llm_call_requires_trigger and self.llm_require_trigger_low_before_call and (not self._trigger_seen_low):
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_trigger_not_armed"
            self._append_vlm_gate_status(
                "skip_trigger_not_armed",
                {"sensor_front_hazard_diag": sensor_front_diag},
            )
            if self.debug_enable:
                rospy.loginfo_throttle(
                    2.0,
                    "[colregs_llm_decision] 【LLM】跳过：尚未观测到 trigger=false 基线，启动期不允许调用",
                )
            return

        if self.llm_disable_after_first_success and self._llm_locked_after_success:
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_locked_after_success"
            self._append_vlm_gate_status(
                "skip_locked_after_success",
                {"sensor_front_hazard_diag": sensor_front_diag},
            )
            return

        if not self._latest_ros_image_available_for_capture():
            if self._use_pointcloud_bev_image():
                missing_image_reason = "skip_no_pointcloud_bev"
            else:
                missing_image_reason = "skip_no_image"
            self._last_llm_called = False
            self._last_llm_call_reason = missing_image_reason
            if (now - self._last_no_image_skip_log_t).to_sec() >= 1.0:
                self._last_no_image_skip_log_t = now
            self._append_vlm_gate_status(missing_image_reason)
            if self.debug_enable:
                rospy.loginfo_throttle(
                    2.0,
                    "[colregs_llm_decision] 【LLM】跳过：%s | diag=%s",
                    self._call_reason_zh(missing_image_reason),
                    json.dumps(self._vla_image_diag(), ensure_ascii=True)[:500],
                )
            return

        self._save_diagnostic_snapshot_image("pre_fov_diagnostic")
        fov_gate_ok, fov_gate_reason, fov_gate_diag = self._vla_fov_gate_ready(now)
        if not fov_gate_ok:
            self._last_llm_called = False
            self._last_llm_call_reason = fov_gate_reason
            if (now - self._last_fov_gate_skip_log_t).to_sec() >= 1.0:
                self._last_fov_gate_skip_log_t = now
            self._save_diagnostic_snapshot_image(fov_gate_reason, fov_gate_diag=fov_gate_diag)
            self._append_vlm_gate_status(
                fov_gate_reason,
                {"fov_gate_diag": fov_gate_diag},
            )
            if self.debug_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[colregs_llm_decision] 【LLM】跳过：%s | fov=%s",
                    self._call_reason_zh(fov_gate_reason),
                    json.dumps(fov_gate_diag, ensure_ascii=True)[:700],
                )
            return

        unknown_gate_ok, unknown_gate_diag = self._fov_unknown_track_gate_ready(fov_gate_diag)
        fov_gate_diag["unknown_track_gate"] = unknown_gate_diag
        if not unknown_gate_ok:
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_no_unknown_fov_track"
            self._append_vlm_gate_status(
                "skip_no_unknown_fov_track",
                {"fov_gate_diag": fov_gate_diag},
            )
            if self.debug_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[colregs_llm_decision] 【LLM】跳过：FOV内没有Unknown track | gate=%s",
                    json.dumps(unknown_gate_diag, ensure_ascii=True),
                )
            return

        image_buffer_ready, image_buffer_diag = self._image_buffer_ready_for_vlm()
        if not image_buffer_ready:
            if self._use_pointcloud_bev_image():
                warming_reason = "skip_pointcloud_bev_warming"
            else:
                warming_reason = "skip_image_buffer_warming"
            self._last_llm_called = False
            self._last_llm_call_reason = warming_reason
            self._append_vlm_gate_status(
                warming_reason,
                {
                    "fov_gate_diag": fov_gate_diag,
                    "image_buffer_diag": image_buffer_diag,
                },
            )
            if self.debug_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[colregs_llm_decision] 【LLM】跳过：%s | buffer=%s",
                    self._call_reason_zh(warming_reason),
                    json.dumps(image_buffer_diag, ensure_ascii=True),
                )
            return

        history_warmup_diag = self._history_warmup_diag_for_targets(fov_gate_diag)
        if not bool(history_warmup_diag.get("ready", False)):
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_target_history_warming"
            self._append_vlm_gate_status(
                "skip_target_history_warming",
                {
                    "fov_gate_diag": fov_gate_diag,
                    "target_history_warmup_diag": history_warmup_diag,
                },
            )
            if self.debug_enable:
                rospy.loginfo_throttle(
                    1.0,
                    "[colregs_llm_decision] 【LLM】跳过：目标历史仍在预热 | history=%s",
                    json.dumps(history_warmup_diag, ensure_ascii=True),
                )
            return

        if not (self.llm_fov_gate_enable and self.llm_fov_gate_bypass_distance):
            distance_gate_m = self.llm_pretrigger_distance_m if pretrigger_active else self.llm_trigger_distance_m
            if d_min is None:
                if self.llm_require_pointcloud_distance:
                    self._last_llm_called = False
                    self._last_llm_call_reason = "skip_no_distance"
                    self._append_vlm_gate_status(
                        "skip_no_distance",
                        {"fov_gate_diag": fov_gate_diag},
                    )
                    if self.debug_enable:
                        rospy.loginfo_throttle(
                            2.0,
                            "[colregs_llm_decision] 【LLM】跳过：尚无点云目标距离信息",
                        )
                    return
            elif d_min > distance_gate_m:
                self._last_llm_called = False
                self._last_llm_call_reason = "skip_distance_gate"
                self._append_vlm_gate_status(
                    "skip_distance_gate",
                    {
                        "pointcloud_trigger_distance_m": d_min,
                        "llm_trigger_distance_m": distance_gate_m,
                        "semantic_pretrigger_active": bool(pretrigger_active),
                        "fov_gate_diag": fov_gate_diag,
                    },
                )
                if self.debug_enable:
                    rospy.loginfo_throttle(
                        2.0,
                        "[colregs_llm_decision] 【LLM】跳过：目标最近距离 %.2fm > 门槛 %.2fm",
                        d_min,
                        distance_gate_m,
                    )
                return

        trigger_generation = int(self._trigger_generation)
        call_generation = None if pretrigger_active else trigger_generation

        if rising_edge:
            dt_since_last_call = (now - self._last_llm_t).to_sec()
            if dt_since_last_call < self.llm_rising_edge_min_interval_s:
                self._last_llm_called = False
                self._last_llm_call_reason = "skip_rising_edge_cooldown"
                self._append_vlm_gate_status(
                    "skip_rising_edge_cooldown",
                    {"dt_since_last_call_s": dt_since_last_call},
                )
                if self.debug_enable:
                    rospy.logwarn_throttle(
                        2.0,
                        "[colregs_llm_decision] 【LLM】触发抖动冷却中：距上次调用 %.2fs < %.2fs，跳过",
                        dt_since_last_call,
                        self.llm_rising_edge_min_interval_s,
                    )
                return

        if now < self._llm_quota_block_until:
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_quota_guard"
            self._append_vlm_gate_status("skip_quota_guard")
            if self.debug_enable:
                remain = (self._llm_quota_block_until - now).to_sec()
                rospy.logwarn_throttle(
                    2.0,
                    "[colregs_llm_decision] 【LLM】配额保护冷却中，剩余 %.1fs，跳过调用",
                    max(0.0, remain),
                )
            return

        horizon = now.to_sec() - 60.0
        self._llm_call_timestamps = [t for t in self._llm_call_timestamps if t >= horizon]
        if len(self._llm_call_timestamps) >= max(1, self.llm_max_calls_per_minute):
            self._llm_quota_block_until = rospy.Time.from_sec(
                now.to_sec() + max(1.0, self.llm_quota_cooldown_s)
            )
            self._last_llm_called = False
            self._last_llm_call_reason = "skip_quota_guard"
            self._append_vlm_gate_status(
                "skip_quota_guard",
                {"calls_last_minute": len(self._llm_call_timestamps)},
                force=True,
            )
            if self.debug_enable:
                rospy.logwarn(
                    "[colregs_llm_decision] 【LLM】过去60秒调用次数=%d，超过阈值=%d，进入 %.1fs 冷却",
                    len(self._llm_call_timestamps),
                    max(1, self.llm_max_calls_per_minute),
                    max(1.0, self.llm_quota_cooldown_s),
                )
            return

        if self.llm_once_per_trigger:
            if self._last_llm_attempt_trigger_generation == trigger_generation:
                self._last_llm_called = False
                self._last_llm_call_reason = "skip_once_per_trigger_already_called"
                self._append_vlm_gate_status("skip_once_per_trigger_already_called")
                return
        else:
            dt = (now - self._last_llm_t).to_sec()
            if (not rising_edge) and (dt < self.llm_update_interval_s):
                self._last_llm_called = False
                self._last_llm_call_reason = "skip_interval_guard"
                self._append_vlm_gate_status(
                    "skip_interval_guard",
                    {"dt_since_last_call_s": dt, "required_interval_s": self.llm_update_interval_s},
                )
                if self.debug_enable:
                    rospy.loginfo_throttle(
                        2.0,
                        "[colregs_llm_decision] 【LLM】跳过：未到周期间隔（已过 %.2f 秒，需 ≥ %.2f 秒）",
                        dt,
                        self.llm_update_interval_s,
                    )
                return

        self._last_llm_t = now
        self._llm_call_timestamps.append(now.to_sec())
        if pretrigger_active:
            reason = "semantic_pretrigger"
        elif fov_gate_reason in (
            "fov_all_tracks_ready",
            "fov_partial_tracks_ready",
            "fov_state_fallback_ready",
            "fov_soft_tracks_ready",
        ):
            reason = fov_gate_reason
        elif rising_edge:
            reason = "trigger_rising_edge"
        elif self.llm_once_per_trigger:
            reason = "trigger_generation_first_call"
        else:
            reason = "periodic_recall"

        if pretrigger_active or (not self.llm_once_per_trigger):
            self._last_snapshot_trigger_generation = -1
        event_image_path = self._capture_gate_snapshot_if_needed(
            trigger_generation,
            reason,
            fov_gate_diag=fov_gate_diag,
        )
        all_prompt_targets = self._build_targets_prompt_context()
        call_context = {
            "event_image_path": event_image_path or self._event_image_path,
            "event_input_image_path": self._event_input_image_path,
            "fov_gate_diag": fov_gate_diag,
            "sensor_front_hazard_diag": sensor_front_diag,
            "targets": all_prompt_targets,
            "semantic_pretrigger_active": bool(pretrigger_active),
            "pointcloud_trigger_distance_m": d_min,
        }

        if self.llm_backend != "dinov2" and self.pending_safety_enable:
            # 先用规则兜底立即出动作，避免阻塞等待远端 LLM 带来的 5s 级延迟。
            pre = self._pending_safety_classification()
            self._last_cls = self._attach_traj_constraints(pre)
            self._activate_traj_plan()

        if self.llm_async_enable:
            started = self._start_async_llm_call(reason, call_generation, call_context=call_context)
            if started:
                self._last_llm_attempt_trigger_generation = trigger_generation
            if (not started) and self.debug_enable:
                rospy.loginfo_throttle(
                    2.0,
                    "[colregs_llm_decision] LLM 异步调用仍在进行，沿用当前分类",
                )
            return

        self._last_llm_attempt_trigger_generation = trigger_generation
        result = self._llm_classify(reason, trigger_generation=call_generation, call_context=call_context)
        if result is not None:
            self._last_cls = result
            self._lock_llm_after_success_if_needed(result)
            self._activate_traj_plan()

    def _build_cmd(self):
        return self._build_continuous_cmd()

    def _build_decision(self):
        now_s = rospy.Time.now().to_sec()
        tcpa = float(self._state.get("tcpa_s", 1e9))
        dcpa = float(self._state.get("dcpa_m", 1e9))
        brief_diag = self._perception_brief()
        classification_source = self._last_cls.get("source", "unknown")
        force_keep_course = bool(self._last_cls.get("force_keep_course", False))
        sensor_front_hazard = bool(
            self._last_cls.get(
                "sensor_front_hazard",
                self._last_sensor_front_hazard_diag.get("hazard", False),
            )
        )
        constraints = self._last_cls.get("trajectory_constraints", {}) or {}
        colreg_action = self._last_cls.get(
            "colreg_action",
            constraints.get("colreg_action", ""),
        )
        course_action = self._last_cls.get(
            "course_action",
            constraints.get("course_action", ""),
        )
        speed_action = self._last_cls.get(
            "speed_action",
            constraints.get("speed_action", ""),
        )
        colreg_weights = self._last_cls.get(
            "colreg_weights",
            constraints.get("colreg_weights", {}),
        )
        course_is_avoidance = self._canonical_course_action(course_action) != "KEEP_COURSE"
        pretrigger_llm_action = bool((not self._trigger) and classification_source == "llm" and course_is_avoidance)
        if force_keep_course:
            action = "KEEP_COURSE"
        elif classification_source == "pending_safety_pre_llm" and self._llm_inflight:
            action = "PENDING_SAFETY"
        elif self._trigger or pretrigger_llm_action:
            action = "AVOIDANCE_ACTION"
        else:
            action = "KEEP_COURSE"
        decision = {
            "stamp": now_s,
            "mode": "llm_vla_track_classifier",
            "llm_backend": self.llm_backend,
            "llm_backend_zh": self._backend_name_zh(self.llm_backend),
            "llm_once_per_trigger": self.llm_once_per_trigger,
            "llm_call_reason_zh": self._call_reason_zh(self._last_llm_call_reason),
            "llm_result_zh": self._llm_result_summary_zh(),
            "action": action,
            "reasoning": self._last_cls.get("reasoning", ""),
            "classification_source": classification_source,
            "confidence": self._last_cls.get("confidence", 0.0),
            "track_classifications": self._last_cls.get("track_classifications", []),
            "targets_json": self._last_cls.get("targets_json", []),
            "course_action": course_action,
            "speed_action": speed_action,
            "colreg_action": colreg_action,
            "colreg_weights": colreg_weights,
            "trajectory_constraints": constraints,
            "llm_error": self._last_llm_error,
            "llm_remote_ok": self._last_llm_remote_ok,
            "llm_model": self._last_llm_model_id,
            "llm_image_source": self._last_llm_image_source,
            "event_image_path": self._event_image_path,
            "event_image_url": self._event_image_url,
            "vlm_score_plot_path": self._last_vlm_score_plot_path,
            "vlm_snapshot_annotated": self._last_vlm_snapshot_annotated,
            "vlm_snapshot_annotation_error": self._last_vlm_snapshot_annotation_error,
            "llm_elapsed_ms": self._last_llm_elapsed_ms,
            "llm_called": self._last_llm_called,
            "llm_call_reason": self._last_llm_call_reason,
            "llm_call_seq": self._last_cls.get("debug_call_id", -1),
            "llm_inflight": self._llm_inflight,
            "llm_locked_after_success": self._llm_locked_after_success,
            "llm_locked_call_id": self._llm_locked_call_id,
            "force_keep_course": force_keep_course,
            "sensor_front_hazard": sensor_front_hazard,
            "sensor_front_hazard_diag": self._last_cls.get(
                "sensor_front_hazard_diag",
                self._last_sensor_front_hazard_diag,
            ),
            "dcpa_m": dcpa,
            "tcpa_s": tcpa,
            "trigger": self._trigger,
            "recovery_active": self._recovery_active,
            "perception_brief": brief_diag,
        }
        if self._should_attach_full_diag(now_s):
            decision["perception_diag"] = self._perception_diag()
            decision["recovery_debug"] = {
                "pre_encounter_heading": self._pre_encounter_heading,
                "pre_encounter_pos": self._pre_encounter_pos,
                "lateral_error_m": self._last_lateral_error,
                "post_recovery_keep_course": self._post_recovery_keep_course,
            }
        return decision

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            self._maybe_update_classification()
            # self._terminal_status_debug()
            cmd = self._build_cmd()
            decision = self._build_decision()
            self._cmd_pub.publish(cmd)
            self._recovery_active_pub.publish(Bool(data=self._recovery_active))
            self._decision_pub.publish(String(data=json.dumps(decision, ensure_ascii=True)))
            if self.debug_enable and self.cmd_log_enable:
                now_s = rospy.Time.now().to_sec()
                sig = (
                    bool(self._trigger),
                    bool(self._recovery_active),
                    str(decision.get("action", "")),
                    str(decision.get("colreg_action", "")),
                    str(decision.get("classification_source", "")),
                    round(float(cmd.linear.x), 2),
                    round(float(cmd.angular.z), 2),
                )
                due = (now_s - self._last_cmd_log_t) >= max(0.5, self.cmd_log_interval_s)
                changed = sig != self._last_cmd_log_sig
                if due or changed:
                    self._last_cmd_log_t = now_s
                    self._last_cmd_log_sig = sig
                    rospy.loginfo(
                        "[colregs_llm_decision] cmd_out v=%.3f w=%.3f trigger=%s recovery=%s action=%s colreg_action=%s src=%s",
                        cmd.linear.x,
                        cmd.angular.z,
                        str(self._trigger),
                        str(self._recovery_active),
                        decision.get("action", ""),
                        decision.get("colreg_action", ""),
                        decision.get("classification_source", ""),
                    )
            rate.sleep()


if __name__ == "__main__":
    ColregsLlmDecisionNode().run()
