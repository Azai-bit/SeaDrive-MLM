#!/usr/bin/env python

import json
import os
import re
import subprocess
import math
import time
from collections import deque

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import String


class VlamissionNode(object):
    def __init__(self):
        self.node_ready = False
        image_topic = rospy.get_param(
            "~image_topic", "/myboat/sensors/cameras/front_camera/image_raw"
        )
        mission_topic = rospy.get_param("~mission_topic", "/vla/mission_text")
        odom_topic = rospy.get_param("~odom_topic", "/myboat/odom")
        cloud_topic = rospy.get_param(
            "~cloud_topic", "/myboat/sensors/lidar_wamv/points"
        )
        explored_grid_topic = rospy.get_param(
            "~explored_grid_topic", "/sdf_map/explored_grid"
        )
        value_map_topic = rospy.get_param("~value_map_topic", "/sdf_map/value_map")
        self.image_save_dir = rospy.get_param("~image_save_dir", os.path.join(os.getcwd(), "tmp"))
        self.value_decay = rospy.get_param("~value_decay", 0.97)
        self.value_paint_radius_m = rospy.get_param("~value_paint_radius_m", 2.0)
        self.value_disk_edge_gain = rospy.get_param("~value_disk_edge_gain", 0.30)
        self.value_region_radius_scale = rospy.get_param("~value_region_radius_scale", 1.8)
        self.value_diffuse_enable = rospy.get_param("~value_diffuse_enable", True)
        self.value_diffuse_kernel = rospy.get_param("~value_diffuse_kernel", 9)
        self.value_diffuse_sigma = rospy.get_param("~value_diffuse_sigma", 2.2)
        self.value_diffuse_mix = rospy.get_param("~value_diffuse_mix", 0.60)
        self.value_diffuse_min_value = rospy.get_param("~value_diffuse_min_value", 4.0)
        self.value_ema_alpha = rospy.get_param("~value_ema_alpha", 0.35)
        self.score_update_period_s = rospy.get_param("~score_update_period_s", 2.0)
        self.vlm_min_interval_s = rospy.get_param("~vlm_min_interval_s", 30.0)
        self.vlm_trigger_mutation_mean_threshold = rospy.get_param(
            "~vlm_trigger_mutation_mean_threshold", 1.2
        )
        self.vlm_trigger_mutation_sum_threshold = rospy.get_param(
            "~vlm_trigger_mutation_sum_threshold", 1800.0
        )
        self.candidate_count = rospy.get_param("~candidate_count", 12)
        self.candidate_sector_count = rospy.get_param("~candidate_sector_count", 12)
        self.region_grid_rows = rospy.get_param("~region_grid_rows", 10)
        self.region_grid_cols = rospy.get_param("~region_grid_cols", 10)
        self.candidate_window_radius_m = rospy.get_param("~candidate_window_radius_m", 2.0)
        self.candidate_min_frontier_ratio = rospy.get_param("~candidate_min_frontier_ratio", 0.03)
        self.candidate_paint_radius_m = rospy.get_param("~candidate_paint_radius_m", 3.0)
        self.candidate_border_margin_m = rospy.get_param("~candidate_border_margin_m", 3.0)
        self.candidate_min_spacing_m = rospy.get_param("~candidate_min_spacing_m", 4.0)
        self.candidate_global_stride_cells = rospy.get_param("~candidate_global_stride_cells", 8)
        self.candidate_geom_mix = rospy.get_param("~candidate_geom_mix", 0.15)
        self.candidate_require_vlm_score = rospy.get_param("~candidate_require_vlm_score", True)
        self.candidate_min_scored_ratio = rospy.get_param("~candidate_min_scored_ratio", 0.50)
        self.candidate_missing_fill_scale = rospy.get_param("~candidate_missing_fill_scale", 0.75)
        self.candidate_semantic_inject_enable = rospy.get_param(
            "~candidate_semantic_inject_enable", False
        )
        self.candidate_debug_log_enable = rospy.get_param("~candidate_debug_log_enable", True)
        self.profile_time_enable = rospy.get_param("~profile_time_enable", True)
        self.debug_unknown_only_mode = rospy.get_param("~debug_unknown_only_mode", False)
        self.debug_unknown_window_m = rospy.get_param("~debug_unknown_window_m", 4.0)
        self.debug_unknown_use_ema = rospy.get_param("~debug_unknown_use_ema", False)
        self.candidate_distance_bins_m = self.parse_float_list_param(
            rospy.get_param("~candidate_distance_bins_m", [6.0, 12.0, 20.0, 30.0]),
            [6.0, 12.0, 20.0, 30.0],
        )
        self.max_region_distance_m = rospy.get_param("~max_region_distance_m", 40.0)
        self.bev_image_path = rospy.get_param(
            "~bev_image_path", os.path.join(self.image_save_dir, "fused_bev_latest.png")
        )
        self.camera_hfov_deg = rospy.get_param("~camera_hfov_deg", 90.0)
        self.camera_vfov_deg = rospy.get_param("~camera_vfov_deg", 60.0)
        self.cloud_proj_max_points = rospy.get_param("~cloud_proj_max_points", 12000)
        self.lidar_mount_x = rospy.get_param("~lidar_mount_x", -0.4)
        self.lidar_mount_y = rospy.get_param("~lidar_mount_y", 0.0)
        self.lidar_mount_z = rospy.get_param("~lidar_mount_z", 0.8)
        self.camera_mount_x = rospy.get_param("~camera_mount_x", 0.75)
        self.camera_mount_y = rospy.get_param("~camera_mount_y", 0.0)
        self.camera_mount_z = rospy.get_param("~camera_mount_z", 0.5)
        self.camera_mount_roll = rospy.get_param("~camera_mount_roll", 0.0)
        self.camera_mount_pitch = rospy.get_param("~camera_mount_pitch", math.radians(15.0))
        self.camera_mount_yaw = rospy.get_param("~camera_mount_yaw", 0.0)
        self.semantic_base_z_min = rospy.get_param("~semantic_base_z_min", -0.30)
        self.use_tf_extrinsics = rospy.get_param("~use_tf_extrinsics", True)
        self.base_link_frame = rospy.get_param("~base_link_frame", "myboat/base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "")
        self.tf_lookup_timeout_s = rospy.get_param("~tf_lookup_timeout_s", 0.05)
        self.camera_frame_is_optical = rospy.get_param("~camera_frame_is_optical", True)
        self.projection_mode = rospy.get_param("~projection_mode", "auto")
        self.projection_flip_horizontal = rospy.get_param(
            "~projection_flip_horizontal", True
        )
        self.semantic_min_range_m = rospy.get_param("~semantic_min_range_m", 1.0)
        self.semantic_max_range_m = rospy.get_param("~semantic_max_range_m", 25.0)
        self.semantic_paint_radius_m = rospy.get_param("~semantic_paint_radius_m", 1.0)
        self.semantic_decay = rospy.get_param("~semantic_decay", 0.92)
        self.semantic_threshold = rospy.get_param("~semantic_threshold", 0.20)
        self.semantic_col_step_px = rospy.get_param("~semantic_col_step_px", 8)
        self.semantic_min_ratio = rospy.get_param("~semantic_min_ratio", 0.06)
        self.semantic_red_min_ratio = rospy.get_param("~semantic_red_min_ratio", 0.06)
        self.semantic_black_min_ratio = rospy.get_param("~semantic_black_min_ratio", 0.05)
        self.semantic_ratio_margin = rospy.get_param("~semantic_ratio_margin", 0.01)
        self.semantic_score_margin = rospy.get_param("~semantic_score_margin", 0.05)
        self.semantic_strength_boost = rospy.get_param("~semantic_strength_boost", 3.5)
        self.semantic_strength_min = rospy.get_param("~semantic_strength_min", 0.35)
        self.event_trigger_enable = rospy.get_param("~event_trigger_enable", False)
        self.red_pixel_trigger_threshold = rospy.get_param(
            "~red_pixel_trigger_threshold", 400
        )
        self.red_pixel_ratio_trigger_threshold = rospy.get_param(
            "~red_pixel_ratio_trigger_threshold", 0.006
        )
        self.event_min_interval_s = rospy.get_param("~event_min_interval_s", 12.0)
        self.event_check_roi_top_ratio = rospy.get_param(
            "~event_check_roi_top_ratio", 0.35
        )
        self.value_trace_steps = rospy.get_param("~value_trace_steps", 4)
        self.value_semantic_boost = rospy.get_param("~value_semantic_boost", 60.0)
        self.value_semantic_min_score = rospy.get_param("~value_semantic_min_score", 0.10)
        self.sync_enable = rospy.get_param("~sync_enable", True)
        self.sync_max_dt_s = rospy.get_param("~sync_max_dt_s", 0.12)
        self.sync_buffer_size = rospy.get_param("~sync_buffer_size", 60)
        self.prompt_min_regions = rospy.get_param("~prompt_min_regions", 6)
        self.prompt_max_regions = rospy.get_param("~prompt_max_regions", 10)
        self.bev_overlay_grid_enable = rospy.get_param("~bev_overlay_grid_enable", True)
        self.bev_overlay_grid_step_cells = rospy.get_param(
            "~bev_overlay_grid_step_cells", 50
        )

        # ollama 命令行配置，可通过参数覆盖
        self.ollama_cmd = rospy.get_param(
            "~ollama_command", ["ollama", "run", "ros_ai_agent","--think=false"]
        )
        self.ollama_timeout = rospy.get_param("~ollama_timeout", 60.0)

        # 默认任务提示词（当还未收到自然语言任务时使用）
        self.default_mission_text = rospy.get_param(
            "~default_mission_text",
            (
                "你是一个水面机器人 VLA 任务规划助手。\n"
                "你只负责给候选区域打分，不生成新候选。\n"
                "命令1：禁止输出任何解释文本。\n"
                "命令2：只输出一行，格式必须为：\n"
                "CANDIDATE_SCORES: [{\"id\":<int>,\"score\":<0到1>}, ...]\n"
                "命令2.1：键名必须是 CANDIDATE_SCORES，不得使用 CAND_MILESTONES 或其他键名。\n"
                "命令3：仅对输入的 candidate id 打分；不要新增 id，不要遗漏高优先级 id。\n"
                "命令4：分数越高表示越值得下一步探索，需兼顾任务相关性、全局增益、避免重复覆盖。\n"
            )
        )

        self.latest_mission_text = None
        self.latest_odom = None
        self.latest_cloud = None
        self.latest_explored_grid = None
        self.latest_camera_image = None
        self.cloud_buffer = deque(maxlen=max(5, int(self.sync_buffer_size)))
        self.last_sync_dt_s = None
        self.value_map = None
        self.semantic_black_score = None
        self.semantic_red_score = None
        self.bridge = CvBridge()
        self.mission_running = False
        self.last_event_trigger_time = rospy.Time(0)
        self.last_vlm_infer_time = rospy.Time(0)
        self.R_base_to_cam = self.rpy_to_rot(
            self.camera_mount_roll, self.camera_mount_pitch, self.camera_mount_yaw
        )
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 确保图像保存目录存在
        if not os.path.exists(self.image_save_dir):
            try:
                os.makedirs(self.image_save_dir)
            except OSError as e:
                rospy.logerr(
                    "[vla_mission_node] 创建图像保存目录失败: %s, error: %s",
                    self.image_save_dir,
                    str(e),
                )

        self.image_sub = rospy.Subscriber(
            image_topic, Image, self.image_cb, queue_size=1
        )
        self.mission_sub = rospy.Subscriber(
            mission_topic, String, self.mission_cb, queue_size=10
        )
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_cb, queue_size=10)
        self.cloud_sub = rospy.Subscriber(
            cloud_topic, PointCloud2, self.cloud_cb, queue_size=1
        )
        self.explored_grid_sub = rospy.Subscriber(
            explored_grid_topic, OccupancyGrid, self.explored_grid_cb, queue_size=1
        )
        self.value_map_pub = rospy.Publisher(
            value_map_topic, OccupancyGrid, queue_size=1, latch=True
        )

        rospy.loginfo(
            "[vla_mission_node] image_topic=%s, mission_topic=%s, odom_topic=%s, cloud_topic=%s",
            image_topic,
            mission_topic,
            odom_topic,
            cloud_topic,
        )
        rospy.loginfo(
            "[vla_mission_node] explored_grid_topic=%s -> value_map_topic=%s",
            explored_grid_topic,
            value_map_topic,
        )
        rospy.loginfo("[vla_mission_node] bev_image_path=%s", self.bev_image_path)
        rospy.loginfo(
            "[vla_mission_node] 使用 Ollama 命令: %s", " ".join(self.ollama_cmd)
        )
        rospy.loginfo(
            "[vla_mission_node] camera_hfov_deg=%.1f semantic_range=[%.1f, %.1f]m",
            self.camera_hfov_deg,
            self.semantic_min_range_m,
            self.semantic_max_range_m,
        )
        rospy.loginfo(
            "[vla_mission_node] camera_vfov_deg=%.1f cloud_proj_max_points=%d",
            self.camera_vfov_deg,
            self.cloud_proj_max_points,
        )
        rospy.loginfo(
            "[vla_mission_node] lidar_mount=(%.2f,%.2f,%.2f) camera_mount=(%.2f,%.2f,%.2f) camera_rpy=(%.2f,%.2f,%.2f)",
            self.lidar_mount_x,
            self.lidar_mount_y,
            self.lidar_mount_z,
            self.camera_mount_x,
            self.camera_mount_y,
            self.camera_mount_z,
            self.camera_mount_roll,
            self.camera_mount_pitch,
            self.camera_mount_yaw,
        )
        rospy.loginfo(
            "[vla_mission_node] use_tf_extrinsics=%s base_link_frame=%s camera_frame=%s optical_convention=%s",
            str(self.use_tf_extrinsics),
            self.base_link_frame,
            self.camera_frame if self.camera_frame else "(from image header)",
            str(self.camera_frame_is_optical),
        )
        rospy.loginfo(
            "[vla_mission_node] projection_mode=%s flip_horizontal=%s semantic_base_z_min=%.2f",
            self.projection_mode,
            str(self.projection_flip_horizontal),
            self.semantic_base_z_min,
        )
        rospy.loginfo(
            "[vla_mission_node] event_trigger_enable=%s(已弃用) red_pixel_trigger_threshold=%d red_pixel_ratio_trigger_threshold=%.4f min_interval=%.1fs",
            str(self.event_trigger_enable),
            self.red_pixel_trigger_threshold,
            self.red_pixel_ratio_trigger_threshold,
            self.event_min_interval_s,
        )
        rospy.loginfo(
            "[vla_mission_node] sync_enable=%s sync_max_dt_s=%.3f sync_buffer_size=%d",
            str(self.sync_enable),
            self.sync_max_dt_s,
            self.sync_buffer_size,
        )
        rospy.loginfo(
            "[vla_mission_node] score_period=%.1fs candidate_count=%d sectors=%d region_grid=%dx%d border_margin=%.1fm min_spacing=%.1fm stride=%d vlm_min_interval=%.1fs mutation(mean>=%.2f or sum>=%.1f) ema_alpha=%.2f geom_mix=%.2f require_vlm=%s min_scored_ratio=%.2f unknown_debug=%s",
            self.score_update_period_s,
            self.candidate_count,
            self.candidate_sector_count,
            self.region_grid_rows,
            self.region_grid_cols,
            self.candidate_border_margin_m,
            self.candidate_min_spacing_m,
            self.candidate_global_stride_cells,
            self.vlm_min_interval_s,
            self.vlm_trigger_mutation_mean_threshold,
            self.vlm_trigger_mutation_sum_threshold,
            self.value_ema_alpha,
            self.candidate_geom_mix,
            str(self.candidate_require_vlm_score),
            self.candidate_min_scored_ratio,
            str(self.debug_unknown_only_mode),
        )
        self.periodic_timer = rospy.Timer(
            rospy.Duration(max(0.5, float(self.score_update_period_s))),
            self.periodic_score_update_cb,
        )
        self.node_ready = True

    def image_cb(self, msg):
        # 仅缓存最新图像；红色像素事件触发机制已停用
        self.latest_camera_image = msg

    def periodic_score_update_cb(self, _event):
        if not self.node_ready or self.mission_running:
            return
        if self.latest_camera_image is None or self.latest_cloud is None:
            return
        if self.sync_enable:
            matched_cloud, dt_s = self.find_nearest_cloud(self.latest_camera_image.header.stamp)
            if matched_cloud is None:
                rospy.logwarn_throttle(
                    2.0,
                    "[vla_mission_node] 周期更新未匹配到近时刻点云，跳过。sync_max_dt_s=%.3f",
                    self.sync_max_dt_s,
                )
                return
            self.latest_cloud = matched_cloud
            self.last_sync_dt_s = dt_s
        self.run_mission_cycle("periodic_5s", None)

    def mission_cb(self, msg):
        self.latest_mission_text = msg.data

    def odom_cb(self, msg):
        self.latest_odom = msg

    def cloud_cb(self, msg):
        self.latest_cloud = msg
        self.cloud_buffer.append(msg)

    def find_nearest_cloud(self, target_stamp):
        if len(self.cloud_buffer) == 0:
            return None, None
        best_msg = None
        best_dt = None
        t_ref = target_stamp.to_sec()
        for m in self.cloud_buffer:
            dt = abs(m.header.stamp.to_sec() - t_ref)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_msg = m
        if best_msg is None or best_dt is None or best_dt > self.sync_max_dt_s:
            return None, None
        return best_msg, best_dt

    def explored_grid_cb(self, msg):
        self.latest_explored_grid = msg
        if self.value_map is None or not self.same_grid_info(msg, self.value_map):
            self.value_map = OccupancyGrid()
            self.value_map.header.frame_id = msg.header.frame_id
            self.value_map.info = msg.info
            self.value_map.data = [0] * (msg.info.width * msg.info.height)
            self.semantic_black_score = np.zeros(
                (msg.info.height, msg.info.width), dtype=np.float32
            )
            self.semantic_red_score = np.zeros(
                (msg.info.height, msg.info.width), dtype=np.float32
            )
            rospy.loginfo(
                "[vla_mission_node] 初始化 value_map: size=%dx%d res=%.3f",
                msg.info.width,
                msg.info.height,
                msg.info.resolution,
            )

    def build_prompt(self, candidates, image_path_override=None):
        parts = [self.default_mission_text]

        if self.latest_mission_text:
            parts.append("\n[自然语言任务]\n%s" % self.latest_mission_text)
        else:
            parts.append("\n[自然语言任务]\n(尚未收到任务指令，使用默认任务描述)")

        if self.latest_explored_grid is not None:
            stamp = self.latest_explored_grid.header.stamp.to_sec()
            image_path = image_path_override
            if image_path is None:
                image_path = self.save_latest_bev_image()
            if image_path:
                parts.append(
                    "\n[传感器信息]\n最近一帧融合 BEV 栅格时间戳: %.3f\n"
                    "输入图像为传感器融合后的上色 BEV 占据图（已保存本地）:\n%s\n"
                    "色彩语义：红色障碍=相机识别为红色目标，黑色障碍=相机识别为黑色目标，"
                    "蓝灰色障碍=仅激光占据未分类，绿色=可通行，灰色=未知，蓝色箭头=船体朝向。"
                    % (stamp, image_path)
                )
            else:
                parts.append(
                    "\n[传感器信息]\n最近一帧融合 BEV 栅格时间戳: %.3f\n"
                    "（BEV 图像保存失败，仅提供栅格信息）" % stamp
                )
        else:
            parts.append("\n[传感器信息]\n尚未收到融合栅格 explored_grid")

        vm_ctx = self.build_value_map_prompt_context()
        if vm_ctx:
            parts.append(vm_ctx)
        scale_ctx = self.build_map_scale_prompt_context()
        if scale_ctx:
            parts.append(scale_ctx)
        parts.append(self.build_candidate_prompt_context(candidates))

        return "\n".join(parts)

    def build_candidate_prompt_context(self, candidates):
        if not candidates:
            return "\n[候选区域]\n无候选。"
        lines = []
        for c in candidates:
            lines.append(
                "id=%d gx=%d gy=%d bearing_deg=%.1f distance_m=%.1f frontier=%.3f unknown=%.3f occ=%.3f red=%.3f black=%.3f geom=%.3f"
                % (
                    c["id"],
                    c["grid_x"],
                    c["grid_y"],
                    c["bearing_deg"],
                    c["distance_m"],
                    c["frontier_ratio"],
                    c["unknown_ratio"],
                    c["occ_ratio"],
                    c["red_sem"],
                    c["black_sem"],
                    c["geom_score"],
                )
            )
        return (
            "\n[候选区域]\n以下为系统生成的候选区域（id 连续且唯一），请仅对这些 id 打分：\n"
            + "\n".join(lines)
        )

    def build_value_map_prompt_context(self):
        if self.value_map is None or self.latest_odom is None:
            return ""
        info = self.value_map.info
        if info.width == 0 or info.height == 0:
            return ""

        data = np.array(self.value_map.data, dtype=np.int16).reshape((info.height, info.width))
        valid = data >= 0
        if not np.any(valid):
            return "\n[历史价值记忆]\nvalue_map 尚无有效单元。"

        nonzero = np.logical_and(valid, data > 0)
        nonzero_ratio = float(np.count_nonzero(nonzero)) / float(np.count_nonzero(valid))

        pose = self.latest_odom.pose.pose
        yaw = self.get_yaw_from_quat(pose.orientation)
        px = pose.position.x
        py = pose.position.y
        sectors = 8
        sector_max = [0.0 for _ in range(sectors)]
        radius_limit = 40.0

        ys, xs = np.where(np.logical_and(valid, data > 5))
        for y, x in zip(ys, xs):
            wx = info.origin.position.x + (x + 0.5) * info.resolution
            wy = info.origin.position.y + (y + 0.5) * info.resolution
            dx = wx - px
            dy = wy - py
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > radius_limit:
                continue
            rel = math.atan2(dy, dx) - yaw
            while rel > math.pi:
                rel -= 2.0 * math.pi
            while rel < -math.pi:
                rel += 2.0 * math.pi
            sid = int((rel + math.pi) / (2.0 * math.pi) * sectors)
            sid = max(0, min(sectors - 1, sid))
            val = float(data[y, x]) / 100.0
            if val > sector_max[sid]:
                sector_max[sid] = val

        hot = sorted(range(sectors), key=lambda i: sector_max[i], reverse=True)[:3]
        hot_desc = ", ".join(
            ["sector%d=%.2f" % (i, sector_max[i]) for i in hot if sector_max[i] > 0.0]
        )
        if not hot_desc:
            hot_desc = "无明显高值扇区"

        return (
            "\n[历史价值记忆]\n"
            "value_map 非零覆盖率=%.3f；当前较高值扇区: %s。\n"
            "请避免继续堆积在这些高值扇区，优先输出低覆盖扇区的新候选区域。"
            % (nonzero_ratio, hot_desc)
        )

    def build_map_scale_prompt_context(self):
        if self.latest_explored_grid is None or self.latest_odom is None:
            return ""
        info = self.latest_explored_grid.info
        if info.width <= 0 or info.height <= 0 or info.resolution <= 1e-6:
            return ""
        map_w_m = info.width * info.resolution
        map_h_m = info.height * info.resolution
        pose = self.latest_odom.pose.pose
        gx, gy = self.world_to_grid(pose.position.x, pose.position.y, info)
        if gx is None:
            return ""
        return (
            "\n[地图尺度与坐标]\n"
            "occupancy_grid: width=%d, height=%d, resolution=%.3f m/cell, size=%.1fm x %.1fm。\n"
            "地图原点(世界坐标)= (%.2f, %.2f)，机器人当前栅格坐标=(%d,%d)。\n"
            "保存图像像素大小为 image_width=%d, image_height=%d（原点左上，u向右,v向下）。\n"
            "命令：仅输出 CANDIDATE_SCORES。"
            % (
                info.width,
                info.height,
                info.resolution,
                map_w_m,
                map_h_m,
                info.origin.position.x,
                info.origin.position.y,
                gx,
                gy,
                info.width,
                info.height,
            )
        )

    def run_mission_cycle(self, trigger_source, red_pixel_count):
        if self.mission_running:
            rospy.logwarn_throttle(
                10.0,
                "[vla_mission_node] 上一次任务尚未完成，跳过本次触发（source=%s）。",
                trigger_source,
            )
            return

        if not self.node_ready:
            rospy.logwarn_throttle(
                30.0,
                "[vla_mission_node] 节点尚未初始化完成，跳过本次定时任务。",
            )
            return
        if self.value_map is None or self.latest_explored_grid is None:
            rospy.logwarn_throttle(
                30.0,
                "[vla_mission_node] 尚未收到 explored_grid，无法更新 value_map。",
            )
            return
        if self.latest_odom is None:
            rospy.logwarn_throttle(
                30.0,
                "[vla_mission_node] 尚未收到 odom，无法投影高价值区域到地图。",
            )
            return
        if self.latest_camera_image is None:
            rospy.logwarn_throttle(
                30.0,
                "[vla_mission_node] 尚未收到相机图像，无法融合黑/红障碍语义。",
            )
            return
        if self.latest_cloud is None:
            rospy.logwarn_throttle(
                30.0,
                "[vla_mission_node] 尚未收到点云，无法进行点云回投语义融合。",
            )
            return
        if self.sync_enable and self.last_sync_dt_s is None:
            rospy.logwarn_throttle(
                5.0,
                "[vla_mission_node] 尚未形成图像-点云同步对，跳过本次触发。",
            )
            return

        self.mission_running = True
        if trigger_source == "periodic_5s":
            rospy.logdebug_throttle(10.0, "[vla_mission_node] 周期执行（source=%s）。", trigger_source)
        else:
            rospy.loginfo("[vla_mission_node] 开始执行（source=%s）。", trigger_source)
        try:
            t_total_start = time.perf_counter()
            stage_ms = {}

            debug_image_path = self.get_debug_bev_image_path(trigger_source)
            projection_debug_path = self.get_debug_projection_image_path(trigger_source)
            t0 = time.perf_counter()
            image_path = self.save_latest_bev_image(
                save_path=debug_image_path,
                projection_debug_path=projection_debug_path,
            )
            stage_ms["bev_fuse_save"] = (time.perf_counter() - t0) * 1000.0
            if image_path:
                if trigger_source == "periodic_5s":
                    rospy.logdebug_throttle(10.0, "[vla_mission_node] BEV 快照已保存: %s", image_path)
                else:
                    rospy.loginfo("[vla_mission_node] 本次调用 BEV 快照已保存: %s", image_path)
            else:
                rospy.logwarn("[vla_mission_node] 本次调用未能生成 BEV 快照。")
            if trigger_source == "periodic_5s":
                rospy.logdebug_throttle(
                    10.0, "[vla_mission_node] 点云回投调试图: %s", projection_debug_path
                )
            else:
                rospy.loginfo(
                    "[vla_mission_node] 本次调用点云回投调试图: %s", projection_debug_path
                )

            # 周期本地更新：固定按 region unknown_ratio 更新 value_map
            if self.debug_unknown_only_mode:
                t0 = time.perf_counter()
                ema_stats = self.update_value_map_from_unknown_ratio()
                stage_ms["value_ema_update"] = (time.perf_counter() - t0) * 1000.0
                self.value_map.header.stamp = rospy.Time.now()
                self.value_map.header.frame_id = self.latest_explored_grid.header.frame_id
                t_pub = time.perf_counter()
                self.value_map_pub.publish(self.value_map)
                stage_ms["publish"] = (time.perf_counter() - t_pub) * 1000.0
            else:
                t0 = time.perf_counter()
                ema_stats = self.update_value_map_from_unknown_ratio()
                stage_ms["value_ema_update"] = (time.perf_counter() - t0) * 1000.0
                self.value_map.header.stamp = rospy.Time.now()
                self.value_map.header.frame_id = self.latest_explored_grid.header.frame_id
                t_pub = time.perf_counter()
                self.value_map_pub.publish(self.value_map)
                stage_ms["publish"] = (time.perf_counter() - t_pub) * 1000.0

            # 根据本地更新后的突变决定是否触发 VLM（并保持最小间隔）
            now = rospy.Time.now()
            dt_since_last = (
                (now - self.last_vlm_infer_time).to_sec()
                if self.last_vlm_infer_time.to_sec() > 0.0
                else 1e9
            )
            mutation_mean = float(ema_stats.get("mutation_mean_abs", 0.0))
            mutation_sum = float(ema_stats.get("mutation_sum_abs", 0.0))
            mutation_trigger = (
                mutation_mean >= self.vlm_trigger_mutation_mean_threshold
                or mutation_sum >= self.vlm_trigger_mutation_sum_threshold
            )
            interval_ok = dt_since_last >= max(30.0, float(self.vlm_min_interval_s))
            if not mutation_trigger or not interval_ok:
                rospy.loginfo_throttle(
                    2.0,
                    "[vla_mission_node] periodic local_update: mutation_mean=%.3f mutation_sum=%.1f vlm=skip(dt=%.1fs)",
                    mutation_mean,
                    mutation_sum,
                    dt_since_last,
                )
                return

            # 只有在需要 VLM 时再生成候选，避免本地周期计算与日志负担
            t0 = time.perf_counter()
            candidates = self.generate_candidates_from_bev()
            stage_ms["candidate_generate"] = (time.perf_counter() - t0) * 1000.0
            if not candidates:
                rospy.logwarn("[vla_mission_node] 候选区域为空，跳过本次 VLM 更新。")
                return
            rospy.loginfo(
                "[vla_mission_node] periodic local_update: mutation_mean=%.3f mutation_sum=%.1f vlm=trigger(dt=%.1fs)",
                mutation_mean,
                mutation_sum,
                dt_since_last,
            )

            t0 = time.perf_counter()
            prompt = self.build_prompt(candidates, image_path_override=image_path)
            stage_ms["prompt_build"] = (time.perf_counter() - t0) * 1000.0
            start_time = rospy.Time.now()

            try:
                t0 = time.perf_counter()
                result = subprocess.run(
                    self.ollama_cmd,
                    input=prompt,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.ollama_timeout,
                    check=True,
                )
                stage_ms["vlm_infer"] = (time.perf_counter() - t0) * 1000.0
            except FileNotFoundError:
                rospy.logerr(
                    "[vla_mission_node] 找不到 'ollama' 命令，请确认已安装并在 PATH 中。"
                )
                return
            except subprocess.TimeoutExpired:
                end_time = rospy.Time.now()
                elapsed = (end_time - start_time).to_sec()
                rospy.logerr(
                    "[vla_mission_node] 调用大模型超时（%.1f s）。", self.ollama_timeout
                )
                rospy.logwarn(
                    "[vla_mission_node] 本次从调用到超时耗时 %.3f 秒", elapsed
                )
                return
            except subprocess.CalledProcessError as e:
                end_time = rospy.Time.now()
                elapsed = (end_time - start_time).to_sec()
                rospy.logerr(
                    "[vla_mission_node] 调用大模型失败，返回码=%d, stderr=%s",
                    e.returncode,
                    e.stderr,
                )
                rospy.logwarn(
                    "[vla_mission_node] 本次从调用到失败耗时 %.3f 秒", elapsed
                )
                return

            end_time = rospy.Time.now()
            elapsed = (end_time - start_time).to_sec()
            output = result.stdout.strip()
            rospy.logdebug(
                "[vla_mission_node] 模型原始输出（耗时 %.3f 秒）: %s", elapsed, output
            )
            self.last_vlm_infer_time = rospy.Time.now()

            t0 = time.perf_counter()
            score_map = self.parse_candidate_scores_from_output(output)
            stage_ms["score_parse"] = (time.perf_counter() - t0) * 1000.0
            if not score_map:
                if self.candidate_require_vlm_score:
                    rospy.logwarn(
                        "[vla_mission_node] 未解析到候选分数且 require_vlm=true，本轮不会写入新候选。"
                    )
                else:
                    rospy.logwarn(
                        "[vla_mission_node] 未解析到候选分数，回退使用几何分数。"
                    )
            t0 = time.perf_counter()
            ema_stats = self.update_value_map_with_candidate_ema(
                candidates, score_map, force_fill_missing=True
            )
            stage_ms["value_ema_update"] = (time.perf_counter() - t0) * 1000.0
            self.value_map.header.stamp = rospy.Time.now()
            self.value_map.header.frame_id = self.latest_explored_grid.header.frame_id
            t_pub = time.perf_counter()
            self.value_map_pub.publish(self.value_map)
            stage_ms["publish"] = (time.perf_counter() - t_pub) * 1000.0
            rospy.loginfo(
                "[vla_mission_node] 已更新并发布 value_map，candidates=%d scored=%d",
                len(candidates),
                len(score_map),
            )
            if self.profile_time_enable:
                total_ms = (time.perf_counter() - t_total_start) * 1000.0
                rospy.loginfo(
                    "[vla_mission_node] 耗时明细(ms): bev=%.1f cand=%.1f prompt=%.1f vlm=%.1f parse=%.1f ema=%.1f pub=%.1f total=%.1f | ema=%s",
                    stage_ms.get("bev_fuse_save", 0.0),
                    stage_ms.get("candidate_generate", 0.0),
                    stage_ms.get("prompt_build", 0.0),
                    stage_ms.get("vlm_infer", 0.0),
                    stage_ms.get("score_parse", 0.0),
                    stage_ms.get("value_ema_update", 0.0),
                    stage_ms.get("publish", 0.0),
                    total_ms,
                    str(ema_stats),
                )
        finally:
            self.mission_running = False

    def get_debug_bev_image_path(self, trigger_source):
        now_sec = rospy.Time.now().to_sec()
        stamp_str = "{:.3f}".format(now_sec).replace(".", "_")
        trigger_tag = re.sub(r"[^a-zA-Z0-9_\\-]", "_", trigger_source)
        parent = os.path.dirname(self.bev_image_path) or self.image_save_dir
        return os.path.join(parent, "fused_bev_{}_{}.png".format(stamp_str, trigger_tag))

    def get_debug_projection_image_path(self, trigger_source):
        now_sec = rospy.Time.now().to_sec()
        stamp_str = "{:.3f}".format(now_sec).replace(".", "_")
        trigger_tag = re.sub(r"[^a-zA-Z0-9_\\-]", "_", trigger_source)
        parent = os.path.dirname(self.bev_image_path) or self.image_save_dir
        return os.path.join(parent, "cloud_proj_{}_{}.png".format(stamp_str, trigger_tag))

    def count_red_pixels_for_event(self, image_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(
                10.0, "[vla_mission_node] 事件触发图像转换失败: %s", str(e)
            )
            return 0, 0.0

        h, w = bgr.shape[:2]
        if h <= 2 or w <= 2:
            return 0, 0.0

        y0 = int(self.event_check_roi_top_ratio * h)
        y0 = max(0, min(h - 1, y0))
        roi = bgr[y0:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 与语义识别保持一致的红色范围
        red_mask1 = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_count = int(np.count_nonzero(red_mask))
        red_ratio = float(red_count) / float(red_mask.size)
        return red_count, red_ratio

    def save_latest_bev_image(self, save_path=None, projection_debug_path=None):
        """
        将融合后的 explored_grid 转成上色 BEV 图并保存本地，返回文件路径。
        """
        if self.latest_explored_grid is None:
            return None

        grid = self.latest_explored_grid
        width = int(grid.info.width)
        height = int(grid.info.height)
        if width <= 0 or height <= 0:
            rospy.logwarn("[vla_mission_node] explored_grid 尺寸非法，无法生成 BEV 图。")
            return None

        data = np.array(grid.data, dtype=np.int16).reshape((height, width))
        bev = np.zeros((height, width, 3), dtype=np.uint8)

        unknown_mask = data < 0
        occupied_mask = data >= 65
        free_mask = data <= 20
        mid_mask = (~unknown_mask) & (~occupied_mask) & (~free_mask)

        # BGR: 未知灰、占据蓝灰（未分类）、空闲绿、中间态黄
        bev[unknown_mask] = (110, 110, 110)
        bev[occupied_mask] = (170, 90, 50)
        bev[free_mask] = (20, 160, 20)
        bev[mid_mask] = (30, 190, 190)
        self.fuse_camera_semantics_to_grid(
            data, grid.info, projection_debug_path=projection_debug_path
        )

        if self.semantic_black_score is not None and self.semantic_red_score is not None:
            red_obstacle = (
                occupied_mask
                & (self.semantic_red_score > self.semantic_black_score)
                & ((self.semantic_red_score - self.semantic_black_score) >= self.semantic_score_margin)
                & (self.semantic_red_score >= self.semantic_threshold)
            )
            black_obstacle = (
                occupied_mask
                & (self.semantic_black_score > self.semantic_red_score)
                & ((self.semantic_black_score - self.semantic_red_score) >= self.semantic_score_margin)
                & (self.semantic_black_score >= self.semantic_threshold)
            )
            bev[red_obstacle] = (30, 30, 230)   # 红色障碍
            bev[black_obstacle] = (20, 20, 20)  # 黑色障碍
            rospy.loginfo_throttle(
                2.0,
                "[vla_mission_node] 语义融合统计: red_cells=%d black_cells=%d unknown_class_cells=%d",
                int(np.count_nonzero(red_obstacle)),
                int(np.count_nonzero(black_obstacle)),
                int(np.count_nonzero(occupied_mask & (~red_obstacle) & (~black_obstacle))),
            )

        # 叠加船位和朝向（若 odom 可用）
        if self.latest_odom is not None:
            pose = self.latest_odom.pose.pose
            gx, gy = self.world_to_grid(pose.position.x, pose.position.y, grid.info)
            if gx is not None:
                img_x = gx
                img_y = gy
                yaw = self.get_yaw_from_quat(pose.orientation)
                arrow_len = max(8, int(2.0 / grid.info.resolution))
                end_x = int(round(img_x + arrow_len * math.cos(yaw)))
                end_y = int(round(img_y + arrow_len * math.sin(yaw)))
                cv2.circle(bev, (img_x, img_y), 3, (255, 255, 0), -1)
                cv2.arrowedLine(bev, (img_x, img_y), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)

        # 转成常见视觉坐标（原点在左上）
        # np.flipud 返回负步长视图，OpenCV 绘制接口要求连续内存
        bev_vis = np.ascontiguousarray(np.flipud(bev))
        if self.bev_overlay_grid_enable:
            step = max(10, int(self.bev_overlay_grid_step_cells))
            h_vis, w_vis = bev_vis.shape[:2]
            for x in range(0, w_vis, step):
                cv2.line(bev_vis, (x, 0), (x, h_vis - 1), (70, 70, 70), 1)
            for y in range(0, h_vis, step):
                cv2.line(bev_vis, (0, y), (w_vis - 1, y), (70, 70, 70), 1)
            cv2.putText(
                bev_vis,
                "u-> right, v-> down",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        target_path = save_path if save_path else self.bev_image_path
        bev_parent = os.path.dirname(target_path)
        if bev_parent and not os.path.exists(bev_parent):
            try:
                os.makedirs(bev_parent)
            except OSError as e:
                rospy.logwarn("[vla_mission_node] 创建 BEV 保存目录失败: %s", str(e))
                return None
        ok = cv2.imwrite(target_path, bev_vis)
        if not ok:
            rospy.logwarn("[vla_mission_node] 保存 BEV 图失败: %s", target_path)
            return None
        # 始终刷新 latest 便于实时查看，同时保留每次调用快照。
        if target_path != self.bev_image_path:
            cv2.imwrite(self.bev_image_path, bev_vis)
        return target_path

    def fuse_camera_semantics_to_grid(self, occ_data, grid_info, projection_debug_path=None):
        if self.latest_camera_image is None or self.latest_odom is None or self.latest_cloud is None:
            return
        if self.semantic_black_score is None or self.semantic_red_score is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(self.latest_camera_image, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(10.0, "[vla_mission_node] 相机图像转换失败: %s", str(e))
            return

        h, w = bgr.shape[:2]
        if h <= 2 or w <= 2:
            return

        # 历史语义衰减，降低瞬时误检影响
        self.semantic_black_score *= self.semantic_decay
        self.semantic_red_score *= self.semantic_decay

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, (0, 100, 60), (12, 255, 255))
        red_mask2 = cv2.inRange(hsv, (168, 100, 60), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 130, 85))

        pose = self.latest_odom.pose.pose
        yaw = self.get_yaw_from_quat(pose.orientation)
        px = pose.position.x
        py = pose.position.y
        hfov = math.radians(max(1e-3, self.camera_hfov_deg))
        vfov = math.radians(max(1e-3, self.camera_vfov_deg))
        cloud_frame = self.latest_cloud.header.frame_id
        image_cam_frame = self.latest_camera_image.header.frame_id
        cam_frame = self.camera_frame if self.camera_frame else image_cam_frame
        is_optical = self.camera_frame_is_optical or ("optical" in cam_frame.lower())

        use_tf = False
        rot_cam_from_cloud = None
        trans_cam_from_cloud = None
        rot_base_from_cloud = None
        trans_base_from_cloud = None
        if self.use_tf_extrinsics and cloud_frame and cam_frame:
            cam_candidates = [cam_frame]
            if image_cam_frame and image_cam_frame != cam_frame:
                cam_candidates.append(image_cam_frame)

            for cam_candidate in cam_candidates:
                rot_try, trans_try = self.lookup_tf_rt(
                    cam_candidate, cloud_frame, self.latest_cloud.header.stamp
                )
                if rot_try is None or trans_try is None:
                    continue
                cam_frame = cam_candidate
                rot_cam_from_cloud = rot_try
                trans_cam_from_cloud = trans_try
                break

            rot_base_from_cloud, trans_base_from_cloud = self.lookup_tf_rt(
                self.base_link_frame, cloud_frame, self.latest_cloud.header.stamp
            )
            use_tf = (
                rot_cam_from_cloud is not None
                and rot_base_from_cloud is not None
                and trans_cam_from_cloud is not None
                and trans_base_from_cloud is not None
            )
            if (
                use_tf
                and self.camera_frame
                and image_cam_frame
                and cam_frame == image_cam_frame
                and self.camera_frame != image_cam_frame
            ):
                rospy.logwarn_throttle(
                    2.0,
                    "[vla_mission_node] 配置 camera_frame=%s 不可用，已回退为图像帧=%s",
                    self.camera_frame,
                    image_cam_frame,
                )
        if not use_tf:
            rospy.logwarn_throttle(
                2.0,
                "[vla_mission_node] 使用手工外参回退（建议检查 TF: %s -> %s）",
                cloud_frame,
                cam_frame,
            )
        if self.projection_mode == "auto":
            if use_tf and is_optical:
                mode_candidates = ["optical", "optical_mirror", "body", "body_mirror"]
            elif use_tf:
                mode_candidates = ["body", "body_mirror", "optical", "optical_mirror"]
            else:
                mode_candidates = ["body", "body_mirror", "optical", "optical_mirror"]
        else:
            mode_candidates = [self.projection_mode]
        mode_hits = {m: 0 for m in mode_candidates}
        rospy.loginfo_throttle(
            5.0,
            "[vla_mission_node] 回投配置: cloud_frame=%s camera_frame=%s tf=%s modes=%s",
            cloud_frame,
            cam_frame,
            str(use_tf),
            ",".join(mode_candidates),
        )

        vis = bgr.copy()
        total_points = int(self.latest_cloud.width * self.latest_cloud.height)
        if total_points <= 0:
            total_points = 1
        step = max(1, int(float(total_points) / float(max(1, int(self.cloud_proj_max_points)))))

        paint_r = max(grid_info.resolution, self.semantic_paint_radius_m)
        paint_cells = int(math.ceil(paint_r / grid_info.resolution))

        projected_points = 0
        classified_points = 0
        fused_points = 0
        fused_red = 0
        fused_black = 0

        for idx, pt in enumerate(
            pc2.read_points(self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True)
        ):
            if idx % step != 0:
                continue
            x_l, y_l, z_l = float(pt[0]), float(pt[1]), float(pt[2])
            p_cloud = np.array([x_l, y_l, z_l], dtype=np.float64)
            if use_tf:
                p_cam = np.dot(rot_cam_from_cloud, p_cloud) + trans_cam_from_cloud
                p_base = np.dot(rot_base_from_cloud, p_cloud) + trans_base_from_cloud
                x_c, y_c, z_c = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
                x_b, y_b, z_b = float(p_base[0]), float(p_base[1]), float(p_base[2])
            else:
                # lidar -> base
                x_b = x_l + self.lidar_mount_x
                y_b = y_l + self.lidar_mount_y
                z_b = z_l + self.lidar_mount_z
                # base -> camera_link
                rel = np.array(
                    [
                        x_b - self.camera_mount_x,
                        y_b - self.camera_mount_y,
                        z_b - self.camera_mount_z,
                    ],
                    dtype=np.float64,
                )
                p_cam = np.dot(self.R_base_to_cam.T, rel)
                x_c, y_c, z_c = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

            if z_b < self.semantic_base_z_min:
                continue

            u = None
            v = None
            used_mode = None
            for mode in mode_candidates:
                uu, vv, ok = self.compute_uv_with_mode(
                    x_c, y_c, z_c, mode, hfov, vfov, w, h
                )
                if ok:
                    u, v = uu, vv
                    used_mode = mode
                    break
            if u is None:
                continue
            projected_points += 1
            mode_hits[used_mode] = mode_hits.get(used_mode, 0) + 1

            is_red = red_mask[v, u] > 0
            is_black = black_mask[v, u] > 0
            if not is_red and not is_black:
                cv2.circle(vis, (u, v), 1, (160, 160, 160), -1)
                continue

            cls = "red" if is_red and not is_black else "black"
            if is_red and is_black:
                # 冲突像素优先按更暗区域判黑，降低高亮反射误判为红
                cls = "black" if hsv[v, u, 2] < 70 else "red"
            classified_points += 1

            strength = max(self.semantic_strength_min, min(1.0, self.semantic_strength_boost * 0.35))
            wx = px + x_b * math.cos(yaw) - y_b * math.sin(yaw)
            wy = py + x_b * math.sin(yaw) + y_b * math.cos(yaw)
            gx, gy = self.world_to_grid(wx, wy, grid_info)
            if gx is None:
                continue

            # 若直接落点不是障碍，尝试 1-cell 邻域吸附到最近障碍。
            hit_gx, hit_gy = gx, gy
            if occ_data[gy, gx] < 65:
                found = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = gx + dx, gy + dy
                        if nx < 0 or ny < 0 or ny >= occ_data.shape[0] or nx >= occ_data.shape[1]:
                            continue
                        if occ_data[ny, nx] >= 65:
                            hit_gx, hit_gy = nx, ny
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue

            for dx in range(-paint_cells, paint_cells + 1):
                for dy in range(-paint_cells, paint_cells + 1):
                    nx, ny = hit_gx + dx, hit_gy + dy
                    if nx < 0 or ny < 0 or ny >= occ_data.shape[0] or nx >= occ_data.shape[1]:
                        continue
                    if occ_data[ny, nx] < 65:
                        continue
                    dist = math.sqrt(dx * dx + dy * dy) * grid_info.resolution
                    if dist > paint_r:
                        continue
                    gain = max(0.0, 1.0 - dist / paint_r) * strength
                    if cls == "red":
                        self.semantic_red_score[ny, nx] += gain
                    else:
                        self.semantic_black_score[ny, nx] += gain

            fused_points += 1
            if cls == "red":
                fused_red += 1
                cv2.circle(vis, (u, v), 2, (0, 0, 255), -1)
            else:
                fused_black += 1
                cv2.circle(vis, (u, v), 2, (0, 0, 0), -1)

        rospy.loginfo_throttle(
            2.0,
            "[vla_mission_node] 点云回投统计: projected=%d classified=%d fused=%d(red=%d,black=%d) mode_hits=%s",
            projected_points,
            classified_points,
            fused_points,
            fused_red,
            fused_black,
            str(mode_hits),
        )

        if projection_debug_path:
            parent = os.path.dirname(projection_debug_path)
            if parent and not os.path.exists(parent):
                try:
                    os.makedirs(parent)
                except OSError:
                    pass
            cv2.imwrite(projection_debug_path, vis)

    def same_grid_info(self, grid_a, grid_b):
        return (
            grid_a.info.width == grid_b.info.width
            and grid_a.info.height == grid_b.info.height
            and abs(grid_a.info.resolution - grid_b.info.resolution) < 1e-9
            and abs(grid_a.info.origin.position.x - grid_b.info.origin.position.x) < 1e-9
            and abs(grid_a.info.origin.position.y - grid_b.info.origin.position.y) < 1e-9
        )

    def generate_candidates_from_bev(self):
        if self.latest_explored_grid is None or self.latest_odom is None:
            return []
        info = self.latest_explored_grid.info
        if info.width <= 0 or info.height <= 0 or info.resolution <= 1e-6:
            return []

        data = np.array(self.latest_explored_grid.data, dtype=np.int16).reshape(
            (info.height, info.width)
        )
        free = data <= 20
        occ = data >= 65
        unknown = data < 0
        free_u8 = free.astype(np.uint8)
        neigh_free = cv2.dilate(free_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
        frontier = np.logical_and(unknown, neigh_free)

        pose = self.latest_odom.pose.pose
        boat_gx, boat_gy = self.world_to_grid(pose.position.x, pose.position.y, info)
        if boat_gx is None:
            return []
        yaw = self.get_yaw_from_quat(pose.orientation)
        sectors = max(4, int(self.candidate_sector_count))
        border_margin_cell = max(0, int(round(self.candidate_border_margin_m / info.resolution)))
        rows = max(1, int(self.region_grid_rows))
        cols = max(1, int(self.region_grid_cols))
        step_y = max(1, int(math.ceil(float(info.height) / float(rows))))
        step_x = max(1, int(math.ceil(float(info.width) / float(cols))))
        candidates = []
        cid = 0
        map_diag_m = math.sqrt((info.width * info.resolution) ** 2 + (info.height * info.resolution) ** 2)

        for ry in range(rows):
            y0 = ry * step_y
            y1 = min(info.height, (ry + 1) * step_y)
            if y1 <= y0:
                continue
            for rx in range(cols):
                x0 = rx * step_x
                x1 = min(info.width, (rx + 1) * step_x)
                if x1 <= x0:
                    continue
                gx = int((x0 + x1 - 1) // 2)
                gy = int((y0 + y1 - 1) // 2)
                if (
                    gx < border_margin_cell
                    or gy < border_margin_cell
                    or gx >= info.width - border_margin_cell
                    or gy >= info.height - border_margin_cell
                ):
                    continue

                area = float((x1 - x0) * (y1 - y0))
                occ_ratio = float(np.count_nonzero(occ[y0:y1, x0:x1])) / area
                free_ratio = float(np.count_nonzero(free[y0:y1, x0:x1])) / area
                unknown_ratio = float(np.count_nonzero(unknown[y0:y1, x0:x1])) / area
                frontier_ratio = float(np.count_nonzero(frontier[y0:y1, x0:x1])) / area
                if frontier_ratio < self.candidate_min_frontier_ratio and unknown_ratio < 0.10:
                    continue
                if occ_ratio > 0.60:
                    continue

                red_sem = 0.0
                black_sem = 0.0
                if self.semantic_red_score is not None:
                    red_sem = float(np.mean(self.semantic_red_score[y0:y1, x0:x1]))
                if self.semantic_black_score is not None:
                    black_sem = float(np.mean(self.semantic_black_score[y0:y1, x0:x1]))

                dx = float(gx - boat_gx)
                dy = float(gy - boat_gy)
                dist_m = math.sqrt(dx * dx + dy * dy) * info.resolution
                dist_norm = max(0.0, min(1.0, dist_m / max(1.0, map_diag_m * 0.7)))
                ang = math.atan2(dy, dx) - yaw
                while ang > math.pi:
                    ang -= 2.0 * math.pi
                while ang < -math.pi:
                    ang += 2.0 * math.pi

                geom_raw = (
                    0.35 * frontier_ratio
                    + 0.35 * unknown_ratio
                    + 0.12 * red_sem
                    - 0.20 * occ_ratio
                    + 0.05 * free_ratio
                    + 0.08 * dist_norm
                )
                geom_score = max(0.0, min(1.0, geom_raw * 2.0))
                candidates.append(
                    {
                        "id": cid,
                        "grid_x": int(gx),
                        "grid_y": int(gy),
                        "bearing_deg": math.degrees(ang),
                        "distance_m": float(dist_m),
                        "frontier_ratio": float(frontier_ratio),
                        "unknown_ratio": float(unknown_ratio),
                        "occ_ratio": float(occ_ratio),
                        "red_sem": float(red_sem),
                        "black_sem": float(black_sem),
                        "geom_score": float(geom_score),
                    }
                )
                cid += 1

        candidates.sort(key=lambda c: c["geom_score"], reverse=True)
        keep_n = max(4, int(self.candidate_count))
        min_spacing_cell = max(1.0, float(self.candidate_min_spacing_m) / info.resolution)
        # 扇区轮询挑选，避免候选集中在单一局部
        sector_buckets = [[] for _ in range(sectors)]
        for c in candidates:
            ang = math.radians(c["bearing_deg"])
            sid = int((ang + math.pi) / (2.0 * math.pi) * sectors)
            sid = max(0, min(sectors - 1, sid))
            sector_buckets[sid].append(c)
        for b in sector_buckets:
            b.sort(key=lambda x: x["geom_score"], reverse=True)

        picked = []
        while len(picked) < keep_n:
            progressed = False
            for sid in range(sectors):
                if len(sector_buckets[sid]) == 0:
                    continue
                c = sector_buckets[sid].pop(0)
                too_close = False
                for p in picked:
                    ddx = float(c["grid_x"] - p["grid_x"])
                    ddy = float(c["grid_y"] - p["grid_y"])
                    if math.sqrt(ddx * ddx + ddy * ddy) < min_spacing_cell:
                        too_close = True
                        break
                if too_close:
                    continue
                picked.append(c)
                progressed = True
                if len(picked) >= keep_n:
                    break
            if not progressed:
                break
        for i, c in enumerate(picked):
            c["id"] = i
        rospy.loginfo(
            "[vla_mission_node] 候选生成完成: total=%d kept=%d",
            len(candidates),
            len(picked),
        )
        return picked

    def parse_candidate_scores_from_output(self, text):
        out = {}
        # 先尝试标准 JSON 数组
        patterns = [
            r"CANDIDATE_SCORES\s*:\s*(\[[\s\S]*?\])",
            r"CAND_MILESTONES\s*:\s*(\[[\s\S]*?\])",
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if not match:
                continue
            arr_text = match.group(1).strip()
            try:
                parsed = json.loads(arr_text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    cid = self.safe_float(item.get("id"), None)
                    score = self.safe_float(item.get("score"), None)
                    if cid is None or score is None:
                        continue
                    out[int(round(cid))] = max(0.0, min(1.0, float(score)))
                if out:
                    return out

        # 兼容非标准格式：CAND_MILESTONES: {"id":2,"score":0.9},{"id":3,"score":0.8}
        block_match = re.search(r"CAND_(?:IDATE_SCORES|MILESTONES)\s*:\s*(.*)", text, re.IGNORECASE)
        block = block_match.group(1) if block_match else text
        obj_pat = re.compile(
            r'\{\s*"id"\s*:\s*(-?\d+)\s*,\s*"score"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}'
        )
        for m in obj_pat.finditer(block):
            cid = int(m.group(1))
            score = max(0.0, min(1.0, float(m.group(2))))
            out[cid] = score
        if not out:
            rospy.logwarn("[vla_mission_node] 未解析到 CANDIDATE_SCORES/CAND_MILESTONES。")
        elif len(out) < max(2, int(0.5 * max(1, int(self.candidate_count)))):
            rospy.logwarn(
                "[vla_mission_node] 候选分数数量偏少: parsed=%d expected~%d，value_map 可能只更新少量区域。",
                len(out),
                int(self.candidate_count),
            )
        return out

    def parse_regions_from_output(self, text):
        """
        期望格式:
        HIGH_VALUE_REGIONS: [{"bearing_deg":15,"distance_m":8.0,"score":0.9}, ...]
        """
        regions = self.parse_regions_as_json(text)
        if regions:
            return regions
        return self.parse_regions_fallback(text)

    def parse_regions_as_json(self, text):
        pattern = re.compile(r"HIGH_VALUE_REGIONS\s*:\s*(\[[\s\S]*?\])", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return []
        array_text = match.group(1).strip()

        try:
            parsed = json.loads(array_text)
        except ValueError as e:
            rospy.logwarn("[vla_mission_node] JSON 解析 HIGH_VALUE_REGIONS 失败: %s", str(e))
            return []

        if not isinstance(parsed, list):
            return []

        regions = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            bearing = self.safe_float(item.get("bearing_deg"), None)
            distance = self.safe_float(item.get("distance_m"), None)
            score = self.safe_float(item.get("score"), None)
            if bearing is None or distance is None or score is None:
                continue
            regions.append(
                {
                    "bearing_deg": bearing,
                    "distance_m": distance,
                    "score": score,
                }
            )
        return regions

    def parse_regions_fallback(self, text):
        regions = []
        line_pattern = re.compile(
            r"HIGH_VALUE_REGION\s*:\s*"
            r"bearing\s*=\s*(-?\d+\.?\d*)\s*,\s*"
            r"distance\s*=\s*(\d+\.?\d*)\s*,\s*"
            r"score\s*=\s*(\d+\.?\d*)",
            re.IGNORECASE,
        )
        for match in line_pattern.finditer(text):
            regions.append(
                {
                    "bearing_deg": float(match.group(1)),
                    "distance_m": float(match.group(2)),
                    "score": float(match.group(3)),
                }
            )
        return regions

    def parse_cells_from_output(self, text):
        pattern = re.compile(r"HIGH_VALUE_CELLS\s*:\s*(\[[\s\S]*?\])", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return []
        array_text = match.group(1).strip()
        try:
            parsed = json.loads(array_text)
        except ValueError as e:
            rospy.logwarn("[vla_mission_node] JSON 解析 HIGH_VALUE_CELLS 失败: %s", str(e))
            return []
        if not isinstance(parsed, list):
            return []

        cells = []
        max_w = int(self.value_map.info.width) if self.value_map is not None else 0
        max_h = int(self.value_map.info.height) if self.value_map is not None else 0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            gx = self.safe_float(item.get("grid_x"), None)
            gy = self.safe_float(item.get("grid_y"), None)
            score = self.safe_float(item.get("score"), None)
            if gx is None or gy is None or score is None:
                continue
            gx_i = int(round(gx))
            gy_i = int(round(gy))
            if max_w > 0 and max_h > 0:
                if gx_i < 0 or gy_i < 0 or gx_i >= max_w or gy_i >= max_h:
                    continue
            cells.append({"grid_x": gx_i, "grid_y": gy_i, "score": float(score)})
        return cells

    def parse_pixels_norm_from_output(self, text):
        pattern = re.compile(r"HIGH_VALUE_PIXELS_NORM\s*:\s*(\[[\s\S]*?\])", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return []
        array_text = match.group(1).strip()
        try:
            parsed = json.loads(array_text)
        except ValueError as e:
            rospy.logwarn(
                "[vla_mission_node] JSON 解析 HIGH_VALUE_PIXELS_NORM 失败: %s", str(e)
            )
            return []
        if not isinstance(parsed, list):
            return []
        pixels = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            u_norm = self.safe_float(item.get("u_norm"), None)
            v_norm = self.safe_float(item.get("v_norm"), None)
            score = self.safe_float(item.get("score"), None)
            if u_norm is None or v_norm is None or score is None:
                continue
            pixels.append(
                {
                    "u_norm": max(0.0, min(1.0, u_norm)),
                    "v_norm": max(0.0, min(1.0, v_norm)),
                    "score": max(0.0, min(1.0, score)),
                }
            )
        return pixels

    def safe_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def parse_float_list_param(self, value, default):
        if isinstance(value, (list, tuple)):
            out = []
            for v in value:
                fv = self.safe_float(v, None)
                if fv is not None:
                    out.append(float(fv))
            return out if out else list(default)
        if isinstance(value, str):
            parts = [x.strip() for x in value.split(",")]
            out = []
            for p in parts:
                if not p:
                    continue
                fv = self.safe_float(p, None)
                if fv is not None:
                    out.append(float(fv))
            return out if out else list(default)
        return list(default)

    def get_yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def rpy_to_rot(self, roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return np.dot(rot_z, np.dot(rot_y, rot_x))

    def quat_to_rot(self, qx, qy, qz, qw):
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def lookup_tf_rt(self, target_frame, source_frame, stamp):
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                rospy.Duration(self.tf_lookup_timeout_s),
            )
            t = np.array(
                [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ],
                dtype=np.float64,
            )
            q = trans.transform.rotation
            r = self.quat_to_rot(q.x, q.y, q.z, q.w)
            return r, t
        except Exception as e_first:
            # 先按消息时间戳查；失败后再退化为最新可用 TF，提高鲁棒性
            try:
                trans = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(self.tf_lookup_timeout_s),
                )
                t = np.array(
                    [
                        trans.transform.translation.x,
                        trans.transform.translation.y,
                        trans.transform.translation.z,
                    ],
                    dtype=np.float64,
                )
                q = trans.transform.rotation
                r = self.quat_to_rot(q.x, q.y, q.z, q.w)
                rospy.logwarn_throttle(
                    2.0,
                    "[vla_mission_node] TF 用消息时间查询失败，已回退最新TF %s <- %s: %s",
                    target_frame,
                    source_frame,
                    str(e_first),
                )
                return r, t
            except Exception as e_second:
                rospy.logwarn_throttle(
                    2.0,
                    "[vla_mission_node] TF 查询失败 %s <- %s: %s | fallback: %s",
                    target_frame,
                    source_frame,
                    str(e_first),
                    str(e_second),
                )
                return None, None

    def compute_uv_with_mode(self, x_c, y_c, z_c, mode, hfov, vfov, w, h):
        if mode == "optical":
            forward = z_c
            right = x_c
            up = -y_c
        elif mode == "optical_mirror":
            forward = z_c
            right = -x_c
            up = -y_c
        elif mode == "body":
            forward = x_c
            right = y_c
            up = z_c
        elif mode == "body_mirror":
            forward = x_c
            right = -y_c
            up = z_c
        else:
            return None, None, False

        planar = math.sqrt(forward * forward + right * right)
        if planar < self.semantic_min_range_m or planar > self.semantic_max_range_m:
            return None, None, False
        if forward <= 1e-3:
            return None, None, False

        bearing = math.atan2(right, forward)
        elev = math.atan2(up, max(1e-6, planar))
        if abs(bearing) > 0.5 * hfov or abs(elev) > 0.5 * vfov:
            return None, None, False

        u = int(round((bearing / hfov + 0.5) * (w - 1)))
        v = int(round((0.5 - elev / vfov) * (h - 1)))
        if self.projection_flip_horizontal:
            u = (w - 1) - u
        if u < 0 or u >= w or v < 0 or v >= h:
            return None, None, False
        return u, v, True

    def world_to_grid(self, x, y, info):
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
            return None, None
        return gx, gy

    def set_cell(self, x, y, value):
        width = self.value_map.info.width
        idx = y * width + x
        if idx < 0 or idx >= len(self.value_map.data):
            return
        old_v = self.value_map.data[idx]
        if old_v < 0:
            old_v = 0
        self.value_map.data[idx] = max(0, min(100, int(max(old_v, value))))

    def paint_value_disk(self, center_gx, center_gy, paint_radius, peak_value):
        info = self.value_map.info
        width = info.width
        height = info.height
        edge_gain = max(0.0, min(0.95, float(self.value_disk_edge_gain)))
        radius_cell = int(math.ceil(paint_radius / info.resolution))
        for dx in range(-radius_cell, radius_cell + 1):
            for dy in range(-radius_cell, radius_cell + 1):
                gx = center_gx + dx
                gy = center_gy + dy
                if gx < 0 or gy < 0 or gx >= width or gy >= height:
                    continue
                dist_cell = math.sqrt(dx * dx + dy * dy)
                dist_m = dist_cell * info.resolution
                if dist_m > paint_radius:
                    continue
                # 让圆斑边缘保留一部分能量，避免只剩稀疏尖点
                gain = edge_gain + (1.0 - edge_gain) * (1.0 - dist_m / paint_radius)
                self.set_cell(gx, gy, int(peak_value * gain))

    def diffuse_value_map(self):
        if not self.value_diffuse_enable or self.value_map is None:
            return
        info = self.value_map.info
        if info.width <= 0 or info.height <= 0:
            return
        arr = np.array(self.value_map.data, dtype=np.float32).reshape((info.height, info.width))
        known_mask = arr >= 0.0
        if not np.any(known_mask):
            return

        pos = np.where(known_mask, np.maximum(arr, 0.0), 0.0)
        kernel = int(self.value_diffuse_kernel)
        if kernel < 3:
            kernel = 3
        if kernel % 2 == 0:
            kernel += 1
        sigma = max(0.01, float(self.value_diffuse_sigma))
        mix = max(0.0, min(1.0, float(self.value_diffuse_mix)))
        blurred = cv2.GaussianBlur(pos, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

        # 与原值混合，并保持峰值不被冲淡
        merged = (1.0 - mix) * pos + mix * blurred
        merged = np.maximum(pos, merged)
        merged = np.where(merged >= float(self.value_diffuse_min_value), merged, 0.0)
        merged = np.clip(merged, 0.0, 100.0)
        self.value_map.data = np.where(known_mask, merged, -1.0).astype(np.int16).flatten().tolist()

    def apply_semantic_memory_to_value_map(self):
        if (
            self.semantic_red_score is None
            or self.semantic_black_score is None
            or self.latest_explored_grid is None
        ):
            return
        occ = np.array(self.latest_explored_grid.data, dtype=np.int16).reshape(
            (self.value_map.info.height, self.value_map.info.width)
        )
        sem = np.maximum(self.semantic_red_score, self.semantic_black_score)
        mask = (occ >= 65) & (sem >= self.value_semantic_min_score)
        ys, xs = np.where(mask)
        if ys.size == 0:
            return
        for y, x in zip(ys, xs):
            v = int(max(0.0, min(100.0, sem[y, x] * self.value_semantic_boost)))
            self.set_cell(int(x), int(y), v)

    def paint_disk_on_array(self, arr, center_gx, center_gy, paint_radius_m, peak_value):
        info = self.value_map.info
        h, w = arr.shape
        radius_cell = int(math.ceil(max(info.resolution, paint_radius_m) / info.resolution))
        for dx in range(-radius_cell, radius_cell + 1):
            for dy in range(-radius_cell, radius_cell + 1):
                gx = center_gx + dx
                gy = center_gy + dy
                if gx < 0 or gy < 0 or gx >= w or gy >= h:
                    continue
                dist = math.sqrt(dx * dx + dy * dy) * info.resolution
                if dist > paint_radius_m:
                    continue
                gain = max(0.0, 1.0 - dist / max(info.resolution, paint_radius_m))
                arr[gy, gx] = max(arr[gy, gx], float(peak_value) * gain)

    def update_value_map_from_unknown_ratio(self):
        if self.value_map is None or self.latest_explored_grid is None:
            return {
                "painted": 0,
                "mode": "unknown_ratio",
                "mutation_mean_abs": 0.0,
                "mutation_sum_abs": 0.0,
            }
        info = self.value_map.info
        if info.width <= 0 or info.height <= 0 or info.resolution <= 1e-6:
            return {
                "painted": 0,
                "mode": "unknown_ratio",
                "mutation_mean_abs": 0.0,
                "mutation_sum_abs": 0.0,
            }

        occ = np.array(self.latest_explored_grid.data, dtype=np.int16).reshape((info.height, info.width))
        unknown = (occ < 0).astype(np.float32)
        rows = max(1, int(self.region_grid_rows))
        cols = max(1, int(self.region_grid_cols))
        step_y = max(1, int(math.ceil(float(info.height) / float(rows))))
        step_x = max(1, int(math.ceil(float(info.width) / float(cols))))
        target = np.zeros((info.height, info.width), dtype=np.float32)
        region_count = 0

        for ry in range(rows):
            y0 = ry * step_y
            y1 = min(info.height, (ry + 1) * step_y)
            if y1 <= y0:
                continue
            for rx in range(cols):
                x0 = rx * step_x
                x1 = min(info.width, (rx + 1) * step_x)
                if x1 <= x0:
                    continue
                region_count += 1
                region_unknown_ratio = float(np.mean(unknown[y0:y1, x0:x1]))
                target[y0:y1, x0:x1] = max(0.0, min(100.0, 100.0 * region_unknown_ratio))

        old_map = np.array(self.value_map.data, dtype=np.float32).reshape((info.height, info.width))
        known_mask = old_map >= 0.0
        old_pos = np.where(known_mask, np.maximum(old_map, 0.0), 0.0)
        if self.debug_unknown_use_ema:
            alpha = max(0.02, min(0.95, float(self.value_ema_alpha)))
            merged = (1.0 - alpha) * old_pos + alpha * target
        else:
            merged = target
        merged = np.clip(merged, 0.0, 100.0)
        mutation = np.abs(merged - old_pos)
        mutation_mean_abs = float(np.mean(mutation)) if mutation.size > 0 else 0.0
        mutation_sum_abs = float(np.sum(mutation))
        self.value_map.data = np.where(known_mask, merged, -1.0).astype(np.int16).flatten().tolist()
        # 本地region打分调试路径不再额外扩散，保持“每个region按unknown_ratio赋值”语义

        painted = int(np.count_nonzero(merged > 0.5))
        return {
            "painted": painted,
            "regions": int(region_count),
            "grid_rows": int(rows),
            "grid_cols": int(cols),
            "mode": "unknown_ratio_10x10",            "mutation_mean_abs": round(float(mutation_mean_abs), 4),
            "mutation_sum_abs": round(float(mutation_sum_abs), 2),
        }

    def update_value_map_with_candidate_ema(self, candidates, score_map, force_fill_missing=False):
        if self.value_map is None or self.latest_explored_grid is None:
            return {
                "painted": 0,
                "skipped_no_score": 0,
                "unmatched_score_ids": 0,
                "mutation_mean_abs": 0.0,
                "mutation_sum_abs": 0.0,
            }
        info = self.value_map.info
        if info.width <= 0 or info.height <= 0:
            return {
                "painted": 0,
                "skipped_no_score": 0,
                "unmatched_score_ids": 0,
                "mutation_mean_abs": 0.0,
                "mutation_sum_abs": 0.0,
            }
        occ = np.array(self.latest_explored_grid.data, dtype=np.int16).reshape((info.height, info.width))
        old_map = np.array(self.value_map.data, dtype=np.float32).reshape((info.height, info.width))
        known_mask = old_map >= 0.0
        target = np.zeros((info.height, info.width), dtype=np.float32)
        geom_mix = max(0.0, min(1.0, float(self.candidate_geom_mix)))
        require_vlm = bool(self.candidate_require_vlm_score)
        min_scored_ratio = max(0.0, min(1.0, float(self.candidate_min_scored_ratio)))
        fill_scale = max(0.0, min(1.0, float(self.candidate_missing_fill_scale)))
        painted = 0
        skipped_no_score = 0

        candidate_id_set = set([int(c["id"]) for c in candidates])
        unmatched_score_ids = [cid for cid in score_map.keys() if int(cid) not in candidate_id_set]
        matched_scored = [cid for cid in score_map.keys() if int(cid) in candidate_id_set]
        scored_ratio = float(len(matched_scored)) / float(max(1, len(candidates)))
        use_missing_fill = force_fill_missing or (require_vlm and scored_ratio < min_scored_ratio)
        if use_missing_fill:
            rospy.logwarn(
                "[vla_mission_node] 候选评分覆盖率过低(%.2f < %.2f)，缺失候选将用几何分补齐(scale=%.2f)。",
                scored_ratio,
                min_scored_ratio,
                fill_scale,
            )
        if len(unmatched_score_ids) > 0:
            rospy.logwarn(
                "[vla_mission_node] score 中存在未匹配候选 id: %s",
                str(sorted(unmatched_score_ids)),
            )

        for c in candidates:
            base = max(0.0, min(1.0, c["geom_score"]))
            vlm = score_map.get(c["id"], None)
            if vlm is None and require_vlm:
                if use_missing_fill:
                    final_score = fill_scale * base
                else:
                    skipped_no_score += 1
                    continue
            if vlm is None:
                final_score = base if not use_missing_fill else fill_scale * base
            else:
                final_score = geom_mix * base + (1.0 - geom_mix) * vlm
            if final_score <= 0.01:
                continue
            gx = int(c["grid_x"])
            gy = int(c["grid_y"])
            if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
                continue
            if self.candidate_debug_log_enable:
                wx = info.origin.position.x + (float(gx) + 0.5) * info.resolution
                wy = info.origin.position.y + (float(gy) + 0.5) * info.resolution
                # 与保存的 BEV 图一致：图像坐标原点左上，因此 v 需要翻转
                bev_u = gx
                bev_v = (info.height - 1) - gy
                rospy.loginfo(
                    "[vla_mission_node] CAND_MAP id=%d -> grid=(%d,%d) world=(%.2f,%.2f) bev_uv=(%d,%d) geom=%.3f vlm=%s final=%.3f",
                    c["id"],
                    gx,
                    gy,
                    wx,
                    wy,
                    bev_u,
                    bev_v,
                    base,
                    "None" if vlm is None else "%.3f" % vlm,
                    final_score,
                )
            peak = 100.0 * final_score
            self.paint_disk_on_array(
                target,
                gx,
                gy,
                max(info.resolution, self.candidate_paint_radius_m),
                peak,
            )
            painted += 1

        # 语义记忆注入 target，保留红/黑障碍长期价值
        if (
            self.candidate_semantic_inject_enable
            and self.semantic_red_score is not None
            and self.semantic_black_score is not None
        ):
            sem = np.maximum(self.semantic_red_score, self.semantic_black_score)
            sem_target = np.clip(sem * float(self.value_semantic_boost), 0.0, 100.0)
            target = np.maximum(target, sem_target.astype(np.float32))

        alpha = max(0.02, min(0.95, float(self.value_ema_alpha)))
        old_pos = np.where(known_mask, np.maximum(old_map, 0.0), 0.0)
        ema = (1.0 - alpha) * old_pos + alpha * target
        ema = np.where(occ >= 0, ema, 0.0)
        ema = np.clip(ema, 0.0, 100.0)
        mutation = np.abs(ema - old_pos)
        mutation_mean_abs = float(np.mean(mutation)) if mutation.size > 0 else 0.0
        mutation_sum_abs = float(np.sum(mutation))
        self.value_map.data = np.where(known_mask, ema, -1.0).astype(np.int16).flatten().tolist()
        self.diffuse_value_map()
        return {
            "painted": int(painted),
            "skipped_no_score": int(skipped_no_score),
            "unmatched_score_ids": int(len(unmatched_score_ids)),
            "scored_ratio": round(float(scored_ratio), 3),
            "use_missing_fill": bool(use_missing_fill),
            "mutation_mean_abs": round(float(mutation_mean_abs), 4),
            "mutation_sum_abs": round(float(mutation_sum_abs), 2),
        }

    def update_value_map(self, regions, cells=None, pixels_norm=None):
        if cells is None:
            cells = []
        if pixels_norm is None:
            pixels_norm = []
        info = self.value_map.info

        # 先做全图轻度衰减，让旧高价值区域逐渐过期
        decayed = []
        for v in self.value_map.data:
            if v < 0:
                decayed.append(v)
            else:
                decayed.append(max(0, min(100, int(v * self.value_decay))))
        self.value_map.data = decayed

        pose = self.latest_odom.pose.pose
        yaw = self.get_yaw_from_quat(pose.orientation)
        boat_x = pose.position.x
        boat_y = pose.position.y

        for region in regions:
            score = max(0.0, min(1.0, region["score"]))
            if score <= 0.0:
                continue
            distance = max(0.0, min(self.max_region_distance_m, region["distance_m"]))
            bearing_rad = math.radians(region["bearing_deg"])
            heading = yaw + bearing_rad

            tx = boat_x + distance * math.cos(heading)
            ty = boat_y + distance * math.sin(heading)
            center_gx, center_gy = self.world_to_grid(tx, ty, info)
            if center_gx is None:
                continue

            paint_radius = max(
                info.resolution,
                self.value_paint_radius_m * max(1.0, float(self.value_region_radius_scale)),
            )
            peak = int(100.0 * score)
            # 目标点打点
            self.paint_value_disk(center_gx, center_gy, paint_radius, peak)

            # 沿机器人到目标连线做稀疏涂抹，增强全局记忆连续性
            steps = max(1, int(self.value_trace_steps))
            for i in range(1, steps):
                alpha = float(i) / float(steps)
                sx = boat_x + alpha * (tx - boat_x)
                sy = boat_y + alpha * (ty - boat_y)
                gx, gy = self.world_to_grid(sx, sy, info)
                if gx is None:
                    continue
                self.paint_value_disk(
                    gx, gy, paint_radius * 0.8, int(peak * (1.0 - 0.5 * alpha))
                )

        # 若模型直接输出了地图栅格坐标，优先按全局坐标落点，避免尺度不对齐
        for cell in cells:
            score = max(0.0, min(1.0, float(cell["score"])))
            if score <= 0.0:
                continue
            gx = int(cell["grid_x"])
            gy = int(cell["grid_y"])
            if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
                continue
            peak = int(100.0 * score)
            self.paint_value_disk(
                gx,
                gy,
                max(
                    info.resolution,
                    self.value_paint_radius_m * max(1.0, float(self.value_region_radius_scale)),
                ),
                peak,
            )

        # 归一化像素坐标 -> 栅格坐标（图像原点左上）
        for p in pixels_norm:
            u = int(round(p["u_norm"] * (info.width - 1)))
            v = int(round(p["v_norm"] * (info.height - 1)))
            gx = u
            gy = (info.height - 1) - v
            if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
                continue
            peak = int(100.0 * p["score"])
            self.paint_value_disk(
                gx,
                gy,
                max(
                    info.resolution,
                    self.value_paint_radius_m * max(1.0, float(self.value_region_radius_scale)),
                ),
                peak,
            )

        # 把累计语义记忆注入 value_map，避免只剩触发点周围小区域
        self.apply_semantic_memory_to_value_map()
        # 最后做一次扩散平滑，输出连续区域价值而非离散点
        self.diffuse_value_map()


def main():
    rospy.init_node("vla_mission_node")
    VlamissionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
