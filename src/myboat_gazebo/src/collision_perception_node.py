#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class CollisionPerceptionNode:
    """LiDAR clustering + lightweight multi-target tracking publisher."""

    def __init__(self):
        rospy.init_node("collision_perception_node")

        self.cloud_topic = rospy.get_param("~cloud_topic", "/myboat/sensors/lidar_wamv/points")
        self.ego_odom_topic = rospy.get_param("~ego_odom_topic", "/myboat/odom")
        self.target_odom_topic = rospy.get_param("~target_odom_topic", "/target_boat/odom")
        self.targets_topic = rospy.get_param("~targets_topic", "/collision/perception/targets")
        self.marker_topic = rospy.get_param("~marker_topic", "/collision/perception/cluster_markers")
        self.decision_topic = rospy.get_param("~decision_topic", "/collision/llm_decision")
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))

        self.pointcloud_min_range = float(rospy.get_param("~pointcloud_min_range", 1.5))
        self.pointcloud_max_range = float(rospy.get_param("~pointcloud_max_range", 120.0))
        self.pointcloud_z_min = float(rospy.get_param("~pointcloud_z_min", -2.0))
        self.pointcloud_z_max = float(rospy.get_param("~pointcloud_z_max", 2.0))
        self.pointcloud_sample_step = int(rospy.get_param("~pointcloud_sample_step", 2))
        self.pointcloud_fov_deg = float(rospy.get_param("~pointcloud_fov_deg", 90.0))

        self.cluster_eps_m = float(rospy.get_param("~cluster_eps_m", 1.8))
        self.cluster_min_points = int(rospy.get_param("~cluster_min_points", 4))
        self.cluster_merge_distance_m = float(
            rospy.get_param("~cluster_merge_distance_m", 1.8)
        )

        self.track_gate_m = float(rospy.get_param("~track_gate_m", 8.0))
        self.track_gate_range_scale = float(rospy.get_param("~track_gate_range_scale", 0.06))
        self.track_max_count = int(rospy.get_param("~track_max_count", 8))
        self.track_init_min_hits = int(rospy.get_param("~track_init_min_hits", 1))
        self.track_max_missed_frames = int(rospy.get_param("~track_max_missed_frames", 10))
        self.track_alpha = float(rospy.get_param("~track_alpha", 0.70))
        self.track_beta = float(rospy.get_param("~track_beta", 0.50))
        self.track_size_alpha = float(rospy.get_param("~track_size_alpha", 0.45))
        self.track_speed_max = float(rospy.get_param("~track_speed_max", 12.0))
        self.track_dt_min = float(rospy.get_param("~track_dt_min", 1e-3))
        self.track_dt_max = float(rospy.get_param("~track_dt_max", 1.0))

        self.use_target_odom_fallback = bool(rospy.get_param("~use_target_odom_fallback", False))
        self.marker_enable = bool(rospy.get_param("~marker_enable", True))
        self.marker_use_tracks = bool(rospy.get_param("~marker_use_tracks", True))
        self.marker_min_hits = int(
            rospy.get_param("~marker_min_hits", self.track_init_min_hits)
        )
        self.marker_max_missed_frames = int(
            rospy.get_param("~marker_max_missed_frames", 2)
        )
        self.marker_box_min_xy_m = float(rospy.get_param("~marker_box_min_xy_m", 0.6))
        self.marker_box_min_z_m = float(rospy.get_param("~marker_box_min_z_m", 0.8))
        self.marker_text_height_m = float(rospy.get_param("~marker_text_height_m", 0.9))
        self.marker_alpha = float(rospy.get_param("~marker_alpha", 0.65))
        self.semantic_buoy_world_speed_max = float(
            rospy.get_param("~semantic_buoy_world_speed_max", 0.18)
        )
        self.semantic_buoy_max_xy_m = float(
            rospy.get_param("~semantic_buoy_max_xy_m", 1.8)
        )
        self.semantic_buoy_max_size_m2 = float(
            rospy.get_param("~semantic_buoy_max_size_m2", 2.5)
        )
        self.semantic_buoy_min_hits = int(
            rospy.get_param("~semantic_buoy_min_hits", 2)
        )
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self.debug_log_interval_s = float(rospy.get_param("~debug_log_interval_s", 1.0))

        self._ego_odom = None
        self._target_odom = None
        self._last_cloud = None
        self._last_cloud_t = None

        self._tracks = {}
        self._track_id_pool = list(range(1, max(3, self.track_max_count) + 1))
        self._semantic_labels = {}
        self._last_debug_t = 0.0

        self._targets_pub = rospy.Publisher(self.targets_topic, String, queue_size=20)
        self._marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=10)

        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        rospy.Subscriber(self.ego_odom_topic, Odometry, self._ego_cb, queue_size=10)
        rospy.Subscriber(self.target_odom_topic, Odometry, self._target_cb, queue_size=10)
        rospy.Subscriber(self.decision_topic, String, self._decision_cb, queue_size=20)

        rospy.loginfo(
            "[collision_perception] cloud=%s ego_odom=%s target_odom=%s decision=%s out=%s marker=%s min_range=%.2f fov=%.1fdeg",
            self.cloud_topic,
            self.ego_odom_topic,
            self.target_odom_topic,
            self.decision_topic,
            self.targets_topic,
            self.marker_topic,
            self.pointcloud_min_range,
            self.pointcloud_fov_deg,
        )
        rospy.loginfo(
            "[collision_perception] cluster(eps=%.2f,min_pts=%d,merge=%.2f) track_assoc=hungarian gate=%.2f+%.3fr max_missed=%d alpha=%.2f beta=%.2f size_alpha=%.2f marker_tracks=%s",
            self.cluster_eps_m,
            self.cluster_min_points,
            self.cluster_merge_distance_m,
            self.track_gate_m,
            self.track_gate_range_scale,
            self.track_max_missed_frames,
            self.track_alpha,
            self.track_beta,
            self.track_size_alpha,
            str(self.marker_use_tracks),
        )

    def _cloud_cb(self, msg):
        self._last_cloud = msg
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        self._last_cloud_t = t if t > 0.0 else rospy.Time.now().to_sec()

    def _ego_cb(self, msg):
        self._ego_odom = msg

    def _target_cb(self, msg):
        self._target_odom = msg

    def _decision_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        items = payload.get("track_classifications") or []
        if not isinstance(items, list):
            return
        updated = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            tid = self._parse_track_id(item.get("target_id", ""))
            if tid is None:
                continue
            if tid not in self._track_id_pool:
                continue
            vessel_type = self._semantic_label_en(item.get("vessel_type", "Unknown"))
            assoc = self._safe_float(item.get("association_confidence", 0.0), 0.0)
            updated[tid] = {
                "vessel_type": vessel_type if vessel_type else "Unknown",
                "association_confidence": self._clamp(float(assoc), 0.0, 1.0),
            }
        self._semantic_labels = updated

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _safe_float(v, default_v=0.0):
        try:
            return float(v)
        except Exception:
            return float(default_v)

    @staticmethod
    def _parse_track_id(raw_tid):
        text = str(raw_tid or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not text:
            return None
        if text.isdigit():
            try:
                return int(text)
            except Exception:
                return None
        match = re.search(r"(?:track|target)?_?(\d+)$", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _semantic_label_en(vessel_type):
        text = str(vessel_type or "").strip().lower().replace("-", " ").replace("_", " ")
        if text in ("lifeboat", "rescue boat", "rescue vessel"):
            return "Lifeboat"
        if text in ("usv", "unmanned boat", "unmanned vessel", "unmanned surface vessel"):
            return "USV"
        if text in ("fishing", "fishing boat", "fishing vessel", "trawler"):
            return "Fishing"
        if text in ("small vessel", "small boat", "smallvessel"):
            return "SmallVessel"
        if text in ("unknown", "unk", "none", "null"):
            return "Unknown"
        return "Unknown"

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _rel_to_world(rel_x, rel_y, ego_x, ego_y, ego_yaw):
        world_x = ego_x + rel_x * math.cos(ego_yaw) - rel_y * math.sin(ego_yaw)
        world_y = ego_y + rel_x * math.sin(ego_yaw) + rel_y * math.cos(ego_yaw)
        return world_x, world_y

    def _update_world_track_state(self, tr, now_t):
        if self._ego_odom is None:
            return
        ego_pos = self._ego_odom.pose.pose.position
        ego_yaw = self._yaw_from_quat(self._ego_odom.pose.pose.orientation)
        world_x, world_y = self._rel_to_world(
            float(tr.get("x", 0.0)),
            float(tr.get("y", 0.0)),
            float(ego_pos.x),
            float(ego_pos.y),
            float(ego_yaw),
        )
        prev_t = tr.get("world_t", None)
        prev_x = tr.get("world_x", None)
        prev_y = tr.get("world_y", None)
        world_vx = float(tr.get("world_vx", 0.0))
        world_vy = float(tr.get("world_vy", 0.0))
        if prev_t is not None and prev_x is not None and prev_y is not None:
            dt = max(self.track_dt_min, float(now_t) - float(prev_t))
            meas_vx = (float(world_x) - float(prev_x)) / dt
            meas_vy = (float(world_y) - float(prev_y)) / dt
            world_vx = self.track_beta * meas_vx + (1.0 - self.track_beta) * world_vx
            world_vy = self.track_beta * meas_vy + (1.0 - self.track_beta) * world_vy
        tr["world_x"] = float(world_x)
        tr["world_y"] = float(world_y)
        tr["world_t"] = float(now_t)
        tr["world_vx"] = float(world_vx)
        tr["world_vy"] = float(world_vy)

    def _track_obstacle_kind(self, tr):
        world_speed = math.hypot(
            float(tr.get("world_vx", 0.0)),
            float(tr.get("world_vy", 0.0)),
        )
        size_x = float(tr.get("size_x_m", 0.0))
        size_y = float(tr.get("size_y_m", 0.0))
        size_m2 = float(tr.get("size_m2", max(0.05, size_x * size_y)))
        hits = int(tr.get("hits", 0))
        if (
            hits >= max(1, self.semantic_buoy_min_hits)
            and world_speed <= max(0.01, self.semantic_buoy_world_speed_max)
            and size_x <= max(0.2, self.semantic_buoy_max_xy_m)
            and size_y <= max(0.2, self.semantic_buoy_max_xy_m)
            and size_m2 <= max(0.05, self.semantic_buoy_max_size_m2)
        ):
            return "static_buoy_like"
        return "vessel_candidate"

    def _extract_valid_xy(self, cloud_msg):
        valid_xy = []
        total_points = 0
        sampled_points = 0
        valid_points = 0
        fov_valid_points = 0
        step = max(1, int(self.pointcloud_sample_step))
        r2_min = self.pointcloud_min_range * self.pointcloud_min_range
        r2_max = self.pointcloud_max_range * self.pointcloud_max_range
        half_fov_rad = 0.5 * math.radians(max(1e-3, self.pointcloud_fov_deg))

        try:
            gen = point_cloud2.read_points(
                cloud_msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            for p in gen:
                total_points += 1
                if ((total_points - 1) % step) != 0:
                    continue
                sampled_points += 1
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
                if z < self.pointcloud_z_min or z > self.pointcloud_z_max:
                    continue
                r2 = x * x + y * y
                if r2 < r2_min or r2 > r2_max:
                    continue
                valid_points += 1
                if abs(math.atan2(y, x)) <= half_fov_rad:
                    fov_valid_points += 1
                valid_xy.append((x, y, z))
        except Exception:
            pass

        return valid_xy, total_points, sampled_points, valid_points, fov_valid_points

    def _cluster_points(self, points):
        if not points:
            return []
        eps2 = self.cluster_eps_m * self.cluster_eps_m
        visited = [False] * len(points)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue
            visited[i] = True
            queue = [i]
            cl = [i]
            while queue:
                cur = queue.pop()
                cx, cy = points[cur][0], points[cur][1]
                for j in range(len(points)):
                    if visited[j]:
                        continue
                    px, py = points[j][0], points[j][1]
                    dx = px - cx
                    dy = py - cy
                    if (dx * dx + dy * dy) <= eps2:
                        visited[j] = True
                        queue.append(j)
                        cl.append(j)
            if len(cl) >= max(1, self.cluster_min_points):
                clusters.append(cl)
        return clusters

    def _cluster_features(self, points, cluster_indices):
        feats = []
        for cl in cluster_indices:
            sx = 0.0
            sy = 0.0
            sz = 0.0
            min_x = 1e9
            max_x = -1e9
            min_y = 1e9
            max_y = -1e9
            min_z = 1e9
            max_z = -1e9
            for idx in cl:
                px, py, pz = points[idx]
                sx += px
                sy += py
                sz += pz
                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)
                min_z = min(min_z, pz)
                max_z = max(max_z, pz)
            n = float(len(cl))
            cx = sx / n
            cy = sy / n
            cz = sz / n
            feats.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "cz": cz,
                    "points": int(len(cl)),
                    "range_m": math.hypot(cx, cy),
                    "size_m2": max(0.05, (max_x - min_x) * (max_y - min_y)),
                    "size_x_m": max(0.05, max_x - min_x),
                    "size_y_m": max(0.05, max_y - min_y),
                    "size_z_m": max(0.05, max_z - min_z),
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "min_z": min_z,
                    "max_z": max_z,
                }
            )
        feats.sort(key=lambda d: d["range_m"])
        return feats

    def _merge_detections(self, detections):
        if not detections:
            return []
        merge_dist = max(0.0, float(self.cluster_merge_distance_m))
        if merge_dist <= 0.0:
            return list(detections)

        visited = [False] * len(detections)
        merged = []

        for index in range(len(detections)):
            if visited[index]:
                continue
            visited[index] = True
            queue = [index]
            group = [index]
            while queue:
                current = queue.pop()
                base = detections[current]
                for other in range(len(detections)):
                    if visited[other]:
                        continue
                    cand = detections[other]
                    if math.hypot(
                        float(base.get("cx", 0.0)) - float(cand.get("cx", 0.0)),
                        float(base.get("cy", 0.0)) - float(cand.get("cy", 0.0)),
                    ) <= merge_dist:
                        visited[other] = True
                        queue.append(other)
                        group.append(other)

            sum_w = 0.0
            sum_x = 0.0
            sum_y = 0.0
            sum_z = 0.0
            min_x = 1e9
            max_x = -1e9
            min_y = 1e9
            max_y = -1e9
            min_z = 1e9
            max_z = -1e9
            total_points = 0
            for det_index in group:
                det = detections[det_index]
                weight = max(1.0, float(det.get("points", 1)))
                sum_w += weight
                sum_x += weight * float(det.get("cx", 0.0))
                sum_y += weight * float(det.get("cy", 0.0))
                sum_z += weight * float(det.get("cz", 0.0))
                min_x = min(min_x, float(det.get("min_x", det.get("cx", 0.0))))
                max_x = max(max_x, float(det.get("max_x", det.get("cx", 0.0))))
                min_y = min(min_y, float(det.get("min_y", det.get("cy", 0.0))))
                max_y = max(max_y, float(det.get("max_y", det.get("cy", 0.0))))
                min_z = min(min_z, float(det.get("min_z", det.get("cz", 0.0))))
                max_z = max(max_z, float(det.get("max_z", det.get("cz", 0.0))))
                total_points += int(det.get("points", 0))

            cx = sum_x / max(1.0, sum_w)
            cy = sum_y / max(1.0, sum_w)
            cz = sum_z / max(1.0, sum_w)
            merged.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "cz": cz,
                    "points": int(total_points),
                    "range_m": math.hypot(cx, cy),
                    "size_m2": max(0.05, (max_x - min_x) * (max_y - min_y)),
                    "size_x_m": max(0.05, max_x - min_x),
                    "size_y_m": max(0.05, max_y - min_y),
                    "size_z_m": max(0.05, max_z - min_z),
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "min_z": min_z,
                    "max_z": max_z,
                }
            )

        merged.sort(key=lambda d: d["range_m"])
        return merged

    def _stable_marker_detections(self):
        detections = []
        for tid, tr in self._tracks.items():
            if int(tr.get("hits", 0)) < max(1, self.marker_min_hits):
                continue
            if int(tr.get("miss", 0)) > max(0, self.marker_max_missed_frames):
                continue
            detections.append(
                {
                    "id": int(tid),
                    "cx": float(tr.get("x", 0.0)),
                    "cy": float(tr.get("y", 0.0)),
                    "cz": float(tr.get("z", 0.0)),
                    "points": int(tr.get("cluster_points", 0)),
                    "range_m": math.hypot(float(tr.get("x", 0.0)), float(tr.get("y", 0.0))),
                    "size_x_m": float(tr.get("size_x_m", self.marker_box_min_xy_m)),
                    "size_y_m": float(tr.get("size_y_m", self.marker_box_min_xy_m)),
                    "size_z_m": float(tr.get("size_z_m", self.marker_box_min_z_m)),
                    "hits": int(tr.get("hits", 0)),
                    "miss": int(tr.get("miss", 0)),
                    "obstacle_kind": str(tr.get("obstacle_kind", "vessel_candidate")),
                    "semantic_candidate": bool(tr.get("semantic_candidate", True)),
                    "vessel_type": str((self._semantic_labels.get(int(tid)) or {}).get("vessel_type", "Unknown")),
                    "association_confidence": float((self._semantic_labels.get(int(tid)) or {}).get("association_confidence", 0.0)),
                }
            )
        detections.sort(key=lambda d: d["range_m"])
        return detections

    def _build_cluster_markers(self, frame_id, stamp, detections):
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = frame_id or "world"
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if (not self.marker_enable) or (not detections):
            return marker_array

        for idx, det in enumerate(detections):
            box = Marker()
            box.header.frame_id = frame_id or "world"
            box.header.stamp = stamp
            box.ns = "cluster_boxes"
            box.id = idx
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(det.get("cx", 0.0))
            box.pose.position.y = float(det.get("cy", 0.0))
            box.pose.position.z = float(det.get("cz", 0.0))
            box.pose.orientation.w = 1.0
            box.scale.x = max(self.marker_box_min_xy_m, float(det.get("size_x_m", 0.5)))
            box.scale.y = max(self.marker_box_min_xy_m, float(det.get("size_y_m", 0.5)))
            box.scale.z = max(self.marker_box_min_z_m, float(det.get("size_z_m", 0.5)))
            if str(det.get("obstacle_kind", "")) == "static_buoy_like":
                box.color.r = 0.95
                box.color.g = 0.72
                box.color.b = 0.18
            else:
                box.color.r = 0.15
                box.color.g = 0.95
                box.color.b = 0.35
            box.color.a = self._clamp(self.marker_alpha, 0.05, 1.0)
            box.lifetime = rospy.Duration(0.0)
            marker_array.markers.append(box)

            text = Marker()
            text.header.frame_id = frame_id or "world"
            text.header.stamp = stamp
            text.ns = "cluster_labels"
            text.id = 1000 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = box.pose.position.x
            text.pose.position.y = box.pose.position.y
            text.pose.position.z = box.pose.position.z + 0.5 * box.scale.z + 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = max(0.2, self.marker_text_height_m)
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.95
            label_id = det.get("id", idx + 1)
            hits = int(det.get("hits", 0))
            vessel_type = str(det.get("vessel_type", "Unknown") or "Unknown").strip()
            assoc = float(det.get("association_confidence", 0.0))
            obstacle_kind = str(det.get("obstacle_kind", "vessel_candidate") or "vessel_candidate")
            text.text = "track_%s  %s\n%.1fm  pts=%d" % (
                str(label_id),
                vessel_type,
                float(det.get("range_m", 0.0)),
                int(det.get("points", 0)),
            )
            if obstacle_kind != "vessel_candidate":
                text.text += "  kind=%s" % obstacle_kind
            if hits > 0:
                text.text += "  h=%d" % hits
            if assoc > 0.0:
                text.text += "  c=%.2f" % assoc
            text.lifetime = rospy.Duration(0.0)
            marker_array.markers.append(text)

        return marker_array

    def _predict_tracks(self, now_t):
        for tr in self._tracks.values():
            dt = now_t - tr["t"]
            if dt <= self.track_dt_min:
                continue
            dt = self._clamp(dt, self.track_dt_min, self.track_dt_max)
            tr["x"] += tr["vx"] * dt
            tr["y"] += tr["vy"] * dt
            tr["t"] = now_t

    @staticmethod
    def _hungarian_assignment(cost_matrix):
        n = len(cost_matrix)
        if n == 0:
            return []

        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, n + 1):
                    if used[j]:
                        continue
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                assignment[p[j] - 1] = j - 1
        return assignment

    def _associate(self, detections):
        track_ids = list(self._tracks.keys())
        if not track_ids and not detections:
            return [], [], []
        if not track_ids:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(track_ids), []

        det_count = len(detections)
        track_count = len(track_ids)
        size = track_count + det_count

        max_gate = self.track_gate_m
        gates = {}
        for tid in track_ids:
            tr = self._tracks[tid]
            gate = self.track_gate_m + self.track_gate_range_scale * math.hypot(tr["x"], tr["y"])
            gates[tid] = gate
            max_gate = max(max_gate, gate)

        unassigned_cost = max_gate + 1.0
        large_cost = unassigned_cost * 4.0
        cost = [[large_cost for _ in range(size)] for _ in range(size)]

        for i, tid in enumerate(track_ids):
            tr = self._tracks[tid]
            gate = gates[tid]
            for j, d in enumerate(detections):
                dx = d["cx"] - tr["x"]
                dy = d["cy"] - tr["y"]
                dist = math.hypot(dx, dy)
                cost[i][j] = dist if dist <= gate else large_cost
            for j in range(det_count, size):
                cost[i][j] = unassigned_cost

        for i in range(track_count, size):
            for j in range(det_count):
                cost[i][j] = unassigned_cost
            for j in range(det_count, size):
                cost[i][j] = 0.0

        assignment = self._hungarian_assignment(cost)

        matches = []
        matched_tracks = set()
        matched_dets = set()
        for i, col in enumerate(assignment[:track_count]):
            tid = track_ids[i]
            if 0 <= col < det_count:
                tr = self._tracks[tid]
                d = detections[col]
                dist = math.hypot(d["cx"] - tr["x"], d["cy"] - tr["y"])
                if dist <= gates[tid] and cost[i][col] < unassigned_cost:
                    matches.append((tid, col, dist))
                    matched_tracks.add(tid)
                    matched_dets.add(col)

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_dets = [j for j in range(det_count) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _update_tracks(self, detections, now_t):
        self._predict_tracks(now_t)
        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for tid, j, _dist in matches:
            tr = self._tracks[tid]
            d = detections[j]
            dt = max(self.track_dt_min, now_t - tr["last_meas_t"])

            pred_x = tr["x"]
            pred_y = tr["y"]
            mx = d["cx"]
            my = d["cy"]

            tr["x"] = self.track_alpha * mx + (1.0 - self.track_alpha) * pred_x
            tr["y"] = self.track_alpha * my + (1.0 - self.track_alpha) * pred_y
            tr["z"] = self.track_size_alpha * float(d.get("cz", 0.0)) + (1.0 - self.track_size_alpha) * float(tr.get("z", 0.0))

            vx_meas = (mx - tr["last_meas_x"]) / dt
            vy_meas = (my - tr["last_meas_y"]) / dt
            spd = math.hypot(vx_meas, vy_meas)
            if spd > self.track_speed_max:
                s = self.track_speed_max / max(1e-6, spd)
                vx_meas *= s
                vy_meas *= s

            tr["vx"] = self.track_beta * vx_meas + (1.0 - self.track_beta) * tr["vx"]
            tr["vy"] = self.track_beta * vy_meas + (1.0 - self.track_beta) * tr["vy"]

            tr["miss"] = 0
            tr["hits"] += 1
            tr["age_s"] += max(0.0, now_t - tr["last_update_t"])
            tr["last_update_t"] = now_t
            tr["last_meas_t"] = now_t
            tr["last_meas_x"] = mx
            tr["last_meas_y"] = my
            tr["cluster_points"] = d["points"]
            tr["size_m2"] = self.track_size_alpha * float(d.get("size_m2", 0.0)) + (1.0 - self.track_size_alpha) * float(tr.get("size_m2", 0.0))
            tr["size_x_m"] = self.track_size_alpha * float(d.get("size_x_m", self.marker_box_min_xy_m)) + (1.0 - self.track_size_alpha) * float(tr.get("size_x_m", self.marker_box_min_xy_m))
            tr["size_y_m"] = self.track_size_alpha * float(d.get("size_y_m", self.marker_box_min_xy_m)) + (1.0 - self.track_size_alpha) * float(tr.get("size_y_m", self.marker_box_min_xy_m))
            tr["size_z_m"] = self.track_size_alpha * float(d.get("size_z_m", self.marker_box_min_z_m)) + (1.0 - self.track_size_alpha) * float(tr.get("size_z_m", self.marker_box_min_z_m))

        for tid in unmatched_tracks:
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            tr["miss"] += 1
            tr["age_s"] += max(0.0, now_t - tr["last_update_t"])
            tr["last_update_t"] = now_t

        used_ids = set(self._tracks.keys())
        free_ids = [tid for tid in self._track_id_pool if tid not in used_ids]
        unmatched_dets_sorted = sorted(
            unmatched_dets,
            key=lambda det_index: float(detections[det_index].get("range_m", 1e9)),
        )

        for j in unmatched_dets_sorted:
            if not free_ids:
                break
            d = detections[j]
            tid = free_ids.pop(0)
            self._tracks[tid] = {
                "id": tid,
                "x": d["cx"],
                "y": d["cy"],
                "z": float(d.get("cz", 0.0)),
                "vx": 0.0,
                "vy": 0.0,
                "hits": 1,
                "miss": 0,
                "age_s": 0.0,
                "t": now_t,
                "last_meas_t": now_t,
                "last_meas_x": d["cx"],
                "last_meas_y": d["cy"],
                "last_update_t": now_t,
                "cluster_points": d["points"],
                "size_m2": d["size_m2"],
                "size_x_m": float(d.get("size_x_m", self.marker_box_min_xy_m)),
                "size_y_m": float(d.get("size_y_m", self.marker_box_min_xy_m)),
                "size_z_m": float(d.get("size_z_m", self.marker_box_min_z_m)),
            }

        for tr in self._tracks.values():
            self._update_world_track_state(tr, now_t)
            obstacle_kind = self._track_obstacle_kind(tr)
            tr["obstacle_kind"] = obstacle_kind
            tr["semantic_candidate"] = obstacle_kind == "vessel_candidate"

        to_del = []
        for tid, tr in self._tracks.items():
            if tr["miss"] > max(0, self.track_max_missed_frames):
                to_del.append(tid)
        for tid in to_del:
            self._tracks.pop(tid, None)
            self._semantic_labels.pop(int(tid), None)

    def _track_confidence(self, tr):
        hit_term = min(0.55, 0.10 * float(tr["hits"]))
        miss_penalty = min(0.35, 0.07 * float(tr["miss"]))
        size_term = min(0.25, 0.01 * float(tr.get("cluster_points", 0)))
        age_term = min(0.15, 0.03 * float(tr.get("age_s", 0.0)))
        c = 0.20 + hit_term + size_term + age_term - miss_penalty
        return self._clamp(c, 0.01, 0.99)

    def _build_targets_payload(self, now_t, cloud_stamp, detections):
        targets = []
        for tid in sorted(self._tracks.keys()):
            tr = self._tracks[tid]
            if tr["hits"] < max(1, self.track_init_min_hits):
                continue
            rel_x = float(tr["x"])
            rel_y = float(tr["y"])
            rvx = float(tr["vx"])
            rvy = float(tr["vy"])
            targets.append(
                {
                    "id": "track_%d" % int(tid),
                    "rel_x": round(rel_x, 3),
                    "rel_y": round(rel_y, 3),
                    "rel_z": round(float(tr.get("z", 0.0)), 3),
                    "range_m": round(math.hypot(rel_x, rel_y), 3),
                    "bearing_deg_sensor": round(math.degrees(math.atan2(rel_y, rel_x)), 2),
                    "rel_vx": round(rvx, 3),
                    "rel_vy": round(rvy, 3),
                    "relative_speed_mps": round(math.hypot(rvx, rvy), 3),
                    "size_m2": round(float(tr.get("size_m2", 0.0)), 3),
                    "size_x_m": round(float(tr.get("size_x_m", self.marker_box_min_xy_m)), 3),
                    "size_y_m": round(float(tr.get("size_y_m", self.marker_box_min_xy_m)), 3),
                    "size_z_m": round(float(tr.get("size_z_m", self.marker_box_min_z_m)), 3),
                    "cluster_points": int(tr.get("cluster_points", 0)),
                    "hits": int(tr["hits"]),
                    "miss": int(tr["miss"]),
                    "age_s": round(float(tr.get("age_s", 0.0)), 3),
                    "world_vx_mps": round(float(tr.get("world_vx", 0.0)), 3),
                    "world_vy_mps": round(float(tr.get("world_vy", 0.0)), 3),
                    "world_speed_mps": round(
                        math.hypot(float(tr.get("world_vx", 0.0)), float(tr.get("world_vy", 0.0))),
                        3,
                    ),
                    "obstacle_kind": str(tr.get("obstacle_kind", "vessel_candidate")),
                    "semantic_candidate": bool(tr.get("semantic_candidate", True)),
                    "confidence": round(self._track_confidence(tr), 3),
                }
            )

        targets.sort(key=lambda t: t.get("range_m", 1e9))

        payload = {
            "stamp": now_t,
            "sensor": "lidar_cluster_tracker",
            "cloud_stamp": cloud_stamp,
            "cluster_count": int(len(detections)),
            "track_count": int(len(targets)),
            "semantic_track_count": int(sum(1 for t in targets if bool(t.get("semantic_candidate", True)))),
            "targets": targets,
        }

        if self.use_target_odom_fallback and (not targets) and (self._ego_odom is not None) and (self._target_odom is not None):
            ex = self._ego_odom.pose.pose.position.x
            ey = self._ego_odom.pose.pose.position.y
            tx = self._target_odom.pose.pose.position.x
            ty = self._target_odom.pose.pose.position.y
            rel_x = tx - ex
            rel_y = ty - ey
            payload["sensor"] = "lidar_cluster_tracker_with_odom_fallback"
            payload["targets"] = [
                {
                    "id": "track_1",
                    "rel_x": round(rel_x, 3),
                    "rel_y": round(rel_y, 3),
                    "rel_z": 0.65,
                    "range_m": round(math.hypot(rel_x, rel_y), 3),
                    "bearing_deg_sensor": round(math.degrees(math.atan2(rel_y, rel_x)), 2),
                    "rel_vx": 0.0,
                    "rel_vy": 0.0,
                    "relative_speed_mps": 0.0,
                    "size_m2": None,
                    "cluster_points": 0,
                    "hits": 0,
                    "miss": 0,
                    "age_s": 0.0,
                    "confidence": 0.2,
                }
            ]
            payload["track_count"] = 1
            payload["fallback"] = "target_odom"

        return payload

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            if self._ego_odom is None:
                rate.sleep()
                continue
            if self._last_cloud is None:
                rate.sleep()
                continue

            now_t = rospy.Time.now().to_sec()
            cloud_t = self._last_cloud_t if self._last_cloud_t is not None else now_t

            valid_xy, total_pts, sampled_pts, valid_pts, fov_valid_pts = self._extract_valid_xy(self._last_cloud)
            clusters = self._cluster_points(valid_xy)
            raw_detections = self._cluster_features(valid_xy, clusters)
            detections = self._merge_detections(raw_detections)
            self._update_tracks(detections, cloud_t)

            payload = self._build_targets_payload(now_t, cloud_t, detections)
            payload["pointcloud_stats"] = {
                "total_points": int(total_pts),
                "sampled_points": int(sampled_pts),
                "valid_points": int(valid_pts),
                "fov_valid_points": int(fov_valid_pts),
                "fov_deg": round(float(self.pointcloud_fov_deg), 2),
                "min_range_m": round(float(self.pointcloud_min_range), 2),
            }
            payload["raw_cluster_count"] = int(len(raw_detections))
            self._targets_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))
            stamp = self._last_cloud.header.stamp if self._last_cloud.header.stamp else rospy.Time.now()
            frame_id = self._last_cloud.header.frame_id or "world"
            marker_detections = self._stable_marker_detections() if self.marker_use_tracks else detections
            self._marker_pub.publish(self._build_cluster_markers(frame_id, stamp, marker_detections))

            if self.debug_enable and (now_t - self._last_debug_t) >= max(0.2, self.debug_log_interval_s):
                self._last_debug_t = now_t
                rngs = [round(d["range_m"], 2) for d in detections[:8]]
                rospy.loginfo(
                    "[collision_perception] points(t/s/v)=%d/%d/%d raw_clusters=%d merged_clusters=%d marker_objs=%d ranges=%s tracks_pub=%d",
                    total_pts,
                    sampled_pts,
                    valid_pts,
                    len(raw_detections),
                    len(detections),
                    len(marker_detections),
                    str(rngs),
                    int(payload.get("track_count", 0)),
                )

            rate.sleep()


if __name__ == "__main__":
    CollisionPerceptionNode().run()
