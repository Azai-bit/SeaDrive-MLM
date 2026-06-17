#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String


class CollisionStateEstimatorNode:
    """Tracking + CPA/TCPA estimation skeleton."""

    def __init__(self):
        rospy.init_node("collision_state_estimator_node")
        ego_odom_topic = rospy.get_param("~ego_odom_topic", "/myboat/odom")
        target_odom_topic = rospy.get_param("~target_odom_topic", "/target_boat/odom")
        self.ego_odom_topic = ego_odom_topic
        self.target_odom_topic = target_odom_topic
        self.state_topic = rospy.get_param("~state_topic", "/collision/state_estimation")
        self.target_name = rospy.get_param("~target_name", "target_boat")
        self.use_target_odom = bool(rospy.get_param("~use_target_odom", False))
        self.pointcloud_topic = rospy.get_param(
            "~pointcloud_topic", "/myboat/sensors/lidar_wamv/points"
        )
        self.pointcloud_min_range = float(rospy.get_param("~pointcloud_min_range", 1.0))
        self.pointcloud_max_range = float(rospy.get_param("~pointcloud_max_range", 120.0))
        self.pointcloud_z_min = float(rospy.get_param("~pointcloud_z_min", -2.0))
        self.pointcloud_z_max = float(rospy.get_param("~pointcloud_z_max", 2.0))
        self.pointcloud_sample_step = int(rospy.get_param("~pointcloud_sample_step", 3))
        self.cluster_eps_m = float(rospy.get_param("~cluster_eps_m", 1.2))
        self.cluster_min_points = int(rospy.get_param("~cluster_min_points", 6))
        self.cluster_gate_m = float(rospy.get_param("~cluster_gate_m", 8.0))
        self.cluster_gate_range_scale = float(rospy.get_param("~cluster_gate_range_scale", 0.03))
        self.track_init_min_hits = int(rospy.get_param("~track_init_min_hits", 3))
        self.kf_process_noise_acc = float(rospy.get_param("~kf_process_noise_acc", 1.2))
        self.kf_measurement_noise = float(rospy.get_param("~kf_measurement_noise", 0.8))
        self.kf_init_pos_var = float(rospy.get_param("~kf_init_pos_var", 4.0))
        self.kf_init_vel_var = float(rospy.get_param("~kf_init_vel_var", 9.0))
        self.kf_max_missed_frames = int(rospy.get_param("~kf_max_missed_frames", 8))
        self.distance_topic = rospy.get_param(
            "~distance_topic", "/collision/debug/distance/myboat_target_boat"
        )
        self.distance_info_topic = rospy.get_param(
            "~distance_info_topic", "/collision/debug/distance_info/myboat_target_boat"
        )
        self.cluster_distance_debug_topic = rospy.get_param(
            "~cluster_distance_debug_topic", "/collision/debug/pointcloud_cluster_distances/myboat_target_boat"
        )
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.window_debug_enable = bool(rospy.get_param("~window_debug_enable", True))
        self.window_debug_interval_s = float(rospy.get_param("~window_debug_interval_s", 1.0))

        self._ego_odom = None
        self._target_odom = None
        self._cloud_rel = None
        self._cloud_stamp = None
        self._cloud_debug = {
            "total_points": 0,
            "sampled_points": 0,
            "valid_points": 0,
            "cluster_count": 0,
            "chosen_cluster_points": 0,
            "assoc_gate_m": None,
            "assoc_residual_m": None,
            "assoc_rejected": False,
            "selected_range_m": None,
            "selected_rel_x": None,
            "selected_rel_y": None,
            "filtered_range_m": None,
            "cluster_ranges_m": [],
        }
        self._cloud_cluster_debug = []
        self._last_rel_meas = None
        self._last_rel_meas_t = None
        self._track_inited = False
        self._kf_x = None
        self._kf_y = None
        self._kf_t = None
        self._kf_missed = 0
        self._track_init_hits = 0
        self._bootstrap_meas = None
        self._last_assoc_gate_m = None
        self._last_assoc_residual_m = None
        self._last_assoc_rejected = False
        self._last_window_debug_t = 0.0

        self._pub = rospy.Publisher(self.state_topic, String, queue_size=20)
        self._distance_pub = rospy.Publisher(self.distance_topic, Float64, queue_size=20)
        self._distance_info_pub = rospy.Publisher(self.distance_info_topic, Float64, queue_size=20)
        self._cluster_debug_pub = rospy.Publisher(self.cluster_distance_debug_topic, String, queue_size=20)
        rospy.Subscriber(ego_odom_topic, Odometry, self._ego_cb, queue_size=10)
        rospy.Subscriber(target_odom_topic, Odometry, self._target_cb, queue_size=10)
        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self._pointcloud_cb, queue_size=2)

        rospy.loginfo(
            "[collision_state_estimator] ego=%s target=%s pointcloud=%s out=%s distance=%s distance_info=%s target_name=%s use_target_odom=%s",
            ego_odom_topic,
            target_odom_topic,
            self.pointcloud_topic,
            self.state_topic,
            self.distance_topic,
            self.distance_info_topic,
            self.target_name,
            str(self.use_target_odom),
        )
        rospy.loginfo(
            "[collision_state_estimator] cluster_distance_debug_topic=%s",
            self.cluster_distance_debug_topic,
        )

    def _ego_cb(self, msg):
        self._ego_odom = msg

    def _target_cb(self, msg):
        self._target_odom = msg

    def _pointcloud_cb(self, msg):
        # Point-cloud tracking: cluster first, then Kalman filter for stable target estimate.
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        if t <= 0.0:
            t = rospy.Time.now().to_sec()

        dt = None
        if self._kf_t is not None:
            dt = t - self._kf_t
            if dt <= 1e-4:
                dt = None

        if self._track_inited and dt is not None:
            self._kalman_predict_xy(dt)

        best = None
        best_r2 = None
        step = max(1, self.pointcloud_sample_step)
        total_points = 0
        sampled_points = 0
        valid_points = 0
        valid_xy = []
        try:
            gen = point_cloud2.read_points(
                msg,
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
                if r2 < (self.pointcloud_min_range * self.pointcloud_min_range):
                    continue
                if r2 > (self.pointcloud_max_range * self.pointcloud_max_range):
                    continue
                valid_points += 1
                valid_xy.append((x, y))
                if best_r2 is None or r2 < best_r2:
                    best_r2 = r2
                    best = (x, y)
        except Exception:
            best = None

        clusters = self._cluster_points(valid_xy)
        cluster_debug = []
        for c in clusters:
            sx = 0.0
            sy = 0.0
            for idx in c:
                px, py = valid_xy[idx]
                sx += px
                sy += py
            n = float(len(c))
            cx = sx / n
            cy = sy / n
            cr = math.hypot(cx, cy)
            cluster_debug.append(
                {
                    "range_m": round(cr, 3),
                    "bearing_deg_sensor": round(math.degrees(math.atan2(cy, cx)), 2),
                    "rel_x_m": round(cx, 3),
                    "rel_y_m": round(cy, 3),
                    "points": int(len(c)),
                }
            )
        cluster_debug.sort(key=lambda x: x["range_m"])
        self._cloud_cluster_debug = cluster_debug

        chosen = self._choose_cluster(valid_xy, clusters)
        if chosen is not None:
            meas_x, meas_y, chosen_cluster_size, assoc_residual_m, assoc_gate_m = chosen
            self._last_assoc_residual_m = assoc_residual_m
            self._last_assoc_gate_m = assoc_gate_m
            self._last_assoc_rejected = False
            if not self._track_inited:
                self._track_init_hits += 1
                self._bootstrap_meas = (meas_x, meas_y)
                if self._track_init_hits >= max(1, self.track_init_min_hits):
                    self._kalman_init_xy(meas_x, meas_y, t)
                    self._track_init_hits = 0
            else:
                self._kalman_update_xy(meas_x, meas_y)
                self._kf_t = t
            self._kf_missed = 0
        else:
            self._last_assoc_rejected = bool(self._track_inited and len(clusters) > 0)
            if self._track_inited:
                self._kf_missed += 1
                self._kf_t = t
                if self._kf_missed > max(0, self.kf_max_missed_frames):
                    self._track_inited = False
                    self._kf_x = None
                    self._kf_y = None
                    self._kf_t = None
            else:
                self._track_init_hits = 0
                self._bootstrap_meas = None
            chosen_cluster_size = 0
            self._last_assoc_residual_m = None
            self._last_assoc_gate_m = None

        if self._track_inited:
            fx = self._kf_x["x"]
            fy = self._kf_y["x"]
            self._cloud_rel = (fx, fy)
            self._cloud_stamp = t
            filtered_range = math.hypot(fx, fy)
        else:
            filtered_range = None
            if best is None:
                self._cloud_rel = None
                self._cloud_stamp = None

        self._cloud_debug = {
            "total_points": int(total_points),
            "sampled_points": int(sampled_points),
            "valid_points": int(valid_points),
            "cluster_count": int(len(clusters)),
            "chosen_cluster_points": int(chosen_cluster_size),
            "assoc_gate_m": self._last_assoc_gate_m,
            "assoc_residual_m": self._last_assoc_residual_m,
            "assoc_rejected": bool(self._last_assoc_rejected),
            "selected_range_m": math.sqrt(best_r2) if best_r2 is not None else None,
            "selected_rel_x": float(best[0]) if best is not None else None,
            "selected_rel_y": float(best[1]) if best is not None else None,
            "filtered_range_m": filtered_range,
            "cluster_ranges_m": [c["range_m"] for c in cluster_debug[:12]],
        }

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
            cluster_idx = [i]
            while queue:
                cur = queue.pop()
                cx, cy = points[cur]
                for j in range(len(points)):
                    if visited[j]:
                        continue
                    px, py = points[j]
                    dx = px - cx
                    dy = py - cy
                    if (dx * dx + dy * dy) <= eps2:
                        visited[j] = True
                        queue.append(j)
                        cluster_idx.append(j)
            if len(cluster_idx) >= max(1, self.cluster_min_points):
                clusters.append(cluster_idx)
        return clusters

    def _choose_cluster(self, points, clusters):
        if not clusters:
            return None
        candidates = []
        for c in clusters:
            sx = 0.0
            sy = 0.0
            for idx in c:
                px, py = points[idx]
                sx += px
                sy += py
            n = float(len(c))
            cx = sx / n
            cy = sy / n
            cr = math.hypot(cx, cy)
            candidates.append((cx, cy, len(c), cr))

        if self._track_inited:
            pred_x = self._kf_x["x"]
            pred_y = self._kf_y["x"]
            pred_r = math.hypot(pred_x, pred_y)
            gate = self.cluster_gate_m + self.cluster_gate_range_scale * pred_r
            gated = []
            for c in candidates:
                dx = c[0] - pred_x
                dy = c[1] - pred_y
                d = math.hypot(dx, dy)
                if d <= gate:
                    gated.append((d, c))
            if gated:
                # Prefer temporally consistent and denser cluster.
                gated.sort(key=lambda x: (x[0] - 0.10 * x[1][2], x[0]))
                d, c = gated[0]
                return c[0], c[1], c[2], d, gate
            # Critical anti-spike rule: when tracking, reject out-of-gate clusters
            # instead of jumping to another nearby clutter cluster.
            return None

        # Bootstrap: prefer largest cluster first, then nearest.
        candidates.sort(key=lambda c: (-c[2], c[3]))
        c = candidates[0]
        return c[0], c[1], c[2], None, None

    def _kalman_init_axis(self, pos):
        return {
            "x": float(pos),
            "v": 0.0,
            "p00": self.kf_init_pos_var,
            "p01": 0.0,
            "p10": 0.0,
            "p11": self.kf_init_vel_var,
        }

    def _kalman_init_xy(self, x, y, t):
        self._kf_x = self._kalman_init_axis(x)
        self._kf_y = self._kalman_init_axis(y)
        self._kf_t = t
        self._track_inited = True

    def _kalman_predict_axis(self, s, dt):
        x = s["x"] + dt * s["v"]
        v = s["v"]
        p00 = s["p00"] + dt * (s["p10"] + s["p01"]) + dt * dt * s["p11"]
        p01 = s["p01"] + dt * s["p11"]
        p10 = s["p10"] + dt * s["p11"]
        p11 = s["p11"]

        q = max(1e-6, self.kf_process_noise_acc)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        p00 += 0.25 * dt4 * q
        p01 += 0.5 * dt3 * q
        p10 += 0.5 * dt3 * q
        p11 += dt2 * q

        s["x"] = x
        s["v"] = v
        s["p00"] = p00
        s["p01"] = p01
        s["p10"] = p10
        s["p11"] = p11

    def _kalman_update_axis(self, s, z):
        r = max(1e-6, self.kf_measurement_noise)
        s_innov = s["p00"] + r
        k0 = s["p00"] / s_innov
        k1 = s["p10"] / s_innov
        y = z - s["x"]

        p00 = s["p00"]
        p01 = s["p01"]
        p10 = s["p10"]
        p11 = s["p11"]

        s["x"] = s["x"] + k0 * y
        s["v"] = s["v"] + k1 * y
        s["p00"] = (1.0 - k0) * p00
        s["p01"] = (1.0 - k0) * p01
        s["p10"] = p10 - k1 * p00
        s["p11"] = p11 - k1 * p01

    def _kalman_predict_xy(self, dt):
        self._kalman_predict_axis(self._kf_x, dt)
        self._kalman_predict_axis(self._kf_y, dt)

    def _kalman_update_xy(self, x, y):
        self._kalman_update_axis(self._kf_x, x)
        self._kalman_update_axis(self._kf_y, y)

    @staticmethod
    def _calc_cpa_tcpa(rel_px, rel_py, rel_vx, rel_vy):
        rel_v_sq = rel_vx * rel_vx + rel_vy * rel_vy
        if rel_v_sq < 1e-6:
            return math.hypot(rel_px, rel_py), float("inf")
        tcpa = -((rel_px * rel_vx + rel_py * rel_vy) / rel_v_sq)
        cpa_x = rel_px + rel_vx * tcpa
        cpa_y = rel_py + rel_vy * tcpa
        dcpa = math.hypot(cpa_x, cpa_y)
        return dcpa, tcpa

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            if self._ego_odom is None:
                rate.sleep()
                continue

            if self.use_target_odom and self._target_odom is None:
                rate.sleep()
                continue

            odom_fallback = False
            if (not self.use_target_odom) and (self._cloud_rel is None):
                if self._target_odom is not None:
                    odom_fallback = True
                else:
                    rate.sleep()
                    continue

            ex = self._ego_odom.pose.pose.position.x
            ey = self._ego_odom.pose.pose.position.y

            evx = self._ego_odom.twist.twist.linear.x
            evy = self._ego_odom.twist.twist.linear.y

            tracking_source = "odom_fusion_stub"
            perception_target_id = self.target_name
            target_confidence = 1.0

            if self.use_target_odom or odom_fallback:
                tx = self._target_odom.pose.pose.position.x
                ty = self._target_odom.pose.pose.position.y
                rel_px = tx - ex
                rel_py = ty - ey
                tvx = self._target_odom.twist.twist.linear.x
                tvy = self._target_odom.twist.twist.linear.y
                rel_vx = tvx - evx
                rel_vy = tvy - evy
                target_yaw = self._yaw_from_quat(self._target_odom.pose.pose.orientation)
                if odom_fallback:
                    tracking_source = "target_odom_fallback_no_cloud"
                    perception_target_id = "target_odom_fallback"
                    target_confidence = 0.55
                else:
                    tracking_source = "odom_fusion_stub"
                    perception_target_id = self.target_name
                    target_confidence = 1.0
            else:
                # Point-cloud mode: use nearest valid point in LiDAR frame,
                # then estimate relative velocity by finite difference.
                if self._cloud_rel is None or self._cloud_stamp is None:
                    rate.sleep()
                    continue
                rel_px = float(self._cloud_rel[0])
                rel_py = float(self._cloud_rel[1])
                now_t = float(self._cloud_stamp)
                tracking_source = "pointcloud_direct_tracking"
                perception_target_id = "pointcloud_nearest"
                target_confidence = 0.35

                rel_vx = 0.0
                rel_vy = 0.0
                rel_dt = None
                if self._last_rel_meas is not None and self._last_rel_meas_t is not None:
                    dt = now_t - self._last_rel_meas_t
                    if dt > 1e-3:
                        rel_vx = (rel_px - self._last_rel_meas[0]) / dt
                        rel_vy = (rel_py - self._last_rel_meas[1]) / dt
                        rel_dt = dt
                self._last_rel_meas = (rel_px, rel_py)
                self._last_rel_meas_t = now_t

                tvx = rel_vx + evx
                tvy = rel_vy + evy
                target_yaw = math.atan2(tvy, tvx) if math.hypot(tvx, tvy) > 0.05 else 0.0

                cloud_age_s = max(0.0, rospy.Time.now().to_sec() - now_t)
                pointcloud_tracking_debug = {
                    "cloud_age_s": round(cloud_age_s, 3),
                    "rel_dt_s": round(rel_dt, 4) if rel_dt is not None else None,
                    "selected_rel_x_m": round(rel_px, 3),
                    "selected_rel_y_m": round(rel_py, 3),
                    "selected_range_m": round(math.hypot(rel_px, rel_py), 3),
                    "selected_bearing_deg_sensor": round(math.degrees(math.atan2(rel_py, rel_px)), 2),
                    "relative_vx_mps": round(rel_vx, 3),
                    "relative_vy_mps": round(rel_vy, 3),
                    "relative_speed_mps": round(math.hypot(rel_vx, rel_vy), 3),
                    "target_vx_mps": round(tvx, 3),
                    "target_vy_mps": round(tvy, 3),
                    "target_speed_mps": round(math.hypot(tvx, tvy), 3),
                    "target_heading_rad": round(target_yaw, 4),
                    "pointcloud_stats": dict(self._cloud_debug),
                    "cluster_obstacles": list(self._cloud_cluster_debug[:12]),
                    "filters": {
                        "min_range_m": self.pointcloud_min_range,
                        "max_range_m": self.pointcloud_max_range,
                        "z_min_m": self.pointcloud_z_min,
                        "z_max_m": self.pointcloud_z_max,
                        "sample_step": int(max(1, self.pointcloud_sample_step)),
                    },
                }
            
            if self.use_target_odom:
                pointcloud_tracking_debug = {
                    "enabled": False,
                    "reason": "use_target_odom_true",
                    "cluster_obstacles": [],
                }
            elif odom_fallback:
                pointcloud_tracking_debug = {
                    "enabled": False,
                    "reason": "fallback_to_target_odom_no_cloud",
                    "cluster_obstacles": list(self._cloud_cluster_debug[:12]),
                    "pointcloud_stats": dict(self._cloud_debug),
                }

            dcpa, tcpa = self._calc_cpa_tcpa(rel_px, rel_py, rel_vx, rel_vy)
            ego_yaw = self._yaw_from_quat(self._ego_odom.pose.pose.orientation)
            rel_bearing = math.atan2(rel_py, rel_px) - ego_yaw
            while rel_bearing > math.pi:
                rel_bearing -= 2.0 * math.pi
            while rel_bearing < -math.pi:
                rel_bearing += 2.0 * math.pi
            state = {
                "stamp": rospy.Time.now().to_sec(),
                "tracking_source": tracking_source,
                "has_perception": self._cloud_rel is not None,
                "use_target_odom": self.use_target_odom,
                "target_id": perception_target_id,
                "target_confidence": target_confidence,
                "relative_position": {"x": rel_px, "y": rel_py},
                "relative_velocity": {"x": rel_vx, "y": rel_vy},
                "ego_velocity": {"x": evx, "y": evy},
                "target_velocity": {"x": tvx, "y": tvy},
                "ego_heading_rad": ego_yaw,
                "target_heading_rad": target_yaw,
                "relative_bearing_rad": rel_bearing,
                "relative_bearing_deg": math.degrees(rel_bearing),
                "range_m": math.hypot(rel_px, rel_py),
                "dcpa_m": dcpa,
                "tcpa_s": tcpa,
                "pointcloud_tracking_debug": pointcloud_tracking_debug,
            }
            distance = state["range_m"]

            now_log_t = rospy.Time.now().to_sec()
            if self.window_debug_enable and (now_log_t - self._last_window_debug_t) >= max(0.2, self.window_debug_interval_s):
                self._last_window_debug_t = now_log_t
                rel_speed = math.hypot(rel_vx, rel_vy)
                dbg = state.get("pointcloud_tracking_debug", {})
                stats = dbg.get("pointcloud_stats", {}) if isinstance(dbg, dict) else {}
                rospy.loginfo(
                    "[collision_state_estimator] src=%s dist=%.2fm bearing=%.1fdeg rel_speed=%.2fm/s dcpa=%.2fm tcpa=%.2fs pts(v/s/t)=%s/%s/%s clusters=%s",
                    tracking_source,
                    distance,
                    state["relative_bearing_deg"],
                    rel_speed,
                    state["dcpa_m"],
                    state["tcpa_s"],
                    str(stats.get("valid_points", "-")),
                    str(stats.get("sampled_points", "-")),
                    str(stats.get("total_points", "-")),
                    str(stats.get("cluster_ranges_m", [])),
                )

            self._distance_pub.publish(Float64(data=distance))
            self._distance_info_pub.publish(Float64(data=distance))
            self._cluster_debug_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "stamp": state["stamp"],
                            "tracking_source": tracking_source,
                            "cluster_count": len(self._cloud_cluster_debug),
                            "cluster_ranges_m": [c["range_m"] for c in self._cloud_cluster_debug],
                            "clusters": self._cloud_cluster_debug,
                        },
                        ensure_ascii=True,
                    )
                )
            )
            self._pub.publish(String(data=json.dumps(state, ensure_ascii=True)))
            rate.sleep()


if __name__ == "__main__":
    CollisionStateEstimatorNode().run()
