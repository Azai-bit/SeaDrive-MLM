#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
import os
import threading

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class TrajectoryRecorderNode:
    def __init__(self):
        rospy.init_node("trajectory_recorder_node")

        self.interval_s = float(rospy.get_param("~interval_s", 0.5))
        default_output_csv = os.path.join(os.getcwd(), "tmp", "boat_tracks.csv")
        self.output_csv = str(rospy.get_param("~output_csv", default_output_csv)).strip()
        self.channel_constraints_enable = bool(rospy.get_param("~channel_constraints_enable", False))
        self.channel_left_y = float(rospy.get_param("~channel_left_y", 254.0))
        self.channel_right_y = float(rospy.get_param("~channel_right_y", 266.0))
        self.channel_x_min = float(rospy.get_param("~channel_x_min", -446.0))
        self.channel_x_max = float(rospy.get_param("~channel_x_max", -422.0))
        self.channel_x_step = float(rospy.get_param("~channel_x_step", 12.0))
        self.goal_x = self._optional_float_param("~goal_x")
        self.goal_y = self._optional_float_param("~goal_y")

        self.ego_odom_topic = str(rospy.get_param("~ego_odom_topic", "/myboat/odom")).strip()
        self.target_odom_topic = str(rospy.get_param("~target_odom_topic", "/target_boat/odom")).strip()
        self.target2_odom_topic = str(rospy.get_param("~target2_odom_topic", "/target_boat_2/odom")).strip()
        self.target3_odom_topic = str(rospy.get_param("~target3_odom_topic", "/target_boat_3/odom")).strip()
        self.llm_decision_topic = str(rospy.get_param("~llm_decision_topic", "/collision/llm_decision")).strip()

        self._poses = {
            "myboat": None,
            "target_boat": None,
            "target_boat_2": None,
            "target_boat_3": None,
        }
        self._channel_points_written = False
        self._goal_point_written = False
        self._last_logged_vlm_call_seq = None
        self._csv_lock = threading.Lock()

        out_dir = os.path.dirname(self.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._fp = open(self.output_csv, "w", newline="")
        self._writer = csv.writer(self._fp)
        self._write_csv_row(["stamp_s", "boat", "x", "y", "yaw", "record_type", "group"], flush=True)

        rospy.Subscriber(self.ego_odom_topic, Odometry, self._ego_cb, queue_size=20)
        rospy.Subscriber(self.target_odom_topic, Odometry, self._target1_cb, queue_size=20)
        rospy.Subscriber(self.target2_odom_topic, Odometry, self._target2_cb, queue_size=20)
        rospy.Subscriber(self.target3_odom_topic, Odometry, self._target3_cb, queue_size=20)
        if self.llm_decision_topic:
            rospy.Subscriber(self.llm_decision_topic, String, self._llm_decision_cb, queue_size=20)

        rospy.loginfo(
            "[trajectory_recorder] interval=%.2fs out=%s ego=%s t1=%s t2=%s t3=%s llm_decision=%s channel_constraints=%s goal=%s",
            self.interval_s,
            self.output_csv,
            self.ego_odom_topic,
            self.target_odom_topic,
            self.target2_odom_topic,
            self.target3_odom_topic,
            self.llm_decision_topic or "disabled",
            str(self.channel_constraints_enable),
            "(%.3f, %.3f)" % (self.goal_x, self.goal_y)
            if self.goal_x is not None and self.goal_y is not None
            else "none",
        )

        rospy.Timer(rospy.Duration(max(0.05, self.interval_s)), self._on_timer)
        rospy.on_shutdown(self._on_shutdown)

    @staticmethod
    def _optional_float_param(name):
        value = rospy.get_param(name, None)
        if value is None or value == "":
            return None
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    def _write_csv_row(self, row, flush=False):
        with self._csv_lock:
            self._writer.writerow(row)
            if flush:
                self._fp.flush()

    def _flush_csv(self):
        with self._csv_lock:
            self._fp.flush()

    @staticmethod
    def _xy_yaw(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(p.x), float(p.y), float(yaw)

    def _ego_cb(self, msg):
        self._poses["myboat"] = self._xy_yaw(msg)

    def _target1_cb(self, msg):
        self._poses["target_boat"] = self._xy_yaw(msg)

    def _target2_cb(self, msg):
        self._poses["target_boat_2"] = self._xy_yaw(msg)

    def _target3_cb(self, msg):
        self._poses["target_boat_3"] = self._xy_yaw(msg)

    @staticmethod
    def _positive_int(value):
        try:
            out = int(value)
        except Exception:
            return None
        return out if out > 0 else None

    @staticmethod
    def _finite_stamp(value):
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def _decision_field(decision, key):
        value = decision.get(key, "")
        if value:
            return str(value).strip()
        constraints = decision.get("trajectory_constraints", {})
        if isinstance(constraints, dict):
            value = constraints.get(key, "")
            if value:
                return str(value).strip()
        return ""

    def _llm_decision_cb(self, msg):
        try:
            decision = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[trajectory_recorder] skip invalid llm_decision json: %s", exc)
            return
        if not isinstance(decision, dict):
            return

        call_seq = self._positive_int(decision.get("llm_call_seq", decision.get("debug_call_id", -1)))
        if call_seq is None or call_seq == self._last_logged_vlm_call_seq:
            return

        stamp_s = self._finite_stamp(decision.get("stamp"))
        if stamp_s is None:
            stamp_s = rospy.Time.now().to_sec()
        course_action = self._decision_field(decision, "course_action")
        speed_action = self._decision_field(decision, "speed_action")

        self._write_csv_row(
            [
                "%.3f" % stamp_s,
                str(call_seq),
                course_action,
                speed_action,
                "",
                "",
                "",
            ],
            flush=True,
        )
        self._last_logged_vlm_call_seq = call_seq

    def _channel_constraint_points(self):
        if not self.channel_constraints_enable:
            return []
        x_lo = min(self.channel_x_min, self.channel_x_max)
        x_hi = max(self.channel_x_min, self.channel_x_max)
        step = max(0.5, abs(self.channel_x_step))
        xs = []
        cur = x_lo
        while cur <= (x_hi + 1e-6):
            xs.append(cur)
            cur += step
        if not xs or abs(xs[-1] - x_hi) > 1e-6:
            xs.append(x_hi)
        points = []
        for idx, x in enumerate(xs, start=1):
            points.append(("channel_left_%d" % idx, x, self.channel_left_y, "channel_left"))
            points.append(("channel_right_%d" % idx, x, self.channel_right_y, "channel_right"))
        return points

    def _write_channel_constraints_once(self):
        if self._channel_points_written:
            return
        wrote = 0
        for name, x, y, group in self._channel_constraint_points():
            self._write_csv_row(
                ["0.000", name, "%.6f" % float(x), "%.6f" % float(y), "", "constraint_point", group]
            )
            wrote += 1
        if wrote > 0:
            self._channel_points_written = True
            self._flush_csv()

    def _write_goal_point_once(self):
        if self._goal_point_written:
            return
        self._goal_point_written = True
        if self.goal_x is None or self.goal_y is None:
            return
        self._write_csv_row(
            [
                "0.000",
                "goal",
                "%.6f" % float(self.goal_x),
                "%.6f" % float(self.goal_y),
                "",
                "goal_point",
                "goal",
            ]
        )
        self._flush_csv()

    def _on_timer(self, _evt):
        self._write_channel_constraints_once()
        self._write_goal_point_once()
        t = rospy.Time.now().to_sec()
        wrote = 0
        for boat, pose in self._poses.items():
            if pose is None:
                continue
            self._write_csv_row(
                [
                    "%.3f" % t,
                    boat,
                    "%.6f" % pose[0],
                    "%.6f" % pose[1],
                    "%.6f" % pose[2],
                    "track",
                    boat,
                ]
            )
            wrote += 1
        if wrote > 0:
            self._flush_csv()

    def _on_shutdown(self):
        try:
            with self._csv_lock:
                self._fp.flush()
                self._fp.close()
        except Exception:
            pass

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    TrajectoryRecorderNode().run()
