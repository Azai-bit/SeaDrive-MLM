#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String


class CollisionRiskAssessorNode:
    """Trigger gate based on manual enable + front-FOV point count."""

    def __init__(self):
        rospy.init_node("collision_risk_assessor_node")
        self.state_topic = rospy.get_param("~state_topic", "/collision/state_estimation")
        self.perception_targets_topic = rospy.get_param(
            "~perception_targets_topic", "/collision/perception/targets"
        )
        self.risk_topic = rospy.get_param("~risk_topic", "/collision/risk")
        self.trigger_topic = rospy.get_param("~trigger_topic", "/collision/trigger")
        self.fov_point_count_on_threshold = int(
            rospy.get_param("~fov_point_count_on_threshold", 18)
        )
        self.fov_point_count_off_threshold = int(
            rospy.get_param("~fov_point_count_off_threshold", 8)
        )
        self.trigger_min_hold_s = float(rospy.get_param("~trigger_min_hold_s", 3.0))
        self.trigger_latch_once_true = bool(
            rospy.get_param("~trigger_latch_once_true", True)
        )
        self.require_manual_enable_from_goal = bool(
            rospy.get_param("~require_manual_enable_from_goal", True)
        )
        self.manual_enable_goal_topic = str(
            rospy.get_param("~manual_enable_goal_topic", "/move_base_simple/goal")
        ).strip()
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self._trigger_state = False
        self._trigger_last_change_t = rospy.Time(0)
        self._manual_trigger_enabled = not self.require_manual_enable_from_goal
        self._last_state = {}
        self._fov_valid_points = 0
        self._pointcloud_stats = {}

        self._risk_pub = rospy.Publisher(self.risk_topic, String, queue_size=20)
        self._trigger_pub = rospy.Publisher(self.trigger_topic, Bool, queue_size=20)
        rospy.Subscriber(self.state_topic, String, self._state_cb, queue_size=20)
        rospy.Subscriber(
            self.perception_targets_topic, String, self._targets_cb, queue_size=20
        )
        if self.require_manual_enable_from_goal and self.manual_enable_goal_topic:
            rospy.Subscriber(
                self.manual_enable_goal_topic,
                PoseStamped,
                self._manual_enable_goal_cb,
                queue_size=1,
            )

        rospy.loginfo(
            "[collision_risk] state=%s perception=%s risk=%s trigger=%s manual_enable_from_goal=%s goal_topic=%s on(fov_pts>=%d) off(fov_pts<=%d) hold=%.1fs latch_once_true=%s",
            self.state_topic,
            self.perception_targets_topic,
            self.risk_topic,
            self.trigger_topic,
            str(self.require_manual_enable_from_goal),
            self.manual_enable_goal_topic,
            self.fov_point_count_on_threshold,
            self.fov_point_count_off_threshold,
            self.trigger_min_hold_s,
            str(self.trigger_latch_once_true),
        )

    def _manual_enable_goal_cb(self, _msg):
        if self._manual_trigger_enabled:
            return
        self._manual_trigger_enabled = True
        if self.debug_enable:
            rospy.loginfo(
                "[collision_risk] 已收到 2D Nav Goal，允许 /collision/trigger 按风险条件打开"
            )
        self._update_trigger_and_publish()

    def _targets_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        self._pointcloud_stats = data.get("pointcloud_stats", {}) if isinstance(data, dict) else {}
        self._fov_valid_points = int(self._pointcloud_stats.get("fov_valid_points", 0) or 0)
        self._update_trigger_and_publish()

    def _state_cb(self, msg):
        try:
            self._last_state = json.loads(msg.data)
        except Exception:
            return
        self._update_trigger_and_publish()

    def _update_trigger_and_publish(self):
        state = self._last_state if isinstance(self._last_state, dict) else {}
        dcpa = float(state.get("dcpa_m", 1e9))
        tcpa = float(state.get("tcpa_s", 1e9))
        now = rospy.Time.now()
        trigger_on = self._fov_valid_points >= max(1, self.fov_point_count_on_threshold)
        trigger_off = self._fov_valid_points <= max(0, self.fov_point_count_off_threshold)

        trigger_candidate = self._trigger_state
        if not self._trigger_state:
            if trigger_on:
                trigger_candidate = True
        else:
            if not self.trigger_latch_once_true:
                held_for = (now - self._trigger_last_change_t).to_sec()
                if held_for >= self.trigger_min_hold_s and trigger_off:
                    trigger_candidate = False

        trigger = trigger_candidate and self._manual_trigger_enabled

        if trigger != self._trigger_state:
            if self.debug_enable:
                rospy.loginfo(
                    "[collision_risk] trigger change %s -> %s (fov_pts=%d on=%s off=%s manual_enabled=%s dcpa=%.3f tcpa=%.3f)",
                    str(self._trigger_state),
                    str(trigger),
                    int(self._fov_valid_points),
                    str(trigger_on),
                    str(trigger_off),
                    str(self._manual_trigger_enabled),
                    dcpa,
                    tcpa,
                )
            self._trigger_state = trigger
            self._trigger_last_change_t = now

        if trigger:
            level = "HIGH"
        elif self._fov_valid_points >= max(1, int(0.5 * self.fov_point_count_on_threshold)):
            level = "MEDIUM"
        else:
            level = "LOW"

        risk = {
            "stamp": rospy.Time.now().to_sec(),
            "level": level,
            "trigger": trigger,
            "metric": "fov_valid_points",
            "fov_valid_points": int(self._fov_valid_points),
            "fov_point_count_on_threshold": int(self.fov_point_count_on_threshold),
            "fov_point_count_off_threshold": int(self.fov_point_count_off_threshold),
            "dcpa_m": dcpa,
            "tcpa_s": tcpa,
            "rule": "manual_goal_gate + hysteresis: on(fov_valid_points>=on_thr), off(fov_valid_points<=off_thr) after hold",
            "trigger_on_candidate": trigger_on,
            "trigger_off_candidate": trigger_off,
            "manual_trigger_enabled": self._manual_trigger_enabled,
            "trigger_state_latched": self._trigger_state,
            "trigger_latch_once_true": self.trigger_latch_once_true,
            "pointcloud_stats": dict(self._pointcloud_stats),
        }

        self._risk_pub.publish(String(data=json.dumps(risk, ensure_ascii=True)))
        self._trigger_pub.publish(Bool(data=trigger))


if __name__ == "__main__":
    CollisionRiskAssessorNode()
    rospy.spin()
