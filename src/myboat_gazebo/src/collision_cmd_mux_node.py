#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_msgs.msg import Bool, String


class CollisionCmdMuxNode:
    """Switch output cmd by trigger and enable_avoidance flag."""

    def __init__(self):
        rospy.init_node("collision_cmd_mux_node")
        self.enable_avoidance = bool(rospy.get_param("~enable_avoidance", False))
        self.planner_mode = self._normalize_planner_mode(rospy.get_param("~planner_mode", "svo"))
        trigger_topic = rospy.get_param("~trigger_topic", "/collision/trigger")
        self.wait_for_start_trigger = bool(rospy.get_param("~wait_for_start_trigger", True))
        self.start_trigger_topic = rospy.get_param("~start_trigger_topic", "/traj_start_trigger")
        normal_cmd_topic = rospy.get_param("~normal_cmd_topic", "/collision/normal_cmd_vel")
        avoid_cmd_topic = rospy.get_param("~avoid_cmd_topic", "/collision/llm_cmd_vel")
        output_cmd_topic = rospy.get_param("~output_cmd_topic", "/myboat/cmd_vel")
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self.debug_topic = rospy.get_param("~debug_topic", "/collision/cmd_mux_debug")

        self._trigger = False
        self._started = not self.wait_for_start_trigger
        self._normal_cmd = Twist()
        self._avoid_cmd = Twist()
        self._last_selected = ""
        self._last_trigger_log_state = None

        self._pub = rospy.Publisher(output_cmd_topic, Twist, queue_size=20)
        self._debug_pub = rospy.Publisher(self.debug_topic, String, queue_size=20)
        rospy.Subscriber(trigger_topic, Bool, self._trigger_cb, queue_size=20)
        if self.wait_for_start_trigger:
            rospy.Subscriber(self.start_trigger_topic, Empty, self._start_trigger_cb, queue_size=1)
        rospy.Subscriber(normal_cmd_topic, Twist, self._normal_cb, queue_size=20)
        rospy.Subscriber(avoid_cmd_topic, Twist, self._avoid_cb, queue_size=20)

        rospy.loginfo(
            "[collision_cmd_mux] enable_avoidance=%s planner_mode=%s trigger=%s start_trigger=%s wait_for_start=%s normal=%s avoid=%s out=%s",
            str(self.enable_avoidance),
            self.planner_mode,
            trigger_topic,
            self.start_trigger_topic,
            str(self.wait_for_start_trigger),
            normal_cmd_topic,
            avoid_cmd_topic,
            output_cmd_topic,
        )
        rospy.loginfo(
            "[collision_cmd_mux] debug_enable=%s debug_topic=%s",
            str(self.debug_enable),
            self.debug_topic,
        )
        if not self.enable_avoidance:
            rospy.logwarn(
                "[collision_cmd_mux] enable_avoidance=false: trigger=true 时也不会采用 LLM 避障命令"
            )

    @staticmethod
    def _normalize_planner_mode(mode_raw):
        mode = str(mode_raw).strip().lower()
        if mode in ("mpc", "dmpc", "smpc", "model_predictive_control", "model-predictive-control"):
            return "mpc"
        if mode in ("vo", "velocity_obstacle", "velocity-obstacle"):
            return "vo"
        if mode in ("svo", "semantic_vo", "semantic-vo", "vlm"):
            return "svo"
        return "svo"

    def _trigger_cb(self, msg):
        self._trigger = bool(msg.data)
        if self.debug_enable and self._last_trigger_log_state != self._trigger:
            self._last_trigger_log_state = self._trigger
            rospy.loginfo(
                "[collision_cmd_mux] trigger state changed -> %s",
                "true" if self._trigger else "false",
            )

    def _start_trigger_cb(self, _msg):
        if not self._started:
            self._started = True
            rospy.loginfo(
                "[collision_cmd_mux] start trigger received from %s, MPC output enabled",
                self.start_trigger_topic,
            )

    def _normal_cb(self, msg):
        self._normal_cmd = msg

    def _avoid_cb(self, msg):
        self._avoid_cmd = msg

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            mpc_active = self.planner_mode == "mpc" and self._started
            use_avoid = self.enable_avoidance and (self._trigger or mpc_active)
            if use_avoid:
                out = self._avoid_cmd
                source = "avoid_cmd"
            else:
                out = self._normal_cmd
                source = "normal_cmd"

            self._pub.publish(out)

            if self.debug_enable:
                dbg = {
                    "stamp": rospy.Time.now().to_sec(),
                    "source": source,
                    "enable_avoidance": self.enable_avoidance,
                    "trigger": self._trigger,
                    "started": self._started,
                    "out_linear_x": out.linear.x,
                    "out_angular_z": out.angular.z,
                    "normal_linear_x": self._normal_cmd.linear.x,
                    "normal_angular_z": self._normal_cmd.angular.z,
                    "avoid_linear_x": self._avoid_cmd.linear.x,
                    "avoid_angular_z": self._avoid_cmd.angular.z,
                }
                self._debug_pub.publish(String(data=str(dbg)))

                if self._last_selected != source:
                    self._last_selected = source
                    rospy.loginfo(
                        "[collision_cmd_mux] source=%s trigger=%s started=%s enable_avoidance=%s out(v=%.3f,w=%.3f)",
                        source,
                        str(self._trigger),
                        str(self._started),
                        str(self.enable_avoidance),
                        out.linear.x,
                        out.angular.z,
                    )

                rospy.loginfo_throttle(
                    2.0,
                    "[collision_cmd_mux] running source=%s trigger=%s started=%s out(v=%.3f,w=%.3f) normal(v=%.3f,w=%.3f) avoid(v=%.3f,w=%.3f)",
                    source,
                    str(self._trigger),
                    str(self._started),
                    out.linear.x,
                    out.angular.z,
                    self._normal_cmd.linear.x,
                    self._normal_cmd.angular.z,
                    self._avoid_cmd.linear.x,
                    self._avoid_cmd.angular.z,
                )
            rate.sleep()


if __name__ == "__main__":
    CollisionCmdMuxNode().run()
