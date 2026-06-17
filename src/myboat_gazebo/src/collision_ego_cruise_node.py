#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String


class CollisionEgoCruiseNode:
    """Point-to-point normal navigation command generator for ego boat."""

    def __init__(self):
        rospy.init_node("collision_ego_cruise_node")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/collision/normal_cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "/myboat/odom")
        self.speed = float(rospy.get_param("~speed", 0.4))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.wait_for_trigger = bool(rospy.get_param("~wait_for_trigger", True))
        self.trigger_topic = rospy.get_param("~trigger_topic", "/traj_start_trigger")
        self.goal_x = float(rospy.get_param("~goal_x", -410.0))
        self.goal_y = float(rospy.get_param("~goal_y", 260.0))
        self.goal_tolerance_m = float(rospy.get_param("~goal_tolerance_m", 2.0))
        self.kp_heading = float(rospy.get_param("~kp_heading", 1.2))
        self.max_turn_rate = float(rospy.get_param("~max_turn_rate", 0.25))
        self.slowdown_radius_m = float(rospy.get_param("~slowdown_radius_m", 12.0))
        self.min_speed = float(rospy.get_param("~min_speed", 0.12))
        self.stop_at_goal = bool(rospy.get_param("~stop_at_goal", True))
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self.debug_topic = rospy.get_param("~debug_topic", "/collision/ego_cruise_debug")

        self._started = not self.wait_for_trigger
        self._odom = None

        if self.wait_for_trigger:
            rospy.Subscriber(self.trigger_topic, Empty, self._trigger_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=20)

        self._pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=20)
        self._debug_pub = rospy.Publisher(self.debug_topic, String, queue_size=20)

        rospy.loginfo(
            "[collision_ego_cruise] cmd=%s odom=%s goal=(%.2f, %.2f) tol=%.2f speed=%.2f wait_for_trigger=%s trigger=%s",
            self.cmd_topic,
            self.odom_topic,
            self.goal_x,
            self.goal_y,
            self.goal_tolerance_m,
            self.speed,
            str(self.wait_for_trigger),
            self.trigger_topic,
        )

    def _trigger_cb(self, _msg):
        if not self._started:
            self._started = True
            rospy.loginfo("[collision_ego_cruise] triggered by %s, start nav", self.trigger_topic)

    def _odom_cb(self, msg):
        self._odom = msg

    @staticmethod
    def _norm_pi(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _build_cmd(self):
        cmd = Twist()
        if (not self._started) or (self._odom is None):
            return cmd, None

        p = self._odom.pose.pose.position
        yaw = self._yaw_from_quat(self._odom.pose.pose.orientation)
        dx = self.goal_x - float(p.x)
        dy = self.goal_y - float(p.y)
        dist = math.hypot(dx, dy)

        if dist <= max(0.1, self.goal_tolerance_m):
            if self.stop_at_goal:
                return cmd, dist
            cmd.linear.x = min(self.speed, self.min_speed)
            cmd.angular.z = 0.0
            return cmd, dist

        des_yaw = math.atan2(dy, dx)
        yaw_err = self._norm_pi(des_yaw - yaw)
        turn = self.kp_heading * yaw_err
        cmd.angular.z = max(-abs(self.max_turn_rate), min(abs(self.max_turn_rate), turn))

        if dist <= self.slowdown_radius_m:
            ratio = max(0.0, min(1.0, dist / max(0.1, self.slowdown_radius_m)))
            cmd.linear.x = max(self.min_speed, self.speed * ratio)
        else:
            cmd.linear.x = self.speed
        return cmd, dist

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            out, dist = self._build_cmd()
            self._pub.publish(out)

            if self.debug_enable:
                dbg = {
                    "stamp": rospy.Time.now().to_sec(),
                    "started": bool(self._started),
                    "goal": {"x": self.goal_x, "y": self.goal_y},
                    "distance_to_goal_m": None if dist is None else round(float(dist), 3),
                    "cmd_linear_x": out.linear.x,
                    "cmd_angular_z": out.angular.z,
                }
                self._debug_pub.publish(String(data=json.dumps(dbg, ensure_ascii=True)))

            rate.sleep()


if __name__ == "__main__":
    CollisionEgoCruiseNode().run()
