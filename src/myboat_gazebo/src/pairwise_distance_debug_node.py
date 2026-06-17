#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String


class PairwiseDistanceDebugNode:
    """Publish absolute distance between two odometry sources.

    This node is intentionally generic so it can be instantiated multiple times
    for multiple targets (one instance per target).
    """

    def __init__(self):
        rospy.init_node("pairwise_distance_debug_node")
        self.ego_odom_topic = rospy.get_param("~ego_odom_topic", "/myboat/odom")
        self.target_odom_topic = rospy.get_param("~target_odom_topic", "/target_boat/odom")
        self.output_topic = rospy.get_param(
            "~output_topic", "/collision/debug/absolute_distance"
        )
        self.info_topic = rospy.get_param("~info_topic", "/collision/debug/absolute_distance_info")
        self.target_name = rospy.get_param("~target_name", "target_boat")
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))

        self._ego_pos = None
        self._target_pos = None

        self._pub = rospy.Publisher(self.output_topic, Float64, queue_size=20)
        self._info_pub = rospy.Publisher(self.info_topic, String, queue_size=20)

        rospy.Subscriber(self.ego_odom_topic, Odometry, self._ego_cb, queue_size=20)
        rospy.Subscriber(self.target_odom_topic, Odometry, self._target_cb, queue_size=20)

        rospy.loginfo(
            "[pairwise_distance_debug] ego=%s target=%s output=%s info=%s target_name=%s",
            self.ego_odom_topic,
            self.target_odom_topic,
            self.output_topic,
            self.info_topic,
            self.target_name,
        )

    def _ego_cb(self, msg):
        p = msg.pose.pose.position
        self._ego_pos = (p.x, p.y, p.z)

    def _target_cb(self, msg):
        p = msg.pose.pose.position
        self._target_pos = (p.x, p.y, p.z)

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            if self._ego_pos is None or self._target_pos is None:
                if self.debug_enable:
                    rospy.loginfo_throttle(
                        2.0,
                        "[pairwise_distance_debug] waiting odom ego_ready=%s target_ready=%s",
                        str(self._ego_pos is not None),
                        str(self._target_pos is not None),
                    )
                rate.sleep()
                continue

            dx = self._target_pos[0] - self._ego_pos[0]
            dy = self._target_pos[1] - self._ego_pos[1]
            dz = self._target_pos[2] - self._ego_pos[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            self._pub.publish(Float64(data=distance))
            info = {
                "stamp": rospy.Time.now().to_sec(),
                "target_name": self.target_name,
                "ego_odom_topic": self.ego_odom_topic,
                "target_odom_topic": self.target_odom_topic,
                "distance_m": distance,
            }
            self._info_pub.publish(String(data=str(info)))

            if self.debug_enable:
                rospy.loginfo_throttle(
                    2.0,
                    "[pairwise_distance_debug] target=%s distance=%.3f m",
                    self.target_name,
                    distance,
                )

            rate.sleep()


if __name__ == "__main__":
    PairwiseDistanceDebugNode().run()
