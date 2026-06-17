#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Empty


class RvizTriggerBridge:
    """Bridge RViz tool clicks to /traj_start_trigger."""

    def __init__(self):
        rospy.init_node("rviz_trigger_bridge")
        self.trigger_topic = rospy.get_param("~trigger_topic", "/traj_start_trigger")
        self.point_topic = rospy.get_param("~point_topic", "/clicked_point")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.min_interval_s = float(rospy.get_param("~min_interval_s", 0.3))
        self._last_pub_t = rospy.Time(0)

        self._pub = rospy.Publisher(self.trigger_topic, Empty, queue_size=1)
        rospy.Subscriber(self.point_topic, PointStamped, self._on_point, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self._on_goal, queue_size=1)

        rospy.loginfo(
            "[rviz_trigger_bridge] point=%s goal=%s -> trigger=%s",
            self.point_topic,
            self.goal_topic,
            self.trigger_topic,
        )

    def _try_publish(self, source):
        now = rospy.Time.now()
        if (now - self._last_pub_t).to_sec() < self.min_interval_s:
            return
        self._last_pub_t = now
        self._pub.publish(Empty())
        rospy.loginfo("[rviz_trigger_bridge] trigger published by %s", source)

    def _on_point(self, _msg):
        self._try_publish("PublishPoint")

    def _on_goal(self, _msg):
        self._try_publish("2DNavGoal")


if __name__ == "__main__":
    RvizTriggerBridge()
    rospy.spin()
