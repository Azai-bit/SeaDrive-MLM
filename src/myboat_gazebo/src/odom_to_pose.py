#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class OdomToPoseNode(object):
    def __init__(self):
        # 允许通过参数重映射输入/输出 topic，默认对接 myboat 和 exploration
        odom_topic = rospy.get_param("~odom_topic", "/myboat/odom")
        pose_topic = rospy.get_param("~pose_topic", "/map_ros/pose")

        self.pub_ = rospy.Publisher(pose_topic, PoseStamped, queue_size=10)
        self.sub_ = rospy.Subscriber(odom_topic, Odometry, self.odom_cb, queue_size=10)
        self.shutting_down_ = False
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("[odom_to_pose] Listening on %s, publishing PoseStamped on %s",
                      odom_topic, pose_topic)

    def odom_cb(self, msg):
        # During node teardown, callbacks may still run while publisher is already closed.
        if self.shutting_down_ or rospy.is_shutdown():
            return

        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        try:
            self.pub_.publish(ps)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                rospy.logwarn_throttle(1.0, "[odom_to_pose] publish skipped: topic closed")

    def on_shutdown(self):
        self.shutting_down_ = True


def main():
    rospy.init_node("odom_to_pose")
    node = OdomToPoseNode()
    rospy.spin()


if __name__ == "__main__":
    main()

